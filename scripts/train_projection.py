#!/usr/bin/env python
"""Projection distillation training (M5b).

Frozen AlignmentModel (M5a) produces canonical tokens -> PerceiverProjection
maps to target LLM space -> InfoNCE distills pseudo tokens toward the target
LLM's own text embeddings of the same synthetic captions.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel, MODALITIES
from framework.models.llm_adapter import LlamaAdapter
from framework.models.alignment import info_nce_loss


def _llm_text_emb(adapter, tokenizer, texts, device):
    """Target-LLM text embedding: mean-pool input embeddings of caption tokens."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt")
    ids = enc["input_ids"]
    model = adapter._load()[0]
    emb_device = model.get_input_embeddings().weight.device
    ids = ids.to(emb_device)
    emb = model.get_input_embeddings()(ids)            # (B, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).to(emb_device)
    pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, dim=-1)


def _action_verb_emb(adapter, tokenizer, labels, device):
    """Action-verb anchor: mean-pool the llama2 sub-token embeddings of the
    action verb word (e.g. 'stretching' -> stretch+ing). 伪 token 被拉向词表
    里真实动作词的位置, 而非整句 caption 的语境 — L3 诊断显示当前伪 token
    落在情境词(arms/room)而非动作词附近."""
    from curation.caption.verbs import LABEL_TO_VERB
    model = adapter._load()[0]
    emb_device = model.get_input_embeddings().weight.device
    emb = model.get_input_embeddings()                 # (vocab, H)
    outs = []
    for lbl in labels:
        verb = LABEL_TO_VERB(lbl).split()[0]
        ids = tokenizer(f" {verb}", add_special_tokens=False).input_ids
        ids = [i for i in ids if i not in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)]
        if not ids:
            ids = [tokenizer.unk_token_id]
        vec = emb.weight[ids].mean(dim=0)              # (H,)
        outs.append(vec)
    pooled = torch.stack(outs).to(emb_device)
    return torch.nn.functional.normalize(pooled, dim=-1)


def _stack_mods(samples, avail, device):
    mods = {}
    first = samples[0]
    for m in MODALITIES:
        if avail.get(m) and m in first.modalities:
            mods[m] = torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in samples]).to(device)
    return mods


def train_epoch(align, adapter, train, opt, text_fn, batch_size=16,
                device="cuda", anchor="caption") -> float:
    align.eval()
    adapter.projection.train()
    total = 0.0; n = 0
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        avail = {m: True for m in MODALITIES}          # 蒸馏阶段用全模态
        mods = _stack_mods(batch, avail, device)
        if not mods:
            continue
        with torch.no_grad():
            ct = align.encode_modalities(mods, avail)  # (B, M, K_max, D)
        pseudo = adapter.project(ct)                   # (B, M*k, H)
        pooled = pseudo.mean(dim=1).float()
        if anchor == "verb":
            t_emb = text_fn([s.label for s in batch]).float()   # 动作词锚
        else:
            texts = [s.text.get("captions") or s.text.get("en", [""]) for s in batch]
            texts = [t[0] if t else "" for t in texts]
            t_emb = text_fn(texts).float()             # 整句 caption 锚
        loss = info_nce_loss(pooled, t_emb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="/tmp/opencode/align_smoke/alignment_seed0.pt",
                    help="M5a alignment checkpoint (smoke default; 真训练需先跑 M5a train_alignment --text-encoder clip)")
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints_projection")
    ap.add_argument("--anchor", choices=["caption", "verb"], default="verb",
                    help="蒸馏锚: verb=动作词(推荐, 伪token落词表动作词附近); caption=整句")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.projection_head = torch.nn.Sequential(
        torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))  # 原型头 (train_alignment --init-prototype)
    align.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"), strict=False)
    align.eval().to(device)
    for p in align.parameters():
        p.requires_grad_(False)

    adapter = LlamaAdapter(model_path=args.llm, k=args.k, device=device)
    model, tok = adapter._load()
    if args.anchor == "verb":
        text_fn = lambda labels: _action_verb_emb(adapter, tok, labels, device)
    else:
        text_fn = lambda texts: _llm_text_emb(adapter, tok, texts, device)
    opt = torch.optim.AdamW(adapter.projection.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        loss = train_epoch(align, adapter, ds.train, opt, text_fn,
                           batch_size=args.batch_size, device=device, anchor=args.anchor)
        print(f"[proj] ep {ep} loss {loss:.4f}", flush=True)
    torch.save(adapter.projection.state_dict(), f"{args.out}/projection_seed0.pt")
    print(f"done -> {args.out}/projection_seed0.pt")


if __name__ == "__main__":
    main()
