#!/usr/bin/env python
"""M6b 复测: 用 llama2 4096 mean-pool 文本侧重测 A-E 矩阵.
循环 6 ckpt, 共享 dataset + PerceiverProjection + llama2 (避免每 ckpt 重载 LLM).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from framework.dataset.loader import load_dataset
from framework.eval.alignment import build_held_out_split
from framework.models.alignment import AlignmentModel
from framework.models.perceiver import PerceiverProjection
from framework.models.llm_adapter import LlamaAdapter
from framework.tokens.tokenizer import MODALITY_ORDER
from train_projection import _llm_text_emb
from eval_alignment_llm import _pool_canonical, _diagnose


def _eval_one(align, proj, adapter, tok, samples, device, batch_size=64):
    align.eval(); proj.eval()
    zs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            mods = {m: torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
                for m in MODALITY_ORDER if m in batch[0].modalities}
            avail = {m: True for m in MODALITY_ORDER if m in mods}
            ct = align.encode_modalities(mods, avail)
            pe = proj(ct)
            zs.append(torch.nn.functional.normalize(_pool_canonical(pe), dim=-1))
            texts = [s.text.get("captions") or s.text.get("en", [""]) for s in batch]
            texts = [t[0] if t else "" for t in texts]
            ts.append(_llm_text_emb(adapter, tok, texts, device).to(device).float())
    Z = torch.cat(zs); T = torch.cat(ts)
    sim = Z @ T.t()
    N = Z.shape[0]
    def r_at_k(k):
        _, topk = sim.topk(k, dim=1)
        hits = (topk == torch.arange(N, device=sim.device)[:, None]).any(dim=1)
        return float(hits.float().mean())
    mean_rank = float((sim > torch.diag(sim)).sum(dim=1).float().mean() + 1)
    tr1 = float((sim.topk(1, dim=0)[1] == torch.arange(N, device=sim.device)).float().mean())
    return {"n": N, "r@1": r_at_k(1), "r@5": r_at_k(5), "r@10": r_at_k(10),
            "tr@1": tr1, "mean_rank": mean_rank}


def _load_align(ckpt_path, device):
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.projection_head = torch.nn.Sequential(
        torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))
    state = torch.load(ckpt_path, map_location="cpu")
    align.load_state_dict(state, strict=False)
    align.eval().to(device)
    for p in align.parameters(): p.requires_grad_(False)
    return align


CKPTS = [
    ("baseline", "checkpoints_alignment/alignment_seed0.pt"),
    ("A_batch32", "checkpoints_alignment/m6b_A_seed0.pt"),
    ("B_batch64", "checkpoints_alignment/m6b_B_seed0.pt"),
    ("C_+CE0.5", "checkpoints_alignment/m6b_C_seed0.pt"),
    ("D_+neg_mine", "checkpoints_alignment/m6b_D_seed0.pt"),
    ("E_all", "checkpoints_alignment/m6b_E_seed0.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--proj-ckpt", default="checkpoints_projection_verb/projection_seed0.pt")
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--out", default="results/m6b_llm4096_sweep.json")
    ap.add_argument("--diagnose-label", action="store_true")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset)

    proj = PerceiverProjection(out_dim=4096, k=8).to(device)
    proj.load_state_dict(torch.load(args.proj_ckpt, map_location="cpu"))
    proj.eval()
    for p in proj.parameters(): p.requires_grad_(False)
    print(f"[sweep] projection loaded from {args.proj_ckpt}", flush=True)

    adapter = LlamaAdapter(model_path=args.llm, k=8, device=device)
    model, tok = adapter._load()
    print(f"[sweep] llama2 loaded from {args.llm}", flush=True)

    train_ids = json.load(open(os.path.join(args.dataset, "splits", "train.json")))
    held, _ = build_held_out_split(train_ids, fraction=args.fraction)
    held_bases = {i for i in held if "__aug" not in i}
    held_samples = [s for s in ds.train if s.id in held_bases]
    print(f"[sweep] held-out base: {len(held_bases)}", flush=True)

    results = {}
    for name, ckpt_path in CKPTS:
        if not os.path.exists(ckpt_path):
            print(f"[sweep] SKIP {name}: {ckpt_path} not found", flush=True)
            continue
        align = _load_align(ckpt_path, device)
        torch.cuda.empty_cache()
        res = _eval_one(align, proj, adapter, tok, held_samples, device=device)
        diag = {}
        if args.diagnose_label:
            zs, ts = [], []
            with torch.no_grad():
                for i in range(0, len(held_samples), 64):
                    b = held_samples[i:i + 64]
                    mods = {m: torch.stack([torch.from_numpy(s.modalities[m].data) for s in b]).to(device)
                            for m in MODALITY_ORDER if m in b[0].modalities}
                    avail = {m: True for m in MODALITY_ORDER if m in mods}
                    ct = align.encode_modalities(mods, avail)
                    pe = proj(ct)
                    zs.append(torch.nn.functional.normalize(_pool_canonical(pe), dim=-1))
                    texts = [s.text.get("captions") or s.text.get("en", [""]) for s in b]
                    texts = [t[0] if t else "" for t in texts]
                    ts.append(_llm_text_emb(adapter, tok, texts, device).to(device).float())
            mean_rank2, same_above = _diagnose(torch.cat(zs).to(device), torch.cat(ts),
                                                [s.label for s in held_samples])
            diag = {"mean_rank_recompute": mean_rank2, "same_label_above": same_above}
        results[name] = {**res, **diag}
        print(f"[sweep] {name:14s} r@1={res['r@1']:.4f} r@5={res['r@5']:.4f} "
              f"r@10={res['r@10']:.4f} tr@1={res['tr@1']:.4f} "
              f"mean_rank={res['mean_rank']:.1f} "
              + (f"same_above={diag.get('same_label_above', 0):.2f}" if diag else ""),
              flush=True)
        del align

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[sweep] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()