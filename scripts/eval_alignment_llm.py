#!/usr/bin/env python
"""L1 retrieval eval in 4096 canonical space (M6a spec §130):
CanonicalToken(4096)池化 vs llama2 mean-pool caption embedding(4096).
用 encoders (alignment checkpoint) + PerceiverProjection(4096) 拼出 4096 表征.
注意: PerceiverProjection 是 M5b 阶段用 baseline encoders 训的——M6b 改了
encoders 但 projection 未重训, 这是"迁移评测"而非"重训评测".
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from framework.dataset.loader import load_dataset
from framework.eval.alignment import build_held_out_split
from framework.models.alignment import AlignmentModel
from framework.models.perceiver import PerceiverProjection
from framework.models.llm_adapter import LlamaAdapter
from framework.tokens.tokenizer import MODALITY_ORDER

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_projection import _llm_text_emb


def _pool_canonical(ct_pe):
    """PerceiverProjection 输出 (B, M*k, 4096) → (B, 4096) mean-pool."""
    return ct_pe.mean(dim=1)


def evaluate_4096(align, proj, adapter, tok, samples, device, batch_size=64):
    align.eval(); proj.eval()
    zs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            mods = {m: torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
                for m in MODALITY_ORDER if m in batch[0].modalities}
            avail = {m: True for m in MODALITY_ORDER if m in mods}
            ct = align.encode_modalities(mods, avail)        # (B, M, 16, 256)
            pe = proj(ct)                                    # (B, M*k, 4096)
            zs.append(torch.nn.functional.normalize(_pool_canonical(pe), dim=-1))
            texts = [s.text.get("captions") or s.text.get("en", [""]) for s in batch]
            texts = [t[0] if t else "" for t in texts]
            ts.append(_llm_text_emb(adapter, tok, texts, device))
    Z = torch.cat(zs); T = torch.cat(ts)
    sim = Z @ T.t()
    N = Z.shape[0]
    def r_at_k(k):
        _, topk = sim.topk(k, dim=1)
        hits = (topk == torch.arange(N, device=sim.device)[:, None]).any(dim=1)
        return float(hits.float().mean())
    mean_rank = float((sim > torch.diag(sim)).sum(dim=1).float().mean() + 1)
    return {"n": N, "r@1": r_at_k(1), "r@5": r_at_k(5), "r@10": r_at_k(10),
            "tr@1": r_at_k(1) if False else float(((sim.topk(1, dim=0)[1]
                  == torch.arange(N, device=sim.device)).float().mean())),
            "mean_rank": mean_rank}


def _diagnose(Z, T, labels):
    sim = Z @ T.t()
    n = sim.shape[0]
    lab = torch.tensor(labels, device=sim.device)
    ranks, same_above = [], []
    for i in range(n):
        above = sim[i] > sim[i, i]
        ranks.append(above.sum().item() + 1)
        same_above.append(
            (above & (lab == labels[i]) & (torch.arange(n, device=sim.device) != i)).sum().item())
    return float(np.mean(ranks)), float(np.mean(same_above))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="checkpoints_alignment/alignment_seed0.pt",
                    help="M5a/M6b alignment ckpt (替换 encoders)")
    ap.add_argument("--proj-ckpt", default="checkpoints_projection_verb/projection_seed0.pt",
                    help="PerceiverProjection 4096 (M5b 用 baseline encoders 训的)")
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--diagnose-label", action="store_true")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset)

    # 加载 encoders (alignment ckpt)
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.projection_head = torch.nn.Sequential(
        torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))  # 原型头
    align.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"), strict=False)
    align.eval().to(device)
    for p in align.parameters(): p.requires_grad_(False)

    # 加载 PerceiverProjection 4096 (冻结, 与 encoders 解耦)
    proj = PerceiverProjection(out_dim=4096, k=8).to(device)
    proj.load_state_dict(torch.load(args.proj_ckpt, map_location="cpu"))
    proj.eval()
    for p in proj.parameters(): p.requires_grad_(False)

    # llama2 (4096) 用于文本侧
    adapter = LlamaAdapter(model_path=args.llm, k=8, device=device)
    model, tok = adapter._load()

    train_ids = json.load(open(os.path.join(args.dataset, "splits", "train.json")))
    held, _ = build_held_out_split(train_ids, fraction=args.fraction)
    held_bases = {i for i in held if "__aug" not in i}
    held_samples = [s for s in ds.train if s.id in held_bases]

    res = evaluate_4096(align, proj, adapter, tok, held_samples, device=device)
    print(f"[eval4096] align={os.path.basename(args.align_ckpt)} "
          f"proj={os.path.basename(args.proj_ckpt)} "
          f"n={res['n']} r@1={res['r@1']:.4f} r@5={res['r@5']:.4f} "
          f"r@10={res['r@10']:.4f} tr@1={res['tr@1']:.4f} "
          f"mean_rank={res['mean_rank']:.1f}")
    if args.diagnose_label:
        with torch.no_grad():
            zs, ts = [], []
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
                ts.append(_llm_text_emb(adapter, tok, texts, device))
            mean_rank2, same_above = _diagnose(torch.cat(zs), torch.cat(ts),
                                               [s.label for s in held_samples])
        print(f"[eval4096] diagnose: mean_rank={mean_rank2:.1f} "
              f"same_label_above={same_above:.2f}")


if __name__ == "__main__":
    main()