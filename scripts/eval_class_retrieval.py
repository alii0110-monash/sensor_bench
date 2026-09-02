#!/usr/bin/env python
"""提议2 方案A: 传感器侧 class-conditional retrieval 评测.

绕开"整句 caption 检索"评测天花板 (M6b: r@1 max 0.0109 ≪ 随机 1/27≈0.037).
纯传感器 embedding, 测同类样本是否在 top-k 内被检索到 (cr@1/5/10 + mean_rank).
对比 baseline + M6b 变体 A-E, 判断训练手段是否其实有效但被整句检索掩盖.

用法:
  python scripts/eval_class_retrieval.py --out results/class_retrieval.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.eval.alignment import build_held_out_split, evaluate_class_retrieval
from framework.models.alignment import AlignmentModel


CKPTS = [
    ("baseline", "checkpoints_alignment/alignment_seed0.pt"),
    ("A_batch32", "checkpoints_alignment/m6b_A_seed0.pt"),
    ("B_batch64", "checkpoints_alignment/m6b_B_seed0.pt"),
    ("C_+CE0.5", "checkpoints_alignment/m6b_C_seed0.pt"),
    ("D_+neg_mine", "checkpoints_alignment/m6b_D_seed0.pt"),
    ("E_all", "checkpoints_alignment/m6b_E_seed0.pt"),
]


def _load_align(ckpt_path, device):
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.projection_head = torch.nn.Sequential(
        torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))
    state = torch.load(ckpt_path, map_location="cpu")
    align.load_state_dict(state, strict=False)
    align.eval().to(device)
    for p in align.parameters(): p.requires_grad_(False)
    return align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--out", default="results/class_retrieval.json")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset)

    train_ids = json.load(open(os.path.join(args.dataset, "splits", "train.json")))
    held, _ = build_held_out_split(train_ids, fraction=args.fraction)
    held_bases = {i for i in held if "__aug" not in i}
    held_samples = [s for s in ds.train if s.id in held_bases]
    print(f"[class-retrieval] held-out base: {len(held_bases)}", flush=True)

    results = {}
    for name, ckpt_path in CKPTS:
        if not os.path.exists(ckpt_path):
            print(f"[class-retrieval] SKIP {name}: {ckpt_path} not found", flush=True)
            continue
        align = _load_align(ckpt_path, device)
        torch.cuda.empty_cache()
        res = evaluate_class_retrieval(align, held_samples, device=device)
        results[name] = res
        print(f"[class-retrieval] {name:14s} cr@1={res['cr@1']:.4f} "
              f"cr@5={res['cr@5']:.4f} cr@10={res['cr@10']:.4f} "
              f"cr_mean_rank={res['cr_mean_rank']:.1f} n={res['n']}", flush=True)
        del align

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[class-retrieval] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
