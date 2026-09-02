"""Precompute per-sample predictions for a split into a JSON file readable by
the GUI (`--predictions`).

Uses a trained token_fusion checkpoint via `predict_batch` (batched, GPU).
Output format: {sample_id: {"pred": int, "conf": float, "source": str}}

Usage:
    python curation/gui/scripts/precompute_predictions.py \
        --dataset datasets/mmfi/v4 --split val \
        --ckpt checkpoints_v4/token_fusion_seed0.pt \
        --out results/predictions_val_v4.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from framework.dataset.loader import load_dataset  # noqa: E402
from framework.models.token_fusion import TokenFusionModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/mmfi/v4")
    p.add_argument("--split", default="val")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-samples", type=int, default=None, help="cap samples (smoke test)")
    args = p.parse_args()

    ds = load_dataset(args.dataset, mode="lazy")
    split = ds.splits.get(args.split)
    if not split:
        raise SystemExit(f"split {args.split!r} empty in {args.dataset}")
    n = len(split)
    if args.max_samples is not None:
        n = min(n, args.max_samples)

    model = TokenFusionModel.load(args.ckpt)
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    available = ds.modalities

    out: dict = {}
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            samples = [split[i] for i in range(start, end)]
            logits = model.predict_batch(samples, available).float()
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            conf = probs.max(dim=-1).values
            for k, s in enumerate(samples):
                out[s.id] = {
                    "pred": int(pred[k].item()),
                    "conf": float(conf[k].item()),
                    "source": os.path.basename(args.ckpt),
                }
            done = end
            if done % (args.batch_size * 4) == 0 or done == n:
                print(f"[precompute] {done}/{n}  ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[precompute] done: {len(out)} predictions -> {args.out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()