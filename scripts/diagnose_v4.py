#!/usr/bin/env python
"""v5 Phase 1 diagnostic: per-subject × per-class accuracy on v4 val.

Focus: class 9/12/14/22 (gold v2 zero-class candidates) + weak modalities.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.dataset.loader import load_dataset
from framework.models.token_fusion import TokenFusionModel


def _predict(model, samples, available, device, batch_size=64):
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logits = model.predict_batch(batch, available)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def _subject_of(sid: str) -> int:
    m = re.match(r"E\d+_S(\d+)_", sid)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--ckpt", default="checkpoints_v4/token_fusion_seed0.pt")
    ap.add_argument("--out", default="results/v5_diagnostic.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset)
    val_samples = list(ds.val)
    print(f"[diag] val: {len(val_samples)}")

    tf = TokenFusionModel()
    tf.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    tf.to(device).eval()

    # Full-model predictions
    full_avail = ["rgb", "depth", "lidar", "mmwave", "wifi"]
    tf_full = _predict(tf, val_samples, full_avail, device)
    print(f"[diag] full-model val acc: "
          f"{(tf_full == np.array([s.label for s in val_samples])).mean():.3f}")

    # Per-modality-only predictions
    per_modality_preds = {}
    for m in full_avail:
        per_modality_preds[m] = _predict(tf, val_samples, [m], device)

    y_val = np.array([s.label for s in val_samples], dtype=np.int64)
    subjects = [_subject_of(s.id) for s in val_samples]
    subject_arr = np.array(subjects)
    val_ids = [s.id for s in val_samples]

    # --- Per-class accuracy: full + per-modality ---
    print("\n=== Per-class accuracy (v4 val, token_fusion) ===")
    class_full_acc = {}
    class_per_modality_acc = {m: {} for m in full_avail}
    for c in range(27):
        mask = y_val == c
        if mask.sum() == 0:
            continue
        class_full_acc[c] = float((tf_full[mask] == c).mean())
        for m in full_avail:
            class_per_modality_acc[m][c] = float(
                (per_modality_preds[m][mask] == c).mean())
    print(f"{'class':>5s} {'n':>4s} {'full':>6s} {'rgb':>6s} {'mmwave':>7s} {'lidar':>6s} {'depth':>6s} {'wifi':>6s}")
    for c in sorted(class_full_acc.keys()):
        n = int((y_val == c).sum())
        row = f"{c:>5d} {n:>4d} {class_full_acc[c]:>6.3f}"
        for m in full_avail:
            row += f" {class_per_modality_acc[m].get(c, 0):>6.3f}"
        print(row)
    # Sort by full acc ascending
    print("\n=== Worst 10 classes by full-model acc ===")
    for c in sorted(class_full_acc, key=class_full_acc.get)[:10]:
        n = int((y_val == c).sum())
        print(f"class {c:>2d} n={n:>4d} full={class_full_acc[c]:.3f} "
              f"rgb={class_per_modality_acc['rgb'].get(c,0):.3f} "
              f"mmwave={class_per_modality_acc['mmwave'].get(c,0):.3f}")

    # --- Gold v2 zero-classes focus ---
    gold = json.load(open("results/gold_subset_v2.json"))
    zero_classes = sorted([int(c) for c, ids in gold["samples_by_label"].items()
                           if len(ids) == 0])
    print(f"\n=== Gold v2 zero-classes: {zero_classes} ===")
    for c in zero_classes:
        n = int((y_val == c).sum())
        full_acc = class_full_acc.get(c, 0)
        # top-3 confusion targets
        wrong_mask = (y_val == c) & (tf_full != c)
        if wrong_mask.sum() > 0:
            from collections import Counter
            top_confusions = Counter(tf_full[wrong_mask].tolist()).most_common(3)
        else:
            top_confusions = []
        print(f"class {c}: n={n} full_acc={full_acc:.3f} "
              f"top_confusions={top_confusions}")

    # --- Per-subject × per-class on zero-classes ---
    print("\n=== Per-subject breakdown for zero-classes ===")
    for c in zero_classes:
        class_mask = y_val == c
        print(f"class {c}:")
        for subj in sorted(set(subjects)):
            subj_mask = class_mask & (subject_arr == subj)
            if subj_mask.sum() == 0:
                continue
            acc = (tf_full[subj_mask] == c).mean()
            print(f"  S{subj}: n={subj_mask.sum():>3d} acc={acc:.3f}")

    # --- Cross-class confusion matrix on zero-classes ---
    print("\n=== Confusion matrix on zero-classes (row=truth, col=pred) ===")
    confusions = defaultdict(lambda: defaultdict(int))
    for c in zero_classes:
        truth_mask = y_val == c
        for true, pred in zip(y_val[truth_mask], tf_full[truth_mask]):
            confusions[int(true)][int(pred)] += 1
    for c in zero_classes:
        total = sum(confusions[c].values())
        items = sorted(confusions[c].items(), key=lambda x: -x[1])[:5]
        items_str = ", ".join(f"{p}:{n}/{total}" for p, n in items)
        print(f"class {c} (n={total}): {items_str}")

    # --- Modality best/worst for zero-classes ---
    print("\n=== Per-modality best modality for zero-classes ===")
    for c in zero_classes:
        per_mod = {m: class_per_modality_acc[m].get(c, 0) for m in full_avail}
        best = max(per_mod, key=per_mod.get)
        worst = min(per_mod, key=per_mod.get)
        print(f"class {c}: best={best} {per_mod[best]:.3f} "
              f"worst={worst} {per_mod[worst]:.3f}")

    # Save
    out = {
        "per_class_full_acc": {str(c): v for c, v in class_full_acc.items()},
        "per_class_per_modality_acc": {
            m: {str(c): v for c, v in d.items()}
            for m, d in class_per_modality_acc.items()
        },
        "zero_classes": zero_classes,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[diag] saved -> {args.out}")


if __name__ == "__main__":
    main()