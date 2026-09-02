#!/usr/bin/env python
"""Build gold subset v1: token_fusion ∩ MLP-rgb-probe both correct.

Source: v4 val (held-out from token_fusion training; held-out from probe's
train subset). For each class, keep up to 5 samples where BOTH predictors agree
with the ground truth label. Save as IDs + meta to results/gold_subset_v1.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality.modality_probe import (
    extract_modality_feature_downsampled, standardize_features,
    train_probe,
)
from framework.models.token_fusion import TokenFusionModel


def build_gold_subset(dataset_root: str = "datasets/mmfi/v4",
                      token_fusion_ckpt: str = "checkpoints_v4/token_fusion_seed0.pt",
                      out_path: str = "results/gold_subset_v1.json",
                      per_class: int = 5,
                      max_train_for_probe: int = 5000,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    ds = load_dataset(dataset_root)
    val_samples = list(ds.val)
    print(f"[gold] v4 val: {len(val_samples)} samples")

    # --- 1. Token_fusion predictions on val ---
    print(f"[gold] loading token_fusion from {token_fusion_ckpt}")
    tf = TokenFusionModel()
    tf.load_state_dict(torch.load(token_fusion_ckpt, map_location="cpu"))
    tf.to(device).eval()
    available = ["rgb", "depth", "lidar", "mmwave", "wifi"]
    tf_preds = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(val_samples), bs):
            batch = val_samples[i:i + bs]
            logits = tf.predict_batch(batch, available)
            tf_preds.append(logits.argmax(dim=-1).cpu().numpy())
    tf_preds = np.concatenate(tf_preds)
    print(f"[gold] token_fusion predicted on val")

    # --- 2. MLP probe on rgb (trained on train subset) ---
    print(f"[gold] training rgb MLP probe (train subset {max_train_for_probe})")
    train_samples = list(ds.train)
    rng = np.random.default_rng(0)
    if len(train_samples) > max_train_for_probe:
        idx = rng.choice(len(train_samples), size=max_train_for_probe, replace=False)
        train_sub = [train_samples[i] for i in idx]
    else:
        train_sub = train_samples
    X_tr = np.stack([extract_modality_feature_downsampled(s, "rgb", pool=8) for s in train_sub])
    y_tr = np.array([s.label for s in train_sub], dtype=np.int64)
    std_stats, X_tr = standardize_features(X_tr)
    probe = train_probe(X_tr, y_tr, num_classes=27, epochs=10, lr=1e-3,
                        batch_size=256, device=device, hidden_dim=256)

    X_ev = np.stack([extract_modality_feature_downsampled(s, "rgb", pool=8) for s in val_samples])
    _, X_ev = standardize_features(X_ev, std_stats)
    with torch.no_grad():
        probe_logits = probe(torch.as_tensor(X_ev, dtype=torch.float32).to(device))
    probe_preds = probe_logits.argmax(dim=-1).cpu().numpy()
    print(f"[gold] probe predicted on val")

    # --- 3. Find intersection: ground truth = tf_pred = probe_pred ---
    y_val = np.array([s.label for s in val_samples], dtype=np.int64)
    val_ids = [s.id for s in val_samples]
    both_correct = (tf_preds == y_val) & (probe_preds == y_val)
    print(f"[gold] both correct: {both_correct.sum()}/{len(val_samples)} "
          f"({both_correct.mean():.3f})")

    # --- 4. Sample per_class per label ---
    gold_ids = {}
    for c in range(27):
        candidates = [val_ids[i] for i in range(len(val_samples))
                      if y_val[i] == c and both_correct[i]]
        gold_ids[str(c)] = candidates[:per_class]
    n_total = sum(len(v) for v in gold_ids.values())
    print(f"[gold] selected {n_total} samples across 27 classes "
          f"(target {27 * per_class})")

    meta = {
        "name": "gold_subset_v1",
        "source": dataset_root,
        "source_split": "val",
        "token_fusion_ckpt": token_fusion_ckpt,
        "probe_modality": "rgb",
        "probe_hidden_dim": 256,
        "probe_max_train": max_train_for_probe,
        "per_class": per_class,
        "selection_rule": "token_fusion AND rgb-MLP-probe both predict label "
                          "correctly on val",
        "n_total": n_total,
        "samples_by_label": gold_ids,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gold] saved -> {out_path}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--ckpt", default="checkpoints_v4/token_fusion_seed0.pt")
    ap.add_argument("--out", default="results/gold_subset_v1.json")
    ap.add_argument("--per-class", type=int, default=5)
    args = ap.parse_args()
    build_gold_subset(dataset_root=args.dataset,
                      token_fusion_ckpt=args.ckpt,
                      out_path=args.out,
                      per_class=args.per_class)