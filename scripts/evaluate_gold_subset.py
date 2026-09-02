#!/usr/bin/env python
"""Empirically validate gold subset v2 as a control variable.

Three checks:
  A: acc on gold_v2 > full val acc → high-confidence proven
  B: classes with 0 gold samples have lowest full-val acc → real difficulty
  C: re-train probes with new seed; same pattern → reproducible, not artifact
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

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


def _probe_predict(model, X, device, batch_size=512):
    X_t = torch.as_tensor(X, dtype=torch.float32).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            preds.append(model(X_t[i:i + batch_size]).argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def _token_fusion_predict(model, samples, device, batch_size=64):
    available = ["rgb", "depth", "lidar", "mmwave", "wifi"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logits = model.predict_batch(batch, available)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--ckpt", default="checkpoints_v4/token_fusion_seed0.pt")
    ap.add_argument("--gold", default="results/gold_subset_v2.json")
    ap.add_argument("--out", default="results/gold_v2_evaluation.json")
    ap.add_argument("--max-train", type=int, default=5000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset)
    val_samples = list(ds.val)
    train_samples = list(ds.train)
    val_ids = [s.id for s in val_samples]
    y_val = np.array([s.label for s in val_samples], dtype=np.int64)
    print(f"[gold-eval] val: {len(val_samples)}")

    # Load gold subset
    gold = json.load(open(args.gold))
    gold_id_set = set()
    for ids in gold["samples_by_label"].values():
        gold_id_set.update(ids)
    gold_idx = np.array([i for i, sid in enumerate(val_ids) if sid in gold_id_set])
    print(f"[gold-eval] gold_v2: {len(gold_idx)} samples in val")

    # ---- token_fusion predictions ----
    print(f"[gold-eval] loading token_fusion")
    tf = TokenFusionModel()
    tf.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    tf.to(device).eval()
    tf_preds = _token_fusion_predict(tf, val_samples, device)
    tf_full_acc = (tf_preds == y_val).mean()
    tf_gold_acc = (tf_preds[gold_idx] == y_val[gold_idx]).mean()
    print(f"[gold-eval] token_fusion: full_val={tf_full_acc:.3f} "
          f"gold_v2={tf_gold_acc:.3f} gap={tf_gold_acc - tf_full_acc:+.3f}")

    # ---- rgb probe (seed 0 to match v2 construction) ----
    print(f"[gold-eval] training rgb probe")
    rng = np.random.default_rng(0)
    if len(train_samples) > args.max_train:
        idx = rng.choice(len(train_samples), size=args.max_train, replace=False)
        train_sub = [train_samples[i] for i in idx]
    else:
        train_sub = train_samples
    X_tr_rgb = np.stack([extract_modality_feature_downsampled(s, "rgb", pool=8)
                         for s in train_sub])
    y_tr = np.array([s.label for s in train_sub], dtype=np.int64)
    stats_rgb, X_tr_rgb = standardize_features(X_tr_rgb)
    rgb_probe = train_probe(X_tr_rgb, y_tr, num_classes=27, epochs=10,
                            batch_size=256, device=device, hidden_dim=256)
    X_ev_rgb = np.stack([extract_modality_feature_downsampled(s, "rgb", pool=8)
                         for s in val_samples])
    _, X_ev_rgb = standardize_features(X_ev_rgb, stats_rgb)
    rgb_preds = _probe_predict(rgb_probe, X_ev_rgb, device)
    rgb_full = (rgb_preds == y_val).mean()
    rgb_gold = (rgb_preds[gold_idx] == y_val[gold_idx]).mean()
    print(f"[gold-eval] rgb-probe:   full_val={rgb_full:.3f} "
          f"gold_v2={rgb_gold:.3f} gap={rgb_gold - rgb_full:+.3f}")

    # ---- mmwave probe (seed 0) ----
    print(f"[gold-eval] training mmwave probe")
    X_tr_mw = np.stack([extract_modality_feature_downsampled(s, "mmwave", pool=8)
                        for s in train_sub])
    stats_mw, X_tr_mw = standardize_features(X_tr_mw)
    mw_probe = train_probe(X_tr_mw, y_tr, num_classes=27, epochs=10,
                           batch_size=256, device=device, hidden_dim=256)
    X_ev_mw = np.stack([extract_modality_feature_downsampled(s, "mmwave", pool=8)
                        for s in val_samples])
    _, X_ev_mw = standardize_features(X_ev_mw, stats_mw)
    mw_preds = _probe_predict(mw_probe, X_ev_mw, device)
    mw_full = (mw_preds == y_val).mean()
    mw_gold = (mw_preds[gold_idx] == y_val[gold_idx]).mean()
    print(f"[gold-eval] mmwave-probe: full_val={mw_full:.3f} "
          f"gold_v2={mw_gold:.3f} gap={mw_gold - mw_full:+.3f}")

    # ---- Check C: rgb probe with different seed ----
    print(f"[gold-eval] Check C: rgb probe with new seed 1")
    rng2 = np.random.default_rng(1)
    if len(train_samples) > args.max_train:
        idx2 = rng2.choice(len(train_samples), size=args.max_train, replace=False)
        train_sub2 = [train_samples[i] for i in idx2]
    else:
        train_sub2 = train_samples
    X_tr2 = np.stack([extract_modality_feature_downsampled(s, "rgb", pool=8)
                      for s in train_sub2])
    y_tr2 = np.array([s.label for s in train_sub2], dtype=np.int64)
    stats2, X_tr2 = standardize_features(X_tr2)
    torch.manual_seed(1)
    rgb_probe2 = train_probe(X_tr2, y_tr2, num_classes=27, epochs=10,
                             batch_size=256, device=device, hidden_dim=256)
    X_ev2 = np.stack([extract_modality_feature_downsampled(s, "rgb", pool=8)
                      for s in val_samples])
    _, X_ev2 = standardize_features(X_ev2, stats2)
    rgb_preds2 = _probe_predict(rgb_probe2, X_ev2, device)
    rgb2_full = (rgb_preds2 == y_val).mean()
    rgb2_gold = (rgb_preds2[gold_idx] == y_val[gold_idx]).mean()
    print(f"[gold-eval] rgb-probe seed1: full_val={rgb2_full:.3f} "
          f"gold_v2={rgb2_gold:.3f} gap={rgb2_gold - rgb2_full:+.3f}")

    # ---- Check B: per-class acc alignment ----
    print(f"[gold-eval] Check B: per-class analysis")
    gold_counts = {int(c): len(v) for c, v in gold["samples_by_label"].items()}
    per_class_acc = defaultdict(list)
    for i in range(len(val_samples)):
        per_class_acc[int(y_val[i])].append(int(tf_preds[i] == y_val[i]))
    per_class_full = {c: float(np.mean(v)) if v else 0.0
                      for c, v in per_class_acc.items()}
    zero_classes = sorted([c for c, n in gold_counts.items() if n == 0])
    full_classes = sorted([c for c, n in gold_counts.items() if n == 5])
    print(f"[gold-eval] classes with 0 gold: {zero_classes}")
    print(f"[gold-eval] classes with 5 gold (full): {full_classes}")
    zero_avg = np.mean([per_class_full[c] for c in zero_classes])
    full_avg = np.mean([per_class_full[c] for c in full_classes])
    print(f"[gold-eval] avg full-val acc on 0-gold classes: {zero_avg:.3f}")
    print(f"[gold-eval] avg full-val acc on 5-gold classes: {full_avg:.3f}")

    # ---- Save results ----
    results = {
        "check_A_acc_gap": {
            "token_fusion": {"full_val_acc": tf_full_acc,
                             "gold_v2_acc": tf_gold_acc,
                             "gap": tf_gold_acc - tf_full_acc},
            "rgb_probe": {"full_val_acc": rgb_full,
                          "gold_v2_acc": rgb_gold,
                          "gap": rgb_gold - rgb_full},
            "mmwave_probe": {"full_val_acc": mw_full,
                             "gold_v2_acc": mw_gold,
                             "gap": mw_gold - mw_full},
        },
        "check_C_seed_sensitivity": {
            "rgb_probe_seed1": {"full_val_acc": rgb2_full,
                                "gold_v2_acc": rgb2_gold,
                                "gap": rgb2_gold - rgb2_full},
            "seed0_vs_seed1_full_val_diff": abs(rgb_full - rgb2_full),
        },
        "check_B_class_difficulty": {
            "zero_gold_classes": zero_classes,
            "five_gold_classes": full_classes,
            "avg_full_val_acc_zero_classes": float(zero_avg),
            "avg_full_val_acc_full_classes": float(full_avg),
            "per_class_full_val_acc_token_fusion": per_class_full,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gold-eval] saved -> {args.out}")


if __name__ == "__main__":
    main()