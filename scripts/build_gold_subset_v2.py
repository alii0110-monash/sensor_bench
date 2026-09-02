#!/usr/bin/env python
"""Build gold subset v2: token_fusion ∩ MLP-rgb-probe ∩ MLP-mmwave-probe all correct.

Strengthens ground truth by requiring independent consensus across two effective
modalities (rgb + mmwave) plus the strong end-to-end model. Avoids the
v1 bias toward rgb-only high confidence.
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


def _train_probe_and_predict(samples_train, samples_eval, modality, label,
                             num_classes, max_train, device):
    rng = np.random.default_rng(0)
    if len(samples_train) > max_train:
        idx = rng.choice(len(samples_train), size=max_train, replace=False)
        train_sub = [samples_train[i] for i in idx]
    else:
        train_sub = samples_train
    X_tr = np.stack([extract_modality_feature_downsampled(s, modality, pool=8)
                     for s in train_sub])
    y_tr = np.array([s.label for s in train_sub], dtype=np.int64)
    stats, X_tr = standardize_features(X_tr)
    model = train_probe(X_tr, y_tr, num_classes=num_classes, epochs=10, lr=1e-3,
                        batch_size=256, device=device, hidden_dim=256)
    X_ev = np.stack([extract_modality_feature_downsampled(s, modality, pool=8)
                     for s in samples_eval])
    _, X_ev = standardize_features(X_ev, stats)
    with torch.no_grad():
        logits = model(torch.as_tensor(X_ev, dtype=torch.float32).to(device))
    preds = logits.argmax(dim=-1).cpu().numpy()
    print(f"[gold2] {label}: train_acc probe self = "
          f"{float((model(torch.as_tensor(X_tr[:64], dtype=torch.float32).to(device)).argmax(-1).cpu().numpy() == y_tr[:64]).mean()):.3f} "
          f"(sanity, 64 samples)")
    return preds


def _token_fusion_predict(model, samples, device, batch_size=64):
    available = ["rgb", "depth", "lidar", "mmwave", "wifi"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logits = model.predict_batch(batch, available)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def build_gold_subset_v2(dataset_root: str = "datasets/mmfi/v4",
                         token_fusion_ckpt: str = "checkpoints_v4/token_fusion_seed0.pt",
                         out_path: str = "results/gold_subset_v2.json",
                         per_class: int = 5,
                         max_train_for_probe: int = 5000,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    ds = load_dataset(dataset_root)
    val_samples = list(ds.val)
    train_samples = list(ds.train)
    print(f"[gold2] v4 val: {len(val_samples)}, train: {len(train_samples)}")

    # token_fusion predictions
    print(f"[gold2] loading token_fusion from {token_fusion_ckpt}")
    tf = TokenFusionModel()
    tf.load_state_dict(torch.load(token_fusion_ckpt, map_location="cpu"))
    tf.to(device).eval()
    tf_preds = _token_fusion_predict(tf, val_samples, device)
    print(f"[gold2] token_fusion val acc: "
          f"{(tf_preds == np.array([s.label for s in val_samples])).mean():.3f}")

    # rgb probe
    rgb_preds = _train_probe_and_predict(
        train_samples, val_samples, "rgb", "rgb", 27,
        max_train_for_probe, device)
    # mmwave probe
    mmwave_preds = _train_probe_and_predict(
        train_samples, val_samples, "mmwave", "mmwave", 27,
        max_train_for_probe, device)

    # individual accuracies on val
    y_val = np.array([s.label for s in val_samples], dtype=np.int64)
    print(f"[gold2] individual val acc: tf={(tf_preds == y_val).mean():.3f} "
          f"rgb={(rgb_preds == y_val).mean():.3f} "
          f"mmwave={(mmwave_preds == y_val).mean():.3f}")

    # 3-way intersection
    all_correct = (tf_preds == y_val) & (rgb_preds == y_val) & (mmwave_preds == y_val)
    print(f"[gold2] 3-way intersection correct: "
          f"{all_correct.sum()}/{len(val_samples)} ({all_correct.mean():.3f})")
    # pairwise intersections for reference
    tf_rgb = (tf_preds == y_val) & (rgb_preds == y_val)
    tf_mmwave = (tf_preds == y_val) & (mmwave_preds == y_val)
    rgb_mmwave = (rgb_preds == y_val) & (mmwave_preds == y_val)
    print(f"[gold2] pairwise: tf∩rgb={tf_rgb.sum()} "
          f"tf∩mmwave={tf_mmwave.sum()} rgb∩mmwave={rgb_mmwave.sum()}")

    val_ids = [s.id for s in val_samples]
    gold_ids = {}
    for c in range(27):
        candidates = [val_ids[i] for i in range(len(val_samples))
                      if y_val[i] == c and all_correct[i]]
        gold_ids[str(c)] = candidates[:per_class]
    n_total = sum(len(v) for v in gold_ids.values())
    n_full = sum(1 for v in gold_ids.values() if len(v) == per_class)
    print(f"[gold2] selected {n_total} samples across 27 classes "
          f"({n_full}/{27} classes at full {per_class})")

    # compare with v1
    v1_path = "results/gold_subset_v1.json"
    v1_overlap = None
    if os.path.exists(v1_path):
        v1 = json.load(open(v1_path))
        v1_ids = set()
        for ids in v1["samples_by_label"].values():
            v1_ids.update(ids)
        v2_ids = set()
        for ids in gold_ids.values():
            v2_ids.update(ids)
        v1_overlap = {
            "v1_total": len(v1_ids),
            "v2_total": len(v2_ids),
            "in_both": len(v1_ids & v2_ids),
            "v1_only": len(v1_ids - v2_ids),
            "v2_only": len(v2_ids - v1_ids),
        }
        print(f"[gold2] vs v1: in_both={v1_overlap['in_both']} "
              f"v1_only={v1_overlap['v1_only']} v2_only={v1_overlap['v2_only']}")

    meta = {
        "name": "gold_subset_v2",
        "source": dataset_root,
        "source_split": "val",
        "token_fusion_ckpt": token_fusion_ckpt,
        "probe_modalities": ["rgb", "mmwave"],
        "probe_hidden_dim": 256,
        "probe_max_train": max_train_for_probe,
        "per_class": per_class,
        "selection_rule": "token_fusion AND rgb-MLP-probe AND mmwave-MLP-probe "
                          "all predict label correctly on val",
        "individual_val_acc": {
            "token_fusion": float((tf_preds == y_val).mean()),
            "rgb_probe": float((rgb_preds == y_val).mean()),
            "mmwave_probe": float((mmwave_preds == y_val).mean()),
        },
        "pairwise_intersect_counts": {
            "tf_and_rgb": int(tf_rgb.sum()),
            "tf_and_mmwave": int(tf_mmwave.sum()),
            "rgb_and_mmwave": int(rgb_mmwave.sum()),
            "all_three": int(all_correct.sum()),
        },
        "n_total": n_total,
        "n_classes_full": n_full,
        "samples_by_label": gold_ids,
        "vs_v1_overlap": v1_overlap,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gold2] saved -> {out_path}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--ckpt", default="checkpoints_v4/token_fusion_seed0.pt")
    ap.add_argument("--out", default="results/gold_subset_v2.json")
    ap.add_argument("--per-class", type=int, default=5)
    args = ap.parse_args()
    build_gold_subset_v2(dataset_root=args.dataset,
                         token_fusion_ckpt=args.ckpt,
                         out_path=args.out,
                         per_class=args.per_class)