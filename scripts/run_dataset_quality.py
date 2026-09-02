#!/usr/bin/env python
"""Dataset quality eval entry (P0-P4 of dataset-quality-eval-design spec).

Three-dimension (info / compact / clean) lightweight Linear-probe evaluation
that measures dataset intrinsic quality, decoupled from downstream task /
LLM / template evaluation.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality.modality_probe import (
    MODALITY_ORDER, train_probe, evaluate_probe,
    compute_info_score, extract_modality_feature_downsampled,
    standardize_features,
)
from framework.eval.dataset_quality.probe_fusion import (
    PerModConcatMLP, train_probe_fusion, predict_fusion,
    PerModCrossAttnMLP, train_probe_crossattn, predict_crossattn,
)
from framework.eval.dataset_quality.compactness import (
    compute_confusion_rate, compute_fisher_ratio,
    compute_leave_one_out_distances,
)
from framework.eval.dataset_quality.cleanliness import (
    compute_anomaly_rate, compute_modality_contribution,
    compute_dup_rate_quantized,
)
from framework.eval.dataset_quality.report import (
    assemble_report, write_report_json,
    plot_per_modality_acc, plot_confusion_matrix, plot_contribution_bar,
)

ALLOWED_EVAL_SPLITS = {"train", "val"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-split", choices=sorted(ALLOWED_EVAL_SPLITS),
                    default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-classes", type=int, default=27)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-batch-size", type=int, default=256)
    ap.add_argument("--anomaly-threshold", type=float, default=0.3)
    ap.add_argument("--js-threshold", type=float, default=0.1)
    ap.add_argument("--hash-decimals", type=int, default=2)
    ap.add_argument("--dup-weight", type=float, default=0.5)
    ap.add_argument("--w-info", type=float, default=0.4)
    ap.add_argument("--w-compact", type=float, default=0.4)
    ap.add_argument("--w-clean", type=float, default=0.2)
    ap.add_argument("--info-w-per-modality", type=float, default=0.7)
    ap.add_argument("--info-w-complement", type=float, default=0.3)
    ap.add_argument("--plots-dir", default=None,
                    help="If set, write diagnostic plots here.")
    ap.add_argument("--max-train-samples", type=int, default=10000,
                    help="Subsample train to this many (stratified by label) "
                         "to bound RAM. Default 10000.")
    ap.add_argument("--probe-hidden-dim", type=int, default=256,
                    help="MLP hidden dim (0 = pure Linear). Default 256.")
    ap.add_argument("--probe-embed-dim", type=int, default=64,
                    help="Per-modality projection dim (PerModConcatMLP). Default 64.")
    ap.add_argument("--probe-fusion-hidden", type=int, default=128,
                    help="PerModConcatMLP head hidden dim. Default 128.")
    ap.add_argument("--probe-dropout-p", type=float, default=0.2,
                    help="Modality dropout probability in PerModConcatMLP. Default 0.2.")
    ap.add_argument("--probe-num-heads", type=int, default=4,
                    help="Attention heads in PerModCrossAttnMLP. Default 4.")
    return ap.parse_args(argv)


def validate_splits(args):
    """P0 guard: test split must never enter probe evaluation."""
    assert args.eval_split != "test", \
        "test split cannot be used for probe evaluation (P0 guard)"
    assert args.eval_split in ALLOWED_EVAL_SPLITS


def run(dataset_root: str, out_path: str, eval_split: str = "val",
        num_classes: int = 27, epochs: int = 20, lr: float = 1e-3,
        batch_size: int = 256, anomaly_threshold: float = 0.3,
        js_threshold: float = 0.1, hash_decimals: int = 2,
        dup_weight: float = 0.5, w_info: float = 0.4,
        w_compact: float = 0.4, w_clean: float = 0.2,
        info_w_per_modality: float = 0.7,
        info_w_complement: float = 0.3,
        plots_dir=None, device: str = "cpu",
        max_train_samples: int = 10000,
        probe_hidden_dim: int = 256,
        probe_embed_dim: int = 64,
        probe_fusion_hidden: int = 128,
        probe_dropout_p: float = 0.2,
        probe_num_heads: int = 4) -> Dict:
    """End-to-end run. Returns the assembled report dict."""
    assert eval_split != "test", "P0 guard: test split cannot be probed"
    ds = load_dataset(dataset_root)
    all_train = list(ds.train)
    eval_samples = list(ds.val if eval_split == "val" else ds.train)

    # Stratified subsample of train (bound RAM)
    if len(all_train) > max_train_samples:
        from collections import defaultdict
        rng = np.random.default_rng(0)
        by_label = defaultdict(list)
        for s in all_train:
            by_label[s.label].append(s)
        per_label = max(1, max_train_samples // len(by_label))
        train_samples = []
        for lbl in sorted(by_label):
            ids = by_label[lbl]
            if len(ids) <= per_label:
                train_samples.extend(ids)
            else:
                idx = rng.choice(len(ids), size=per_label, replace=False)
                train_samples.extend([ids[i] for i in idx])
    else:
        train_samples = all_train

    available_modalities = [m for m in MODALITY_ORDER
                            if any(m in s.modalities for s in train_samples)]
    print(f"[dq] modalities={available_modalities} "
          f"train={len(train_samples)}/{len(all_train)} eval={len(eval_samples)}")

    # --- Feature preparation (downsampled depth + per-modality z-score) ---
    def _feat_extract(samples):
        """Returns {modality: (N, dim)} for all samples."""
        feats = {}
        for m in available_modalities:
            feats[m] = np.stack(
                [extract_modality_feature_downsampled(s, m, pool=8)
                 for s in samples if m in s.modalities])
        return feats

    feats_tr = _feat_extract(train_samples)
    feats_ev = _feat_extract(eval_samples)
    y_tr_all = np.array([s.label for s in train_samples], dtype=np.int64)
    y_ev_all = np.array([s.label for s in eval_samples], dtype=np.int64)

    # Standardize per-modality with train stats; record concat column slices
    std_stats = {}
    concat_slices = {}  # modality -> (start, end) column indices in concat feature
    offset = 0
    for m in available_modalities:
        std_stats[m], feats_tr[m] = standardize_features(feats_tr[m])
        _, feats_ev[m] = standardize_features(feats_ev[m], std_stats[m])
        d = feats_tr[m].shape[1]
        concat_slices[m] = (offset, offset + d)
        offset += d
    concat_tr = np.concatenate([feats_tr[m] for m in available_modalities], axis=1)
    concat_ev = np.concatenate([feats_ev[m] for m in available_modalities], axis=1)

    # --- Dimension 1: info (per-modality MLP + concat MLP) ---
    acc_per_modality = {}
    for m in available_modalities:
        model = train_probe(feats_tr[m], y_tr_all, num_classes=num_classes,
                            epochs=epochs, lr=lr, batch_size=batch_size,
                            device=device, hidden_dim=probe_hidden_dim)
        acc_per_modality[m] = evaluate_probe(model, feats_ev[m], y_ev_all,
                                             device=device)

    concat_model = train_probe_fusion(concat_tr, y_tr_all, concat_slices,
                                    num_classes=num_classes,
                                    embed_dim=probe_embed_dim,
                                    hidden=probe_fusion_hidden,
                                    epochs=epochs, lr=lr,
                                    batch_size=batch_size, device=device,
                                    dropout_p=probe_dropout_p,
                                    class_weighted=True)
    acc_concat = (predict_fusion(concat_model, concat_ev, concat_slices,
                                 device=device) == y_ev_all).mean()

    info = compute_info_score(acc_per_modality, acc_concat,
                              w_per_modality=info_w_per_modality,
                              w_complement=info_w_complement)
    info["acc_per_modality"] = acc_per_modality
    info["acc_concat"] = acc_concat

    # --- Dimension 2: compact ---
    preds = predict_fusion(concat_model, concat_ev, concat_slices, device=device)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_ev_all, preds):
        cm[t, p] += 1
    confusion_rate = compute_confusion_rate(y_ev_all, preds, num_classes)
    fisher = compute_fisher_ratio(concat_ev, y_ev_all)
    loo = compute_leave_one_out_distances(concat_ev, y_ev_all)
    compact = {
        "confusion_matrix": cm.tolist(),
        "confusion_rate": confusion_rate,
        "CompactScore": float(1.0 - confusion_rate),
        "fisher_ratio": fisher,
        "leave_one_out_dist_p90": float(np.percentile(loo, 90)),
    }

    # --- Dimension 3: clean ---
    # Anomaly rate: concat probe on train
    full_train_preds = predict_fusion(concat_model, concat_tr, concat_slices,
                                      device=device)
    train_probs = np.eye(num_classes)[full_train_preds]  # one-hot (we need probs for anomaly)
    # Actually we need softmax probs for anomaly_rate. Use predict_fusion? it returns argmax.
    # Let's compute probs separately:
    with torch.no_grad():
        Xt_tr = torch.as_tensor(concat_tr, dtype=torch.float32).to(device)
        train_probs = torch.softmax(
            concat_model(Xt_tr, concat_slices), dim=-1).cpu().numpy()
    anomaly_rate = compute_anomaly_rate(train_probs, y_tr_all,
                                        anomaly_threshold=anomaly_threshold)

    # Drop-modality contribution: drop modality m in the projected space
    # (the model's projection for m is zeroed via avail=False). More correct
    # than zeroing raw features because high-dim modalities dominated input.
    with torch.no_grad():
        Xt_ev = torch.as_tensor(concat_ev, dtype=torch.float32).to(device)
        full_logits = concat_model(Xt_ev, concat_slices)
        full_probs = torch.softmax(full_logits, dim=-1).cpu().numpy()
    contribution_per_modality = {}
    for m in available_modalities:
        avail_drop = {mm: (mm != m) for mm in available_modalities}
        with torch.no_grad():
            drop_logits = concat_model(Xt_ev, concat_slices, avail=avail_drop)
            drop_probs = torch.softmax(drop_logits, dim=-1).cpu().numpy()
        contribution_per_modality[m] = compute_modality_contribution(
            full_probs, drop_probs)
    modality_contribution = float(np.mean(list(contribution_per_modality.values())))

    dup_rate = compute_dup_rate_quantized(concat_ev, decimals=hash_decimals)
    # CleanScore: three "good" components. anomaly low, contribution high, dup low.
    good = ((1.0 - anomaly_rate)
            + modality_contribution
            + (1.0 - dup_weight * dup_rate)) / 2.0
    clean = {
        "anomaly_rate": anomaly_rate,
        "modality_contribution": modality_contribution,
        "contribution_per_modality": contribution_per_modality,
        "dup_rate": dup_rate,
        "CleanScore": float(max(0.0, min(1.0, good))),
    }

    args_dict = {
        "dataset": dataset_root, "eval_split": eval_split,
        "num_classes": num_classes, "probe_epochs": epochs,
        "probe_lr": lr, "probe_batch_size": batch_size,
        "anomaly_threshold": anomaly_threshold,
        "js_threshold": js_threshold, "hash_decimals": hash_decimals,
        "dup_weight": dup_weight, "w_info": w_info,
        "w_compact": w_compact, "w_clean": w_clean,
        "info_weights": {"per_modality": info_w_per_modality,
                         "complement": info_w_complement},
        "val_sample_count": len(eval_samples),
        "train_sample_count": len(train_samples),
        "max_train_samples_requested": max_train_samples,
        "probe_hidden_dim": probe_hidden_dim,
        "probe_embed_dim": probe_embed_dim,
        "probe_fusion_hidden": probe_fusion_hidden,
        "probe_dropout_p": probe_dropout_p,
        "probe_num_heads": probe_num_heads,
        "depth_pool": 8,
    }
    report = assemble_report(args_dict, info, compact, clean)
    write_report_json(report, out_path)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plot_per_modality_acc(acc_per_modality,
                              os.path.join(plots_dir, "per_modality_acc.png"))
        plot_confusion_matrix(cm.tolist(),
                              os.path.join(plots_dir, "confusion_matrix.png"))
        plot_contribution_bar(contribution_per_modality,
                              os.path.join(plots_dir, "modality_contribution.png"))
    return report


def main():
    args = parse_args()
    validate_splits(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args.dataset, args.out, eval_split=args.eval_split,
        num_classes=args.num_classes, epochs=args.probe_epochs,
        lr=args.probe_lr, batch_size=args.probe_batch_size,
        anomaly_threshold=args.anomaly_threshold,
        js_threshold=args.js_threshold, hash_decimals=args.hash_decimals,
        dup_weight=args.dup_weight, w_info=args.w_info,
        w_compact=args.w_compact, w_clean=args.w_clean,
        info_w_per_modality=args.info_w_per_modality,
        info_w_complement=args.info_w_complement,
        plots_dir=args.plots_dir, device=device,
        max_train_samples=args.max_train_samples,
        probe_hidden_dim=args.probe_hidden_dim,
        probe_embed_dim=args.probe_embed_dim,
        probe_fusion_hidden=args.probe_fusion_hidden,
        probe_dropout_p=args.probe_dropout_p,
        probe_num_heads=args.probe_num_heads)


if __name__ == "__main__":
    main()