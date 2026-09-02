"""JSON metadata assembly + matplotlib diagnostic plots."""
from __future__ import annotations

import json
import os
from typing import Dict


def build_metadata(args: Dict) -> Dict:
    """Persist all hyperparameters and weights for reproducibility."""
    return {
        "dataset": args.get("dataset"),
        "eval_split": args.get("eval_split"),
        "num_classes": args.get("num_classes"),
        "probe_epochs": args.get("probe_epochs"),
        "probe_lr": args.get("probe_lr"),
        "probe_batch_size": args.get("probe_batch_size"),
        "anomaly_threshold": args.get("anomaly_threshold"),
        "js_threshold": args.get("js_threshold"),
        "hash_decimals": args.get("hash_decimals"),
        "dup_weight": args.get("dup_weight"),
        "w_info": args.get("w_info"),
        "w_compact": args.get("w_compact"),
        "w_clean": args.get("w_clean"),
        "info_weights": args.get("info_weights"),
        "val_sample_count": args.get("val_sample_count"),
        "train_sample_count": args.get("train_sample_count"),
        "probe_hidden_dim": args.get("probe_hidden_dim"),
        "depth_pool": args.get("depth_pool", 8),
    }


def assemble_report(args: Dict, info: Dict, compact: Dict,
                    clean: Dict) -> Dict:
    quality = (args.get("w_info", 0.4) * info["InfoScore"]
               + args.get("w_compact", 0.4) * compact["CompactScore"]
               + args.get("w_clean", 0.2) * clean["CleanScore"])
    return {
        "dataset": args.get("dataset"),
        "metadata": build_metadata(args),
        "info": info,
        "compact": compact,
        "clean": clean,
        "quality": float(quality),
    }


def write_report_json(report: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def plot_per_modality_acc(acc_per_modality, out_path: str) -> None:
    """Bar chart of per-modality accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = list(acc_per_modality.items())
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(6, 4))
    plt.bar(names, vals)
    plt.ylabel("val top-1 acc")
    plt.title("Per-modality probe accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(cm), cmap="Blues")
    plt.colorbar()
    plt.title("Confusion matrix (concat probe, val)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_js_histogram(js_per_sample, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.hist(js_per_sample, bins=30)
    plt.xlabel("JS divergence")
    plt.ylabel("count")
    plt.title("Cross-modal JS distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_contribution_bar(contribution_per_modality: dict, out_path: str) -> None:
    """Bar chart of drop-modality contribution per modality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = list(contribution_per_modality.items())
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(6, 4))
    plt.bar(names, vals)
    plt.ylabel("argmax change rate")
    plt.title("Drop-modality contribution (concat probe)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()