"""Tests for report module: metadata, assembly, JSON write."""
import json

import pytest

from framework.eval.dataset_quality.report import (
    build_metadata, assemble_report, write_report_json,
)


def _toy_args():
    return {
        "dataset": "datasets/mmfi/v4",
        "eval_split": "val",
        "num_classes": 27,
        "probe_epochs": 20,
        "probe_lr": 1e-3,
        "probe_batch_size": 256,
        "anomaly_threshold": 0.3,
        "js_threshold": 0.1,
        "hash_decimals": 2,
        "dup_weight": 0.5,
        "w_info": 0.4,
        "w_compact": 0.4,
        "w_clean": 0.2,
        "info_weights": {"per_modality": 0.7, "complement": 0.3},
        "val_sample_count": 3500,
        "train_sample_count": 46509,
    }


def test_build_metadata_includes_all_keys():
    md = build_metadata(_toy_args())
    for k in ["num_classes", "probe_epochs", "anomaly_threshold",
              "js_threshold", "hash_decimals", "w_info", "w_compact",
              "w_clean", "val_sample_count", "probe_hidden_dim"]:
        assert k in md, f"missing {k}"


def test_assemble_report_includes_all_scores():
    info = {"mean_acc": 0.5, "complement_gain": 0.1, "InfoScore": 0.4}
    compact = {"confusion_rate": 0.3, "CompactScore": 0.7,
               "fisher_ratio": 1.5, "leave_one_out_dist_p90": 2.1}
    clean = {"anomaly_rate": 0.1, "inconsistency_rate": 0.05,
             "dup_rate": 0.02, "CleanScore": 0.9}
    rep = assemble_report(_toy_args(), info, compact, clean)
    assert rep["quality"] == pytest.approx(0.4 * 0.4 + 0.4 * 0.7 + 0.2 * 0.9)
    for k in ["info", "compact", "clean", "metadata", "quality"]:
        assert k in rep


def test_write_report_json(tmp_path):
    rep = assemble_report(_toy_args(),
                          {"mean_acc": 0.5, "complement_gain": 0.1, "InfoScore": 0.4},
                          {"confusion_rate": 0.3, "CompactScore": 0.7,
                           "fisher_ratio": 1.5, "leave_one_out_dist_p90": 2.1},
                          {"anomaly_rate": 0.1, "inconsistency_rate": 0.05,
                           "dup_rate": 0.02, "CleanScore": 0.9})
    out = tmp_path / "quality.json"
    write_report_json(rep, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["quality"] == rep["quality"]