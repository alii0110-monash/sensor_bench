"""Tests for leaderboard module."""
import json

from framework.eval.dataset_quality.leaderboard import (
    load_reports, render_markdown, aggregate_quality,
)


def _write_report(path, dataset, info, compact, clean, quality):
    rep = {
        "dataset": dataset,
        "metadata": {"val_sample_count": 100, "num_classes": 27},
        "info": info,
        "compact": compact,
        "clean": clean,
        "quality": quality,
    }
    with open(path, "w") as f:
        json.dump(rep, f)


def test_load_reports(tmp_path):
    p1 = tmp_path / "q1.json"
    p2 = tmp_path / "q2.json"
    _write_report(str(p1), "v1", {"InfoScore": 0.3},
                  {"CompactScore": 0.4}, {"CleanScore": 0.5}, 0.4)
    _write_report(str(p2), "v2", {"InfoScore": 0.5},
                  {"CompactScore": 0.6}, {"CleanScore": 0.7}, 0.6)
    reps = load_reports([str(p1), str(p2)])
    assert "v1" in reps and "v2" in reps


def test_render_markdown():
    reports = {
        "v1": {"info": {"InfoScore": 0.3},
               "compact": {"CompactScore": 0.4},
               "clean": {"CleanScore": 0.5}, "quality": 0.4},
        "v2": {"info": {"InfoScore": 0.5},
               "compact": {"CompactScore": 0.6},
               "clean": {"CleanScore": 0.7}, "quality": 0.6},
    }
    md = render_markdown(reports)
    assert "v1" in md and "v2" in md
    assert "InfoScore" in md and "Quality" in md
    assert md.startswith("# ")


def test_aggregate_quality_uses_scores():
    reports = {"v1": {"quality": 0.5}, "v2": {"quality": 0.7}}
    scores = aggregate_quality(reports)
    assert scores == {"v1": 0.5, "v2": 0.7}