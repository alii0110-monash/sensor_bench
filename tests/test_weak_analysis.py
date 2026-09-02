import json
from pathlib import Path

import pytest

from scripts.weak_analysis import aggregate, load_protocol


class FakeSample:
    def __init__(self, sid, label, subject, env="E01"):
        self.id = sid
        self.label = label
        self.meta = {"subject": subject, "env": env}


class FakeModel:
    def __init__(self, prob_map):
        self.prob_map = prob_map

    def predict_batch(self, samples):
        import torch
        return torch.tensor([[self.prob_map[s.id][c] for c in sorted(self.prob_map[s.id])]
                             for s in samples])


def test_load_protocol(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"profiles": [{"id": "full", "available": ["w"]}]}))
    assert load_protocol(str(p)) == [{"id": "full", "available": ["w"]}]


def test_aggregate_per_class_subject():
    samples = [
        FakeSample("s1", 0, "S01"),
        FakeSample("s2", 0, "S01"),
        FakeSample("s3", 1, "S01"),
        FakeSample("s4", 1, "S02"),
    ]
    full = FakeModel({"s1": {0: 0.9, 1: 0.1}, "s2": {0: 0.1, 1: 0.9},
                      "s3": {1: 0.9, 0: 0.1}, "s4": {1: 0.1, 0: 0.9}})
    miss = FakeModel({"s1": {0: 0.9, 1: 0.1}, "s2": {0: 0.9, 1: 0.1},
                      "s3": {1: 0.1, 0: 0.9}, "s4": {1: 0.9, 0: 0.1}})
    res = aggregate(samples, {"full": full, "miss-mmwave": miss})
    # class 0: full 1/2=0.5, miss 2/2=1.0
    assert res["per_class"]["0"]["full"] == 0.5
    assert res["per_class"]["0"]["miss-mmwave"] == 1.0
    # class 1: full 1/2=0.5 (s3✓), miss 1/2=0.5 (s4✓)
    assert res["per_class"]["1"]["full"] == 0.5
    assert res["per_class"]["1"]["miss-mmwave"] == 0.5
    # subject S01: full 2/3, miss 2/3
    assert res["per_subject"]["S01"]["full"] == 2 / 3
    # top_weak_classes 按 deg 降序: class0 deg=-0.5 < class1 deg=0.0 → class1 在前
    assert res["top_weak_classes"][0]["cls"] == "1"
