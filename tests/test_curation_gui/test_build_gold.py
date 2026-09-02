from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest

from framework.dataset.loader import load_dataset
from framework.dataset.sample import Modality, Sample

from curation.gui.core.edit_log import EditLog
from curation.gui.scripts import build_gold


def _make_dataset(root: str, n: int = 6) -> None:
    """Write a tiny Dataset-protocol root: n base samples + splits."""
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    ids = []
    for i in range(n):
        sid = f"E01_S01_A{min(i + 1, 27):02d}_f1-3"
        ids.append(sid)
        sample = Sample(
            id=sid,
            label=i % 27,
            modalities={
                "rgb": Modality(data=np.random.rand(3, 17, 2).astype(np.float32),
                                frame_indices=[1, 2, 3], sample_rate=20, name="rgb"),
                "wifi": Modality(data=np.random.rand(3, 3, 114, 10).astype(np.float32),
                                 frame_indices=[1, 2, 3], sample_rate=1000, name="wifi"),
            },
            text={"captions": [f"original caption for {sid}"]},
            meta={"subject": "S01", "env": "E01"},
        )
        with open(os.path.join(root, "data", f"{sid}.pkl"), "wb") as f:
            pickle.dump(sample.to_dict(), f)
    with open(os.path.join(root, "splits", "val.json"), "w") as f:
        json.dump(ids, f)
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"name": "mmfi", "version": "test", "n_samples": n, "changelog": []}, f)
    with open(os.path.join(root, "modalities.yaml"), "w") as f:
        f.write("modalities:\n- rgb\n- wifi\n")


def _write_edits(path: str, edits: list) -> None:
    log = EditLog(str(path))
    for e in edits:
        log.save(**e)
    return log


def test_build_gold_dry_run_no_write(tmp_path):
    src = str(tmp_path / "src")
    _make_dataset(src)
    edits = str(tmp_path / "edits.jsonl")
    _write_edits(edits, [
        {"sample_id": "E01_S01_A01_f1-3",
         "fields": {"text": ["corrected text"], "quality": "golden"},
         "changed": {"text": [[], ["corrected text"]]}},
        {"sample_id": "E01_S01_A02_f1-3",
         "fields": {"quality": "ok"}},
    ])
    out = str(tmp_path / "gold")
    report = build_gold.build(__import__("argparse").Namespace(
        dataset=src, split="val", edits=str(edits), out=out,
        out_split="gold", dry_run=True))
    assert report["n_golden"] == 1
    assert report["n_edited"] == 2
    assert report["text_changes"] == 1
    assert not os.path.exists(out)  # dry-run writes nothing


def test_build_gold_applies_corrections(tmp_path):
    src = str(tmp_path / "src")
    _make_dataset(src)
    edits = str(tmp_path / "edits.jsonl")
    _write_edits(edits, [
        {"sample_id": "E01_S01_A01_f1-3",
         "fields": {"text": ["corrected text"], "label": 10, "quality": "golden", "note": "fix label"},
         "changed": {"text": [[], ["corrected text"]], "label": [0, 10]}},
        {"sample_id": "E01_S01_A02_f1-3",
         "fields": {"quality": "reject"}},
    ])
    out = str(tmp_path / "gold")
    report = build_gold.build(__import__("argparse").Namespace(
        dataset=src, split="val", edits=str(edits), out=out,
        out_split="gold", dry_run=False))
    assert report["n_golden"] == 1
    assert report["label_changes"] == 1
    assert report["text_changes"] == 1
    assert os.path.exists(os.path.join(out, "data", "E01_S01_A01_f1-3.pkl"))
    assert os.path.exists(os.path.join(out, "splits", "gold.json"))
    assert os.path.exists(os.path.join(out, "meta.json"))

    ds = load_dataset(out, mode="lazy")
    assert len(ds.gold) == 1
    s = ds.gold[0]
    assert s.label == 10
    assert s.text["captions"] == ["corrected text"]
    assert s.meta.get("golden") is True
    assert s.meta.get("curation_note") == "fix label"
    assert set(ds.modalities) == {"rgb", "wifi"}


def test_build_gold_keeps_original_label_when_unchanged(tmp_path):
    src = str(tmp_path / "src")
    _make_dataset(src)
    edits = str(tmp_path / "edits.jsonl")
    _write_edits(edits, [
        {"sample_id": "E01_S01_A01_f1-3", "fields": {"quality": "golden"}},
    ])
    out = str(tmp_path / "gold")
    build_gold.build(__import__("argparse").Namespace(
        dataset=src, split="val", edits=str(edits), out=out,
        out_split="gold", dry_run=False))
    ds = load_dataset(out, mode="lazy")
    assert ds.gold[0].label == 0
    assert ds.gold[0].text["captions"] == ["original caption for E01_S01_A01_f1-3"]