from __future__ import annotations

import json
import os
import pickle

import numpy as np

from framework.dataset.sample import Modality, Sample

from curation.gui.core.dataset_service import (
    compute_health,
    dataset_name,
    find_quality_json,
    find_raw_root,
    list_dataset_roots,
    open_dataset,
    parse_sample_id,
    raw_depth_frames,
    split_ids,
)


def test_parse_sample_id():
    assert parse_sample_id("E04_S33_A14_f37-46") == {"ep": 4, "subject": "S33", "action": 14}
    assert parse_sample_id("E01_S01_A01_f1-7__aug0") == {"ep": 1, "subject": "S01", "action": 1}
    assert parse_sample_id("garbage") is None


def _fake_sample(sid, label):
    return Sample(
        id=sid,
        label=label,
        modalities={
            "rgb": Modality(data=np.random.rand(3, 17, 2).astype(np.float32),
                            frame_indices=[1, 2, 3]),
            "wifi": Modality(data=np.random.rand(3, 3, 114, 10).astype(np.float32),
                             frame_indices=[1, 2, 3]),
        },
        text={},
        meta={},
    )


def test_compute_health_distributions_and_anomalies():
    samples = [
        _fake_sample("E01_S01_A01_f1-3", 0),
        _fake_sample("E01_S02_A02_f1-3", 1),
        _fake_sample("E02_S33_A03_f1-3", 2),
    ]
    h = compute_health(iter(samples), max_samples=None)
    assert h["n_scanned"] == 3
    assert h["label_dist"] == {0: 1, 1: 1, 2: 1}
    assert h["subject_dist"] == {"S01": 1, "S02": 1, "S33": 1}
    assert h["frame_dist"]["rgb"][3] == 3
    assert h["anomalies"]["rgb"] == {}  # no anomalies


def test_compute_health_anomaly_detection():
    const = _fake_sample("E01_S01_A01_f1-3", 0)
    const.modalities["rgb"] = Modality(
        data=np.zeros((3, 17, 2), dtype=np.float32), frame_indices=[1, 2, 3])
    h = compute_health(iter([const]))
    assert h["anomalies"]["rgb"]["allzero"] == 1
    assert h["anomalies"]["rgb"]["constant"] == 1


def test_compute_health_max_samples():
    samples = [_fake_sample(f"E01_S01_A{i:02d}_f1-3", i) for i in range(10)]
    h = compute_health(iter(samples), max_samples=4)
    assert h["n_scanned"] == 4


def test_list_and_open(tmp_path):
    root = str(tmp_path / "v1")
    os.makedirs(os.path.join(root, "data"))
    os.makedirs(os.path.join(root, "splits"))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"name": "mmfi", "version": "v1"}, f)
    with open(os.path.join(root, "modalities.yaml"), "w") as f:
        f.write("modalities:\n- rgb\n")
    sid = "E01_S01_A01_f1-3"
    with open(os.path.join(root, "splits", "val.json"), "w") as f:
        json.dump([sid], f)
    sample = _fake_sample(sid, 0)
    with open(os.path.join(root, "data", f"{sid}.pkl"), "wb") as f:
        pickle.dump(sample.to_dict(), f)

    roots = list_dataset_roots(str(tmp_path))
    assert root in roots
    assert dataset_name(root) == "mmfi_v1"
    assert split_ids(root, "val") == [sid]
    ds = open_dataset(root, "val")
    assert len(ds.val) == 1
    with open(os.path.join(root, "splits", "bad.json"), "w") as f:
        json.dump([sid], f)
    assert split_ids(root, "bad") == [sid]


def test_open_dataset_invalid_root(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        open_dataset(str(tmp_path / "nope"))


def test_find_quality_json(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    (rdir / "quality_v1.json").write_text(json.dumps({"dataset": "datasets/mmfi/v1"}))
    (rdir / "quality_v2.json").write_text(json.dumps({"dataset": "datasets/mmfi/v2"}))
    assert find_quality_json("datasets/mmfi/v1", str(rdir)) == str(rdir / "quality_v1.json")
    assert find_quality_json("datasets/mmfi/nope", str(rdir)) is None


def test_find_raw_root_explicit_and_default():
    assert find_raw_root("/no/such/dir") is None
    # default candidates point at the real machine layout
    root = find_raw_root()
    assert root is None or os.path.isdir(root)


def _fake_raw_depth_root(tmp_path):
    """Build E04/S33/A01/depth/frame001-003.png (480x640 uint16) under tmp_path."""
    import cv2

    ddir = tmp_path / "E04" / "S33" / "A01" / "depth"
    ddir.mkdir(parents=True)
    for i in (1, 2, 3):
        img = np.full((480, 640), 3000 + i, dtype=np.uint16)
        cv2.imwrite(str(ddir / f"frame{i:03d}.png"), img)
    return str(tmp_path)


def test_raw_depth_frames_loads_native_resolution(tmp_path):
    raw_root = _fake_raw_depth_root(tmp_path)
    sample = _fake_sample("E04_S33_A01_f1-3", 0)  # label 0 -> A01
    sample.meta = {"env": "E04", "subject": "S33"}
    sample.modalities["depth"] = Modality(
        data=np.zeros((3, 1, 224, 224), dtype=np.float32),
        frame_indices=[1, 2, 3])
    raw = raw_depth_frames(sample, raw_root)
    assert raw is not None
    assert raw.shape == (3, 1, 480, 640)
    assert raw.dtype == np.float32
    # frame1 = 3001 * 0.001 = 3.001m
    assert abs(float(raw[0, 0, 240, 320]) - 3.001) < 1e-4


def test_raw_depth_frames_missing_returns_none(tmp_path):
    raw_root = _fake_raw_depth_root(tmp_path)
    sample = _fake_sample("E04_S33_A02_f1-3", 1)  # label 1 -> A02 (dir absent)
    sample.meta = {"env": "E04", "subject": "S33"}
    sample.modalities["depth"] = Modality(
        data=np.zeros((3, 1, 224, 224), dtype=np.float32), frame_indices=[1, 2, 3])
    assert raw_depth_frames(sample, raw_root) is None
    assert raw_depth_frames(sample, None) is None