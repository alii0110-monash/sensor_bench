"""End-to-end test for run_dataset_quality on a synthetic toy dataset."""
import json
import pickle

import numpy as np
import pytest

from framework.dataset.sample import Sample, Modality


def _toy_dataset(tmp_path, n=80, dim=8, n_classes=4, seed=0):
    """Build a tiny synthetic dataset on disk mimicking v4 layout."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "toy_v0"
    data_dir = root / "data"
    splits_dir = root / "splits"
    data_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    samples = []
    ids = []
    for i in range(n):
        sid = f"S{i:03d}"
        ids.append(sid)
        cls = i % n_classes
        feats = rng.normal(size=(5, dim)).astype(np.float32)
        feats[:, 0] += cls * 2.0
        # mmwave: (T=5, max_points=64, 5 attributes) — sparse, ~22% nonzero.
        # Shape matches v4 real data so extract_mmwave_features accepts it.
        mmwave = rng.normal(size=(5, 64, 5)).astype(np.float32)
        mmwave[rng.random(size=mmwave.shape) < 0.78] = 0.0  # ~22% nonzero
        mmwave[:, :, 0] += cls * 1.5  # class signal in x-coord
        mods = {
            "rgb": Modality(data=feats, frame_indices=[0, 1, 2, 3, 4]),
            "depth": Modality(data=feats[:, :4], frame_indices=[0, 1, 2, 3, 4]),
            "lidar": Modality(data=feats[:, :3], frame_indices=[0, 1, 2, 3, 4]),
            "mmwave": Modality(data=mmwave, frame_indices=[0, 1, 2, 3, 4]),
            "wifi": Modality(data=feats[:, :6], frame_indices=[0, 1, 2, 3, 4]),
        }
        sample = Sample(id=sid, label=cls, modalities=mods)
        samples.append(sample)
    for s in samples:
        with open(data_dir / f"{s.id}.pkl", "wb") as f:
            pickle.dump(s.to_dict(), f)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    (splits_dir / "train.json").write_text(json.dumps(ids[:n_train]))
    (splits_dir / "val.json").write_text(json.dumps(ids[n_train:n_train + n_val]))
    (splits_dir / "test.json").write_text(json.dumps(ids[n_train + n_val:]))
    (root / "meta.json").write_text(json.dumps({
        "name": "toy", "version": "v0",
        "modalities": ["rgb", "depth", "lidar", "mmwave", "wifi"],
    }))
    return str(root)


def test_run_end_to_end(tmp_path):
    from scripts.run_dataset_quality import run
    root = _toy_dataset(tmp_path)
    out = tmp_path / "quality.json"
    run(root, str(out), num_classes=4, epochs=5, batch_size=16)
    rep = json.loads(out.read_text())
    assert "info" in rep and "compact" in rep and "clean" in rep
    assert 0.0 <= rep["quality"] <= 1.0
    assert rep["metadata"]["val_sample_count"] > 0


def test_run_rejects_test_split(tmp_path):
    from scripts.run_dataset_quality import run
    root = _toy_dataset(tmp_path)
    with pytest.raises(AssertionError):
        run(root, str(tmp_path / "q.json"), eval_split="test", num_classes=4)