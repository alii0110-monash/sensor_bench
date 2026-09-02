# tests/test_loader.py
import json, os, pickle
import numpy as np
from framework.dataset.loader import (LazySplit, available_memory_bytes,
                                      estimate_dataset_bytes, load_dataset,
                                      preflight_dataset)
from framework.dataset.sample import Sample, Modality

def _make_tiny(tmp_path, n=4, mods=("wifi",)):
    root = tmp_path / "tiny" / "v1"
    (root / "data").mkdir(parents=True)
    ids = []
    for i in range(n):
        mm = {m: Modality(data=np.zeros((2, 2, 2), dtype=np.float32),
                          frame_indices=[1, 2], sample_rate=10) for m in mods}
        s = Sample(id=f"s{i}", label=i % 3, modalities=mm)
        with open(root / "data" / f"{s.id}.pkl", "wb") as f:
            pickle.dump(s.to_dict(), f)
        ids.append(s.id)
    (root / "splits").mkdir(exist_ok=True)
    with open(root / "splits" / "train.json", "w") as f:
        json.dump(ids[:2], f)
    with open(root / "splits" / "test.json", "w") as f:
        json.dump(ids[2:], f)
    return root

def test_load_dataset(tmp_path):
    root = _make_tiny(tmp_path)
    ds = load_dataset(str(root))
    assert len(ds.train) == 2
    assert len(ds.test) == 2
    assert set(ds.splits.keys()) == {"train", "test"}
    assert ds.modalities == ["wifi"]
    s = ds.test[0]
    assert isinstance(s, Sample)
    assert set(s.available_modalities()) == {"wifi"}

def test_load_dataset_missing_split(tmp_path):
    root = _make_tiny(tmp_path)
    (root / "splits" / "val.json").write_text("[]")
    ds = load_dataset(str(root))
    assert ds.val == []

# ---- memory pre-flight + lazy loading ----

def test_estimate_dataset_bytes(tmp_path):
    root = _make_tiny(tmp_path, n=10)
    est = estimate_dataset_bytes(str(root))
    assert est > 0

def test_available_memory_bytes():
    assert available_memory_bytes() > 0

def test_preflight_dataset(tmp_path):
    root = _make_tiny(tmp_path, n=4)
    pre = preflight_dataset(str(root))
    assert pre["n_files"] == 4
    assert pre["estimated_bytes"] > 0
    assert pre["available_bytes"] > 0

def test_load_lazy_matches_eager(tmp_path):
    root = _make_tiny(tmp_path, n=6)
    eager = load_dataset(str(root), mode="eager")
    lazy = load_dataset(str(root), mode="lazy")
    assert len(eager.train) == len(lazy.train)
    assert len(eager.test) == len(lazy.test)
    assert eager.modalities == lazy.modalities
    assert [s.id for s in eager.train] == [s.id for s in lazy.train]
    for es, ls in zip(eager.train, lazy.train):
        assert es.id == ls.id and es.label == ls.label
        assert set(es.available_modalities()) == set(ls.available_modalities())

def test_lazy_split_slice_and_index(tmp_path):
    root = _make_tiny(tmp_path, n=6)
    lazy = load_dataset(str(root), mode="lazy")
    split = lazy.test
    assert len(split) == 4
    assert isinstance(split[0], Sample)
    assert split[0].id == "s2"
    batch = split[0:2]
    assert len(batch) == 2 and all(isinstance(s, Sample) for s in batch)
    assert split[-1].id == "s5"
    assert "s3" in split

def test_lazy_split_shared_cache_no_reload(tmp_path, monkeypatch):
    root = _make_tiny(tmp_path, n=4)
    lazy = load_dataset(str(root), mode="lazy")
    split = lazy.test
    reads = []
    import pickle as _pickle
    orig = _pickle.load
    def spy(f, **kw):
        reads.append(f.name)
        return orig(f, **kw)
    monkeypatch.setattr("framework.dataset.loader.pickle.load", spy)
    _ = split[0]; _ = split[0]
    assert len(reads) == 1  # second access served from cache

def test_eager_oversize_refuses(tmp_path, monkeypatch):
    root = _make_tiny(tmp_path, n=4)
    monkeypatch.setattr("framework.dataset.loader.available_memory_bytes", lambda: 1)
    with __import__("pytest").raises(MemoryError):
        load_dataset(str(root), mode="eager")
    ds = load_dataset(str(root))  # auto → lazy, no raise
    assert len(ds.train) == 2

# ---- v4 variant dedup (delta files referencing base) ----

def _make_variant_tiny(tmp_path, n=2):
    """Like _make_tiny but train variants stored as deltas (kind=variant)."""
    import numpy as np
    from framework.dataset.sample import Sample, Modality
    root = tmp_path / "tiny" / "v4"
    (root / "data").mkdir(parents=True)
    ids = []
    for i in range(n):
        mm = {m: Modality(data=np.zeros((2, 2, 2), dtype=np.float32),
                          frame_indices=[1, 2], sample_rate=10) for m in ("wifi", "rgb")}
        s = Sample(id=f"s{i}", label=i, modalities=mm)
        with open(root / "data" / f"{s.id}.pkl", "wb") as f:
            pickle.dump(s.to_dict(), f)
        ids.append(s.id)
        for k in range(2):
            vid = f"{s.id}__aug{k}"
            delta = {"kind": "variant", "id": vid, "base_id": s.id, "label": s.label,
                     "rgb": {"data": np.zeros((2, 2, 2), dtype=np.float32) + k,
                             "frame_indices": [1, 2], "sample_rate": 10},
                     "aug": k}
            with open(root / "data" / f"{vid}.pkl", "wb") as f:
                pickle.dump(delta, f)
            ids.append(vid)
    (root / "splits").mkdir(exist_ok=True)
    with open(root / "splits" / "train.json", "w") as f:
        json.dump(ids, f)
    with open(root / "splits" / "test.json", "w") as f:
        json.dump([], f)
    return root

def test_variant_delta_resolved_lazy(tmp_path):
    root = _make_variant_tiny(tmp_path)
    ds = load_dataset(str(root), mode="lazy")
    assert len(ds.train) == 6
    v = ds.train[1]
    assert v.id == "s0__aug0"
    assert v.label == 0
    assert set(v.available_modalities()) == {"wifi", "rgb"}
    assert float(v.modalities["rgb"].data[0, 0, 0]) == 0.0  # aug0 delta
    v2 = ds.train[2]
    assert v2.id == "s0__aug1"
    assert float(v2.modalities["rgb"].data[0, 0, 0]) == 1.0  # aug1 delta
    assert ds.modalities == ["wifi", "rgb"]

def test_variant_delta_resolved_eager(tmp_path):
    root = _make_variant_tiny(tmp_path)
    ds = load_dataset(str(root), mode="eager")
    assert len(ds.train) == 6
    assert {s.id for s in ds.train} == {f"s{i}" for i in range(2)} | {f"s{i}__aug{k}" for i in range(2) for k in range(2)}
    v = [s for s in ds.train if s.id == "s1__aug1"][0]
    assert v.label == 1
    assert float(v.modalities["rgb"].data[0, 0, 0]) == 1.0

def test_variant_delta_lazy_matches_eager(tmp_path):
    root = _make_variant_tiny(tmp_path)
    eager = load_dataset(str(root), mode="eager")
    lazy = load_dataset(str(root), mode="lazy")
    assert len(eager.train) == len(lazy.train)
    for es, ls in zip(eager.train, lazy.train):
        assert es.id == ls.id
        for m in es.available_modalities():
            assert float(es.modalities[m].data[0, 0, 0]) == float(ls.modalities[m].data[0, 0, 0])
