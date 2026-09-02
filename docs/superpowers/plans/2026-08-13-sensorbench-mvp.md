# SensorBench MVP Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development (if subagents available) or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SensorBench framework MVP — a data/model-decoupled benchmark framework that ingests MMFi (4 modalities), trains pluggable sensor-fusion models, evaluates missing-modality robustness, and runs one data-improvement flywheel loop (v1 → v2).

**Architecture:** Dataset (canonical sample format + splits + versioning) is fully decoupled from models (`SensorModel` interface). Evaluation is a fixed protocol (15 modality profiles, seeds 0-2, Robustness Score = mean accuracy over all profiles). The MVP runs on MMFi `wifi/mmwave/lidar/depth` using the cross-subject (cs) split.

**Tech Stack:** Python 3.12 (reuse `/home/li/projects/holollm/.venv`, which has torch 2.9.1+cu128, torchvision, scipy, opencv, numpy), pytest, PyYAML, pickle per-sample storage.

---

## Environment & Data Facts (for implementer)

- Python: `/home/li/projects/holollm/.venv/bin/python` (has all deps; do NOT create a new venv for MVP)
- MMFi raw root: `/home/li/datasets/MMFi_dataset/data/MMFi_Dataset` layout `E01..E04/S01..S40/A01..A27/{depth,lidar,mmwave,wifi-csi,rgb,infra1,infra2}`
- Annotations: `/home/li/datasets/holollm_annotations/textual_annotations/mmfi/mmficap/mmfi_{train,test}_cs_full.json` (train 11657 / test 4791, subjects disjoint: train 28 subs, test 12 subs)
- Action label = `int(path.split('/')[-1][1:]) - 1` (A01→0 ... A27→26)
- Raw per-frame formats (verified):
  - wifi: `wifi-csi/frameNNN.mat` → `loadmat(...)['CSIamp']` shape `(3,114,10)` float64
  - depth: `depth/frameNNN.png` → `cv2.imread(IMREAD_UNCHANGED)` `(480,640)` uint16, ×0.001 → meters
  - lidar: `lidar/frameNNN.bin` → `np.frombuffer(open(f,'rb').read(), np.float64).reshape(-1,3)`, N≤1536
  - mmwave: `mmwave/frameNNN.bin` → reshape(-1,5), N≤64
- Annotation sample: `{"video_path": "./datasets/MMFi/E02/S19/A03", "start_index": 65, "end_index": 101, "conversations": [...]}`
- Output dataset root: `/home/li/projects/sensorbench/datasets/mmfi/v1` (NOT committed to git)
- GPU: RTX 5060 Ti 16GB single — models are ~10M params, train in minutes

---

## File Structure

```
sensorbench/
├── requirements.txt
├── pyproject.toml                 # package metadata + pytest config
├── .gitignore                     # datasets/, *.pkl, __pycache__, checkpoints/
├── framework/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── sample.py              # Sample dataclass + contract validation
│   │   ├── loader.py              # load_dataset(root) → Dataset with splits
│   │   └── splits.py              # build splits from MMFi annotations (subject-stratified)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # SensorModel protocol + TrainConfig
│   │   ├── encoders.py            # shared per-modality token encoders
│   │   ├── token_fusion.py        # main model (统一 token 融合 + 缺模态)
│   │   └── late_fusion.py         # baseline (late fusion)
│   └── harness/
│       ├── __init__.py
│       ├── protocol.py            # build 15-profile protocol.json
│       ├── evaluate.py            # run model across profiles
│       └── leaderboard.py         # leaderboard + degradation matrix JSON/report
├── curation/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── mmfi.py                # MMFi raw → canonical Sample pkl
│   │   └── readers.py             # per-modality raw file readers (tested)
│   ├── clean/
│   │   ├── __init__.py
│   │   └── align_check.py         # frame_indices consistency verification
│   └── version/
│       ├── __init__.py
│       └── version.py             # meta.json writer, changelog
├── scripts/
│   ├── ingest_mmfi.py             # CLI: raw → datasets/mmfi/v1
│   ├── train.py                   # CLI: train a SensorModel, multi-seed
│   └── run_eval.py                # CLI: protocol → leaderboard JSON + report
├── tests/
│   ├── test_sample.py
│   ├── test_loader.py
│   ├── test_splits.py
│   ├── test_readers.py            # reads REAL MMFi files (smoke)
│   ├── test_ingest.py             # tiny synthetic + --limit real ingest
│   ├── test_align_check.py
│   ├── test_protocol.py
│   ├── test_models.py             # toy-data forward + missing-modality predict
│   └── test_harness.py
└── docs/superpowers/plans/2026-08-13-sensorbench-mvp.md   # this plan
```

---

### Task 1: Project skeleton, deps, git hygiene

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `framework/__init__.py`, `framework/dataset/__init__.py`, `framework/models/__init__.py`, `framework/harness/__init__.py`, `curation/__init__.py`, `curation/ingest/__init__.py`, `curation/clean/__init__.py`, `curation/version/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
datasets/
checkpoints/
*.pkl
.pytest_cache/
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[project]
name = "sensorbench"
version = "0.1.0"
requires-python = ">=3.12"
```

- [ ] **Step 3: Write `requirements.txt`** (all present in holollm venv; documented for reproducibility)

```
numpy
scipy
opencv-python
pyyaml
pytest
torch>=2.0
torchvision>=0.15
```

- [ ] **Step 4: Create the empty `__init__.py` files listed above**

Run: `touch framework/__init__.py framework/dataset/__init__.py framework/models/__init__.py framework/harness/__init__.py curation/__init__.py curation/ingest/__init__.py curation/clean/__init__.py curation/version/__init__.py tests/__init__.py`

- [ ] **Step 5: Verify imports work**

Run: `/home/li/projects/holollm/.venv/bin/python -c "import framework.dataset, framework.models, framework.harness, curation.ingest, pytest; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "chore: project skeleton, deps, gitignore"
```

---

### Task 2: Sample dataclass + contract validation

**Files:**
- Create: `framework/dataset/sample.py`
- Test: `tests/test_sample.py`

The `Sample` is the single contract between dataset and models. All modality arrays are stored per-frame at ingest time; shape convention `data.shape[0] == len(frame_indices)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sample.py
import numpy as np
import pytest
from framework.dataset.sample import Sample, Modality

def test_sample_roundtrip():
    mod = Modality(data=np.zeros((5, 3, 114, 10), dtype=np.float32),
                   frame_indices=[65, 74, 83, 92, 101], sample_rate=1000)
    s = Sample(id="x", label=2, modalities={"wifi": mod})
    assert s.modalities["wifi"].shape[0] == 5
    assert s.label == 2

def test_sample_rejects_mismatched_frames():
    mod = Modality(data=np.zeros((5, 3, 114, 10), dtype=np.float32),
                   frame_indices=[65, 66, 67], sample_rate=20)
    with pytest.raises(ValueError):
        Sample(id="x", label=0, modalities={"lidar": mod})

def test_sample_rejects_empty_modalities():
    with pytest.raises(ValueError):
        Sample(id="x", label=0, modalities={})

def test_sample_requires_label_in_range():
    mod = Modality(data=np.zeros((5, 3, 1, 1), dtype=np.float32),
                   frame_indices=[1,2,3,4,5], sample_rate=1)
    with pytest.raises(ValueError):
        Sample(id="x", label=-1, modalities={"depth": mod})
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_sample.py -v`
Expected: FAIL — `ModuleNotFoundError: framework.dataset.sample`

- [ ] **Step 3: Implement `framework/dataset/sample.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Modality:
    """One sensor stream of one sample. data.shape[0] == len(frame_indices)."""
    data: np.ndarray
    frame_indices: List[int]
    sample_rate: int = 0
    name: str = ""

    @property
    def shape(self):
        return list(self.data.shape)

    def __post_init__(self):
        self.frame_indices = [int(i) for i in self.frame_indices]
        if len(self.frame_indices) != self.data.shape[0]:
            raise ValueError(
                f"frame_indices ({len(self.frame_indices)}) must match data.shape[0] "
                f"({self.data.shape[0]})")


@dataclass
class Sample:
    """Canonical sample contract. Models consume this dict-like object."""
    id: str
    label: int
    modalities: Dict[str, Modality]
    text: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.modalities:
            raise ValueError("sample must have at least one modality")
        if not (0 <= self.label < 1000):
            raise ValueError(f"label out of range: {self.label}")
        for name, mod in self.modalities.items():
            mod.name = name

    def available_modalities(self) -> List[str]:
        return list(self.modalities.keys())

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "modalities": {
                name: {"data": m.data, "frame_indices": m.frame_indices,
                       "sample_rate": m.sample_rate, "name": name}
                for name, m in self.modalities.items()
            },
            "text": self.text,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d):
        mods = {
            name: Modality(data=np.asarray(mm["data"]),
                           frame_indices=mm["frame_indices"],
                           sample_rate=mm.get("sample_rate", 0), name=name)
            for name, mm in d["modalities"].items()
        }
        return cls(id=d["id"], label=d["label"], modalities=mods,
                   text=d.get("text", {}), meta=d.get("meta", {}))
```

- [ ] **Step 4: Run tests, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_sample.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: Sample/Modality contract with validation"
```

---

### Task 3: Dataset loader

**Files:**
- Create: `framework/dataset/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loader.py
import json, os, pickle
import numpy as np
from framework.dataset.loader import load_dataset
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
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_loader.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `framework/dataset/loader.py`**

```python
from __future__ import annotations
import json, os, pickle
from typing import Dict, List
from .sample import Sample

_SPLIT_NAMES = ["train", "val", "test"]


class Dataset:
    def __init__(self, root: str, splits: Dict[str, List[Sample]], modalities: List[str]):
        self.root = root
        self.splits: Dict[str, List[Sample]] = splits
        self.modalities = modalities

    def __getattr__(self, name):
        if name in _SPLIT_NAMES:
            return self.splits.get(name, [])
        raise AttributeError(name)

    @property
    def meta(self) -> dict:
        p = os.path.join(self.root, "meta.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def __repr__(self):
        n = {k: len(v) for k, v in self.splits.items()}
        return f"Dataset(root={self.root}, splits={n}, modalities={self.modalities})"


def _read_sample(p: str) -> Sample:
    with open(p, "rb") as f:
        return Sample.from_dict(pickle.load(f))


def load_dataset(root: str) -> Dataset:
    data_dir = os.path.join(root, "data")
    files = sorted(os.listdir(data_dir))
    cache: Dict[str, Sample] = {}
    for fn in files:
        if fn.endswith(".pkl"):
            s = _read_sample(os.path.join(data_dir, fn))
            cache[s.id] = s

    splits: Dict[str, List[Sample]] = {}
    for name in _SPLIT_NAMES:
        p = os.path.join(root, "splits", f"{name}.json")
        if os.path.exists(p):
            ids = json.load(open(p))
            splits[name] = [cache[i] for i in ids if i in cache]

    modalities = []
    for s in cache.values():
        for m in s.modalities:
            if m not in modalities:
                modalities.append(m)
    return Dataset(root, splits, modalities)
```

- [ ] **Step 4: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_loader.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: dataset loader with splits"
```

---

### Task 4: Split builder (subject-stratified, reuses MMFi cs annotations)

**Files:**
- Create: `framework/dataset/splits.py`
- Test: `tests/test_splits.py`

Splits come from MMFi's cross-subject annotation files. Train (28 subjects) is further carved into train/val by subject (last 5 subjects → val).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
import json
from framework.dataset.splits import split_annotations, _subject_of

def test_subject_of():
    assert _subject_of("./datasets/MMFi/E02/S19/A03") == "S19"

def test_split_annotations_disjoint_subjects(tmp_path):
    anns = []
    # 3 subjects x 2 actions
    for sub in ["S01", "S02", "S03"]:
        for act in ["A01", "A02"]:
            anns.append({"video_path": f"./datasets/MMFi/E01/{sub}/{act}",
                         "start_index": 0, "end_index": 9})
    train, val = split_annotations(anns, val_subjects=["S03"])
    train_subs = {_subject_of(a["video_path"]) for a in train}
    val_subs = {_subject_of(a["video_path"]) for a in val}
    assert val_subs == {"S03"}
    assert train_subs == {"S01", "S02"}
    assert train_subs.isdisjoint(val_subs)

def test_split_annotations_id_uses_frame_window(tmp_path):
    anns = [{"video_path": "./datasets/MMFi/E01/S01/A05", "start_index": 3, "end_index": 12}]
    train, _ = split_annotations(anns, val_subjects=[])
    assert train[0]["sample_id"] == "E01_S01_A05_f3-12"
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_splits.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `framework/dataset/splits.py`**

```python
from __future__ import annotations
import os
from typing import Dict, List, Tuple


def _subject_of(video_path: str) -> str:
    return video_path.split("/")[-2]


def _env_of(video_path: str) -> str:
    return video_path.split("/")[-3]


def _sample_id(video_path: str, start: int, end: int) -> str:
    return f"{video_path.split('/')[-3]}_{_subject_of(video_path)}_{video_path.split('/')[-1]}_f{start}-{end}"


def split_annotations(anns: List[dict], val_subjects: List[str]) -> Tuple[List[dict], List[dict]]:
    """Split annotation list into (train, val). val is defined by val_subjects.
    Adds 'sample_id' to each annotation. Returns copies, does not mutate input."""
    train, val = [], []
    for a in anns:
        item = dict(a)
        item["sample_id"] = _sample_id(a["video_path"], a["start_index"], a["end_index"])
        if _subject_of(a["video_path"]) in set(val_subjects):
            val.append(item)
        else:
            train.append(item)
    return train, val


def build_val_subjects(train_anns: List[dict], n_val_subjects: int = 5) -> List[str]:
    subs = sorted({_subject_of(a["video_path"]) for a in train_anns})
    return subs[-n_val_subjects:]
```

- [ ] **Step 4: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_splits.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: subject-stratified split builder"
```

---

### Task 5: Per-modality raw readers

**Files:**
- Create: `curation/ingest/readers.py`
- Test: `tests/test_readers.py` (uses REAL MMFi files, smoke)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_readers.py
import glob
import numpy as np
import pytest
from curation.ingest import readers

RAW = "/home/li/datasets/MMFi_dataset/data/MMFi_Dataset/E01/S01/A01"


@pytest.mark.skipif(len(glob.glob(f"{RAW}/wifi-csi/*.mat")) == 0, reason="MMFi data not present")
def test_read_wifi():
    f = sorted(glob.glob(f"{RAW}/wifi-csi/*.mat"))[0]
    x = readers.read_wifi_frame(f)
    assert x.shape == (3, 114, 10)
    assert np.isfinite(x).all()
    assert x.dtype == np.float32


@pytest.mark.skipif(len(glob.glob(f"{RAW}/lidar/*.bin")) == 0, reason="MMFi data not present")
def test_read_lidar():
    f = sorted(glob.glob(f"{RAW}/lidar/*.bin"))[0]
    x = readers.read_lidar_frame(f)
    assert x.shape[1] == 3 and x.shape[0] <= 1536


@pytest.mark.skipif(len(glob.glob(f"{RAW}/mmwave/*.bin")) == 0, reason="MMFi data not present")
def test_read_mmwave():
    f = sorted(glob.glob(f"{RAW}/mmwave/*.bin"))[0]
    x = readers.read_mmwave_frame(f)
    assert x.shape[1] == 5 and x.shape[0] <= 64


@pytest.mark.skipif(len(glob.glob(f"{RAW}/depth/*.png")) == 0, reason="MMFi data not present")
def test_read_depth():
    f = sorted(glob.glob(f"{RAW}/depth/*.png"))[0]
    x = readers.read_depth_frame(f)
    assert x.shape == (224, 224)
    assert x.dtype == np.float32
    assert x.max() <= 20.0  # meters, sanity
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_readers.py -v`
Expected: FAIL — module not found (or skip if data missing)

- [ ] **Step 3: Implement `curation/ingest/readers.py`**

```python
from __future__ import annotations
import cv2
import numpy as np
import scipy.io as scio

LIDAR_MAX = 1536
MMWAVE_MAX = 64


def read_wifi_frame(path: str) -> np.ndarray:
    """Returns (3, 114, 10) float32, finite, min-max normalized per frame."""
    f = scio.loadmat(path)["CSIamp"].astype(np.float64)
    f[np.isinf(f)] = np.nan
    for i in range(f.shape[-1]):
        col = f[:, :, i]
        nans = np.isnan(col)
        if nans.all():
            f[:, :, i] = 0.0
        elif nans.any():
            col[nans] = np.nanmean(col)
    mn, mx = float(f.min()), float(f.max())
    f = (f - mn) / (mx - mn + 1e-9)
    return f.astype(np.float32)


def read_lidar_frame(path: str) -> np.ndarray:
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
    pts = raw.reshape(-1, 3).astype(np.float32)
    n = min(pts.shape[0], LIDAR_MAX)
    pts = pts[:n]
    if n < LIDAR_MAX:
        pts = np.pad(pts, ((0, LIDAR_MAX - n), (0, 0)))
    return pts  # (1536, 3)


def read_mmwave_frame(path: str) -> np.ndarray:
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
    pts = raw.copy().reshape(-1, 5).astype(np.float32)
    n = min(pts.shape[0], MMWAVE_MAX)
    pts = pts[:n]
    if n < MMWAVE_MAX:
        pts = np.pad(pts, ((0, MMWAVE_MAX - n), (0, 0)))
    return pts  # (64, 5)


def read_depth_frame(path: str) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    d = cv2.resize(d, (224, 224), interpolation=cv2.INTER_AREA)
    return d  # (224, 224) meters
```

- [ ] **Step 4: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_readers.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: MMFi per-modality raw readers"
```

---

### Task 6: Ingest pipeline (annotation → canonical Sample pkl)

**Files:**
- Create: `curation/ingest/mmfi.py`, `curation/version/version.py`, `scripts/ingest_mmfi.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
import json, os, tempfile
import numpy as np
from curation.ingest.mmfi import ingest_annotation
from framework.dataset.loader import load_dataset


def _fake_ann(tmp_path, raw_root, sub, act, start, end, sample_id):
    (raw_root / sub / act / "wifi-csi").mkdir(parents=True)
    (raw_root / sub / act / "depth").mkdir(parents=True)
    for i in range(start, end + 1):
        # wifi .mat is the only hard one; skip actually writing real mat, test with --limit skip
        pass
    return {"video_path": f"./datasets/MMFi/E01/{sub}/{act}",
            "start_index": start, "end_index": end, "sample_id": sample_id}
```

- [ ] **Step 2: Implement `curation/ingest/mmfi.py`** (ingest is data-heavy; the smoke test calls it with `write=False` to validate the sampling logic only)

```python
from __future__ import annotations
import os
import pickle
from typing import Dict, List
import numpy as np

from . import readers
from framework.dataset.sample import Sample, Modality


def sample_frames(start: int, end: int, n: int = 5) -> List[int]:
    """Deterministic uniform frame sampling across [start, end] (reproducible)."""
    if end - start + 1 <= n:
        idx = list(range(start, end + 1))
        while len(idx) < n:
            idx.append(idx[-1])
        return idx
    return sorted(np.unique(np.linspace(start, end, n, dtype=int)).tolist())


def frame_paths(act_dir: str, modal: str, frames: List[int]) -> List[str]:
    exts = {"wifi-csi": ".mat", "lidar": ".bin", "mmwave": ".bin", "depth": ".png"}
    return [os.path.join(act_dir, modal, f"frame{i:03d}{exts[modal]}") for i in frames]


def _read_frames(reader, paths: List[str], shape_tail) -> np.ndarray:
    arrs = [reader(p) for p in paths]
    stacked = np.stack(arrs, axis=0)          # (T, ...)
    assert stacked.shape[1:] == shape_tail, (stacked.shape, shape_tail)
    return stacked


import re


def rel_video_path(video_path: str) -> str:
    """Map annotation video_path (./datasets/MMFi/E02/S19/A03) to the raw-root
    relative path (E02/S19/A03). Robust to any leading prefix."""
    parts = video_path.replace("./", "").split("/")
    for i, p in enumerate(parts):
        if re.match(r"^E\d+$", p):
            return "/".join(parts[i:])
    raise ValueError(f"cannot parse video_path: {video_path}")


def annotation_to_sample(ann: dict, raw_root: str, action_labels: Dict[str, int]) -> Sample:
    rel = rel_video_path(ann["video_path"])
    act_dir = os.path.join(raw_root, rel)
    frames = sample_frames(ann["start_index"], ann["end_index"])
    sid = ann["sample_id"]
    action = rel.split("/")[-1]
    label = action_labels[action]

    wifi = _read_frames(readers.read_wifi_frame, frame_paths(act_dir, "wifi-csi", frames), (3, 114, 10))
    depth = _read_frames(readers.read_depth_frame, frame_paths(act_dir, "depth", frames), (224, 224))
    lidar = _read_frames(readers.read_lidar_frame, frame_paths(act_dir, "lidar", frames), (1536, 3))
    mmwave = _read_frames(readers.read_mmwave_frame, frame_paths(act_dir, "mmwave", frames), (64, 5))

    mods = {
        "wifi":   Modality(data=wifi,   frame_indices=frames, sample_rate=1000),
        "depth":  Modality(data=depth,  frame_indices=frames, sample_rate=20),
        "lidar":  Modality(data=lidar,  frame_indices=frames, sample_rate=20),
        "mmwave": Modality(data=mmwave, frame_indices=frames, sample_rate=20),
    }
    return Sample(
        id=sid, label=label, modalities=mods,
        text={"captions": [c["value"] for c in ann.get("conversations", []) if c["from"] == "gpt"]},
        meta={"subject": rel.split("/")[-2], "env": rel.split("/")[-3],
              "source": "mmfi", "start_index": ann["start_index"], "end_index": ann["end_index"]},
    )


def write_sample(root: str, sample: Sample) -> None:
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f"{sample.id}.pkl"), "wb") as f:
        pickle.dump(sample.to_dict(), f)


def action_labels() -> Dict[str, int]:
    return {f"A{i:02d}": i - 1 for i in range(1, 28)}
```

- [ ] **Step 3: Implement `curation/version/version.py`**

```python
from __future__ import annotations
import json
import os
from typing import List


def write_meta(root: str, name: str, version: str, changelog: List[str],
               n_samples: int, n_modalities: int, source: dict,
               license: str = "unknown", collection_protocol: dict = None) -> None:
    os.makedirs(root, exist_ok=True)
    meta = {
        "name": name, "version": version,
        "changelog": changelog,
        "n_samples": n_samples, "n_modalities": n_modalities,
        "source": source, "license": license,
        "collection_protocol": collection_protocol or {},
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: Implement `scripts/ingest_mmfi.py`** (the full ingest driver)

```python
#!/usr/bin/env python
import argparse, json, os, time
from curation.ingest.mmfi import annotation_to_sample, action_labels, write_sample
from framework.dataset.splits import split_annotations, build_val_subjects
from curation.version.version import write_meta
import yaml


def write_modalities(root: str, modalities: list) -> None:
    with open(os.path.join(root, "modalities.yaml"), "w") as f:
        yaml.safe_dump({"modalities": modalities, "note": "modalities derived from samples; list is authoritative registry"}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-train", required=True)
    ap.add_argument("--annotations-test", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only ingest first N (smoke)")
    args = ap.parse_args()

    train = json.load(open(args.annotations_train))
    test = json.load(open(args.annotations_test))
    if args.limit:
        train = train[:args.limit]
        test = test[:args.limit]

    val_subs = build_val_subjects(train, n_val_subjects=5)
    train_a, val_a = split_annotations(train, val_subjects=val_subs)
    test_a, _ = split_annotations(test, val_subjects=[])
    labels = action_labels()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    for name, anns in [("train", train_a), ("val", val_a), ("test", test_a)]:
        n = 0
        for ann in anns:
            try:
                s = annotation_to_sample(ann, args.raw_root, labels)
                write_sample(args.out, s)
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {ann['sample_id']}: {e}")
        os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)
        with open(os.path.join(args.out, "splits", f"{name}.json"), "w") as f:
            json.dump([ann["sample_id"] for ann in anns], f)
        print(f"{name}: {n} samples ({time.time()-t0:.0f}s)")

    write_meta(args.out, name="mmfi", version="v1",
               changelog=["initial ingest of wifi/mmwave/lidar/depth, cs split",
                          "known simplifications: splits are plain id lists (subject/env are in sample.meta)"],
               n_samples=len(train_a) + len(val_a) + len(test_a),
               n_modalities=4, source={"dataset": "MMFi", "split": "cs"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"envs": "E01-E04", "subjects": 40, "actions": 27,
                                    "frames_per_sample": 5, "sample_frames": "deterministic uniform"})
    write_modalities(args.out, ["wifi", "depth", "lidar", "mmwave"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write smoke test `tests/test_ingest.py` (replaces Step 1 stub)**

```python
# tests/test_ingest.py
import numpy as np
from curation.ingest.mmfi import sample_frames, action_labels, annotation_to_sample


def test_sample_frames_short_window_pads():
    assert sample_frames(1, 3) == [1, 2, 3, 3, 3]

def test_sample_frames_long_window_returns_5():
    idx = sample_frames(1, 40)
    assert len(idx) == 5 and idx[0] == 1 and idx[-1] == 40

def test_sample_frames_reproducible():
    assert sample_frames(5, 30) == sample_frames(5, 30)

def test_action_labels():
    assert action_labels()["A01"] == 0 and action_labels()["A27"] == 26
```

- [ ] **Step 6: Run tests, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 7: Smoke-run full CLI with --limit 2 (validates the real file I/O path end-to-end)**

Run:
```bash
cd /home/li/projects/sensorbench && mkdir -p /tmp/sensorbench_smoke && \
/home/li/projects/holollm/.venv/bin/python scripts/ingest_mmfi.py \
  --annotations-train /home/li/datasets/holollm_annotations/textual_annotations/mmfi/mmficap/mmfi_train_cs_full.json \
  --annotations-test /home/li/datasets/holollm_annotations/textual_annotations/mmfi/mmficap/mmfi_test_cs_full.json \
  --raw-root /home/li/datasets/MMFi_dataset/data/MMFi_Dataset \
  --out /tmp/sensorbench_smoke --limit 30
```
Expected: no tracebacks (optional `[warn]` lines OK). First 30 train annotations span several subjects → some go to train, last-5 subjects to val, so `train: ~N ... val: ~M ... test: 30` (N≈16, M≈14). Real signal: `ls /tmp/sensorbench_smoke/data | wc -l` → 60.

- [ ] **Step 8: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: MMFi ingest pipeline with versioning + CLI"
```

---

### Task 7: Align check (clean step)

**Files:**
- Create: `curation/clean/align_check.py`
- Test: `tests/test_align_check.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_align_check.py
import numpy as np
from curation.clean.align_check import verify_alignment
from framework.dataset.sample import Sample, Modality


def test_alignment_ok():
    mods = {"wifi": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 5]),
            "depth": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 5])}
    issues = verify_alignment(Sample(id="x", label=0, modalities=mods))
    assert issues == []

def test_alignment_mismatch():
    mods = {"wifi": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 5]),
            "lidar": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 6])}
    issues = verify_alignment(Sample(id="x", label=0, modalities=mods))
    assert len(issues) == 1
    assert issues[0]["modality"] == "lidar"
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_align_check.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `curation/clean/align_check.py`**

```python
from __future__ import annotations
from typing import List
from framework.dataset.sample import Sample


def verify_alignment(sample: Sample) -> List[dict]:
    """Returns a list of issues; empty list means aligned.
    All modalities of a sample must reference the same frame_indices window."""
    if not sample.modalities:
        return []
    ref = None
    issues = []
    for name, mod in sample.modalities.items():
        if ref is None:
            ref = list(mod.frame_indices)
        elif list(mod.frame_indices) != ref:
            issues.append({"id": sample.id, "modality": name,
                           "expected": ref, "got": list(mod.frame_indices)})
    return issues
```

- [ ] **Step 4: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_align_check.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: frame alignment verification"
```

---

### Task 8: SensorModel protocol + shared encoders

**Files:**
- Create: `framework/models/base.py`, `framework/models/encoders.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import numpy as np
from framework.models.base import SensorModel, TrainConfig


class Dummy(SensorModel):
    name = "dummy"
    def fit(self, train, val, cfg): pass
    def predict(self, sample, available):
        return {0: 0.5, 1: 0.5}

def test_protocol_interface():
    m = Dummy()
    assert callable(m.fit) and callable(m.predict)
    assert m.name == "dummy"
    probs = m.predict(None, ["wifi"])
    assert abs(sum(probs.values()) - 1.0) < 1e-6

def test_train_config_defaults():
    c = TrainConfig(epochs=10)
    assert c.epochs == 10 and c.lr > 0 and c.seed is not None
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `framework/models/base.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: Optional[int] = 0
    device: str = "cuda"
    out_dir: str = "checkpoints"
    modality_dropout_p: float = 0.25
    eval_steps: int = 100
    patience: int = 5


class SensorModel:
    """The ONLY contract between framework and any model implementation.
    Models are trained on a Dataset, predict per-sample given an `available`
    modality list. Missing-modality behavior is entirely the model's concern."""

    name: str = "sensor_model"

    def fit(self, train, val, cfg: TrainConfig) -> None:
        raise NotImplementedError

    def predict(self, sample, available: List[str]) -> Dict[int, float]:
        """Returns {class_id: prob}. available = subset of dataset.modalities."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "SensorModel":
        raise NotImplementedError
```

- [ ] **Step 4: Implement `framework/models/encoders.py`** (shared token encoders, dim=256, 16 tokens/modality)

```python
from __future__ import annotations
import torch
import torch.nn as nn

D = 256
N_TOK = 16


class WifiEncoder(nn.Module):
    """(B,5,3,114,10) -> (B,16,D)"""
    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x).view(B, T, N_TOK, -1).mean(dim=1)

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.proj = nn.Linear(64, D)


class DepthEncoder(nn.Module):
    """(B,5,1,224,224) -> (B,16,D)"""
    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])
        x = self.conv(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x).view(B, T, N_TOK, -1).mean(dim=1)

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.proj = nn.Linear(64, D)


class PointEncoder(nn.Module):
    """(B,5,P,C) -> (B,16,D). lidar: C=3, mmwave: C=5."""
    def forward(self, x):
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:]).transpose(1, 2)  # (BT,C,P)
        x = self.mlp(x)
        x = torch.nn.functional.adaptive_avg_pool1d(x, N_TOK)  # (BT,C,16)
        return self.proj(x.transpose(1, 2)).view(B, T, N_TOK, -1).mean(dim=1)

    def __init__(self, in_c: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.ReLU())
        self.proj = nn.Linear(64, D)
```

- [ ] **Step 5: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_models.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: SensorModel protocol + shared token encoders"
```

---

### Task 9: token_fusion model (main) with modality dropout

**Files:**
- Create: `framework/models/token_fusion.py`
- Modify: `tests/test_models.py` (add model tests)

- [ ] **Step 1: Add the failing test**

```python
# tests/test_models.py (append)
import torch
from framework.models.token_fusion import TokenFusionModel
from framework.dataset.sample import Sample, Modality
import numpy as np


def _toy_sample():
    mods = {
        "wifi": Modality(np.zeros((2, 3, 114, 10), dtype=np.float32), [1, 2], 1000),
        "depth": Modality(np.zeros((2, 1, 224, 224), dtype=np.float32), [1, 2], 20),
        "lidar": Modality(np.zeros((2, 1536, 3), dtype=np.float32), [1, 2], 20),
        "mmwave": Modality(np.zeros((2, 64, 5), dtype=np.float32), [1, 2], 20),
    }
    return Sample(id="toy", label=3, modalities=mods)


def test_token_fusion_full_modalities():
    m = TokenFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi", "depth", "lidar", "mmwave"])
    assert len(probs) == 27
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_token_fusion_missing_modality():
    m = TokenFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi"])  # 3 modalities missing
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_token_fusion_dropout_batch_trains():
    m = TokenFusionModel(num_classes=27)
    s = _toy_sample()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    logits = m(batch, avail={"wifi": True, "depth": True, "lidar": True, "mmwave": False})
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `framework/models/token_fusion.py`**

```python
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn

from .base import SensorModel, TrainConfig
from .encoders import D, N_TOK, WifiEncoder, DepthEncoder, PointEncoder

MODALITIES = ["wifi", "depth", "lidar", "mmwave"]
MODALITY_IN_C = {"lidar": 3, "mmwave": 5}


class TokenFusionModel(nn.Module, SensorModel):
    """Unified token fusion: per-modality encoder -> 16 tokens each, shared
    transformer, mean-pool, classification head. Missing modality = learned
    [MISSING] embedding + masked in attention. Trained with modality dropout."""

    name = "token_fusion"

    def __init__(self, num_classes: int = 27, d: int = D, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.encoders = nn.ModuleDict({
            "wifi": WifiEncoder(), "depth": DepthEncoder(),
            "lidar": PointEncoder(3), "mmwave": PointEncoder(5)})
        self.missing = nn.ParameterDict({
            m: nn.Parameter(torch.randn(N_TOK, d) * 0.02) for m in MODALITIES})
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=4 * d, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.1)
        self.fusion = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, num_classes)

    def forward(self, mods: Dict[str, torch.Tensor], avail: Dict[str, bool]) -> torch.Tensor:
        B = next(iter(mods.values())).shape[0]
        toks, masks = [], []
        for m in MODALITIES:
            if avail.get(m):
                toks.append(self.encoders[m](mods[m]))
                masks += [1] * N_TOK
            else:
                toks.append(self.missing[m].unsqueeze(0).expand(B, -1, -1))
                masks += [0] * N_TOK
        x = torch.cat(toks, dim=1)                                    # (B, 4*16, D)
        pad = torch.tensor(masks, device=x.device, dtype=torch.bool)[None].expand(B, -1)
        x = self.fusion(x, src_key_padding_mask=~pad)
        return self.head(x.mean(dim=1))

    # ---- SensorModel interface ----

    def fit(self, train, val, cfg: TrainConfig) -> None:
        torch.manual_seed(cfg.seed)
        self.train().to(cfg.device)
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.CrossEntropyLoss()
        rng = torch.Generator().manual_seed(cfg.seed)
        best = -1.0
        patience = 0
        for ep in range(cfg.epochs):
            self.train()
            for i in range(0, len(train), cfg.batch_size):
                batch = train[i:i + cfg.batch_size]
                avail = self._dropout_mask(cfg, rng)          # batch-level mask (shared across batch)
                mods = self._stack_mods(batch, avail, cfg)
                lbl = torch.tensor([s.label for s in batch], device=cfg.device)
                loss = crit(self(mods, avail), lbl)
                opt.zero_grad(); loss.backward(); opt.step()
            v = self._evaluate(val, cfg)
            if v > best:
                best = v; patience = 0
                self.save(f"{cfg.out_dir}/{self.name}_seed{cfg.seed}.pt")
            else:
                patience += 1
                if patience >= cfg.patience:
                    break
            print(f"[{self.name}] ep {ep} val {v:.3f} (best {best:.3f})")

    def _dropout_mask(self, cfg: TrainConfig, rng: torch.Generator) -> Dict[str, bool]:
        avail = {m: bool(torch.rand(1, generator=rng).item() > cfg.modality_dropout_p)
                 for m in MODALITIES}
        if not any(avail.values()):
            avail[list(avail)[0]] = True
        return avail

    def _stack_mods(self, samples, avail: Dict[str, bool], cfg: TrainConfig):
        mods = {}
        for m in MODALITIES:
            if avail.get(m):
                mods[m] = torch.stack(
                    [torch.from_numpy(s.modalities[m].data) for s in samples]).to(cfg.device)
        return mods

    @torch.no_grad()
    def _evaluate(self, samples, cfg: TrainConfig) -> float:
        self.eval()
        ok = tot = 0
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i:i + cfg.batch_size]
            avail = {m: True for m in MODALITIES}
            mods = self._stack_mods(batch, avail, cfg)
            preds = self(mods, avail).argmax(-1).cpu().tolist()
            ok += sum(p == s.label for p, s in zip(preds, batch))
            tot += len(batch)
        return ok / max(tot, 1)

    @torch.no_grad()
    def predict(self, sample, available: List[str]) -> Dict[int, float]:
        self.eval()
        avail = {m: m in available for m in MODALITIES}
        mods = {m: torch.from_numpy(sample.modalities[m].data)[None] for m in available}
        logits = self(mods, avail)
        probs = torch.softmax(logits[0], dim=-1)
        return {i: float(p) for i, p in enumerate(probs)}

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "TokenFusionModel":
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m
```

- [ ] **Step 4: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_models.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: token_fusion model with modality dropout"
```

---

### Task 10: late_fusion baseline model

**Files:**
- Create: `framework/models/late_fusion.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add the failing test**

```python
# tests/test_models.py (append)
from framework.models.late_fusion import LateFusionModel

def test_late_fusion_missing_modality():
    m = LateFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["depth"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4

def test_late_fusion_full():
    m = LateFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi", "depth", "lidar", "mmwave"])
    assert abs(sum(probs.values()) - 1.0) < 1e-4
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `framework/models/late_fusion.py`**

```python
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn

from .base import SensorModel, TrainConfig
from .encoders import D, WifiEncoder, DepthEncoder, PointEncoder
from .token_fusion import MODALITIES, MODALITY_IN_C


class LateFusionModel(nn.Module, SensorModel):
    """Baseline: per-modality encoder -> single vector; missing -> zero vector;
    concat -> MLP head. No alignment mechanism (control baseline)."""

    name = "late_fusion"

    def __init__(self, num_classes: int = 27):
        super().__init__()
        self.num_classes = num_classes
        self.encoders = nn.ModuleDict({
            "wifi": WifiEncoder(), "depth": DepthEncoder(),
            "lidar": PointEncoder(3), "mmwave": PointEncoder(5)})
        self.head = nn.Sequential(
            nn.Linear(D * len(MODALITIES), 512), nn.ReLU(), nn.Linear(512, num_classes))

    def forward(self, mods, avail):
        B = next(iter(mods.values())).shape[0] if mods else 1
        feats = []
        for m in MODALITIES:
            if avail.get(m):
                feats.append(self.encoders[m](mods[m]).mean(dim=1))
            else:
                feats.append(torch.zeros(B, D, device=next(self.parameters()).device))
        return self.head(torch.cat(feats, dim=1))

    def fit(self, train, val, cfg: TrainConfig) -> None:
        torch.manual_seed(cfg.seed)
        self.train().to(cfg.device)
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.CrossEntropyLoss()
        best = -1.0
        for ep in range(cfg.epochs):
            self.train()
            for i in range(0, len(train), cfg.batch_size):
                batch = train[i:i + cfg.batch_size]
                avail = {m: True for m in MODALITIES}
                mods = self._stack_mods(batch, avail, cfg)
                lbl = torch.tensor([s.label for s in batch], device=cfg.device)
                loss = crit(self(mods, avail), lbl)
                opt.zero_grad(); loss.backward(); opt.step()
            v = self._evaluate(val, cfg)
            if v > best:
                best = v
                self.save(f"{cfg.out_dir}/{self.name}_seed{cfg.seed}.pt")
            print(f"[{self.name}] ep {ep} val {v:.3f} (best {best:.3f})")

    def _stack_mods(self, samples, avail, cfg):
        return {m: torch.stack([torch.from_numpy(s.modalities[m].data) for s in samples]).to(cfg.device)
                for m in MODALITIES if avail.get(m)}

    @torch.no_grad()
    def _evaluate(self, samples, cfg):
        self.eval()
        ok = tot = 0
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i:i + cfg.batch_size]
            avail = {m: True for m in MODALITIES}
            mods = self._stack_mods(batch, avail, cfg)
            preds = self(mods, avail).argmax(-1).cpu().tolist()
            ok += sum(p == s.label for p, s in zip(preds, batch))
            tot += len(batch)
        return ok / max(tot, 1)

    @torch.no_grad()
    def predict(self, sample, available):
        self.eval()
        avail = {m: m in available for m in MODALITIES}
        mods = {m: torch.from_numpy(sample.modalities[m].data)[None] for m in available}
        probs = torch.softmax(self(mods, avail)[0], dim=-1)
        return {i: float(p) for i, p in enumerate(probs)}

    def save(self, path): torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m
```

- [ ] **Step 4: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_models.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: late_fusion baseline model"
```

---

### Task 11: Training script + smoke train

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Implement `scripts/train.py`**

```python
#!/usr/bin/env python
import argparse, os, json, torch
from framework.dataset.loader import load_dataset
from framework.models.base import TrainConfig
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel

MODELS = {"token_fusion": TokenFusionModel, "late_fusion": LateFusionModel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      seed=args.seed, out_dir=args.out_dir)
    model = MODELS[args.model](num_classes=27)
    model.fit(ds.train, ds.val, cfg)
    print(f"trained {args.model} seed {args.seed} -> {cfg.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-train on the smoke dataset (3 epochs — validates training loop incl. backward end-to-end fast)**

Run:
```bash
cd /home/li/projects/sensorbench && \
/home/li/projects/holollm/.venv/bin/python scripts/train.py \
  --dataset /tmp/sensorbench_smoke --model token_fusion --epochs 3 --out-dir /tmp/sensorbench_ckpt
```
Expected: prints 3 epoch lines with finite val accuracy (train split has ~16 samples → ~1 batch/epoch), checkpoint saved.

- [ ] **Step 3: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: training CLI + smoke train"
```

---

### Task 12: Protocol builder + evaluate harness + leaderboard

**Files:**
- Create: `framework/harness/protocol.py`, `framework/harness/evaluate.py`, `framework/harness/leaderboard.py`, `scripts/run_eval.py`
- Test: `tests/test_protocol.py`, `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol.py
from framework.harness.protocol import build_protocol

def test_build_protocol_15_profiles():
    mods = ["wifi", "depth", "lidar", "mmwave"]
    p = build_protocol(mods, seeds=[0, 1, 2])
    profiles = p["profiles"]
    ids = [x["id"] for x in profiles]
    assert "full" in ids
    assert sum("miss-" in i for i in ids) == 4
    assert sum("only-" in i for i in ids) == 4
    assert sum("miss2-" in i for i in ids) == 6
    assert len(profiles) == 15
    assert p["seeds"] == [0, 1, 2]

def test_protocol_full_available_all():
    mods = ["wifi", "depth", "lidar", "mmwave"]
    p = build_protocol(mods, seeds=[0])
    full = [x for x in p["profiles"] if x["id"] == "full"][0]
    assert set(full["available"]) == set(mods)
```

```python
# tests/test_harness.py
from framework.harness.evaluate import evaluate_model, accuracy
from framework.models.token_fusion import TokenFusionModel
from tests.test_models import _toy_sample


def test_accuracy():
    preds = [{0: 0.9, 1: 0.1}, {0: 0.1, 1: 0.9}]
    labels = [0, 1]
    assert accuracy(preds, labels) == 1.0


def test_evaluate_model_single_sample():
    m = TokenFusionModel(num_classes=27)
    s = _toy_sample()
    res = evaluate_model(m, [s], profile={"id": "full", "available": ["wifi", "depth", "lidar", "mmwave"]})
    assert 0.0 <= res["accuracy"] <= 1.0


def test_leaderboard_mean_std():
    from framework.harness.leaderboard import build_leaderboard
    results = {
        "m": [
            {"profile": "full", "available": [], "accuracy": 0.8, "seed": 0},
            {"profile": "full", "available": [], "accuracy": 0.9, "seed": 1},
            {"profile": "miss-wifi", "available": [], "accuracy": 0.6, "seed": 0},
            {"profile": "miss-wifi", "available": [], "accuracy": 0.8, "seed": 1},
        ]
    }
    lb = build_leaderboard(results)
    assert abs(lb["m"]["profiles"]["full"]["mean"] - 0.85) < 1e-6
    assert abs(lb["m"]["profiles"]["full"]["std"] - 0.05) < 1e-6
    assert lb["m"]["robustness"] == 0.775
    assert "degradation" in lb["m"]
```

- [ ] **Step 2: Run, verify fail**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_protocol.py tests/test_harness.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `framework/harness/protocol.py`**

```python
from __future__ import annotations
import itertools
from typing import List


def build_protocol(modalities: List[str], seeds: List[int]) -> dict:
    profiles = []
    profiles.append({"id": "full", "available": list(modalities)})
    for m in modalities:
        profiles.append({"id": f"miss-{m}", "available": [x for x in modalities if x != m]})
    for a, b in itertools.combinations(modalities, 2):
        profiles.append({"id": f"miss2-{a}-{b}",
                         "available": [x for x in modalities if x not in (a, b)]})
    for m in modalities:
        profiles.append({"id": f"only-{m}", "available": [m]})
    return {"modalities": modalities, "seeds": seeds, "profiles": profiles}
```

- [ ] **Step 4: Implement `framework/harness/evaluate.py`**

```python
from __future__ import annotations
from typing import Dict, List


def accuracy(preds: List[Dict[int, float]], labels: List[int]) -> float:
    if not labels:
        return 0.0
    ok = sum(1 for p, l in zip(preds, labels) if max(p, key=p.get) == l)
    return ok / len(labels)


def evaluate_model(model, samples, profile: dict) -> dict:
    preds = [model.predict(s, profile["available"]) for s in samples]
    labels = [s.label for s in samples]
    return {"profile": profile["id"], "available": profile["available"],
            "accuracy": accuracy(preds, labels)}
```

- [ ] **Step 5: Implement `framework/harness/leaderboard.py`**

```python
from __future__ import annotations
import json, os
from typing import Dict, List


def robust_score(profile_results: List[dict]) -> float:
    return sum(r["accuracy"] for r in profile_results) / max(len(profile_results), 1)


def build_leaderboard(model_results: Dict[str, List[dict]]) -> dict:
    """model_results: {model: [{profile, available, accuracy, seed}, ...]}.
    Groups per-profile across seeds; reports mean + std + per-seed array
    (spec §6.2 requires mean ± std)."""
    lb = {}
    for model, results in model_results.items():
        by_profile = {}
        for r in results:
            by_profile.setdefault(r["profile"], []).append(r["accuracy"])
        profiles = {}
        for p, accs in by_profile.items():
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            profiles[p] = {"mean": round(mean, 4), "std": round(var ** 0.5, 4),
                           "per_seed": [round(a, 4) for a in accs]}
        full = profiles["full"]["mean"]
        rob_mean = sum(v["mean"] for v in profiles.values()) / len(profiles)
        rob_std = (sum(v["std"] ** 2 for v in profiles.values()) / len(profiles)) ** 0.5
        lb[model] = {
            "robustness": round(rob_mean, 4),
            "robustness_std": round(rob_std, 4),
            "acc_full": full,
            "profiles": profiles,
            "degradation": {p: round(full - v["mean"], 4) for p, v in profiles.items()},
        }
    return lb


def save_leaderboard(lb: dict, path: str, protocol: dict, dataset_root: str) -> None:
    out = {"protocol": protocol, "dataset": dataset_root,
           "leaderboard": lb}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
```

- [ ] **Step 6: Implement `scripts/run_eval.py`**

```python
#!/usr/bin/env python
import argparse, json
from framework.dataset.loader import load_dataset
from framework.harness.protocol import build_protocol
from framework.harness.evaluate import evaluate_model
from framework.harness.leaderboard import build_leaderboard, save_leaderboard
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel

MODELS = {"token_fusion": TokenFusionModel, "late_fusion": LateFusionModel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True, help="path to protocol.json")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", default="leaderboard_v1.json")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--seeds", default="0", help="comma-separated")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    protocol = json.load(open(args.protocol))
    models = args.model or list(MODELS)
    seeds = [int(x) for x in args.seeds.split(",")]

    results = {}
    for name in models:
        results[name] = []
        for seed in seeds:
            m = MODELS[name].load(f"{args.ckpt_dir}/{name}_seed{seed}.pt")
            for profile in protocol["profiles"]:
                r = evaluate_model(m, ds.test, profile)
                r["seed"] = seed
                results[name].append(r)

    lb = build_leaderboard(results)
    save_leaderboard(lb, args.out, protocol, args.dataset)
    print(json.dumps(lb, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run tests, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "feat: protocol builder, evaluate harness, leaderboard, eval CLI"
```

---

### Task 13: FULL MMFi ingest (background, monitored)

This is a long-running data task. Run it detached per AGENTS.md rules, monitor progress.

- [ ] **Step 1: Generate the protocol file**

Run:
```bash
cd /home/li/projects/sensorbench && \
/home/li/projects/holollm/.venv/bin/python -c "
from framework.harness.protocol import build_protocol
import json
p = build_protocol(['wifi','depth','lidar','mmwave'], seeds=[0,1,2])
json.dump(p, open('protocol.json','w'), indent=2)
print('profiles:', len(p['profiles']), 'seeds:', p['seeds'])
"
```
Expected: `profiles: 15 seeds: [0, 1, 2]`

- [ ] **Step 2: Launch full ingest in background**

```bash
cd /home/li/projects/sensorbench && mkdir -p logs && \
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/ingest_mmfi.py \
  --annotations-train /home/li/datasets/holollm_annotations/textual_annotations/mmfi/mmficap/mmfi_train_cs_full.json \
  --annotations-test /home/li/datasets/holollm_annotations/textual_annotations/mmfi/mmficap/mmfi_test_cs_full.json \
  --raw-root /home/li/datasets/MMFi_dataset/data/MMFi_Dataset \
  --out datasets/mmfi/v1 > logs/ingest.log 2>&1' < /dev/null &
echo "launched, pid $!"
```

- [ ] **Step 3: Monitor until done (poll `logs/ingest.log`; ingest ~16k samples, allow 1-3h)**

Run (repeat):
```bash
tail -5 /home/li/projects/sensorbench/logs/ingest.log; ls /home/li/projects/sensorbench/datasets/mmfi/v1/data 2>/dev/null | wc -l
```
Expected when done: `train: ~9576 samples ... val: ~2082 ... test: ~4791` (total ≈ 16448 = 11657 train-anns + 4791 test-anns; val = last 5 of the 28 train subjects), `data` dir ≈ 16.4k files.

- [ ] **Step 4: Verify dataset integrity via align check + load**

Run:
```bash
cd /home/li/projects/sensorbench && \
/home/li/projects/holollm/.venv/bin/python -c "
from framework.dataset.loader import load_dataset
from curation.clean.align_check import verify_alignment
ds = load_dataset('datasets/mmfi/v1')
print('splits', {k: len(v) for k,v in ds.splits.items()}, 'modalities', ds.modalities)
issues = 0
for s in ds.test[:1000]:
    issues += len(verify_alignment(s))
print('alignment issues in first 1000 test samples:', issues)
"
```
Expected: splits ~ (train≈9576, val≈2082, test≈4791), 0 alignment issues.

- [ ] **Step 5: Commit protocol.json**

```bash
cd /home/li/projects/sensorbench && git add protocol.json && git commit -m "chore: add mmfi v1 eval protocol"
```

---

### Task 14: Train both models × 3 seeds (background), run full evaluation → leaderboard_v1

- [ ] **Step 1: Launch training (background; ~5-15 min/model/seed on 16GB GPU — models are ~10M params, batched)**

```bash
cd /home/li/projects/sensorbench && \
for m in token_fusion late_fusion; do
  for s in 0 1 2; do
    setsid bash -c "/home/li/projects/holollm/.venv/bin/python scripts/train.py \
      --dataset datasets/mmfi/v1 --model $m --seed $s --epochs 30 \
      --out-dir checkpoints > logs/train_${m}_${s}.log 2>&1" < /dev/null &
  done
done
echo "launched 6 training jobs"
```
NOTE: 6 concurrent jobs on one 16GB card is tight. Monitor `nvidia-smi`; if OOM or memory ≥90%, kill and re-launch sequentially (`for s in 0 1 2; do ...; wait; done` per model).

- [ ] **Step 2: Monitor GPU + training logs**

Run (repeat):
```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv; \
tail -3 /home/li/projects/sensorbench/logs/train_token_fusion_0.log
```
Expected: GPU util high during training; logs show `val X.XXX (best ...)` increasing.

- [ ] **Step 3: Verify checkpoints exist**

Run: `ls /home/li/projects/sensorbench/checkpoints/`
Expected: `late_fusion_seed0.pt late_fusion_seed1.pt ... token_fusion_seed2.pt` (6 files)

- [ ] **Step 4: Run full evaluation (background; 2 models × 15 profiles × ~4.8k test samples of per-sample CPU inference — expect ~30-60 min, backgrounded)**

```bash
cd /home/li/projects/sensorbench && \
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/run_eval.py \
  --dataset datasets/mmfi/v1 --protocol protocol.json \
  --ckpt-dir checkpoints --out leaderboard_v1.json > logs/eval_v1.log 2>&1' < /dev/null &
echo "eval launched"
```
(Optional speedup if needed: run `run_eval.py` on GPU by loading checkpoints with `map_location="cuda"` — but CPU is fine since it's backgrounded.)
Monitor: `tail -5 logs/eval_v1.log` until `leaderboard_v1.json` appears. Then print it:
```bash
/home/li/projects/holollm/.venv/bin/python -c "import json; print(json.dumps(json.load(open('leaderboard_v1.json'))['leaderboard'], indent=2))"
```
Expected: leaderboard with robustness scores (mean ± std) for both models.

- [ ] **Step 5: Commit checkpoints meta + leaderboard**

```bash
cd /home/li/projects/sensorbench && git add leaderboard_v1.json && git commit -m "results: v1 leaderboard (token_fusion vs late_fusion, 3 seeds)"
```

---

### Task 15: v2 data improvement (consistency filtering) + re-run

Uses the trained token_fusion model (now exists) to flag cross-modality-inconsistent samples; removes them → v2 dataset; retrain; compare Robustness.

- [ ] **Step 1: Implement `curation/clean/consistency.py`**

```python
from __future__ import annotations
from typing import List
from framework.models.base import SensorModel


def flag_inconsistent(model: SensorModel, samples: List, drop_rate: float = 0.05) -> List[str]:
    """Use per-modality marginal predictions; flag samples where the max class
    under full-modality differs from single-best-modality predictions (top-1
    disagreement) as suspect. Keep the worst `drop_rate` fraction."""
    scored = []
    all_mods = sorted({m for s in samples for m in s.modalities})
    for s in samples:
        full = model.predict(s, all_mods)
        best = max(full, key=full.get)
        disagree = 0
        for m in all_mods:
            marg = model.predict(s, [m])
            if max(marg, key=marg.get) != best:
                disagree += 1
        scored.append((s.id, disagree / max(len(all_mods), 1)))
    scored.sort(key=lambda x: -x[1])
    n = int(len(scored) * drop_rate)
    return [sid for sid, _ in scored[:n]]
```

- [ ] **Step 2: Add test**

```python
# tests/test_consistency.py
from curation.clean.consistency import flag_inconsistent
from framework.models.token_fusion import TokenFusionModel
from tests.test_models import _toy_sample

def test_flag_inconsistent_empty_when_perfect():
    m = TokenFusionModel(num_classes=27)
    samples = [_toy_sample() for _ in range(4)]
    # untrained model: random but consistent-enough; just check API returns list
    out = flag_inconsistent(m, samples, drop_rate=0.5)
    assert isinstance(out, list) and len(out) <= 2
```

- [ ] **Step 3: Run, verify pass**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_consistency.py -v`
Expected: 1 PASS

- [ ] **Step 4: Implement v2 script `scripts/make_v2.py`**

```python
#!/usr/bin/env python
import argparse, json, os, shutil
from framework.dataset.loader import load_dataset
from curation.clean.consistency import flag_inconsistent
from curation.version.version import write_meta
from framework.models.token_fusion import TokenFusionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="datasets/mmfi/v1")
    ap.add_argument("--v2", default="datasets/mmfi/v2")
    ap.add_argument("--ckpt", default="checkpoints/token_fusion_seed0.pt")
    ap.add_argument("--drop-rate", type=float, default=0.05)
    args = ap.parse_args()

    ds = load_dataset(args.v1)
    model = TokenFusionModel.load(args.ckpt)
    dropped = set()
    # Clean train/val only. Test is NEVER filtered: v1-vs-v2 must be
    # evaluated on the identical, unchanged test set (apples-to-apples).
    for split in ["train", "val"]:
        flagged = flag_inconsistent(model, ds.splits[split], drop_rate=args.drop_rate)
        dropped.update(flagged)
        print(f"{split}: flagged {len(flagged)}")

    # copy data files minus dropped
    os.makedirs(f"{args.v2}/data", exist_ok=True)
    kept = 0
    for fn in os.listdir(f"{args.v1}/data"):
        if fn.replace(".pkl", "") not in dropped:
            shutil.copy(os.path.join(args.v1, "data", fn), os.path.join(args.v2, "data", fn))
            kept += 1
    shutil.copytree(f"{args.v1}/splits", f"{args.v2}/splits", dirs_exist_ok=True)
    shutil.copy(os.path.join(args.v1, "modalities.yaml"), os.path.join(args.v2, "modalities.yaml"))
    json.dump({"v1_to_v2": {"dropped": len(dropped), "reason": "cross-modal consistency filter"},
               "kept": kept}, open(f"{args.v2}/changes.json", "w"), indent=2)
    write_meta(args.v2, name="mmfi", version="v2",
               changelog=[f"v2: dropped {len(dropped)} train/val samples flagged as cross-modality "
                          "inconsistent by token_fusion model (drop_rate=0.05); test unchanged"],
               n_samples=kept, n_modalities=4,
               source={"dataset": "MMFi", "split": "cs", "parent": "mmfi/v1"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"based_on": "mmfi/v1"})
    print(f"v2: kept {kept} samples")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run v2 creation (background — consistency pass is ~5 predict × ~11.6k train/val samples)**

Run:
```bash
cd /home/li/projects/sensorbench && \
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/make_v2.py \
  --v1 datasets/mmfi/v1 --v2 datasets/mmfi/v2 > logs/make_v2.log 2>&1' < /dev/null &
```
Monitor `tail -5 logs/make_v2.log`. Expected: prints flagged counts per split (~5% of train/val each), `datasets/mmfi/v2/` created with `meta.json`, `changes.json`, `modalities.yaml`, and an unchanged test split.

- [ ] **Step 6: Retrain both models on v2 (background) and run eval on v2**

Train with `--dataset datasets/mmfi/v2 --out-dir checkpoints_v2` (separate dir so v1 checkpoints are preserved). Then eval explicitly with the v2 checkpoint dir:
```bash
cd /home/li/projects/sensorbench && \
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/run_eval.py \
  --dataset datasets/mmfi/v2 --protocol protocol.json \
  --ckpt-dir checkpoints_v2 --out leaderboard_v2.json > logs/eval_v2.log 2>&1' < /dev/null &
```
Test split is identical to v1's (never filtered), so v1-vs-v2 comparisons are apples-to-apples.

- [ ] **Step 7: Compare v1 vs v2**

Run:
```bash
cd /home/li/projects/sensorbench && \
/home/li/projects/holollm/.venv/bin/python -c "
import json
for f in ['leaderboard_v1.json','leaderboard_v2.json']:
    d = json.load(open(f))['leaderboard']
    for m, v in d.items():
        print(f, m, 'robustness', round(v['robustness'],4), 'acc_full', round(v['acc_full'],4))
"
```
Expected: report v1 vs v2. If v2 robustness ≤ v1, document the finding honestly in the report (the loop result is the deliverable, not a guaranteed win).

- [ ] **Step 8: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "results: v2 leaderboard + consistency filtering"
```

---

### Task 16: Docs + reproducibility

**Files:**
- Create: `README.md`, `docs/reports/robustness_v1_v2.md`

- [ ] **Step 1: Write `README.md`** — project intro, quickstart (ingest/train/eval commands verbatim), the 15-profile protocol explanation, how to add a model (implement SensorModel), how to add a dataset (add an ingest adapter).

- [ ] **Step 2: Write `docs/reports/robustness_v1_v2.md`** — paste the v1/v2 leaderboard, Degradation matrix per modality, what was flagged in v2, conclusion.

- [ ] **Step 3: Commit**

```bash
cd /home/li/projects/sensorbench && git add -A && git commit -m "docs: README + v1/v2 robustness report"
```

---

## Out of Scope (deferred, per spec §8.2)

- XRF55 ingest, contrastive model, VLM recaptioning, self-collection hardware, cross-dataset fusion training
- `predict_batch` optimization (harness loops per-sample; fine for ~4.8k test samples × 15 profiles)
