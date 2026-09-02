"""Dataset access service: load any Dataset-protocol root lazily, derive
health statistics, and parse sample ids.

All reads go through `framework.dataset.loader` (mode='lazy'); the GUI never
writes to the dataset root.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from framework.dataset.loader import Dataset, load_dataset

ID_RE = re.compile(r"E(\d+)_S(\d+)_A(\d+)")


def parse_sample_id(sid: str) -> Optional[dict]:
    """'E04_S33_A01_f37-46' -> {ep, subject, action}. Variant suffix ignored."""
    m = ID_RE.search(sid)
    if not m:
        return None
    return {"ep": int(m.group(1)), "subject": f"S{m.group(2)}", "action": int(m.group(3))}


def list_dataset_roots(base: str = "datasets") -> List[str]:
    """Roots that look like Dataset-protocol dirs (meta.json + data/ + splits/)."""
    out = []
    for root, dirs, files in os.walk(base):
        if root == base:
            continue
        if {"meta.json", "data"} <= set(dirs + files) and os.path.isdir(os.path.join(root, "splits")):
            out.append(root)
            dirs[:] = []  # don't descend further into a dataset root
    return sorted(out)


def dataset_name(root: str) -> str:
    """'datasets/mmfi/v4' -> 'mmfi_v4' (from meta.json, fallback to basename)."""
    try:
        meta = json.load(open(os.path.join(root, "meta.json")))
        return f"{meta.get('name', 'dataset')}_{meta.get('version', os.path.basename(root))}"
    except (OSError, ValueError):
        return os.path.basename(root).replace(os.sep, "_")


def open_dataset(root: str, split: Optional[str] = None) -> Dataset:
    if not os.path.isdir(os.path.join(root, "data")):
        raise ValueError(f"not a dataset root: {root}")
    ds = load_dataset(root, mode="lazy")
    if split and split not in ds.splits:
        raise ValueError(f"split {split!r} not in {list(ds.splits)}")
    return ds


def split_ids(root: str, split: str) -> List[str]:
    """Ids for a split, filtered to files that exist on disk."""
    p = os.path.join(root, "splits", f"{split}.json")
    if not os.path.exists(p):
        return []
    ids = json.load(open(p))
    return [i for i in ids if os.path.exists(os.path.join(root, "data", f"{i}.pkl"))]


def sample_summary(sample) -> dict:
    """Compact per-sample info for the review table."""
    return {
        "id": sample.id,
        "label": sample.label,
        "modalities": list(sample.modalities.keys()),
        "frames": {m: len(mod.frame_indices) for m, mod in sample.modalities.items()},
        "n_captions": len(sample.text.get("captions", [])) if isinstance(sample.text, dict) else 0,
    }


def compute_health(sample_iter, max_samples: Optional[int] = None) -> dict:
    """Aggregate label / subject / frame-length distributions and per-modality
    anomaly counts (NaN/Inf, all-zero, constant). Iterates lazily and samples
    at most `max_samples` samples (None = all)."""
    label_counter: Counter = Counter()
    subject_counter: Counter = Counter()
    frame_counter: Dict[str, Counter] = {}
    anomalies: Dict[str, Counter] = {}
    total = 0
    for sample in sample_iter:
        if max_samples is not None and total >= max_samples:
            break
        total += 1
        label_counter[sample.label] += 1
        p = parse_sample_id(sample.id)
        subject_counter[p["subject"] if p else "unknown"] += 1
        for name, mod in sample.modalities.items():
            frame_counter.setdefault(name, Counter())[len(mod.frame_indices)] += 1
            an = anomalies.setdefault(name, Counter())
            d = mod.data
            try:
                if d.size == 0:
                    an["empty"] += 1
                    continue
                if not np.isfinite(d).all():
                    an["nan_inf"] += 1
                flat = d.reshape(-1)
                if flat.size and (flat.max() == flat.min()):
                    an["constant"] += 1
                if flat.size and np.count_nonzero(flat) == 0:
                    an["allzero"] += 1
            except (ValueError, TypeError):
                an["unreadable"] += 1
    return {
        "n_scanned": total,
        "label_dist": dict(sorted(label_counter.items())),
        "subject_dist": dict(sorted(subject_counter.items())),
        "frame_dist": {m: dict(sorted(c.items())) for m, c in frame_counter.items()},
        "anomalies": {m: dict(c) for m, c in anomalies.items()},
    }


def make_label_lookup(session_state, dataset: str, split: str,
                      ds: Dataset, ids: List[str]):
    """Return callable(sid)->label with a per-(dataset,split) cache in
    session_state. The full id->label map is built on first miss (loads every
    sample once) and reused afterwards, so label/pred filters stay fast.
    Uses only `in`/`[]`/`del` — Streamlit's session_state proxy has no .get()."""
    cache_key = f"labels_{dataset}_{split}"

    def _lookup(sid: str) -> Optional[int]:
        m = session_state[cache_key] if cache_key in session_state else None
        if m is None or len(m) < len(ids):
            built = {iid: ds.splits[split][i].label for i, iid in enumerate(ids)}
            session_state[cache_key] = built
            m = built
        return m.get(sid)

    return _lookup


_RAW_ROOT_CANDIDATES = (
    "/home/li/datasets/MMFi_dataset/data/MMFi_Dataset",
    "/home/li/datasets/MMFi_dataset",
)


def find_raw_root(explicit: Optional[str] = None) -> Optional[str]:
    """Locate the raw MMFi dataset root (contains E*/S*/A*/*/depth/*.png)."""
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for cand in _RAW_ROOT_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return None


def raw_depth_frames(sample, raw_root: Optional[str]) -> Optional[np.ndarray]:
    """Load a sample's depth frames at native resolution (480x640) from the raw
    dataset, as (T,1,H,W) float32 meters. Returns None if unavailable."""
    if not raw_root or "depth" not in sample.modalities:
        return None
    try:
        env = sample.meta.get("env")
        subj = sample.meta.get("subject")
        if not env or not subj:
            return None
        action = f"A{sample.label + 1:02d}"
        ddir = os.path.join(raw_root, env, subj, action, "depth")
        if not os.path.isdir(ddir):
            return None
        import cv2  # lazy import: keep GUI startup fast

        frames = []
        for fi in sample.modalities["depth"].frame_indices:
            p = os.path.join(ddir, f"frame{int(fi):03d}.png")
            if not os.path.exists(p):
                return None
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
            frames.append(img[None])
        return np.stack(frames).astype(np.float32)
    except Exception:
        return None


def find_quality_json(root: str, results_dir: str = "results") -> Optional[str]:
    """Find a results/quality_*.json whose 'dataset' matches the root."""
    if not os.path.isdir(results_dir):
        return None
    for fn in sorted(os.listdir(results_dir)):
        if not fn.startswith("quality_") or not fn.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(results_dir, fn)))
        except (OSError, ValueError):
            continue
        if data.get("dataset") == root:
            return os.path.join(results_dir, fn)
    return None