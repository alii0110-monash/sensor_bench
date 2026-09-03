"""Base-sample loading with caption-id join + per-selection memory preflight."""
from __future__ import annotations
import json
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch

from framework.dataset.loader import LazySplit, Sample, available_memory_bytes


def _read_sample(p: str) -> Sample:
    with open(p, "rb") as f:
        return Sample.from_dict(pickle.load(f))


def load_caption_ids(captions_jsonl: str) -> set:
    ids = set()
    with open(captions_jsonl) as f:
        for line in f:
            ids.add(json.loads(line)["id"])
    return ids


def load_split_base(root: str, split: str, mode: str = "auto",
                    caption_ids: Optional[set] = None,
                    cache_size: int = 256) -> Tuple[List[Sample], List[str], dict]:
    """Load base (non-__aug) samples of a split, filtered to files that exist.

    caption_ids: when given, the split is joined on these ids (train protocol,
    9205 base); otherwise plain __aug filtering applies (val protocol, 1870).
    Returns (samples, missing_ids, preflight) — preflight stats only the
    selected files (先量后跑), not the whole dataset."""
    data_dir = os.path.join(root, "data")
    split_p = os.path.join(root, "splits", f"{split}.json")
    ids = json.load(open(split_p)) if os.path.exists(split_p) else []
    if caption_ids is not None:
        ids = [i for i in ids if i in caption_ids]
    else:
        ids = [i for i in ids if "__aug" not in i]
    existing = [i for i in ids if os.path.exists(os.path.join(data_dir, f"{i}.pkl"))]
    missing = [i for i in ids if i not in set(existing)]

    sizes = [os.path.getsize(os.path.join(data_dir, f"{i}.pkl")) for i in existing]
    estimated = sum(sizes)
    avail = available_memory_bytes()
    fits = estimated <= avail
    preflight = {"n_selected": len(existing), "n_missing": len(missing),
                 "estimated_bytes": estimated, "available_bytes": avail, "fits": fits}
    if mode == "auto":
        mode = "eager" if fits else "lazy"
    if mode not in ("eager", "lazy"):
        raise ValueError(f"mode must be auto/eager/lazy, got {mode!r}")
    if mode == "eager" and not fits:
        raise MemoryError(
            f"[llm_sft] selected base samples need ~{estimated/1e9:.1f}GB but only "
            f"{avail/1e9:.1f}GB available; use mode='lazy' or a GPU node")
    if mode == "lazy":
        return LazySplit(data_dir, existing, cache_size=cache_size), missing, preflight
    return [_read_sample(os.path.join(data_dir, f"{i}.pkl")) for i in existing], missing, preflight


def collate_mods(samples, device):
    """Stack all 5 modalities (full profile) to device tensors + labels."""
    from framework.models.alignment import MODALITIES
    mods = {}
    for m in MODALITIES:
        arrs = [s.modalities[m].data for s in samples]
        mods[m] = torch.from_numpy(np.stack(arrs).astype("float32")).to(device)
    labels = torch.tensor([s.label for s in samples], dtype=torch.long, device=device)
    return mods, labels
