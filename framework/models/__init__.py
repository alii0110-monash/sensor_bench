"""Model package. SensorModel protocol + pluggable fusion models.

`detect_structured_features` inspects a dataset's first sample to decide which
modalities carry 1-D structured features (v5_structfeat) vs raw multi-dim
data, so train.py can build the right encoders automatically.
"""
from __future__ import annotations
from typing import Dict


def detect_structured_features(dataset) -> Dict[str, int]:
    """Return {modality: feat_dim} for modalities whose data is 1-D
    (structured features, e.g. v5_structfeat). Raw modalities (multi-dim
    point clouds / images) are excluded. Uses the first sample of the first
    non-empty split."""
    for split in ("train", "val", "test"):
        samples = getattr(dataset, split, None)
        if samples:
            s = samples[0]
            return {m: int(mod.data.shape[0])
                    for m, mod in s.modalities.items()
                    if mod.data.ndim == 1}
    return {}
