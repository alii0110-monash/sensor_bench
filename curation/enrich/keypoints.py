# curation/enrich/keypoints.py
"""Keypoint normalization + spatial augmentation for body-keypoint modalities.

Operates on (T, 17, 2) keypoint arrays (COCO 17 joints: 0=noise, 11/12=hips,
15/16=ankles). Normalization is deterministic (hip-center + torso-length scale)
and applied to ALL splits; augmentation (flip/translate/scale) is applied only
to training data, producing offline variants stored in the dataset.
"""
from __future__ import annotations

import numpy as np

COCO_LEFT_RIGHT = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7,
                   9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
FLIP_IDX = [0] + [COCO_LEFT_RIGHT[i] for i in range(1, 17)]  # position i <- joint swap(i)


def _hip_center(kpts: np.ndarray) -> np.ndarray:
    """(T,17,2) -> (T,2) mid-point of hips (joints 11,12)."""
    return (kpts[:, 11, :] + kpts[:, 12, :]) / 2.0


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Deterministic: translate hip-center to origin, scale by mean torso
    length (nose-to-hip distance). Output ~ unit scale, camera-invariant."""
    kpts = np.asarray(kpts, dtype=np.float32)
    hip = _hip_center(kpts)
    centered = kpts - hip[:, None, :]
    torso = np.linalg.norm(centered[:, 0, :], axis=-1)      # nose-to-hip per frame
    scale = float(np.mean(torso)) + 1e-9
    return centered / scale


def augment_keypoints(kpts: np.ndarray, rng: np.random.Generator,
                      flip_p: float = 0.5, trans_frac: float = 0.1,
                      scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Stochastic spatial augmentation (training only). Flip (with L/R joint
    swap), translate, scale. kpts assumed already normalized."""
    out = np.asarray(kpts, dtype=np.float32).copy()
    if rng.random() < flip_p:
        out[:, :, 0] *= -1            # mirror x
        out = out[:, FLIP_IDX]        # swap L/R joints (position i <- joint FLIP_IDX[i])
    tx, ty = rng.uniform(-trans_frac, trans_frac, size=2)
    out += np.array([tx, ty], dtype=np.float32)
    s = rng.uniform(*scale_range)
    out *= s
    return out


def make_variant_id(sample_id: str, k: int) -> str:
    return f"{sample_id}__aug{k}"
