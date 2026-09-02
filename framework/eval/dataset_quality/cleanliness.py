"""Cleanliness evaluation: anomaly rate + JS cross-modal consistency + quantized-hash dup."""
from __future__ import annotations

import numpy as np


def compute_anomaly_rate(probs: np.ndarray, y_true: np.ndarray,
                         anomaly_threshold: float = 0.3) -> float:
    """Fraction of train samples whose true-class prob < (1 - threshold)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    true_probs = probs[np.arange(n), y_true]
    anomaly_scores = 1.0 - true_probs
    return float((anomaly_scores > anomaly_threshold).mean())


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric JS divergence between two probability distributions."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def compute_inconsistency_rate(js_per_sample: np.ndarray,
                               js_threshold: float = 0.1) -> float:
    """Fraction of samples whose mean cross-modal JS exceeds threshold."""
    if len(js_per_sample) == 0:
        return 0.0
    return float((js_per_sample > js_threshold).mean())


def compute_dup_rate_quantized(features: np.ndarray, decimals: int = 2) -> float:
    """Fraction of samples sharing a quantized-feature hash with another sample.

    Quantization: round to `decimals` decimals, then hash per-row.
    """
    if features.shape[0] == 0:
        return 0.0
    rounded = np.round(features.astype(np.float64), decimals=decimals)
    keys = [tuple(r) for r in rounded]
    seen = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    total_dups = sum(c for c in seen.values() if c > 1)
    return float(total_dups / len(keys))


def compute_modality_contribution(probs_full: np.ndarray,
                                  probs_drop: np.ndarray) -> float:
    """Fraction of samples where argmax changes when a modality is dropped.

    Replaces the broken per-modality-independent JS divergence metric.
    Uses ONE calibrated model's outputs: full vs drop_m. High value means
    modality m was contributing unique information to the prediction.
    """
    if probs_full.shape != probs_drop.shape:
        raise ValueError(f"shape mismatch: {probs_full.shape} vs {probs_drop.shape}")
    if probs_full.shape[0] == 0:
        return 0.0
    full_argmax = probs_full.argmax(axis=-1)
    drop_argmax = probs_drop.argmax(axis=-1)
    return float((full_argmax != drop_argmax).mean())