"""Tests for cleanliness: anomaly rate, JS divergence, quantized-hash dup, modality contribution."""
import numpy as np
import pytest

from framework.eval.dataset_quality.cleanliness import (
    compute_anomaly_rate, jensen_shannon, compute_inconsistency_rate,
    compute_dup_rate_quantized, compute_modality_contribution,
)


def test_anomaly_rate_with_threshold():
    # probs shape (N, num_classes); true-class prob per sample = probs[i, y_true[i]]
    n, n_cls = 4, 5
    probs = np.full((n, n_cls), 0.1)
    probs[0, 0] = 0.9   # true prob 0.9 → score 0.1
    probs[1, 1] = 0.5   # 0.5 → 0.5
    probs[2, 2] = 0.2   # 0.2 → 0.8
    probs[3, 3] = 0.05  # 0.05 → 0.95
    y_true = np.array([0, 1, 2, 3])
    rate = compute_anomaly_rate(probs, y_true, anomaly_threshold=0.3)
    # scores [0.1, 0.5, 0.8, 0.95]; > 0.3 → 3 flagged → 0.75
    assert rate == pytest.approx(0.75)


def test_anomaly_rate_all_correct():
    n, n_cls = 3, 4
    probs = np.full((n, n_cls), 0.05)
    probs[0, 0] = probs[1, 1] = probs[2, 2] = 0.95
    y_true = np.array([0, 1, 2])
    rate = compute_anomaly_rate(probs, y_true, anomaly_threshold=0.3)
    assert rate == 0.0


def test_jensen_shannon_symmetric():
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    js_pq = jensen_shannon(p, q)
    js_qp = jensen_shannon(q, p)
    assert js_pq == pytest.approx(js_qp)


def test_jensen_shannon_identical_zero():
    p = np.array([0.3, 0.3, 0.4])
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-6)


def test_inconsistency_rate_with_threshold():
    js_per_sample = np.array([0.05, 0.2, 0.4, 0.6])
    rate = compute_inconsistency_rate(js_per_sample, js_threshold=0.3)
    assert rate == 0.5


def test_dup_rate_quantized_identifies_duplicates():
    feats = np.array([
        [1.2341, 2.3441],
        [1.2342, 2.3442],
        [1.2343, 2.3443],
        [5.678, 9.012],
    ], dtype=np.float32)
    rate = compute_dup_rate_quantized(feats, decimals=3)
    # rows 0/1/2 all round to (1.234, 2.344) → 3 dups → 3/4 = 0.75
    assert rate == pytest.approx(0.75)


def test_dup_rate_quantized_no_dups():
    feats = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rate = compute_dup_rate_quantized(feats, decimals=2)
    assert rate == 0.0


def test_dup_rate_quantized_empty():
    rate = compute_dup_rate_quantized(np.zeros((0, 4), np.float32), decimals=2)
    assert rate == 0.0

# --- modality contribution (drop-modality argmax shift) ---

def test_modality_contribution_zero_when_identical():
    """If dropping has no effect, contribution = 0."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=10).astype(np.float32)
    rate = compute_modality_contribution(probs, probs.copy())
    assert rate == 0.0


def test_modality_contribution_all_when_completely_flipped():
    """If dropping always flips argmax, contribution = 1."""
    probs_full = np.eye(4, dtype=np.float32)  # argmax = 0,1,2,3
    probs_drop = np.flip(probs_full, axis=1)  # argmax = 3,2,1,0
    rate = compute_modality_contribution(probs_full, probs_drop)
    assert rate == 1.0


def test_modality_contribution_partial():
    """Half the samples flip argmax → contribution ≈ 0.5."""
    rng = np.random.default_rng(0)
    n, n_cls = 20, 4
    probs = np.zeros((n, n_cls), dtype=np.float32)
    probs[range(n), rng.integers(0, n_cls, n)] = 1.0  # one-hot
    # First half: shift argmax by +1 (always changes)
    # Second half: keep argmax the same
    probs_drop = probs.copy()
    for i in range(n // 2):
        old = probs_drop[i].argmax()
        probs_drop[i] = 0
        probs_drop[i, (old + 1) % n_cls] = 1.0
    rate = compute_modality_contribution(probs, probs_drop)
    assert rate == pytest.approx(0.5)


def test_modality_contribution_shape_mismatch_raises():
    probs_a = np.zeros((10, 4), np.float32)
    probs_b = np.zeros((8, 4), np.float32)
    with pytest.raises(ValueError, match="shape"):
        compute_modality_contribution(probs_a, probs_b)
