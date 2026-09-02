"""Tests for compactness module: confusion rate, Fisher ratio, leave-one-out distance."""
import numpy as np
import pytest

from framework.eval.dataset_quality.compactness import (
    compute_confusion_rate, compute_fisher_ratio, compute_leave_one_out_distances,
)


def test_confusion_rate_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = y_true.copy()
    rate = compute_confusion_rate(y_true, y_pred, num_classes=3)
    assert rate == 0.0


def test_confusion_rate_off_diagonal_only():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 2, 0])
    rate = compute_confusion_rate(y_true, y_pred, num_classes=3)
    assert rate == pytest.approx(1.0)


def test_confusion_rate_half_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    rate = compute_confusion_rate(y_true, y_pred, num_classes=2)
    assert rate == pytest.approx(0.5)


def test_fisher_ratio_separable_higher():
    rng = np.random.default_rng(0)
    n, dim = 200, 4
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X[:n // 2, 0] -= 2
    X[n // 2:, 0] += 2
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    r_sep = compute_fisher_ratio(X, y)
    y_rand = rng.permutation(y)
    r_rand = compute_fisher_ratio(X, y_rand)
    assert r_sep > r_rand


def test_leave_one_out_distances_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = rng.integers(0, 3, size=50)
    dists = compute_leave_one_out_distances(X, y)
    assert dists.shape == (50,)


def test_compact_score_in_unit_range():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 2, 0])
    rate = compute_confusion_rate(y_true, y_pred, num_classes=3)
    score = 1.0 - rate
    assert 0.0 <= score <= 1.0