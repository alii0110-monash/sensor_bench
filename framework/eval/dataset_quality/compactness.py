"""Compactness evaluation: confusion rate (main), Fisher ratio, leave-one-out distance (diagnostic)."""
from __future__ import annotations

import numpy as np


def compute_confusion_rate(y_true: np.ndarray, y_pred: np.ndarray,
                           num_classes: int) -> float:
    """Off-diagonal fraction of the confusion matrix. In [0, 1]."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    total = cm.sum()
    if total == 0:
        return 0.0
    diag = np.trace(cm)
    return float((total - diag) / total)


def compute_fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """tr(S_b) / tr(S_w) where S_b = between-class, S_w = within-class covariance.

    Diagnostic only (depends on feature dim); not used in CompactScore.
    """
    X = X.astype(np.float32)
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    S_b = np.zeros((X.shape[1],), dtype=np.float32)
    S_w = np.zeros((X.shape[1],), dtype=np.float32)
    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] < 2:
            continue
        nc = Xc.shape[0]
        mean_c = Xc.mean(axis=0)
        S_b += nc * (mean_c - overall_mean) ** 2
        S_w += ((Xc - mean_c) ** 2).sum(axis=0)
    eps = 1e-8
    return float(S_b.sum() / (S_w.sum() + eps))


def compute_leave_one_out_distances(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each sample, Euclidean distance to its class centroid (mean of others).

    Diagnostic only (90th percentile reported alongside).
    """
    X = X.astype(np.float32)
    classes = np.unique(y)
    centroids = {c: X[y == c].mean(axis=0) for c in classes}
    return np.linalg.norm(X - np.array([centroids[y[i]] for i in range(len(y))]),
                          axis=1)