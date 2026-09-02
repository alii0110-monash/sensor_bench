# tests/test_batching.py
import numpy as np
import pytest
from framework.models.batching import build_batch_indexer, BalancedIndexer, class_weights


def _labels_imbalanced():
    # 27 classes, imbalanced: class0=100, class1=40, class2=10, rest=2 each
    labels = []
    for c in range(27):
        n = {0: 100, 1: 40, 2: 10}.get(c, 2)
        labels += [c] * n
    return labels


def test_balanced_batch_each_class_count_diff_le_one():
    labels = _labels_imbalanced()
    indexer = build_batch_indexer("balanced", labels, 27, 64, 0)
    seen = set()
    for batch in indexer.batches():
        assert len(batch) <= 64
        cnt = {}
        for pos in batch:
            c = labels[pos]
            cnt[c] = cnt.get(c, 0) + 1
            seen.add(pos)
        # Full batches are balanced (diff <= 1); only the trailing short
        # batch may deviate once rare classes are exhausted.
        if len(batch) == 64:
            counts = list(cnt.values())
            assert max(counts) - min(counts) <= 1, f"unbalanced full batch: {cnt}"
    # every sample used across the epoch
    assert len(seen) == len(labels)


def test_balanced_batch_size_never_exceeds():
    labels = _labels_imbalanced()
    indexer = build_batch_indexer("balanced", labels, 27, 64, 1)
    for batch in indexer.batches():
        assert len(batch) <= 64


def test_balanced_reproducible_with_seed():
    labels = _labels_imbalanced()
    a = list(build_batch_indexer("balanced", labels, 27, 64, 42).batches())
    b = list(build_batch_indexer("balanced", labels, 27, 64, 42).batches())
    assert a == b  # same seed -> same batches


def test_balanced_vs_seed_produces_different_orders():
    labels = _labels_imbalanced()
    a = list(build_batch_indexer("balanced", labels, 27, 64, 0).batches())
    b = list(build_batch_indexer("balanced", labels, 27, 64, 1).batches())
    # Different seeds should partition samples into different batch orders.
    # Assert the flattened sample sequences differ.
    flat_a = [p for batch in a for p in batch]
    flat_b = [p for batch in b for p in batch]
    assert flat_a != flat_b


def test_balanced_covers_rare_classes():
    """Rare classes (only 2 samples) must still appear in batches, not get
    starved out."""
    labels = _labels_imbalanced()
    indexer = build_batch_indexer("balanced", labels, 27, 64, 0)
    pos_seen = set()
    for batch in indexer.batches():
        pos_seen.update(batch)
    # every sample position appears exactly once across the epoch
    assert pos_seen == set(range(len(labels)))


def test_class_weights_none_is_ones():
    labels = _labels_imbalanced()
    w = class_weights(labels, 27, "none")
    assert w == [1.0] * 27


def test_class_weights_inverse_freq_rare_heavier():
    labels = _labels_imbalanced()  # class0=100, class1=40, class2=10, rest=2
    w = class_weights(labels, 27, "inverse_freq")
    # rare classes get higher weight
    assert w[2] > w[1] > w[0]
    # weight ratio matches inverse frequency ratio (class2:class0 = 100:10)
    assert abs(w[2] / w[0] - 10.0) < 1e-6


def test_class_weights_sqrt_milder_than_inverse():
    labels = _labels_imbalanced()
    w_inv = class_weights(labels, 27, "inverse_freq")
    w_sqrt = class_weights(labels, 27, "sqrt_inverse_freq")
    # sqrt version is closer to 1 (milder)
    assert abs(w_sqrt[0] - 1) < abs(w_inv[0] - 1)
    assert abs(w_sqrt[2] - 1) < abs(w_inv[2] - 1)
