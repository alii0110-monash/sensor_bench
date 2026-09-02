"""Batch construction strategies for the training loop.

- "shuffle": per-epoch random shuffle, then contiguous slices. This de-biases
  the data order so batches become unbiased random samples of the dataset.
- "balanced": stratified batches — each batch contains roughly equal samples
  per class (any two classes *present* in a batch differ by at most 1). This
  forces class diversity within a batch, unlike plain shuffle which only
  guarantees it on average. For imbalanced datasets this oversamples rare
  classes. Per-epoch shuffling keeps the grouping varied.
"""
from __future__ import annotations
import random
from typing import Iterator, List


class BalancedIndexer:
    """Build balanced batches by pulling evenly from each class that still has
    samples. The batch's per-class counts differ by at most 1 among the
    classes present; shortfall when rare classes deplete leaves a shorter
    batch rather than unbalancing it."""

    def __init__(self, labels: List[int], num_classes: int, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.rng = random.Random(seed)
        by_class = {}
        for pos, lbl in enumerate(labels):
            by_class.setdefault(lbl, []).append(pos)
        for c in range(num_classes):
            self.rng.shuffle(by_class.setdefault(c, []))
        self.by_class = by_class

    def batches(self) -> Iterator[List[int]]:
        rng = self.rng
        while True:
            # Classes that still have samples.
            active = [c for c in range(self.num_classes) if self.by_class[c]]
            if not active:
                return
            per_class = max(1, self.batch_size // len(active))
            batch = []
            # Pull `per_class` from each active class.
            for c in active:
                pool = self.by_class[c]
                take = min(per_class, len(pool))
                batch += pool[:take]
                del pool[:take]
            rng.shuffle(batch)
            # Top up toward batch_size with 1 extra from distinct active
            # classes (keeps diff <= 1).
            need = self.batch_size - len(batch)
            if need > 0:
                fill_from = [c for c in active if self.by_class[c]]
                rng.shuffle(fill_from)
                for c in fill_from[:need]:
                    batch.append(self.by_class[c].pop())
            rng.shuffle(batch)
            yield batch


def build_batch_indexer(strategy: str, labels: List[int], num_classes: int,
                        batch_size: int, seed: int):
    """Return an object with .batches() yielding lists of sample positions."""
    if strategy == "balanced":
        return BalancedIndexer(labels, num_classes, batch_size, seed)
    raise ValueError(f"unknown batch strategy: {strategy}")


def class_weights(labels: List[int], num_classes: int, mode: str) -> List[float]:
    """Per-class loss weights for imbalanced data. Does NOT change sampling —
    only scales each class's gradient contribution in the loss.

    - "none": all ones.
    - "inverse_freq": w_c = N / (num_classes * n_c)  (sums to 1 over classes).
    - "sqrt_inverse_freq": w_c = sqrt(N / (num_classes * n_c)) — milder.
    """
    if mode == "none":
        return [1.0] * num_classes
    counts = [0] * num_classes
    for lbl in labels:
        counts[lbl] += 1
    n = len(labels)
    if mode == "inverse_freq":
        return [n / (num_classes * max(c, 1)) for c in counts]
    if mode == "sqrt_inverse_freq":
        return [(n / (num_classes * max(c, 1))) ** 0.5 for c in counts]
    raise ValueError(f"unknown class_weight mode: {mode}")
