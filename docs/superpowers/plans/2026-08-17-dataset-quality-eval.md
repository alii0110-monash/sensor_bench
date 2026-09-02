# 数据集质量评测系统 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-dimension (info/compact/clean) lightweight Linear-probe evaluation system that measures dataset intrinsic quality, decoupled from downstream task / LLM / template evaluation.

**Architecture:** Five modules under `framework/eval/dataset_quality/` (modality_probe, compactness, cleanliness, report, leaderboard) + one entry script `scripts/run_dataset_quality.py`. P0 guard enforces test-split isolation. All hyperparameters and weights persisted in JSON metadata for reproducibility.

**Tech Stack:** PyTorch (Linear + Adam + CrossEntropy), NumPy, scikit-learn (only for `confusion_matrix` utility), Matplotlib (diagnostic plots), existing `framework/dataset/loader.py`.

**Spec:** `docs/superpowers/specs/2026-08-17-dataset-quality-eval-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `framework/eval/dataset_quality/__init__.py` | Package init, re-export main API |
| `framework/eval/dataset_quality/modality_probe.py` | Per-modality Linear + Concat-Linear training + feature extraction + InfoScore |
| `framework/eval/dataset_quality/compactness.py` | Confusion matrix + Fisher ratio + leave-one-out distance |
| `framework/eval/dataset_quality/cleanliness.py` | Anomaly rate + JS divergence cross-modal consistency + quantized-hash dup detection |
| `framework/eval/dataset_quality/report.py` | JSON metadata assembly + matplotlib diagnostic plots |
| `framework/eval/dataset_quality/leaderboard.py` | Cross-version aggregation + markdown rendering |
| `scripts/run_dataset_quality.py` | CLI entry: load dataset, run probes, write JSON |
| `tests/test_dataset_quality/__init__.py` | Test package |
| `tests/test_dataset_quality/test_modality_probe.py` | Linear probe shape, train loop, acc computation |
| `tests/test_dataset_quality/test_compactness.py` | Confusion matrix / Fisher / distance |
| `tests/test_dataset_quality/test_cleanliness.py` | JS symmetric, hash quantization, anomaly threshold |
| `tests/test_dataset_quality/test_info_score.py` | Clipping behavior, formula range |
| `tests/test_dataset_quality/test_p0_guard.py` | test-split rejected |
| `tests/test_dataset_quality/test_run_e2e.py` | Full pipeline on mock dataset |
| `tests/test_dataset_quality/test_leaderboard.py` | Markdown render format |
| `results/quality_v{1,2,4}.json` | Per-version outputs |
| `leaderboard_quality.md` | Cross-version table |
| `docs/reports/dataset_quality_v1_v2_v4.md` | Diagnostic report |

---

## Task 1: P0 入口护栏 + 入口脚本骨架

**Files:**
- Create: `framework/eval/dataset_quality/__init__.py`
- Create: `scripts/run_dataset_quality.py`
- Create: `tests/test_dataset_quality/__init__.py`
- Create: `tests/test_dataset_quality/test_p0_guard.py`

- [ ] **Step 1: Write failing test for P0 guard**

```python
# tests/test_dataset_quality/test_p0_guard.py
import pytest

def test_test_split_rejected():
    """Passing test split to probe evaluation must raise."""
    from scripts.run_dataset_quality import parse_args, validate_splits
    args = parse_args(["--dataset", "datasets/mmfi/v4",
                       "--eval-split", "test"])
    with pytest.raises(AssertionError, match="test split"):
        validate_splits(args)


def test_val_split_accepted():
    from scripts.run_dataset_quality import parse_args, validate_splits
    args = parse_args(["--dataset", "datasets/mmfi/v4",
                       "--eval-split", "val"])
    validate_splits(args)  # must not raise


def test_train_split_accepted():
    from scripts.run_dataset_quality import parse_args, validate_splits
    args = parse_args(["--dataset", "datasets/mmfi/v4",
                       "--eval-split", "train"])
    validate_splits(args)  # must not raise
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_dataset_quality/test_p0_guard.py -v`
Expected: ImportError or AttributeError (functions not yet defined).

- [ ] **Step 3: Implement entry script skeleton with P0 guard**

```python
# scripts/run_dataset_quality.py
#!/usr/bin/env python
"""Dataset quality eval (P0-P4 of dataset-quality-eval-design spec)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_EVAL_SPLITS = {"train", "val"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-split", choices=sorted(ALLOWED_EVAL_SPLITS),
                    default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-classes", type=int, default=27)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-batch-size", type=int, default=256)
    ap.add_argument("--anomaly-threshold", type=float, default=0.3)
    ap.add_argument("--js-threshold", type=float, default=0.1)
    ap.add_argument("--hash-decimals", type=int, default=2)
    ap.add_argument("--dup-weight", type=float, default=0.5)
    ap.add_argument("--w-info", type=float, default=0.4)
    ap.add_argument("--w-compact", type=float, default=0.4)
    ap.add_argument("--w-clean", type=float, default=0.2)
    ap.add_argument("--info-w-per-modality", type=float, default=0.7)
    ap.add_argument("--info-w-complement", type=float, default=0.3)
    ap.add_argument("--plots-dir", default=None,
                    help="If set, write diagnostic plots here.")
    return ap.parse_args(argv)


def validate_splits(args):
    """P0 guard: test split must never enter probe evaluation."""
    assert args.eval_split != "test", \
        "test split cannot be used for probe evaluation (P0 guard)"
    assert args.eval_split in ALLOWED_EVAL_SPLITS


def main():
    args = parse_args()
    validate_splits(args)
    print(f"[dq] dataset={args.dataset} eval_split={args.eval_split}")
    # TODO: subsequent tasks wire in modality_probe, compactness, cleanliness


if __name__ == "__main__":
    main()
```

Also create empty `framework/eval/dataset_quality/__init__.py` and `tests/test_dataset_quality/__init__.py`.

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_dataset_quality/test_p0_guard.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/__init__.py scripts/run_dataset_quality.py tests/test_dataset_quality/
git commit -m "feat(dq-eval): P0 guard + entry script skeleton"
```

---

## Task 2: 模态特征提取

**Files:**
- Create: `framework/eval/dataset_quality/modality_probe.py`
- Test: `tests/test_dataset_quality/test_modality_probe.py`

- [ ] **Step 1: Write failing tests for feature extraction**

```python
# tests/test_dataset_quality/test_modality_probe.py
import numpy as np
import torch
from framework.dataset.sample import Sample, Modality
from framework.eval.dataset_quality.modality_probe import (
    extract_modality_feature, extract_concat_feature, MODALITY_ORDER,
)


def _toy_sample():
    return Sample(
        id="S1",
        label=0,
        modalities={
            "rgb": Modality(data=np.random.rand(5, 3, 8, 8).astype(np.float32),
                            frame_indices=[0, 1, 2, 3, 4]),
            "depth": Modality(data=np.random.rand(5, 224, 224).astype(np.float32),
                              frame_indices=[0, 1, 2, 3, 4]),
            "lidar": Modality(data=np.random.rand(5, 100, 3).astype(np.float32),
                              frame_indices=[0, 1, 2, 3, 4]),
        },
    )


def test_modality_feature_shape():
    s = _toy_sample()
    feat = extract_modality_feature(s, "rgb")
    assert isinstance(feat, np.ndarray)
    assert feat.ndim == 1
    assert feat.shape[0] == 3 * 8 * 8  # 192


def test_modality_feature_rgb_channels_preserved():
    """Channel dim (last) must be preserved; mean over time/spatial only."""
    s = _toy_sample()
    rgb = s.modalities["rgb"].data
    # build deterministic sample
    rgb[:] = 0.0
    rgb[0, 0, 0, 0] = 1.0  # one pixel lit
    feat = extract_modality_feature(s, "rgb")
    # Channel 0 should sum to 1/(5*8*8)
    assert feat[0] > 0
    assert feat.shape[0] == 3 * 8 * 8


def test_modality_feature_depth_2d_no_channel():
    s = _toy_sample()
    s.modalities["depth"].data[:] = 0.0
    s.modalities["depth"].data[0, 0, 0] = 1.0
    feat = extract_modality_feature(s, "depth")
    # depth shape (5, 224, 224) → mean over axis 0 → (224, 224)
    assert feat.shape == (224 * 224,)


def test_concat_feature_concatenates_modalities():
    s = _toy_sample()
    feat = extract_concat_feature(s, ["rgb", "depth", "lidar"])
    rgb_d = 3 * 8 * 8
    depth_d = 224 * 224
    lidar_d = 100 * 3
    assert feat.shape == (rgb_d + depth_d + lidar_d,)


def test_modality_order_contains_five():
    assert set(MODALITY_ORDER) == {"rgb", "depth", "lidar", "mmwave", "wifi"}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_dataset_quality/test_modality_probe.py -v`
Expected: ImportError (extract_modality_feature not defined).

- [ ] **Step 3: Implement feature extraction**

```python
# framework/eval/dataset_quality/modality_probe.py
"""Per-modality Linear + Concat-Linear probes for dataset intrinsic quality.

The probe trains on raw modality data (no pretrained encoder) so the eval
measures dataset properties, not model quality. Feature extraction: mean over
time/frame axis (axis 0), keeping all other dims as a flat vector.
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np

# Modality order is fixed (matches framework.tokens.tokenizer.MODALITY_ORDER).
MODALITY_ORDER = ["rgb", "depth", "lidar", "mmwave", "wifi"]


def extract_modality_feature(sample, modality: str) -> np.ndarray:
    """Return a 1-D float feature vector for one modality.

    Convention: mean over axis 0 (time/frames), flatten the rest.
    Channel dim is preserved (last dim kept, not reduced).
    """
    data = sample.modalities[modality].data  # (T, ...)
    if data.ndim == 1:
        return data.astype(np.float32)
    feat = data.mean(axis=0)
    return feat.reshape(-1).astype(np.float32)


def extract_concat_feature(sample, modalities: Sequence[str]) -> np.ndarray:
    """Concatenate per-modality features into one flat vector."""
    feats = [extract_modality_feature(sample, m) for m in modalities
             if m in sample.modalities]
    return np.concatenate(feats).astype(np.float32)


def stack_split(samples, modalities: Sequence[str], concat: bool = False):
    """Build (X, y) tensors from a sample list.

    Returns:
        X_dict: {modality: np.ndarray (N, dim_m)} if not concat
                {"concat": np.ndarray (N, sum_dims)} if concat
        y: np.ndarray (N,) int64
    """
    y = np.array([s.label for s in samples], dtype=np.int64)
    if concat:
        X = np.stack([extract_concat_feature(s, modalities) for s in samples])
        return {"concat": X}, y
    X_dict = {}
    for m in modalities:
        feats = [extract_modality_feature(s, m) for s in samples
                 if m in s.modalities]
        X_dict[m] = np.stack(feats) if feats else np.zeros((0, 1), np.float32)
    return X_dict, y
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_dataset_quality/test_modality_probe.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/modality_probe.py tests/test_dataset_quality/test_modality_probe.py
git commit -m "feat(dq-eval): per-modality and concat feature extraction"
```

---

## Task 3: Linear probe 训练 + per-modality acc

**Files:**
- Modify: `framework/eval/dataset_quality/modality_probe.py`
- Modify: `tests/test_dataset_quality/test_modality_probe.py`

- [ ] **Step 1: Add failing tests for probe training**

Append to `tests/test_dataset_quality/test_modality_probe.py`:

```python
import torch
from framework.eval.dataset_quality.modality_probe import (
    train_probe, evaluate_probe, _to_tensor,
)


def _toy_classification_data(n=200, dim=16, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return X, y


def test_to_tensor_conversion():
    arr = np.zeros((3, 4), np.float32)
    t = _to_tensor(arr)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 4)


def test_train_probe_returns_model():
    X, y = _toy_classification_data()
    model = train_probe(X, y, num_classes=4, epochs=2, lr=1e-2, batch_size=32)
    assert isinstance(model, torch.nn.Linear)
    assert model.out_features == 4


def test_evaluate_probe_accuracy_in_bounds():
    X, y = _toy_classification_data()
    model = train_probe(X, y, num_classes=4, epochs=3, lr=1e-2, batch_size=32)
    acc = evaluate_probe(model, X, y)
    assert 0.0 <= acc <= 1.0


def test_evaluate_probe_better_than_random_on_separable():
    """Linear probe should learn a separable toy problem above random."""
    rng = np.random.default_rng(0)
    n, dim, n_classes = 600, 8, 4
    # Make class labels correlate with first dim
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) * 2  # 0 or 2; add 1 for 1,3 noise
    y[X[:, 0] < -0.5] = 1
    y[X[:, 0] > 0.5] = 3
    model = train_probe(X, y, num_classes=4, epochs=10, lr=1e-1, batch_size=64)
    acc = evaluate_probe(model, X, y)
    assert acc > 0.4  # clearly above 1/4 = 0.25 random
```

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_dataset_quality/test_modality_probe.py -v -k "to_tensor or train_probe or evaluate_probe"`
Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement probe training + evaluation**

Append to `framework/eval/dataset_quality/modality_probe.py`:

```python
import torch
import torch.nn.functional as F


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32)


def train_probe(X: np.ndarray, y: np.ndarray, num_classes: int,
                epochs: int = 20, lr: float = 1e-3,
                batch_size: int = 256, device: str = "cpu") -> torch.nn.Linear:
    """Train a single Linear layer with Adam + cross-entropy.

    Returns the trained Linear module (eval mode).
    """
    in_dim = X.shape[1]
    model = torch.nn.Linear(in_dim, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = _to_tensor(X).to(device)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    model.to(device)
    n = X_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_probe(model: torch.nn.Linear, X: np.ndarray, y: np.ndarray,
                   device: str = "cpu", batch_size: int = 1024) -> float:
    """Return top-1 accuracy."""
    model.eval()
    X_t = _to_tensor(X).to(device)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    correct, total = 0, 0
    for i in range(0, X_t.shape[0], batch_size):
        logits = model(X_t[i:i + batch_size])
        pred = logits.argmax(dim=-1)
        correct += (pred == y_t[i:i + batch_size]).sum().item()
        total += pred.shape[0]
    return correct / max(total, 1)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_modality_probe.py -v`
Expected: 9 passed (5 from Task 2 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/modality_probe.py tests/test_dataset_quality/test_modality_probe.py
git commit -m "feat(dq-eval): Linear probe training and evaluation"
```

---

## Task 4: InfoScore 公式（clipping）

**Files:**
- Modify: `framework/eval/dataset_quality/modality_probe.py`
- Modify: `tests/test_dataset_quality/test_modality_probe.py`

- [ ] **Step 1: Add failing tests for InfoScore**

Append to `tests/test_dataset_quality/test_modality_probe.py`:

```python
from framework.eval.dataset_quality.modality_probe import compute_info_score


def test_info_score_basic():
    acc_per_modality = {"rgb": 0.8, "depth": 0.5, "lidar": 0.6,
                        "mmwave": 0.4, "wifi": 0.1}
    acc_concat = 0.9
    res = compute_info_score(acc_per_modality, acc_concat,
                             w_per_modality=0.7, w_complement=0.3)
    assert res["mean_acc"] == 0.48
    assert res["complement_gain"] == 0.5  # 0.9 - 0.8
    assert 0.0 <= res["InfoScore"] <= 1.0


def test_info_score_clips_negative_gain():
    """Complement gain < 0 should be clipped to 0, not subtracted."""
    acc_per_modality = {"rgb": 0.9}
    acc_concat = 0.5  # worse than single modality
    res = compute_info_score(acc_per_modality, acc_concat,
                             w_per_modality=0.7, w_complement=0.3)
    assert res["complement_gain"] == -0.4
    assert res["InfoScore"] == pytest.approx(0.7 * 0.9)  # only per-modality


def test_info_score_clamps_to_unit_interval():
    """High gain must not push InfoScore above 1."""
    acc_per_modality = {"a": 0.5}
    acc_concat = 1.0
    res = compute_info_score(acc_per_modality, acc_concat,
                             w_per_modality=0.7, w_complement=0.3)
    assert res["InfoScore"] <= 1.0
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run new tests, verify they fail**

Run: `pytest tests/test_dataset_quality/test_modality_probe.py -v -k "info_score"`
Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement InfoScore formula**

Append to `framework/eval/dataset_quality/modality_probe.py`:

```python
from typing import Dict


def compute_info_score(acc_per_modality: Dict[str, float],
                       acc_concat: float,
                       w_per_modality: float = 0.7,
                       w_complement: float = 0.3) -> Dict[str, float]:
    """Bounded InfoScore per spec.

    InfoScore = w_per_modality * mean(acc_per_modality)
              + w_complement * clamp(complement_gain, 0, 1 - mean(acc_per_modality))
    """
    mean_acc = sum(acc_per_modality.values()) / max(len(acc_per_modality), 1)
    complement_gain = acc_concat - max(acc_per_modality.values()
                                       or [0.0])
    clipped_gain = max(0.0, min(complement_gain, max(0.0, 1.0 - mean_acc)))
    info = w_per_modality * mean_acc + w_complement * clipped_gain
    return {"mean_acc": mean_acc,
            "complement_gain": complement_gain,
            "InfoScore": float(min(1.0, max(0.0, info)))}
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_modality_probe.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/modality_probe.py tests/test_dataset_quality/test_modality_probe.py
git commit -m "feat(dq-eval): InfoScore with clipping + range clamp"
```

---

## Task 5: 紧致度 CompactScore

**Files:**
- Create: `framework/eval/dataset_quality/compactness.py`
- Create: `tests/test_dataset_quality/test_compactness.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset_quality/test_compactness.py
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
    y_pred = np.array([1, 2, 0])  # all wrong but cyclic
    rate = compute_confusion_rate(y_true, y_pred, num_classes=3)
    assert rate == pytest.approx(1.0)


def test_confusion_rate_half_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # 2/4 wrong
    rate = compute_confusion_rate(y_true, y_pred, num_classes=2)
    assert rate == pytest.approx(0.5)


def test_fisher_ratio_separable_higher():
    rng = np.random.default_rng(0)
    # Class 0 cluster around -2, class 1 around +2 on first axis
    n, dim = 200, 4
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X[:n // 2, 0] -= 2
    X[n // 2:, 0] += 2
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    r_sep = compute_fisher_ratio(X, y)
    # Now mess it up: random labels
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_dataset_quality/test_compactness.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement compactness module**

```python
# framework/eval/dataset_quality/compactness.py
"""Compactness evaluation: confusion rate (main), Fisher ratio, leave-one-out distance (diagnostic)."""
from __future__ import annotations
from typing import Tuple

import numpy as np


def compute_confusion_rate(y_true: np.ndarray, y_pred: np.ndarray,
                           num_classes: int) -> float:
    """Off-diagonal fraction of the confusion matrix. ∈ [0, 1]."""
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
    """For each sample, distance to its predicted class centroid (excluding itself).

    Diagnostic only (90th percentile reported alongside).
    """
    X = X.astype(np.float32)
    classes = np.unique(y)
    centroids = {c: X[y == c].mean(axis=0) for c in classes}
    return np.linalg.norm(X - np.array([centroids[y[i]] for i in range(len(y))]), axis=1)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_compactness.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/compactness.py tests/test_dataset_quality/test_compactness.py
git commit -m "feat(dq-eval): compactness module (confusion rate, Fisher, LOO dist)"
```

---

## Task 6: 纯净度 Cleanliness（异常率 + JS 散度 + 量化 hash）

**Files:**
- Create: `framework/eval/dataset_quality/cleanliness.py`
- Create: `tests/test_dataset_quality/test_cleanliness.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset_quality/test_cleanliness.py
import numpy as np
import pytest
from framework.eval.dataset_quality.cleanliness import (
    compute_anomaly_rate, jensen_shannon, compute_inconsistency_rate,
    compute_dup_rate_quantized,
)


def test_anomaly_rate_with_threshold():
    # 4 samples, true-class probs = [0.9, 0.5, 0.2, 0.05]
    # anomaly_score = 1 - prob; threshold 0.3 → flags scores > 0.3
    probs = np.array([0.9, 0.5, 0.2, 0.05])
    y_true = np.array([0, 1, 2, 3])
    rate = compute_anomaly_rate(probs, y_true, anomaly_threshold=0.3)
    # scores [0.1, 0.5, 0.8, 0.95]; > 0.3 → 3 flagged
    assert rate == pytest.approx(0.75)


def test_anomaly_rate_all_correct():
    probs = np.array([1.0, 1.0, 1.0])
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
    # 4 samples, 2 modalities. JS values [0.05, 0.2, 0.4, 0.6].
    # threshold 0.3 → 2 flagged → rate 0.5
    js_per_sample = np.array([0.05, 0.2, 0.4, 0.6])
    rate = compute_inconsistency_rate(js_per_sample, js_threshold=0.3)
    assert rate == 0.5


def test_dup_rate_quantized_identifies_duplicates():
    feats = np.array([
        [1.234, 2.345],
        [1.234, 2.345],   # exact duplicate
        [1.001, 2.001],   # near-duplicate (within 2 decimals)
        [5.678, 9.012],   # distinct
    ], dtype=np.float32)
    rate = compute_dup_rate_quantized(feats, decimals=2)
    # 4 samples, 1 duplicate pair → 2/4
    assert rate == pytest.approx(0.5)


def test_dup_rate_quantized_no_dups():
    feats = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rate = compute_dup_rate_quantized(feats, decimals=2)
    assert rate == 0.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_dataset_quality/test_cleanliness.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement cleanliness module**

```python
# framework/eval/dataset_quality/cleanliness.py
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
    # Use tuple-of-row as hash key
    keys = [tuple(r) for r in rounded]
    seen = {}
    dup_count = 0
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    total_dups = sum(c for c in seen.values() if c > 1)
    return float(total_dups / len(keys))
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_cleanliness.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/cleanliness.py tests/test_dataset_quality/test_cleanliness.py
git commit -m "feat(dq-eval): cleanliness module (anomaly, JS, dup)"
```

---

## Task 7: Report 模块（JSON metadata + 诊断图）

**Files:**
- Create: `framework/eval/dataset_quality/report.py`
- Create: `tests/test_dataset_quality/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset_quality/test_report.py
import json
import os
import numpy as np
from framework.eval.dataset_quality.report import (
    build_metadata, assemble_report, write_report_json,
)


def _toy_args():
    return {
        "dataset": "datasets/mmfi/v4",
        "eval_split": "val",
        "num_classes": 27,
        "probe_epochs": 20,
        "probe_lr": 1e-3,
        "probe_batch_size": 256,
        "anomaly_threshold": 0.3,
        "js_threshold": 0.1,
        "hash_decimals": 2,
        "dup_weight": 0.5,
        "w_info": 0.4,
        "w_compact": 0.4,
        "w_clean": 0.2,
        "info_weights": {"per_modality": 0.7, "complement": 0.3},
        "val_sample_count": 3500,
        "train_sample_count": 46509,
    }


def test_build_metadata_includes_all_keys():
    md = build_metadata(_toy_args())
    for k in ["num_classes", "probe_epochs", "anomaly_threshold",
              "js_threshold", "hash_decimals", "w_info", "w_compact",
              "w_clean", "val_sample_count"]:
        assert k in md, f"missing {k}"


def test_assemble_report_includes_all_scores():
    info = {"mean_acc": 0.5, "complement_gain": 0.1, "InfoScore": 0.4}
    compact = {"confusion_rate": 0.3, "CompactScore": 0.7,
               "fisher_ratio": 1.5, "leave_one_out_dist_p90": 2.1}
    clean = {"anomaly_rate": 0.1, "inconsistency_rate": 0.05,
             "dup_rate": 0.02, "CleanScore": 0.9}
    rep = assemble_report(_toy_args(), info, compact, clean)
    assert rep["quality"] == pytest.approx(0.4 * 0.4 + 0.4 * 0.7 + 0.2 * 0.9)
    for k in ["info", "compact", "clean", "metadata", "quality"]:
        assert k in rep


def test_write_report_json(tmp_path):
    rep = assemble_report(_toy_args(),
                          {"mean_acc": 0.5, "complement_gain": 0.1, "InfoScore": 0.4},
                          {"confusion_rate": 0.3, "CompactScore": 0.7,
                           "fisher_ratio": 1.5, "leave_one_out_dist_p90": 2.1},
                          {"anomaly_rate": 0.1, "inconsistency_rate": 0.05,
                           "dup_rate": 0.02, "CleanScore": 0.9})
    out = tmp_path / "quality.json"
    write_report_json(rep, str(out))
    loaded = json.loads(out.read_text())
    assert loaded["quality"] == rep["quality"]
```

Add `import pytest` to the test file.

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_dataset_quality/test_report.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement report module**

```python
# framework/eval/dataset_quality/report.py
"""JSON metadata assembly + matplotlib diagnostic plots."""
from __future__ import annotations

import json
import os
from typing import Dict


def build_metadata(args: Dict) -> Dict:
    """Persist all hyperparameters and weights for reproducibility."""
    return {
        "dataset": args.get("dataset"),
        "eval_split": args.get("eval_split"),
        "num_classes": args.get("num_classes"),
        "probe_epochs": args.get("probe_epochs"),
        "probe_lr": args.get("probe_lr"),
        "probe_batch_size": args.get("probe_batch_size"),
        "anomaly_threshold": args.get("anomaly_threshold"),
        "js_threshold": args.get("js_threshold"),
        "hash_decimals": args.get("hash_decimals"),
        "dup_weight": args.get("dup_weight"),
        "w_info": args.get("w_info"),
        "w_compact": args.get("w_compact"),
        "w_clean": args.get("w_clean"),
        "info_weights": args.get("info_weights"),
        "val_sample_count": args.get("val_sample_count"),
        "train_sample_count": args.get("train_sample_count"),
    }


def assemble_report(args: Dict, info: Dict, compact: Dict,
                    clean: Dict) -> Dict:
    quality = (args.get("w_info", 0.4) * info["InfoScore"]
               + args.get("w_compact", 0.4) * compact["CompactScore"]
               + args.get("w_clean", 0.2) * clean["CleanScore"])
    return {
        "dataset": args.get("dataset"),
        "metadata": build_metadata(args),
        "info": info,
        "compact": compact,
        "clean": clean,
        "quality": float(quality),
    }


def write_report_json(report: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def plot_per_modality_acc(acc_per_modality: Dict[str, float],
                          out_path: str) -> None:
    """Bar chart of per-modality accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = list(acc_per_modality.items())
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(6, 4))
    plt.bar(names, vals)
    plt.ylabel("val top-1 acc")
    plt.title("Per-modality probe accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm: list, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(cm), cmap="Blues")
    plt.colorbar()
    plt.title("Confusion matrix (concat probe, val)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_js_histogram(js_per_sample, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.hist(js_per_sample, bins=30)
    plt.xlabel("JS divergence")
    plt.ylabel("count")
    plt.title("Cross-modal JS distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_report.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/report.py tests/test_dataset_quality/test_report.py
git commit -m "feat(dq-eval): report module (JSON metadata + plots)"
```

---

## Task 8: Leaderboard 跨版本聚合

**Files:**
- Create: `framework/eval/dataset_quality/leaderboard.py`
- Create: `tests/test_dataset_quality/test_leaderboard.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dataset_quality/test_leaderboard.py
import json
import os
from framework.eval.dataset_quality.leaderboard import (
    load_reports, render_markdown, aggregate_quality,
)


def _write_report(path, dataset, info, compact, clean, quality):
    rep = {
        "dataset": dataset,
        "metadata": {"val_sample_count": 100, "num_classes": 27},
        "info": info,
        "compact": compact,
        "clean": clean,
        "quality": quality,
    }
    with open(path, "w") as f:
        json.dump(rep, f)


def test_load_reports(tmp_path):
    p1 = tmp_path / "q1.json"
    p2 = tmp_path / "q2.json"
    _write_report(str(p1), "v1", {"InfoScore": 0.3},
                  {"CompactScore": 0.4}, {"CleanScore": 0.5}, 0.4)
    _write_report(str(p2), "v2", {"InfoScore": 0.5},
                  {"CompactScore": 0.6}, {"CleanScore": 0.7}, 0.6)
    reps = load_reports([str(p1), str(p2)])
    assert "v1" in reps and "v2" in reps


def test_render_markdown(tmp_path):
    reports = {
        "v1": {"info": {"InfoScore": 0.3, "acc_per_modality": {}},
               "compact": {"CompactScore": 0.4},
               "clean": {"CleanScore": 0.5}, "quality": 0.4},
        "v2": {"info": {"InfoScore": 0.5, "acc_per_modality": {}},
               "compact": {"CompactScore": 0.6},
               "clean": {"CleanScore": 0.7}, "quality": 0.6},
    }
    md = render_markdown(reports)
    assert "v1" in md and "v2" in md
    assert "InfoScore" in md and "Quality" in md
    assert md.startswith("# ")


def test_aggregate_quality_uses_weights():
    reports = {"v1": {"quality": 0.5}, "v2": {"quality": 0.7}}
    scores = aggregate_quality(reports)
    assert scores == {"v1": 0.5, "v2": 0.7}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_dataset_quality/test_leaderboard.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement leaderboard module**

```python
# framework/eval/dataset_quality/leaderboard.py
"""Cross-version aggregation and markdown rendering."""
from __future__ import annotations

import json
from typing import Dict, List


def load_reports(paths: List[str]) -> Dict[str, Dict]:
    out = {}
    for p in paths:
        with open(p) as f:
            rep = json.load(f)
        out[rep["dataset"]] = rep
    return out


def aggregate_quality(reports: Dict[str, Dict]) -> Dict[str, float]:
    return {k: v["quality"] for k, v in reports.items()}


def render_markdown(reports: Dict[str, Dict]) -> str:
    """Render a Markdown table comparing versions."""
    lines = ["# Dataset Quality Leaderboard", ""]
    header = "| dataset | InfoScore | CompactScore | CleanScore | Quality |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for name, rep in reports.items():
        lines.append(
            f"| {name} | {rep['info']['InfoScore']:.3f} | "
            f"{rep['compact']['CompactScore']:.3f} | "
            f"{rep['clean']['CleanScore']:.3f} | "
            f"{rep['quality']:.3f} |"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_leaderboard.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add framework/eval/dataset_quality/leaderboard.py tests/test_dataset_quality/test_leaderboard.py
git commit -m "feat(dq-eval): leaderboard aggregation + markdown render"
```

---

## Task 9: 端到端 run_dataset_quality.py 整合 + 集成测试

**Files:**
- Modify: `scripts/run_dataset_quality.py`
- Create: `tests/test_dataset_quality/test_run_e2e.py`

- [ ] **Step 1: Write failing e2e test**

```python
# tests/test_dataset_quality/test_run_e2e.py
import json
import os
import numpy as np
import pytest
from framework.dataset.sample import Sample, Modality


def _toy_dataset(tmp_path, n=80, dim=8, n_classes=4, seed=0):
    """Build a tiny synthetic dataset on disk mimicking v4 layout."""
    rng = np.random.default_rng(seed)
    root = tmp_path / "toy_v0"
    data_dir = root / "data"
    splits_dir = root / "splits"
    data_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    samples = []
    ids = []
    for i in range(n):
        sid = f"S{i:03d}"
        ids.append(sid)
        # Class-correlated signal in feature
        cls = i % n_classes
        feats = rng.normal(size=(5, dim)).astype(np.float32)
        feats[:, 0] += cls * 2.0
        mods = {
            "rgb": Modality(data=feats, frame_indices=[0, 1, 2, 3, 4]),
            "depth": Modality(data=feats[:, :4], frame_indices=[0, 1, 2, 3, 4]),
            "lidar": Modality(data=feats[:, :3], frame_indices=[0, 1, 2, 3, 4]),
            "mmwave": Modality(data=feats[:, :5], frame_indices=[0, 1, 2, 3, 4]),
            "wifi": Modality(data=feats[:, :6], frame_indices=[0, 1, 2, 3, 4]),
        }
        sample = Sample(id=sid, label=cls, modalities=mods)
        samples.append(sample)
    # Save individual pkl
    import pickle
    for s in samples:
        with open(data_dir / f"{s.id}.pkl", "wb") as f:
            pickle.dump(s.to_dict(), f)
    # splits
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    (splits_dir / "train.json").write_text(json.dumps(ids[:n_train]))
    (splits_dir / "val.json").write_text(json.dumps(ids[n_train:n_train + n_val]))
    (splits_dir / "test.json").write_text(json.dumps(ids[n_train + n_val:]))
    (root / "meta.json").write_text(json.dumps({
        "name": "toy", "version": "v0",
        "modalities": ["rgb", "depth", "lidar", "mmwave", "wifi"],
    }))
    return str(root)


def test_run_end_to_end(tmp_path):
    from scripts.run_dataset_quality import run
    root = _toy_dataset(tmp_path)
    out = tmp_path / "quality.json"
    run(root, str(out), num_classes=4, epochs=5, batch_size=16)
    rep = json.loads(out.read_text())
    assert "info" in rep and "compact" in rep and "clean" in rep
    assert 0.0 <= rep["quality"] <= 1.0
    assert rep["metadata"]["val_sample_count"] > 0


def test_run_rejects_test_split(tmp_path):
    from scripts.run_dataset_quality import run
    root = _toy_dataset(tmp_path)
    with pytest.raises(AssertionError):
        run(root, str(tmp_path / "q.json"), eval_split="test", num_classes=4)
```

- [ ] **Step 2: Run e2e test, verify it fails**

Run: `pytest tests/test_dataset_quality/test_run_e2e.py -v`
Expected: ImportError on `run`.

- [ ] **Step 3: Implement run() in entry script**

Replace `scripts/run_dataset_quality.py` with:

```python
#!/usr/bin/env python
"""Dataset quality eval entry (P0-P4 of dataset-quality-eval-design spec)."""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality.modality_probe import (
    MODALITY_ORDER, stack_split, train_probe, evaluate_probe,
    compute_info_score, extract_modality_feature, extract_concat_feature,
)
from framework.eval.dataset_quality.compactness import (
    compute_confusion_rate, compute_fisher_ratio, compute_leave_one_out_distances,
)
from framework.eval.dataset_quality.cleanliness import (
    compute_anomaly_rate, jensen_shannon, compute_inconsistency_rate,
    compute_dup_rate_quantized,
)
from framework.eval.dataset_quality.report import (
    assemble_report, write_report_json,
    plot_per_modality_acc, plot_confusion_matrix, plot_js_histogram,
)

ALLOWED_EVAL_SPLITS = {"train", "val"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-split", choices=sorted(ALLOWED_EVAL_SPLITS),
                    default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-classes", type=int, default=27)
    ap.add_argument("--probe-epochs", type=int, default=20)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-batch-size", type=int, default=256)
    ap.add_argument("--anomaly-threshold", type=float, default=0.3)
    ap.add_argument("--js-threshold", type=float, default=0.1)
    ap.add_argument("--hash-decimals", type=int, default=2)
    ap.add_argument("--dup-weight", type=float, default=0.5)
    ap.add_argument("--w-info", type=float, default=0.4)
    ap.add_argument("--w-compact", type=float, default=0.4)
    ap.add_argument("--w-clean", type=float, default=0.2)
    ap.add_argument("--info-w-per-modality", type=float, default=0.7)
    ap.add_argument("--info-w-complement", type=float, default=0.3)
    ap.add_argument("--plots-dir", default=None)
    return ap.parse_args(argv)


def validate_splits(args):
    assert args.eval_split != "test", \
        "test split cannot be used for probe evaluation (P0 guard)"
    assert args.eval_split in ALLOWED_EVAL_SPLITS


def run(dataset_root: str, out_path: str, eval_split: str = "val",
        num_classes: int = 27, epochs: int = 20, lr: float = 1e-3,
        batch_size: int = 256, anomaly_threshold: float = 0.3,
        js_threshold: float = 0.1, hash_decimals: int = 2,
        dup_weight: float = 0.5, w_info: float = 0.4,
        w_compact: float = 0.4, w_clean: float = 0.2,
        info_w_per_modality: float = 0.7, info_w_complement: float = 0.3,
        plots_dir=None, device: str = "cpu") -> Dict:
    """End-to-end run. Returns the assembled report dict."""
    assert eval_split != "test", "P0 guard: test split cannot be probed"
    ds = load_dataset(dataset_root)
    train_samples = list(ds.train)
    eval_samples = list(ds.val if eval_split == "val" else ds.train)
    available_modalities = [m for m in MODALITY_ORDER
                            if any(m in s.modalities for s in train_samples)]
    print(f"[dq] modalities={available_modalities} "
          f"train={len(train_samples)} eval={len(eval_samples)}")

    # --- Dimension 1: info ---
    acc_per_modality = {}
    for m in available_modalities:
        X_tr, y_tr = stack_split(train_samples, [m])
        X_ev, y_ev = stack_split(eval_samples, [m])
        model = train_probe(X_tr[m], y_tr, num_classes=num_classes,
                            epochs=epochs, lr=lr, batch_size=batch_size,
                            device=device)
        acc_per_modality[m] = evaluate_probe(model, X_ev[m], y_ev, device=device)

    X_tr_concat, y_tr_concat = stack_split(train_samples, available_modalities,
                                           concat=True)
    X_ev_concat, y_ev_concat = stack_split(eval_samples, available_modalities,
                                           concat=True)
    concat_model = train_probe(X_tr_concat["concat"], y_tr_concat,
                               num_classes=num_classes,
                               epochs=epochs, lr=lr, batch_size=batch_size,
                               device=device)
    acc_concat = evaluate_probe(concat_model, X_ev_concat["concat"],
                                y_ev_concat, device=device)

    info = compute_info_score(acc_per_modality, acc_concat,
                              w_per_modality=info_w_per_modality,
                              w_complement=info_w_complement)
    info.update({"acc_per_modality": acc_per_modality, "acc_concat": acc_concat})

    # --- Dimension 2: compact ---
    with torch.no_grad():
        Xt = torch.as_tensor(X_ev_concat["concat"], dtype=torch.float32)
        preds = concat_model(Xt).argmax(dim=-1).numpy()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_ev_concat, preds):
        cm[t, p] += 1
    confusion_rate = compute_confusion_rate(y_ev_concat, preds, num_classes)
    fisher = compute_fisher_ratio(X_ev_concat["concat"], y_ev_concat)
    loo = compute_leave_one_out_distances(X_ev_concat["concat"], y_ev_concat)
    compact = {
        "confusion_matrix": cm.tolist(),
        "confusion_rate": confusion_rate,
        "CompactScore": float(1.0 - confusion_rate),
        "fisher_ratio": fisher,
        "leave_one_out_dist_p90": float(np.percentile(loo, 90)),
    }

    # --- Dimension 3: clean ---
    # anomaly rate: train probs
    with torch.no_grad():
        Xt_tr = torch.as_tensor(X_tr_concat["concat"], dtype=torch.float32)
        train_probs = torch.softmax(concat_model(Xt_tr), dim=-1).numpy()
    anomaly_rate = compute_anomaly_rate(train_probs, y_tr_concat,
                                        anomaly_threshold=anomaly_threshold)
    # cross-modal JS on val
    js_per_sample = []
    val_per_mod_probs = {}
    for m in available_modalities:
        X_ev_m, _ = stack_split(eval_samples, [m])
        with torch.no_grad():
            Xt_m = torch.as_tensor(X_ev_m[m], dtype=torch.float32)
            m_model = train_probe(*stack_split(train_samples, [m]),
                                 num_classes=num_classes,
                                 epochs=epochs, lr=lr, batch_size=batch_size,
                                 device=device)
            val_per_mod_probs[m] = torch.softmax(m_model(Xt_m), dim=-1).numpy()
    for i in range(len(eval_samples)):
        js_vals = []
        for m1, m2 in combinations(available_modalities, 2):
            js_vals.append(jensen_shannon(val_per_mod_probs[m1][i],
                                          val_per_mod_probs[m2][i]))
        js_per_sample.append(np.mean(js_vals))
    js_per_sample = np.array(js_per_sample)
    inconsistency_rate = compute_inconsistency_rate(js_per_sample,
                                                     js_threshold=js_threshold)
    dup_rate = compute_dup_rate_quantized(X_ev_concat["concat"],
                                          decimals=hash_decimals)
    # Weight dup by dup_weight (per spec: lower dup weight to reduce noise)
    eff_clean = (anomaly_rate
                 + inconsistency_rate
                 + dup_weight * dup_rate) / (1 + dup_weight)
    clean = {
        "anomaly_rate": anomaly_rate,
        "inconsistency_rate": inconsistency_rate,
        "dup_rate": dup_rate,
        "CleanScore": float(max(0.0, min(1.0, 1.0 - eff_clean))),
    }

    args_dict = {
        "dataset": dataset_root, "eval_split": eval_split,
        "num_classes": num_classes, "probe_epochs": epochs,
        "probe_lr": lr, "probe_batch_size": batch_size,
        "anomaly_threshold": anomaly_threshold,
        "js_threshold": js_threshold, "hash_decimals": hash_decimals,
        "dup_weight": dup_weight, "w_info": w_info,
        "w_compact": w_compact, "w_clean": w_clean,
        "info_weights": {"per_modality": info_w_per_modality,
                         "complement": info_w_complement},
        "val_sample_count": len(eval_samples),
        "train_sample_count": len(train_samples),
    }
    report = assemble_report(args_dict, info, compact, clean)
    write_report_json(report, out_path)

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        plot_per_modality_acc(acc_per_modality,
                              os.path.join(plots_dir, "per_modality_acc.png"))
        plot_confusion_matrix(cm.tolist(),
                              os.path.join(plots_dir, "confusion_matrix.png"))
        plot_js_histogram(js_per_sample,
                          os.path.join(plots_dir, "cross_modal_js.png"))
    return report


def main():
    args = parse_args()
    validate_splits(args)
    device = "cuda" if (args.probe_batch_size  # noqa
                         and torch.cuda.is_available()) else "cpu"
    run(args.dataset, args.out, eval_split=args.eval_split,
        num_classes=args.num_classes, epochs=args.probe_epochs,
        lr=args.probe_lr, batch_size=args.probe_batch_size,
        anomaly_threshold=args.anomaly_threshold,
        js_threshold=args.js_threshold, hash_decimals=args.hash_decimals,
        dup_weight=args.dup_weight, w_info=args.w_info,
        w_compact=args.w_compact, w_clean=args.w_clean,
        info_w_per_modality=args.info_w_per_modality,
        info_w_complement=args.info_w_complement,
        plots_dir=args.plots_dir, device=device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run e2e tests, verify they pass**

Run: `pytest tests/test_dataset_quality/test_run_e2e.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full dq test suite, verify all pass**

Run: `pytest tests/test_dataset_quality/ -v`
Expected: All tests pass (~33 tests across 7 files).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_dataset_quality.py tests/test_dataset_quality/test_run_e2e.py
git commit -m "feat(dq-eval): end-to-end pipeline + e2e tests on toy dataset"
```

---

## Task 10: 跑 v4 → results/quality_v4.json + 诊断图

**Files:**
- Create: `results/quality_v4.json`
- Create: `results/plots_v4/{per_modality_acc,confusion_matrix,cross_modal_js}.png`

- [ ] **Step 1: Pre-flight resource check**

```bash
free -h | head -2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Confirm ≥ 8GB RAM available and GPU/CPU choice.

- [ ] **Step 2: Launch v4 eval in background**

```bash
setsid bash -c '/home/li/miniconda/envs/rfgen/bin/python scripts/run_dataset_quality.py \
  --dataset datasets/mmfi/v4 \
  --out results/quality_v4.json \
  --plots-dir results/plots_v4 \
  --probe-epochs 10 \
  --probe-batch-size 256 \
  > logs/quality_v4.log 2>&1' < /dev/null > /dev/null 2>&1 &
```

Estimate: 5 Linear probes × ~30s each + concat-Linear ~2min + JS ~1min ≈ ~5 min total.

- [ ] **Step 3: Monitor (poll every 30s)**

```bash
while ps -p $PID > /dev/null 2>&1; do
  sleep 30
  tail -5 logs/quality_v4.log
  free -h | head -2
done
```

Watch for: OOM (RAM swap), process death, log stall > 2min.

- [ ] **Step 4: Verify output**

```bash
cat results/quality_v4.json | python -m json.tool | head -50
ls -la results/plots_v4/
```

Confirm: quality ∈ [0,1], all three scores present, 5 per-modality acc, plots written.

- [ ] **Step 5: Commit**

```bash
git add results/quality_v4.json results/plots_v4/ logs/quality_v4.log
git commit -m "feat(dq-eval): v4 baseline quality scores"
```

---

## Task 11: 跑 v1 + v2 → leaderboard_quality.md

**Files:**
- Create: `results/quality_v1.json`
- Create: `results/quality_v2.json`
- Create: `leaderboard_quality.md`

- [ ] **Step 1: Run v1 eval**

```bash
setsid bash -c '/home/li/miniconda/envs/rfgen/bin/python scripts/run_dataset_quality.py \
  --dataset datasets/mmfi/v1 \
  --out results/quality_v1.json \
  --plots-dir results/plots_v1 \
  --probe-epochs 10 \
  --probe-batch-size 256 \
  > logs/quality_v1.log 2>&1' < /dev/null > /dev/null 2>&1 &
```

Monitor as in Task 10.

- [ ] **Step 2: Run v2 eval**

```bash
setsid bash -c '/home/li/miniconda/envs/rfgen/bin/python scripts/run_dataset_quality.py \
  --dataset datasets/mmfi/v2 \
  --out results/quality_v2.json \
  --plots-dir results/plots_v2 \
  --probe-epochs 10 \
  --probe-batch-size 256 \
  > logs/quality_v2.log 2>&1' < /dev/null > /dev/null 2>&1 &
```

- [ ] **Step 3: Generate leaderboard**

```bash
/home/li/miniconda/envs/rfgen/bin/python -c "
from framework.eval.dataset_quality.leaderboard import load_reports, render_markdown
reps = load_reports([
  'results/quality_v1.json',
  'results/quality_v2.json',
  'results/quality_v4.json',
])
with open('leaderboard_quality.md', 'w') as f:
    f.write(render_markdown(reps))
print(open('leaderboard_quality.md').read())
"
```

- [ ] **Step 4: Verify leaderboard content**

Read `leaderboard_quality.md`. Confirm: 3 datasets listed, all four scores present, scores in [0, 1].

- [ ] **Step 5: Commit**

```bash
git add results/quality_v1.json results/quality_v2.json results/plots_v1/ results/plots_v2/ leaderboard_quality.md logs/quality_v1.log logs/quality_v2.log
git commit -m "feat(dq-eval): v1/v2 quality + cross-version leaderboard"
```

---

## Task 12: 数据质量演变报告 docs/reports/

**Files:**
- Create: `docs/reports/dataset_quality_v1_v2_v4.md`

- [ ] **Step 1: Draft report**

Write `docs/reports/dataset_quality_v1_v2_v4.md` with sections:

1. **背景与目标**：引用 spec，解释为何引入 dataset_quality
2. **方法**：三个维度公式 + 轻量 probe + P0 隔离
3. **结果**：复制 leaderboard 表
4. **per-modality 分析**：哪个模态从 v1→v4 提升最大、哪个一直是瓶颈
5. **与主流程 robustness 关系**：
   - 主流程 v4 评估：rgb+mmwave 强、wifi/depth/lidar 弱
   - dataset_quality v4：wifi/depth/lidar per-modality acc 数值是多少
   - 是否一致？若一致 → 验证评测独立；若不一致 → 暴露主流程的"被掩盖"问题
6. **结论**：dataset_quality 已成为数据飞轮的独立判据；下一步可指导 v5 数据改进方向

- [ ] **Step 2: Commit**

```bash
git add docs/reports/dataset_quality_v1_v2_v4.md
git commit -m "docs(dq-eval): v1→v4 data quality evolution report"
```

---

## Task 13: 更新 STATUS.md 与断点标记

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: Update STATUS.md judgment layer**

Mark P0-P4 complete in the milestone evidence list. Add a new milestone M7 (or add to existing) noting dataset-quality-eval as a deliverable. Update decision layer with:
- M6b 复测结论（已完成）
- 新 milestone: 数据集质量评测系统（已完成）
- 下一步行动（[提议]）：用 dataset_quality 指导 v5 数据改进方向

- [ ] **Step 2: Refresh facts layer**

Run: `python tools/project_status.py scan STATUS.md`

- [ ] **Step 3: Commit**

```bash
git add STATUS.md
git commit -m "docs(status): dataset-quality-eval milestone + v1/v2/v4 results"
```

---

## Verification Checklist

After all tasks complete, run:

```bash
pytest tests/test_dataset_quality/ -v          # all unit + e2e tests pass
cat leaderboard_quality.md                    # cross-version table populated
cat results/quality_v4.json | python -m json.tool | head -30   # structure valid
ls results/plots_v4/                          # 3 PNG diagnostic plots
```

Expected: ~33 tests pass, leaderboard shows v1/v2/v4 with all four scores in [0,1], plots written.

---

## Notes on Test Data Sizes

For real datasets, val size is ~3500 samples and full feature concat is ~59k dim. Linear probe is fast — `train_probe` on 46509 train × 59k concat float32 in batches of 256 should finish in ~2 minutes on CPU. Per-modality is smaller. JS computation adds ~30s. Total: ~5 min per dataset.

For tests, toy dataset is 80 samples × small dim — completes in seconds.