"""Tests for MLP probe upgrade: standardization, depth downsampling, MLP."""
import numpy as np
import pytest
import torch

from framework.dataset.sample import Sample, Modality
from framework.eval.dataset_quality.modality_probe import (
    extract_modality_feature, extract_concat_feature, MODALITY_ORDER,
    stack_split, train_probe, evaluate_probe, _to_tensor,
    compute_info_score, standardize_features, downsample_depth,
    train_probe_mlp, extract_modality_feature_downsampled,
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
            "mmwave": Modality(data=np.random.rand(5, 64, 5).astype(np.float32),
                               frame_indices=[0, 1, 2, 3, 4]),
            "wifi": Modality(data=np.random.rand(5, 3, 114, 10).astype(np.float32),
                             frame_indices=[0, 1, 2, 3, 4]),
        },
    )


# --- 1. Standardization ---

def test_standardize_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(100, 3)).astype(np.float32)
    stats = {"mean": X.mean(axis=0), "std": X.std(axis=0)}
    _, Xs = standardize_features(X, stats)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-5)


def test_standardize_uses_provided_stats():
    """val features standardized with train stats, not val stats."""
    rng = np.random.default_rng(0)
    X_tr = rng.normal(loc=10, scale=1, size=(50, 2)).astype(np.float32)
    X_ev = rng.normal(loc=10, scale=1, size=(20, 2)).astype(np.float32)
    stats = {"mean": X_tr.mean(axis=0), "std": X_tr.std(axis=0)}
    _, Xs_ev = standardize_features(X_ev, stats)
    # val mean won't be exactly 0 (uses train stats), but should be near
    assert np.allclose(Xs_ev.mean(axis=0), (X_ev.mean(axis=0) - stats["mean"]) / stats["std"], atol=1e-5)


def test_standardize_zero_std_guarded():
    X = np.zeros((10, 3), np.float32)
    stats = {"mean": np.zeros(3), "std": np.zeros(3)}
    _, Xs = standardize_features(X, stats)
    assert np.isnan(Xs).sum() == 0
    assert np.allclose(Xs, 0)


def test_standardize_returns_stats():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    stats, _ = standardize_features(X)
    assert "mean" in stats and "std" in stats
    assert stats["mean"].shape == (4,)


# --- 2. Depth downsampling ---

def test_downsample_depth_reduces_dim():
    rng = np.random.default_rng(0)
    depth = rng.normal(size=(5, 224, 224)).astype(np.float32)
    out = downsample_depth(depth, pool=8)
    # 224/8 = 28 → 28*28 = 784
    assert out.shape == (5, 28, 28)


def test_downsample_depth_matches_pool():
    rng = np.random.default_rng(0)
    depth = rng.normal(size=(5, 224, 224)).astype(np.float32)
    out = downsample_depth(depth, pool=8)
    # max of first 8x8 block should match (max-pool)
    block = depth[0, :8, :8].max()
    assert np.allclose(out[0, 0, 0], block, atol=1e-6)


def test_extract_depth_downsampled_feature():
    s = _toy_sample()
    feat = extract_modality_feature_downsampled(s, "depth", pool=8)
    assert feat.shape == (28 * 28,)


def test_downsample_identity_pool1():
    rng = np.random.default_rng(0)
    depth = rng.normal(size=(5, 224, 224)).astype(np.float32)
    out = downsample_depth(depth, pool=1)
    assert out.shape == depth.shape


# --- 3. MLP probe ---

def test_train_probe_mlp_returns_module():
    X = np.random.rand(100, 16).astype(np.float32)
    y = np.random.randint(0, 4, size=100).astype(np.int64)
    model = train_probe_mlp(X, y, num_classes=4, epochs=2, lr=1e-2,
                            batch_size=32, hidden_dim=32)
    assert isinstance(model, torch.nn.Module)
    assert model.out_features == 4


def test_train_probe_mlp_better_than_random_on_separable():
    rng = np.random.default_rng(0)
    n, dim, n_classes = 600, 8, 4
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) * 2
    y[X[:, 0] < -0.5] = 1
    y[X[:, 0] > 0.5] = 3
    model = train_probe_mlp(X, y, num_classes=4, epochs=10, lr=1e-1,
                            batch_size=64, hidden_dim=32)
    acc = evaluate_probe(model, X, y)
    assert acc > 0.4


def test_train_probe_mlp_nonlinear_capacity():
    """MLP should beat Linear on XOR-like data (nonlinear separator needed)."""
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(size=(n, 2)).astype(np.float32)
    # XOR: class 0 if signs differ, class 1 if same
    y = np.zeros(n, dtype=np.int64)
    y[((X[:, 0] > 0) == (X[:, 1] > 0))] = 1
    X = np.hstack([X, rng.normal(size=(n, 2)).astype(np.float32)])
    model_mlp = train_probe_mlp(X, y, num_classes=2, epochs=50, lr=1e-1,
                                batch_size=128, hidden_dim=32)
    acc_mlp = evaluate_probe(model_mlp, X, y)
    model_lin = train_probe(X, y, num_classes=2, epochs=50, lr=1e-1,
                            batch_size=128)
    acc_lin = evaluate_probe(model_lin, X, y)
    assert acc_mlp > acc_lin
    assert acc_mlp > 0.7
