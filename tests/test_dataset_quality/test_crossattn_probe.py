"""Tests for PerModCrossAttnMLP: attention-based probe for weak modalities.

Structured features suit attention better than concat because each modality
becomes a "token" that learns to attend to others.
"""
import pytest
import torch

from framework.eval.dataset_quality.probe_fusion import (
    PerModCrossAttnMLP, train_probe_crossattn,
)


def _toy_slices():
    return {"rgb": 4, "depth": 8, "lidar": 16, "mmwave": 12, "wifi": 4}


def _toy_concat(N=20):
    return torch.randn(N, 4 + 8 + 16 + 12 + 4)


def _toy_slices_with_idx():
    return {"rgb": (0, 4), "depth": (4, 12), "lidar": (12, 28),
            "mmwave": (28, 40), "wifi": (40, 44)}


def test_model_creation():
    model = PerModCrossAttnMLP({"a": 4, "b": 8}, embed_dim=8,
                                 num_heads=2, hidden=16, num_classes=3)
    assert "a" in model.projs and "b" in model.projs
    assert model.attn.embed_dim == 8
    assert model.head[0].in_features == 8  # mean pool back to embed_dim


def test_forward_shape():
    model = PerModCrossAttnMLP({"a": 4, "b": 8}, embed_dim=8,
                                 num_heads=2, hidden=16, num_classes=3,
                                 dropout_p=0.0)
    model.eval()
    X = torch.randn(5, 12)
    slices = {"a": (0, 4), "b": (4, 12)}
    out = model(X, slices)
    assert out.shape == (5, 3)


def test_forward_with_modality_dropped():
    """avail[m]=False forces projection of m to zero."""
    model = PerModCrossAttnMLP({"a": 4, "b": 8}, embed_dim=8,
                                 num_heads=2, hidden=16, num_classes=3,
                                 dropout_p=0.0)
    model.eval()
    X = torch.randn(5, 12)
    slices = {"a": (0, 4), "b": (4, 12)}
    out_full = model(X, slices, avail={"a": True, "b": True})
    X_drop = X.clone(); X_drop[:, :4] = 0
    out_drop = model(X_drop, slices, avail={"a": False, "b": True})
    assert not torch.allclose(out_full, out_drop)
    # Setting raw input to zero != forcing projection zero
    X_keep = X.clone(); X_keep[:, :4] = 999.0
    out_proj = model(X_keep, slices, avail={"a": False, "b": True})
    X_zero = X.clone(); X_zero[:, :4] = 0
    out_both = model(X_zero, slices, avail={"a": False, "b": True})
    assert torch.allclose(out_proj, out_both, atol=1e-5)


def test_train_returns_model():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn(60, 28)
    y = torch.randint(0, 4, (60,))
    slices = {"rgb": (0, 4), "depth": (4, 12), "lidar": (12, 28)}
    model = train_probe_crossattn(X, y, slices, num_classes=4,
                                   embed_dim=16, num_heads=2,
                                   hidden=32, epochs=3, batch_size=16)
    assert isinstance(model, PerModCrossAttnMLP)
    assert model.head[-1].out_features == 4


def test_train_better_than_random():
    rng = torch.Generator().manual_seed(0)
    n, dim, n_classes = 400, 12, 3
    X = torch.randn(n, dim)
    y = (X[:, 0] > 0).long()
    slices = {"a": (0, 6), "b": (6, 12)}
    model = train_probe_crossattn(X, y, slices, num_classes=n_classes,
                                   embed_dim=16, num_heads=2,
                                   hidden=32, epochs=20, lr=1e-1,
                                   batch_size=32)
    model.eval()
    with torch.no_grad():
        preds = model(X, slices).argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    assert acc > 0.4  # above 1/3 random


def test_attention_weights_computable():
    """PerModCrossAttnMLP returns (B, M+1, M+1) attention weights (CLS + modalities)."""
    model = PerModCrossAttnMLP({"a": 4, "b": 8, "c": 12}, embed_dim=8,
                                 num_heads=2, hidden=16, num_classes=3,
                                 dropout_p=0.0)
    model.eval()
    X = torch.randn(2, 24)
    slices = {"a": (0, 4), "b": (4, 12), "c": (12, 24)}
    with torch.no_grad():
        weights = model.attention_weights(X, slices)
    # 3 modalities + 1 CLS token = 4 tokens
    assert weights.shape == (2, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-4)