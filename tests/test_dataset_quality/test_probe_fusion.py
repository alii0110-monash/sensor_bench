"""Tests for PerModConcatMLP probe (per-modality projection + modality dropout)."""
import pytest
import torch

from framework.eval.dataset_quality.probe_fusion import (
    PerModConcatMLP, train_probe_fusion,
)


def _toy_slices():
    return {"rgb": 4, "depth": 8, "lidar": 16}


def _toy_concat_X(N=20):
    """Build a concat tensor matching _toy_slices() column order."""
    return torch.randn(N, 4 + 8 + 16)


def _toy_slices_with_idx():
    return {"rgb": (0, 4), "depth": (4, 12), "lidar": (12, 28)}


def test_model_creation_with_slice_dims():
    model = PerModConcatMLP({"rgb": 4, "depth": 8}, embed_dim=8,
                             hidden=16, num_classes=3)
    assert "rgb" in model.projs and "depth" in model.projs
    # projs is nn.Sequential(Linear, BatchNorm1d) when use_batchnorm=True
    assert model.projs["rgb"][0].out_features == 8
    assert model.head[0].in_features == 16  # 2 mods × 8


def test_forward_shape_all_modalities():
    model = PerModConcatMLP({"rgb": 4, "depth": 8}, embed_dim=8,
                             hidden=16, num_classes=3, dropout_p=0.0)
    model.eval()
    X = torch.randn(5, 12)
    slices = {"rgb": (0, 4), "depth": (4, 12)}
    out = model(X, slices)
    assert out.shape == (5, 3)


def test_forward_with_modality_dropped():
    """If avail[m]=False, projection of m is zeroed (not just raw input)."""
    model = PerModConcatMLP({"rgb": 4, "depth": 8}, embed_dim=8,
                             hidden=16, num_classes=3, dropout_p=0.0)
    model.eval()
    X = torch.randn(5, 12)
    slices = {"rgb": (0, 4), "depth": (4, 12)}
    out_full = model(X, slices, avail={"rgb": True, "depth": True})
    # zero the rgb raw input — should NOT equal dropping the projection
    X_drop_rgb = X.clone()
    X_drop_rgb[:, 0:4] = 0
    out_drop = model(X_drop_rgb, slices, avail={"rgb": True, "depth": True})
    # Both should be different from out_full
    assert not torch.allclose(out_full, out_drop)
    # avail=False forces zero projection regardless of raw input
    X_with_rgb = X.clone()
    X_with_rgb[:, 0:4] = 999.0
    out_proj_drop = model(X_with_rgb, slices, avail={"rgb": False, "depth": True})
    # Verify projection is forced zero by comparing to a case where raw rgb is also zero
    X_drop_both = X.clone()
    X_drop_both[:, 0:4] = 0
    out_drop_both = model(X_drop_both, slices, avail={"rgb": False, "depth": True})
    assert torch.allclose(out_proj_drop, out_drop_both, atol=1e-5)


def test_dropout_only_in_train_mode():
    model = PerModConcatMLP({"a": 4, "b": 4}, embed_dim=4, hidden=4,
                             num_classes=2, dropout_p=0.5)
    X = torch.randn(100, 8)
    slices = {"a": (0, 4), "b": (4, 8)}
    model.eval()
    out_eval = model(X, slices)
    model.train()
    out_train = model(X, slices)
    # Should differ (modality dropout active in train)
    assert not torch.allclose(out_eval, out_train)


def test_train_probe_fusion_returns_module():
    rng = torch.Generator().manual_seed(0)
    X = torch.randn(60, 28)
    y = torch.randint(0, 4, (60,))
    slices = {"rgb": (0, 4), "depth": (4, 12), "lidar": (12, 28)}
    model = train_probe_fusion(X, y, slices, num_classes=4, embed_dim=8,
                               hidden=16, epochs=3, batch_size=16)
    assert isinstance(model, PerModConcatMLP)
    assert model.head[-1].out_features == 4


def test_train_probe_fusion_better_than_random():
    rng = torch.Generator().manual_seed(0)
    n, dim, n_classes = 400, 12, 3
    # Class signal in first modality only
    X = torch.randn(n, dim)
    y = (X[:, 0] > 0).long()
    slices = {"a": (0, 6), "b": (6, 12)}
    model = train_probe_fusion(X, y, slices, num_classes=n_classes,
                               embed_dim=16, hidden=32, epochs=20, lr=1e-1,
                               batch_size=32)
    model.eval()
    with torch.no_grad():
        preds = model(X, slices).argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    assert acc > 0.7  # clearly above 1/3 random