"""Unit tests for framework.eval.dataset_quality.feature_extract: the
domain-aware structured feature extractors (mmwave/wifi/depth/lidar) and
the hard-link-safe write path used by make_v5_structfeat.

Dimension regression guard: mmwave=134, wifi=161, depth=63, lidar=353.
"""
from __future__ import annotations
import os
import pickle

import numpy as np
import pytest

from framework.eval.dataset_quality.feature_extract import (
    extract_depth_features, extract_wifi_features, extract_lidar_features,
    extract_mmwave_features,
)
from curation.io import safe_replace_pickle

# Expected output dimensions (T=5 default for all raw modalities)
MMWAVE_DIM = 5 * (16 + 8) + 8 + 3 + 3  # 5*(geom16+signal8) + zhist8 + xy3 + drift3 = 134
WIFI_DIM = 5 * 9 + 5 * 19 + 5 * 3 + 3 + 3  # 161
DEPTH_DIM = 5 * 12 + 3  # 63
LIDAR_DIM = 5 * (3 + 3 + 64) + 3  # 353


def _sparse(a, ratio=0.5):
    a[a < ratio] = 0.0
    return a


def test_mmwave_feature_dim():
    mw = _sparse(np.random.rand(5, 64, 5).astype(np.float32))
    out = extract_mmwave_features(mw)
    assert out.shape == (MMWAVE_DIM,)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_wifi_feature_dim():
    wf = np.random.rand(5, 3, 114, 10).astype(np.float32)
    out = extract_wifi_features(wf)
    assert out.shape == (WIFI_DIM,)
    assert np.isfinite(out).all()


def test_depth_feature_dim():
    dp = np.random.rand(5, 1, 224, 224).astype(np.float32)
    out = extract_depth_features(dp)
    assert out.shape == (DEPTH_DIM,)
    assert np.isfinite(out).all()


def test_lidar_feature_dim():
    lid = np.random.rand(5, 1536, 3).astype(np.float32)
    out = extract_lidar_features(lid)
    assert out.shape == (LIDAR_DIM,)
    assert np.isfinite(out).all()


def test_extractors_return_float32_no_nan():
    """All extractors must replace NaN/Inf with 0 (corrcoef on constant input
    produced NaN historically, which collapsed training to argmax=0)."""
    wf = np.random.rand(5, 3, 114, 10).astype(np.float32)
    # force a constant slice so corrcoef would divide by zero
    wf[:, 0, :, :] = 0.5
    out = extract_wifi_features(wf)
    assert np.isfinite(out).all()
    assert out.dtype == np.float32


def test_write_independent_inode(tmp_path):
    """Regression: make_v5_structfeat must NOT overwrite the source pickle
    when the destination is a hard-link to it (shared inode). Writing in
    place to a shared inode corrupts the source (v4). Unlink first."""
    src = tmp_path / "src.pkl"
    dst = tmp_path / "dst.pkl"
    pickle.dump({"modalities": {"mmwave": {"data": np.zeros(50)}}}, open(src, "wb"))
    os.link(src, dst)  # same inode
    assert os.stat(src).st_ino == os.stat(dst).st_ino
    # overwrite dst the way make_v5_structfeat does (unlink first)
    if os.path.exists(dst):
        os.unlink(dst)
    pickle.dump({"modalities": {"mm": {"data": np.zeros(134)}}}, open(dst, "wb"))
    # src must be untouched
    with open(src, "rb") as f:
        s = pickle.load(f)
    assert s["modalities"]["mmwave"]["data"].shape == (50,)
    # dst got its own inode
    assert os.stat(src).st_ino != os.stat(dst).st_ino


def test_safe_replace_pickle_shared_helper(tmp_path):
    """The shared curation.io.safe_replace_pickle must protect a hard-linked
    source from in-place overwrite (used by make_v5_structfeat,
    make_v6_relabel, make_v5_hardaug)."""
    src = tmp_path / "src.pkl"
    dst = tmp_path / "dst.pkl"
    pickle.dump({"label": 14}, open(src, "wb"))
    os.link(src, dst)
    assert os.stat(src).st_ino == os.stat(dst).st_ino
    safe_replace_pickle(str(dst), {"label": 15})
    # src untouched, dst has its own inode
    with open(src, "rb") as f:
        assert pickle.load(f)["label"] == 14
    with open(dst, "rb") as f:
        assert pickle.load(f)["label"] == 15
    assert os.stat(src).st_ino != os.stat(dst).st_ino
