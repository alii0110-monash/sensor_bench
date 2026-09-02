# tests/test_dataset_quality/test_finegrained.py
import re
import numpy as np
import pytest

from framework.dataset.sample import Sample, Modality
from framework.eval.dataset_quality import finegrained


def _make_sample_id(sid, label):
    """带真实 id 的样本（分组/信号用）。rgb 特征带 label 信号。"""
    feats = np.zeros((5, 4), dtype=np.float32)
    feats[:, 0] += float(label)  # 特征首维编码类别
    mods = {"rgb": Modality(data=feats, frame_indices=[0,1,2,3,4])}
    return Sample(id=sid, label=label, modalities=mods)


def _identity_feature(s):
    """恒等特征提取器（测试用）。"""
    return s.modalities["rgb"].data.reshape(-1).astype(np.float32)


def test_parse_sample_id():
    assert finegrained.parse_sample_id("E01_S01_A01_f1-7") == ("E01", "S01", "A01")


def test_is_variant():
    assert not finegrained.is_variant("E01_S01_A01_f1-7")
    assert finegrained.is_variant("E01_S01_A01_f105-110__aug1")


def test_group_by_class_subject_excludes_variants():
    samples = [
        _make_sample_id("E01_S01_A01_f1-7", 0),
        _make_sample_id("E01_S01_A01_f8-14", 0),
        _make_sample_id("E01_S01_A02_f1-7", 1),
        _make_sample_id("E01_S01_A01_f105-110__aug1", 0),  # 变体应排除
        _make_sample_id("E02_S02_A01_f1-7", 0),
    ]
    groups = finegrained.group_by_class_subject(samples)
    # 3 格：(A01,S01)=2、(A02,S01)=1、(A01,S02)=1
    assert len(groups) == 3
    assert len(groups[("A01", "S01")]) == 2  # 变体被排除


# ---- 信号计算 ----

def test_compute_cell_signals_compactness():
    # 格内特征紧凑（离散度低）→ compactness 高
    tight = np.array([[1.0, 1.0], [1.01, 1.0]], dtype=np.float32)
    sig_tight = finegrained.compute_cell_signals(tight, None, None)
    # 格内特征分散 → compactness 低
    spread = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    sig_spread = finegrained.compute_cell_signals(spread, None, None)
    assert sig_tight["compactness"] > sig_spread["compactness"]
    assert 0.0 <= sig_tight["compactness"] <= 1.0


def test_compute_cell_signals_consistency_and_separability():
    # 两类样本，特征可分
    cls0 = [_make_sample_id(f"E01_S01_A01_f{i}-{i+5}", 0) for i in range(5)]
    cls1 = [_make_sample_id(f"E01_S01_A02_f{i}", 1) for i in range(5)]
    X0 = np.stack([_identity_feature(s) for s in cls0])
    X1 = np.stack([_identity_feature(s) for s in cls1])
    sig = finegrained.compute_cell_signals(X0, None, X1)
    assert 0.0 <= sig["separability"] <= 1.0
    assert 0.0 <= sig["consistency"] <= 1.0


def test_separability_distinguishes_separable_from_overlapping():
    # 可分：cell 特征远离子其他类
    far_cell = np.array([[10.0, 0.0], [10.0, 0.1]], dtype=np.float32)
    other = np.array([[0.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    sig_far = finegrained.compute_cell_signals(far_cell, None, other)

    # 不可分：cell 与其他类重叠
    overlap_cell = np.array([[0.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    sig_overlap = finegrained.compute_cell_signals(overlap_cell, None, other)

    assert sig_far["separability"] > sig_overlap["separability"]


# ---- 加权合成 + 矩阵构建 ----

def test_synthesize_quality_weighted():
    signals = {"compactness": 0.8, "consistency": 0.6, "separability": 0.7}
    w = {"compactness": 0.4, "consistency": 0.3, "separability": 0.3}
    q = finegrained.synthesize_quality(signals, w)
    assert q == pytest.approx(0.4*0.8 + 0.3*0.6 + 0.3*0.7, rel=1e-3)
    assert 0.0 <= q <= 1.0


def test_build_matrix_structure():
    groups = {("A01","S01"): [_make_sample_id("E01_S01_A01_f1-7", 0),
                              _make_sample_id("E01_S01_A01_f8-14", 0)],
              ("A02","S01"): [_make_sample_id("E01_S01_A02_f1-7", 1)]}
    result = finegrained.build_matrix(groups, extract_fn=_identity_feature, weights=None)
    assert "global" in result and "matrix" in result and "low_quality" in result
    assert "A01_S01" in result["matrix"]
    assert "per_class" in result["global"] and "per_subject" in result["global"]
    assert "per_env" in result["global"]
