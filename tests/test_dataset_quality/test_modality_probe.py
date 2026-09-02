"""Tests for per-modality and concat feature extraction + Linear probe."""
import numpy as np
import pytest
import torch

from framework.dataset.sample import Sample, Modality
from framework.eval.dataset_quality.modality_probe import (
    extract_modality_feature, extract_concat_feature, MODALITY_ORDER,
    stack_split, train_probe, evaluate_probe, _to_tensor,
    compute_info_score,
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


def test_modality_feature_shape_rgb():
    s = _toy_sample()
    feat = extract_modality_feature(s, "rgb")
    assert isinstance(feat, np.ndarray)
    assert feat.ndim == 1
    assert feat.shape[0] == 3 * 8 * 8  # 192


def test_modality_feature_depth_2d_no_channel():
    s = _toy_sample()
    s.modalities["depth"].data[:] = 0.0
    s.modalities["depth"].data[0, 0, 0] = 1.0
    feat = extract_modality_feature(s, "depth")
    # depth shape (5, 224, 224) → mean over axis 0 → (224, 224) → flatten
    assert feat.shape == (224 * 224,)


def test_modality_feature_wifi_4d():
    s = _toy_sample()
    feat = extract_modality_feature(s, "wifi")
    # wifi shape (5, 3, 114, 10) → mean over axis 0 → (3, 114, 10) → flatten
    assert feat.shape == (3 * 114 * 10,)


def test_concat_feature_concatenates_modalities():
    s = _toy_sample()
    feat = extract_concat_feature(s, ["rgb", "depth", "lidar"])
    rgb_d = 3 * 8 * 8
    depth_d = 224 * 224
    lidar_d = 100 * 3
    assert feat.shape == (rgb_d + depth_d + lidar_d,)


def test_modality_order_contains_five():
    assert set(MODALITY_ORDER) == {"rgb", "depth", "lidar", "mmwave", "wifi"}


def test_stack_split_per_modality():
    samples = [_toy_sample() for _ in range(3)]
    X_dict, y = stack_split(samples, ["rgb", "depth"])
    assert X_dict["rgb"].shape == (3, 3 * 8 * 8)
    assert X_dict["depth"].shape == (3, 224 * 224)
    assert y.shape == (3,)
    assert (y == 0).all()


def test_stack_split_concat():
    samples = [_toy_sample() for _ in range(2)]
    X_dict, y = stack_split(samples, ["rgb", "depth"], concat=True)
    assert "concat" in X_dict
    assert X_dict["concat"].shape == (2, 3 * 8 * 8 + 224 * 224)
    assert y.shape == (2,)

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


def test_train_probe_returns_linear():
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
    rng = np.random.default_rng(0)
    n, dim, n_classes = 600, 8, 4
    X = rng.normal(size=(n, dim)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) * 2
    y[X[:, 0] < -0.5] = 1
    y[X[:, 0] > 0.5] = 3
    model = train_probe(X, y, num_classes=4, epochs=10, lr=1e-1, batch_size=64)
    acc = evaluate_probe(model, X, y)
    assert acc > 0.4  # clearly above 1/4 = 0.25 random


def test_info_score_basic():
    acc_per_modality = {"rgb": 0.8, "depth": 0.5, "lidar": 0.6,
                        "mmwave": 0.4, "wifi": 0.1}
    acc_concat = 0.9
    res = compute_info_score(acc_per_modality, acc_concat,
                             w_per_modality=0.7, w_complement=0.3)
    assert res["mean_acc"] == pytest.approx(0.48)
    # complement_gain = acc_concat - max(acc_per_modality) = 0.9 - 0.8 = 0.1
    assert res["complement_gain"] == pytest.approx(0.1)
    # InfoScore = 0.7 * 0.48 + 0.3 * 0.1 = 0.366
    assert res["InfoScore"] == pytest.approx(0.366)
    assert 0.0 <= res["InfoScore"] <= 1.0


def test_info_score_clips_negative_gain():
    acc_per_modality = {"rgb": 0.9}
    acc_concat = 0.5
    res = compute_info_score(acc_per_modality, acc_concat,
                             w_per_modality=0.7, w_complement=0.3)
    assert res["complement_gain"] == pytest.approx(-0.4)
    assert res["InfoScore"] == pytest.approx(0.7 * 0.9)


def test_info_score_clamps_to_unit_interval():
    acc_per_modality = {"a": 0.5}
    acc_concat = 1.0
    res = compute_info_score(acc_per_modality, acc_concat,
                             w_per_modality=0.7, w_complement=0.3)
    assert res["InfoScore"] <= 1.0
    assert res["InfoScore"] >= 0.0
