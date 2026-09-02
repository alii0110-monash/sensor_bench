import numpy as np
import pytest

from curation.enrich.keypoints import (FLIP_IDX, augment_keypoints,
                                       make_variant_id, normalize_keypoints)


def _fake_kpts():
    """Simple standing pose: nose top, hips middle, ankles bottom."""
    k = np.zeros((2, 17, 2), dtype=np.float32)
    k[:, 0] = [100, 50]       # nose
    k[:, 11] = [90, 150]      # left hip
    k[:, 12] = [110, 150]     # right hip
    k[:, 15] = [95, 250]      # left ankle
    k[:, 16] = [105, 250]     # right ankle
    return k


def test_normalize_hip_center_zero():
    k = _fake_kpts()
    n = normalize_keypoints(k)
    # hip midpoint should be at origin
    hip_mid = (n[0, 11] + n[0, 12]) / 2
    assert np.allclose(hip_mid, 0, atol=1e-5)
    # scale ~1: nose-to-hip distance normalized
    assert abs(np.linalg.norm(n[0, 0]) - 1.0) < 1e-2


def test_normalize_invariant_to_translation():
    k1 = _fake_kpts()
    k2 = _fake_kpts() + np.array([300.0, -80.0])
    assert np.allclose(normalize_keypoints(k1), normalize_keypoints(k2), atol=1e-4)


def test_normalize_preserves_shape():
    k = _fake_kpts()
    n = normalize_keypoints(k)
    assert n.shape == k.shape
    assert n.dtype == np.float32


def test_augment_flip_swaps_left_right():
    k = normalize_keypoints(_fake_kpts())
    # flip: joint 11 (left hip) <- mirrored joint 12 (right hip); trans/scale neutral
    a = augment_keypoints(k, np.random.default_rng(0), flip_p=1.0, trans_frac=0.0, scale_range=(1, 1))
    expected = k[:, FLIP_IDX]
    expected[:, :, 0] *= -1
    assert np.allclose(a, expected, atol=1e-5)


def test_augment_no_flip_keeps_identity():
    rng = np.random.default_rng(1)
    k = normalize_keypoints(_fake_kpts())
    # with flip_p=0, no flip; with trans=scale neutral, unchanged
    a = augment_keypoints(k, rng, flip_p=0.0, trans_frac=0.0, scale_range=(1, 1))
    assert np.allclose(a, k)


def test_make_variant_id():
    assert make_variant_id("E01_S01_A01_f1-7", 0) == "E01_S01_A01_f1-7__aug0"
    assert make_variant_id("E01_S01_A01_f1-7", 3) == "E01_S01_A01_f1-7__aug3"
