from __future__ import annotations

import os

import numpy as np

from framework.dataset.sample import Modality, Sample

from curation.gui.core.playback import action_frames, canonical_segment, segment_frames


def _make_raw_root(tmp_path):
    """Build a raw action dir E04/S33/A01 with frames 1..6 for all modalities."""
    import cv2
    import scipy.io as scio

    root = tmp_path / "raw"
    ddir = root / "E04" / "S33" / "A01"
    for sub in ("depth", "lidar", "mmwave", "rgb", "wifi-csi"):
        (ddir / sub).mkdir(parents=True)
    for i in range(1, 7):
        n = f"frame{i:03d}"
        cv2.imwrite(str(ddir / "depth" / f"{n}.png"),
                    np.full((224, 224), 3000 + i, dtype=np.uint16))
        (ddir / "lidar" / f"{n}.bin").write_bytes(np.random.rand(10, 3).astype(np.float64).tobytes())
        (ddir / "mmwave" / f"{n}.bin").write_bytes(np.random.rand(8, 5).astype(np.float64).tobytes())
        np.save(ddir / "rgb" / f"{n}.npy", np.random.rand(17, 2).astype(np.float64))
        scio.savemat(str(ddir / "wifi-csi" / f"{n}.mat"),
                     {"CSIamp": np.random.rand(3, 114, 10)})
    return str(root)


def _fake_sample():
    return Sample(
        id="E04_S33_A01_f1-6",
        label=0,
        modalities={
            "wifi": Modality(data=np.zeros((5, 3, 114, 10), dtype=np.float32), frame_indices=[1, 2, 3, 3, 3]),
            "depth": Modality(data=np.zeros((5, 1, 224, 224), dtype=np.float32), frame_indices=[1, 2, 3, 3, 3]),
            "lidar": Modality(data=np.zeros((5, 1536, 3), dtype=np.float32), frame_indices=[1, 2, 3, 3, 3]),
            "mmwave": Modality(data=np.zeros((5, 64, 5), dtype=np.float32), frame_indices=[1, 2, 3, 3, 3]),
            "rgb": Modality(data=np.zeros((5, 17, 2), dtype=np.float32), frame_indices=[1, 2, 3, 3, 3]),
        },
        meta={"subject": "S33", "env": "E04", "start_index": 1, "end_index": 3},
    )


def test_segment_frames_loads_segment_only(tmp_path):
    raw = _make_raw_root(tmp_path)
    s = _fake_sample()
    seg = segment_frames(s, raw)
    assert seg is not None
    assert seg["__source__"] == "raw_segment"
    for mod in ("wifi", "depth", "lidar", "mmwave", "rgb"):
        data, fis = seg[mod]
        assert len(fis) == 3 and fis == [1, 2, 3]  # segment is 1-3
        assert data.shape[0] == 3


def test_action_frames_loads_every_raw_frame(tmp_path):
    raw = _make_raw_root(tmp_path)
    s = _fake_sample()
    seg = action_frames(s, raw)
    assert seg is not None
    assert seg["__source__"] == "raw_full"
    for mod in ("wifi", "depth", "lidar", "mmwave", "rgb"):
        data, fis = seg[mod]
        assert len(fis) == 6 and fis == [1, 2, 3, 4, 5, 6]  # whole action
        assert data.shape[0] == 6


def test_segment_frames_none_without_raw_root():
    assert segment_frames(_fake_sample(), None) is None
    assert action_frames(_fake_sample(), None) is None


def test_canonical_segment_fallback():
    s = _fake_sample()
    seg = canonical_segment(s)
    assert seg["__source__"] == "canonical"
    assert seg["depth"][0].shape[0] == 5