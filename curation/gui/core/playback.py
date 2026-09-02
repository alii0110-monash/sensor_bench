"""Load a sample's full action segment (ALL raw frames) for video-like playback.

The canonical samples hold only 5 frames per action; the raw dataset holds the
whole segment [start_index, end_index]. This module loads every raw frame of
the segment, per modality, at canonical processed resolutions (depth 224x224,
wifi CSI, lidar point cloud, mmwave radar point cloud (T,64,5) = [x,y,z,
doppler,intensity], rgb keypoints).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from curation.ingest.readers import (
    read_depth_frame,
    read_keypoint_frame,
    read_lidar_frame,
    read_mmwave_frame,
    read_wifi_frame,
)

# modality -> (reader, file extension, raw subdir)
_MODALITY_READERS = {
    "wifi": (read_wifi_frame, ".mat", "wifi-csi"),
    "depth": (read_depth_frame, ".png", "depth"),
    "lidar": (read_lidar_frame, ".bin", "lidar"),
    "mmwave": (read_mmwave_frame, ".bin", "mmwave"),
    "rgb": (read_keypoint_frame, ".npy", "rgb"),
}


def _load(sample, raw_root, fis):
    """Load the given raw frame indices for every present modality."""
    env = sample.meta.get("env")
    subj = sample.meta.get("subject")
    if not env or not subj:
        return None
    action = f"A{sample.label + 1:02d}"
    out = {}
    for mod, (reader, ext, subdir) in _MODALITY_READERS.items():
        if mod not in sample.modalities:
            continue
        ddir = os.path.join(raw_root, env, subj, action, subdir)
        arrs = []
        ok = True
        for fi in fis:
            p = os.path.join(ddir, f"frame{fi:03d}{ext}")
            if not os.path.exists(p):
                ok = False
                break
            try:
                arrs.append(reader(p))
            except Exception:
                ok = False
                break
        if ok and arrs:
            out[mod] = (np.stack(arrs, axis=0).astype(np.float32), list(fis))
    return out


def segment_frames(sample, raw_root: Optional[str]) -> Optional[dict]:
    """Load every raw frame of the sample's action segment.

    Returns {modality: (data (T,...), frame_indices [start..end])} or None if
    the raw segment cannot be loaded. Only modalities present in the sample are
    loaded; a modality missing any raw file is skipped.
    """
    if not raw_root:
        return None
    start = sample.meta.get("start_index")
    end = sample.meta.get("end_index")
    if start is None or end is None:
        return None
    out = _load(sample, raw_root, list(range(int(start), int(end) + 1)))
    if not out:
        return None
    out["__source__"] = "raw_segment"
    return out


def action_frames(sample, raw_root: Optional[str], max_frames: int = 297) -> Optional[dict]:
    """Load the ENTIRE action recording (all available frames, capped at
    max_frames, i.e. every repetition of the action type for this subject/env).
    This is the 'whole A01' view the reviewer sees the action as one stream."""
    if not raw_root:
        return None
    env = sample.meta.get("env")
    subj = sample.meta.get("subject")
    if not env or not subj:
        return None
    action = f"A{sample.label + 1:02d}"
    # available frame count = min across present modality dirs (keep sync)
    n_avail = None
    for mod, (_, ext, subdir) in _MODALITY_READERS.items():
        if mod not in sample.modalities:
            continue
        ddir = os.path.join(raw_root, env, subj, action, subdir)
        if os.path.isdir(ddir):
            cnt = sum(1 for f in os.listdir(ddir) if f.endswith(ext))
            n_avail = cnt if n_avail is None else min(n_avail, cnt)
    if not n_avail:
        return None
    n = min(n_avail, max_frames)
    out = _load(sample, raw_root, list(range(1, n + 1)))
    if not out:
        return None
    out["__source__"] = "raw_full"
    return out


def canonical_segment(sample) -> Optional[dict]:
    """Fallback: the sample's own modalities (usually only 5 frames)."""
    if not sample.modalities:
        return None
    out = {
        mod: (mod_obj.data, list(mod_obj.frame_indices))
        for mod, mod_obj in sample.modalities.items()
    }
    out["__source__"] = "canonical"
    return out