from __future__ import annotations
import os
import pickle
import re
from typing import Dict, List
import numpy as np

from . import readers
from framework.dataset.sample import Sample, Modality


def sample_frames(start: int, end: int, n: int = 5) -> List[int]:
    """Deterministic uniform frame sampling across [start, end] (reproducible)."""
    if end - start + 1 <= n:
        idx = list(range(start, end + 1))
        while len(idx) < n:
            idx.append(idx[-1])
        return idx
    return sorted(np.unique(np.linspace(start, end, n, dtype=int)).tolist())


def frame_paths(act_dir: str, modal: str, frames: List[int]) -> List[str]:
    exts = {"wifi-csi": ".mat", "lidar": ".bin", "mmwave": ".bin", "depth": ".png"}
    return [os.path.join(act_dir, modal, f"frame{i:03d}{exts[modal]}") for i in frames]


def _read_frames(reader, paths: List[str], shape_tail) -> np.ndarray:
    arrs = [reader(p) for p in paths]
    stacked = np.stack(arrs, axis=0)          # (T, ...)
    assert stacked.shape[1:] == shape_tail, (stacked.shape, shape_tail)
    return stacked


def rel_video_path(video_path: str) -> str:
    """Map annotation video_path (./datasets/MMFi/E02/S19/A03) to the raw-root
    relative path (E02/S19/A03). Robust to any leading prefix."""
    parts = video_path.replace("./", "").split("/")
    for i, p in enumerate(parts):
        if re.match(r"^E\d+$", p):
            return "/".join(parts[i:])
    raise ValueError(f"cannot parse video_path: {video_path}")


def annotation_to_sample(ann: dict, raw_root: str, action_labels: Dict[str, int]) -> Sample:
    rel = rel_video_path(ann["video_path"])
    act_dir = os.path.join(raw_root, rel)
    frames = sample_frames(ann["start_index"], ann["end_index"])
    sid = ann["sample_id"]
    action = rel.split("/")[-1]
    label = action_labels[action]

    wifi = _read_frames(readers.read_wifi_frame, frame_paths(act_dir, "wifi-csi", frames), (3, 114, 10))
    depth = _read_frames(readers.read_depth_frame, frame_paths(act_dir, "depth", frames), (1, 224, 224))
    lidar = _read_frames(readers.read_lidar_frame, frame_paths(act_dir, "lidar", frames), (1536, 3))
    mmwave = _read_frames(readers.read_mmwave_frame, frame_paths(act_dir, "mmwave", frames), (64, 5))

    mods = {
        "wifi":   Modality(data=wifi,   frame_indices=frames, sample_rate=1000),
        "depth":  Modality(data=depth,  frame_indices=frames, sample_rate=20),
        "lidar":  Modality(data=lidar,  frame_indices=frames, sample_rate=20),
        "mmwave": Modality(data=mmwave, frame_indices=frames, sample_rate=20),
    }
    return Sample(
        id=sid, label=label, modalities=mods,
        text={"captions": [c["value"] for c in ann.get("conversations", []) if c["from"] == "gpt"]},
        meta={"subject": rel.split("/")[-2], "env": rel.split("/")[-3],
              "source": "mmfi", "start_index": ann["start_index"], "end_index": ann["end_index"]},
    )


def write_sample(root: str, sample: Sample) -> None:
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, f"{sample.id}.pkl"), "wb") as f:
        pickle.dump(sample.to_dict(), f)


def action_labels() -> Dict[str, int]:
    return {f"A{i:02d}": i - 1 for i in range(1, 28)}
