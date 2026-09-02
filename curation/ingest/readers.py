from __future__ import annotations
import cv2
import numpy as np
import scipy.io as scio

LIDAR_MAX = 1536
MMWAVE_MAX = 64


def read_wifi_frame(path: str) -> np.ndarray:
    """Returns (3, 114, 10) float32, finite, min-max normalized per frame."""
    f = scio.loadmat(path)["CSIamp"].astype(np.float64)
    f[np.isinf(f)] = np.nan
    for i in range(f.shape[-1]):
        col = f[:, :, i]
        nans = np.isnan(col)
        if nans.all():
            f[:, :, i] = 0.0
        elif nans.any():
            col[nans] = np.nanmean(col)
    mn, mx = float(f.min()), float(f.max())
    f = (f - mn) / (mx - mn + 1e-9)
    return f.astype(np.float32)


def read_lidar_frame(path: str) -> np.ndarray:
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
    pts = raw.reshape(-1, 3).astype(np.float32)
    n = min(pts.shape[0], LIDAR_MAX)
    pts = pts[:n]
    if n < LIDAR_MAX:
        pts = np.pad(pts, ((0, LIDAR_MAX - n), (0, 0)))
    return pts  # (1536, 3)


def read_mmwave_frame(path: str) -> np.ndarray:
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.float64)
    pts = raw.copy().reshape(-1, 5).astype(np.float32)
    n = min(pts.shape[0], MMWAVE_MAX)
    pts = pts[:n]
    if n < MMWAVE_MAX:
        pts = np.pad(pts, ((0, MMWAVE_MAX - n), (0, 0)))
    return pts  # (64, 5)


def read_depth_frame(path: str) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    d = cv2.resize(d, (224, 224), interpolation=cv2.INTER_AREA)
    return d[None]  # (1, 224, 224) meters


def read_keypoint_frame(path: str) -> np.ndarray:
    """Reads a body-keypoint frame (rgb/infra): (17, 2) float64 -> float32."""
    return np.load(path).astype(np.float32)  # (17, 2)
