"""Domain-aware structured features for weak modalities.

Goal: replace raw high-dim sensor data with low-dim domain features
that capture motion/silhouette patterns, similar to how v3 added rgb
keypoints (17,2) to expose discriminative structure.

Each extractor returns a fixed-size 1-D feature vector per sample.
NaN/Inf values are replaced with 0.0 to avoid corrupting downstream
training (NaN propagation through Linear → NaN logits → argmax=0 collapse).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _safe(feat: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0.0 to prevent downstream propagation."""
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---- Depth (T, 1, H, W) or (T, H, W) ----------------------------------
def extract_depth_features(depth: np.ndarray,
                          body_range: tuple = (1.0, 3.5)) -> np.ndarray:
    """Domain features from depth maps: body silhouette stats + motion.

    Returns a 1-D float32 vector. Concept: a person is a closer-depth blob
    (typically 1.0-3.5m) on a farther background. Frame-wise stats + motion
    characterize the action.
    """
    if depth.ndim == 4:
        depth = depth[:, 0]  # (T, H, W)
    T, H, W = depth.shape
    feats = []
    lo, hi = body_range
    body_mask = (depth >= lo) & (depth <= hi) & np.isfinite(depth)
    for t in range(T):
        f = depth[t]
        m = body_mask[t]
        body_frac = float(m.mean())
        body_mean = float(f[m].mean()) if m.any() else 0.0
        body_std = float(f[m].std()) if m.any() else 0.0
        bg_mean = float(f[~m].mean()) if (~m).any() else 0.0
        # silhouette bbox
        ys, xs = np.where(m)
        if len(xs) > 0:
            bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
            bbox_frac = bbox_area / (H * W)
            cy = float((ys.mean() - H / 2) / H)
            cx = float((xs.mean() - W / 2) / W)
        else:
            bbox_frac = cy = cx = 0.0
        # depth histogram (5 bins covering 0..6m)
        hist, _ = np.histogram(f[~np.isnan(f)], bins=5, range=(0, 6))
        feats.extend([body_frac, body_mean, body_std, bg_mean,
                      bbox_frac, cy, cx, *hist.tolist()])
    # motion: frame-to-frame depth change statistics
    if T > 1:
        diff = np.diff(depth, axis=0)
        motion_mean = float(np.abs(diff).mean())
        motion_std = float(np.abs(diff).std())
        motion_max = float(np.abs(diff).max())
    else:
        motion_mean = motion_std = motion_max = 0.0
    feats.extend([motion_mean, motion_std, motion_max])
    return _safe(np.asarray(feats, dtype=np.float32))


# ---- Wifi (T, 3, 114, 10) -----------------------------------------------
def extract_wifi_features(wifi: np.ndarray) -> np.ndarray:
    """Improved wifi features (wifi_v2, 2026-08-21).

    wifi: (T, 3, 114, 10) = frames × antennas × subcarriers × time-samples.

    The pre-2026-08-21 extractor (129d) scored 0.080 on probe — *below* the
    raw mean-over-time baseline (0.093), i.e. the feature engineering was
    actively hurting. wifi_v2 (161d) focuses on temporal variation (body
    motion modulates CSI over time) + per-frame antenna/subcarrier stats +
    cross-antenna correlation + inter-frame motion, and scores 0.110
    (+37% vs the old extractor). Verified by
    `scripts/probe_wifi_feat_v2.py`.

    Note: wifi remains a weak modality (probe ~0.11 vs random 1/27=0.037);
    the improvement is real but bounded by the data's intrinsic low
    discriminative power (only-wifi ≈ random in the main pipeline).
    """
    if wifi.ndim != 4:
        wifi = wifi.reshape(wifi.shape[0], -1, 114, 10)
    T, A, S, F = wifi.shape
    feats = []
    # 1. Per-frame antenna temporal stats (mean/std over subcarriers+time)
    for t in range(T):
        frame = wifi[t]  # (A, S, F)
        ant_mean = frame.mean(axis=(1, 2))  # (A,)
        ant_std = frame.std(axis=(1, 2))     # (A,)
        feats.extend(ant_mean.tolist())
        feats.extend(ant_std.tolist())
        # temporal variance (over time-samples, per antenna)
        temp_var = frame.var(axis=-1).mean(axis=1)  # (A,)
        feats.extend(temp_var.tolist())
    # 2. Subcarrier profile (mean over antennas+time, per frame, downsampled)
    for t in range(T):
        sc_mean = wifi[t].mean(axis=(0, 2))  # (S,)
        sc_bins = sc_mean.reshape(6, 19).mean(axis=0)  # (19,)
        feats.extend(sc_bins.tolist())
    # 3. Cross-antenna correlation (per frame)
    for t in range(T):
        flat = wifi[t].reshape(A, -1)  # (A, S*F)
        for i in range(A):
            for j in range(i + 1, A):
                vi, vj = flat[i], flat[j]
                si, sj = vi.std(), vj.std()
                if si < 1e-9 or sj < 1e-9:
                    feats.append(0.0)
                else:
                    feats.append(float(np.corrcoef(vi, vj)[0, 1]))
    # 4. Inter-frame motion (frame-to-frame abs diff)
    if T > 1:
        diffs = [np.abs(wifi[t + 1] - wifi[t]).mean() for t in range(T - 1)]
        feats.extend([float(np.mean(diffs)), float(np.max(diffs)),
                      float(np.std(diffs))])
    else:
        feats.extend([0.0, 0.0, 0.0])
    # 5. Global stats
    feats.append(float(wifi.mean()))
    feats.append(float(wifi.std()))
    feats.append(float(wifi.max()))
    return _safe(np.asarray(feats, dtype=np.float32))


# ---- Lidar (T, 1536, 3) --------------------------------------------------
def extract_lidar_features(lidar: np.ndarray) -> np.ndarray:
    """Point cloud spatial statistics + motion.

    Concept: lidar captures 3D body shape per frame; bounding box,
    point density, and motion magnitude discriminate actions.
    """
    T, N, _ = lidar.shape
    feats = []
    for t in range(T):
        pts = lidar[t]
        valid = np.isfinite(pts).all(axis=-1)
        pts = pts[valid]
        if pts.shape[0] == 0:
            pts = np.zeros((1, 3))
        # bbox (range per axis)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = (maxs - mins)
        # centroid
        centroid = pts.mean(axis=0)
        # point density in 4x4x4 grid
        if pts.shape[0] > 0:
            mins_g = mins
            maxs_g = maxs + 1e-6
            grid = ((pts - mins_g) / (maxs_g - mins_g) * 4).astype(int)
            grid = np.clip(grid, 0, 3)
            density = np.zeros(64)
            for x, y, z in grid:
                density[x * 16 + y * 4 + z] += 1
        else:
            density = np.zeros(64)
        feats.extend(ranges.tolist())
        feats.extend(centroid.tolist())
        feats.extend(density.tolist())
    # Motion: inter-frame centroid shift magnitude
    if T > 1:
        diffs = []
        for t in range(T - 1):
            c1 = lidar[t][np.isfinite(lidar[t]).all(axis=-1)].mean(axis=0)
            c2 = lidar[t + 1][np.isfinite(lidar[t + 1]).all(axis=-1)].mean(axis=0)
            diffs.append(float(np.linalg.norm(c2 - c1)))
        feats.extend([float(np.mean(diffs)), float(np.max(diffs)),
                      float(np.std(diffs))])
    else:
        feats.extend([0.0, 0.0, 0.0])
    return _safe(np.asarray(feats, dtype=np.float32))


# ---- mmwave (T, 64, 5) ---------------------------------------------------
def _percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else 0.0


def _geom_per_frame(pts: np.ndarray) -> list[float]:
    """Per-frame XYZ stats (16 features): point count + 5 stats per dim."""
    f = [float(len(pts))] if len(pts) else [0.0]
    if len(pts) == 0:
        return f + [0.0] * 15
    for d in range(3):  # x, y, z
        col = pts[:, d]
        f.append(float(col.mean()))
        f.append(float(col.std()) if len(col) > 1 else 0.0)
        f.append(float(col.max() - col.min()))
        f.append(_percentile(col, 25))
        f.append(_percentile(col, 75))
    return f  # 16 features


def _signal_per_frame(pts: np.ndarray) -> list[float]:
    """Per-frame doppler (dim3) + intensity (dim4) stats (8 features)."""
    f = []
    if len(pts) == 0:
        return [0.0] * 8
    for d in (3, 4):
        col = pts[:, d]
        f.append(float(col.mean()))
        f.append(float(np.abs(col).mean()))  # magnitude
        f.append(float(col.std()) if len(col) > 1 else 0.0)
        f.append(float(col.max() - col.min()))
    return f  # 8 features


def _z_histogram(pts: np.ndarray, bins: int = 8, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    """8-bin histogram of dim2 (z). Captures vertical distribution shape."""
    if len(pts) == 0:
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(pts[:, 2], bins=bins, range=(lo, hi))
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def _xy_extent(pts: np.ndarray) -> list[float]:
    """XY extent + covariance eigenvalues for spatial shape."""
    if len(pts) < 2:
        return [0.0, 0.0, 0.0]
    xy = pts[:, :2]
    xr = float(xy[:, 0].max() - xy[:, 0].min())
    yr = float(xy[:, 1].max() - xy[:, 1].min())
    cov = np.cov(xy.T)
    eigs = np.linalg.eigvalsh(cov) if cov.shape == (2, 2) else np.array([0.0, 0.0])
    return [xr * yr, float(eigs[0]), float(eigs[1])]


def extract_mmwave_features(mmwave: np.ndarray) -> np.ndarray:
    """Cartesian-geometry mmwave feature extractor (geom_v2, 2026-08-20).

    MMFi mmwave columns: `[x, y, z, doppler, intensity]` — Cartesian 3D
    coordinates + radial velocity + reflection strength (confirmed against
    MMFi official `mmwave_vae.py` which uses [:,:3] as 3D point cloud for
    FPS sampling and Chamfer losses).

    This is the *correct* (geom_v2) extractor: it focuses on XYZ geometric
    distribution (the dimension that the dim-ablation experiment identified
    as the dominant signal — drop_dim2_z costs -27.5% probe val acc). It
    complements with doppler/intensity summary stats.

    Replaces the pre-2026-08-20 spherical misread (50-dim, probe val_acc
    0.359) which was lower than raw (0.376) because the direction was
    wrong. New probe val_acc (Linear probe, 20 epochs × 3 seeds on v3):
    **0.709** (+88.5% vs raw 0.376). Verified by
    `scripts/probe_mmwave_geom_v2.py`.

    Output (134-dim float32):
      - T frames × 16 geom_per_frame  = 16T  (T=5 → 80; runtime adapts)
      - T frames × 8  signal_per_frame = 8T   (T=5 → 40)
      - 8   z histogram bins (avg across frames)
      - 3   xy_extent (avg across frames)
      - 3   centroid_stds (x, y, z inter-frame drift)

    Note: feature dimension depends on T (number of frames per sample).
    Current dataset uses T=5 → 80+40+8+3+3 = 134 dims.
    """
    T, N, A = mmwave.shape
    assert A == 5, f"expected 5 attributes per point, got {A}"
    geom, signal, z_hists, xy_exts, centroids = [], [], [], [], [[], [], []]
    for t in range(T):
        valid = ~np.all(mmwave[t] == 0, axis=-1)
        pts = mmwave[t][valid]
        geom.extend(_geom_per_frame(pts))
        signal.extend(_signal_per_frame(pts))
        z_hists.append(_z_histogram(pts))
        xy_exts.append(_xy_extent(pts))
        if len(pts):
            for d in range(3):
                centroids[d].append(float(pts[:, d].mean()))
    z_hist_avg = (np.mean(z_hists, axis=0) if z_hists
                  else np.zeros(8, dtype=np.float32))
    xy_avg = np.mean(xy_exts, axis=0) if xy_exts else [0.0, 0.0, 0.0]
    centroid_stds = [float(np.std(c)) if c else 0.0 for c in centroids]
    feat = np.concatenate([
        np.asarray(geom, dtype=np.float32),
        np.asarray(signal, dtype=np.float32),
        z_hist_avg.astype(np.float32),
        np.asarray(xy_avg, dtype=np.float32),
        np.asarray(centroid_stds, dtype=np.float32),
    ])
    return _safe(feat)


# ---- Dispatch ----
_EXTRACTORS = {
    "depth": extract_depth_features,
    "wifi": extract_wifi_features,
    "lidar": extract_lidar_features,
    "mmwave": extract_mmwave_features,
}


def extract_structured_feature(sample, modality: str) -> np.ndarray:
    if modality not in sample.modalities:
        return np.zeros(1, dtype=np.float32)
    fn = _EXTRACTORS.get(modality)
    if fn is None:
        return sample.modalities[modality].data.reshape(-1).astype(np.float32)
    return fn(sample.modalities[modality].data)


if __name__ == "__main__":
    import pickle
    # Quick smoke test
    with open("datasets/mmfi/v4/data/E01_S01_A01_f1-7.pkl", "rb") as f:
        s = pickle.load(f)
    Sample = __import__("framework.dataset.sample", fromlist=["Sample"]).Sample
    s = Sample.from_dict(s)
    for m in ["wifi", "depth", "lidar", "mmwave"]:
        f = extract_structured_feature(s, m)
        print(f"{m}: shape={f.shape} dtype={f.dtype} "
              f"range=[{f.min():.3f}, {f.max():.3f}]")