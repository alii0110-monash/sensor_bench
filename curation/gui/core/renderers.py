"""Per-modality plotly renderers + registry.

Each renderer has signature  render(data, frame_indices, frame) -> Figure | None
  - data: np.ndarray of the modality (first axis = frames)
  - frame: int -> single-frame view; None -> aggregate view
  - None return means "cannot render" (the review page falls back to text stats).

Unknown modalities are handled by the page-level fallback (shape + statistics).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RENDERERS: Dict[str, Callable] = {}

# Semantic section layout for known structured-feature vectors (v5_structfeat).
# Format: modality name -> {length: [(start, end, label), ...]}
_FEATURE_SECTIONS: Dict[str, Dict[int, List[tuple]]] = {
    "depth": {
        63: [
            (0, 12, "帧1 身体统计+深度直方图(5)"),
            (12, 24, "帧2"),
            (24, 36, "帧3"),
            (36, 48, "帧4"),
            (48, 60, "帧5"),
            (60, 63, "帧间运动量(均值/std/最大)"),
        ],
    },
    "wifi": {
        161: [
            (0, 9, "帧1 (天线均值3+std3+时间方差3)"),
            (9, 18, "帧2"),
            (18, 27, "帧3"),
            (27, 36, "帧4"),
            (36, 45, "帧5"),
            (45, 140, "子载波剖面 (5帧×19bin)"),
            (140, 155, "天线间相关 (5帧×3对)"),
            (155, 158, "帧间运动 (均值/最大/std)"),
            (158, 161, "全局统计 (mean/std/max)"),
        ],
    },
    "lidar": {
        353: [
            (0, 70, "帧1 点云统计(范围3+质心3+密度64)"),
            (70, 140, "帧2"),
            (140, 210, "帧3"),
            (210, 280, "帧4"),
            (280, 350, "帧5"),
            (350, 353, "帧间运动量(均值/最大/std)"),
        ],
    },
    "mmwave": {
        134: [
            (0, 24, "帧1 (XYZ几何16 + doppler/intensity8)"),
            (24, 48, "帧2"),
            (48, 72, "帧3"),
            (72, 96, "帧4"),
            (96, 120, "帧5"),
            (120, 128, "z 直方图(8bin)"),
            (128, 131, "xy 范围(面积+协方差特征值2)"),
            (131, 134, "质心漂移 std(x,y,z)"),
        ],
    },
}


def _sections_for(name: str, length: int) -> Optional[List[tuple]]:
    by_len = _FEATURE_SECTIONS.get(name)
    if not by_len:
        return None
    sections = by_len.get(length)
    if not sections:
        return None
    if sections[0][0] != 0 or sections[-1][1] != length:
        return None
    prev = 0
    for s, e, _ in sections:
        if s != prev:
            return None
        prev = e
    return sections


def register(name: str):
    def deco(fn: Callable) -> Callable:
        RENDERERS[name] = fn
        return fn

    return deco


def render_modality(name: str, data: np.ndarray, frame_indices=None,
                    frame: Optional[int] = None):
    fn = RENDERERS.get(name)
    if fn is not None:
        try:
            fig = fn(data, frame_indices, frame)
            if fig is not None:
                return fig
        except Exception:
            pass
def render_animated(name: str, data: np.ndarray, frame_indices=None,
                    max_frames: int = 60, depth_downsample: int = 4,
                    lidar_max_points: int = 600,
                    camera: Optional[dict] = None) -> Optional[go.Figure]:
    """Client-side animation: one plotly figure with per-frame data + built-in
    play/pause/slider. Runs entirely in the browser (no server reruns).

    This is the PLAYING-mode renderer in the dual-path player (Plan A):
    frames advance smoothly client-side with the built-in play button driving
    `Plotly.animate`. The play button uses redraw:True so EVERY modality
    (line charts, heatmaps, WebGL 3D point clouds) visibly animates. Trade-off:
    redraw rebuilds the gl3d scene each frame, so a user-rotated 3D camera
    resets to the default during playback; the paused static-frame view still
    preserves the rotated view for inspection.

    `camera` (optional): dict like {"eye": {"x":..,"y":..,"z":..}} applied to the
    3D scene (lidar/mmwave) so the reviewer can set a fixed view via manual
    eye coordinates; playback keeps this camera (redraw rebuilds to it).

    Data volume control: depth is downsampled in the spatial dims and lidar
    point counts are capped, so the full-sequence spec stays small enough to
    transfer once.
    """
    if data is None or getattr(data, "ndim", 0) < 2:
        return None
    T = data.shape[0]
    n = min(T, max_frames)
    if n <= 1:
        return render_modality(name, data, frame_indices, frame=0)

    frame_data = data[:n]
    if name == "depth" and depth_downsample > 1 and frame_data.ndim == 4:
        frame_data = frame_data[:, :, ::depth_downsample, ::depth_downsample]
    elif name == "lidar" and frame_data.ndim == 3 and frame_data.shape[1] > lidar_max_points:
        idx = np.linspace(0, frame_data.shape[1] - 1, lidar_max_points).astype(int)
        frame_data = frame_data[:, idx, :]

    base = render_modality(name, frame_data, frame_indices, frame=0)
    if base is None:
        return None
    # Only apply camera to real 3D scenes (lidar/mmwave). Check the raw layout
    # dict for a scene with an xaxis (a real 3D subplot) — accessing
    # base.layout.scene would auto-create an empty Scene in plotly.
    if camera:
        scene = base.layout.to_plotly_json().get("scene")
        if isinstance(scene, dict) and scene.get("xaxis") is not None:
            base.update_layout(scene_camera=camera)
    frames = []
    for i in range(1, n):
        fi = render_modality(name, frame_data, frame_indices, frame=i)
        if fi is None:
            continue
        frames.append(go.Frame(data=list(fi.data), name=str(i)))
    if not frames:
        return base
    base.frames = frames

    labels = [str(i) for i in range(n)]
    base.update_layout(
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[lab], {"mode": "immediate",
                                      "frame": {"duration": 0, "redraw": True}}],
                        label=lab) for lab in labels],
            currentvalue=dict(prefix="帧 ", font=dict(size=11)),
            pad=dict(t=28), len=1.0, x=0)],
        updatemenus=[dict(
            type="buttons", direction="left", pad=dict(r=10, t=8), x=0.0,
            xanchor="left", y=1.14, yanchor="top",
            buttons=[
                dict(label="▶ 播放", method="animate",
                     args=[None, {"frame": {"duration": 70, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 0}}]),
                dict(label="⏸ 暂停", method="animate",
                     args=[[None], {"mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True}}]),
            ])],
        margin=dict(t=70))
    return base


def render_modality(name: str, data: np.ndarray, frame_indices=None,
                    frame: Optional[int] = None):
    fn = RENDERERS.get(name)
    if fn is not None:
        try:
            fig = fn(data, frame_indices, frame)
            if fig is not None:
                return fig
        except Exception:
            pass
    # Generic fallback by shape: 1-D feature vectors (e.g. v5_structfeat).
    arr = np.asarray(data)
    if arr.ndim == 1:
        sections = _sections_for(name, arr.shape[0])
        if sections is not None:
            return render_segmented_vector(arr, sections, name)
        return render_vector(arr)
    return None




def render_segmented_vector(vec: np.ndarray, sections: List[tuple],
                            modality: str = "feature") -> Optional[go.Figure]:
    """1-D feature vector drawn as one labeled panel per semantic section —
    the human-readable view for structured features (v5_structfeat)."""
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim != 1:
        return None
    clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    rows = len(sections)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=False, vertical_spacing=0.06,
        subplot_titles=[lbl for _, _, lbl in sections])
    for r, (s, e, lbl) in enumerate(sections, start=1):
        xs = list(range(s, e))
        ys = clean[s:e]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(width=1.2, color="#17becf"),
            showlegend=False,
            hovertemplate="idx %{x} = %{y:.4g}<extra></extra>"), row=r, col=1)
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray", opacity=0.5,
                      row=r, col=1)
    fig.update_layout(
        title=f"{modality} 结构化特征（按语义分块，共 {len(v)} 维）",
        height=max(200, rows * 85 + 60),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
    return fig


@register("__vector__")
def render_vector(vec: np.ndarray, frame_indices=None, frame=None):
    """Generic 1-D feature-vector view (line + zero reference + NaN markers).
    Capped to 2000 points for plotly performance."""
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim != 1:
        return None
    n = len(v)
    if n == 0:
        return None
    x = np.arange(n)
    if n > 2000:
        keep = np.linspace(0, n - 1, 2000).astype(int)
        x, v = x[keep], v[keep]
    clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    fig = go.Figure(go.Scatter(
        x=x, y=clean, mode="lines", name="feature",
        line=dict(width=1.2, color="#17becf")))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", opacity=0.6)
    if np.isnan(v).any() or np.isinf(v).any():
        bad = x[np.isnan(v) | np.isinf(v)]
        if len(bad):
            fig.add_trace(go.Scatter(
                x=bad, y=[0.0] * len(bad), mode="markers",
                marker=dict(symbol="x", color="red", size=6), name="NaN/Inf"))
    fig.update_layout(
        title=f"feature vector (len={n})",
        height=300, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        xaxis=dict(title="feature index"), yaxis=dict(title="value"))
    return fig


def _n_frames(data: np.ndarray) -> int:
    return int(data.shape[0])


def _clamp(frame: Optional[int], n: int) -> Optional[int]:
    if frame is None:
        return None
    return max(0, min(int(frame), n - 1))


def _base_layout(title: str, height: int = 380):
    return go.Figure().update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))


def _fixed_scene(limit: float = 3.5, camera_eye_x: bool = False,
                 ticks: bool = True) -> dict:
    """A scene with FIXED axis ranges and cube aspect, so the 3D view does not
    rescale with the data each frame. Optionally starts the camera looking
    along +X (from (0,0,0) toward (1,0,0))."""
    axis = dict(range=[-limit, limit], showticklabels=ticks)
    scene = dict(
        xaxis=dict(**axis, title="x (m)" if ticks else ""),
        yaxis=dict(**axis, title="y (m)" if ticks else ""),
        zaxis=dict(**axis, title="z (m)" if ticks else ""),
        aspectmode="cube")
    if camera_eye_x:
        # look from +X toward the origin: camera sits far out on +X axis
        scene["camera"] = dict(
            eye=dict(x=1.8, y=0.0, z=0.0),
            up=dict(x=0.0, y=0.0, z=1.0),
            center=dict(x=0.0, y=0.0, z=0.0))
    return scene


@register("rgb")
def render_rgb(data: np.ndarray, frame_indices, frame=None):
    """Skeleton keypoints (T,17,2) in COCO 17-joint layout. Aggregate =
    centroid trajectory + first-frame skeleton; single frame = that frame's
    skeleton with anatomical connections. Y axis is reversed so the body
    displays upright (v4 keypoints follow image convention: nose has the most
    negative y)."""
    if data.ndim != 3 or data.shape[1:] != (17, 2):
        return None
    T = _n_frames(data)
    frame = _clamp(frame, T)
    if frame is None:
        fig = _base_layout("rgb keypoints — centroid trajectory + 骨架(帧0)")
        centroid = data.mean(axis=1)  # (T,2)
        fig.add_trace(go.Scatter(
            x=centroid[:, 0], y=centroid[:, 1], mode="lines",
            name="centroid trajectory", line=dict(width=1, color="#1f77b4")))
        _add_skeleton(fig, data[0], color="rgba(215,39,40,0.9)", name="skeleton frame 0")
        fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
        return fig
    fig = _base_layout(f"rgb keypoints — frame {frame}/{T - 1}")
    _add_skeleton(fig, data[frame], color="#d62728", name=f"skeleton {frame}")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    return fig


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders + arms
    (5, 11), (6, 12), (11, 12),           # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


def _add_skeleton(fig: go.Figure, kpts: np.ndarray, color: str, name: str) -> None:
    """Add one skeleton frame's anatomical edges + joint markers to a figure."""
    for i, j in COCO_SKELETON:
        fig.add_trace(go.Scatter(
            x=[kpts[i, 0], kpts[j, 0]], y=[kpts[i, 1], kpts[j, 1]],
            mode="lines", line=dict(width=1.6, color=color), name=name,
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=kpts[:, 0], y=kpts[:, 1], mode="markers",
        marker=dict(size=5, color=color), name=name, showlegend=False))


@register("depth")
def render_depth(data: np.ndarray, frame_indices, frame=None):
    """Depth video (T,1,224,224): single frame = heatmap (downsampled); aggregate
    = mean-depth heatmap over time."""
    if data.ndim != 4 or data.shape[1] != 1:
        return None
    T = _n_frames(data)
    frame = _clamp(frame, T)
    if frame is None:
        d = data.mean(axis=0)[0]  # (H,W)
    else:
        d = data[frame, 0]
    fig = go.Figure(go.Heatmap(
        z=d, colorscale="Viridis",
        colorbar=dict(title="depth"),
        hovertemplate="x=%{x} y=%{y} d=%{z:.2f}<extra></extra>"))
    fig.update_layout(
        title=f"depth — {'mean over frames' if frame is None else f'frame {frame}/{T-1}'}",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        xaxis=dict(showticklabels=False, title=""),
        yaxis=dict(showticklabels=False, title="", autorange="reversed",
                   scaleanchor="x", scaleratio=1))
    return fig


@register("wifi")
def render_wifi(data: np.ndarray, frame_indices, frame=None):
    """CSI (T,A,114,S): aggregate = frame×subcarrier mean-power heatmap;
    single frame = subcarrier×symbol heatmap (mean over antennas)."""
    if data.ndim < 3:
        return None
    T = _n_frames(data)
    frame = _clamp(frame, T)
    if frame is None:
        d = data.mean(axis=(1, 3))  # (T,114) mean over antennas+symbols
        fig = go.Figure(go.Heatmap(
            z=d, colorscale="Viridis",
            x=[f"sc{i}" for i in range(d.shape[1])],
            y=[f"t{t}" for t in range(d.shape[0])],
            colorbar=dict(title="power")))
        title = "wifi CSI — frame × subcarrier (mean power)"
    else:
        d = data[frame].mean(axis=(0, 2))  # (114,) mean over antennas+symbols
        fig = go.Figure(go.Scatter(
            x=list(range(d.shape[0])), y=d, mode="lines",
            line=dict(width=1.5, color="#17becf")))
        fig.add_hline(y=float(np.median(d)), line_dash="dash", line_color="gray")
        title = f"wifi CSI — frame {frame}/{T - 1} subcarrier power profile"
    fig.update_layout(
        title=title, height=380, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
    return fig


@register("lidar")
def render_lidar(data: np.ndarray, frame_indices, frame=None):
    """Point clouds (T,N,3): scatter3d of one frame, or aggregated across frames."""
    if data.ndim != 3 or data.shape[2] != 3:
        return None
    T = _n_frames(data)
    frame = _clamp(frame, T)
    if frame is None:
        pts = data.reshape(-1, 3)
        title = f"lidar — all points ({T} frames)"
    else:
        pts = data[frame]
        title = f"lidar — frame {frame}/{T - 1}"
    if len(pts) > 4000:  # cap for plotly performance
        idx = np.linspace(0, len(pts) - 1, 4000).astype(int)
        pts = pts[idx]
    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=1.5, color=pts[:, 2], colorscale="Viridis", opacity=0.6)))
    fig.update_layout(
        title=title, height=420, margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        scene=_fixed_scene(limit=4.5, camera_eye_x=True))
    return fig


@register("mmwave")
def render_mmwave(data: np.ndarray, frame_indices, frame=None):
    """Sparse TI IWR radar point cloud (T, 64, 5). Each point =
    [x, y, z, doppler, intensity] — Cartesian 3D coordinates (meters) +
    radial velocity + reflection strength.

    Confirmed 2026-08-20 against MMFi official mmwave_vae.py:
      - Line 65:  `farthest_point_sample(x[:, :, :3])` — 3D FPS on first 3 cols
      - Line 104: `pred_xyz = pred_points[:, :, :3]` — Chamfer 3D loss
      - V5_architecture.md:80: "mmwave has 5 channels (XYZ+velocity)"

    3D spatial view: Cartesian x/y/z are already in the point cloud; no
    coordinate transform is needed. Single frame colored by doppler;
    aggregate overlays all frames colored by frame index (trajectory).
    """
    if data.ndim != 3 or data.shape[2] != 5:
        return None
    T = _n_frames(data)
    frame = _clamp(frame, T)
    if frame is None:
        return _mmwave_aggregate(data, T)
    return _mmwave_single(data[frame], frame, T)


def _mmwave_active(pts: np.ndarray) -> np.ndarray:
    """Rows that carry a real detection (not zero-padding)."""
    return pts[np.any(pts != 0, axis=1)]


def _mmwave_xyz(pts: np.ndarray) -> np.ndarray:
    """Extract Cartesian XYZ from a frame's valid mmwave points.

    MMFi mmwave columns are `[x, y, z, doppler, intensity]` (Cartesian m +
    radial velocity + reflection strength, confirmed 2026-08-20 against
    MMFi official `mmwave_vae.py` which uses [:,:3] as 3D point cloud for
    FPS sampling and Chamfer losses). So the 3D positions are just the
    first three columns — no coordinate transform needed.
    """
    return pts[:, :3].copy()


def _mmwave_scene(fig, xyz: np.ndarray, color, colorbar_title, name,
                  customdata, hovertemplate) -> None:
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        customdata=customdata, hovertemplate=hovertemplate,
        marker=dict(size=4, color=color, colorscale="Viridis", opacity=0.85,
                    colorbar=dict(title=colorbar_title)),
        name=name))




def _mmwave_single(pts: np.ndarray, frame: int, T: int):
    pts = _mmwave_active(pts)
    if len(pts) == 0:
        return None
    xyz = _mmwave_xyz(pts)
    fig = go.Figure()
    _mmwave_scene(
        fig, xyz, color=pts[:, 3], colorbar_title="doppler (m/s)", name=f"frame {frame}",
        customdata=np.stack([pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 4]], axis=1),
        hovertemplate=("x=%{customdata[0]:.2f}m y=%{customdata[1]:.2f}m "
                       "z=%{customdata[2]:.2f}m "
                       "intensity=%{customdata[3]:.2f} "
                       "doppler=%{marker.color:.2f}m/s<extra></extra>"))
    fig.update_layout(
        title=f"mmwave — frame {frame}/{T - 1} 雷达点云 3D (点色=doppler)",
        height=460, margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        scene=_fixed_scene())
    return fig


def _mmwave_aggregate(data: np.ndarray, T: int):
    """All frames overlaid; color encodes frame index (action trajectory)."""
    fig = go.Figure()
    for t in range(T):
        pts = _mmwave_active(data[t])
        if len(pts) == 0:
            continue
        xyz = _mmwave_xyz(pts)
        fr = np.full(len(pts), t)
        _mmwave_scene(
            fig, xyz, color=fr, colorbar_title="帧", name=f"t{t}",
            customdata=np.stack([pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], pts[:, 4]], axis=1),
            hovertemplate=("frame=%{marker.color:.0f} "
                           "x=%{customdata[0]:.2f}m y=%{customdata[1]:.2f}m "
                           "z=%{customdata[2]:.2f}m "
                           "doppler=%{customdata[3]:.2f}m/s "
                           "intensity=%{customdata[4]:.2f}<extra></extra>"))
    fig.update_layout(
        title="mmwave — 全部帧叠加 (点色=帧序，动作轨迹)",
        height=460, margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        scene=_fixed_scene())
    return fig


def modality_stats(data: np.ndarray) -> str:
    """Text fallback for unknown / unrenderable modalities."""
    try:
        arr = np.asarray(data, dtype=np.float32)
        return (f"shape={list(arr.shape)} dtype={data.dtype} "
                f"min={float(np.nanmin(arr)):.3g} max={float(np.nanmax(arr)):.3g} "
                f"nan={int(np.isnan(arr).sum())} zeros={int(np.count_nonzero(arr) == 0)}")
    except Exception:
        return f"shape={list(np.shape(data))} dtype={getattr(data, 'dtype', '?')}"