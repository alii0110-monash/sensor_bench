from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from curation.gui.core import renderers


def _check_figure(fig):
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_rgb_aggregate():
    data = np.random.rand(10, 17, 2).astype(np.float32)
    fig = renderers.render_modality("rgb", data, list(range(10)), frame=None)
    _check_figure(fig)


def test_rgb_single_frame():
    data = np.random.rand(10, 17, 2).astype(np.float32)
    fig = renderers.render_modality("rgb", data, list(range(10)), frame=4)
    _check_figure(fig)


def test_rgb_skeleton_uses_coco_connections():
    data = np.random.rand(4, 17, 2).astype(np.float32)
    fig = renderers.render_modality("rgb", data, list(range(4)), frame=2)
    n_edges = len(renderers.COCO_SKELETON)
    line_traces = [t for t in fig.data if t.mode == "lines"]
    assert len(line_traces) == n_edges
    # y axis reversed so the body displays upright
    assert fig.layout.yaxis.autorange == "reversed"


def test_depth_square_aspect():
    data = np.random.rand(4, 1, 224, 224).astype(np.float32)
    fig = renderers.render_modality("depth", data, list(range(4)), frame=1)
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == 1


def test_mmwave_radar_point_cloud_layout():
    data = np.random.rand(4, 64, 5).astype(np.float32)
    data[data < 0.2] = 0.0  # sparse: most rows zero-padded
    fig = renderers.render_modality("mmwave", data, list(range(4)), frame=1)
    assert fig is not None
    assert len(fig.data) == 1  # single 3D scatter
    assert fig.data[0].type == "scatter3d"
    # aggregate overlays frames
    fig_agg = renderers.render_modality("mmwave", data, list(range(4)), frame=None)
    assert fig_agg is not None
    assert len(fig_agg.data) == 4
    assert all(t.type == "scatter3d" for t in fig_agg.data)


def test_lidar_initial_camera_looks_along_x():
    data = np.random.rand(4, 100, 3).astype(np.float32)
    fig = renderers.render_modality("lidar", data, list(range(4)), frame=1)
    cam = fig.layout.scene.camera.eye
    # eye on +X, looking toward origin
    assert cam.x > 0 and abs(cam.y) < 1e-6 and abs(cam.z) < 1e-6
    assert fig.layout.scene.aspectmode == "cube"
    assert fig.layout.scene.xaxis.range == (-4.5, 4.5)


def test_mmwave_fixed_cube_range():
    data = np.random.rand(4, 64, 5).astype(np.float32)
    data[data < 0.2] = 0.0
    fig = renderers.render_modality("mmwave", data, list(range(4)), frame=1)
    assert fig.layout.scene.aspectmode == "cube"
    assert fig.layout.scene.xaxis.range == (-3.5, 3.5)
    fig_agg = renderers.render_modality("mmwave", data, list(range(4)), frame=None)
    assert fig_agg.layout.scene.aspectmode == "cube"


def test_mmwave_xyz_extraction_cartesian():
    """_mmwave_xyz extracts the first three columns of valid points (Cartesian
    x, y, z). MMFi mmwave columns are `[x, y, z, doppler, intensity]`
    (confirmed 2026-08-20 against `mmwave_vae.py` which uses [:,:3] as 3D
    point cloud for FPS sampling and Chamfer losses). No coordinate
    transform is applied.
    """
    pts = np.array([[3.0, 1.0, -2.0, 0.4, 5.0],   # x=3, y=1, z=-2
                    [0.0, 0.0, 0.0, 0.0, 0.0]])   # zero-padding, filtered
    xyz = renderers._mmwave_xyz(renderers._mmwave_active(pts))
    assert xyz.shape == (1, 3)
    assert abs(xyz[0, 0] - 3.0) < 1e-6
    assert abs(xyz[0, 1] - 1.0) < 1e-6
    assert abs(xyz[0, 2] + 2.0) < 1e-6


def test_rgb_clamps_out_of_range_frame():
    data = np.random.rand(10, 17, 2).astype(np.float32)
    fig = renderers.render_modality("rgb", data, list(range(10)), frame=99)
    _check_figure(fig)


def test_depth_aggregate_and_frame():
    data = np.random.rand(5, 1, 224, 224).astype(np.float32)
    _check_figure(renderers.render_modality("depth", data, list(range(5)), frame=None))
    _check_figure(renderers.render_modality("depth", data, list(range(5)), frame=2))


def test_wifi_aggregate_and_frame():
    data = np.random.rand(5, 3, 114, 10).astype(np.float32)
    _check_figure(renderers.render_modality("wifi", data, list(range(5)), frame=None))
    _check_figure(renderers.render_modality("wifi", data, list(range(5)), frame=2))


def test_lidar_aggregate_and_frame():
    data = np.random.rand(5, 1536, 3).astype(np.float32)
    _check_figure(renderers.render_modality("lidar", data, list(range(5)), frame=None))
    _check_figure(renderers.render_modality("lidar", data, list(range(5)), frame=2))


def test_mmwave_aggregate_and_frame():
    data = np.random.rand(5, 64, 5).astype(np.float32)
    _check_figure(renderers.render_modality("mmwave", data, list(range(5)), frame=None))
    _check_figure(renderers.render_modality("mmwave", data, list(range(5)), frame=2))


def test_unknown_modality_returns_none():
    data = np.random.rand(5, 8).astype(np.float32)
    assert renderers.render_modality("infrared", data, list(range(5)), frame=None) is None


def test_1d_vector_fallback_render():
    data = np.random.rand(100).astype(np.float32)  # no known section layout
    fig = renderers.render_modality("wifi", data, list(range(100)), frame=None)
    _check_figure(fig)
    assert len(fig.data) == 1  # line only, no NaN markers
    assert fig.data[0].mode == "lines"


def test_1d_vector_fallback_with_nan():
    data = np.random.rand(63).astype(np.float32)
    data[5] = np.nan
    fig = renderers.render_modality("depth", data, list(range(63)), frame=None)
    assert len(fig.data) >= 2  # line + NaN markers


def test_structured_feature_uses_segmented_view():
    for name, length, n_sections in (("wifi", 161, 9), ("depth", 63, 6),
                                     ("lidar", 353, 6), ("mmwave", 134, 8)):
        data = np.random.rand(length).astype(np.float32)
        fig = renderers.render_modality(name, data, list(range(length)), frame=None)
        assert fig is not None
        assert len(fig.layout.annotations) == n_sections  # one panel per section
        assert fig.layout.title.text.startswith(f"{name} 结构化特征")


def test_unknown_length_falls_back_to_plain_line():
    data = np.random.rand(100).astype(np.float32)  # no known section layout
    fig = renderers.render_modality("depth", data, list(range(100)), frame=None)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"


def test_sections_cover_whole_range():
    for name, by_len in renderers._FEATURE_SECTIONS.items():
        for length, sections in by_len.items():
            assert sections[0][0] == 0
            assert sections[-1][1] == length
            prev = 0
            for s, e, _ in sections:
                assert s == prev
                assert e > s
                prev = e


def test_1d_vector_caps_at_2000_points():
    data = np.random.rand(5000).astype(np.float32)
    fig = renderers.render_modality("lidar", data, list(range(5000)), frame=None)
    assert len(fig.data[0].x) <= 2000


def test_bad_shape_returns_none():
    data = np.random.rand(5, 17, 2, 3).astype(np.float32)
    assert renderers.render_modality("rgb", data, list(range(5)), frame=None) is None


def test_modality_stats_text():
    data = np.zeros((3, 4), dtype=np.float32)
    s = renderers.modality_stats(data)
    assert "shape=[3, 4]" in s and "zeros=" in s


def test_registry_has_all_known():
    for name in ("rgb", "depth", "wifi", "lidar", "mmwave"):
        assert name in renderers.RENDERERS


def test_render_animated_single_frame_falls_back():
    # only 1 frame -> static single-frame figure (no animation)
    data = np.random.rand(1, 17, 2).astype(np.float32)
    fig = renderers.render_animated("rgb", data, list(range(1)))
    assert fig is not None
    assert fig.frames is None or len(fig.frames) == 0


def test_render_animated_has_slider_and_play_button():
    data = np.random.rand(20, 17, 2).astype(np.float32)
    fig = renderers.render_animated("rgb", data, list(range(20)), max_frames=20)
    assert fig is not None
    assert len(fig.frames) == 19  # frames 1..19
    assert len(fig.layout.sliders) == 1
    assert len(fig.layout.updatemenus) == 1
    # play button uses redraw:true so every modality (line/heatmap/3D) animates
    play_btn = fig.layout.updatemenus[0].buttons[0]
    assert play_btn.method == "animate"
    assert play_btn.args[1]["frame"]["redraw"] is True


def test_render_animated_depth_downsampled():
    data = np.random.rand(10, 1, 224, 224).astype(np.float32)
    fig = renderers.render_animated("depth", data, list(range(10)), max_frames=10,
                                    depth_downsample=4)
    assert fig is not None
    # base figure z shape reflects downsampled 56x56 (via heatmap z)
    z = fig.data[0].z
    assert z.shape == (56, 56)


def test_render_animated_lidar_points_capped():
    data = np.random.rand(10, 2000, 3).astype(np.float32)
    fig = renderers.render_animated("lidar", data, list(range(10)), max_frames=10,
                                    lidar_max_points=600)
    assert fig is not None
    assert len(fig.data[0].x) <= 600


def test_render_animated_respects_max_frames():
    data = np.random.rand(100, 17, 2).astype(np.float32)
    fig = renderers.render_animated("rgb", data, list(range(100)), max_frames=60)
    assert fig is not None
    assert len(fig.frames) == 59  # capped at 60 frames -> 59 frames after base


def test_render_animated_applies_camera_to_3d():
    data = np.random.rand(10, 600, 3).astype(np.float32)
    cam = {"eye": {"x": 2.5, "y": 1.0, "z": 2.0},
           "up": {"x": 0.0, "y": 0.0, "z": 1.0},
           "center": {"x": 0.0, "y": 0.0, "z": 0.0}}
    fig = renderers.render_animated("lidar", data, list(range(10)), max_frames=10,
                                    camera=cam)
    assert fig is not None
    assert fig.layout.scene is not None
    assert fig.layout.scene.camera.eye.x == 2.5
    assert fig.layout.scene.camera.eye.y == 1.0
    assert fig.layout.scene.camera.eye.z == 2.0
    assert fig.layout.scene.camera.up.z == 1.0
    assert fig.layout.scene.camera.center.x == 0.0


def test_render_animated_camera_ignored_for_2d():
    import json
    data = np.random.rand(10, 17, 2).astype(np.float32)
    cam = {"eye": {"x": 2.5, "y": 1.0, "z": 2.0}}
    fig = renderers.render_animated("rgb", data, list(range(10)), max_frames=10,
                                    camera=cam)
    assert fig is not None
    # rgb has no 3D scene; camera must NOT be applied (no scene.xaxis in layout)
    layout = json.loads(fig.to_json())["layout"]
    scene = layout.get("scene")
    assert scene is None or scene.get("xaxis") is None