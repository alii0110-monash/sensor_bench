"""Headless smoke tests for the Streamlit app (streamlit.testing.v1.AppTest).

These run the real app against datasets/mmfi/v4 (present in the repo) and
assert the two pages render without exceptions.
"""
from __future__ import annotations

import os

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "..", "curation", "gui", "app.py")


@pytest.fixture
def fast_health(monkeypatch):
    monkeypatch.setenv("CURATION_HEALTH_MAX", "50")


@pytest.mark.slow
def test_review_page_smoke():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    assert any(s.label == "样本" for s in at.selectbox)  # sample picker present
    # switching pages works (index 0 is the review page's quality radio)
    at.radio[1].set_value("聚合看板").run()
    assert not at.exception


@pytest.mark.slow
def test_dashboard_page_smoke(fast_health):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio[1].set_value("聚合看板").run()
    assert not at.exception
    # quality metrics render (quality_v4.json matches datasets/mmfi/v4)
    assert any(m.label == "Quality" for m in at.metric)
    # health stats render at least one chart
    assert len(at.get("plotly_chart")) >= 1


@pytest.mark.slow
def test_nav_buttons_after_filter_change():
    """Regression: prev/next must never raise StreamlitAPIException on the
    widget key, including after the filter set changes."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    label_box = next(s for s in at.selectbox if s.label == "类别")
    label_box.set_value(0).run()
    assert not at.exception
    at.button(key="gui_next").click().run()
    assert not at.exception
    label_box = next(s for s in at.selectbox if s.label == "类别")
    label_box.set_value(1).run()
    assert not at.exception
    at.button(key="gui_next").click().run()
    assert not at.exception
    at.button(key="gui_prev").click().run()
    assert not at.exception


@pytest.mark.slow
def test_dataset_switch_via_sidebar():
    """Switching dataset/split from the sidebar reloads the session cleanly."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    dsbox = next(s for s in at.selectbox if s.label == "数据集")
    spbox = next(s for s in at.selectbox if s.label == "split")
    assert spbox.value == "val"
    # switch to the structured-feature dataset
    if "datasets/mmfi/v5_structfeat" in dsbox.options:
        dsbox.set_value("datasets/mmfi/v5_structfeat").run()
        assert not at.exception
        # structured modalities render as segmented feature views
        assert any(s.label == "样本" for s in at.selectbox)
        # switch back to v4
        dsbox = next(s for s in at.selectbox if s.label == "数据集")
        dsbox.set_value("datasets/mmfi/v4").run()
        assert not at.exception


@pytest.mark.slow
def test_v5_structfeat_all_modalities_render():
    """On the structured-feature dataset, all 5 modalities must render a
    plotly chart (wifi/depth/lidar/mmwave as segmented feature views, rgb as
    skeleton). Guards the mmwave 134d / wifi 161d feature-dim changes."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    dsbox = next(s for s in at.selectbox if s.label == "数据集")
    if "datasets/mmfi/v5_structfeat" not in dsbox.options:
        pytest.skip("v5_structfeat dataset not present")
    dsbox.set_value("datasets/mmfi/v5_structfeat").run()
    assert not at.exception
    # review page renders one plotly chart per modality (5 total). When the
    # raw MMFi dataset is present the review page shows raw frames (wifi CSI,
    # depth, lidar, mmwave 3D, rgb skeleton); otherwise it falls back to the
    # canonical structured-feature segmented views. Either way all 5
    # modalities must render without exception.
    charts = at.get("plotly_chart")
    assert len(charts) >= 5, f"expected >=5 modality charts, got {len(charts)}"
    import json as _json
    titles = []
    for c in charts:
        try:
            spec = _json.loads(c.proto.spec)
            titles.append(spec.get("layout", {}).get("title", {}).get("text", ""))
        except Exception:
            continue
    joined = " ".join(titles)
    # every modality must appear in some chart title
    for mod in ("wifi", "depth", "lidar", "mmwave", "rgb"):
        assert mod in joined, f"missing render for {mod}"