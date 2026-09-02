"""Aggregate dashboard page: dataset overview + quality metrics + health stats."""
from __future__ import annotations

import json
import os

import plotly.graph_objects as go

from curation.caption.verbs import LABEL_TO_VERB
from curation.gui.core.cache import get_dataset, get_quality_json
from curation.gui.core.dataset_service import compute_health


def _render_quality(q: dict, root: str):
    import streamlit as st

    meta = q.get("metadata", {})
    st.subheader("质量指标")
    info = q.get("info", {})
    compact = q.get("compact", {})
    clean = q.get("clean", {})
    cols = st.columns(5)
    cols[0].metric("Quality", f"{q.get('quality', float('nan')):.3f}")
    cols[1].metric("InfoScore", f"{info.get('InfoScore', float('nan')):.3f}")
    cols[2].metric("CompactScore", f"{compact.get('CompactScore', float('nan')):.3f}")
    cols[3].metric("CleanScore", f"{clean.get('CleanScore', float('nan')):.3f}")
    cols[4].metric("val 样本", meta.get("val_sample_count", "—"))
    if meta.get("eval_split"):
        st.caption(f"probe eval split: {meta['eval_split']} | epochs {meta.get('probe_epochs')} "
                   f"| hidden {meta.get('probe_hidden_dim')}")

    acc = info.get("acc_per_modality")
    if acc:
        st.markdown("**per-modality probe acc**（MLP probe，val 评测）")
        fig = go.Figure(go.Bar(
            x=list(acc.keys()), y=[acc[k] for k in acc],
            text=[f"{acc[k]:.3f}" for k in acc], textposition="outside"))
        fig.add_hline(y=1 / 27, line_dash="dash", line_color="red",
                      annotation_text="random 1/27")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        st.plotly_chart(fig, width="stretch")

    cm = compact.get("confusion_matrix")
    if cm:
        st.markdown("**compactness 混淆矩阵**（concat probe，val）")
        labels = [LABEL_TO_VERB(i) for i in range(len(cm))]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Viridis",
            hovertemplate="true=%{y}<br>pred=%{x}<br>n=%{z}<extra></extra>"))
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(size=9))
        st.plotly_chart(fig, width="stretch")


def _render_health(health: dict):
    import streamlit as st

    st.subheader("数据健康统计")
    st.caption(f"扫描 {health['n_scanned']} 个样本（懒加载，最多 2000）")

    label_dist = health["label_dist"]
    if label_dist:
        labels = [f"{k} {LABEL_TO_VERB(k)}" for k in sorted(label_dist)]
        fig = go.Figure(go.Bar(x=labels, y=[label_dist[k] for k in sorted(label_dist)],
                               text=[label_dist[k] for k in sorted(label_dist)],
                               textposition="outside"))
        fig.update_layout(title="label 分布", height=360, xaxis_tickangle=-45,
                          margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(size=9))
        st.plotly_chart(fig, width="stretch")

    subj = health["subject_dist"]
    if subj:
        fig = go.Figure(go.Bar(x=list(subj.keys()), y=list(subj.values())))
        fig.update_layout(title="subject 分布", height=280, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        st.plotly_chart(fig, width="stretch")

    if health["frame_dist"]:
        fig = go.Figure()
        for mod, dist in health["frame_dist"].items():
            xs = sorted(dist)
            fig.add_trace(go.Scatter(x=xs, y=[dist[x] for x in xs], mode="lines+markers", name=mod))
        fig.update_layout(title="帧数分布（每模态）", height=300,
                          margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        st.plotly_chart(fig, width="stretch")

    anom = health["anomalies"]
    if anom:
        st.markdown("**异常检查**（NaN/Inf · 全零 · 全同值 · 空 · 不可读）")
        rows = []
        for mod, c in anom.items():
            rows.append({"modality": mod, **{k: v for k, v in c.items()}})
        if rows:
            st.dataframe(rows, width="stretch")


def render(ses) -> None:
    import streamlit as st

    ds, root = ses.ds, ses.dataset
    st.header(f"聚合看板 — `{root}` / `{ses.split}`")
    meta = ds.meta
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("version", meta.get("version", "?"))
    c2.metric("train", len(ds.splits.get("train", [])))
    c3.metric("val", len(ds.splits.get("val", [])))
    c4.metric("test", len(ds.splits.get("test", [])))
    st.caption("modalities: " + ", ".join(ds.modalities))
    if meta.get("changelog"):
        st.caption("changelog: " + " | ".join(meta["changelog"]))

    qpath = get_quality_json(root)
    if qpath:
        try:
            _render_quality(json.load(open(qpath)), root)
        except Exception as e:  # quality json shape changed -> degrade gracefully
            st.warning(f"quality JSON 读取失败: {e}")
    else:
        st.info("未找到匹配的 `results/quality_*.json`（按 dataset 路径匹配），仅显示数据健康统计。")

    @st.cache_data(show_spinner=False, ttl=600)
    def _health(root: str, split: str, max_samples: int):
        dsx = get_dataset(root, split)
        it = (dsx.splits[split][i] for i in range(len(dsx.splits[split])))
        return compute_health(it, max_samples=max_samples)

    health = _health(root, ses.split, int(os.environ.get("CURATION_HEALTH_MAX", "2000")))
    _render_health(health)