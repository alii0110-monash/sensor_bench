"""Per-sample review page: modality visualization + text/label/quality/note editing."""
from __future__ import annotations

from typing import Dict

from curation.caption.verbs import LABEL_TO_VERB
from curation.gui.core.cache import get_split_ids
from curation.gui.core.dataset_service import (
    make_label_lookup,
    parse_sample_id,
)
from curation.gui.core.filters import FilterSpec, apply_filters
from curation.gui.core.renderers import modality_stats, render_modality

_QUALITY_OPTIONS = ["", "golden", "ok", "reject", "flagged"]
_STATUS_BADGES = {
    "unreviewed": "gray", "edited": "orange", "golden": "green",
    "ok": "blue", "reject": "red", "flagged": "purple",
}


def _init_widget(key: str, value):
    import streamlit as st

    if key not in st.session_state:
        st.session_state[key] = value


def _widget_keys(sid: str):
    return {k: f"gui_{k}_{sid}" for k in ("text", "label", "quality", "note")}


def _reset_widgets(keys: Dict[str, str]):
    import streamlit as st

    for k in keys.values():
        st.session_state.pop(k, None)


def _status_badge(status: str) -> str:
    return f":{_STATUS_BADGES.get(status, 'gray')}[{status}]"


def _animated_cached(modality: str, data, frame_indices, max_frames: int,
                     camera):
    """Build one modality's animated figure. Wrapped with cache_resource so the
    figure survives browser refreshes (session_state is cleared on refresh, but
    a process-level cache does not). Callers clear a modality's cache entry
    when the camera changes."""
    from curation.gui.core.renderers import render_animated
    return render_animated(modality, data, frame_indices, max_frames=max_frames,
                           camera=camera)


import streamlit as st  # noqa: E402
_animated_cached = st.cache_resource(show_spinner=False, max_entries=64)(_animated_cached)


def _rebuild_anim_modality(ses, cam_key: str, act: str, mod_name: str) -> None:
    """Rebuild ONLY one modality's animated figure in the per-sample cache,
    so camera changes don't force a full rebuild of all 5 modalities (which
    causes a visible stall). Other modalities' figures are reused."""
    seg = ses.get("pb_seg")
    if not seg or mod_name not in seg:
        return
    data, fis = seg[mod_name][0], seg[mod_name][1]
    T = max(len(seg[m][1]) for m in seg if m != "__source__")
    cam = ses.get(cam_key, {}).get(mod_name) if mod_name in ("lidar", "mmwave") else None
    figs = ses.get("pb_anim") or {}
    figs[mod_name] = _animated_cached(mod_name, data, fis, min(T, 297), cam)
    ses["pb_anim"] = figs


def _render_camera_inputs(ses, cam_key: str, mod_name: str, act: str,
                          dataset_name: str, sid: str) -> None:
    """Manual 3D camera inputs for a 3D modality. Reviewer fills eye/up/center
    vectors and clicks 应用 to set a fixed view; playback keeps this camera.

    A plotly 3D camera is fully described by three vectors:
      - eye:    camera position (where we look from)
      - up:     camera up direction (rotation about the view axis)
      - center: look-at point (where we look toward)
    Only eye cannot express rotation, so all three are exposed.

    The current view lives in session_state (per sample). '保存视角' persists it
    to disk (view_store) so it survives restarts; '加载已存视角' reloads it.
    """
    import streamlit as st

    cur = ses[cam_key].get(mod_name) or {}
    eye = cur.get("eye", {})
    up = cur.get("up", {})
    center = cur.get("center", {})

    with st.expander(f"**{mod_name} 视角 (eye/up/center)**", expanded=False):
        st.caption("eye=相机位置 · up=朝上方向 · center=注视点")
        c1, c2, c3 = st.columns(3)
        ex = c1.number_input("eye.x", value=float(eye.get("x", 1.8)), key=f"cam_{act}_{mod_name}_ex", step=0.1)
        ey = c2.number_input("eye.y", value=float(eye.get("y", 0.0)), key=f"cam_{act}_{mod_name}_ey", step=0.1)
        ez = c3.number_input("eye.z", value=float(eye.get("z", 0.0)), key=f"cam_{act}_{mod_name}_ez", step=0.1)
        ux = c1.number_input("up.x", value=float(up.get("x", 0.0)), key=f"cam_{act}_{mod_name}_ux", step=0.1)
        uy = c2.number_input("up.y", value=float(up.get("y", 0.0)), key=f"cam_{act}_{mod_name}_uy", step=0.1)
        uz = c3.number_input("up.z", value=float(up.get("z", 1.0)), key=f"cam_{act}_{mod_name}_uz", step=0.1)
        cx = c1.number_input("center.x", value=float(center.get("x", 0.0)), key=f"cam_{act}_{mod_name}_cx", step=0.1)
        cy = c2.number_input("center.y", value=float(center.get("y", 0.0)), key=f"cam_{act}_{mod_name}_cy", step=0.1)
        cz = c3.number_input("center.z", value=float(center.get("z", 0.0)), key=f"cam_{act}_{mod_name}_cz", step=0.1)
        if st.button("应用视角", key=f"cam_{act}_{mod_name}_apply"):
            ses[cam_key][mod_name] = {
                "eye": {"x": float(ex), "y": float(ey), "z": float(ez)},
                "up": {"x": float(ux), "y": float(uy), "z": float(uz)},
                "center": {"x": float(cx), "y": float(cy), "z": float(cz)},
            }
            # rebuild only this modality's animated figure (not all 5)
            _rebuild_anim_modality(ses, cam_key, act, mod_name)
            st.rerun()
        if st.button("重置为默认视角", key=f"cam_{act}_{mod_name}_reset"):
            ses[cam_key][mod_name] = None
            # rebuild only this modality (reset to default camera)
            _rebuild_anim_modality(ses, cam_key, act, mod_name)
            st.rerun()
        b1, b2 = st.columns(2)
        if b1.button("💾 保存视角", key=f"cam_{act}_{mod_name}_save"):
            from curation.gui.core.view_store import save_view
            save_view(dataset_name, sid, mod_name, ses[cam_key][mod_name])
            st.toast(f"已保存 {mod_name} 视角")
        if b2.button("📂 加载已存视角", key=f"cam_{act}_{mod_name}_load"):
            from curation.gui.core.view_store import load_view
            saved = load_view(dataset_name, sid, mod_name)
            if saved:
                ses[cam_key][mod_name] = saved
                _rebuild_anim_modality(ses, cam_key, act, mod_name)
                st.rerun()
            else:
                st.toast("没有已保存的视角", icon="ℹ️")


def _render_modalities(ses, sample, raw_root, dataset_name: str) -> None:
    """Main modality view: whole-action player (all raw frames when available,
    else segment / sample frames).

    Each modality is a client-side animation figure (`render_animated`) with its
    own built-in play/pause/slider, so the reviewer plays any modality directly.
    No top-level global play toggle. Animated figures are cached per sample
    (built once on first render of this action, reused across reruns).
    """
    import streamlit as st

    from curation.gui.core.playback import (
        action_frames,
        canonical_segment,
        segment_frames,
    )
    from curation.gui.core.renderers import render_animated

    sid = sample.id
    act = f"{sample.meta.get('env')}_{sample.meta.get('subject')}_A{sample.label + 1:02d}"
    if ses.get("pb_cur_act") != act:
        for k in ("pb_seg", "pb_anim", "pb_anim_act"):
            ses.pop(k, None)
        ses["pb_cur_act"] = act
    if "pb_seg" not in ses:
        seg = (action_frames(sample, raw_root)
               or segment_frames(sample, raw_root)
               or canonical_segment(sample))
        if not seg:
            st.info("无可用帧数据")
            return
        ses["pb_seg"] = seg
    seg = ses["pb_seg"]
    src_label = {"raw_full": "整个动作",
                 "raw_segment": "动作段",
                 "canonical": "样本帧"}.get(seg.get("__source__"), "?")
    T = max(len(seg[mod][1]) for mod in seg if mod != "__source__")

    st.caption(f"A{sample.label + 1:02d} · {src_label} · {T} 帧 · 每个模态直接点 ▶ 播放")

    # Manual 3D camera (eye) for lidar/mmwave — reviewer can set a fixed view.
    # Stored per action in session_state; applied to the animated figure.
    # On first view of this sample, seed from any persisted view on disk.
    _3D_MODALS = ("lidar", "mmwave")
    cam_key = f"pb_cam_{sid}"
    if cam_key not in ses:
        from curation.gui.core.view_store import load_view
        cam = {}
        for m in _3D_MODALS:
            cam[m] = load_view(dataset_name, sid, m)
        ses[cam_key] = cam

    # Animated figures are cached per sample (built once on first render of
    # this action, reused across reruns). Rebuild cache if action changed.
    if ses.get("pb_anim_act") != act:
        ses["pb_anim"] = None
        ses["pb_anim_act"] = act

    for mod_name in seg:
        if mod_name == "__source__":
            continue
        data, fis = seg[mod_name][0], seg[mod_name][1]
        if ses.get("pb_anim") is None:
            # build all animated figures once per action
            figs = {}
            for m in seg:
                if m == "__source__":
                    continue
                d, f = seg[m][0], seg[m][1]
                cam = ses[cam_key].get(m) if m in _3D_MODALS else None
                af = _animated_cached(m, d, f, min(T, 297), cam)
                if af is not None:
                    figs[m] = af
            ses["pb_anim"] = figs
        figs = ses["pb_anim"]
        fig = figs.get(mod_name)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key=f"pb_anim_{sid}_{mod_name}")
        else:
            with st.expander(f"{mod_name} — 无法播放渲染", expanded=False):
                st.code(modality_stats(data))
        # Manual camera controls for 3D modalities
        if mod_name in _3D_MODALS:
            _render_camera_inputs(ses, cam_key, mod_name, act, dataset_name, sid)


def render(ses) -> None:
    import streamlit as st

    ds, root, split = ses.ds, ses.dataset, ses.split
    edit_log, preds = ses.edit_log, ses.preds
    ids = get_split_ids(root, split)
    idx = {sid: i for i, sid in enumerate(ids)}

    def lookup(sid):
        i = idx.get(sid)
        return ds.splits[split][i] if i is not None else None

    label_lookup = make_label_lookup(ses, ses.dataset, split, ds, ids)
    statuses = {sid: edit_log.status(sid) for sid in ids}

    st.header(f"逐样本审查 — `{root}` / `{split}`  ({len(ids)} 样本)")

    # ------------------------------------------------------- filters
    with st.sidebar:
        st.subheader("过滤器")
        label = st.selectbox(
            "类别", [None] + list(range(27)),
            format_func=lambda l: "全部" if l is None else f"{l} {LABEL_TO_VERB(l)}")
        subject = st.text_input("subject（如 S33）", value="")
        status = st.selectbox(
            "审阅状态", [""] + ["unreviewed", "edited", "golden", "ok", "reject", "flagged"],
            format_func=lambda s: "全部" if s == "" else s)
        pred_wrong = None
        if preds:
            pw = st.radio("预测对照", ["全部", "仅预测错", "仅预测对"], index=0, horizontal=True)
            pred_wrong = None if pw == "全部" else (pw == "仅预测错")
        has_note = st.checkbox("仅看有备注的", value=False)
    spec = FilterSpec(label=label, subject=subject.strip() or None,
                      status=status or None, pred_wrong=pred_wrong,
                      has_note=has_note or None)
    filtered = apply_filters(ids, label_lookup, preds, statuses, edit_log.fields, spec)
    if not filtered:
        st.warning("没有匹配的样本")
        return
    st.caption(f"过滤后 {len(filtered)} / {len(ids)}")

    # ------------------------------------------------------- picker + nav
    # Navigation never writes the widget key directly (Streamlit forbids that
    # once the widget exists). Buttons only set a plain `gui_nav_pending` key;
    # the script body applies it to the widget key BEFORE the widget is
    # instantiated in this run, which Streamlit allows.
    nav_pending = st.session_state.pop("gui_nav_pending", None)
    if nav_pending is not None:
        st.session_state.gui_picker = nav_pending
    if st.session_state.get("gui_picker") not in filtered:
        st.session_state.gui_picker = filtered[0]
    sel = st.selectbox("样本", filtered, key="gui_picker",
                       format_func=lambda s: f"{s}  [{statuses[s]}]")

    def _nav(delta: int) -> None:
        try:
            i = filtered.index(st.session_state.gui_picker)
        except ValueError:
            i = 0
        st.session_state.gui_nav_pending = filtered[(i + delta) % len(filtered)]

    pos = idx[sel]
    cprev, cnext, cpos = st.columns([1, 1, 3])
    cprev.button("◀ 上一个", key="gui_prev", on_click=_nav, args=(-1,))
    cnext.button("下一个 ▶", key="gui_next", on_click=_nav, args=(1,))
    cpos.caption(f"位置 {pos + 1} / {len(ids)}（全集） · 过滤内 {filtered.index(sel) + 1}/{len(filtered)}")

    sid = sel
    s = lookup(sid)
    if s is None:
        st.error("样本读取失败（文件可能缺失或损坏）")
        return

    # ------------------------------------------------------- header
    p = parse_sample_id(sid)
    st.subheader(f"{sid}  {_status_badge(statuses[sid])}")
    meta_extra = ", ".join(f"{k}={v}" for k, v in s.meta.items() if k in ("env", "source"))
    st.caption(f"label {s.label} {LABEL_TO_VERB(s.label)}"
               + (f" | subject {p['subject']} | env {p['ep']}" if p else "")
               + (f" | {meta_extra}" if meta_extra else ""))
    if preds.get(sid):
        pr = preds[sid]
        verdict = "✓ 对" if pr["pred"] == s.label else "✗ 错"
        st.markdown(f"**模型预测**: {pr['pred']} {LABEL_TO_VERB(pr['pred'])} "
                    f"(conf {pr['conf']:.3f}, {pr['source']}) **{verdict}**")
    orig_text = s.text.get("captions", []) if isinstance(s.text, dict) else []
    if orig_text:
        with st.expander("原始文本（未改）", expanded=False):
            for i, c in enumerate(orig_text):
                st.write(f"{i + 1}. {c}")

    # ------------------------------------------------------- modalities (full-segment player)
    from curation.gui.core.dataset_service import dataset_name
    _render_modalities(ses, s, getattr(ses, "raw_root", None),
                       dataset_name(ses.dataset))

    # ------------------------------------------------------- edit form
    keys = _widget_keys(sid)
    prev = edit_log.fields(sid)
    _init_widget(keys["text"], "\n".join(prev.get("text", orig_text)))
    _init_widget(keys["label"], prev.get("label", s.label))
    _init_widget(keys["quality"], prev.get("quality", ""))
    _init_widget(keys["note"], prev.get("note", ""))

    st.divider()
    st.subheader("修正")
    st.text_area("文本（每行一条 caption）", key=keys["text"], height=160,
                 help="原文本供参考；留空行会被删除")
    c1, c2 = st.columns(2)
    c1.selectbox("标签", list(range(27)), key=keys["label"],
                 format_func=lambda l: f"{l} {LABEL_TO_VERB(l)}")
    c2.radio("质量标记", _QUALITY_OPTIONS, key=keys["quality"], horizontal=True,
             format_func=lambda q: "不标记" if q == "" else q)
    st.text_area("备注", key=keys["note"], height=80)
    b1, b2, b3 = st.columns(3)
    if b1.button("💾 保存修改", key=f"save_{sid}", type="primary"):
        _save(ses, sid, s, keys)
        st.rerun()
    if b2.button("仅记录审阅状态", key=f"flag_{sid}"):
        q = st.session_state[keys["quality"]] or None
        note = (st.session_state[keys["note"]] or "").strip() or None
        edit_log.flag(sid, q, note)
        st.toast("已记录审阅状态")
        st.rerun()
    if b3.button("↩ 回滚最近修改", key=f"rollback_{sid}"):
        if edit_log.rev(sid) > 0:
            edit_log.rollback(sid)
            _reset_widgets(keys)
            st.toast("已回滚")
        else:
            st.toast("没有可回滚的修改")
        st.rerun()

    if edit_log.rev(sid) > 0:
        ch = edit_log.changes(sid)
        if ch:
            with st.expander("本样本的修改记录", expanded=False):
                for k, (old, new) in ch.items():
                    st.write(f"**{k}**: `{old}` → `{new}`")


def _save(ses, sid: str, sample, keys: Dict[str, str]) -> None:
    import streamlit as st

    edit_log = ses.edit_log
    prev = edit_log.fields(sid)
    new_text = [l.strip() for l in (st.session_state[keys["text"]] or "").splitlines() if l.strip()]
    new_label = int(st.session_state[keys["label"]])
    new_quality = st.session_state[keys["quality"]] or None
    new_note = (st.session_state[keys["note"]] or "").strip() or None

    orig_text = sample.text.get("captions", []) if isinstance(sample.text, dict) else []
    fields: Dict[str, object] = {}
    changed: Dict[str, list] = {}

    cur_text = prev.get("text", orig_text)
    if new_text != cur_text:
        fields["text"] = new_text
        changed["text"] = [cur_text, new_text]
    cur_label = prev.get("label", sample.label)
    if new_label != cur_label:
        fields["label"] = new_label
        changed["label"] = [cur_label, new_label]
    cur_q = prev.get("quality")
    if new_quality != cur_q:
        fields["quality"] = new_quality
        changed["quality"] = [cur_q, new_quality]
    cur_note = prev.get("note")
    if new_note != cur_note:
        fields["note"] = new_note
        changed["note"] = [cur_note, new_note]

    if fields:
        edit_log.save(sid, fields, changed)
        st.toast("已保存")
    else:
        st.toast("无改动", icon="ℹ️")