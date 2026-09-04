"""SFTMVP Playground — 传感器伪 token 问答演示页.

Launch: streamlit run demo/playground.py --server.port 8560 --server.headless true
Set SFTMVP_DEMO_FAKE=1 for a stub engine (UI smoke tests).
"""
import os
import random
import sys

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from curation.gui.core.renderers import render_modality
from framework.llm_sft.demo import DEFAULT_QUESTION, make_engine
from framework.models.alignment import MODALITIES

st.set_page_config(page_title="SFTMVP Playground", page_icon="🛰", layout="wide")

MOD_LABELS = {"wifi": "WiFi CSI", "depth": "Depth 深度图", "lidar": "LiDAR 点云",
              "mmwave": "mmWave 雷达", "rgb": "RGB 骨架关键点"}


@st.cache_resource(show_spinner="加载模型与数据（首次 ~1 分钟）...")
def get_engine():
    return make_engine(_ROOT)


engine = get_engine()
ids = engine.sample_ids()

st.title("🛰 SensorBench 伪 token 问答演示")
st.caption(
    "传感器信号 → 80 个伪 token → Qwen2.5-0.5B（LoRA SFT）。"
    "模型只会回答 27 类动作短语；自由提问与关闭传感器均属演示性外推。"
    f"验证样本数：{len(ids)}")

with st.sidebar:
    st.header("样本")
    if st.button("🎲 随机样本"):
        st.session_state["sid"] = random.choice(ids)
    sid = st.selectbox("验证集样本", ids,
                       index=st.session_state.get("sid_idx", 0),
                       key="sid_selector")
    st.session_state["sid_idx"] = ids.index(sid) if sid in ids else 0

    st.header("传感器开关（缺模态演示）")
    toggles = {m: st.checkbox(MOD_LABELS[m], value=True, key=f"tg_{m}")
               for m in MODALITIES}
    n_off = sum(1 for v in toggles.values() if not v)

    st.header("提问")
    question = st.text_area("你的问题", value=DEFAULT_QUESTION, height=80)
    compare = st.checkbox("同时显示「全模态」回答对比", value=True)
    go = st.button("🚀 提问", type="primary", use_container_width=True, key="go")

sample = engine.get_sample(sid)
gold = engine.class_map.get(sample.label, str(sample.label))
st.markdown(f"**样本** `{sid}` ｜ **真实动作**：{gold}")

if go or st.session_state.get("auto_run", False):
    st.session_state["auto_run"] = True
    avail = toggles
    res = engine.answer(sample, avail=avail, question=question)
    warn = []
    if n_off:
        warn.append(f"⚠ 已关闭 {n_off} 个传感器——该配置模型从未训练过，回答用于观察退化")
    if question.strip() != DEFAULT_QUESTION:
        warn.append("⚠ 自由提问：模型只按固定问题模板训练过，回答仅供参考")
    for w in warn:
        st.warning(w)

    if compare and n_off:
        full = engine.answer(sample, avail={m: True for m in MODALITIES},
                             question=question)
        c1, c2 = st.columns(2)
        c1.subheader("全部传感器开启")
        c1.success(f"**{full['text']}**\n\n判定类别：{full['class_name'] or '(未匹配)'}")
        c2.subheader(f"当前开关（关 {n_off} 个）")
        tone = st.error if res["label"] != full["label"] else st.success
        tone(f"**{res['text']}**\n\n判定类别：{res['class_name'] or '(未匹配)'}")
        if res["label"] != full["label"]:
            c2.caption("答案随缺模态改变——缺模态鲁棒性的直观展示")
    else:
        st.success(f"**{res['text']}**\n\n判定类别：{res['class_name'] or '(未匹配)'}")
    if res["top3"]:
        st.caption("匹配分数 Top3：" + " ｜ ".join(
            f"{name} ({s})" for name, s in res["top3"]))

    st.divider()
    st.subheader("传感器信号（当前可用模态）")
    for m in MODALITIES:
        if not avail[m]:
            st.info(f"{MOD_LABELS[m]}：已关闭")
            continue
        fig = render_modality(m, sample.modalities[m].data,
                              sample.modalities[m].frame_indices)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"fig_{m}")
else:
    st.info("左侧选样本、开/关传感器，点「🚀 提问」开始。")
