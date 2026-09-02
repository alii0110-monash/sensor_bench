"""Dataset Curation GUI — Streamlit entry point.

Run:
    streamlit run curation/gui/app.py -- --dataset datasets/mmfi/v4 --split val
Optional:
    --predictions results/predictions_val_v4.json
The dataset / split can also be switched freely from the sidebar at runtime.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the project root importable regardless of CWD.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st  # noqa: E402

from curation.gui.core.cache import get_dataset, get_roots  # noqa: E402
from curation.gui.core.dataset_service import dataset_name, find_raw_root  # noqa: E402
from curation.gui.core.edit_log import EditLog  # noqa: E402
from curation.gui.core.prediction_loader import load_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Dataset curation GUI")
    p.add_argument("--dataset", default="datasets/mmfi/v4", help="dataset root (Dataset protocol)")
    p.add_argument("--split", default="val", help="split name (any split in splits/)")
    p.add_argument("--predictions", default=None, help="optional predictions JSON file")
    p.add_argument("--raw-root", default=None,
                   help="raw MMFi dataset root (auto-detected in ~/datasets/MMFi_dataset)")
    return p.parse_args(argv)


def _split_names(root: str) -> list:
    splits_dir = os.path.join(root, "splits")
    if not os.path.isdir(splits_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(splits_dir) if f.endswith(".json"))


def _load_dataset_session(dataset: str, split: str) -> None:
    """(Re)build the session for a (dataset, split) pair. Clears per-sample
    navigation state so stale picker values never leak across datasets."""
    ds = get_dataset(dataset, split)
    if split not in ds.splits or not ds.splits[split]:
        raise ValueError(f"split {split!r} 为空或不存在。可用 splits: {list(ds.splits)}")
    for k in ("gui_picker", "gui_nav_pending"):
        st.session_state.pop(k, None)
    st.session_state.ds = ds
    st.session_state.dataset = dataset
    st.session_state.split = split
    st.session_state.edit_log = EditLog(
        os.path.join(PROJECT_ROOT, "curation", "gui", "edits",
                     f"{dataset_name(dataset)}-{split}.jsonl"))


def main() -> None:
    st.set_page_config(page_title="Dataset Curation GUI", layout="wide")
    args = parse_args()
    roots = get_roots()
    if not roots:
        st.error("未发现任何数据集根（datasets/ 下需要 meta.json + data/ + splits/）")
        st.stop()

    # ------------------------------------------------------- sidebar
    with st.sidebar:
        st.title("Dataset Curation GUI")
        if args.dataset in roots:
            dflt_ds = args.dataset
        else:
            dflt_ds = roots[0]
        dataset = st.selectbox("数据集", roots, index=roots.index(dflt_ds), key="sel_dataset")
        ds_splits = _split_names(dataset)
        if not ds_splits:
            st.error(f"{dataset} 没有 split 文件")
            st.stop()
        dflt_split = args.split if (dataset == args.dataset and args.split in ds_splits) else ds_splits[0]
        split = st.selectbox("split", ds_splits, index=ds_splits.index(dflt_split), key="sel_split")

        try:
            if (st.session_state.get("active_dataset") != dataset
                    or st.session_state.get("active_split") != split):
                _load_dataset_session(dataset, split)
                st.session_state.active_dataset = dataset
                st.session_state.active_split = split
                # predictions only make sense for the dataset they were computed for
                if dataset == args.dataset:
                    st.session_state.preds = load_predictions(args.predictions)
                else:
                    st.session_state.preds = {}
        except ValueError as e:
            st.error(str(e))
            st.stop()

        ses = st.session_state
        ses.raw_root = find_raw_root(args.raw_root)
        if ses.raw_root:
            st.caption(f"raw 数据: `{ses.raw_root}` (depth 原始分辨率)")
        st.caption(f"编辑日志: `{os.path.basename(ses.edit_log.path)}`")
        st.caption(f"已编辑样本: **{len(ses.edit_log.edited_ids())}**")
        if ses.preds:
            st.caption(f"预测文件: **{os.path.basename(args.predictions or '')}** ({len(ses.preds)} 样本)")
        page = st.radio("页面", ["逐样本审查", "聚合看板"])

    # ------------------------------------------------------- page body
    if page == "聚合看板":
        from curation.gui.pages import dashboard
        dashboard.render(ses)
    else:
        from curation.gui.pages import review
        review.render(ses)


if __name__ == "__main__":
    main()