#!/usr/bin/env python
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality import finegrained
from framework.eval.dataset_quality.modality_probe import extract_modality_feature_downsampled


def _concat_features(s):
    """样本 → 拼接各模态原始数据特征（depth 降采样 + 其余 mean-over-time）。
    与现有 dataset_quality (InfoScore) 同款特征，测纯数据属性，不做领域特征工程。
    不依赖任何主模型。"""
    return np.concatenate([extract_modality_feature_downsampled(s, m) for m in
                           ["wifi", "depth", "lidar", "mmwave", "rgb"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-split", default="train")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-cell", type=int, default=3)
    ap.add_argument("--w-compactness", type=float, default=0.4)
    ap.add_argument("--w-consistency", type=float, default=0.3)
    ap.add_argument("--w-separability", type=float, default=0.3)
    ap.add_argument("--mode", choices=["auto", "eager", "lazy"], default="eager",
                    help="dataset loading mode. Default eager: 集群 CPU 节点内存充足，"
                         "一次性载入避免 lazy 逐样本读盘（GPFS 慢）。")
    args = ap.parse_args()
    ds = load_dataset(args.dataset, mode=args.mode)
    samples = getattr(ds, args.eval_split)
    groups = finegrained.group_by_class_subject(samples)
    weights = {"compactness": args.w_compactness, "consistency": args.w_consistency,
               "separability": args.w_separability}
    result = finegrained.build_matrix(groups, extract_fn=_concat_features,
                                      weights=weights,
                                      top_k=args.top_k, min_cell=args.min_cell)
    result["dataset"] = args.dataset
    # version 从 meta.json 读（spec §五），fallback 到数据集目录名
    meta_path = os.path.join(args.dataset, "meta.json")
    v = json.load(open(meta_path)).get("version", "")
    result["version"] = v if v.startswith("v") else f"v{v}"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
