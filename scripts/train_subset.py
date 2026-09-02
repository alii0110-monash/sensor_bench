#!/usr/bin/env python
"""缩小样本验证 sensorbench 训练流程。只取 v4 前 N 个样本跑 1 epoch，验证 GPU 训练正确性。"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel
from framework.models.base import TrainConfig

MODELS = {"token_fusion": TokenFusionModel, "late_fusion": LateFusionModel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="token_fusion", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=500, help="train 样本数上限")
    ap.add_argument("--n-val", type=int, default=100, help="val 样本数上限")
    ap.add_argument("--out-dir", default="checkpoints_smoke")
    args = ap.parse_args()

    # lazy 加载，只取子集，避免 18G 全量 eager
    ds = load_dataset(args.dataset, mode="lazy")

    train = ds.train[: args.n_train] if hasattr(ds.train, "__getitem__") else list(ds.train)[: args.n_train]
    val = ds.val[: args.n_val] if hasattr(ds.val, "__getitem__") else list(ds.val)[: args.n_val]
    # 确保含 rgb（5 模态），否则报 KeyError
    print(f"train subset: {len(train)}  val subset: {len(val)}")
    print(f"modalities: {list(train[0].modalities.keys())}")
    assert "rgb" in train[0].modalities, "v4 需要 rgb 模态"

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=1e-3,
                      seed=args.seed, out_dir=args.out_dir)
    model = MODELS[args.model](num_classes=27)
    model.fit(train, val, cfg)
    print(f"trained {args.model} seed {args.seed} on subset -> {cfg.out_dir}")


if __name__ == "__main__":
    main()
