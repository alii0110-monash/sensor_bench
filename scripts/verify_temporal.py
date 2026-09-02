#!/usr/bin/env python
"""快速机制验证：时间建模是否让模型感知帧序。

用 v4 raw 小样本子集，lazy 加载，训练 temporal=True 和 temporal=False 两个
token_fusion，然后各自打乱帧序看 acc 变化：
- non-temporal: 打乱帧序 acc 应不变（mean 坍缩 T 维）
- temporal:     打乱帧序 acc 应下降（RoPE + 时序注意力感知顺序）

用法：conda run -n sensorbench python scripts/verify_temporal.py
"""
import argparse, os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.dataset.sample import Sample
from framework.harness.evaluate import evaluate_model
from framework.models.base import TrainConfig
from framework.models.token_fusion import TokenFusionModel, MODALITIES


def shuffled_sample(sample, rng):
    """Permute frame axis (axis 0) of multi-frame modalities."""
    T = None
    for mod in sample.modalities.values():
        if mod.data.ndim >= 2:
            T = mod.data.shape[0]
            break
    if T is None:
        return sample
    perm = rng.permutation(T)
    mods = {}
    for name, mod in sample.modalities.items():
        if mod.data.ndim >= 2:
            mods[name] = mod.__class__(
                data=mod.data[perm],
                frame_indices=[int(mod.frame_indices[p]) for p in perm],
                sample_rate=mod.sample_rate, name=name)
        else:
            mods[name] = mod
    return Sample(id=sample.id, label=sample.label, modalities=mods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints_temporal_verify")
    args = ap.parse_args()

    ds = load_dataset("datasets/mmfi/v4", mode="lazy")
    train = list(ds.train)[:args.n_train]
    val = list(ds.val)[:args.n_val]
    print(f"train {len(train)}, val {len(val)}")

    for temporal in (False, True):
        tag = "temporal" if temporal else "baseline"
        out_dir = os.path.join(args.out_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        m = TokenFusionModel(num_classes=27, temporal=temporal)
        cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch, seed=args.seed,
                          out_dir=out_dir, device="cuda", patience=3)
        m.fit(train, val, cfg)
        m = TokenFusionModel.load(f"{cfg.out_dir}/token_fusion_seed{args.seed}.pt")
        # original acc
        orig = evaluate_model(m, val, {"id": "full", "available": MODALITIES})["accuracy"]
        # shuffled acc (x3)
        rng = np.random.default_rng(0)
        shuf_accs = []
        for _ in range(3):
            shuf_val = [shuffled_sample(s, rng) for s in val]
            shuf_accs.append(evaluate_model(m, shuf_val, {"id": "s", "available": MODALITIES})["accuracy"])
        mean_shuf = float(np.mean(shuf_accs))
        delta = mean_shuf - orig
        verdict = "感知时间" if abs(delta) > 0.01 else "不感知时间"
        print(f"[{tag}] orig={orig:.4f} shuffled={mean_shuf:.4f} delta={delta:+.4f} => {verdict}")


if __name__ == "__main__":
    main()
