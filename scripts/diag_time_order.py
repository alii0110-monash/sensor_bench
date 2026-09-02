#!/usr/bin/env python
"""诊断：token_fusion 是否理解时间顺序？

把每个样本的时间帧顺序打乱（shuffle frame axis），对比 acc 是否变化。
- 若 acc 几乎不变 → 模型没看时间顺序（encoder mean 坍缩了 T 维）
- 若 acc 显著下降 → 模型抓到了时序

用法：conda run -n sensorbench python scripts/diag_time_order.py \
  --dataset datasets/mmfi/v5_structfeat --ckpt checkpoints_v5_structfeat_v2/token_fusion_seed0.pt
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.dataset.sample import Sample
from framework.harness.evaluate import evaluate_model
from framework.models.token_fusion import TokenFusionModel, MODALITIES


def shuffled_sample(sample, rng):
    """Return a copy of sample with frame axis (axis 0) permuted.
    Only multi-frame modalities (ndim >= 2) have a time axis to shuffle;
    1-D features (v5_structfeat) are left unchanged."""
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
    ap.add_argument("--dataset", default="datasets/mmfi/v5_structfeat")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, mode="lazy")
    val = list(ds.val)[:args.limit]
    m = TokenFusionModel.load(args.ckpt)
    print(f"dataset={args.dataset.split('/')[-1]}  ckpt={os.path.basename(args.ckpt)}  "
          f"n_val={len(val)}  T={next(iter(val[0].modalities.values())).data.shape[0]}")

    # original
    orig = evaluate_model(m, val, {"id": "full", "available": MODALITIES})["accuracy"]
    print(f"original acc_full:      {orig:.4f}")

    # shuffled (3 independent permutations, averaged)
    rng = np.random.default_rng(0)
    shuf_accs = []
    for _ in range(3):
        shuf_val = [shuffled_sample(s, rng) for s in val]
        shuf_accs.append(
            evaluate_model(m, shuf_val, {"id": "shuf", "available": MODALITIES})["accuracy"])
    mean_shuf = float(np.mean(shuf_accs))
    print(f"shuffled acc_full (x3):  {[f'{a:.4f}' for a in shuf_accs]}  mean={mean_shuf:.4f}")
    print(f"delta = {mean_shuf - orig:+.4f}")
    if abs(mean_shuf - orig) < 0.01:
        print("=> 模型不感知时间顺序（encoder mean 坍缩 T 维）")
    else:
        print("=> 模型部分依赖时间顺序")


if __name__ == "__main__":
    main()
