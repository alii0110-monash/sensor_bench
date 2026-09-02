"""Sanity controls for layer_cka_v4 values (2026-09-02).

回答"CKA 数值是否合理"的三个对照:
  1. Permutation null — 打乱样本配对 (30 次) → 在当前 N/D 下的经验随机地板
  2. Random-weight model — 同架构随机初始化 → 分离"架构同质化"与"学到的融合"
  3. Sample-size sensitivity — N ∈ {200,500,1000,1870} → 偏差随 N 是否稳定

Run: sbatch jobs/layer_cka_controls.slurm  (~5-8 min, normal_test CPU)
Out: results/layer_cka_controls.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framework.dataset.loader import load_dataset  # noqa: E402
from framework.models.token_fusion import TokenFusionModel, MODALITIES  # noqa: E402
from framework.eval.dataset_quality.layer_cka import (  # noqa: E402
    extract_layerwise_features, linear_cka)

CKPT = ROOT / 'checkpoints_v4_temporal' / 'token_fusion_seed0.pt'
DATASET = ROOT / 'datasets' / 'mmfi' / 'v4'
OUT = ROOT / 'results' / 'layer_cka_controls.json'
HOOKS = ['enc_out', 'layer1_out']
N_PERM = 30
N_SENS = [200, 500, 1000, 1870]


def main() -> None:
    t0 = time.time()
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    val = ds.splits['val']
    n = len(val)
    print(f'[ctrl] prewarming {n} samples...', flush=True)
    for i in range(n):
        _ = val[i]
    print(f'[ctrl] prewarm done in {time.time()-t0:.0f}s', flush=True)

    # ---- trained model features ----
    model = TokenFusionModel.load(str(CKPT))
    print('[ctrl] extracting trained features...', flush=True)
    feats_t = extract_layerwise_features(model, val, device='cpu', batch_size=64,
                                         hook_points=HOOKS)

    # ---- random-weight control (same architecture/config as seed0) ----
    rand_model = TokenFusionModel(num_classes=model.num_classes,
                                  temporal=model.temporal,
                                  structured=model.structured,
                                  domain=model.domain,
                                  domain_dims=model.domain_dims)
    rand_model.eval()
    print('[ctrl] extracting random-model features...', flush=True)
    feats_r = extract_layerwise_features(rand_model, val, device='cpu', batch_size=64,
                                         hook_points=HOOKS)

    rng = np.random.default_rng(0)
    pairs: dict = {}
    print(f'[ctrl] {"pair":<22}{"obs":>7}{"null":>7}{"±":>7}{"z":>7}{"p":>7}{"rand":>7}',
          flush=True)
    for h in HOOKS:
        for ia, ma in enumerate(MODALITIES):
            for mb in MODALITIES[ia + 1:]:
                X = feats_t[h][ma]
                Y = feats_t[h][mb]
                obs = linear_cka(X, Y)
                null_vals = []
                for _ in range(N_PERM):
                    idx = rng.permutation(Y.shape[0])
                    null_vals.append(linear_cka(X, Y[idx]))
                null_vals = np.asarray(null_vals)
                r = linear_cka(feats_r[h][ma], feats_r[h][mb])
                z = (obs - null_vals.mean()) / (null_vals.std() + 1e-12)
                p = float((null_vals >= obs).mean())
                pairs[f'{h}|{ma}__{mb}'] = {
                    'observed': obs,
                    'null_mean': float(null_vals.mean()),
                    'null_std': float(null_vals.std()),
                    'z': float(z),
                    'p_null_ge_obs': p,
                    'random_model': r,
                }
                print(f'[ctrl] {h[:4]}|{ma}__{mb:<10}{obs:7.3f}{null_vals.mean():7.3f}'
                      f'{null_vals.std():7.3f}{z:7.1f}{p:7.2f}{r:7.3f}', flush=True)

    # ---- sample-size sensitivity (headline pair) ----
    sens: dict = {}
    for h in HOOKS:
        X = feats_t[h]['mmwave']
        Y = feats_t[h]['rgb']
        sens[h] = {str(k): linear_cka(X[:k], Y[:k]) for k in N_SENS}
    print(f'[ctrl] N-sensitivity mmwave×rgb: {sens}', flush=True)

    out = {
        'checkpoint': str(CKPT),
        'dataset': str(DATASET),
        'n_val': n,
        'n_perm': N_PERM,
        'n_sensitivity_grid': N_SENS,
        'pairs': pairs,
        'n_sensitivity_mmwave_rgb': sens,
        'elapsed_s': time.time() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f'[ctrl] saved {OUT}  ({time.time()-t0:.0f}s total)', flush=True)


if __name__ == '__main__':
    main()