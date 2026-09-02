"""Per-layer per-modality linear probe (MVP iteration #2).

区分"语义对齐" vs "平凡塌缩"（layer_cka_v4.md §9.3 判据）:
  - 深层 CKA 高 + 各模态 probe acc 高且接近 → 真对齐
  - 深层 CKA 高 + probe acc 低             → 塌缩（平凡共享成分）
  - mmwave acc 高 + CKA 低                 → 独立 oracle（判别信息在自己的几何里）

Protocol:
  - features: extract_layerwise_features (同 layer_cka.py), 3 seeds
  - probe train: v4 train 分层子集 3000; eval: val 1870 (与 dataset_quality P0 护栏一致)
  - probe: z-score(train统计) → Linear(256→27), AdamW lr 1e-3, 30 epochs, batch 256
  - 附加: label-CKA = linear_cka(feature, one-hot label) — 类结构含量

Run: sbatch jobs/layer_probe.slurm   (~15-20 min, normal_test)
Out: results/layer_probe_v4.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framework.dataset.loader import load_dataset  # noqa: E402
from framework.models.token_fusion import TokenFusionModel, MODALITIES  # noqa: E402
from framework.eval.dataset_quality.layer_cka import (  # noqa: E402
    extract_layerwise_features, linear_cka)

CKPT_DIR = ROOT / 'checkpoints_v4_temporal'
DATASET = ROOT / 'datasets' / 'mmfi' / 'v4'
OUT = ROOT / 'results' / 'layer_probe_v4.json'
HOOKS = ['enc_out', 'layer1_out']
SEEDS = [0, 1, 2]
N_TRAIN_SUB = 3000
PROBE_EPOCHS = 30
PROBE_LR = 1e-3
BATCH = 256
NUM_CLASSES = 27


def stratified_subset(labels: list, n_total: int, seed: int = 0) -> np.ndarray:
    """Per-class round-robin subset of indices, ~n_total samples."""
    rng = np.random.default_rng(seed)
    by_cls: dict = {}
    for i, l in enumerate(labels):
        by_cls.setdefault(l, []).append(i)
    for l in by_cls:
        rng.shuffle(by_cls[l])
    per_cls = max(1, n_total // len(by_cls))
    idx = []
    for l in sorted(by_cls):
        idx.extend(by_cls[l][:per_cls])
    return np.array(sorted(idx))


def label_from_id(sid: str) -> int:
    """v4/MMFi id format E.._S.._A{action}_f.. → label = action-1 (实测验证)."""
    a = sid.split('_')[2]  # 'A01'
    return int(a[1:]) - 1


def extract(samples, device: str = 'cpu') -> dict:
    """{hook: {mod: (N,256)}} for one split."""
    out = {}
    for seed in SEEDS:
        ckpt = CKPT_DIR / f'token_fusion_seed{seed}.pt'
        model = TokenFusionModel.load(str(ckpt))
        print(f'[probe] extracting seed{seed} n={len(samples)}...', flush=True)
        feats = extract_layerwise_features(model, samples, device=device,
                                           batch_size=64, hook_points=HOOKS)
        for h in HOOKS:
            out.setdefault(h, {}).setdefault('by_seed', {})
            out[h]['by_seed'][seed] = feats[h]
    return out


def probe_acc(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
              seed: int = 0) -> float:
    """Linear probe on z-scored features; returns best val acc."""
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xva_t = torch.as_tensor((Xva - mu) / sd, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long)
    yva_t = torch.as_tensor(yva, dtype=torch.long)

    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], NUM_CLASSES)
    opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    n = len(Xtr_t)
    for ep in range(PROBE_EPOCHS):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            loss = crit(head(Xtr_t[b]), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xva_t).argmax(-1) == yva_t).float().mean().item()
        best = max(best, acc)
    return best


def main() -> None:
    t0 = time.time()
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    print(f'[probe] train={len(train_all)} val={len(val)}', flush=True)

    # label 从 id 解析（不触发样本加载）；用 LazySplit._ids 保证与位置对齐
    tr_ids = list(train_all._ids)
    tr_labels = [label_from_id(s) for s in tr_ids]
    tr_idx = stratified_subset(tr_labels, N_TRAIN_SUB, seed=0)
    train_sub = [train_all[int(i)] for i in tr_idx]
    print(f'[probe] train subset: {len(train_sub)} (stratified) — loading...',
          flush=True)
    for s in list(train_sub) + list(val):
        _ = s  # touch → cache
    print(f'[probe] prewarm done {time.time()-t0:.0f}s', flush=True)

    ytr = np.array([s.label for s in train_sub])
    yva = np.array([s.label for s in val])

    tr_feats = extract(train_sub)
    va_feats = extract(list(val))
    print(f'[probe] extraction done {time.time()-t0:.0f}s', flush=True)

    onehot_tr = np.eye(NUM_CLASSES, dtype=np.float32)[ytr]

    results = {'n_train': len(train_sub), 'n_val': len(val), 'seeds': SEEDS,
               'probe': {}, 'label_cka': {}}
    print(f'[probe] {"hook/mod":<18}' + ''.join(f's{s:<7}' for s in SEEDS) + 'mean',
          flush=True)
    for h in HOOKS:
        for m in MODALITIES:
            accs = []
            for seed in SEEDS:
                Xtr = tr_feats[h]['by_seed'][seed][m]
                Xva = va_feats[h]['by_seed'][seed][m]
                accs.append(probe_acc(Xtr, ytr, Xva, yva, seed=seed))
                results['probe'].setdefault(h, {}).setdefault(m, {})[str(seed)] = accs[-1]
            mean = float(np.mean(accs))
            results['probe'][h][m]['mean'] = mean
            results['probe'][h][m]['std'] = float(np.std(accs))
            print(f'[probe] {h[:4]}/{m:<12}' + ''.join(f'{a:<8.3f}' for a in accs)
                  + f'{mean:.3f}', flush=True)

    # label-CKA on seed0 features (class-structure content)
    for h in HOOKS:
        for m in MODALITIES:
            X = tr_feats[h]['by_seed'][0][m]
            results['label_cka'].setdefault(h, {})[m] = linear_cka(X, onehot_tr)
        print(f'[probe] label-CKA {h}: '
              + ' '.join(f'{m}={results["label_cka"][h][m]:.3f}'
                         for m in MODALITIES), flush=True)

    results['elapsed_s'] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=2))
    print(f'[probe] saved {OUT}  ({time.time()-t0:.0f}s total)', flush=True)


if __name__ == '__main__':
    main()