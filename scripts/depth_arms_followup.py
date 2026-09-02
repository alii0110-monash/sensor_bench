"""Follow-up arms for depth_arms (protocol-fairness + undertraining check).

- vit_mae_ft_lowlr : MAE init + finetune at lr 1e-4 (1e-3 nuked pretrained weights)
- vit_raw_long    : from scratch, 150 epochs (30 may be undertrained for ViT)

Run: sbatch jobs/depth_arms_followup.slurm (GPU)
Out: results/depth_arms_followup.json
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
from framework.models.depth_vit import ViTDepthEncoder  # noqa: E402
from scripts.depth_arms_experiment import (  # noqa: E402
    CKPT_DIR, DATASET, TRAIN_SUB, label_from_id, stratified_subset, train_ce,
    eval_acc, mae_pretrain)

OUT = ROOT / 'results' / 'depth_arms_followup.json'


def main() -> None:
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[fu] device={device}', flush=True)
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    tr_idx = stratified_subset(list(train_all._ids),
                               [label_from_id(s) for s in train_all._ids],
                               TRAIN_SUB, seed=0)
    train_sub = [train_all[i] for i in tr_idx]
    for s in list(train_sub) + list(val):
        _ = s
    print(f'[fu] prewarm done {time.time()-t0:.0f}s', flush=True)

    results: dict = {'arms': {}}

    # ---- arm: vit_mae_ft_lowlr (MAE init, lr 1e-4) ----
    enc = ViTDepthEncoder()
    enc = mae_pretrain(enc, train_sub, device)  # resumes from checkpoint
    enc_ft, head_ft = train_ce(enc, train_sub, device, lr=1e-4)
    acc = eval_acc(enc_ft, head_ft, val, device)
    results['arms']['vit_mae_ft_lowlr'] = acc
    print(f'[fu] vit_mae_ft_lowlr {acc:.4f}', flush=True)

    # ---- arm: vit_raw_long (150 epochs from scratch) ----
    enc_l, head_l = train_ce(ViTDepthEncoder(), train_sub, device, epochs=150)
    acc = eval_acc(enc_l, head_l, val, device)
    results['arms']['vit_raw_long'] = acc
    print(f'[fu] vit_raw_long {acc:.4f}', flush=True)

    results['elapsed_s'] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=2))
    print(f'[fu] saved {OUT} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()