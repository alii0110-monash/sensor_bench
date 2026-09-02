"""Route B salvage: motion-difference channels at T=5 (re-ingest is blocked).

B 的重 ingest 部分因原始 82G tar 损坏删除而阻塞；但 DMM/帧差分思想不依赖
长时序 — depth 输入 1ch -> 5ch [d_0, d_1-d_0, d_2-d_1, d_3-d_2, d_4-d_3]，
把手写特征里最有效的信号（motion stats, 0.27）显式喂给 encoder。

Arm: vit_motion_raw (from scratch, same protocol as depth_arms vit_raw 0.078)
Out: results/depth_motion_channels.json
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
from framework.models.depth_vit import D, GRID  # noqa: E402
from scripts.depth_arms_experiment import (  # noqa: E402
    DATASET, TRAIN_SUB, BATCH, EPOCHS, LR, NUM_CLASSES, label_from_id,
    stratified_subset)

OUT = ROOT / 'results' / 'depth_motion_channels.json'


class ViTMotionEncoder(nn.Module):
    """5-channel depth+diffs -> (B,T,16,256) same token contract."""

    def __init__(self, in_ch: int = 5, d: int = 256, n_layers: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, d, kernel_size=16, stride=16)
        self.pos = nn.Parameter(torch.randn(1, GRID * GRID, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=4 * d,
                                           batch_first=True, activation="gelu",
                                           norm_first=True, dropout=0.1)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        xt = x.reshape(B * T, *x.shape[2:])
        p = self.patch_embed(xt).flatten(2).transpose(1, 2)
        p = self.blocks(p + self.pos)
        p = self.norm(p)
        g = p.transpose(1, 2).reshape(-1, D, GRID, GRID)
        g = torch.nn.functional.adaptive_avg_pool2d(g, (4, 4))
        tok = g.flatten(2).transpose(1, 2)
        return tok.view(B, T, 16, -1)


def make_motion(depth: np.ndarray) -> np.ndarray:
    """(T,1,224,224) -> (T,T,224,224): [d_0, d_1-d_0, ..., d_{T-1}-d_{T-2}].
    T=5 → 5 channels (1 raw + 4 diffs)."""
    d = depth[:, 0]  # (T,224,224)
    chans = [d]
    for t in range(1, d.shape[0]):
        chans.append(d[t] - d[t - 1])
    return np.stack(chans, axis=1).astype(np.float32)


def batch_input(batch_samples):
    return np.stack([make_motion(s.modalities['depth'].data)
                     for s in batch_samples])


def main() -> None:
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[motion] device={device}', flush=True)
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    tr_idx = stratified_subset(list(train_all._ids),
                               [label_from_id(s) for s in train_all._ids],
                               TRAIN_SUB, seed=0)
    train_sub = [train_all[i] for i in tr_idx]
    for s in list(train_sub) + list(val):
        _ = s
    print(f'[motion] prewarm done {time.time()-t0:.0f}s', flush=True)

    torch.manual_seed(0)
    enc = ViTMotionEncoder().to(device).train()
    head = nn.Linear(D, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR)
    labels = [s.label for s in train_sub]
    for ep in range(EPOCHS):
        perm = np.random.default_rng(ep).permutation(len(train_sub))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            x = torch.as_tensor(batch_input([train_sub[j] for j in idx]),
                                dtype=torch.float32, device=device)
            y = torch.as_tensor([labels[j] for j in idx], dtype=torch.long,
                                device=device)
            loss = nn.functional.cross_entropy(enc(x).mean(dim=1).mean(dim=1), y)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    ok = tot = 0
    with torch.no_grad():
        for i in range(0, len(val), BATCH):
            batch = val[i:i + BATCH]
            x = torch.as_tensor(batch_input(batch), dtype=torch.float32,
                                device=device)
            y = torch.as_tensor([s.label for s in batch], dtype=torch.long,
                                device=device)
            pred = enc(x).mean(dim=1).mean(dim=1).argmax(-1)
            ok += (pred == y).sum().item(); tot += len(batch)
    acc = ok / tot
    results = {'arm': 'vit_motion_raw', 'val_acc': acc, 'epochs': EPOCHS,
               'elapsed_s': time.time() - t0}
    OUT.write_text(json.dumps(results, indent=2))
    print(f'[motion] vit_motion_raw {acc:.4f} — saved {OUT} '
          f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()