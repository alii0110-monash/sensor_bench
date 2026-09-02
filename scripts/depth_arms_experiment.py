"""Three-arm depth encoder experiment (MVP: model-based foreground handling).

问题: depth 语义丰富但 encoder 从零学不出 (enc probe ≈ 0.07-0.10)。
检验三个假设:
  H1 容量不足: tiny DepthEncoder (2 conv) vs ViTDepthEncoder (4-layer, 196 patches)
  H2 背景拖累: raw vs person-masked (Mask2Former, Group A)
  H3 缺先验:  MAE 自监督预训练 init (Group B) vs 从零

Arms (identical downstream protocol: CE depth-only classification, 3000 stratified
train → val 1870, AdamW lr 1e-3, 30 epochs, batch 64):
  tiny_raw      : DepthEncoder (existing, 2-conv) on raw depth
  vit_raw       : ViTDepthEncoder from scratch on raw depth
  vit_masked    : ViTDepthEncoder from scratch on person-masked depth
  vit_mae_ft    : ViTDepthEncoder + MAE pretrain (75% patch reconstruction on
                  the same 3000 samples' frames, no labels) then same CE finetune
  vit_mae_probe : MAE-pretrained encoder FROZEN + linear probe (pure prior quality)

Run: sbatch jobs/depth_arms.slurm  (GPU required for MAE/ViT speed)
Out: results/depth_arms_v4.json
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
from framework.models.encoders import DepthEncoder  # noqa: E402
from framework.models.depth_vit import (  # noqa: E402
    ViTDepthEncoder, MAEDecoder, mae_loss, D, N_TOK)

DATASET = ROOT / 'datasets' / 'mmfi' / 'v4'
OUT = ROOT / 'results' / 'depth_arms_v4.json'
CKPT_DIR = ROOT / 'results' / 'depth_arms_ckpt'
TRAIN_SUB = 3000
EPOCHS = 30
LR = 1e-3
BATCH = 64
MAE_EPOCHS = 50
MAE_MASK = 0.75
NUM_CLASSES = 27


def label_from_id(sid: str) -> int:
    return int(sid.split('_')[2][1:]) - 1


def stratified_subset(ids: list, labels: list, n_total: int, seed: int = 0):
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
    return sorted(int(i) for i in idx)


def load_mask(sample_id: str, mask_dir: Path) -> np.ndarray:
    f = mask_dir / f'{sample_id}.npz'
    if f.exists():
        return np.load(f)['mask']  # (T,224,224) uint8
    return None


def apply_mask(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """depth (T,1,224,224) × mask (T,224,224) — background → 0m (远背景)."""
    return (depth[:, 0] * mask)[:, None].astype(np.float32)


def train_ce(encoder: nn.Module, samples, device: str, seed: int = 0,
             masked: bool = False, mask_dir: Path = None,
             lr: float = LR, epochs: int = EPOCHS) -> nn.Module:
    """Single-modality depth classification: encoder → mean tokens → linear."""
    torch.manual_seed(seed)
    encoder = encoder.to(device).train()
    head = nn.Linear(D, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()
    labels = [s.label for s in samples]
    for ep in range(epochs):
        perm = np.random.default_rng(seed * 100 + ep).permutation(len(samples))
        for i in range(0, len(perm), BATCH):
            batch = [samples[j] for j in perm[i:i + BATCH]]
            d = np.stack([s.modalities['depth'].data for s in batch])
            if masked:
                ms = [load_mask(s.id, mask_dir) for s in batch]
                d = np.stack([
                    apply_mask(di, mi if mi is not None else np.ones(
                        di.shape[0], dtype=np.uint8))
                    for di, mi in zip(d, ms)])
            x = torch.as_tensor(d, dtype=torch.float32, device=device)
            y = torch.as_tensor([labels[j] for j in perm[i:i + BATCH]],
                                dtype=torch.long, device=device)
            loss = crit(head(encoder(x).mean(dim=1)), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return encoder, head


@torch.no_grad()
def eval_acc(encoder: nn.Module, head: nn.Module, samples, device: str,
             masked: bool = False, mask_dir: Path = None) -> float:
    encoder.eval(); head.eval()
    ok = tot = 0
    for i in range(0, len(samples), BATCH):
        batch = samples[i:i + BATCH]
        d = np.stack([s.modalities['depth'].data for s in batch])
        if masked:
            ms = [load_mask(s.id, mask_dir) for s in batch]
            d = np.stack([
                apply_mask(di, mi if mi is not None else np.ones(
                    di.shape[0], dtype=np.uint8)) for di, mi in zip(d, ms)])
        x = torch.as_tensor(d, dtype=torch.float32, device=device)
        y = torch.as_tensor([s.label for s in batch], dtype=torch.long,
                            device=device)
        pred = head(encoder(x).mean(dim=1)).argmax(-1)
        ok += (pred == y).sum().item(); tot += len(batch)
    return ok / tot


def frozen_probe(encoder: nn.Module, train, val, device: str) -> float:
    """Linear probe on frozen encoder features (mean-pooled 256-d)."""
    def feats(samples):
        fs, ys = [], []
        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(samples), BATCH):
                batch = samples[i:i + BATCH]
                d = np.stack([s.modalities['depth'].data for s in batch])
                x = torch.as_tensor(d, dtype=torch.float32, device=device)
                fs.append(encoder(x).mean(dim=1).cpu().numpy())
                ys.extend(s.label for s in batch)
        return np.concatenate(fs), np.array(ys)
    Xtr, ytr = feats(train)
    Xva, yva = feats(val)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    torch.manual_seed(0)
    head = nn.Linear(D, NUM_CLASSES).to(device)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(100):
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = crit(head(Xtr_t[b]), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xva_t).argmax(-1) == yva_t).float().mean().item()
        best = max(best, acc)
    return best


def mae_pretrain(encoder: nn.Module, samples, device: str) -> nn.Module:
    encoder = encoder.to(device)
    ckpt = CKPT_DIR / 'vit_mae.pt'
    if ckpt.exists():
        print(f'[mae] resume from {ckpt}', flush=True)
        encoder.load_state_dict(torch.load(ckpt, map_location=device))
        return encoder
    decoder = MAEDecoder().to(device)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=3e-4)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(0)
    frames = np.concatenate(
        [s.modalities['depth'].data for s in samples])  # (Nf,1,224,224)
    print(f'[mae] frames={len(frames)}', flush=True)
    for ep in range(MAE_EPOCHS):
        perm = np.random.default_rng(ep).permutation(len(frames))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = frames[perm[i:i + 256]]
            x = torch.as_tensor(b, dtype=torch.float32, device=device)
            loss = mae_loss(encoder, decoder, x, MAE_MASK, rng)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 10 == 0:
            print(f'[mae] ep {ep+1}/{MAE_EPOCHS} loss {tot/len(perm):.4f}',
                  flush=True)
    return encoder


def main() -> None:
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[arms] device={device}', flush=True)
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    tr_idx = stratified_subset(list(train_all._ids),
                               [label_from_id(s) for s in train_all._ids],
                               TRAIN_SUB, seed=0)
    train_sub = [train_all[i] for i in tr_idx]
    print(f'[arms] train {len(train_sub)} val {len(val)} — prewarm...', flush=True)
    for s in list(train_sub) + list(val):
        _ = s
    print(f'[arms] prewarm done {time.time()-t0:.0f}s', flush=True)

    mask_dir = DATASET / 'masks_m2f' / 'train'
    mask_dir_val = DATASET / 'masks_m2f' / 'val'
    mask_dir_val.mkdir(parents=True, exist_ok=True)

    results = {'epochs': EPOCHS, 'train_n': len(train_sub), 'val_n': len(val),
               'arms': {}}

    # ---- H3: MAE pretrain (no labels) — done once, reused by two arms ----
    enc_mae = ViTDepthEncoder()
    mae_pretrain(enc_mae, train_sub, device)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(enc_mae.state_dict(), CKPT_DIR / 'vit_mae.pt')

    # ---- arm: vit_mae_probe (frozen) ----
    acc = frozen_probe(enc_mae.to(device).eval(), train_sub, val, device)
    results['arms']['vit_mae_probe'] = acc
    print(f'[arms] vit_mae_probe {acc:.4f}', flush=True)

    # ---- arm: vit_mae_ft (MAE init + same CE protocol) ----
    enc, head = train_ce(enc_mae, train_sub, device)
    acc = eval_acc(enc, head, val, device)
    results['arms']['vit_mae_ft'] = acc
    print(f'[arms] vit_mae_ft {acc:.4f}', flush=True)

    # ---- arm: vit_raw ----
    enc_vr, head_vr = train_ce(ViTDepthEncoder(), train_sub, device)
    acc = eval_acc(enc_vr, head_vr, val, device)
    results['arms']['vit_raw'] = acc
    print(f'[arms] vit_raw {acc:.4f}', flush=True)

    # ---- arm: vit_masked ----
    enc_vm, head_vm = train_ce(ViTDepthEncoder(), train_sub, device, masked=True,
                               mask_dir=mask_dir)
    acc = eval_acc(enc_vm, head_vm, val, device, masked=True,
                   mask_dir=mask_dir_val)
    results['arms']['vit_masked'] = acc
    print(f'[arms] vit_masked {acc:.4f}', flush=True)

    # ---- arm: tiny_raw (existing 2-conv DepthEncoder, capacity reference) ----
    enc_t, head_t = train_ce(DepthEncoder(), train_sub, device)
    acc = eval_acc(enc_t, head_t, val, device)
    results['arms']['tiny_raw'] = acc
    print(f'[arms] tiny_raw {acc:.4f}', flush=True)

    # ---- arm: tiny_masked ----
    enc_tm, head_tm = train_ce(DepthEncoder(), train_sub, device, masked=True,
                               mask_dir=mask_dir)
    acc = eval_acc(enc_tm, head_tm, val, device, masked=True,
                   mask_dir=mask_dir_val)
    results['arms']['tiny_masked'] = acc
    print(f'[arms] tiny_masked {acc:.4f}', flush=True)

    results['elapsed_s'] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=2))
    print(f'[arms] saved {OUT} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()