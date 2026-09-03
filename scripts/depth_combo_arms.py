"""Combination arms: motion channels x MAE init x distillation (orthogonality test).

B' established motion channels as depth's biggest lever (vit_motion_raw 0.474
from scratch). A established distillation (+MAE) stack. Test combinations:

  motion_mae_ft          : MAE init (blocks+pos; patch_embed 2ch re-init) + CE lr 1e-4
  motion_distill_ft      : InfoNCE distill (rgb-keypoint teacher) + CE lr 1e-4
  motion_mae_distill_ft  : MAE init + InfoNCE + CE lr 1e-4

Protocol: same as depth_arms (train 2997 stratified -> val 1870, sample-level eval).
Baselines: motion_scratch 0.474 (lr 1e-3), distill_only 0.223, mae_only 0.146.

Run: sbatch jobs/depth_combo.slurm
Out: results/depth_combo_arms.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framework.dataset.loader import load_dataset  # noqa: E402
from scripts.depth_motion_channels import (  # noqa: E402
    ViTMotionEncoder, make_motion, batch_input)
from scripts.distill_depth_route_a import (  # noqa: E402
    KeypointTeacher, infonce, frame_pairs, linear_probe, _feats_depth,
    LR_DISTILL, LR_FT, PROJ_DIM, TEMP, EPOCHS, DISTILL_EPOCHS)
from scripts.depth_arms_experiment import (  # noqa: E402
    DATASET, TRAIN_SUB, BATCH, NUM_CLASSES, label_from_id, stratified_subset,
    CKPT_DIR)

OUT = ROOT / 'results' / 'depth_combo_arms.json'


def load_mae_init(enc: nn.Module) -> nn.Module:
    """Load MAE weights into motion encoder: blocks/pos/norm transfer,
    patch_embed (1ch->2ch) stays fresh."""
    ckpt = CKPT_DIR / 'vit_mae.pt'
    if not ckpt.exists():
        print('[combo] no MAE ckpt, skip init', flush=True)
        return enc
    state = torch.load(ckpt, map_location='cpu')
    adapted = {k: v for k, v in state.items() if not k.startswith('patch_embed')}
    missing, unexpected = enc.load_state_dict(adapted, strict=False)
    assert all(k.startswith('patch_embed') for k in missing), missing
    nn.init.kaiming_normal_(enc.patch_embed.weight, nonlinearity='relu')
    print(f'[combo] MAE init applied (patch_embed fresh 2ch)', flush=True)
    return enc


def train_ce_motion(encoder, train_sub, val, device, lr: float,
                    epochs: int = EPOCHS, seed: int = 0):
    torch.manual_seed(seed)
    encoder = encoder.to(device).train()
    head = nn.Linear(256, NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=lr)
    labels = [s.label for s in train_sub]
    for ep in range(epochs):
        perm = np.random.default_rng(seed * 100 + ep).permutation(len(train_sub))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            x = torch.as_tensor(batch_input([train_sub[j] for j in idx]),
                                dtype=torch.float32, device=device)
            y = torch.as_tensor([labels[j] for j in idx], dtype=torch.long,
                                device=device)
            loss = F.cross_entropy(encoder(x).mean(dim=1).mean(dim=1), y)
            opt.zero_grad(); loss.backward(); opt.step()
    encoder.eval(); head.eval()
    ok = tot = 0
    with torch.no_grad():
        for i in range(0, len(val), BATCH):
            batch = val[i:i + BATCH]
            x = torch.as_tensor(batch_input(batch), dtype=torch.float32,
                                device=device)
            y = torch.as_tensor([s.label for s in batch], dtype=torch.long,
                                device=device)
            pred = encoder(x).mean(dim=1).mean(dim=1).argmax(-1)
            ok += (pred == y).sum().item(); tot += len(batch)
    return encoder, head, ok / tot


def train_teacher(kps_tr, yfr_tr, device, epochs: int = 10):
    teacher = KeypointTeacher().to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(len(kps_tr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            k = torch.as_tensor(kps_tr[idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(yfr_tr[idx], dtype=torch.long, device=device)
            loss = F.cross_entropy(teacher.ce_head(teacher.embed(k)), y)
            opt.zero_grad(); loss.backward(); opt.step()
    teacher.eval()
    tck = ROOT / 'results' / 'distill_teacher.pt'
    torch.save(teacher.state_dict(), tck)
    print(f'[combo] teacher trained & saved {tck}', flush=True)
    return teacher


def distill_motion(student, teacher, train_sub, device,
                   epochs: int = DISTILL_EPOCHS):
    """Sample-level InfoNCE: student = motion-depth embedding (T mean-pooled),
    teacher = rgb-keypoint embedding (T mean-pooled). Positive = same sample."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    student = student.to(device).train()
    proj_s = nn.Linear(256, PROJ_DIM).to(device)
    proj_t = nn.Linear(256, PROJ_DIM).to(device)
    opt = torch.optim.AdamW(list(student.parameters())
                            + list(proj_s.parameters())
                            + list(proj_t.parameters()), lr=LR_DISTILL)
    n = len(train_sub)
    for ep in range(epochs):
        student.train()
        perm = np.random.default_rng(100 + ep).permutation(n)
        for i in range(0, n, BATCH):
            batch = [train_sub[j] for j in perm[i:i + BATCH]]
            x = torch.as_tensor(batch_input(batch), dtype=torch.float32,
                                device=device)  # (B,T,2,224,224)
            k = torch.as_tensor(np.stack([s.modalities['rgb'].data for s in batch]),
                                dtype=torch.float32, device=device)  # (B,T,17,2)
            B, T = k.shape[:2]
            with torch.no_grad():
                t_emb = teacher.embed(k.reshape(B * T, 17, 2)).reshape(B, T, -1).mean(1)
            s_emb = student(x).mean(dim=1).mean(dim=1)  # (B,256)
            loss = infonce(proj_s(s_emb), proj_t(t_emb))
            opt.zero_grad(); loss.backward(); opt.step()
    return student


def main() -> None:
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[combo] device={device}', flush=True)
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    tr_idx = stratified_subset(list(train_all._ids),
                               [label_from_id(s) for s in train_all._ids],
                               TRAIN_SUB, seed=0)
    train_sub = [train_all[i] for i in tr_idx]
    for s in list(train_sub) + list(val):
        _ = s
    dep_tr, kps_tr, yfr_tr = frame_pairs(train_sub)
    print(f'[combo] prewarm done {time.time()-t0:.0f}s', flush=True)

    teacher = train_teacher(kps_tr, yfr_tr, device)
    results: dict = {'arms': {'motion_scratch_ref': 0.4738}}

    # ---- arm 1: motion + MAE init ----
    enc1 = load_mae_init(ViTMotionEncoder())
    _, _, acc = train_ce_motion(enc1, train_sub, val, device, lr=LR_FT)
    results['arms']['motion_mae_ft'] = acc
    print(f"[combo] motion_mae_ft {acc:.4f}", flush=True)

    # ---- arm 2: motion + distill ----
    enc2 = distill_motion(ViTMotionEncoder(), teacher, train_sub, device)
    _, _, acc = train_ce_motion(enc2, train_sub, val, device, lr=LR_FT)
    results['arms']['motion_distill_ft'] = acc
    print(f"[combo] motion_distill_ft {acc:.4f}", flush=True)

    # ---- arm 3: motion + MAE + distill ----
    enc3 = load_mae_init(ViTMotionEncoder())
    enc3 = distill_motion(enc3, teacher, train_sub, device)
    _, _, acc = train_ce_motion(enc3, train_sub, val, device, lr=LR_FT)
    results['arms']['motion_mae_distill_ft'] = acc
    print(f"[combo] motion_mae_distill_ft {acc:.4f}", flush=True)

    results['elapsed_s'] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=2))
    print(f'[combo] saved {OUT} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()