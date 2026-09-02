"""Route A (revised): cross-modal distillation rgb-keypoints -> depth encoder.

原方案（depth→关键点回归）需要 rgb→depth 外参投影，但原始 MMFi 标定已随
82G tar 删除，v4 只保留归一化 2D 关键点 → 改为对比蒸馏（无需标定）：
  teacher: rgb keypoints (17,2) -> MLP -> 256
  student: depth (1,224,224)  -> ViTDepthEncoder(MAE init) -> 256
  InfoNCE: positives = same frame; negatives = other frames in batch

Phases:
  1. teacher_ce      : rgb teacher + CE frame-level (sanity, expect ~0.78)
  2. distill         : InfoNCE alignment of student to frozen teacher
  3. distill_probe   : student frozen -> linear probe   (baseline vit_mae_probe 0.095)
  4. distill_ft      : student + CE low-lr, sample-level
                       (baseline vit_mae_ft_lowlr 0.146, handcrafted 0.27)

Run: sbatch jobs/distill_route_a.slurm
Out: results/distill_route_a.json
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
from framework.models.depth_vit import ViTDepthEncoder  # noqa: E402
from scripts.depth_arms_experiment import (  # noqa: E402
    CKPT_DIR, DATASET, TRAIN_SUB, BATCH, NUM_CLASSES, label_from_id,
    stratified_subset, train_ce, eval_acc)

OUT = ROOT / 'results' / 'distill_route_a.json'
EPOCHS = 30
DISTILL_EPOCHS = 30
LR_TEACHER = 1e-3
LR_DISTILL = 1e-3
LR_FT = 1e-4
PROJ_DIM = 128
TEMP = 0.1


class KeypointTeacher(nn.Module):
    """rgb keypoints (17,2)->256 embedding + projection head + CE head."""

    def __init__(self, d: int = 256, proj: int = PROJ_DIM, nc: int = NUM_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(34, d), nn.ReLU(), nn.LayerNorm(d))
        self.proj = nn.Linear(d, proj)
        self.ce_head = nn.Linear(d, nc)

    def embed(self, kps: torch.Tensor) -> torch.Tensor:
        return self.mlp(kps.flatten(1))


def frame_pairs(samples):
    """Flatten samples to per-frame pairs: depth (N,1,224,224), kps (N,17,2), labels."""
    dep, kps, ys = [], [], []
    for s in samples:
        d = s.modalities['depth'].data   # (T,1,224,224)
        k = s.modalities['rgb'].data     # (T,17,2)
        dep.append(d); kps.append(k)
        ys.extend([s.label] * d.shape[0])
    return (np.concatenate(dep).astype(np.float32),
            np.concatenate(kps).astype(np.float32),
            np.array(ys))


def infonce(s_proj: torch.Tensor, t_proj: torch.Tensor, temp: float = TEMP):
    """Symmetric InfoNCE over batch (positives on diagonal)."""
    s = F.normalize(s_proj, dim=-1)
    t = F.normalize(t_proj, dim=-1)
    logits = s @ t.T / temp
    labels = torch.arange(len(s), device=s.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def _feats_depth(encoder, dep: np.ndarray, device):
    outs = []
    encoder.eval()
    for i in range(0, len(dep), BATCH):
        x = torch.as_tensor(dep[i:i + BATCH, None], dtype=torch.float32,
                            device=device)  # (B,1,1,224,224) T=1
        outs.append(encoder(x).mean(dim=1).cpu().numpy())
    return np.concatenate(outs)


def linear_probe(Xtr, ytr, Xva, yva, device, epochs=100) -> float:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor((Xva - mu) / sd, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
    torch.manual_seed(0)
    head = nn.Linear(Xtr.shape[1], NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    best = 0.0
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(head(Xtr_t[b]), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (head(Xva_t).argmax(-1) == yva_t).float().mean().item()
        best = max(best, acc)
    return best


def main() -> None:
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[distill] device={device}', flush=True)
    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    tr_idx = stratified_subset(list(train_all._ids),
                               [label_from_id(s) for s in train_all._ids],
                               TRAIN_SUB, seed=0)
    train_sub = [train_all[i] for i in tr_idx]
    for s in list(train_sub) + list(val):
        _ = s
    dep_tr, kps_tr, yfr_tr = frame_pairs(train_sub)      # frame-level
    dep_va, kps_va, yfr_va = frame_pairs(val)
    y_smp_tr = np.array([s.label for s in train_sub])
    y_smp_va = np.array([s.label for s in val])
    print(f'[distill] frames train={len(dep_tr)} val={len(dep_va)} '
          f'({time.time()-t0:.0f}s)', flush=True)

    results: dict = {'arms': {}}

    # ---- Phase 1: teacher CE sanity ----
    torch.manual_seed(0)
    teacher = KeypointTeacher().to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=LR_TEACHER)
    for ep in range(EPOCHS):
        perm = np.random.default_rng(ep).permutation(len(dep_tr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            k = torch.as_tensor(kps_tr[idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(yfr_tr[idx], dtype=torch.long, device=device)
            loss = F.cross_entropy(teacher.ce_head(teacher.embed(k)), y)
            opt.zero_grad(); loss.backward(); opt.step()
    teacher.eval()
    with torch.no_grad():
        correct = 0
        for j in range(0, len(dep_va), BATCH):
            k = torch.as_tensor(kps_va[j:j + BATCH], dtype=torch.float32,
                                device=device)
            pred = teacher.ce_head(teacher.embed(k)).argmax(-1).cpu().numpy()
            correct += int((pred == yfr_va[j:j + len(pred)]).sum())
    results['arms']['teacher_ce_frame'] = correct / len(dep_va)
    print(f"[distill] teacher_ce_frame {results['arms']['teacher_ce_frame']:.4f}",
          flush=True)

    # student init from MAE checkpoint
    student = ViTDepthEncoder()
    ckpt = CKPT_DIR / 'vit_mae.pt'
    if ckpt.exists():
        student.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f'[distill] student init from {ckpt}', flush=True)
    student = student.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Phase 2: InfoNCE distillation ----
    proj_s = nn.Linear(256, PROJ_DIM).to(device)
    proj_t = nn.Linear(256, PROJ_DIM).to(device)
    # train proj_t on teacher embeddings too (same objective), teacher frozen
    opt = torch.optim.AdamW(list(student.parameters())
                            + list(proj_s.parameters())
                            + list(proj_t.parameters()), lr=LR_DISTILL)
    for ep in range(DISTILL_EPOCHS):
        student.train()
        perm = np.random.default_rng(100 + ep).permutation(len(dep_tr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            x = torch.as_tensor(dep_tr[idx, None], dtype=torch.float32,
                                device=device)
            k = torch.as_tensor(kps_tr[idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                t_emb = teacher.embed(k)
            s_emb = student(x).mean(dim=1)
            loss = infonce(proj_s(s_emb), proj_t(t_emb))
            opt.zero_grad(); loss.backward(); opt.step()
    print(f'[distill] distill done ({time.time()-t0:.0f}s)', flush=True)

    # ---- Phase 3: frozen probe ----
    Ftr = _feats_depth(student, dep_tr, device)
    Fva = _feats_depth(student, dep_va, device)
    results['arms']['distill_probe'] = linear_probe(Ftr, yfr_tr, Fva, yfr_va, device)
    print(f"[distill] distill_probe {results['arms']['distill_probe']:.4f}",
          flush=True)

    # ---- Phase 4: CE low-lr finetune (sample-level, same protocol as depth_arms) ----
    enc_ft, head_ft = train_ce(student, train_sub, device, lr=LR_FT)
    acc = eval_acc(enc_ft, head_ft, val, device)
    results['arms']['distill_ft'] = acc
    print(f"[distill] distill_ft {acc:.4f}", flush=True)

    results['frames'] = {'train': len(dep_tr), 'val': len(dep_va)}
    results['elapsed_s'] = time.time() - t0
    OUT.write_text(json.dumps(results, indent=2))
    print(f'[distill] saved {OUT} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()