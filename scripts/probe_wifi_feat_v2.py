"""wifi 特征工程 v2：对比当前 v5_structfeat 特征 vs 改进版。

当前 extract_wifi_features (129d) 在 probe 上 0.080，比 raw (0.093) 还低。
本脚本设计更丰富的 wifi 特征，重点捕捉：
  - 时间变化（动作导致 CSI 时间波动）
  - 子载波间相关性
  - 天线间差异
  - 频域/时域统计

对比（v3 base 9205/1870，Linear probe 20 epochs × 3 seeds）：
  A. raw baseline
  B. 当前 v5_structfeat 特征 (129d)
  C. wifi_v2 改进特征
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality.feature_extract import extract_wifi_features


def _raw_feat(s):
    return s.modalities["wifi"].data.mean(axis=0).reshape(-1).astype(np.float32)


def _standardize(Xtr, Xv):
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    safe = np.where(std < 1e-8, 1.0, std)
    return (Xtr - mean) / safe, (Xv - mean) / safe


def _train_probe(X, y, n_classes, *, epochs, lr, batch_size, seed, device):
    torch.manual_seed(seed)
    in_dim = X.shape[1]
    model = torch.nn.Linear(in_dim, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32).to(device)
    yt = torch.as_tensor(y, dtype=torch.long).to(device)
    model.to(device)
    n = Xt.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = torch.nn.functional.cross_entropy(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def _eval_probe(model, X, y, device, batch_size=1024) -> float:
    Xt = torch.as_tensor(X, dtype=torch.float32).to(device)
    yt = torch.as_tensor(y, dtype=torch.long).to(device)
    correct = total = 0
    for i in range(0, Xt.shape[0], batch_size):
        pred = model(Xt[i:i + batch_size]).argmax(dim=-1)
        correct += (pred == yt[i:i + batch_size]).sum().item()
        total += pred.shape[0]
    return correct / max(total, 1)


def _safe(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def wifi_v2(wifi: np.ndarray) -> np.ndarray:
    """Improved wifi features. wifi: (T, 3, 114, 10) = frames × antennas ×
    subcarriers × time-samples."""
    T, A, S, F = wifi.shape
    feats = []
    # 1. Per-antenna temporal stats (mean/std over time-samples, per frame)
    for t in range(T):
        frame = wifi[t]  # (A, S, F)
        # mean over subcarriers+time
        ant_mean = frame.mean(axis=(1, 2))  # (A,)
        ant_std = frame.std(axis=(1, 2))    # (A,)
        feats.extend(ant_mean.tolist())
        feats.extend(ant_std.tolist())
        # temporal variance (over time-samples, per antenna)
        temp_var = frame.var(axis=-1).mean(axis=1)  # (A,)
        feats.extend(temp_var.tolist())
    # 2. Subcarrier profile (mean over antennas+time, per frame)
    for t in range(T):
        sc_mean = wifi[t].mean(axis=(0, 2))  # (S,)
        # downsample to 19 bins (114/6)
        sc_bins = sc_mean.reshape(6, 19).mean(axis=0)  # (19,)
        feats.extend(sc_bins.tolist())
    # 3. Cross-antenna correlation (per frame)
    for t in range(T):
        flat = wifi[t].reshape(A, -1)  # (A, S*F)
        for i in range(A):
            for j in range(i + 1, A):
                vi, vj = flat[i], flat[j]
                si, sj = vi.std(), vj.std()
                if si < 1e-9 or sj < 1e-9:
                    feats.append(0.0)
                else:
                    feats.append(float(np.corrcoef(vi, vj)[0, 1]))
    # 4. Inter-frame motion (frame-to-frame abs diff)
    if T > 1:
        diffs = [np.abs(wifi[t + 1] - wifi[t]).mean() for t in range(T - 1)]
        feats.extend([float(np.mean(diffs)), float(np.max(diffs)),
                      float(np.std(diffs))])
    else:
        feats.extend([0.0, 0.0, 0.0])
    # 5. Global stats
    feats.append(float(wifi.mean()))
    feats.append(float(wifi.std()))
    feats.append(float(wifi.max()))
    return _safe(np.asarray(feats, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/mmfi/v3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/wifi_feat_v2_compare.json")
    args = ap.parse_args()

    ds = load_dataset(args.data_root, mode="lazy")
    train = list(ds.splits["train"])
    val = list(ds.splits["val"])
    ytr = np.array([s.label for s in train], dtype=np.int64)
    yv = np.array([s.label for s in val], dtype=np.int64)
    n_classes = int(max(ytr.max(), yv.max())) + 1
    print(f"[wifi-v2] train={len(train)} val={len(val)} n_classes={n_classes}")

    variants = {
        "A_raw": lambda s: _raw_feat(s),
        "B_v5_current": lambda s: extract_wifi_features(s.modalities["wifi"].data),
        "C_wifi_v2": lambda s: wifi_v2(s.modalities["wifi"].data),
    }

    results = {"config": {"data_root": args.data_root, "epochs": args.epochs,
                          "seeds": args.seeds}, "variants": {}}

    for name, fn in variants.items():
        Xtr = np.stack([fn(s) for s in train])
        Xv = np.stack([fn(s) for s in val])
        Xtr_s, Xv_s = _standardize(Xtr, Xv)
        accs = []
        for seed in args.seeds:
            model = _train_probe(Xtr_s, ytr, n_classes, epochs=args.epochs,
                                 lr=args.lr, batch_size=args.batch_size,
                                 seed=seed, device=args.device)
            accs.append(_eval_probe(model, Xv_s, yv, device=args.device))
        mean = float(np.mean(accs))
        results["variants"][name] = {
            "dim": int(Xtr.shape[1]), "val_acc": accs, "val_acc_mean": mean,
        }
        print(f"  [{name}] dim={Xtr.shape[1]:4d} val_acc={mean:.4f} "
              f"per_seed={[f'{a:.4f}' for a in accs]}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[wifi-v2] saved → {args.out}")


if __name__ == "__main__":
    main()