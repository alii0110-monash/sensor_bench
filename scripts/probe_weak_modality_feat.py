"""depth/wifi/lidar 特征工程复核：raw baseline vs 当前 v5_structfeat 特征。

对比实验（v3 base 9205 train / 1870 val，Linear probe 20 epochs × 3 seeds）：
  A. raw baseline（mean over time + flatten）
  B. 当前 v5_structfeat 特征（extract_{depth,wifi,lidar}_features）

目的：判断这三个模态是否也像 mmwave 一样"特征工程方向错了"。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality.feature_extract import (
    extract_depth_features, extract_wifi_features, extract_lidar_features,
)


def _raw_feat(s, m):
    d = s.modalities[m].data
    if m == "depth":
        d = d[:, 0]  # (T, H, W)
    return d.mean(axis=0).reshape(-1).astype(np.float32)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/mmfi/v3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/weak_modality_feat_compare.json")
    args = ap.parse_args()

    ds = load_dataset(args.data_root, mode="lazy")
    train = list(ds.splits["train"])
    val = list(ds.splits["val"])
    ytr = np.array([s.label for s in train], dtype=np.int64)
    yv = np.array([s.label for s in val], dtype=np.int64)
    n_classes = int(max(ytr.max(), yv.max())) + 1
    print(f"[weak-feat] train={len(train)} val={len(val)} n_classes={n_classes}")

    extractors = {
        "depth": extract_depth_features,
        "wifi": extract_wifi_features,
        "lidar": extract_lidar_features,
    }

    results = {"config": {"data_root": args.data_root, "epochs": args.epochs,
                          "seeds": args.seeds}, "modalities": {}}

    for m in ["depth", "wifi", "lidar"]:
        # A. raw baseline
        Xtr_raw = np.stack([_raw_feat(s, m) for s in train])
        Xv_raw = np.stack([_raw_feat(s, m) for s in val])
        Xtr_s, Xv_s = _standardize(Xtr_raw, Xv_raw)
        accs_raw = []
        for seed in args.seeds:
            model = _train_probe(Xtr_s, ytr, n_classes, epochs=args.epochs,
                                 lr=args.lr, batch_size=args.batch_size,
                                 seed=seed, device=args.device)
            accs_raw.append(_eval_probe(model, Xv_s, yv, device=args.device))
        raw_mean = float(np.mean(accs_raw))

        # B. current v5_structfeat features
        Xtr_cur = np.stack([extractors[m](s.modalities[m].data) for s in train])
        Xv_cur = np.stack([extractors[m](s.modalities[m].data) for s in val])
        Xtr_cs, Xv_cs = _standardize(Xtr_cur, Xv_cur)
        accs_cur = []
        for seed in args.seeds:
            model = _train_probe(Xtr_cs, ytr, n_classes, epochs=args.epochs,
                                 lr=args.lr, batch_size=args.batch_size,
                                 seed=seed, device=args.device)
            accs_cur.append(_eval_probe(model, Xv_cs, yv, device=args.device))
        cur_mean = float(np.mean(accs_cur))

        results["modalities"][m] = {
            "raw_dim": int(Xtr_raw.shape[1]),
            "cur_dim": int(Xtr_cur.shape[1]),
            "raw_val_acc": accs_raw,
            "raw_val_acc_mean": raw_mean,
            "cur_val_acc": accs_cur,
            "cur_val_acc_mean": cur_mean,
            "delta_cur_vs_raw": cur_mean - raw_mean,
        }
        print(f"  [{m}] raw_dim={Xtr_raw.shape[1]:5d} cur_dim={Xtr_cur.shape[1]:4d} "
              f"raw={raw_mean:.4f} cur={cur_mean:.4f} Δ={cur_mean-raw_mean:+.4f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[weak-feat] saved → {args.out}")


if __name__ == "__main__":
    main()