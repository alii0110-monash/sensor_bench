"""mmwave 几何特征工程 v2: 基于消融实验的发现（dim2/z 最重要，
doppler/intensity 只贡献 ~6%）重设计特征提取。

对比实验：
  A. raw baseline           (T*64*5 = 320 维 flatten mean over time)
  B. v5_structfeat current  (50 维 既有 extract_mmwave_features)
  C. geom_v2_xyz_only       (~33 维 只用 dim0,1,2 几何分布)
  D. geom_v2_xyz_plus_dim34 (~45 维 几何 + doppler/intensity)

数据：v3 base 9689 train + 1968 val（避免 v4 variant 内存膨胀）。
评测：Linear probe, 20 epochs, 3 seeds 取 mean ± std。
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

DIM_NAMES = ["x", "y", "z", "doppler", "intensity"]


# ========== Feature extractors ==========

def feat_raw(X: np.ndarray) -> np.ndarray:
    """A. raw baseline: mean over time + flatten → 64*5=320 dim."""
    return X.mean(axis=1).reshape(X.shape[0], -1).astype(np.float32)


def feat_v5_current(X: np.ndarray) -> np.ndarray:
    """B. v5_structfeat current: extract_mmwave_features on raw mmwave."""
    from framework.eval.dataset_quality.feature_extract import extract_mmwave_features
    out = np.stack([extract_mmwave_features(x) for x in X])
    return out.astype(np.float32)


def _percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else 0.0


def _geom_per_frame(pts: np.ndarray) -> list[float]:
    """Per-frame geometric stats for valid points (T, n, 5).

    Returns: [n, x_mean, x_std, x_range, x_p25, x_p75,
              y_mean, y_std, y_range, y_p25, y_p75,
              z_mean, z_std, z_range, z_p25, z_p75]
    """
    f = []
    if len(pts) == 0:
        return [0.0] * 16
    f.append(float(len(pts)))
    for d in range(3):  # dim0,1,2 (x, y, z)
        col = pts[:, d]
        f.append(float(col.mean()))
        f.append(float(col.std()) if len(col) > 1 else 0.0)
        f.append(float(col.max() - col.min()))
        f.append(_percentile(col, 25))
        f.append(_percentile(col, 75))
    return f


def _z_histogram(pts: np.ndarray, bins: int = 8, lo: float = -3.0, hi: float = 3.0) -> list[float]:
    """8-bin histogram of dim2 (z) values. Captures vertical distribution shape."""
    if len(pts) == 0:
        return [0.0] * bins
    z = pts[:, 2]
    hist, _ = np.histogram(z, bins=bins, range=(lo, hi))
    return (hist / max(hist.sum(), 1)).tolist()


def _xy_extent(pts: np.ndarray) -> list[float]:
    """XY extent (range x * range y), covariance eigenvalues for shape."""
    if len(pts) < 2:
        return [0.0, 0.0, 0.0]
    xy = pts[:, :2]
    xr = xy[:, 0].max() - xy[:, 0].min()
    yr = xy[:, 1].max() - xy[:, 1].min()
    cov = np.cov(xy.T)
    eigs = np.linalg.eigvalsh(cov) if cov.shape == (2, 2) else np.array([0.0, 0.0])
    return [float(xr * yr), float(eigs[0]), float(eigs[1])]


def _signal_per_frame(pts: np.ndarray) -> list[float]:
    """Per-frame stats for dim3 (doppler) and dim4 (intensity)."""
    f = []
    if len(pts) == 0:
        return [0.0] * 8
    for d in (3, 4):
        col = pts[:, d]
        f.append(float(col.mean()))
        f.append(float(np.abs(col).mean()))  # magnitude
        f.append(float(col.std()) if len(col) > 1 else 0.0)
        f.append(float(col.max() - col.min()))
    return f


def feat_geom_xyz_only(X: np.ndarray) -> np.ndarray:
    """C. geom_v2 xyz-only: 几何分布统计, 忽略 doppler/intensity."""
    out = []
    for x in X:  # x: (T, 64, 5)
        T = x.shape[0]
        per_frame = []
        centroids = [[], [], []]
        for t in range(T):
            valid = ~np.all(x[t] == 0, axis=-1)
            pts = x[t][valid]
            per_frame.extend(_geom_per_frame(pts))
            # z histogram + xy extent per frame (avg across frames)
            if t == 0:
                z_hists = []
                xy_exts = []
            z_hists.append(_z_histogram(pts))
            xy_exts.append(_xy_extent(pts))
            if len(pts):
                for d in range(3):
                    centroids[d].append(float(pts[:, d].mean()))
        # aggregate z hist across frames
        z_hist_avg = np.mean(z_hists, axis=0) if z_hists else np.zeros(8)
        # aggregate xy extent across frames
        xy_avg = np.mean(xy_exts, axis=0) if xy_exts else [0, 0, 0]
        # inter-frame centroid drift
        centroid_stds = [float(np.std(c)) if c else 0.0 for c in centroids]
        feat = np.concatenate([
            np.array(per_frame, dtype=np.float32),       # T*16
            z_hist_avg.astype(np.float32),                 # 8
            np.array(xy_avg, dtype=np.float32),            # 3
            np.array(centroid_stds, dtype=np.float32),     # 3
        ])
        out.append(feat)
    return np.stack(out)


def feat_geom_xyz_plus_signal(X: np.ndarray) -> np.ndarray:
    """D. geom_v2 xyz + dim3,4: 几何 + doppler/intensity 弱信号."""
    out = []
    for x in X:
        T = x.shape[0]
        per_frame = []
        signal_per_frame = []
        centroids = [[], [], []]
        z_hists, xy_exts = [], []
        for t in range(T):
            valid = ~np.all(x[t] == 0, axis=-1)
            pts = x[t][valid]
            per_frame.extend(_geom_per_frame(pts))
            signal_per_frame.extend(_signal_per_frame(pts))
            z_hists.append(_z_histogram(pts))
            xy_exts.append(_xy_extent(pts))
            if len(pts):
                for d in range(3):
                    centroids[d].append(float(pts[:, d].mean()))
        z_hist_avg = np.mean(z_hists, axis=0) if z_hists else np.zeros(8)
        xy_avg = np.mean(xy_exts, axis=0) if xy_exts else [0, 0, 0]
        centroid_stds = [float(np.std(c)) if c else 0.0 for c in centroids]
        feat = np.concatenate([
            np.array(per_frame, dtype=np.float32),
            np.array(signal_per_frame, dtype=np.float32),
            z_hist_avg.astype(np.float32),
            np.array(xy_avg, dtype=np.float32),
            np.array(centroid_stds, dtype=np.float32),
        ])
        out.append(feat)
    return np.stack(out)


# ========== Probe + eval (same as probe_mmwave_ablation.py) ==========

def _standardize(Xtr: np.ndarray, Xv: np.ndarray):
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


def _load_split(data_dir: str, split_ids: list[str]):
    Xs, ys = [], []
    for sid in split_ids:
        p = os.path.join(data_dir, f"{sid}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            s = pickle.load(f)
        m = s["modalities"]["mmwave"]["data"]
        if m.shape != (5, 64, 5):
            continue
        Xs.append(m)
        ys.append(s["label"])
    return np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/mmfi/v3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/mmwave_geom_v2_compare.json")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    splits_dir = os.path.join(args.data_root, "splits")
    train_ids = json.load(open(os.path.join(splits_dir, "train.json")))
    val_ids = json.load(open(os.path.join(splits_dir, "val.json")))
    data_dir = os.path.join(args.data_root, "data")

    print(f"[geom-v2] loading train ({len(train_ids)})…", flush=True)
    Xtr_raw, ytr = _load_split(data_dir, train_ids)
    print(f"[geom-v2] loading val ({len(val_ids)})…", flush=True)
    Xv_raw, yv = _load_split(data_dir, val_ids)
    n_classes = int(max(ytr.max(), yv.max())) + 1
    print(f"[geom-v2] n_classes={n_classes}  Xtr={Xtr_raw.shape}  Xv={Xv_raw.shape}",
          flush=True)

    # Extract all 4 feature variants on BOTH splits (training is fair: each
    # variant uses ONLY its own training samples for the model).
    variants = [
        ("A_raw_baseline_320d",         feat_raw),
        ("B_v5_current_50d",            feat_v5_current),
        ("C_geom_v2_xyz_only",          feat_geom_xyz_only),
        ("D_geom_v2_xyz_plus_signal",   feat_geom_xyz_plus_signal),
    ]
    print("[geom-v2] extracting features…", flush=True)
    feats_tr, feats_v = {}, {}
    for name, fn in variants:
        t0 = time.time()
        feats_tr[name] = fn(Xtr_raw)
        feats_v[name] = fn(Xv_raw)
        print(f"  {name:32s}  dim={feats_tr[name].shape[1]}  ({time.time()-t0:.1f}s)",
              flush=True)

    results = {
        "config": {
            "data_root": args.data_root,
            "n_train": len(ytr),
            "n_val": len(yv),
            "n_classes": n_classes,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
        },
        "variants": {},
    }

    for name in feats_tr:
        # standardize with TRAIN stats only (avoid val leak)
        Xtr_s, Xv_s = _standardize(feats_tr[name], feats_v[name])
        accs = []
        for seed in args.seeds:
            model = _train_probe(Xtr_s, ytr, n_classes,
                                 epochs=args.epochs, lr=args.lr,
                                 batch_size=args.batch_size,
                                 seed=seed, device=args.device)
            accs.append(_eval_probe(model, Xv_s, yv, device=args.device))
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        results["variants"][name] = {
            "dim": int(feats_tr[name].shape[1]),
            "val_acc_per_seed": accs,
            "val_acc_mean": mean,
            "val_acc_std": std,
        }
        print(f"  [{name:32s}] dim={feats_tr[name].shape[1]:4d}  "
              f"val_acc={mean:.4f}±{std:.4f}  per_seed={[f'{a:.4f}' for a in accs]}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("[geom-v2] ranking by val_acc_mean (high → low):")
    ranked = sorted(results["variants"].items(),
                    key=lambda kv: -kv[1]["val_acc_mean"])
    for name, info in ranked:
        print(f"  {name:32s} dim={info['dim']:4d}  acc={info['val_acc_mean']:.4f}±{info['val_acc_std']:.4f}")
    print(f"[geom-v2] saved → {args.out}")


if __name__ == "__main__":
    main()