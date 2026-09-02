"""mmwave 5 维消融实验：probe val acc vs 维度组合。

mmwave shape (T, 64, 5)，5 维按 MMFi 约定顺序 (x, y, z, doppler, intensity)。

数据策略：用 v3 数据集（9689 train + 1968 val base，无 variant），
直接 pickle.load 一次性提 mmwave + label 到 numpy，
避免 lazy loader 在循环里反复读盘 + 解码 variant 导致 OOM。

输出 results/mmwave_dim_ablation.json：每种 ablation 的 val acc 与 baseline 差值。
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


def _load_mmwave_and_labels(data_dir: str, split_ids: list[str]):
    """Load mmwave (N, T, 64, 5) + label (N,) from pickle files in data_dir.

    Skips samples whose pickle file is missing (filtered in upstream curation).
    Memory: ~62 MB for 9689 samples.
    """
    Xs, ys, kept = [], [], []
    skipped = 0
    for sid in split_ids:
        p = os.path.join(data_dir, f"{sid}.pkl")
        if not os.path.exists(p):
            skipped += 1
            continue
        with open(p, "rb") as f:
            s = pickle.load(f)
        m = s["modalities"]["mmwave"]["data"]
        if m.shape != (5, 64, 5):
            print(f"  WARN: {sid} mmwave shape {m.shape} — skip", flush=True)
            skipped += 1
            continue
        Xs.append(m)
        ys.append(s["label"])
        kept.append(sid)
        if len(kept) % 1000 == 0:
            print(f"  loaded {len(kept)}/{len(split_ids)}", flush=True)
    if skipped:
        print(f"  [skip {skipped} missing/bad samples]", flush=True)
    return (np.stack(Xs).astype(np.float32),
            np.array(ys, dtype=np.int64),
            kept)


def _feat(X: np.ndarray, keep_dims: list[int]) -> np.ndarray:
    """Mask dims NOT in keep_dims to 0, then mean over time axis → flatten.

    Input: (N, T, 64, 5). Output: (N, 64*5=320) with masked dims = 0.
    """
    arr = X.copy()
    mask = np.zeros(5, dtype=bool)
    for d in keep_dims:
        mask[d] = True
    arr[..., ~mask] = 0.0
    feat = arr.mean(axis=1)        # (N, 64, 5)
    return feat.reshape(arr.shape[0], -1).astype(np.float32)


def _standardize(Xtr: np.ndarray, Xv: np.ndarray):
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    safe = np.where(std < 1e-8, 1.0, std)
    return (Xtr - mean) / safe, (Xv - mean) / safe


def _train_probe(X: np.ndarray, y: np.ndarray, n_classes: int, *, epochs, lr,
                 batch_size, hidden_dim, seed, device):
    torch.manual_seed(seed)
    in_dim = X.shape[1]
    if hidden_dim <= 0:
        model = torch.nn.Linear(in_dim, n_classes)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_classes),
        )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32).to(device)
    yt = torch.as_tensor(y, dtype=torch.long).to(device)
    model.to(device)
    n = Xt.shape[0]
    for ep in range(epochs):
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
        logits = model(Xt[i:i + batch_size])
        pred = logits.argmax(dim=-1)
        correct += (pred == yt[i:i + batch_size]).sum().item()
        total += pred.shape[0]
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/mmfi/v3")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=0,
                    help="0 = Linear probe (default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/mmwave_dim_ablation.json")
    args = ap.parse_args()

    splits_dir = os.path.join(args.data_root, "splits")
    train_ids = json.load(open(os.path.join(splits_dir, "train.json")))
    val_ids = json.load(open(os.path.join(splits_dir, "val.json")))
    data_dir = os.path.join(args.data_root, "data")

    print(f"[mmwave-ablation] data_root={args.data_root}")
    print(f"[mmwave-ablation] loading train ({len(train_ids)})…", flush=True)
    Xtr_raw, ytr, train_kept = _load_mmwave_and_labels(data_dir, train_ids)
    print(f"[mmwave-ablation] loading val ({len(val_ids)})…", flush=True)
    Xv_raw, yv, val_kept = _load_mmwave_and_labels(data_dir, val_ids)
    n_classes = int(max(ytr.max(), yv.max())) + 1
    print(f"[mmwave-ablation] n_classes={n_classes}  Xtr={Xtr_raw.shape}  "
          f"Xv={Xv_raw.shape}  epochs={args.epochs}", flush=True)

    ablations = [
        ("baseline_all_5",          [0, 1, 2, 3, 4]),
        ("drop_dim0_x",             [1, 2, 3, 4]),
        ("drop_dim1_y",             [0, 2, 3, 4]),
        ("drop_dim2_z",             [0, 1, 3, 4]),
        ("drop_dim3_doppler",       [0, 1, 2, 4]),
        ("drop_dim4_intensity",     [0, 1, 2, 3]),
        ("drop_geom_xyz",           [3, 4]),
        ("only_doppler",            [3]),
        ("only_intensity",          [4]),
        ("only_geom_xyz",           [0, 1, 2]),
        ("only_doppler_intensity",  [3, 4]),
        ("only_doppler_geom",       [0, 1, 2, 3]),
        ("drop_doppler_AND_intensity", [0, 1, 2]),
    ]

    results = {
        "config": {
            "data_root": args.data_root,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_classes": n_classes,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "device": args.device,
        },
        "dim_names": DIM_NAMES,
        "ablations": [],
    }

    # Cache features & standardized tensors per ablation.
    cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    for label, keep in ablations:
        t0 = time.time()
        key = tuple(keep)
        if key not in cache:
            Ftr = _feat(Xtr_raw, list(keep))
            Fv = _feat(Xv_raw, list(keep))
            Xtr_std, Xv_std = _standardize(Ftr, Fv)
            cache[key] = (Xtr_std, Xv_std)
        Xtr_s, Xv_s = cache[key]

        model = _train_probe(Xtr_s, ytr, n_classes,
                             epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size,
                             hidden_dim=args.hidden_dim,
                             seed=args.seed, device=args.device)
        acc = _eval_probe(model, Xv_s, yv, device=args.device)
        dt = time.time() - t0
        results["ablations"].append({
            "label": label, "keep_dims": list(keep),
            "kept_names": [DIM_NAMES[d] for d in keep],
            "val_acc": acc, "elapsed_s": round(dt, 1),
        })
        print(f"  [{label:32s}] keep={keep}  val_acc={acc:.4f}  ({dt:.1f}s)",
              flush=True)

    baseline = next(a["val_acc"] for a in results["ablations"]
                    if a["label"] == "baseline_all_5")
    for a in results["ablations"]:
        a["delta_vs_baseline"] = a["val_acc"] - baseline
        a["delta_pct"] = (a["delta_vs_baseline"] / baseline * 100) if baseline > 0 else 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print(f"[mmwave-ablation] baseline_all_5 val_acc = {baseline:.4f}")
    print(f"[mmwave-ablation] ranking by val_acc (low → high):")
    for a in sorted(results["ablations"], key=lambda x: x["val_acc"]):
        s = "+" if a["delta_vs_baseline"] >= 0 else ""
        print(f"  {a['label']:32s} acc={a['val_acc']:.4f}  "
              f"Δ={s}{a['delta_vs_baseline']:+.4f} ({s}{a['delta_pct']:+.1f}%)")
    print(f"[mmwave-ablation] saved → {args.out}")


if __name__ == "__main__":
    main()