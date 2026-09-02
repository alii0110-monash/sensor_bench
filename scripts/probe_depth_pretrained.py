#!/usr/bin/env python
"""P0-2 预训练 encoder 验证：resnet50 冻结特征 vs 当前 63d 结构化特征（depth）。

对比 v4 原始 depth 图像经 resnet50 冻结特征后的 probe acc，与当前
v5_structfeat 63d 结构化特征 probe acc（0.23）对比，判断预训练 encoder
是否值得接入主流程。

用法：conda run -n sensorbench python scripts/probe_depth_pretrained.py
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality.modality_probe import (
    train_probe, evaluate_probe, standardize_features)


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """(T,1,224,224) or (T,224,224) -> (T,3,224,224) float32 [0,1]."""
    arr = depth.astype(np.float32)
    if arr.ndim == 4:
        arr = arr[:, 0]
    # normalize per-sample to [0,1]
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-6)
    return np.repeat(arr[:, None], 3, axis=1)  # (T,3,224,224)


@torch.no_grad()
def extract_resnet_features(samples, model, device, batch_frames=32):
    """Extract resnet50 avgpool (2048-d) features per sample (mean over frames)."""
    model.to(device).eval()
    feats = []
    for s in samples:
        depth = s.modalities["depth"].data
        rgb = _depth_to_rgb(depth)  # (T,3,224,224)
        T = rgb.shape[0]
        frame_feats = []
        for i in range(0, T, batch_frames):
            x = torch.from_numpy(rgb[i:i + batch_frames]).to(device)
            # resnet50 expects (B,3,224,224), normalize with ImageNet stats
            x = (x - 0.5) / 0.5
            f = model(x)  # (B,2048) avgpool
            frame_feats.append(f.cpu().numpy())
        feats.append(np.concatenate(frame_feats, axis=0).mean(axis=0))
    return np.stack(feats).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="0=all train base")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, mode="lazy")
    train = [s for s in ds.train if "__aug" not in s.id]
    val = list(ds.val)
    if args.limit:
        train = train[:args.limit]
    print(f"train base {len(train)}, val {len(val)}", flush=True)

    import torchvision
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # keep avgpool (2048-d)
    model.eval()

    print("extracting resnet features (train)...", flush=True)
    X_train = extract_resnet_features(train, model, args.device)
    print("extracting resnet features (val)...", flush=True)
    X_val = extract_resnet_features(val, model, args.device)
    y_train = np.array([s.label for s in train], dtype=np.int64)
    y_val = np.array([s.label for s in val], dtype=np.int64)
    print(f"X_train {X_train.shape}, X_val {X_val.shape}", flush=True)

    stats, Xs_train = standardize_features(X_train)
    _, Xs_val = standardize_features(X_val, stats)

    probe = train_probe(Xs_train, y_train, num_classes=27, epochs=args.epochs,
                        device=args.device, hidden_dim=256)
    acc = evaluate_probe(probe, Xs_val, y_val, device=args.device)
    print(f"resnet50 depth probe val_acc: {acc:.4f}", flush=True)

    # also linear probe (no MLP) for fair comparison with 63d structured
    probe_lin = train_probe(Xs_train, y_train, num_classes=27, epochs=args.epochs,
                            device=args.device, hidden_dim=0)
    acc_lin = evaluate_probe(probe_lin, Xs_val, y_val, device=args.device)
    print(f"resnet50 depth probe val_acc (linear): {acc_lin:.4f}", flush=True)


if __name__ == "__main__":
    main()
