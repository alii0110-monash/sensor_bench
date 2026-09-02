"""Per-modality Linear + Concat-Linear probes for dataset intrinsic quality.

The probe trains on raw modality data (no pretrained encoder) so the eval
measures dataset properties, not model quality. Feature extraction: mean over
time/frame axis (axis 0), keeping all other dims as a flat vector.
"""
from __future__ import annotations
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Modality order is fixed (matches framework.tokens.tokenizer.MODALITY_ORDER).
MODALITY_ORDER = ["rgb", "depth", "lidar", "mmwave", "wifi"]


def extract_modality_feature(sample, modality: str) -> np.ndarray:
    """Return a 1-D float feature vector for one modality.

    Convention: mean over axis 0 (time/frames), flatten the rest.
    Channel dim is preserved (last dim kept, not reduced).
    """
    data = sample.modalities[modality].data
    if data.ndim == 1:
        return data.astype(np.float32)
    feat = data.mean(axis=0)
    return feat.reshape(-1).astype(np.float32)


def extract_concat_feature(sample, modalities: Sequence[str]) -> np.ndarray:
    """Concatenate per-modality features into one flat vector."""
    feats = [extract_modality_feature(sample, m) for m in modalities
             if m in sample.modalities]
    return np.concatenate(feats).astype(np.float32)


# --- MLP probe upgrade: standardization + depth downsampling ---

def downsample_depth(data: np.ndarray, pool: int = 8) -> np.ndarray:
    """Max-pool depth over spatial dims to reduce feature dimension.

    data: (T, 224, 224) or (T, 1, 224, 224). pool=8 → 28x28.
    Returns shape (T, H/pool, W/pool).
    """
    arr = data.astype(np.float32)
    if arr.ndim == 4:  # (T, 1, H, W)
        arr = arr[:, 0]
    if pool <= 1:
        return arr
    t = arr.shape[0]
    h, w = arr.shape[1], arr.shape[2]
    nh, nw = h // pool, w // pool
    arr = arr[:, :nh * pool, :nw * pool]
    arr = arr.reshape(t, nh, pool, nw, pool)
    return arr.max(axis=(2, 4))  # (T, nh, nw)


def extract_modality_feature_downsampled(sample, modality: str,
                                         pool: int = 8) -> np.ndarray:
    """Feature extraction with per-modality dispatch.

    - depth: max-pool over spatial dims then mean over time (224×224→28×28).
    - mmwave: 134-d Cartesian-geometry features via `extract_mmwave_features`
      (geom_v2 design, probe val_acc 0.71 — replaces the raw mean-over-time
      path which gave only 0.38 because raw mmwave is 22% non-zero and
      mean-fills the rest).
    - other modalities: mean over time axis + flatten.

    The `pool` argument is depth-only; ignored for other modalities.
    """
    data = sample.modalities[modality].data
    if modality == "depth" and data.ndim >= 3:
        down = downsample_depth(data, pool=pool)
        return down.mean(axis=0).reshape(-1).astype(np.float32)
    if modality == "mmwave":
        # Lazy import to avoid top-level coupling between probe and feature_extract.
        from framework.eval.dataset_quality.feature_extract import extract_mmwave_features
        # v5_structfeat datasets store mmwave as 1D pre-extracted features;
        # only call extract_mmwave_features when the raw 3D point cloud
        # (T, N, 5) is available (e.g. v3 / v4 datasets).
        if data.ndim == 3:
            return extract_mmwave_features(data)
        return data.astype(np.float32)
    return extract_modality_feature(sample, modality)


def standardize_features(X: np.ndarray, stats: Dict = None):
    """Z-score standardize (feature-wise). Returns (stats, X_standardized).

    If stats is None, compute from X (train). stats = {"mean", "std"}.
    Zero-std guarded (leave as 0).
    """
    X = X.astype(np.float32)
    if stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        stats = {"mean": mean, "std": std}
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    safe_std = np.where(std < 1e-8, 1.0, std)
    Xs = (X - mean) / safe_std
    return stats, Xs


def stack_split(samples, modalities: Sequence[str], concat: bool = False):
    """Build (X, y) tensors from a sample list.

    Returns:
        X_dict: {modality: np.ndarray (N, dim_m)} if not concat
                {"concat": np.ndarray (N, sum_dims)} if concat
        y: np.ndarray (N,) int64
    """
    y = np.array([s.label for s in samples], dtype=np.int64)
    if concat:
        X = np.stack([extract_concat_feature(s, modalities) for s in samples])
        return {"concat": X}, y
    X_dict = {}
    for m in modalities:
        feats = [extract_modality_feature(s, m) for s in samples
                 if m in s.modalities]
        X_dict[m] = np.stack(feats) if feats else np.zeros((0, 1), np.float32)
    return X_dict, y


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32)


def _build_probe(in_dim: int, num_classes: int, hidden_dim: int = 0):
    """Return Linear (hidden_dim=0) or 2-layer MLP (hidden_dim>0)."""
    if hidden_dim <= 0:
        return torch.nn.Linear(in_dim, num_classes)
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, num_classes),
    )


def train_probe(X: np.ndarray, y: np.ndarray, num_classes: int,
                epochs: int = 20, lr: float = 1e-3,
                batch_size: int = 256, device: str = "cpu",
                hidden_dim: int = 0,
                class_weighted: bool = False) -> torch.nn.Module:
    """Train a Linear (hidden_dim=0) or small MLP with Adam + CE.

    If class_weighted=True, applies inverse-frequency class weights to break
    the "predict majority class" local minimum.
    """
    in_dim = X.shape[1]
    model = _build_probe(in_dim, num_classes, hidden_dim)
    model.out_features = num_classes
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = _to_tensor(X).to(device)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    model.to(device)
    n = X_t.shape[0]
    weight = None
    if class_weighted:
        counts = np.bincount(y, minlength=num_classes).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1)
        # Normalize so mean weight = 1 (avoid overall loss scale change)
        weight = torch.as_tensor(inv * num_classes / inv.sum(), dtype=torch.float32).to(device)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_t[idx])
            loss = F.cross_entropy(logits, y_t[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def train_probe_mlp(X: np.ndarray, y: np.ndarray, num_classes: int,
                    epochs: int = 20, lr: float = 1e-3,
                    batch_size: int = 256, device: str = "cpu",
                    hidden_dim: int = 256) -> torch.nn.Module:
    """Convienience wrapper: MLP probe (hidden_dim=256 default)."""
    return train_probe(X, y, num_classes=num_classes, epochs=epochs,
                       lr=lr, batch_size=batch_size, device=device,
                       hidden_dim=hidden_dim)


@torch.no_grad()
def evaluate_probe(model: torch.nn.Linear, X: np.ndarray, y: np.ndarray,
                   device: str = "cpu", batch_size: int = 1024) -> float:
    """Return top-1 accuracy."""
    model.eval()
    X_t = _to_tensor(X).to(device)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    correct, total = 0, 0
    for i in range(0, X_t.shape[0], batch_size):
        logits = model(X_t[i:i + batch_size])
        pred = logits.argmax(dim=-1)
        correct += (pred == y_t[i:i + batch_size]).sum().item()
        total += pred.shape[0]
    return correct / max(total, 1)


def compute_info_score(acc_per_modality: Dict[str, float],
                       acc_concat: float,
                       w_per_modality: float = 0.7,
                       w_complement: float = 0.3) -> Dict[str, float]:
    """Bounded InfoScore per spec.

    InfoScore = w_per_modality * mean(acc_per_modality)
              + w_complement * clamp(complement_gain, 0, 1 - mean(acc_per_modality))
    """
    if not acc_per_modality:
        return {"mean_acc": 0.0, "complement_gain": 0.0, "InfoScore": 0.0}
    mean_acc = sum(acc_per_modality.values()) / len(acc_per_modality)
    best_single = max(acc_per_modality.values())
    complement_gain = acc_concat - best_single
    clipped_gain = max(0.0, min(complement_gain, max(0.0, 1.0 - mean_acc)))
    info = w_per_modality * mean_acc + w_complement * clipped_gain
    return {"mean_acc": mean_acc,
            "complement_gain": complement_gain,
            "InfoScore": float(min(1.0, max(0.0, info)))}