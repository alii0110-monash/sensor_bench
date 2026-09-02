"""Per-modality projection + modality dropout concat MLP probe.

Fixes the dimension-dominance problem of naive concat: each modality m is
projected to a common `embed_dim` via its own Linear, then concatenated in the
projected space. Modality dropout during training forces the probe to learn
robust multi-modal features rather than relying on whichever modality has the
largest raw feature dimension.
"""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerModConcatMLP(nn.Module):
    """Per-modality Linear+BN → concat → MLP head.

    Args:
        slice_dims: ordered dict {modality: input_dim}. Order determines
            column ordering of the concat input.
        embed_dim: per-modality projection output dim.
        hidden: MLP hidden dim.
        num_classes: classifier output dim.
        dropout_p: per-modality dropout probability (training only).
        use_batchnorm: if True (default), BatchNorm1d normalizes each modality's
            projection output to mean=0 std=1, preventing one modality from
            dominating due to magnitude differences.
    """

    def __init__(self, slice_dims: Dict[str, int],
                 embed_dim: int = 64,
                 hidden: int = 128,
                 num_classes: int = 27,
                 dropout_p: float = 0.2,
                 use_batchnorm: bool = True):
        super().__init__()
        self.slice_dims = dict(slice_dims)
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.projs = nn.ModuleDict({
                m: nn.Sequential(
                    nn.Linear(d, embed_dim),
                    nn.BatchNorm1d(embed_dim),
                ) for m, d in slice_dims.items()
            })
        else:
            self.projs = nn.ModuleDict({
                m: nn.Linear(d, embed_dim) for m, d in slice_dims.items()
            })
        in_dim = embed_dim * len(slice_dims)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor,
                slices: Dict[str, Tuple[int, int]],
                avail: Dict[str, bool] = None) -> torch.Tensor:
        embs = []
        for m, (s, e) in slices.items():
            proj = self.projs[m]
            if avail is None or avail.get(m, True):
                emb = proj(x[:, s:e])
                if self.training and self.dropout_p > 0:
                    keep = (torch.rand((), device=x.device)
                            > self.dropout_p).float()
                    emb = emb * keep / (1.0 - self.dropout_p)
            else:
                emb = torch.zeros(x.shape[0], self.embed_dim,
                                  device=x.device, dtype=x.dtype)
            embs.append(emb)
        return self.head(torch.cat(embs, dim=-1))


def _to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.float()
    return torch.as_tensor(arr, dtype=torch.float32)


# ===========================================================================
# PerModCrossAttnMLP — attention-based probe (replaces concat for structured
# features). Each modality becomes a "token"; tokens attend cross-modality
# before mean-pool + MLP head. Avoids concat's magnitude-mismatch pathologies.
# ===========================================================================

class PerModCrossAttnMLP(nn.Module):
    """Per-modality projection → cross-modal self-attention → mean pool → MLP.

    Args:
        slice_dims: ordered {modality: input_dim}.
        embed_dim: per-modality projection output dim (must be divisible by num_heads).
        num_heads: attention heads.
        hidden: MLP head hidden dim.
        num_classes: classifier output.
        dropout_p: per-modality (token-level) dropout during training.
    """

    def __init__(self, slice_dims: Dict[str, int],
                 embed_dim: int = 64, num_heads: int = 4,
                 hidden: int = 128, num_classes: int = 27,
                 dropout_p: float = 0.2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.slice_dims = dict(slice_dims)
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.projs = nn.ModuleDict({
            m: nn.Linear(d, embed_dim) for m, d in slice_dims.items()
        })
        # Learnable [CLS]-like modality summary token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads,
                                           batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def _embed_tokens(self, x: torch.Tensor,
                      slices: Dict[str, Tuple[int, int]],
                      avail: Dict[str, bool] = None) -> torch.Tensor:
        """Project each modality to embed_dim; stack as (B, M, embed_dim)."""
        toks = []
        for m, (s, e) in slices.items():
            proj = self.projs[m]
            if avail is None or avail.get(m, True):
                emb = proj(x[:, s:e])
                if self.training and self.dropout_p > 0:
                    keep = (torch.rand((), device=x.device)
                            > self.dropout_p).float()
                    emb = emb * keep / (1.0 - self.dropout_p)
            else:
                emb = torch.zeros(x.shape[0], proj.out_features,
                                  device=x.device, dtype=x.dtype)
            toks.append(emb)
        stacked = torch.stack(toks, dim=1)  # (B, M, embed_dim)
        # prepend learnable [CLS] token
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, stacked], dim=1)  # (B, M+1, embed_dim)

    def forward(self, x: torch.Tensor,
                slices: Dict[str, Tuple[int, int]],
                avail: Dict[str, bool] = None) -> torch.Tensor:
        tokens = self._embed_tokens(x, slices, avail)  # (B, M+1, D)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        # Mean pool over M+1 tokens (CLS + modalities)
        pooled = attn_out.mean(dim=1)  # (B, D)
        return self.head(pooled)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor,
                          slices: Dict[str, Tuple[int, int]],
                          avail: Dict[str, bool] = None) -> torch.Tensor:
        """Return (B, M+1, M+1) attention weights (avg over heads)."""
        tokens = self._embed_tokens(x, slices, avail)
        _, weights = self.attn(tokens, tokens, tokens,
                               need_weights=True, average_attn_weights=True)
        return weights


def train_probe_crossattn(X, y, slices: Dict[str, Tuple[int, int]],
                           num_classes: int = 27,
                           embed_dim: int = 64, num_heads: int = 4,
                           hidden: int = 128, epochs: int = 20,
                           lr: float = 1e-3, batch_size: int = 256,
                           device: str = "cpu",
                           dropout_p: float = 0.2,
                           avail: Dict[str, bool] = None) -> PerModCrossAttnMLP:
    """Train a PerModCrossAttnMLP with Adam + cross-entropy."""
    slice_dims = {m: e - s for m, (s, e) in slices.items()}
    model = PerModCrossAttnMLP(slice_dims, embed_dim=embed_dim,
                                 num_heads=num_heads, hidden=hidden,
                                 num_classes=num_classes,
                                 dropout_p=dropout_p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = _to_tensor(X).to(device)
    y_t = _to_tensor(y).long().to(device)
    model.to(device)
    n = X_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_t[idx], slices, avail=avail)
            loss = F.cross_entropy(logits, y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_crossattn(model: PerModCrossAttnMLP, X,
                       slices: Dict[str, Tuple[int, int]],
                       device: str = "cpu",
                       avail: Dict[str, bool] = None,
                       batch_size: int = 1024) -> np.ndarray:
    """Return argmax predictions."""
    model.eval()
    X_t = _to_tensor(X).to(device)
    preds = []
    for i in range(0, X_t.shape[0], batch_size):
        logits = model(X_t[i:i + batch_size], slices, avail=avail)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def train_probe_fusion(X, y, slices: Dict[str, Tuple[int, int]],
                       num_classes: int = 27,
                       embed_dim: int = 64, hidden: int = 128,
                       epochs: int = 20, lr: float = 1e-3,
                       batch_size: int = 256, device: str = "cpu",
                       dropout_p: float = 0.2,
                       avail: Dict[str, bool] = None,
                       class_weighted: bool = False) -> PerModConcatMLP:
    """Train a PerModConcatMLP with Adam + cross-entropy.

    slices must match the column layout of X.
    class_weighted=True applies inverse-frequency class weights to break
    "predict majority class" local minima.
    """
    slice_dims = {m: e - s for m, (s, e) in slices.items()}
    model = PerModConcatMLP(slice_dims, embed_dim=embed_dim, hidden=hidden,
                             num_classes=num_classes, dropout_p=dropout_p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = _to_tensor(X).to(device)
    y_t = _to_tensor(y).long().to(device)
    model.to(device)
    n = X_t.shape[0]
    weight = None
    if class_weighted:
        import numpy as _np
        counts = _np.bincount(_np.asarray(y), minlength=num_classes).astype(_np.float32)
        inv = 1.0 / _np.maximum(counts, 1)
        weight = torch.as_tensor(
            inv * num_classes / inv.sum(), dtype=torch.float32).to(device)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_t[idx], slices, avail=avail)
            loss = F.cross_entropy(logits, y_t[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_fusion(model: PerModConcatMLP, X,
                   slices: Dict[str, Tuple[int, int]],
                   device: str = "cpu",
                   avail: Dict[str, bool] = None,
                   batch_size: int = 1024) -> np.ndarray:
    """Return argmax predictions."""
    model.eval()
    X_t = _to_tensor(X).to(device)
    preds = []
    for i in range(0, X_t.shape[0], batch_size):
        logits = model(X_t[i:i + batch_size], slices, avail=avail)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)