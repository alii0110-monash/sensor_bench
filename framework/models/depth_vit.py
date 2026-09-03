"""ViT-based depth encoder with the same token contract as DepthEncoder.

Contract (framework/models/encoders.py::DepthEncoder):
  forward((B, T, 1, 224, 224)) -> (B, T, 16, D) if temporal else (B, 16, D)

Design:
  - patch 16x16 -> 196 tokens (14x14), dim 256, 4-layer transformer, 4 heads
  - 196 tokens -> adaptive-avg-pool reshape to 4x4 = 16 tokens (grid preserved)
  - class token NOT used (token contract needs spatial tokens for fusion)

MAE support:
  - `features_patches(x)` returns (B*T, 196, D) pre-pool patch features for
    masked reconstruction pretraining
  - MAEDecoder: 2-layer light transformer reconstructing 16x16 depth patches
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn

D = 256
N_TOK = 16
PATCH = 16
GRID = 224 // PATCH  # 14
N_PATCH = GRID * GRID  # 196


class ViTDepthEncoder(nn.Module):
    def __init__(self, d: int = D, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, d, kernel_size=PATCH, stride=PATCH)
        self.pos = nn.Parameter(torch.randn(1, N_PATCH, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d,
                                           batch_first=True, activation="gelu",
                                           norm_first=True, dropout=0.1)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def features_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(..., 1, 224, 224) -> (N, 196, d) patch features (pre-pool).
        Accepts (B,T,1,H,W) or (N,1,H,W); leading dims are flattened."""
        x = x.reshape(-1, *x.shape[-3:])  # (N, 1, H, W)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)  # (N, 196, d)
        p = self.blocks(p + self.pos)
        return self.norm(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        p = self.features_patches(x)                       # (B*T, 196, d)
        n = p.shape[0]
        g = p.transpose(1, 2).reshape(n, D, GRID, GRID)    # (N, d, 14, 14)
        g = torch.nn.functional.adaptive_avg_pool2d(g, (4, 4))  # (N, d, 4, 4)
        tok = g.flatten(2).transpose(1, 2)                 # (N, 16, d)
        out = tok.view(B, T, N_TOK, -1)
        return out if self.temporal else out.mean(dim=1)

    @property
    def temporal(self) -> bool:
        return getattr(self, "_temporal", False)

    @temporal.setter
    def temporal(self, v: bool) -> None:
        self._temporal = v


def make_motion(depth: np.ndarray) -> np.ndarray:
    """(T,1,H,W) -> (T,2,H,W): per-frame [d_t, d_t-d_{t-1}] (Δ_0 = 0).

    Frame-difference channel (DMM-style): depth's action signal lives in
    cross-frame motion — explicit differencing is depth's biggest single
    lever (0.078 -> 0.474 from scratch, see depth_revival_ab_v4.md).
    """
    import numpy as np
    d = depth[:, 0]
    diff = np.zeros_like(d)
    diff[1:] = d[1:] - d[:-1]
    return np.stack([d, diff], axis=1).astype(np.float32)


class ViTMotionEncoder(nn.Module):
    """2-channel [d_t, Δ_t] ViT depth encoder, DepthEncoder-style contract.

    forward((B,T,2,224,224)) -> (B,T,16,D) if temporal else (B,16,D).
    Drop-in replacement for DepthEncoder when motion_depth=True.
    """

    def __init__(self, in_ch: int = 2, d: int = D, n_layers: int = 4,
                 n_heads: int = 4, temporal: bool = False):
        super().__init__()
        self.temporal = temporal
        self.patch_embed = nn.Conv2d(in_ch, d, kernel_size=PATCH, stride=PATCH)
        self.pos = nn.Parameter(torch.randn(1, N_PATCH, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d,
                                           batch_first=True, activation="gelu",
                                           norm_first=True, dropout=0.1)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        xt = x.reshape(B * T, *x.shape[2:])
        p = self.patch_embed(xt).flatten(2).transpose(1, 2)
        p = self.blocks(p + self.pos)
        p = self.norm(p)
        g = p.transpose(1, 2).reshape(-1, D, GRID, GRID)
        g = torch.nn.functional.adaptive_avg_pool2d(g, (4, 4))
        tok = g.flatten(2).transpose(1, 2)               # (BT, 16, D)
        out = tok.view(B, T, N_TOK, -1)
        return out if self.temporal else out.mean(dim=1)


class MAEDecoder(nn.Module):
    """Light decoder: reconstruct normalized depth patches from encoder patches.

    Masked positions are replaced by a learned mask token; shared positional
    embeddings give every position a distinct identity (без pos emb a zeroed
    token yields identical predictions everywhere — unlearnable).
    """

    def __init__(self, d: int = D, patch: int = PATCH, n_layers: int = 2):
        super().__init__()
        self.patch = patch
        self.out_dim = patch * patch
        self.pos = nn.Parameter(torch.randn(1, N_PATCH, d) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, dim_feedforward=d * 2,
                                           batch_first=True, activation="gelu",
                                           norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.head = nn.Linear(d, self.out_dim)

    def forward(self, tokens: torch.Tensor,
                keep_mask: torch.Tensor = None) -> torch.Tensor:
        """tokens (N, 196, d): full features; keep_mask (N, 196) bool.
        Masked positions → mask_token. Returns (N, 196, patch²) predictions."""
        if keep_mask is not None:
            tokens = torch.where(keep_mask.unsqueeze(-1), tokens,
                                 self.mask_token.expand_as(tokens))
        return self.head(self.blocks(tokens + self.pos))


def mae_loss(encoder: ViTDepthEncoder, decoder: MAEDecoder,
             depth: torch.Tensor, mask_ratio: float = 0.75,
             rng: torch.Generator = None) -> torch.Tensor:
    """depth: (B*T, 1, 224, 224). Returns MSE loss on masked patches.

    Per-sample patch normalization (MAE convention): each patch normalized by
    its own mean/std before reconstruction, so the task is shape/structure
    reconstruction rather than absolute depth regression.
    """
    BT = depth.shape[0]
    patches = depth.reshape(BT, 1, GRID, PATCH, GRID, PATCH)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(BT, N_PATCH, -1)
    mean = patches.mean(-1, keepdim=True)
    std = patches.std(-1, keepdim=True) + 1e-6
    targets = (patches - mean) / std  # (BT, 196, 256)

    full = encoder.features_patches(depth)  # (BT, 196, d) — no masking inside
    # random mask per sample
    noise = torch.rand(BT, N_PATCH, generator=rng).to(depth.device)
    keep_mask = noise > mask_ratio  # bool, True = keep
    # decoder sees mask_token at masked positions + positional embeddings
    pred = decoder(full, keep_mask)  # (BT, 196, 256)
    loss = (pred - targets).pow(2).mean(-1)  # (BT, 196)
    masked = ~keep_mask
    return loss[masked].mean()