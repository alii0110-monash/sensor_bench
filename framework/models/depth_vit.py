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