"""Per-modality shared-weight Perceiver projection (spec M5b §79-81).

Compresses each modality's K_max=16 canonical tokens into k latent queries via
cross-attention, then maps to the target LLM's hidden dim. Weights are SHARED
across modalities. k is fixed at training; inference may prefix-truncate to k'.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import D


class PerceiverProjection(nn.Module):
    def __init__(self, in_dim: int = D, out_dim: int = 4096, k: int = 8,
                 n_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.latents = nn.Parameter(torch.randn(k, in_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(in_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim * 2), nn.GELU(),
                                 nn.Linear(in_dim * 2, in_dim))
        self.to_llm = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, K_max, D) -> (B, M*k, out_dim). Missing-modality slots
        (all-zero rows) are masked in attention AND their output rows are
        explicitly zeroed (spec §82: 投影层对缺失模态不产生输出)."""
        B, M, K, D = x.shape
        x = x.reshape(B * M, K, D)                     # per-modality tokens
        missing = (x.abs().sum(dim=-1) == 0)           # (BM, K)
        mask = missing.clone()
        # Fully-masked rows would make MHA softmax(all -inf) -> NaN.
        # Give such rows a visible placeholder token AND un-mask them; the
        # explicit zeroing below then clears the whole modality's output.
        attn_in = x
        row_missing = mask.all(dim=1)                  # (BM,)
        if row_missing.any():
            attn_in = x.clone()
            attn_in[row_missing] = 1.0
            mask[row_missing] = False
        q = self.latents.unsqueeze(0).expand(B * M, -1, -1)  # (BM, k, D)
        attn_out, _ = self.cross_attn(q, attn_in, attn_in,
                                      key_padding_mask=mask)  # (BM, k, D)
        h = self.norm(attn_out + q)
        h = self.mlp(h) + h
        h = self.to_llm(h)                             # (BM, k, out_dim)
        h = h.reshape(B, M * self.k, self.out_dim)
        # zero out rows corresponding to missing modalities
        missing_mod = row_missing.reshape(B, M, 1)     # (B, M, 1)
        mask_out = missing_mod.expand(B, M, self.k).reshape(B, M * self.k, 1)
        h = h * (~mask_out).to(h.dtype)
        return h
