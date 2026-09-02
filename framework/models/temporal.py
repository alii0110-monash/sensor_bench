"""TemporalAggregator: 模态内时序聚合（保留时间顺序）。

背景：encoder 输出 (B, T, N_TOK, D)，此前 `mean(dim=1)` 把 T 帧坍缩成平均姿态，
丢弃了时间顺序（诊断：打乱帧序 acc 不变，delta=+0.0000）。

本模块对 T 帧做：
1. RoPE 旋转位置编码（默认均匀 0..T-1；传 frame_indices 可用真实时间戳）
2. 模态内时序自注意力（full attention，分类任务看全程）
3. 时间池化 -> (B, N_TOK, D)

下游 token 预算不变。MLPEncoder（1D 特征，无时间轴）不经过本模块。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import D


def _precompute_rope(dim: int, max_pos: int, theta: float = 10000.0,
                     device="cpu", dtype=torch.float32):
    """Precompute RoPE cos/sin: shape (max_pos, dim/2) each. dim should be even."""
    assert dim % 2 == 0, "RoPE requires even dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (max_pos, dim/2)
    return (freqs.cos().to(device=device, dtype=dtype),
            freqs.sin().to(device=device, dtype=dtype))


def _apply_rope(x, cos, sin):
    """x: (..., seq, dim); cos/sin: (seq, dim/2) each (half of the dims).
    Standard RoPE: split dim into two halves, rotate one by the angle.
    """
def _apply_rope(x, cos, sin):
    """x: (..., seq, dim); cos/sin: (..., seq, dim/2) — same leading dims as x
    (already gathered per-position). Rotate two dim-halves by the angle."""
    D = x.shape[-1]
    half = D // 2
    x1 = x[..., :half]          # (..., seq, half)
    x2 = x[..., half:]          # (..., seq, half)
    cos = cos.to(x.device)
    sin = sin.to(x.device)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return torch.cat([rot1, rot2], dim=-1)


class TemporalAggregator(nn.Module):
    """RoPE + 因果时序自注意力 + 时间池化。

    forward(x, frame_indices=None):
        x: (B, T, N_TOK, D)
        frame_indices: optional (B, T) real timestamps for RoPE positions.
    Returns (B, N_TOK, D).

    因果结构（对标 MiniMind-O 的时间推断器）：
    - RoPE 编码相对位置，告诉模型"谁先谁后、相隔多远"。
    - **因果掩码** 强制每帧只能 attend 到自身及之前的帧（frame k 只能看到
      ≤ k），模型被训练成"用过去预测当前"，而不是双向看全程。时间遮蔽增强
      正是利用这条因果链——抹掉中间某帧后，模型被迫从更早的上下文重建它。
    - 时间池化取**最后一帧**（而非 mean）：因果下最后一帧聚合了全部可用的
      过去信息，语义上最接近"截至当前时刻的完整表征"。
    """

    def __init__(self, d: int = D, n_heads: int = 4, n_layers: int = 1,
                 max_pos: int = 512, theta: float = 10000.0):
        super().__init__()
        self.d = d
        self.theta = theta
        self.max_pos = max_pos
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=4 * d, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                 enable_nested_tensor=False)

    def forward(self, x, frame_indices=None):
        B, T, N, D = x.shape
        if T > self.max_pos:
            raise ValueError(f"T={T} > max_pos={self.max_pos}")
        x = x.reshape(B * N, T, D)                     # (BN, T, D)
        # RoPE positions: default 0..T-1; use frame_indices if provided
        if frame_indices is not None:
            pos = frame_indices.long().clamp(0, self.max_pos - 1)  # (B,T)
            pos = pos.unsqueeze(1).expand(B, N, T).reshape(B * N, T)  # (BN,T)
        else:
            pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B * N, T)
        cos_all, sin_all = _precompute_rope(self.d, self.max_pos, self.theta,
                                            device=x.device, dtype=x.dtype)
        cos = cos_all[pos]                             # (BN, T, D)
        sin = sin_all[pos]
        x = _apply_rope(x, cos, sin)
        # Causal mask: frame k attends only to frames 0..k (triu(diagonal=1) = -inf).
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1)
        # Pass src_mask so torch's TransformerEncoderLayer applies the causal
        # constraint; is_causal hint alone requires attn_mask in torch >= 2.9.
        x = self.transformer(x, mask=mask)             # (BN, T, D)
        # Last-frame pooling: causal chain makes the final frame the complete
        # "present" summary of all past frames.
        return x[:, -1].view(B, N, D)
