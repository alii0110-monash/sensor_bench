"""Two-stage stage-1: multimodal token encoder + InfoNCE alignment to a frozen
text encoder. The encoders (framework/models/encoders.py) output per-modality
token sequences (B, N_TOK, D); AlignmentModel aggregates available modalities,
applies modality dropout during training, and projects a pooled vector into the
text-encoder's dimension for InfoNCE."""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import D, WifiEncoder, DepthEncoder, PointEncoder

MODALITIES = ["wifi", "depth", "lidar", "mmwave", "rgb"]
N_TOK = 16


def info_nce_loss(z: torch.Tensor, t: torch.Tensor, temperature: float = 0.07,
                  labels: torch.Tensor | None = None, min_negatives: int = 8) -> torch.Tensor:
    """InfoNCE between sensor-pooled vectors z and text vectors t (both (B, dim)).
    Positives are the same index; negatives are the rest of the batch.
    labels 提供时排除同 label 负样本 (label-aware); 对角线正样本恒保留;
    每行可用负样本不足 min_negatives 时不排除 (保底防梯度消失).
    不传 labels 时与旧版行为完全一致."""
    z = F.normalize(z, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = z @ t.t() / temperature          # (B, B)
    if labels is not None:
        logits = _masked_logits(logits, labels, min_negatives)
    idx = torch.arange(z.shape[0], device=z.device)
    return F.cross_entropy(logits, idx)


def _masked_logits(logits: torch.Tensor, labels: torch.Tensor,
                   min_negatives: int = 8) -> torch.Tensor:
    """把同 label 非对角线对的 logits 置 -inf (排除为负样本).
    对角线正样本保留; 每行可用负样本 < min_negatives 时该行不 mask."""
    B = labels.shape[0]
    same = labels[:, None] == labels[None, :]          # (B,B) 同 label
    same[torch.arange(B), torch.arange(B)] = False     # 对角线(正样本)保留
    n_neg = (~same).sum(dim=1) - 1                     # 每行可用负样本数(减对角线)
    guard = n_neg >= min_negatives                     # 保底
    mask = same & guard[:, None]
    return logits.masked_fill(mask, float("-inf"))


class AlignmentModel(nn.Module):
    """Stage-1: multimodal token encoder + projection head for InfoNCE."""

    def __init__(self, num_modalities: int = 5, text_dim: int = 512,
                 dropout_p: float = 0.25, num_classes: int | None = None):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "wifi": WifiEncoder(), "depth": DepthEncoder(),
            "lidar": PointEncoder(3), "mmwave": PointEncoder(5),
            "rgb": PointEncoder(2)})
        self.text_dim = text_dim
        self.dropout_p = dropout_p
        self.projection_head = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, text_dim))
        self.classification_head = nn.Linear(D, num_classes) if num_classes else None

    def encode_modalities(self, mods: Dict[str, torch.Tensor],
                          avail: Dict[str, bool]) -> torch.Tensor:
        """Stack per-modality token sequences into (B, M, N_TOK, D).
        Missing modalities contribute a zero slot (router removes them downstream)."""
        B = next(iter(mods.values())).shape[0]
        toks = []
        for m in MODALITIES:
            if avail.get(m) and m in mods:
                toks.append(self.encoders[m](mods[m]))       # (B, N_TOK, D)
            else:
                toks.append(torch.zeros(B, N_TOK, D, device=mods[list(mods)[0]].device))
        return torch.stack(toks, dim=1)                       # (B, M, N_TOK, D)

    def pool(self, toks: torch.Tensor) -> torch.Tensor:
        """Pool token sequences to one vector per sample (mean over tokens+mods;
        CLS 为 spec 首选但实现锁定为 mean 作备选)。"""
        return toks.mean(dim=(1, 2))                          # (B, D)

    def forward_loss(self, mods: Dict[str, torch.Tensor],
                     text_emb: torch.Tensor, avail: Dict[str, bool],
                     labels: torch.Tensor | None = None,
                     neg_mine: bool = False) -> tuple:
        """InfoNCE (+ 可选分类辅助 CE). 返回 (info_nce, ce); ce 为 None 当无分类头
        或未传 labels. neg_mine=True 时 info_nce 用 labels 排除同 label 负样本."""
        toks = self.encode_modalities(mods, avail)
        pooled = self.pool(toks)
        z = self.projection_head(pooled)
        info_nce = info_nce_loss(z, text_emb, labels=labels if neg_mine else None)
        ce = None
        if self.classification_head is not None and labels is not None:
            ce = F.cross_entropy(self.classification_head(pooled), labels)
        return info_nce, ce
