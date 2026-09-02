"""DomainEncoder: 领域知识作为评测模型的可插拔能力（而非数据层预处理）。

背景：v5_structfeat 把领域特征（depth/wifi/lidar/mmwave 的 extract_*_features）
预提取后写进数据集文件。这污染了"数据集中立"原则——probe 不再测纯数据质量，
评测模型也被绑死到特定特征维度。

迁移：本 encoder 让 token_fusion 在 forward 时对 raw 数据【现场】调用领域
特征提取（numpy，非可微），再接可学习 MLP → (B, N_TOK, D)。数据集保持 raw
中立；领域知识成为评测模型的一个可插拔先验 encoder。非可微的领域提取部分
如同预训练特征提取器（token 网络只学习 MLP 投影）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .encoders import D, N_TOK, MLPEncoder

# 原始模态 -> 领域特征提取函数（lazy import 避免模型/评测层强耦合）
_DOMAIN_EXTRACTORS = {
    "depth": "extract_depth_features",
    "wifi": "extract_wifi_features",
    "lidar": "extract_lidar_features",
    "mmwave": "extract_mmwave_features",
}


def _get_extractor(modality: str):
    """Lazy import the extractor to avoid importing eval code at model import."""
    from framework.eval.dataset_quality import feature_extract as _fe
    return getattr(_fe, _DOMAIN_EXTRACTORS[modality])


class DomainEncoder(nn.Module):
    """Per-modality encoder: raw data → (non-diff) domain features → MLP → (B,N_TOK,D).

    forward(x): x is a raw batched tensor of shape (B, ...) for this modality.
    Applies the domain feature extractor per sample (numpy), then an MLP projection.

    The MLP input dim is the extracted feature dim, which is deterministic for a
    given dataset (fixed frame count) and passed in via ``feat_dim``.
    """

    def __init__(self, modality: str, feat_dim: int):
        super().__init__()
        self.modality = modality
        self.feat_dim = int(feat_dim)
        self.extractor = _get_extractor(modality)
        self.mlp = MLPEncoder(self.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        B = x_np.shape[0]
        feats = np.stack([self.extractor(x_np[i]) for i in range(B)])  # (B, F)
        return self.mlp(torch.from_numpy(feats).to(x.device))
