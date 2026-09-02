"""CanonicalToken: 规范空间伪 token 的可移植载体 (spec M6a 组件 1).

data 固定 4096 维 float32, modality-major; 与任何 LLM 无关."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

CANONICAL_DIM = 4096


@dataclass
class CanonicalToken:
    id: str
    label: int
    data: np.ndarray            # (M*k, 4096) float32
    modality_order: List[str]   # 与 data 行对齐
    k: int                      # 每模态 token 数
    meta: Dict = field(default_factory=dict)   # 仅 encoder_version

    def validate(self):
        if self.data.ndim != 2 or self.data.shape[1] != CANONICAL_DIM:
            raise ValueError(f"data must be (M*k, {CANONICAL_DIM}), got {self.data.shape}")
        if self.data.dtype != np.float32:
            raise ValueError(f"data must be float32, got {self.data.dtype}")
        n_rows = len(self.modality_order) * self.k
        if self.data.shape[0] != n_rows:
            raise ValueError(f"rows {self.data.shape[0]} != len(modality)*k {n_rows}")
        if not all(isinstance(m, str) and m for m in self.modality_order):
            raise ValueError("modality_order must be non-empty strings")
        return None
