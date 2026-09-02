"""TokenToLLM: 规范空间 → 目标 LLM 空间投影 (spec M6a 组件 4).

project(CanonicalToken) 是唯一 LLM 相关层; inject 复用 LLMAdapter.
每 LLM 一个 Linear(4096 -> llm_hidden)."""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .canonical import CanonicalToken


class TokenToLLM(ABC):
    @property
    @abstractmethod
    def llm_hidden(self) -> int:
        ...

    @abstractmethod
    def project(self, canonical: CanonicalToken) -> torch.Tensor:
        """(M*k, 4096) -> (1, n, llm_hidden) 伪 token."""


class LinearTokenToLLM(TokenToLLM):
    """轻量线性投影: Linear(4096 -> llm_hidden). 换 LLM 只换这一层."""

    def __init__(self, llm_hidden: int, device: str = "cpu"):
        self._hidden = llm_hidden
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.linear = nn.Linear(4096, llm_hidden).to(self.device)
        self.linear.eval()

    @property
    def llm_hidden(self) -> int:
        return self._hidden

    @torch.no_grad()
    def project(self, canonical: CanonicalToken) -> torch.Tensor:
        canonical.validate()
        x = torch.from_numpy(canonical.data)[None].to(self.device)   # (1, M*k, 4096)
        return self.linear(x)                                        # (1, M*k, llm_hidden)
