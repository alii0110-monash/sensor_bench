"""Frozen text encoders anchoring the contrastive alignment.

TextEncoder is the interface. HashTextEncoder is a deterministic CPU mock for
unit tests (no model download). CLIPTextEncoder wraps a frozen CLIP text model
(transformers) for the real stage-1 training; it downloads weights once.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn.functional as F


class TextEncoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """texts -> (B, dim) normalized embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class HashTextEncoder(TextEncoder):
    """Deterministic bag-of-words feature-hashing encoder (unit-test mock).
    Not semantically meaningful — only exercises the interface/shapes."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> torch.Tensor:
        import hashlib
        out = torch.zeros(len(texts), self._dim)
        for i, txt in enumerate(texts):
            for tok in txt.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self._dim] += 1.0
        return F.normalize(out, dim=-1)


class CLIPTextEncoder(TextEncoder):
    """Frozen CLIP text encoder (transformers). Model may be a local path or
    a HF repo id. Weights loaded frozen."""

    def __init__(self, model_name: str = "/home/li/datasets/models/clip-vit-base-patch32",
                 device: str = "cpu"):
        from transformers import CLIPTextModel, CLIPTokenizer
        self._device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=77,
                             return_tensors="pt").to(self._device)
        emb = self.model(**enc).pooler_output   # (B, 512) CLIP 标准 pooled (含 projection)
        return F.normalize(emb, dim=-1)
