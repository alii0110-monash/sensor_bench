"""LLMAdapter abstraction + LlamaAdapter (local llama2-7b, spec M5b §84-85).

project(): canonical tokens -> pseudo tokens in target LLM space.
inject():   pseudo tokens as prefix merged with text token embeddings.
The target LLM stays frozen; only the projection layer is trained.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .perceiver import PerceiverProjection


class LLMAdapter(ABC):
    """Per-target-LLM adapter. Each LLM family implements project + inject."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        ...

    @abstractmethod
    def project(self, canonical_tokens: torch.Tensor) -> torch.Tensor:
        """(B, M, K_max, D) -> (B, M*k, hidden_dim) pseudo tokens."""

    @abstractmethod
    def inject(self, prefix_embs: torch.Tensor, input_ids: torch.Tensor,
               embed_fn) -> torch.Tensor:
        """Merge pseudo-token prefix embeddings with text embeddings."""


class LlamaAdapter(LLMAdapter):
    """Local llama2-7b adapter. LLM loaded lazily on first use."""

    def __init__(self, model_path: str = "/home/li/datasets/models/llama2-7b",
                 k: int = 8, device: str = "cuda", dtype=torch.bfloat16):
        self.model_path = model_path
        self.k = k
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self.projection = PerceiverProjection(out_dim=self._probe_hidden(), k=k)
        if device != "cpu":
            self.projection = self.projection.to(device)

    def _probe_hidden(self) -> int:
        """Hidden dim without loading weights: read from config.json."""
        import json
        cfg = json.load(open(f"{self.model_path}/config.json"))
        return int(cfg["hidden_size"])

    @property
    def hidden_dim(self) -> int:
        return self._probe_hidden()

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            kwargs = {"torch_dtype": self.dtype}
            if self.device == "cuda":
                # 模型全部装 CPU (避免 16GB 显存满载/冻 WSL).
                # GPU 仅留 1.5 GiB 余量给其他小张量; 调用者按需把 input/output embedding 移到 GPU.
                kwargs["device_map"] = "cpu"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **kwargs).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model, self._tokenizer

    def project(self, canonical_tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(canonical_tokens)

    def inject(self, prefix_embs: torch.Tensor, input_ids: torch.Tensor,
               embed_fn=None) -> torch.Tensor:
        model, _ = self._load()
        if embed_fn is None:
            embed_fn = lambda ids: model.get_input_embeddings()(ids.to(model.device))
        text_embs = embed_fn(input_ids)
        prefix_embs = prefix_embs.to(text_embs.dtype).to(text_embs.device)
        return torch.cat([prefix_embs, text_embs], dim=1)
