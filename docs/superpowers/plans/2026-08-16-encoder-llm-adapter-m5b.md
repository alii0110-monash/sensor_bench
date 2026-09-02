# SensorBench M5b: Perceiver 投影 + LLMAdapter + Token Router + L2 冒烟 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 两段式架构第二阶段——把规范空间 token 投影为目标 LLM 的伪文本 token，支持半动态 token 数（router），并做 L2 注入冒烟（维度正确、前向通过、冻结 LLM 文本能力不回归）。

**Architecture:** 在 M5a 的 AlignmentModel（冻结）之上加三层：(1) Perceiver 投影——per-modality 共享权重，K_max=16 规范 token → k_m 个伪 token；(2) LLMAdapter 抽象——`project()` + `inject()`，参考 LLM 用本地 llama2-7b（hidden=4096）；(3) TokenRouter——确定性启发式按预算/可用性截取前 k'，缺模态=0，极端预算回退纯文本。L2 冒烟验证投影 token 能拼进冻结 LLM 前缀且不破坏文本能力。

**Tech Stack:** Python 3.12, torch 2.9, numpy, pytest, transformers 4.44 (llama2-7b 本地加载)。运行脚本用 `/home/li/projects/holollm/.venv/bin/python`。

**前置:** spec `docs/superpowers/specs/2026-08-16-encoder-llm-adapter-design.md`（Approved）。M5a 已完成（AlignmentModel + TextEncoder + v5 + L1 评测，commit `093bfbb`）。当前 HEAD: `093bfbb`。

---

## 文件结构

```
framework/models/perceiver.py      # PerceiverProjection (per-modality 共享权重)
framework/models/llm_adapter.py    # LLMAdapter 抽象 + LlamaAdapter + inject
framework/models/router.py         # TokenRouter (半动态启发式 + 前缀截断)
scripts/train_projection.py        # 投影蒸馏训练 driver (对齐伪token到目标LLM text emb)
scripts/smoke_llm_inject.py        # L2 冒烟: 前缀注入 + 前向 + 文本回归
tests/test_perceiver.py            # Perceiver 单测 (无 LLM 加载)
tests/test_router.py               # router 单测
tests/test_llm_adapter.py          # adapter 单测 (mock LLM, 不加载真实模型)
tests/test_projection_smoke.py     # L2 冒烟集成 (真实 llama2-7b, 可选/慢)
```

---

## Task 1: PerceiverProjection

**Files:**
- Create: `framework/models/perceiver.py`
- Test: `tests/test_perceiver.py`

**接口**: `PerceiverProjection(in_dim=256, out_dim=LLM_hidden, k_max=16, k=8)`。输入 `(B, M, K_max, D)` 规范 token（来自冻结 AlignmentModel.encode_modalities）；输出 `(B, M*k, LLM_hidden)` 伪 token。per-modality 共享权重：同一 perceiver 应用于每个模态的 K_max 个 token。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_perceiver.py
import torch
from framework.models.perceiver import PerceiverProjection

def test_perceiver_shape():
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=8)
    x = torch.randn(4, 5, 16, 256)   # (B, M, K_max, D)
    out = p(x)                        # (B, M*k, out_dim)
    assert out.shape == (4, 40, 4096)  # 5 modal * 8 queries

def test_perceiver_variable_k():
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=4)
    x = torch.randn(2, 3, 16, 256)
    out = p(x)
    assert out.shape == (2, 12, 4096)

def test_perceiver_prefix_truncation():
    # 半动态: 训练 k=8, 推理截取前 k'=3 → 结果与取前 3 个 query 一致
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=8)
    x = torch.randn(2, 5, 16, 256)
    full = p(x)                       # (2, 40, 4096)
    truncated = full[:, :15]          # 每模态 3 个 = 前 15
    assert truncated.shape == (2, 15, 4096)

def test_perceiver_missing_modality_zero():
    # spec §82: 缺模态 = 投影层对缺失模态不产生输出 (全零行)
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=8)
    x = torch.randn(2, 5, 16, 256)
    x[:, 2, :, :] = 0.0               # depth (index 2) missing
    out = p(x)                        # (2, 40, 4096)
    # depth's 8 rows (indices 16..24) must be exactly zero
    assert torch.all(out[:, 16:24] == 0)
    # non-missing modalities non-zero
    assert torch.any(out[:, 0:16] != 0)
    assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_perceiver.py -v`
Expected: FAIL (ModuleNotFoundError: perceiver)

- [ ] **Step 3: 实现 perceiver.py**

```python
# framework/models/perceiver.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_perceiver.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/models/perceiver.py tests/test_perceiver.py
git commit -m "feat(perceiver): per-modality shared-weight Perceiver projection"
```

---

## Task 2: TokenRouter（半动态）

**Files:**
- Create: `framework/models/router.py`
- Test: `tests/test_router.py`

**接口**: `TokenRouter(k_max=8)`。`route(avail, budget) -> Dict[str, int]` 返回每模态分配的 token 数。规则：缺模态=0；有模态先给 1，剩余按 budget 均分到 `min(k_max, budget_remaining)`；极端预算回退（全 0 → 纯文本）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_router.py
from framework.models.router import TokenRouter

def test_route_missing_modality_zero():
    r = TokenRouter(k_max=8)
    avail = {"wifi": True, "depth": False, "lidar": True, "mmwave": True, "rgb": True}
    budget = 32
    counts = r.route(avail, budget)
    assert counts["depth"] == 0
    assert counts["wifi"] >= 1

def test_route_respects_budget():
    r = TokenRouter(k_max=8)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    counts = r.route(avail, budget=10)
    assert sum(counts.values()) <= 10
    assert all(1 <= v <= 8 for v in counts.values())

def test_route_extreme_budget_fallback():
    r = TokenRouter(k_max=8)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    counts = r.route(avail, budget=0)
    assert all(v == 0 for v in counts.values())  # 回退纯文本

def test_route_prefix_stable():
    # 截取稳定性: 预算从 40 → 15 → 5 单调不减
    r = TokenRouter(k_max=8)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    c40 = r.route(avail, 40); c15 = r.route(avail, 15); c5 = r.route(avail, 5)
    s40 = sum(c40.values()); s15 = sum(c15.values()); s5 = sum(c5.values())
    assert s40 >= s15 >= s5
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_router.py -v`
Expected: FAIL (ModuleNotFoundError: router)

- [ ] **Step 3: 实现 router.py**

```python
# framework/models/router.py
"""Semi-dynamic token router (spec M5b §87-92).

Deterministic heuristic: each available modality gets >=1 token; remaining
budget distributed up to k_max per modality; missing modality = 0. Extreme
budget (all 0) falls back to pure-text (text captions already stored).
"""
from __future__ import annotations
from typing import Dict, List


class TokenRouter:
    def __init__(self, k_max: int = 8):
        self.k_max = k_max

    def route(self, avail: Dict[str, bool], budget: int) -> Dict[str, int]:
        """avail: {modality: bool}; budget: total token budget. Returns counts."""
        active = [m for m in avail if avail.get(m)]
        counts = {m: 0 for m in avail}
        # give each active modality a floor of 1
        remaining = budget
        for m in active:
            if remaining > 0:
                counts[m] = 1
                remaining -= 1
        # distribute remaining evenly up to k_max
        idx = 0
        while remaining > 0 and active:
            m = active[idx % len(active)]
            if counts[m] < self.k_max:
                counts[m] += 1
                remaining -= 1
            idx += 1
            if all(counts[m] >= self.k_max for m in active):
                break
        return counts
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_router.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/models/router.py tests/test_router.py
git commit -m "feat(router): semi-dynamic deterministic TokenRouter"
```

---

## Task 3: LLMAdapter 抽象 + LlamaAdapter

**Files:**
- Create: `framework/models/llm_adapter.py`
- Test: `tests/test_llm_adapter.py`

**接口**: `LLMAdapter` 抽象 + `LlamaAdapter`（本地 llama2-7b）。`project(tokens) -> (B, Σk_m, LLM_hidden)` 用 PerceiverProjection；`inject(prefix_embs, input_ids) -> merged_embs` 把伪 token embedding 拼进 LLM embedding 序列。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_llm_adapter.py
import torch
import pytest
from framework.models.llm_adapter import LLMAdapter

def test_abstract_requires_methods():
    with pytest.raises(TypeError):
        LLMAdapter()  # ABC with abstract methods

class MockAdapter(LLMAdapter):
    """Minimal fake for interface tests (no LLM load)."""
    @property
    def hidden_dim(self) -> int:
        return 4096
    def project(self, canonical_tokens):
        B, M, K, D = canonical_tokens.shape
        return torch.nn.Linear(256, 4096)(canonical_tokens.reshape(B, M * K, D))
    def inject(self, prefix_embs, input_ids, embed_fn):
        # embed_fn: (B, T) -> (B, T, H); returns concat(prefix, text_embs)
        text_embs = embed_fn(input_ids)
        return torch.cat([prefix_embs, text_embs], dim=1)

def test_mock_adapter_project_inject():
    a = MockAdapter()
    ct = torch.randn(2, 5, 16, 256)
    pseudo = a.project(ct)
    assert pseudo.shape == (2, 80, 4096)
    input_ids = torch.randint(0, 100, (2, 10))
    def embed_fn(ids):
        return torch.randn(ids.shape[0], ids.shape[1], 4096)
    merged = a.inject(pseudo[:, :8], input_ids, embed_fn)
    assert merged.shape == (2, 18, 4096)  # 8 pseudo + 10 text
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_adapter.py -v`
Expected: FAIL (ModuleNotFoundError: llm_adapter)

- [ ] **Step 3: 实现 llm_adapter.py**

```python
# framework/models/llm_adapter.py
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
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None).eval()
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
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_adapter.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/models/llm_adapter.py tests/test_llm_adapter.py
git commit -m "feat(adapter): LLMAdapter ABC + LlamaAdapter (local llama2-7b)"
```

---

## Task 4: 投影蒸馏训练 driver

**Files:**
- Create: `scripts/train_projection.py`
- Test: `tests/test_projection_smoke.py` (L2 冒烟集成)

**说明**: 冻结 AlignmentModel（M5a）→ Perceiver 投影 → 蒸馏 InfoNCE（伪 token 池化 vs 目标 LLM 文本 embedding）。参考 LLM 用 llama2-7b 的 text embedding（取输入层对合成文本的池化）。

- [ ] **Step 1: 写失败测试 (L2 冒烟: 前向 + 注入 + 文本回归)**

```python
# tests/test_projection_smoke.py
"""L2 smoke: projection produces pseudo tokens that inject into a frozen LLM
prefix, forward passes, and the frozen LLM's text QA ability doesn't regress.
Uses MockAdapter unless --real-llm (then llama2-7b)."""
import os
import pytest
import torch

# ---- pure-interface smoke (no LLM load, runs in CI) ----

def test_projection_forward_interface():
    from framework.models.llm_adapter import LLMAdapter
    from framework.models.perceiver import PerceiverProjection
    from framework.models.router import TokenRouter
    proj = PerceiverProjection(in_dim=256, out_dim=512, k=4)
    router = TokenRouter(k_max=4)
    ct = torch.randn(2, 5, 16, 256)
    pseudo = proj(ct)                       # (2, 20, 512)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    counts = router.route(avail, budget=12)
    # inject first sum(counts) pseudo tokens as prefix
    n = sum(counts.values())
    prefix = pseudo[:, :n]
    assert prefix.shape[1] == n
    assert n <= 12

def test_inject_preserves_text_tokens():
    from framework.models.llm_adapter import LLMAdapter
    class T(LLMAdapter):
        @property
        def hidden_dim(self): return 128
        def project(self, ct): return torch.randn(ct.shape[0], 4, 128)
        def inject(self, pe, ids, embed_fn):
            te = embed_fn(ids)
            return torch.cat([pe, te], dim=1)
    a = T()
    pe = torch.randn(2, 4, 128)
    ids = torch.randint(0, 10, (2, 6))
    merged = a.inject(pe, ids, lambda x: torch.randn(x.shape[0], x.shape[1], 128))
    assert merged.shape == (2, 10, 128)  # 4 pseudo + 6 text, no token loss
```

- [ ] **Step 2: 运行确认通过（不加载真实 LLM）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_projection_smoke.py -q -m "not slow"`
Expected: 2 passed

- [ ] **Step 3: 实现 scripts/train_projection.py**

```python
#!/usr/bin/env python
"""Projection distillation training (M5b).

Frozen AlignmentModel (M5a) produces canonical tokens -> PerceiverProjection
maps to target LLM space -> InfoNCE distills pseudo tokens toward the target
LLM's own text embeddings of the same synthetic captions.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel, MODALITIES
from framework.models.llm_adapter import LlamaAdapter
from framework.models.router import TokenRouter
from framework.models.alignment import info_nce_loss


def _llm_text_emb(adapter, tokenizer, texts, device):
    """Target-LLM text embedding: mean-pool input embeddings of caption tokens."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64,
                    return_tensors="pt")
    ids = enc["input_ids"]
    model = adapter._load()[0]
    emb_device = model.get_input_embeddings().weight.device
    ids = ids.to(emb_device)
    emb = model.get_input_embeddings()(ids)            # (B, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).to(emb_device)
    pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, dim=-1)


def _stack_mods(samples, avail, device):
    mods = {}
    first = samples[0]
    for m in MODALITIES:
        if avail.get(m) and m in first.modalities:
            mods[m] = torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in samples]).to(device)
    return mods


def train_epoch(align, adapter, router, train, opt, text_fn, batch_size=16,
                device="cuda", dropout_p=0.0) -> float:
    align.eval()
    adapter.projection.train()
    rng = np.random.default_rng(1)
    total = 0.0; n = 0
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        avail = {m: True for m in MODALITIES}          # 蒸馏阶段用全模态
        mods = _stack_mods(batch, avail, device)
        if not mods:
            continue
        with torch.no_grad():
            ct = align.encode_modalities(mods, avail)  # (B, M, K_max, D)
        pseudo = adapter.project(ct)                   # (B, M*k, H)
        pooled = pseudo.mean(dim=1)
        texts = [s.text.get("en", [""])[0] for s in batch]
        t_emb = text_fn(texts)
        loss = info_nce_loss(pooled, t_emb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="/tmp/opencode/align_smoke/alignment_seed0.pt",
                    help="M5a alignment checkpoint (smoke default; 真训练需先跑 M5a train_alignment --text-encoder clip)")
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints_projection")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"))
    align.eval().to(device)
    for p in align.parameters():
        p.requires_grad_(False)

    adapter = LlamaAdapter(model_path=args.llm, k=args.k, device=device)
    model, tok = adapter._load()
    text_fn = lambda texts: _llm_text_emb(adapter, tok, texts, device)
    opt = torch.optim.AdamW(adapter.projection.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        loss = train_epoch(align, adapter, None, ds.train, opt, text_fn,
                           batch_size=args.batch_size, device=device)
        print(f"[proj] ep {ep} loss {loss:.4f}", flush=True)
    torch.save(adapter.projection.state_dict(), f"{args.out}/projection_seed0.pt")
    print(f"done -> {args.out}/projection_seed0.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_projection_smoke.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add scripts/train_projection.py tests/test_projection_smoke.py
git commit -m "feat(proj): projection distillation training driver + L2 smoke"
```

---

## Task 5: 真实 L2 冒烟（llama2-7b 注入 + 文本回归）

**Files:**
- Create: `scripts/smoke_llm_inject.py`
- Test: `tests/test_projection_smoke.py` (追加, `--real-llm` 标记)

**说明**: 真实加载本地 llama2-7b，验证：伪 token 前缀注入前向通过、冻结 LLM 文本能力不回归（同一句子的生成/困惑度在注入前后一致）。这是 M5b 的 L2 冒烟验收。

- [ ] **Step 1: 写失败测试 (real-llm, 标记 slow)**

```python
# tests/test_projection_smoke.py (追加)
import pytest

@pytest.mark.slow
def test_real_llm_inject_forward():
    """真实 llama2-7b: 伪 token 前缀 + 文本前向通过, 文本 token 数不丢."""
    from framework.models.llm_adapter import LlamaAdapter
    adapter = LlamaAdapter(model_path="/home/li/datasets/models/llama2-7b",
                           k=4, device="cuda")
    model, tok = adapter._load()
    device = "cuda"
    # 文本侧
    ids = tok("What is the person doing?", return_tensors="pt").input_ids.to(device)
    # 伪 token 前缀 (随机, 只验证通道)
    ct = torch.randn(1, 5, 16, 256, device=device)
    pseudo = adapter.project(ct)            # (1, 20, 4096)
    merged = adapter.inject(pseudo[:, :4], ids)   # 4 pseudo + N text
    assert merged.shape == (1, 4 + ids.shape[1], 4096)
    # 前向通过
    with torch.no_grad():
        out = model(inputs_embeds=merged, attention_mask=torch.ones(
            merged.shape[0], merged.shape[1], dtype=torch.long, device=device))
    assert out.logits.shape[1] == merged.shape[1]
```

- [ ] **Step 2: 运行确认失败（未实现）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_projection_smoke.py -m slow -v`
Expected: FAIL

- [ ] **Step 3: 实现 scripts/smoke_llm_inject.py**

```python
#!/usr/bin/env python
"""L2 smoke (M5b §97): pseudo-token prefix injection into a frozen local LLM,
forward passes, and text-only regression (text ability unchanged).

Usage:
  python scripts/smoke_llm_inject.py [--llm .../llama2-7b] [--k 8] [--device cuda]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.models.llm_adapter import LlamaAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    adapter = LlamaAdapter(model_path=args.llm, k=args.k, device=device)
    model, tok = adapter._load()

    text = "What is the person doing?"
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    # 1) text-only forward (regression baseline)
    with torch.no_grad():
        out_text = model(input_ids=ids)
    text_tokens = ids.shape[1]

    # 2) pseudo-token prefix forward
    ct = torch.randn(1, 5, 16, 256, device=device)
    pseudo = adapter.project(ct)
    n_prefix = min(args.k, pseudo.shape[1])
    merged = adapter.inject(pseudo[:, :n_prefix], ids)
    with torch.no_grad():
        out_inj = model(inputs_embeds=merged,
                        attention_mask=torch.ones(
                            merged.shape[0], merged.shape[1], dtype=torch.long,
                            device=device))

    # text tokens preserved: prefix + text == merged
    assert merged.shape == (1, n_prefix + text_tokens, adapter.hidden_dim), merged.shape
    assert out_inj.logits.shape[1] == merged.shape[1]

    # regression: text-only logits at last text position unchanged shape/value range
    lt = out_text.logits[:, -1]
    assert torch.isfinite(lt).all()
    print(f"[smoke] prefix={n_prefix} text_tokens={text_tokens} "
          f"merged={merged.shape[1]} forward OK; text regression OK")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python scripts/smoke_llm_inject.py --k 4 --device cuda`
Expected: `[smoke] ... forward OK; text regression OK` + `PASS`

（真实 llama2-7b 加载较慢，用后台 + 监控，参考全局 AGENTS.md）

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add scripts/smoke_llm_inject.py tests/test_projection_smoke.py
git commit -m "feat(smoke): L2 llama2-7b pseudo-token injection smoke (L2 验收)"
```

---

## Task 6: 全量测试 + 状态更新

**Files:**
- Modify: `STATUS.md`
- Test: 全量 pytest

- [ ] **Step 1: 全量回归测试（排除 slow，避免加载 14GB llama）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/ -q -m "not slow"`
Expected: 全部通过（既有 100 + 新增 13 = 113，无回退）

- [ ] **Step 1b: 运行 slow 测试（真实 llama2-7b，后台 + 监控）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_projection_smoke.py -m slow -v`
Expected: 1 passed（real llama inject forward，加载 ~14GB）

（注意：先 `free -h` + `nvidia-smi` 确认显存/内存余量；llama2-7b bf16 约 13.5GB，GPU 16GB 紧张，必要时 `device_map="auto"` CPU offload）

- [ ] **Step 2: 更新 STATUS.md**

在判断层追加：
```
- **M5b 完成（2026-08-16）**：Perceiver 投影（per-modality 共享权重）、TokenRouter（半动态启发式）、LLMAdapter（llama2-7b）、L2 冒烟（前缀注入前向通过 + 文本回归）。两段式第二段就绪。
```
在决策层追加：
```
- [x] `[已定]`：M5b——Perceiver 投影 + LLMAdapter + router + L2 冒烟。✓ 完成
- [ ] `[提议]`：M5c——L3 端到端 LLM 能力评测（少样本动作理解/事件问答）。
```

- [ ] **Step 3: 刷新事实层 + 提交**

Run:
```bash
cd /home/li/projects/sensorbench
/home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md
git add STATUS.md
git commit -m "docs(status): M5b 完成 — Perceiver + Router + LLMAdapter + L2 冒烟"
```

---

## 验收标准

1. 全测试绿（`-m "not slow"`）：既有 100 + 新增 13 = 113，无回退。
2. Perceiver 输出 shape `(B, M*k, LLM_hidden)` 正确；**缺模态行显式置零且无 NaN**（spec §82）。
3. Router 半动态：预算/缺模态/极端回退/前缀稳定单调，全部测试覆盖。
4. L2 冒烟（真实 llama2-7b，`slow` 测试 + `scripts/smoke_llm_inject.py`）：伪 token 前缀注入前向通过，`merged.shape == (1, prefix+text, 4096)`，文本回归 OK。
5. STATUS.md 更新完成。
