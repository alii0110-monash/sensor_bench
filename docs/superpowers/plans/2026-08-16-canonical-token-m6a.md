# SensorBench M6a: CanonicalToken 可移植性架构 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 定义 CanonicalToken 协议（4096 维规范空间）+ CanonicalTokenizer + 伪 token 资产化落盘（npz + index.json）+ LinearTokenToLLM per-LLM 投影层，让编码器与 LLM 空间彻底解耦。

**Architecture:** 伪 token 作为可移植跨模态统一表征（B' 路线）。冻结编码器（M5a alignment）+ Perceiver（M5b projection）→ CanonicalToken(4096) → 落盘 npz 资产化 → 任意消费者（检索/分类/LinearTokenToLLM 注入）。LLM 相关逻辑收敛到 `LinearTokenToLLM`（每 LLM 一个 4096→hidden 线性投影），复用现有 `LLMAdapter.inject`。

**Tech Stack:** Python 3.12, torch 2.9, numpy, pytest。checkpoints: `checkpoints_alignment/alignment_seed0.pt`（原型头 256→27→512）、`checkpoints_projection_verb/projection_seed0.pt`（Perceiver 4096）。v5 数据集带官方 captions。运行脚本用 `/home/li/projects/holollm/.venv/bin/python`。

**前置:** spec `docs/superpowers/specs/2026-08-16-canonical-token-portability-design.md`（2 轮评审 Approved）。当前 HEAD: `840defd`。

---

## 文件结构

```
framework/tokens/__init__.py       # tokens 包
framework/tokens/canonical.py      # CanonicalToken dataclass + 序列化/反序列化 + 校验
framework/tokens/tokenizer.py      # CanonicalTokenizer (传感器 ↔ CanonicalToken)
framework/tokens/assets.py         # 资产化: 落盘 npz + index.json + 版本化
framework/tokens/llm_proj.py       # TokenToLLM ABC + LinearTokenToLLM (复用 LLMAdapter.inject)
scripts/make_tokens.py             # 资产化生成 driver (v5 → v5tokens/)
tests/test_canonical.py            # CanonicalToken 协议单测
tests/test_tokenizer.py            # encode/decode round-trip
tests/test_assets.py               # 落盘/加载/版本化
tests/test_llm_proj.py             # LinearTokenToLLM 维度投影
tests/test_make_tokens_e2e.py      # mini v5 → 资产化 → 检索跑通
datasets/mmfi/v5tokens/            # 生成 (make_tokens 产出)
```

---

## Task 1: CanonicalToken 协议

**Files:**
- Create: `framework/tokens/__init__.py`
- Create: `framework/tokens/canonical.py`
- Test: `tests/test_canonical.py`

**说明**: CanonicalToken 是规范空间伪 token 的可移植载体（spec 组件 1）。data 固定 4096 维 float32，modality-major。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_canonical.py
import numpy as np
import pytest
from framework.tokens.canonical import CanonicalToken

def _tok():
    data = np.random.randn(40, 4096).astype(np.float32)   # M=5, k=8
    return CanonicalToken(id="s0", label=0, data=data,
                          modality_order=["wifi","depth","lidar","mmwave","rgb"],
                          k=8, meta={"encoder_version": "v0"})

def test_canonical_shape_and_fields():
    t = _tok()
    assert t.data.shape == (40, 4096)
    assert t.data.dtype == np.float32
    assert t.k == 8 and t.label == 0
    assert t.meta["encoder_version"] == "v0"

def test_canonical_modality_alignment():
    # validate() 校验: data 维度 + 行数 == len(modality_order)*k
    t = _tok()
    assert t.validate() is None  # 合法不抛

def test_canonical_invalid_dim():
    t = _tok()
    t.data = np.random.randn(40, 512).astype(np.float32)   # 错维
    with pytest.raises(ValueError):
        t.validate()

def test_canonical_roundtrip(tmp_path):
    import pickle
    t = _tok()
    p = tmp_path / "t.pkl"
    with open(p, "wb") as f:
        pickle.dump(t, f)
    with open(p, "rb") as f:
        t2 = pickle.load(f)
    assert np.array_equal(t.data, t2.data)
    assert t.id == t2.id and t.k == t2.k
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_canonical.py -v`
Expected: FAIL (ModuleNotFoundError: canonical)

- [ ] **Step 3: 实现 canonical.py**

```python
# framework/tokens/__init__.py
"""Portable canonical-token layer (M6a): CanonicalToken protocol + assets."""

# framework/tokens/canonical.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_canonical.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/tokens/ tests/test_canonical.py
git commit -m "feat(tokens): CanonicalToken 协议 (4096-dim, modality-major, 校验)"
```

---

## Task 2: CanonicalTokenizer（传感器 ↔ CanonicalToken）

**Files:**
- Create: `framework/tokens/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**说明**: 冻结加载 AlignmentModel（原型头）+ PerceiverProjection，传感器样本 → CanonicalToken（spec 组件 2）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tokenizer.py
import numpy as np
import pytest
from framework.tokens.canonical import CanonicalToken
from framework.tokens.tokenizer import CanonicalTokenizer

def test_tokenizer_encode_shape(tmp_path):
    """用真实 checkpoint 对 mini 样本编码 (CPU)."""
    import sys, os
    from framework.dataset.sample import Sample, Modality
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # 构造一个 5 模态样本 (形状与真实 v5 一致)
    mm = {
        "wifi": Modality(np.zeros((5,3,114,10), dtype=np.float32), [1,2,3,4,5], 10),
        "depth": Modality(np.zeros((5,1,224,224), dtype=np.float32), [1,2,3,4,5], 10),
        "lidar": Modality(np.zeros((5,1536,3), dtype=np.float32), [1,2,3,4,5], 10),
        "mmwave": Modality(np.zeros((5,64,5), dtype=np.float32), [1,2,3,4,5], 10),
        "rgb": Modality(np.zeros((5,17,2), dtype=np.float32), [1,2,3,4,5], 10),
    }
    s = Sample(id="s0", label=3, modalities=mm)
    tok = CanonicalTokenizer(align_ckpt="checkpoints_alignment/alignment_seed0.pt",
                             proj_ckpt="checkpoints_projection_verb/projection_seed0.pt",
                             k=8, device="cpu")
    ct = tok.encode(s)
    assert isinstance(ct, CanonicalToken)
    assert ct.data.shape == (40, 4096)   # 5 modal * 8
    assert ct.id == "s0" and ct.label == 3
    ct.validate()   # 通过校验

def test_tokenizer_decode_shape():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    tok = CanonicalTokenizer(align_ckpt="checkpoints_alignment/alignment_seed0.pt",
                             proj_ckpt="checkpoints_projection_verb/projection_seed0.pt",
                             k=8, device="cpu")
    # decode 不需要传感器, 直接返回 data 张量
    ct = CanonicalToken(id="s0", label=0, data=np.random.randn(40,4096).astype(np.float32),
                        modality_order=["wifi","depth","lidar","mmwave","rgb"], k=8)
    out = tok.decode(ct)
    assert out.shape == (1, 40, 4096)   # (1, M*k, H)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_tokenizer.py -v`
Expected: FAIL (ModuleNotFoundError: tokenizer)

- [ ] **Step 3: 实现 tokenizer.py**

```python
# framework/tokens/tokenizer.py
"""CanonicalTokenizer: 传感器样本 ↔ CanonicalToken (spec M6a 组件 2).

冻结 AlignmentModel + PerceiverProjection, 传感器 → 规范空间 token."""
from __future__ import annotations

import torch
import numpy as np

from ..models.alignment import AlignmentModel
from ..models.perceiver import PerceiverProjection
from .canonical import CanonicalToken

MODALITY_ORDER = ["wifi", "depth", "lidar", "mmwave", "rgb"]


class CanonicalTokenizer:
    def __init__(self, align_ckpt: str, proj_ckpt: str, k: int = 8, device: str = "cpu"):
        self.k = k
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.align = AlignmentModel(num_modalities=5, text_dim=512)
        self.align.projection_head = torch.nn.Sequential(
            torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))   # 原型头
        self.align.load_state_dict(torch.load(align_ckpt, map_location="cpu"), strict=False)
        self.align.eval().to(self.device)
        for p in self.align.parameters():
            p.requires_grad_(False)
        self.proj = PerceiverProjection(out_dim=4096, k=k).to(self.device)
        self.proj.load_state_dict(torch.load(proj_ckpt, map_location="cpu"))
        self.proj.eval()
        for p in self.proj.parameters():
            p.requires_grad_(False)

    def encode(self, sample) -> CanonicalToken:
        """传感器样本 → CanonicalToken."""
        mods = {m: torch.from_numpy(sample.modalities[m].data)[None].to(self.device)
                for m in MODALITY_ORDER if m in sample.modalities}
        avail = {m: True for m in mods}
        with torch.no_grad():
            ct = self.align.encode_modalities(mods, avail)   # (1, M, 16, 256)
            pe = self.proj(ct)                               # (1, M*k, 4096)
        data = pe[0].cpu().numpy().astype(np.float32)
        return CanonicalToken(id=sample.id, label=sample.label, data=data,
                              modality_order=MODALITY_ORDER, k=self.k,
                              meta={"encoder_version": "alignment_seed0+proj_verb"})

    def decode(self, tok: CanonicalToken) -> torch.Tensor:
        """CanonicalToken → (1, M*k, 4096) 张量.
        注: 返回带 batch 维 (1, M*k, H) 以便与 LLMAdapter.inject 拼接一致;
        spec 签名写 np.ndarray (M*k,4096) 为简化, 此处以张量实现为准."""
        tok.validate()
        return torch.from_numpy(tok.data)[None].to(self.device)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_tokenizer.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/tokens/tokenizer.py tests/test_tokenizer.py
git commit -m "feat(tokens): CanonicalTokenizer — 冻结编码器+Perceiver 传感器↔CanonicalToken"
```

---

## Task 3: 资产化（落盘 npz + index.json）

**Files:**
- Create: `framework/tokens/assets.py`
- Test: `tests/test_assets.py`

**说明**: CanonicalToken 落盘为 `{root}/tokens/{id}.npz`（data）+ `{root}/index.json`（版本化元数据）。参考 `write_meta` 模式（spec 组件 3）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_assets.py
import json, numpy as np
from framework.tokens.canonical import CanonicalToken
from framework.tokens.assets import write_tokens, load_tokens, write_index

def _toks(n=3):
    out = []
    for i in range(n):
        out.append(CanonicalToken(id=f"s{i}", label=i % 3,
            data=np.random.randn(40, 4096).astype(np.float32),
            modality_order=["wifi","depth","lidar","mmwave","rgb"], k=8,
            meta={"encoder_version": "v0"}))
    return out

def test_write_load_tokens(tmp_path):
    root = tmp_path / "tokens_root"
    toks = _toks()
    d0 = toks[0].data.copy()   # 捕获写入前数据 (避免 _toks() 每次随机)
    write_tokens(toks, str(root), version="v1", encoder_ckpt="ckpt0")
    # index.json
    idx = json.load(open(root / "index.json"))
    assert idx["version"] == "v1"
    assert idx["encoder_ckpt"] == "ckpt0"
    assert idx["n_samples"] == 3
    assert set(idx["samples"].keys()) == {"s0", "s1", "s2"}
    # npz 加载
    loaded = load_tokens(str(root))
    assert len(loaded) == 3
    assert np.array_equal(loaded["s0"].data, d0)   # 与写入前的数据一致
    assert loaded["s0"].label == 0

def test_load_tokens_missing_file(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    loaded = load_tokens(str(root))
    assert loaded == {}   # 空目录安全返回
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_assets.py -v`
Expected: FAIL (ModuleNotFoundError: assets)

- [ ] **Step 3: 实现 assets.py**

```python
# framework/tokens/assets.py
"""伪 token 资产化: CanonicalToken 落盘 npz + index.json (spec M6a 组件 3)."""
from __future__ import annotations
import json
import os
from typing import Dict, List

import numpy as np

from .canonical import CanonicalToken


def write_tokens(tokens: List[CanonicalToken], root: str, version: str,
                 encoder_ckpt: str) -> Dict:
    """落盘: {root}/tokens/{id}.npz + {root}/index.json. 返回 index dict."""
    tok_dir = os.path.join(root, "tokens")
    os.makedirs(tok_dir, exist_ok=True)
    samples = {}
    for t in tokens:
        t.validate()
        np.savez_compressed(os.path.join(tok_dir, f"{t.id}.npz"),
                            data=t.data, modality_order=t.modality_order,
                            label=t.label, k=t.k)
        samples[t.id] = {"label": t.label, "k": t.k,
                         "modality_order": t.modality_order}
    index = {"version": version, "encoder_ckpt": encoder_ckpt,
             "generated_at": __import__("datetime").datetime.now().isoformat(),
             "n_samples": len(tokens), "samples": samples}
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return index


def load_tokens(root: str) -> Dict[str, CanonicalToken]:
    """加载: {root}/index.json + {root}/tokens/*.npz → {id: CanonicalToken}."""
    idx_p = os.path.join(root, "index.json")
    if not os.path.exists(idx_p):
        return {}
    index = json.load(open(idx_p))
    tok_dir = os.path.join(root, "tokens")
    out = {}
    for sid in index.get("samples", {}):
        npz = np.load(os.path.join(tok_dir, f"{sid}.npz"))
        meta = index["samples"][sid]
        out[sid] = CanonicalToken(
            id=sid, label=int(meta["label"]),
            data=npz["data"].astype(np.float32),
            modality_order=list(npz["modality_order"]),
            k=int(meta["k"]),
            meta={"encoder_version": index.get("encoder_ckpt", "")})
    return out


def write_index(index: Dict, root: str) -> None:
    """独立更新 index.json (溯源单源)."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_assets.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/tokens/assets.py tests/test_assets.py
git commit -m "feat(tokens): 资产化 — npz + index.json 版本化落盘/加载"
```

---

## Task 4: LinearTokenToLLM（per-LLM 投影）

**Files:**
- Create: `framework/tokens/llm_proj.py`
- Test: `tests/test_llm_proj.py`

**说明**: CanonicalToken(4096) → 目标 LLM hidden 的线性投影（spec 组件 4）。复用现有 `LLMAdapter.inject` 拼接逻辑，本类只新增 `project(CanonicalToken)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_llm_proj.py
import numpy as np
import torch
from framework.tokens.canonical import CanonicalToken
from framework.tokens.llm_proj import TokenToLLM, LinearTokenToLLM

def _tok():
    return CanonicalToken(id="s0", label=0,
        data=np.random.randn(40, 4096).astype(np.float32),
        modality_order=["wifi","depth","lidar","mmwave","rgb"], k=8)

def test_linear_project_dim():
    proj = LinearTokenToLLM(llm_hidden=2048)
    ct = _tok()
    out = proj.project(ct)
    assert out.shape == (1, 40, 2048)   # (1, M*k, llm_hidden)

def test_linear_project_different_hidden():
    for h in (4096, 2048, 1024):
        proj = LinearTokenToLLM(llm_hidden=h)
        out = proj.project(_tok())
        assert out.shape == (1, 40, h)

def test_llm_hidden_property():
    assert LinearTokenToLLM(llm_hidden=2048).llm_hidden == 2048

def test_abstract_requires_project():
    import pytest
    with pytest.raises(TypeError):
        TokenToLLM()   # ABC
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_proj.py -v`
Expected: FAIL (ModuleNotFoundError: llm_proj)

- [ ] **Step 3: 实现 llm_proj.py**

```python
# framework/tokens/llm_proj.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_proj.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/tokens/llm_proj.py tests/test_llm_proj.py
git commit -m "feat(tokens): TokenToLLM + LinearTokenToLLM — per-LLM 维度投影"
```

---

## Task 5: make_tokens.py 资产化生成 driver

**Files:**
- Create: `scripts/make_tokens.py`
- Test: `tests/test_make_tokens_e2e.py`

**说明**: 遍历 v5 数据集（train base），CanonicalTokenizer.encode → 资产化落盘 `v5tokens/`（spec 组件 3 的生成流程）。

- [ ] **Step 1: 写失败测试 (mini e2e)**

```python
# tests/test_make_tokens_e2e.py
import json, os
import numpy as np
from framework.tokens.canonical import CanonicalToken

def test_make_tokens_mini(tmp_path):
    """mini v5 → 资产化 → 加载 → 检索跑通 (CPU, 用真实 checkpoint)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from make_tokens import make_tokens
    from framework.dataset.loader import load_dataset
    from framework.tokens.assets import load_tokens
    from framework.tokens.canonical import CANONICAL_DIM

    # 用真实 v5, 但只取前 2 个 train base 样本
    ds = load_dataset("datasets/mmfi/v5", mode="lazy")
    samples = []
    for s in ds.train:
        if "__aug" not in s.id:
            samples.append(s)
            if len(samples) >= 2:
                break
    out = make_tokens(samples, "checkpoints_alignment/alignment_seed0.pt",
                      "checkpoints_projection_verb/projection_seed0.pt",
                      str(tmp_path / "toks"), k=8, device="cpu")
    assert out["n_samples"] == 2
    loaded = load_tokens(str(tmp_path / "toks"))
    assert len(loaded) == 2
    for sid, t in loaded.items():
        assert t.data.shape == (40, CANONICAL_DIM)
        t.validate()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_make_tokens_e2e.py -v`
Expected: FAIL (ModuleNotFoundError: make_tokens)

- [ ] **Step 3: 实现 make_tokens.py**

```python
#!/usr/bin/env python
"""M6a: v5 传感器 → CanonicalToken 资产化落盘 (datasets/mmfi/v5tokens/)."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.tokens.assets import write_tokens
from framework.tokens.tokenizer import CanonicalTokenizer


def make_tokens(samples, align_ckpt, proj_ckpt, out_root, k=8, device="cpu") -> dict:
    tok = CanonicalTokenizer(align_ckpt=align_ckpt, proj_ckpt=proj_ckpt, k=k, device=device)
    tokens = []
    for s in samples:
        tokens.append(tok.encode(s))
    return write_tokens(tokens, out_root, version="v1",
                        encoder_ckpt=f"{align_ckpt}+{proj_ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="checkpoints_alignment/alignment_seed0.pt")
    ap.add_argument("--proj-ckpt", default="checkpoints_projection_verb/projection_seed0.pt")
    ap.add_argument("--out", default="datasets/mmfi/v5tokens")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0=全部 train base")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, mode="lazy")
    samples = [s for s in ds.train if "__aug" not in s.id]
    if args.limit:
        samples = samples[:args.limit]
    idx = make_tokens(samples, args.align_ckpt, args.proj_ckpt, args.out,
                      k=args.k, device=args.device)
    print(f"v5tokens: wrote {idx['n_samples']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_make_tokens_e2e.py -v`
Expected: 1 passed

- [ ] **Step 5: 运行真实资产化生成（后台 + 监控）**

Run:
```bash
cd /home/li/projects/sensorbench
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/make_tokens.py \
  --dataset datasets/mmfi/v5 --out datasets/mmfi/v5tokens --k 8 --device cuda \
  > /tmp/opencode/make_tokens.log 2>&1' < /dev/null > /dev/null 2>&1 &
```
Wait for completion, verify:
```bash
tail -1 /tmp/opencode/make_tokens.log   # expect: v5tokens: wrote 9205
du -sh datasets/mmfi/v5tokens
ls datasets/mmfi/v5tokens/tokens | wc -l
```
Expected: wrote 9205, ~4-5GB (9205×40×4096×4B ≈ 6GB, float32 压缩有限), 9205 npz files

- [ ] **Step 6: Commit**

```bash
cd /home/li/projects/sensorbench
git add scripts/make_tokens.py tests/test_make_tokens_e2e.py
git commit -m "feat(tokens): make_tokens — v5→v5tokens 资产化生成 driver"
```

---

## Task 6: 全量测试 + 状态更新

**Files:**
- Modify: `STATUS.md`
- Test: 全量 pytest

- [ ] **Step 1: 全量回归测试**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/ -q`
Expected: 全部通过（既有 122 + 新增 13 = 135，无回退）

- [ ] **Step 2: 更新 STATUS.md**

判断层追加：
```
- **M6a 完成（2026-08-16）**：CanonicalToken 协议（4096-dim 规范空间）+ CanonicalTokenizer + 资产化（npz+index.json）+ LinearTokenToLLM。伪 token 作为可移植跨模态统一表征，与 LLM 空间解耦。v5tokens 生成（9205 base）。
```
决策层追加：
```
- [x] `[已定]`：M6a——CanonicalToken 可移植性架构（协议+资产化）。✓ 完成
- [ ] `[提议]`：M6b——提编码器对齐质量（大 batch/分类辅助 loss/锚对比），L1 检索提升。
```

- [ ] **Step 3: 刷新事实层 + 提交**

Run:
```bash
cd /home/li/projects/sensorbench
/home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md
git add STATUS.md
git commit -m "docs(status): M6a 完成 — CanonicalToken 可移植性架构"
```

---

## 验收标准

1. 全测试绿：既有 122 + 新增 13 = 135，无回退。
2. CanonicalToken 协议：4096 维 float32、modality-major、校验拒绝错维。
3. CanonicalTokenizer encode/decode round-trip 正确（真实 checkpoint，CPU）。
4. 资产化：npz + index.json 落盘/加载，版本化，空目录安全。
5. LinearTokenToLLM：4096→任意 hidden 投影，4 个维度全测。
6. v5tokens 真实生成：9205 个 train base → npz，可被 load_tokens 加载。
7. STATUS.md 更新完成。
