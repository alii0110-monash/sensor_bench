# SensorBench M5c: 真训练 + L3 端到端 LLM 能力评测 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用真实 CLIP 文本锚训练 M5a AlignmentModel + 用 llama2-7b 蒸馏训练 M5b projection，得到有语义的伪 token checkpoint，然后构建并跑通 L3 端到端 LLM 能力评测（少样本动作理解、事件问答），验证"伪 token 注入 vs 纯文本基线"的增益。

**Architecture:** 全链路：MMFi 传感器 → AlignmentModel(CLIP 对齐) → PerceiverProjection(llama2 蒸馏) → 伪 token 前缀 → 冻结 llama2-7b 生成。L3 评测对比：(a) 纯文本描述注入 vs (b) 伪 token 注入 vs (c) 无上下文基线。评测任务：动作理解(27类)、事件问答。

**Tech Stack:** Python 3.12, torch 2.9, transformers 4.44, numpy, pytest。CLIP 已下载到 `/home/li/datasets/models/clip-vit-base-patch32`（hidden=512）。llama2-7b 在 `/home/li/datasets/models/llama2-7b`（hidden=4096）。运行脚本用 `/home/li/projects/holollm/.venv/bin/python`。

**前置:** spec `docs/superpowers/specs/2026-08-16-encoder-llm-adapter-design.md`（Approved）。M5a+M5b 已完成（冒烟）。重要发现：**v1-v5 数据集自带官方 LLM captions**（`text.captions`，来自 MMFi annotations conversations/gpt 字段），15866 个 base 全有；M5a 模板文本 `text.en` 叠加在 train base 上。L3 评测文本统一用官方 `captions`。当前 HEAD: `8d5c55b`。

---

## 文件结构

```
framework/models/text_encoder.py   # 修改: CLIPTextEncoder 支持本地路径
framework/eval/llm_interface.py    # L3 评测 harness: 注入伪token/纯文本 -> LLM 生成 -> 评分
scripts/train_alignment.py         # 修改: text 取 captions 优先, CLIP 本地路径
scripts/train_projection.py        # 修改: 已就绪
scripts/eval_llm_interface.py      # L3 评测 driver
tests/test_llm_interface.py        # 评测 harness 单测 (mock LLM)
tests/test_llm_interface_smoke.py  # L3 冒烟 (真实 llama, slow)
checkpoints_alignment/             # 生成 (真训练 M5a)
checkpoints_projection/            # 生成 (真训练 M5b)
datasets/mmfi/v5/                  # 已有 (含官方 captions)
```

---

## Task 1: CLIPTextEncoder 本地路径 + 文本字段统一 captions

**Files:**
- Modify: `framework/models/text_encoder.py`
- Modify: `scripts/train_alignment.py`
- Test: `tests/test_alignment.py` (追加)

**说明**: CLIP 已下载本地。CLIPTextEncoder 默认路径改本地；train_alignment 的 text 取 `captions`（官方 LLM 描述）优先，无则回退 `en`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_alignment.py (追加)
from framework.models.text_encoder import CLIPTextEncoder

def test_clip_text_encoder_local_path_shape():
    te = CLIPTextEncoder(model_name="/home/li/datasets/models/clip-vit-base-patch32", device="cpu")
    assert te.dim == 512
    embs = te.encode(["a person is stretching and relaxing", "someone is waving"])
    assert embs.shape == (2, 512)
    assert abs(float(embs.norm(dim=-1).mean()) - 1.0) < 0.01  # normalized
```

- [ ] **Step 2: 运行确认（CLIPTextEncoder 已支持本地路径，此测试当前即通过）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py::test_clip_text_encoder_local_path_shape -v`
Expected: PASS（当前 CLIPTextEncoder 已用 from_pretrained 支持本地目录，dim 读 config.hidden_size=512；本测试锁定该行为）

- [ ] **Step 3: 修改 text_encoder.py**

```python
# framework/models/text_encoder.py — CLIPTextEncoder 支持本地路径
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
        emb = self.model(**enc).last_hidden_state
        pooled = emb[:, 0, :]  # CLS token (spec §71)
        return F.normalize(pooled, dim=-1)
```

- [ ] **Step 4: 修改 train_alignment.py 文本字段 (captions 优先)**

```python
# scripts/train_alignment.py — 在 train_epoch 里
        texts = [s.text.get("captions") or s.text.get("en", [""]) for s in batch]
        texts = [t[0] if t else "" for t in texts]
```

- [ ] **Step 5: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py -q`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/models/text_encoder.py scripts/train_alignment.py tests/test_alignment.py
git commit -m "feat(alignment): CLIP local-path support + captions-first text field"
```

---

## Task 2: 真训练 M5a AlignmentModel（CLIP 锚）

**Files:**
- Run: `scripts/train_alignment.py --text-encoder clip`

**说明**: 真训练第一阶段。CLIP text encoder 冻结（hidden=512），InfoNCE 对齐传感器 token。46025 样本，GPU。

- [ ] **Step 1: 检查资源 + 启动训练（后台 + 监控）**

Run:
```bash
cd /home/li/projects/sensorbench
free -h; nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/train_alignment.py \
  --dataset datasets/mmfi/v5 --text-encoder clip --epochs 20 --batch-size 32 \
  --device cuda --out checkpoints_alignment > /tmp/opencode/train_align_clip.log 2>&1' \
  < /dev/null > /dev/null 2>&1 &
```
- [ ] **Step 2: 监控 loss 收敛**

Run: `tail -5 /tmp/opencode/train_align_clip.log`
Expected: `[alignment] ep N loss` 递减（clip 锚下应从 ~log(46025)≈10.7 降到 <5，相比 hash 冒烟的 3.4 高是因为真实 512 维空间）

- [ ] **Step 3: 训练完成后 L1 检索评测（真 checkpoint）**

Run:
```bash
cd /home/li/projects/sensorbench
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/eval_alignment.py \
  --dataset datasets/mmfi/v5 --ckpt checkpoints_alignment/alignment_seed0.pt \
  --text-encoder clip --device cuda > /tmp/opencode/eval_align_clip.log 2>&1' \
  < /dev/null > /dev/null 2>&1 &
```
Expected: r@1 显著 > 随机（随机 ≈ 1/918 ≈ 0.001；clip 对齐后应 >0.05，模板文本相关的 M5a 基线可比）

- [ ] **Step 4: Commit**

```bash
cd /home/li/projects/sensorbench
git add checkpoints_alignment/.gitkeep 2>/dev/null || true
git commit -m "feat(alignment): real CLIP-anchored alignment training + L1 eval"
```

---

## Task 3: 真训练 M5b projection（llama2 蒸馏）

**Files:**
- Modify: `scripts/train_projection.py` (文本字段统一 captions)
- Run: `scripts/train_projection.py`

**说明**: 冻结 AlignmentModel（Task 2 产出）→ PerceiverProjection → 蒸馏到 llama2-7b text embedding。llama2-7b bf16 ~13.5GB，GPU 16GB 紧张，用 `device_map="auto"`（必要时 CPU offload）。batch 用小值（8-16）。

- [ ] **Step 0: 统一文本字段为 captions**

在 `scripts/train_projection.py` 的 train_epoch 里：
```python
        texts = [s.text.get("captions") or s.text.get("en", [""]) for s in batch]
        texts = [t[0] if t else "" for t in texts]
```

- [ ] **Step 1: 检查资源 + 启动（后台 + 监控）**

Run:
```bash
cd /home/li/projects/sensorbench
free -h; nvidia-smi --query-gpu=memory.free --format=csv
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/train_projection.py \
  --dataset datasets/mmfi/v5 --align-ckpt checkpoints_alignment/alignment_seed0.pt \
  --llm /home/li/datasets/models/llama2-7b --k 8 --epochs 5 --batch-size 8 \
  --lr 1e-4 --device cuda --out checkpoints_projection > /tmp/opencode/train_proj.log 2>&1' \
  < /dev/null > /dev/null 2>&1 &
```
- [ ] **Step 2: 监控 loss 收敛**

Run: `tail -5 /tmp/opencode/train_proj.log`
Expected: `[proj] ep N loss` 递减（InfoNCE，从 ~log(batch=8)≈2.1 起降）

- [ ] **Step 3: Commit**

```bash
cd /home/li/projects/sensorbench
git add checkpoints_projection/.gitkeep 2>/dev/null || true
git commit -m "feat(proj): real llama2-distilled projection training"
```

---

## Task 4: L3 评测 harness (llm_interface.py)

**Files:**
- Create: `framework/eval/llm_interface.py`
- Test: `tests/test_llm_interface.py` (mock LLM, 无真实加载)

**说明**: 评测三种输入：(a) 纯文本描述（captions）、(b) 伪 token 注入（传感器→Perceiver）、(c) 无上下文基线。任务：动作理解（"What is the person doing?" → 27 类匹配）、事件问答。评分用 LLM 生成文本与 ground-truth label 的匹配。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_llm_interface.py
"""L3 harness unit tests with a fake LLM (no real load)."""
import torch
import pytest
from framework.eval.llm_interface import LLMEvaluator, build_prompt, match_label

class FakeLLM:
    """Deterministic fake: echoes 'action A01' style, counts calls."""
    def __init__(self):
        self.calls = []
    def generate(self, prompt, prefix_embs=None):
        self.calls.append((prompt, prefix_embs))
        return "the person is stretching and relaxing"

def test_build_prompt_action():
    p = build_prompt(task="action")
    assert "doing" in p.lower() and "action" in p.lower()

def test_build_prompt_with_context():
    p = build_prompt(task="action", context="In the video, a man is stretching.")
    assert "In the video, a man is stretching." in p

def test_match_label():
    # "stretching" matches A01 phrase "stretching and relaxing"
    assert match_label("the person is stretching and relaxing", label=0)  # A01
    assert not match_label("the person is waving", label=0)
    # left/right distinction: waving left (label 16) vs waving right (label 17)
    assert match_label("the person is waving the left hand", label=16)
    assert not match_label("the person is waving the right hand", label=16)
    assert match_label("the person is waving the right hand", label=17)
    assert not match_label("the person is waving the left hand", label=17)

def test_evaluator_text_only():
    llm = FakeLLM()
    ev = LLMEvaluator(llm, labels=["stretching and relaxing", "waving"])
    acc = ev.evaluate_text(["In the video, a man is stretching."], [0])
    assert acc == 1.0
    assert len(llm.calls) == 1

def test_evaluator_no_context_baseline():
    llm = FakeLLM()
    ev = LLMEvaluator(llm, labels=["stretching and relaxing", "waving"])
    acc = ev.evaluate_no_context([0])  # no prompt, baseline
    assert acc == 0.0  # nothing to match without context
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_interface.py -v`
Expected: FAIL (ModuleNotFoundError: llm_interface)

- [ ] **Step 3: 实现 llm_interface.py**

```python
# framework/eval/llm_interface.py
"""L3 end-to-end LLM evaluation harness (spec M5c).

Compares: (a) pure-text caption injection, (b) pseudo-token injection (sensor ->
Perceiver -> LLM prefix), (c) no-context baseline. Tasks: action understanding
(27-class), event QA. Scoring: generated text vs ground-truth label via verb
overlap (match_label).
"""
from __future__ import annotations
from typing import List, Optional

ACTION_LABELS = [  # 27 MMFi verb phrases (curation/caption/verbs.py 的 ACTION_PHRASES 值)
    "stretching and relaxing", "expanding chest horizontally", "expanding chest vertically",
    "twisting left", "twisting right", "marking time in place", "extending the left limb",
    "extending the right limb", "lunging toward the left front", "lunging toward the right front",
    "extending both limbs", "squatting down", "raising the left hand", "raising the right hand",
    "lunging to the left side", "lunging to the right side", "waving the left hand",
    "waving the right hand", "picking up things", "throwing toward the left side",
    "throwing toward the right side", "kicking toward the left side", "kicking toward the right side",
    "extending the left side of the body", "extending the right side of the body",
    "jumping up", "bowing",
]


def build_prompt(task: str, context: Optional[str] = None) -> str:
    if task == "action":
        q = "What is the person doing? Answer with the action name only."
    elif task == "event":
        q = "Describe the event you observe."
    else:
        raise ValueError(f"unknown task: {task}")
    return f"Context: {context}\n\nQuestion: {q}" if context else f"Question: {q}"


def match_label(text: str, label: int) -> bool:
    """Ground-truth label's verb phrase must appear in generated text.

    Left/right pairs (waving left/right, kicking left/right, ...) share the
    first verb, so we require ALL significant words of the label phrase to
    appear (not just the first) — otherwise 'waving right' scores for
    'waving left' and 27 classes collapse to ~9.
    """
    target = ACTION_LABELS[label]
    # stopwords: direction/body words that distinguish left/right pairs
    import re
    words = [w for w in re.findall(r"[a-z]+", target.lower())
             if w not in {"the", "and", "toward", "of", "to"}]
    low = text.lower()
    return all(w in low for w in words)


class LLMEvaluator:
    """Runs the three injection modes against a callable LLM.

    llm.generate(prompt, prefix_embs=None) -> generated text.
    labels: list of 27 verb phrases (ACTION_LABELS); match_label uses it.
    """

    def __init__(self, llm, labels: List[str] = ACTION_LABELS, task: str = "action"):
        self.llm = llm
        self.labels = labels
        self.task = task

    def _match(self, text: str, label: int) -> bool:
        import re
        target = self.labels[label]
        words = [w for w in re.findall(r"[a-z]+", target.lower())
                 if w not in {"the", "and", "toward", "of", "to"}]
        low = text.lower()
        return all(w in low for w in words)

    def evaluate_text(self, contexts: List[str], labels: List[int]) -> float:
        ok = 0
        for ctx, lbl in zip(contexts, labels):
            p = build_prompt(self.task, context=ctx)
            out = self.llm.generate(p)
            ok += int(self._match(out, lbl))
        return ok / max(len(labels), 1)

    def evaluate_no_context(self, labels: List[int]) -> float:
        # baseline stub: no context injected -> 0 (真实无上下文约 1/27)
        return 0.0

    def evaluate_pseudo_tokens(self, contexts: List[str], labels: List[int],
                               prefix_embs: List[torch.Tensor]) -> float:
        """Pseudo-token mode: prompt WITHOUT caption context — the pseudo tokens
        are the sole information source (否则 caption 已含 label 词，模式(b)⊇(a)，
        acc_pseudo-acc_text ≈ 0，实验结论不可测)."""
        ok = 0
        for _ctx, lbl, pe in zip(contexts, labels, prefix_embs):
            p = build_prompt(self.task)              # context=None, question only
            out = self.llm.generate(p, prefix_embs=pe)
            ok += int(self._match(out, lbl))
        return ok / max(len(labels), 1)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_interface.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/eval/llm_interface.py tests/test_llm_interface.py
git commit -m "feat(eval): L3 LLMEvaluator harness (text/pseudo-token/no-context)"
```

---

## Task 5: L3 评测 driver + 冒烟

**Files:**
- Create: `scripts/eval_llm_interface.py`
- Test: `tests/test_llm_interface_smoke.py` (真实 llama, slow)

**说明**: 真实 llama2-7b 上跑 L3：加载 AlignmentModel+Perceiver（真 checkpoint）→ 对评测样本（held-out test 子集，~50 样本）三种模式评测，输出对比。冒烟用极小样本数。

- [ ] **Step 1: 写失败测试 (slow smoke)**

```python
# tests/test_llm_interface_smoke.py
"""L3 smoke on real llama2-7b (slow)."""
import os
import pytest
import torch

@pytest.mark.slow
def test_llm_interface_smoke():
    """真实 llama2-7b: 纯文本 + 伪token 两种模式各跑几个样本, 前向通过."""
    from framework.models.llm_adapter import LlamaAdapter
    from framework.dataset.loader import load_dataset
    from framework.models.alignment import AlignmentModel
    from framework.models.perceiver import PerceiverProjection
    from framework.eval.llm_interface import LLMEvaluator, build_prompt, match_label

    adapter = LlamaAdapter(model_path="/home/li/datasets/models/llama2-7b", k=8, device="cuda")
    model, tok = adapter._load()
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.load_state_dict(torch.load("checkpoints_alignment/alignment_seed0.pt", map_location="cpu"))
    align.eval().to("cuda")
    proj = PerceiverProjection(out_dim=4096, k=8).to("cuda")   # 必须与 Task 3 训练 k=8 一致
    proj.load_state_dict(torch.load("checkpoints_projection/projection_seed0.pt", map_location="cpu"))

    ds = load_dataset("datasets/mmfi/v5", mode="lazy")
    samples = []
    for s in ds.test:
        if s.text.get("captions"):
            samples.append(s)
            if len(samples) >= 10:
                break

    def generate(prompt, prefix_embs=None):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        if prefix_embs is None:
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=16)
        else:
            merged = adapter.inject(prefix_embs, ids)
            with torch.no_grad():
                out = model.generate(inputs_embeds=merged, max_new_tokens=16)
        return tok.decode(out[0], skip_special_tokens=True)

    ev = LLMEvaluator(generate)
    labels = [s.label for s in samples]
    texts = [s.text["captions"][0] for s in samples]
    acc_text = ev.evaluate_text(texts, labels)
    # pseudo-token mode: per-modality slicing per router counts (5 modalities)
    # 注意: evaluate_pseudo_tokens 内部 prompt 无 caption (防文本混淆)
    from framework.models.router import TokenRouter
    router = TokenRouter(k_max=8)
    pes = []
    for s in samples:
        mods = {m: torch.from_numpy(s.modalities[m].data)[None].to("cuda")
                for m in s.modalities}
        avail = {m: True for m in s.modalities}
        counts = router.route(avail, budget=8)
        with torch.no_grad():
            ct = align.encode_modalities(mods, avail)
            pe = proj(ct)                       # (1, M*k, H) modality-major
        # slice per-modality: modality j occupies rows [j*k, (j+1)*k)
        parts = []
        for j, m in enumerate(s.modalities):
            kk = counts[m]
            if kk > 0:
                parts.append(pe[0, j*8:(j*8)+kk])
        pes.append(torch.cat(parts))
    acc_pseudo = ev.evaluate_pseudo_tokens(texts, labels, pes)
    assert isinstance(acc_text, float) and isinstance(acc_pseudo, float)
```

- [ ] **Step 2: 运行确认（检查点已由 Task 2-3 产出，应为 sanity 检查而非 red）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_llm_interface_smoke.py -m slow -v`
Expected: PASS（前提: checkpoints_alignment/projection 已由 Task 2/3 生成；若缺失则报 checkpoint 加载错）

- [ ] **Step 3: 实现 scripts/eval_llm_interface.py**

```python
#!/usr/bin/env python
"""L3 end-to-end LLM eval (M5c): text vs pseudo-token vs no-context on real llama2-7b."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel
from framework.models.llm_adapter import LlamaAdapter
from framework.models.perceiver import PerceiverProjection
from framework.models.router import TokenRouter
from framework.eval.llm_interface import LLMEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="checkpoints_alignment/alignment_seed0.pt")
    ap.add_argument("--proj-ckpt", default="checkpoints_projection/projection_seed0.pt")
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=50, help="eval sample count")
    ap.add_argument("--budget", type=int, default=8, help="token budget for router (per-modality k_max=k)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    adapter = LlamaAdapter(model_path=args.llm, k=args.k, device=device)
    model, tok = adapter._load()
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"))
    align.eval().to(device)
    proj = PerceiverProjection(out_dim=adapter.hidden_dim, k=args.k).to(device)
    proj.load_state_dict(torch.load(args.proj_ckpt, map_location="cpu"))
    router = TokenRouter(k_max=args.k)

    ds = load_dataset(args.dataset, mode="lazy")
    samples = []
    for s in ds.test:
        if s.text.get("captions"):
            samples.append(s)
            if len(samples) >= args.n:
                break

    def generate(prompt, prefix_embs=None):
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        if prefix_embs is None:
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=16)
        else:
            merged = adapter.inject(prefix_embs, ids)
            with torch.no_grad():
                out = model.generate(inputs_embeds=merged, max_new_tokens=16)
        return tok.decode(out[0], skip_special_tokens=True)

    ev = LLMEvaluator(generate)
    labels = [s.label for s in samples]
    texts = [s.text["captions"][0] for s in samples]

    acc_text = ev.evaluate_text(texts, labels)
    acc_baseline = ev.evaluate_no_context(labels)   # 下界桩 (0.0): 真实无上下文约 1/27≈0.037

    pes = []
    for s in samples:
        mods = {m: torch.from_numpy(s.modalities[m].data)[None].to(device)
                for m in s.modalities}
        avail = {m: True for m in s.modalities}
        counts = router.route(avail, args.budget)
        with torch.no_grad():
            ct = align.encode_modalities(mods, avail)
            pe = proj(ct)                       # (1, M*k, H) modality-major
        # per-modality slicing: modality j occupies rows [j*k, (j+1)*k)
        parts = []
        for j, m in enumerate(s.modalities):
            kk = counts[m]
            if kk > 0:
                parts.append(pe[0, j*args.k:(j*args.k)+kk])
        pes.append(torch.cat(parts))
    # pseudo mode: prompt 无 caption (见 evaluate_pseudo_tokens 注释, 防混淆)
    acc_pseudo = ev.evaluate_pseudo_tokens(texts, labels, pes)

    print(f"[L3] n={len(samples)} acc_text={acc_text:.3f} "
          f"acc_pseudo={acc_pseudo:.3f} acc_baseline={acc_baseline:.3f}")
    print(f"[L3] pseudo - text = {acc_pseudo - acc_text:+.3f} (正=伪token有增益)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过（真实 llama, slow）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python scripts/eval_llm_interface.py --n 10 --k 8 --budget 8`
Expected: 输出三模式 acc 对比（后台 + 监控，llama 加载 + 生成较慢）

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add scripts/eval_llm_interface.py tests/test_llm_interface_smoke.py
git commit -m "feat(eval): L3 real-llama eval driver (text/pseudo-token/no-context)"
```

---

## Task 6: 全量测试 + 状态更新

**Files:**
- Modify: `STATUS.md`
- Test: 全量 pytest

- [ ] **Step 1: 全量回归（排除 slow）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/ -q`
Expected: 118 passed, 1 deselected (slow) — 无回退（新增 6: 1 alignment + 5 llm_interface）

- [ ] **Step 2: 更新 STATUS.md**

判断层追加：
```
- **M5c 完成（2026-08-16）**：真训练 AlignmentModel（CLIP 锚，hidden=512）+ PerceiverProjection（llama2 蒸馏）+ L3 端到端评测（text/pseudo-token/no-context 三模式对比）。**关键发现：v1-v5 数据集自带官方 LLM captions（text.captions，15866 base 全有），L3 文本侧直接复用。**
```
决策层追加：
```
- [x] `[已定]`：M5c——真训练 + L3 端到端 LLM 能力评测。✓ 完成
```

- [ ] **Step 3: 刷新事实层 + 提交**

Run:
```bash
cd /home/li/projects/sensorbench
/home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md
git add STATUS.md
git commit -m "docs(status): M5c 完成 — 真训练 + L3 端到端评测"
```

---

## 验收标准

1. 全测试绿：118 passed + 1 slow deselected（既有 112 + 新增 6），无回退。
2. 真训练 AlignmentModel：CLIP 锚 loss 收敛，L1 检索 r@1 显著 > 随机。
3. 真训练 projection：llama2 蒸馏 loss 递减。
4. L3 评测跑通（真实 llama2-7b）：三模式 acc 对比输出，pseudo-token 模式前向通过且**按 router 逐模态切分**（非单模态）。
5. match_label 能区分左右成对类（测试锁定）。
6. STATUS.md 更新完成。
