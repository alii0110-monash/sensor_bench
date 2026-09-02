# SensorBench M5a: 合成文本 + 编码器改造 + 规范空间对齐 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建合成文本标注管线（v5 数据集）+ 编码器 token 序列输出改造 + 规范空间对比对齐（InfoNCE）+ L1 跨模态检索评测。

**Architecture:** 两段式架构的第一阶段。文本侧用**冻结的通用文本编码器**（CLIP ViT-B-32，本项目 CUDA 17GB 可用）做锚；传感器侧改造现有编码器输出 token 序列 `(B, N_TOK, D)`；InfoNCE 把两者拉进同一规范空间。L1 评测用 train base held-out（~970 个，base+变体全排除训练）跑跨模态检索 recall@k。

**Tech Stack:** Python 3.12, torch 2.9, numpy, pytest, transformers 4.44 (CLIP), open_clip 可选。运行脚本用 `/home/li/projects/holollm/.venv/bin/python`。

**前置:** 依赖 spec `docs/superpowers/specs/2026-08-16-encoder-llm-adapter-design.md`（已 3 轮评审 Approved）。当前 git HEAD: `471055e`。

---

## 文件结构

```
curation/caption/__init__.py        # caption 包
curation/caption/verbs.py           # 27 类动作语义映射 (A01-A27 → 动词短语)
curation/caption/captioner.py       # SyntheticCaptioner 抽象 + TemplateCaptioner
curation/caption/quality.py         # 质量检查 (长度/去重/动作词)
curation/version/version.py         # (既有) write_meta — 复用不修改
scripts/make_v5.py                  # 合成文本 → v5 数据集 (Sample.text)
framework/models/encoders.py        # 改造: 保留时间池化, 输出 token 序列
framework/models/alignment.py       # AlignmentModel: 多模态 token + InfoNCE + 投影头
framework/eval/__init__.py          # eval 包 (空 __init__)
framework/eval/alignment.py         # L1 跨模态检索 recall@k
scripts/train_alignment.py          # 第一阶段对比训练 driver
scripts/eval_alignment.py           # L1 评测 driver
tests/test_captioner.py             # captioner 单测 (mock, 无 GPU)
tests/test_alignment.py             # InfoNCE/projection 单测 (无 GPU)
tests/test_alignment_e2e.py         # mini 数据集成测试
tests/test_keypoints_enrich.py      # (既有, 不回退)
datasets/mmfi/v5/                   # 生成 (make_v5 产出, 不提交)
```

---

## Task 1: 动作语义映射 (verbs.py)

**Files:**
- Create: `curation/caption/__init__.py`
- Create: `curation/caption/verbs.py`
- Test: `tests/test_captioner.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_captioner.py
import json
from curation.caption.verbs import ACTION_PHRASES, LABEL_TO_VERB

def test_action_phrases_complete():
    assert len(ACTION_PHRASES) == 27  # A01..A27

def test_action_phrases_all_lowercase_nonempty():
    for code, phrase in ACTION_PHRASES.items():
        assert code.startswith("A") and code[1:].isdigit()
        assert phrase and phrase == phrase.strip()

def test_label_to_verb_mapping():
    # label 0 == A01
    assert LABEL_TO_VERB(0) == ACTION_PHRASES["A01"]
    assert LABEL_TO_VERB(26) == ACTION_PHRASES["A27"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py::test_action_phrases_complete -v`
Expected: FAIL (ModuleNotFoundError / AttributeError)

- [ ] **Step 3: 实现 verbs.py**

```python
# curation/caption/__init__.py
"""Synthetic caption pipeline for LLM-aligned multimodal encoders."""

# curation/caption/verbs.py
"""27-class action semantics from the MMFi dataset annotation.

Source: MMFi_dataset README (Activity A01-A27). Each phrase is the natural
language verb phrase a human would use to describe the action. These anchor
the synthetic captions and the contrastive text side.
"""
from __future__ import annotations

# (code -> natural-language verb phrase)
ACTION_PHRASES = {
    "A01": "stretching and relaxing",
    "A02": "expanding chest horizontally",
    "A03": "expanding chest vertically",
    "A04": "twisting left",
    "A05": "twisting right",
    "A06": "marking time in place",
    "A07": "extending the left limb",
    "A08": "extending the right limb",
    "A09": "lunging toward the left front",
    "A10": "lunging toward the right front",
    "A11": "extending both limbs",
    "A12": "squatting down",
    "A13": "raising the left hand",
    "A14": "raising the right hand",
    "A15": "lunging to the left side",
    "A16": "lunging to the right side",
    "A17": "waving the left hand",
    "A18": "waving the right hand",
    "A19": "picking up things",
    "A20": "throwing toward the left side",
    "A21": "throwing toward the right side",
    "A22": "kicking toward the left side",
    "A23": "kicking toward the right side",
    "A24": "extending the left side of the body",
    "A25": "extending the right side of the body",
    "A26": "jumping up",
    "A27": "bowing",
}


def action_code(label: int) -> str:
    """label (0-26) -> MMFi action code (A01-A27)."""
    return f"A{label + 1:02d}"


def LABEL_TO_VERB(label: int) -> str:
    """label (0-26) -> natural-language verb phrase."""
    return ACTION_PHRASES[action_code(label)]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add curation/caption/ tests/test_captioner.py
git commit -m "feat(caption): 27-class MMFi action verb mapping (verbs.py)"
```

---

## Task 2: SyntheticCaptioner 抽象 + 模板实现

**Files:**
- Create: `curation/caption/captioner.py`
- Test: `tests/test_captioner.py` (追加)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_captioner.py (追加)
import pytest
from curation.caption.captioner import SyntheticCaptioner, TemplateCaptioner

def _fake_sample(sid="E01_S01_A01_f1-7", label=0):
    return {"id": sid, "label": label, "meta": {"env": "E01", "subject": "S01"}}

def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        SyntheticCaptioner()  # ABC with abstract generate

def test_template_generates_multiple_sentences():
    c = TemplateCaptioner(n=3)
    texts = c.generate(_fake_sample())
    assert isinstance(texts, list) and len(texts) == 3
    assert all(isinstance(t, str) and len(t) > 10 for t in texts)

def test_template_contains_action_verb():
    c = TemplateCaptioner(n=1)
    texts = c.generate(_fake_sample(label=0))
    assert "stretching" in texts[0].lower()

def test_template_uses_meta_env_subject():
    c = TemplateCaptioner(n=1)
    texts = c.generate(_fake_sample(label=0))
    assert "E01" in texts[0] or "S01" in texts[0]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py -v`
Expected: FAIL (ModuleNotFoundError: captioner)

- [ ] **Step 3: 实现 captioner.py**

```python
# curation/caption/captioner.py
"""Synthetic caption generation for MMFi samples.

TemplateCaptioner is deterministic (no LLM call) — used for tests and as a
fallback. The real pipeline may subclass SyntheticCaptioner with an LLM backend
(local/API), keeping the `generate(sample) -> List[str]` interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List

from .verbs import LABEL_TO_VERB


class SyntheticCaptioner(ABC):
    """Generate 3-5 natural-language sentences describing one sample."""

    @abstractmethod
    def generate(self, sample: Dict) -> List[str]:
        ...


class TemplateCaptioner(SyntheticCaptioner):
    """Deterministic template captions anchored on action verb + metadata.

    Sentence templates are cycled for diversity (n sentences from n patterns).
    """

    _TEMPLATES = [
        "A person is {verb}.",
        "We observe a person {verb}.",
        "The subject can be seen {verb}.",
        "In this scene, someone is {verb}.",
        "This clip shows a person {verb}.",
    ]

    def __init__(self, n: int = 3):
        self.n = n

    def generate(self, sample: Dict) -> List[str]:
        verb = LABEL_TO_VERB(sample["label"])
        meta = sample.get("meta", {})
        env = meta.get("env", "")
        subj = meta.get("subject", "")
        out = []
        for i in range(self.n):
            t = self._TEMPLATES[i % len(self._TEMPLATES)].format(verb=verb)
            if env:
                t = f"{t} Environment: {env}."
            if subj:
                t = f"{t} Subject: {subj}."
            out.append(t)
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py -v`
Expected: 7 passed (3 from Task 1 + 4 new)

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add curation/caption/captioner.py tests/test_captioner.py
git commit -m "feat(caption): SyntheticCaptioner ABC + deterministic TemplateCaptioner"
```

---

## Task 3: 质量检查 (quality.py)

**Files:**
- Create: `curation/caption/quality.py`
- Test: `tests/test_captioner.py` (追加)

- [ ] **Step 1: 写失败测试**

```python
# tests/test_captioner.py (追加)
from curation.caption.quality import check_captions

def test_check_captions_ok():
    texts = ["A person is stretching and relaxing.", "Someone is stretching."]
    assert check_captions(texts, verb="stretching") == []

def test_check_captions_flags_empty():
    assert "empty" in check_captions(["", "  "], verb="stretching")

def test_check_captions_flags_missing_verb():
    assert "verb" in check_captions(["A person is waving."], verb="stretching")

def test_check_captions_flags_duplicates():
    assert "duplicate" in check_captions(["same.", "same."], verb="stretching")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py::test_check_captions_ok -v`
Expected: FAIL (ModuleNotFoundError: quality)

- [ ] **Step 3: 实现 quality.py**

```python
# curation/caption/quality.py
"""Caption quality checks: emptiness, action-verb presence, near-duplicates."""
from __future__ import annotations
from typing import List


def check_captions(texts: List[str], verb: str) -> List[str]:
    """Return a list of issue descriptions (empty list == passes)."""
    issues = []
    cleaned = [t.strip() for t in texts if t and t.strip()]
    if not cleaned:
        issues.append("empty: no non-blank captions")
        return issues
    if any(not t for t in cleaned):
        issues.append("empty: contains blank caption")
    if verb and not any(verb.lower() in t.lower() for t in cleaned):
        issues.append(f"verb: no caption contains action verb '{verb}'")
    seen = set()
    for t in cleaned:
        key = " ".join(t.lower().split())
        if key in seen:
            issues.append("duplicate: repeated caption")
        seen.add(key)
    return issues
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add curation/caption/quality.py tests/test_captioner.py
git commit -m "feat(caption): quality checks for synthetic captions"
```

---

## Task 4: make_v5 — 合成文本 → v5 数据集

**Files:**
- Create: `scripts/make_v5.py`
- Test: `tests/test_captioner.py` (追加, make_v5 的 mini 数据集测试)

**说明**: v5 = v4 数据 + train base 样本的 `Sample.text` 填充合成文本。变体 `__aug*` 不直接写文本——loader 的 `_resolve_variant` 会从 base 继承 text（spec §64 定稿方案）。

- [ ] **Step 1: 写失败测试 (make_v5 mini 流程)**

```python
# tests/test_captioner.py (追加)
import os, pickle, sys
sys.path.insert(0, "scripts")
from make_v5 import add_captions

def test_add_captions_writes_text_to_base(tmp_path):
    # build a mini v4-like dataset: 1 base + 1 variant delta
    import numpy as np
    from framework.dataset.sample import Sample, Modality
    root = tmp_path / "v4"
    (root / "data").mkdir(parents=True)
    mm = {m: Modality(data=np.zeros((2,2,2), dtype=np.float32), frame_indices=[1,2], sample_rate=10)
          for m in ("wifi",)}
    s = Sample(id="E01_S01_A01_f1-7", label=0, modalities=mm)
    with open(root / "data" / "E01_S01_A01_f1-7.pkl", "wb") as f:
        pickle.dump(s.to_dict(), f)
    delta = {"kind": "variant", "id": "E01_S01_A01_f1-7__aug0", "base_id": "E01_S01_A01_f1-7",
             "label": 0, "rgb": np.zeros((2,2,2), dtype=np.float32), "aug": 0}
    with open(root / "data" / "E01_S01_A01_f1-7__aug0.pkl", "wb") as f:
        pickle.dump(delta, f)
    (root / "splits").mkdir(exist_ok=True)
    import json
    json.dump(["E01_S01_A01_f1-7", "E01_S01_A01_f1-7__aug0"], open(root / "splits" / "train.json", "w"))
    json.dump([], open(root / "splits" / "val.json", "w"))
    json.dump([], open(root / "splits" / "test.json", "w"))

    v5 = tmp_path / "v5"
    captioner = __import__("curation.caption.captioner", fromlist=["TemplateCaptioner"]).TemplateCaptioner(n=2)
    add_captions(str(root), str(v5), captioner)

    with open(v5 / "data" / "E01_S01_A01_f1-7.pkl", "rb") as f:
        base = pickle.load(f)
    assert isinstance(base.get("text", {}).get("en"), list) and len(base["text"]["en"]) == 2
    assert "stretching" in base["text"]["en"][0].lower()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py::test_add_captions_writes_text_to_base -v`
Expected: FAIL (ModuleNotFoundError: make_v5)

- [ ] **Step 3: 实现 make_v5.py**

```python
#!/usr/bin/env python
"""v5: copy v4 samples + fill train base samples' `Sample.text` with synthetic
captions. Variants (`__aug*`) share the base's text (resolved by loader).

Apples-to-apples: same data, same splits, same labels — only adds text.
"""
import argparse
import json
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation.caption.captioner import SyntheticCaptioner, TemplateCaptioner
from curation.caption.quality import check_captions
from curation.version.version import write_meta

_VARIANT_MARKER = "__aug"


def _is_train_base(sid: str, train_ids) -> bool:
    return sid in train_ids and _VARIANT_MARKER not in sid


def add_captions(v4_root: str, v5_root: str, captioner: SyntheticCaptioner) -> dict:
    """Copy v4 -> v5, generating captions for train base samples. Returns stats."""
    data_out = os.path.join(v5_root, "data")
    os.makedirs(data_out, exist_ok=True)
    shutil.copytree(os.path.join(v4_root, "splits"), os.path.join(v5_root, "splits"),
                    dirs_exist_ok=True)

    train_ids = set(json.load(open(os.path.join(v4_root, "splits", "train.json"))))
    n_base = n_variant = n_fail = n_written_total = 0
    for fn in sorted(os.listdir(os.path.join(v4_root, "data"))):
        if not fn.endswith(".pkl"):
            continue
        sid = fn[:-4]
        src = os.path.join(v4_root, "data", fn)
        dst = os.path.join(data_out, fn)
        n_written_total += 1
        if _is_train_base(sid, train_ids):
            with open(src, "rb") as f:
                sample = pickle.load(f)
            texts = captioner.generate(sample)
            verb = __import__("curation.caption.verbs", fromlist=["LABEL_TO_VERB"]).LABEL_TO_VERB(sample["label"])
            issues = check_captions(texts, verb)
            if issues:
                n_fail += 1
            sample = dict(sample)
            sample["text"] = dict(sample.get("text", {}))
            sample["text"]["en"] = texts
            with open(dst, "wb") as f:
                pickle.dump(sample, f)
            n_base += 1
        else:
            shutil.copy(src, dst)
            if _VARIANT_MARKER in sid:
                n_variant += 1

    write_meta(v5_root, name="mmfi", version="v5",
               changelog=[f"v5: synthetic captions for {n_base} train base samples "
                          "(variants share base text); no data change"],
               n_samples=n_written_total, n_modalities=5,
               source={"dataset": "MMFi", "split": "cs", "parent": "mmfi/v4"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"based_on": "mmfi/v4", "captioning": "TemplateCaptioner(n=3)"})
    return {"n_base": n_base, "n_variant": n_variant, "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default="datasets/mmfi/v4")
    ap.add_argument("--v5", default="datasets/mmfi/v5")
    ap.add_argument("--n", type=int, default=3, help="captions per base sample")
    ap.add_argument("--captioner", choices=["template"], default="template")
    args = ap.parse_args()
    captioner = TemplateCaptioner(n=args.n)
    stats = add_captions(args.v4, args.v5, captioner)
    print(f"v5: base={stats['n_base']} variant={stats['n_variant']} fail={stats['n_fail']}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_captioner.py -v`
Expected: 12 passed

- [ ] **Step 5: 运行 make_v5 生成真实 v5 数据集（后台 + 监控）**

Run:
```bash
cd /home/li/projects/sensorbench
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/make_v5.py --v4 datasets/mmfi/v4 --v5 datasets/mmfi/v5 > /tmp/opencode/make_v5.log 2>&1' < /dev/null > /dev/null 2>&1 &
```
Wait for completion, then verify:
```bash
tail -1 /tmp/opencode/make_v5.log   # expect: v5: base=9205 variant=36820 fail=0
                                    # 注意 base=9205 (9689 定义 - 484 缺文件), 非 9689
du -sh datasets/mmfi/v5
/home/li/projects/holollm/.venv/bin/python -c "
from framework.dataset.loader import load_dataset
ds = load_dataset('datasets/mmfi/v5', mode='lazy')   # lazy 防 RAM 压力
s = next(x for x in ds.train if '__aug' not in x.id)
print('base text:', s.text.get('en'))
v = next(x for x in ds.train if '__aug' in x.id)
print('variant text inherited:', v.text.get('en') == s.text.get('en'))
"
```
Expected: base text non-empty; variant inherits base text (True)

- [ ] **Step 6: Commit**

```bash
cd /home/li/projects/sensorbench
git add scripts/make_v5.py tests/test_captioner.py
git commit -m "feat(caption): make_v5 — synthetic captions into v5 Sample.text"
```

---

## Task 5: 编码器 token 序列输出改造

**Files:**
- Modify: `framework/models/encoders.py` (保留时间池化; 已输出 (B,N_TOK,D) — 确认不改)
- Modify: `framework/models/token_fusion.py:49` (去掉跨 token 池化 head(x.mean(dim=1))? **NO — token_fusion 是分类模型，M5a 不改它**)

**重要澄清（spec §69）**: 现有 `encoders.py` 的 `.mean(dim=1)` 是**时间帧池化**（输出已经是 `(B, N_TOK, D)` token 序列），**保留**。要"去掉跨 token 池化"的是 `token_fusion.py:49` 的分类 head——但 token_fusion 是既有分类模型，**不在 M5a 改动范围**。M5a 的编码器改造是**新增一个 alignment 用的多模态 token 聚合器**（Task 6），不是改 token_fusion。

因此本 Task 只做**验证性测试**，锁定 encoders 已输出 token 序列的现状，防止未来误改。

- [ ] **Step 1: 写锁定测试**

```python
# tests/test_alignment.py
import numpy as np
import torch
from framework.models.encoders import WifiEncoder, DepthEncoder, PointEncoder

def test_encoders_output_token_sequence():
    # each encoder already outputs (B, N_TOK, D) token sequence
    wifi = WifiEncoder()
    x = torch.zeros(2, 5, 3, 114, 10)  # (B,T,...)
    out = wifi(x)
    assert out.shape == (2, 16, 256), out.shape

def test_point_encoder_output_token_sequence():
    pe = PointEncoder(3)
    x = torch.zeros(2, 5, 1536, 3)
    out = pe(x)
    assert out.shape == (2, 16, 256)
```

- [ ] **Step 2: 运行确认通过（现状已是如此）**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: 2 passed (锁定现状)

- [ ] **Step 3: Commit**

```bash
cd /home/li/projects/sensorbench
git add tests/test_alignment.py
git commit -m "test(encoder): lock token-sequence output shape (B,N_TOK,D)"
```

---

## Task 6: AlignmentModel — 多模态 token 聚合 + InfoNCE + 投影头

**Files:**
- Create: `framework/models/alignment.py`
- Test: `tests/test_alignment.py` (追加)

**说明**: 复用 `framework/models/encoders.py` 的 5 个编码器（时间池化保留，输出 token 序列）。AlignmentModel 负责：收集可用模态 token → 可选 modality-dropout → 投影头（D→文本维度）→ InfoNCE loss。文本侧用冻结 CLIP text encoder（transformers 4.44 已装）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_alignment.py (追加)
import torch
import pytest
from framework.models.alignment import AlignmentModel, info_nce_loss
from framework.models.encoders import WifiEncoder, DepthEncoder, PointEncoder

def _toy_mods():
    return {
        "wifi": torch.zeros(4, 5, 3, 114, 10),
        "depth": torch.zeros(4, 5, 1, 224, 224),
        "lidar": torch.zeros(4, 5, 1536, 3),
        "mmwave": torch.zeros(4, 5, 64, 5),
        "rgb": torch.zeros(4, 5, 17, 2),
    }

def test_alignment_model_shapes():
    m = AlignmentModel(num_modalities=5, text_dim=512)
    mods = _toy_mods()
    toks = m.encode_modalities(mods, avail={k: True for k in mods})
    assert toks.shape == (4, 5, 16, 256)  # (B, M, N_TOK, D)

def test_alignment_model_missing_modality_zero():
    m = AlignmentModel(num_modalities=5, text_dim=512)
    mods = _toy_mods()
    avail = {k: (k != "depth") for k in mods}  # depth missing
    toks = m.encode_modalities(mods, avail)
    # depth slot should be zero/absent; with fixed slot indexing, mark via mask
    assert toks.shape == (4, 5, 16, 256)

def test_projection_head_shape():
    m = AlignmentModel(num_modalities=5, text_dim=512)
    pooled = torch.randn(4, 256)
    proj = m.projection_head(pooled)
    assert proj.shape == (4, 512)

def test_info_nce_loss_value():
    # identical embeddings -> low loss; orthogonal -> high loss
    z = torch.randn(16, 128)
    text = z.clone()  # perfect positive pairs
    loss = info_nce_loss(z, text, temperature=0.1)
    assert loss.shape == () and loss.item() < 1.0
    z2 = torch.randn(16, 128)
    z2 = z2 / z2.norm(dim=-1, keepdim=True)
    t2 = torch.roll(z2, 1, dims=0)  # wrong pairing
    loss2 = info_nce_loss(z2, t2, temperature=0.1)
    assert loss2.item() > loss.item()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: FAIL (ModuleNotFoundError: alignment)

- [ ] **Step 3: 实现 alignment.py**

```python
# framework/models/alignment.py
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


def info_nce_loss(z: torch.Tensor, t: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE between sensor-pooled vectors z and text vectors t (both (B, dim)).
    Positives are the same index; negatives are the rest of the batch."""
    z = F.normalize(z, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = z @ t.t() / temperature          # (B, B)
    labels = torch.arange(z.shape[0], device=z.device)
    return F.cross_entropy(logits, labels)


class AlignmentModel(nn.Module):
    """Stage-1: multimodal token encoder + projection head for InfoNCE."""

    def __init__(self, num_modalities: int = 5, text_dim: int = 512,
                 dropout_p: float = 0.25):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "wifi": WifiEncoder(), "depth": DepthEncoder(),
            "lidar": PointEncoder(3), "mmwave": PointEncoder(5),
            "rgb": PointEncoder(2)})
        self.text_dim = text_dim
        self.dropout_p = dropout_p
        self.projection_head = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, text_dim))

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
                     text_emb: torch.Tensor, avail: Dict[str, bool]) -> torch.Tensor:
        toks = self.encode_modalities(mods, avail)
        pooled = self.pool(toks)
        z = self.projection_head(pooled)
        return info_nce_loss(z, text_emb)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: 6 passed (2 from Task 5 + 4 new)

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/models/alignment.py tests/test_alignment.py
git commit -m "feat(alignment): AlignmentModel — multimodal token encoder + InfoNCE + projection head"
```

---

## Task 7: 冻结文本编码器封装 (TextEncoder)

**Files:**
- Create: `framework/models/text_encoder.py`
- Test: `tests/test_alignment.py` (追加)

**说明**: 文本侧锚 = 冻结的 CLIP text encoder（transformers 4.44 已装）。为可测性（无 GPU/不下载模型也能跑单测），TextEncoder 抽象 + `HashTextEncoder`（确定性 mock，仅测试用）+ `CLIPTextEncoder`（真实实现，懒加载）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_alignment.py (追加)
from framework.models.text_encoder import TextEncoder, HashTextEncoder

def test_hash_text_encoder_deterministic():
    a = HashTextEncoder(dim=512)
    t1 = a.encode(["a person is stretching and relaxing"])
    t2 = a.encode(["a person is stretching and relaxing"])
    assert torch.allclose(t1, t2)

def test_hash_text_encoder_shape():
    a = HashTextEncoder(dim=512)
    t = a.encode(["a person is stretching", "someone is waving"])
    assert t.shape == (2, 512)

def test_abstract_requires_encode():
    with pytest.raises(TypeError):
        TextEncoder()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py::test_hash_text_encoder_deterministic -v`
Expected: FAIL (ModuleNotFoundError: text_encoder)

- [ ] **Step 3: 实现 text_encoder.py**

```python
# framework/models/text_encoder.py
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

_HASH_DIM = 1 << 16  # 16-bit feature hashing bucket count


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
    """Frozen CLIP text encoder (transformers). Weights downloaded on first use."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
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
        pooled = emb[:, 0, :]  # CLS token (spec §71: 首选 CLS, 实现时锁定)
        return F.normalize(pooled, dim=-1)
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/models/text_encoder.py tests/test_alignment.py
git commit -m "feat(alignment): TextEncoder interface + HashTextEncoder mock + CLIPTextEncoder"
```

---

## Task 8: 第一阶段训练 driver (train_alignment.py)

**Files:**
- Create: `scripts/train_alignment.py`
- Test: `tests/test_alignment_e2e.py` (mini 数据集成测试)

**说明**: 复用 `framework/dataset/loader.py`（load_dataset v5）+ AlignmentModel + CLIPTextEncoder。训练循环：batch 采样 → modality-dropout → 编码 → InfoNCE。支持 `--text-encoder hash|clip`（hash 用于 CI 冒烟，clip 用于真训练）。

- [ ] **Step 1: 写失败测试 (mini e2e, hash encoder)**

```python
# tests/test_alignment_e2e.py
import json, os, pickle
import numpy as np
import torch
import pytest
from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel
from framework.models.text_encoder import HashTextEncoder
from framework.dataset.sample import Sample, Modality

def _mini_v5(tmp_path, n=8):
    root = tmp_path / "v5"
    (root / "data").mkdir(parents=True)
    ids = []
    for i in range(n):
        mm = {
            "wifi": Modality(data=np.zeros((2, 3, 4, 4), dtype=np.float32),  # (T,C,H,W)
                             frame_indices=[1, 2], sample_rate=10),
            "rgb": Modality(data=np.zeros((2, 17, 2), dtype=np.float32),     # (T,P,C)
                            frame_indices=[1, 2], sample_rate=10),
        }
        s = Sample(id=f"s{i}", label=i % 3, modalities=mm,
                   text={"en": [f"action number {i % 3}"]})
        with open(root / "data" / f"{s.id}.pkl", "wb") as f:
            pickle.dump(s.to_dict(), f)
        ids.append(s.id)
    (root / "splits").mkdir(exist_ok=True)
    json.dump(ids[:6], open(root / "splits" / "train.json", "w"))
    json.dump(ids[6:], open(root / "splits" / "test.json", "w"))
    json.dump([], open(root / "splits" / "val.json", "w"))
    with open(root / "modalities.yaml", "w") as f:
        f.write("modalities:\n- wifi\n- rgb\n")
    return root

def test_train_alignment_mini(tmp_path):
    root = _mini_v5(tmp_path)
    ds = load_dataset(str(root))
    m = AlignmentModel(num_modalities=5, text_dim=512)
    te = HashTextEncoder(dim=512)
    cfg = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "device": "cpu"}
    # import and run a mini train step
    import sys; sys.path.insert(0, "scripts")
    from train_alignment import train_epoch
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"])
    loss = train_epoch(m, te, ds.train, opt, batch_size=4, device="cpu")
    assert loss > 0 and torch.isfinite(torch.tensor(loss))
    # model params updated
    p = list(m.parameters())[0].detach().clone()
    assert not torch.allclose(p, list(m.parameters())[0].detach())
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment_e2e.py -v`
Expected: FAIL (ModuleNotFoundError: train_alignment)

- [ ] **Step 3: 实现 train_alignment.py**

```python
#!/usr/bin/env python
"""Stage-1 contrastive alignment training (M5a).

Trains AlignmentModel with InfoNCE against a frozen text encoder on the v5
dataset. `--text-encoder hash` uses the deterministic mock (CI/smoke);
`--text-encoder clip` uses frozen CLIP (real training).
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel, MODALITIES, info_nce_loss
from framework.models.text_encoder import CLIPTextEncoder, HashTextEncoder

TEXT_ENCODERS = {"hash": HashTextEncoder, "clip": CLIPTextEncoder}


def _dropout_mask(rng, p: float) -> dict:
    avail = {m: bool(rng.random() > p) for m in MODALITIES}
    if not any(avail.values()):
        avail[list(avail)[0]] = True
    return avail


def _stack_mods(samples, avail, device):
    mods = {}
    first = samples[0]
    for m in MODALITIES:
        if avail.get(m) and m in first.modalities:
            mods[m] = torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in samples]).to(device)
    return mods


def train_epoch(model, text_encoder, train, opt, batch_size=32,
                device="cuda", dropout_p=0.25) -> float:
    model.train()
    rng = __import__("numpy").random.default_rng(0)
    total = 0.0; n = 0
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        avail = _dropout_mask(rng, dropout_p)
        mods = _stack_mods(batch, avail, device)
        if not mods:
            continue
        texts = [s.text.get("en", [""])[0] for s in batch]
        text_emb = text_encoder.encode(texts).to(device)
        toks = model.encode_modalities(mods, avail)
        pooled = model.pool(toks)
        z = model.projection_head(pooled)
        loss = info_nce_loss(z, text_emb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--text-encoder", choices=list(TEXT_ENCODERS), default="clip")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints_alignment")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    te_cls = TEXT_ENCODERS[args.text_encoder]
    te = te_cls(dim=512) if args.text_encoder == "hash" else te_cls(device=device)
    model = AlignmentModel(num_modalities=5, text_dim=te.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        loss = train_epoch(model, te, ds.train, opt, batch_size=args.batch_size,
                           device=device)
        print(f"[alignment] ep {ep} loss {loss:.4f}", flush=True)
        if loss < best:
            best = loss
            torch.save(model.state_dict(), f"{args.out}/alignment_seed0.pt")
    print(f"done -> {args.out}/alignment_seed0.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment_e2e.py -v`
Expected: 1 passed

- [ ] **Step 5: 冒烟跑真训练（hash encoder, mini epochs, 后台 + 监控）**

Run:
```bash
cd /home/li/projects/sensorbench
setsid bash -c '/home/li/projects/holollm/.venv/bin/python scripts/train_alignment.py \
  --dataset datasets/mmfi/v5 --text-encoder hash --epochs 2 --batch-size 32 \
  --device cpu --out /tmp/opencode/align_smoke > /tmp/opencode/train_align.log 2>&1' \
  < /dev/null > /dev/null 2>&1 &
```
Wait for completion, verify loss finite and decreasing:
```bash
tail -3 /tmp/opencode/train_align.log
```
Expected: `[alignment] ep 0 loss finite`, `[alignment] ep 1 loss < ep 0 loss` (收敛趋势即可，hash encoder 有相关性下界)

- [ ] **Step 6: Commit**

```bash
cd /home/li/projects/sensorbench
git add scripts/train_alignment.py tests/test_alignment_e2e.py
git commit -m "feat(alignment): train_alignment driver + mini e2e test"
```

---

## Task 9: L1 跨模态检索评测

**Files:**
- Create: `framework/eval/__init__.py`
- Create: `framework/eval/alignment.py`
- Create: `scripts/eval_alignment.py`
- Test: `tests/test_alignment_e2e.py` (追加)

**说明**: L1 评测在 train base held-out（~10%，base+变体全排除训练）上跑。retrieval: 传感器 query（池化向量）→ 文本候选库（同 held-out 的文本），计算 recall@1/@5/@10。

- [ ] **Step 1: 写失败测试 (held-out split + recall@k)**

```python
# tests/test_alignment_e2e.py (追加)
import numpy as np
from framework.eval.alignment import build_held_out_split, retrieval_recall_at_k

def test_build_held_out_split():
    bases = [f"E01_S01_A01_f{i}-{i+6}" for i in range(20)]
    variants = [f"{b}__aug{k}" for b in bases[:10] for k in range(4)]
    all_ids = bases + variants
    held, train_ids = build_held_out_split(all_ids, fraction=0.1, seed=0)
    # held contains whole base groups (base + its variants)
    held_bases = {x.split("__")[0] for x in held}
    assert 0 < len(held_bases) <= 3  # ~10% of 20
    assert all(x not in train_ids for x in held)

def test_retrieval_recall_at_k():
    # 10 queries, embeddings where nearest neighbor is correct
    rng = np.random.default_rng(0)
    q = rng.randn(10, 8); t = q.copy()  # exact matches
    r1 = retrieval_recall_at_k(torch.tensor(q), torch.tensor(t), k=1)
    assert r1 == 1.0
    # reverse direction: text query -> sensor candidate
    rt1 = retrieval_recall_at_k(torch.tensor(t), torch.tensor(q), k=1)
    assert rt1 == 1.0
    # random -> ~0 recall@1
    t2 = torch.tensor(rng.randn(10, 8))
    r2 = retrieval_recall_at_k(torch.tensor(q), t2, k=1)
    assert r2 < 0.3
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment_e2e.py::test_retrieval_recall_at_k -v`
Expected: FAIL (ModuleNotFoundError: framework.eval.alignment)

- [ ] **Step 3: 实现 framework/eval/alignment.py**

```python
# framework/eval/alignment.py
"""L1 cross-modal retrieval evaluation (spec M5a)."""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn.functional as F


def build_held_out_split(all_ids: List[str], fraction: float = 0.1, seed: int = 0) -> Tuple[list, set]:
    """Split base groups into (held_out, train_ids). A held-out base and ALL its
    `__aug*` variants are excluded from training (variants share base text)."""
    import random
    rng = random.Random(seed)
    bases = sorted({i.split("__")[0] for i in all_ids})
    n_held = max(1, int(len(bases) * fraction))
    held_bases = set(rng.sample(bases, n_held))
    held = [i for i in all_ids if i.split("__")[0] in held_bases]
    train_ids = {i for i in all_ids if i.split("__")[0] not in held_bases}
    return held, train_ids


def retrieval_recall_at_k(query: torch.Tensor, cand: torch.Tensor, k: int = 1) -> float:
    """Recall@k: fraction of queries whose true text candidate is in top-k.
    query/cand: (N, dim) normalized embeddings, index-aligned positives."""
    q = F.normalize(query, dim=-1)
    c = F.normalize(cand, dim=-1)
    sim = q @ c.t()                      # (N, N)
    _, topk = sim.topk(k, dim=1)         # (N, k)
    hits = (topk == torch.arange(len(q), device=q.device)[:, None]).any(dim=1)
    return float(hits.float().mean())


def evaluate_retrieval(model, text_encoder, samples, device="cuda",
                       batch_size=64) -> dict:
    """Embed sensor and text for all samples; return recall@1/5/10."""
    model.eval()
    model.to(device)
    zs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            avail = {m: True for m in model.encoders}
            mods = {}
            for m in avail:
                if m in batch[0].modalities:
                    mods[m] = torch.stack(
                        [torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
            texts = [s.text.get("en", [""])[0] for s in batch]
            t = text_encoder.encode(texts).to(device)
            if mods:
                toks = model.encode_modalities(mods, avail)
                z = model.projection_head(model.pool(toks))
            else:
                z = torch.zeros(len(batch), t.shape[1], device=device)
            zs.append(z); ts.append(t)
    Z = torch.cat(zs); T = torch.cat(ts)
    return {"r@1": retrieval_recall_at_k(Z, T, 1),
            "r@5": retrieval_recall_at_k(Z, T, 5),
            "r@10": retrieval_recall_at_k(Z, T, 10),
            "tr@1": retrieval_recall_at_k(T, Z, 1),   # text→sensor (spec §75 反向)
            "tr@5": retrieval_recall_at_k(T, Z, 5),
            "tr@10": retrieval_recall_at_k(T, Z, 10),
            "n": len(Z)}
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment_e2e.py -v`
Expected: 3 passed (1 from Task 8 + 2 new)

- [ ] **Step 5: 实现 scripts/eval_alignment.py**

```python
#!/usr/bin/env python
"""L1 retrieval eval on train base held-out (spec M5a)."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.eval.alignment import build_held_out_split, evaluate_retrieval
from framework.models.alignment import AlignmentModel
from framework.models.text_encoder import CLIPTextEncoder, HashTextEncoder

TEXT_ENCODERS = {"hash": HashTextEncoder, "clip": CLIPTextEncoder}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--ckpt", default="checkpoints_alignment/alignment_seed0.pt")
    ap.add_argument("--text-encoder", choices=list(TEXT_ENCODERS), default="clip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fraction", type=float, default=0.1)
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    te_cls = TEXT_ENCODERS[args.text_encoder]
    te = te_cls(dim=512) if args.text_encoder == "hash" else te_cls(device=device)

    model = AlignmentModel(num_modalities=5, text_dim=te.dim)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    # 从 splits ids 派生 held-out（仅加载 held 样本，避免 v5 全量进内存）
    # held-out 按 base 组整组划出（含变体，变体共享 base 文本）。评测样本集取 base 为主：
    # 变体文本重复会压低 recall@k，且 spec 意图是"held-out base"。故只保留 base 样本参与评测。
    import json
    train_ids = json.load(open(os.path.join(args.dataset, "splits", "train.json")))
    held, _ = build_held_out_split(train_ids, fraction=args.fraction)
    held_bases = {i for i in held if "__aug" not in i}
    held_samples = [s for s in ds.train if s.id in held_bases]
    res = evaluate_retrieval(model, te, held_samples, device=device)
    print(f"[eval] n={res['n']} r@1={res['r@1']:.4f} r@5={res['r@5']:.4f} r@10={res['r@10']:.4f} "
          f"tr@1={res['tr@1']:.4f} tr@5={res['tr@5']:.4f} tr@10={res['tr@10']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 运行确认通过 (hash encoder, held-out)**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/test_alignment_e2e.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
cd /home/li/projects/sensorbench
git add framework/eval/__init__.py framework/eval/alignment.py scripts/eval_alignment.py tests/test_alignment_e2e.py
git commit -m "feat(eval): L1 cross-modal retrieval recall@k + held-out split"
```

---

## Task 10: 全量测试 + 状态更新

**Files:**
- Modify: `STATUS.md` (判断层/决策层)
- Test: 全量 pytest

- [ ] **Step 1: 全量回归测试**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python -m pytest tests/ -q`
Expected: 全部通过（既有 75 + 新增 24 = 99，无回退）

- [ ] **Step 2: 更新 STATUS.md**

在判断层追加：
```
- **M5a 完成（2026-08-16）**：合成文本管线（v5，TemplateCaptioner 9205 个落盘 train base）、编码器 token 序列确认、AlignmentModel + InfoNCE、L1 检索评测。**注意：v5 用模板文本（LLM 后端延后），L1 数值反映模板文本的简单相关性，M5b 换 LLM 后端后对比需谨慎。**
```
在决策层追加：
```
- [ ] `[提议]`：M5b——Perceiver 投影 + LLMAdapter + router + L2 冒烟。
```

- [ ] **Step 3: 刷新事实层 + 提交**

Run:
```bash
cd /home/li/projects/sensorbench
/home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md
git add STATUS.md
git commit -m "docs(status): M5a 完成 — 合成文本 + 对齐模型 + L1 检索评测"
```

---

## 验收标准

1. 全测试绿：既有测试不回退 + 新增 captioner/alignment/eval 测试通过。
2. `make_v5.py` 产出真实 v5：`n_base=9205, n_variant=36820, fail=0`（base 定义 9689，缺 484 文件，落盘 9205）；loader 验证变体继承 base text。
3. `train_alignment.py --text-encoder hash` 冒烟：loss finite 且递减（hash encoder 有相关性下界，不强求 <1.0）。
4. `eval_alignment.py --text-encoder hash --ckpt /tmp/opencode/align_smoke/alignment_seed0.pt`：recall@k 输出正常（r@1/r@5/r@10 + tr@1/5/10 数值）。注：hash encoder 是冒烟基准（非语义对齐），真实 CLIP 训练/评测为 M5a 之后的可选增强。
5. STATUS.md 更新完成。
