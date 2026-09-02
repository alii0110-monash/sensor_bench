# 细粒度数据集质量矩阵（Fine-Grained Quality Matrix）实现计划

> **For agentic workers:** REQUIRED: 按本计划的 Task 顺序执行，每步用 checkbox 追踪。项目非 git 仓库，无 commit 步骤。

**Goal:** 为每个 (类别×受试者) 子组计算可跨版本追踪的质量分矩阵，输出结构化 JSON。

**Architecture:** 新增 `framework/eval/dataset_quality/finegrained.py`（独立于现有三维评分），从样本 id 解析 subject/env/action（排除 `__aug` 变体），按 (class,subject) 分组（621 格），每格算 3 信号（主模型识别 + 类内一致性 + 类间可分性），归一化后加权合成质量分。配套 `scripts/run_finegrained.py` CLI 与 `tools/compare_quality_matrix.py` 跨版本对比。

**Tech Stack:** Python 3.12, torch 2.4, numpy, pytest。复用现有 `feature_extract`、`compactness`、`token_fusion`。

**Spec:** `docs/superpowers/specs/2026-08-25-finegrained-quality-matrix-design.md`

---

## 文件结构

- 新建 `framework/eval/dataset_quality/finegrained.py` — 核心逻辑（分组/信号/合成/矩阵）
- 新建 `scripts/run_finegrained.py` — CLI 入口
- 新建 `tools/compare_quality_matrix.py` — 跨版本对比
- 新建 `tests/test_dataset_quality/test_finegrained.py` — 测试

---

### Task 1: id 解析 + 变体过滤

**Files:**
- Create: `framework/eval/dataset_quality/finegrained.py`
- Test: `tests/test_dataset_quality/test_finegrained.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dataset_quality/test_finegrained.py
import re
import numpy as np
import torch
import pytest

from framework.dataset.sample import Sample, Modality
from framework.eval.dataset_quality import finegrained


def _make_sample(sid, label):
    feats = np.zeros((5, 4), dtype=np.float32)
    mods = {"rgb": Modality(data=feats, frame_indices=[0,1,2,3,4])}
    return Sample(id=sid, label=label, modalities=mods)


def _make_sample_id(sid, label):
    """带真实 id 的样本（分组/信号用）。rgb 特征带 label 信号。"""
    feats = np.zeros((5, 4), dtype=np.float32)
    feats[:, 0] += float(label)  # 特征首维编码类别
    mods = {"rgb": Modality(data=feats, frame_indices=[0,1,2,3,4])}
    return Sample(id=sid, label=label, modalities=mods)


def _identity_feature(s):
    """恒等特征提取器（测试用）。"""
    return s.modalities["rgb"].data.reshape(-1).astype(np.float32)


def test_parse_sample_id():
    assert finegrained.parse_sample_id("E01_S01_A01_f1-7") == ("E01", "S01", "A01")


def test_is_variant():
    assert not finegrained.is_variant("E01_S01_A01_f1-7")
    assert finegrained.is_variant("E01_S01_A01_f105-110__aug1")


def test_group_by_class_subject_excludes_variants():
    samples = [
        _make_sample_id("E01_S01_A01_f1-7", 0),
        _make_sample_id("E01_S01_A01_f8-14", 0),
        _make_sample_id("E01_S01_A02_f1-7", 1),
        _make_sample_id("E01_S01_A01_f105-110__aug1", 0),  # 变体应排除
        _make_sample_id("E02_S02_A01_f1-7", 0),
    ]
    groups = finegrained.group_by_class_subject(samples)
    # 3 格：(A01,S01)=2、(A02,S01)=1、(A01,S02)=1
    assert len(groups) == 3
    assert len(groups[("A01", "S01")]) == 2  # 变体被排除
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /seu_share2/home/wangshuai02/220255046/sensorbench && .conda/envs/test/bin/python -m pytest tests/test_dataset_quality/test_finegrained.py -v`
Expected: FAIL (import finegrained 不存在)

- [ ] **Step 3: 实现解析 + 过滤 + 分组**

```python
# framework/eval/dataset_quality/finegrained.py
from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, Tuple

_ID_RE = re.compile(r"(E\d+)_(S\d+)_(A\d+)_")


def parse_sample_id(sid: str) -> Tuple[str, str, str]:
    m = _ID_RE.match(sid)
    if not m:
        raise ValueError(f"cannot parse sample id: {sid}")
    return m.group(1), m.group(2), m.group(3)  # env, subject, action


def is_variant(sid: str) -> bool:
    return "__aug" in sid


def group_by_class_subject(samples: List) -> Dict[Tuple[str, str], List]:
    groups = defaultdict(list)
    for s in samples:
        if is_variant(s.id):
            continue  # 排除变体
        env, subj, action = parse_sample_id(s.id)
        groups[(action, subj)].append(s)
    return dict(groups)
```

- [ ] **Step 4: 运行测试确认通过**
Expected: PASS (3 tests)

---

### Task 2: 每格信号计算（主模型识别 + 一致性 + 可分性）

**Files:**
- Modify: `framework/eval/dataset_quality/finegrained.py`

**设计**：信号1 用主模型 `predict_batch`（spec §三 信号1），信号2/3 用特征距离。为可测试性，`compute_cell_signals` 接受注入的 `main_model`（可 mock）。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 test_finegrained.py
class _FakeModel:
    """Mock 主模型：可编程预测 logits。"""
    def __init__(self, correct_ids):
        self.correct_ids = set(correct_ids)
    def predict_batch(self, samples, available):
        return torch.tensor(
            [[1.0, 0.0] if s.id in self.correct_ids else [0.0, 1.0] for s in samples])


def _extract_fn(s):
    return np.ones(4, dtype=np.float32)  # 特征恒为1（一致性=1）


def test_compute_cell_signals_main_model_used():
    cell = [_make_sample_id(f"E01_S01_A01_f{i}-{i+5}", 0) for i in range(5)]
    other_cls = [_make_sample_id(f"E01_S01_A02_f{i}", 1) for i in range(5)]  # 其他类
    model = _FakeModel(correct_ids={s.id for s in cell})  # 全部识别对
    sig = finegrained.compute_cell_signals(cell, [], other_cls, _extract_fn, main_model=model)
    assert sig["main_acc"] == pytest.approx(1.0, rel=1e-3)  # 全对 -> 1.0
    assert 0.0 <= sig["consistency"] <= 1.0
    assert 0.0 <= sig["separability"] <= 1.0


def test_compute_cell_signals_main_model_wrong():
    cell = [_make_sample_id(f"E01_S01_A01_f{i}", 0) for i in range(5)]
    model = _FakeModel(correct_ids=set())  # 全错
    sig = finegrained.compute_cell_signals(cell, [], [], _extract_fn, main_model=model)
    assert sig["main_acc"] == pytest.approx(0.0, abs=1e-3)
```

- [ ] **Step 2: 运行测试确认失败**
Expected: FAIL (compute_cell_signals 不存在)

- [ ] **Step 3: 实现信号计算**

```python
# framework/eval/dataset_quality/finegrained.py 追加
import numpy as np
import torch

from .compactness import compute_fisher_ratio


def _features(samples, extract_fn):
    return np.stack([extract_fn(s) for s in samples])


def _sigmoid_norm(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_cell_signals(cell, other_same_class, other_classes, extract_fn,
                         main_model=None) -> Dict[str, float]:
    """三信号，全部归一化到 0-1（spec §三）。

    信号1 main_acc：主模型识别该格为正确类的比例（spec §三 信号1）。
    信号2 consistency：格内 vs 同类其他受试者的余弦相似度（spec §三 信号2）。
    信号3 separability：格内 vs 其他类特征的距离（spec §三 信号3），
       复用 compactness.compute_fisher_ratio（类间/类内协方差比），sigmoid 归一化。
    """
    if main_model is not None:
        # 主模型识别：每个样本预测 argmax，统计正确比例
        # 注意：argmax 比较的是整数类别索引 s.label；分组 key 是字符串动作名 (A01)。
        # 两者独立但都对应同一样本，勿混淆。
        # 传显式模态列表（predict_batch 会遍历 available，不能传 None）
        available = ["wifi", "depth", "lidar", "mmwave", "rgb"]
        logits = main_model.predict_batch(cell, available)
        preds = logits.argmax(-1).cpu().tolist()
        labels = [s.label for s in cell]
        main_acc = float(sum(p == l for p, l in zip(preds, labels)) / max(len(cell), 1))
        conf = float(torch.softmax(logits, dim=-1).max(-1).values.mean())
    else:
        # 无主模型：退化为格内一致性（features 的类内紧凑度）。
        # 注意：spec §六提到 probe 兜底，但此处用特征 std 近似，是刻意简化（避免
        # 每格训 probe 的开销）；如需 probe 兜底可在 main_model 传一个线性 probe 适配器。
        X = _features(cell, extract_fn)
        main_acc = float(max(0.0, 1.0 - X.std(axis=0).mean())) if X.shape[0] > 1 else 1.0
        conf = main_acc

    X = _features(cell, extract_fn)
    if other_same_class:
        X_other = _features(other_same_class, extract_fn)
        sims = []
        for x in X:
            for y in X_other:
                denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
                sims.append(np.dot(x, y) / denom)
        consistency = float(np.clip(np.mean(sims), 0.0, 1.0)) if sims else 1.0
    else:
        consistency = 1.0

    # separability：格内 vs 其他类（类间/类内 fisher 比），sigmoid 归一化到 0-1
    if other_classes:
        X_cell = _features(cell, extract_fn)
        X_other_cls = _features(other_classes, extract_fn)
        y = np.concatenate([np.zeros(len(X_cell)), np.ones(len(X_other_cls))])
        X_all = np.concatenate([X_cell, X_other_cls], axis=0)
        fisher = compute_fisher_ratio(X_all, y)  # 类间/类内
        separability = float(_sigmoid_norm(np.log(max(fisher, 1e-8))))
    else:
        sep_in = X.std(axis=0).mean() if X.shape[0] > 1 else 0.0
        separability = float(np.clip(1.0 - _sigmoid_norm(sep_in), 0.0, 1.0))
    return {"main_acc": main_acc, "conf": conf,
            "consistency": consistency, "separability": separability}
```

- [ ] **Step 4: 运行测试确认通过**

---

### Task 3: 加权合成 + 完整矩阵构建

**Files:**
- Modify: `framework/eval/dataset_quality/finegrained.py`

- [ ] **Step 1: 写失败测试**

```python
def test_synthesize_quality_weighted():
    signals = {"main_acc": 0.8, "consistency": 0.6, "separability": 0.7}
    w = {"main": 0.4, "consistency": 0.3, "separability": 0.3}
    q = finegrained.synthesize_quality(signals, w)
    assert q == pytest.approx(0.4*0.8 + 0.3*0.6 + 0.3*0.7, rel=1e-3)
    assert 0.0 <= q <= 1.0


def test_build_matrix_structure():
    groups = {("A01","S01"): [_make_sample_id("E01_S01_A01_f1-7", 0)],
              ("A02","S01"): [_make_sample_id("E01_S01_A02_f1-7", 1)]}
    result = finegrained.build_matrix(groups, extract_fn=_identity_feature, weights=None)
    assert "global" in result and "matrix" in result and "low_quality" in result
    assert "A01_S01" in result["matrix"]
    assert "per_class" in result["global"] and "per_subject" in result["global"]
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现合成 + 矩阵构建**

```python
DEFAULT_WEIGHTS = {"main": 0.4, "consistency": 0.3, "separability": 0.3}


def synthesize_quality(signals, weights=None):
    w = weights or DEFAULT_WEIGHTS
    return float(w["main"] * signals["main_acc"]
                 + w["consistency"] * signals["consistency"]
                 + w["separability"] * signals["separability"])


def build_matrix(groups, extract_fn, weights=None, main_model=None,
                 min_cell=3, top_k=20):
    matrix, per_class, per_subject = {}, defaultdict(list), defaultdict(list)
    cell_env = {}  # key -> env (from id parse)
    for (cls, subj), cell in groups.items():
        n = len(cell)
        env = parse_sample_id(cell[0].id)[0]  # E01...
        cell_env[f"{cls}_{subj}"] = env
        other_same = [s for (c, s_), group in groups.items()
                      if c == cls and s_ != subj for s in group]
        other_classes = [s for (c, _), group in groups.items()
                         if c != cls for s in group]
        if n < min_cell:
            # 样本太少：标记 low_confidence，不参与全局聚合（spec §六）
            # quality=null 显式表示"不可靠"，区别于真正的低质量
            matrix[f"{cls}_{subj}"] = {"n": n, "quality": None, "low_confidence": True}
            continue
        signals = compute_cell_signals(cell, other_same, other_classes, extract_fn, main_model)
        q = synthesize_quality(signals, weights)
        matrix[f"{cls}_{subj}"] = {**signals, "n": n, "quality": q, "low_confidence": False}
        per_class[cls].append(q)
        per_subject[subj].append(q)
    conf_cells = [v for v in matrix.values() if not v.get("low_confidence")]
    global_q = float(np.mean([v["quality"] for v in conf_cells])) if conf_cells else 0.0
    low_quality = sorted([k for k in matrix if not matrix[k].get("low_confidence")],
                         key=lambda k: matrix[k]["quality"])[:top_k]
    per_env = defaultdict(list)
    for k, v in matrix.items():
        if not v.get("low_confidence"):
            per_env[cell_env[k]].append(v["quality"])
    return {
        "global": {"quality": global_q,
                   "per_class": {k: float(np.mean(v)) for k, v in per_class.items()},
                   "per_subject": {k: float(np.mean(v)) for k, v in per_subject.items()},
                   "per_env": {k: float(np.mean(v)) for k, v in per_env.items()}},
        "matrix": matrix,
        "low_quality": low_quality,
    }
```

- [ ] **Step 4: 运行测试确认通过**

---

### Task 4: CLI 入口

**Files:**
- Create: `scripts/run_finegrained.py`

- [ ] **Step 1: 实现 CLI**

```python
#!/usr/bin/env python
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.dataset.loader import load_dataset
from framework.eval.dataset_quality import finegrained
from framework.eval.dataset_quality.feature_extract import extract_structured_feature
from framework.models.token_fusion import TokenFusionModel


def _concat_features(s):
    """样本 → 拼接各模态结构化特征 (F,)"""
    return np.concatenate([extract_structured_feature(s, m) for m in
                           ["wifi", "depth", "lidar", "mmwave", "rgb"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="主模型 checkpoint（可选）")
    ap.add_argument("--eval-split", default="train")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-cell", type=int, default=3)
    ap.add_argument("--w-main", type=float, default=0.4)
    ap.add_argument("--w-consistency", type=float, default=0.3)
    ap.add_argument("--w-separability", type=float, default=0.3)
    args = ap.parse_args()
    ds = load_dataset(args.dataset, mode="lazy")
    samples = getattr(ds, args.eval_split)
    groups = finegrained.group_by_class_subject(samples)
    main_model = TokenFusionModel.load(args.ckpt) if args.ckpt else None
    weights = {"main": args.w_main, "consistency": args.w_consistency, "separability": args.w_separability}
    result = finegrained.build_matrix(groups, extract_fn=_concat_features,
                                      weights=weights, main_model=main_model,
                                      top_k=args.top_k, min_cell=args.min_cell)
    result["dataset"] = args.dataset
    # version 从 meta.json 读（spec §五），fallback 到数据集目录名
    import json as _json
    meta_path = os.path.join(args.dataset, "meta.json")
    result["version"] = "v" + _json.load(open(meta_path)).get("version", "")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟测试（本地小样本）**
Run: `python scripts/run_finegrained.py --dataset datasets/mmfi/v4 --out /tmp/matrix_test.json --top-k 5`
Expected: 生成 /tmp/matrix_test.json，含 matrix 和 low_quality

---

### Task 5: 跨版本对比工具

**Files:**
- Create: `tools/compare_quality_matrix.py`

- [ ] **Step 1: 实现对比**

```python
#!/usr/bin/env python
import argparse, json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base quality_matrix json")
    ap.add_argument("--new", required=True, help="new quality_matrix json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b = json.load(open(args.base)); n = json.load(open(args.new))
    common = set(b["matrix"]) & set(n["matrix"])
    diff = {}
    for k in common:
        dq = n["matrix"][k]["quality"] - b["matrix"][k]["quality"]
        diff[k] = dq
    result = {
        "base": args.base, "new": args.new,
        "improved": sorted([(k, v) for k, v in diff.items() if v > 0], key=lambda x: -x[1])[:20],
        "regressed": sorted([(k, v) for k, v in diff.items() if v < 0], key=lambda x: x[1])[:20],
        "mean_delta": float(sum(diff.values()) / len(diff)) if diff else 0.0,
    }
    json.dump(result, open(args.out, "w"), indent=2)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟**（用两个 toy JSON）

---

## 执行顺序与验证

1. Task 1→2→3→4→5 依次实现，每步 TDD（先测试后实现）
2. 全部测试：`python -m pytest tests/test_dataset_quality/ -v`
3. 真实数据冒烟：`python scripts/run_finegrained.py --dataset datasets/mmfi/v4 --out results/quality_matrix_v4.json`
