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


# ---- 信号计算 ----

import numpy as np


def _sigmoid_norm(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_cell_signals(cell_feats, other_same_feats, other_class_feats) -> Dict[str, float]:
    """三信号，全部归一化到 0-1（spec §三），完全基于纯特征，不依赖任何主模型。

    `cell_feats`: (N, F) 该格特征矩阵（预提取）。
    `other_same_feats`: 同类其他受试者的特征矩阵。
    `other_class_feats`: 其他类样本的特征矩阵。

    信号1 compactness：格内特征紧凑度（离散度低 = 质量高），纯数据属性。
    信号2 consistency：格内 vs 同类其他受试者的余弦相似度。
    信号3 separability：格内 vs 其他类（质心距离/类内离散度）。
    """
    X = cell_feats
    cell_center = X.mean(axis=0)
    # 信号1：格内紧凑度 = 1 - 离散度（std），0-1
    compactness = float(max(0.0, 1.0 - X.std(axis=0).mean())) if X.shape[0] > 1 else 1.0

    # 信号2：与同类其他受试者的余弦相似度
    if other_same_feats is not None and len(other_same_feats):
        sims = []
        for x in X:
            for y in other_same_feats:
                denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
                sims.append(np.dot(x, y) / denom)
        consistency = float(np.clip(np.mean(sims), 0.0, 1.0)) if sims else 1.0
    else:
        consistency = 1.0

    # 信号3：格内 vs 其他类（质心距离/类内离散度），sigmoid 归一化到 0-1。
    # sep = (该格质心到其他类质心的距离) / (格内离散度 + 常数)，越大越可分。
    if other_class_feats is not None and len(other_class_feats):
        other_center = other_class_feats.mean(axis=0)
        between = float(np.linalg.norm(cell_center - other_center))
        within = float(np.mean(np.linalg.norm(X - cell_center, axis=1))) + 1e-8
        ratio = between / within
        separability = float(_sigmoid_norm(np.log(max(ratio, 1e-8))))
    else:
        sep_in = X.std(axis=0).mean() if X.shape[0] > 1 else 0.0
        separability = float(np.clip(1.0 - _sigmoid_norm(sep_in), 0.0, 1.0))
    return {"compactness": compactness,
            "consistency": consistency, "separability": separability}


# ---- 加权合成 + 矩阵构建 ----

DEFAULT_WEIGHTS = {"compactness": 0.4, "consistency": 0.3, "separability": 0.3}


def synthesize_quality(signals, weights=None):
    w = weights or DEFAULT_WEIGHTS
    return float(w["compactness"] * signals["compactness"]
                 + w["consistency"] * signals["consistency"]
                 + w["separability"] * signals["separability"])


def build_matrix(groups, extract_fn, weights=None, min_cell=3, top_k=20):
    """构建细粒度质量矩阵（纯特征，不依赖主模型）。

    关键性能优化：先对所有样本**预提取一次**特征并缓存，避免每个格子对
    other_classes（其余全部样本）重复提取（原实现 O(格数×样本数) 次提取）。
    """
    matrix, per_class, per_subject = {}, defaultdict(list), defaultdict(list)
    cell_env = {}  # key -> env (from id parse)

    # 预提取：样本 id -> 特征
    feat_cache = {}
    cell_sample_ids = {}  # cell_key -> list of sample ids
    for (cls, subj), cell in groups.items():
        ids = []
        for s in cell:
            if s.id not in feat_cache:
                feat_cache[s.id] = extract_fn(s)
            ids.append(s.id)
        cell_sample_ids[f"{cls}_{subj}"] = ids
        cell_env[f"{cls}_{subj}"] = parse_sample_id(cell[0].id)[0]

    for (cls, subj), cell in groups.items():
        key = f"{cls}_{subj}"
        n = len(cell)
        env = parse_sample_id(cell[0].id)[0]
        cell_env[key] = env
        if n < min_cell:
            matrix[key] = {"n": n, "quality": None, "low_confidence": True}
            continue
        # 该格特征矩阵
        X_cell = np.stack([feat_cache[s.id] for s in cell])
        # 同类其他受试者的特征（key 是 "A01_S01" 字符串，解析 cls/subj）
        same_ids = []
        for k, ids in cell_sample_ids.items():
            c_cls, c_subj = k.split("_", 1)
            if c_cls == cls and c_subj != subj:
                same_ids.extend(ids)
        X_same = np.array([feat_cache[i] for i in same_ids]) if same_ids else np.zeros((0, X_cell.shape[1]))
        # 其他类特征（预提取复用）
        other_ids = []
        for k, ids in cell_sample_ids.items():

            c_cls, _ = k.split("_", 1)
            if c_cls != cls:
                other_ids.extend(ids)
        X_other = np.array([feat_cache[i] for i in other_ids]) if other_ids else np.zeros((0, X_cell.shape[1]))

        signals = compute_cell_signals(
            X_cell, X_same if len(X_same) else None, X_other if len(X_other) else None)
        q = synthesize_quality(signals, weights)
        matrix[key] = {**signals, "n": n, "quality": q, "low_confidence": False}
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
