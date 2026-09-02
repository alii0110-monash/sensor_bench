#!/usr/bin/env python3
"""弱项定位分析：per-sample 评测 → (profile × class × subject) 聚合矩阵。

用法:
    python scripts/weak_analysis.py --dataset datasets/mmfi/v2 \
      --protocol protocol.json --ckpt-dir checkpoints_v2 \
      --model late_fusion --seeds 0,1,2 --out docs/reports/weak_points_v2.json

输出 JSON:
    per_class[cls]    = {full, miss-mmwave, ..., deg}  每 profile 该类准确率 + 降幅
    per_subject[subj] = 同上
    top_weak_classes  = 按 full→miss-mmwave 降幅排序
    top_weak_subjects = 同上
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel

MODELS = {"token_fusion": TokenFusionModel, "late_fusion": LateFusionModel}


class BoundModel:
    """把 (model, available) 绑定的适配器：aggregate 无差别调用 predict_batch。"""

    def __init__(self, model, available: list):
        self.model = model
        self.available = available

    def predict_batch(self, samples):
        return self.model.predict_batch(samples, self.available)


def load_protocol(path: str) -> list:
    return json.load(open(path))["profiles"]


def _argmax(preds: dict) -> int:
    return max(preds, key=preds.get)


def aggregate(samples, models_by_profile, batch_size: int = 64) -> dict:
    """models_by_profile: {profile_id: predict_batch(samples) -> (B, C) logits}.
    对每个 profile 全样本预测, 按 class / subject 聚合准确率。"""
    per_class: dict = {}
    per_subject: dict = {}

    for pid, model in models_by_profile.items():
        correct_by_cls: dict = {}
        total_by_cls: dict = {}
        correct_by_subj: dict = {}
        total_by_subj: dict = {}
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            logits = model.predict_batch(batch)
            for j, s in enumerate(batch):
                pred = int(logits[j].argmax().item())
                cls = str(s.label)
                subj = str(s.meta.get("subject"))
                total_by_cls[cls] = total_by_cls.get(cls, 0) + 1
                total_by_subj[subj] = total_by_subj.get(subj, 0) + 1
                if pred == s.label:
                    correct_by_cls[cls] = correct_by_cls.get(cls, 0) + 1
                    correct_by_subj[subj] = correct_by_subj.get(subj, 0) + 1
        for cls, tot in total_by_cls.items():
            per_class.setdefault(cls, {})[pid] = correct_by_cls.get(cls, 0) / tot
        for subj, tot in total_by_subj.items():
            per_subject.setdefault(subj, {})[pid] = correct_by_subj.get(subj, 0) / tot

    for d in per_class.values():
        d["deg"] = round(d.get("full", 0) - d.get("miss-mmwave", 0), 4)
    for d in per_subject.values():
        d["deg"] = round(d.get("full", 0) - d.get("miss-mmwave", 0), 4)

    top_classes = sorted(per_class.items(), key=lambda kv: kv[1]["deg"], reverse=True)
    top_subjects = sorted(per_subject.items(), key=lambda kv: kv[1]["deg"], reverse=True)
    return {
        "per_class": per_class,
        "per_subject": per_subject,
        "top_weak_classes": [{"cls": c, **d} for c, d in top_classes],
        "top_weak_subjects": [{"subject": s, **d} for s, d in top_subjects],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--model", default="late_fusion", choices=list(MODELS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="weak_points.json")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    split = getattr(ds, args.split)
    profiles = load_protocol(args.protocol)
    seeds = [int(x) for x in args.seeds.split(",")]
    cls = MODELS[args.model]

    # seed0 模型做细粒度聚合（弱项定位用单个代表性 seed）
    sample_models = {}
    for pid in profiles:
        m = cls.load(f"{args.ckpt_dir}/{args.model}_seed0.pt")
        sample_models[pid["id"]] = BoundModel(m, pid["available"])
        # 多 seed 准确率只用于汇报
        accs = []
        for seed in seeds:
            mm = cls.load(f"{args.ckpt_dir}/{args.model}_seed{seed}.pt")
            bound = BoundModel(mm, pid["available"])
            ok = 0
            for i in range(0, len(split), 64):
                logits = bound.predict_batch(split[i:i + 64])
                ok += sum(int(logits[j].argmax().item()) == s.label
                          for j, s in enumerate(split[i:i + 64]))
            accs.append(ok / len(split))
        print(f"{pid['id']}: acc(mean)={sum(accs)/len(accs):.4f} seeds={[round(a,4) for a in accs]}")

    res = aggregate(split, sample_models)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n写入 {args.out}")
    print("\n=== top weak classes (full→miss-mmwave deg) ===")
    for r in res["top_weak_classes"][:10]:
        print(f"  class {r['cls']}: full={r.get('full'):.3f} miss-mmwave={r.get('miss-mmwave'):.3f} deg={r['deg']:.3f}")
    print("\n=== top weak subjects ===")
    for r in res["top_weak_subjects"][:5]:
        print(f"  {r['subject']}: full={r.get('full'):.3f} miss-mmwave={r.get('miss-mmwave'):.3f} deg={r['deg']:.3f}")


if __name__ == "__main__":
    main()
