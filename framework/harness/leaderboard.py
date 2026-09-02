from __future__ import annotations
import json
import os
from typing import Dict, List


def robust_score(profile_results: List[dict]) -> float:
    return sum(r["accuracy"] for r in profile_results) / max(len(profile_results), 1)


def build_leaderboard(model_results: Dict[str, List[dict]]) -> dict:
    """model_results: {model: [{profile, available, accuracy, seed}, ...]}.
    Groups per-profile across seeds; reports mean + std + per-seed array
    (spec §6.2 requires mean ± std)."""
    lb = {}
    for model, results in model_results.items():
        by_profile = {}
        for r in results:
            by_profile.setdefault(r["profile"], []).append(r["accuracy"])
        profiles = {}
        for p, accs in by_profile.items():
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            profiles[p] = {"mean": round(mean, 4), "std": round(var ** 0.5, 4),
                           "per_seed": [round(a, 4) for a in accs]}
        full = profiles["full"]["mean"]
        rob_mean = sum(v["mean"] for v in profiles.values()) / len(profiles)
        rob_std = (sum(v["std"] ** 2 for v in profiles.values()) / len(profiles)) ** 0.5
        lb[model] = {
            "robustness": round(rob_mean, 4),
            "robustness_std": round(rob_std, 4),
            "acc_full": full,
            "profiles": profiles,
            "degradation": {p: round(full - v["mean"], 4) for p, v in profiles.items()},
        }
    return lb


def save_leaderboard(lb: dict, path: str, protocol: dict, dataset_root: str) -> None:
    out = {"protocol": protocol, "dataset": dataset_root,
           "leaderboard": lb}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
