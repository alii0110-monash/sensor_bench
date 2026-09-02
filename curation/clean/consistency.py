from __future__ import annotations
from typing import List
from framework.models.base import SensorModel


def flag_inconsistent(model: SensorModel, samples: List, drop_rate: float = 0.05) -> List[str]:
    """Use per-modality marginal predictions; flag samples where the max class
    under full-modality differs from single-best-modality predictions (top-1
    disagreement) as suspect. Keep the worst `drop_rate` fraction."""
    scored = []
    all_mods = sorted({m for s in samples for m in s.modalities})
    for s in samples:
        full = model.predict(s, all_mods)
        best = max(full, key=full.get)
        disagree = 0
        for m in all_mods:
            marg = model.predict(s, [m])
            if max(marg, key=marg.get) != best:
                disagree += 1
        scored.append((s.id, disagree / max(len(all_mods), 1)))
    scored.sort(key=lambda x: -x[1])
    n = int(len(scored) * drop_rate)
    return [sid for sid, _ in scored[:n]]
