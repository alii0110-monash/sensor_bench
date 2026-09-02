"""Semi-dynamic token router (spec M5b §87-92).

Deterministic heuristic: each available modality gets >=1 token; remaining
budget distributed up to k_max per modality; missing modality = 0. Extreme
budget (all 0) falls back to pure-text (text captions already stored).
"""
from __future__ import annotations
from typing import Dict


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
