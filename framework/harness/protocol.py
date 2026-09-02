from __future__ import annotations
import itertools
from typing import List


def build_protocol(modalities: List[str], seeds: List[int]) -> dict:
    profiles = []
    profiles.append({"id": "full", "available": list(modalities)})
    for m in modalities:
        profiles.append({"id": f"miss-{m}", "available": [x for x in modalities if x != m]})
    for a, b in itertools.combinations(modalities, 2):
        profiles.append({"id": f"miss2-{a}-{b}",
                         "available": [x for x in modalities if x not in (a, b)]})
    for m in modalities:
        profiles.append({"id": f"only-{m}", "available": [m]})
    return {"modalities": modalities, "seeds": seeds, "profiles": profiles}
