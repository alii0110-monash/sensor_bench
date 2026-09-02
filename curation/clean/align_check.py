from __future__ import annotations
from typing import List
from framework.dataset.sample import Sample


def verify_alignment(sample: Sample) -> List[dict]:
    """Returns a list of issues; empty list means aligned.
    All modalities of a sample must reference the same frame_indices window."""
    if not sample.modalities:
        return []
    ref = None
    issues = []
    for name, mod in sample.modalities.items():
        if ref is None:
            ref = list(mod.frame_indices)
        elif list(mod.frame_indices) != ref:
            issues.append({"id": sample.id, "modality": name,
                           "expected": ref, "got": list(mod.frame_indices)})
    return issues
