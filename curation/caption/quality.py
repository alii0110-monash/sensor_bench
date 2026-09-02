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
