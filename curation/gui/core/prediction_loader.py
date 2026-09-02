"""Optional per-sample prediction file loading.

File format (JSON): {sample_id: {"pred": int, "conf": float, "source": str}}
The GUI only reads these files; they are produced separately by
scripts/precompute_predictions.py.
"""
from __future__ import annotations

import json
import os
from typing import Dict


def load_predictions(path: Optional[str]) -> Dict[str, dict]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return {}
    out: Dict[str, dict] = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        try:
            out[str(k)] = {
                "pred": int(v["pred"]),
                "conf": float(v.get("conf", 0.0)),
                "source": str(v.get("source", "")),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out