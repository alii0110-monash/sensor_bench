"""Persist 3D camera views (eye/up/center) to disk, keyed by sample id + modality.

The review page lets the reviewer set a fixed 3D camera for lidar/mmwave. By
default that lives only in session_state (lost on restart / sample switch).
This module persists it so the chosen view survives across sessions.

Storage: curation/gui/views/{dataset_name}.json  (gitignored)
  { "<sample_id>": { "lidar": {"eye":{...},"up":{...},"center":{...}}, ... } }
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "views")


def _path(dataset_name: str, views_dir: Optional[str] = None) -> str:
    d = views_dir or _DEFAULT_DIR
    return os.path.join(d, f"{dataset_name}.json")


def _load(dataset_name: str, views_dir: Optional[str] = None) -> Dict[str, dict]:
    p = _path(dataset_name, views_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(dataset_name: str, data: Dict[str, dict],
          views_dir: Optional[str] = None) -> None:
    p = _path(dataset_name, views_dir)
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_view(dataset_name: str, sample_id: str, modality: str,
              camera: dict, views_dir: Optional[str] = None) -> None:
    """Persist one modality's camera for a sample id."""
    data = _load(dataset_name, views_dir)
    data.setdefault(sample_id, {})[modality] = camera
    _save(dataset_name, data, views_dir)


def load_view(dataset_name: str, sample_id: str, modality: str,
              views_dir: Optional[str] = None) -> Optional[dict]:
    """Return the saved camera for (sample_id, modality) or None."""
    return _load(dataset_name, views_dir).get(sample_id, {}).get(modality)


def clear_view(dataset_name: str, sample_id: str, modality: str,
               views_dir: Optional[str] = None) -> None:
    """Remove the saved camera for (sample_id, modality)."""
    data = _load(dataset_name, views_dir)
    if sample_id in data and modality in data[sample_id]:
        del data[sample_id][modality]
        if not data[sample_id]:
            del data[sample_id]
        _save(dataset_name, data, views_dir)
