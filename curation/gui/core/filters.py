"""Filtering helpers for the review page."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .dataset_service import parse_sample_id


@dataclass
class FilterSpec:
    label: Optional[int] = None
    subject: Optional[str] = None
    status: Optional[str] = None       # unreviewed/edited/golden/ok/reject/flagged
    pred_wrong: Optional[bool] = None  # True -> pred != label; False -> pred == label
    has_note: Optional[bool] = None


def apply_filters(ids: List[str],
                  label_of: Callable[[str], Optional[int]],
                  preds: Dict[str, dict],
                  statuses: Dict[str, str],
                  fields_lookup: Callable[[str], dict],
                  spec: FilterSpec) -> List[str]:
    """Filter sample ids. `label_of(sid)` returns the sample's label (or None
    if the sample cannot be loaded); `statuses[sid]` the review status;
    `fields_lookup(sid)` the edit-log fields.

    Sample data is ONLY loaded when a filter actually needs it (label or
    pred_wrong). Subject/status/note filters run on the id / edit log alone,
    so the common no-filter case never reads a pickle from disk."""
    needs_sample = spec.label is not None or spec.pred_wrong is not None
    out = []
    for sid in ids:
        if spec.status and statuses.get(sid, "unreviewed") != spec.status:
            continue
        if spec.subject is not None:
            p = parse_sample_id(sid)
            if not p or p["subject"] != spec.subject:
                continue
        if spec.has_note:
            if not (fields_lookup(sid).get("note") or "").strip():
                continue
        if needs_sample:
            label = label_of(sid)
            if label is None:
                continue
            if spec.label is not None and label != spec.label:
                continue
            if spec.pred_wrong is not None:
                pred = preds.get(sid)
                if pred is None:
                    continue
                if (pred["pred"] != label) != spec.pred_wrong:
                    continue
        out.append(sid)
    return out