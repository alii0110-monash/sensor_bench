"""Append-only JSONL edit log for dataset curation.

Each line is one event (edit / flag / rollback). Loading folds events per
sample_id into the latest effective state; a rollback event marks the
targeted revision as reverted so folding skips it. The log is append-only:
GUI edits never touch the underlying dataset pickle files.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Dict, List, Optional

EVENTS = ("edit", "flag", "rollback")
QUALITY_CHOICES = ("golden", "ok", "reject", "flagged")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class EditLog:
    """Append-only JSONL edit log with per-sample state folding."""

    def __init__(self, path: str):
        self.path = path
        self._events: List[dict] = []
        self._state: Dict[str, dict] = {}
        self._max_revs: Dict[str, int] = {}
        self.load()

    # ------------------------------------------------------------- IO
    def load(self) -> None:
        self._events = []
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate corrupt lines
                if ev.get("event") in EVENTS and ev.get("sample_id"):
                    self._events.append(ev)
        self._fold()

    def _fold(self) -> None:
        """Recompute latest effective state per sample_id from events.

        Rollbacks are resolved in a prepass (a rollback reverts the referenced
        revision, so edits before it must also be reconsidered), then surviving
        edits are folded in order.
        """
        reverted: Dict[str, set] = {}
        revs: Dict[str, int] = {}
        for ev in self._events:
            sid = ev["sample_id"]
            revs[sid] = revs.get(sid, 0) + 1
            if ev["event"] == "rollback":
                rset = reverted.setdefault(sid, set())
                if ev.get("rolled_back_rev") is not None:
                    rset.add(ev["rolled_back_rev"])
                rset.add(revs[sid])
        self._max_revs = revs

        state: Dict[str, dict] = {}
        revs = {}
        for ev in self._events:
            sid = ev["sample_id"]
            revs[sid] = revs.get(sid, 0) + 1
            if ev["event"] == "rollback":
                continue
            if revs[sid] in reverted.get(sid, ()):
                continue  # this edit was reverted
            s = dict(state.get(sid, {"rev": 0, "ts": "", "fields": {}, "changes": {}}))
            s["rev"] = revs[sid]
            s["ts"] = ev.get("ts", "")
            fields = dict(s.get("fields", {}))
            for k, v in (ev.get("fields") or {}).items():
                if v is not None:
                    fields[k] = v
            s["fields"] = fields
            changes = dict(s.get("changes", {}))
            for k, v in (ev.get("changed") or {}).items():
                changes[k] = v
            s["changes"] = changes
            state[sid] = s
        self._state = state

    def _append(self, ev: dict) -> None:
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self._events.append(ev)
        self._fold()

    # ------------------------------------------------------------- API
    def next_rev(self, sample_id: str) -> int:
        return self._max_revs.get(sample_id, 0) + 1

    def status(self, sample_id: str) -> str:
        """unreviewed / edited / golden / ok / reject / flagged"""
        s = self._state.get(sample_id)
        if not s:
            return "unreviewed"
        q = s.get("fields", {}).get("quality")
        if q in QUALITY_CHOICES:
            return q
        if s.get("fields"):
            return "edited"
        return "unreviewed"

    def fields(self, sample_id: str) -> dict:
        return dict(self._state.get(sample_id, {}).get("fields", {}))

    def changes(self, sample_id: str) -> dict:
        return dict(self._state.get(sample_id, {}).get("changes", {}))

    def rev(self, sample_id: str) -> int:
        return self._state.get(sample_id, {}).get("rev", 0)

    def edited_ids(self) -> List[str]:
        return [sid for sid, s in self._state.items() if s.get("fields")]

    def save(self, sample_id: str, fields: dict, changed: Optional[dict] = None) -> dict:
        """Append an edit event. Returns the effective state for the sample."""
        ev = {
            "event": "edit",
            "sample_id": sample_id,
            "rev": self.next_rev(sample_id),
            "ts": _now(),
            "fields": fields,
            "changed": changed or {},
        }
        self._append(ev)
        return self._state[sample_id]

    def flag(self, sample_id: str, quality: str, note: Optional[str] = None) -> dict:
        """Mark a sample as reviewed without editing content fields."""
        fields = {"quality": quality}
        if note is not None:
            fields["note"] = note
        return self.save(sample_id, fields, changed={})

    def rollback(self, sample_id: str) -> Optional[dict]:
        """Revert the latest effective edit for a sample. Returns its new state."""
        s = self._state.get(sample_id)
        if not s:
            return None
        ev = {
            "event": "rollback",
            "sample_id": sample_id,
            "rev": self.next_rev(sample_id),
            "rolled_back_rev": s["rev"],
            "ts": _now(),
        }
        self._append(ev)
        return self._state.get(sample_id)