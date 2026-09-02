from __future__ import annotations

import json
import os

import pytest

from curation.gui.core.edit_log import EditLog


def _tmp_log(tmp_path):
    return str(tmp_path / "edits.jsonl")


def test_save_fold_latest_state(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    log.save("s1", {"text": ["v1"], "quality": "ok"}, changed={"text": [[], ["v1"]]})
    log.save("s1", {"text": ["v2"], "label": 5}, changed={"label": [3, 5]})
    assert log.status("s1") == "ok"  # quality mark takes priority
    assert log.fields("s1") == {"text": ["v2"], "label": 5, "quality": "ok"}
    assert log.rev("s1") == 2
    assert log.edited_ids() == ["s1"]


def test_quality_status_priority(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    log.save("s1", {"text": ["a"]})
    assert log.status("s1") == "edited"
    log.save("s1", {"quality": "golden"})
    assert log.status("s1") == "golden"


def test_rollback(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    log.save("s1", {"text": ["v1"], "label": 3, "quality": "ok"})
    log.save("s1", {"text": ["v2"], "label": 5})
    log.rollback("s1")
    assert log.fields("s1") == {"text": ["v1"], "label": 3, "quality": "ok"}
    # rollback again -> reverts the first effective edit, leaving no fields
    log.rollback("s1")
    assert log.fields("s1") == {}


def test_rollback_rev_never_reused(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    log.save("s1", {"text": ["v1"]})
    log.rollback("s1")
    log.save("s1", {"text": ["v2"]})  # must NOT reuse rev 1
    assert log.rev("s1") == 3
    assert log.fields("s1") == {"text": ["v2"]}


def test_persistence_across_reload(tmp_path):
    path = _tmp_log(tmp_path)
    log = EditLog(path)
    log.save("s1", {"text": ["keep"], "quality": "golden"}, changed={"text": [[], ["keep"]]})
    log2 = EditLog(path)
    assert log2.fields("s1") == {"text": ["keep"], "quality": "golden"}
    assert log2.status("s1") == "golden"
    assert log2.rev("s1") == 1


def test_corrupt_lines_tolerated(tmp_path):
    path = _tmp_log(tmp_path)
    log = EditLog(path)
    log.save("s1", {"label": 2})
    with open(path, "a") as f:
        f.write("{not json\n")
    log2 = EditLog(path)
    assert log2.fields("s1") == {"label": 2}


def test_flag(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    log.flag("s1", "flagged", note="needs review")
    assert log.status("s1") == "flagged"
    assert log.fields("s1") == {"quality": "flagged", "note": "needs review"}
    assert log.changes("s1") == {}


def test_next_rev_starts_at_one(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    assert log.next_rev("s1") == 1


def test_status_unreviewed(tmp_path):
    log = EditLog(_tmp_log(tmp_path))
    assert log.status("nope") == "unreviewed"
    assert log.rev("nope") == 0