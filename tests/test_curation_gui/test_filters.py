from __future__ import annotations

from curation.gui.core.filters import FilterSpec, apply_filters
from curation.gui.core.prediction_loader import load_predictions


class _FakeSample:
    def __init__(self, sid, label):
        self.id = sid
        self.label = label


def _lookup(samples):
    return lambda sid: samples.get(sid).label if sid in samples else None


def _fields_of(fields_map):
    return lambda sid: fields_map.get(sid, {})


IDS = ["E01_S01_A01_f1-3", "E01_S02_A05_f1-3", "E02_S33_A14_f1-3", "E02_S33_A14_f4-6"]
SAMPLES = {sid: _FakeSample(sid, i % 27) for i, sid in enumerate(IDS)}
PREDS = {
    "E01_S01_A01_f1-3": {"pred": 0, "conf": 0.9},
    "E01_S02_A05_f1-3": {"pred": 6, "conf": 0.5},  # wrong (label 1)
    "E02_S33_A14_f1-3": {"pred": 13, "conf": 0.8},
}
STATUSES = {s: "unreviewed" for s in IDS}
STATUSES["E01_S01_A01_f1-3"] = "golden"
STATUSES["E02_S33_A14_f4-6"] = "reject"
FIELDS = {"E02_S33_A14_f1-3": {"note": "check"}}
NO_FIELDS = {}


def test_no_filter_returns_all():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(NO_FIELDS), FilterSpec())
    assert set(out) == set(IDS)


def test_filter_label():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(NO_FIELDS),
                        FilterSpec(label=2))
    assert set(out) == {"E02_S33_A14_f1-3"}


def test_filter_subject():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(NO_FIELDS),
                        FilterSpec(subject="S33"))
    assert set(out) == {"E02_S33_A14_f1-3", "E02_S33_A14_f4-6"}


def test_filter_status():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(NO_FIELDS),
                        FilterSpec(status="golden"))
    assert out == ["E01_S01_A01_f1-3"]


def test_filter_pred_wrong():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(NO_FIELDS),
                        FilterSpec(pred_wrong=True))
    # E01_S02_A05 (label 1, pred 6) and E02_S33_A14 (label 2, pred 13) are wrong
    assert set(out) == {"E01_S02_A05_f1-3", "E02_S33_A14_f1-3"}


def test_filter_pred_wrong_drops_missing_preds():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(NO_FIELDS),
                        FilterSpec(pred_wrong=False))
    # E01_S01_A01 (label 0, pred 0) correct; others wrong or missing prediction
    assert set(out) == {"E01_S01_A01_f1-3"}


def test_filter_has_note():
    out = apply_filters(IDS, _lookup(SAMPLES), PREDS, STATUSES, _fields_of(FIELDS),
                        FilterSpec(has_note=True))
    assert out == ["E02_S33_A14_f1-3"]


def test_missing_sample_skipped():
    samples = dict(SAMPLES)
    del samples["E01_S01_A01_f1-3"]
    # no-filter: ids are trusted (split_ids already filters to existing files)
    out = apply_filters(IDS, _lookup(samples), PREDS, STATUSES, _fields_of(NO_FIELDS), FilterSpec())
    assert "E01_S01_A01_f1-3" in out
    # label filter needs the sample -> missing one is dropped
    out = apply_filters(IDS, _lookup(samples), PREDS, STATUSES, _fields_of(NO_FIELDS),
                        FilterSpec(label=0))
    assert "E01_S01_A01_f1-3" not in out


def test_load_predictions(tmp_path):
    p = tmp_path / "preds.json"
    p.write_text('{"s1": {"pred": 3, "conf": 0.7, "source": "ckpt"}, "bad": "x", "s2": {"pred": "9"}}')
    out = load_predictions(str(p))
    assert out["s1"] == {"pred": 3, "conf": 0.7, "source": "ckpt"}
    assert "bad" not in out  # non-dict dropped
    assert "s2" not in out or out["s2"]["pred"] == 9  # coerce int


def test_load_predictions_missing_file():
    assert load_predictions(None) == {}
    assert load_predictions("/no/such/file.json") == {}