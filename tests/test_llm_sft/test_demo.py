"""Demo engine + playground page tests (FakeEngine, no model weights)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("SFTMVP_DEMO_FAKE", "1")


def test_fake_engine_answer_and_toggles():
    from framework.llm_sft.demo import FakeEngine
    from framework.models.alignment import MODALITIES
    eng = FakeEngine()
    sid = eng.sample_ids()[0]
    s = eng.get_sample(sid)
    full = eng.answer(s)
    assert full["label"] == 0 and "fake action" in full["text"]
    avail = {m: (m != "mmwave") for m in MODALITIES}
    degraded = eng.answer(s, avail=avail)
    assert "mmwave" in degraded["text"]


def test_fake_engine_sample_shapes():
    from framework.llm_sft.demo import FakeEngine
    from framework.models.alignment import MODALITIES
    s = FakeEngine().get_sample("FAKE_S01_A01_f1-5")
    assert set(s.modalities.keys()) == set(MODALITIES)
    for m in MODALITIES:
        assert s.modalities[m].data.ndim >= 2


def test_match_scores_top3():
    from framework.llm_sft.classmap import match_scores
    cm = {0: "stretching and relaxing", 1: "marking time in place"}
    sc = match_scores("the person bends arms for stretching and relaxation", cm)
    assert sc[0][0] == 0
    assert all(s2 <= s1 for (_, s1), (_, s2) in zip(sc, sc[1:]))


@pytest.mark.skipif(not os.path.isdir(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "curation")), reason="curation gui not present")
def test_playground_renders_with_fake_engine():
    from streamlit.testing.v1 import AppTest
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    page = os.path.join(root, "demo", "playground.py")
    at = AppTest.from_file(page, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("🛰")
    at.button(key="go").click().run()
    assert not at.exception
