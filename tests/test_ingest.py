# tests/test_ingest.py
import numpy as np
from curation.ingest.mmfi import sample_frames, action_labels


def test_sample_frames_short_window_pads():
    assert sample_frames(1, 3) == [1, 2, 3, 3, 3]

def test_sample_frames_long_window_returns_5():
    idx = sample_frames(1, 40)
    assert len(idx) == 5 and idx[0] == 1 and idx[-1] == 40

def test_sample_frames_reproducible():
    assert sample_frames(5, 30) == sample_frames(5, 30)

def test_action_labels():
    assert action_labels()["A01"] == 0 and action_labels()["A27"] == 26
