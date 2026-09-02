# tests/test_sample.py
import numpy as np
import pytest
from framework.dataset.sample import Sample, Modality

def test_sample_roundtrip():
    mod = Modality(data=np.zeros((5, 3, 114, 10), dtype=np.float32),
                   frame_indices=[65, 74, 83, 92, 101], sample_rate=1000)
    s = Sample(id="x", label=2, modalities={"wifi": mod})
    assert s.modalities["wifi"].shape[0] == 5
    assert s.label == 2

def test_sample_rejects_mismatched_frames():
    with pytest.raises(ValueError):
        Modality(data=np.zeros((5, 3, 114, 10), dtype=np.float32),
                 frame_indices=[65, 66, 67], sample_rate=20)

def test_sample_rejects_empty_modalities():
    with pytest.raises(ValueError):
        Sample(id="x", label=0, modalities={})

def test_sample_requires_label_in_range():
    mod = Modality(data=np.zeros((5, 3, 1, 1), dtype=np.float32),
                   frame_indices=[1,2,3,4,5], sample_rate=1)
    with pytest.raises(ValueError):
        Sample(id="x", label=-1, modalities={"depth": mod})
