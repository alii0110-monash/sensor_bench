# tests/test_align_check.py
import numpy as np
from curation.clean.align_check import verify_alignment
from framework.dataset.sample import Sample, Modality


def test_alignment_ok():
    mods = {"wifi": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 5]),
            "depth": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 5])}
    issues = verify_alignment(Sample(id="x", label=0, modalities=mods))
    assert issues == []

def test_alignment_mismatch():
    mods = {"wifi": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 5]),
            "lidar": Modality(np.zeros((5, 1), dtype=np.float32), [1, 2, 3, 4, 6])}
    issues = verify_alignment(Sample(id="x", label=0, modalities=mods))
    assert len(issues) == 1
    assert issues[0]["modality"] == "lidar"
