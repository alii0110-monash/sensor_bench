# tests/test_canonical.py
import numpy as np
import pytest
from framework.tokens.canonical import CanonicalToken

def _tok():
    data = np.random.randn(40, 4096).astype(np.float32)   # M=5, k=8
    return CanonicalToken(id="s0", label=0, data=data,
                          modality_order=["wifi","depth","lidar","mmwave","rgb"],
                          k=8, meta={"encoder_version": "v0"})

def test_canonical_shape_and_fields():
    t = _tok()
    assert t.data.shape == (40, 4096)
    assert t.data.dtype == np.float32
    assert t.k == 8 and t.label == 0
    assert t.meta["encoder_version"] == "v0"

def test_canonical_modality_alignment():
    # validate() 校验: data 维度 + 行数 == len(modality_order)*k
    t = _tok()
    assert t.validate() is None  # 合法不抛

def test_canonical_invalid_dim():
    t = _tok()
    t.data = np.random.randn(40, 512).astype(np.float32)   # 错维
    with pytest.raises(ValueError):
        t.validate()

def test_canonical_roundtrip(tmp_path):
    import pickle
    t = _tok()
    p = tmp_path / "t.pkl"
    with open(p, "wb") as f:
        pickle.dump(t, f)
    with open(p, "rb") as f:
        t2 = pickle.load(f)
    assert np.array_equal(t.data, t2.data)
    assert t.id == t2.id and t.k == t2.k
