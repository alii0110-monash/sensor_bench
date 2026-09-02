# tests/test_llm_proj.py
import numpy as np
import torch
from framework.tokens.canonical import CanonicalToken
from framework.tokens.llm_proj import TokenToLLM, LinearTokenToLLM

def _tok():
    return CanonicalToken(id="s0", label=0,
        data=np.random.randn(40, 4096).astype(np.float32),
        modality_order=["wifi","depth","lidar","mmwave","rgb"], k=8)

def test_linear_project_dim():
    proj = LinearTokenToLLM(llm_hidden=2048)
    ct = _tok()
    out = proj.project(ct)
    assert out.shape == (1, 40, 2048)   # (1, M*k, llm_hidden)

def test_linear_project_different_hidden():
    for h in (4096, 2048, 1024):
        proj = LinearTokenToLLM(llm_hidden=h)
        out = proj.project(_tok())
        assert out.shape == (1, 40, h)

def test_llm_hidden_property():
    assert LinearTokenToLLM(llm_hidden=2048).llm_hidden == 2048

def test_abstract_requires_project():
    import pytest
    with pytest.raises(TypeError):
        TokenToLLM()   # ABC
