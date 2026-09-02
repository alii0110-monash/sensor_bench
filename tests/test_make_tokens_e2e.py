# tests/test_make_tokens_e2e.py
import json, os
import numpy as np
from framework.tokens.canonical import CanonicalToken

def test_make_tokens_mini(tmp_path):
    """mini v5 → 资产化 → 加载 → 检索跑通 (CPU, 用真实 checkpoint)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from make_tokens import make_tokens
    from framework.dataset.loader import load_dataset
    from framework.tokens.assets import load_tokens
    from framework.tokens.canonical import CANONICAL_DIM

    # 用真实 v5, 但只取前 2 个 train base 样本
    ds = load_dataset("datasets/mmfi/v5", mode="lazy")
    samples = []
    for s in ds.train:
        if "__aug" not in s.id:
            samples.append(s)
            if len(samples) >= 2:
                break
    out = make_tokens(samples, "checkpoints_alignment/alignment_seed0.pt",
                      "checkpoints_projection_verb/projection_seed0.pt",
                      str(tmp_path / "toks"), k=8, device="cpu")
    assert out["n_samples"] == 2
    loaded = load_tokens(str(tmp_path / "toks"))
    assert len(loaded) == 2
    for sid, t in loaded.items():
        assert t.data.shape == (40, CANONICAL_DIM)
        t.validate()
