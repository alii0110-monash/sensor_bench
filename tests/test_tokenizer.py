# tests/test_tokenizer.py
import numpy as np
import pytest
from framework.tokens.canonical import CanonicalToken
from framework.tokens.tokenizer import CanonicalTokenizer

def test_tokenizer_encode_shape(tmp_path):
    """用真实 checkpoint 对 mini 样本编码 (CPU)."""
    import sys, os
    from framework.dataset.sample import Sample, Modality
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    mm = {
        "wifi": Modality(np.zeros((5,3,114,10), dtype=np.float32), [1,2,3,4,5], 10),
        "depth": Modality(np.zeros((5,1,224,224), dtype=np.float32), [1,2,3,4,5], 10),
        "lidar": Modality(np.zeros((5,1536,3), dtype=np.float32), [1,2,3,4,5], 10),
        "mmwave": Modality(np.zeros((5,64,5), dtype=np.float32), [1,2,3,4,5], 10),
        "rgb": Modality(np.zeros((5,17,2), dtype=np.float32), [1,2,3,4,5], 10),
    }
    s = Sample(id="s0", label=3, modalities=mm)
    tok = CanonicalTokenizer(align_ckpt="checkpoints_alignment/alignment_seed0.pt",
                             proj_ckpt="checkpoints_projection_verb/projection_seed0.pt",
                             k=8, device="cpu")
    ct = tok.encode(s)
    assert isinstance(ct, CanonicalToken)
    assert ct.data.shape == (40, 4096)   # 5 modal * 8
    assert ct.id == "s0" and ct.label == 3
    ct.validate()   # 通过校验

def test_tokenizer_decode_shape():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    tok = CanonicalTokenizer(align_ckpt="checkpoints_alignment/alignment_seed0.pt",
                             proj_ckpt="checkpoints_projection_verb/projection_seed0.pt",
                             k=8, device="cpu")
    ct = CanonicalToken(id="s0", label=0, data=np.random.randn(40,4096).astype(np.float32),
                        modality_order=["wifi","depth","lidar","mmwave","rgb"], k=8)
    out = tok.decode(ct)
    assert out.shape == (1, 40, 4096)   # (1, M*k, H)
