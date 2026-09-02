# tests/test_assets.py
import json, numpy as np
from framework.tokens.canonical import CanonicalToken
from framework.tokens.assets import write_tokens, load_tokens

def _toks(n=3):
    out = []
    for i in range(n):
        out.append(CanonicalToken(id=f"s{i}", label=i % 3,
            data=np.random.randn(40, 4096).astype(np.float32),
            modality_order=["wifi","depth","lidar","mmwave","rgb"], k=8,
            meta={"encoder_version": "v0"}))
    return out

def test_write_load_tokens(tmp_path):
    root = tmp_path / "tokens_root"
    toks = _toks()
    d0 = toks[0].data.copy()   # 捕获写入前数据 (避免 _toks() 每次随机)
    write_tokens(toks, str(root), version="v1", encoder_ckpt="ckpt0")
    # index.json
    idx = json.load(open(root / "index.json"))
    assert idx["version"] == "v1"
    assert idx["encoder_ckpt"] == "ckpt0"
    assert idx["n_samples"] == 3
    assert set(idx["samples"].keys()) == {"s0", "s1", "s2"}
    # npz 加载
    loaded = load_tokens(str(root))
    assert len(loaded) == 3
    assert np.array_equal(loaded["s0"].data, d0)   # 与写入前的数据一致
    assert loaded["s0"].label == 0

def test_load_tokens_missing_file(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    loaded = load_tokens(str(root))
    assert loaded == {}   # 空目录安全返回
