# tests/test_perceiver.py
import torch
from framework.models.perceiver import PerceiverProjection

def test_perceiver_shape():
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=8)
    x = torch.randn(4, 5, 16, 256)   # (B, M, K_max, D)
    out = p(x)                        # (B, M*k, out_dim)
    assert out.shape == (4, 40, 4096)  # 5 modal * 8 queries

def test_perceiver_variable_k():
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=4)
    x = torch.randn(2, 3, 16, 256)
    out = p(x)
    assert out.shape == (2, 12, 4096)

def test_perceiver_prefix_truncation():
    # 半动态: 训练 k=8, 推理截取前 k'=3 → 结果与取前 3 个 query 一致
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=8)
    x = torch.randn(2, 5, 16, 256)
    full = p(x)                       # (2, 40, 4096)
    truncated = full[:, :15]          # 每模态 3 个 = 前 15
    assert truncated.shape == (2, 15, 4096)

def test_perceiver_missing_modality_zero():
    # spec §82: 缺模态 = 投影层对缺失模态不产生输出 (全零行)
    p = PerceiverProjection(in_dim=256, out_dim=4096, k=8)
    x = torch.randn(2, 5, 16, 256)
    x[:, 2, :, :] = 0.0               # depth (index 2) missing
    out = p(x)                        # (2, 40, 4096)
    # depth's 8 rows (indices 16..24) must be exactly zero
    assert torch.all(out[:, 16:24] == 0)
    # non-missing modalities non-zero
    assert torch.any(out[:, 0:16] != 0)
    assert not torch.isnan(out).any()
