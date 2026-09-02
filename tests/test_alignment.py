# tests/test_alignment.py
import numpy as np
import torch
from framework.models.encoders import WifiEncoder, DepthEncoder, PointEncoder

def test_encoders_output_token_sequence():
    # each encoder already outputs (B, N_TOK, D) token sequence
    wifi = WifiEncoder()
    x = torch.zeros(2, 5, 3, 114, 10)  # (B,T,...)
    out = wifi(x)
    assert out.shape == (2, 16, 256), out.shape

def test_point_encoder_output_token_sequence():
    pe = PointEncoder(3)
    x = torch.zeros(2, 5, 1536, 3)
    out = pe(x)
    assert out.shape == (2, 16, 256)


import torch
import pytest
from framework.models.alignment import AlignmentModel, info_nce_loss
from framework.models.encoders import WifiEncoder, DepthEncoder, PointEncoder

def _toy_mods():
    return {
        "wifi": torch.zeros(4, 5, 3, 114, 10),
        "depth": torch.zeros(4, 5, 1, 224, 224),
        "lidar": torch.zeros(4, 5, 1536, 3),
        "mmwave": torch.zeros(4, 5, 64, 5),
        "rgb": torch.zeros(4, 5, 17, 2),
    }

def test_alignment_model_shapes():
    m = AlignmentModel(num_modalities=5, text_dim=512)
    mods = _toy_mods()
    toks = m.encode_modalities(mods, avail={k: True for k in mods})
    assert toks.shape == (4, 5, 16, 256)  # (B, M, N_TOK, D)

def test_alignment_model_missing_modality_zero():
    m = AlignmentModel(num_modalities=5, text_dim=512)
    mods = _toy_mods()
    avail = {k: (k != "depth") for k in mods}  # depth missing
    toks = m.encode_modalities(mods, avail)
    assert toks.shape == (4, 5, 16, 256)

def test_projection_head_shape():
    m = AlignmentModel(num_modalities=5, text_dim=512)
    pooled = torch.randn(4, 256)
    proj = m.projection_head(pooled)
    assert proj.shape == (4, 512)

def test_info_nce_loss_value():
    # identical embeddings -> low loss; orthogonal -> high loss
    z = torch.randn(16, 128)
    text = z.clone()  # perfect positive pairs
    loss = info_nce_loss(z, text, temperature=0.1)
    assert loss.shape == () and loss.item() < 1.0
    z2 = torch.randn(16, 128)
    z2 = z2 / z2.norm(dim=-1, keepdim=True)
    t2 = torch.roll(z2, 1, dims=0)  # wrong pairing
    loss2 = info_nce_loss(z2, t2, temperature=0.1)
    assert loss2.item() > loss.item()


from framework.models.alignment import _masked_logits


def test_info_nce_label_aware_mask():
    # B=16, 2 类 × 8 → 每行可用负样本 = 8 ≥ min_negatives=8, 保底不触发, 全部 mask
    z = torch.randn(16, 128)
    t = z.clone()
    labels = torch.tensor([i // 8 for i in range(16)])   # 8×label0, 8×label1
    z_n = torch.nn.functional.normalize(z, dim=-1)
    t_n = torch.nn.functional.normalize(t, dim=-1)
    logits = z_n @ t_n.t() / 0.07
    masked_logits = _masked_logits(logits, labels)
    # 对角线必须保留 (正样本)
    assert torch.isfinite(torch.diag(masked_logits)).all()
    # 同 label 非对角 (-inf) 被排除 (8 个同 label, 减自身=7 个被 mask)
    for i in range(16):
        for j in range(16):
            if i != j and labels[i] == labels[j]:
                assert masked_logits[i, j] == float("-inf")
    # 跨 label 保留
    for i in range(16):
        for j in range(16):
            if i != j and labels[i] != labels[j]:
                assert torch.isfinite(masked_logits[i, j])


def test_info_nce_label_aware_min_negatives():
    # 每行负样本数 < min_negatives 时不 mask (保底)
    z = torch.randn(4, 128)
    t = z.clone()
    labels = torch.tensor([0, 0, 0, 0])  # 每行同 label 排除后剩 0 负样本
    masked = _masked_logits(z @ t.t() / 0.07, labels)
    # 全部保留 (保底触发, 退化为普通 InfoNCE)
    assert torch.isfinite(masked).all()


def test_alignment_aux_cls_forward():
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    assert m.classification_head is not None
    assert m.classification_head.in_features == 256
    assert m.classification_head.out_features == 27
    mods = _toy_mods()   # 复用文件内已有 helper (4 样本)
    avail = {k: True for k in mods}
    text = torch.randn(4, 512)
    labels = torch.randint(0, 27, (4,))
    info_nce, ce = m.forward_loss(mods, text, avail, labels=labels, neg_mine=False)
    assert info_nce.shape == () and torch.isfinite(info_nce)
    assert ce.shape == () and torch.isfinite(ce)


def test_alignment_aux_cls_forward_neg_mine():
    # neg_mine=True 时 info_nce 用 labels 排除同 label 负样本, 仍返回 (info_nce, ce)
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    mods = _toy_mods()
    avail = {k: True for k in mods}
    labels = torch.tensor([0, 0, 1, 1])
    info_nce, ce = m.forward_loss(mods, torch.randn(4, 512), avail,
                                  labels=labels, neg_mine=True)
    assert info_nce.shape == () and torch.isfinite(info_nce)
    assert ce.shape == () and torch.isfinite(ce)


def test_alignment_no_classification_head():
    m = AlignmentModel(num_modalities=5, text_dim=512)  # 默认无分类头
    assert m.classification_head is None
    mods = _toy_mods()
    avail = {k: True for k in mods}
    info_nce, ce = m.forward_loss(mods, torch.randn(4, 512), avail)
    assert ce is None and info_nce.shape == ()


from framework.models.text_encoder import TextEncoder, HashTextEncoder

def test_hash_text_encoder_deterministic():
    a = HashTextEncoder(dim=512)
    t1 = a.encode(["a person is stretching and relaxing"])
    t2 = a.encode(["a person is stretching and relaxing"])
    assert torch.allclose(t1, t2)

def test_hash_text_encoder_shape():
    a = HashTextEncoder(dim=512)
    t = a.encode(["a person is stretching", "someone is waving"])
    assert t.shape == (2, 512)

def test_abstract_requires_encode():
    with pytest.raises(TypeError):
        TextEncoder()


from framework.models.text_encoder import CLIPTextEncoder

def test_clip_text_encoder_local_path_shape():
    te = CLIPTextEncoder(model_name="/home/li/datasets/models/clip-vit-base-patch32", device="cpu")
    assert te.dim == 512
    embs = te.encode(["a person is stretching and relaxing", "someone is waving"])
    assert embs.shape == (2, 512)
    assert abs(float(embs.norm(dim=-1).mean()) - 1.0) < 0.01  # normalized


def test_encoder_warmstart_from_token_fusion():
    """分类预热: AlignmentModel 的 encoders.* 可从 token_fusion checkpoint 加载."""
    import torch
    from framework.models.token_fusion import TokenFusionModel
    from framework.models.alignment import AlignmentModel
    tf = TokenFusionModel(num_classes=27)
    am = AlignmentModel(num_modalities=5, text_dim=512)
    src = {k: v for k, v in tf.state_dict().items() if k.startswith("encoders.")}
    missing, unexpected = am.load_state_dict(src, strict=False)
    assert len(unexpected) == 0            # 无多余键
    assert all("encoders." in k or "projection" in k for k in missing)  # 只缺投影头
    assert torch.equal(tf.encoders["wifi"].conv[0].weight,
                       am.encoders["wifi"].conv[0].weight)


def test_prototype_init_projection():
    """原型初始化投影头: 分类头(256→27) + 27原型→CLIP文本, 使正确类 sim > 平均."""
    import torch
    from framework.models.alignment import AlignmentModel
    from framework.models.token_fusion import TokenFusionModel
    from framework.models.text_encoder import CLIPTextEncoder
    from curation.caption.verbs import ACTION_PHRASES
    m = AlignmentModel(num_modalities=5, text_dim=512)
    tf = TokenFusionModel(num_classes=27)
    tf.load_state_dict(torch.load("checkpoints_v4/token_fusion_seed0.pt", map_location="cpu"))
    src = {k: v for k, v in tf.state_dict().items() if k.startswith("encoders.")}
    m.load_state_dict(src, strict=False)
    # 构造原型投影头: Linear(256→27) 用分类头, 27→512 用 CLIP 原型
    te = CLIPTextEncoder(device="cpu")
    protos = te.encode(["a person is " + v for v in ACTION_PHRASES.values()])  # (27,512)
    head = torch.nn.Sequential(
        torch.nn.Linear(256, 27),
        torch.nn.Linear(27, 512))
    with torch.no_grad():
        head[0].weight.copy_(tf.head.weight)
        head[0].bias.copy_(tf.head.bias)
        head[1].weight.copy_(protos.t())   # Linear(27,512) weight=(512,27)
        head[1].bias.zero_()
    m.projection_head = head
    # 验证: 分类 acc 高的特征, 投影后应靠近其类别原型
    from framework.dataset.loader import load_dataset
    ds = load_dataset("datasets/mmfi/v5", mode="lazy")
    batch = ds.train[0:16]
    toks = m.encode_modalities(
        {mm: torch.stack([torch.from_numpy(s.modalities[mm].data) for s in batch])
         for mm in ("wifi", "depth", "lidar", "mmwave", "rgb")},
        {mm: True for mm in ("wifi", "depth", "lidar", "mmwave", "rgb")})
    z = m.projection_head(m.pool(toks))
    zn = torch.nn.functional.normalize(z, dim=-1)
    pn = torch.nn.functional.normalize(protos, dim=-1)
    sim = zn @ pn.t()  # (16, 27)
    correct = [sim[i, batch[i].label].item() for i in range(16)]
    mean = sim.mean(dim=1).tolist()
    gap = sum(correct) / 16 - sum(mean) / 16
    assert gap > 0.05, f"原型初始化后正确类 sim 应高于平均 (gap={gap:.3f})"


def test_clip_distinguishes_semantics():
    """CLIP 文本锚必须能区分不同语义 (回归: 曾因错误 pooling 全 sim=1.0)."""
    import torch
    from framework.models.text_encoder import CLIPTextEncoder
    te = CLIPTextEncoder(device="cpu")
    emb = te.encode(["a photo of a cat", "a photo of a dog", "a photo of a car"])
    sim = emb @ emb.t()
    # 猫vs狗 (同域) 应 < 猫vs车 或至少可区分
    assert sim[0, 1] < 0.95, f"CLIP 应区分猫和狗 (sim={sim[0,1]:.3f})"
    # 所有不同语义句子应 < 1.0
    off = sim[~torch.eye(3, dtype=bool)]
    assert off.max() < 1.0, f"CLIP 不同语义 sim 应 < 1.0 (got {off.max():.3f})"
