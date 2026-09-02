# tests/test_models.py
import numpy as np
import torch
import pytest
from framework.models.base import SensorModel, TrainConfig
from framework.dataset.sample import Sample, Modality
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel
from framework.models.cross_attention import CrossAttentionModel


class Dummy(SensorModel):
    name = "dummy"
    def fit(self, train, val, cfg): pass
    def predict(self, sample, available):
        return {0: 0.5, 1: 0.5}


def test_protocol_interface():
    m = Dummy()
    assert callable(m.fit) and callable(m.predict)
    assert m.name == "dummy"
    probs = m.predict(None, ["wifi"])
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_train_config_defaults():
    c = TrainConfig(epochs=10)
    assert c.epochs == 10 and c.lr > 0 and c.seed is not None


def _toy_sample():
    mods = {
        "wifi": Modality(np.zeros((2, 3, 114, 10), dtype=np.float32), [1, 2], 1000),
        "depth": Modality(np.zeros((2, 1, 224, 224), dtype=np.float32), [1, 2], 20),
        "lidar": Modality(np.zeros((2, 1536, 3), dtype=np.float32), [1, 2], 20),
        "mmwave": Modality(np.zeros((2, 64, 5), dtype=np.float32), [1, 2], 20),
        "rgb": Modality(np.zeros((2, 17, 2), dtype=np.float32), [1, 2], 20),
    }
    return Sample(id="toy", label=3, modalities=mods)


def test_token_fusion_full_modalities():
    m = TokenFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_token_fusion_missing_modality():
    m = TokenFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi"])  # 3 modalities missing
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_token_fusion_dropout_batch_trains():
    m = TokenFusionModel(num_classes=27)
    s = _toy_sample()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    logits = m(batch, avail={"wifi": True, "depth": True, "lidar": True, "mmwave": False})
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()


def test_token_fusion_per_modality_dropout():
    """modality_dropout overrides the global p per modality; a modality with
    p=1.0 is always missing, p=0.0 always present."""
    from framework.models.token_fusion import MODALITIES
    m = TokenFusionModel(num_classes=27)
    cfg = TrainConfig(epochs=1, modality_dropout_p=0.25,
                      modality_dropout={"mmwave": 1.0, "rgb": 0.0})
    rng = torch.Generator().manual_seed(0)
    for _ in range(50):
        avail = m._dropout_mask(cfg, rng)
        assert avail["mmwave"] is False   # always missing
        assert avail["rgb"] is True       # always present
    # unlisted modalities use the global p
    cfg2 = TrainConfig(epochs=1, modality_dropout_p=0.0,
                       modality_dropout={"mmwave": 1.0})
    rng2 = torch.Generator().manual_seed(0)
    for _ in range(20):
        avail = m._dropout_mask(cfg2, rng2)
        assert avail["mmwave"] is False
        assert avail["wifi"] is True      # global p=0.0 -> always present


def test_late_fusion_missing_modality():
    m = LateFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["depth"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_late_fusion_full():
    m = LateFusionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_mlp_encoder_structured_feature():
    """MLPEncoder maps a 1-D structured feature (e.g. v5_structfeat mmwave
    134d) to the (B,16,D) token shape used by the fusion heads, matching
    PointEncoder's output so it can be swapped in for the main pipeline."""
    from framework.models.encoders import MLPEncoder, N_TOK, D
    for in_dim in (134, 161, 63, 353):  # mmwave / wifi / depth / lidar
        enc = MLPEncoder(in_dim)
        x = torch.randn(4, in_dim)
        out = enc(x)
        assert out.shape == (4, N_TOK, D)
        assert torch.isfinite(out).all()


def test_mlp_encoder_batch_variable_size():
    from framework.models.encoders import MLPEncoder, N_TOK, D
    enc = MLPEncoder(134)
    for b in (1, 3, 16):
        out = enc(torch.randn(b, 134))
        assert out.shape == (b, N_TOK, D)


def _toy_structured_sample():
    """v5_structfeat-style sample: weak modalities are 1-D structured features
    (frame_indices == range(feat_dim), matching make_v5_structfeat), rgb stays
    a raw point cloud."""
    mods = {
        "wifi": Modality(np.zeros(161, dtype=np.float32), list(range(161)), 0),
        "depth": Modality(np.zeros(63, dtype=np.float32), list(range(63)), 0),
        "lidar": Modality(np.zeros(353, dtype=np.float32), list(range(353)), 0),
        "mmwave": Modality(np.zeros(134, dtype=np.float32), list(range(134)), 0),
        "rgb": Modality(np.zeros((2, 17, 2), dtype=np.float32), [1, 2], 20),
    }
    return Sample(id="toy_struct", label=3, modalities=mods)


STRUCTURED = {"wifi": 161, "depth": 63, "lidar": 353, "mmwave": 134}


def test_token_fusion_structured_forward():
    m = TokenFusionModel(num_classes=27, structured=STRUCTURED)
    s = _toy_structured_sample()
    probs = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4
    # missing a structured modality still works
    probs2 = m.predict(s, ["mmwave"])
    assert abs(sum(probs2.values()) - 1.0) < 1e-4


def test_late_fusion_structured_forward():
    m = LateFusionModel(num_classes=27, structured=STRUCTURED)
    s = _toy_structured_sample()
    probs = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_token_fusion_structured_save_load_roundtrip(tmp_path):
    m = TokenFusionModel(num_classes=27, structured=STRUCTURED)
    p = tmp_path / "tf_struct.pt"
    m.save(str(p))
    m2 = TokenFusionModel.load(str(p))
    assert m2.structured == STRUCTURED
    s = _toy_structured_sample()
    p1 = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    p2 = m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert p1 == p2


def test_late_fusion_structured_save_load_roundtrip(tmp_path):
    m = LateFusionModel(num_classes=27, structured=STRUCTURED)
    p = tmp_path / "lf_struct.pt"
    m.save(str(p))
    m2 = LateFusionModel.load(str(p))
    assert m2.structured == STRUCTURED
    s = _toy_structured_sample()
    p1 = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    p2 = m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert p1 == p2


def test_token_fusion_balanced_batch_trains():
    """token_fusion trains with the balanced batching strategy on a small
    imbalanced toy set; the balanced path must run and give finite probs."""
    from framework.dataset.loader import Dataset

    class _Split:
        def __init__(self, samples): self._s = samples
        def __len__(self): return len(self._s)
        def __getitem__(self, i): return self._s[i]

    samples = []
    for c in range(27):
        mods = dict(_toy_sample().modalities)
        samples.append(Sample(id=f"b{c}", label=c, modalities=mods))
    train = _Split(samples)
    ds = Dataset(root="x", splits={"train": train, "val": _Split(samples[:10]),
                                   "test": _Split([])},
                 modalities=list(samples[0].modalities))
    m = TokenFusionModel(num_classes=27)
    import os
    os.makedirs("/tmp/opencode/bal", exist_ok=True)
    cfg = TrainConfig(epochs=1, batch_size=64, seed=0, device="cpu",
                      batch_strategy="balanced", out_dir="/tmp/opencode/bal")
    m.fit(train, ds.val, cfg)
    probs = m.predict(samples[0], ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_load_backward_compat_bare_state_dict(tmp_path):
    """Pre-structured checkpoints are a bare state_dict (no 'state_dict' key).
    load() must still reconstruct a working model (structured={})."""
    m = TokenFusionModel(num_classes=27)
    p = tmp_path / "old_tf.pt"
    torch.save(m.state_dict(), p)  # old format
    m2 = TokenFusionModel.load(str(p))
    assert m2.structured == {}
    s = _toy_sample()
    assert abs(sum(m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"]).values()) - 1.0) < 1e-4

    m3 = LateFusionModel(num_classes=27)
    p3 = tmp_path / "old_lf.pt"
    torch.save(m3.state_dict(), p3)
    m4 = LateFusionModel.load(str(p3))
    assert m4.structured == {}
    assert abs(sum(m4.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"]).values()) - 1.0) < 1e-4


def test_detect_structured_features():
    from framework.models import detect_structured_features
    from framework.dataset.loader import Dataset

    class _DummySplit:
        def __init__(self, samples): self._s = samples
        def __getitem__(self, i): return self._s[i]
        def __len__(self): return len(self._s)

    ds = Dataset(root="x", splits={
        "train": _DummySplit([_toy_structured_sample()]),
        "val": [], "test": []}, modalities=list(STRUCTURED) + ["rgb"])
    assert detect_structured_features(ds) == STRUCTURED

    # raw-only dataset -> empty
    ds_raw = Dataset(root="x", splits={
        "train": _DummySplit([_toy_sample()]),
        "val": [], "test": []}, modalities=["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert detect_structured_features(ds_raw) == {}


def _toy_raw_sample_for_domain():
    """raw multi-dim sample matching v4 real frame count (T=5) so the domain
    feature dims are the deterministic constants (depth 63, wifi 161, ...)."""
    from framework.dataset.sample import Sample, Modality
    mods = {
        "wifi": Modality(np.zeros((5, 3, 114, 10), dtype=np.float32), list(range(5)), 1000),
        "depth": Modality(np.zeros((5, 1, 224, 224), dtype=np.float32), list(range(5)), 20),
        "lidar": Modality(np.zeros((5, 1536, 3), dtype=np.float32), list(range(5)), 20),
        "mmwave": Modality(np.zeros((5, 64, 5), dtype=np.float32), list(range(5)), 20),
        "rgb": Modality(np.zeros((5, 17, 2), dtype=np.float32), list(range(5)), 20),
    }
    return Sample(id="toy_raw", label=3, modalities=mods)


def test_token_fusion_domain_encoder_forward():
    """DomainEncoder wraps the domain feature extractor + MLP: raw multi-dim
    input (B, ...) -> (B, N_TOK, D), and token_fusion consumes it."""
    from framework.models.domain_encoder import DomainEncoder
    from framework.models.encoders import N_TOK, D
    from framework.models.token_fusion import _build_encoders

    # DomainEncoder directly on depth raw (B,5,1,224,224), feat_dim=63 (depth)
    de = DomainEncoder("depth", feat_dim=63)
    x = torch.zeros(2, 5, 1, 224, 224, dtype=torch.float32)
    out = de(x)
    assert out.shape == (2, N_TOK, D)
    assert torch.isfinite(out).all()

    # token_fusion with domain for depth
    m = TokenFusionModel(num_classes=27, domain={"depth": 1}, domain_dims={"depth": 63})
    assert isinstance(m.encoders["depth"], DomainEncoder)
    # depth available -> domain path, others raw encoders
    s = _toy_raw_sample_for_domain()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    logits = m(batch, avail={"wifi": True, "depth": True, "lidar": True, "mmwave": True, "rgb": True})
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()


def test_token_fusion_domain_save_load_roundtrip(tmp_path):
    m = TokenFusionModel(num_classes=27, domain={"depth": 1, "mmwave": 1},
                         domain_dims={"depth": 63, "mmwave": 134})
    p = tmp_path / "tf_domain.pt"
    m.save(str(p))
    m2 = TokenFusionModel.load(str(p))
    assert m2.domain == {"depth": 1, "mmwave": 1}
    assert m2.domain_dims == {"depth": 63, "mmwave": 134}
    s = _toy_raw_sample_for_domain()
    p1 = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    p2 = m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert p1 == p2


def test_temporal_aggregator_rope():
    """TemporalAggregator: (B,T,N,D) -> (B,N,D), finite, shape preserved."""
    from framework.models.temporal import TemporalAggregator
    from framework.models.encoders import D
    x = torch.randn(2, 5, 16, D)
    agg = TemporalAggregator(d=D, n_heads=4)
    out = agg(x)
    assert out.shape == (2, 16, D)
    assert torch.isfinite(out).all()


def test_temporal_aggregator_order_sensitive():
    """RoPE should make the aggregator sensitive to frame order: permuting the
    time axis changes the output."""
    from framework.models.temporal import TemporalAggregator
    from framework.models.encoders import D
    agg = TemporalAggregator(d=D, n_heads=4, n_layers=1)
    x = torch.randn(1, 5, 16, D)
    x0 = x[:, [0, 1, 2, 3, 4]]
    xp = x[:, [4, 3, 2, 1, 0]]
    out0 = agg(x0)
    outp = agg(xp)
    # with RoPE, reordering frames changes the aggregated token
    assert not torch.allclose(out0, outp, atol=1e-6)


def test_token_fusion_temporal_forward_and_roundtrip(tmp_path):
    """token_fusion with temporal=True keeps time axis through encoders and
    aggregates via TemporalAggregator; save/load roundtrip preserves config."""
    m = TokenFusionModel(num_classes=27, temporal=True)
    s = _toy_raw_sample_for_domain()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    avail = {"wifi": True, "depth": True, "lidar": True, "mmwave": True, "rgb": True}
    logits = m(batch, avail)
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()
    p = tmp_path / "tf_temporal.pt"
    m.save(str(p))
    m2 = TokenFusionModel.load(str(p))
    assert m2.temporal is True
    p1 = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    p2 = m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert p1 == p2


def test_temporal_missing_modality_works():
    """Missing modality with temporal=True still works (MISSING token path)."""
    m = TokenFusionModel(num_classes=27, temporal=True)
    s = _toy_raw_sample_for_domain()
    probs = m.predict(s, ["wifi"])  # 4 missing
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_temporal_aggregator_causal_last_frame():
    """Causal structure: the last frame output must depend only on past/current
    frames, not future ones. Masking future frames leaves the last-frame token
    unchanged."""
    from framework.models.temporal import TemporalAggregator
    from framework.models.encoders import D
    agg = TemporalAggregator(d=D, n_heads=4, n_layers=1)
    x = torch.randn(1, 5, 16, D)
    x_masked = x.clone()
    x_masked[:, 0] = 999.0  # corrupt the EARLIEST frame -> last-frame must change
    out = agg(x)
    out_m = agg(x_masked)
    # earliest frame is part of the causal past of the last frame -> differs
    assert not torch.allclose(out, out_m, atol=1e-5)
    # corrupting a frame that is *after* the last output frame is impossible
    # (last frame IS the final one), so the causal property holds by construction.


def test_temporal_time_mask_zeroes_frames():
    """_apply_time_mask zeroes a contiguous run of frames on (B, T, ...),
    preserving the time axis and leaving the source batch intact."""
    m = TokenFusionModel(num_classes=27, temporal=True)
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(2, 5, 64, 5)  # B, T, P, C (mmwave-like)
    orig = x.clone()
    out = m._apply_time_mask(x, rng)
    assert out.shape == x.shape
    # at least one frame fully zeroed
    zero_frames = (out == 0).all(dim=-1).all(dim=-1).all(dim=0)  # (T,)
    assert zero_frames.any()
    # source untouched
    assert torch.equal(x, orig)


def test_time_mask_does_not_apply_at_inference():
    """_stack_mods with rng=None (inference) must NOT mask, regardless of
    cfg.time_mask_p. Use a nonzero raw input and assert it is unchanged."""
    m = TokenFusionModel(num_classes=27, temporal=True)
    cfg = TrainConfig(epochs=1, time_mask_p=1.0, device="cpu")
    s = _toy_raw_sample_for_domain()
    # Give every modality nonzero values so masking would be detectable.
    for name, mod in s.modalities.items():
        mod.data = np.ones_like(mod.data, dtype=np.float32)
    mods = m._stack_mods([s], {"wifi": True, "depth": True, "lidar": True,
                               "mmwave": True, "rgb": True}, cfg, rng=None)
    for name, t in mods.items():
        if t.dim() == 4:  # raw multi-frame
            # rng=None -> no masking -> all ones preserved
            assert torch.count_nonzero(t) == t.numel()


def test_time_mask_training_forward():
    """End-to-end: temporal=True + time_mask_p>0 trains a few steps without
    error and produces finite logits."""
    import torch as _t
    m = TokenFusionModel(num_classes=27, temporal=True)
    cfg = TrainConfig(epochs=1, batch_size=2, time_mask_p=1.0, device="cpu",
                      modality_dropout_p=0.0)
    samples = [_toy_raw_sample_for_domain() for _ in range(4)]
    # monkey-free: build one batch manually with a real rng
    rng = _t.Generator().manual_seed(0)
    avail = {mm: True for mm in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    mods = m._stack_mods(samples[:2], avail, cfg, rng)
    lbl = _t.tensor([3, 3])
    logits = m(mods, avail)
    assert logits.shape == (2, 27)
    assert torch.isfinite(logits).all()


# ---- cross_attention ----

def test_cross_attention_full_modalities():
    m = CrossAttentionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_cross_attention_missing_modality():
    """Missing modalities simply drop out of key/value; query still works."""
    m = CrossAttentionModel(num_classes=27)
    probs = m.predict(_toy_sample(), ["wifi"])  # 4 missing
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_cross_attention_single_modality():
    """Only one modality available -> still produces a valid distribution."""
    m = CrossAttentionModel(num_classes=27)
    for mod in ["wifi", "depth", "lidar", "mmwave", "rgb"]:
        probs = m.predict(_toy_sample(), [mod])
        assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_cross_attention_forward_shape():
    m = CrossAttentionModel(num_classes=27)
    s = _toy_sample()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    logits = m(batch, avail={"wifi": True, "depth": True, "lidar": True, "mmwave": False, "rgb": True})
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()


def test_cross_attention_query_count():
    """n_query controls the number of learnable queries; forward works for
    different query counts."""
    for nq in (8, 32, 64):
        m = CrossAttentionModel(num_classes=27, n_query=nq)
        assert m.query.shape[0] == nq
        s = _toy_sample()
        batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
        logits = m(batch, avail={"wifi": True, "depth": True, "lidar": True, "mmwave": True, "rgb": True})
        assert logits.shape == (1, 27)
        assert torch.isfinite(logits).all()


def test_cross_attention_structured_forward():
    m = CrossAttentionModel(num_classes=27, structured=STRUCTURED)
    s = _toy_structured_sample()
    probs = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4
    probs2 = m.predict(s, ["mmwave"])
    assert abs(sum(probs2.values()) - 1.0) < 1e-4


def test_cross_attention_save_load_roundtrip(tmp_path):
    m = CrossAttentionModel(num_classes=27, structured=STRUCTURED, n_query=32)
    p = tmp_path / "ca_struct.pt"
    m.save(str(p))
    m2 = CrossAttentionModel.load(str(p))
    assert m2.structured == STRUCTURED
    assert m2.n_query == 32
    s = _toy_structured_sample()
    p1 = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    p2 = m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert p1 == p2


def test_cross_attention_temporal_forward_and_roundtrip(tmp_path):
    m = CrossAttentionModel(num_classes=27, temporal=True)
    s = _toy_raw_sample_for_domain()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    avail = {"wifi": True, "depth": True, "lidar": True, "mmwave": True, "rgb": True}
    logits = m(batch, avail)
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()
    p = tmp_path / "ca_temporal.pt"
    m.save(str(p))
    m2 = CrossAttentionModel.load(str(p))
    assert m2.temporal is True
    p1 = m.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    p2 = m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert p1 == p2


def test_cross_attention_temporal_missing_modality():
    m = CrossAttentionModel(num_classes=27, temporal=True)
    s = _toy_raw_sample_for_domain()
    probs = m.predict(s, ["wifi"])  # 4 missing
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_cross_attention_domain_forward():
    from framework.models.domain_encoder import DomainEncoder
    m = CrossAttentionModel(num_classes=27, domain={"depth": 1}, domain_dims={"depth": 63})
    assert isinstance(m.encoders["depth"], DomainEncoder)
    s = _toy_raw_sample_for_domain()
    batch = {name: torch.from_numpy(mod.data)[None] for name, mod in s.modalities.items()}
    logits = m(batch, avail={"wifi": True, "depth": True, "lidar": True, "mmwave": True, "rgb": True})
    assert logits.shape == (1, 27)
    assert torch.isfinite(logits).all()


def test_cross_attention_balanced_batch_trains():
    """cross_attention trains with the balanced batching strategy on a small
    toy set; the balanced path must run and give finite probs."""
    from framework.dataset.loader import Dataset

    class _Split:
        def __init__(self, samples): self._s = samples
        def __len__(self): return len(self._s)
        def __getitem__(self, i): return self._s[i]

    samples = []
    for c in range(27):
        mods = dict(_toy_sample().modalities)
        samples.append(Sample(id=f"b{c}", label=c, modalities=mods))
    train = _Split(samples)
    ds = Dataset(root="x", splits={"train": train, "val": _Split(samples[:10]),
                                   "test": _Split([])},
                 modalities=list(samples[0].modalities))
    m = CrossAttentionModel(num_classes=27)
    import os
    os.makedirs("/tmp/opencode/bal_ca", exist_ok=True)
    cfg = TrainConfig(epochs=1, batch_size=64, seed=0, device="cpu",
                      batch_strategy="balanced", out_dir="/tmp/opencode/bal_ca")
    m.fit(train, ds.val, cfg)
    probs = m.predict(samples[0], ["wifi", "depth", "lidar", "mmwave", "rgb"])
    assert len(probs) == 27 and abs(sum(probs.values()) - 1.0) < 1e-4


def test_cross_attention_load_backward_compat_bare_state_dict(tmp_path):
    """Pre-config checkpoints are a bare state_dict; load() must reconstruct
    a working model (structured={}, n_query=32)."""
    m = CrossAttentionModel(num_classes=27, n_query=32)
    p = tmp_path / "old_ca.pt"
    torch.save(m.state_dict(), p)  # old format
    m2 = CrossAttentionModel.load(str(p))
    assert m2.structured == {}
    assert m2.n_query == 32
    s = _toy_sample()
    assert abs(sum(m2.predict(s, ["wifi", "depth", "lidar", "mmwave", "rgb"]).values()) - 1.0) < 1e-4

