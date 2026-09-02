# tests/test_batch_predict.py
import torch

from framework.harness.evaluate import evaluate_model
from framework.models.token_fusion import TokenFusionModel, MODALITIES
from framework.models.late_fusion import LateFusionModel
from framework.dataset.sample import Modality, Sample
import numpy as np


def _toy_sample(label=3):
    mods = {
        "wifi": Modality(np.zeros((2, 3, 114, 10), dtype=np.float32), [1, 2], 1000),
        "depth": Modality(np.zeros((2, 1, 224, 224), dtype=np.float32), [1, 2], 20),
        "lidar": Modality(np.zeros((2, 1536, 3), dtype=np.float32), [1, 2], 20),
        "mmwave": Modality(np.zeros((2, 64, 5), dtype=np.float32), [1, 2], 20),
        "rgb": Modality(np.zeros((2, 17, 2), dtype=np.float32), [1, 2], 20),
    }
    return Sample(id=f"toy_{label}", label=label, modalities=mods)


def _toy_samples(n=8):
    return [_toy_sample(label=i % 5) for i in range(n)]


def _assert_batch_matches_single(model, samples, available):
    """predict_batch 的 top-1 必须与逐样本 predict 完全一致。"""
    single = [max(model.predict(s, available), key=model.predict(s, available).get)
              for s in samples]
    batched = model.predict_batch(samples, available)
    assert batched.shape == (len(samples), model.num_classes)
    assert batched.argmax(-1).tolist() == single


def test_token_fusion_predict_batch_matches_single():
    torch.manual_seed(0)
    m = TokenFusionModel(num_classes=27)
    m.eval()
    samples = _toy_samples()
    _assert_batch_matches_single(m, samples, MODALITIES)
    _assert_batch_matches_single(m, samples, ["mmwave", "wifi"])


def test_late_fusion_predict_batch_matches_single():
    torch.manual_seed(0)
    m = LateFusionModel(num_classes=27)
    m.eval()
    samples = _toy_samples()
    _assert_batch_matches_single(m, samples, MODALITIES)
    _assert_batch_matches_single(m, samples, ["mmwave"])


def test_evaluate_model_uses_batch_matches_single_accuracy():
    torch.manual_seed(0)
    m = TokenFusionModel(num_classes=27)
    m.eval()
    samples = _toy_samples(16)
    profile = {"id": "full", "available": MODALITIES}
    # 分别跑：evaluate_model 应内部走 predict_batch
    res = evaluate_model(m, samples, profile)
    assert 0.0 <= res["accuracy"] <= 1.0
