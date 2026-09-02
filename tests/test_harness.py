# tests/test_harness.py
from framework.harness.evaluate import evaluate_model, accuracy
from framework.harness.leaderboard import build_leaderboard
from framework.models.token_fusion import TokenFusionModel
from tests.test_models import _toy_sample


def test_accuracy():
    preds = [{0: 0.9, 1: 0.1}, {0: 0.1, 1: 0.9}]
    labels = [0, 1]
    assert accuracy(preds, labels) == 1.0


def test_evaluate_model_single_sample():
    m = TokenFusionModel(num_classes=27)
    s = _toy_sample()
    res = evaluate_model(m, [s], profile={"id": "full", "available": ["wifi", "depth", "lidar", "mmwave"]})
    assert 0.0 <= res["accuracy"] <= 1.0


def test_leaderboard_mean_std():
    results = {
        "m": [
            {"profile": "full", "available": [], "accuracy": 0.8, "seed": 0},
            {"profile": "full", "available": [], "accuracy": 0.9, "seed": 1},
            {"profile": "miss-wifi", "available": [], "accuracy": 0.6, "seed": 0},
            {"profile": "miss-wifi", "available": [], "accuracy": 0.8, "seed": 1},
        ]
    }
    lb = build_leaderboard(results)
    assert abs(lb["m"]["profiles"]["full"]["mean"] - 0.85) < 1e-6
    assert abs(lb["m"]["profiles"]["full"]["std"] - 0.05) < 1e-6
    assert lb["m"]["robustness"] == 0.775
    assert "degradation" in lb["m"]
