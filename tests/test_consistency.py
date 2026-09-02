# tests/test_consistency.py
from curation.clean.consistency import flag_inconsistent
from framework.models.token_fusion import TokenFusionModel
from tests.test_models import _toy_sample


def test_flag_inconsistent_returns_sorted_subset():
    m = TokenFusionModel(num_classes=27)
    samples = [_toy_sample() for _ in range(4)]
    out = flag_inconsistent(m, samples, drop_rate=0.5)
    assert isinstance(out, list)
    assert len(out) <= 2
