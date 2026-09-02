# tests/test_llm_adapter.py
import torch
import pytest
from framework.models.llm_adapter import LLMAdapter

def test_abstract_requires_methods():
    with pytest.raises(TypeError):
        LLMAdapter()  # ABC with abstract methods

class MockAdapter(LLMAdapter):
    """Minimal fake for interface tests (no LLM load)."""
    @property
    def hidden_dim(self) -> int:
        return 4096
    def project(self, canonical_tokens):
        B, M, K, D = canonical_tokens.shape
        return torch.nn.Linear(256, 4096)(canonical_tokens.reshape(B, M * K, D))
    def inject(self, prefix_embs, input_ids, embed_fn):
        # embed_fn: (B, T) -> (B, T, H); returns concat(prefix, text_embs)
        text_embs = embed_fn(input_ids)
        return torch.cat([prefix_embs, text_embs], dim=1)

def test_mock_adapter_project_inject():
    a = MockAdapter()
    ct = torch.randn(2, 5, 16, 256)
    pseudo = a.project(ct)
    assert pseudo.shape == (2, 80, 4096)
    input_ids = torch.randint(0, 100, (2, 10))
    def embed_fn(ids):
        return torch.randn(ids.shape[0], ids.shape[1], 4096)
    merged = a.inject(pseudo[:, :8], input_ids, embed_fn)
    assert merged.shape == (2, 18, 4096)  # 8 pseudo + 10 text
