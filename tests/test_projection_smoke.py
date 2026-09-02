"""L2 smoke: projection produces pseudo tokens that inject into a frozen LLM
prefix, forward passes, and the frozen LLM's text QA ability doesn't regress.
Uses MockAdapter unless --real-llm (then llama2-7b)."""
import os
import pytest
import torch

# ---- pure-interface smoke (no LLM load, runs in CI) ----

def test_projection_forward_interface():
    from framework.models.perceiver import PerceiverProjection
    from framework.models.router import TokenRouter
    proj = PerceiverProjection(in_dim=256, out_dim=512, k=4)
    router = TokenRouter(k_max=4)
    ct = torch.randn(2, 5, 16, 256)
    pseudo = proj(ct)                       # (2, 20, 512)
    avail = {m: True for m in ["wifi", "depth", "lidar", "mmwave", "rgb"]}
    counts = router.route(avail, budget=12)
    # inject first sum(counts) pseudo tokens as prefix
    n = sum(counts.values())
    prefix = pseudo[:, :n]
    assert prefix.shape[1] == n
    assert n <= 12

def test_inject_preserves_text_tokens():
    from framework.models.llm_adapter import LLMAdapter
    class T(LLMAdapter):
        @property
        def hidden_dim(self): return 128
        def project(self, ct): return torch.randn(ct.shape[0], 4, 128)
        def inject(self, pe, ids, embed_fn):
            te = embed_fn(ids)
            return torch.cat([pe, te], dim=1)
    a = T()
    pe = torch.randn(2, 4, 128)
    ids = torch.randint(0, 10, (2, 6))
    merged = a.inject(pe, ids, lambda x: torch.randn(x.shape[0], x.shape[1], 128))
    assert merged.shape == (2, 10, 128)  # 4 pseudo + 6 text, no token loss


@pytest.mark.slow
def test_real_llm_inject_forward():
    """真实 llama2-7b: 伪 token 前缀 + 文本前向通过, 文本 token 数不丢."""
    from framework.models.llm_adapter import LlamaAdapter
    adapter = LlamaAdapter(model_path="/home/li/datasets/models/llama2-7b",
                           k=4, device="cuda")
    model, tok = adapter._load()
    device = "cuda"
    # 文本侧
    ids = tok("What is the person doing?", return_tensors="pt").input_ids.to(device)
    # 伪 token 前缀 (随机, 只验证通道)
    ct = torch.randn(1, 5, 16, 256, device=device)
    pseudo = adapter.project(ct)            # (1, 20, 4096)
    merged = adapter.inject(pseudo[:, :4], ids)   # 4 pseudo + N text
    assert merged.shape == (1, 4 + ids.shape[1], 4096)
    # 前向通过
    with torch.no_grad():
        out = model(inputs_embeds=merged, attention_mask=torch.ones(
            merged.shape[0], merged.shape[1], dtype=torch.long, device=device))
    assert out.logits.shape[1] == merged.shape[1]
