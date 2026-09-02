"""L3 smoke on real llama2-7b (slow)."""
import os
import pytest
import torch

@pytest.mark.slow
def test_llm_interface_smoke():
    """真实 llama2-7b: 纯文本 + 伪token 两种模式各跑几个样本, 前向通过."""
    from framework.models.llm_adapter import LlamaAdapter
    from framework.dataset.loader import load_dataset
    from framework.models.alignment import AlignmentModel
    from framework.models.perceiver import PerceiverProjection
    from framework.models.router import TokenRouter
    from framework.eval.llm_interface import LLMEvaluator

    adapter = LlamaAdapter(model_path="/home/li/datasets/models/llama2-7b", k=8, device="cuda")
    model, tok = adapter._load()
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.projection_head = torch.nn.Sequential(
        torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))  # 原型头
    align.load_state_dict(torch.load("checkpoints_alignment/alignment_seed0.pt", map_location="cpu"), strict=False)
    align.eval().to("cuda")
    proj = PerceiverProjection(out_dim=4096, k=8).to("cuda")   # 必须与 Task 3 训练 k=8 一致
    proj.load_state_dict(torch.load("checkpoints_projection/projection_seed0.pt", map_location="cpu"))

    ds = load_dataset("datasets/mmfi/v5", mode="lazy")
    samples = []
    for s in ds.test:
        if s.text.get("captions"):
            samples.append(s)
            if len(samples) >= 10:
                break

    def _generate(prompt, prefix_embs=None):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        if prefix_embs is None:
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=16)
        else:
            merged = adapter.inject(prefix_embs, ids)
            with torch.no_grad():
                out = model.generate(inputs_embeds=merged, max_new_tokens=16)
        return tok.decode(out[0], skip_special_tokens=True)

    class _LLM:
        def __init__(self, fn):
            self._fn = fn
        def generate(self, prompt, prefix_embs=None):
            return self._fn(prompt, prefix_embs)

    ev = LLMEvaluator(_LLM(_generate))
    labels = [s.label for s in samples]
    texts = [s.text["captions"][0] for s in samples]
    acc_text = ev.evaluate_text(texts, labels)
    # pseudo-token mode: per-modality slicing per router counts (5 modalities)
    # 注意: evaluate_pseudo_tokens 内部 prompt 无 caption (防文本混淆)
    from framework.models.router import TokenRouter
    router = TokenRouter(k_max=8)
    pes = []
    for s in samples:
        mods = {m: torch.from_numpy(s.modalities[m].data)[None].to("cuda")
                for m in s.modalities}
        avail = {m: True for m in s.modalities}
        counts = router.route(avail, budget=8)
        with torch.no_grad():
            ct = align.encode_modalities(mods, avail)
            pe = proj(ct)                       # (1, M*k, H) modality-major
        # slice per-modality: modality j occupies rows [j*k, (j+1)*k)
        # 保留 batch 维 (1, ...) 以便 inject 拼接
        parts = []
        for j, m in enumerate(s.modalities):
            kk = counts[m]
            if kk > 0:
                parts.append(pe[:, j*8:(j*8)+kk])
        pes.append(torch.cat(parts, dim=1))
    acc_pseudo = ev.evaluate_pseudo_tokens(texts, labels, pes)
    assert isinstance(acc_text, float) and isinstance(acc_pseudo, float)
