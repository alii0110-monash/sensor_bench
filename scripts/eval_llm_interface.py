#!/usr/bin/env python
"""L3 end-to-end LLM eval (M5c): text vs pseudo-token vs no-context on real llama2-7b."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel
from framework.models.llm_adapter import LlamaAdapter
from framework.models.perceiver import PerceiverProjection
from framework.models.router import TokenRouter
from framework.eval.llm_interface import LLMEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="checkpoints_alignment/alignment_seed0.pt")
    ap.add_argument("--proj-ckpt", default="checkpoints_projection_verb/projection_seed0.pt")
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=50, help="eval sample count")
    ap.add_argument("--budget", type=int, default=8, help="token budget for router (per-modality k_max=k)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    adapter = LlamaAdapter(model_path=args.llm, k=args.k, device=device)
    model, tok = adapter._load()
    align = AlignmentModel(num_modalities=5, text_dim=512)
    align.projection_head = torch.nn.Sequential(
        torch.nn.Linear(256, 27), torch.nn.Linear(27, 512))  # 原型头
    align.load_state_dict(torch.load(args.align_ckpt, map_location="cpu"), strict=False)
    align.eval().to(device)
    proj = PerceiverProjection(out_dim=adapter.hidden_dim, k=args.k).to(device)
    proj.load_state_dict(torch.load(args.proj_ckpt, map_location="cpu"))
    router = TokenRouter(k_max=args.k)

    ds = load_dataset(args.dataset, mode="lazy")
    samples = []
    for s in ds.test:
        if s.text.get("captions"):
            samples.append(s)
            if len(samples) >= args.n:
                break

    def _generate(prompt, prefix_embs=None):
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        if prefix_embs is None:
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=16)
        else:
            merged = adapter.inject(prefix_embs, ids)
            with torch.no_grad():
                out = model.generate(inputs_embeds=merged, max_new_tokens=16)
        return tok.decode(out[0], skip_special_tokens=True)

    class _LLM:
        """适配: LLMEvaluator 期望 llm.generate(prompt, prefix_embs)."""
        def __init__(self, fn):
            self._fn = fn
        def generate(self, prompt, prefix_embs=None):
            return self._fn(prompt, prefix_embs)

    ev = LLMEvaluator(_LLM(_generate))
    labels = [s.label for s in samples]
    texts = [s.text["captions"][0] for s in samples]

    acc_text = ev.evaluate_text(texts, labels)
    acc_baseline = ev.evaluate_no_context(labels)   # 下界桩 (0.0): 真实无上下文约 1/27≈0.037

    pes = []
    for s in samples:
        mods = {m: torch.from_numpy(s.modalities[m].data)[None].to(device)
                for m in s.modalities}
        avail = {m: True for m in s.modalities}
        counts = router.route(avail, args.budget)
        with torch.no_grad():
            ct = align.encode_modalities(mods, avail)
            pe = proj(ct)                       # (1, M*k, H) modality-major
        # per-modality slicing: modality j occupies rows [j*k, (j+1)*k)
        # 保留 batch 维 (1, ...) 以便 inject 拼接 (prefix (1,n,H) + text (1,T,H))
        parts = []
        for j, m in enumerate(s.modalities):
            kk = counts[m]
            if kk > 0:
                parts.append(pe[:, j*args.k:(j*args.k)+kk])
        pes.append(torch.cat(parts, dim=1))
    # pseudo mode: prompt 无 caption (见 evaluate_pseudo_tokens 注释, 防混淆)
    acc_pseudo = ev.evaluate_pseudo_tokens(texts, labels, pes)

    print(f"[L3] n={len(samples)} acc_text={acc_text:.3f} "
          f"acc_pseudo={acc_pseudo:.3f} acc_baseline={acc_baseline:.3f}")
    print(f"[L3] pseudo - text = {acc_pseudo - acc_text:+.3f} (正=伪token有增益)")


if __name__ == "__main__":
    main()
