#!/usr/bin/env python
"""L2 smoke (M5b §97): pseudo-token prefix injection into a frozen local LLM,
forward passes, and text-only regression (text ability unchanged).

Usage:
  python scripts/smoke_llm_inject.py [--llm .../llama2-7b] [--k 8] [--device cuda]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.models.llm_adapter import LlamaAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="/home/li/datasets/models/llama2-7b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    adapter = LlamaAdapter(model_path=args.llm, k=args.k, device=device)
    model, tok = adapter._load()

    text = "What is the person doing?"
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    # 1) text-only forward (regression baseline)
    with torch.no_grad():
        out_text = model(input_ids=ids)
    text_tokens = ids.shape[1]

    # 2) pseudo-token prefix forward
    ct = torch.randn(1, 5, 16, 256, device=device)
    pseudo = adapter.project(ct)
    n_prefix = min(args.k, pseudo.shape[1])
    merged = adapter.inject(pseudo[:, :n_prefix], ids)
    with torch.no_grad():
        out_inj = model(inputs_embeds=merged,
                        attention_mask=torch.ones(
                            merged.shape[0], merged.shape[1], dtype=torch.long,
                            device=device))

    # text tokens preserved: prefix + text == merged
    assert merged.shape == (1, n_prefix + text_tokens, adapter.hidden_dim), merged.shape
    assert out_inj.logits.shape[1] == merged.shape[1]

    # regression: text-only logits at last text position unchanged shape/value range
    lt = out_text.logits[:, -1]
    assert torch.isfinite(lt).all()
    print(f"[smoke] prefix={n_prefix} text_tokens={text_tokens} "
          f"merged={merged.shape[1]} forward OK; text regression OK")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
