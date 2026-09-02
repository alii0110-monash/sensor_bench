#!/usr/bin/env python
"""M6a: v5 传感器 → CanonicalToken 资产化落盘 (datasets/mmfi/v5tokens/)."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.tokens.assets import write_tokens
from framework.tokens.tokenizer import CanonicalTokenizer


def make_tokens(samples, align_ckpt, proj_ckpt, out_root, k=8, device="cpu") -> dict:
    tok = CanonicalTokenizer(align_ckpt=align_ckpt, proj_ckpt=proj_ckpt, k=k, device=device)
    tokens = []
    for s in samples:
        tokens.append(tok.encode(s))
    return write_tokens(tokens, out_root, version="v1",
                        encoder_ckpt=f"{align_ckpt}+{proj_ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--align-ckpt", default="checkpoints_alignment/alignment_seed0.pt")
    ap.add_argument("--proj-ckpt", default="checkpoints_projection_verb/projection_seed0.pt")
    ap.add_argument("--out", default="datasets/mmfi/v5tokens")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0=全部 train base")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, mode="lazy")
    samples = [s for s in ds.train if "__aug" not in s.id]
    if args.limit:
        samples = samples[:args.limit]
    idx = make_tokens(samples, args.align_ckpt, args.proj_ckpt, args.out,
                      k=args.k, device=args.device)
    print(f"v5tokens: wrote {idx['n_samples']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
