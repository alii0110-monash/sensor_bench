#!/usr/bin/env python
"""SFT MVP evaluation CLI."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.llm_sft.eval_sft import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_sftmvp")
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--anchors", default="results/sftmvp/class_anchors.json")
    ap.add_argument("--out", default="results/sftmvp/eval_mvp.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mode", choices=["auto", "eager", "lazy"], default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--collect-gens", type=int, default=200)
    args = ap.parse_args()

    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        args.device = "cpu"
        print("[sftmvp] no CUDA, falling back to CPU", flush=True)
    res = evaluate(ckpt_dir=args.ckpt, dataset_root=args.dataset,
                   anchors_path=args.anchors, out_path=args.out,
                   device=args.device, batch_size=args.batch_size,
                   load_mode=args.mode, max_new_tokens=args.max_new_tokens,
                   collect_gens=args.collect_gens)
    print(f"[sftmvp] POSITIVE={res['positive']}")


if __name__ == "__main__":
    main()
