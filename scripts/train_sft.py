#!/usr/bin/env python
"""SFT MVP training CLI."""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.llm_sft.train_sft import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v4")
    ap.add_argument("--encoders", default="checkpoints_alignment/m6b_v4text_seed0.pt")
    ap.add_argument("--captions", default="results/captions_route_c_train.jsonl")
    ap.add_argument("--anchors", default="results/sftmvp/class_anchors.json")
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/qwen2.5-0.5b-instruct"))
    ap.add_argument("--out", default="checkpoints_sftmvp")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["auto", "eager", "lazy"], default="auto")
    ap.add_argument("--max-train", type=int, default=0, help=">0: cap train samples (smoke)")
    ap.add_argument("--log", default="results/sftmvp/train_log.json")
    args = ap.parse_args()

    if args.device == "cuda" and not __import__("torch").cuda.is_available():
        args.device = "cpu"
        print("[sftmvp] no CUDA, falling back to CPU", flush=True)
    train(dataset_root=args.dataset, encoders_ckpt=args.encoders,
          captions_jsonl=args.captions, anchors_path=args.anchors,
          model_dir=args.model_dir, out_dir=args.out, epochs=args.epochs,
          batch_size=args.batch_size, seed=args.seed, lr_proj=args.lr_proj,
          lr_lora=args.lr_lora, device=args.device, load_mode=args.mode,
          max_train=args.max_train, log_path=args.log)


if __name__ == "__main__":
    main()
