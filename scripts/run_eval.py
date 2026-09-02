#!/usr/bin/env python
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.harness.protocol import build_protocol
from framework.harness.evaluate import evaluate_model
from framework.harness.leaderboard import build_leaderboard, save_leaderboard
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel
from framework.models.cross_attention import CrossAttentionModel

MODELS = {"token_fusion": TokenFusionModel, "late_fusion": LateFusionModel,
          "cross_attention": CrossAttentionModel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True, help="path to protocol.json")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", default="leaderboard_v1.json")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--seeds", default="0", help="comma-separated")
    ap.add_argument("--mode", choices=["auto", "eager", "lazy"], default=None,
                    help="dataset loading mode. Default auto; use lazy for raw "
                         "multi-frame datasets (v4 ~18GB) that do not fit the "
                         "cgroup and would OOM-kill the whole group.")
    args = ap.parse_args()

    mode = args.mode or "auto"
    ds = load_dataset(args.dataset, mode=mode)
    protocol = json.load(open(args.protocol))
    models = args.model or list(MODELS)
    seeds = [int(x) for x in args.seeds.split(",")]

    results = {}
    for name in models:
        results[name] = []
        for seed in seeds:
            m = MODELS[name].load(f"{args.ckpt_dir}/{name}_seed{seed}.pt")
            for profile in protocol["profiles"]:
                r = evaluate_model(m, ds.test, profile)
                r["seed"] = seed
                results[name].append(r)

    lb = build_leaderboard(results)
    save_leaderboard(lb, args.out, protocol, args.dataset)
    print(json.dumps(lb, indent=2))


if __name__ == "__main__":
    main()
