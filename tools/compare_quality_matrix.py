#!/usr/bin/env python
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base quality_matrix json")
    ap.add_argument("--new", required=True, help="new quality_matrix json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    b = json.load(open(args.base))
    n = json.load(open(args.new))
    common = set(b["matrix"]) & set(n["matrix"])
    diff = {}
    for k in common:
        bq = b["matrix"][k].get("quality")
        nq = n["matrix"][k].get("quality")
        if bq is not None and nq is not None:
            diff[k] = nq - bq
    result = {
        "base": args.base, "new": args.new,
        "n_cells": len(common),
        "improved": sorted([(k, v) for k, v in diff.items() if v > 0], key=lambda x: -x[1])[:20],
        "regressed": sorted([(k, v) for k, v in diff.items() if v < 0], key=lambda x: x[1])[:20],
        "mean_delta": float(sum(diff.values()) / len(diff)) if diff else 0.0,
    }
    json.dump(result, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
