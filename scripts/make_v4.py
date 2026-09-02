#!/usr/bin/env python
"""v4: normalize keypoints (all splits) + offline spatial augmentation (train only).

- normalize: hip-center + torso-length scale (deterministic, camera-invariant)
- augment (train): per-train-sample `n_aug` stochastic variants (flip/translate/scale)
- STORAGE (dedup): a variant is written as a small DELTA file that references its
  base sample by id and only carries the augmented rgb modality + meta, instead of
  duplicating the 4 unchanged modalities (~1.17MB) per variant. The loader
  (framework/dataset/loader.py) reconstructs the full variant Sample by resolving
  the base and swapping in the delta rgb. This cuts v4 disk from ~58GB to ~19GB.
- val/test: normalized only, id/sample count unchanged.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from curation.enrich.keypoints import (augment_keypoints, make_variant_id,
                                       normalize_keypoints)
from curation.version.version import write_meta

_VARIANT_MARKER = "__aug"


def transform_rgb(sample: dict, mode: str, rng=None, n_aug: int = 4) -> list:
    """Normalize rgb of one sample; if mode=='train', also produce n_aug variants.
    Returns list of (sample_dict, is_variant) to write."""
    rgb = sample["modalities"]["rgb"]["data"]
    rgb_n = normalize_keypoints(rgb)
    base = dict(sample)
    base["modalities"] = dict(base["modalities"])
    base["modalities"]["rgb"] = dict(base["modalities"]["rgb"])
    base["modalities"]["rgb"]["data"] = rgb_n
    base["meta"] = dict(base.get("meta", {}))
    base["meta"]["keypoint_norm"] = "hip_center+torso_len"
    out = [(base, False)]
    if mode == "train":
        for k in range(n_aug):
            delta = {
                "kind": "variant",
                "id": make_variant_id(base["id"], k),
                "base_id": base["id"],
                "label": base["label"],
                "rgb": {"data": augment_keypoints(rgb_n, rng),
                        "frame_indices": list(base["modalities"]["rgb"]["frame_indices"]),
                        "sample_rate": base["modalities"]["rgb"]["sample_rate"]},
                "aug": k,
            }
            out.append((delta, True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="datasets/mmfi/v3")
    ap.add_argument("--v4", default="datasets/mmfi/v4")
    ap.add_argument("--n-aug", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = json.load(open(os.path.join(args.v3, "splits", f"{name}.json")))

    os.makedirs(f"{args.v4}/data", exist_ok=True)
    n_written = 0
    train_variants = []
    for fn in sorted(os.listdir(f"{args.v3}/data")):
        if not fn.endswith(".pkl"):
            continue
        sid = fn.replace(".pkl", "")
        if sid in splits["train"]:
            mode = "train"
        elif sid in splits["val"]:
            mode = "val"
        elif sid in splits["test"]:
            mode = "test"
        else:
            continue  # not in any split (shouldn't happen)
        with open(os.path.join(args.v3, "data", fn), "rb") as f:
            sample = pickle.load(f)
        for s, is_var in transform_rgb(sample, mode, rng=rng, n_aug=args.n_aug):
            with open(os.path.join(args.v4, "data", f"{s['id']}.pkl"), "wb") as f:
                pickle.dump(s, f)
            n_written += 1
            if is_var:
                train_variants.append(s["id"])

    # extended train split = original + variants
    os.makedirs(f"{args.v4}/splits", exist_ok=True)
    splits["train"] = list(splits["train"]) + sorted(train_variants)
    for name, ids in splits.items():
        with open(os.path.join(args.v4, "splits", f"{name}.json"), "w") as f:
            json.dump(ids, f)

    with open(os.path.join(args.v4, "modalities.yaml"), "w") as f:
        f.write("modalities:\n- wifi\n- depth\n- lidar\n- mmwave\n- rgb\n"
                "note: v4 = v3 rgb normalized (hip-center+torso-len) all splits; "
                "train augmented with spatial variants (flip/translate/scale)\n")
    json.dump({"v3_to_v4": {"rgb_norm": "hip_center+torso_len",
                            "augment": {"n_aug": args.n_aug, "seed": args.seed,
                                        "flip_p": 0.5, "trans_frac": 0.1,
                                        "scale_range": [0.9, 1.1]},
                            "train_orig": len(splits["train"]) - len(train_variants),
                            "train_variants": len(train_variants)},
               "n_written": n_written},
              open(f"{args.v4}/changes.json", "w"), indent=2)
    write_meta(args.v4, name="mmfi", version="v4",
               changelog=["v4: rgb keypoints normalized (hip-center+torso-length) on all "
                          "splits; train extended with offline spatial variants (n_aug "
                          f"={args.n_aug}: flip/translate/scale). val/test unchanged ids."],
               n_samples=n_written, n_modalities=5,
               source={"dataset": "MMFi", "split": "cs", "parent": "mmfi/v3"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"based_on": "mmfi/v3"})
    print(f"v4: written {n_written} (train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])})", flush=True)


if __name__ == "__main__":
    main()
