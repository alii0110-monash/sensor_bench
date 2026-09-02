#!/usr/bin/env python
"""v3: copy v2 samples + inject `rgb` body-keypoint modality (17,2) per frame.

Apples-to-apples: same splits, same labels, same 4 sensor modalities. Only
adds the rgb keypoints so we can test "does a vision-derived modality lift
robustness" without changing anything else.
"""
import argparse
import json
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from curation.ingest.mmfi import sample_frames
from curation.ingest.readers import read_keypoint_frame
from curation.version.version import write_meta


def inject_rgb(sample: dict, raw_root: str) -> dict:
    """Returns a copy of sample dict with `rgb` modality added."""
    env, subj, act = sample["id"].split("_")[:3]
    act_dir = os.path.join(raw_root, env, subj, act)
    frames = sample["modalities"]["mmwave"]["frame_indices"]
    keypoints = np.stack([read_keypoint_frame(
        os.path.join(act_dir, "rgb", f"frame{f:03d}.npy")) for f in frames], axis=0)
    sample = dict(sample)
    mods = dict(sample["modalities"])
    mods["rgb"] = {"data": keypoints, "frame_indices": frames,
                   "sample_rate": 20, "name": "rgb"}
    sample["modalities"] = mods
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="datasets/mmfi/v2")
    ap.add_argument("--v3", default="datasets/mmfi/v3")
    ap.add_argument("--raw-root", required=True)
    args = ap.parse_args()

    os.makedirs(f"{args.v3}/data", exist_ok=True)
    missing = []
    n = 0
    for fn in sorted(os.listdir(f"{args.v2}/data")):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(args.v2, "data", fn), "rb") as f:
            sample = pickle.load(f)
        try:
            sample = inject_rgb(sample, args.raw_root)
        except Exception as e:  # noqa: BLE001
            missing.append((fn, str(e)))
            continue
        with open(os.path.join(args.v3, "data", fn), "wb") as f:
            pickle.dump(sample, f)
        n += 1

    shutil.copytree(f"{args.v2}/splits", f"{args.v3}/splits", dirs_exist_ok=True)
    with open(os.path.join(args.v3, "modalities.yaml"), "w") as f:
        f.write("modalities:\n- wifi\n- depth\n- lidar\n- mmwave\n- rgb\n"
                "note: v3 = v2 samples + rgb body-keypoints (17,2) modality\n")
    json.dump({"v2_to_v3": {"added": "rgb keypoints (17,2) per frame",
                            "n_missing": len(missing), "missing": missing[:10]},
               "kept": n}, open(f"{args.v3}/changes.json", "w"), indent=2)
    write_meta(args.v3, name="mmfi", version="v3",
               changelog=["v3: injected rgb body-keypoint modality (17,2) from "
                          "MMFi raw (ResNet-48 keypoints); splits/labels unchanged"],
               n_samples=n, n_modalities=5,
               source={"dataset": "MMFi", "split": "cs", "parent": "mmfi/v2"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"based_on": "mmfi/v2", "rgb_source": "frameNNN.npy keypoints"})
    print(f"v3: kept {n} samples, {len(missing)} missing", flush=True)


if __name__ == "__main__":
    main()
