#!/usr/bin/env python
"""v5 = v4 + weak-modality raw → structured features.

Replaces raw depth/wifi/lidar/mmwave data with domain-aware structured
features in each pickle (sizes drop from ~1MB to <1KB per modality). Keeps
rgb unchanged (already structured keypoints). Variants (which only carry
an rgb delta + reference base) automatically pick up the new base data via
the existing loader.

Probe validation has shown these features lift per-modality val acc:
  depth  0.077 → 0.243
  wifi   0.048 → 0.109
  lidar  0.095 → 0.249
  mmwave 0.348 → 0.509

Strategy:
- Hard-link v4 pickles into v5/data/
- Overwrite only base-sample pickles with new structured data
- Variants inherit via loader reconstruction
- Splits + meta copied from v4, meta changelog notes the change
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation.io import safe_replace_pickle
from curation.version.version import write_meta
from framework.dataset.sample import Sample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framework.eval.dataset_quality.feature_extract import (
    extract_depth_features, extract_wifi_features,
    extract_lidar_features, extract_mmwave_features,
)

# Modalities whose raw data we replace with structured features.
_STRUCTURED_MODALITIES = ("depth", "wifi", "lidar", "mmwave")

_EXTRACTORS = {
    "depth": extract_depth_features,
    "wifi": extract_wifi_features,
    "lidar": extract_lidar_features,
    "mmwave": extract_mmwave_features,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/mmfi/v4")
    ap.add_argument("--dst", default="datasets/mmfi/v5_structfeat")
    ap.add_argument("--report", default="results/v5_structfeat_report.json")
    args = ap.parse_args()

    src_data = os.path.join(args.src, "data")
    dst_data = os.path.join(args.dst, "data")
    os.makedirs(dst_data, exist_ok=True)

    # Copy splits + meta + modalities.yaml + changes.json
    for sub in ["splits", "meta.json", "modalities.yaml", "changes.json"]:
        src_p = os.path.join(args.src, sub)
        dst_p = os.path.join(args.dst, sub)
        if not os.path.exists(src_p):
            continue
        if os.path.isdir(src_p):
            if os.path.exists(dst_p):
                shutil.rmtree(dst_p)
            shutil.copytree(src_p, dst_p)
        else:
            shutil.copy2(src_p, dst_p)

    # Hard-link ALL pickles from v4 → v5
    n_linked = 0
    for fn in sorted(os.listdir(src_data)):
        if not fn.endswith(".pkl"):
            continue
        src_p = os.path.join(src_data, fn)
        dst_p = os.path.join(dst_data, fn)
        if os.path.exists(dst_p):
            continue
        try:
            os.link(src_p, dst_p)
            n_linked += 1
        except OSError:
            shutil.copy2(src_p, dst_p)
    print(f"[v5-sf] hard-linked {n_linked} v4 pickles (variants stay raw-rgb delta)")

    # Find base samples (no __aug marker) and replace weak modalities
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = json.load(open(os.path.join(args.src, "splits", f"{name}.json")))
    base_ids = set()
    for split_ids in splits.values():
        for sid in split_ids:
            if "__aug" not in sid:
                base_ids.add(sid)

    n_replaced = 0
    feature_dims = {}
    for base_id in sorted(base_ids):
        src_p = os.path.join(src_data, f"{base_id}.pkl")
        if not os.path.exists(src_p):
            continue
        with open(src_p, "rb") as f:
            d = pickle.load(f)
        if not isinstance(d, dict) or "modalities" not in d:
            continue
        s = Sample.from_dict(d)
        new_mods = dict(s.modalities)
        for m in _STRUCTURED_MODALITIES:
            if m in new_mods:
                feat = _EXTRACTORS[m](new_mods[m].data)
                new_mods[m] = new_mods[m].__class__(
                    data=feat,
                    frame_indices=list(range(feat.shape[0])),
                    sample_rate=new_mods[m].sample_rate,
                    name=m,
                )
                feature_dims[m] = int(feat.shape[0])
        new_s = Sample(id=s.id, label=s.label, modalities=new_mods,
                       text=s.text, meta=s.meta)
        dst_p = os.path.join(dst_data, f"{base_id}.pkl")
        # CRITICAL: dst_p may be a hard-link to the v4 source pickle (shared
        # inode). Writing in place would overwrite v4 too. Unlink first so the
        # new file gets its own inode, leaving v4 untouched.
        safe_replace_pickle(dst_p, new_s.to_dict())
        n_replaced += 1

    print(f"[v5-sf] replaced weak modalities in {n_replaced} base samples")
    print(f"[v5-sf] feature dims: {feature_dims}")

    # meta.json + changes.json
    meta_p = os.path.join(args.dst, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        meta.setdefault("changelog", []).insert(
            0,
            f"v5_structfeat: weak modalities (depth/wifi/lidar/mmwave) replaced "
            f"with domain-aware structured features ({feature_dims}). rgb "
            f"unchanged (keypoints). Probe v4 val: depth 0.08→0.24, "
            f"wifi 0.05→0.11, lidar 0.10→0.25, mmwave 0.35→0.51.")
        meta["version"] = "v5_structfeat"
        meta["n_samples"] = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        meta["source"] = dict(meta.get("source", {}), parent="mmfi/v4")
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)

    json.dump({
        "v4_to_v5_structfeat": {
            "strategy": "weak_modalities_to_structured_features",
            "structured_modalities": list(_STRUCTURED_MODALITIES),
            "rgb_unchanged": True,
            "feature_dims": feature_dims,
            "n_base_replaced": n_replaced,
        },
    }, open(os.path.join(args.dst, "changes.json"), "w"), indent=2)

    # report
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump({
        "src": args.src, "dst": args.dst,
        "n_base_replaced": n_replaced,
        "feature_dims": feature_dims,
    }, open(args.report, "w"), indent=2)
    print(f"[v5-sf] saved -> {args.report}")


if __name__ == "__main__":
    main()