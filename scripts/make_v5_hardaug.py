#!/usr/bin/env python
"""v5: v4 + extra spatial augmentation for hard-class training base samples.

Hard classes identified by v5 Phase 1 diagnostic (acc < 0.55 on v4 val with
token_fusion): 4, 9, 14, 20, 21, 22. For each train base sample in these
classes, add 8 more variants (aug indices 4..11) using the existing
augment_keypoints function (flip/translate/scale).

Storage: hard-link v4 pickles (no copy of base + existing variants), only
write new delta pickles for the extra variants. Updated train.json adds the
new variant ids. val/test unchanged.

Expected gain: v4 val acc_full 0.808 → v5 should improve hard-class acc;
overall acc_full expected to rise (modest) and Quality on dataset_quality to
reflect richer augmentation.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation.enrich.keypoints import augment_keypoints, normalize_keypoints
from curation.io import safe_replace_pickle
from curation.version.version import write_meta


_HARD_CLASSES = [4, 9, 14, 20, 21, 22]


def _variant_id(base_id: str, k: int) -> str:
    return f"{base_id}__aug{k}"


def _subject_of(sid: str) -> int:
    m = re.match(r"E\d+_S(\d+)_", sid)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/mmfi/v4")
    ap.add_argument("--dst", default="datasets/mmfi/v5_hardaug")
    ap.add_argument("--n-extra-aug", type=int, default=8,
                    help="Extra variants per hard-class base sample (aug indices "
                         "4..3+n_extra_aug)")
    ap.add_argument("--aug-start", type=int, default=4,
                    help="Start index for new aug ids (v4 uses 0..3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="results/v5_creation_report.json")
    args = ap.parse_args()

    src_data = os.path.join(args.src, "data")
    dst_data = os.path.join(args.dst, "data")
    os.makedirs(dst_data, exist_ok=True)

    # Copy splits dir + meta + modalities.yaml (will modify train.json later)
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

    # Load splits
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = json.load(open(os.path.join(args.src, "splits", f"{name}.json")))
    train_ids = set(splits["train"])
    print(f"[v5] train: {len(train_ids)}, val: {len(splits['val'])}, "
          f"test: {len(splits['test'])}")

    # Hard-link ALL source pickles into dst (saves disk + time)
    n_linked = 0
    for fn in sorted(os.listdir(src_data)):
        if not fn.endswith(".pkl"):
            continue
        src_p = os.path.join(src_data, fn)
        dst_p = os.path.join(dst_data, fn)
        if os.path.exists(dst_p):
            continue  # already linked/copied (e.g. base used for delta)
        try:
            os.link(src_p, dst_p)
            n_linked += 1
        except OSError:
            shutil.copy2(src_p, dst_p)
    print(f"[v5] hard-linked {n_linked} source pickles")

    # Walk train base samples; identify hard-class bases
    train_base = [sid for sid in splits["train"]
                  if "__aug" not in sid]
    print(f"[v5] train base samples: {len(train_base)}")

    hard_base_with_label = []
    n_missing_base = 0
    for sid in train_base:
        src_p = os.path.join(src_data, f"{sid}.pkl")
        if not os.path.exists(src_p):
            n_missing_base += 1
            continue
        with open(src_p, "rb") as f:
            d = pickle.load(f)
        if d["label"] in _HARD_CLASSES:
            hard_base_with_label.append((sid, d["label"]))
    print(f"[v5] train base samples missing pickle (skipped): {n_missing_base}")
    print(f"[v5] hard-class base samples: {len(hard_base_with_label)} "
          f"(classes {sorted(set(l for _, l in hard_base_with_label))})")

    # Generate extra variants per hard-class base
    rng = np.random.default_rng(args.seed)
    new_variant_ids = []
    n_new = 0
    n_skip_conflict = 0
    for base_id, label in hard_base_with_label:
        with open(os.path.join(src_data, f"{base_id}.pkl"), "rb") as f:
            base_sample = pickle.load(f)
        # Normalize rgb again (deterministic — same as v4)
        rgb_norm = normalize_keypoints(base_sample["modalities"]["rgb"]["data"])
        for k_offset in range(args.n_extra_aug):
            k = args.aug_start + k_offset
            vid = _variant_id(base_id, k)
            dst_p = os.path.join(dst_data, f"{vid}.pkl")
            if os.path.exists(dst_p):
                n_skip_conflict += 1
                continue
            rgb_aug = augment_keypoints(rgb_norm, rng)
            delta = {
                "kind": "variant",
                "id": vid,
                "base_id": base_id,
                "label": label,
                "rgb": {"data": rgb_aug,
                        "frame_indices": list(base_sample["modalities"]["rgb"]["frame_indices"]),
                        "sample_rate": base_sample["modalities"]["rgb"]["sample_rate"]},
                "aug": k,
            }
            # dst_p is a fresh variant id (never hard-linked), but use the
            # safe write path so a stale/conflicting file can't clobber a
            # hard-linked source.
            safe_replace_pickle(dst_p, delta)
            new_variant_ids.append(vid)
            n_new += 1

    print(f"[v5] new variants written: {n_new} (skipped conflicts: {n_skip_conflict})")

    # Update train.json
    new_train = list(splits["train"]) + sorted(new_variant_ids)
    splits["train"] = new_train
    with open(os.path.join(args.dst, "splits", "train.json"), "w") as f:
        json.dump(new_train, f)
    print(f"[v5] new train total: {len(new_train)} "
          f"(was {len(train_ids)}, added {len(new_variant_ids)})")

    # Update meta.json
    meta_p = os.path.join(args.dst, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        meta.setdefault("changelog", []).insert(
            0,
            f"v5: extra {args.n_extra_aug} spatial variants per hard-class train "
            f"base (classes {_HARD_CLASSES}, total +{n_new} samples). "
            f"Built on v4 (v3 + rgb-norm + n_aug=4).")
        meta["version"] = "v5"
        meta["n_samples"] = len(new_train) + len(splits["val"]) + len(splits["test"])
        meta["source"] = dict(meta.get("source", {}), parent="mmfi/v4")
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)

    # changes.json
    json.dump({
        "v4_to_v5": {
            "strategy": "extra_aug_for_hard_classes",
            "hard_classes": _HARD_CLASSES,
            "n_extra_aug_per_base": args.n_extra_aug,
            "aug_start": args.aug_start,
            "aug_kind": "flip+translate+scale (same as v4)",
            "n_hard_base": len(hard_base_with_label),
            "n_new_variants": n_new,
            "n_conflict_skip": n_skip_conflict,
        },
    }, open(os.path.join(args.dst, "changes.json"), "w"), indent=2)

    # Report
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump({
        "src": args.src, "dst": args.dst,
        "hard_classes": _HARD_CLASSES,
        "n_hard_base": len(hard_base_with_label),
        "n_new_variants": n_new,
        "n_total_train_after": len(new_train),
        "per_class_new_counts": dict(_class_counts(new_variant_ids, splits)),
    }, open(args.report, "w"), indent=2)
    print(f"[v5] saved -> {args.report}")


def _class_counts(new_variant_ids, splits):
    """Count new variants per label by reading the delta pickle."""
    from framework.dataset.loader import _VARIANT_MARKER  # noqa: F401
    counts = defaultdict(int)
    for vid in new_variant_ids:
        base_id = vid.split("__aug")[0]
        p = f"datasets/mmfi/v4/data/{base_id}.pkl"
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        counts[d["label"]] += 1
    return counts


if __name__ == "__main__":
    main()