"""Build a golden dataset from an edit log.

Reads a JSONL edit log produced by the GUI, keeps samples whose final quality
mark is `golden`, applies their corrected text / label / note, and writes a
new Dataset-protocol root that the loader can read back.

The source dataset is never modified.

Usage:
    python curation/gui/scripts/build_gold.py \
        --dataset datasets/mmfi/v4 --split val \
        --edits curation/gui/edits/mmfi_v4-val.jsonl \
        --out datasets/mmfi/gold --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from framework.dataset.loader import load_dataset  # noqa: E402
from framework.dataset.sample import Sample  # noqa: E402

from curation.gui.core.edit_log import EditLog, QUALITY_CHOICES  # noqa: E402


def _copy_modalities_yaml(src_root: str, out_root: str) -> None:
    src = os.path.join(src_root, "modalities.yaml")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_root, "modalities.yaml"))


def build(args: argparse.Namespace) -> dict:
    ds = load_dataset(args.dataset, mode="lazy")
    split = ds.splits.get(args.split)
    if not split:
        raise SystemExit(f"split {args.split!r} empty in {args.dataset}")

    log = EditLog(args.edits)
    report = {
        "source": args.dataset,
        "source_split": args.split,
        "edits_file": args.edits,
        "out": args.out,
        "n_samples_source": len(split),
        "n_edited": len(log.edited_ids()),
        "n_golden": 0,
        "label_changes": 0,
        "text_changes": 0,
        "kept_ids": [],
        "quality_counts": {},
    }
    for q in QUALITY_CHOICES:
        report["quality_counts"][q] = 0

    kept: list = []
    for i in range(len(split)):
        s = split[i]
        fields = log.fields(s.id)
        q = fields.get("quality")
        if q not in QUALITY_CHOICES:
            continue
        report["quality_counts"][q] += 1
        if q != "golden":
            continue
        # apply corrections
        d = s.to_dict()
        if "text" in fields and fields["text"]:
            d["text"] = {"captions": list(fields["text"])}
            report["text_changes"] += 1
        if "label" in fields and fields["label"] is not None and fields["label"] != s.label:
            d["label"] = int(fields["label"])
            report["label_changes"] += 1
        d["meta"] = dict(d.get("meta", {}))
        d["meta"]["golden"] = True
        if fields.get("note"):
            d["meta"]["curation_note"] = fields["note"]
        kept.append((s.id, Sample.from_dict(d)))

    report["n_golden"] = len(kept)
    report["kept_ids"] = [sid for sid, _ in kept]

    if args.dry_run:
        return report

    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)
    for sid, sample in kept:
        with open(os.path.join(args.out, "data", f"{sid}.pkl"), "wb") as f:
            pickle.dump(sample.to_dict(), f)
    out_split = args.out_split
    with open(os.path.join(args.out, "splits", f"{out_split}.json"), "w") as f:
        json.dump([sid for sid, _ in kept], f)
    _copy_modalities_yaml(args.dataset, args.out)

    src_meta = ds.meta
    meta = {
        "name": src_meta.get("name", "mmfi"),
        "version": "gold",
        "changelog": [
            f"gold: human-curated golden subset from {args.dataset}/{args.split} "
            f"({len(kept)} samples, quality=golden)",
            f"edits file: {args.edits}",
        ],
        "n_samples": len(kept),
        "n_modalities": len(ds.modalities),
        "source": {"dataset": args.dataset, "split": args.split, "parent": args.dataset},
        "license": src_meta.get("license", ""),
        "collection_protocol": {
            "based_on": args.dataset,
            "curation": {
                "tool": "build_gold.py",
                "edits_file": args.edits,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="datasets/mmfi/v4")
    p.add_argument("--split", default="val")
    p.add_argument("--edits", required=True)
    p.add_argument("--out", default="datasets/mmfi/gold")
    p.add_argument("--out-split", default="gold")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    report = build(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("\n[dry-run] 未写盘。移除 --dry-run 以实际构建。")


if __name__ == "__main__":
    main()