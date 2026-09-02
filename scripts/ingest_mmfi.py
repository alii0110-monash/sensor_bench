#!/usr/bin/env python
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation.ingest.mmfi import annotation_to_sample, action_labels, write_sample
from framework.dataset.splits import split_annotations, build_val_subjects
from curation.version.version import write_meta
import yaml


def write_modalities(root: str, modalities: list) -> None:
    with open(os.path.join(root, "modalities.yaml"), "w") as f:
        yaml.safe_dump({"modalities": modalities,
                        "note": "modalities derived from samples; list is authoritative registry"}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-train", required=True)
    ap.add_argument("--annotations-test", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only ingest first N (smoke)")
    ap.add_argument("--no-val", action="store_true", help="skip building a val split (all train -> train)")
    args = ap.parse_args()

    train = json.load(open(args.annotations_train))
    test = json.load(open(args.annotations_test))
    if args.limit:
        train = train[:args.limit]
        test = test[:args.limit]

    val_subs = [] if args.no_val else build_val_subjects(train, n_val_subjects=5)
    train_a, val_a = split_annotations(train, val_subjects=val_subs)
    test_a, _ = split_annotations(test, val_subjects=[])
    labels = action_labels()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    for name, anns in [("train", train_a), ("val", val_a), ("test", test_a)]:
        n = 0
        for ann in anns:
            try:
                s = annotation_to_sample(ann, args.raw_root, labels)
                write_sample(args.out, s)
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {ann['sample_id']}: {e}")
        os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)
        with open(os.path.join(args.out, "splits", f"{name}.json"), "w") as f:
            json.dump([ann["sample_id"] for ann in anns], f)
        print(f"{name}: {n} samples ({time.time()-t0:.0f}s)")

    write_meta(args.out, name="mmfi", version="v1",
               changelog=["initial ingest of wifi/mmwave/lidar/depth, cs split",
                          "known simplifications: splits are plain id lists (subject/env are in sample.meta)"],
               n_samples=len(train_a) + len(val_a) + len(test_a),
               n_modalities=4, source={"dataset": "MMFi", "split": "cs"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"envs": "E01-E04", "subjects": 40, "actions": 27,
                                    "frames_per_sample": 5, "sample_frames": "deterministic uniform"})
    write_modalities(args.out, ["wifi", "depth", "lidar", "mmwave"])


if __name__ == "__main__":
    main()
