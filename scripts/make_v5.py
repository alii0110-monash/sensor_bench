#!/usr/bin/env python
"""v5: copy v4 samples + fill train base samples' `Sample.text` with synthetic
captions. Variants (`__aug*`) share the base's text (resolved by loader).

Apples-to-apples: same data, same splits, same labels — only adds text.
"""
import argparse
import json
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation.caption.captioner import SyntheticCaptioner, TemplateCaptioner
from curation.caption.quality import check_captions
from curation.caption.verbs import LABEL_TO_VERB
from curation.version.version import write_meta

_VARIANT_MARKER = "__aug"


def _is_train_base(sid: str, train_ids) -> bool:
    return sid in train_ids and _VARIANT_MARKER not in sid


def add_captions(v4_root: str, v5_root: str, captioner: SyntheticCaptioner) -> dict:
    """Copy v4 -> v5, generating captions for train base samples. Returns stats."""
    data_out = os.path.join(v5_root, "data")
    os.makedirs(data_out, exist_ok=True)
    shutil.copytree(os.path.join(v4_root, "splits"), os.path.join(v5_root, "splits"),
                    dirs_exist_ok=True)

    train_ids = set(json.load(open(os.path.join(v4_root, "splits", "train.json"))))
    n_base = n_variant = n_fail = n_written_total = 0
    for fn in sorted(os.listdir(os.path.join(v4_root, "data"))):
        if not fn.endswith(".pkl"):
            continue
        sid = fn[:-4]
        src = os.path.join(v4_root, "data", fn)
        dst = os.path.join(data_out, fn)
        n_written_total += 1
        if _is_train_base(sid, train_ids):
            with open(src, "rb") as f:
                sample = pickle.load(f)
            texts = captioner.generate(sample)
            verb = LABEL_TO_VERB(sample["label"])
            issues = check_captions(texts, verb)
            if issues:
                n_fail += 1
            sample = dict(sample)
            sample["text"] = dict(sample.get("text", {}))
            sample["text"]["en"] = texts
            with open(dst, "wb") as f:
                pickle.dump(sample, f)
            n_base += 1
        else:
            shutil.copy(src, dst)
            if _VARIANT_MARKER in sid:
                n_variant += 1

    write_meta(v5_root, name="mmfi", version="v5",
               changelog=[f"v5: synthetic captions for {n_base} train base samples "
                          "(variants share base text); no data change"],
               n_samples=n_written_total, n_modalities=5,
               source={"dataset": "MMFi", "split": "cs", "parent": "mmfi/v4"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"based_on": "mmfi/v4", "captioning": "TemplateCaptioner(n=3)"})
    return {"n_base": n_base, "n_variant": n_variant, "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", default="datasets/mmfi/v4")
    ap.add_argument("--v5", default="datasets/mmfi/v5")
    ap.add_argument("--n", type=int, default=3, help="captions per base sample")
    ap.add_argument("--captioner", choices=["template"], default="template")
    args = ap.parse_args()
    captioner = TemplateCaptioner(n=args.n)
    stats = add_captions(args.v4, args.v5, captioner)
    print(f"v5: base={stats['n_base']} variant={stats['n_variant']} fail={stats['n_fail']}", flush=True)


if __name__ == "__main__":
    main()
