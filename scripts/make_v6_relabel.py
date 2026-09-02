#!/usr/bin/env python
"""v6 = v5 + label corrections (Phase 1 diagnostic → Phase 2 action).

Diagnosis (results/v5_diagnostic.json) showed:
- class 14: 45 samples, 100% wrong across all subjects, all predicted as 15.
  -> relabel to 15 (strong evidence).
- class 9: 50% wrong, S35+S36 all zero, others fine.
  -> relabel S35+S36 samples to top-confusion target (class 8).

v6 reuses v5 (caption preserved) but writes new pickles only for affected
samples. Unaffected samples are hard-linked (same filesystem), so disk usage
is minimal.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curation.io import safe_replace_pickle


def _subject_of(sid: str) -> int:
    m = re.match(r"E\d+_S(\d+)_", sid)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/mmfi/v5")
    ap.add_argument("--dst", default="datasets/mmfi/v6")
    ap.add_argument("--report", default="results/v6_relabel_report.json")
    args = ap.parse_args()

    src_data = os.path.join(args.src, "data")
    dst_data = os.path.join(args.dst, "data")
    os.makedirs(dst_data, exist_ok=True)

    # Copy splits + meta unchanged
    for sub in ["splits", "meta.json", "modalities.yaml", "changes.json"]:
        src_p = os.path.join(args.src, sub)
        dst_p = os.path.join(args.dst, sub)
        if not os.path.exists(src_p):
            continue
        if os.path.isdir(src_p):
            if not os.path.exists(dst_p):
                shutil.copytree(src_p, dst_p)
        else:
            shutil.copy2(src_p, dst_p)

    # Walk all pickles; relabel if matched; hard-link otherwise
    relabel_log = []
    n_total = 0
    n_relabeled = 0
    n_linked = 0
    n_copied = 0
    for fn in sorted(os.listdir(src_data)):
        if not fn.endswith(".pkl"):
            continue
        n_total += 1
        src_p = os.path.join(src_data, fn)
        dst_p = os.path.join(dst_data, fn)

        # Load to check label (cheap on SSD; 18GB total over 52k files ~ 350KB avg)
        with open(src_p, "rb") as f:
            d = pickle.load(f)
        old_label = d["label"]
        sid = d["id"]
        subj = _subject_of(sid)

        new_label = old_label
        action = None
        if old_label == 14:
            new_label = 15
            action = "class14->15"
        elif old_label == 9 and subj in (35, 36):
            new_label = 8
            action = "class9_S35/S36->8"

        if new_label != old_label:
            d["label"] = new_label
            # dst_p may be a hard-link to src_p (shared inode); unlink first
            # so the relabeled file gets its own inode, leaving src untouched.
            safe_replace_pickle(dst_p, d)
            n_relabeled += 1
            relabel_log.append({"id": sid, "old": old_label,
                                "new": new_label, "action": action})
        else:
            try:
                os.link(src_p, dst_p)
                n_linked += 1
            except OSError:
                shutil.copy2(src_p, dst_p)
                n_copied += 1

    print(f"[v6] total pickles: {n_total}")
    print(f"[v6] relabeled: {n_relabeled}")
    print(f"[v6] hard-linked (no change): {n_linked}")
    print(f"[v6] copied (fallback): {n_copied}")

    # Write report
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({
            "src": args.src, "dst": args.dst,
            "n_total": n_total,
            "n_relabeled": n_relabeled,
            "n_linked": n_linked,
            "n_copied": n_copied,
            "relabels": relabel_log,
        }, f, indent=2)
    print(f"[v6] saved -> {args.report}")

    # Patch meta.json changelog
    meta_p = os.path.join(args.dst, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            meta = json.load(f)
        meta.setdefault("changelog", []).insert(
            0,
            f"v6: relabel {n_relabeled} samples based on v5 Phase 1 diagnostic "
            f"(class 14 -> 15 strong evidence; class 9 S35/S36 -> 8).")
        meta["version"] = "v6"
        meta["n_samples"] = n_total
        meta["source"] = dict(meta.get("source", {}), parent="mmfi/v5")
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[v6] meta.json updated")


if __name__ == "__main__":
    main()