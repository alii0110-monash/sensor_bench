#!/usr/bin/env python
import argparse, json, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from curation.clean.consistency import flag_inconsistent
from curation.version.version import write_meta
from framework.models.token_fusion import TokenFusionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="datasets/mmfi/v1")
    ap.add_argument("--v2", default="datasets/mmfi/v2")
    ap.add_argument("--ckpt", default="checkpoints/token_fusion_seed0.pt")
    ap.add_argument("--drop-rate", type=float, default=0.05)
    args = ap.parse_args()

    ds = load_dataset(args.v1)
    model = TokenFusionModel.load(args.ckpt)
    dropped = set()
    # Clean train/val only. Test is NEVER filtered: v1-vs-v2 must be
    # evaluated on the identical, unchanged test set (apples-to-apples).
    for split in ["train", "val"]:
        flagged = flag_inconsistent(model, ds.splits[split], drop_rate=args.drop_rate)
        dropped.update(flagged)
        print(f"{split}: flagged {len(flagged)}", flush=True)

    # copy data files minus dropped
    os.makedirs(f"{args.v2}/data", exist_ok=True)
    kept = 0
    for fn in os.listdir(f"{args.v1}/data"):
        if fn.replace(".pkl", "") not in dropped:
            shutil.copy(os.path.join(args.v1, "data", fn), os.path.join(args.v2, "data", fn))
            kept += 1
    shutil.copytree(f"{args.v1}/splits", f"{args.v2}/splits", dirs_exist_ok=True)
    shutil.copy(os.path.join(args.v1, "modalities.yaml"), os.path.join(args.v2, "modalities.yaml"))
    json.dump({"v1_to_v2": {"dropped": len(dropped), "reason": "cross-modal consistency filter"},
               "kept": kept}, open(f"{args.v2}/changes.json", "w"), indent=2)
    write_meta(args.v2, name="mmfi", version="v2",
               changelog=[f"v2: dropped {len(dropped)} train/val samples flagged as cross-modality "
                          "inconsistent by token_fusion model (drop_rate=0.05); test unchanged"],
               n_samples=kept, n_modalities=4,
               source={"dataset": "MMFi", "split": "cs", "parent": "mmfi/v1"},
               license="MMFi dataset license (NTU); see https://github.com/ybhbingo/MMFi_dataset",
               collection_protocol={"based_on": "mmfi/v1"})
    print(f"v2: kept {kept} samples", flush=True)


if __name__ == "__main__":
    main()
