from __future__ import annotations
import json
import os
from typing import List


def write_meta(root: str, name: str, version: str, changelog: List[str],
               n_samples: int, n_modalities: int, source: dict,
               license: str = "unknown", collection_protocol: dict = None) -> None:
    os.makedirs(root, exist_ok=True)
    meta = {
        "name": name, "version": version,
        "changelog": changelog,
        "n_samples": n_samples, "n_modalities": n_modalities,
        "source": source, "license": license,
        "collection_protocol": collection_protocol or {},
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
