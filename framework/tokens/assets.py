"""伪 token 资产化: CanonicalToken 落盘 npz + index.json (spec M6a 组件 3)."""
from __future__ import annotations
import json
import os
from typing import Dict, List

import numpy as np

from .canonical import CanonicalToken


def write_tokens(tokens: List[CanonicalToken], root: str, version: str,
                 encoder_ckpt: str) -> Dict:
    """落盘: {root}/tokens/{id}.npz + {root}/index.json. 返回 index dict."""
    tok_dir = os.path.join(root, "tokens")
    os.makedirs(tok_dir, exist_ok=True)
    samples = {}
    for t in tokens:
        t.validate()
        np.savez_compressed(os.path.join(tok_dir, f"{t.id}.npz"),
                            data=t.data, modality_order=t.modality_order,
                            label=t.label, k=t.k)
        samples[t.id] = {"label": t.label, "k": t.k,
                         "modality_order": t.modality_order}
    index = {"version": version, "encoder_ckpt": encoder_ckpt,
             "generated_at": __import__("datetime").datetime.now().isoformat(),
             "n_samples": len(tokens), "samples": samples}
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return index


def load_tokens(root: str) -> Dict[str, CanonicalToken]:
    """加载: {root}/index.json + {root}/tokens/*.npz → {id: CanonicalToken}."""
    idx_p = os.path.join(root, "index.json")
    if not os.path.exists(idx_p):
        return {}
    index = json.load(open(idx_p))
    tok_dir = os.path.join(root, "tokens")
    out = {}
    for sid in index.get("samples", {}):
        npz = np.load(os.path.join(tok_dir, f"{sid}.npz"))
        meta = index["samples"][sid]
        out[sid] = CanonicalToken(
            id=sid, label=int(meta["label"]),
            data=npz["data"].astype(np.float32),
            modality_order=[str(m) for m in npz["modality_order"]],
            k=int(meta["k"]),
            meta={"encoder_version": index.get("encoder_ckpt", "")})
    return out


def write_index(index: Dict, root: str) -> None:
    """独立更新 index.json (溯源单源)."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
