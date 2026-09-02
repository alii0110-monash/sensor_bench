"""Safe filesystem helpers for dataset build scripts.

The make_* scripts hard-link source pickles into a destination dataset to
save disk/time. Writing in place to a hard-linked path shares the source
inode, so overwriting the destination also corrupts the source (this bit us
twice on v4). Always unlink the destination first so the new file gets its
own inode, leaving the source untouched.
"""
from __future__ import annotations

import os
import pickle
from typing import Any


def safe_replace_pickle(path: str, obj: Any) -> None:
    """Write ``obj`` to ``path`` without clobbering a hard-linked source.

    If ``path`` is a hard-link to another file (shared inode), unlink it
    first so the write creates a fresh inode. The linked source is left
    untouched.
    """
    if os.path.exists(path):
        os.unlink(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
