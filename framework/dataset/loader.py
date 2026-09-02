from __future__ import annotations
import json
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Union

from .sample import Sample

_SPLIT_NAMES = ["train", "val", "test"]


def _split_names(root: str) -> List[str]:
    """Split names from splits/*.json (all of them, not just train/val/test),
    so curated splits like 'gold' load through the standard path."""
    splits_dir = os.path.join(root, "splits")
    if os.path.isdir(splits_dir):
        names = [f[:-5] for f in sorted(os.listdir(splits_dir)) if f.endswith(".json")]
        if names:
            return names
    return list(_SPLIT_NAMES)


class Dataset:
    def __init__(self, root: str, splits: Dict[str, Union[List[Sample], "LazySplit"]],
                 modalities: List[str]):
        self.root = root
        self.splits: Dict[str, Union[List[Sample], "LazySplit"]] = splits
        self.modalities = modalities

    def __getattr__(self, name):
        if name in self.splits:
            return self.splits[name]
        raise AttributeError(name)

    @property
    def meta(self) -> dict:
        p = os.path.join(self.root, "meta.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def __repr__(self):
        n = {k: len(v) for k, v in self.splits.items()}
        return f"Dataset(root={self.root}, splits={n}, modalities={self.modalities})"


def _read_sample(p: str) -> Sample:
    with open(p, "rb") as f:
        return Sample.from_dict(pickle.load(f))


_VARIANT_MARKER = "__aug"


def _is_variant_raw(raw: dict) -> bool:
    return isinstance(raw, dict) and raw.get("kind") == "variant"


def _resolve_variant(raw: dict, base: Sample) -> Sample:
    """Reconstruct a variant Sample from its base + rgb delta."""
    d = base.to_dict()
    d["id"] = raw["id"]
    d["label"] = raw.get("label", base.label)
    d["modalities"]["rgb"] = raw["rgb"]
    d["meta"] = dict(d.get("meta", {}))
    d["meta"]["aug"] = raw["aug"]
    return Sample.from_dict(d)


# ---------------------------------------------------------------------------
# Memory pre-flight (先量后跑): estimate before loading anything.
# ---------------------------------------------------------------------------

def estimate_dataset_bytes(root: str) -> int:
    """Sum of on-disk pickle sizes in data/ — a tight upper bound of the
    in-RAM numpy footprint (pickle overhead is ~0 for ndarrays)."""
    data_dir = os.path.join(root, "data")
    total = 0
    if not os.path.isdir(data_dir):
        return 0
    for fn in os.listdir(data_dir):
        if fn.endswith(".pkl"):
            total += os.path.getsize(os.path.join(data_dir, fn))
    return total


def available_memory_bytes() -> int:
    """MemAvailable + SwapFree (Linux). Falls back to sysconf on other OSes."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0]) * 1024
        return info.get("MemAvailable", 0) + info.get("SwapFree", 0)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return pages * size if pages > 0 and size > 0 else 0
    except (ValueError, OSError):
        return 0


def preflight_dataset(root: str) -> dict:
    """Return {n_files, estimated_bytes, available_bytes, fits, mode}. Cheap:
    file stat only, no pickle loading."""
    data_dir = os.path.join(root, "data")
    files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")] if os.path.isdir(data_dir) else []
    estimated = sum(os.path.getsize(os.path.join(data_dir, f)) for f in files)
    avail = available_memory_bytes()
    fits = estimated <= avail
    return {"n_files": len(files), "estimated_bytes": estimated,
            "available_bytes": avail, "fits": fits,
            "mode": "lazy" if not fits else "eager"}


# ---------------------------------------------------------------------------
# Lazy split: on-demand per-sample loading with a bounded LRU cache. Keeps the
# list-like interface (len / int-index / slice / iteration) so models and
# harness code work unchanged.
# ---------------------------------------------------------------------------

class LazySplit:
    def __init__(self, data_dir: str, ids: List[str], cache_size: int = 256,
                 base_loader=None):
        self._data_dir = data_dir
        self._ids = list(ids)
        self._cache: "OrderedDict[str, Sample]" = OrderedDict()
        self._cache_size = max(cache_size, 1)
        self._base_loader = base_loader  # callable(base_id) -> Sample, for variants

    def __len__(self) -> int:
        return len(self._ids)

    def _load(self, i: int) -> Sample:
        sid = self._ids[i]
        cached = self._cache.get(sid)
        if cached is not None:
            self._cache.move_to_end(sid)
            return cached
        with open(os.path.join(self._data_dir, f"{sid}.pkl"), "rb") as f:
            raw = pickle.load(f)
        if _is_variant_raw(raw):
            base = self._base_loader(raw["base_id"])
            s = _resolve_variant(raw, base)
        else:
            s = Sample.from_dict(raw)
        self._cache[sid] = s
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return s

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._load(i) for i in range(*key.indices(len(self)))]
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(key)
            return self._load(key)
        raise TypeError(f"LazySplit indices must be int or slice, got {type(key)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self._load(i)

    def __contains__(self, item):
        return item in self._ids


def _existing_ids(root: str, split_name: str) -> List[str]:
    """Read a split's ids, filtered to files that actually exist on disk."""
    p = os.path.join(root, "splits", f"{split_name}.json")
    if not os.path.exists(p):
        return []
    ids = json.load(open(p))
    return [i for i in ids if os.path.exists(os.path.join(root, "data", f"{i}.pkl"))]


def _resolve_modalities(root: str, data_dir: str) -> List[str]:
    """Prefer modalities.yaml (cheap); fall back to the first sample on disk."""
    yaml_p = os.path.join(root, "modalities.yaml")
    if os.path.exists(yaml_p):
        mods = []
        for line in open(yaml_p):
            line = line.strip()
            if line.startswith("- "):
                mods.append(line[2:].strip())
        if mods:
            return mods
    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".pkl") and _VARIANT_MARKER not in fn:
            return list(_read_sample(os.path.join(data_dir, fn)).modalities.keys())
    return []


def _make_base_loader(data_dir: str, cache_size: int):
    """Return callable(base_id) -> Sample for variant reconstruction, with its own
    bounded LRU cache. Base files are never variants themselves."""
    cache: "OrderedDict[str, Sample]" = OrderedDict()

    def load(base_id: str) -> Sample:
        cached = cache.get(base_id)
        if cached is not None:
            cache.move_to_end(base_id)
            return cached
        s = _read_sample(os.path.join(data_dir, f"{base_id}.pkl"))
        cache[base_id] = s
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return s

    return load


def load_dataset(root: str, mode: str = "auto", cache_size: int = 256) -> Dataset:
    """Load a dataset from `root`.

    mode:
      - "auto":  estimate in-RAM footprint vs MemAvailable+SwapFree; use lazy
                 loading when it would not fit (default). Refuses to eager-load
                 an oversized dataset.
      - "eager": load everything into RAM as before. Raises MemoryError if the
                 estimate exceeds available memory (先量后跑 guardrail).
      - "lazy":  always use LazySplit (on-demand per-sample loading).

    v4 variant files (kind="variant", written by make_v4.py) hold only the rgb
    delta + base_id; the loader reconstructs the full Sample from its base."""
    if mode not in ("auto", "eager", "lazy"):
        raise ValueError(f"mode must be auto/eager/lazy, got {mode!r}")

    data_dir = os.path.join(root, "data")
    pre = preflight_dataset(root)

    if mode == "eager" and not pre["fits"]:
        raise MemoryError(
            f"[loader] dataset needs ~{pre['estimated_bytes']/1e9:.1f}GB but only "
            f"{pre['available_bytes']/1e9:.1f}GB available (RAM+swap). "
            f"Use mode='lazy' (on-demand loading) or shrink the dataset.")
    if mode == "auto":
        mode = "lazy" if not pre["fits"] else "eager"

    if mode == "lazy":
        if not pre["fits"]:
            print(f"[loader] WARN: {pre['n_files']} samples ~{pre['estimated_bytes']/1e9:.1f}GB "
                  f"> {pre['available_bytes']/1e9:.1f}GB available → lazy loading "
                  f"(on-demand per batch, memory-safe)")
        base_loader = _make_base_loader(data_dir, cache_size)
        splits: Dict[str, Union[List[Sample], LazySplit]] = {}
        for name in _split_names(root):
            ids = _existing_ids(root, name)
            splits[name] = LazySplit(data_dir, ids, cache_size=cache_size,
                                     base_loader=base_loader) if ids else []
        return Dataset(root, splits, _resolve_modalities(root, data_dir))

    cache: Dict[str, Sample] = {}
    files = sorted(os.listdir(data_dir))
    for fn in files:
        if fn.endswith(".pkl"):
            sid = fn[:-4]
            if _VARIANT_MARKER in sid:
                continue  # variants resolved in a second pass against bases
            s = _read_sample(os.path.join(data_dir, fn))
            cache[s.id] = s

    for fn in files:
        if not fn.endswith(".pkl") or _VARIANT_MARKER not in fn:
            continue
        with open(os.path.join(data_dir, fn), "rb") as f:
            raw = pickle.load(f)
        if _is_variant_raw(raw) and raw["base_id"] in cache:
            cache[raw["id"]] = _resolve_variant(raw, cache[raw["base_id"]])

    splits = {}
    for name in _split_names(root):
        p = os.path.join(root, "splits", f"{name}.json")
        if os.path.exists(p):
            ids = json.load(open(p))
            splits[name] = [cache[i] for i in ids if i in cache]

    modalities = []
    for s in cache.values():
        for m in s.modalities:
            if m not in modalities:
                modalities.append(m)
    return Dataset(root, splits, modalities)
