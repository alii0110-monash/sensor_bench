"""Streamlit caching wrappers for the GUI app.

Heavy, deterministic work (filesystem scans, dataset opening, split-id
resolution) is cached across reruns so a single widget interaction doesn't
re-scan disk or re-instantiate the lazy loader. The underlying pure functions
in dataset_service stay untouched and unit-testable; only these wrappers carry
the caching decorators, so importing them outside a Streamlit runtime is safe
(the decorated calls just execute once).
"""
from __future__ import annotations

from typing import List, Optional

import streamlit as st

from curation.gui.core.dataset_service import (
    find_quality_json,
    list_dataset_roots,
    open_dataset,
    split_ids,
)
from framework.dataset.loader import Dataset


@st.cache_resource(show_spinner=False, max_entries=8)
def get_dataset(root: str, split: Optional[str]) -> Dataset:
    """Cached dataset handle. A Dataset is a heavy lazy object reused across
    reruns for the same (dataset, split); caching avoids re-scanning the
    splits/ dir and re-instantiating the loader on every interaction."""
    return open_dataset(root, split)


@st.cache_data(show_spinner=False, max_entries=64)
def get_split_ids(root: str, split: str) -> List[str]:
    """Cached split ids (existing-on-disk filter included). Deterministic for
    a fixed dataset root; saves 46k `os.path.exists` calls per review render."""
    return split_ids(root, split)


@st.cache_data(show_spinner=False, max_entries=8)
def get_roots() -> List[str]:
    """Cached dataset-root discovery."""
    return list_dataset_roots()


@st.cache_data(show_spinner=False, max_entries=64)
def get_quality_json(root: str) -> Optional[str]:
    """Cached quality json path for a dataset root."""
    return find_quality_json(root)
