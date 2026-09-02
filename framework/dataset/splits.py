from __future__ import annotations
import os
from typing import Dict, List, Tuple


def _subject_of(video_path: str) -> str:
    return video_path.split("/")[-2]


def _env_of(video_path: str) -> str:
    return video_path.split("/")[-3]


def _sample_id(video_path: str, start: int, end: int) -> str:
    return f"{video_path.split('/')[-3]}_{_subject_of(video_path)}_{video_path.split('/')[-1]}_f{start}-{end}"


def split_annotations(anns: List[dict], val_subjects: List[str]) -> Tuple[List[dict], List[dict]]:
    """Split annotation list into (train, val). val is defined by val_subjects.
    Adds 'sample_id' to each annotation. Returns copies, does not mutate input."""
    train, val = [], []
    val_set = set(val_subjects)
    for a in anns:
        item = dict(a)
        item["sample_id"] = _sample_id(a["video_path"], a["start_index"], a["end_index"])
        if _subject_of(a["video_path"]) in val_set:
            val.append(item)
        else:
            train.append(item)
    return train, val


def build_val_subjects(train_anns: List[dict], n_val_subjects: int = 5) -> List[str]:
    subs = sorted({_subject_of(a["video_path"]) for a in train_anns})
    return subs[-n_val_subjects:]
