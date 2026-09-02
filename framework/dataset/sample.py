from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Modality:
    """One sensor stream of one sample. data.shape[0] == len(frame_indices)."""
    data: np.ndarray
    frame_indices: List[int]
    sample_rate: int = 0
    name: str = ""

    @property
    def shape(self):
        return list(self.data.shape)

    def __post_init__(self):
        self.frame_indices = [int(i) for i in self.frame_indices]
        if len(self.frame_indices) != self.data.shape[0]:
            raise ValueError(
                f"frame_indices ({len(self.frame_indices)}) must match data.shape[0] "
                f"({self.data.shape[0]})")


@dataclass
class Sample:
    """Canonical sample contract. Models consume this dict-like object."""
    id: str
    label: int
    modalities: Dict[str, Modality]
    text: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.modalities:
            raise ValueError("sample must have at least one modality")
        if not (0 <= self.label < 1000):
            raise ValueError(f"label out of range: {self.label}")
        for name, mod in self.modalities.items():
            mod.name = name

    def available_modalities(self) -> List[str]:
        return list(self.modalities.keys())

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "modalities": {
                name: {"data": m.data, "frame_indices": m.frame_indices,
                       "sample_rate": m.sample_rate, "name": name}
                for name, m in self.modalities.items()
            },
            "text": self.text,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d):
        mods = {
            name: Modality(data=np.asarray(mm["data"]),
                           frame_indices=mm["frame_indices"],
                           sample_rate=mm.get("sample_rate", 0), name=name)
            for name, mm in d["modalities"].items()
        }
        return cls(id=d["id"], label=d["label"], modalities=mods,
                   text=d.get("text", {}), meta=d.get("meta", {}))
