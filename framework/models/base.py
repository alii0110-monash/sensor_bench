from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: Optional[int] = 0
    device: str = "cuda"
    out_dir: str = "checkpoints"
    modality_dropout_p: float = 0.25
    eval_steps: int = 100
    patience: int = 5
    batch_strategy: str = "shuffle"  # "shuffle" | "balanced"
    class_weight: str = "none"  # "none" | "inverse_freq" | "sqrt_inverse_freq"
    modality_dropout: Optional[Dict[str, float]] = None  # per-modality dropout p override
    time_mask_p: float = 0.0  # per-frame time masking probability (temporal=True).
        # During training, zero out a random contiguous run of frames on raw
        # multi-frame modalities, forcing the causal aggregator to reconstruct
        # the masked frame from earlier context (对标 MiniMind-O 时间遮蔽).


class SensorModel:
    """The ONLY contract between framework and any model implementation.
    Models are trained on a Dataset, predict per-sample given an `available`
    modality list. Missing-modality behavior is entirely the model's concern."""

    name: str = "sensor_model"

    def fit(self, train, val, cfg: TrainConfig) -> None:
        raise NotImplementedError

    def predict(self, sample, available: List[str]) -> Dict[int, float]:
        """Returns {class_id: prob}. available = subset of dataset.modalities."""
        raise NotImplementedError

    @torch.no_grad()
    def predict_batch(self, samples: List, available: List[str]) -> "torch.Tensor":
        """Batched predict -> (B, num_classes) logits.

        Default: loop over predict(). Models override for GPU utilization
        (batch=1 inference underutilizes the GPU)."""
        import torch
        return torch.stack([torch.as_tensor(
            list(self.predict(s, available).values())) for s in samples])

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "SensorModel":
        raise NotImplementedError
