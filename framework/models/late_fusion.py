from __future__ import annotations
import random
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from .base import SensorModel, TrainConfig
from .batching import build_batch_indexer, class_weights
from .encoders import D, WifiEncoder, DepthEncoder, PointEncoder, MLPEncoder
from .token_fusion import MODALITIES, _build_encoders


class LateFusionModel(nn.Module, SensorModel):
    """Baseline: per-modality encoder -> single vector; missing -> zero vector;
    concat -> MLP head. No alignment mechanism (control baseline).

    `structured` (dict modality -> feat_dim) switches those modalities to
    MLPEncoder for 1-D structured features (v5_structfeat); persisted in the
    checkpoint so load() reconstructs the same architecture."""

    name = "late_fusion"

    def __init__(self, num_classes: int = 27, structured: dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.structured = dict(structured) if structured else {}
        self.encoders = _build_encoders(self.structured)
        self.head = nn.Sequential(
            nn.Linear(D * len(MODALITIES), 512), nn.ReLU(), nn.Linear(512, num_classes))

    def forward(self, mods, avail):
        B = next(iter(mods.values())).shape[0] if mods else 1
        feats = []
        for m in MODALITIES:
            if avail.get(m):
                feats.append(self.encoders[m](mods[m]).mean(dim=1))
            else:
                feats.append(torch.zeros(B, D, device=next(self.parameters()).device))
        return self.head(torch.cat(feats, dim=1))

    def fit(self, train, val, cfg: TrainConfig) -> None:
        torch.manual_seed(cfg.seed)
        self.train().to(cfg.device)
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        labels = [s.label for s in train]
        w = class_weights(labels, self.num_classes, cfg.class_weight)
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32,
                                device=torch.device(cfg.device)))
        rng_shuffle = random.Random(cfg.seed)
        best = -1.0
        for ep in range(cfg.epochs):
            self.train()
            if cfg.batch_strategy == "balanced":
                indexer = build_batch_indexer(
                    "balanced", labels, self.num_classes, cfg.batch_size,
                    cfg.seed * 1000 + ep)
                batches = indexer.batches()
            else:
                order = list(range(len(train)))
                rng_shuffle.shuffle(order)
                batches = ([order[i:i + cfg.batch_size]
                            for i in range(0, len(order), cfg.batch_size)])
            for idx in batches:
                batch = [train[j] for j in idx]
                avail = {m: True for m in MODALITIES}
                mods = self._stack_mods(batch, avail, cfg)
                lbl = torch.tensor([s.label for s in batch], device=cfg.device)
                loss = crit(self(mods, avail), lbl)
                opt.zero_grad(); loss.backward(); opt.step()
            v = self._evaluate(val, cfg)
            if v > best:
                best = v
                self.save(f"{cfg.out_dir}/{self.name}_seed{cfg.seed}.pt")
            print(f"[{self.name}] ep {ep} val {v:.3f} (best {best:.3f})")

    def _stack_mods(self, samples, avail, cfg):
        mods = {}
        for m in MODALITIES:
            if avail.get(m):
                arrs = [s.modalities[m].data for s in samples]
                if m in self.structured:
                    mods[m] = torch.from_numpy(np.stack(arrs)).to(cfg.device)
                else:
                    mods[m] = torch.stack(
                        [torch.from_numpy(a) for a in arrs]).to(cfg.device)
        return mods

    @torch.no_grad()
    def _evaluate(self, samples, cfg):
        self.eval()
        ok = tot = 0
        for i in range(0, len(samples), cfg.batch_size):
            batch = samples[i:i + cfg.batch_size]
            avail = {m: True for m in MODALITIES}
            mods = self._stack_mods(batch, avail, cfg)
            preds = self(mods, avail).argmax(-1).cpu().tolist()
            ok += sum(p == s.label for p, s in zip(preds, batch))
            tot += len(batch)
        return ok / max(tot, 1)

    def _device(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if next(self.parameters()).device.type != dev:
            self.to(dev)
        return next(self.parameters()).device

    @torch.no_grad()
    def predict(self, sample, available):
        self.eval()
        avail = {m: m in available for m in MODALITIES}
        dev = self._device()
        mods = {}
        for m in available:
            data = sample.modalities[m].data
            mods[m] = torch.from_numpy(data)[None].to(dev)
        probs = torch.softmax(self(mods, avail)[0], dim=-1)
        return {i: float(p) for i, p in enumerate(probs)}

    @torch.no_grad()
    def predict_batch(self, samples, available):
        self.eval()
        avail = {m: m in available for m in MODALITIES}
        dev = self._device()
        mods = {}
        for m in available:
            arrs = [s.modalities[m].data for s in samples]
            if m in self.structured:
                mods[m] = torch.from_numpy(np.stack(arrs)).to(dev)
            else:
                mods[m] = torch.stack(
                    [torch.from_numpy(a) for a in arrs]).to(dev)
        return self(mods, avail)

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "structured": self.structured,
                    "num_classes": self.num_classes}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")
        # Backward compat: pre-structured checkpoints are a bare state_dict.
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state, structured, num_classes = (
                ckpt["state_dict"], ckpt.get("structured", {}), ckpt.get("num_classes", 27))
        else:
            state, structured, num_classes = ckpt, {}, 27
        m = cls(num_classes=num_classes, structured=structured)
        m.load_state_dict(state)
        return m
