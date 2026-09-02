from __future__ import annotations
import random
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from .base import SensorModel, TrainConfig
from .batching import build_batch_indexer, class_weights
from .encoders import D, N_TOK, WifiEncoder, DepthEncoder, PointEncoder, MLPEncoder
from .domain_encoder import DomainEncoder, _DOMAIN_EXTRACTORS
from .temporal import TemporalAggregator

MODALITIES = ["wifi", "depth", "lidar", "mmwave", "rgb"]


def _build_encoders(structured: dict, domain: dict = None,
                    domain_dims: dict = None, temporal: bool = False) -> nn.ModuleDict:
    """Build per-modality encoders.

    Priority per modality:
      1. `domain` (modality in domain): DomainEncoder — raw data → domain features
         (extract_*_features on the fly) → MLP. ``domain_dims[modality]`` gives the
         extracted feature dim (deterministic per dataset).
      2. `structured` (modality in structured): MLPEncoder for pre-extracted 1-D
         structured features (v5_structfeat).
      3. else: the raw-data encoder (Wifi/Depth/Point). With `temporal=True`,
         raw encoders keep the time axis and return (B, T, N_TOK, D) for later
         TemporalAggregator.
    """
    encoders = {}
    for m in MODALITIES:
        if domain and m in domain:
            encoders[m] = DomainEncoder(m, domain_dims.get(m) if domain_dims else 64)
        elif structured and m in structured:
            encoders[m] = MLPEncoder(structured[m])
        elif m == "wifi":
            encoders[m] = WifiEncoder(temporal=temporal)
        elif m == "depth":
            encoders[m] = DepthEncoder(temporal=temporal)
        elif m == "lidar":
            encoders[m] = PointEncoder(3, temporal=temporal)
        elif m == "mmwave":
            encoders[m] = PointEncoder(5, temporal=temporal)
        else:  # rgb
            encoders[m] = PointEncoder(2, temporal=temporal)
    return nn.ModuleDict(encoders)


class TokenFusionModel(nn.Module, SensorModel):
    """Unified token fusion: per-modality encoder -> 16 tokens each, shared
    transformer, mean-pool, classification head. Missing modality = learned
    [MISSING] embedding + masked in attention. Trained with modality dropout.

    `structured` (dict modality -> feat_dim) switches those modalities to
    MLPEncoder for 1-D structured features (v5_structfeat); raw modalities
    keep their native encoders. The config is persisted in the checkpoint so
    load() reconstructs the same architecture."""

    name = "token_fusion"

    def __init__(self, num_classes: int = 27, d: int = D, n_layers: int = 2,
                 n_heads: int = 4, structured: dict = None, domain: dict = None,
                 domain_dims: dict = None, temporal: bool = False):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.temporal = temporal
        self.structured = dict(structured) if structured else {}
        self.domain = dict(domain) if domain else {}
        self.domain_dims = dict(domain_dims) if domain_dims else {}
        self.encoders = _build_encoders(self.structured, self.domain, self.domain_dims,
                                        temporal=temporal)
        self.missing = nn.ParameterDict({
            m: nn.Parameter(torch.randn(N_TOK, d) * 0.02) for m in MODALITIES})
        # per-modality temporal aggregator (only when temporal=True; raw multi-frame
        # encoders return (B,T,N,D) then aggregate here). Only built when temporal=True
        # so old non-temporal checkpoints keep a clean state_dict.
        self.temporal_agg = nn.ModuleDict({
            m: TemporalAggregator(d, n_heads) for m in MODALITIES}) if temporal else nn.ModuleDict()
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=4 * d, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.1)
        self.fusion = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.head = nn.Linear(d, num_classes)

    def forward(self, mods: Dict[str, torch.Tensor], avail: Dict[str, bool]) -> torch.Tensor:
        B = next(iter(mods.values())).shape[0]
        toks, masks = [], []
        for m in MODALITIES:
            if avail.get(m):
                enc_out = self.encoders[m](mods[m])
                if self.temporal and m not in self.structured and m not in self.domain \
                        and enc_out.dim() == 4:
                    # (B, T, N, D) -> (B, N, D) via temporal aggregation
                    enc_out = self.temporal_agg[m](enc_out)
                toks.append(enc_out)
                masks += [1] * N_TOK
            else:
                toks.append(self.missing[m].unsqueeze(0).expand(B, -1, -1))
                masks += [0] * N_TOK
        x = torch.cat(toks, dim=1)                                    # (B, 5*16, D)
        pad = torch.tensor(masks, device=x.device, dtype=torch.bool)[None].expand(B, -1)
        x = self.fusion(x, src_key_padding_mask=~pad)
        return self.head(x.mean(dim=1))

    # ---- SensorModel interface ----

    def fit(self, train, val, cfg: TrainConfig) -> None:
        torch.manual_seed(cfg.seed)
        self.train().to(cfg.device)
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        labels = [s.label for s in train]
        w = class_weights(labels, self.num_classes, cfg.class_weight)
        crit = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32,
                                device=torch.device(cfg.device)))
        rng = torch.Generator().manual_seed(cfg.seed)
        rng_shuffle = random.Random(cfg.seed)
        best = -1.0
        patience = 0
        for ep in range(cfg.epochs):
            self.train()
            if cfg.batch_strategy == "balanced":
                indexer = build_batch_indexer(
                    "balanced", labels, self.num_classes, cfg.batch_size,
                    cfg.seed * 1000 + ep)
                for idx in indexer.batches():
                    batch = [train[j] for j in idx]
                    avail = self._dropout_mask(cfg, rng)
                    mods = self._stack_mods(batch, avail, cfg, rng)
                    lbl = torch.tensor([s.label for s in batch], device=cfg.device)
                    loss = crit(self(mods, avail), lbl)
                    opt.zero_grad(); loss.backward(); opt.step()
            else:
                order = list(range(len(train)))
                rng_shuffle.shuffle(order)
                for i in range(0, len(order), cfg.batch_size):
                    batch = [train[j] for j in order[i:i + cfg.batch_size]]
                    avail = self._dropout_mask(cfg, rng)
                    mods = self._stack_mods(batch, avail, cfg, rng)
                    lbl = torch.tensor([s.label for s in batch], device=cfg.device)
                    loss = crit(self(mods, avail), lbl)
                    opt.zero_grad(); loss.backward(); opt.step()
            v = self._evaluate(val, cfg)
            if v > best:
                best = v; patience = 0
                self.save(f"{cfg.out_dir}/{self.name}_seed{cfg.seed}.pt")
            else:
                patience += 1
                if patience >= cfg.patience:
                    break
            print(f"[{self.name}] ep {ep} val {v:.3f} (best {best:.3f})")

    def _dropout_mask(self, cfg: TrainConfig, rng: torch.Generator) -> Dict[str, bool]:
        avail = {}
        for m in MODALITIES:
            p = cfg.modality_dropout_p
            if cfg.modality_dropout and m in cfg.modality_dropout:
                p = cfg.modality_dropout[m]
            avail[m] = bool(torch.rand(1, generator=rng).item() > p)
        if not any(avail.values()):
            avail[list(avail)[0]] = True
        return avail

    def _stack_mods(self, samples, avail: Dict[str, bool], cfg: TrainConfig,
                    rng: torch.Generator = None):
        mods = {}
        for m in MODALITIES:
            if avail.get(m):
                arrs = [s.modalities[m].data for s in samples]
                if m in self.structured:
                    # 1-D structured feature: (B, F) — no time axis.
                    mods[m] = torch.from_numpy(
                        np.stack(arrs)).to(cfg.device)
                else:
                    x = torch.stack(
                        [torch.from_numpy(a) for a in arrs]).to(cfg.device)
                    # Time masking (training augmentation): zero out one random
                    # contiguous frame run on raw multi-frame modalities, forcing
                    # the causal aggregator to reconstruct from earlier context.
                    if (self.temporal and m not in self.domain and x.dim() == 4
                            and cfg.time_mask_p > 0 and rng is not None
                            and torch.rand(1, generator=rng).item() < cfg.time_mask_p):
                        x = self._apply_time_mask(x, rng)
                    mods[m] = x
        return mods

    def _apply_time_mask(self, x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        """Zero a random contiguous run of frames on (B, T, ...).

        Uses a short center-weighted run (1-2 frames) so masking stays realistic
        for short sequences (T=5). Returns a new tensor; inputs untouched.
        """
        B, T = x.shape[:2]
        length = min(2, max(1, int(round(torch.rand(1, generator=rng).item() * 2))))
        if length >= T:
            start = 0
        else:
            start = int(torch.randint(0, T - length + 1, (1,), generator=rng).item())
        out = x.clone()
        out[:, start:start + length] = 0
        return out

    @torch.no_grad()
    def _evaluate(self, samples, cfg: TrainConfig) -> float:
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
    def predict(self, sample, available: List[str]) -> Dict[int, float]:
        self.eval()
        avail = {m: m in available for m in MODALITIES}
        dev = self._device()
        mods = {}
        for m in available:
            data = sample.modalities[m].data
            if m in self.structured:
                mods[m] = torch.from_numpy(data)[None].to(dev)
            else:
                mods[m] = torch.from_numpy(data)[None].to(dev)
        logits = self(mods, avail)
        probs = torch.softmax(logits[0], dim=-1)
        return {i: float(p) for i, p in enumerate(probs)}

    @torch.no_grad()
    def predict_batch(self, samples, available: List[str]) -> "torch.Tensor":
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

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(),
                    "structured": self.structured,
                    "domain": self.domain,
                    "domain_dims": self.domain_dims,
                    "temporal": self.temporal,
                    "num_classes": self.num_classes}, path)

    @classmethod
    def load(cls, path: str) -> "TokenFusionModel":
        ckpt = torch.load(path, map_location="cpu")
        # Backward compat: pre-structured checkpoints are a bare state_dict.
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state, structured, num_classes = (
                ckpt["state_dict"], ckpt.get("structured", {}), ckpt.get("num_classes", 27))
            domain = ckpt.get("domain", {})
            domain_dims = ckpt.get("domain_dims", {})
            temporal = ckpt.get("temporal", False)
        else:
            state, structured, num_classes = ckpt, {}, 27
            domain, domain_dims, temporal = {}, {}, False
        m = cls(num_classes=num_classes, structured=structured,
                domain=domain, domain_dims=domain_dims, temporal=temporal)
        m.load_state_dict(state)
        return m
