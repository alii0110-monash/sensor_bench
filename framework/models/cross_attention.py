from __future__ import annotations
import random
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from .base import SensorModel, TrainConfig
from .batching import build_batch_indexer, class_weights
from .encoders import D, N_TOK
from .token_fusion import MODALITIES, _build_encoders
from .temporal import TemporalAggregator


class CrossAttentionModel(nn.Module, SensorModel):
    """Cross-attention fusion: per-modality encoder -> 16 tokens each, a fixed
    set of learnable global queries read the modality tokens as key/value,
    mean-pool, classification head. Missing modality = its tokens simply are
    not provided as key/value (no [MISSING] embedding, no attention mask).

    Unlike token_fusion's concat + shared transformer (O((M*16)^2)), the
    cross-attention cost is O(Q * M*16) with Q fixed, so it scales *linearly*
    with the number of modalities and naturally supports arbitrary
    add/remove of modalities at inference.

    `structured` / `domain` / `domain_dims` / `temporal` mirror token_fusion
    and are persisted in the checkpoint so load() reconstructs the same
    architecture.
    """

    name = "cross_attention"

    def __init__(self, num_classes: int = 27, d: int = D, n_layers: int = 3,
                 n_heads: int = 4, n_query: int = 64, structured: dict = None,
                 domain: dict = None, domain_dims: dict = None,
                 temporal: bool = False):
        super().__init__()
        self.d = d
        self.num_classes = num_classes
        self.n_query = n_query
        self.temporal = temporal
        self.structured = dict(structured) if structured else {}
        self.domain = dict(domain) if domain else {}
        self.domain_dims = dict(domain_dims) if domain_dims else {}
        self.encoders = _build_encoders(self.structured, self.domain,
                                        self.domain_dims, temporal=temporal)
        # learnable global queries, not bound to any modality
        self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)
        # per-modality temporal aggregator (only when temporal=True)
        self.temporal_agg = nn.ModuleDict({
            m: TemporalAggregator(d, n_heads) for m in MODALITIES}) if temporal else nn.ModuleDict()
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=0.1)
            for _ in range(n_layers)])
        self.self_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(n_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        # Direct path: head also sees the mean-pooled modality tokens, so it can
        # classify even before the queries learn to read them well.
        self.head = nn.Linear(2 * d, num_classes)

    def forward(self, mods: Dict[str, torch.Tensor], avail: Dict[str, bool]) -> torch.Tensor:
        B = next(iter(mods.values())).shape[0]
        kvs = []
        for m in MODALITIES:
            if avail.get(m):
                enc_out = self.encoders[m](mods[m])
                if self.temporal and m not in self.structured and m not in self.domain \
                        and enc_out.dim() == 4:
                    # (B, T, N, D) -> (B, N, D) via temporal aggregation
                    enc_out = self.temporal_agg[m](enc_out)
                kvs.append(enc_out)
        if not kvs:
            # degenerate: no modality available (shouldn't happen in practice)
            kv = torch.zeros(B, 1, self.d, device=next(self.parameters()).device)
        else:
            kv = torch.cat(kvs, dim=1)                       # (B, M*16, D)
        q = self.query.unsqueeze(0).expand(B, -1, -1)        # (B, Q, D)
        # Perceiver-style: each layer does cross-attention (queries read the
        # modality tokens) then self-attention (queries exchange info).
        for attn, anorm, sattn, snorm, ff, fnorm in zip(
                self.cross, self.cross_norms, self.self_attn, self.self_norms,
                self.ffs, self.ff_norms):
            a, _ = attn(q, kv, kv)                          # (B, Q, D)
            q = anorm(a + q)
            s, _ = sattn(q, q, q)                           # query self-attention
            q = snorm(s + q)
            q = fnorm(ff(q) + q)
        pooled = kv.mean(dim=1)                             # (B, D) direct path
        fused = q.mean(dim=1)                               # (B, D)
        return self.head(torch.cat([fused, pooled], dim=-1))

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
                    mods[m] = torch.from_numpy(
                        np.stack(arrs)).to(cfg.device)
                else:
                    x = torch.stack(
                        [torch.from_numpy(a) for a in arrs]).to(cfg.device)
                    if (self.temporal and m not in self.domain and x.dim() == 4
                            and cfg.time_mask_p > 0 and rng is not None
                            and torch.rand(1, generator=rng).item() < cfg.time_mask_p):
                        x = self._apply_time_mask(x, rng)
                    mods[m] = x
        return mods

    def _apply_time_mask(self, x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
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
                    "n_query": self.n_query,
                    "num_classes": self.num_classes}, path)

    @classmethod
    def load(cls, path: str) -> "CrossAttentionModel":
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state, structured, num_classes = (
                ckpt["state_dict"], ckpt.get("structured", {}), ckpt.get("num_classes", 27))
            domain = ckpt.get("domain", {})
            domain_dims = ckpt.get("domain_dims", {})
            temporal = ckpt.get("temporal", False)
            n_query = ckpt.get("n_query", 32)
        else:
            state, structured, num_classes = ckpt, {}, 27
            domain, domain_dims, temporal, n_query = {}, {}, False, 32
        m = cls(num_classes=num_classes, structured=structured,
                domain=domain, domain_dims=domain_dims, temporal=temporal,
                n_query=n_query)
        m.load_state_dict(state)
        return m
