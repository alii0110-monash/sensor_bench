"""L1 cross-modal retrieval evaluation (spec M5a)."""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn.functional as F


def build_held_out_split(all_ids: List[str], fraction: float = 0.1, seed: int = 0) -> Tuple[list, set]:
    """Split base groups into (held_out, train_ids). A held-out base and ALL its
    `__aug*` variants are excluded from training (variants share base text)."""
    import random
    rng = random.Random(seed)
    bases = sorted({i.split("__")[0] for i in all_ids})
    n_held = max(1, int(len(bases) * fraction))
    held_bases = set(rng.sample(bases, n_held))
    held = [i for i in all_ids if i.split("__")[0] in held_bases]
    train_ids = {i for i in all_ids if i.split("__")[0] not in held_bases}
    return held, train_ids


def retrieval_recall_at_k(query: torch.Tensor, cand: torch.Tensor, k: int = 1) -> float:
    """Recall@k: fraction of queries whose true text candidate is in top-k.
    query/cand: (N, dim) normalized embeddings, index-aligned positives."""
    q = F.normalize(query, dim=-1)
    c = F.normalize(cand, dim=-1)
    sim = q @ c.t()                      # (N, N)
    _, topk = sim.topk(k, dim=1)         # (N, k)
    hits = (topk == torch.arange(len(q), device=q.device)[:, None]).any(dim=1)
    return float(hits.float().mean())


def class_retrieval_recall_at_k(query: torch.Tensor, labels: torch.Tensor,
                                k: int = 1) -> float:
    """Class-conditional recall@k: fraction of queries whose top-k candidates
    contain ANY sample of the same label (not just the index-aligned positive).

    This measures whether sensor embeddings cluster by action class, bypassing
    the whole-sentence caption retrieval ceiling (M6b: r@1 max 0.0109 ≪ random
    1/27≈0.037). query: (N, dim) normalized embeddings; labels: (N,) int.
    """
    q = F.normalize(query, dim=-1)
    sim = q @ q.t()                          # (N, N)
    N = q.shape[0]
    same = labels[:, None] == labels[None, :]  # (N, N) 同 label
    same[torch.arange(N), torch.arange(N)] = False  # 排除自身
    _, topk = sim.topk(k, dim=1)              # (N, k)
    hits = same.gather(1, topk).any(dim=1)    # top-k 内是否有同类
    return float(hits.float().mean())


def class_retrieval_mean_rank(query: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean rank of the first same-label candidate (1-indexed). Lower = tighter
    clustering. Excludes self. Returns mean over queries."""
    q = F.normalize(query, dim=-1)
    sim = q @ q.t()
    N = q.shape[0]
    same = labels[:, None] == labels[None, :]
    same[torch.arange(N), torch.arange(N)] = False
    # 对每行, 找第一个同类候选的 rank (按 sim 降序)
    _, order = sim.sort(dim=1, descending=True)   # (N, N) 索引
    ranks = []
    for i in range(N):
        cands = order[i]
        pos = (same[i][cands]).nonzero(as_tuple=True)[0]
        ranks.append(pos[0].item() + 1 if len(pos) else N)
    return float(sum(ranks) / len(ranks))


def evaluate_class_retrieval(model, samples, device="cuda", batch_size=64) -> dict:
    """Sensor-only class-conditional retrieval (proposal 2, scheme A).
    Embeds all samples, then measures whether same-label samples are retrieved
    in top-k. No text encoder needed — bypasses caption template ceiling."""
    model.eval()
    model.to(device)
    zs, labs = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            avail = {m: True for m in model.encoders}
            mods = {}
            for m in avail:
                if m in batch[0].modalities:
                    mods[m] = torch.stack(
                        [torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
            if mods:
                toks = model.encode_modalities(mods, avail)
                z = model.projection_head(model.pool(toks))
            else:
                z = torch.zeros(len(batch), model.text_dim, device=device)
            zs.append(z)
            labs.append(torch.tensor([s.label for s in batch], device=device))
    Z = torch.cat(zs)
    L = torch.cat(labs)
    return {"cr@1": class_retrieval_recall_at_k(Z, L, 1),
            "cr@5": class_retrieval_recall_at_k(Z, L, 5),
            "cr@10": class_retrieval_recall_at_k(Z, L, 10),
            "cr_mean_rank": class_retrieval_mean_rank(Z, L),
            "n": len(Z)}


def evaluate_retrieval(model, text_encoder, samples, device="cuda",
                       batch_size=64) -> dict:
    """Embed sensor and text for all samples; return recall@1/5/10 both directions."""
    model.eval()
    model.to(device)
    zs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            avail = {m: True for m in model.encoders}
            mods = {}
            for m in avail:
                if m in batch[0].modalities:
                    mods[m] = torch.stack(
                        [torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
            texts = [s.text.get("en") or s.text.get("captions") or [""] for s in batch]
            texts = [t[0] if t else "" for t in texts]
            t = text_encoder.encode(texts).to(device)
            if mods:
                toks = model.encode_modalities(mods, avail)
                z = model.projection_head(model.pool(toks))
            else:
                z = torch.zeros(len(batch), t.shape[1], device=device)
            zs.append(z); ts.append(t)
    Z = torch.cat(zs); T = torch.cat(ts)
    return {"r@1": retrieval_recall_at_k(Z, T, 1),
            "r@5": retrieval_recall_at_k(Z, T, 5),
            "r@10": retrieval_recall_at_k(Z, T, 10),
            "tr@1": retrieval_recall_at_k(T, Z, 1),   # text→sensor (spec §75 反向)
            "tr@5": retrieval_recall_at_k(T, Z, 5),
            "tr@10": retrieval_recall_at_k(T, Z, 10),
            "n": len(Z)}
