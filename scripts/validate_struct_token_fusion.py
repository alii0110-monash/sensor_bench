"""Minimal validation: train token_fusion on v5_structfeat structured features.

v5_structfeat replaces weak-modality raw data with 1-D structured features
(mw 134, wifi 161, depth 63, lidar 353). The main-pipeline token_fusion uses
PointEncoder expecting (B,5,P,C) point clouds — mismatched. This script uses
MLPEncoder for the structured modalities and keeps PointEncoder for rgb,
then measures only-mmwave + robustness vs the v4 baseline (0.296 / 0.496).

Run with few epochs for a quick signal; full run uses 30.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.models.base import TrainConfig
from framework.models.encoders import D, N_TOK, MLPEncoder, PointEncoder

MODALITIES = ["wifi", "depth", "lidar", "mmwave", "rgb"]
FEAT_DIM = {"wifi": 161, "depth": 63, "lidar": 353, "mmwave": 134}


class StructTokenFusion(nn.Module):
    """token_fusion adapted to structured 1-D features. rgb stays a raw point
    cloud (PointEncoder); the other four use MLPEncoder. Same fusion head."""

    name = "struct_token_fusion"

    def __init__(self, num_classes: int = 27, d: int = D, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.d = d
        self.encoders = nn.ModuleDict({
            "wifi": MLPEncoder(FEAT_DIM["wifi"]),
            "depth": MLPEncoder(FEAT_DIM["depth"]),
            "lidar": MLPEncoder(FEAT_DIM["lidar"]),
            "mmwave": MLPEncoder(FEAT_DIM["mmwave"]),
            "rgb": PointEncoder(2),
        })
        self.missing = nn.ParameterDict({
            m: nn.Parameter(torch.randn(N_TOK, d) * 0.02) for m in MODALITIES})
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=4 * d, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.1)
        self.fusion = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.head = nn.Linear(d, num_classes)

    def forward(self, mods, avail):
        B = next(iter(mods.values())).shape[0]
        toks, masks = [], []
        for m in MODALITIES:
            if avail.get(m):
                toks.append(self.encoders[m](mods[m]))
                masks += [1] * N_TOK
            else:
                toks.append(self.missing[m].unsqueeze(0).expand(B, -1, -1))
                masks += [0] * N_TOK
        x = torch.cat(toks, dim=1)
        pad = torch.tensor(masks, device=x.device, dtype=torch.bool)[None].expand(B, -1)
        x = self.fusion(x, src_key_padding_mask=~pad)
        return self.head(x.mean(dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/mmfi/v5_structfeat")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/v5_structfeat_tf_val.json")
    args = ap.parse_args()

    print(f"[struct-tf] loading {args.data_root}…", flush=True)
    ds = load_dataset(args.data_root, mode="lazy")
    train = list(ds.splits["train"])
    val = list(ds.splits["val"])
    test = list(ds.splits["test"])
    print(f"[struct-tf] train={len(train)} val={len(val)} test={len(test)}", flush=True)

    torch.manual_seed(args.seed)
    model = StructTokenFusion(num_classes=27).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    def _stack(samples, avail):
        mods = {}
        for m in avail:
            arrs = [torch.from_numpy(s.modalities[m].data) for s in samples]
            if m == "rgb":
                mods[m] = torch.stack(arrs).to(args.device)
            else:
                # (B, F) — pad to (B,1,F) so MLPEncoder sees (B,1,F)? No: MLPEncoder expects (B,F).
                mods[m] = torch.stack(arrs).to(args.device)
        return mods

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        import random
        random.shuffle(order)
        t0 = time.time()
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            batch = [train[j] for j in idx]
            avail = {m: (torch.rand(1).item() > 0.25) for m in MODALITIES}
            if not any(avail.values()):
                avail["rgb"] = True
            mods = _stack(batch, avail)
            lbl = torch.tensor([s.label for s in batch], device=args.device)
            loss = crit(model(mods, avail), lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ok = tot = 0
            for i in range(0, len(val), 128):
                batch = val[i:i + 128]
                avail = {m: True for m in MODALITIES}
                mods = _stack(batch, avail)
                pred = model(mods, avail).argmax(-1).cpu().tolist()
                ok += sum(p == s.label for p, s in zip(pred, batch))
                tot += len(batch)
            v = ok / tot
        print(f"[struct] ep {ep} val {v:.3f} ({time.time()-t0:.1f}s)", flush=True)
        if v > best:
            best = v

    # Evaluate key profiles on test
    def _acc(mods_avail):
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for i in range(0, len(test), 128):
                batch = test[i:i + 128]
                avail = {m: m in mods_avail for m in MODALITIES}
                mods = _stack(batch, avail)
                pred = model(mods, avail).argmax(-1).cpu().tolist()
                ok += sum(p == s.label for p, s in zip(pred, batch))
                tot += len(batch)
        return ok / max(tot, 1)

    results = {"seed": args.seed, "epochs": args.epochs, "best_val": best,
               "test_full": _acc(MODALITIES),
               "test_only_mmwave": _acc(["mmwave"]),
               "test_only_rgb": _acc(["rgb"]),
               "test_miss_mmwave": _acc([m for m in MODALITIES if m != "mmwave"])}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("[struct] results:", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()