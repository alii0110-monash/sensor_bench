#!/usr/bin/env python
"""Stage-1 contrastive alignment training (M5a).

Trains AlignmentModel with InfoNCE against a frozen text encoder on the v5
dataset. `--text-encoder hash` uses the deterministic mock (CI/smoke);
`--text-encoder clip` uses frozen CLIP (real training).
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from framework.dataset.loader import load_dataset
from framework.models.alignment import AlignmentModel, MODALITIES, info_nce_loss
from framework.models.text_encoder import CLIPTextEncoder, HashTextEncoder

TEXT_ENCODERS = {"hash": HashTextEncoder, "clip": CLIPTextEncoder}


def _dropout_mask(rng, p: float) -> dict:
    avail = {m: bool(rng.random() > p) for m in MODALITIES}
    if not any(avail.values()):
        avail[list(avail)[0]] = True
    return avail


def _stack_mods(samples, avail, device):
    mods = {}
    first = samples[0]
    for m in MODALITIES:
        if avail.get(m) and m in first.modalities:
            mods[m] = torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in samples]).to(device)
    return mods


def train_epoch(model, text_encoder, train, opt, batch_size=32,
                device="cuda", dropout_p=0.25, aux_cls_weight=0.0,
                neg_mine=False) -> tuple:
    """返回 (avg_total_loss, avg_info_nce, avg_ce). ce 在未启用辅助时按 0 累计."""
    model.train()
    rng = np.random.default_rng(0)
    total = 0.0; total_nce = 0.0; total_ce = 0.0; n = 0
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        avail = _dropout_mask(rng, dropout_p)
        mods = _stack_mods(batch, avail, device)
        if not mods:
            continue
        texts = [s.text.get("captions") or s.text.get("en", [""]) for s in batch]
        texts = [t[0] if t else "" for t in texts]
        text_emb = text_encoder.encode(texts).to(device)
        labels = torch.tensor([s.label for s in batch], device=device)
        info_nce, ce = model.forward_loss(mods, text_emb, avail,
                                          labels=labels, neg_mine=neg_mine)
        loss = info_nce if ce is None else info_nce + aux_cls_weight * ce
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); total_nce += info_nce.item()
        if ce is not None:
            total_ce += ce.item()
        n += 1
    return total / max(n, 1), total_nce / max(n, 1), total_ce / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--text-encoder", choices=list(TEXT_ENCODERS), default="clip")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints_alignment")
    ap.add_argument("--init-encoders", default="checkpoints_v4/token_fusion_seed0.pt",
                    help="用分类预热的 token_fusion 编码器权重初始化（冷启动对齐更稳）")
    ap.add_argument("--init-prototype", action="store_true",
                    help="原型初始化投影头: 分类头(256→27) + 27 CLIP原型(27→512)")
    ap.add_argument("--aux-cls-weight", type=float, default=0.0,
                    help="分类辅助 loss 权重 (0=关闭, 如 0.5)")
    ap.add_argument("--neg-mine", action="store_true",
                    help="label-aware 负样本排除")
    ap.add_argument("--out-tag", default="",
                    help="checkpoint 名后缀: {out}/m6b_{tag}_seed0.pt (空=alignment_seed0.pt)")
    ap.add_argument("--cache-size", type=int, default=256,
                    help="lazy loader cache_size (batch=256 时传 4096 防 cache thrash)")
    args = ap.parse_args()

    torch.manual_seed(0)
    ds = load_dataset(args.dataset, cache_size=args.cache_size)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    te_cls = TEXT_ENCODERS[args.text_encoder]
    te = te_cls(dim=512) if args.text_encoder == "hash" else te_cls(device=device)
    model = AlignmentModel(num_modalities=5, text_dim=te.dim,
                           num_classes=27 if args.aux_cls_weight > 0 else None).to(device)

    # 冷启动: 复用分类预热的 token_fusion 编码器权重 (encoders.* 键名一致)
    if args.init_encoders and os.path.exists(args.init_encoders):
        from framework.models.token_fusion import TokenFusionModel
        tf = TokenFusionModel(num_classes=27)
        tf.load_state_dict(torch.load(args.init_encoders, map_location="cpu"))
        src = {k: v for k, v in tf.state_dict().items() if k.startswith("encoders.")}
        missing, _ = model.load_state_dict(src, strict=False)
        if missing:
            print(f"[alignment] 未初始化的参数: {len(missing)} (投影头, 正常)", flush=True)
        print(f"[alignment] 编码器从 {args.init_encoders} 预热", flush=True)

        # 原型初始化投影头 (仅 clip 锚下有意义)
        if args.init_prototype and args.text_encoder == "clip":
            from curation.caption.verbs import ACTION_PHRASES
            protos = te.encode(["a person is " + v for v in ACTION_PHRASES.values()])
            head = torch.nn.Sequential(
                torch.nn.Linear(256, 27), torch.nn.Linear(27, te.dim))
            with torch.no_grad():
                head[0].weight.copy_(tf.head.weight)
                head[0].bias.copy_(tf.head.bias)
                head[1].weight.copy_(protos.t().to(head[1].weight.dtype))
                head[1].bias.zero_()
            model.projection_head = head.to(device)
            print("[alignment] 投影头原型初始化 (分类头 + 27 CLIP 原型)", flush=True)

        # 分类头预热 (aux 启用时从 tf.head 复制; 依赖本块内的 tf 作用域)
        if args.aux_cls_weight > 0:
            with torch.no_grad():
                model.classification_head.weight.copy_(tf.head.weight)
                model.classification_head.bias.copy_(tf.head.bias)
            print("[alignment] 分类头从 token_fusion head 预热", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    for ep in range(args.epochs):
        loss, nce, ce = train_epoch(model, te, ds.train, opt,
                                    batch_size=args.batch_size, device=device,
                                    aux_cls_weight=args.aux_cls_weight,
                                    neg_mine=args.neg_mine)
        ce_str = f", ce {ce:.4f}" if ce > 0 else ""
        print(f"[alignment] ep {ep} loss {loss:.4f} (info_nce {nce:.4f}{ce_str})", flush=True)
        if nce < best:                       # 按 info_nce 分量选 best (跨变体可比)
            best = nce
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("classification_head.")}   # 剔除分类头
            name = f"m6b_{args.out_tag}_seed0.pt" if args.out_tag else "alignment_seed0.pt"
            torch.save(state, f"{args.out}/{name}")
    name = f"m6b_{args.out_tag}_seed0.pt" if args.out_tag else "alignment_seed0.pt"
    print(f"done -> {args.out}/{name}")


if __name__ == "__main__":
    main()
