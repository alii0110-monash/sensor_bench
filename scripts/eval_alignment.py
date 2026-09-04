#!/usr/bin/env python
"""L1 retrieval eval on train base held-out (spec M5a)."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from framework.dataset.loader import load_dataset
from framework.eval.alignment import build_held_out_split, evaluate_retrieval
from framework.models.alignment import AlignmentModel
from framework.models.text_encoder import CLIPTextEncoder, HashTextEncoder

TEXT_ENCODERS = {"hash": HashTextEncoder, "clip": CLIPTextEncoder}


def _diagnose_label(model, te, samples, labels, device, batch_size=64):
    """每 query 正样本 rank + 排在前面的同 label 负样本数 (均值)."""
    import numpy as np
    zs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            avail = {m: True for m in model.encoders}
            mods = {m: torch.stack(
                [torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
                for m in avail if m in batch[0].modalities}
            texts = [s.text.get("en", [""])[0] for s in batch]
            toks = model.encode_modalities(mods, avail)
            zs.append(model.projection_head(model.pool(toks)))
            ts.append(te.encode(texts).to(device))
    Z = torch.cat(zs); T = torch.cat(ts)
    Z = torch.nn.functional.normalize(Z, dim=-1)
    T = torch.nn.functional.normalize(T, dim=-1)
    sim = Z @ T.t()                                  # (N,N)
    n = sim.shape[0]
    lab = torch.tensor(labels, device=sim.device)
    ranks, same_above = [], []
    for i in range(n):
        above = (sim[i] > sim[i, i])
        ranks.append(above.sum().item() + 1)
        same_above.append(
            (above & (lab == labels[i]) & (torch.arange(n, device=sim.device) != i)).sum().item())
    return float(np.mean(ranks)), float(np.mean(same_above))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mmfi/v5")
    ap.add_argument("--ckpt", default="checkpoints_alignment/alignment_seed0.pt")
    ap.add_argument("--text-encoder", choices=list(TEXT_ENCODERS), default="clip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--prototype-head", action="store_true",
                    help="checkpoint 使用原型头 (256→27→text_dim), 如 train_alignment --init-prototype")
    ap.add_argument("--diagnose-label", action="store_true",
                    help="输出诊断: 正样本平均 rank + 排前面的同 label 负样本数")
    ap.add_argument("--captions-override", default=None,
                    help="JSONL {id,caption}: 覆盖评测文本（schema 臂）")
    ap.add_argument("--clip-model", default="/home/li/datasets/models/clip-vit-base-patch32",
                    help="CLIP 权重路径（本地目录或 HF repo id）")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    te_cls = TEXT_ENCODERS[args.text_encoder]
    te = (te_cls(dim=512) if args.text_encoder == "hash"
          else te_cls(model_name=args.clip_model, device=device))
    override = None
    if args.captions_override:
        override = {}
        for line in open(args.captions_override):
            r = json.loads(line)
            override[r["id"]] = r["caption"]

    model = AlignmentModel(num_modalities=5, text_dim=te.dim)
    if args.prototype_head:
        model.projection_head = torch.nn.Sequential(
            torch.nn.Linear(256, 27), torch.nn.Linear(27, te.dim))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    # 从 splits ids 派生 held-out（仅加载 held 样本，避免 v5 全量进内存）
    # held-out 按 base 组整组划出（含变体，变体共享 base 文本）。评测样本集取 base 为主：
    # 变体文本重复会压低 recall@k，且 spec 意图是"held-out base"。故只保留 base 样本参与评测。
    train_ids = json.load(open(os.path.join(args.dataset, "splits", "train.json")))
    held, _ = build_held_out_split(train_ids, fraction=args.fraction)
    held_bases = {i for i in held if "__aug" not in i}
    held_samples = [s for s in ds.train if s.id in held_bases]
    if override is not None:
        for s in held_samples:
            cap = override.get(s.id) or override.get(s.id.split("__aug")[0])
            if cap:
                s.text = {"en": [cap]}
    res = evaluate_retrieval(model, te, held_samples, device=device)
    print(f"[eval] n={res['n']} r@1={res['r@1']:.4f} r@5={res['r@5']:.4f} r@10={res['r@10']:.4f} "
          f"tr@1={res['tr@1']:.4f} tr@5={res['tr@5']:.4f} tr@10={res['tr@10']:.4f}")
    if args.diagnose_label:
        labels = [s.label for s in held_samples]
        mean_rank, same_above = _diagnose_label(model, te, held_samples, labels, device)
        print(f"[eval] diagnose: mean_rank={mean_rank:.1f} same_label_above_pos={same_above:.2f}")


if __name__ == "__main__":
    main()
