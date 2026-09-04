#!/usr/bin/env python
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.dataset.loader import load_dataset
from framework.models import detect_structured_features
from framework.models.base import TrainConfig
from framework.models.token_fusion import TokenFusionModel
from framework.models.late_fusion import LateFusionModel
from framework.models.cross_attention import CrossAttentionModel

MODELS = {"token_fusion": TokenFusionModel, "late_fusion": LateFusionModel,
          "cross_attention": CrossAttentionModel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--batch-strategy", choices=["shuffle", "balanced"], default="shuffle")
    ap.add_argument("--class-weight", choices=["none", "inverse_freq", "sqrt_inverse_freq"],
                    default="none")
    ap.add_argument("--modality-dropout", default=None,
                    help="per-modality dropout p as JSON dict, e.g. "
                         '{"mmwave":0.5,"rgb":0.5} (overrides --modality-dropout-p)')
    ap.add_argument("--modality-dropout-p", type=float, default=0.25)
    ap.add_argument("--domain", default=None,
                    help="comma-separated modalities to use DomainEncoder on "
                         "(raw data -> extract_*_features on the fly -> MLP), "
                         "e.g. 'depth,wifi,lidar,mmwave'. Keeps the dataset raw.")
    ap.add_argument("--temporal", action="store_true",
                    help="keep time axis through raw encoders and aggregate via "
                         "TemporalAggregator (RoPE + intra-modality temporal "
                         "attention). Requires a raw multi-frame dataset (v4).")
    ap.add_argument("--motion-depth", action="store_true",
                    help="token_fusion only: swap DepthEncoder for ViTMotionEncoder "
                         "(2ch [d_t, Δ_t] frame-difference input, DMM-style). "
                         "Requires --temporal and raw depth (not structured/domain).")
    ap.add_argument("--motion-depth-layernorm", action="store_true",
                    help="with --motion-depth: LayerNorm on depth tokens after the "
                         "encoder, aligning ViT token statistics with other "
                         "modalities before the shared fusion transformer.")
    ap.add_argument("--adaptive-pool", action="store_true",
                    help="token_fusion only: mean-pool only AVAILABLE tokens after "
                         "fusion (1 modality -> /16, 2 -> /32 ...). Fixes only-* "
                         "profiles where [MISSING]-token outputs dilute the "
                         "present modality's signal. Full-profile behavior is "
                         "mathematically identical to unconditional mean.")
    ap.add_argument("--mode", choices=["auto", "eager", "lazy"], default=None,
                    help="dataset loading mode. Default auto; when --temporal, "
                         "defaults to lazy (full v4 raw does not fit the 18GB "
                         "cgroup and eager-loading OOM-kills the whole group).")
    ap.add_argument("--time-mask-p", type=float, default=0.0,
                    help="probability of masking a random contiguous frame run "
                         "per sample during training (temporal=True, raw frames). "
                         "0.0 disables. 对标 MiniMind-O 时间遮蔽增强.")
    args = ap.parse_args()

    mode = args.mode or ("lazy" if args.temporal else "auto")
    ds = load_dataset(args.dataset, mode=mode)
    os.makedirs(args.out_dir, exist_ok=True)
    import json as _json
    md = _json.loads(args.modality_dropout) if args.modality_dropout else None
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      seed=args.seed, out_dir=args.out_dir,
                      batch_strategy=args.batch_strategy,
                      class_weight=args.class_weight,
                      modality_dropout_p=args.modality_dropout_p,
                      modality_dropout=md, time_mask_p=args.time_mask_p)
    structured = detect_structured_features(ds)
    domain = {m: 1 for m in (args.domain.split(",") if args.domain else [])}
    # compute domain_dims from the actual extractors on the first train sample
    domain_dims = {}
    if domain:
        from framework.eval.dataset_quality.feature_extract import _EXTRACTORS
        s0 = ds.train[0]
        for m in domain:
            if m in s0.modalities:
                domain_dims[m] = _EXTRACTORS[m](s0.modalities[m].data).shape[0]
        print(f"[train] domain encoders: {domain_dims}")
    if domain and structured:
        print("[train] WARNING: --domain overrides structured-feature encoders "
              "for listed modalities")
    if structured:
        print(f"[train] detected structured features: {structured}")
    if args.model == "late_fusion":
        # late_fusion takes structured only; domain/temporal are token_fusion-only.
        model = MODELS[args.model](num_classes=27, structured=structured)
    else:
        model = MODELS[args.model](num_classes=27, structured=structured, domain=domain,
                                   domain_dims=domain_dims, temporal=args.temporal,
                                   motion_depth=args.motion_depth,
                                   motion_depth_layernorm=args.motion_depth_layernorm,
                                   adaptive_pool=args.adaptive_pool)
    model.fit(ds.train, ds.val, cfg)
    print(f"trained {args.model} seed {args.seed} -> {cfg.out_dir}")


if __name__ == "__main__":
    main()
