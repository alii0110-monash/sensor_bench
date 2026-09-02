"""Runner entry: run_layer_cka + plot.

Usage:
    python scripts/run_layer_cka.py --checkpoint_dir checkpoints_v4_temporal \
        --dataset_root datasets/mmfi/v4 --output_dir results
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# 让脚本能从 sensorbench/ 根目录运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framework.eval.dataset_quality.layer_cka import run_layer_cka  # noqa: E402
from scripts.plot_layer_cka import plot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', default='checkpoints_v4_temporal')
    ap.add_argument('--dataset_root', default='datasets/mmfi/v4')
    ap.add_argument('--output_dir', default='results')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device', default='auto',
                    help='cuda | cpu | auto (default: cuda if available)')
    args = ap.parse_args()

    import torch
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"[run_layer_cka] device={device}")

    data = run_layer_cka(
        checkpoint_dir=args.checkpoint_dir,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=device,
    )

    # plot
    json_path = Path(args.output_dir) / 'layer_cka_v4.json'
    plot_path = Path(args.output_dir) / 'plots_v4' / 'layer_cka_curve.png'
    plot(str(json_path), str(plot_path))
    print(f"[run_layer_cka] done.  JSON={json_path}  PNG={plot_path}")


if __name__ == '__main__':
    main()