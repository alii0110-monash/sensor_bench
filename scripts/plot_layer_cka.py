"""Plot layer-wise CKA curves from results/layer_cka_v4.json.

读取 JSON，画 [hook x CKA] 曲线（按模态对分组），3 seeds 置信带。
输出: results/plots_v4/layer_cka_curve.png
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# 重点观察的模态对（按声明第 4 节决策 3）
FOCUS_PAIRS = [
    ('mmwave', 'rgb'),
    ('lidar', 'rgb'),
    ('wifi', 'rgb'),
    ('mmwave', 'lidar'),
]


def plot(json_path: str, out_path: str) -> None:
    with open(json_path) as f:
        data = json.load(f)
    hooks = data['hooks']
    modalities = data['modalities']
    aggregated = data['aggregated']
    per_seed = data['per_seed']

    fig, axes = plt.subplots(1, len(FOCUS_PAIRS), figsize=(4 * len(FOCUS_PAIRS), 4),
                             sharey=True)
    if len(FOCUS_PAIRS) == 1:
        axes = [axes]

    for ax, (ma, mb) in zip(axes, FOCUS_PAIRS):
        means = []
        stds = []
        for h in hooks:
            v = aggregated[h].get(f"{ma}__{mb}", {})
            means.append(v.get('mean', 0.0))
            stds.append(v.get('std', 0.0))
        # 横轴: 0..len(hooks)-1，标 "enc_out", "layer1_out"
        x = np.arange(len(hooks))
        ax.plot(x, means, 'o-', linewidth=2, markersize=8, color='#1f77b4',
                label='mean (3 seeds)')
        ax.fill_between(x,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color='#1f77b4')
        # 也画 3 个 seed 的原始曲线
        for s, cka in per_seed.items():
            ys = [cka[h].get(f"{ma}__{mb}", 0.0) for h in hooks]
            ax.plot(x, ys, 'o--', linewidth=0.8, alpha=0.5, label=f'seed {s}')
        ax.set_xticks(x)
        ax.set_xticklabels(hooks, rotation=15)
        ax.set_title(f'{ma}  vs  {mb}')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Linear CKA')
        ax.grid(True, alpha=0.3)
        # legend 仅第一个子图（避免重复）
        if ma == FOCUS_PAIRS[0][0]:
            ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('MMFi v4 — Layer-wise Cross-Modal CKA\n'
                 'enc_out = per-modality encoder, layer1_out = transformer final layer',
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='results/layer_cka_v4.json')
    ap.add_argument('--out', default='results/plots_v4/layer_cka_curve.png')
    args = ap.parse_args()
    plot(args.json, args.out)


if __name__ == '__main__':
    main()