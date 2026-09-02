"""Visualize cached person masks vs depth for quality inspection.

Loads a few samples' depth + cached masks, renders panels:
  [depth pseudo-RGB | mask overlay | masked depth | body-band mask]
Out: results/plots_v4/mask_quality_panel.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / 'datasets' / 'mmfi' / 'v4'
MASK_DIR = DATASET / 'masks_m2f'
OUT = ROOT / 'results' / 'plots_v4' / 'mask_quality_panel.png'
N_SAMPLES = 6


def pseudo_rgb(d: np.ndarray) -> np.ndarray:
    return (1.0 - np.clip(d / 5.0, 0, 1))


def main() -> None:
    from framework.dataset.loader import load_dataset  # noqa: E402
    ds = load_dataset(str(DATASET), mode='lazy')
    val = ds.splits['val']
    fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(12, 3 * N_SAMPLES))
    shown = 0
    for s in val:
        if shown >= N_SAMPLES:
            break
        f = MASK_DIR / 'val' / f'{s.id}.npz'
        if not f.exists():
            continue
        d = s.modalities['depth'].data[:, 0]          # (T,224,224)
        m = np.load(f)['mask']                         # (T,224,224)
        t = d.shape[0] // 2                            # middle frame
        img = pseudo_rgb(d[t])
        band = ((d[t] >= 1.0) & (d[t] <= 3.5)).astype(float)
        masked = img * m[t]
        axes[shown, 0].imshow(img, cmap='gray'); axes[shown, 0].set_title(
            f'{s.id} depth(pseudo)', fontsize=8)
        axes[shown, 1].imshow(img, cmap='gray')
        axes[shown, 1].imshow(np.ma.masked_where(m[t] == 0, m[t]),
                              cmap='autumn', alpha=0.5)
        axes[shown, 1].set_title('mask overlay', fontsize=8)
        axes[shown, 2].imshow(masked, cmap='gray'); axes[shown, 2].set_title(
            'masked depth', fontsize=8)
        axes[shown, 3].imshow(band, cmap='gray'); axes[shown, 3].set_title(
            'naive body-band 1-3.5m', fontsize=8)
        for c in range(4):
            axes[shown, c].axis('off')
        shown += 1
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print(f'saved {OUT} ({shown} samples)')


if __name__ == '__main__':
    main()