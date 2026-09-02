"""Cache person masks for v4 depth frames (Group A preprocessing).

- depth (T,1,224,224) meters -> pseudo-RGB (near=bright, clip [0,5]m)
- torchvision Mask R-CNN R50-FPN COCO (label 1 = person, score>0.5, union masks)
  (Mask2Former 换掉的原因: hf-mirror 大文件重定向 us.aws.cdn.hf.co 被墙;
   torchvision 权重在 download.pytorch.org 直连可达)
- saves per-sample uint8 masks (T,224,224) {0,1} to masks_m2f/{split}/{sid}.npz
- quality stats: detection rate + IoU vs naive body_range band (1-3.5m)

Run (GPU): sbatch jobs/depth_masks.slurm
Out: datasets/mmfi/v4/masks_m2f/ + results/mask_quality.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / 'datasets' / 'mmfi' / 'v4'
MASK_DIR = DATASET / 'masks_m2f'
OUT_JSON = ROOT / 'results' / 'mask_quality.json'
TRAIN_SUB = 3000
BATCH_FRAMES = 16
SCORE_THR = 0.5


def label_from_id(sid: str) -> int:
    return int(sid.split('_')[2][1:]) - 1


def stratified_subset(ids: list, labels: list, n_total: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    by_cls: dict = {}
    for i, l in enumerate(labels):
        by_cls.setdefault(l, []).append(i)
    for l in by_cls:
        rng.shuffle(by_cls[l])
    per_cls = max(1, n_total // len(by_cls))
    idx = []
    for l in sorted(by_cls):
        idx.extend(by_cls[l][:per_cls])
    return sorted(int(i) for i in idx)


def to_pseudo_rgb(depth_frame: np.ndarray) -> torch.Tensor:
    """(1,224,224) meters -> (3,224,224) float 0..1, near=bright."""
    d = np.clip(depth_frame[0] / 5.0, 0.0, 1.0)
    g = (1.0 - d).astype(np.float32)
    return torch.from_numpy(np.repeat(g[None], 3, axis=0))


def main() -> None:
    from framework.dataset.loader import load_dataset  # noqa: E402
    from torchvision.models.detection import maskrcnn_resnet50_fpn, \
        MaskRCNN_ResNet50_FPN_Weights  # noqa: E402

    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[m2f] device={device}', flush=True)

    ds = load_dataset(str(DATASET), mode='lazy', cache_size=10000)
    train_all = ds.splits['train']
    val = ds.splits['val']
    tr_idx = stratified_subset(list(train_all._ids),
                               [label_from_id(s) for s in train_all._ids],
                               TRAIN_SUB, seed=0)
    targets = {'train': [int(i) for i in tr_idx], 'val': list(range(len(val)))}
    print(f'[m2f] targets: {[(k, len(v)) for k, v in targets.items()]}', flush=True)

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    stats = {'det_rate': [], 'iou_band': [], 'n_frames': 0}

    # person = COCO label 1; keep ONLY the highest-scoring instance
    # (union 会被背景假阳性实例污染 — 首轮可视化发现书架/墙面被标成 person)
    def infer_batch(frames):
        with torch.no_grad():
            x = torch.stack(frames).to(device)
            out = model(x)
        masks = []
        for o in out:
            keep = (o['labels'] == 1) & (o['scores'] > SCORE_THR)
            if keep.any():
                best = o['scores'][keep].argmax()
                m = o['masks'][keep][best, 0]  # (H,W) 0..1
                masks.append((m > 0.5).cpu().numpy().astype(np.uint8))
            else:
                masks.append(np.zeros((224, 224), dtype=np.uint8))
        return masks

    for split, indices in targets.items():
        split_dir = MASK_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for n, pos in enumerate(indices):
            sample = train_all[pos] if split == 'train' else val[pos]
            d = sample.modalities['depth'].data  # (T,1,224,224)
            T = d.shape[0]
            frames = [to_pseudo_rgb(d[t]) for t in range(T)]
            masks = []
            for i in range(0, len(frames), BATCH_FRAMES):
                masks.extend(infer_batch(frames[i:i + BATCH_FRAMES]))
            m = np.stack(masks).astype(np.uint8)  # (T,224,224)
            np.savez_compressed(split_dir / f'{sample.id}.npz', mask=m)
            # stats
            det = m.reshape(T, -1).sum(1) > 50  # ≥50 px person
            band = ((d[:, 0] >= 1.0) & (d[:, 0] <= 3.5))
            inter = (m.astype(bool) & band).reshape(T, -1).sum(1)
            union = (m.astype(bool) | band).reshape(T, -1).sum(1)
            iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
            stats['det_rate'].extend(det.tolist())
            stats['iou_band'].extend(iou.tolist())
            stats['n_frames'] += T
            if (n + 1) % 200 == 0:
                print(f'[m2f] {split} {n+1}/{len(indices)} '
                      f'det={np.mean(stats["det_rate"]):.3f} '
                      f'iou_band={np.mean(stats["iou_band"]):.3f} '
                      f'({time.time()-t0:.0f}s)', flush=True)

    summary = {
        'model': 'torchvision maskrcnn_resnet50_fpn (COCO, person=label1)',
        'score_thr': SCORE_THR,
        'n_frames': stats['n_frames'],
        'detection_rate': float(np.mean(stats['det_rate'])),
        'iou_vs_body_band_mean': float(np.mean(stats['iou_band'])),
        'iou_vs_body_band_p50': float(np.percentile(stats['iou_band'], 50)),
        'elapsed_s': time.time() - t0,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f'[m2f] saved {OUT_JSON}: {summary}', flush=True)


if __name__ == '__main__':
    main()