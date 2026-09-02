"""Layer-wise Cross-Modal CKA on trained TokenFusionModel.

目的: 用训练好的 token_fusion 模型在数据集 val split 上提取逐层表征，
计算跨模态 Linear CKA 曲线，回答"浅层特异/深层融合"的诊断问题。

Hook 点 (3 个):
  - enc_out:    per-modality encoder 输出 (cat 前，模态完全独立)
  - layer1_out: 第 2 层 transformer 输出 (共享语义开始出现)
  - pool_out:   mean pool 输出 (head 输入，与最终分类器对齐)

输出: results/layer_cka_{dataset}.json  +  results/plots_{dataset}/layer_cka_curve.png

用法:
  from framework.eval.dataset_quality.layer_cka import run_layer_cka
  run_layer_cka(
      checkpoint_dir='checkpoints_v4_temporal',
      dataset_root='datasets/mmfi/v4',
      output_dir='results',
      seeds=[0,1,2],
  )

参考: Kornblith et al. (2019) "Similarity of Neural Network Representations
Revisited" — Linear CKA 实现。
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Linear CKA (Kornblith et al. 2019)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two sample×feature matrices.

    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    X: (N, D_x), Y: (N, D_y), N samples. Per-feature centering.
    Returns float in [0, 1].
    """
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, ord='fro') ** 2
    den_x = np.linalg.norm(Xc.T @ Xc, ord='fro')
    den_y = np.linalg.norm(Yc.T @ Yc, ord='fro')
    den = den_x * den_y
    if den < 1e-12:
        return 0.0
    return float(num / den)


# ---------------------------------------------------------------------------
# Layer-wise feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_layerwise_features(
    model,
    samples,
    device: str = 'cuda',
    batch_size: int = 64,
    hook_points: List[str] = ('enc_out', 'layer1_out', 'pool_out'),
) -> Dict[str, Dict[str, np.ndarray]]:
    """对每个 (sample, hook_point, modality) 抽 (N, D) 特征矩阵。

    Strategy:
      1) 强制全模态可用 (eval=True, avail all True)，保证 batch 间 token 顺序稳定
      2) per-modality encoder 输出 = 直接 forward 每个 encoder
      3) transformer 中间层用 forward_hook 抓取 (B, 80, D) 输出，按 MODALITIES 切回各模态 (B, 16, D)
      4) mean pool 每模态的 16 token -> (B, D_model)
      5) 沿 batch 拼接 -> (N, D_model)

    Returns: {hook_name: {modality: np.ndarray (N, D)}}
    """
    from framework.models.token_fusion import MODALITIES  # type: ignore
    model.eval().to(device)

    # 验证每个样本都有 5 模态（强制全模态的前提）
    for s in samples[:5]:
        missing = [m for m in MODALITIES if m not in s.modalities]
        if missing:
            raise ValueError(
                f"Sample {s.id} 缺失模态 {missing}，layer_cka 要求 val 全模态可用。"
                f"（drop-modality robustness 不在本工具范围）")

    # Hook buffer: {hook_name: {modality: list of (B, D)}}
    buffers: Dict[str, Dict[str, List[np.ndarray]]] = {
        h: {m: [] for m in MODALITIES} for h in hook_points
    }

    # ---------- per-modality encoder 输出（最简单：直接调 self.encoders[m]）----------
    enc_out_per_mod: Dict[str, List[np.ndarray]] = {m: [] for m in MODALITIES}

    # ---------- transformer hook：拿 layer1_out (即 fusion 输出) ----------
    # nn.TransformerEncoder 没有原生 per-layer hook——但它的 layers 是 nn.ModuleList，
    # 我们手动把最后一层 (layers[-1]) 包一层让它在 forward 时存结果
    fusion = model.fusion
    # 先确认是 2 层
    assert hasattr(fusion, 'layers'), "model.fusion 不是 nn.TransformerEncoder"
    last_layer = fusion.layers[-1]
    captured: Dict[str, torch.Tensor] = {}

    def last_layer_hook(_mod, _inp, out):
        # TransformerEncoderLayer 输出: (B, seq, D)
        captured['layer1_out'] = out.detach()

    h_last = last_layer.register_forward_hook(last_layer_hook)

    # ---------- 跑 batch ----------
    try:
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            avail = {m: True for m in MODALITIES}
            # 不调 model._stack_mods（避免训练期 augmentation / numpy ABI 兼容问题），
            # 自己实现 eval 版的 stack — 干净输入 + 全模态可用。
            mods = _eval_stack_mods(model, batch, avail, device)

            # ---- 1) per-modality encoder ----
            for m in MODALITIES:
                e_out = model.encoders[m](mods[m])
                # temporal path: (B, T, N, D) -> (B, N, D)
                if e_out.dim() == 4 and model.temporal:
                    e_out = model.temporal_agg[m](e_out)
                # (B, N, D) -> mean pool over N -> (B, D)
                feat = e_out.mean(dim=1).cpu().numpy().astype(np.float32)
                enc_out_per_mod[m].append(feat)

            # ---- 2) full forward (拿 layer1_out + pool_out) ----
            _ = model(mods, avail)
            if 'layer1_out' not in captured:
                raise RuntimeError("hook 未捕获 layer1_out——检查 model.fusion 结构")
            t_out = captured['layer1_out']  # (B, 80, D)
            # 按 MODALITIES 切回各模态 (每模态 16 token)
            for i_m, m in enumerate(MODALITIES):
                seg = t_out[:, i_m * 16:(i_m + 1) * 16, :]  # (B, 16, D)
                feat = seg.mean(dim=1).cpu().numpy().astype(np.float32)
                buffers['layer1_out'][m].append(feat)

            # pool_out = t_out.mean(dim=1) 与原 head 决策对齐
            pool = t_out.mean(dim=1).cpu().numpy().astype(np.float32)
            # 注: pool_out 是**融合后**的全局表征——不分模态。
            # 我们用 fusion 输入前的 per-modality encoder 输出 (enc_out) 作为
            # "浅层模态特异基线"，layer1_out 作为"深层共享表征"。
            # pool_out 留作辅助（如果用户要看 head 决策空间）。
    finally:
        h_last.remove()

    # ---- 收尾：cat 沿 batch ----
    out: Dict[str, Dict[str, np.ndarray]] = {}
    # enc_out: 直接来自 enc_out_per_mod（per-modality encoder 单独 forward，无 hook）
    out['enc_out'] = {m: np.concatenate(enc_out_per_mod[m], axis=0)
                      for m in MODALITIES}
    # layer1_out / pool_out: 从 buffers 取
    for h in hook_points:
        if h == 'enc_out':
            continue  # 上面已处理
        out[h] = {}
        for m in MODALITIES:
            if h == 'pool_out':
                # pool_out 不分模态（融合后）；跳过 per-modality 累加
                continue
            arrs = buffers[h][m]
            if not arrs:
                raise RuntimeError(f"hook={h} mod={m} 无样本")
            out[h][m] = np.concatenate(arrs, axis=0)  # (N, D)
    return out


# ---------------------------------------------------------------------------
# 简易 eval 版 _stack_mods（不调原方法，避免训练期 augmentation / numpy ABI）
# ---------------------------------------------------------------------------

def _eval_stack_mods(model, batch, avail, device: str) -> Dict[str, torch.Tensor]:
    """Eval-only stack：每个模态纯 torch tensor、无 dropout / 无 time mask / 全模态可用。
    保持与 _stack_mods 相同的 tensor 形状约定。"""
    from framework.models.token_fusion import MODALITIES  # type: ignore
    import numpy as np
    mods = {}
    for m in MODALITIES:
        if not avail.get(m, False):
            continue
        arrs = [s.modalities[m].data for s in batch]
        if m in model.structured:
            mods[m] = torch.as_tensor(
                np.stack(arrs), dtype=torch.float32, device=device)
        else:
            mods[m] = torch.as_tensor(
                np.stack(arrs), dtype=torch.float32, device=device)
    return mods


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def compute_layer_cka_matrix(
    feats: Dict[str, Dict[str, np.ndarray]],
    modalities: List[str],
    hook_points: List[str],
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """对每个 hook_point，计算所有 (mod_a, mod_b) CKA 矩阵 (含对称 + 自身=1)。

    Returns: {hook_point: {(mod_a, mod_b): cka_value}}
    """
    out: Dict[str, Dict[Tuple[str, str], float]] = {}
    for h in hook_points:
        out[h] = {}
        for ma in modalities:
            for mb in modalities:
                key = (ma, mb)
                if ma == mb:
                    out[h][key] = 1.0
                elif (mb, ma) in out[h]:
                    out[h][key] = out[h][(mb, ma)]
                else:
                    X = feats[h][ma]
                    Y = feats[h][mb]
                    # 截断到较短样本数
                    n = min(X.shape[0], Y.shape[0])
                    out[h][key] = linear_cka(X[:n], Y[:n])
    return out


def run_layer_cka(
    checkpoint_dir: str,
    dataset_root: str,
    output_dir: str,
    seeds: List[int] = (0, 1, 2),
    batch_size: int = 64,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict:
    """完整跑一遍：load checkpoints + extract features + compute CKA + save."""
    from framework.dataset.loader import load_dataset  # type: ignore
    from framework.models.token_fusion import TokenFusionModel, MODALITIES  # type: ignore

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[layer_cka] loading dataset {dataset_root}...", flush=True)
    # 用大 cache 把 val 全部 cache 住（避免 3 seeds 重复 IO）
    ds = load_dataset(dataset_root, mode='lazy', cache_size=10000)
    val = ds.splits['val']
    n_val = len(val)
    print(f"[layer_cka] val split: {n_val} samples — prewarming cache...", flush=True)
    # 预热：访问所有 val 样本一次。LazySplit.__getitem__ 会触发 load 并 cache。
    t0 = time.time()
    for i in range(n_val):
        _ = val[i]
    print(f"[layer_cka] cache prewarmed: {n_val} samples in {time.time()-t0:.1f}s", flush=True)

    hooks = ['enc_out', 'layer1_out']  # pool_out 不分模态，留给 head input 视角
    modalities = MODALITIES

    # per-seed CKA + per-modality features for diagnostic
    per_seed_cka: Dict[int, Dict[str, Dict[Tuple[str, str], float]]] = {}

    for seed in seeds:
        ckpt = os.path.join(checkpoint_dir, f'token_fusion_seed{seed}.pt')
        if not os.path.exists(ckpt):
            print(f"[layer_cka] WARN: missing {ckpt}, skip seed {seed}")
            continue
        print(f"[layer_cka] seed {seed}: loading {ckpt}...")
        model = TokenFusionModel.load(ckpt)
        feats = extract_layerwise_features(
            model, val, device=device, batch_size=batch_size, hook_points=hooks)
        cka = compute_layer_cka_matrix(feats, modalities, hooks)
        per_seed_cka[seed] = cka
        # 打印关键对
        for h in hooks:
            r = cka[h].get(('mmwave', 'rgb'))
            m = cka[h].get(('mmwave', 'lidar'))
            print(f"  seed{seed} {h}: mmwave×rgb={r:.3f}  mmwave×lidar={m:.3f}")

    # aggregate across seeds (mean ± std)
    aggregated: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
    for h in hooks:
        aggregated[h] = {}
        for ma in modalities:
            for mb in modalities:
                vals = [per_seed_cka[s][h][(ma, mb)]
                        for s in per_seed_cka if (ma, mb) in per_seed_cka[s][h]]
                if vals:
                    aggregated[h][(ma, mb)] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'n_seeds': len(vals),
                    }

    # save JSON
    out_json = {
        'dataset': dataset_root,
        'checkpoint_dir': checkpoint_dir,
        'seeds_used': list(per_seed_cka.keys()),
        'hooks': hooks,
        'modalities': modalities,
        'aggregated': {
            h: {f"{a}__{b}": v for (a, b), v in d.items()}
            for h, d in aggregated.items()
        },
        'per_seed': {
            str(s): {h: {f"{a}__{b}": v for (a, b), v in d.items()}
                     for h, d in cka.items()}
            for s, cka in per_seed_cka.items()
        },
    }
    json_path = os.path.join(output_dir, 'layer_cka_v4.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"[layer_cka] saved {json_path}")
    return out_json