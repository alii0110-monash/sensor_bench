# SensorBench M6b: 编码器对齐质量提升（大 batch / 分类辅助 loss / 负样本挖掘）Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 5 变体实验矩阵（batch 256 / 分类辅助 loss / label-aware 负样本挖掘）提升 AlignmentModel 编码器对齐质量，L1 检索 recall@k 相对基线（r@1=0.0066）显著提升。

**Architecture:** 三个独立可开关的改进手段叠加到现有 InfoNCE 对齐训练上：(1) batch 32→256（负样本 8x）；(2) `AlignmentModel` 增加独立 `classification_head`（256→27，从 token_fusion 预热），训练 `L = info_nce + λ·CE`；(3) `info_nce_loss` 增加可选 `labels`，排除同 label 负样本（保留对角线正样本 + 最少负样本保底）。5 变体增量对比，统一 seed/epochs/lr/init 保证公平。

**Tech Stack:** Python 3.12, torch 2.7.1+cu128, numpy, pytest。运行/测试用 `/home/li/bin/python3` 与 `/home/li/bin/pytest`（已验证 torch+CUDA 可用）。checkpoints: `checkpoints_alignment/`、预热权重 `checkpoints_v4/token_fusion_seed0.pt`。

**前置:** spec `docs/superpowers/specs/2026-08-16-alignment-quality-m6b-design.md`（2 轮评审 Approved）。当前 HEAD: `451b465`。

---

## 文件结构

```
framework/models/alignment.py       # 修改: info_nce_loss 加 labels; AlignmentModel 加 classification_head + forward_loss
scripts/train_alignment.py          # 修改: 新参数 --aux-cls-weight/--neg-mine/--out-tag/--cache-size; seed; 组合 loss; checkpoint 剔除
scripts/eval_alignment.py           # 修改: --diagnose-label 诊断子指标
tests/test_alignment.py             # 修改: 新增 label-aware / forward_loss / min-negatives 单测
tests/test_alignment_e2e.py         # 修改: mini 组合训练 + checkpoint roundtrip
docs/reports/m6b_alignment_matrix.md # 生成: 实验对比表（评测流程产出）
checkpoints_alignment/m6b_{A..E}_seed0.pt  # 生成: 5 变体 checkpoint
```

---

## Task 1: info_nce_loss 支持 label-aware 负样本排除

**Files:**
- Modify: `framework/models/alignment.py:19-26`
- Test: `tests/test_alignment.py`（新增测试函数）

**说明**: 为 `info_nce_loss` 增加可选 `labels`。当提供时，同 label 非对角线对的 logits 置 `-inf`（排除）；**对角线正样本必须保留**（`mask[arange, arange]=False`，否则整行 -inf → NaN）；每行可用负样本数 < `min_negatives` 时该行不 mask（保底防梯度消失）。不传 labels 时行为与原来完全一致（向后兼容）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_alignment.py 追加 (顶部已 import torch; 新增 import 如下)
from framework.models.alignment import info_nce_loss, _masked_logits

def test_info_nce_label_aware_mask():
    # B=16, 2 类 × 8 → 每行可用负样本 = 8 ≥ min_negatives=8, 保底不触发, 全部 mask
    z = torch.randn(16, 128)
    t = z.clone()
    labels = torch.tensor([i // 8 for i in range(16)])   # 8×label0, 8×label1
    z_n = torch.nn.functional.normalize(z, dim=-1)
    t_n = torch.nn.functional.normalize(t, dim=-1)
    logits = z_n @ t_n.t() / 0.07
    masked_logits = _masked_logits(logits, labels)
    # 对角线必须保留 (正样本)
    assert torch.isfinite(torch.diag(masked_logits)).all()
    # 同 label 非对角 (-inf) 被排除 (8 个同 label, 减自身=7 个被 mask)
    for i in range(16):
        for j in range(16):
            if i != j and labels[i] == labels[j]:
                assert masked_logits[i, j] == float("-inf")
    # 跨 label 保留
    for i in range(16):
        for j in range(16):
            if i != j and labels[i] != labels[j]:
                assert torch.isfinite(masked_logits[i, j])

def test_info_nce_label_aware_min_negatives():
    # 每行负样本数 < min_negatives 时不 mask (保底)
    z = torch.randn(4, 128)
    t = z.clone()
    labels = torch.tensor([0, 0, 0, 0])  # 每行同 label 排除后剩 0 负样本
    masked = _masked_logits(z @ t.t() / 0.07, labels)
    # 全部保留 (保底触发, 退化为普通 InfoNCE)
    assert torch.isfinite(masked).all()
```

- [ ] **Step 2: 跑测试验证失败**

Run: `/home/li/bin/pytest tests/test_alignment.py -v -k label_aware`
Expected: FAIL（`ImportError: cannot import name '_masked_logits'`）

- [ ] **Step 3: 实现 label-aware InfoNCE**

```python
# framework/models/alignment.py — 替换 info_nce_loss (保留 F.normalize 向后兼容)
def info_nce_loss(z: torch.Tensor, t: torch.Tensor, temperature: float = 0.07,
                  labels: torch.Tensor | None = None, min_negatives: int = 8) -> torch.Tensor:
    """InfoNCE. labels 提供时排除同 label 负样本 (label-aware);
    对角线正样本恒保留; 负样本不足 min_negatives 的行不排除.
    与旧版行为一致 (z/t 均 L2 归一化, 温度生效)."""
    z = F.normalize(z, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = z @ t.t() / temperature
    if labels is not None:
        logits = _masked_logits(logits, labels, min_negatives)
    B = logits.shape[0]
    idx = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, idx)


def _masked_logits(logits: torch.Tensor, labels: torch.Tensor,
                   min_negatives: int = 8) -> torch.Tensor:
    B = labels.shape[0]
    same = labels[:, None] == labels[None, :]          # (B,B) 同 label
    same[torch.arange(B), torch.arange(B)] = False     # 对角线(正样本)保留
    n_neg = (~same).sum(dim=1) - 1                     # 每行可用负样本数(减对角线)
    guard = n_neg >= min_negatives                     # 保底
    mask = same & guard[:, None]
    return logits.masked_fill(mask, float("-inf"))
```

> 注: `_masked_logits` 为模块级辅助函数（测试直接导入）。`temperature` 已归一化进 logits，`_masked_logits` 不再接收。保底默认 `min_negatives=8`。

- [ ] **Step 4: 跑测试验证通过**

Run: `/home/li/bin/pytest tests/test_alignment.py -v -k label_aware`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add framework/models/alignment.py tests/test_alignment.py
git commit -m "feat(align): info_nce_loss 支持 label-aware 负样本排除 (对角线保留+保底)"
```

---

## Task 2: AlignmentModel 增加 classification_head + forward_loss

**Files:**
- Modify: `framework/models/alignment.py:29-67`
- Test: `tests/test_alignment.py`（新增测试函数）

**说明**: `AlignmentModel` 增加可选 `num_classes` 参数；非 None 时创建独立 `classification_head = nn.Linear(D=256, num_classes)`（不共享 projection_head 参数）。`forward_loss` 改为返回 `(info_nce, ce)` 元组（无分类头或未传 labels 时 ce=None）。训练时若启用 CE 且给了 labels，返回 `cross_entropy(classification_head(pooled), labels)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_alignment.py 追加
import torch
import pytest
from framework.models.alignment import AlignmentModel

def test_alignment_aux_cls_forward():
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    assert m.classification_head is not None
    assert m.classification_head.in_features == 256
    assert m.classification_head.out_features == 27
    mods = _toy_mods()   # 复用文件内已有 helper (4 样本)
    avail = {k: True for k in mods}
    text = torch.randn(4, 512)
    labels = torch.randint(0, 27, (4,))
    info_nce, ce = m.forward_loss(mods, text, avail, labels=labels, neg_mine=False)
    assert info_nce.shape == () and torch.isfinite(info_nce)
    assert ce.shape == () and torch.isfinite(ce)

def test_alignment_aux_cls_forward_neg_mine():
    # neg_mine=True 时 info_nce 用 labels 排除同 label 负样本, 仍返回 (info_nce, ce)
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    mods = _toy_mods()
    avail = {k: True for k in mods}
    labels = torch.tensor([0, 0, 1, 1])
    info_nce, ce = m.forward_loss(mods, torch.randn(4, 512), avail,
                                  labels=labels, neg_mine=True)
    assert info_nce.shape == () and torch.isfinite(info_nce)
    assert ce.shape == () and torch.isfinite(ce)

def test_alignment_no_classification_head():
    m = AlignmentModel(num_modalities=5, text_dim=512)  # 默认无分类头
    assert m.classification_head is None
    mods = _toy_mods()
    avail = {k: True for k in mods}
    info_nce, ce = m.forward_loss(mods, torch.randn(4, 512), avail)
    assert ce is None and info_nce.shape == ()
```

- [ ] **Step 2: 跑测试验证失败**

Run: `/home/li/bin/pytest tests/test_alignment.py -v -k "aux_cls or classification_head"`
Expected: FAIL（`AttributeError: ... forward_loss` 返回元组 / `classification_head` 不存在）

- [ ] **Step 3: 实现**

```python
# framework/models/alignment.py — __init__ 增加 num_classes
    def __init__(self, num_modalities: int = 5, text_dim: int = 512,
                 dropout_p: float = 0.25, num_classes: int | None = None):
        ...
        self.classification_head = nn.Linear(D, num_classes) if num_classes else None

# — forward_loss 改为返回 (info_nce, ce)
    def forward_loss(self, mods, text_emb, avail, labels=None, neg_mine=False):
        toks = self.encode_modalities(mods, avail)
        pooled = self.pool(toks)
        z = self.projection_head(pooled)
        info_nce = info_nce_loss(z, text_emb, labels=labels if neg_mine else None)
        ce = None
        if self.classification_head is not None and labels is not None:
            ce = torch.nn.functional.cross_entropy(self.classification_head(pooled), labels)
        return info_nce, ce
```

- [ ] **Step 4: 跑测试验证通过**

Run: `/home/li/bin/pytest tests/test_alignment.py -v`
Expected: PASS（全部通过，含旧测试——`_toy_mods` 为 4 样本，兼容）

- [ ] **Step 5: 提交**

```bash
git add framework/models/alignment.py tests/test_alignment.py
git commit -m "feat(align): AlignmentModel 增加 classification_head + forward_loss 返回 (info_nce, ce)"
```

---

## Task 3: train_alignment.py 新参数 + 组合 loss + checkpoint 兼容

**Files:**
- Modify: `scripts/train_alignment.py`
- Test: `tests/test_alignment_e2e.py`（新增/修改）

**说明**: 
- 新参数：`--aux-cls-weight`（float，默认 0.0，>0 启用 CE 辅助）、`--neg-mine`（flag，启用 label-aware 排除）、`--out-tag`（str，默认空；非空时 checkpoint 名为 `{out}/m6b_{tag}_seed0.pt`）、`--cache-size`（int，默认 256；batch=256 变体传 4096）。
- 补 `torch.manual_seed(0)`（当前脚本无 seed，变体间不可复现）。
- 模型构造：`AlignmentModel(num_modalities=5, text_dim=te.dim, num_classes=27 if args.aux_cls_weight > 0 else None)`。
- 分类头预热：`--init-encoders` 已加载 tf 权重时，若 `args.aux_cls_weight > 0`，从 `tf.head`（`Linear(256,27)`）复制 weight/bias 到 `model.classification_head`。
- 训练循环改用 `model.forward_loss(mods, text_emb, avail, labels=labels, neg_mine=args.neg_mine)`，`labels = torch.tensor([s.label for s in batch])`。组合 `L = info_nce + args.aux_cls_weight * ce`。
- **best 按 info_nce 分量选**（C/E 组合 loss 与 A/B/D 单 InfoNCE 判据不一致，统一按 info_nce 分量保存最优，保证跨变体可比）。
- **checkpoint 剔除 `classification_head.*` 键**（只存 encoders+projection_head，eval 用默认 AlignmentModel strict 加载兼容）。

- [ ] **Step 1: 写/改失败测试**

```python
# tests/test_alignment_e2e.py — 修改 test_train_alignment_mini (适配 train_epoch 返回 tuple)
def test_train_alignment_mini(tmp_path):
    root = _mini_v5(tmp_path)
    ds = load_dataset(str(root))
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    te = HashTextEncoder(dim=512)
    cfg = {"epochs": 2, "batch_size": 4, "lr": 1e-3, "device": "cpu"}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from train_alignment import train_epoch
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"])
    p0 = list(m.parameters())[0].detach().clone()
    loss, nce, ce = train_epoch(m, te, ds.train, opt, batch_size=4, device="cpu",
                                aux_cls_weight=0.5, neg_mine=True)
    assert loss > 0 and torch.isfinite(torch.tensor(loss))
    assert torch.isfinite(torch.tensor(nce)) and torch.isfinite(torch.tensor(ce))
    assert not torch.allclose(p0, list(m.parameters())[0].detach())

# tests/test_alignment_e2e.py — 新增 checkpoint roundtrip
def test_checkpoint_roundtrip_strips_cls_head(tmp_path):
    import torch
    from framework.models.alignment import AlignmentModel
    m = AlignmentModel(num_modalities=5, text_dim=512, num_classes=27)
    state = m.state_dict()
    assert any(k.startswith("classification_head.") for k in state)
    stripped = {k: v for k, v in state.items()
                if not k.startswith("classification_head.")}
    ev = AlignmentModel(num_modalities=5, text_dim=512)   # eval 默认无分类头
    ev.load_state_dict(stripped)                          # strict=True 应通过
```

- [ ] **Step 2: 跑测试验证失败**

Run: `/home/li/bin/pytest tests/test_alignment_e2e.py -v`
Expected: FAIL（`train_epoch` 返回 float，不能解包 3 元组；`forward_loss`/`num_classes` 缺失）

- [ ] **Step 3: 实现 train_epoch 改造**

```python
# scripts/train_alignment.py — train_epoch 替换
def train_epoch(model, text_encoder, train, opt, batch_size=32, device="cuda",
                dropout_p=0.25, aux_cls_weight=0.0, neg_mine=False) -> tuple:
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
```

- [ ] **Step 4: 实现 main() 改造**

```python
# scripts/train_alignment.py — main() 关键改动
    ap.add_argument("--aux-cls-weight", type=float, default=0.0,
                    help="分类辅助 loss 权重 (0=关闭, 如 0.5)")
    ap.add_argument("--neg-mine", action="store_true",
                    help="label-aware 负样本排除")
    ap.add_argument("--out-tag", default="",
                    help="checkpoint 名后缀: {out}/m6b_{tag}_seed0.pt (空=alignment_seed0.pt)")
    ap.add_argument("--cache-size", type=int, default=256,
                    help="lazy loader cache_size (batch=256 时传 4096 防 cache thrash)")

    torch.manual_seed(0)                       # 新增: 变体间可复现
    ds = load_dataset(args.dataset, cache_size=args.cache_size)
    ...
    model = AlignmentModel(num_modalities=5, text_dim=te.dim,
                           num_classes=27 if args.aux_cls_weight > 0 else None).to(device)
    ...
    # 分类头预热 (aux 启用时从 tf.head 复制; 必须放在 --init-encoders 块之后,
    # 依赖该块内的 tf 作用域; 若 init-encoders 缺失会 NameError, 本次 5 变体均存在该文件)
    if args.aux_cls_weight > 0:
        with torch.no_grad():
            model.classification_head.weight.copy_(tf.head.weight)
            model.classification_head.bias.copy_(tf.head.bias)
        print("[alignment] 分类头从 token_fusion head 预热", flush=True)
    ...
    best = 1e9
    for ep in range(args.epochs):
        loss, nce, ce = train_epoch(model, te, ds.train, opt,
                                    batch_size=args.batch_size, device=device,
                                    aux_cls_weight=args.aux_cls_weight,
                                    neg_mine=args.neg_mine)
        print(f"[alignment] ep {ep} loss {loss:.4f} (info_nce {nce:.4f}"
              f"{', ce ' + f'{ce:.4f}' if ce > 0 else ''})", flush=True)
        if nce < best:                          # 按 info_nce 分量选 best (跨变体可比)
            best = nce
            state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("classification_head.")}   # 剔除分类头
            name = f"m6b_{args.out_tag}_seed0.pt" if args.out_tag else "alignment_seed0.pt"
            torch.save(state, f"{args.out}/{name}")
    print(f"done -> {args.out}/{name if args.out_tag else 'alignment_seed0.pt'}")
```

- [ ] **Step 5: 跑测试验证通过**

Run: `/home/li/bin/pytest tests/test_alignment_e2e.py -v`
Expected: PASS（全部通过）

- [ ] **Step 6: 提交**

```bash
git add scripts/train_alignment.py tests/test_alignment_e2e.py
git commit -m "feat(train): --aux-cls-weight/--neg-mine/--out-tag/--cache-size + seed + checkpoint 剔除分类头"
```

---

## Task 4: eval_alignment.py --diagnose-label 诊断子指标

**Files:**
- Modify: `scripts/eval_alignment.py`

**说明**: 增加 `--diagnose-label` 标志。启用时额外统计：(1) 每个 query 正样本的 rank（1-indexed）均值；(2) 每个 query 排在其正样本之前的**同 label 负样本数**均值。用于理解"同 label 负样本是否构成 r@1 主要干扰、label-aware 排除是变好还是变坏"。

- [ ] **Step 1: 实现**

```python
# scripts/eval_alignment.py — main() 增加参数与逻辑
    ap.add_argument("--diagnose-label", action="store_true",
                    help="输出诊断: 正样本平均 rank + 排前面的同 label 负样本数")

    res = evaluate_retrieval(model, te, held_samples, device=device)
    print(f"[eval] n={res['n']} r@1={res['r@1']:.4f} ...")
    if args.diagnose_label:
        labels = [s.label for s in held_samples]
        mean_rank, same_above = _diagnose_label(model, te, held_samples, labels, device)
        print(f"[eval] diagnose: mean_rank={mean_rank:.1f} same_label_above_pos={same_above:.2f}")


def _diagnose_label(model, te, samples, labels, device, batch_size=64):
    """每 query 正样本 rank + 排在前面的同 label 负样本数 (均值)."""
    import numpy as np
    zs, ts = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            avail = {m: True for m in model.encoders}
            mods = {m: torch.stack([torch.from_numpy(s.modalities[m].data) for s in batch]).to(device)
                    for m in avail if m in batch[0].modalities}
            texts = [s.text.get("en", [""])[0] for s in batch]
            zs.append(model.projection_head(model.pool(model.encode_modalities(mods, avail))))
            ts.append(te.encode(texts).to(device))
    Z = torch.cat(zs); T = torch.cat(ts)
    Z = torch.nn.functional.normalize(Z, dim=-1); T = torch.nn.functional.normalize(T, dim=-1)
    sim = Z @ T.t()                                  # (N,N)
    n = sim.shape[0]
    lab = torch.tensor(labels)
    ranks, same_above = [], []
    for i in range(n):
        above = (sim[i] > sim[i, i])
        ranks.append(above.sum().item() + 1)
        same_above.append((above & (lab == labels[i]) & (torch.arange(n) != i)).sum().item())
    return float(np.mean(ranks)), float(np.mean(same_above))
```

- [ ] **Step 2: 冒烟验证（不用 GPU 训练，直接对 baseline checkpoint 跑）**

Run: `/home/li/bin/python3 scripts/eval_alignment.py --ckpt checkpoints_alignment/alignment_seed0.pt --prototype-head --diagnose-label --fraction 0.02`
Expected: 正常输出 r@k + diagnose 两行（数值合理，无异常）

- [ ] **Step 3: 提交**

```bash
git add scripts/eval_alignment.py
git commit -m "feat(eval): --diagnose-label 输出正样本 rank + 同label负样本前置数"
```

---

## Task 5: 实验矩阵运行 + 评测 + 对比表

**Files:**
- Run: `scripts/train_alignment.py`（5 变体）
- Run: `scripts/eval_alignment.py`（每变体 + baseline）
- Create: `docs/reports/m6b_alignment_matrix.md`

**说明**: 训练 5 变体（后台 + 资源监控），统一控制变量：epochs=20、lr=1e-3、init-encoders=checkpoints_v4/token_fusion_seed0.pt、dropout_p=0.25（默认）、`torch.manual_seed(0)`、全部 `--init-prototype`。变体 C/E 用 `--aux-cls-weight 0.5`。B-E 用 `--batch-size 256 --cache-size 4096`。

- [ ] **Step 1: 训练变体 A（基线复现，batch=32）**

```bash
cd /home/li/projects/sensorbench
setsid bash -c '/home/li/bin/python3 scripts/train_alignment.py \
  --out-tag A --batch-size 32 --init-prototype \
  > logs/train_m6b_A.log 2>&1' < /dev/null > /dev/null 2>&1 &
```
> 预期：~20 epochs，batch=32 全 train 46509 → 1454 step/epoch。监控 `logs/train_m6b_A.log` + GPU/内存。完成后验证 `checkpoints_alignment/m6b_A_seed0.pt` 存在。

- [ ] **Step 2: 训练变体 B（batch=256）**

```bash
setsid bash -c '/home/li/bin/python3 scripts/train_alignment.py \
  --out-tag B --batch-size 256 --cache-size 4096 --init-prototype \
  > logs/train_m6b_B.log 2>&1' < /dev/null > /dev/null 2>&1 &
```
> 预期：~182 step/epoch。监控。

- [ ] **Step 3: 训练变体 C（batch=256 + 分类辅助 loss λ=0.5）**

```bash
setsid bash -c '/home/li/bin/python3 scripts/train_alignment.py \
  --out-tag C --batch-size 256 --cache-size 4096 --init-prototype \
  --aux-cls-weight 0.5 > logs/train_m6b_C.log 2>&1' < /dev/null > /dev/null 2>&1 &
```

- [ ] **Step 4: 训练变体 D（batch=256 + 负样本挖掘）**

```bash
setsid bash -c '/home/li/bin/python3 scripts/train_alignment.py \
  --out-tag D --batch-size 256 --cache-size 4096 --init-prototype \
  --neg-mine > logs/train_m6b_D.log 2>&1' < /dev/null > /dev/null 2>&1 &
```

- [ ] **Step 5: 训练变体 E（全组合）**

```bash
setsid bash -c '/home/li/bin/python3 scripts/train_alignment.py \
  --out-tag E --batch-size 256 --cache-size 4096 --init-prototype \
  --aux-cls-weight 0.5 --neg-mine > logs/train_m6b_E.log 2>&1' < /dev/null > /dev/null 2>&1 &
```
> 资源控制：5 变体串行跑（避免 5× 并发占满 16GB GPU 与 25GB RAM）。每个变体训练时前台轮询监控（日志尾部 + `nvidia-smi` + `free -h`），完成一个再启下一个。卡住/异常立即报告并给选项。

- [ ] **Step 6: 评测全部变体 + 基线**

```bash
for V in A B C D E; do
  /home/li/bin/python3 scripts/eval_alignment.py \
    --ckpt checkpoints_alignment/m6b_${V}_seed0.pt --prototype-head \
    --diagnose-label --fraction 0.1 >> logs/eval_m6b.log 2>&1
done
# 基线对齐（对比参考）
/home/li/bin/python3 scripts/eval_alignment.py \
  --ckpt checkpoints_alignment/alignment_seed0.pt --prototype-head \
  --diagnose-label --fraction 0.1 >> logs/eval_m6b.log 2>&1
```

- [ ] **Step 7: 判定 + 写对比表**

判定规则（spec）：r@1 提升 ≥ 2SE（n≈969, p≈0.0066 → SE≈0.0026，阈值≈0.005；从严取 0.0075）且 r@5/10 与 tr@k 同向才采信；否则记录负结果并分析诊断指标。

写 `docs/reports/m6b_alignment_matrix.md`：
```markdown
# M6b 对齐质量实验矩阵
| 变体 | 配置 | r@1 | r@5 | r@10 | tr@1 | mean_rank | same_label_above |
|---|---|---|---|---|---|---|---|
| baseline | CLIP512 batch32 | 0.0066 | ... | ... | ... | ... | ... |
| A | 复现 | ... | ... | ... | ... | ... | ... |
| B | batch256 | ... | ... | ... | ... | ... | ... |
| C | +CE0.5 | ... | ... | ... | ... | ... | ... |
| D | +neg-mine | ... | ... | ... | ... | ... | ... |
| E | 全 | ... | ... | ... | ... | ... | ... |
**结论**: [最优变体 / 无显著提升原因分析]
```

- [ ] **Step 8: 提交**

```bash
git add docs/reports/m6b_alignment_matrix.md
git commit -m "docs(report): M6b 对齐质量实验矩阵 (A-E 变体 L1 评测)"
```

---

## Task 6: STATUS.md 决策层更新

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: 更新决策层与判断层**

把 M6b 决策项改为 `[已定]` 并记录实验结果（最优变体、r@1 提升或负结果），补判断层结论。用项目 status 流程（`python tools/project_status.py scan STATUS.md` 刷新事实层）。

- [ ] **Step 2: 提交**

```bash
git add STATUS.md
git commit -m "docs(status): M6b 完成 — 编码器对齐质量实验结论"
```

---

## 评审

- [ ] plan-document-reviewer 评审通过后执行
