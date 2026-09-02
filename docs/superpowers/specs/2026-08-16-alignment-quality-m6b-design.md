# M6b 编码器对齐质量提升（大 batch / 分类辅助 loss / 负样本挖掘）设计

- 日期: 2026-08-16
- 状态: 设计定稿
- 前置: M6a 完成（CanonicalToken 协议 + 资产化 + LinearTokenToLLM）。当前对齐基线 L1 r@1=0.0066（弱）。
- 目标: 提升编码器（AlignmentModel.encoders）对齐质量，L1 检索 recall@k 提升。

## 背景

M5a 建立的 stage-1 对齐：CLIP 512 维文本锚 + InfoNCE（batch=32, 20 epochs, lr=1e-3），编码器从 v4 token_fusion 预热。当前 L1 r@1=0.0066 偏弱，编码器对齐质量是 CanonicalToken 表征质量的上限（M6a spec 背景：伪 token 作为可移植跨模态统一表征，编码器质量是核心）。

## 已确认的关键决策

| 决策点 | 选择 |
|---|---|
| 对齐锚空间 | **保持 CLIP 512**（不改 llama2 4096；训练锚与评测锚一致，手段效果可闭环对比） |
| 手段范围 | **三种手段都实验**（大 batch / 分类辅助 loss / 负样本挖掘），增量对比各自贡献 |
| 评测口径 | CLIP 512 L1（held-out base, r@1/5/10，复用现有 eval_alignment.py） |
| 产出 | 5 变体 checkpoint + 对比表，选最优作为 M6b 新编码器 |
| 范围缩减 | **"锚对比"从 M6b 移除**（M6a 决策表中列为 M6b 候选）——锚已定 CLIP 512，锚对比无意义；推迟到 M6c 一并评估多锚 |

**评测口径限制（必须承认）**：CLIP-512 r@1 是 **4096 canonical 空间质量的代理指标**——编码器经 PerceiverProjection 映射到 4096 后，CLIP-512 空间的对齐提升不一定 1:1 迁移。本迭代以 CLIP-512 闭环对比手段效果；4096 空间迁移验证列入产出 spot-check（见评测流程第 4 步）。

## 三个实验变量

### 1. 大 batch（batch 32→256）

InfoNCE 靠 batch 内负样本（当前 31 个/样本）。batch=256 → 255 负样本（8x）。对齐质量通常随负样本量提升。

- 显存核算：单样本全模态 ~1.12MB × 256 ≈ 287MB 输入 + 小模型（编码器+投影头），16GB GPU 充足。
- 数据量：train 46509（base 9689），batch=256 → ~182 step/epoch，20 epochs 可接受。
- **cache 核算**：loader 默认 `cache_size=256` == 新 batch → 缓存命中率塌陷、逐 epoch 从盘重读（重演 v4 磁盘瓶颈）。batch=256 时必须 `load_dataset(..., cache_size=4096)`（≈4.6GB RAM，可用 25GB 充裕）或核算 eager 可行性（46509×1.12MB≈52GB ≫ 可用，不可 eager）。
- 实施：`train_alignment.py --batch-size 256` + `--cache-size 4096`。

### 2. 分类辅助 loss（L = InfoNCE + λ·CE）

编码器预热自 token_fusion（含分类头 256→27），对齐训练可能破坏动作判别力。加 27 类分类 CE 作为辅助 loss 保持判别力。

- **分类头实现**：`AlignmentModel` 新增**独立** `classification_head = nn.Sequential(Linear(256, 27))`（不共享 projection_head 参数），从 token_fusion 的 `head.weight/bias`（256→27）复制预热。CE 输入 = `pooled`（256 维，`AlignmentModel.pool` 输出）。
  - 注：tf.head 在 token_fusion **融合 transformer** 特征上训练，而 AlignmentModel.pool 是**无融合 mean-pool**，特征分布不同——预期 CE 头需重新适应，预热只是起点。
- `forward_loss(mods, text_emb, avail, labels)` 返回 `(info_nce, ce)` 元组；训练脚本 `L = info_nce + λ * ce`。
- λ 可调（默认 0.5，范围 0.1~1.0）。

### 3. 负样本挖掘（label-aware InfoNCE，实现=排除）

**动机纠正**：L1 检索正样本是 **id 匹配**（同一样本 vs 自身 text），不是同 label。batch 内**同 label 不同实例**的样本是**最硬的真负样本**（模板 caption 近似相同，embedding 接近），正是 r@1 需要分辨的对象。把它们当普通负样本是合理的；把它们排除是"降低难度"——**可能削弱而非增强判别力**。本变体正是为了验证这一点（实验给出答案，不预设结论）。

- 实现（**定死为"排除"**，不做降权）：`info_nce_loss(z, t, labels)` 增加可选 `labels`；对 `(i,j)` 且 `label_i == label_j` 且 `i != j` 的 logits 置 `-inf`。
  - **必须显式保留对角线**：`mask[arange, arange] = False`（对角线是正样本对，label 必相同，否则整行全 -inf → NaN）。
- 保底：某样本排除后负样本数 < 8 时**不排除该行**（退化为普通 InfoNCE），防梯度消失。
- 数量核算：27 类均匀（9689/27≈359/类），batch=256 排除后仍有 ~255×(26/27)≈245 负样本，充足。

## 实验矩阵（增量对比）

| 变体 | batch | 辅助 loss | 挖掘 | 目的 |
|---|---|---|---|---|
| A 基线 | 32 | 无 | 无 | 复现当前 0.0066 |
| B | 256 | 无 | 无 | 大 batch 单独贡献 |
| C | 256 | 有 (λ=0.5) | 无 | +分类辅助 |
| D | 256 | 无 | 有 | +负样本挖掘 |
| E 全 | 256 | 有 | 有 | 全部叠加 |

**控制变量（全变体一致）**：epochs=20、lr=1e-3、init-encoders=checkpoints_v4/token_fusion_seed0.pt、dropout_p=0.25、**`torch.manual_seed(0)`（train_alignment 当前无 seed，需补）**、**init-prototype 全变体同一策略**（基线 A 用原型头复现 0.0066；B-E 统一 `--init-prototype` 保证投影头初始化一致）。

**结论判定（防噪声误判）**：held-out base ~969 → r@1 单点约 6 命中，±0.005 内是纯噪声。判定规则：
1. 先看 r@1，要求提升 **≥ 2SE**（SE≈sqrt(p(1-p)/n)，n≈969，p≈0.0066 → SE≈0.0026，2SE≈0.005）才算显著；
2. 同时综合 r@5/r@10 与 tr@k（多指标同向才采信）；
3. 选最优变体作 M6b 编码器；若所有变体 ≤ 基线+2SE，记录负结果并分析。

**诊断子指标（理解挖掘效果）**：评测时对每个 query 统计**正样本 rank** 及**排在其前面的同 label 负样本数**（均值）。用于回答"同 label 负样本是否构成 r@1 的主要干扰、排除后到底变好还是变坏"。

## 改动点

| 文件 | 改动 |
|---|---|
| `framework/models/alignment.py` | `info_nce_loss` 增加可选 `labels`（label-aware 排除 + 对角线保留 + 最少负样本保底）；`AlignmentModel` 增加 `classification_head` + `forward_loss` 返回 `(info_nce, ce)` |
| `scripts/train_alignment.py` | 新增 `--aux-cls-weight`（默认 0.0，>0 启用）、`--neg-mine`（flag）、`--out-tag`（变体名）、`--cache-size`（默认 256，batch=256 时传 4096）；补 `torch.manual_seed(0)`；分类头预热（从 tf.head 复制）；训练循环组合 loss；**保存 checkpoint 时剔除 `classification_head.*` 键**（只存 encoders+projection_head，eval 兼容） |
| `scripts/eval_alignment.py` | 增加 `--diagnose-label`：输出正样本平均 rank + 同 label 负样本排在正样本之前的平均数（诊断挖掘效果） |
| `tests/test_alignment_e2e.py` 或新测试 | 见测试策略 |

## 评测流程

1. 训练 5 变体（每变体 ~20 epochs，后台跑，监控资源；batch=256 + cache_size=4096）。
2. `scripts/eval_alignment.py --ckpt checkpoints_alignment/m6b_{V}.pt --prototype-head`（CLIP 512 L1，含 r@1/5/10 + tr@1/5/10 + 诊断指标）。
3. 汇总 `docs/reports/m6b_alignment_matrix.md` 对比表。
4. **4096 空间 spot-check**（可选，若步骤 2 有显著提升）：用最优变体 + 现有 PerceiverProjection 重生成 mini v5tokens，跑一次 4096 空间 L1（llama2 文本侧），验证提升迁移。

## 测试策略

- 单元测试（无 GPU/mock，CPU + HashTextEncoder）：
  - `test_info_nce_label_aware`：**对角线正样本 logits 保留为正**；同 label 非对角负样本被 mask（logits=-inf）；不同 label 保留。
  - `test_info_nce_label_aware_min_negatives`：负样本数 < 保底时该行不 mask。
  - `test_alignment_aux_cls_forward`：`forward_loss` 返回 (info_nce, ce) 且 shape 正确。
  - `test_train_alignment_mini_combined`：扩展 mini e2e，`--aux-cls-weight` + `--neg-mine` 组合路径可训练（loss 有限）。
  - `test_checkpoint_roundtrip`：train mini → save（剔除 classification_head）→ eval_alignment 能加载。
- 现有 135 测试保持通过（label-aware 为向后兼容可选参数）。

## 里程碑

- **M6b（本迭代）**：3 手段实验矩阵 + 最优编码器产出 + 对比表。
- M6c：数据质量（弱模态、infra1/infra2 → 7 模态；含多锚评估）。

## 开放问题

- 若大 batch 下 20 epochs 训练时间超预期（>1h），epochs 可减半并记录（变体间保持同 epochs 才能公平对比）。
- λ 敏感性（0.5 vs 1.0）：C/E 变体可另存 λ=1.0 快照作敏感性参考（非必需）。
- 分类辅助 loss 的 CE 头需重新适应 mean-pool 特征（见 §2 注），若 CE 不收敛可降 λ 或冻结前几层再观察。
