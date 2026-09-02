# 声明：用层级 CKA 诊断 MMFi v4 多模态融合机制

> 关联：M6 数据集质量评测系统（[STATUS.md](../../STATUS.md) · `framework/eval/dataset_quality/`）
> 关联背景：用户问"通过已训练网络看浅层是否有特异性、深层是否融合来反推数据质量"
> 生成日期：2026-09-02 · 状态：`[提议]`

---

## 1. 目标（Goal）

**在 `checkpoints_v4_temporal/token_fusion_seed{0,1,2}.pt` 上提取 token_fusion 的逐层中间表征，计算跨模态 CKA 曲线，回答一个具体问题：**

> **v4 上 mmwave 之所以是"真正的互补"（contribution 0.625 + 单 probe 0.348，源自 `dataset_quality_v1_v2_v4.md`），到底是因为它在 transformer 浅层保持模态特异性、深层才与 rgb 对齐；还是它从未对齐、只是被 attention mask 隔离着？**

次要但同源目标：
- 验证"浅层特异 / 深层融合"的理想曲线是否在 v4 上成立
- 把"lidar 单模态弱（acc=0.095）但融合贡献高（0.502）"这个反常归因到具体层

预期产出物（具体到路径）：
- `framework/eval/dataset_quality/layer_cka.py` — 逐层 hook + Linear CKA 计算
- `results/layer_cka_v4.json` — 4 模态两两 CKA × 3 hook 点 × 3 seeds
- `results/plots_v4/layer_cka_curve.png` — **层数 × CKA** 曲线
- `docs/reports/layer_cka_v4.md` — 数据质量诊断报告（含"用户原话"对照）

---

## 2. 现状（Current State）

| 维度 | 现状 | 来源 |
|---|---|---|
| 评测工具 | 仅输入特征 + 最终特征两个端点 | `framework/eval/dataset_quality/{feature_extract,modality_probe,probe_fusion}.py` |
| 已训练模型 | `token_fusion_seed{0,1,2}.pt`（v4 + temporal） | `checkpoints_v4_temporal/` |
| 数据集 | v4 = 4 模态 wifi/depth/lidar/mmwave + 可选 rgb，27 类 | `datasets/mmfi/v4/` |
| 已知结论 | mmwave 是 v4 唯一非冗余互补；wifi/depth/lidar ≈随机但 lidar 仍贡献融合 | `dataset_quality_v1_v2_v4.md` L83-89 |
| M6 缺口 | M6a（v5tokens 可移植性） + M6b（训练手段实验）均未完成 | `STATUS.md` |

**关键缺口**：现有评测是"端点式"的——只看了"模型入口"（probe 输入层）和"模型出口"（concat probe 分类层），中间 2 层 transformer encoder 的表征几何完全没探过。这就是"浅层/深层是否融合"无法回答的根因。

**好消息**：`token_fusion.py` 是 2 层 `nn.TransformerEncoder` + per-modality encoder + Linear head。结构简单清晰、hook 点明确，不需要改模型就能拿到层 0 入、层 0 出、层 1 入、层 1 出、head 入——5 个 hook 点足够画完整曲线。

---

## 3. 问题分析（Problem Analysis）

### 3.1 complement_gain 为负的根因还没定位

v4 朴素版 concat acc=0.211 < rgb 单模 0.819 ——**融合反而比单模态更差**。PerModConcatMLP 修到了 0.450，但仍 < rgb。这说明：

- 网络确实学到了 mmwave 互补（contribution 0.625），但**总融合收益被其它模态拖累了**
- 弱模态（wifi/depth/lidar）究竟在浅层/深层贡献了什么，目前无法观测

### 3.2 三种可能机制 → 三种不同曲线形态

| 可能 | 对应曲线 | 数据集质量诊断 |
|---|---|---|
| **A. 浅层特异、深层融合**（理想） | CKA 从浅层 0.1-0.3 单调上升到深层 0.7-0.95 | 数据集**信息互补 + 语义一致**——质量好 |
| **B. 浅层就高 CKA** | 全程 CKA > 0.7 | **模态冗余**——多个传感器给重复信息（rgb+depth 退化） |
| **C. 深层升不上去** | 全程 CKA ≤ 0.3-0.4 | **时间/语义错位**（致命）——多模态根本没对齐 |

v4 的真实曲线是哪种，决定了 M6b 训练手段实验（数据对齐 vs 模态 dropout）的方向。

### 3.3 mmwave vs rgb 是最值得聚焦的一对

- rgb 单 probe 0.819（独立判别最强）+ 融合 contribution 0.426（中等）
- mmwave 单 probe 0.348（中等）+ 融合 contribution 0.625（最强）
- 这两个的 **互补结构** 是 v4 数据集的核心信号——诊断清楚这一对，wifi/depth/lidar 的反常自然可推

### 3.4 风险点（避坑）

- **不能用未收敛模型**：M6 token_fusion 在 v4 上的 val acc 已知 0.450+（PerModConcatMLP 视角），模型本身有效
- **不能 MSE 原始特征**：必须 Linear CKA（缩放+正交不变性）
- **不能只看 seed0**：3 seeds 平均 + std，画置信带

---

## 4. 路径取舍与选择（Trade-offs）

### 决策 1：Linear CKA vs Hilbert CKA

| 选项 | 速度 | 准确度 | 解释性 | 选择 |
|---|---|---|---|---|
| Linear CKA | O(n²d)，5000 样本 < 1s/对 | 够用 | 强 | **[已定]** 主路径 |
| Hilbert CKA | 核运算，10-50× 慢 | 略高 | 弱（核空间） | 备选——若 Linear 曲线太平滑可补 |

### 决策 2：hook 点选择

`token_fusion` 结构：per-modality encoder → concat 16×D tokens → 2 层 transformer → mean pool → head

- 选 5 个 hook 点：`enc_out`（per-mod encoder 出） + `layer0_in/out` + `layer1_in/out` + `pool_out` + `head_in`
- 简化（如果 v4 temporal 显存紧张）：3 个 hook 点 = `enc_out` / `layer1_out` / `pool_out`
- **[已定]** 先做 3 点验证曲线形态，曲线有意义再补 5 点

### 决策 3：模态对组合

4 模态全两两 = C(4,2) = 6 对 + rgb×4 = 4 对 = 10 对 × 3 seeds × 3 hook 点 = 90 个 CKA 值

- **[已定]** 全部计算，但**主报告只画**：`mmwave×rgb`、`lidar×rgb`、`wifi×rgb`、`mmwave×lidar` 4 对（最有信息量）
- 其它 6 对进附录 / JSON

### 决策 4：是否引入线性探针

| 选项 | 价值 | 成本 | 选择 |
|---|---|---|---|
| 仅 CKA | 够回答"特异/融合" | — | **[已定]** MVP 范围 |
| 加 per-layer linear probe | 直接看每层分类能力 | +30 分钟训练 | 备选 |

声明先止于 CKA，linear probe 作为"如时间允许"的迭代项。

### 决策 5：报告形式

`docs/reports/layer_cka_v4.md` 对照 `dataset_quality_v1_v2_v4.md` 的格式：
- 一张曲线图（4 对 × 3 层）+ 置信带
- 表格：每对模态在每层的 CKA 值
- 三段结论：
  1. v4 属于哪种曲线形态（A/B/C）
  2. mmwave vs rgb 的具体机制
  3. 给 M6b 训练手段实验的建议（**dropout 比例？层归一化？模态对齐损失？**）

---

## 5. 范围（Out of Scope）

- ❌ 不改 token_fusion 模型结构
- ❌ 不重训 token_fusion（仅用现有 checkpoint）
- ❌ 不引入 v5/v6 数据集（聚焦 v4 一次闭环）
- ❌ 不画 Hilbert CKA 对照（除非 Linear 失效）
- ❌ 不引入跨数据集验证（MMFi 单数据集足够回答"质量"问题）

---

## 6. AI 提议的最重要问题

> **v4 的 mmwave 之所以是"真正的互补"，是因为它在 transformer 浅层保持模态特异性、深层才与 rgb 对齐；还是它从未对齐、只是被 attention mask 保护了起来？**

理由：
- 这是 v4 数据集"complement_gain 由负到正"机制的核心
- 一旦回答清楚，wifi/depth/lidar 的反常行为可类推
- 直接对应用户原话"浅层是否特异 / 深层是否融合"
- 答案会驱动 M6b 训练手段实验的设计（**如果浅层特异深层融合** → 数据本身没问题，可加对齐损失强化；**如果从未对齐** → 数据时间戳错位是根因，需重新 ingest）

---

## 7. 验收标准

- [ ] `framework/eval/dataset_quality/layer_cka.py` 实现 Linear CKA + 多 hook 提取
- [ ] 在 v4 val split 上跑完 3 seeds × 3 hook 点 × 10 模态对
- [ ] 曲线图明确归类（A / B / C）
- [ ] mmwave vs rgb 浅层/深层 CKA 数值有明确结论
- [ ] 报告与 `dataset_quality_v1_v2_v4.md` 交叉验证一致
- [ ] 给 M6b 的下一步建议具体到训练手段（dropout / norm / 对齐损失）

---

## 8. 时间预算

| 步骤 | 估时 | 说明 |
|---|---|---|
| 实现 layer_cka.py | 1-2 h | 含 Linear CKA + hook + 批量提取 |
| 跑 v4 + 3 seeds | 30-60 min | 在 GPU 节点，5000 样本 × 3 seeds |
| 出图 + JSON | 15 min | matplotlib 已有 |
| 写报告 | 1 h | 对照 dataset_quality_v1_v2_v4.md 格式 |
| **总计** | **~3-4 h 编码 + ~1 h 集群跑批** | 一天可闭环 |

---

## 9. 决策层记录

`[提议]` 是否同意按以上声明进入实施？

- **默认参数**：3 hook 点、Linear CKA、10 模态对全跑、仅 CKA 不带 linear probe
- **可调整项**：hook 点数（3↔5）、是否画 Hilbert 对照、是否加 linear probe
- **不需要确认的**：模型 checkpoint、数据集版本、报告存放路径（已有约定）