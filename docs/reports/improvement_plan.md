# SensorBench 改进建议（数据构成 / 增强 / 主题架构）

> 结合 omni-model 调研（`docs/reports/omni_model_comparison.md`）与本项目现状（v5_structfeat 主流程 + M6 对齐/CanonicalToken 管线）整理。
> 生成日期：2026-08-23。所有条目为 `[提议]`，需人拍板后转 `[已定]`。

## 0. 现状基线（事实）

- **数据**：MMFi 室内人体活动，27 类，5 模态（wifi/depth/lidar/mmwave/rgb 关键点）。v1→v6 演进：清洗→关键点归一化+空间增强(v4)→结构化特征(v5_structfeat)→标签修正(v6)。v5 train 46509（含增强变体）。
- **增强**：仅 `augment_keypoints`（flip/translate/scale），只作用于 rgb 关键点模态；v5_hardaug 对 hard-class 加量（失败，无新信息维度）。
- **主题架构**：token_fusion（每模态16 token→共享Transformer）+ late_fusion 基线 + M6 的 alignment(InfoNCE→文本)/perceiver/router/LLM adapter 的 CanonicalToken 可移植管线。
- **性能**：v5_structfeat token_fusion robustness **0.6858** / acc_full **0.9184**。
- **鲁棒性短板**（v5 leaderboard）：miss-rgb **0.765**、miss-mmwave **0.824**、miss-lidar **0.88**，而 miss-wifi/depth **0.92**。→ rgb 与 mmwave 是鲁棒性瓶颈。
- **弱模态独立判别力**（probe 多口径共振）：wifi/depth/lidar 单模 ≈ 随机 0.05-0.11；mmwave 已靠 geom_v2 特征提到 0.71-0.77。

## 1. 数据构成改进

### 1.1 真实缺模态样本（最高优先，直击 miss-rgb/mmwave 短板）
- **现状**：缺模态靠训练期 dropout 模拟（`_dropout_mask`），模型只见过"随机丢单模态"，未见"某传感器整段缺失"。
- **建议**：在数据层生成**真实缺模态样本**——对部分 train 样本物理去掉 mmwave（或 rgb），让模型学到"无 mmwave 时靠 wifi+depth"的强先验，而非仅靠 MISSING token 泛化。
- **预期**：直接针对 miss-mmwave 0.824、miss-rgb 0.765 两个最大降幅缺口。
- **成本**：低，可复用现有 make_* 管线 + delta 存储（只存缺失标记，不复制数据）。

### 1.2 组合缺失增强
- **现状**：训练 dropout 是单模态随机丢。
- **建议**：训练时模拟**组合缺失**（如同时缺 rgb+mmwave），与评测协议的双缺失 profile 对齐，让模型在评测的 miss2-* 场景下更稳。

### 1.3 模态感知 token 预算（OmniScope 洞见）
- **现状**：所有模态固定 16 token 均衡预算（`N_TOK=16`）。
- **建议**：引入**模态感知 token 预算**（`TokenRouter` 已有雏形，`framework/models/router.py`）——高信息模态（rgb）在缺模态时获得更多 token，低信息模态（wifi/depth）少给。OmniScope 证明"query 共享但各模态 salience 独立"优于均衡。

## 2. 增强改进

### 2.1 多模态增强（直击弱模态）
- **现状**：只增强 rgb 关键点。
- **建议**：
  - **mmwave/lidar 点云增强**：旋转、抖动、点 dropout（mmwave 是当前第二瓶颈，0.824）。
  - **wifi 时域增强**：时间缩放、帧扰动。
  - **depth 空间增强**：与 rgb 同步的几何变换（当前 rgb 增强后 depth 未同步，跨模态几何不一致）。
- **预期**：提升弱模态独立判别力 + 缺模态鲁棒性。

### 2.2 跨模态一致增强
- **现状**：v4 只对 rgb 做 flip/translate/scale，depth/lidar/mmwave 未同步变换 → 增强后跨模态几何不一致。
- **建议**：对共享几何的模态（rgb 关键点 + depth + lidar + mmwave 点云）做**同步空间变换**，保持跨模态一致性。

## 3. 主题架构改进

### 3.1 query 条件融合（OmniPack 洞见）
- **现状**：token_fusion 无 query 引导，纯分类。
- **建议**：引入文本/意图作为 query 锚点，让融合聚焦任务相关 token，向 OmniPack 的 in-LLM 精修靠拢。可复用 M6 已有的文本侧（caption/verb 锚）。

### 3.2 分层融合
- **现状**：单层共享 Transformer。
- **建议**：借鉴 OmniPack 的"pre-LLM 结构压缩 + in-LLM 语义精修"，在 token_fusion 中增加一层 query 条件精修，而非单层共享 Transformer。

### 3.3 缺模态建模显式化（本项目差异点，强化）
- **现状**：MISSING token + attention mask + dropout 已实现，但未系统化。
- **建议**：将缺模态建模作为**一等公民**文档化 + 评测，这是 omni 文献未系统化的部分，正是本项目基准的独特价值。可考虑缺模态专用 loss 或显式"模态可用性"条件输入。

## 4. 优先级排序

| 优先级 | 条目 | 预期收益 | 成本 |
|--------|------|----------|------|
| P0 | 1.1 真实缺模态样本 | 直击 miss-rgb/mmwave 短板 | 低 |
| P0 | 2.1 多模态增强（mmwave 点云） | 提升弱模态 + 鲁棒性 | 中 |
| P1 | 1.2 组合缺失增强 | 对齐 miss2-* 评测 | 低 |
| P1 | 2.2 跨模态一致增强 | 增强一致性 | 中 |
| P1 | 1.3 模态感知 token 预算 | 缺模态时高信息模态更多 token | 中 |
| P2 | 3.1/3.2 query 条件 + 分层融合 | 架构升级 | 高 |
| P2 | 3.3 缺模态建模显式化 | 差异化价值 | 低 |

## 5. 验证方式
- 所有数据改进在 **gold_subset_v2**（85 样本，三方共识）上复测归因（数据 vs 架构）。
- 主流程用 leaderboard_v5 协议（5 模态 21 profiles × 3 seeds）对比 robustness / acc_full / per-profile degradation。
- 弱模态用 dataset_quality probe（per-modality acc + concat contribution）交叉验证。

## 6. P0-1.1 机制验证结果（2026-08-23，负结果）

**假设**：提高瓶颈模态（mmwave/rgb）的缺失暴露率能提升缺模态鲁棒性。

**方法**：实现按模态偏置 dropout（`TrainConfig.modality_dropout` + token_fusion `_dropout_mask` + train.py `--modality-dropout`），mmwave/rgb 缺失率 0.5，token_fusion × 3 seeds，v5_structfeat。

**结果**（vs baseline 0.6858/0.9184）：

| profile | MD | baseline | Δ |
|---|---|---|---|
| full | 0.9032 | 0.9184 | **-0.015** |
| miss-mmwave | 0.7950 | 0.8237 | **-0.029** |
| miss-rgb | 0.7888 | 0.7650 | **+0.024** |
| miss-lidar | 0.8629 | 0.8800 | -0.017 |
| miss-wifi | 0.9095 | 0.9169 | -0.007 |
| miss-depth | 0.9043 | 0.9151 | -0.011 |

**结论（负结果）**：
- 偏置 dropout 只对 miss-rgb 有 +0.024 提升，但 miss-mmwave 反而降 -0.029，full 降 -0.015，整体 robustness 0.6858 → ~0.66（净负）。
- **根因**：提高缺失暴露率以牺牲 full 精度为代价（每个 batch 里 mmwave/rgb 有 50% 概率被丢，full 训练样本减少）。
- **关键洞察**：token_fusion 的 MISSING token 机制已能处理缺失，偏置 dropout 与"真实缺模态样本"在机制上等价（都是 avail=False → MISSING token）。因此 **P0-1.1 真实缺模态数据管线收益有限，不投入构建 v7**。
- **代码保留**：`modality_dropout` 配置已实现 + 测试（`test_token_fusion_per_modality_dropout`），作为可调超参保留，但默认不启用。

**下一步建议**：转向 P0-2.1 多模态增强（mmwave 点云增强），或接受"缺模态鲁棒性已接近架构上限"的现状，聚焦弱模态独立判别力（wifi/depth/lidar）。

## 7. 预训练 encoder 验证结果（2026-08-23，有限收益）

**假设**（omni 模型普遍接入成熟预训练 encoder）：用预训练 encoder 替换从零训练的小 CNN，能提升弱模态（depth）判别力。

**方法**：resnet50（ImageNet 预训练，冻结，avgpool 2048-d）在 v4 原始 depth 图像 (224,224) 上提取特征 → probe（MLP 256 + linear），对比当前 v5_structfeat 63d 结构化特征。

**结果**（depth 单模 probe val_acc）：

| 特征 | probe acc |
|---|---|
| v5_structfeat 63d 结构化特征（当前） | **0.305** |
| resnet50 冻结特征（MLP probe） | 0.308 |
| resnet50 冻结特征（linear probe） | 0.337 |

**结论（有限收益）**：
- 预训练 encoder 相对当前结构化特征提升仅 **+0.003~+0.032**，几乎打平。
- **根因**：depth 单模判别力瓶颈在**数据本身**（depth 单模信息量有限，probe 多口径共振 0.07-0.31），不在 encoder 架构。预训练先验无法凭空补足数据缺失的信息。
- **决策**：**不值得为接入预训练 encoder 改变主流程数据管线**（v5_structfeat 已把 depth 换成 63d 结构化特征，数据 18GB→272MB）。预训练 encoder 的收益不足以抵消数据管线复杂度。
- **与 omni 模型的差异澄清**：omni 模型接预训练 encoder（CLIP/CLAP）是因为其输入是自然图像/音频，预训练先验与任务域匹配；本项目 depth 是深度图、rgb 是 17 关节关键点、wifi 是 CSI，**与预训练 encoder 的预训练域不匹配**，先验迁移收益有限。
- **脚本保留**：`scripts/probe_depth_pretrained.py`（resnet50 冻结特征 depth probe），可复用于其它模态/encoder 验证。

**下一步建议**：预训练 encoder 路线收益有限，不投入。聚焦 P0-2.1 多模态增强（mmwave 点云增强）或弱模态数据本身。
