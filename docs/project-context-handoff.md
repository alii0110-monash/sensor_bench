# SensorBench 项目完整背景交接文档

> 用途：供另一个 AI 模型独立分析项目当前困境。本文档包含从项目起源到最新实验的全部前因后果、数据、结论与开放问题。全部事实可回溯到 git 历史（117 个 commit）与 `docs/` 下各 spec/report。

---

## 1. 项目是什么

**SensorBench**：一个"数据/模型解耦的跨模态融合基准框架"，核心目标是用**缺模态鲁棒性（Robustness Score）量化多模态数据集质量**。

- **数据**：MMFi 数据集（动作识别，27 类），跨主体（cross-subject）划分，5 个模态：
  - `wifi`、`depth`、`lidar`、`mmwave`（4 个射频/视觉传感器，原始数据）
  - `rgb`（v3 加入，人体关键点 17×2/frame，用 ResNet-48 从视频提取）
- **主流程**：protocol.json 定义 15 个"缺模态 profile"（全模态/单缺/双缺/单模）→ 训练 token_fusion / late_fusion → 评测每个 profile 的 acc → Robustness Score = 所有 profile acc 均值 + acc_full。
- **数据版本**：v1 → v2 → v3 → v4 → v5，每个版本是数据改进实验，leaderboard 记录结果。

## 2. 数据版本演进（前因后果链）

| 版本 | 改动 | 效果 | 结论 |
|---|---|---|---|
| **v1** | 初始 4 模态 | token_fusion robustness 0.1462 | 基线 |
| **v2** | 用 token_fusion 标记"跨模态不一致"样本，drop 5% train/val | robustness 持平或略降（late_fusion 0.2288→0.2138） | **"删样本"方向错误**：过滤器与评测模型相关性太强，且删的样本太少 |
| **v3** | 注入 `rgb` 人体关键点模态（17,2/frame），不删样本 | token_fusion robustness 0.1425→**0.3167**（2.2x）、acc_full 0.2382→0.5759 | **"加信息"方向正确**。核心假设判明：弱模态缺独立判别力，不是任务太难 |
| **v4** | rgb 规范化（髋部居中+躯干长，全 split）+ train 离线增强（翻转/平移/缩放，n_aug=4，46509 样本） | token_fusion acc_full **0.7632**、robustness **0.4961**；late_fusion 0.7084/0.3823 | **规范化+增强显著有效**（v3 基础上 +32%/+57%） |
| **v5** | 合成文本 caption（TemplateCaptioner），9205 个 train base | 用于对齐训练，不改变数据本身 | 见 §5 |

**v4 数据质量诊断（主流程视角）**：
- 有效信息支柱只有 rgb + mmwave：miss-rgb 掉 0.402、miss-mmwave 掉 0.108
- miss-wifi/depth/lidar 几乎不掉（<0.01）——**不是因为弱模态好，而是强模态掩盖**
- 单模态独立判别力：only-mmwave 0.296（唯一有效非视觉模态）、only-wifi 0.032 / only-depth 0.036 / only-lidar 0.050（≈随机 1/27=0.037）
- 丢 rgb+mmwave（miss2）直接崩到 0.037 → **数据实质是"rgb+mmwave 双模态"**

## 3. 架构与模型

- **token_fusion**：每模态 token 序列（16×256）→ 注意力融合。对齐机制占优（robustness 更高）。
- **late_fusion**：各模态独立编码 → 拼接 → 分类。稳定性好（std 低）。
- **AlignmentModel**：两段式——
  - 第一段（M5a）：InfoNCE 对齐传感器编码 → 文本锚（CLIP 512 或 hash mock）
  - 第二段（M5b）：PerceiverProjection 投影 → 目标 LLM 空间（llama2-7b，4096 维）
  - **CanonicalToken**（M6a）：4096 维规范空间伪 token，资产化落盘（npz + index.json），换 LLM 只换线性投影层。v5tokens 已生成 9205 个 train base（5.3GB）。

## 4. 评测体系（三层）

### 4.1 主流程 robustness（M1-M4，✅ 跑通）
- 测"丢模态代价"：15-21 个 profile × 3 seeds → leaderboard
- **局限**：① acc 是"模型×数据"耦合，非纯数据属性 ② 强模态冗余掩盖弱模态坏数据 ③ 测不出语义可分性
- → 这是引入 L1 检索（M5）和 dataset_quality（M6）的动机

### 4.2 L1 跨模态检索（M5-M6b，❌ 双负结果）
- 目标：测表征可分性。传感器伪 token ↔ 文本 caption 互检索（recall@k）
- **CLIP 512 空间**（M6b 第一次评测）：全部噪声级，baseline r@1=0.0022 ≪ 随机 0.037
- **llama2 4096 空间**（M6b 复测，2026-08-17）：baseline r@1=0.0076（3.5x 于 CLIP），但 A-E 变体差异仍 < 2SE，无显著提升
- **根因**：v5 模板文本（TemplateCaptioner 生成）语义趋同，在 CLIP/llama2 空间都几乎不可检索 → **评测天花板**，测不出编码器质量差异

### 4.3 dataset_quality 轻量 probe（M6c 前置，✅ 刚上线）
- 目的：彻底解耦——用轻量 probe（Linear/MLP）直接测"数据本身的判别力"，不看任何下游模型
- 三维度：InfoScore（每模态独立判别力+互补增益）、CompactScore（类内紧致）、CleanScore（异常+跨模态一致性+重复）
- **P0 护栏**：test split 全程不进入 probe eval
- **v4 结果（MLP probe）**：rgb 0.828 / mmwave 0.326 / depth 0.077 / lidar 0.090 / wifi 0.051；acc_concat 0.211；quality 0.228（v1 0.194 / v2 0.181）
- **独立验证了主流程的弱模态判断**：wifi/depth/lidar 在 v1→v4 始终 ≈ 随机，不是被 rgb 掩盖，而是**数据本身缺独立判别力**

## 5. M5-M6b 对齐质量实验（核心负结果链）

### 5.1 M5a：合成文本 + InfoNCE 对齐（✅ 完成）
- TemplateCaptioner 生成 9205 个 train base 的 caption
- AlignmentModel 用 InfoNCE 把传感器编码对齐到 CLIP 文本锚
- L1 检索评测：r@1=0.0066（有信号但弱）

### 5.2 M5b：Perceiver 投影 + LLM 适配（✅ 完成）
- PerceiverProjection：per-modality 共享权重，缺模态显式置零无 NaN
- TokenRouter：半动态启发式路由
- LlamaAdapter：本地 llama2-7b（bf16，CPU offload）
- L2 冒烟 PASS：prefix=4+text=7→merged=11 前向通过

### 5.3 M5c：真训练 + 端到端 LLM 评测（❌ 负结果，已归档）
- 真训练 M5a（CLIP 锚）+ M5b projection（llama2 蒸馏）+ L3 三模式评测
- 关键发现链：
  - **CLIP 文本锚 bug**：`CLIPTextEncoder` 曾用 `last_hidden_state[:,0]`（BOS）导致 27 类动作 sim=1.0 不可分；修复为 `pooler_output` 后可分性 0.316
  - 原型初始化投影头使 InfoNCE 从卡死（3.4649）转为收敛（3.28）
  - 伪 token 最近邻诊断：caption 锚下落情境词（arms/room）；改 verb 锚后落动作词域（jump/throwing/lung），但仍不精确
  - **核心负结论**：伪 token embedding 即使与 llama2 词表动作词相近，**冻结 llama2 仍读不懂**（L3 acc_pseudo=0）——冻结 LLM 从未被训练过"使用"伪 token 前缀。生成式理解需要 LoRA 微调，当前不做。

### 5.4 M6b：训练手段提升对齐质量（❌ 负结果×2）
- 3 手段：大 batch（64/128/256）/ 分类辅助 CE / label-aware 负样本挖掘
- 5 变体矩阵 A-E（batch32 复现 / batch64 / +CE0.5 / +neg-mine / 全组合）
- 训练结果：neg-mine 使 InfoNCE loss 降 33%（3.8→2.5）、CE 辅助让 r@5/10 略升，但 **r@1 完全没跟上**
- **CLIP 512 评测**：r@1 A-E ≈ 0.003-0.004，全部噪声级
- **llama2 4096 复测**：baseline 0.0076 → A 0.0109（1SE），变体差异仍不达 2SE
- **核心负结论**：① 优化了训练目标，没优化评测指标 ② 评测天花板（模板 caption 在 CLIP/llama2 都不可检索）③ CE 在双评测空间同步最差（过拟合 27-way，拖累文本对齐）

## 6. 最新工作：probe 升级（2026-08-17，今天）

- 问题：Linear probe 在 v4 上 concat acc=0.050 < rgb 单模态 0.418，CompactScore/CleanScore 失真
- 根因诊断：depth 占 concat 维度 86%（50176/58558）且量纲最大（std=4.0 vs 其他 <3）
- **三管齐下修复**：① per-modality z-score 标准化（train 统计，val 套用）② depth 最大池化 224→28（50176→784）③ Linear → 2 层 MLP（256 隐）
- **效果**：v4 rgb 0.418→**0.828**、acc_concat 0.050→0.211、anomaly_rate 0.93→0.0、quality 0.070→**0.228**
- **遗留死指标**：`inconsistency_rate` 恒 1.0（per-modality 独立 probe 无校准，跨模态 JS 无意义）→ CleanScore=0.333 是地板值，不作对比依据

## 7. 当前困境汇总（供分析）

### 困境 A：数据本身弱模态缺判别力
- wifi/depth/lidar 三个模态在 v1→v4 **始终** ≈ 随机（0.05-0.09），dataset_quality 独立确认
- 数据实质是"rgb + mmwave 双模态"，其余三个形同虚设
- 已提 M6c：补弱模态独立判别力 或 加 infra1/infra2 → 7 模态

### 困境 B：评测天花板（L1 检索）
- 模板 caption（TemplateCaptioner）在 CLIP 512 和 llama2 4096 都不可检索
- 训练手段被评测天花板压制：手段可能有效但评测不敏感
- 换评测空间只"缓解"不"解决"（llama2 4096 baseline 3.5x 但仍噪声级）

### 困境 C：对齐质量提不上去（M6b）
- 3 训练手段（大batch/CE/neg-mine）双评测空间均无显著提升
- CE 反而最差（过拟合 27-way 分类）
- 伪 token 已有（CanonicalToken 4096 可移植），但没人能"用"它

### 困境 D：dataset_quality 评测系统的一只腿还瘸着
- InfoScore/CompactScore 已可用（MLP 修复）
- **CleanScore 的 inconsistency 指标死掉**（per-modality 独立 probe 无校准）
- 已提重设计：drop-modality 预测差异 或 单模态预测熵

## 8. 开放方向（决策层 [提议]，待拍板）

1. **inconsistency 指标重设计**：改为"全模态 concat probe 的 drop-modality 预测差异"
2. **换评测目标**：动作-动作 verb 相似度 / 类内紧致度，绕开"整句 caption 模板化"陷阱
3. **M6c 数据改进**：focus wifi/depth/lidar 独立判别力（dataset_quality 已指明靶点）
4. **LLM caption 落地**：M5b 延后项，模板 caption → LLM 真实 caption，L1 上限应抬升
5. **接入主流程 robustness 交叉验证**：dataset_quality 高分模态 ↔ 主流程 miss 该模态高代价

## 9. 关键资源索引

| 资源 | 路径 |
|---|---|
| 项目状态唯一入口 | `STATUS.md` |
| 主流程 robustness 报告 | `docs/reports/robustness_v1_v2.md` |
| M6b 实验矩阵报告 | `docs/reports/m6b_alignment_matrix.md` |
| dataset_quality 报告 | `docs/reports/dataset_quality_v1_v2_v4.md` |
| M6b llama2 复测数据 | `results/m6b_llm4096_sweep.json` |
| dataset_quality 结果 | `results/quality_v{1,2,4}.json` + `leaderboard_quality.md` |
| 踩坑记录（17 条+） | `docs/LESSONS.md` |
| spec | `docs/superpowers/specs/2026-08-17-dataset-quality-eval-design.md` 等 6 份 |
| plan | `docs/superpowers/plans/2026-08-17-dataset-quality-eval.md` 等 7 份 |
| 代码 | `framework/`（models/ eval/ dataset/ tokens/）+ `scripts/` |

## 10. 关键数字速查

- 数据：5 模态，27 类动作，跨主体划分；v4 训练 46509（9689 原 + 36820 增强变体），val 1870，test 4791
- 主流程最佳（v4）：token_fusion robustness 0.4961 / acc_full 0.7632
- L1 最佳：llama2 4096 A 变体 r@1=0.0109（随机 0.037）
- dataset_quality 最佳（v4 MLP）：rgb 0.828 / mmwave 0.326 / quality 0.228
- 环境：RTX 5060 Ti 16GB，27GB RAM，llama2-7b bf16 CPU offload（GPU 仅 1.5GB）
- 测试：dataset_quality 51 个；全项目 ~135+ 个