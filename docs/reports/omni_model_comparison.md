# Omni Model 调研 vs SensorBench 模型结构对比

> 目的：收集近期 omni-modal / omnimodal 模型的架构信息，与 SensorBench 的融合模型（token_fusion / late_fusion）做结构对照。
> 信息来源：paperhub 库内检索 + 新增下载分析 3 篇（OmniPack / OmniScope / Ex-Omni-2D）。
> 生成日期：2026-08-23

## 1. 本项目的模型结构（SensorBench）

- **协议**：`SensorModel`（`framework/models/base.py`）—— 训练 + 单样本 `predict` + 批量化 `predict_batch`，缺模态行为完全由模型自决。
- **模态**：`["wifi", "depth", "lidar", "mmwave", "rgb"]`，每模态独立 encoder。
- **TokenFusion**（`framework/models/token_fusion.py`）：每模态 encoder → 16 token，共享 Transformer，mean-pool → 分类头。**缺模态用 `[MISSING]` embedding + 在 attention 中 mask**；训练用模态 dropout（`modality_dropout_p`）。→ 这是"token 级融合 + 显式缺模态建模"。
- **LateFusion**（`framework/models/late_fusion.py`）：每模态 encoder → 单向量，**缺模态填零向量**，concat → MLP。无对齐机制，作控制基线。
- **编码器**（`framework/models/encoders.py`）：Wifi/Depth 用 CNN，lidar/mmwave/rgb 用 PointNet 式 MLP，统一输出 `(B,16,D=256)`；MLPEncoder 处理结构化特征。

## 2. Omni 模型调研要点

### 2.1 统一大模型（Omni-LLM）方向
| 论文 | 结构特点 |
|------|----------|
| **Dynin-Omni** (2604.00007) | 首个**掩码扩散**统一基础模型，文本/图像/语音/视频共享离散 token 空间上的掩码扩散，非自回归、双向上下文，消除模态特定生成专家。 |
| **OpenOmni** (2501.04561) | 开源全模态 LLM，两阶段：以文本为枢纽的渐进式多模态对齐 + 轻量端到端语音解码器（+DPO）。7B。 |
| **AudioPaLM** (2306.12925) | 语音-文本统一模型，融合 PaLM-2 与 AudioLM。 |
| **RoboEgo / EgoMem** (2506.01934/2509.11914) | 全双工原生 + 终身记忆代理（人/声识别、事实检索）。 |
| **Ex-Omni-2D** (2608.10720) | 全模态对话 + 原生视觉在场：生成文本+个性化语音+参考条件视频；共享声学-时序接口。 |

### 2.2 Token 压缩 / 融合机制（与本项目最相关）
| 论文 | 核心机制 |
|------|----------|
| **OmniPack** (2608.03812) | 训练-free 两段式：**LLM 前**结构压缩（模态特异重要性+全局覆盖+相似合并）→ **LLM 内** query 条件语义精炼（文本引导 + 音视频协作）。Qwen2.5-Omni-7B 上 FLOPs 降到 16.7% 仍保 98% 性能。 |
| **OmniScope** (2607.23193) | **模态解耦 token 压缩**：用 query 作共享语义锚，但音频/视频**分别估 salience、独立分配 token 预算**。核心洞见：同一 query 音频/视频的 salience 峰值往往不同步，单向跨模态引导会丢关键线索。25% 保留率下仅掉 0.35 点。 |

### 2.3 其它 omni 相关
- **Omni2LoRA** (2608.09227)：LoRA 参数内存做高效 omni LLM 记忆。
- **Deferred Audio Pruning** (2608.08794)：延迟音频剪枝 + 局部音视动态。
- **Orchestra-o1** (2606.13707)：全模态智能体编排。
- 其余多为应用/基准（OmniScientist、OmniReasoner、Listen-See-Track、Ex-Omni-2D 等）。

## 3. 与本项目结构的对比分析

| 维度 | SensorBench (token_fusion) | Omni 统一模型 |
|------|------------------------------|----------------|
| **粒度** | 每模态 16 token（固定小 token 预算，几乎天然"压缩"） | 视频/音频产生大量 token，需显式压缩（OmniPack/OmniScope） |
| **融合位置** | 共享 Transformer（早期/单层融合）+ mean-pool | 进入 LLM 前编码 + LLM 内 query 条件交互（分层融合） |
| **缺模态处理** | `[MISSING]` embedding + attention mask + 训练期 dropout | Omni 模型鲜见显式缺模态设计（默认全模态在线） |
| **模态独立性** | 各模态共享 transformer 处理，mask 控制参与 | OmniScope 强调"query 共享但 salience 各自独立"，避免跨模态错误主导 |
| **query 条件** | 无（纯分类，无 query 引导） | query 作为语义锚点参与 salience 估计与压缩 |

### 对本项目的关键启发
1. **缺模态建模是差异点（优势）**：Omni 模型基本假设模态齐全；本项目显式处理缺失（MISSING token + attention mask + dropout），这是 Omni 文献中较少系统化处理的部分，正是本项目"缺模态鲁棒性"基准的独特价值。
2. **salience 解耦观点与 token 预算呼应**：OmniScope 的"各模态独立 salience"与 SensorBench 天然把每模态限制为固定 16 token 的"均衡预算"一致——不必让任一模态主导融合。
3. **分层融合（pre-LLM + in-LLM）值得借鉴**：可考虑在 token_fusion 中增加"query 条件"路径（若未来有语义/意图输入），向 OmniPack 的 in-LLM 精修靠拢。
4. **压缩经验可迁移**：若未来引入高 token 密度模态（如高清 depth/rgb 视频流），OmniPack/OmniScope 的"先结构压缩、后语义精修"可直接套用。

## 4. 待补充 / 下一步
- 可下载分析 Qwen2.5-Omni / MiniGPT-Omni 等代表性开源 omni 骨架，做更深入的逐模块对照。
- Omni 侧缺少"缺模态鲁棒性"的量化评测，是两项目潜在结合点。
