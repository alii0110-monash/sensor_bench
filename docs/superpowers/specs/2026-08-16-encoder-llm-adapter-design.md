# 多模态编码器 → LLM 动态适配层设计（SensorBench 迭代 5）

- 日期: 2026-08-16
- 状态: 设计定稿
- 前置: v4 数据管线（关键点规范化 + 增强 + 去重）已完成，v4 训练/评测证明数据质量飞轮有效（token_fusion robustness 0.3167→0.4961）

## 背景与目标

最终目标：构建**多模态对齐数据集 + 好编码器**，输出可**动态适配任意 LLM** 的伪文本 token，供下游 LLM 理解使用。

现状差距（v4 体系是分类基准，不是 LLM 接口）：
1. 对齐是隐式的（token_fusion 训练分类时顺带对齐），无显式对齐评测。
2. 编码器为 27 类分类设计，输出 token 被 mean 池化，与 LLM embedding 空间无可比性。
3. 无文本侧，无"伪文本 token"注入能力。
4. robustness 指标是间接信号，无法直接度量编码器对齐质量。

## 已确认的关键决策

| 决策点 | 选择 |
|---|---|
| LLM 输入形态 | **伪文本 token 走 LLM 词表**（embedding 前缀注入，非直接 embedding 序列） |
| LLM 适配范围 | **适配任意 LLM**（两段式，per-LLM 投影层） |
| 对齐信号 | **合成文本标注 + CLIP 式对比**（InfoNCE） |
| 文本侧锚 | **固定冻结的通用文本编码器**（第一阶段锚），per-LLM 蒸馏用目标 LLM 自己的 text encoder |
| 合成文本生成 | **离线批量生成**（调 LLM），存回 `Sample.text` |
| Token 动态化 | **半动态**：训练固定 k，推理按可用性/预算截取前 k' |

## 架构（两段式，6 个独立单元）

```
第一阶段（一次性）：数据集 + 编码器
  MMFi 5模态 ─→ 模态编码器 ─→ 规范token (K_max=16/模态, 256维)
                      ↕ InfoNCE（冻结通用文本编码器做锚）
  合成文本 ─→ 通用文本编码器 ─→ 文本embedding

第二阶段（per-LLM）：轻量可插拔
  规范token ─→ Perceiver投影 ─→ 伪token (k, LLM_hidden)
                    ↕ 蒸馏监督（同一批合成文本, 目标LLM text encoder）
  推理: router 按(可用性, 预算)截取前 k_m ─→ [BOS][伪token×k'][文本] → 冻结LLM
```

| 单元 | 职责 | 训练 | 依赖 |
|---|---|---|---|
| 1. 模态编码器 | 原始传感器 → 每模态 K_max token | 第一阶段对比 | 数据管线 |
| 2. 合成文本生成器 | 样本 → 多句自然描述 | 离线脚本 | 标注数据 |
| 3. 通用文本编码器 | 文本 → 锚定 embedding | 冻结 | 合成文本 |
| 4. 规范空间对齐层 | 对比 loss 训练编码器 | 第一阶段 | 2+3 |
| 5. Perceiver 投影 | 规范token → 目标 LLM 空间 | per-LLM 蒸馏 | 1(冻结) |
| 6. Token router | 推理时动态截取 k_m | 启发式（可后续可训） | 5 |

接口约定：
- 单元 1 输出：`(B, M, K_max, D)` 张量
- 单元 5 输出：`(B, Σk_m, LLM_hidden)` 注入序列
- 单元 2 输出：`Sample.text = {"en": List[str]}`（多句描述）

## 数据管线与合成文本标注

- 输入：MMFi v4（52686 样本，5 模态；train base 9689 + 变体 36820、val 1968、test 4791；实际落盘可加载 train 46025 / val 1870，因 base 缺文件 train 484 + val 98）。
- 合成文本：每训练 **base 样本（9689 个）**生成 **3-5 句**自然描述，存入 `Sample.text`。
  - 变体（`__aug*`）共享 base 的 text（空间变换不改变语义），节省 ~4/5 生成成本 → 总生成调用 ≈ 9689 × 3-5 次。
  - 描述必须含**动作语义**（27 类 label 的动作动词）+ **可选情境**（subject/env 元数据）。
  - val/test 后补，不在第一阶段必做范围。
- `SyntheticCaptioner` 抽象接口：`generate(sample) -> List[str]`，底层可换 API/本地模型；测试用固定模板 mock。
- 版本化：落盘方案**二选一定稿为「写入 v5 的 Sample.text 字段」**（经 loader `_resolve_variant` 变体自动继承 base text，无需 loader 外置合并）。可复现，不重复调用。
- 质量检查：长度、去重、是否含动作词 + 人工抽检。

## 模态编码器 + 规范空间对齐

- 编码器改造澄清（避免误删）：现有 `encoders.py` 每编码器内 `(B,T,N_TOK,D).mean(dim=1)` 是对**时间帧 T** 求均值，已输出 `(B, N_TOK, D)` token 序列——**保留**。要改的是 `token_fusion.py` 的 `head(x.mean(dim=1))` 跨 token 池化——**去掉**，让编码器输出保留 token 序列 `(B, N_TOK, D)` 供投影层消费。
- 缺模态 = 0 个 token（自然支持半动态，投影层不产生输出）。
- 对齐 loss（InfoNCE）：传感器侧池化成一个向量参与对比（**首选 CLS，实现时锁定**；均值作备选），编码器输出仍是 token 序列，池化只在 loss 内部。
- 维度对齐：传感器侧 D=256 与通用文本编码器维度（如 CLIP 512）不同，InfoNCE 前需一个轻量**投影头（D→文本维度）**，归入单元 4（规范空间对齐层）。
- 文本编码器冻结（CLIP 或类似，维度固定）。
- 第一阶段对比训练**沿用 v4 的 modality-dropout**（缺模态训练），保住本项目 robustness 核心卖点。
- 验收：跨模态检索 recall@k（传感器→文本 及 反向）。

## Perceiver 投影 + 伪文本 token 注入

- Perceiver 投影：**per-modality 共享权重 latent**——每模态 K_max=16 个规范 token 各自过一个共享权重的 Perceiver，输出 k_m 个伪 token；多模态拼接后总注入数 = Σk_m（即接口约定 `(B, Σk_m, LLM_hidden)`）。
  - k_m 默认 4-8，**按每模态计算**（非全局总数）；K_max=16 是 per-modality 的上限，Perceiver 把 16 个压缩到 k_m 个。
  - 半动态核心：训练固定 k（每模态 8），推理 router 在每模态内截取前 k'（query 按重要性排序）。
  - 缺模态 = 该模态 0 个 token（投影层对缺失模态不产生输出）。
- 蒸馏监督：同一批合成文本用目标 LLM 自己的 text encoder 编码，InfoNCE 对齐伪 token 池化表征。
- 注入：伪 token 作为前缀序列 `[BOS][伪token×k'][用户指令/文本]`，目标 LLM **冻结**，只训练投影层。
- `LLMAdapter` 抽象：`project(tokens) -> (k, LLM_hidden)` + `inject(prefix_embs)`，每个目标 LLM 一份实现。

## Token Router（半动态）

- 训练固定 k，推理截取前 k'（query 按重要性排序，截取是合理近似）。
- 起步用**确定性启发式**：`k_m = min(k_max, budget_remaining)`，缺模态自然为 0。截取机制 = **plain 前缀截断**（取每模态前 k' 个 query），重要度排序留作 router 增强；用测试锁定截断稳定性（不同预算下 `predict_batch` 一致）。
- 极端预算回退：k'=0 时只喂文本描述（合成文本已存），任何预算下都可用。
- 可训 router（Gumbel）为后续增强，当前不做（YAGNI）。

## 评测体系（精简）

- **L1（必做）**：跨模态检索 recall@k（编码器对齐质量），复用现有 harness 批处理。**评测数据源**：从已标注的 9689 个 train base 中划出 held-out 子集（如 10%，~970 个）作为 L1 检索评测集——**该子集的 base 及其 `__aug*` 变体全部排除出对比训练**（变体共享 base 文本，否则近重复样本泄漏进训练集、虚高 recall）；val/test 的文本标注不阻塞 M5a（L1 用 train base held-out 即可，val/test 标注随后续里程碑补全）。
- **L2（冒烟）**：投影层能拼进目标 LLM 前缀、一次前向、维度正确、冻结 LLM 文本能力不回归（2-3 个文本 QA 样例）。
- **L3（推后）**：少样本动作理解、事件问答等端到端 LLM 能力评测，移入后续里程碑。

## 测试策略

- 单元测试（无 GPU，mock 重依赖）：
  - `test_captioner.py`：模板 mock 生成、text 写入、变体共享、质量检查
  - `test_alignment_loss.py`：InfoNCE 数值正确性、负对构造
  - `test_perceiver.py`：shape、变长输入、缺模态=0 token
  - `test_router.py`：预算收缩、缺模态、极端回退
  - `test_llm_adapter.py`：前缀注入 shape/拼接（mock LLM）
- 集成测试（真数据子集）：
  - `test_alignment_e2e.py`：mini 数据集跑第一阶段，recall@k 高于随机
  - `test_projection_smoke.py`：真实 LLM 加载 + 注入 + 前向 + 文本回归
- 验收：全测试绿（含既有 72 个不回退）；L1 recall@k 显著 > 随机；投影冒烟通过。

## 里程碑划分

- M5a（本迭代）：数据管线（合成文本）+ 编码器改造 + 规范空间对齐 + L1 评测
- M5b：Perceiver 投影 + LLMAdapter + router + L2 冒烟
- M5c（推后）：L3 端到端 LLM 能力评测

## M5a-M5c 执行结果与负结论（2026-08-16 归档）

- **M5a ✅**：v5 数据集（官方 LLM captions `text.captions` 自 v1 起全量存在，非 M5a 生成）、AlignmentModel + InfoNCE、L1 评测。
- **M5b ✅**：Perceiver 投影 + TokenRouter + LlamaAdapter + L2 注入冒烟 PASS。
- **M5c ⚠️ 负结果**：真训练 + L3 评测完成，但 **冻结 llama2 无法读懂伪 token**（L3 acc_pseudo=0）。
  - 关键 bug：CLIPTextEncoder 曾用 BOS 位置导致 27 类 sim=1.0，改 pooler_output 后可分（0.316）。
  - 原型初始化投影头让 InfoNCE 从卡死转收敛，L1 r@1=0.0066（弱）。
  - 伪 token 最近邻从情境词(arms/room) → verb 锚后动作词域(jump/throwing/lung)，但冻结 LLM 仍不可读。
  - **结论：生成式 LLM 理解伪 token 需 LoRA 微调（Qwen-VL/LLaVA 式），纯冻结不可行。** 该路线当前不实施（YAGNI）。

## 开放问题

- 通用文本编码器的具体选择（CLIP ViT-B-32 / Sentence-BERT / 其他），需在实现时对比。—— **已实测：CLIP pooler_output 对 27 类动作短语可分性 0.316，可用但弱；llama2 文本 embedding 可分性 0.45 更强。**
- 合成文本生成的 LLM 后端（本地可用模型 / API），需在实现时确认可用资源。—— **已确认：官方 captions 自 v1 起存在，无需生成。**
- 目标 LLM 的具体清单（决定 LLMAdapter 实现个数），建议先定 1 个参考 LLM。—— **已用 llama2-7b（hidden=4096）。**
- **新增：若未来要让 LLM 生成式理解伪 token，需 LoRA 微调 LLM；或将评测改为隐式（跨模态检索/动作分类，不依赖生成）。**
