---
project: SensorBench
goal: 数据/模型解耦的跨模态融合基准框架，以缺模态鲁棒性（Robustness Score）量化数据质量
milestones:
  - id: M1
    name: 项目骨架 + 规范格式 + MMFi ingest 管线
    evidence: [framework/dataset/sample.py, datasets/mmfi/v1/meta.json, datasets/mmfi/v2/meta.json]
  - id: M2
    name: SensorModel 协议 + token_fusion/late_fusion
    evidence: [framework/models/base.py, framework/models/token_fusion.py, framework/models/late_fusion.py]
  - id: M3
    name: 评测 harness + 排行榜
    evidence: [protocol.json, framework/harness/evaluate.py, leaderboard_v1.json]
  - id: M4
    name: v2 数据改进闭环（清洗 v2 数据 → 重训 → v2 评测对比）
    evidence: [datasets/mmfi/v2/meta.json, checkpoints_v2/late_fusion_seed2.pt, leaderboard_v2.json]
  - id: M5
    name: 文档与可复现性
    evidence: [README.md, docs/reports/robustness_v1_v2.md]
  - id: M6
    name: M6a CanonicalToken 可移植性 + M6b 训练手段实验 + 数据集质量评测系统
    evidence: [datasets/mmfi/v5tokens/, results/quality_v4.json, framework/eval/dataset_quality/]
artifacts:
  - name: v1 leaderboard
    path: leaderboard_v1.json
    expect: nonempty
  - name: v2 leaderboard
    path: leaderboard_v2.json
    expect: nonempty
  - name: v2 eval 日志
    path: logs/eval_v2.log
    expect: nonempty
  - name: 评测协议
    path: protocol.json
    expect: nonempty
  - name: 数据集质量 v1
    path: results/quality_v1.json
    expect: nonempty
  - name: 数据集质量 v2
    path: results/quality_v2.json
    expect: nonempty
  - name: 数据集质量 v4
    path: results/quality_v4.json
    expect: nonempty
  - name: 数据集质量 leaderboard
    path: leaderboard_quality.md
    expect: nonempty
protocol_fingerprint: protocol.json
anomaly_scan:
  - pattern: "_pending_"
    path: docs/reports/robustness_v1_v2.md
log_dirs: [logs]
---

# STATUS — SensorBench

> 项目状态唯一入口。事实层由脚本生成（勿手写）；判断层由 AI 会话维护；决策层由人拍板（AI 提议用 `[提议]` 前缀，人确认后改 `[已定]`）。

<!-- FACTS:START -->
## ⚙ 事实层（脚本生成 — `project-status scan` 维护, 勿手写）

- 生成时间: 2026-08-25 13:56
- 项目: **SensorBench** — 数据/模型解耦的跨模态融合基准框架，以缺模态鲁棒性（Robustness Score）量化数据质量
- 协议指纹: `7cdc8c3a5636`

### 里程碑
| id | 状态 | 名称 | 证据缺口 |
|----|------|------|----------|
| M1 | ✅ DONE | 项目骨架 + 规范格式 + MMFi ingest 管线 | — |
| M2 | ✅ DONE | SensorModel 协议 + token_fusion/late_fusion | — |
| M3 | ✅ DONE | 评测 harness + 排行榜 | — |
| M4 | ⚠ INCOMPLETE | v2 数据改进闭环（清洗 v2 数据 → 重训 → v2 评测对比） | checkpoints_v2/late_fusion_seed2.pt |
| M5 | ✅ DONE | 文档与可复现性 | — |
| M6 | ⚠ INCOMPLETE | M6a CanonicalToken 可移植性 + M6b 训练手段实验 + 数据集质量评测系统 | datasets/mmfi/v5tokens/ |

### 关键产物
| 名称 | 路径 | mtime | 状态 |
|------|------|-------|------|
| v1 leaderboard | `leaderboard_v1.json` | 2026-08-24 12:35 | ok |
| v2 leaderboard | `leaderboard_v2.json` | 2026-08-24 12:35 | ok |
| v2 eval 日志 | `logs/eval_v2.log` | — | MISSING |
| 评测协议 | `protocol.json` | 2026-08-24 12:35 | ok |
| 数据集质量 v1 | `results/quality_v1.json` | 2026-08-24 12:35 | ok |
| 数据集质量 v2 | `results/quality_v2.json` | 2026-08-24 12:35 | ok |
| 数据集质量 v4 | `results/quality_v4.json` | 2026-08-24 12:35 | ok |
| 数据集质量 leaderboard | `leaderboard_quality.md` | 2026-08-24 12:35 | ok |

### ⚠ 异常清单
- [ ] 缺失: checkpoints_v2/late_fusion_seed2.pt
- [ ] 缺失: datasets/mmfi/v5tokens/
- [ ] 缺失: logs/eval_v2.log
- [ ] 日志目录缺失: logs

<!-- FACTS:END -->

## 🧠 判断层

- 当前阶段：**C2 修复版全量闭环完成（2026-09-03，§14）——A/B/C 排序定案（单 seed）**。C1 后置检发现两洞（60.8% 复读、13% 幻觉过弱过滤）→ C2 管线修复（严格动作词过滤+复读拒绝+重试升级+num_predict+行缓冲修 GPFS 缓冲写丢失），3 分片 3.5h 全量生成（9205、均值 2.37 变体、strict-valid 100%、多样性 0.0712≈v4）。三臂重训：**排序 A > C2 > B 两级复现**（样本 r@1 0.0131/0.0109/0.0087；类级 0.4651/0.4444/0.3911），C2 仍救活 B 但未越过 A——v4 优势=多样性+信息内容（真实 RGB 视频来源），C2 只是槽位事实的同义重排。**A 自身跨轮方差 0.0185→0.0131 吞掉样本级 A/C 差距；最终定论必须多 seed。** 对齐绝对水平仍弱信号区。报告 §14，结果 `results/alignment_caption_c2.json`+`alignment_class_c2.json`，checkpoint `m6b_routec2_seed0.pt`，文本 `results/captions_route_c2_train.jsonl`（3 分片拼接）。
- 当前阶段（历史）：**sftmvp 伪 token SFT 反转实验 POSITIVE（2026-09-03，§15）**——LoRA(Qwen2.5-0.5B-Instruct, r16 qkvo) + projector 联合 SFT（冻结 m6b_v4text encoder 在线提 5×16 token；9205 train base × 4ep；作业 1062679 冒烟 / 1062693 全量，gpu_v100 29min）：**acc_pseudo 0.1396 vs text-only 0.0000（Δ=+0.1396，N=1870，双判据 0.074/+0.05 均过）**——M5c"冻结 LLM 读不懂伪 token"根因确认为**缺 SFT 教学段**，伪 token 本身可读。类结构发现：lunging 前向镜像对 0.98/0.91（传感器 token 空间比 CLIP 文本塔更可分），小幅静态动作（expanding chest ×2 / marking time / extending limb ×2）全 0，与弱模态短板类重叠。工程记录：① worktree 整树 datasets symlink 被 git 跟踪元数据破坏（splits/meta 入库 data/ 忽略）→ 改 data/ 逐个链接 + loader 0 样本 fail-fast；② minimind-o env 实测 = transformers 4.57.6 + torch 2.6.0+cu124 无 peft（补装 peft 0.20 + accelerate 1.14）；`conda run`/共享 base 解析有污染，作业一律用 env python 绝对路径。spec `docs/superpowers/specs/2026-09-03-sftmvp-design.md`（r2，含声明差距节），报告 `docs/reports/sftmvp_mvp_report.md`，产物 `checkpoints_sftmvp/` + `results/sftmvp/`。已合并 main（原 agent/sftmvp 分支 7 commit）。
- 当前阶段（历史）：**路线 C 全量三臂闭环完成（2026-09-02，§13）**——全量生成 9205×2 改写（3h06m，全部 ≥2 变体零回退）→ 三臂重训（作业 1061674/1061723）：**排序稳定 A > C > B 两级一致**（样本级 r@1 A 0.0185 / C 0.0153 / B 0.0076；类级 0.507 / 0.431 / 0.398）。**C 把 B 救活**（r@1 翻倍，§11 多样性机制获正向因果证据），但 A 的无上限原生多样性仍领先；A vs C 样本级差距在单 seed 方差内，类级更实在。C 不可替代属性：动词锚 100%/零填充语/侧别 98.6%/零幻觉（A 无保证）。单 seed 方差大（±0.003-0.005），最终结论需多 seed。报告 §13，结果 `results/alignment_caption_ab.json`（三臂）+ `results/captions_route_c_train.jsonl`，checkpoint `checkpoints_alignment/m6b_routec_seed0.pt`。当日主线洞见：**文本侧类内多样性=监督信号宽度，一致性=静态锚质量，二者是一对代价，最优解在中间**。
- 当前阶段（历史）：**Alignment 重训 A/B 完成（2026-09-02，§11）**——同协议双臂（v4text vs schematext，冷启动 base-only 9205×20ep，作业 1059804/1059857）：**B 臂训练 InfoNCE 更低（2.8581 vs 2.9618）但样本级与类级检索全输**（r@1 0.0109 vs 0.0163；类级 0.407 vs 0.437）。机制确认：**类内文本一致性在对比训练中是缺陷**（近重复目标=任务变易+监督变粗），v4 逐样本改写=免费文本增广。与 §10 合并：schema=更好的静态文本锚（CLIP 可分 0.893），v4=更好的训练目标；M5a"模板趋同→检索失败"假设获机制级确认。报告 §11，结果 `results/alignment_{caption_ab,class_ab}.json`，checkpoint `checkpoints_alignment/m6b_{v4text,schematext}_seed0.pt`。
- 当前阶段（历史）：**Caption 区分性 MVP 完成 + CLIP 嵌入验证（2026-09-02）**——keypoint schema caption（路线 B）3 轮迭代：lexical gate FAIL（distinct-2 结构性差距，如实报告），但 full_centroid 0.864 反超 v4 0.846。**补充实验（§10）**：CLIP text tower 嵌入空间 27 类质心 acc **v4 0.675 → schema 0.893（+21.8pp）**，类间余弦全面改善——**文本锚路线（M5 方向）证据复活**；镜像对余弦双方 ≈0.97 → 左右方向语义是 CLIP 文本塔固有限制，措辞无法解决。报告 `docs/reports/preflight_caption_mvp_report.md`（含 §10），结果 `results/clip_separability_v4_vs_schema.json`，脚本 `scripts/eval_caption_clip_separability.py`。**基础设施备忘：hf-mirror LFS 数据面（us.aws.cdn.hf.co）集群不可达，权重下载改走 ModelScope（openai-mirror org），CLIP 权重已落 `sensorbench/.models/`，评测用 minimind-o 环境（transformers 4.57 兼容 torch 2.6；test 环境 transformers 5.14 与 torch 2.4 冲突）。**
- [提议] gate 修订：distinct-2 对 anchor 型 caption 是错配指标（类内一致性本应是特性），改为 full_centroid 主指标 + 镜像对单列 + distinct-2 仅作下限（报告 §5）。
- [提议] 下一步（CLIP 验证已正向）：① gate 修订拍板后全量生成 v7 并落盘数据集版本；② 用 schema caption 重启 alignment 对照实验（v4 文本 vs schema 文本，同 AlignmentModel+InfoNCE，看 L1 r@1 是否突破 0.011 天花板）；③ 路线 C（LLM 润色 schema 骨架）可与 ② 并行评估。
- [提议] ② 已执行（§11，负结果+机制确认）——修正后的下一步：**路线 C 多样性版**（schema 骨架 + LLM 每样本 2-3 改写，控 distinct-2 与类锚并存），需 LLM 后端选型（ollama qwen2.5-vl 本地 vs deepseek-v4-flash:cloud）与全量 9205×3 生成成本估算后再提交；或先小样（200×3）过 A/B 闭环验证假设。
- 当前阶段（接上）：**路线 C 小样完成（§12，2026-09-02）**——ollama qwen2.5-vl-3b 作业内生成（3.7s/条），216×2 改写：多样性达标（distinct-2 0.054→0.255）、侧别保留 98.6%、动词过滤内联（19% 改写丢锚被拒）；**静态 CLIP 质心 acc 0.839→0.598**（改写表面变化抹平质心，反证 schema 高分离度来自词汇刚性）。判定：静态分离度与多样性是一对代价，裁决只能靠全量闭环训练。全量生成作业（9205×2，~2.4h，`gen_route_c_full.py`+`sb_route_c_full.slurm`，作业 1061391）排队中；产出后跑 C 臂训练（variants 轮转，`run_caption_alignment_ab.py` 需加 --captions-multi 支持）。
- 当前阶段（历史）：**v5_structfeat 主流程重训完成（A 完整落地）**——token_fusion robustness 0.4961→0.6858、acc_full 0.7632→0.9184；late_fusion robustness 0.3823→0.4979、acc_full 0.7084→0.8841。结构化特征让主流程直接用了 probe 已验证的可分特征。M5a/b/c 完成（M5c 负结果已归档）；M6a 完成——伪 token 作为可移植跨模态统一表征（CanonicalToken 4096-dim + 资产化），与 LLM 空间解耦。
- 发现与结论：
  - **v3 是首次真正提升**：token_fusion robustness 0.1425→0.3167（2.2x），acc_full 0.2382→0.5759；late_fusion 0.2138→0.2615。核心假设判明：弱模态缺独立判别力，不是任务难。
  - v2 全局过滤方向错误；数据 flywheel 靠"加信息"而非"删样本"。
  - **v4 = v3 rgb 规范化（髋部居中+躯干长，全 split）+ train 离线增强（翻转/平移/缩放，n_aug=4）**：train 46509（9689 原+36820 变体），val/test 不变。
  - **v4 评测结果（2026-08-16 00:10）**：token_fusion acc_full **0.7632**（v3 0.5759，+32%）、robustness **0.4961**（v3 0.3167，+57%）；late_fusion acc_full **0.7084**（v3 ~0.26，2.7x）、robustness **0.3823**（+47%）。关键点规范化+增强显著有效。
  - **v4 最大降幅缺口（token_fusion）**：miss-rgb 0.4024、miss-mmwave 0.1084；单模 only-wifi/only-depth 各降 ~0.73（弱模态仍缺独立判别力）。token_fusion robustness 0.496 > late_fusion 0.382（对齐机制仍占优）。
  - **v4 OOM 事故（2026-08-15 14:09，内核日志实锤）**：单样本 numpy ≈1.12MB，52686 样本全量进 RAM ≈ 61.7GB ≫ 27GB RAM + 7GB swap → swap 耗尽、页面分配失败、整个 WSL 冻结重启。`logs/train_v4_token_fusion_0.log` 0 字节、checkpoints_v4 为空，进程死在 `load_dataset`（framework/dataset/loader.py:40 cache dict 全量加载），训练未开始。v2/v3 之所以没爆：15866 × 1.12MB ≈ 18.6GB 能装下。
  - **v4 磁盘膨胀根因**：make_v4.py 的变体只改了 rgb（680B），却把 depth/wifi/lidar/mmwave 四模态（~1.17MB）随 pickle 全量复制 4 份 → 58GB（v2/v3 仅 18GB）。
  - **文档教训**：STATUS.md 原"load_dataset 需 2-3 分钟"只估了时间成本没估内存成本，属于误导性预期；"先量后跑"（启动前估算内存需求）已补进全局 AGENTS.md 长任务资源监控第 0 条。
  - **loader 已修复（2026-08-15）**：`framework/dataset/loader.py` 新增 `preflight_dataset`（文件 stat 估算 `样本数×单样本字节` vs MemAvailable+SwapFree）+ `load_dataset(mode=auto/lazy/eager)`。v4 实测：61.8GB 需求 > 34.2GB 可用 → 自动 lazy 逐批加载，峰值 RSS 1.07GB（修复前 62GB OOM）；train/val/test 语义与 eager 一致（9 个 loader 测试 + 全量 72 测试通过）。`mode='eager'` 且超限时直接 `MemoryError` 拒绝启动。
  - **v4 数据去重（2026-08-15）**：make_v4.py 变体改为只存 rgb 差异（delta 文件，~1KB vs 1.17MB）+ base_id 引用，loader 加载时解析 base 重建全模态变体。磁盘 58GB → **18GB**，每 epoch 从盘读 54GB → 读 0（eager 全进内存），训练从 ~25min/epoch（GPU 5%，磁盘瓶颈）→ **~2.6min/epoch（GPU 90%，计算瓶颈）**。旧 58GB 数据保留在 `datasets/mmfi/v4_big`。
  - **M5a 完成（2026-08-16）**：合成文本管线（v5，TemplateCaptioner 9205 个落盘 train base）、编码器 token 序列确认、AlignmentModel + InfoNCE、L1 检索评测。**注意：v5 用模板文本（LLM 后端延后），L1 数值反映模板文本的简单相关性，M5b 换 LLM 后端后对比需谨慎。**
  - **M5b 完成（2026-08-16）**：Perceiver 投影（per-modality 共享权重，缺模态显式置零无 NaN）、TokenRouter（半动态启发式）、LLMAdapter + LlamaAdapter（本地 llama2-7b）、L2 冒烟 PASS（prefix=4+text=7→merged=11 前向通过 + 文本回归）。两段式第二段就绪。
  - **M5c 完成但为负结果（2026-08-16，已归档）**：真训练 M5a（CLIP 锚，L1 r@1=0.0066 有信号但弱）+ 真训练 M5b projection（llama2 蒸馏）+ L3 三模式评测（text/pseudo-token/no-context）。关键发现链：
    - CLIP 文本锚 bug：`CLIPTextEncoder` 曾用 `last_hidden_state[:,0]`(BOS) 导致 27 类动作 sim=1.0 不可分；修复为 `pooler_output` 后可分性 0.316。
    - 原型初始化投影头（分类头 256→27 + 27 CLIP 原型）使 InfoNCE 从卡死(3.4649)转为收敛(3.28)，L1 r@1 有信号。
    - **伪 token 最近邻诊断**：caption 锚下落在情境词(arms/room)；改 verb 锚（动作词）后落在动作词域(jump/throwing/lung)，但仍不精确。
    - **核心负结论**：伪 token embedding 即使与 llama2 词表动作词相近，**冻结 llama2 仍读不懂**（L3 acc_pseudo=0）——冻结 LLM 从未被训练过"使用"伪 token 前缀。生成式 LLM 理解伪 token 需要 LoRA 微调（Qwen-VL/LLaVA 式），纯冻结不可行。
    - **架构教训**：CLIP 通用文本锚对细粒度动作语义有失效风险（需先验可分性）；"embedding 落点接近词表" ≠ "LLM 可利用"。
  - **M6a 完成（2026-08-16）**：CanonicalToken 协议（4096-dim 规范空间，modality-major，校验）+ CanonicalTokenizer（冻结编码器+Perceiver）+ 资产化（npz + index.json 版本化）+ LinearTokenToLLM（per-LLM 4096→hidden 线性投影）。**v5tokens 已生成（9205 个 train base，5.3GB）**。伪 token 成为可移植跨模态统一表征，换 LLM 只换维度投影层。135 测试通过。
  - **Concat MLP 升级 → PerModConcatMLP（2026-08-17）**：`framework/eval/dataset_quality/probe_fusion.py`——per-modality Linear(dim→64) → concat 320 维 → modality dropout 0.2 → MLP head 128。**修复朴素 concat MLP 两大问题**：① 维度支配（depth 50176/lidar 4608 主导）→ per-modality 投影均衡 ② rgb 几乎不用（贡献 0.06）→ modality dropout 强制学多模态。**v4 提升**：acc_concat 0.211 → **0.450**、CompactScore 0.211 → 0.450、rgb contribution 0.064 → **0.426**、lidar contribution 0.825 → 0.502（脱离维度主导）、**Quality 0.355 → 0.456**。contribution 现在与 per-modality probe acc 相关性更好（rgb 0.43, mmwave 0.63 强模态高；lidar 0.50, depth/wifi 0.23-0.25 弱模态下调）。61 测试通过（6 个 probe_fusion 新测试）。
  - **数据集质量评测系统完成（2026-08-17）**：`framework/eval/dataset_quality/` 三维度（info/compact/clean）+ 轻量 probe，**与下游任务/LLM/模板完全解耦**——probe 只看原始模态数据，测的是数据集固有属性。P0 护栏：test split 全程不进入 probe eval（argparse choices + validate_splits 双层防御）。51 单元 + e2e 测试通过。**v1/v2/v4 leaderboard 已出**（results/quality_v{1,2,4}.json + leaderboard_quality.md）。
  - **probe 升级完成（2026-08-17，Linear → MLP）**：① z-score 标准化 ② depth 最大池化 224→28（50176→784 维，消除 concat 维度支配）③ Linear → 2 层 MLP（256 隐）。**MLP 结果**：v4 acc_per_modality = rgb **0.828**（Linear 0.418）/ mmwave 0.326 / depth 0.077 / lidar 0.090 / wifi 0.051；acc_concat 0.050（Linear）→ 0.211（MLP）。**v4 quality 0.070 → 0.228**（v1 0.194 / v2 0.181）。rgb 是 v4 唯一新增的有效模态（v1/v2 无 rgb），wifi/depth/lidar 三个模态 v1→v4 始终 ≈ 随机 0.05-0.09——**独立验证主流程 robustness 弱模态判断**。MLP 修复 concat 退化 + CompactScore/anomaly 失真（anomaly 0.93→0.0）。**inconsistency_rate 仍为 1.0 死指标**（per-modality 独立 probe 无校准，JS 无意义，CleanScore=0.333 为地板值不作对比依据）。报告 `docs/reports/dataset_quality_v1_v2_v4.md` + spec + 计划已提交。
  - **A 高阶增益空间验证完成（2026-08-17）**：跨 leaderboard_v4.json 与 dataset_quality 交叉对比——**probe 下界**：rgb 0.828；**端到端上界**：token_fusion acc_full 0.7632、only-rgb 0.5861、only-mmwave 0.2955、late_fusion only-rgb 0.1796 / only-mmwave 0.4004。**核心结论**：① rgb 单模在 probe 比 token_fusion 的 only-rgb 强（架构内 rgb 处理有压缩损失）② concat probe 0.211 ≪ token_fusion acc_full 0.76 → 强架构在多模融合上 +55% 增益，**确实是 probe 看不到的高阶信号** ③ mmwave 在三种口径下都 ≈0.3-0.4（共识） ④ wifi/depth/lidar 在 probe 0.05-0.09 + token_fusion/late_fusion 0.03-0.05 一致接近随机（**跨架构多口径共振证明数据缺陷**）⑤ v4 信息空间定锚：probe 下界 0.83 rgb / 端到端上界 0.76 acc_full / mmwave 高阶互补 +0.18，其它三个模态**几乎没有信息贡献**。
  - **B 黄金子集完成（2026-08-17）**：`scripts/build_gold_subset.py` + `results/gold_subset_v1.json`——129 个 v4 val 样本（27 类各 ≤5），构造规则 token_fusion ∩ MLP-rgb-probe 双预测正确 → 高置信 ground truth 控制变量。class 14 0 个（极难）、class 22 4 个（其余 27 个类 5 个）。**用法**：跨数据集/模型改进复测归因（数据 vs 架构）。
  - **黄金子集 v2 完成（2026-08-17）**：`scripts/build_gold_subset_v2.py` + `results/gold_subset_v2.json`——**三方共识**（token_fusion ∩ rgb-probe ∩ mmwave-probe），85 个 val 样本（4 类为 0：class 9/12/14/22）。**mmwave 是瓶颈**（probe val acc 37% ≪ rgb 80% / tf 81%）；v1 滤掉的 85 个样本即"rgb 单模高置信"偏倚产物；v2 是 v1 严格子集+11 个新增。
  - **黄金子集 v2 实证完成（2026-08-17）**：`scripts/evaluate_gold_subset.py` + `results/gold_v2_evaluation.json`——**三项检验全过**：**A 高置信**：所有模型在 gold_v2 上 acc 远超 full val（tf 0.808→1.000 +0.19 / rgb-probe 0.809→0.988 +0.18 / mmwave-probe 0.372→0.776 +0.40）**C seed 不敏感**：rgb-probe seed1 在 gold_v2 上仍 0.953（gap +0.14）**B 难度对齐**：0-gold 类 (9/12/14/22) 在 full val 上 acc 0.46 ≪ 5-gold 类 0.87（差 0.41）。**gold v2 是合格的控制变量**——未来 v5 数据改进应在 gold_v2 上复测归因。报告 `docs/reports/gold_subset_v2_empirical.md`。
  - **黄金子集 v2 完成（2026-08-17）**：`scripts/build_gold_subset_v2.py` + `results/gold_subset_v2.json`——**三方共识**（token_fusion ∩ rgb-probe ∩ mmwave-probe），85 个 val 样本（4 类为 0：class 9/12/14/22）。**mmwave 是瓶颈**（probe val acc 37% ≪ rgb 80% / tf 81%）；v1 滤掉的 85 个样本即"rgb 单模高置信"偏倚产物；v2 是 v1 严格子集+11 个新增。
  - **inconsistency 重设计完成（2026-08-17）**：`compute_modality_contribution` 替换 `compute_inconsistency_rate`——根因是 per-modality **独立训练** probe 校准不可比，JS 恒 1.0；新方案用**单一 concat MLP** 视角，drop 模态 m 置零其特征 → argmax 变化率 = contribution_m。CleanScore 摆脱 0.333 地板 → **v4 Quality 0.228 → 0.355**。v4 contribution per-modality: rgb 0.064（concat 几乎不用 rgb）/ lidar 0.825（lidar 50k 维主导）/ mmwave 0.569 / wifi 0.579 / depth 0.249；**解读注意**：contribution 是 concat-probe 视角，与 per-modality probe acc 交叉读——rgb-contribution 0.06 + rgb-probe 0.83 → 揭示 MLP concat 融合瓶颈（非 rgb 数据缺陷）。
  - **M6b 复测（2026-08-17，llama2 4096 文本侧，n=918）**：原 CLIP 评测空间确认"评测天花板"——llama2 4096 替换后 baseline r@1 提升 3.5x（0.0022→0.0076）、变体 r@1 极差扩大 1.3x→2.5x；A r@1=0.0109 最高但 Δ vs baseline 仅 1SE（0.0033<0.0066），仍**无显著差异**。C（CE）在双评测空间同步最差（过拟合 27-way 分类，拖累文本对齐）。模板 caption 细粒度瓶颈仍在（r@1 max 0.0109 ≪ 随机 1/27=0.037）。脚本 `scripts/eval_alignment_llm_sweep.py`，结果 `results/m6b_llm4096_sweep.json`，报告 `docs/reports/m6b_alignment_matrix.md`（复测段）。**结论不变**：训练手段路线未救活，换评测空间只能"缓解"不能"解决"。
  - **M6b 实验完成但为负结果（2026-08-17 03:00）**：3 手段（大 batch / 分类辅助 CE / label-aware 负样本挖掘）5 变体矩阵（A-E），CLIP 512 L1 评测（n=918 held-out base）：
    - **L1 全部噪声级**：r@1 A-E ≈ 0.003-0.004（baseline 0.0022），远低于随机 1/27≈0.037 的一半；r@1 提升 0.0022 ≪ 2SE≈0.005 → **无显著提升**。
    - **训练 loss 与评测脱节（关键负结论）**：neg-mine 让 InfoNCE loss 降 33%（3.8→2.5，D/E），CE 辅助让 r@5/10 略升（C 0.0218/0.0447），但 r@1 完全没跟上——**优化了训练目标，没优化评测指标**。
    - **根因=评测天花板**：v5 模板文本（TemplateCaptioner）在 CLIP 512 空间几乎不可检索（918 base caption 模板化、语义趋同）→ L1 测不出编码器质量差异。手段可能有效但评测不敏感。
    - **batch 偏差**：spec 定 256，16GB 卡满载（97%）卡死、batch=128 慢速路径（bwd 2.32s/step）→ 改 **batch=64**（负样本 63/样本）。坑记录见 docs/LESSONS.md。
    - **代码已提交**（5 feat commit：label-aware InfoNCE + classification_head + train/eval 参数），实验报告 docs/reports/m6b_alignment_matrix.md。
    - **下一步方向**：换评测口径（llama2 4096 文本侧，M6a spec 定案）或等 LLM 后端 caption 落地后重测；训练手段（neg-mine/CE）保留作复测基线。
  - **主流程数据好坏评判机制（2026-08-16 复盘）**：protocol.json（15 profiles × 3 seeds）→ leaderboard（robustness=15 profile acc 均值 + acc_full + per-profile degradation）→ 数据质量结论。核心闭环已跑通 v1→v4。
  - **v4 数据质量诊断（2026-08-16，基于 leaderboard_v4.json 明细）**：
    - 有效信息支柱只有 rgb + mmwave：miss-rgb 掉 0.402、miss-mmwave 掉 0.108；miss-wifi/depth/lidar 几乎不掉（<0.01）——**不是因为弱模态好，而是强模态掩盖**。
    - 单模态独立判别力：only-mmwave 0.296（唯一有效非视觉模态），only-wifi 0.032 / only-depth 0.036 / only-lidar 0.050（≈随机 1/27=0.037）。**wifi/depth/lidar 数据形同虚设**。
    - 丢 rgb+mmwave（miss2-mmwave-rgb）直接崩到 0.037：数据实质是"rgb+mmwave 双模态"。
  - **主流程评判局限（robustness 指标盲区）**：① 测的是"丢模态代价"，强模态冗余会掩盖弱模态坏数据；② acc 是"模型×数据"耦合，非纯数据属性；③ 测不出语义可分性。→ 这就是 M6 补 L1 检索（表征可分性）的动机。
  - 弱项定位（v2）：缺 mmWave 时 class 0/5/17/25 全崩、S20/S40 最弱。
  - 评测已批处理（predict_batch 3.1x）；后台任务前台监控已写入全局 AGENTS.md。
  - **数据审查 GUI 工具（2026-08-18）**：`curation/gui/` Streamlit 工具——聚合看板（自动匹配 results/quality_*.json + 数据健康统计）+ 逐样本审查（5 模态可视化：rgb 骨架/depth 热图/wifi CSI/lidar 点云/mmwave 谱 + 文本/标签/质量/备注全功能编辑 + JSONL 追加式编辑日志 + 回滚）。构建脚本 `build_gold.py` 把 quality=golden 样本 + 修正文本/标签落盘为 datasets/mmfi/gold/（符合 Dataset 协议）。预计算脚本 `precompute_predictions.py` 用 token_fusion checkpoint 批量产预测供错误分析（真实冒烟发现 E04_S33_A01 低置信错分——正是人工复核对象）。loader 加向后兼容改动：split 发现改为读 splits/*.json 全部文件（支持 gold 等自定义 split）。42 单测 + 2 AppTest 冒烟通过；全量 250 测试通过。与另一 agent 的 dq-eval 工作零交叉（只读共享 loader/verbs，写仅限 curation/gui/ 与 tests/test_curation_gui/）。设计文档 docs/superpowers/specs/2026-08-18-dataset-gui-design.md。
  - **审查页播放器 + 3D 视角重构（2026-08-23）**：逐样本审查页播放器从"服务器驱动整页重跑"改为**每模态独立的 plotly 客户端动画图**（去掉全局播放总开关，每个模态自带 ▶播放/⏸暂停/滑块，直接点即播）。播放用 `redraw:true`，全模态（含 depth 热图、lidar/mmwave WebGL 3D）都能动画。动画图改用 **`st.cache_resource` 进程级缓存**，浏览器刷新不再重建全部 5 模态 297 帧图（实测刷新 11s→~5s，余下为数据集惰性加载）。**3D 视角**：支持手动填 eye/up/center 三向量（完整表达旋转）+ 应用/重置按钮；按样本持久化保存/加载到磁盘（`curation/gui/core/view_store.py`，`curation/gui/views/{dataset}.json` gitignored）；应用/重置只重建该模态不卡顿。**历史教训**：plotly 客户端动画此前因"拖动滑块图表消失"被回退（`2e2401a`→`0ea6c8e`），本次用"播放态客户端动画 + 静态帧可旋转"双路径绕开该坑；`redraw:false` 只更新 2D 线图、热图/3D 不刷新，故改 `redraw:true`（代价是播放时 3D 相机回默认，暂停态静态帧可旋转）。全量 303 测试通过。设计文档 docs/superpowers/specs/2026-08-23-playback-dual-path-design.md。
  - **环境统一到 conda（2026-08-18）**：新建专用 conda env `sensorbench`（python 3.12 + torch 2.9.1+cu128 + numpy 2.5 + scipy + opencv + pyyaml + pytest + transformers 4.44.2 + streamlit 1.61 + plotly 6.9），替代原 README 指定的 holollm/.venv（跨项目 venv 依赖）。镜像源安装：conda 用 USTC/NJU 镜像（tuna 的 pkgs/main/linux-64 repodata 缺失导致 conda 找不到 python 3.12），pip 用 Tsinghua pypi（torch 900MB 下载 ~43MB/s）。requirements.txt 补全 transformers/tokenizers/streamlit/plotly（缺它们 4 个 alignment/LLM 测试 ModuleNotFoundError）。**注意**：原 conda envs（mmfi/mmclip/RadarLLM/mmexpert 等）多为无 python 空壳，仅 mmwave 可用（py3.10 + torch 2.12.1，版本过新不采用）。
- 卡点 / 风险：**M5c 负结果：冻结 LLM + 伪 token 路线不成立**（生成式理解需 LoRA 微调，当前不做）；**M6b 负结果×2（CLIP 512 + llama2 4096 双侧确认）：3 训练手段仍无显著提升**，换 llama2 4096 评测仅缓解模板瓶颈、未救活训练手段；infra1/infra2 尚未加入（→7 模态）；弱模态（wifi/depth/lidar）仍缺独立判别力。
- **omni 调研 + 改进方向（2026-08-23）**：检索 paperhub 库内 omni 论文 + 新增下载分析 3 篇（OmniPack/OmniScope/Ex-Omni-2D），对比报告 `docs/reports/omni_model_comparison.md`。核心洞见：① Omni 模型基本假设模态齐全，本项目显式缺模态建模（MISSING token + mask + dropout）是差异化价值；② OmniScope"query 共享但各模态 salience 独立"与固定 16 token 均衡预算思路一致；③ OmniPack"pre-LLM 结构压缩 + in-LLM 语义精修"可借鉴。改进建议整理进 `docs/reports/improvement_plan.md`（P0：真实缺模态样本 + 多模态增强，直击 miss-rgb/mmwave 短板）。
- 🚨 **temporal 训练 cgroup OOM 事故（2026-08-24，整组冻结）**：`train.py --temporal` 启动 `train_v4_temporal_seed0.log`（0 字节即死）——根因 `train.py` 用默认 `load_dataset(mode=auto)` 对 v4 走 **eager 全量载入内存**，单进程 RSS **17.2GB** 撞 cgroup 硬上限 18GB（`/agents/opencode-main`，`memory.oom.group`）→ **整个 cgroup 组（含 opencode 会话 + 多个 streamlit）一起被 OOM 击杀**，表现为整终端冻结。**教训**：与 DomainEncoder 那次 RSS 17.8GB 近 OOM 同一坑——任何吃整集内存的 v4 raw 训练**必须 `mode="lazy"`**（`verify_temporal.py` 一直写对，`train.py` 漏了）。**已修复**：`scripts/train.py` 新增 `--mode`（auto/eager/lazy），`--temporal` 默认强制 `lazy`。运行时监控（GPU/内存/RSS）必须前置，不能只看时间成本。
- ✅ **时间建模（temporal）机制验证通过（2026-08-24）**——OOM 前已训好 `checkpoints_v4_temporal/token_fusion_seed0.pt`（temporal=True, raw v4），恢复会话后直接对 500 val 样本打乱帧序验证：
  | 模型 | 打乱帧序 delta |
  |---|---|
  | non-temporal（baseline） | **0.0000**（不感知时间） |
  | **temporal** | **-0.0453**（acc 0.854→0.809）|
  - **结论：temporal 建模生效**——`framework/models/temporal.py`（TemporalAggregator: RoPE + 模态内时序自注意力 + 时间池化）让模型真正依赖了时间顺序，修复了"encoder mean 坍缩 T 维、打乱帧序 acc 不变"的盲区（`diag_time_order.py` 诊断：v4/v5_structfeat 均 delta=+0.0000）。temporal=True 时 encoders 保留 T 轴（(B,T,N,D)）→ TemporalAggregator→(B,N,D)，MISSING token 路径兼容（`test_temporal_missing_modality_works`）。集成 + 5 测试通过（23 测试全绿）。
- ✅ **temporal 全量 30 epochs 完成（2026-08-25，集群 gpu_v100）**——`checkpoints_v4_temporal/` 3 seeds 训完（seed0 best val 0.904 / seed1 0.894 / seed2 0.853），`run_eval.py --protocol protocol_v5.json`（5 模态 21 profiles）生成 `leaderboard_temporal_full.json`：
  | 模型 | robustness | acc_full |
  |---|---|---|
  | v4 (old) | 0.4961 | 0.7632 |
  | v4_shuffle | 0.6265 | 0.8746 |
  | v5_structfeat | 0.6858 | 0.9184 |
  | **temporal (新)** | **0.6904 ± 0.0435** | **0.9451** |
  - **结论：temporal 是当前最佳**——acc_full 0.9451（超越 v5_structfeat 0.9184，历史最高）、robustness 0.6904（也超过 v5_structfeat 0.6858）。temporal 建模不仅让模型感知时间，还显著提升缺模态鲁棒性与全模态精度。
  - **关键 profile**：full 0.9451 / miss-mmwave 0.8135 / miss-rgb 0.7413 / miss2-mmwave-rgb 0.1472 / only-mmwave 0.6826 / only-rgb 0.7431。
  - **集群经验**：gpu_v100 每卡 60G 内存，v4 eager 18G 完全装得下，**用 eager 而非 lazy**（lazy 逐样本读 GPFS 慢）。两个 seed 并行会共用 GPU（CUDA_VISIBLE_DEVICES=0），GPU 利用率约 30-60%，可用 batch=64 提升。完整训练 30 epochs 单 seed 约 45-68 分钟（数据加载 ~13min + 训练 ~40min）。
- ✅ **细粒度数据集质量矩阵完成（2026-08-25，集群 2680v4 CPU）**——`framework/eval/dataset_quality/finegrained.py`（方案 A，测纯数据属性，不做领域特征工程）+ `scripts/run_finegrained.py` + `tools/compare_quality_matrix.py`。按 (类别×受试者) 分组（621 格，**排除 __aug 变体**，从样本 id 解析 subject/env），每格 3 信号合成质量分（**纯特征，不依赖任何主模型**）：
  - **信号1 compactness**：格内特征紧凑度（`1 - std`），纯数据属性。**设计注记**：最初用主模型 `main_acc`（temporal token_fusion 识别该格正确率），但发现**"在 v4 上训练的模型评价 v4 数据"构成自我验证偏差**——模型见过这些样本，main_acc 反映"记没记住"而非"数据可分性"，且 acc 是"模型×数据耦合"（M6 motivation 原话）。故去掉主模型，改纯特征紧凑度，真正测数据集固有属性。
  - **信号2 consistency**：该格 vs 同类其他受试者的余弦相似度
  - **信号3 separability**：质心欧氏距离/类内离散度（`between/within`），sigmoid 归一化。**实现注记**：原计划用 `compute_fisher_ratio`，实测在 8980 维原始特征下塌缩到 ~0（无区分度），改用质心欧氏距离度量。
  - 合成 `quality = 0.4·compactness + 0.3·consistency + 0.3·separability`
  - 输出 `results/quality_matrix_v4.json`（global quality 0.860，per_class 27 / per_subject 23 / per_env 4 / matrix 619 格 / low_quality）。
  - **关键发现（纯特征版）**：A22/A01/A19 类数据质量问题较突出（compactness 0.81-0.86、separability 0.60-0.70），是 v5 数据改进的明确靶点。对比主模型版（A05 类低）——两者反映不同侧面：纯特征版反映"数据本身不可分"，主模型版反映"模型难学"。
  - **性能教训**：① 特征预提取一次并缓存（原实现 O(格数×样本数) 次提取，卡死；改为 O(样本数) 次）；② 集群 CPU 节点内存充足，**用 eager 而非 lazy**（lazy 逐样本读 GPFS 慢，全量 46509 样本 eager 加载 ~10min + 特征/矩阵 ~8min）；③ 用价格最低的 2680v4 队列。

## 🗂 决策层

- [提议]（sftmvp §15）：多 seed 复跑（seed 1/2，~30min/次）确认 acc_pseudo 方差后，再决定是否投入 7B 同构复现。
- [提议]（sftmvp §15）：A/B 文本臂迁移——SFT 答案措辞用 v4text vs route-C caption 对照，验证"监督信号宽度"洞见在 SFT 范式下是否成立。
- [x] 下一步行动（已定）：重跑 v2 评估，生成 leaderboard_v2.json，更新 robustness 报告。✓ 完成
- [x] `[已定]`：v3 迭代第一步——针对 miss-mmwave 降幅定位弱项（(profile × class × subject) 细粒度矩阵），不做全局样本过滤；评估 late_fusion 作为主模型。✓ 完成
- [x] `[已定]`：v3 迭代第二步——加入 RGB/红外模态评估（用户选轻量方案：先只加 rgb）。✓ 完成，结论=数据信息量不足
- [x] `[已定]`：v4——关键点规范化（髋部居中+躯干长，全 split）+ train 离线空间增强（翻转/平移/缩放）。✓ 训练+评测完成（token_fusion robustness 0.3167→0.4961，acc_full 0.5759→0.7632；late_fusion 0.26→0.38/0.71）
- [x] `[已定]`：v4 训练前先修 loader——惰性/分块加载 + 启动前内存预检（样本数×单样本字节 vs 可用 RAM），避免再次 OOM。✓ 完成（`load_dataset(mode=lazy/auto)` + `preflight_dataset`，实测 v4 峰值 RSS 1.07GB）
- [x] `[提议]`：make_v4 变体不再复制全量模态（仅 rgb 差异），58GB 磁盘膨胀可降至 ~20GB。✓ 完成（delta 文件 ~1KB，磁盘 58GB→18GB，训练 25min/epoch→2.6min/epoch）
- [x] `[已定]`：M5a——合成文本 + 编码器 token 序列 + 规范空间对齐（InfoNCE）+ L1 检索评测。✓ 完成（v5 数据集 + AlignmentModel + train/eval_alignment，spec 3 轮评审 Approved）
- [x] `[已定]`：M5b——Perceiver 投影 + LLMAdapter + router + L2 冒烟。✓ 完成（per-modality Perceiver + TokenRouter 半动态 + LlamaAdapter(2-7b) + L2 inject 冒烟 PASS，112 测试）
- [x] `[已定]`：M5c——真训练 + L3 端到端 LLM 能力评测。✓ 完成但**负结果**（伪 token 落动作词域但冻结 llama2 不可读 acc=0；已归档，结论=生成式理解需 LoRA 微调）
- [x] `[已定]`：M6a——CanonicalToken 可移植性架构（协议+资产化+LinearTokenToLLM）。✓ 完成（v5tokens 9205 base，135 测试）
- [x] `[已定]`：M6b——提编码器对齐质量（大 batch/分类辅助 loss/锚对比），L1 检索提升。✓ 完成但**负结果**（3 手段均未提升 CLIP 512 L1，评测天花板；batch 256→64，报告 docs/reports/m6b_alignment_matrix.md）
- [x] `[已定]`：M6b 复测 llama2 4096 文本侧，验证"换评测空间"能否救活训练手段。✓ 完成（baseline r@1 3.5x 但变体差异仍未达 2SE，路线整体仍负）
- [x] `[已定]`：双轨指标体系（probe 下界 vs 端到端耦合上界）已写入 spec §双轨指标 + 黄金子集 + 决策规则。✓ 完成
- [x] `[已定]`：高阶增益空间 A：probe 下界 vs 端到端上界交叉对比。✓ 完成（v4 信息空间定锚：rgb 下界 0.83 / 上界 0.76 / mmwave 高阶 +0.18）
- [x] `[已定]`：黄金子集 B：v4 val 上 token_fusion ∩ probe-rgb 双预测正确，129 样本。✓ 完成
- [x] `[已定]`：黄金子集 v2：三方共识（token_fusion ∩ rgb-probe ∩ mmwave-probe），85 样本（4 类为 0）。✓ 完成。**mmwave 是瓶颈**（probe val acc 37% ≪ rgb 80%/tf 81%）→ v2 比 v1 滤掉了 85 个"rgb-only 高置信"样本（v1 only），**显著减少了 rgb 偏倚**
- [x] `[已定]`：黄金子集 v2 实证（高置信/可重复/难度对齐三项检验全过）✓ 完成 — 类 9/12/14/22 是 v5 数据改进的明确靶点
- [ ] `[提议]`：黄金子集 v3——加 late_fusion 做四方共识，或扩到 depth-probe 共识（虽然弱但增加多样性）；当前 v2 (85 样本) 已足够大多数诊断场景。
- [x] `[已定]`：v5 数据改进——**两次实验**：
  - **v5_hardaug（失败）**：6 难类各 +8 变体（flip+translate+scale，与 v4 同策略）→ 13464 新样本；**训练失败**：ep 0-5 val 0.04-0.05（random），与 v4 ep 5=0.276 对比明显退化。**根因**：增广策略与 v4 完全相同（无新信息维度）+ 给不可学类（class 14 acc 0%）加样本只会让模型在不可学数据上多过拟合。**教训**：增广不能解决数据根本问题（标签/语义），v5 必须换方向（弱模态特征 / 新模态）。
  - **v5_structfeat（部分赢部分输 + concat 退化未解决）**：depth/wifi/lidar/mmwave raw → 领域结构化特征，数据 18GB → 272MB（65x 压缩）。**单模 probe 大幅提升**：depth 0.08→0.27 / wifi 0.05→0.11 / lidar 0.10→0.21 / mmwave 0.35→0.51。**但 concat 退化**：v4 acc_concat 0.358 → v5 0.109，contribution per modality 全 0（drop modal 不影响 argmax → 模型退化为预测 class 0）。**尝试 PerModCrossAttnMLP（CLS + multi-head attention）救场**：v4 上反而退化到 0.258，已回退到 ConcatMLP 默认。**Quality**：v4 0.423 / v5_structfeat 0.351（单模强但 concat 拖后腿）。**根因**：结构化特征量纲跨度差异（depth max 13000 vs rgb max 1）使 PerModConcatMLP 的 per-modality projection 训练崩溃。**未解方案**：需要在投影前加 LayerNorm/BatchNorm，或为结构化特征加专用 normalization。
- [x] `[已定]`：🚨 **数据事故**：make_v5_structfeat.py 用 hard-link 写 v4 共享 Inode → 覆盖了原始 v4（probe 全跑 raw 数据看上去变成结构化数据）。**已恢复**：从 v3 重跑 make_v4.py，v4 完整恢复 (rgb normalized, depth/wifi/lidar/mmwave 原始)。**教训**：硬链接写入是新数据时危险的——必须先 unlink 再写，或用独立 inode（cp + rm）。

- [x] `[已定]`：**v5_structfeat concat 退化修复完成（三层修复）**：
  - **根因**：`extract_wifi_features` 的 `np.corrcoef` 在恒定输入时返回 NaN → v5_structfeat 创建时 NaN 写入 pickle → 训练时 NaN 输入 → logits 全 NaN → argmax 总是 0 → acc_concat 0.109（恰好 = val class 0 占比）。
  - **修复 1**：corrcoef 加 `std<1e-9` guard；所有 extractor 用 `_safe` wrap（NaN/Inf→0）；v5_structfeat 数据重新生成（替换 stale NaN pickles）。
  - **修复 2**：PerModConcatMLP per-modality projection 后加 BatchNorm1d（均衡量纲）。
  - **修复 3**：cross-entropy 加 class_weighted（inverse-frequency），打破 "predict class 0" 局部最优。
  - **结果**：v5_structfeat Quality 0.351 → **0.620**；acc_concat 0.109 → **0.788**；contribution per modality 全 0 → rgb 0.42 / mmwave 0.28 / lidar 0.15 / depth 0.11 / wifi 0.09。v4 也顺带受益（0.423 → 0.620）。**新 leaderboard**：v1 0.356 / v2 0.337 / v4 0.620 / v5_structfeat 0.620。
- [ ] `[提议]`：换评测目标——动作-动作 verb 相似度 / 类内紧致度，绕开"整句 caption 模板化"陷阱（M6b 报告复测结论 §3 剩余路径 #1）
- [ ] `[提议]`：**数据构成/增强/主题架构改进（2026-08-23，omni 调研后整理，见 `docs/reports/improvement_plan.md`）**：
  - **P0-1.1 真实缺模态样本**：数据层物理去掉 mmwave/rgb 生成真实缺模态样本，让模型学"无 mmwave 靠 wifi+depth"强先验，直击 miss-mmwave 0.824 / miss-rgb 0.765 短板。
  - **P0-2.1 多模态增强**：mmwave/lidar 点云增强（旋转/抖动/点 dropout）+ wifi 时域增强 + depth 与 rgb 同步几何变换。
  - **P1-1.2 组合缺失增强**：训练模拟组合缺失，对齐评测 miss2-* profile。
  - **P1-2.2 跨模态一致增强**：共享几何模态（rgb/depth/lidar/mmwave）同步空间变换。
  - **P1-1.3 模态感知 token 预算**：复用 TokenRouter，缺模态时高信息模态（rgb）更多 token（OmniScope 洞见）。
  - **P2-3.1/3.2 query 条件 + 分层融合**：引入文本 query 锚点 + pre/in-LLM 分层精修（OmniPack 洞见）。
  - **P2-3.3 缺模态建模显式化**：将缺模态建模作为一等公民文档化+评测（本项目差异化价值）。
- [x] `[已定]`：**P0-1.1 真实缺模态样本机制验证（2026-08-23，负结果）**——实现按模态偏置 dropout（`TrainConfig.modality_dropout` + token_fusion `_dropout_mask` + train.py `--modality-dropout`），mmwave/rgb 缺失率 0.5，token_fusion × 3 seeds 对比 baseline：
  | profile | MD | baseline | Δ |
  |---|---|---|---|
  | full | 0.9032 | 0.9184 | **-0.015** |
  | miss-mmwave | 0.7950 | 0.8237 | **-0.029** |
  | miss-rgb | 0.7888 | 0.7650 | **+0.024** |
  | miss-lidar | 0.8629 | 0.8800 | -0.017 |
  - **结论**：偏置 dropout 只对 miss-rgb +0.024，但 miss-mmwave -0.029、full -0.015，整体 robustness 0.6858→~0.66（净负）。**根因**：提高缺失暴露率以牺牲 full 精度为代价。**关键洞察**：token_fusion 的 MISSING token 机制已能处理缺失，偏置 dropout 与"真实缺模态样本"机制等价（avail=False→MISSING token）→ **P0-1.1 不投入构建 v7**。代码保留（`modality_dropout` 配置 + 测试 `test_token_fusion_per_modality_dropout`），默认不启用。详见 `docs/reports/improvement_plan.md` §6。
- [x] `[已定]`：**预训练 encoder 验证（2026-08-23，有限收益）**——omni 模型普遍接成熟预训练 encoder（CLIP/CLAP），本项目 encoder 全从零训练。用 resnet50（ImageNet 冻结，avgpool 2048-d）在 v4 原始 depth 图像上提取特征 → probe，对比当前 v5_structfeat 63d 结构化特征：
  | 特征 | depth probe acc |
  |---|---|
  | v5_structfeat 63d 结构化特征（当前） | **0.305** |
  | resnet50 冻结特征（MLP probe） | 0.308 |
  | resnet50 冻结特征（linear probe） | 0.337 |
  - **与 omni 差异澄清**：omni 接预训练 encoder 因输入是自然图像/音频（预训练域匹配）；本项目 depth 深度图/rgb 关键点/wifi CSI 与预训练域不匹配，先验迁移收益有限。脚本 `scripts/probe_depth_pretrained.py` 保留可复用。详见 `docs/reports/improvement_plan.md` §7。
- [ ] `[提议]`：**架构迁移：主流程回归 v4 raw + DomainEncoder（2026-08-23，编码器已实现，实验因性能暂停）**——为修正 v5_structfeat"领域知识固化进数据层"对目标1（probe 测纯数据质量）和目标2（通用评测模型）的偏离，计划把领域特征从数据层迁到评测模型 encoder：
  - **已实现**：`framework/models/domain_encoder.py`（DomainEncoder：raw 数据 → extract_*_features 现场提取 → MLP → (B,N_TOK,D)）+ token_fusion 增加 `domain`/`domain_dims` 配置（save/load 持久化 + 向后兼容）+ train.py `--domain` 标志。4 个新测试（forward/save-load roundtrip）。**全量 306 测试通过**。
  - **方向确认（人拍板）**：v5_structfeat 保留作对照数据集（特别标注"领域特征非中立"）；主流程回归 v4 raw；新增 DomainEncoder 作为可插拔先验 encoder（领域知识 = 模型能力，不污染数据层）。
  - **暂停原因**：DomainEncoder 的 numpy 逐样本特征提取（`[self.extractor(x_np[i]) for i in range(B)]`）是严重性能瓶颈——v4 raw 上 1 epoch 10+ 分钟未完成、CPU 117% GPU 长期空置、v4 18GB 全量进内存 RSS 17.8GB 近 OOM。违背"GPU 利用率优先"原则。
  - **下一步选项**（未定）：① 加 functools.cache 按样本 id 缓存提取结果（只算一次，但数据集仍需保存 raw id）② 只对单模态用 DomainEncoder 验证价值 ③ 预提取缓存到磁盘（回到数据层但保留 raw 备份）。
- [ ] `[提议]`：M6c——数据质量改进（先补弱模态独立判别力：wifi/depth/lidar，或加 infra1/infra2 → 7 模态），主流程 robustness 直接受益。dataset_quality v1/v2/v4 leaderboard 已指明弱模态（wifi/depth/lidar acc ≈ 随机）→ v5 数据改进的明确靶点。
- [x] `[已定]`：v5_structfeat concat 退化修复——**三层修复已完成**（corrcoef NaN guard + PerModConcatMLP per-modality BatchNorm + class_weighted CE，line 194），Quality 0.423 → 0.620。归入历史。
- [x] `[已定]`：**mmwave 维度消融实验**（2026-08-20，`scripts/probe_mmwave_ablation.py`，v3 base 9689/1968，Linear probe 20 epochs × 3 seeds）。**核心结论**（违反假设）：
  - **Doppler 不是最重要**：drop_dim3_doppler 只 -6.0%±0.2%，only_doppler -37.1%±1.1%（远不及 baseline）
  - **几何坐标（特别是 dim2）是核心**：drop_dim2_z -27.5%±0.7%，drop_geom_xyz -25.2%
  - intensity 是最弱信号（only_intensity -53.2% ≈ 随机）
  - **物理意义**：mmwave 点云本质是人体几何（姿态/位置），Doppler 维度被帧间噪声抹平
- [x] `[已定]`：**geom_v2 特征工程对比**（`scripts/probe_mmwave_geom_v2.py`，3 seeds × 20 epochs）——验证消融结论能否转化为特征工程改进：
  - **A. raw 320d**: 0.376±0.002（baseline）
  - **B. v5_current 50d**: 0.359±0.002（**比 raw 还低**——方向错）
  - **C. geom_v2 xyz_only 94d**: 0.616±0.004（**+63.7%** vs raw）
  - **D. geom_v2 xyz+signal 134d**: **0.709±0.002（+88.5% vs raw）**
  - v5_current 的 50 维特征没起正面作用；geom_v2 关注 dim2 几何分布 + dim3,4 统计后，**mmwave probe 接近 0.71**——是当前最强单模态特征
- [x] `[已定]`：**v5_structfeat 修复 mmwave 特征**——`extract_mmwave_features` 已替换为 geom_v2 版本（134d 笛卡尔几何），mmwave probe val 0.36 → 0.71，v5_structfeat Quality 0.589→0.679。✓ 完成（见下方"mmwave 实现修复"）
- [x] `[已定]`：**mmwave 5 维真实语义确认（2026-08-20）**——基于 MMFi 官方代码 (`/home/li/datasets/MMFi_dataset/mmfi_omni/codecs/mmwave_vae.py`) 与 V5 报告：
  - **真实列序**：`[x, y, z, doppler, intensity]`（TI IWR 雷达点云格式）
  - 前 3 列 (x/y/z) 是**笛卡尔 3D 坐标**，被官方代码当 3D 点云处理（FPS 采样 line 65 / Chamfer Distance 损失 line 104-106 / per-frame 中心化 train_mmwave_codec.py:44）
  - **v5_structfeat 注释的 (range, velocity, azimuth, SNR, elevation) 完全错**——spherical 错读。`extract_mmwave_features` 抽取的特征方向反了
  - 消融实验结论与此一致：dim2 (z 坐标) 是核心信号，doppler/intensity 是辅助
- [x] `[已定]`：**mmwave 语义全面更新（2026-08-20）**——所有错读 spherical 注释/docstring 已修正为正确笛卡尔语义：
  - `framework/eval/dataset_quality/feature_extract.py:167-200`——`extract_mmwave_features` docstring 重写，附完整证据链 + 警示
  - `curation/gui/core/renderers.py:338-385, 393, 416`——`render_mmwave`/`_mmwave_xyz` docstring 重写 + 两个 hovertemplate 改为 x/y/z/doppler/intensity
  - `tests/test_curation_gui/test_renderers.py:77`——测试函数改名 `test_mmwave_polar_to_cartesian` → `test_mmwave_xyz_extraction_legacy_polar`，docstring 标注 LEGACY + 待修
  - `docs/LESSONS.md #19`——二次修正，补充笛卡尔语义证据 + 教训
- [x] `[已定]`：**mmwave 实现修复（2026-08-20）**——两份关键函数从 spherical 错读转为正确笛卡尔几何：
  - `curation/gui/core/renderers.py::_mmwave_xyz` 现在返回 `pts[:, :3].copy()`，移除球→笛卡尔转换；GUI 3D 视图从错位→正确（之前 x' = x·cos(el)·cos(az) 的假变换没了）
  - `framework/eval/dataset_quality/feature_extract.py::extract_mmwave_features` 替换为 geom_v2 设计（134 维）：T 帧 × (16 XYZ 几何 + 8 doppler/intensity 统计) + z 直方图(8) + xy 范围(3) + 质心漂移(3)
  - `curation/gui/core/renderers.py::_FEATURE_SECTIONS["mmwave"]` 更新 56→134 维分段（8 sections：5 帧 + z 直方图 + xy 范围 + 质心漂移）
  - 测试：`test_mmwave_xyz_extraction_cartesian`（断言直接取 x,y,z）+ `test_structured_feature_uses_segmented_view` 更新 (mmwave, 134, 8)。**全量 267 测试通过（6 deselected 是环境跳过）**
  - **mmwave probe val_acc 0.359→0.7093**（+98%，脚本 `probe_mmwave_geom_v2.py`，20 epochs × 3 seeds）
  - `render_mmwave` + `_mmwave_single` docstring/hovertemplate/title 同步更新：颜色变量 `pts[:,3]` 是 doppler 不是 intensity，colorbar_title 改"doppler (m/s)"
- [x] `[已定]`：**dataset_quality 路径 + v5_structfeat 数据集升级（2026-08-20）**——端到端 mmwave 修复落地：
  - `framework/eval/dataset_quality/modality_probe.py::extract_modality_feature_downsampled` 对 mmwave 分支调用 `extract_mmwave_features`（ndim=3 时）；ndim=1（已是结构化特征）直通
  - 重跑 `make_v5_structfeat.py`：52686 v4 pickles hardlink + 15866 base 重抽（mmwave 50→134d），旧版备份到 `datasets/mmfi/v5_structfeat_old50d/`
  - 重跑 `run_dataset_quality.py` 在新 v5_structfeat：**Quality 0.589→0.679**（+15%）；mmwave probe 0.337→0.765；acc_concat 0.788→0.869；InfoScore 0.251→0.330；CompactScore 0.723→0.869
  - `leaderboard_quality.md` 更新：v5_structfeat row 0.589→**0.679**
  - `tests/test_dataset_quality/test_run_e2e.py` 修复：toy dataset 给 mmwave 3D shape (5, 64, 5)，原 2D 触发 shape 错误；67 test_dataset_quality 测试通过
- [x] `[已定]`：🚨 **数据事故（2026-08-20，v4 被污染后恢复）**——重跑 `make_v5_structfeat.py` 时 hard-link 让 v5_structfeat 的 base 与 v4 共享 inode，重写 v5 时**覆盖了 v4**（depth/wifi/lidar/mmwave 全变成结构化特征 63/129/353/134，非原始数据）。**已恢复**：从 v3 重跑 `make_v4.py`，v4 完整恢复（depth 5,1,224,224 / wifi 5,3,114,10 / lidar 5,1536,3 / mmwave 5,64,5 / rgb 5,17,2），52686 样本（train 46509 / val 1968 / test 4791）与原始一致。污染备份已删除。**教训**：make_v5_structfeat 的 hard-link 写入是危险的——必须先 unlink 再写，或用独立 inode（cp + rm）。**注意**：v4 恢复后，`results/quality_v4.json`（基于污染数据）已失效，需重跑 `run_dataset_quality.py` 重新生成
- [x] `[已定]`：**v4 恢复后重评（2026-08-21）**——`run_dataset_quality.py` 在恢复的 v4（原始数据 + 新 mmwave 特征）上重跑：**Quality 0.489→0.646**（+32%）；mmwave probe 0.337→0.773；acc_concat 0.529→0.848；InfoScore 0.193→0.267；CompactScore 0.529→0.848。`leaderboard_quality.md` 更新 v4 row 0.489→**0.646**。结果 `results/quality_v4_after_recover.json`
- [x] `[已定]`：**主流程 v4 leaderboard 重训**——`leaderboard_v4.json` 当前 mmwave probe 0.35（raw 320d）来自 v4 主流程训练数据集评测；token_fusion/late_fusion 模型用 raw mmwave 输入。要不要把 v4 → v5_structfeat 升级也覆盖到主流程评测？需要：(1) make_v5_structfeat_from_v4 把 mmwave 替换 (2) 重训 token_fusion/late_fusion (3) 重跑 leaderboard_v5。✓ 完成（即"主流程 v4 leaderboard 全量重训（A 完整落地，2026-08-21）"：checkpoints_v5_structfeat_v2/ 6 个 checkpoint + leaderboard_v5.json token_fusion 0.6858/0.9184、late_fusion 0.4979/0.8841 + protocol_v5.json 5 模态 21 profiles）
- [x] `[已定]`：**depth/lidar/wifi 特征工程复核（2026-08-21）**——`scripts/probe_weak_modality_feat.py`（v3 base 9205/1870，Linear probe 20 epochs × 3 seeds）：
  - **depth**：raw 0.106 → 当前特征 0.230（+0.124）✅ 方向对，无需改
  - **lidar**：raw 0.080 → 当前特征 0.289（+0.209）✅ 方向对，无需改
  - **wifi**：raw 0.093 → 当前特征 0.080（-0.013）❌ 方向错
  - **wifi_v2 落地**：`extract_wifi_features` 替换为 wifi_v2（161d，捕捉时间变化/子载波/天线相关/帧间运动），probe 0.080→0.110（+37%）。GUI `_FEATURE_SECTIONS["wifi"]` 更新 129→161d（9 sections）。重跑 make_v5_structfeat（wifi 129→161d）+ quality：**Quality 0.679→0.683**，wifi probe 0.080→0.091，acc_concat 0.869→0.878。`leaderboard_quality.md` 更新 v5_structfeat row 0.679→**0.683**
  - **结论**：wifi 提升有限（数据本身信息量低，only-wifi ≈ 随机），但特征工程方向已修正
- [x] `[已定]`：🚨 **make_v5_structfeat hard-link 事故修复（2026-08-21）**——重跑 make_v5_structfeat 时 hard-link 再次污染 v4（base 共享 inode，重写覆盖 v4）。**已修复脚本**：base 重写前先 `os.unlink(dst_p)` 解除 hard-link，再写独立文件（v4 不再被污染）。v4 已从 v3 重跑 make_v4.py 恢复。**教训**：make_v5_structfeat 的 hard-link 写入必须 unlink 再写，或用独立 inode
- [x] `[已定]`：**主流程结构化特征验证（2026-08-21，A 提议最小验证）**——`scripts/validate_struct_token_fusion.py` + `framework/models/encoders.py::MLPEncoder`（1D 结构化特征→(B,16,D)，与 PointEncoder 兼容）。在 v5_structfeat 上用 MLPEncoder（wifi/depth/lidar/mmwave）+ PointEncoder（rgb）训练 token_fusion，15 epochs × 3 seeds：
  - **only-mmwave 0.296 → 0.657/0.655/0.681（均值 0.664，+124%）**——结构化 mmwave 特征让模型直接用了 probe 已验证的可分特征，而非从稀疏点云重新学
  - **acc_full 0.763 → 0.901/0.901/0.903（均值 0.902，+18%）**
  - test_only_rgb 0.586→0.626/0.681/0.707；test_miss_mmwave 0.655→0.809/0.844/0.840
  - **3 seeds 稳健**（only-mmwave 0.65-0.68 窄区间）。**结论：主流程重训收益明确巨大，值得全量落地**
  - 注意：15 epochs（非完整 30）；best_val 0.83-0.86 已有过拟合迹象（ep13 掉到 0.72），完整 30 epochs 需早停
- [x] `[已定]`：**主流程 v4 leaderboard 全量重训（A 完整落地，2026-08-21）**——给 token_fusion/late_fusion 正式加 MLPEncoder 结构化特征路径（`structured` 配置 + save/load 持久化 + 向后兼容裸 state_dict），train.py 自动检测结构化特征。在 v5_structfeat 上全量 30 epochs × 2 模型 × 3 seeds 重训，重跑 leaderboard_v5：
  - **token_fusion：robustness 0.4961 → 0.6858（+0.19），acc_full 0.7632 → 0.9184（+0.16）**；only-mmwave 0.2955 → 0.6686（+0.37）、miss-mmwave 0.6548 → 0.8237、miss-rgb 0.3608 → 0.7650
  - **late_fusion：robustness 0.3823 → 0.4979（+0.12），acc_full 0.7084 → 0.8841（+0.18）**；miss-mmwave 0.1961 → 0.4684、only-rgb 0.1796 → 0.3004
  - **关键 bug 修复**：`fit()` 缺 per-epoch shuffle 导致 acc_full 只有 0.72（validate 脚本有 shuffle 达 0.90）；补上 `random.Random(seed).shuffle` 后复现 0.90。**教训：训练循环必须 shuffle，否则固定 batch 顺序让模型过拟合到 batch 内分布**
  - 产物：`checkpoints_v5_structfeat_v2/`、`leaderboard_v5.json`、`protocol_v5.json`（5 模态 21 profiles）
  - **2×2 归因（2026-08-22，补跑 v4 raw+shuffle 对照组 `checkpoints_v4_shuffle/` + `leaderboard_v4_shuffle.json`）**：
    | config (token_fusion) | robustness | acc_full |
    |---|---|---|
    | v4 raw, NO shuffle（旧） | 0.4961 | 0.7632 |
    | v4 raw, +shuffle（新） | 0.6265 | **0.8746** |
    | v5 struct, +shuffle（最终） | 0.6858 | **0.9184** |
    - **shuffle 贡献：+0.1114 acc_full（0.7632→0.8746）——主因**
    - **结构化特征贡献：+0.0438 acc_full（0.8746→0.9184）——次要**
    - **结论修正**：之前把"结构化特征"当主因是错的。真正主因是 shuffle bug 修复（+0.11），结构化特征单独只贡献 +0.04（且无 shuffle 时甚至略降）。validate 脚本一直显示 0.90 正因为它一直有 shuffle，掩盖了 fit() 缺 shuffle 的缺陷。
  - **类别均衡采样 vs 随机 shuffle（2026-08-22，负结果）**——`framework/models/batching.py::BalancedIndexer`（强制 batch 内每类数量差 ≤1）+ `TrainConfig.batch_strategy`（shuffle/balanced）+ `scripts/train.py --batch-strategy`。在 v5_structfeat 上 token_fusion × 3 seeds 对比：
    | batch 策略 | robustness | acc_full |
    |---|---|---|
    | shuffle（保留自然分布） | 0.6858 | **0.9184** |
    | balanced（强制每类相等） | 0.4344 | **0.6807** |
    - **结论：类别均衡是负结果**。逐类看，shuffle 多数类 acc 0.9+，balanced 多个类崩到 0.2-0.6（class 3/6/23 等），仅 class 24 受益（0.757→0.962）。
    - **根因**：本数据类别差仅 4.7x（4685 vs 1005），不算极端不平衡；强制均衡 = 对稀有类过度采样，扭曲真实分布。batch 64/27=每类仅~2 个，均衡把头部类梯度稀释 + 稀有类重复过拟合。
    - **纠正口头解释**：shuffle 有效**不是**因为"让 batch 多类"，而是去偏 + 保留自然类别分布；"刻意均衡"反而破坏类别先验。测试 `tests/test_batching.py`（5）+ `test_token_fusion_balanced_batch_trains`，全量 288 passed。
  - **类别权重 CE vs 采样（2026-08-22，中性结果）**——`batching.py::class_weights`（inverse_freq / sqrt_inverse_freq）+ `TrainConfig.class_weight` + `train.py --class-weight`。在 v5_structfeat 上 token_fusion × 3 seeds 对比：
    | 策略 | robustness | acc_full |
    |---|---|---|
    | shuffle（baseline） | 0.6858 | **0.9184** |
    | class-weight inverse_freq | 0.6828 | 0.9141 |
    | class-weight sqrt_inverse_freq | 0.6707 | 0.9133 |
    - **结论：类别权重 CE 基本中性**（0.914 vs 0.918 baseline，-0.004），远好于 balanced（0.68）。它不扭曲采样，只在 loss 层温和调权，所以不崩。
    - **逐类看**：确实帮到部分稀有类（class 19 +0.125、class 24 +0.232），但也伤到另一些（class 26 -0.206、class 23 -0.121），净效果 ≈ 0。
    - **综合结论**：本项目类别不平衡仅 4.7x，**改采样（balanced）有害、改 loss 权重（class-weight）中性**——都不值得采用。shuffle + 无权重是最优。测试 `test_class_weights_*`（4），全量 291 passed。
- [x] `[已定]`：**新增测试覆盖（2026-08-21）**——补足本次会话改动缺失的自动化测试：
  - `tests/test_models.py`：`test_mlp_encoder_structured_feature`（MLPEncoder 4 个维度 134/161/63/353 → (B,16,D)）+ `test_mlp_encoder_batch_variable_size`
  - `tests/test_dataset_quality/test_feature_extract.py`（新文件）：mmwave 134d / wifi 161d / depth 63d / lidar 353d 维度回归 + NaN/Inf 清除 + **make_v5_structfeat hard-link 防污染回归**（unlink 后写独立 inode，src 不被覆盖）
  - **全量测试 267 → 275 passed**（+8），GUI 冒烟 4 → 5 passed
  - `tests/test_curation_gui/test_app_smoke.py`：新增 `test_v5_structfeat_all_modalities_render`——切到 v5_structfeat 后验证 5 模态（wifi/depth/lidar/mmwave/rgb）都渲染 plotly chart（raw 数据存在时显示原始帧，否则 fallback 结构化特征分段视图）。**注意**：review 页优先用 raw 数据（`action_frames`/`segment_frames`），结构化特征分段视图仅在 raw 不可用时出现
- [x] `[已定]`：inconsistency 指标重设计——drop-modality contribution。✓ 完成（CleanScore 摆脱 0.333 地板，v4 Quality 0.355）
- [x] `[已定]`：改进 concat MLP 容量（更深 / 跨模态注意力）→ ✓ 完成（PerModConcatMLP：per-modality projection 64 + modality dropout 0.2 + MLP head 128，v4 Quality 0.456）
- [ ] `[提议]`：黄金子集 v3——加 late_fusion 做四方共识，或扩到 depth-probe 共识（虽然弱但增加多样性）；当前 v2 (85 样本) 已足够大多数诊断场景。
- [ ] `[提议]`：v4 评测后，若规范化+增强有效，考虑加 infra1/infra2 关键点（→7 模态）或跨模态对齐。
- [x] `[已定]`：**temporal 全量评测（2026-08-24→25）**——✓ 完成：v4 raw token_fusion temporal=True × 3 seeds 全量训练（`train.py --temporal --mode eager`）→ `leaderboard_temporal_full.json`（protocol_v5 21 profiles）。**temporal 当前最佳**：robustness 0.6904（>v5_structfeat 0.6858）、acc_full 0.9451（历史最高，>v5_structfeat 0.9184）。详见判断层 2026-08-25 条目。
- [x] `[已定]`：数据集可视化 + 文本修正 GUI（Streamlit）——聚合看板 + 逐样本审查 + JSONL 编辑日志 + build_gold 构建黄金数据集。✓ 完成（curation/gui/，42 单测 + 2 AppTest 冒烟，全量 250 测试通过）。黄金数据集的下一步：在 GUI 中人工复核 gold_subset v2 的 85 个 val 样本，修正其文本/标签后 build_gold 产出 v3 黄金集。
- [x] `[已定]`：**hard-link 防污染写入统一（2026-08-21）**——把散落在各 make_* 脚本的 unlink 保护抽成公共 helper `curation/io.py::safe_replace_pickle`（目标若与源共享 inode 先 unlink 再写，源不受污染）。三处脚本改用：`make_v5_structfeat.py`（原内联 unlink 逻辑）、`make_v6_relabel.py`（relabel 写路径，原裸 open(wb) 有污染风险）、`make_v5_hardaug.py`（变体写路径，原裸 open(wb)）。新增回归测试 `test_safe_replace_pickle_shared_helper`（断言共享 inode 下源不被覆盖）。**全量测试 275 → 276 passed**（+1）。✓ 完成
- [x] `[已定]`：**逐层跨模态 CKA 诊断 v4 融合机制（2026-09-02）**——实现 `framework/eval/dataset_quality/layer_cka.py`（Linear CKA + per-modality encoder hook + transformer layer1 hook，3 seeds mean±std），用 `checkpoints_v4_temporal/token_fusion_seed{0,1,2}.pt` 在 v4 val (1870 样本) 上跑完。声明：`docs/reports/cka_layerwise_v4_proposal.md`。
  - **关键发现 — mmwave×rgb 从不对齐**：enc_out CKA=0.177（3 seed 均值）、layer1_out CKA=0.219。**Δ = +0.04**——是 10 个模态对里唯一 Δ < 0.05 的。其它 9 对在 layer1_out 都飙升 0.2-0.6（如 wifi×rgb 0.013→0.269、depth×wifi 0.037→0.458）。
  - **回答 AI 提议的最重要问题**（v4 mmwave 是"浅层特异深层融合"还是"从未对齐"）：**从未对齐**。mmwave 在 transformer 浅层就几何独立，深层 attention 也没把它与 rgb 拉到同一表征空间——但 mmwave contribution 0.625（v4 dataset_quality）说明它提供了 rgb 没有的判别信号。
  - **反讽结论**：mmwave 是 v4 的"互补"信号源，**但互补机制不是几何融合而是 attention mask 隔离的独立 oracle**。对 M6b 训练手段实验的含义：加 cross-modal alignment loss（CKA-based）会**伤害** mmwave 的判别力，应改用"特征独立正则"（惩罚 mmwave×rgb CKA 升高）。
  - **理想曲线 A.浅层特异/深层融合**：**局部成立**。浅层（enc_out）10 对全部 < 0.2 → 所有模态都保特异；但深层融合仅对 9/10 对成立，mmwave×rgb 例外。
  - 产物：`results/layer_cka_v4.json`（12683 B）+ `results/plots_v4/layer_cka_curve.png` + 报告 `docs/reports/layer_cka_v4.md`（迭代报告）。Runner `scripts/run_layer_cka.py` + `scripts/plot_layer_cka.py`，slurm `jobs/layer_cka_v4.slurm`（normal_test CPU 队列，sglang-0.5.10-cuda12.8 env，~7min/3seeds）
- [ ] `[提议]`：**M6c 训练手段实验方向修正（2026-09-02）**——基于 layer_cka 发现，传统 alignment loss（CKA-based / MMD / contrastive）会破坏 mmwave 独立 oracle 的判别力。提议实验集：
  1. **mmwave 独立性正则**：loss = -λ · CKA(mmwave, rgb)·pool  + acc_loss（明确"保住 mmwave 不与 rgb 对齐"）
  2. **per-modality 早停**：每个模态独立的 val head 监控，超过 plateau 单独停
  3. **mmwave-aware dropout**：训练时强制 rgb-miss 而 mmwave-pres 比率 ≥ 0.5，强迫模型依赖 mmwave 决策路径
  4. 与现有 modality_dropout（已验证负结果）正交 — 上述都是**主动**正则，不是被动 dropout
  - **⚠ 2026-09-02 controls 修正**（`results/layer_cka_controls.json`，job 1059481）：随机初始化同架构模型 layer1_out CKA=0.97-1.0（全塌缩），训练后 0.15-0.48 → **深层 CKA 上升主要是架构塌缩倾向被训练部分抵消，不是"学出融合"**。原前提"其他模态融合、mmwave 不融合"不成立（实为"训练让所有对保持分化、程度不同：wifi×rgb 0.15 最分化 … wifi×depth 0.48 最接近塌缩"）。**M6c 实验设计需先补 per-layer linear probe 区分"对齐 vs 塌缩"再定**，上述 3 候选降优先级保留
  - **⚠ 2026-09-02 probe 迭代完成**（`results/layer_probe_v4.json`，job 1059621）：深层是**功能性信息混写**——wifi/depth/lidar 浅层探针 ≈ 随机（0.04-0.10）但深层 0.66-0.70（attention 把类信息写进各模态 token 段），同时 CKA 0.15-0.48 ≪ 随机 0.97（几何不塌缩）。**mmwave 独立 oracle 获探针证据**（enc 0.422 / layer1 0.574 各自可解码 + 与 rgb CKA 仅 0.23）。浅层探针独立复现 dataset_quality 模态层级（rgb 0.78/mmwave 0.42/其余随机）。**对 M6c：alignment loss 仍应避免**；wifi 深层高分是借来的（miss-wifi 不掉分三角印证），v5 方向仍是弱模态自有信息。masked-context probe（仅 wifi 可用）为下步
  - **退出条件**：① val acc 提升 > 0.02 或 miss-mmwave robustness 提升 > 0.05 → 进入主流程；② 3 个变体均无显著差异 → 收尾 M6c
- [x] `[已定]`：**Depth encoder 三臂诊断——模型式前景/先验注入（2026-09-02）**——回答"depth 语义丰富为何 encoder ≈ 随机、能否用模型替代传统分割"。8 臂同协议对比（train 2997 分层 → val 1870）：
  | arm | val acc |
  |---|---|
  | tiny_raw / tiny_masked | 0.063 / 0.065 |
  | vit_raw / vit_masked / vit_raw_long(150ep) | 0.078 / 0.054 / 0.079 |
  | vit_mae_probe(冻结) / vit_mae_ft(lr1e-3) | 0.095 / 0.059 |
  | **vit_mae_ft_lowlr(lr1e-4)** | **0.146** |
  | 参照：手写运动统计(v5_structfeat) | 0.27 |
  - **H1 容量 FAIL / H2 背景 FAIL（Mask R-CNN mask 干净但零收益，不进主流程）/ H3 MAE 先验 PARTIAL WIN（2×，但 lr 敏感：1e-3 打崩 0.059，1e-4 才 0.146）**
  - **核心修正**：depth 动作信号在**跨帧运动**而非单帧外观——手写帧差分统计 0.27 碾压一切学习方案；推荐 depth 输入加运动通道（帧差分）作为 M6 数据改进方向
  - 产物：`docs/reports/depth_foreground_prior_v4.md` + `results/depth_arms_ckpt/vit_mae.pt`（可复用 MAE 权重）+ `framework/models/depth_vit.py`
  - 基建坑（已修）：GPU 节点需 `LD_LIBRARY_PATH` 完全覆盖为 env lib（否则 cudnn 符号冲突 abort）；hf-mirror 大文件重定向 us.aws.cdn.hf.co 被墙，torchvision 权重走 download.pytorch.org 并行 range 下载
- [x] `[已定]`：**Depth 振兴路线 A 执行完成——rgb关键点对比蒸馏（2026-09-02，job 1060087）**——原提案"关键点回归"因标定缺失降级为 InfoNCE 对比蒸馏（teacher=rgb keypoints MLP 帧级 sanity 0.604；student=MAE init ViTDepthEncoder）：
  - **distill_probe（冻结）0.133**（纯 MAE 0.095 的 1.4×）
  - **distill_ft（低lr微调）0.223**（纯 MAE 0.146 的 **1.53×**、从零 0.078 的 2.9×），逼近手写 0.27
  - **语义中间表示假设在 MMFi 成立**；MAE 先验与蒸馏两级可加（0.078→0.146→0.223）；蒸馏管线可推广（teacher 换 mmwave/wifi）
  - 产物：`results/distill_route_a.json` + `scripts/distill_depth_route_a.py` + 报告 `docs/reports/depth_revival_ab_v4.md`
- [x] `[已定]`：**Depth 振兴路线 B 阻塞确认（2026-09-02）**——原始 82G tar 解压时已损坏（`exp/mmfi_extract3.log` Unexpected EOF）且删除，`~/MMFi_dataset/` 已空，官方分发（GitHub→GDrive/百度）集群无可靠再获取路径 → **T=16/32 重 ingest 无源可用**。若未来重获数据按 STATUS 路线 B 原案执行
- [ ] `[提议]`：**路线 B'（T=5 运动通道）+ 组合拳（2026-09-02）**——B'：depth 输入 1ch→5ch `[d_0, 4×帧差]`（DMM 思路，作业 1060121 进行中）；组合拳：运动通道 × MAE init × 蒸馏三者正交可叠加，预期 depth 单模态突破 0.27 后接回 token_fusion 主流程重训
- [x] `[已定]`：**路线 B' 完成——帧差分通道是 depth 的最大单项杠杆（2026-09-02，job 1060783）**——depth 输入 1ch→2ch `[d_t, d_t-d_{t-1}]`（Δ_0=0），vit_motion_raw 从零训练：
  - **val acc 0.4738**：vs vit_raw 0.078（**6.1×**）、vs 手写运动统计 0.27（**1.75×**）、vs 蒸馏最优 0.223（2.1×）
  - **确证**：depth 判别信号在跨帧运动，显式给帧差分后无需任何先验；mask/MAE/蒸馏在原始 1ch 输入上的努力全部被这一行预处理超越
  - **下一步**：组合臂（motion × MAE init × distill 正交性未测，预期继续上探）+ motion 通道接回 token_fusion 主流程重训（8 通道×5 模态融合重评 leaderboard）
  - 产物：`results/depth_motion_channels.json` + `scripts/depth_motion_channels.py`（`ViTMotionEncoder` 2ch 契约兼容）
- [x] `[已定]`：**组合臂完成——motion 通道包含全部先验收益（2026-09-03，job 1061406）**——2×2 正交检验（raw/motion × MAE/蒸馏）：
  | input \ init | 从零 | +MAE | +蒸馏 | +MAE+蒸馏 |
  |---|---|---|---|---|
  | raw 1ch | 0.078 | 0.146 | 0.223 | — |
  | motion 2ch | **0.474** | 0.467 | 0.407 | 0.387 |
  - **结论：信息 > 先验**——MAE（静态重建先验）/rgb 蒸馏（外观语义先验）只对信息匮乏的 raw 输入有效，motion 通道给出后无增益甚至轻微干扰（种子方差内单调降）。先验注入是信息不足时的拐杖
  - **下一步**：`ViTMotionEncoder`（2ch）接入 token_fusion 主流程（DepthEncoder 同契约可替换）+ 全量重训 + leaderboard 重评；wifi/lidar 试运动通道
  - 产物：`results/depth_combo_arms.json` + `results/distill_teacher.pt` + `scripts/depth_combo_arms.py`
- [x] `[已定]`：**motion 通道接入主流程——负结果（2026-09-03，job 1061450，3h52m）**——token_fusion 新增 `motion_depth` 契约（DepthEncoder→ViTMotionEncoder 2ch，save/load 持久化 + `train.py --motion-depth`），v4 raw temporal 全量 3 seeds + 21 profiles 评测：
  - **robustness 0.6399 vs 基线 0.6904（-0.050）；acc_full 0.9159 vs 0.9451（-0.029）**——单模态 6.1× 增益**未迁移**到融合
  - 劣化集中：mmwave/rgb 缺失 profiles 均值 -0.092、only-lidar -0.082（depth 缺席的 profile 也劣化）
  - **根因**：共享 transformer 全模态共享权重，depth token 分布改变重塑全部模态学习动态（非局部手术）；motion 版早停过早（ep11-16）也可能截断
  - **新增独立问题**：only-depth 进融合后 ≈ 随机（0.058，基线同）——[MISSING]-token 淹没单模态 depth 信号，encoder 探针 0.474 无法通过融合路径兑现
  - **修正结论链**：信息>先验（单模态）→ **单模态增益 ≠ 融合增益（共享 fusion 中换 encoder 不是局部手术）**
  - 下一步选项：① encoder 输出 LayerNorm 对齐 token 统计后重验 ② 保守变体（tiny conv + diff 通道）③ mmwave/rgb 缺失时路由 motion-depth 专家头 ④ 修 only-* 悖论（单模态可用时旁路 MISSING 机制）
  - 产物：`docs/reports/motion_main_pipeline_v4.md` + `leaderboard_motion_v4.json` + `checkpoints_motion_v4/`
- [x] `[已定]`：**选项 1（LayerNorm token 对齐）验证完成——无效（2026-09-03，job 1062327）**——`motion_depth_layernorm` 开关（depth token encoder 后 LayerNorm，已合入契约持久化），1 seed 验证：
  - robustness 0.6352 vs 无 LN 3-seed 0.6399（持平略降，种子方差内）；acc_full 0.9057 vs 0.9159；val best 0.837 vs 0.820（+0.017 未兑现到 leaderboard）
  - **排除假设：劣化不是 token 统计（范数）问题，而是信息内容**——强 motion-depth token 使共享 transformer 训练时重新分配注意力、边缘化弱模态路径；LayerNorm 无法改变信息内容
  - 选项 1 关闭。剩余：② 保守变体（tiny conv + diff 通道，分布扰动最小）③ 专家路由 ④ only-* 悖论修复
  - 产物：`leaderboard_motion_ln_seed0.json` + `checkpoints_motion_ln/`
- [x] `[已定]`：**only-* 悖论修复（adaptive_pool）——大成功，motion 变体转正（2026-09-04，job 1063019）**——`adaptive_pool` 开关（fusion 后只平均在场 token：1 模态→/16、2→/32...，full profile 数学等价旧 mean，已 smoke 验证；契约持久化 + `train.py --adaptive-pool`）。motion_depth + adaptive_pool 1 seed：
  - **robustness 0.7068**：vs 无 AP 0.6399（**+0.067**）、vs temporal 基线 0.6904（**+0.016，motion 变体首次转正**）
  - **acc_full 0.9395**（vs 无 AP 0.9159 +0.024；vs 基线 0.9451 持平）
  - mmwave/rgb 缺失 profiles 全面大幅回升（miss-rgb +0.144 / miss2-wifi-rgb +0.158 / miss-mmwave +0.117）；only-lidar 0.184 **超基线 0.143**、only-mmwave/only-wifi 也超基线
  - **残留**：only-depth 仍 0.059——池化修复不覆盖它（训练中 depth-only 组合概率 ~0.3%，共享 transformer 未学过 depth-only 模式），独立后续问题
  - 注意：1 seed 结果，扩 3 seeds 确认后建议 AP 进入主流程默认
  - 产物：`leaderboard_motion_ap_seed0.json` + `checkpoints_motion_ap/`
- [ ] `[提议]`：**Depth 振兴路线 A——rgb关键点→depth 跨模态蒸馏 MVP（2026-09-02）**——业内统治级范式是"语义中间表示"（NTU SOTA 全是 skeleton-based）；rgb 关键点 (17,2) 已在 v4 数据中当现成老师：
  1. 训 depth→关键点热图回归（小 CNN/ViT，逐帧任务**不依赖 T=5**），监督 = 同帧 rgb 关键点经外参投影对齐（MMFi 提供标定）
  2. 产出 depth-keypoints 新模态 → 复用现有 PointEncoder(2) 管线，与 rgb-keypoints 平行入融合模型
  3. 验收：depth-keypoints 单模态 probe ≥ 手写特征 0.27；miss-rgb 时融合模型掉分收窄
  - 成本：~1 天（蒸馏训练 + probe 验证）；风险：rgb 老师自身遮挡误差、外参精度
- [ ] `[提议]`：**Depth/Lidar 振兴路线 B——T=16/32 重 ingest + 运动通道 + 大规模 MAE（2026-09-02）**——T=5 帧协议把时序信息掐死（业内点云动作 SOTA 用 32-64 帧），是数据协议缺陷非模型缺陷：
  1. 重 ingest MMFi 原始序列到 T=16（或 32），帧率/采样间隔对齐 27 类动作时长分布
  2. depth 输入加显式运动通道 `[d_t, d_t-d_{t-1}×k]`（DMM 思路，业内验证；手写帧差分 0.27 已证明该信号存在）
  3. MAE 预训练扩规模：全 46k train 的 ~230k 帧 + 100 epochs（当前 15k 帧/50ep 玩具规模已验证管线，权重 `results/depth_arms_ckpt/vit_mae.pt`）
  - 成本：重 ingest ~1 天 + 预训练 ~0.5 天 GPU；**决策点**：T 升级会让数据体积 ×3-6，需先量后跑确认磁盘/内存预算（preflight_dataset 会把关）
  - 退出条件：depth 单模态 probe 0.27→≥0.4 视为路线成立；否则 depth 维持 v5_structfeat 手写特征方案
