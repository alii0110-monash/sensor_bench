# sftmvp — 伪 token SFT 反转实验报告（MVP）

> 分支 `agent/sftmvp` ｜ 作业 1062679（smoke, normal_test）/ 1062693（train+eval, gpu_v100/gpu10）
> spec：`docs/superpowers/specs/2026-09-03-sftmvp-design.md`（r2）｜ 2026-09-03

## 1. 结论：POSITIVE（M5c 负结果被反转）

| 指标 | 值 | 判据 | 通过 |
|---|---|---|---|
| **acc_pseudo**（with-token, 1870 val base） | **0.1396** | ≥ 0.074 (2×random) | ✅ (3.8×random) |
| **acc_text**（text-only 配对基线） | **0.0000** | — | — |
| **Δ = acc_pseudo − acc_text** | **+0.1396** | ≥ +0.05 | ✅ (2.8×阈值) |
| unmatched 生成 | 32/1870 (1.7%) | — | 格式学习良好 |

**一句话**：冻结 Qwen2.5-0.5B-Instruct 经 LoRA+projector SFT 后，能从 80 个传感器伪 token 中读出动作类别（文本先验下完全瞎猜 acc=0）；M5c"冻结 LLM 读不懂伪 token"的根因确认为**缺少 SFT 教学段**，而非伪 token 本身不可读。

## 2. 训练与评测事实

- 训练：9205 train base（captions join）× 4 epochs × batch 32 × seed 0，fp32，1.1 steps/s，V100 单卡全程 29m13s
- val loss 走势：0.0947 → 0.0662 → 0.0820 → 0.0365（epoch 2 有回升，epoch 3 最佳）
- 评测：held-out 1870 val base（98 个 id 磁盘缺失，清单在 `results/sftmvp/eval_mvp.json`）
- 产物：`checkpoints_sftmvp/`（projector.pt + adapter/ + run_config.json）、`results/sftmvp/{eval_mvp.json,eval_mvp_generations.jsonl,train_log.json,class_anchors.json}`

## 3. 类别结构（有信息的发现）

- **接近解决**：lunging toward the left/right front **0.98/0.91**、twisting right 0.40、throwing left 0.37、picking up 0.35——镜像对（CLIP 文本塔余弦 0.97 的左右方向问题）在**传感器 token 空间**反而是最可分的
- **完全失败（0.0）**：expanding chest horizontally/vertically、marking time in place、extending the left/right limb——多为小幅/静态上半身动作，与 v4 主流程"弱模态缺独立判别力"的短板类高度重叠
- 生成样例：class 0（stretching and relaxing）大量被答成 twisting left/right——混淆集中在低幅度运动簇

## 4. 与声明的差距（模板第 4 步）

**达成**：主判据 POSITIVE 且双条件均超阈值；MVP 边界全部遵守（未做 L1 复测/部署/多 seed/主流程改动）；单 seed 如声明。

**偏差与修正**（全部发生在实现期，已回写 spec r2）：
1. **环境（评审 CRITICAL）**：spec v1 的环境表来自被污染的 `conda run` 解析；实测用户 env `minimind-o` = torch 2.6.0+cu124 / transformers 4.57.6 且**无 peft** → 补装 peft 0.20 + accelerate 1.14（用户 env 内，未动共享 base；base 的 numpy 误升级已还原 1.26.4）
2. **val 基数（评审 CRITICAL）**：1968 → **1870**（98 个 id 磁盘缺失，集中在 E04_S33 块）；评测报告实际 N + 缺失清单
3. **train 基数机制**：split 排除 `__aug` = 9689，其中 484 无 caption/文件 → 与 captions jsonl **按 id join** = 9205
4. **worktree 数据符号链接坑**：`datasets/mmfi/*/splits` 等元数据被 git 跟踪，`git worktree add` 物化真实目录使整树 symlink 失效（data/ 为空 → 全部样本"缺失"）→ 改为**只对 data/ 目录逐个链接**；`load_split_base` 加 0 样本 fail-fast
5. **评测解码**：手动贪心（KV cache）而非 `generate(inputs_embeds=...)`，规避 4.57 兼容风险——实际未踩坑，属防御性偏差
6. **估算偏差**：全量训练实测 17.5min（估 1-2h，显著高估）；两次冒烟共 ~21min（第一次因 4 暴露问题）

**未验证/遗留**：单 seed 方差未知；0.5B 结论对 7B 的可迁移性未知；缺模态 profile 泛化未测；per-class 0.0 类是否可通过 route-C 多样性文本（A 臂优势）改善未测。

## 5. 下一步（[提议]，待拍板）

1. **多 seed 复跑**（seed 1/2，每次 ~30min）确认 acc_pseudo 方差
2. **A/B 文本臂迁移**：用 v4text vs route-C caption 作为 SFT 答案措辞，验证"监督信号宽度"洞见在 SFT 范式下是否成立
3. **7B 同构复现**（先 A 后 B 的 B 步）：llama2-7b + LinearTokenToLLM，数字与 M6b 评测空间对话
4. **与主流程对话**：SFT 后的 LLM 表征是否可通过 L2 注入改善主流程（M5b TokenRouter 复活路径）
