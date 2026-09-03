# sftmvp — 伪 token SFT 反转实验设计（MVP）

> 日期：2026-09-03。分支 `agent/sftmvp`，worktree `.worktrees/sftmvp`。
> 对应声明：本文件即声明（目标/现状/问题分析/路径取舍）的定稿版。

## 1. 目标与判据（写死，不可事后移动）

**一句话**：反转 M5c 负结果——验证"经 SFT 的 LLM 能否学会使用传感器伪 token"，拿到 0/1 信号。

- **主判据**：`acc_pseudo`（held-out val 上，给伪 token 前缀，LLM 贪心生成动作短语并匹配 27 类）
- **POSITIVE 条件**：`acc_pseudo ≥ 0.074`（2×random）**且** `acc_pseudo − acc_text ≥ +0.05`（配对：同一样本去掉 token 重问）
- **NEGATIVE 处理**：如实报告，给出 loss/生成样例的机制分析
- **MVP 边界（明确不做）**：L1 检索复测、caption 生成质量、部署、多 seed、主流程 token_fusion 改动

## 2. 现状（已核实）

| 资产 | 状态 |
|---|---|
| 冻结 encoder 权重 | `checkpoints_alignment/m6b_v4text_seed0.pt`：34 张量，5 模态 encoder 全在，输出 (B,16,256)/模态 |
| CanonicalToken 资产 v5tokens | 缺失 → 在线提取（encoder 前向，无资产依赖） |
| 指令数据 | `results/captions_route_c_train.jsonl`：9205 条 `{id,label,variants[2]}`，27 类锚定短语可多数投票提取 |
| LLM 基座 | Qwen2.5-0.5B-Instruct（ModelScope，用户拍板 A 路线），本地无 HF 权重需下载 |
| 环境 | conda env `minimind-o`：peft 0.19 + accelerate + transformers 5.2 + torch 2.4 + modelscope 1.23 |
| 算力 | gpu_v100 单卡（32G 显存 / 60G RAM），提交前查余量 |

## 3. 架构

```
framework/llm_sft/
  classmap.py    # captions → class_id→动作短语（多数投票）；答案规范化与匹配
  dataset.py     # load_split_base（只读 base，预过滤 __aug，先量后跑）；collate
  projector.py   # SensorTokenProjector: Linear(256→hidden) + LayerNorm + 模态类型嵌入
  prompting.py   # chat 序列构造（占位符 span 注入 embeds）、左填充批处理、答案匹配
  train_sft.py   # LoRA(r=16 q/k/v/o) + projector 联合训练；loss 仅 assistant 段
  eval_sft.py    # 配对评测：with-token vs text-only；手动贪心解码（带 KV cache）
scripts/{train_sft,eval_sft}.py   # CLI 入口
tests/test_llm_sft/               # 单测（tiny Qwen2 config，无外部权重）
jobs/sftmvp_{train,eval}.slurm    # gpu_v100 作业
```

**数据流**：v4 raw（train base 9205 / val base 1968，排除 `__aug`）→ 冻结 m6b_v4text encoder
→ (B,5,16,256) → projector → (B,80,hidden) 注入 user 段占位符位置 → Qwen2.5-0.5B（LoRA）
→ 生成动作短语 → 规范化匹配 → acc。

**冻结边界**：encoders 全程冻结（量尺不变）；LLM 底座冻结；只训 projector（lr 1e-3）+ LoRA（lr 1e-4）。

## 4. 关键决策记录

- **基座 = Qwen2.5-0.5B-Instruct**（用户拍板）：MVP 原则下最快拿 0/1 信号；llama2-7b 同构复现留作信号为正后的第二轮
- **在线提 token**：不重建 v5tokens 资产（缺失），encoder 前向成本可忽略
- **训练只测 full profile**（5 模态全在）：MVP 只回答"LLM 能否读懂伪 token"，缺模态泛化留给后续
- **监督目标 = 类锚定动作短语**（从 route-C captions 多数投票提取），匹配用字符串规范化，不引入外部映射表
- **内存**：只加载 base 样本（9205×~1.1MB≈10GB），GPU 节点 60G RAM eager 安全；登录节点（8G cgroup）绝不跑全量
- **fp32 训练**：V100 (sm_70) 无 bf16 加速，fp16 需 scaler；0.5B fp32 显存/速度均无压力，选稳定优先
- **评测用手动贪心解码**（DynamicCache + 左填充批处理），不用 `generate(inputs_embeds=...)`，规避 transformers 5.x 兼容性风险

## 5. 迭代计划

1. 单测全绿（tiny config，CPU 可跑）
2. 冒烟：normal_test 队列（30min 上限），100 样本 2 epochs——loss 必须下降 + 生成格式正确
3. 全量：gpu_v100，epochs 4 / batch 32 / seed 0，预计 <1h
4. 评测：val base 1968 × 2 条件，产物 `results/sftmvp/eval_mvp.json` + 生成样例 jsonl
5. 差距报告：`docs/reports/sftmvp_mvp_report.md` + STATUS.md 更新 + claim 置 done

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| projector 从零训 + LoRA 联合不稳 | 冒烟先单独看 projector 收敛；必要时前 1 epoch 冻结 LoRA 只训 projector |
| 0.5B 读不出细粒度语义 | 这本身就是有效 NEGATIVE 信号；第二轮升基座（先 A 后 B 两步走的 B） |
| 生成答案不匹配任何类 | match_answer 返回 -1 计错，样例落盘供人工分析 |
| transformers 5.x API 变动 | 单测锁定关键接口（chat 构造/解码循环）；Qwen2 路径为 transformers 核心路径，风险低 |
