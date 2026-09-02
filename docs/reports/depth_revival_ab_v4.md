# Depth 振兴路线 A/B 迭代报告（2026-09-02）

> 前置：`depth_foreground_prior_v4.md` 结论——depth 信号在跨帧运动与语义中间表示；
> 业内调研：NTU SOTA 全是 skeleton-based（语义中间表示统治级范式）。
> 路线 A：rgb关键点→depth 蒸馏（STATUS `[提议]` 路线 A）· 路线 B：T=16 重 ingest（`[提议]` 路线 B）
> 作业：1060087（A 蒸馏）· 1060121（B' 运动通道）· 脚本 `scripts/distill_depth_route_a.py` · `scripts/depth_motion_channels.py`

---

## TL;DR

- **路线 A（对比蒸馏）大捷**：rgb-keypoints 当老师、InfoNCE 对齐后 depth encoder
  - 冻结探针 **0.133**（纯 MAE 0.095 的 1.4×）
  - 低 lr 微调 **0.223**（纯 MAE 0.146 的 **1.53×**、从零 0.078 的 **2.9×**）
  - **逼近手写特征 0.27**——学习式方案首次与手写工程同量级，且无需任何手工规则
- **路线 B（T=16 重 ingest）阻塞**：原始 82G tar 解压时已损坏（`exp/mmfi_extract3.log` "Unexpected EOF"）且已删除；无可靠再获取渠道
- **路线 B'（T=5 运动通道）作为抢救方案执行中**：`[d_0, 4×帧差]` 5 通道输入

---

## 1. 路线 A：对比蒸馏（协议修正说明）

STATUS 原提案是"depth→关键点回归"（需外参投影）。侦察发现：v4 的 rgb 是**归一化 2D 关键点**（±1），原始像素坐标与标定已不可得 → **降级为对比蒸馏**，无需任何标定：

- **Teacher**：rgb keypoints (17,2) → MLP → 256d（先 CE 训练，帧级 sanity acc 0.604；样本级 rgb 参照 0.78）
- **Student**：depth (1,224,224) → ViTDepthEncoder（**MAE 预训练权重初始化**）→ 256d
- **InfoNCE**（温度 0.1）：正样本 = 同帧 rgb↔depth 配对；负样本 = batch 内其他帧；投影头 128d
- 蒸馏 30 epochs（lr 1e-3）→ 冻结探针 / CE 低 lr 微调（lr 1e-4，样本级协议与 depth_arms 一致）

### 结果

| 阶段 | val acc | 参照 |
|---|---|---|
| teacher 帧级 sanity | 0.604 | rgb 样本级 0.78（layer_probe） |
| **distill_probe**（冻结） | **0.133** | vit_mae_probe 0.095（+40%） |
| **distill_ft**（低 lr 微调） | **0.223** | vit_mae_ft_lowlr 0.146（**+53%**）· 从零 0.078（2.9×）· 手写 0.27 |

### 解读

1. **语义中间表示假设在 MMFi 上成立**：rgb 骨架的判别结构可以被蒸馏进 depth encoder——这正是业内"depth→skeleton"范式的无标定版本
2. **MAE 先验 + 蒸馏叠加生效**：0.078（从零）→ 0.146（+MAE）→ 0.223（+MAE+蒸馏）——两级先验可加
3. 与手写特征 0.27 的剩余差距（0.223 vs 0.27）：手写特征含显式运动统计（帧差分），蒸馏老师（静态关键点）不含运动信息——**下一级叠加：B' 运动通道**

---

## 2. 路线 B：阻塞记录

- 原始 82G tar 解压损坏（`exp/mmfi_extract3.log`：`Unexpected EOF in archive`）且已删除
- `~/MMFi_dataset/` 已空；MMFi 官方分发走 GitHub（Google Drive/百度网盘），集群无可靠再获取路径
- **结论**：T=16/32 重 ingest 无源可用。若未来重获原始数据，方案见 STATUS `[提议]` 路线 B 原文

## 3. 路线 B'：T=5 运动通道（抢救方案）

帧差分不依赖长时序——depth 输入 1ch → 5ch `[d_0, d_1-d_0, d_2-d_1, d_3-d_2, d_4-d_3]`（DMM 思路）。

- 臂：vit_motion_raw（从零，与 vit_raw 0.078 同协议对照）
- 结果：**见 `results/depth_motion_channels.json`**（作业 1060121）
- 若有效（≥0.15）：与蒸馏正交，可叠加（motion 通道 + MAE init + 蒸馏）

---

## 4. 对 M6 的建议（下一步）

1. **主推组合拳**：depth 输入 = 运动通道（B'）× encoder = MAE init（已验证）× 训练 = 蒸馏对齐（A）——三者正交，预期逐级叠加
2. 蒸馏管线可复用：teacher 换 mmwave（高阶互补）或 wifi，任意弱模态都能"借"强模态的判别结构
3. 蒸馏后的 depth encoder 接回 token_fusion 主流程重训（当前仅单模态验证）

## 5. 产物

- `results/distill_route_a.json` · `results/depth_motion_channels.json`
- `scripts/distill_depth_route_a.py` · `scripts/depth_motion_channels.py`
- `jobs/distill_route_a.slurm` · `jobs/depth_motion.slurm`