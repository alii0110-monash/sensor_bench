# Motion 通道接入主流程：负结果报告（2026-09-03）

> 前置链：`depth_foreground_prior_v4.md`（单模态 motion 通道 0.474，6.1×）→ 本实验假设融合模型同样受益
> 作业：1061450（3h52m，gpu11）· 训练 `checkpoints_motion_v4/token_fusion_seed{0,1,2}.pt`
> 评测：`leaderboard_motion_v4.json`（protocol_v5，21 profiles × 3 seeds）
> 对照：`leaderboard_temporal_full.json`（v4 raw temporal，robustness 0.6904 / acc_full 0.9451）

---

## TL;DR

**单模态 6.1× 的增益没有迁移到融合模型——motion 版融合全面小幅劣于基线**：

| 指标 | motion | temporal 基线 | Δ |
|---|---|---|---|
| robustness | 0.6399 | 0.6904 | **-0.050** |
| acc_full | 0.9159 | 0.9451 | -0.029 |
| mean Δ (强模态缺失 profiles) | — | — | **-0.092** |
| mean Δ (weak-only profiles) | — | — | -0.031 |
| mean Δ (其余) | — | — | -0.024 |

劣化集中在 **mmwave/rgb 缺失场景**（miss-mmwave -0.10、miss2-mmwave-rgb -0.086）和 **only-lidar（-0.082）**。

---

## 1. 实验设置

- `token_fusion.py` 新增 `motion_depth` 开关：DepthEncoder → `ViTMotionEncoder`（2ch `[d_t, Δ_t]`，4 层 ViT，temporal 契约兼容，save/load 持久化；`train.py --motion-depth`）
- v4 raw + temporal + eager，batch 32，30 epochs（patience 5），3 seeds 串行
- 各 seed best val：0.820（早停 ep11）/ 0.847（ep14）/ 0.874（ep16）——**均低于基线 acc_full 0.945**

## 2. 根因分析（三个机制）

### 2.1 共享 transformer 扰动（主因，解释 most）

fusion 的 transformer 层是**全模态共享权重**的。depth encoder 从 2 层小 conv 换成 4 层 ViT 后：
- depth token 的分布/范数/信息密度完全改变
- 共享 transformer 的注意力学习动态被整体改变——**连 depth 缺席的 profile（only-lidar -0.08、miss2-mmwave-rgb 走 wifi+lidar -0.086）都劣化**，说明劣化不在 depth 路径本身，而在共享层的训练被"带偏"
- 弱模态（wifi/lidar）受影响最大：它们的 token 本就弱，共享空间被强 depth token 重新校准后更加边缘化

### 2.2 早停 + 收敛动态

motion 版 3 seeds 均在 ep11-16 早停（best 0.82-0.87），基线 acc_full 0.945 说明基线训练到了更高的平台。motion 版的 depth token 过强可能让 CE 更快进入平台，patience=5 提前终止——**容量/时长因素未排除**。

### 2.3 only-depth 依旧 ≈ 随机（0.058，基线 0.056）

单模态 encoder 探针 0.474，但进融合后 only-depth ≈ 随机——**[MISSING]-token 机制在单模态可用时反而淹没 depth 信号**。这是融合框架的独立问题（基线同样存在），与 motion 无关，值得单独修。

---

## 3. 修正后的结论链

1. **信息 > 先验**（上一轮）——在单模态评测下成立
2. **单模态增益 ≠ 融合增益**（本轮新增）——共享 fusion 中换 encoder 不是局部手术，会通过共享 transformer 重塑所有模态的学习
3. depth 单模态 0.474 的正确用法可能不是"换进融合"，而是：**miss-rgb/miss-mmwave 时的独立兜底头**（专家/路由），或经 token 统计对齐（encoder 后 LayerNorm）后再融入

## 4. 下一步选项（按预期收益排序）

1. ~~**token 统计对齐**：ViTMotionEncoder 输出加 LayerNorm~~ → **已验证无效（2026-09-03，job 1062327）**，见 §6
2. **保守变体**：保留 tiny conv 架构只加 diff 通道（2ch conv），token 分布扰动最小
3. **专家路由**：mmwave/rgb 缺失时路由到 motion-depth 专家头（绕过共享 transformer 干扰）
4. **修 only-depth 悖论**：单模态可用时旁路 [MISSING] 机制（对 5 个 only-* profile 是普遍提升点）

## 6. 选项 1 验证：LayerNorm token 对齐——无效（2026-09-03，job 1062327）

**假设**：motion 版劣化源于 ViT depth token 的数值分布（范数/尺度）与其他模态不匹配。

**干预**：`motion_depth_layernorm` 开关（`token_fusion.py` 已合入）——depth token 在 encoder 之后、共享 transformer 之前过 LayerNorm。1 seed 验证（对照 motion seed0 best val 0.820 / 3-seed robustness 0.6399 / 基线 0.6904）：

| 指标 | LN (1 seed) | motion 无 LN | temporal 基线 |
|---|---|---|---|
| best val | 0.837 | 0.820 | — |
| robustness | 0.6352 | 0.6399 (3-seed) | 0.6904 |
| acc_full | 0.9057 | 0.9159 | 0.9451 |
| only-depth | **0.0908** | 0.0580 | 0.0555 |

**判定：无效**。robustness/acc_full 与无 LN 持平（种子方差内），val 上的 +0.017 未兑现到 leaderboard。

**排除的假设**：劣化不是 token **统计**（数值范数/尺度）问题，而是**信息内容**问题——强 motion-depth token 让共享 transformer 在训练时重新分配注意力、边缘化弱模态路径；LayerNorm 抹平数值分布，抹不平信息竞争。唯一亮点：only-depth 0.058→0.091（LN 稍微帮了 depth 自己的路径）。

**选项 1 关闭。** 剩余：保守 tiny conv 变体 / 专家路由 / only-* 悖论修复。

## 5. 产物

- `leaderboard_motion_v4.json` · `checkpoints_motion_v4/`
- `framework/models/{token_fusion,depth_vit}.py`（motion_depth 契约，已合入）· `scripts/train.py --motion-depth`