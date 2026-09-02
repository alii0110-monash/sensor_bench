# Depth encoder 诊断：模型式前景分离 vs 先验注入（三臂 + 补充臂）

> 背景：`layer_cka_v4.md` §10 显示 depth 浅层探针 ≈ 随机（0.102），但理论上 depth 语义丰富；
> `v5_structfeat` 手写运动统计特征 0.27。
> 问题：不写传统分割/规则，用**模型**（分割模型 / 自监督预训练）能否救活 depth encoder？
> 日期：2026-09-02 · GPU 作业：1059726/1059753（mask 缓存）+ 1059858（三臂）+ 1059902（补充臂）
> 脚本：`scripts/cache_depth_masks.py` · `scripts/depth_arms_experiment.py` · `scripts/depth_arms_followup.py`
> 编码器：`framework/models/depth_vit.py`（ViTDepthEncoder，与 DepthEncoder 同 token 契约）

---

## TL;DR

**三条"模型自己搞定"的路，两条失败一条部分成功**：

1. **模型式前景分离（Mask R-CNN）→ 无效**：mask 技术上干净（人物轮廓贴合、检测率 83%），但 masked 训练 acc 0.054-0.065 ≈ raw 0.063-0.078。**背景不是瓶颈**。
2. **加容量（2-conv → 4 层 ViT）→ 无效**：0.063 → 0.078，且 150 epochs 长训（0.079）排除欠训练。
3. **MAE 自监督先验 → 部分有效**：低 lr（1e-4）微调下 **0.146**，是 from-scratch 的 ~2×；但 lr 1e-3 会把预训练权重打崩（0.059，灾难性遗忘）——**先验有效但脆弱，协议敏感**。
4. **所有学习式方案（最高 0.146）仍远低于手写运动统计特征（0.27）**——depth 的动作判别信号主要在**时序运动计算**（帧差分/轮廓跟踪）里，而非静态外观；这个规模的数据/预算下，模型自己"悟"不出来。

---

## 1. 实验设计

### 1.1 数据

- v4 train 分层子集 2997（label 从 id 的 `A{xx}` 解析，27 类）→ val 1870
- MAE 预训练：同 2997 样本的 14985 帧（无标签）

### 1.2 前景 mask（模型式，零规则）

- **torchvision Mask R-CNN R50-FPN (COCO)**，person=label1，score>0.5，取最高分实例
- depth → 伪 RGB（clip [0,5]m，near=bright）
- 首轮教训：**union 所有实例会被背景假阳性污染**（书架/墙面被认成人，IoU 0.042）；改 top-1 后 mask 干净贴合人体（可视化 `results/plots_v4/mask_quality_panel.png`）
- 最终质量：24335 帧，检测率 **0.832**，与 1-3.5m 带 IoU 0.035（带占 45% 像素，IoU 低是预期的）
- 下载坑：hf-mirror 大文件重定向 us.aws.cdn.hf.co 被墙 → 换 download.pytorch.org（8 路并行 range 下载，178MB/100s）

### 1.3 六臂 + 两补充臂（同协议：CE depth 单模态分类，AdamW，val acc）

| arm | encoder | init | input | epochs | lr |
|---|---|---|---|---|---|
| tiny_raw | DepthEncoder (2 conv) | 从零 | raw | 30 | 1e-3 |
| tiny_masked | DepthEncoder | 从零 | **masked** | 30 | 1e-3 |
| vit_raw | ViTDepthEncoder (4L, 196 patch) | 从零 | raw | 30 | 1e-3 |
| vit_masked | ViTDepthEncoder | 从零 | **masked** | 30 | 1e-3 |
| vit_mae_probe | ViTDepthEncoder | **MAE 预训练**（冻结） | raw | probe only | 1e-3 |
| vit_mae_ft | ViTDepthEncoder | MAE 预训练 | raw | 30 | 1e-3 |
| vit_mae_ft_lowlr | ViTDepthEncoder | MAE 预训练 | raw | 30 | **1e-4** |
| vit_raw_long | ViTDepthEncoder | 从零 | raw | **150** | 1e-3 |

MAE：75% patch mask + learned mask token + 位置编码，重建 per-patch 标准化 depth（loss 0.98→0.30）。

---

## 2. 结果

| arm | val acc | vs 随机 0.037 |
|---|---|---|
| tiny_raw | 0.0626 | 1.7× |
| tiny_masked | 0.0647 | 1.7× |
| vit_raw | 0.0775 | 2.1× |
| vit_masked | 0.0535 | 1.4× |
| vit_mae_probe（冻结） | 0.0947 | 2.6× |
| vit_mae_ft（lr 1e-3） | 0.0594 | 1.6× |
| **vit_mae_ft_lowlr（lr 1e-4）** | **0.1455** | **3.9×** |
| vit_raw_long（150 ep） | 0.0786 | 2.1× |
| 参照：手写运动统计（v5_structfeat 63d） | **0.27** | 7.3× |

---

## 3. 三个假设的判定

| 假设 | 检验 | 判定 |
|---|---|---|
| H1 容量不足 | tiny vs vit（0.063 vs 0.078）+ 150ep（0.079） | **FAIL** |
| H2 背景拖累 | masked vs raw（两架构均无提升，甚至 -0.02） | **FAIL** |
| H3 缺先验 | MAE probe 0.095 / MAE+低lr微调 **0.146** | **PARTIAL WIN**（2×，但协议敏感） |

**H3 协议敏感性**：lr 1e-3 微调把 MAE 权重打崩（0.146 → 0.059）。预训练权重必须配 1e-4 级微调。

---

## 4. 为什么手写特征（0.27）碾压所有学习方案？

v5_structfeat 的 63 维特征里起作用的成分（`feature_extract.py`）：**帧间差分统计（motion_mean/std/max）+ 轮廓 bbox 位置 + 身体深度分布直方图**——都是**显式的时序-几何计算**。

学习式 encoder 拿到的是 5 帧静态 depth 堆叠，理论上能自己学出帧差分，但在 3k 样本 × 30-150 epochs 规模下没有。**结论修正**：depth 的"语义丰富"不在单帧外观（分割再干净也没用），而在**跨帧运动模式**；这个信号需要：(a) 显式给它（手写帧差分/多通道输入），或 (b) 大一个数量级的预训练。

---

## 5. 对 M6/数据管线的建议

1. **短期（最低成本）**：depth 输入加运动通道——`[d_t, d_t - d_{t-1} (×4 帧)]` 5 通道喂 encoder，把手写特征里最有效的帧差分显式化。预期直接对齐手写特征收益。
2. **中斯**：MAE 预训练扩规模（全 46k train 的 ~230k 帧、更长 schedule）+ 低 lr 微调，作为 depth encoder 的标准初始化流程（本次已验证管线可复用：`results/depth_arms_ckpt/vit_mae.pt`）。
3. **跨模态蒸馏**（未做）：rgb 关键点当老师训 depth→keypoints，输出直接对接现有 rgb-keypoints 管线。
4. **mask 不值得**：模型式分割管线已建好（`masks_m2f/`），但精度收益为零——不推荐进入主流程。

---

## 6. 产物清单

- `results/depth_arms_v4.json` · `results/depth_arms_followup.json` · `results/mask_quality.json`
- `results/depth_arms_ckpt/vit_mae.pt`（可复用的 MAE 预训练权重）
- `results/plots_v4/mask_quality_panel.png`（mask 质量可视化）
- `framework/models/depth_vit.py` · `scripts/{cache_depth_masks,depth_arms_experiment,depth_arms_followup,viz_depth_masks}.py`
- `jobs/{depth_masks,depth_arms,depth_arms_followup,gpu_smoke}.slurm`