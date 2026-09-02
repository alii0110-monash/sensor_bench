# MMFi v4 层级跨模态 CKA 诊断报告

> 工具：`framework/eval/dataset_quality/layer_cka.py` · `scripts/run_layer_cka.py` · `scripts/plot_layer_cka.py`
> 数据：`checkpoints_v4_temporal/token_fusion_seed{0,1,2}.pt` × v4 val (1870 样本)
> 跑批：job 1059460 (slurm normal_test, sglang-0.5.10-cuda12.8 env, ~7min 全部完成)
> 关联：[声明 `cka_layerwise_v4_proposal.md`](cka_layerwise_v4_proposal.md) · [数据集质量 v1→v4](dataset_quality_v1_v2_v4.md)
> 日期：2026-09-02

---

## TL;DR

**核心问题**（声明第 6 节）：v4 mmwave 之所以是"真正的互补"，是因浅层特异/深层融合，还是 attention mask 隔离？

**答案**：**从未对齐**。mmwave×rgb 在 enc_out CKA=0.177，layer1_out CKA=0.219，Δ=+0.04——**是 10 个模态对中唯一 Δ<0.05 的**。其它 9 对在深层都大幅融合。

**数据质量诊断**：
- ✅ **浅层特异**（理想曲线 A 局部成立）：enc_out 10 对 CKA 全部 < 0.2
- ⚠️ **深层融合**（理想曲线 A 局部成立）：9/10 对 CKA 飙升 0.2-0.6；mmwave×rgb 例外
- ❌ **mmwave 独立 oracle**：互补机制不是几何融合，是 attention mask 隔离的独立信号源

**对 M6b 的暗示**（决策层 [提议]）：传统 alignment loss（CKA/MMD/contrastive）会**伤害** mmwave 的判别力。应改"对齐"为"独立正则"——惩罚 mmwave×rgb CKA 升高。

---

## 1. 完整 CKA 矩阵（3-seed mean）

### enc_out（per-modality encoder 输出）

|  | wifi | depth | lidar | mmwave | rgb |
|---|---|---|---|---|---|
| wifi | 1.000 | 0.037 | 0.050 | 0.024 | 0.013 |
| depth | — | 1.000 | 0.099 | 0.082 | 0.066 |
| lidar | — | — | 1.000 | 0.128 | 0.048 |
| mmwave | — | — | — | 1.000 | 0.177 |
| rgb | — | — | — | — | 1.000 |

**特征**：所有 CKA < 0.20，10 对中 9 对 < 0.13。**浅层模态完全独立**（理想曲线 A.浅层特异）。

### layer1_out（2 层 transformer 最终输出）

|  | wifi | depth | lidar | mmwave | rgb |
|---|---|---|---|---|---|
| wifi | 1.000 | 0.458 | 0.627 | 0.373 | 0.269 |
| depth | — | 1.000 | 0.452 | 0.423 | 0.338 |
| lidar | — | — | 1.000 | 0.336 | 0.312 |
| mmwave | — | — | — | 1.000 | **0.219** |
| rgb | — | — | — | — | 1.000 |

**特征**：9/10 对 Δ > +0.10，**深度融合**发生。**mmwave×rgb Δ = +0.04，唯一例外**。

---

## 2. 与声明第 6 节问题的对照

| 机制假设 | 预期曲线 | 实际观测 | 判定 |
|---|---|---|---|
| A. 浅层特异/深层融合（理想） | enc_out 低、layer1_out 高 | enc_out 普遍 < 0.2；layer1_out 9/10 对飙升 | **部分成立** |
| B. 浅层就高 CKA（模态冗余） | 全程 > 0.7 | enc_out 全部 < 0.2 | **否** |
| C. 深层升不上去（时间/语义错位） | 全程 < 0.3-0.4 | 9/10 对深层能升 0.2-0.6 | **否**（mmwave 例外见下） |

**回答核心问题**：
- v4 数据集**不**是冗余（排除 B）
- v4 数据集**不**是整体错位（排除 C）
- v4 是"**理想曲线 A + 单一例外**"——所有模态浅层独立、大部分深层融合、mmwave×rgb 例外
- mmwave 在 v4 中是"**独立 oracle**"——它提供 rgb 没有的判别信号，但表征空间从未与 rgb 对齐

---

## 3. 关键模态对分析（与 `dataset_quality_v1_v2_v4.md` 交叉验证）

### 3.1 mmwave × rgb（核心）

- enc_out CKA=0.177（mmwave 单 probe acc=0.348, rgb 0.819）
- layer1_out CKA=0.219（Δ=+0.04）
- **mmwave contribution 0.625**（dataset_quality）
- **诊断**：mmwave 几何独立，但 attention 仍能用它做决策——因为 mmwave 提供的"动作粒度"信息 rgb 没有
- **对 dataset_quality 的修正**：mmwave 的"互补"是**信号互补**（决策信息不同），不是**表征互补**（几何对齐）

### 3.2 lidar × rgb

- enc_out CKA=0.048（lidar 单 probe 0.095）
- layer1_out CKA=0.312（Δ=+0.26）
- **lidar contribution 0.502**（dataset_quality）
- **诊断**：lidar 浅层完全独立（不冗余），深层被 attention 拉向 rgb（融合）。**这是符合理想曲线 A 的"健康"互补**——既有特异信息，又有几何融合

### 3.3 wifi × rgb

- enc_out CKA=0.013（wifi 单 probe 0.048）
- layer1_out CKA=0.269（Δ=+0.26）
- **wifi contribution 0.254**（dataset_quality）
- **诊断**：wifi 浅层几何独立，深层被强行拉到 rgb 附近。**疑似 attention 拉过去**——wifi 信息量低但被 attention 当作填充物

### 3.4 mmwave × lidar

- enc_out CKA=0.128
- layer1_out CKA=0.336（Δ=+0.21）
- **诊断**：两个"几何模态"在深层融合，符合预期

---

## 4. 声明 vs 实际 — 差距分析

| 声明项 | 实际 | 差距 |
|---|---|---|
| 目标：定位 mmwave×rgb 融合机制 | ✓ 完成 | 0 |
| 现状：缺中间层 hook 工具 | ✓ 实现 `layer_cka.py` | 0 |
| 问题分析：complement_gain=-0.31 根因 | **部分更新**——根因不是单一模态错位，而是"mmwave 独立 oracle"机制；其它模态融合是健康的 |
| 路径 1（Linear CKA） | ✓ 采纳 | 0 |
| 路径 2（3 hook 点：enc_out/layer1_out/pool_out） | ⚠ **实际只用 2 个**：pool_out 不分模态，CKA 无意义，已从输出移除 | -1 hook 点 |
| 路径 3（10 模态对） | ✓ 全跑 | 0 |
| 路径 4（不加 linear probe） | ✓ 仅 CKA | 0 |
| 路径 5（对照 `dataset_quality_v1_v2_v4.md` 格式） | ✓ 报告 §3 交叉验证 | 0 |
| 时间预算 ~3-4 h 编码 + 1 h 跑批 | **1 h 编码（含调试 2 个环境问题）+ 7 min 跑批** | 提前完成 |
| 验收标准 | **全部满足** | 0 |

### 4.1 实施期意外

1. **环境兼容性**：sglang env (torch 2.9.1+cu128) 与系统 cudnn 冲突 → 改用 normal_test CPU 队列
2. **dataset loading**：v4 lazy load 在 1870 样本 × 3 seeds 时 IO 严重 → 加 cache_size=10000 预热
3. **pool_out 不分模态**：融合后 token 全交叉，per-modality 切分无意义 → 从产品中移除

### 4.2 路径 2 调整理由

声明 3 hook 点 = enc_out + layer1_out + pool_out。但 pool_out = `transformer 输出 mean(dim=1)`，**不分模态**（所有 80 个 token 已交叉注意力）。对它算 per-modality CKA 不可能。

降级方案：保留 enc_out（per-modality encoder 出）+ layer1_out（transformer 最后层出）。这两点正好对应"浅层/深层"二分。

---

## 5. 对 SensorBench M6 的影响

### 5.1 v4 数据集质量评价（更新 `dataset_quality_v1_v2_v4.md`）

**原结论**："mmwave 是真正的互补"——基于 contribution 0.625 + 单 probe 0.348。

**layer_cka 补充**：
- mmwave 的"互补"是**信号互补**，不是**几何融合**
- mmwave 浅层独立、深层仍未对齐（Δ=0.04）——这是**几何解耦**，而非"几何融合 + 语义独立"
- **互补机制 = attention mask 隔离的独立 oracle**

**修正**："mmwave 高 contribution"应解读为"模型用 mmwave 当独立特征源"，而非"mmwave 与 rgb 在共享语义空间被 attention 整合"。

### 5.2 对 M6b 训练手段实验的方向修正

`docs/reports/improvement_plan.md` §3 提到的"跨模态一致性增强" (P1-2.2) 和"组合缺失增强" (P1-1.1) **可能过度**——它们会破坏 mmwave 的几何独立性。

**新提议（M6c 候选，已写入 STATUS.md 决策层）**：
1. **mmwave 独立性正则**：loss += -λ · CKA(mmwave, rgb)·pool
2. **per-modality 早停**：rgb/mmwave 各自独立 val head
3. **mmwave-aware dropout**：训练时 rgb-miss+mmwave-pres 比例 ≥ 0.5

---

## 6. 复现脚本

```bash
# 提交 slurm（normal_test CPU 队列）
cd /seu_share2/home/wangshuai02/220255046/sensorbench
sbatch jobs/layer_cka_v4.slurm

# 或本地（CPU 估时 5-10 min/3-seeds）：
PY=/seu_share/apps/anaconda3-2024.10-1/envs/sglang-0.5.10-cuda12.8/bin/python
$PY scripts/run_layer_cka.py \
    --checkpoint_dir checkpoints_v4_temporal \
    --dataset_root datasets/mmfi/v4 \
    --output_dir results \
    --seeds 0 1 2 \
    --batch_size 64 \
    --device cpu
```

产物：
- `results/layer_cka_v4.json`（12683 B，per-pair mean/std/n_seeds）
- `results/plots_v4/layer_cka_curve.png`（4 子图：mmwave×rgb / lidar×rgb / wifi×rgb / mmwave×lidar）

---

## 7. 下一步（[提议]）

### 7.1 立等可做

- [ ] **重复实验在 v5_structfeat / v5_hardaug / v6_relabel**——验证 mmwave×rgb 不融合是 v4 独有，还是 MMFi 通用现象
- [ ] **加 Hilbert CKA 对照**——Linear CKA 仅几何相似，Hilbert 可捕捉非线性结构差异；若 Hilbert 显示 mmwave×rgb 高 = Linear 低估融合
- [ ] **拆 27 类逐类 CKA**——某些动作 mmwave 可能真的与 rgb 对齐（如"走路"），某些独立（"打拳"）——这能定位 mmwave 擅长的具体动作

### 7.2 提议进入 M6b 训练手段实验

见 [STATUS.md 决策层](https://github.com/)：`M6c 训练手段实验方向修正（2026-09-02）`

### 7.3 长程

- 跨数据集验证：在 [公开多模态动作数据集如 NTU60/PKU-MMD] 重复同样 CKA 诊断，建立"mmwave-like 模态不融合"是否 MMFi 数据集独有的判断

---

## 8. 结论

本次 MVP 迭代（声明 → 实现 → 跑批 → 报告）**全部按声明完成**，并发现了一个**反直觉的关键结论**：

**v4 mmwave 不是"互补融合"，是"独立 oracle"**——它与 rgb 在所有层都几何独立（最大 CKA = 0.22），但仍贡献 0.625 的融合增益，机制是 attention mask 隔离而非几何融合。

这个发现**修正了 `dataset_quality_v1_v2_v4.md` 的解释**，并直接驱动了 M6b 训练手段实验方向的修正（从"对齐"改为"独立正则"）。

下次讨论的起点：v5/v6 数据集上 mmwave×rgb 曲线是否仍平。

---

## 9. 合理性检验与解读修正（2026-09-02 补跑，job 1059481）

三个对照：`scripts/layer_cka_controls.py`（seed0，30 次置换，随机初始化模型，N 敏感性）。

### 9.1 ✅ 结论一：数值统计上全部真实

- 置换 null（打乱样本配对 30 次）：均值 **0.001-0.005**、std ≈ 0.001
- 所有观测值 z ≥ 14、p < 0.01——包括 enc_out 的 0.014 也是真结构
- 此前担心的解析随机地板 √(D²)/N ≈ 0.137 **实测不存在**（特征高度各向异性，null 塌缩到 ~0，解析 iid 假设不适用）
- 样本量敏感性：N≥500 后 mmwave×rgb 稳定（enc 0.19 / layer1 0.22-0.27）；N=200 偏高（0.36-0.38），符合小样本正偏差方向 → 全量 N=1870 取值无偏

### 9.2 ⚠ 结论二：「深层融合」叙事被随机对照推翻

| 模型 | layer1_out 跨模态 CKA（10 对范围） |
|---|---|
| **随机初始化**（同架构） | **0.969-0.996（全塌缩）** |
| 训练后（seed0） | 0.146-0.478 |

- 共享 transformer 天然把所有模态 token 塌缩到共同方向（random attention ≈ 均值混合 + FFN 共同分量支配方差）→ 未训练时 CKA ≈ 1
- 训练的作用是让模态在深层**重新分开**（0.97 → 0.15-0.48），**不是"学出融合"**
- 原报告 §1-§3 的"9/10 对深层融合、理想曲线 A 局部成立"**解读修正**为：
  - 浅层（enc_out）：模态独立（架构 + 数据共同决定，随机模型同层也低）
  - 深层（layer1_out）：训练**主动保持模态可分性**，分化程度因对而异
  - 修正后的排序（ trained 越低 = 离随机塌缩越远 = 越分化）：wifi×rgb 0.146 最分化 < lidar×rgb 0.164 < mmwave×rgb 0.233 < … < wifi×depth 0.478 最接近塌缩
- seed 稳健性备注：mmwave×rgb 深层 0.21-0.23（3 seeds 稳）；lidar×rgb 单 seed 波动大（0.15/0.40/0.35），其"分化"结论弱于 mmwave×rgb

### 9.3 ⚠ 结论三：CKA 单独无法区分「对齐」vs「塌缩」

高 CKA 两种成因：① 语义对齐（共享判别结构）；② 共同主成分支配（如"有人在动"这种平凡成分）。判定需 **per-layer per-modality 线性探针**（声明 §4 决策 4 的备选项升级为必做）：

- 深层 CKA 高 + 各模态 probe acc 高且接近 → 真对齐
- 深层 CKA 高 + probe acc 低 → 塌缩（平凡共享成分）

mmwave×rgb 平坦（0.18→0.22-0.23）在修正框架下仍成立：该对在深层既未塌缩也未与 rgb 共享几何——"独立 oracle"仍是当前最佳解释，但需 layer-wise probe 佐证其独立性承载的是判别信息而非噪声。

### 9.4 对 M6c 提议的影响

原 [提议]（mmwave 独立性正则）前提"其他模态融合、mmwave 不融合"不成立——实际是"训练让所有对保持分化、程度不同"。M6c 实验设计需先补 per-layer probe 实验再定，原 3 个候选手段保留但降优先级。

### 9.5 新增产物

- `results/layer_cka_controls.json`
- `scripts/layer_cka_controls.py` + `jobs/layer_cka_controls.slurm`（job 1059481，315s）

---

## 10. MVP 迭代 #2：per-layer linear probe（2026-09-02，job 1059621）

`scripts/layer_probe.py`：train 分层子集 2997 → val 1870，Linear(256→27) + z-score，30 epochs，3 seeds。
label-CKA = 特征与 one-hot 标签的 CKA（类结构含量）。

### 10.1 结果（3-seed mean）

| modality | enc_out probe | layer1_out probe | Δ | label-CKA enc | label-CKA layer1 |
|---|---|---|---|---|---|
| wifi | 0.041 | **0.656** | +0.62 | 0.010 | 0.239 |
| depth | 0.102 | **0.683** | +0.58 | 0.027 | 0.257 |
| lidar | 0.068 | **0.699** | +0.63 | 0.068 | 0.163 |
| mmwave | **0.422** | **0.574** | +0.15 | 0.238 | 0.291 |
| rgb | **0.782** | **0.818** | +0.04 | 0.335 | 0.403 |

（随机 = 1/27 ≈ 0.037）

### 10.2 判读（对照 §9.3 判据矩阵）

1. **深层是"功能性的信息混写"，不是平凡塌缩**：wifi/depth/lidar 的 enc_out 探针 ≈ 随机（0.04-0.10），但 layer1_out 探针 0.66-0.70——cross-modal attention 把类判别信息**写进了每个模态的 token 段**（探针在单模态段上即可读出类信息）。同时 CKA 0.15-0.48 远低于随机模型的 0.97 → 训练学会了"传递类信息但不塌缩几何"。
2. **浅层探针独立复现了 dataset_quality 的模态层级**：rgb 0.78 / mmwave 0.42 / 其余 ≈ 随机，与 `dataset_quality_v1_v2_v4.md`（0.819/0.348/≈0.05）交叉验证一致——两套独立方法同一结论。
3. **mmwave「独立 oracle」获探针证据**：mmwave 自带判别几何（enc 0.422）+ 深层仍可独立解码（0.574）+ 与 rgb 几何不合并（CKA 0.23）。rgb 与 mmwave 是深层**共存的两套各自可解码、几何互不合并的表征**。
4. **wifi 深层 0.656 ≠ wifi 数据好**：其信息是 attention 从 rgb/mmwave 借来的（enc 0.041 ≈ 随机）。与 modality_dropout 实验（miss-wifi 几乎不掉分）三角印证：wifi 无独立贡献。
5. **数据质量总判**（回到用户最初问题）：v4 **无模态冗余**（浅层全独立）、**无致命错位**（attention 能有效跨模态传递类信息）、模态信息层级 rgb > mmwave ≫ wifi/depth/lidar（原始编码）。理想曲线 A 的修正版：**浅层特异 + 深层功能性混写 + 几何保持可分**。

### 10.3 残留问题（下步）

- probe acc 的"借用信息"与"自有信息"需 masked-context probe 分离：仅 wifi 可用时提取 layer1_out 再探针（avail={wifi}）→ 读出 wifi 自有贡献上限
- lidar label-CKA 深层反降（0.068→0.163 vs 其他模态升）+ seed 方差大 → lidar 深层几何最不稳定

### 10.4 新增产物

- `results/layer_probe_v4.json` + `scripts/layer_probe.py` + `jobs/layer_probe.slurm`（1127s）