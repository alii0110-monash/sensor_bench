# 黄金子集 v2 实证报告

> 工具：`scripts/evaluate_gold_subset.py` → `results/gold_v2_evaluation.json`
> 评估数据：v4 val 1870 样本 → gold_v2 子集 85 样本
> 评估模型：token_fusion seed0（v4 ckpt）、MLP-rgb-probe seed0/1、MLP-mmwave-probe seed0

## 检验 A：高置信度证明

**问题**：gold_v2 是否真的"高置信"，还是只是采样伪影？
**判据**：如果 gold_v2 是真正高置信子集，**所有**模型在它上面的 acc 应显著 > full val acc。

| 模型 | full val acc | gold_v2 acc | gap |
|---|---|---|---|
| token_fusion | 0.808 | **1.000** | +0.192 |
| rgb-probe (seed 0) | 0.809 | **0.988** | +0.180 |
| mmwave-probe (seed 0) | 0.372 | **0.776** | +0.404 |

**结论**：
- 三个模型在 gold_v2 上 acc **全部大幅 > full val**（gap 0.18-0.40）
- mmwave probe gap 巨大（0.40）尤其有意义：mmwave 在 full val 上仅 37%，但在 gold_v2 上 **78%**——这是真正可信的 ground truth 信号，证明 mmwave 在 gold_v2 样本上确实含有充分信息
- **gold_v2 是合格的高置信 ground truth 控制变量**

## 检验 C：seed 敏感性（可重复性）

**问题**：A 中的结果是 single-seed 偶然吗？
**判据**：换 seed 1 重训 rgb-probe，gap 是否仍然存在。

| 配置 | full val acc | gold_v2 acc | gap |
|---|---|---|---|
| rgb-probe seed 0 | 0.809 | 0.988 | +0.180 |
| rgb-probe seed 1 | 0.812 | 0.953 | +0.141 |

**结论**：
- 两个 seed 在 full val 上 acc 差 < 0.005（稳定）
- 两个 seed 在 gold_v2 上 acc 都 > 0.95（gap 显著）
- **gold_v2 信号可重复，不是单次实验产物**

## 检验 B：困难类对齐（难度真实性）

**问题**：v2 中 4 个 0 gold 样本的类（9/12/14/22）——这些类**真的更难**，还是 v2 构造的偶然？
**判据**：这些类在 full val 上的 token_fusion acc 是否也最低。

| 组别 | 类 | full val acc (token_fusion) |
|---|---|---|
| 0-gold classes | 9, 12, 14, 22 | **0.463 (平均)** |
| 5-gold classes (14 类) | 0, 1, 2, 5, 6, 7, 11, 15, 16, 17, 19, 20, 25, 26 | **0.874 (平均)** |
| gap | — | **-0.411** |

**结论**：
- 0-gold 类在 full val 上 acc **0.46 ≪ 5-gold 类 0.87**（差 0.41）
- **"难度"是数据真实属性**，gold v2 准确捕获了这一点
- 类 9/12/14/22 应当是 v5 数据改进的**优先目标**——增样本、加标注、或检查是否标注错误

## 综合结论

**gold v2 满足控制变量的三项核心要求**：

1. ✅ **高置信**——所有模型 acc gap +0.18-+0.40
2. ✅ **可重复**——seed 1 验证 gap 不消失
3. ✅ **捕获真实难度**——0-gold 类在 full val 上也确实最难（gap -0.41）

**用法确立**：
- **未来数据集改进 v5/v6**：在 gold_v2 上重新跑 token_fusion/rgb-probe → 如果 acc 上升说明数据真改进；acc 不变说明改进未触及困难样本
- **新模型架构**：在 gold_v2 与 full val 上对比 acc 差——差大说明新架构只在 easy 样本有效，差小说明架构泛化强
- **归因控制变量**：任何"改进"的复现，都应在 gold_v2 上同时验证

**衍生洞察**：类 9/12/14/22 是 v5 数据飞轮的明确靶点——4 类 acc 0.46，远低于其他类 0.87。

## 产物索引

| 文件 | 内容 |
|---|---|
| `results/gold_subset_v1.json` | v1 双模型共识，129 样本 |
| `results/gold_subset_v2.json` | v2 三模型共识，85 样本 |
| `results/gold_v2_evaluation.json` | 实证结果（A/B/C 三个检验） |
| `scripts/build_gold_subset.py` | v1 构造脚本 |
| `scripts/build_gold_subset_v2.py` | v2 构造脚本 |
| `scripts/evaluate_gold_subset.py` | 实证脚本（A/B/C） |