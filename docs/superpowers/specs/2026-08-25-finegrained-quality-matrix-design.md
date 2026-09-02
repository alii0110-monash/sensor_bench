# 细粒度数据集质量矩阵（Fine-Grained Quality Matrix）

> 日期：2026-08-25
> 目标：让数据集评价从"3 个全局总分"细化为**可跨版本追踪的 (类别×受试者) 质量分矩阵**。

## 背景

当前数据集评价（`run_dataset_quality.py`）产出 3 个全局评分：
- **InfoScore**（信息量，权重 0.4）
- **CompactScore**（紧致度，权重 0.4）
- **CleanScore**（纯净度，权重 0.2）

合成 `quality` 总分。这些是**数据集级**的宏观健康度，无法回答"哪个类别、哪个受试者的数据质量差"。

**动机**：主流程 temporal 评测暴露短板——`miss2-mmwave-rgb` 崩到 0.147、weak 模态（wifi/depth/lidar）单模几乎随机。需要细粒度定位病灶，支撑数据改进闭环（v4→v5→...）的跨版本追踪。

## 一、范围与目标

**目标**：为每个 (类别 × 受试者) 子组计算一个可追踪的质量分（0~1），供跨版本对比数据改进效果。

**粒度**：类别×受试者为主（27 类 × 23 受试者 = 621 格），环境作为辅助维度。

**数据来源**：subject/env/action **从样本 id 解析**（`E01_S01_A01_f1-7` → env=E01/subject=S01/action=A01），**不从 meta 读**——变体样本（`__aug`）的 meta 是 `None`。解析规则见 `framework/dataset/splits.py::_sample_id`。

**变体处理**：v4 train 46509 样本中 36820 个是 `__aug` 变体（79%，仅存 rgb delta，subject/env/label 同 base）。**细粒度分组必须排除变体，只用 base 样本**——否则近重复变体会让一致性/可分性信号失真、格子计数虚高。**621 格按非变体 base 统计（每格 5-61 样本）。**

**信号**：多信号合成（主模型识别 + 类内一致性 + 类间可分性）。

**产出**：结构化 JSON（`results/quality_matrix_v{version}.json`），供程序化对比/追踪。

**放置**：扩展 `framework/eval/dataset_quality/`，独立于现有三维评分。

## 二、架构

新增模块 `framework/eval/dataset_quality/finegrained.py`，复用现有：
- `feature_extract.extract_structured_feature` / `modality_probe.extract_modality_feature_downsampled`（特征提取，方案 A 用原始数据特征）
- 主模型 `TokenFusionModel.predict_batch`（识别）

```
输入: 数据集 (datasets/mmfi/v4) + 主模型 checkpoint
        │  样本 id 解析: subject / env / action   (排除 __aug 变体)
        ▼
按 (类别, 受试者) 分组样本 (621 格)
        ▼
每格算 3 信号:
   ├─ 信号1: 主模型识别 (main_acc + conf)
   ├─ 信号2: 类内一致性 (该受试者 vs 同类其他受试者距离)
   └─ 信号3: 类间可分性 (该格 vs 其他类距离)
        ▼
加权合成 → 每格质量分 (0~1)
        ▼
输出 JSON: quality_matrix_{version}.json
```

## 三、信号定义（每格）

对每个格子 `(class=A01, subject=S01)`，样本全部为该受试者的 A01 类动作。**以下所有信号（含一致性/可分性）都只用 base 非变体样本计算，排除 `__aug` 变体**（见 §一）。

**信号 1：主模型识别**
- 主模型 `predict_batch`，available = 全部模态
- `main_acc` = 正确识别为 A01 的比例
- `conf` = 预测为 A01 的平均置信度
- 反映：模型能否识别该受试者的该类动作

**信号 2：类内一致性**
- 该格样本特征 vs 所有其他受试者同类(A01)样本特征的距离
- 高 = 该受试者的动作形态与同类一致（无异常形态/标签问题）
- `consistency = similarity`（距离转相似度）

**信号 3：类间可分性**
- 该格样本特征 vs 其他 26 类的距离
- 衡量该受试者的 A01 是否容易被误判为其他类
- `separability`：**用质心欧氏距离/类内离散度**（`between/within` 比），sigmoid 归一化到 0-1。
  - `between` = 该格质心到其他类质心的欧氏距离；`within` = 该格样本到自身质心的平均距离 + 1e-8
  - `separability = sigmoid(log(between/within))`
  - **实现注记**：原计划用 `compute_fisher_ratio`，但实测在 8980 维原始特征下类间/类内协方差比塌缩到 ~0（无区分度）。改用质心欧氏距离度量更稳健（验证：可分样本 separability > 重叠样本）。

**归一化**：三信号量纲不一（main_acc 0-1、consistency 相似度 0-1、separability 无界），合成前**统一归一化到 0-1**：
- main_acc / conf 天然 0-1
- consistency：相似度已 0-1
- separability：用 `sigmoid(log(between/within))` 归一化到 0-1

**合成**：`quality = w1·main_acc + w2·consistency + w3·separability`，默认 `(0.4, 0.3, 0.3)`，可配置。

## 四、输出 JSON 结构

```json
{
  "dataset": "datasets/mmfi/v4",
  "version": "v4",
  "metadata": {
    "dataset": "datasets/mmfi/v4", "version": "v4",
    "eval_split": "train", "signals": {"main": 0.4, "consistency": 0.3, "separability": 0.3},
    "generated": "2026-08-25T12:00:00"
  },
  "global": {
    "quality": 0.72,
    "per_class": {"A01": 0.85, "A02": 0.67, ...},
    "per_subject": {"S01": 0.75, ...},
    "per_env": {"E01": 0.70, ...}
  },
  "matrix": {
    "A01_S01": {
      "n": 65, "main_acc": 0.80, "conf": 0.90,
      "consistency": 0.70, "separability": 0.75,
      "quality": 0.77, "low_confidence": false
    }
  },
  "low_quality": ["A14_S03", ...]
}
```

## 五、跨版本追踪

- 命名：`results/quality_matrix_v{version}.json`，`version` 从 `meta.json` 取（v4/v5...），与现有 `quality_v*.json` 一致。
- 跨版本对比：`tools/compare_quality_matrix.py --base v4 --new v5 --out diff.json`
  - 相同 `(class, subject)` 格求质量分差值
  - 输出：总改进量、top 提升格、top 恶化格、低质量格改善情况

## 六、错误处理

- 格子样本数 < 阈值（默认 5，**实际用 base 非变体后每格 5-61，阈值取 5 会标满**，建议取 3）→ `low_confidence: true`，不参与全局聚合
- 主模型 checkpoint 缺失 → 降级为 probe 单信号（仅 main_acc 用 probe 替代）
- 特征提取失败（NaN/空）→ 复用 `_safe` guard，置为 0

## 七、测试

新增 `tests/test_dataset_quality/test_finegrained.py`：
- 分组：构造含 noise/错标签样本，验证"低质量格子分数更低"
- 单元：分组、信号计算、加权合成、JSON 结构
- 冒烟：在 v4 小样本上跑通完整流程

## 八、CLI

```
python scripts/run_finegrained.py \
  --dataset datasets/mmfi/v4 \
  --ckpt checkpoints_v4_temporal/token_fusion_seed0.pt \
  --out results/quality_matrix_v4.json \
  [--eval-split train] [--top-k 20] [--w-main 0.4] [--w-consistency 0.3] [--w-separability 0.3]
```

## 九、关键文件

- 新：`framework/eval/dataset_quality/finegrained.py`
- 新：`scripts/run_finegrained.py`
- 新：`tools/compare_quality_matrix.py`
- 新：`tests/test_dataset_quality/test_finegrained.py`
- 复用：`feature_extract.py`、`compactness.py`、`token_fusion.py`、`modality_probe.py`
