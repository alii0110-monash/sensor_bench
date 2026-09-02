# 数据集质量评测系统（Dataset Quality Eval）设计

- 日期: 2026-08-17
- 状态: 设计定稿
- 前置: M1-M6a 完成。M5c / M6b 双负结果——LLM/CLIP 检索评测被"数据集 + 模板 + 编码器"三者耦合污染，无法独立判定"数据集是否真的好"。本迭代把评测主体从"下游任务性能"切换为"数据集固有属性"，用轻量 Linear-probe 完成。

## 背景与目标

**最终目标**：构建一个好的多模态数据集。判据有三条——
1. **信息量大**：每模态独立判别力强，模态间互补而非冗余
2. **类内紧致 / 类间分离**：同类样本聚集，不同类远离
3. **纯净无瑕**：无错配、无重复、跨模态语义一致

**当前痛点**：现有评测（robustness leaderboard + L1 retrieval）都把"数据集 + 模型"耦合，无法独立判定数据本身的好坏。`v4 leaderboard` 里 wifi/depth/lidar only-X ≈ 随机 0.03，**未必**是数据烂——也可能是被 rgb 掩盖后分类器学不动；而 L1 retrieval 在 CLIP/llama2 双侧都跑不出信号，又分不清是模板烂还是编码器烂。

**本迭代目标**：建立独立的数据集质量评测系统，三个维度量化数据固有属性，与下游任务/编码器/文本模板解耦。同时输出对外 leaderboard + 对内诊断报告，服务"数据飞轮"迭代。

## 已确认的关键决策

| 决策点 | 选择 |
|---|---|
| 评测对象 | **数据集本身固有属性**，不是"数据集 + 模型"耦合性能 |
| 评测主体 | **轻量训练 probe**：`per-modality Linear` (×5) + `Concat-Linear` (×1) |
| 三个维度 | 信息量 InfoScore / 紧致度 CompactScore / 纯净度 CleanScore |
| 数据流 | **严格隔离**：probe 只用 train 训、val 评；**test split 全程不进入 probe 评估**（留给下游任务） |
| 辅助指标 | Fisher ratio / 90 分位留一距 / per-sample 异常分数 → **仅诊断，不进聚合分数** |
| 分数范围 | 全部钳位到 [0, 1]；InfoScore 用加权 + clamp 防止越界 |
| 跨模态一致性 | **JS 散度**（对称），阈值化后算 `inconsistency_rate` |
| 重复检测 | **离散化 + hash**（特征四舍五入到固定小数位），不是 raw mean |
| 总分 quality | **加权合成**，`w_info=0.4, w_compact=0.4, w_clean=0.2`（CLI 可覆写） |
| 超参可复现 | 全部阈值 / 权重 / val 样本数 / num_classes 写进 JSON metadata |
| 与 L1 关系 | **互补不替代**：高质量数据集应在 L1 下游表现更好；若不满足说明下游瓶颈 |

## 架构

```
framework/eval/dataset_quality/
  ├── modality_probe.py     # 维度1: 每模态 Linear + Concat-Linear
  ├── compactness.py        # 维度2: Fisher ratio + 混淆矩阵 + 留一距
  ├── cleanliness.py        # 维度3: 异常样本 + JS 一致性 + dup_rate
  ├── report.py             # 报告聚合 + 诊断图 (matplotlib 文本/直方图/热图)
  └── leaderboard.py        # 跨版本聚合 + 排序 + 渲染 markdown
scripts/
  └── run_dataset_quality.py  # 主入口: --dataset X --out results/quality_X.json
results/
  ├── quality_v1.json
  ├── quality_v2.json
  └── quality_v4.json
docs/reports/
  └── dataset_quality_v1_v2_v4.md  # 跨版本数据质量演变结论
```

依赖：`framework/dataset/loader.py`、`torch.nn.Linear`、`matplotlib`。

## 维度 1：信息量 InfoScore

**目标**：每模态独立判别力 + 模态互补增益。

### 算法

```python
# 训练 5 个 per-modality Linear + 1 个 Concat-Linear
for m in modalities:                  # 5 个
    probe[m] = Linear(m_data_dim, num_classes).fit(train_x[m], train_y)
probe["concat"] = Linear(Σ m_data_dim, num_classes).fit(train_concat, train_y)

# 评估（全部在 val split）
acc_per_modality[m] = top1(probe[m].predict(val_x[m]))        # 单模态
acc_concat          = top1(probe["concat"].predict(val_concat)) # 全模态
complement_gain     = acc_concat - max(acc_per_modality)       # 互补增益
```

### 分数（钳位 + 加权）

```python
InfoScore = 0.7 * mean(acc_per_modality) + 0.3 * clamp(complement_gain, 0, 1 - mean(acc_per_modality))
```

- `mean(acc_per_modality)` 是基础分（单模态判别力）
- `complement_gain` 是加分项，**负值 clip 到 0**（模态冲突不倒扣基础分）
- 加权系数 0.7/0.3 保证 InfoScore ∈ [0, 1]

### 关键约束

- **缺失模态处理**：val 样本中某模态缺失时，对应 acc_per_modality 用 0 向量填充 Linear（baseline 处理，spec 不强制 mask）
- **未启用 dropout**：probe 训练纯监督，与下游 robustness profile 的"丢模态评估"解耦
- **批量大小、epochs、lr 写入 metadata**

### 诊断输出

- `acc_per_modality`: dict（5 模态原始 acc）
- `complement_gain`: 原始值（可能为负）
- `mean(acc_per_modality)`: 原始值

## 维度 2：紧致度 CompactScore

**目标**：类内紧致 + 类间分离的可度量版本。

### 主分数（聚合）

```python
# 用 Concat-Linear 的 val 预测计算混淆矩阵
confusion_matrix = ConfusionMatrix(val_y, probe["concat"].predict(val_x_concat))
confusion_rate   = (confusion_matrix.sum() - trace(confusion_matrix)) / confusion_matrix.sum()
CompactScore = 1 - confusion_rate    # ∈ [0, 1]
```

### 诊断输出（不参与聚合）

- **Fisher ratio**：`tr(S_b) / tr(S_w)`，S_b = 类间协方差，S_w = 类内协方差（用 concat Linear 中间特征或 per-modality 拼接）。**绝对值受特征维度影响，仅作诊断图，不聚合**。
- **混淆矩阵热图**：png 渲染，跨版本对比（v1/v2/v4 三栏并排）
- **90 分位留一距**：每样本到其预测类中心的欧氏距离 → 长尾指标，定位类内离群样本

### 关键约束

- 主分数只用 `1 - confusion_rate`，避免 Fisher 绝对值跨版本不可比
- 诊断图必须输出，方便人眼发现"哪几类最容易混"

## 维度 3：纯净度 CleanScore

**目标**：错配样本 + 跨模态不一致 + 重复样本。

### 算法

```python
# 1. 异常样本率
probs = probe["concat"].predict_proba(train_x_concat)
anomaly_score = 1 - probs[range(len(train_y)), train_y]   # 1 - 真实类概率
anomaly_rate  = mean(anomaly_score > anomaly_threshold)   # 默认 0.3，CLI 可配

# 2. 跨模态一致性（JS 散度）
probs_per_modality = {m: probe[m].predict_proba(val_x[m]) for m in modalities}
js_matrix = {m1, m2: jensen_shannon(probs_per_modality[m1], probs_per_modality[m2]) for m1, m2 in pairs}
js_per_sample = mean of js over modality pairs
inconsistency_rate = mean(js_per_sample > js_threshold)   # 默认 0.1，CLI 可配

# 3. 重复检测（量化 hash）
def quantize(x, decimals=2): return np.round(x, decimals)  # 固定小数位量化
hashes = {quantize(val_x[m].mean(axis=tuple)) for m}       # 每个模态一个指纹
dup_rate = mean any collision in sample-level hashes       # 默认权重 0.5（调低避免扰动总分）
```

### 主分数

```python
CleanScore = 1 - mean(anomaly_rate, inconsistency_rate, dup_rate)   # 三项等权
```

### 关键约束

- **JS 而非 KL**（对称，避免方向性）
- **JS 必须先 softmax 转概率**
- **重复检测用量化 hash**，不用 raw mean；特征 round 到 2 位小数再 hash，避免微小噪声扰动
- **dup_rate 权重调低**（默认 0.5），避免 hash 噪声扰动总分——可在 metadata 标注
- **anomaly_threshold / js_threshold / hash_decimals 全做 CLI 参数 + metadata**

## 总分 quality

```python
quality = w_info * InfoScore + w_compact * CompactScore + w_clean * CleanScore
```

默认权重：`w_info=0.4, w_compact=0.4, w_clean=0.2`（CLI 可覆写 `--w-info 0.5 --w-compact 0.3 --w-clean 0.2`）

允许不同数据集迭代阶段用不同权重（早期更看 clean，后期更看 info+compact）。

## 数据流（严格隔离）

1. `run_dataset_quality.py --dataset datasets/mmfi/vX --probe-out results/quality_vX.json`
2. **P0 护栏**（入口断言）：
   ```python
   assert "test" not in args.eval_split, "test split 不能进 probe 评估"
   ds = load_dataset(args.dataset)
   train_data = ds.train           # probe 训练
   eval_data  = ds.val             # probe 评估（绝不取 ds.test）
   ```
3. 加载 train → 训 6 个 Linear → 在 val 上评估 → 算 3 个 Score
4. 写 JSON（含 metadata block）+ 渲染诊断图
5. `leaderboard.py` 聚合所有 results → `leaderboard_quality.md`

**test split 全程不进入 dataset-quality 评估流程**——留给下游真实任务（主流程 robustness、未来 L1 retrieval 等），避免污染。

## 输出形态

### JSON 结构（每个 dataset 一个）

```json
{
  "dataset": "datasets/mmfi/v4",
  "metadata": {
    "val_sample_count": 3500,
    "train_sample_count": 46509,
    "num_classes": 27,
    "probe_epochs": 20,
    "probe_lr": 1e-3,
    "probe_batch_size": 256,
    "anomaly_threshold": 0.3,
    "js_threshold": 0.1,
    "hash_decimals": 2,
    "dup_weight": 0.5,
    "w_info": 0.4,
    "w_compact": 0.4,
    "w_clean": 0.2,
    "info_weights": {"per_modality": 0.7, "complement": 0.3}
  },
  "info": {
    "acc_per_modality": {"rgb": 0.x, "depth": 0.x, "lidar": 0.x, "mmwave": 0.x, "wifi": 0.x},
    "acc_concat": 0.x,
    "complement_gain": 0.x,
    "mean_acc": 0.x,
    "InfoScore": 0.x
  },
  "compact": {
    "confusion_matrix": [[...]],
    "confusion_rate": 0.x,
    "CompactScore": 0.x,
    "fisher_ratio": 0.x,
    "leave_one_out_dist_p90": 0.x
  },
  "clean": {
    "anomaly_rate": 0.x,
    "inconsistency_rate": 0.x,
    "dup_rate": 0.x,
    "CleanScore": 0.x
  },
  "quality": 0.x
}
```

### Leaderboard（跨版本聚合）

`leaderboard_quality.md`：

| dataset | InfoScore | CompactScore | CleanScore | Quality | per-modality (rgb/mmwave/depth/lidar/wifi) |
|---|---|---|---|---|---|
| v1 | ... | ... | ... | ... | ... |
| v2 | ... | ... | ... | ... | ... |
| v4 | ... | ... | ... | ... | ... |

### 诊断报告

- `per_modality_acc.png`：5 模态 × N 版本柱状图
- `confusion_matrix.png`：v1/v2/v4 三栏混淆矩阵对比
- `cross_modal_js.png`：每样本 JS 直方图
- `top_anomalies.json`：top-100 异常样本 id + 分数

## 与 L1-retrieval 的关系（关键澄清）

**dataset_quality ≠ L1 替代品**：

| | dataset_quality | L1-retrieval |
|---|---|---|
| 评测对象 | 数据集固有属性 | 下游跨模态检索任务性能 |
| 依赖 | 只用 train/val + 标签 | 数据集 + 文本模板 + 编码器（LLM/CLIP） |
| 输出 | info/compact/clean 分数 | recall@k |
| 失败原因 | 数据本身烂（如果分数低） | 模板烂 / 编码器烂 / 数据烂（三者耦合） |

**预期假设**：高质量数据集（高 info/compact/clean）应当**在 L1 下游取得更好结果**。如果不满足 → 说明下游任务（模板、编码器）存在瓶颈，**正好解释此前 M6b 在 CLIP/llama2 双侧负结果的现象**：数据本身未必烂，是下游评测系统有陷阱。

两套体系**互证**：L1 跑不出信号时，先看 dataset_quality——如果分数高，是下游问题；如果分数低，才是真数据问题。

## 双轨指标体系（2026-08-17 制度化）

**核心论断**：项目所有评测指标分属两个不同层级，**不能混为一谈**：

| 层级 | 指标 | 衡量什么 | 依赖 |
|---|---|---|---|
| **底层数据指标**（probe） | dataset_quality：InfoScore / CompactScore / CleanScore | 数据集固有的信息量与纯净度（**下界**） | 仅轻量 probe（MLP/Linear）+ 标签 |
| **端到端耦合指标**（model） | 主流程 Robustness Score / L1 retrieval r@k | 数据集 × 当前模型架构 的**联合性能**（**上界**） | 数据集 + 强模型（token_fusion / late_fusion / LLM） |

**关键约束**：
1. **probe 给下界**——浅层 MLP 看不见高阶跨模态互补，所以 dataset_quality **只证明"数据里有信号"**，**不证明"任何模型都能榨出"**
2. **端到端给上界**——强架构能挖出 probe 看不到的高阶信号，但同时引入架构偏差
3. **黄金子集**（`results/gold_subset_v1.json`，129 个 val 样本，两个高置信预测交集）：
   - 作为**控制变量**——两个独立模型都正确预测 → 这是已知的"答案"，用来归因
   - 用法：① 新数据集改进后，gold 子集 acc 应保持/提升 ② 新模型架构，gold 子集 acc 对比 probe 看架构加成

**两轨互相校验**（决策规则）：
- **数据改进了** → dataset_quality 涨，端到端也应该涨（理想）；只涨一边要归因
- **架构升级了** → 端到端涨，dataset_quality 不变（理想）；端到端涨但 dataset_quality 跌 = 过拟合，警惕
- **改了模型/数据都无效** → 检查黄金子集是否仍然是 ground truth，验证评测没坏

**已知耦合陷阱**：
- 不要用 Robustness Score 单独判"数据集质量"（M5a→M6b 的教训：acc=0.76 是 token_fusion × v4 联合结果）
- 不要用 dataset_quality 单独判"模型上限"（v4 rgb 0.828 是单 MLP 探针，token_fusion acc_full 0.76 来自 5 模融合不是单 rgb）
- 两轨一起用 → **围出真实信息空间**（v4：下界 0.83 rgb / 上界 0.76 acc_full / 真实信息在 rgb+mmwave）

### 黄金子集（控制变量）

- **位置**：
  - `results/gold_subset_v1.json`（129 个 val 样本，27 类各 ≤5）— token_fusion ∩ rgb-probe 双正
  - `results/gold_subset_v2.json`（85 个 val 样本，27 类各 ≤5，4 类为 0）— token_fusion ∩ rgb-probe ∩ mmwave-probe **三方**正
- **构造规则**：v4 val 上，多个独立预测都正确 → 高置信 ground truth
- **v2 比 v1 更严**：v2 引入 mmwave-probe 作独立证据，滤掉了"rgb-only 高置信"的 85 个样本（v1 only）
- **用法**：跨数据集/模型改进时复测 → 控制变量；归因时 → 区分数据缺陷 vs 架构缺陷
- **建议优先用 v2**：跨两个有效模态（rgb+mmwave）验证 ground truth，比 v1 更少偏倚

## 测试策略

### 单元测试（无 GPU / mock）

- `test_modality_probe.py`：per-modality Linear + Concat-Linear 输入输出形状、acc 计算正确性
- `test_compactness.py`：混淆矩阵 / Fisher ratio / 留一距正确性（toy 数据）
- `test_cleanliness.py`：JS 散度对称性 / 量化 hash / 异常率阈值化
- `test_info_score_clipping.py`：complement_gain 负值 clip 到 0、InfoScore ∈ [0,1]
- `test_p0_guard.py`：test split 传入时断言触发

### 集成测试

- `test_run_dataset_quality.py`：mini v4 (小规模 mock 数据集) → 跑完整 pipeline → 验证 JSON 结构 + metadata 完整

### 跨版本一致性测试

- `test_leaderboard.py`：多个 results → 渲染 markdown → 表格格式正确

## 里程碑

- **P0**：`run_dataset_quality.py` 入口加护栏（禁止 test split 进入 probe eval）
- **P1**：维度 1（per-modality + Concat-Linear）跑通 v4 出 InfoScore
- **P2**：维度 2（CompactScore）+ 维度 3（CleanScore，含 JS 散度 + 量化 hash）跑通 v4
- **P3**：v1/v2/v4 leaderboard 聚合 + 全套诊断图
- **P4**：数据质量演变结论写入 `docs/reports/dataset_quality_v1_v2_v4.md`

## 开放问题（不阻塞本迭代）

- **num_classes 做成 CLI 参数**：当前固定 27；未来新类别时支持 `--num-classes N`
- **probe 训练是否加数据增强**：当前不加（保持"测数据本身"纯净度）；若分数对 augment 敏感再加开关
- **类别不平衡处理**：当前 val acc 直接算；如有显著不平衡加 per-class F1 作为补充
- **JS 阈值 0.1 是否合理**：先用默认值，跨版本对比后微调
- **重复检测的 hash_decimals=2**：先用 2，跑 v1/v2/v4 看 dup_rate 量级是否合理
- **probe 是否要支持不同骨架**（如 small MLP）：当前只用 Linear；若 Linear 表达力不够再升级