# SensorBench：可复现的跨模态传感融合基准框架设计

日期：2026-08-13
状态：Draft（待评审）

## 1. 背景与目标

### 1.1 问题

我们希望构建一个**高质量的多模态对齐数据集**，用于室内人体活动感知的**跨模态传感融合**（多传感器互相对齐，文本为辅助监督）。数据集的构建不是一次性交付，而是**持续收集、迭代优化**的资产。

已有的 MMFi / XRF55 等公开数据集存在明显不足（模板化 caption、类不均衡、无细粒度 grounding、规模小），不能直接作为最终资产使用。

### 1.2 核心洞察

**数据集是产品，模型是测量尺。** 数据质量的好坏，通过"缺模态鲁棒性"量化——即全模态训练后，缺某模态时的性能降幅越小，说明模态间的对齐与互补越好。

因此本项目的核心不是训练一个 SOTA 模型，而是构建一个**数据与模型完全解耦的基准框架（SensorBench）**：

- 数据集侧负责：规范格式、清洗、增强、版本化、评测协议
- 模型侧负责：通过统一接口 `SensorModel` 即插即用
- 评测驱动数据迭代：Degradation 矩阵指出数据弱项 → 补数据 → 新版本 → Robustness Score 应单调上升

### 1.3 约束

- 暂无采集硬件，先用公开数据（MMFi 为主）
- 单卡 RTX 5060 Ti 16GB，模型必须轻量（参数几十 M 级）
- 数据与模型完全分离，模型可任意更换
- 评测协议必须可复现（固定 splits、固定组合表、多 seed）

## 2. 架构总览

```
sensorbench/
├── framework/
│   ├── dataset/          # 规范格式 + 加载器 + 划分
│   ├── models/           # SensorModel 协议 + 适配器
│   └── harness/          # 评测协议 + 排行榜
├── curation/
│   ├── ingest/           # mmfi.py, xrf55.py, custom.py
│   ├── clean/            # nan.py, align_check.py, consistency.py
│   ├── enrich/           # recaption.py
│   └── version/          # version.py, registry
└── datasets/
    ├── mmfi/v1/          # MVP 只产出这一份
    ├── xrf55/v1/         # 未来结构（延期）
    └── mmfi+xrf55/v2/    # 未来结构（延期，跨数据集训练）
```

## 3. 设计原则

1. **数据与模型完全解耦**：数据集只定义"数据本身 + 评测协议"；模型通过 `SensorModel` 接口即插即用，框架不依赖任何具体模型实现。
2. **缺模态由调用方声明**：评测 harness 决定"这次推理可用哪些模态"，模型只负责适配，不决定协议。
3. **数据是演进的资产**：版本化 + 变更日志 + 协议 hash 绑定，排行榜只能跑在固定版本上。
4. **YAGNI**：MVP 只做最小闭环，重活（recaption、自采、对比学习）验证后再加。

## 4. 设计 Section 1：数据集规范格式

### 4.1 目录布局

```
datasets/<dataset_name>/<version>/
├── data/
│   ├── sample_000001.pkl     # 每条样本 = 一个 dict
│   ├── sample_000002.pkl
│   └── ...
├── splits/
│   ├── train.json            # 样本 id 列表 + 划分标签(subject/env)
│   ├── val.json
│   └── test.json
├── modalities.yaml           # 传感器注册表（声明式）
└── meta.json                 # 数据集版本、来源、许可、采集协议
```

### 4.2 样本 dict 结构（模型输入契约）

```python
{
  "id": "mmfi_E02_S19_A03_f065-101",
  "label": 2,                          # 动作类别 id（数据集定义，与模型无关）
  "modalities": {
    "wifi":   {"data": np.ndarray, "shape": [5,114,32,3],   "sample_rate": 1000, "frame_indices": [65..101]},
    "mmwave": {"data": np.ndarray, "shape": [5,64,5],       "sample_rate": 20,   "frame_indices": [65..101]},
    "lidar":  {"data": np.ndarray, "shape": [5,1536,3],     "sample_rate": 20,   "frame_indices": [65..101]},
    "depth":  {"data": np.ndarray, "shape": [5,1,224,224],  "sample_rate": 20,   "frame_indices": [65..101]},
  },
  "text": {                            # 辅助监督，可选
    "captions": [...],
    "vqa": [{"q": ..., "a": ...}]
  },
  "meta": {"subject": 19, "env": 2, "sensor_settings": {...}}
}
```

**`frame_indices` 的确切语义（重要）**：

- `frame_indices` 是**全局同步参考时间线上的帧索引**（如 MMFi 采集时各传感器硬件同步，`frame003.png/.bin/.mat` 对应同一参考时刻），覆盖该样本对应的动作窗（如 `[65..101]` 共 37 个参考帧）。
- 它**不是**各传感器自身的采样时间戳。`sample_rate` 是传感器帧内的原始采样率（元信息），仅用于说明；数据在摄取时已重采样到共享参考帧序列。
- 因此 `data.shape[0]` = 从参考窗抽出的帧数（MVP 中按 MMFi 的 5 段抽样约定为 5），各模态 shape 差异来自**帧内分辨率**（CSI 子载波数 / 点云点数 / 像素），不来自时间窗不同。
- 一条样本内所有模态的 `frame_indices` 必须**完全相同**（引用同一个参考窗）。若某模态掉帧/错位导致窗口不一致，该样本视为异常（见 7.1 清洗），剔除并记录。

### 4.3 关键点

- `modalities` 是 dict，模型按 key 取它需要的模态——天然支持换模型和缺模态
- 每模态自带 `frame_indices`（全局同步参考帧索引，见上）和 `sample_rate`（帧内原始采样率），时间对齐信息显式存储，不藏在加载代码里
- 一条样本内所有模态必须引用同一 `frame_indices` 参考窗（一致性由 7.1 清洗校验）
- 标签、splits、协议全部由数据集侧定义，模型侧不能改
- `modalities.yaml` 声明数据集有哪些传感器；新增传感器 = 注册一个 entry

## 5. 设计 Section 2：模型适配器接口

### 5.1 SensorModel 协议

```python
# framework/models/base.py
class SensorModel:
    name: str

    def fit(self, train: Dataset, val: Dataset, cfg: TrainConfig) -> None:
        """在数据集上训练。框架不关心内部实现。"""

    def predict(self, sample: Sample, available: list[str]) -> dict[str, float]:
        """对单条样本预测。
        available: 本次推理可用的模态列表（缺模态由调用方决定）
        返回: {class_id: prob, ...}
        """
```

### 5.2 关键决策

- **缺模态由调用方声明**：评测 harness 说"这次只有 wifi+mmwave"，模型就只收到这两个模态的 data。鲁棒性策略（`[MISSING]` token、zero-pad、impute）是模型内部的事。
- **适配器包装**：每种模型实现 `SensorModel` 即可注册进框架。
- **约定优于配置**：模型收到未注册的模态 key 就报错，强制数据与模型不偷偷耦合。
- **统一 checkpoint 格式**：训练好的模型保存为统一格式，排行榜只认接口。

### 5.3 内置模型

- `token_fusion`（默认主模型）：统一 token 融合网络。每模态一小 encoder → 固定数量 token → 共享浅 Transformer（2-4 层）→ 分类头。缺模态用可学习 `[MISSING]` token + mask。参数量几十 M。
- `late_fusion`（基线）：每模态 encoder 各出一个特征 → concat/加权求和 → MLP 分类。缺模态 zero-pad。
- `contrastive`（可选，后期）：对比对齐 + 融合双头。

## 6. 设计 Section 3：评测协议

### 6.1 缺模态组合表（Modality Profile Matrix）

K 个模态，4 层评测：

| 层级 | 组合数 | 说明 |
|------|--------|------|
| Full | 1 | 全部 K 个模态 |
| Single-Missing | K | 每次缺一个 |
| Double-Missing | C(K,2) | 每次缺两个（MVP K=4 → 6 种全量枚举；K 大时可配置为随机抽样子集） |
| Single-Modal | K | 只剩一个模态 |

组合表由 `modalities.yaml` + 协议配置自动生成，写进 `protocol.json`：

```json
{
  "modalities": ["wifi","mmwave","lidar","depth"],
  "seeds": [0, 1, 2],
  "profiles": [
    {"id": "full", "available": ["wifi","mmwave","lidar","depth"]},
    {"id": "miss-wifi", "available": ["mmwave","lidar","depth"]},
    {"id": "only-wifi", "available": ["wifi"]}
  ]
}
```

（K=4 时表会生成全部 15 个 profile：1 Full + 4 Single-Missing + 6 Double-Missing + 4 Single-Modal；`seeds` 列表固定多 seed 复现。）

### 6.2 指标

- `Acc_full`：全模态准确率
- `Acc[profile]`：每个组合的准确率
- **Robustness Score（核心）**：`mean(Acc[profile])`，对 `protocol.json` 里**所有** profile（含 Full）取平均——固定的定义，保证跨模型/跨版本可比
- **Degradation 矩阵**：`Acc_full − Acc[profile]`，按模态列出——指出最缺一不可/最可有可无的传感器
- 多 seed 报告 mean ± std + bootstrap 置信区间

### 6.3 排行榜

固定协议、固定 splits、固定 seed，分数只认 `SensorModel.predict` 输出。存 JSON，可复现比对。

### 6.4 统计严谨性

- splits 按 subject/env 分层，防泄漏（复用 MMFi 的 cs/ce 设计）
- 多 seed 报告均值±std，模型间差异做配对显著性检验
- 协议 hash 归档，`--seed` 固定

## 7. 设计 Section 4：数据管线

### 7.1 四段流水线

```
原始数据 ─► [1 摄取] ─► [2 清洗] ─► [3 增强] ─► [4 版本化] ─► 规范数据集
```

- **摄取（ingest）**：每种数据源一个适配器（`ingest/mmfi.py`、`ingest/xrf55.py`、`ingest/custom.py`），输出统一 Sample dict，元数据写进 meta.json。
- **清洗（clean）**：NaN/Inf 检测与插值、帧对齐校验（各模态 `frame_indices` 必须完全一致，不一致剔除并记录）。跨模态一致性过滤（用基线模型交叉验证，预测 top-2 或概率差超阈值即标记可疑供人工抽检）**依赖已训练模型**——v1 阶段关闭（尚无模型），v2 迭代（M4，重训后）才启用。
- **增强（enrich）**：可选。用轻量 VLM 对 caption 重写，解决模板化问题。增强字段与原始标注并存。
- **版本化（version）**：每个版本一个目录，meta.json 记录变更日志；协议 hash 绑定版本。

### 7.2 数据 flywheel

```
持续收集新数据 ─► 摄取+清洗+增强 ─► 新版数据集 ─► 训基线 ─► 缺模态评测
     ▲                                                            │
     │                                                            ▼
     └──── 弱项定位：哪些 (模态组合, 类别, 受试者) 在缺模态下崩 ─────┘
```

数据好不好 = 每次版本迭代后 Robustness Score 是否单调上升。

## 8. 设计 Section 5：MVP 闭环

### 8.1 MVP 范围

**输入**：MMFi 的 4 个模态 `wifi / mmwave / lidar / depth`（本地齐全，无需 RGB 下载）。

**首个版本链路**：

```
v1: 摄取 4 模态 → 规范格式 → 训 token_fusion 基线 → 缺模态评测 → 出 Robustness 报告
     │
     ▼ （根据报告做一次数据改进）
v2: 针对性清洗/增强 → 重训 → 对比 Robustness v1 vs v2（应单调上升）
```

**交付物**：
1. `datasets/mmfi/v1/` + `v2/`——规范格式数据集，带版本元数据
2. `SensorModel` 协议 + `token_fusion`（主）+ `late_fusion`（基线）
3. `protocol.json` + 评测 harness → `leaderboard_v1.json`
4. Degradation 矩阵报告 + v2 改进说明

### 8.2 明确不做（延期）

| 延期项 | 原因 |
|--------|------|
| XRF55 接入 | 先跑通 MMFi 单圈闭环，再横向扩展 |
| contrastive 模型 | 数据质量验证到位前不引入训练复杂度 |
| VLM recaptioning | 重，文本非核心；先量化缺模态鲁棒性 |
| 自采硬件/真实数据收集 | 框架和评测协议验证后再谈 |
| 跨数据集融合训练 | v2 后再考虑 |

## 9. 里程碑

| 里程碑 | 内容 | 完成标志 |
|--------|------|----------|
| M1 | 项目骨架 + 规范格式 + MMFi ingest 管线 | 管线从 MMFi 原始数据产出 `datasets/mmfi/v1/`，`framework/dataset` 可加载 |
| M2 | SensorModel 协议 + token_fusion/late_fusion | 两个适配器能在数据集上训练 |
| M3 | 评测 harness + 排行榜 | 跑出完整 Robustness 报告 |
| M4 | v2 数据改进 | Robustness v2 > v1 得到验证 |
| M5 | 文档与可复现性 | 复现指南 + 排行榜归档 |

## 10. 待确认项

- 项目命名：sensorbench（可改）
- 样本存储格式：pickle vs zarr（MVP 用 pickle，数据量大后再评估 zarr）
- 是否引入 config 管理框架（MVP 用 argparse + yaml，不引额外依赖）
