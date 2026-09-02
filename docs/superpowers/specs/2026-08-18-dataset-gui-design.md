# 数据集可视化 + 文本修正 GUI（Dataset Curation GUI）设计

- 日期: 2026-08-18
- 状态: 设计定稿（用户确认方案 A 后授权自主执行）
- 前置: M6 全部完成。M5a/b/c、M6b 双负结果已归档——LLM 文本侧受"模板 caption 语义趋同"瓶颈限制；gold_subset v1/v2 是模型共识筛选，**缺人工复核的文本与质量标注**。
- 并行约束: 另一 agent 正在改进 `framework/eval/dataset_quality/probe_fusion.py`、`framework/models/llm_adapter.py`、`scripts/run_dataset_quality.py`、`scripts/make_v6_relabel.py`、`curation/enrich/`、`tests/test_keypoints_enrich.py`、`tests/test_dataset_quality/`。**本工具不读写这些路径。**

## 背景与目标

**最终目标**：人工构建"黄金数据集"——同时具备 ① 人工确认的正确文本标注（修掉模板 caption 的趋同/粒度粗问题）② 人工确认的高质量样本（修掉标签错，如 class 14/9 已知问题）。产出既可作为 LLM 文本侧的黄金标注，也可作为评测控制变量（对 gold_subset 的模型共识做人工复核）。

**本迭代目标**：一个**通用**的本地 GUI 工具（Streamlit）——
1. **聚合看板**：查看任意数据集根的质量指标与数据健康统计
2. **逐样本审查**：可视化 5 个模态 + 编辑文本/标签 + 质量标记 + 备注
3. **JSONL 编辑日志**：追加式、可回滚、可追溯
4. **构建脚本**：编辑日志 → 新黄金数据集版本（不动原始数据）

## 已确认的关键决策（用户拍板）

| 决策点 | 选择 |
|---|---|
| 黄金数据集目标 | **文本 + 样本双修**（既改 caption 文本，也人工筛选/复核样本质量） |
| 数据范围 | **通用**——符合 Dataset 协议的任何根目录（v1/v2/v4/v5/…/gold），任意 split |
| 质量可视化 | **两页**：聚合看板 + 逐样本审查 |
| 逐样本操作 | **全功能**：编辑文本 + 改标签 + 质量标记（golden/ok/reject/flagged）+ 备注 |
| 技术栈 | **Streamlit**（纯 Python，本地起服务，浏览器访问） |
| 落盘方式 | **JSONL 编辑日志 + 构建脚本** → 新数据集版本；原始 pickle 只读 |
| 模型预测 | **可选加载预测文件**（id→{pred,conf}）；另配预计算脚本；不内置实时推理 |
| 并行隔离 | GUI 代码全部在 `curation/gui/`；edits 输出也在其下；共享的只有只读的 framework loader 与数据集根 |

## 架构

```
curation/gui/
  ├── app.py                  # Streamlit 入口：--dataset --split --predictions；页面路由 + 会话状态
  ├── pages/
  │   ├── dashboard.py        # 聚合看板页
  │   └── review.py           # 逐样本审查页
  ├── core/
  │   ├── __init__.py
  │   ├── dataset_service.py  # 通过 framework/dataset/loader.py 加载任意数据集根（mode=lazy）
  │   ├── renderers.py        # 5 模态渲染器（plotly）+ 注册表 RENDERERS: dict[name->fn(data)->fig]
  │   ├── filters.py          # 按类/标签/审阅状态/subject/预测错误过滤
  │   ├── edit_log.py         # JSONL 追加式编辑日志（读/写/回滚/状态恢复）
  │   └── prediction_loader.py# 可选预测文件加载与校验
  ├── scripts/
  │   ├── precompute_predictions.py  # token_fusion 批量跑 split → predictions JSON（GPU）
  │   └── build_gold.py       # 编辑日志 → datasets/mmfi/gold/（dry-run 支持）
  └── edits/                  # 编辑日志输出（.gitignore）
tests/
  └── test_curation_gui/      # 单测 + 冒烟
```

**运行方式**：
```bash
streamlit run curation/gui/app.py -- \
  --dataset datasets/mmfi/v4 --split val \
  --predictions results/predictions_val_v4.json   # 可选
```

## 聚合看板页（dashboard.py）

- **数据集概览卡**：meta.json 的 name/version/changelog、各 split 样本数、模态列表
- **质量指标**（若存在对应 `results/quality_{ver}.json`，按 dataset name 自动匹配或手动指定）：
  - 总分 Quality + Info/Compact/Clean 三分解横条
  - per-modality probe acc 条形图（rgb/depth/lidar/mmwave/wifi）
  - concat probe 混淆矩阵热图（compactness）
- **数据健康统计**（从数据本身算，懒加载只扫一 subset 或按需）：
  - label 分布直方图（27 类）
  - subject 分布（从 id `E{ep}_S{subject}_...` 解析）
  - 帧数分布（每模态 frame_indices 长度）
  - 每模态异常检查：NaN/Inf、全零、全同帧（数据值无变化）的样本计数
- **快速入口**：点击某类/某 subject → 跳转到审查页预置过滤器

## 逐样本审查页（review.py）

**过滤器（sidebar）**：split、class（label/动词短语）、审阅状态（未审/已改/golden/ok/reject/flagged）、subject、预测错误（仅加载了 predictions 时，pred≠label）、备注非空

**样本列表**：表格（id / label / 动作 / 审阅状态 / 预测+置信 / 是否有文本改动），点击选择当前样本

**主面板（当前样本）**：
- **头部**：sample id、真实 label + 动作短语、预测（若加载）、当前审阅状态徽章
- **模态可视化**（renderers，各有独立渲染器，注册进 RENDERERS）：
  - `rgb`：关键点 2D 骨架动画——frame slider 拖动看逐帧姿态 (T,17,2) + 质心轨迹叠加
  - `depth`：frame slider 逐帧深度热图 (T,224,224)，下采样到 ~56×56 以保性能
  - `wifi`：CSI 热图 (frame × subcarrier，均值化天线) + 子载波方差剖面
  - `lidar`：frame slider 逐帧 3D 点云 scatter（降采样 ≤4000 点）
  - `mmwave`：frame slider 逐帧 Range 谱热图 / 点云
  - 未知模态：降级为"原始 shape + 基础统计"文本展示（通用性兜底）
- **编辑区**：
  - 文本（多行文本框，初始值为样本现有 text 的拼接或列表）
  - 标签（27 类下拉，显示动作短语）
  - 质量标记（radio: golden / ok / reject / flagged）
  - 备注（多行）
- **保存按钮**：append 一条 edit 事件到 JSONL；`_rev` 递增；状态徽章即时更新
- **撤销按钮**：回滚该样本最近一条 edit（写 rollback 事件）

## 编辑日志 schema（core/edit_log.py）

```json
{"event": "edit", "sample_id": "E04_S33_A01_f37-46", "rev": 2,
 "ts": "2026-08-18T00:40:00Z",
 "fields": {"text": ["…", "…"], "label": 0, "quality": "golden", "note": "…"},
 "changed": {"text": [[旧, 新]], "label": [7, 0]}}
```
- 事件类型：`edit` / `rollback` / `flag`（仅标记审阅状态无字段改动）
- 文件：`curation/gui/edits/{dataset_name}-{split}.jsonl`
- 读：加载时按 sample_id 折叠到**最新**状态（rev 最大）；写：追加
- 回滚：`rollback` 事件记录被回滚的 rev，加载时跳过其字段

## 构建脚本（scripts/build_gold.py）

```
python scripts/build_gold.py --dataset datasets/mmfi/v4 --split val \
  --edits curation/gui/edits/mmfi_v4-val.jsonl --out datasets/mmfi/gold \
  [--dry-run]
```
- 只取 quality=golden 的样本；应用其修正 text / label / note
- 输出新数据集根：`data/*.pkl`（仅 golden 样本，用 framework Sample 序列化）、`splits/gold.json`、`meta.json`（changelog 注明来源 + edit 引用）、`modalities.yaml`
- `--dry-run`：只输出统计（n_golden、label 分布、修正统计）不写盘
- 未审/非 golden 样本不进 gold 集

## 预计算脚本（scripts/precompute_predictions.py）

- 复用 `framework/models/token_fusion.py` 训练好的 checkpoint（`--ckpt checkpoints_v4/token_fusion_seed0.pt`）
- `predict_batch` 批量跑指定 split → 输出 `{id: {"pred": int, "conf": float, "source": "token_fusion_seed0"}}` 到 JSON
- GPU 任务按全局规范：先量基线、后台运行、前台监控；结果与 GUI 解耦

## 错误处理

- 数据集根无效 / 缺 meta.json → 启动时报错并列出可用数据集根
- 单样本 pickle 损坏 → 审查页跳过并标记，不崩页面
- 模态 shape 异常 → 渲染器内部 try/except，降级为文本统计展示
- 编辑日志文件被外部修改 → 每次保存前重读，以磁盘为准（防多窗口冲突）

## 测试

- `tests/test_curation_gui/`：
  - `test_edit_log.py`：append/折叠最新状态/rollback/损坏行容错
  - `test_renderers.py`：5 模态渲染器各产出一个 plotly Figure（含未知模态降级）
  - `test_filters.py`：各过滤条件组合
  - `test_build_gold.py`：golden 过滤、label/text 覆盖、dry-run 不写盘、输出可被 loader 重读
  - `test_prediction_loader.py`：加载/校验/损坏容错
- 运行：`.venv/bin/python -m pytest tests/test_curation_gui/ -q`
- 不做 GPU 测试（precompute 脚本冒烟用 2 样本）

## 并行隔离清单（不触碰的文件/目录）

`framework/eval/dataset_quality/`、`framework/models/llm_adapter.py`、`scripts/run_dataset_quality.py`、`scripts/make_v6_relabel.py`、`scripts/make_v5*.py`、`curation/enrich/`、`curation/caption/`、`results/`、`datasets/`（只读）、`tests/test_dataset_quality/`、`tests/test_keypoints_enrich.py`