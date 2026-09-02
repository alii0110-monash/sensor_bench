# Dataset Curation GUI

Streamlit 工具：可视化数据集质量 + 逐样本审查 + 文本/标签修正，构建人工复核的黄金数据集。

与数据集质量评测系统（`framework/eval/dataset_quality/`）完全解耦：**只读**原始数据，所有修正写入独立 JSONL 编辑日志，不触碰任何原始 pickle。

## 运行

```bash
# 1. 环境：sensorbench conda env（torch 2.9.1+cu128 + streamlit + plotly）
conda activate sensorbench
# 2. 启动（默认 val split）
streamlit run curation/gui/app.py -- --dataset datasets/mmfi/v4 --split val

# 可选：加载模型预测做错误分析（先跑 precompute 生成）
streamlit run curation/gui/app.py -- --dataset datasets/mmfi/v4 --split val \
  --predictions results/predictions_val_v4.json
```

浏览器打开 `http://localhost:8501`。两个页面（侧栏切换）：**逐样本审查** / **聚合看板**。

**侧栏数据集切换**：`--dataset`/`--split` 只是启动默认值；运行中可在侧栏自由切换任意数据集（自动发现 `datasets/*/` 下符合协议的所有根目录，含 v1/v2/v3/v4/v5/v5_structfeat）与任意 split。切换后编辑日志按 `{dataset}-{split}.jsonl` 自动对应加载，预测文件仅在数据集与启动时一致时才生效。

## 逐样本审查页

- 侧栏过滤器：类别 / subject / 审阅状态 / 预测对错（加载了 predictions 时）/ 有备注
- 样本表格选择 → 看 5 模态可视化（rgb 骨架、depth 热图、wifi CSI、lidar 点云、mmwave 雷达点云 3D）：
  - 多帧样本有帧滑块；聚合视图看全序列趋势
  - 未知/不可渲染模态降级为原始统计
- 修正区：**文本**（每行一条 caption，可整段重写）、**标签**（27 类下拉）、**质量标记**（golden/ok/reject/flagged）、**备注**
- 按钮：保存修改 / 仅记录审阅状态 / 回滚最近修改
- 修改记录可展开查看（字段级 old → new）

## 聚合看板页

- 数据集概览（version / split 大小 / 模态 / changelog）
- 质量指标（自动匹配 `results/quality_{ver}.json`）：Quality + 三维分解、per-modality probe acc 条形图、confusion matrix 热图
- 数据健康统计（懒加载，最多 `CURATION_HEALTH_MAX` 个样本，默认 2000）：
  label 分布 / subject 分布 / 帧数分布 / 异常检查（NaN·全零·全同值·空）

## 编辑日志

- 位置：`curation/gui/edits/{dataset_name}-{split}.jsonl`（已 gitignore）
- 追加式 JSONL，每行一个事件：`edit` / `flag` / `rollback`，含 rev 递增与时间戳
- 加载时折叠到每样本最新有效状态；回滚写入新事件（历史可追溯、可重放）

## 构建黄金数据集

```bash
# 只统计，不写盘
python curation/gui/scripts/build_gold.py --dataset datasets/mmfi/v4 --split val \
  --edits curation/gui/edits/mmfi_v4-val.jsonl --out /tmp/gold_dry --dry-run

# 实际构建（只保留 quality=golden 的样本，应用修正文本/标签，生成新数据集根）
python curation/gui/scripts/build_gold.py --dataset datasets/mmfi/v4 --split val \
  --edits curation/gui/edits/mmfi_v4-val.jsonl --out datasets/mmfi/gold
```

输出 `datasets/mmfi/gold/` 符合 Dataset 协议，可被 `load_dataset` 直接读取（`splits/gold.json`）。

## 预计算模型预测（可选，GPU）

```bash
python curation/gui/scripts/precompute_predictions.py \
  --dataset datasets/mmfi/v4 --split val \
  --ckpt checkpoints_v4/token_fusion_seed0.pt \
  --out results/predictions_val_v4.json
```

按 GPU 规范跑（先量基线、前台监控）；`--max-samples N` 可冒烟。

## 测试

```bash
python -m pytest tests/test_curation_gui/ -q        # 单元测试
python -m pytest tests/test_curation_gui/ -m slow   # AppTest 冒烟（真实 v4 数据）
```

## 设计

见 `docs/superpowers/specs/2026-08-18-dataset-gui-design.md`。

## 并行隔离

本工具读写范围：`curation/gui/`（含 edits/） + `tests/test_curation_gui/`。
只读：`framework/dataset/loader.py`、`curation/caption/verbs.py`、数据集根。
不触碰：`framework/eval/dataset_quality/`、`framework/models/llm_adapter.py`、`scripts/run_dataset_quality.py`、`scripts/make_v*.py`、`curation/enrich/`、`results/`。

> 注：`framework/dataset/loader.py` 增加了一处向后兼容改动——split 发现改为读取 `splits/*.json` 全部文件（原硬编码 train/val/test），使 gold 等自定义 split 可经标准路径加载。全部 248 测试通过。