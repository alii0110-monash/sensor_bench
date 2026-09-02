# Cross-Attention Fusion Model — Design

**日期**: 2026-08-25
**状态**: 已确认（用户拍板）

## 背景与动机

当前主模型 `token_fusion`（`framework/models/token_fusion.py`）用**拼接 + 共享 transformer** 做跨模态融合：5 模态各 16 token 拼成 80 token，过 2 层 TransformerEncoder。缺失模态用 `[MISSING]` embedding 占位 + attention mask。

问题：
1. 拼接后自注意力复杂度是 **O((模态数×16)²)**，随模态数**二次**增长。
2. 缺失模态的 `[MISSING]` 其实被 mask 掉不参与 attention，鲁棒性全靠训练期模态 dropout 隐式学。
3. 固定 16 token/模态，token 预算不反映信息量。

目标：用**交叉注意力**替代拼接融合，让计算量随模态数**线性**增长，且天然支持任意模态接入/去除。

## 设计决策（用户已确认）

| 决策点 | 选择 |
|---|---|
| 落地方式 | 新增独立模型 `cross_attention`，保留 `token_fusion` 作基线对比 |
| query 来源 | **可学习全局 query × 32**（Perceiver/Q-Former 风格），不绑定任何模态 |
| temporal | 支持，复用 `TemporalAggregator` 做模态内时序聚合 |
| 训练增强 | 完全复用模态 dropout + 时间遮蔽，与基线训练配置一致 |

## 架构

```
输入 (B, T, ...) 各模态
  │
  ├─ per-modality encoder ──► (B, T, 16, D)   [Wifi/Depth/Point/MLP/Domain]
  │        │
  │        └─ TemporalAggregator (temporal=True) ──► (B, 16, D)
  │
  ▼
  key/value: 各模态 token 拼接 ──► (B, 模态数×16, D)
  query:     可学习 latent ──► (B, 32, D)
  │
  ▼
  Cross-Attention (query 读 key/value) ──► (B, 32, D)
  │
  ▼
  mean-pool ──► (B, D) ──► Linear(D, num_classes)
```

### 缺失模态处理
- 缺失模态的 token **不进入 key/value**（不提供即可），query 照常。
- 无需 `[MISSING]` embedding、无需 attention mask——这是交叉注意力相对拼接的核心优势。
- 训练期模态 dropout 让模型学会在任意缺失组合下用剩余模态融合。

### 计算量
- 交叉注意力 = **O(Q × K)**，Q=32 固定，K=模态数×16。
- 5 模态：32×80=2560，比拼接的 80²=6400 省。
- 随模态数**线性**增长，接入/去除模态只改 K，query 不变。

## 组件

新增 `framework/models/cross_attention.py`，实现 `CrossAttentionModel(nn.Module, SensorModel)`：

- `__init__(num_classes, d, n_heads, n_layers, n_query=32, structured, domain, domain_dims, temporal)`：
  - 复用 `_build_encoders`（`token_fusion.py`）构建 per-modality encoder。
  - `self.query = nn.Parameter(torch.randn(n_query, d) * 0.02)` 可学习全局 query。
  - `self.temporal_agg`：temporal=True 时复用 `TemporalAggregator`。
  - `self.cross`：`nn.MultiheadAttention` 或 TransformerEncoderLayer（query 作 src，key/value 作 memory）。
  - `self.head = nn.Linear(d, num_classes)`。
- `forward(mods, avail)`：编码 → 时序聚合 → 拼 key/value → 交叉注意力 → mean-pool → head。
- `fit` / `_dropout_mask` / `_stack_mods` / `_apply_time_mask` / `_evaluate` / `predict` / `predict_batch` / `save` / `load`：复用 `token_fusion` 的既有实现模式（含 checkpoint 持久化 structured/domain/domain_dims/temporal）。

## 注册

- `scripts/train.py` + `scripts/run_eval.py` 的 `MODELS` dict 加入 `"cross_attention": CrossAttentionModel`。

## 验证

- 与 `token_fusion` temporal 基线（robustness 0.6904, full acc 0.9451）同配置对比。
- 跑 `protocol.json` 的 21 个 profile × 3 seeds，比较 robustness score 与各缺失 profile 的 degradation。
- 单元测试：`tests/` 下新增 cross_attention 的 forward 形状、缺失模态、temporal 路径测试。

## 风险

- 交叉注意力参数量略增（query 参数 + 注意力层），但远小于 encoder 主体。
- 若 query 数过少可能容量不足；32 是合理起点，可调。
