# 缺模态处理机制（Missing-Modality Handling）

> 记录 SensorBench 当前如何建模"传感器缺失"，供未来改进参考。
> 更新：2026-08-25（temporal 完整评测后）。

## 一、核心设计：数据/模型解耦 + 缺模态鲁棒性

SensorBench 的基准假设是**数据质量 = 缺模态鲁棒性**。评测协议 `protocol_v5.json` 定义了 21 个 profile：
- `full`：5 模态全在
- `miss-*`：缺 1 模态（5 种）
- `miss2-*`：缺 2 模态（10 种）
- `only-*`：只剩 1 模态（5 种）

**Robustness Score** = 21 个 profile 的 mean acc 跨 3 seeds 平均；`acc_full` = 全模态 profile 的 acc。

## 二、当前缺模态处理方式（3 层）

### 1. 数据层：只传"可用"模态

评测 `framework/harness/evaluate.py` 把 `profile["available"]` 传给模型，**缺失模态物理上不给数据**：

```python
avail = {m: m in available for m in MODALITIES}   # 缺模态 -> avail[m]=False
for m in available:                                # 只取可用模态数据
    mods[m] = torch.stack([...])                  # 缺失模态无 data
```

数据不造假——模型必须自己处理"看不到某传感器"。

### 2. TokenFusion 内部：MISSING token + 注意力 mask

`framework/models/token_fusion.py` forward：

```python
for m in MODALITIES:
    if avail.get(m):                         # 模态存在 -> 正常编码
        enc_out = self.encoders[m](mods[m])
        toks.append(enc_out)
        masks += [1] * N_TOK                  # mask=1 参与注意力
    else:                                    # 模态缺失
        toks.append(self.missing[m].unsqueeze(0).expand(B, -1, -1))
        masks += [0] * N_TOK                  # mask=0 注意力屏蔽
x = torch.cat(toks, dim=1)                   # (B, 5*16, D)
x = self.fusion(x, src_key_padding_mask=~pad)
```

- **MISSING 嵌入**：`self.missing` 是每模态可学习参数（`N_TOK × D`，`token_fusion.py:74-75`），缺模态时填充占位。
- **注意力 mask**：缺失 token 被 `src_key_padding_mask` 屏蔽，不参与 attention。仅保持 token 数量对齐（每模态固定 16 token）。
- **保持架构不变**：无论缺失几个模态，token 序列恒为 80（5×16），只有 mask 变化。

### 3. LateFusion 简化：零向量填充

`framework/models/late_fusion.py:39`：缺模态时用 `torch.zeros(B, D)` 填充，纯 concat + MLP，无对齐机制（对照组）。

### 训练期"预演"：modality dropout

`token_fusion.py _dropout_mask`：训练时每个样本以 p=0.25 随机丢每个模态，强制模型适应缺失：

```python
avail[m] = bool(torch.rand(1).item() > p)   # p=0.25 随机缺模态
if not any(avail.values()):
    avail[list(avail)[0]] = True              # 至少保留一个模态
```

- `modality_dropout` 可对特定模态偏置（`--modality-dropout '{"mmwave":0.5,"rgb":0.5}'`）。
- **实验结论（improvement_plan.md §6）**：偏置 dropout 只对 miss-rgb +0.024，但 miss-mmwave -0.029、full -0.015，整体净负。MISSING token 已能处理缺失，偏置 dropout 与"真实缺模态样本"机制等价 → **P0-1.1 真实缺模态数据管线收益有限，不投入**。

### temporal 特有：时间遮蔽

`time_mask_p`：训练时抹掉随机连续帧，强制因果 TemporalAggregator 从上下文重建（对标 MiniMind-O）。temporal=True 时缺模态仍走 MISSING token 路径（兼容）。

## 三、当前表现

temporal token_fusion（2026-08-25，`leaderboard_temporal_full.json`）：

| profile | acc |
|---------|-----|
| full | 0.9451 |
| miss-mmwave | 0.8135 |
| miss-rgb | 0.7413 |
| miss2-mmwave-rgb | 0.1472 |
| only-mmwave | 0.6826 |
| only-rgb | 0.7431 |
| **robustness** | **0.6904** |

**短板**：`miss2-mmwave-rgb` 崩到 0.1472（同缺两大主模态）；only-wifi/depth/lidar 极低（弱模态缺独立判别力）。

## 四、未来改进方向（待探索）

> 以下均为 [提议]，未投入。来自 omni 调研（`docs/reports/omni_model_comparison.md`）与 improvement_plan。

1. **缺模态建模显式化（P2）**：将缺模态作为一等公民——缺模态专用 loss 或显式"模态可用性"条件输入。当前 MISSING token + mask + dropout 已实现但未系统化。
2. **模态感知 token 预算（P1）**：缺模态时高信息模态（rgb）更多 token，低信息模态（wifi/depth）少给（TokenRouter 雏形 `framework/models/router.py`，对标 OmniScope）。
3. **组合缺失增强（P1）**：训练模拟 miss2-* 组合缺失，对齐评测协议（当前 dropout 独立丢，未见组合缺 2 模态）。
4. **真实缺模态样本（P0，已判定收益有限）**：数据层物理去掉 mmwave/rgb 生成真实缺模态样本。
5. **缺模态路由/动态融合**：对可用模态集做条件融合，而非固定 5×16 均匀 token 预算。
6. **弱模态独立判别力**：wifi/depth/lidar 单模几乎随机（only-* ≈0.05-0.09），缺模态鲁棒性受此拖累；改进需提升弱模态数据信息量（非缺模态机制本身）。

## 五、关键文件

- 评测入口：`framework/harness/evaluate.py`、`scripts/run_eval.py`
- leaderboard：`framework/harness/leaderboard.py`
- 协议：`protocol.json`（4 模态 15 profiles）、`protocol_v5.json`（5 模态 21 profiles）
- 模型：`framework/models/token_fusion.py`（MISSING+mask）、`late_fusion.py`（零填充）
- 时序：`framework/models/temporal.py`（TemporalAggregator）
- 改进方向：`docs/reports/improvement_plan.md`、`docs/reports/omni_model_comparison.md`
