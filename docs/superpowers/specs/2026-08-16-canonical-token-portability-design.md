# 伪 Token 可移植性架构（CanonicalToken 协议 + 资产化）设计

- 日期: 2026-08-16
- 状态: 设计定稿
- 前置: M5a-M5c 完成。M5c 负结论（冻结 LLM 读不懂伪 token）后转向 B' 路线——伪 token 作为**跨模态统一表征**（可移植），而非给 LLM 生成用。

## 背景与目标

最终目标：构建多模态对齐数据集 + 好编码器。M5c 证明"冻结 LLM 生成式理解伪 token"不可行，但编码器质量仍是核心。B' 路线把伪 token 定位为**可移植的跨模态统一表征**：wifi/depth/lidar/mmwave/rgb → 规范空间向量，供检索/分类/后续任意 LLM 消费。

**本迭代（M6a）目标**：把"规范空间"与"LLM 空间投影"彻底解耦，定义 CanonicalToken 协议 + 伪 token 资产化（落盘交换），为将来"编码器一次训好、换 LLM 只换投影"打地基。

## 已确认的关键决策

| 决策点 | 选择 |
|---|---|
| 路线 | **B'**：伪 token = 跨模态统一表征，不追求生成式 LLM 理解 |
| 规范空间维度 | **4096**（=llama2 hidden；换 LLM 只需 Linear(4096→目标hidden)） |
| 可移植性抽象 | **接口协议 + 伪 token 资产化（落盘交换）** |

## 架构（两段解耦）

```
┌────── 规范空间（与 LLM 无关，一次训好） ──────┐
│                                               │
│ 传感器 → 编码器 → Perceiver → 规范token (4096) │
│                (冻结)         CanToken 协议    │
│                                    │           │
│                伪 token 资产化落盘（npz + index.json, 版本化）│
└────────────────────────────────────┼───────────┘
                                     ▼
┌────── LLM 空间（可插拔，每 LLM 一份） ──────┐
│                                             │
│ CanToken(4096) → Linear(4096→hidden) → 注入 │
│                  per-LLM 投影层              │
└─────────────────────────────────────────────┘
```

## 组件 1：CanonicalToken 协议

**定义**：规范空间伪 token 的标准格式，与任何 LLM 无关。

```python
@dataclass
class CanonicalToken:
    """规范空间伪 token 的可移植载体."""
    id: str                    # 样本 id
    label: int                 # 动作类别 (0-26)
    data: np.ndarray           # (M*k, 4096) float32, modality-major
    modality_order: List[str]  # 每模态 k 个 token 的顺序 (对应 data 行)
    k: int                     # 每模态 token 数
    meta: Dict                 # encoder_version 溯源 (权威溯源在 index.json)
```

**协议约束**：
- `data` 固定 4096 维 float32（规范空间）
- `modality_order` 与 `data` 行对齐（modality j 占 `[j*k, (j+1)*k)`）
- `meta` 仅存 `encoder_version`（编码器 checkpoint 标识；完整溯源在 index.json，不重复维护）

## 组件 2：CanonicalTokenizer（编码器 → CanonicalToken）

**职责**：把传感器样本经编码器 + Perceiver 转成 CanonicalToken，或从落盘加载。

```python
class CanonicalTokenizer:
    """传感器样本 ↔ CanonicalToken 双向转换."""

    def __init__(self, align_ckpt: str, proj_ckpt: str, k: int = 8):
        # 冻结加载 AlignmentModel + PerceiverProjection (M5a/b checkpoint)

    def encode(self, sample) -> CanonicalToken:
        """传感器 → 规范 token."""

    def decode(self, tok: CanonicalToken) -> np.ndarray:
        """CanonicalToken → (M*k, 4096) 张量 (供注入)."""
```

## 组件 3：伪 Token 资产化（落盘交换）

**格式**：`CanonicalToken` 序列化到 `{root}/tokens/{sample_id}.npz`（数据）+ 一个 `index.json`（元数据/版本）。

```
datasets/mmfi/v5tokens/
  index.json          # {id: {label, k, modality_order, meta}, ...} 版本化
  tokens/             # {sample_id}.npz: data (M*k,4096)
```

**版本化**：`index.json` 含 `version`, `encoder_ckpt`, `generated_at`, `n_samples`。参考现有 `write_meta` 模式（curation/version/version.py）。

**生成流程**：`scripts/make_tokens.py` —— 遍历 v5 数据集（train base），用 CanonicalTokenizer.encode → 落盘。一次性离线，可复现。

## 组件 4：LLM 空间投影（per-LLM 可插拔）

**职责**：CanonicalToken(4096) → 目标 LLM hidden，注入其输入。

**与现有 `LLMAdapter`/`LlamaAdapter` 的关系**：`LLMAdapter.inject`（llm_adapter.py:16-32）已实现"伪 token 前缀 + 文本 embedding 拼接"，语义相同。区别在输入形态：新层消费**落盘的 CanonicalToken**（MaterializedToken），LLMAdapter 消费**张量**。规划时复用 `LLMAdapter.inject` 的拼接逻辑，`TokenToLLM` 只新增 `project(CanonicalToken)` 的物化→张量转换，不重复实现 inject。

```python
class TokenToLLM(ABC):
    """规范空间 → 目标 LLM 空间投影. project() 复用 LLMAdapter.inject 拼接."""

    @property
    @abstractmethod
    def llm_hidden(self) -> int:
        ...

    @abstractmethod
    def project(self, canonical: CanonicalToken) -> torch.Tensor:
        """(M*k, 4096) -> (1, n, llm_hidden) 伪 token."""

    # inject 复用 LLMAdapter (见 llm_adapter.py), 不在本类重复定义


class LinearTokenToLLM(TokenToLLM):
    """轻量线性投影: Linear(4096 -> llm_hidden). 换 LLM 只换这一层."""
```

**关键解耦**：CanonicalToken(4096) 与 LLM 无关；`LinearTokenToLLM` 是唯一的 LLM 相关层，per-LLM 一份。

**溯源单源**：`CanonicalToken.meta` 只存 `encoder_version`（编码器 checkpoint 标识）；`index.json` 是权威溯源（version/generated_at/n_samples）。不在两处重复维护同一字段。

## 数据流（端到端）

1. **生成**（离线一次）：`make_tokens.py` → v5 传感器 → CanonicalToken → 落盘 `v5tokens/`
2. **交换**：CanonicalToken 可被任意消费者加载（检索/分类/LLM 投影）
3. **LLM 注入**（可选，per-LLM）：加载 CanonicalToken → `LinearTokenToLLM` → 注入

## 评测（B' 隐式，复用 L1 体系）

- **跨模态检索 recall@k**：CanonicalToken(4096) 池化 vs 文本 embedding。**文本侧定案**：用 llama2 mean-pool caption embedding（4096 维，与规范空间匹配），复用 `scripts/train_projection.py` 的 `_llm_text_emb` 模式（需加载 llama2，标记 slow）。不用 CLIP 512 维（维度不匹配）。
- **动作分类**：CanonicalToken → 线性分类头（27 类）
- **可移植性验证**：同一份 CanonicalToken 用 `LinearTokenToLLM` 投影到不同 hidden（如 4096→2048→1024），验证维度投影正确。**信息保留 = 投影后跨模态检索 recall@k 稳定性**（投影后 r@1 相对投影前下降 ≤ 2pt 视为通过）。

## 测试策略

- 单元测试（无 GPU/mock）：
  - `test_canonical_token.py`：CanonicalToken dataclass 序列化/反序列化、维度/顺序校验
  - `test_tokenizer.py`：encode/decode round-trip（mock 编码器）
  - `test_assets.py`：落盘 npz + index.json 生成/加载、版本化
  - `test_token_to_llm.py`：LinearTokenToLLM 维度投影、换 hidden 可插拔
- 集成测试：
  - `test_make_tokens_e2e.py`：mini v5 → 生成 CanonicalToken → 加载 → 检索 recall@k 跑通

## 里程碑

- **M6a（本迭代）**：CanonicalToken 协议 + CanonicalTokenizer + 资产化 + LinearTokenToLLM + 测试
- M6b：提编码器对齐质量（大 batch/分类辅助 loss/锚对比），L1 检索提升
- M6c：数据质量（弱模态、infra1/infra2 → 7 模态）

## 开放问题

- 资产化根目录：`datasets/mmfi/v5tokens/` 与 v5 数据集并列（vs 内嵌 v5 目录）。先定并列。
- CanonicalToken 是否存 `k` 变化的变体（半动态截取）：先存全量 `k=8`，截取在消费端做。
