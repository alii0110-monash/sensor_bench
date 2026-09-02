# Project Status Hub：实时项目现状文档规范（设计）

日期：2026-08-14
状态：Draft（待评审）
范围：跨项目通用约定，SensorBench 为首个实践

## 1. 背景与问题

- 现有规划文档（superpowers 的 design/plan）是**静态快照**，不随代码演进，无法反映真实状态。
- 报告的 `_pending_` 项无人补、中断的任务（如 v2 eval）留空日志，只能靠 git log / 手工翻产物才知道项目到哪了。
- 没有一个"打开就知道项目当前状态"的单一入口。

## 2. 目标与决策

**Status Hub**：项目根目录一个 `STATUS.md`，作为项目状态唯一入口，让人和 AI 助手快速看懂——当前状态、卡在哪、下一步是什么。

核心决策（已与用户确认）：

| # | 决策点 | 选择 |
|---|--------|------|
| 1 | 用途 | **状态入口（Status Hub）**——现状 / 卡点 / 下一步，非演进档案、非项目管理指挥台 |
| 2 | 维护模型 | **混合**：脚本管"可验证的事实"，AI 管"需要推理的判断"，人拍板"决策" |
| 3 | 落地机制 | **AGENTS.md 强制约定 + skill 提供流程** |
| 4 | 脚本配置 | **内嵌声明式配置**：YAML front-matter 写在 STATUS.md 头部 |

**命名统一**：脚本与 skill 统一用 `project-status`（脚本入口 `project-status scan`）。

## 3. 文档结构：三层

```
STATUS.md
├── ⚙ 事实层（脚本生成，机器可验证）
│   ├── 项目：名称 / 一句话目标 / 协议指纹
│   ├── 里程碑 & 目标：每个目标 + 完成状态 + 证据（路径/时间戳/规则）
│   ├── 关键产物：datasets / checkpoints / leaderboard / reports（存在性 + 更新时间）
│   ├── 近期活动：最近 N 次 commit、最近运行的任务
│   └── 异常清单：脚本可抓的问题（产物缺失、空日志、_pending_ 残留、证据不全）
│
├── 🧠 判断层（AI 会话维护：每次开工读 / 收工更新）
│   ├── 当前阶段：在哪个里程碑、完成度多少
│   ├── 发现与结论：如"mmWave 是主导传感器""token_fusion 不稳定"
│   └── 卡点 / 风险：如"v2 eval 中断，需重跑"
│
└── 🗂 决策层（人工拍板）
    ├── 下一步行动（明确的人/任务）
    └── 待确认项 / 暂停项
```

**三条铁律**：

1. **事实层不许手写**——由脚本生成，防止过期与撒谎。
2. **判断层不许凭空编造**——AI 的结论必须基于事实层 + git 历史 + 日志。
3. **决策层只有人能拍板**——AI 可提议下一步，但必须用 `[提议]` 前缀标记，人确认后改为 `[已定]`，不能冒充已定决策。

## 4. 脚本设计：`project_status`

通用扫描器（对项目零知识，放 `~/bin` 或 skill 内），配置从 STATUS.md front-matter 读。

### 4.1 front-matter schema

```yaml
---
project: SensorBench
goal: 数据/模型解耦的跨模态融合基准框架，以缺模态鲁棒性量化数据质量
milestones:                      # 每个里程碑 = 一个目标 + 可验证的证据
  - id: M1
    name: MMFi ingest 管线
    evidence: [datasets/mmfi/v1/meta.json, datasets/mmfi/v2/meta.json]
  - id: M4
    name: v2 数据改进闭环
    evidence: [checkpoints_v2/late_fusion_seed2.pt, leaderboard_v2.json]
artifacts:                       # 关键产物，报告存在性 + 更新时间
  - name: v1 leaderboard
    path: leaderboard_v1.json
    expect: nonempty
  - name: v2 eval 日志
    path: logs/eval_v2.log
    expect: nonempty          # 0 字节 → 脚本自动标 ⚠
  - name: 评测协议
    path: protocol.json
    expect: nonempty
    fresh_hours: 720        # 协议指纹锚点，30 天未动即提示复查
protocol_fingerprint: protocol.json  # 哈希锚定协议版本
anomaly_scan:                    # 全文扫这些模式，命中即标记
  - pattern: "_pending_"
    path: docs/reports/robustness_v1_v2.md
log_dirs: [logs/]                # 扫描最近活动日志 + mtime
---
```

### 4.2 检查规则类型

| 规则 | 说明 |
|------|------|
| `exists`（默认） | 路径存在性 |
| `expect: nonempty` | 文件大小 > 0 |
| `fresh_hours` | 证据/产物新鲜度阈值（如 `fresh_hours: 24`），mtime 超过阈值 → 标 ⚠"过期"；不设置则不检查 |
| `anomaly_scan.pattern` | 指定文件中扫正则，命中即入异常清单 |
| `log_dirs` | 列出最近活动日志及 mtime |

### 4.3 脚本行为

- 读 front-matter → 逐条检查 → 生成事实层。
- 只写入文档 `<!-- FACTS:START -->` / `<!-- FACTS:END -->` 标记之间，**绝不触碰判断层与决策层**。
- 产出**异常清单**：缺失产物、空日志、过期产物、`_pending_` 残留、里程碑证据不全。异常清单是给 AI 和人的"该干活了"信号。

### 4.4 协议指纹

事实层的"协议指纹" = `protocol.json` 的 SHA-256（或文件 mtime），用于锁定评测协议版本，防止排行榜跑在未归档的协议上。无 protocol.json 的项目可省略。

### 4.4 效果示例

M4 的 `leaderboard_v2.json` 不存在 → 脚本自动将 M4 标黄并列出缺失证据；`docs/reports/robustness_v1_v2.md` 中 `_pending_` 残留 → 自动报警。

## 5. 落地机制

### 5.1 全局 AGENTS.md 约定（强制力，**分阶段启用**）

约定内容如下，但**当前仅在 SensorBench 项目内试行**（通过项目级约定 + skill），待验证成功后再写入 `~/.config/opencode/AGENTS.md` 推广到所有项目：

1. 项目根目录的 `STATUS.md` 是项目状态唯一入口。会话开始必须读它；涉及状态的事实变化（跑了脚本、完成/中断任务、产物变化）必须更新它。
2. 事实层由 `project-status scan` 生成，不手写；判断层由 AI 维护；决策层只有人能拍板，AI 提议须用 `[提议]` 前缀。
3. 收工时若有卡点，在决策层留下"下一步行动"（AI 用 `[提议]`）。

**推广门槛（Gate）**：SensorBench 上 STATUS.md 连续正常使用（≥2 次 scan + 判断层如实反映中断任务）后，将上述约定追加到 `~/.config/opencode/AGENTS.md`，并把本项目作为首个已接入案例。

### 5.2 skill：`project-status`（可复现流程）

- `scan`：跑 `project-status scan` 脚本刷新事实层
- `review`：读 STATUS.md + 事实层，判断当前阶段、找卡点、定位异常清单对应原因
- `update`：更新判断层（当前阶段/发现/卡点），把下一步提议写进决策层（`[提议]` 前缀）
- `onboarding`：新项目初始化 STATUS.md（生成模板 + 引导填 front-matter）

## 6. 首个实践：SensorBench

- 为 SensorBench 创建 `STATUS.md`，front-matter 填 5 个里程碑（M1–M5）的证据。
- M2/M3/M5 的证据路径在**实施计划阶段**从 `2026-08-13-sensorbench-design.md` §9 里程碑定义中枚举（如 M2=`framework/models/*`、M3=`leaderboard_v1.json`、M5=`README.md`+归档），不留给实现时自行发挥。
- 跑一次 `scan`，把真实状态（v2 eval 中断、报告 `_pending_`）固化为"卡点 + 下一步"。
- 作为模板参照，验证流程是否顺滑，再推广到其他项目。

## 7. 明确不做（延期）

| 延期项 | 原因 |
|--------|------|
| Web / 可视化 dashboard | YAGNI，markdown 足够 |
| git hook 自动触发 scan | 复杂度高，先手动/会话驱动 |
| 与 GSD `.planning/` 体系整合 | 先独立跑通，验证价值后再谈 |
| 演进档案（Living Log） | 用途定为状态入口，不混入 |

## 8. 待确认项

- skill / 脚本存放位置：候选 `~/.config/opencode/skills/`（与 gsd-* 同处，易被 opencode 发现）——实施前确认
- 脚本实现语言：Python（通用、跨项目）
- ~~模板文件名~~（已定：`STATUS.md`）
- 全局 AGENTS.md 写入时机：以 §5.1 的推广门槛为准，验证成功后再推广
