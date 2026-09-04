# SensorBench 项目约定

- `STATUS.md` 是项目状态唯一入口。会话开始必须读它；涉及状态的事实变化（跑了脚本、完成/中断任务、产物变化）必须更新它。
- 事实层由 `python tools/project_status.py scan STATUS.md` 生成，**不手写**；判断层由 AI 维护；决策层只有人能拍板，AI 提议须用 `[提议]` 前缀，人确认后改 `[已定]`。
- 收工时若有卡点，在决策层留下"下一步行动"（AI 用 `[提议]`）。

## 并行开发纪律（多 agent 同仓协作）

多个 opencode 实例可能同时开发本仓库，任何会话**动手前**必须走完以下流程：

### 1. 开工三步（先认领后动手）
1. 读 `STATUS.md`（见上）。
2. 读 `.agents/claims/` 下所有 `*.json`，了解其他 agent 的占用范围。
3. 写自己的认领文件 `.agents/claims/<agent-id>.json`，字段：
   ```json
   {"agent_id": "...", "started": "<ISO时间>", "task": "<一句话任务>",
    "branch": "agent/<agent-id>", "worktree": ".worktrees/<agent-id>",
    "claimed_paths": ["framework/eval/...", "scripts/..."],
    "artifact_dir": "<短名，如 c2s0>", "job_prefix": "<slurm作业名前缀>",
    "status": "active"}
   ```

### 2. 代码隔离：一律在 worktree 里改
- **禁止在主检出（~/sensorbench 的 main 工作区）直接编辑代码**。主检出只做合并集成，归人/编排会话所有。
- 开工即建：
  ```bash
  git -C ~/sensorbench worktree add .worktrees/<agent-id> -b agent/<agent-id>
  ln -s ~/sensorbench/datasets .worktrees/<agent-id>/datasets   # 大数据文件不入 git，软链共用
  ```
- `claimed_paths` 与已有 claim 重叠 → 换任务，或到 STATUS.md 决策层用 `[提议]` 协调，禁止抢写。

### 3. 实验产物命名空间
- 结果只写 `results/<artifact_dir>/`，checkpoint 只写 `checkpoints_<artifact_dir>*`，脚本参数显式指定路径；**禁止**写裸 `results/` 根、`leaderboard_*.json`、无前缀 `checkpoints_*`。
- Slurm 作业名 = `<job_prefix>_<任务>`（如 `c2s0_eval`）；提交前先 `squeue -u $USER` + `nvidia-smi` 确认余量。

### 4. main 保护与合并
- 禁止直接向 main 提交/合并；只在 `agent/<agent-id>` 分支上原子提交。
- 合并回 main 由人（或明确授权的编排会话）**串行**执行，逐分支验证后合并。
- 会话收工：按 STATUS.md 协议更新状态，claim 置 `"status": "done"`；claim 超 7 天无新 commit 且非 done 视为失效，可接管（在原 claim 写 `taken_over_by`）。

### 5. 共享文件豁免
- `.agents/`、`.worktrees/` 不入库（已 gitignore）。
- `STATUS.md` / `AGENTS.md` / `docs/reports/` 是共享区：reports 文件名须带 `<artifact_dir>` 前缀避撞。
