# Project Status Hub 实施计划（STATUS.md 规范）

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 Status Hub 规范——实现 `project-status` 脚本（扫描产物生成 STATUS.md 事实层）、SensorBench 首个 STATUS.md 实例、项目级约定与 skill，验证成功后留待推广全局。

**Architecture:** STATUS.md 三层（事实层脚本生成 / 判断层 AI 维护 / 决策层人拍板）。脚本 `tools/project_status.py` 读 STATUS.md 头部 YAML front-matter（内嵌声明式配置），逐条检查里程碑证据、产物规则、异常模式、日志目录，把结果写入 `<!-- FACTS:START -->` / `<!-- FACTS:END -->` 标记之间。脚本通用、对项目零知识，配置随文档走。

**Tech Stack:** Python 3.12（复用 `/home/li/projects/holollm/.venv`，已有 pyyaml + pytest 9.1.1），stdlib only（re/subprocess/hashlib/argparse），pytest TDD。

---

## 环境事实（实施者必读）

- Python: `/home/li/projects/holollm/.venv/bin/python`
- pytest 配置: `pyproject.toml` 已设 `testpaths=["tests"]` + `pythonpath=["."]` → 测试可 `from tools.project_status import ...`
- 本计划开发脚本于 repo 内 `tools/project_status.py`（可测试、可 git 追踪）；skill 引用它。全局推广时才把脚本复制到共享位置（本计划**不做**）。
- SensorBench 现状（用于端到端验证的事实）：
  - `leaderboard_v2.json` **不存在**（v2 eval 中断于 07:44，`logs/eval_v2.log` 0 字节）→ 验证脚本能标黄 M4 + 报空日志
  - `docs/reports/robustness_v1_v2.md` 有 `_pending_` 残留 → 验证 anomaly_scan
- **不做**：修改全局 `~/.config/opencode/AGENTS.md`（设验证门槛，见 spec §5.1）；Web dashboard；git hook 自动触发；与 GSD 整合。

## 文件结构

```
tools/
├── __init__.py              # 使 tools 成为可导入包
└── project_status.py        # 通用扫描器（本计划核心产物）
tests/test_project_status.py # TDD 测试
STATUS.md                    # SensorBench 首个实例（三层结构）
AGENTS.md                    # 项目级约定（推广到全局前先在本地试行）
docs/superpowers/plans/2026-08-14-status-hub.md   # 本计划
```

---

### Task 1: 脚本骨架 + front-matter 解析

**Files:**
- Create: `tools/__init__.py`
- Create: `tests/test_project_status.py`（本 task 部分）
- Create: `tools/project_status.py`（骨架 + `parse_front_matter`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_project_status.py
import re
from pathlib import Path
from textwrap import dedent

import pytest

from tools.project_status import FACTS_END, FACTS_START, parse_front_matter, render_facts, write_facts


def make_status(tmp_path, front: str = "", body: str = "") -> Path:
    if not front:
        front = dedent("""\
            project: Demo
            goal: demo goal
            milestones: []
            artifacts: []
            anomaly_scan: []
            log_dirs: []
            """)
    p = tmp_path / "STATUS.md"
    p.write_text(f"---\n{front}\n---\n{body}")
    return p


def test_parse_front_matter(tmp_path):
    p = make_status(tmp_path, "project: Demo\ngoal: hello\n")
    cfg = parse_front_matter(p.read_text())
    assert cfg["project"] == "Demo"
    assert cfg["goal"] == "hello"


def test_parse_front_matter_missing():
    with pytest.raises(ValueError, match="front-matter"):
        parse_front_matter("# no front matter here\n")
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools'`（tools 目录未建）

- [ ] **Step 3: 创建包与最小骨架**

`tools/__init__.py` — 空文件。

`tools/project_status.py`（骨架，含 front-matter 解析）:

```python
#!/usr/bin/env python3
"""project-status: 生成并维护项目 STATUS.md 的事实层。

用法:
    python tools/project_status.py scan [STATUS.md] [--root PATH]

事实层由脚本根据 STATUS.md 头部 YAML front-matter 生成，
写入 <!-- FACTS:START --> / <!-- FACTS:END --> 之间。
退出码: 0 = 无异常; 1 = 有异常; 2 = 用法/文件错误。
"""
import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

FACTS_START = "<!-- FACTS:START -->"
FACTS_END = "<!-- FACTS:END -->"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def parse_front_matter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError("STATUS.md 缺少 YAML front-matter (--- ... ---)")
    if yaml is None:
        raise RuntimeError("需要 PyYAML (pip install pyyaml)")
    cfg = yaml.safe_load(m.group(1))
    return cfg if isinstance(cfg, dict) else {}
```

- [ ] **Step 4: 运行确认通过**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/__init__.py tools/project_status.py tests/test_project_status.py
git commit -m "feat(project-status): 脚本骨架 + front-matter 解析"
```

---

### Task 2: 路径检查规则（exists / nonempty / fresh_hours）

**Files:**
- Modify: `tools/project_status.py`（加 `StatusScanner` + `check_path`）
- Test: `tests/test_project_status.py`

- [ ] **Step 1: 写失败测试**

```python
def test_check_path_exists(tmp_path):
    from tools.project_status import StatusScanner
    (tmp_path / "a.pt").write_text("x")
    s = StatusScanner(tmp_path, {})
    r = s.check_path("a.pt", {})
    assert r["ok"] is True
    assert r["note"] == ""


def test_check_path_missing(tmp_path):
    from tools.project_status import StatusScanner
    s = StatusScanner(tmp_path, {})
    r = s.check_path("nope.pt", {})
    assert r["ok"] is False
    assert s.anomalies == ["缺失: nope.pt"]


def test_check_path_nonempty_empty(tmp_path):
    from tools.project_status import StatusScanner
    (tmp_path / "e.log").write_text("")
    s = StatusScanner(tmp_path, {})
    r = s.check_path("e.log", {"expect": "nonempty"})
    assert r["ok"] is True          # 文件存在
    assert r["note"] == "EMPTY"
    assert any("空文件" in a for a in s.anomalies)


def test_check_path_fresh_hours_stale(tmp_path, monkeypatch):
    from tools.project_status import StatusScanner
    from datetime import datetime, timedelta
    f = tmp_path / "old.json"
    f.write_text("{}")
    old = datetime.now() - timedelta(hours=50)
    import os
    os.utime(f, (old.timestamp(), old.timestamp()))
    s = StatusScanner(tmp_path, {})
    r = s.check_path("old.json", {"fresh_hours": 24})
    assert r["note"].startswith("STALE")
    assert any("过期" in a for a in s.anomalies)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: FAIL — `ImportError: cannot import name 'StatusScanner'`

- [ ] **Step 3: 实现 `StatusScanner.check_path`**

```python
def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


class StatusScanner:
    """对 front-matter 配置执行检查, 汇总异常清单（去重）。"""

    def __init__(self, root: Path, cfg: dict):
        self.root = root
        self.cfg = cfg or {}
        self.anomalies: list[str] = []

    def _flag(self, msg: str) -> None:
        """追加异常, 自动去重（同一文件可能同时出现在证据与 artifacts 中）。"""
        if msg not in self.anomalies:
            self.anomalies.append(msg)

    def check_path(self, rel: str, rules: dict) -> dict:
        p = self.root / rel
        if not p.exists():
            self._flag(f"缺失: {rel}")
            return {"path": rel, "ok": False, "mtime": None, "note": "MISSING"}
        note = ""
        st = p.stat()
        if rules.get("expect") == "nonempty" and st.st_size == 0:
            self._flag(f"空文件: {rel}")
            note = "EMPTY"
        fh = rules.get("fresh_hours")
        if fh:
            age_h = (datetime.now() - datetime.fromtimestamp(st.st_mtime)).total_seconds() / 3600
            if age_h > fh:
                self._flag(f"过期: {rel} ({age_h:.0f}h > {fh}h)")
                note = f"STALE {age_h:.0f}h"
        return {"path": rel, "ok": True, "mtime": _mtime(p), "note": note}
```

- [ ] **Step 4: 运行确认通过**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/project_status.py tests/test_project_status.py
git commit -m "feat(project-status): 路径检查规则 exists/nonempty/fresh_hours"
```

---

### Task 3: 聚合——里程碑 / 产物 / 异常模式 / 日志

**Files:**
- Modify: `tools/project_status.py`（`milestones`/`artifacts`/`scan_patterns`/`logs`）
- Test: `tests/test_project_status.py`

- [ ] **Step 1: 写失败测试**

```python
def test_milestones_done(tmp_path):
    from tools.project_status import StatusScanner
    (tmp_path / "m.json").write_text("{}")
    (tmp_path / "o.pt").write_text("x")
    s = StatusScanner(tmp_path, {"milestones": [
        {"id": "M1", "name": "n", "evidence": ["m.json", "o.pt"]}]})
    rows = s.milestones()
    assert rows[0]["done"] is True
    assert rows[0]["missing"] == []


def test_milestones_incomplete(tmp_path):
    from tools.project_status import StatusScanner
    s = StatusScanner(tmp_path, {"milestones": [
        {"id": "M4", "name": "n", "evidence": ["ok.pt", "missing.json"]}]})
    rows = s.milestones()
    assert rows[0]["done"] is False
    assert rows[0]["missing"] == ["missing.json"]
    assert "缺失: missing.json" in s.anomalies


def test_artifacts_named(tmp_path):
    from tools.project_status import StatusScanner
    (tmp_path / "lb.json").write_text("{}")
    s = StatusScanner(tmp_path, {"artifacts": [
        {"name": "v1 leaderboard", "path": "lb.json", "expect": "nonempty"}]})
    rows = s.artifacts()
    assert rows[0]["name"] == "v1 leaderboard"
    assert rows[0]["ok"] is True


def test_scan_patterns_hit(tmp_path):
    from tools.project_status import StatusScanner
    r = tmp_path / "report.md"
    r.write_text("row: _pending_\n")
    s = StatusScanner(tmp_path, {"anomaly_scan": [
        {"pattern": "_pending_", "path": "report.md"}]})
    s.scan_patterns()
    assert any("_pending_" in a and "report.md" in a for a in s.anomalies)


def test_logs_lists_files(tmp_path):
    from tools.project_status import StatusScanner
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "a.log").write_text("x")
    s = StatusScanner(tmp_path, {"log_dirs": ["logs"]})
    rows = s.logs()
    assert rows[0]["path"] == "logs/a.log"
    assert rows[0]["size"] == 1


def test_logs_missing_dir(tmp_path):
    from tools.project_status import StatusScanner
    s = StatusScanner(tmp_path, {"log_dirs": ["logs"]})
    assert s.logs() == []
    assert any("日志目录缺失" in a for a in s.anomalies)


def test_anomalies_dedup(tmp_path):
    from tools.project_status import StatusScanner
    s = StatusScanner(tmp_path, {
        "milestones": [{"id": "M4", "name": "n", "evidence": ["missing.json"]}],
        "artifacts": [{"name": "a", "path": "missing.json"}]})
    s.milestones()
    s.artifacts()
    assert s.anomalies.count("缺失: missing.json") == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: FAIL — `AttributeError: 'StatusScanner' object has no attribute 'milestones'`

- [ ] **Step 3: 实现聚合方法**

```python
    def milestones(self) -> list[dict]:
        rows = []
        for m in self.cfg.get("milestones", []):
            evs = [self.check_path(e, {}) for e in m.get("evidence", [])]
            missing = [e["path"] for e in evs if not e["ok"]]
            rows.append({"id": m.get("id"), "name": m.get("name"),
                         "done": not missing, "missing": missing, "evidence": evs})
        return rows

    def artifacts(self) -> list[dict]:
        rows = []
        for a in self.cfg.get("artifacts", []):
            rules = {k: a[k] for k in ("expect", "fresh_hours") if k in a}
            r = self.check_path(a["path"], rules)
            r["name"] = a.get("name", a["path"])
            rows.append(r)
        return rows

    def scan_patterns(self) -> None:
        for s in self.cfg.get("anomaly_scan", []):
            p = self.root / s["path"]
            if not p.exists():
                self._flag(f"anomaly 目标缺失: {s['path']}")
                continue
            text = p.read_text(errors="replace")
            hits = len(re.findall(s["pattern"], text))
            if hits:
                self._flag(f"发现 {hits} 处 `{s['pattern']}`: {s['path']}")

    def logs(self) -> list[dict]:
        rows = []
        for d in self.cfg.get("log_dirs", []):
            dd = self.root / d
            if not dd.is_dir():
                self._flag(f"日志目录缺失: {d}")
                continue
            for f in sorted(dd.iterdir()):
                if not f.is_file():
                    continue
                st = f.stat()
                rows.append({"path": f.relative_to(self.root).as_posix(),
                             "size": st.st_size,
                             "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")})
        return rows
```

- [ ] **Step 4: 运行确认通过**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: PASS（13 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/project_status.py tests/test_project_status.py
git commit -m "feat(project-status): 里程碑/产物/异常模式/日志聚合"
```

---

### Task 4: 事实层渲染 + 写入标记区 + CLI + 退出码

**Files:**
- Modify: `tools/project_status.py`（`render_facts` / `write_facts` / `main`）
- Test: `tests/test_project_status.py`

- [ ] **Step 1: 写失败测试**

```python
def test_render_contains_sections(tmp_path):
    from tools.project_status import StatusScanner, render_facts
    (tmp_path / "ok.pt").write_text("x")
    cfg = {"project": "Demo", "goal": "g",
           "milestones": [{"id": "M1", "name": "n", "evidence": ["ok.pt"]}],
           "artifacts": [{"name": "a", "path": "ok.pt"}],
           "log_dirs": ["logs"], "anomaly_scan": []}
    out = render_facts(StatusScanner(tmp_path, cfg))
    assert "事实层" in out
    assert "M1" in out
    assert "✅ DONE" in out


def test_render_lists_anomaly(tmp_path):
    from tools.project_status import StatusScanner, render_facts
    cfg = {"project": "P", "goal": "g", "milestones": [],
           "artifacts": [{"name": "a", "path": "missing.json"}], "anomaly_scan": [], "log_dirs": []}
    out = render_facts(StatusScanner(tmp_path, cfg))
    assert "⚠ 异常清单" in out
    assert "缺失: missing.json" in out


def test_write_facts_inserts_after_frontmatter(tmp_path):
    from tools.project_status import write_facts
    p = make_status(tmp_path, "project: Demo\ngoal: g\n", body="\n## 判断层\n")
    write_facts(p, "FACTS_BLOCK")
    text = p.read_text()
    assert FACTS_START in text and "FACTS_BLOCK" in text
    assert text.index("---\n") < text.index(FACTS_START) < text.index("## 判断层")


def test_write_facts_replaces_between_markers(tmp_path):
    from tools.project_status import write_facts
    p = make_status(tmp_path, "project: Demo\ngoal: g\n", body=f"\n{FACTS_START}\nOLD\n{FACTS_END}\n## 判断层\n")
    write_facts(p, "NEW")
    text = p.read_text()
    assert "OLD" not in text
    assert "NEW" in text
    assert text.index(FACTS_START) < text.index("NEW") < text.index(FACTS_END)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: FAIL — `ImportError: cannot import name 'render_facts'`

- [ ] **Step 3: 实现渲染与写入**

`render_facts` 会调用 `scanner.protocol_fingerprint()` 与 `scanner.git_activity()`（Task 5 才实现），故本步同时加两个**空方法桩**：

```python
    def git_activity(self, n=5):          # 桩, Task 5 实现
        return []

    def protocol_fingerprint(self):        # 桩, Task 5 实现
        return None
```

```python
def render_facts(scanner: "StatusScanner") -> str:
    cfg = scanner.cfg
    L = ["## ⚙ 事实层（脚本生成 — `project-status scan` 维护, 勿手写）", ""]
    L.append(f"- 生成时间: {now_str()}")
    L.append(f"- 项目: **{cfg.get('project', '?')}** — {cfg.get('goal', '')}")

    fp = scanner.protocol_fingerprint()
    if fp:
        L.append(f"- 协议指纹: `{fp}`")
    L.append("")
    L.append("### 里程碑")
    L.append("| id | 状态 | 名称 | 证据缺口 |")
    L.append("|----|------|------|----------|")
    for m in scanner.milestones():
        st = "✅ DONE" if m["done"] else "⚠ INCOMPLETE"
        gap = ", ".join(m["missing"]) if m["missing"] else "—"
        L.append(f"| {m['id']} | {st} | {m['name']} | {gap} |")

    arts = scanner.artifacts()
    if arts:
        L.append("")
        L.append("### 关键产物")
        L.append("| 名称 | 路径 | mtime | 状态 |")
        L.append("|------|------|-------|------|")
        for a in arts:
            L.append(f"| {a['name']} | `{a['path']}` | {a['mtime'] or '—'} | {a['note'] or 'ok'} |")

    logs = scanner.logs()
    if logs:
        L.append("")
        L.append("### 近期日志")
        L.append("| 文件 | 大小 | mtime |")
        L.append("|------|------|-------|")
        for l in logs[-8:]:
            L.append(f"| `{l['path']}` | {l['size']} | {l['mtime']} |")

    git = scanner.git_activity()
    if git:
        L.append("")
        L.append("### 近期提交")
        for line in git:
            L.append(f"- `{line}`")

    L.append("")
    L.append("### ⚠ 异常清单")
    if scanner.anomalies:
        for a in scanner.anomalies:
            L.append(f"- [ ] {a}")
    else:
        L.append("- 无")
    return "\n".join(L) + "\n"


def write_facts(path: Path, facts_block: str) -> None:
    text = path.read_text()
    block = f"{FACTS_START}\n{facts_block}\n{FACTS_END}"
    start = text.find(FACTS_START)
    end = text.find(FACTS_END)
    if start != -1 and end != -1 and end > start:
        text = text[:start] + block + text[end + len(FACTS_END):]
    else:
        m = re.match(r"^---\n.*?\n---\n", text, re.S)
        if m:
            text = text[: m.end()] + block + "\n" + text[m.end():]
        else:
            text = block + "\n" + text
    path.write_text(text)
```

- [ ] **Step 4: 写 CLI 入口测试（失败）**

```python
def test_cli_end_to_end_anomalies(tmp_path, capsys):
    from tools.project_status import main
    (tmp_path / "logs").mkdir()
    p = make_status(tmp_path, dedent("""\
        project: Demo
        goal: g
        milestones:
          - id: M4
            name: n
            evidence: [ok.pt, missing.json]
        artifacts:
          - name: lb
            path: missing.json
        anomaly_scan: []
        log_dirs: [logs]
        """))
    code = main(["scan", str(p), "--root", str(tmp_path)])
    assert code == 1
    assert "缺失: missing.json" in capsys.readouterr().out
    assert "M4" in p.read_text()


def test_cli_clean_exit_zero(tmp_path):
    from tools.project_status import main
    (tmp_path / "ok.pt").write_text("x")
    p = make_status(tmp_path, dedent("""\
        project: Demo
        goal: g
        milestones:
          - id: M1
            name: n
            evidence: [ok.pt]
        artifacts: []
        anomaly_scan: []
        log_dirs: []
        """))
    code = main(["scan", str(p), "--root", str(tmp_path)])
    assert code == 0
    assert "✅ DONE" in p.read_text()
```

- [ ] **Step 5: 实现 CLI**

```python
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="project-status 扫描器: 生成 STATUS.md 事实层")
    ap.add_argument("cmd", choices=["scan"])
    ap.add_argument("status", nargs="?", default="STATUS.md")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    spath = root / args.status
    if not spath.exists():
        print(f"错误: {spath} 不存在", file=sys.stderr)
        return 2
    cfg = parse_front_matter(spath.read_text())
    scanner = StatusScanner(root, cfg)
    scanner.scan_patterns()
    block = render_facts(scanner)
    write_facts(spath, block)
    print(f"[project-status] 事实层已更新: {spath.name}")
    if scanner.anomalies:
        print(f"[project-status] ⚠ {len(scanner.anomalies)} 项异常:")
        for a in scanner.anomalies:
            print(f"    - {a}")
        return 1
    print("[project-status] ✓ 无异常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 运行全部测试确认通过**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: PASS（19 passed）

- [ ] **Step 7: 提交**

```bash
git add tools/project_status.py tests/test_project_status.py
git commit -m "feat(project-status): 事实层渲染 + FACTS 标记写入 + CLI + 退出码"
```

---

### Task 5: git 近期活动 + 协议指纹

**Files:**
- Modify: `tools/project_status.py`（`git_activity` / `protocol_fingerprint`）
- Test: `tests/test_project_status.py`

- [ ] **Step 0: 确认桩方法已存在**

Task 4 Step 3 已加空方法桩。若上一步被跳过导致 `AttributeError`，先补桩（见 Task 4 Step 3）：

```python
    def git_activity(self, n=5):
        return []

    def protocol_fingerprint(self):
        return None
```

- [ ] **Step 1: 写失败测试**

```python
def test_protocol_fingerprint(tmp_path):
    from tools.project_status import StatusScanner
    (tmp_path / "protocol.json").write_bytes(b'{"p":1}')
    s = StatusScanner(tmp_path, {"protocol_fingerprint": "protocol.json"})
    fp = s.protocol_fingerprint()
    assert fp and len(fp) == 12


def test_protocol_fingerprint_missing(tmp_path):
    from tools.project_status import StatusScanner
    s = StatusScanner(tmp_path, {"protocol_fingerprint": "nope.json"})
    assert s.protocol_fingerprint() is None
    assert any("协议指纹缺失" in a for a in s.anomalies)


def test_git_activity_lists(tmp_path):
    import subprocess
    from tools.project_status import StatusScanner
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "feat: first"], cwd=tmp_path, check=True)
    s = StatusScanner(tmp_path, {})
    lines = s.git_activity(3)
    assert any("first" in l for l in lines)
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: FAIL — `test_protocol_fingerprint` 等断言失败（桩返回 None/[]）

- [ ] **Step 3: 实现**

```python
    def git_activity(self, n=5) -> list[str]:
        try:
            out = subprocess.run(["git", "-C", str(self.root), "log", "--oneline", f"-{n}"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                return []
            return [l for l in out.stdout.splitlines() if l]
        except (subprocess.SubprocessError, FileNotFoundError):
            return []

    def protocol_fingerprint(self) -> str | None:
        rel = self.cfg.get("protocol_fingerprint")
        if not rel:
            return None
        p = self.root / rel
        if not p.exists():
            self._flag(f"协议指纹缺失: {rel}")
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()[:12]
```

- [ ] **Step 4: 运行全部测试确认通过**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest tests/test_project_status.py -v`
Expected: PASS（22 passed）

- [ ] **Step 5: 提交**

```bash
git add tools/project_status.py tests/test_project_status.py
git commit -m "feat(project-status): git 近期活动 + 协议指纹"
```

---

### Task 6: 初始化 SensorBench 的 STATUS.md + 项目 AGENTS.md

**Files:**
- Create: `STATUS.md`
- Create: `AGENTS.md`

- [ ] **Step 1: 写 `STATUS.md`（三层结构 + front-matter）**

```markdown
---
project: SensorBench
goal: 数据/模型解耦的跨模态融合基准框架，以缺模态鲁棒性（Robustness Score）量化数据质量
milestones:
  - id: M1
    name: 项目骨架 + 规范格式 + MMFi ingest 管线
    evidence: [framework/dataset/sample.py, datasets/mmfi/v1/meta.json, datasets/mmfi/v2/meta.json]
  - id: M2
    name: SensorModel 协议 + token_fusion/late_fusion
    evidence: [framework/models/base.py, framework/models/token_fusion.py, framework/models/late_fusion.py]
  - id: M3
    name: 评测 harness + 排行榜
    evidence: [protocol.json, framework/harness/evaluate.py, leaderboard_v1.json]
  - id: M4
    name: v2 数据改进闭环（清洗 v2 数据 → 重训 → v2 评测对比）
    evidence: [datasets/mmfi/v2/meta.json, checkpoints_v2/late_fusion_seed2.pt, leaderboard_v2.json]
  - id: M5
    name: 文档与可复现性
    evidence: [README.md, docs/reports/robustness_v1_v2.md]
artifacts:
  - name: v1 leaderboard
    path: leaderboard_v1.json
    expect: nonempty
  - name: v2 leaderboard
    path: leaderboard_v2.json
    expect: nonempty
  - name: v2 eval 日志
    path: logs/eval_v2.log
    expect: nonempty
  - name: 评测协议
    path: protocol.json
    expect: nonempty
protocol_fingerprint: protocol.json
anomaly_scan:
  - pattern: "_pending_"
    path: docs/reports/robustness_v1_v2.md
log_dirs: [logs]
---

# STATUS — SensorBench

> 项目状态唯一入口。事实层由脚本生成（勿手写）；判断层由 AI 会话维护；决策层由人拍板（AI 提议用 `[提议]` 前缀，人确认后改 `[已定]`）。

<!-- FACTS:START -->
<!-- FACTS:END -->

## 🧠 判断层

- 当前阶段：_pending_（首次 scan 后填写，见 Task 8）
- 发现与结论：_pending_
- 卡点 / 风险：_pending_

## 🗂 决策层

- [ ] 下一步行动：_pending_（收工时 AI 填 `[提议]`）
```

> 注：`_pending_` 出现在 STATUS.md 自身属正常占位（不是事实层的 anomaly 目标文件）。

- [ ] **Step 2: 写 `AGENTS.md`（项目级试行约定，不碰全局）**

```markdown
# SensorBench 项目约定

- `STATUS.md` 是项目状态唯一入口。会话开始必须读它；涉及状态的事实变化（跑了脚本、完成/中断任务、产物变化）必须更新它。
- 事实层由 `python tools/project_status.py scan STATUS.md` 生成，**不手写**；判断层由 AI 维护；决策层只有人能拍板，AI 提议须用 `[提议]` 前缀，人确认后改 `[已定]`。
- 收工时若有卡点，在决策层留下"下一步行动"（AI 用 `[提议]`）。
```

- [ ] **Step 3: 运行一次 scan 验证实例**

Run: `cd /home/li/projects/sensorbench && /home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md`
Expected: 退出码 1；输出含 `⚠ 3 项异常`（去重后，`leaderboard_v2.json` 虽同时在 M4 证据与 artifacts 中，缺失只报一次）：缺失 leaderboard_v2.json、空文件 logs/eval_v2.log、发现 8 处 `_pending_`（在 robustness_v1_v2.md）。`STATUS.md` 事实层已生成：M1/M2/M3/M5 ✅ DONE，M4 ⚠ INCOMPLETE（缺 leaderboard_v2.json）。

- [ ] **Step 4: 提交**

```bash
git add STATUS.md AGENTS.md
git commit -m "feat(status-hub): SensorBench STATUS.md 实例 + 项目级约定 AGENTS.md"
```

---

### Task 7: 创建 `project-status` skill

**Files:**
- Create: `~/.config/opencode/skills/project-status/SKILL.md`
- Create: `~/.config/opencode/skills/project-status/STATUS.template.md`

- [ ] **Step 1: 写 `SKILL.md`**

```markdown
---
name: project-status
description: 维护项目 STATUS.md（状态唯一入口）。当用户询问项目现状/进度/卡点、需要生成或刷新 STATUS.md、或要按 Status Hub 规范初始化新项目时使用。提供 scan/review/update/onboarding 四个流程。
---

# project-status

维护 STATUS.md：事实层脚本生成、判断层 AI 维护、决策层人拍板。

## 前置

- 脚本: 项目内 `tools/project_status.py`（本仓库首个实例）。推广到全局后改用共享脚本。
- Python: `/home/li/projects/holollm/.venv/bin/python`

## 流程

### scan — 刷新事实层

运行 `python tools/project_status.py scan STATUS.md`（在项目根目录）。
事实层会写回 `<!-- FACTS:START -->` / `<!-- FACTS:END -->` 之间；退出码 1 = 有异常清单。**不手写事实层。**

### review — 读状态找卡点

1. 读 `STATUS.md`（含事实层）与 `git log --oneline -10`。
2. 定位异常清单每一项对应的根因（如 v2 eval 中断 → logs/eval_v2.log 0 字节）。
3. 报告：当前在哪个里程碑、完成度、卡点、下一步。

### update — 更新判断层 / 提议决策层

1. 事实变化（任务完成/中断、产物变动）时，先跑 `scan` 刷新事实层。
2. 更新判断层（当前阶段/发现与结论/卡点风险）——只写有事实依据的结论。
3. 在决策层写"下一步行动"，用 `[提议]` 前缀；不得冒充已定决策。

### onboarding — 初始化新项目 STATUS.md

> **注意**：脚本目前仅存在于本仓库（`tools/project_status.py`）。全局推广门槛（spec §5.1）通过前，onboarding 只能在本仓库运行；推广后脚本随 skill 分发，其他项目方可使用。

1. 复制 `STATUS.template.md` 到项目根目录为 `STATUS.md`。
2. 引导填写 front-matter：project / goal / milestones（每个带 evidence 路径）/ artifacts（expect/fresh_hours）/ protocol_fingerprint / anomaly_scan / log_dirs。
3. 跑 `scan` 验证事实层生成，再填判断层与决策层占位。
```

- [ ] **Step 2: 写 `STATUS.template.md`**

```markdown
---
project: <项目名>
goal: <一句话目标>
milestones: []
artifacts: []
protocol_fingerprint:
anomaly_scan: []
log_dirs: []
---

# STATUS — <项目名>

> 项目状态唯一入口。事实层由脚本生成（勿手写）；判断层由 AI 会话维护；决策层由人拍板（AI 提议用 `[提议]` 前缀，人确认后改 `[已定]`）。

<!-- FACTS:START -->
<!-- FACTS:END -->

## 🧠 判断层

- 当前阶段：
- 发现与结论：
- 卡点 / 风险：

## 🗂 决策层

- [ ] 下一步行动：
```

- [ ] **Step 3: 验证 skill 可被 opencode 发现**

Run: `ls ~/.config/opencode/skills/project-status/`
Expected: `SKILL.md` 与 `STATUS.template.md` 存在。

- [ ] **Step 4: 提交（skill 在仓库外，不入 git）**

Run: `git -C /home/li/projects/sensorbench status --short`
Expected: 干净（skill 文件位于 home，不在仓库内，无需提交）。

---

### Task 8: 端到端验证 + 收尾

**Files:**
- Modify: `STATUS.md`（判断层/决策层填真实状态）

- [ ] **Step 1: 全量测试**

Run: `/home/li/projects/holollm/.venv/bin/python -m pytest -q`
Expected: 原有用例 + 22 个 project-status 用例全部 PASS。

- [ ] **Step 2: 再跑一次 scan，确认事实层幂等（重复执行结果一致）**

Run: `/home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md && /home/li/projects/holollm/.venv/bin/python tools/project_status.py scan STATUS.md`
Expected: 两次输出一致，事实层内容稳定（无重复累积）。

- [ ] **Step 3: 填真实状态到判断层/决策层**

把 Task 6 Step 3 的 scan 结果固化为人工/AI 可读状态，例如：

```markdown
## 🧠 判断层

- 当前阶段：**M4（v2 数据改进闭环）进行中**——v1 全部完成，v2 训练完成、评估中断。
- 发现与结论：v1 显示 mmWave 是主导传感器（miss-mmwave 降幅最大），token_fusion 不如 late_fusion 稳定。
- 卡点 / 风险：v2 eval 于 08-14 07:44 中断（eval_v2.log 0 字节、leaderboard_v2.json 未生成）；报告 robustness_v1_v2.md 的 v2 结果 pending。

## 🗂 决策层

- [x] 下一步行动（已定）：重跑 v2 评估，生成 leaderboard_v2.json，更新 robustness 报告。
- [ ] `[提议]`：v2 结果出来后，把 v1 vs v2 对比写进报告并归档 leaderboard_v2.json。
```

- [ ] **Step 4: 最终提交**

```bash
git add STATUS.md
git commit -m "feat(status-hub): 判断层/决策层填入真实状态（v2 eval 中断为当前卡点）"
```

- [ ] **Step 5: 验证收尾**

Run: `git -C /home/li/projects/sensorbench log --oneline -12`
Expected: 本计划 7 次提交按序存在（Task 1–6、8）。skill 在 `~/.config/opencode/skills/project-status/` 就绪。推广全局的门槛（≥2 次 scan + 判断层如实反映中断任务）在本计划 Task 6/8 的 scan 后**部分达成**，剩一次后续 scan 满足后即可按 spec §5.1 推广。

---

## 不做（边界）

- 不改 `~/.config/opencode/AGENTS.md`（推广门槛未到）
- 不做 Web dashboard、不做 git hook 自动 scan
- 不整合 GSD `.planning/`（先独立验证）
- 脚本不引入 pyyaml 之外的依赖（yaml 已在 venv）
