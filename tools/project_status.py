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
