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
    import os
    f = tmp_path / "old.json"
    f.write_text("{}")
    old = datetime.now() - timedelta(hours=50)
    os.utime(f, (old.timestamp(), old.timestamp()))
    s = StatusScanner(tmp_path, {})
    r = s.check_path("old.json", {"fresh_hours": 24})
    assert r["note"].startswith("STALE")
    assert any("过期" in a for a in s.anomalies)


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
    (tmp_path / "ok.pt").write_text("x")
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
