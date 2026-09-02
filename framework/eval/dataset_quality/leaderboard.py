"""Cross-version aggregation and markdown rendering."""
from __future__ import annotations

import json
from typing import Dict, List


def load_reports(paths: List[str]) -> Dict[str, Dict]:
    out = {}
    for p in paths:
        with open(p) as f:
            rep = json.load(f)
        out[rep["dataset"]] = rep
    return out


def aggregate_quality(reports: Dict[str, Dict]) -> Dict[str, float]:
    return {k: v["quality"] for k, v in reports.items()}


def render_markdown(reports: Dict[str, Dict]) -> str:
    """Render a Markdown table comparing versions."""
    lines = ["# Dataset Quality Leaderboard", ""]
    header = "| dataset | InfoScore | CompactScore | CleanScore | Quality |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for name, rep in reports.items():
        lines.append(
            f"| {name} | {rep['info']['InfoScore']:.3f} | "
            f"{rep['compact']['CompactScore']:.3f} | "
            f"{rep['clean']['CleanScore']:.3f} | "
            f"{rep['quality']:.3f} |"
        )
    return "\n".join(lines) + "\n"