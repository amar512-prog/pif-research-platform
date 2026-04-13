from __future__ import annotations

import re
from pathlib import Path


SECTION_ORDER = [
    "Executive Summary",
    "Research Question and Context",
    "Literature Review",
    "Indicator and Data Summary",
    "Methodology",
    "Key Findings",
    "Policy Recommendations",
    "Limitations",
    "References",
]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def extract_citation_ids(text: str) -> tuple[set[str], set[str]]:
    literature = set(re.findall(r"\[(S\d+)\]", text))
    analysis = set(re.findall(r"\[(A\d+)\]", text))
    return literature, analysis


def render_markdown_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join([header_row, separator] + body)


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def section_title_to_markdown(title: str) -> str:
    return f"## {title}"

