#!/usr/bin/env python3
"""Audit publication artifacts for overclaiming language."""

from __future__ import annotations

import re
from pathlib import Path

AUDIT_DIRECTORIES = (
    Path("paper/arxiv"),
    Path("paper/tmlr"),
    Path("paper/joss"),
    Path("docs/publication"),
)

TEXT_SUFFIXES = {".md", ".tex", ".txt", ".rst"}

RISKY_PHRASES = (
    "Kalman beats mean",
    "Kalman improves retrieval",
    "Kalman outperforms",
    "proved superior",
    "state of the art",
    "statistically significant improvement",
    "robustly improves",
)

ALLOWED_CONTEXT_PATTERNS = (
    re.compile(r"\bavoid\b", re.IGNORECASE),
    re.compile(r"\bprohibited\b", re.IGNORECASE),
    re.compile(r"\bsuch\s+as\b", re.IGNORECASE),
    re.compile(r"\boverclaiming\b", re.IGNORECASE),
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bunsupported\b", re.IGNORECASE),
    re.compile(r"\bcannot\s+support\b", re.IGNORECASE),
    re.compile(r"\bfails?\s+to\s+support\b", re.IGNORECASE),
    re.compile(r"\bhypothesis\s+was\s+not\s+supported\b", re.IGNORECASE),
)


def _iter_candidate_files(repo_root: Path):
    for rel_dir in AUDIT_DIRECTORIES:
        directory = repo_root / rel_dir
        if not directory.exists():
            continue
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in TEXT_SUFFIXES:
                yield file_path


def _line_has_allowed_context(line: str, recent_context: str = "") -> bool:
    context = f"{recent_context} {line}".strip()
    return any(pattern.search(context) for pattern in ALLOWED_CONTEXT_PATTERNS)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[tuple[str, int, str, str]] = []

    for file_path in _iter_candidate_files(repo_root):
        rel_path = str(file_path.relative_to(repo_root))
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for lineno, line in enumerate(lines, start=1):
            for phrase in RISKY_PHRASES:
                if not re.search(re.escape(phrase), line, re.IGNORECASE):
                    continue
                context_start = max(0, lineno - 6)
                recent_context = " ".join(lines[context_start : lineno - 1])
                if line.strip().startswith('- "') and re.search(
                    r"overclaiming|prohibited|avoid", recent_context, re.IGNORECASE
                ):
                    continue
                if _line_has_allowed_context(line, recent_context=recent_context):
                    continue
                violations.append((rel_path, lineno, phrase, line.strip()))

    if violations:
        print("Risky paper claims detected:")
        for rel_path, lineno, phrase, line in violations:
            print(f" - {rel_path}:{lineno}: matched '{phrase}'")
            print(f"   line: {line}")
        return 1

    print("No risky paper claims detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
