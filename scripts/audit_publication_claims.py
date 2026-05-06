#!/usr/bin/env python3
"""Audit publication markdown for overclaiming language."""

from __future__ import annotations

import re
from pathlib import Path

PROHIBITED_PHRASES = (
    "Kalman beats mean",
    "Kalman improves retrieval",
    "Kalman outperforms",
    "proved superior",
    "state-of-the-art",
)

ALLOWED_CONTEXT_PATTERNS = (
    re.compile(r"\bdoes\s+not\s+beat\b", re.IGNORECASE),
    re.compile(r"\bhypothesis\s+was\s+not\s+supported\b", re.IGNORECASE),
    re.compile(r"\bnot\s+supported\b", re.IGNORECASE),
    re.compile(r"\bunsupported\b", re.IGNORECASE),
    re.compile(r"\bno\s+evidence\b", re.IGNORECASE),
)


def _is_allowed_context(line: str) -> bool:
    return any(pattern.search(line) for pattern in ALLOWED_CONTEXT_PATTERNS)


def _iter_markdown_files(root: Path):
    for relative_base in (Path("docs/publication"), Path("paper")):
        base = root / relative_base
        if not base.exists():
            continue
        yield from sorted(base.rglob("*.md"))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations: list[tuple[str, int, str]] = []

    for file_path in _iter_markdown_files(repo_root):
        for lineno, line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _is_allowed_context(line):
                continue

            for phrase in PROHIBITED_PHRASES:
                if re.search(re.escape(phrase), line, flags=re.IGNORECASE):
                    rel_path = str(file_path.relative_to(repo_root))
                    violations.append((rel_path, lineno, phrase))

    if violations:
        print("Risky publication claims detected:")
        for path, lineno, phrase in violations:
            print(f" - {path}:{lineno}: {phrase}")
        return 1

    print("No risky publication claim language found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
