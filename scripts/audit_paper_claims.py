#!/usr/bin/env python3
"""Audit publication text for unsupported superiority claims."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_DIRECTORIES = [
    "paper/arxiv",
    "paper/tmlr",
    "paper/joss",
    "docs/publication",
]

TEXT_EXTENSIONS = {".md", ".tex", ".txt", ".rst"}

RISKY_PATTERNS = [
    r"kalman beats mean",
    r"kalman improves retrieval",
    r"kalman outperforms",
    r"proved superior",
    r"state of the art",
    r"statistically significant improvement",
    r"robustly improves",
]

ALLOWANCE_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"\bunsupported\b",
    r"\bwas not supported\b",
    r"\bdid not\b",
    r"\bfailed to\b",
    r"\bcannot support\b",
    r"\bcan't support\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag risky superiority claims in publication text."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_DIRECTORIES,
        help="Directories or files to audit.",
    )
    return parser.parse_args()


def iter_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)
            continue
        if path.is_dir():
            for file in path.rglob("*"):
                if file.is_file() and file.suffix.lower() in TEXT_EXTENSIONS:
                    files.append(file)
    return sorted(files)


def sentence_bounds(line: str) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]", line):
        end = match.end()
        bounds.append((start, end))
        start = end
    if start < len(line):
        bounds.append((start, len(line)))
    return bounds


def containing_sentence(line: str, start_idx: int) -> str:
    for start, end in sentence_bounds(line):
        if start <= start_idx < end:
            return line[start:end]
    return line


def is_allowed(context: str) -> bool:
    lowered = context.lower()
    return any(re.search(pattern, lowered) for pattern in ALLOWANCE_PATTERNS)


def audit_file(
    path: Path, risky_regexes: list[re.Pattern[str]]
) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        line_number = idx + 1
        lowered = line.lower()
        for regex in risky_regexes:
            match = regex.search(lowered)
            if not match:
                continue
            sentence_context = containing_sentence(lowered, match.start())
            window_start = max(0, idx - 5)
            window_end = min(len(lines), idx + 6)
            neighborhood = " ".join(lines[window_start:window_end]).lower()
            if is_allowed(sentence_context):
                continue
            if any(
                tag in neighborhood
                for tag in [
                    "risky",
                    "disallowed",
                    "not supported",
                    "unsupported",
                    "only acceptable when",
                    "do not use",
                    "avoid",
                    "unqualified phrases",
                ]
            ):
                continue
            findings.append((line_number, regex.pattern, line.strip()))
    return findings


def main() -> int:
    args = parse_args()
    files = iter_files(args.paths)
    risky_regexes = [
        re.compile(pattern, flags=re.IGNORECASE) for pattern in RISKY_PATTERNS
    ]

    issues: list[tuple[Path, int, str, str]] = []
    for file in files:
        for line_number, pattern, line in audit_file(file, risky_regexes):
            issues.append((file, line_number, pattern, line))

    if issues:
        print("Risky publication claims detected:")
        for file, line_number, pattern, line in issues:
            print(f"- {file}:{line_number}: matched '{pattern}' -> {line}")
        return 1

    print("No unsupported superiority claims detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
