#!/usr/bin/env python3
"""Fail CI if benchmark result files contain obvious hardcoded fabricated metrics."""

from __future__ import annotations

import json
from pathlib import Path

DISALLOWED_EXACT = {
    "1": 0.99,
    "5": 1.0,
    "10": 1.0,
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    candidate_files = list((repo_root / "results").rglob("*.json"))
    candidate_files.extend((repo_root / "experiments").rglob("*.json"))

    violations = []
    for file_path in sorted(set(candidate_files)):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if isinstance(payload, dict):
            fused = payload.get("fused_recalls")
            if isinstance(fused, dict) and fused == DISALLOWED_EXACT:
                violations.append(str(file_path.relative_to(repo_root)))

    if violations:
        print(
            "Found hardcoded fabricated benchmark pattern {'1': 0.99, '5': 1.0, '10': 1.0} in:"
        )
        for path in violations:
            print(f" - {path}")
        return 1

    print("No fabricated hardcoded fused benchmark patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
