#!/usr/bin/env python3
"""Compare ranking shifts between two canonical summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["ndcg@5", "ndcg@10", "mrr@5", "recall@1", "top1_success"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank(summary: dict, metric: str) -> list[tuple[str, float]]:
    rows = []
    for method, payload in summary["methods"].items():
        metrics = payload.get("metrics", {})
        if metric not in metrics:
            continue
        rows.append((method, float(metrics[metric]["mean"])))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def _pos_map(ranked: list[tuple[str, float]]) -> dict[str, int]:
    return {name: idx + 1 for idx, (name, _) in enumerate(ranked)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = _load(args.old)
    new = _load(args.new)
    lines = [
        "# Canonical Ranking Shift Comparison",
        "",
        f"- Old summary: `{args.old}`",
        f"- New summary: `{args.new}`",
        "",
    ]

    for metric in METRICS:
        old_rank = _rank(old, metric)
        new_rank = _rank(new, metric)
        old_pos = _pos_map(old_rank)
        new_pos = _pos_map(new_rank)
        methods = sorted(set(old_pos).intersection(new_pos))

        lines.extend(
            [
                f"## {metric}",
                "",
                "| Method | Old rank | New rank | Shift |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in methods:
            shift = old_pos[method] - new_pos[method]
            lines.append(
                f"| {method} | {old_pos[method]} | {new_pos[method]} | {shift:+d} |"
            )
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
