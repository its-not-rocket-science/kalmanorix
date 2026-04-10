#!/usr/bin/env python3
"""Generate bucketed Kalman error analysis report from query-level benchmark details."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalmanorix.benchmarks.kalman_error_analysis import (
    generate_kalman_error_analysis_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details-json",
        type=Path,
        default=Path("results/canonical_benchmark/runner_details.json"),
        help="Path to benchmark details JSON with top-level query_level payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kalman_error_analysis/report.md"),
    )
    args = parser.parse_args()

    details = json.loads(args.details_json.read_text(encoding="utf-8"))
    report = generate_kalman_error_analysis_report(details, args.output)
    print(report)


if __name__ == "__main__":
    main()
