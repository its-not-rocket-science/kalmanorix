"""Run uncertainty-method ablation and write summary/report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.uncertainty_ablation import run_uncertainty_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/uncertainty_ablation"),
        help="Output directory for summary.json and report.md",
    )
    args = parser.parse_args()
    run_uncertainty_ablation(args.output_dir)


if __name__ == "__main__":
    main()
