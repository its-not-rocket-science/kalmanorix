#!/usr/bin/env python3
"""Run Kalman prior ablation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.kalman_prior_ablation import run_kalman_prior_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kalman_prior_ablation"),
        help="Artifact output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_kalman_prior_ablation(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
