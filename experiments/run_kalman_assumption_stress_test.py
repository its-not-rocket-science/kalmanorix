#!/usr/bin/env python3
"""Run Kalman assumption stress-test slice benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.kalman_assumption_stress_test import (
    AssumptionStressConfig,
    run_kalman_assumption_stress_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kalman_assumption_stress_test"),
        help="Artifact output directory.",
    )
    parser.add_argument("--seed", type=int, default=41, help="Random seed.")
    parser.add_argument(
        "--n-per-case-type",
        type=int,
        default=120,
        help="Number of queries generated for each assumption case type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_kalman_assumption_stress_test(
        output_dir=args.output_dir,
        config=AssumptionStressConfig(
            random_seed=args.seed,
            n_per_case_type=args.n_per_case_type,
        ),
    )


if __name__ == "__main__":
    main()
