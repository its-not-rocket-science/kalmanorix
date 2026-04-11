#!/usr/bin/env python3
"""Run correlation-aware Kalman fusion benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.correlation_aware_fusion import (
    CorrelationAwareFusionConfig,
    run_correlation_aware_fusion_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/correlation_aware_fusion"),
        help="Artifact output directory.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_correlation_aware_fusion_benchmark(
        output_dir=args.output_dir,
        config=CorrelationAwareFusionConfig(random_seed=args.seed),
    )


if __name__ == "__main__":
    main()
