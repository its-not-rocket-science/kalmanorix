#!/usr/bin/env python3
"""Run retrieval-aware Kalman + learned correction ablation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kalmanorix.benchmarks.kalman_learned_correction import (
    LearnedCorrectionConfig,
    run_kalman_learned_correction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kalman_learned_correction"),
        help="Directory where benchmark artifacts are written",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp"],
        default="linear",
        help="Correction model family",
    )
    args = parser.parse_args()

    cfg = LearnedCorrectionConfig(model_type=args.model_type)
    summary = run_kalman_learned_correction(output_dir=args.output_dir, config=cfg)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
