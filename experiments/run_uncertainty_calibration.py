"""Run validation-only uncertainty calibration benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.uncertainty_calibration import run_uncertainty_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/uncertainty_calibration"),
        help="Output directory for uncertainty calibration artifacts.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="rank_error_proxy",
        choices=[
            "rank_error_proxy",
            "topk_miss_indicator",
            "distance_to_relevant_doc_centroid",
            "score_quality_residual",
        ],
        help="Held-out objective used to fit sigma2 calibrators.",
    )
    args = parser.parse_args()
    run_uncertainty_calibration(output_dir=args.output_dir, objective=args.objective)


if __name__ == "__main__":
    main()
