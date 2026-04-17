"""Run validation-only uncertainty calibration benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.uncertainty_calibration import (
    ValidationPowerConfig,
    run_uncertainty_calibration,
)


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
    parser.add_argument("--min-validation-total", type=int, default=8)
    parser.add_argument("--min-validation-per-domain", type=int, default=2)
    parser.add_argument("--min-effective-support-per-specialist", type=int, default=6)
    parser.add_argument("--calibrator-min-samples", type=int, default=8)
    args = parser.parse_args()
    run_uncertainty_calibration(
        output_dir=args.output_dir,
        objective=args.objective,
        power_config=ValidationPowerConfig(
            min_validation_total=args.min_validation_total,
            min_validation_per_domain=args.min_validation_per_domain,
            min_effective_support_per_specialist=args.min_effective_support_per_specialist,
            calibrator_min_samples=args.calibrator_min_samples,
        ),
    )


if __name__ == "__main__":
    main()
