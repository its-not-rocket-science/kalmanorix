"""CLI entrypoints for packaged benchmark runners."""

from __future__ import annotations

import argparse
from pathlib import Path

from .correlation_aware_fusion import (
    CorrelationAwareFusionConfig,
    run_correlation_aware_fusion_benchmark,
)
from .uncertainty_calibration import (
    ValidationPowerConfig,
    run_uncertainty_calibration,
)


def correlation_aware_fusion_main() -> None:
    """Run the correlation-aware fusion benchmark and emit report artifacts."""
    parser = argparse.ArgumentParser(description="Run the correlation-aware Kalman fusion benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/correlation_aware_fusion"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    cfg = CorrelationAwareFusionConfig(random_seed=args.seed)
    run_correlation_aware_fusion_benchmark(output_dir=args.output_dir, cfg=cfg)


def uncertainty_calibration_main() -> None:
    """Run uncertainty calibration study and emit report artifacts."""
    parser = argparse.ArgumentParser(description="Run uncertainty calibration benchmark.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/uncertainty_calibration"))
    parser.add_argument("--objective", type=str, default="rank_error_proxy")
    parser.add_argument("--sigma2-method", type=str, default="centroid_distance_sigma2")
    parser.add_argument("--min-validation-total", type=int, default=24)
    parser.add_argument("--min-validation-per-domain", type=int, default=4)
    parser.add_argument("--min-effective-support-per-specialist", type=int, default=18)
    parser.add_argument("--calibrator-min-samples", type=int, default=20)
    args = parser.parse_args()

    run_uncertainty_calibration(
        output_dir=args.output_dir,
        sigma2_method=args.sigma2_method,
        objective=args.objective,
        power_config=ValidationPowerConfig(
            min_validation_total=args.min_validation_total,
            min_validation_per_domain=args.min_validation_per_domain,
            min_effective_support_per_specialist=args.min_effective_support_per_specialist,
            calibrator_min_samples=args.calibrator_min_samples,
        ),
    )
