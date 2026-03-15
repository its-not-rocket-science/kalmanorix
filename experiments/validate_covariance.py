"""
Validation of covariance estimation on synthetic data with known noise levels.

This script corresponds to Milestone 1.1: Diagonal Covariance Estimation.
It generates embeddings with known diagonal covariance and tests whether
our estimation methods recover it accurately.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # pylint: disable=wrong-import-position

from kalmanorix.kalman_engine.covariance import EmpiricalCovariance, DiagonalCovariance  # pylint: disable=wrong-import-position


def run_validation(
    dimension: int = 100,
    n_samples: int = 1000,
    noise_levels: tuple = (0.1, 1.0, 10.0),
    seed: int = 42,
):
    """
    Validate covariance estimation.

    Args:
        dimension: Embedding dimension d
        n_samples: Number of validation samples
        noise_levels: Tuple of (low, medium, high) noise multipliers
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    print("=" * 70)
    print("Covariance Estimation Validation (Milestone 1.1)")
    print("=" * 70)
    print(f"Dimension: {dimension}, Samples: {n_samples}")
    print(f"Noise levels: {noise_levels}")
    print()

    results = []

    for level_name, multiplier in zip(["Low", "Medium", "High"], noise_levels):
        print(f"\n--- {level_name} noise (multiplier = {multiplier}) ---")

        # Generate true diagonal covariance
        # Use log‑normal distribution for realistic variance patterns
        true_cov = np.exp(np.random.randn(dimension)) * multiplier

        # Generate mean vector
        true_mean = np.random.randn(dimension)

        # Generate validation embeddings
        embeddings = np.zeros((n_samples, dimension))
        for i in range(dimension):
            embeddings[:, i] = np.random.normal(
                true_mean[i], np.sqrt(true_cov[i]), n_samples
            )

        # Estimate covariance
        estimator = EmpiricalCovariance(embeddings)
        estimated_cov = estimator.fixed_covariance

        # Compute errors
        abs_error = np.abs(estimated_cov - true_cov)
        rel_error = abs_error / (true_cov + 1e-12)

        # Statistics
        stats = {
            "level": level_name,
            "multiplier": multiplier,
            "mean_true": np.mean(true_cov),
            "mean_estimated": np.mean(estimated_cov),
            "mean_abs_error": np.mean(abs_error),
            "mean_rel_error": np.mean(rel_error),
            "max_rel_error": np.max(rel_error),
            "median_rel_error": np.median(rel_error),
        }

        results.append(stats)

        print(f"  True mean variance:      {stats['mean_true']:.6f}")
        print(f"  Estimated mean variance: {stats['mean_estimated']:.6f}")
        print(f"  Mean absolute error:     {stats['mean_abs_error']:.6f}")
        print(f"  Mean relative error:     {stats['mean_rel_error']:.3f}")
        print(f"  Median relative error:   {stats['median_rel_error']:.3f}")
        print(f"  Max relative error:      {stats['max_rel_error']:.3f}")

        # Check that relative error is below threshold for most dimensions
        threshold = 0.3  # 30% error allowed due to finite samples
        prop_within_threshold = np.mean(rel_error < threshold)
        print(
            f"  % dimensions < {threshold * 100:.0f}% error: {prop_within_threshold * 100:.1f}%"
        )

        if prop_within_threshold < 0.8:
            print(
                f"  ⚠️  Warning: Only {prop_within_threshold * 100:.1f}% of dimensions "
                f"within {threshold * 100:.0f}% error"
            )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for stats in results:
        print(
            f"{stats['level']:8} noise: mean error {stats['mean_rel_error']:.3f}, "
            f"max {stats['max_rel_error']:.3f}"
        )

    # Test DiagonalCovariance container
    print("\n--- DiagonalCovariance container test ---")
    test_cov = np.exp(np.random.randn(5))
    container = DiagonalCovariance(test_cov)
    print(f"Dimension: {container.dimension}")
    print(f"Uncertainty score: {container.uncertainty_score():.4f}")
    print(f"Confidence score:  {container.confidence_score():.4f}")
    assert np.allclose(container.diagonal, test_cov), "Container mismatch"

    # Final verdict
    all_within_threshold = all(r["mean_rel_error"] < 0.5 for r in results)
    if all_within_threshold:
        print(
            "\n[PASS] Validation PASSED: Covariance estimation works within expected tolerances."
        )
    else:
        print("\n[FAIL] Validation FAILED: Some noise levels show large errors.")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
