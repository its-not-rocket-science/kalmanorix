from __future__ import annotations

import numpy as np

from kalmanorix.calibration import (
    calibration_summary,
    compute_embedding_calibration,
)


def test_embedding_calibration_uses_absolute_tolerance_not_batch_normalization() -> None:
    specialist = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    reference = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    variances = np.array([0.1, 0.1], dtype=float)

    result = compute_embedding_calibration(
        specialist_embeddings=specialist,
        reference_embeddings=reference,
        predicted_variances=variances,
        norm="l2",
        error_tolerance=0.5,
        n_bins=5,
    )

    # First sample error=0 => accurate, second error=1 => inaccurate.
    assert result.mean_accuracy == 0.5


def test_embedding_confidence_decreases_with_higher_sigma2() -> None:
    specialist = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    reference = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)

    low_sigma2 = np.array([0.1, 0.1], dtype=float)
    high_sigma2 = np.array([1.5, 1.5], dtype=float)

    low_res = compute_embedding_calibration(
        specialist_embeddings=specialist,
        reference_embeddings=reference,
        predicted_variances=low_sigma2,
        norm="l2",
        error_tolerance=0.5,
    )
    high_res = compute_embedding_calibration(
        specialist_embeddings=specialist,
        reference_embeddings=reference,
        predicted_variances=high_sigma2,
        norm="l2",
        error_tolerance=0.5,
    )

    assert low_res.mean_confidence > high_res.mean_confidence


def test_calibration_summary_contains_overconfidence_gap() -> None:
    specialist = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    reference = np.zeros_like(specialist)
    variances = np.array([0.1, 1.0], dtype=float)

    result = compute_embedding_calibration(
        specialist_embeddings=specialist,
        reference_embeddings=reference,
        predicted_variances=variances,
        norm="l2",
        error_tolerance=0.5,
    )
    summary = calibration_summary(result)

    assert "overconfidence_gap" in summary
    assert np.isfinite(summary["overconfidence_gap"])
