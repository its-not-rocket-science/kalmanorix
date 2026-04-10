"""Tests for Kalman prior ablation and residual fusion modes."""

from __future__ import annotations

import numpy as np

from kalmanorix.benchmarks.kalman_prior_ablation import (
    fit_learned_linear_prior,
    kalman_fuse_with_prior_modes,
    run_kalman_prior_ablation,
)
from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal


def test_residual_mode_equals_standard_when_prior_is_zero() -> None:
    rng = np.random.default_rng(0)
    d = 16
    embeddings = [rng.normal(size=d) for _ in range(3)]
    covariances = [np.full(d, 0.12 + 0.03 * i, dtype=np.float64) for i in range(3)]

    standard_x, standard_p = kalman_fuse_diagonal(embeddings, covariances)
    residual_x, residual_p = kalman_fuse_with_prior_modes(
        embeddings,
        covariances,
        prior_mode="single_generalist",
        generalist_prior=np.zeros(d, dtype=np.float64),
        residual_mode=True,
    )

    assert np.allclose(residual_x, standard_x)
    assert np.allclose(residual_p, standard_p)


def test_prior_only_mode_reproducible() -> None:
    rng = np.random.default_rng(1)
    d = 12
    embeddings = [rng.normal(size=d) for _ in range(2)]
    covariances = [np.full(d, 0.2, dtype=np.float64) for _ in range(2)]
    prior = rng.normal(size=d)

    out1, cov1 = kalman_fuse_with_prior_modes(
        embeddings,
        covariances,
        prior_mode="single_generalist",
        generalist_prior=prior,
        prior_only=True,
    )
    out2, cov2 = kalman_fuse_with_prior_modes(
        embeddings,
        covariances,
        prior_mode="single_generalist",
        generalist_prior=prior,
        prior_only=True,
    )

    assert np.allclose(out1, prior)
    assert np.allclose(out1, out2)
    assert np.allclose(cov1, cov2)


def test_learned_prior_fit_train_split_only() -> None:
    rng = np.random.default_rng(2)
    n = 30
    n_specialists = 3
    d = 10
    specialists = rng.normal(size=(n, n_specialists, d))
    truth = rng.normal(size=(n, d))
    split_labels = (["train"] * 10) + (["validation"] * 10) + (["test"] * 10)

    weights, meta = fit_learned_linear_prior(
        specialists,
        truth,
        split_labels,
        fit_split="train",
    )

    assert weights.shape == (n_specialists,)
    assert np.isclose(np.sum(weights), 1.0)
    assert set(meta["fit_indices"]) == set(range(10))
    assert all(idx < 10 for idx in meta["fit_indices"])


def test_run_kalman_prior_ablation_writes_artifacts(tmp_path) -> None:
    summary = run_kalman_prior_ablation(output_dir=tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert "kalman_generalist_prior" in summary["metrics"]
    assert "kalman_learned_linear_prior" in summary["metrics"]
    assert "kalman_residuals" in summary["metrics"]
