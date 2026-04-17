from __future__ import annotations

import numpy as np

from kalmanorix.benchmarks.kalman_covariance_ablation import (
    fit_diagonal_variance,
    fit_structured_covariance,
    project_psd,
)


def test_project_psd_enforces_nonnegative_eigenvalues() -> None:
    mat = np.array([[1.0, 2.0], [2.0, -0.5]], dtype=np.float64)
    psd = project_psd(mat)
    eigs = np.linalg.eigvalsh(psd)
    assert np.min(eigs) >= -1e-10


def test_fit_structured_covariance_shapes_and_nonnegative_diagonal() -> None:
    rng = np.random.default_rng(7)
    residuals = rng.normal(size=(60, 12))

    cov = fit_structured_covariance(residuals, rank=3, diagonal_floor=1e-5)

    assert cov.diagonal.shape == (12,)
    assert np.all(cov.diagonal >= 1e-5)
    assert cov.lowrank_factor is not None
    assert cov.lowrank_factor.shape == (12, 3)


def test_shrinkage_behavior_reduces_offdiagonal_energy() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(size=(100, 6))
    correlated = base.copy()
    correlated[:, 1] = 0.9 * correlated[:, 0] + 0.1 * correlated[:, 1]

    low_shrink = fit_structured_covariance(
        correlated,
        rank=2,
        shrinkage_to_diagonal=0.0,
    )
    high_shrink = fit_structured_covariance(
        correlated,
        rank=2,
        shrinkage_to_diagonal=0.9,
    )

    low_off = low_shrink.to_full() - np.diag(np.diag(low_shrink.to_full()))
    high_off = high_shrink.to_full() - np.diag(np.diag(high_shrink.to_full()))

    assert (
        np.linalg.norm(high_off, ord="fro") <= np.linalg.norm(low_off, ord="fro") + 1e-8
    )


def test_rank_zero_falls_back_to_diagonal() -> None:
    rng = np.random.default_rng(11)
    residuals = rng.normal(size=(40, 10))

    cov = fit_structured_covariance(residuals, rank=0)

    assert cov.rank == 0
    assert cov.lowrank_factor is None


def test_fit_diagonal_variance_shape() -> None:
    rng = np.random.default_rng(13)
    residuals = rng.normal(size=(20, 5))
    diag = fit_diagonal_variance(residuals)
    assert diag.shape == (5,)
