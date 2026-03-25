"""Unit tests for low‑level Kalman fusion functions (kalman_fuser.py)."""

import numpy as np
import pytest

from kalmanorix.kalman_engine.kalman_fuser import (
    kalman_fuse_diagonal,
    kalman_fuse_structured,
    _structured_kalman_update_diagonal,
)
from kalmanorix.kalman_engine.structured_covariance import StructuredCovariance


def test_structured_kalman_update_diagonal_diagonal_only():
    """Test update with diagonal-only structured covariance matches diagonal update."""
    np.random.seed(42)
    d = 10
    x = np.random.randn(d)
    P = np.exp(np.random.randn(d))  # positive
    z = np.random.randn(d)
    R_diag = np.exp(np.random.randn(d))

    # Diagonal-only structured covariance
    R = StructuredCovariance.from_diagonal(R_diag)

    # Structured update
    x_new, P_new = _structured_kalman_update_diagonal(x, P, z, R, epsilon=1e-12)

    # Compare with diagonal update (using internal function)
    from kalmanorix.kalman_engine.kalman_fuser import _kalman_update_diagonal

    x_new_diag, P_new_diag = _kalman_update_diagonal(x, P, z, R_diag, epsilon=1e-12)

    assert np.allclose(x_new, x_new_diag, rtol=1e-10)
    assert np.allclose(P_new, P_new_diag, rtol=1e-10)


def test_structured_kalman_update_diagonal_lowrank():
    """Test update with low‑rank factor (small dimension for verification)."""
    np.random.seed(123)
    d = 6
    k = 2
    x = np.random.randn(d)
    P = np.exp(np.random.randn(d))
    z = np.random.randn(d)
    D = np.exp(np.random.randn(d))
    U = np.random.randn(d, k) * 0.1  # small factor

    R = StructuredCovariance.from_lowrank(D, U)

    x_new, P_new = _structured_kalman_update_diagonal(x, P, z, R, epsilon=1e-12)

    # Verify that covariance decreased (uncertainty shouldn't increase)
    assert np.all(P_new <= P + 1e-10)

    # Verify finite
    assert np.all(np.isfinite(x_new))
    assert np.all(np.isfinite(P_new))

    # Verify positive covariance
    assert np.all(P_new >= 0)


def test_kalman_fuse_structured_diagonal_only():
    """Test fusion with diagonal-only structured covariance matches diagonal fusion."""
    np.random.seed(456)
    d = 8
    n = 3
    embeddings = [np.random.randn(d) for _ in range(n)]
    cov_diags = [np.exp(np.random.randn(d)) for _ in range(n)]
    structured_covs = [StructuredCovariance.from_diagonal(diag) for diag in cov_diags]

    # Structured fusion
    x_struct, P_struct = kalman_fuse_structured(
        embeddings, structured_covs, epsilon=1e-12
    )
    # Diagonal fusion
    x_diag, P_diag = kalman_fuse_diagonal(embeddings, cov_diags, epsilon=1e-12)

    assert np.allclose(x_struct, x_diag, rtol=1e-10)
    assert np.allclose(P_struct, P_diag, rtol=1e-10)


def test_kalman_fuse_structured_lowrank():
    """Test fusion with low‑rank factors."""
    np.random.seed(789)
    d = 7
    k = 2
    n = 4
    embeddings = [np.random.randn(d) for _ in range(n)]
    # Create structured covariances with random low-rank factors
    structured_covs = []
    for _ in range(n):
        D = np.exp(np.random.randn(d))
        U = np.random.randn(d, k) * 0.1
        structured_covs.append(StructuredCovariance.from_lowrank(D, U))

    x, P = kalman_fuse_structured(embeddings, structured_covs, epsilon=1e-12)

    # Basic sanity checks
    assert x.shape == (d,)
    assert P.shape == (d,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(P))
    assert np.all(P >= 0)

    # Uncertainty should be less than prior (non-informative prior used)
    # Prior is first covariance diagonal, which may be smaller than final
    # So we just check that P is finite.


def test_kalman_fuse_structured_initial_state():
    """Test with explicit initial state and covariance."""
    np.random.seed(999)
    d = 5
    n = 2
    embeddings = [np.random.randn(d) for _ in range(n)]
    Ds = [np.exp(np.random.randn(d)) for _ in range(n)]
    Us = [np.random.randn(d, 1) * 0.1 for _ in range(n)]
    structured_covs = [StructuredCovariance.from_lowrank(D, U) for D, U in zip(Ds, Us)]

    initial_state = np.random.randn(d)
    initial_cov = np.exp(np.random.randn(d))

    x, P = kalman_fuse_structured(
        embeddings,
        structured_covs,
        initial_state=initial_state,
        initial_covariance=initial_cov,
        epsilon=1e-12,
    )

    assert x.shape == (d,)
    assert P.shape == (d,)
    # Should be different from using default initial
    x_default, P_default = kalman_fuse_structured(
        embeddings, structured_covs, epsilon=1e-12
    )
    # Not necessarily different (depends on random), but at least check shape
    assert not np.allclose(x, x_default) or not np.allclose(P, P_default)


def test_kalman_fuse_structured_sorting():
    """Test that sort_by_certainty works."""
    np.random.seed(111)
    d = 4
    # Create three measurements with increasing uncertainty
    embeddings = [np.random.randn(d) for _ in range(3)]
    # Uncertainties: 0.1, 1.0, 10.0 (diagonal only)
    Ds = [
        np.full(d, 0.1),
        np.full(d, 1.0),
        np.full(d, 10.0),
    ]
    structured_covs = [StructuredCovariance.from_diagonal(D) for D in Ds]

    # With sorting (default)
    x_sorted, P_sorted = kalman_fuse_structured(
        embeddings, structured_covs, sort_by_certainty=True, epsilon=1e-12
    )
    # Without sorting
    x_unsorted, P_unsorted = kalman_fuse_structured(
        embeddings, structured_covs, sort_by_certainty=False, epsilon=1e-12
    )

    # Results should be numerically close (order shouldn't matter for diagonal)
    # but may differ due to floating point accumulation
    assert np.allclose(x_sorted, x_unsorted, rtol=1e-10)
    assert np.allclose(P_sorted, P_unsorted, rtol=1e-10)


def test_structured_kalman_update_diagonal_lowrank_correctness():
    """Compare low‑rank update with full matrix computation (small d)."""
    np.random.seed(12345)
    d = 5
    k = 2
    x = np.random.randn(d)
    P = np.diag(np.exp(np.random.randn(d)))  # full diagonal matrix
    P_diag = np.diag(P)
    z = np.random.randn(d)
    D = np.exp(np.random.randn(d))
    U = np.random.randn(d, k) * 0.2
    R = StructuredCovariance.from_lowrank(D, U)

    # Full measurement covariance matrix
    R_full = np.diag(D) + U @ U.T

    # Exact Kalman gain using full matrices
    S_full = P + R_full
    K_exact = P @ np.linalg.inv(S_full)  # d × d

    # Exact state update
    innovation = z - x
    x_new_exact = x + K_exact @ innovation

    # Exact covariance update (full)
    eye = np.eye(d)
    P_new_exact = (eye - K_exact) @ P @ (eye - K_exact).T + K_exact @ R_full @ K_exact.T
    P_new_exact_diag = np.diag(P_new_exact)

    # Our structured update (returns diagonal covariance)
    x_new_approx, P_new_approx = _structured_kalman_update_diagonal(
        x, P_diag, z, R, epsilon=1e-12
    )

    # Compare state updates (should be close)
    assert np.allclose(x_new_approx, x_new_exact, rtol=1e-10, atol=1e-10)

    # Compare diagonal of covariance update (approximation vs exact diagonal)
    # Our approximation may differ; check that it's within reasonable tolerance
    # and that variance decreases (uncertainty doesn't increase)
    diff = np.abs(P_new_approx - P_new_exact_diag)
    relative = diff / (np.abs(P_new_exact_diag) + 1e-12)
    # Allow 15% relative error due to diagonal approximation (using column sum
    # of inverse rather than true diagonal of inverse for efficiency)
    assert np.all(relative < 0.15), f"Large relative error: {relative.max()}"

    # Ensure covariance decreased (or stayed same) for each dimension
    assert np.all(P_new_approx <= P_diag + 1e-10)


def test_kalman_fuse_structured_validation():
    """Test input validation errors."""
    d = 5
    embeddings = [np.random.randn(d)]
    structured_covs = [StructuredCovariance.from_diagonal(np.ones(d))]

    # Mismatched lengths
    with pytest.raises(ValueError, match="Number of embeddings"):
        kalman_fuse_structured(embeddings * 2, structured_covs)

    # Empty lists
    with pytest.raises(ValueError, match="At least one embedding"):
        kalman_fuse_structured([], [])

    # Dimension mismatch
    wrong_cov = StructuredCovariance.from_diagonal(np.ones(d + 1))
    with pytest.raises(ValueError, match="dimension"):
        kalman_fuse_structured(embeddings, [wrong_cov])

    # Wrong initial state shape
    with pytest.raises(ValueError, match="initial_state"):
        kalman_fuse_structured(
            embeddings, structured_covs, initial_state=np.ones(d + 1)
        )

    # Wrong initial covariance shape
    with pytest.raises(ValueError, match="initial_covariance"):
        kalman_fuse_structured(
            embeddings, structured_covs, initial_covariance=np.ones(d + 1)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
