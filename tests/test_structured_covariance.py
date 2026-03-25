"""
Tests for low‑rank structured covariance representation (Milestone 3.2).
"""

import numpy as np
import pytest  # type: ignore  # pylint: disable=import-error

from kalmanorix.kalman_engine.structured_covariance import (
    StructuredCovariance,
    woodbury_inverse,
)


def test_structured_covariance_diagonal_only():
    """Test diagonal‑only structured covariance (backward compatibility)."""
    d = 10
    diag = np.exp(np.random.randn(d))

    cov = StructuredCovariance(diag, lowrank_factor=None)

    assert cov.dimension == d
    assert cov.rank == 0
    assert cov.is_diagonal
    assert np.allclose(cov.diagonal, diag)
    assert cov.lowrank_factor is None

    # Full matrix conversion
    R = cov.to_full()
    assert R.shape == (d, d)
    assert np.allclose(np.diag(R), diag)
    assert np.allclose(R - np.diag(np.diag(R)), 0)

    # Uncertainty score
    assert np.isclose(cov.uncertainty_score(), np.sum(diag))
    assert 0 < cov.confidence_score() < 1

    # Diagonal-only copy
    cov2 = cov.diagonal_only()
    assert cov2.is_diagonal
    assert np.allclose(cov2.diagonal, diag)


def test_structured_covariance_with_lowrank():
    """Test structured covariance with low‑rank factor."""
    d = 12
    k = 3
    diag = np.exp(np.random.randn(d))
    U = np.random.randn(d, k)

    cov = StructuredCovariance(diag, lowrank_factor=U)

    assert cov.dimension == d
    assert cov.rank == k
    assert not cov.is_diagonal
    assert np.allclose(cov.diagonal, diag)
    assert np.allclose(cov.lowrank_factor, U)

    # Full matrix conversion
    R = cov.to_full()
    expected = np.diag(diag) + U @ U.T
    assert R.shape == (d, d)
    assert np.allclose(R, expected)

    # Uncertainty score includes trace(UUᵀ)
    trace_UUT = np.sum(U**2)
    assert np.isclose(cov.uncertainty_score(), np.sum(diag) + trace_UUT)


def test_structured_covariance_validation():
    """Test validation of inputs."""
    d = 5
    diag = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Negative diagonal raises ValueError
    with pytest.raises(ValueError):
        StructuredCovariance(np.array([1.0, -0.1, 2.0]))

    # Wrong shape of lowrank_factor
    U_wrong = np.random.randn(d + 1, 2)
    with pytest.raises(ValueError):
        StructuredCovariance(diag, lowrank_factor=U_wrong)

    # 1‑D lowrank_factor (should become 2‑D?)
    U_1d = np.random.randn(d)
    with pytest.raises(ValueError):
        StructuredCovariance(diag, lowrank_factor=U_1d)

    # Valid lowrank_factor
    U = np.random.randn(d, 2)
    cov = StructuredCovariance(diag, lowrank_factor=U)
    assert cov.rank == 2


def test_woodbury_solve_diagonal():
    """Test Woodbury solve for diagonal‑only covariance."""
    d = 8
    diag = np.exp(np.random.randn(d))
    prior_diag = np.exp(np.random.randn(d))
    v = np.random.randn(d)

    cov = StructuredCovariance(diag, lowrank_factor=None)

    # Diagonal case reduces to elementwise division: (prior + D) x = v
    x = cov.woodbury_solve(prior_diag, v)
    expected = v / (prior_diag + diag + 1e-8)
    assert np.allclose(x, expected)

    # Batch solve (multiple RHS vectors)
    m = 4
    V = np.random.randn(d, m)
    X = cov.woodbury_solve(prior_diag, V)
    expected_batch = V / (prior_diag[:, None] + diag[:, None] + 1e-8)
    assert np.allclose(X, expected_batch)


def test_woodbury_solve_lowrank():
    """Test Woodbury solve with low‑rank factor."""
    d = 6
    k = 2
    diag = np.exp(np.random.randn(d))
    U = np.random.randn(d, k) * 0.1  # Small factor for stability
    prior_diag = np.exp(np.random.randn(d))
    v = np.random.randn(d)

    cov = StructuredCovariance(diag, lowrank_factor=U)

    # Solve (prior + D + UUᵀ) x = v
    x = cov.woodbury_solve(prior_diag, v, epsilon=1e-12)

    # Verify by direct computation (small d)
    S = np.diag(prior_diag)
    R = np.diag(diag) + U @ U.T
    x_expected = np.linalg.solve(S + R, v)
    assert np.allclose(x, x_expected, rtol=1e-10)

    # Batch solve
    m = 3
    V = np.random.randn(d, m)
    X = cov.woodbury_solve(prior_diag, V, epsilon=1e-12)
    for i in range(m):
        x_i_expected = np.linalg.solve(S + R, V[:, i])
        assert np.allclose(X[:, i], x_i_expected, rtol=1e-10)


def test_woodbury_solve_numerical_stability():
    """Test Woodbury solve with near‑zero prior_diag."""
    d = 5
    diag = np.ones(d)
    U = np.random.randn(d, 2) * 0.01
    prior_diag = np.array([1e-12, 1.0, 2.0, 0.0, 1e-8])
    v = np.random.randn(d)

    cov = StructuredCovariance(diag, lowrank_factor=U)

    # Should not raise division‑by‑zero due to epsilon
    x = cov.woodbury_solve(prior_diag, v, epsilon=1e-8)
    assert np.all(np.isfinite(x))

    # Compare with direct solve using prior_diag + epsilon
    S = np.diag(prior_diag + 1e-8)
    R = np.diag(diag) + U @ U.T
    x_expected = np.linalg.solve(S + R, v)
    assert np.allclose(x, x_expected, rtol=1e-6)


def test_diagonal_only_method():
    """Test dropping low‑rank factor."""
    d = 7
    k = 2
    diag = np.exp(np.random.randn(d))
    U = np.random.randn(d, k)

    cov = StructuredCovariance(diag, lowrank_factor=U)
    cov_diag = cov.diagonal_only()

    assert cov_diag.is_diagonal
    assert cov_diag.lowrank_factor is None
    assert cov_diag.rank == 0
    assert np.allclose(cov_diag.diagonal, diag)

    # Full matrix should be just diagonal
    R_diag = cov_diag.to_full()
    assert np.allclose(R_diag, np.diag(diag))


def test_factory_methods():
    """Test from_diagonal and from_lowrank factory methods."""
    d = 9
    k = 3
    diag = np.exp(np.random.randn(d))
    U = np.random.randn(d, k)

    cov1 = StructuredCovariance.from_diagonal(diag)
    assert cov1.is_diagonal
    assert np.allclose(cov1.diagonal, diag)

    cov2 = StructuredCovariance.from_lowrank(diag, U)
    assert not cov2.is_diagonal
    assert np.allclose(cov2.diagonal, diag)
    assert np.allclose(cov2.lowrank_factor, U)


def test_uncertainty_confidence_scores():
    """Test scalar uncertainty and confidence metrics."""
    diag = np.array([0.1, 0.2, 0.3, 0.4])
    U = np.array([[0.1, 0.0], [0.0, 0.1], [0.0, 0.0], [0.0, 0.0]])

    cov_diag = StructuredCovariance.from_diagonal(diag)
    assert np.isclose(cov_diag.uncertainty_score(), 1.0)  # sum = 1.0
    assert np.isclose(cov_diag.confidence_score(), 1.0 / (1.0 + 1.0))

    cov_lowrank = StructuredCovariance.from_lowrank(diag, U)
    trace_UUT = 0.01 + 0.01  # 0.1² + 0.1²
    expected = 1.0 + trace_UUT
    assert np.isclose(cov_lowrank.uncertainty_score(), expected)
    assert np.isclose(cov_lowrank.confidence_score(), 1.0 / (1.0 + expected))


def test_woodbury_inverse_helper():
    """Test woodbury_inverse helper function."""
    d = 6
    k = 2
    S_diag = np.exp(np.random.randn(d))
    U = np.random.randn(d, k)

    S_inv, B, M_inv = woodbury_inverse(S_diag, U, epsilon=1e-12)

    # Check shapes
    assert S_inv.shape == (d,)
    assert B.shape == (d, k)
    assert M_inv.shape == (k, k)

    # Check values
    S = S_diag + 1e-12
    expected_S_inv = 1.0 / S
    assert np.allclose(S_inv, expected_S_inv)

    expected_B = expected_S_inv[:, None] * U
    assert np.allclose(B, expected_B)

    expected_M = np.eye(k) + U.T @ expected_B
    expected_M_inv = np.linalg.inv(expected_M)
    assert np.allclose(M_inv, expected_M_inv)

    # Diagonal case (U = None)
    S_inv_diag, B_diag, M_inv_diag = woodbury_inverse(S_diag, None)
    assert np.allclose(S_inv_diag, expected_S_inv)
    assert B_diag is None
    assert M_inv_diag is None


def test_woodbury_inverse_singular_fallback():
    """Test pseudo‑inverse fallback when inv fails."""
    from unittest import mock

    d = 5
    k = 2
    S_diag = np.ones(d)
    U = np.random.randn(d, k)

    # Make np.linalg.inv raise LinAlgError
    with mock.patch(
        "numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular matrix")
    ):
        # Should fall back to pseudo‑inverse
        S_inv, B, M_inv = woodbury_inverse(S_diag, U, epsilon=1e-8)
        # Verify pseudo‑inverse was used (M_inv computed via pinv)
        # We can't directly check, but ensure no exception and shapes match
        assert S_inv.shape == (d,)
        assert B.shape == (d, k)
        assert M_inv.shape == (k, k)
        # M_inv should be a pseudo‑inverse (M @ M_inv @ M ≈ M)
        M = np.eye(k) + U.T @ B
        assert np.allclose(M @ M_inv @ M, M, atol=1e-6)


def test_repr():
    """Test string representation."""
    d = 3
    diag = np.array([0.1, 0.2, 0.3])
    U = np.random.randn(d, 2) * 0.1

    cov_diag = StructuredCovariance.from_diagonal(diag)
    repr_diag = repr(cov_diag)
    assert "diagonal" in repr_diag
    assert f"d={d}" in repr_diag
    assert "uncertainty" in repr_diag

    cov_lowrank = StructuredCovariance.from_lowrank(diag, U)
    repr_lowrank = repr(cov_lowrank)
    assert "diagonal+lowrank" in repr_lowrank
    assert f"d={d}" in repr_lowrank
    assert f"k={2}" in repr_lowrank
    assert "uncertainty" in repr_lowrank


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
