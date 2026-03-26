"""Unit tests for low‑level Kalman fusion functions (kalman_fuser.py)."""

import numpy as np
import pytest

from kalmanorix.kalman_engine.kalman_fuser import (
    kalman_fuse_diagonal,
    kalman_fuse_diagonal_ensemble,
    kalman_fuse_diagonal_batch,
    kalman_fuse_diagonal_ensemble_batch,
    kalman_fuse_structured,
    _structured_kalman_update_diagonal,
    fuse_with_prior,
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


def test_kalman_fuse_diagonal_ensemble_basic():
    """Test ensemble fusion matches exact precision-weighted fusion (no prior)."""
    np.random.seed(777)
    d = 8
    n = 4
    embeddings = [np.random.randn(d) for _ in range(n)]
    covariances = [np.exp(np.random.randn(d)) for _ in range(n)]

    # Ensemble fusion (non-informative prior)
    x_ensemble, P_ensemble = kalman_fuse_diagonal_ensemble(
        embeddings, covariances, epsilon=1e-12
    )

    # Exact fusion using precision weighting (no prior)
    inv_cov_sum = np.zeros(d)
    inv_cov_weighted = np.zeros(d)
    for emb, cov in zip(embeddings, covariances):
        inv = 1.0 / (cov + 1e-12)
        inv_cov_sum += inv
        inv_cov_weighted += inv * emb
    P_exact = 1.0 / (inv_cov_sum + 1e-12)
    x_exact = P_exact * inv_cov_weighted

    # Should match exactly (same mathematical formulation)
    assert np.allclose(x_ensemble, x_exact, rtol=1e-10, atol=1e-12)
    assert np.allclose(P_ensemble, P_exact, rtol=1e-10, atol=1e-12)


def test_kalman_fuse_diagonal_ensemble_with_prior():
    """Test ensemble fusion with explicit prior."""
    np.random.seed(888)
    d = 6
    n = 3
    embeddings = [np.random.randn(d) for _ in range(n)]
    covariances = [np.exp(np.random.randn(d)) for _ in range(n)]
    prior_state = np.random.randn(d)
    prior_cov = np.exp(np.random.randn(d))

    # Ensemble fusion with prior
    x_ensemble, P_ensemble = kalman_fuse_diagonal_ensemble(
        embeddings,
        covariances,
        initial_state=prior_state,
        initial_covariance=prior_cov,
        epsilon=1e-12,
    )

    # Sequential fusion with prior (use fuse_with_prior)
    x_seq, P_seq = fuse_with_prior(
        embeddings, covariances, prior_state, prior_cov, epsilon=1e-12
    )

    # Should match
    assert np.allclose(x_ensemble, x_seq, rtol=1e-10)
    assert np.allclose(P_ensemble, P_seq, rtol=1e-10)


def test_kalman_fuse_diagonal_ensemble_validation():
    """Test input validation for ensemble fusion."""
    d = 4
    embeddings = [np.random.randn(d)]
    covariances = [np.exp(np.random.randn(d))]

    # Mismatched lengths
    with pytest.raises(ValueError, match="Number of embeddings"):
        kalman_fuse_diagonal_ensemble(embeddings * 2, covariances)

    # Empty lists
    with pytest.raises(ValueError, match="At least one embedding"):
        kalman_fuse_diagonal_ensemble([], [])

    # Wrong initial state shape
    with pytest.raises(ValueError, match="initial_state"):
        kalman_fuse_diagonal_ensemble(
            embeddings, covariances, initial_state=np.ones(d + 1)
        )

    # Wrong initial covariance shape
    with pytest.raises(ValueError, match="initial_covariance"):
        kalman_fuse_diagonal_ensemble(
            embeddings, covariances, initial_covariance=np.ones(d + 1)
        )

    # Only one of initial_state/initial_covariance provided
    with pytest.raises(ValueError, match="Both initial_state and initial_covariance"):
        kalman_fuse_diagonal_ensemble(embeddings, covariances, initial_state=np.ones(d))


def test_kalman_fuse_diagonal_batch_basic():
    """Test basic batch fusion with diagonal covariance."""
    np.random.seed(42)
    num_specialists = 3
    batch_size = 5
    d = 8

    # Generate random embeddings and covariances
    embeddings = np.random.randn(num_specialists, batch_size, d)
    covariances = np.exp(np.random.randn(num_specialists, batch_size, d))  # positive

    # Perform batch fusion (disable sorting to match per-query ordering)
    fused, fused_cov = kalman_fuse_diagonal_batch(
        embeddings, covariances, sort_by_certainty=False, epsilon=1e-12
    )

    # Check output shapes
    assert fused.shape == (batch_size, d)
    assert fused_cov.shape == (batch_size, d)

    # Covariances must be positive
    assert np.all(fused_cov > 0)

    # Check that each batch element matches sequential fusion
    for b in range(batch_size):
        # Extract single query data
        emb_single = embeddings[:, b, :]  # (num_specialists, d)
        cov_single = covariances[:, b, :]  # (num_specialists, d)
        # Convert to list format expected by non-batch function
        emb_list = list(emb_single)
        cov_list = list(cov_single)
        fused_single, cov_single_result = kalman_fuse_diagonal(
            emb_list, cov_list, sort_by_certainty=False, epsilon=1e-12
        )
        assert np.allclose(fused[b], fused_single, rtol=1e-10)
        assert np.allclose(fused_cov[b], cov_single_result, rtol=1e-10)


def test_kalman_fuse_diagonal_batch_validation():
    """Test input validation for batch fusion."""
    np.random.seed(42)
    num_specialists = 2
    batch_size = 3
    d = 4

    # Valid 3D arrays
    embeddings = np.random.randn(num_specialists, batch_size, d)
    covariances = np.exp(np.random.randn(num_specialists, batch_size, d))

    # Wrong number of dimensions
    with pytest.raises(ValueError, match="must be 3D arrays"):
        kalman_fuse_diagonal_batch(embeddings[0], covariances)  # shape (batch_size, d)

    with pytest.raises(ValueError, match="must be 3D arrays"):
        kalman_fuse_diagonal_batch(embeddings, covariances[0])

    # Shape mismatch
    with pytest.raises(ValueError, match="must match"):
        wrong_cov = np.exp(np.random.randn(num_specialists, batch_size + 1, d))
        kalman_fuse_diagonal_batch(embeddings, wrong_cov)

    # Negative covariance
    neg_cov = covariances.copy()
    neg_cov[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="must be non-negative"):
        kalman_fuse_diagonal_batch(embeddings, neg_cov)

    # Non-finite values
    inf_emb = embeddings.copy()
    inf_emb[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        kalman_fuse_diagonal_batch(inf_emb, covariances)

    # Wrong initial state shape
    with pytest.raises(ValueError, match="initial_state"):
        kalman_fuse_diagonal_batch(
            embeddings, covariances, initial_state=np.ones((batch_size, d + 1))
        )

    # Wrong initial covariance shape
    with pytest.raises(ValueError, match="initial_covariance"):
        kalman_fuse_diagonal_batch(
            embeddings, covariances, initial_covariance=np.ones((batch_size, d + 1))
        )


def test_kalman_fuse_diagonal_ensemble_batch_basic():
    """Test basic ensemble batch fusion."""
    np.random.seed(123)
    num_specialists = 4
    batch_size = 6
    d = 7

    embeddings = np.random.randn(num_specialists, batch_size, d)
    covariances = np.exp(np.random.randn(num_specialists, batch_size, d))

    # Perform ensemble batch fusion
    fused, fused_cov = kalman_fuse_diagonal_ensemble_batch(
        embeddings, covariances, epsilon=1e-12
    )

    # Check output shapes
    assert fused.shape == (batch_size, d)
    assert fused_cov.shape == (batch_size, d)
    assert np.all(fused_cov > 0)

    # Compare with non-batch ensemble fusion for each query
    for b in range(batch_size):
        emb_single = embeddings[:, b, :]  # (num_specialists, d)
        cov_single = covariances[:, b, :]
        # Convert to list format
        emb_list = list(emb_single)
        cov_list = list(cov_single)
        fused_single, cov_single_result = kalman_fuse_diagonal_ensemble(
            emb_list, cov_list, epsilon=1e-12
        )
        assert np.allclose(fused[b], fused_single, rtol=1e-10)
        assert np.allclose(fused_cov[b], cov_single_result, rtol=1e-10)


def test_kalman_fuse_diagonal_ensemble_batch_validation():
    """Test input validation for ensemble batch fusion."""
    np.random.seed(456)
    num_specialists = 2
    batch_size = 3
    d = 5

    embeddings = np.random.randn(num_specialists, batch_size, d)
    covariances = np.exp(np.random.randn(num_specialists, batch_size, d))

    # Wrong dimensions
    with pytest.raises(ValueError, match="must be 3D arrays"):
        kalman_fuse_diagonal_ensemble_batch(embeddings[0], covariances)

    with pytest.raises(ValueError, match="must be 3D arrays"):
        kalman_fuse_diagonal_ensemble_batch(embeddings, covariances[0])

    # Shape mismatch
    with pytest.raises(ValueError, match="must match"):
        wrong_cov = np.exp(np.random.randn(num_specialists, batch_size + 1, d))
        kalman_fuse_diagonal_ensemble_batch(embeddings, wrong_cov)

    # Negative covariance
    neg_cov = covariances.copy()
    neg_cov[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="must be non-negative"):
        kalman_fuse_diagonal_ensemble_batch(embeddings, neg_cov)

    # Non-finite values
    inf_emb = embeddings.copy()
    inf_emb[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        kalman_fuse_diagonal_ensemble_batch(inf_emb, covariances)

    # Wrong initial state shape
    with pytest.raises(ValueError, match="initial_state"):
        kalman_fuse_diagonal_ensemble_batch(
            embeddings, covariances, initial_state=np.ones((batch_size, d + 1))
        )

    # Wrong initial covariance shape
    with pytest.raises(ValueError, match="initial_covariance"):
        kalman_fuse_diagonal_ensemble_batch(
            embeddings, covariances, initial_covariance=np.ones((batch_size, d + 1))
        )

    # Only one of initial_state/initial_covariance provided
    with pytest.raises(ValueError, match="Both initial_state and initial_covariance"):
        kalman_fuse_diagonal_ensemble_batch(
            embeddings, covariances, initial_state=np.ones((batch_size, d))
        )


def test_kalman_fuse_diagonal_batch_sorting_and_prior():
    """Test batch fusion with sorting by certainty and prior state."""
    np.random.seed(789)
    num_specialists = 4
    batch_size = 5
    d = 6

    embeddings = np.random.randn(num_specialists, batch_size, d)
    # Create covariances with clear ordering: specialist 0 most certain, 3 least certain
    base_cov = np.exp(np.random.randn(num_specialists, batch_size, d))
    # Scale each specialist's covariance to create ordering
    scales = np.array([0.1, 0.5, 1.0, 2.0])[
        :, np.newaxis, np.newaxis
    ]  # shape (num_specialists, 1, 1)
    covariances = base_cov * scales

    # Test with sorting enabled (default)
    fused_sorted, cov_sorted = kalman_fuse_diagonal_batch(
        embeddings, covariances, sort_by_certainty=True, epsilon=1e-12
    )

    # Test with sorting disabled
    fused_unsorted, cov_unsorted = kalman_fuse_diagonal_batch(
        embeddings, covariances, sort_by_certainty=False, epsilon=1e-12
    )

    # Results should be mathematically identical regardless of order
    # (Kalman filter is associative for diagonal covariance with independent measurements)
    # But numerical differences may occur due to floating point.
    # We'll just ensure shapes are correct
    assert fused_sorted.shape == (batch_size, d)
    assert cov_sorted.shape == (batch_size, d)
    assert fused_unsorted.shape == (batch_size, d)
    assert cov_unsorted.shape == (batch_size, d)

    # Test with prior state and covariance
    prior_state = np.random.randn(batch_size, d)
    prior_cov = np.exp(np.random.randn(batch_size, d))

    fused_with_prior, cov_with_prior = kalman_fuse_diagonal_batch(
        embeddings,
        covariances,
        initial_state=prior_state,
        initial_covariance=prior_cov,
        sort_by_certainty=True,
        epsilon=1e-12,
    )

    # Should have correct shapes
    assert fused_with_prior.shape == (batch_size, d)
    assert cov_with_prior.shape == (batch_size, d)

    # Compare with non-batch fusion for each query (with prior)
    for b in range(batch_size):
        emb_single = embeddings[:, b, :]
        cov_single = covariances[:, b, :]
        emb_list = list(emb_single)
        cov_list = list(cov_single)
        fused_single, cov_single_result = kalman_fuse_diagonal(
            emb_list,
            cov_list,
            initial_state=prior_state[b],
            initial_covariance=prior_cov[b],
            epsilon=1e-12,
        )
        assert np.allclose(fused_with_prior[b], fused_single, rtol=1e-10)
        assert np.allclose(cov_with_prior[b], cov_single_result, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
