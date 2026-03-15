"""
Tests for covariance estimation (Milestone 1.1).
"""

import numpy as np
import pytest  # type: ignore  # pylint: disable=import-error

from kalmanorix.kalman_engine.covariance import (
    CovarianceEstimator,
    EmpiricalCovariance,
    DistanceBasedCovariance,
    DiagonalCovariance,
    ConstantCovariance,
    ScalarCovariance,
    KNNBasedCovariance,
    DomainBasedCovariance,
)


def test_empirical_covariance_basic():
    """Test EmpiricalCovariance with synthetic data."""
    np.random.seed(42)
    d = 10
    n_samples = 500  # Enough for stable estimates

    # Generate data with known diagonal covariance
    true_mean = np.random.randn(d)
    true_cov = np.exp(np.random.randn(d))  # Positive variances

    embeddings = np.zeros((n_samples, d))
    for i in range(d):
        embeddings[:, i] = np.random.normal(
            true_mean[i], np.sqrt(true_cov[i]), n_samples
        )

    estimator = EmpiricalCovariance(embeddings, epsilon=1e-12)

    # Shape and positivity
    assert estimator.fixed_covariance.shape == (d,)
    assert np.all(estimator.fixed_covariance > 0)

    # Unbiased within reasonable tolerance (due to finite samples)
    relative_error = np.abs(estimator.fixed_covariance - true_cov) / true_cov
    assert np.all(relative_error < 0.3), f"Large relative error: {relative_error.max()}"

    # Estimate method returns same vector
    def dummy_model(text):  # pylint: disable=unused-argument
        return np.zeros(d)

    cov = estimator.estimate(dummy_model, "any text")
    assert np.allclose(cov, estimator.fixed_covariance)


def test_empirical_covariance_small_samples():
    """Test with few samples (should still work with epsilon)."""
    d = 5
    embeddings = np.random.randn(2, d)  # Only 2 samples
    estimator = EmpiricalCovariance(embeddings)
    assert estimator.fixed_covariance.shape == (d,)
    assert np.all(estimator.fixed_covariance >= 0)


def test_diagonal_covariance_container():
    """Test DiagonalCovariance utility class."""
    d = 7
    diag = np.exp(np.random.randn(d))
    container = DiagonalCovariance(diag)

    assert container.dimension == d
    assert np.allclose(container.diagonal, diag)
    assert container.uncertainty_score() == np.sum(diag)
    assert 0 < container.confidence_score() < 1

    # Full matrix conversion
    full = container.to_full()
    assert full.shape == (d, d)
    assert np.allclose(np.diag(full), diag)
    assert np.allclose(full - np.diag(np.diag(full)), 0)  # Off-diagonals zero

    # Negative variance raises error
    with pytest.raises(ValueError):
        DiagonalCovariance(np.array([1.0, -0.1, 2.0]))


def test_distance_based_covariance():
    """Test distance‑based uncertainty scaling."""
    d = 8
    n_ref = 30
    np.random.seed(123)

    # Reference set
    ref_embeddings = np.random.randn(n_ref, d)
    # Normalise for cosine distance
    ref_embeddings = ref_embeddings / np.linalg.norm(
        ref_embeddings, axis=1, keepdims=True
    )

    # Base covariance
    base_estimator = EmpiricalCovariance(ref_embeddings * 0.1)

    estimator = DistanceBasedCovariance(
        base_estimator=base_estimator,
        reference_texts=[f"text{i}" for i in range(n_ref)],
        reference_embeddings=ref_embeddings,
        alpha=2.0,
        distance_metric="cosine",
    )

    # Close point (near reference)
    close_emb = ref_embeddings[0] + 0.01 * np.random.randn(d)
    close_emb = close_emb / np.linalg.norm(close_emb)

    def dummy_close(text):  # pylint: disable=unused-argument
        return close_emb

    # Far point (random direction)
    far_emb = np.random.randn(d)
    far_emb = far_emb / np.linalg.norm(far_emb)

    def dummy_far(text):  # pylint: disable=unused-argument
        return far_emb

    cov_close = estimator.estimate(dummy_close, "close")
    cov_far = estimator.estimate(dummy_far, "far")

    # Far point should have higher uncertainty
    assert np.sum(cov_far) > np.sum(cov_close)

    # Scaling should preserve shape (same relative variances)
    assert np.allclose(cov_close / cov_far, cov_close[0] / cov_far[0])


def test_covariance_estimator_abc():
    """CovarianceEstimator is an ABC and cannot be instantiated."""
    with pytest.raises(TypeError):
        CovarianceEstimator()  # type: ignore  # pylint: disable=abstract-class-instantiated


@pytest.mark.skipif(
    not hasattr(np, "errstate"),
    reason="NumPy error state context manager not available",
)
def test_numerical_stability():
    """Test that zero‑variance dimensions are handled safely."""
    d = 6
    # One dimension has zero variance (all same value)
    embeddings = np.random.randn(100, d)
    embeddings[:, 2] = 5.0  # Constant column

    estimator = EmpiricalCovariance(embeddings, epsilon=1e-8)
    cov = estimator.fixed_covariance
    assert cov[2] >= 1e-8  # Should be clipped to epsilon
    assert np.all(np.isfinite(cov))


def test_constant_covariance():
    """Test ConstantCovariance with fixed diagonal."""
    d = 7
    diag = np.exp(np.random.randn(d))
    estimator = ConstantCovariance(diag, epsilon=1e-8)

    def dummy_model(text):  # pylint: disable=unused-argument
        return np.zeros(d)

    cov = estimator.estimate(dummy_model, "any text")

    assert cov.shape == (d,)
    assert np.allclose(cov, diag)
    assert np.all(cov >= 1e-8)

    # Negative variance raises error
    with pytest.raises(ValueError):
        ConstantCovariance(np.array([1.0, -0.1, 2.0]))


def test_scalar_covariance():
    """Test ScalarCovariance with constant and callable sigma2."""
    d = 5

    def dummy_model(text):  # pylint: disable=unused-argument
        return np.zeros(d)

    # Constant scalar
    estimator = ScalarCovariance(0.3, epsilon=1e-8)
    cov = estimator.estimate(dummy_model, "any")
    assert cov.shape == (d,)
    assert np.allclose(cov, 0.3)

    # Callable scalar
    def sigma2_func(query: str) -> float:
        return 0.5 if "certain" in query else 2.0

    estimator = ScalarCovariance(sigma2_func, epsilon=1e-8)
    cov1 = estimator.estimate(dummy_model, "certain query")
    assert np.allclose(cov1, 0.5)
    cov2 = estimator.estimate(dummy_model, "unknown query")
    assert np.allclose(cov2, 2.0)

    # Epsilon clipping
    estimator = ScalarCovariance(-1.0, epsilon=0.1)
    cov = estimator.estimate(dummy_model, "any")
    assert np.all(cov >= 0.1)


def test_knn_based_covariance():
    """Test KNNBasedCovariance with synthetic reference data."""
    np.random.seed(123)
    d = 6
    n_ref = 20
    k = 3

    # Generate reference embeddings and covariances
    reference_embeddings = np.random.randn(n_ref, d)
    reference_covariances = np.exp(np.random.randn(n_ref, d))  # positive

    estimator = KNNBasedCovariance(
        reference_embeddings,
        reference_covariances,
        k=k,
        distance_metric="cosine",
        epsilon=1e-8,
    )

    def dummy_model(text):  # pylint: disable=unused-argument
        return np.random.randn(d)

    cov = estimator.estimate(dummy_model, "any text")

    # Shape and positivity
    assert cov.shape == (d,)
    assert np.all(cov >= 0)

    # Test with euclidean distance
    estimator_eucl = KNNBasedCovariance(
        reference_embeddings,
        reference_covariances,
        k=k,
        distance_metric="euclidean",
    )
    cov_eucl = estimator_eucl.estimate(dummy_model, "any")
    assert cov_eucl.shape == (d,)

    # Test k >= n_ref (uses all reference points with distance weighting)
    estimator_all = KNNBasedCovariance(
        reference_embeddings,
        reference_covariances,
        k=n_ref + 5,
    )
    cov_all = estimator_all.estimate(dummy_model, "any")
    # Should be weighted average of all reference covariances
    # Check each dimension is between min and max of reference values
    for dim in range(d):
        dim_min = np.min(reference_covariances[:, dim])
        dim_max = np.max(reference_covariances[:, dim])
        assert dim_min <= cov_all[dim] <= dim_max


def test_domain_based_covariance():
    """Test DomainBasedCovariance scaling by domain hint."""
    np.random.seed(42)
    d = 8
    n_samples = 100

    # Create base estimator
    embeddings = np.random.randn(n_samples, d)
    base = EmpiricalCovariance(embeddings, epsilon=1e-8)
    base_cov = base.fixed_covariance

    # Domain factors
    domain_factors = {"medical": 0.5, "legal": 2.0, "tech": 1.0}
    default_factor = 1.5

    estimator = DomainBasedCovariance(
        base,
        domain_factors,
        default_factor=default_factor,
        epsilon=1e-8,
    )

    def dummy_model(text):  # pylint: disable=unused-argument
        return np.zeros(d)

    # Known domain: medical (factor 0.5)
    cov_medical = estimator.estimate(dummy_model, "any", domain_hint="medical")
    assert np.allclose(cov_medical, base_cov * 0.5)

    # Known domain: legal (factor 2.0)
    cov_legal = estimator.estimate(dummy_model, "any", domain_hint="legal")
    assert np.allclose(cov_legal, base_cov * 2.0)

    # Unknown domain: uses default factor
    cov_unknown = estimator.estimate(dummy_model, "any", domain_hint="finance")
    assert np.allclose(cov_unknown, base_cov * default_factor)

    # No domain hint: uses default factor
    cov_none = estimator.estimate(dummy_model, "any", domain_hint=None)
    assert np.allclose(cov_none, base_cov * default_factor)

    # Zero factor clipping (epsilon ensures positivity)
    estimator_zero = DomainBasedCovariance(
        base,
        {"zero": 0.0},
        default_factor=1.0,
        epsilon=0.1,
    )
    cov_zero = estimator_zero.estimate(dummy_model, "any", domain_hint="zero")
    # Factor should be clipped to epsilon (0.1)
    assert np.allclose(cov_zero, base_cov * 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
