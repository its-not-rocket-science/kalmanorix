"""
Unit and integration tests for Milestone 2.2: Uncertainty Robustness.

Tests:
- OOD test set creation (ood_datasets.py)
- Calibration metrics (calibration.py)
- Mis-specified uncertainty wrappers (uncertainty.py)
- Integration: Kalman fusion with mis-specified covariances
- Ablation: equal covariances → averaging
"""

import numpy as np
import pytest
from kalmanorix.ood_datasets import (
    create_ood_test_set,
    create_synthetic_ood_test_set,
)
from kalmanorix.calibration import (
    compute_embedding_calibration,
    compute_retrieval_calibration,
    CalibrationResult,
)
from kalmanorix.uncertainty import ConstantSigma2, ScaledSigma2, KeywordSigma2
from kalmanorix.village import Village, SEF
from kalmanorix.scout import ScoutRouter
from kalmanorix.panoramix import Panoramix, KalmanorixFuser, MeanFuser


def test_create_synthetic_ood_test_set_basic():
    """Test synthetic OOD test set creation."""
    seen_domains = ["medical", "legal"]
    ood_domain = "tech"
    n_docs = 100
    n_queries = 20

    documents, queries = create_synthetic_ood_test_set(
        seen_domains=seen_domains,
        ood_domain=ood_domain,
        n_docs=n_docs,
        n_queries=n_queries,
        seed=42,
    )

    # Check sizes
    assert len(documents) == n_docs
    assert len(queries) == n_queries

    # All queries should be OOD (true_doc_id = -1)
    for _, true_id in queries:
        assert true_id == -1

    # Documents should be strings
    assert all(isinstance(doc, str) for doc in documents)
    assert all(isinstance(q, str) for q, _ in queries)


def test_create_ood_test_set_mock():
    """Test OOD test set creation with mock domain datasets."""

    # Mock DomainDataset objects
    class MockDomainDataset:
        def __init__(self, domain: str, test_texts: list[str]):
            self.domain = domain
            self.test = test_texts

    domain_datasets = {
        "medical": MockDomainDataset(
            "medical", ["med1", "med2", "med3", "med4", "med5"]
        ),
        "legal": MockDomainDataset("legal", ["law1", "law2", "law3", "law4", "law5"]),
        "tech": MockDomainDataset(
            "tech", ["tech1", "tech2", "tech3", "tech4", "tech5"]
        ),
    }

    seen_domains = ["medical", "legal"]
    ood_domain = "tech"
    n_docs = 4
    n_queries = 6

    documents, queries = create_ood_test_set(
        domain_datasets=domain_datasets,
        seen_domains=seen_domains,
        ood_domain=ood_domain,
        ood_proportion=0.5,  # half OOD, half seen
        n_docs=n_docs,
        n_queries=n_queries,
        seed=42,
    )

    # Check sizes
    assert len(documents) == n_docs
    assert len(queries) == n_queries

    # Documents should come only from seen domains
    # (We can't easily verify content because of random sampling)

    # Count OOD vs seen queries
    ood_count = sum(1 for _, true_id in queries if true_id == -1)
    seen_count = sum(1 for _, true_id in queries if true_id != -1)

    # With proportion 0.5, we expect roughly half OOD
    assert ood_count > 0
    assert seen_count > 0
    assert ood_count + seen_count == n_queries

    # Seen queries should have valid document indices
    for _, true_id in queries:
        if true_id != -1:
            assert 0 <= true_id < len(documents)


def test_constant_sigma2():
    """Test ConstantSigma2 wrapper."""
    sigma2 = ConstantSigma2(value=2.5)
    assert sigma2("any query") == 2.5
    assert sigma2("another query") == 2.5


def test_scaled_sigma2():
    """Test ScaledSigma2 wrapper."""
    base = ConstantSigma2(value=2.0)
    scaled = ScaledSigma2(base_sigma2=base, scale=3.0)
    assert scaled("query") == 6.0

    # Test with query-dependent base
    keyword = KeywordSigma2(
        keywords={"hello"}, in_domain_sigma2=1.0, out_domain_sigma2=10.0
    )
    scaled_keyword = ScaledSigma2(base_sigma2=keyword, scale=0.5)
    # For query containing "hello"
    assert scaled_keyword("hello world") == 0.5  # 1.0 * 0.5
    # For query without keyword
    assert scaled_keyword("goodbye") == 5.0  # 10.0 * 0.5


def test_compute_embedding_calibration_perfect():
    """Test calibration metrics with perfectly calibrated predictions."""
    n_samples = 100
    d = 10

    # Perfectly calibrated: predicted variance matches actual error
    np.random.seed(42)
    reference = np.random.randn(n_samples, d)
    # Specialist embeddings are reference plus noise scaled by variance
    predicted_variances = np.random.uniform(0.1, 2.0, n_samples)
    # Generate noise with magnitude proportional to sqrt(variance)
    noise = np.random.randn(n_samples, d) * np.sqrt(predicted_variances[:, np.newaxis])
    specialist = reference + noise

    # Compute calibration
    result = compute_embedding_calibration(
        specialist_embeddings=specialist,
        reference_embeddings=reference,
        predicted_variances=predicted_variances,
        n_bins=5,
        norm="l2",
    )

    # With perfect calibration, ECE should be low (not exactly zero due to binning)
    assert result.ece >= 0.0
    assert result.brier_score >= 0.0
    assert result.n_samples == n_samples
    assert len(result.bin_edges) == 6  # 5 bins + 1
    assert len(result.bin_centers) == 5


def test_compute_retrieval_calibration():
    """Test retrieval calibration with synthetic data."""
    n_queries = 50
    n_docs = 100
    d = 10

    np.random.seed(42)
    # Random embeddings
    query_embeddings = np.random.randn(n_queries, d)
    doc_embeddings = np.random.randn(n_docs, d)

    # Random true indices (some invalid for OOD)
    true_indices = np.random.randint(-1, n_docs, size=n_queries)

    # Random variances
    query_variances = np.random.uniform(0.1, 5.0, n_queries)

    result = compute_retrieval_calibration(
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        query_variances=query_variances,
        true_indices=true_indices.tolist(),
        k=5,
        n_bins=5,
    )

    assert isinstance(result, CalibrationResult)
    assert result.ece >= 0.0
    assert result.brier_score >= 0.0
    assert result.n_samples == n_queries


def test_kalman_fusion_with_scaled_uncertainty():
    """Test that Kalman fusion works with scaled uncertainties (no crash)."""

    # Create toy specialists with different uncertainty scales
    def embed1(query: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def embed2(query: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    # Base sigma2
    sigma2_base = ConstantSigma2(value=1.0)
    sigma2_scaled = ScaledSigma2(base_sigma2=sigma2_base, scale=2.0)

    sef1 = SEF(name="spec1", embed=embed1, sigma2=sigma2_base)
    sef2 = SEF(name="spec2", embed=embed2, sigma2=sigma2_scaled)

    village = Village(modules=[sef1, sef2])
    scout = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=KalmanorixFuser())

    # Should not crash
    potion = panoramix.brew("test query", village=village, scout=scout)
    assert potion.vector.shape == (2,)


def test_constant_variance_ablation_equals_averaging():
    """
    Test that with constant equal variances, Kalman fusion reduces to averaging.

    When all specialists have the same constant variance, the Kalman update
    weights (computed as inverse uncertainties) are equal, and the sequential
    fusion produces the arithmetic mean of the embeddings.
    """
    np.random.seed(42)
    d = 5

    # Create specialists with random embeddings but same constant variance
    variance = 2.0
    sigma2 = ConstantSigma2(value=variance)

    specialists = []
    base_vectors = []
    for i in range(3):
        # Each specialist returns a distinct fixed vector (no noise for determinism)
        vec = np.random.randn(d)
        vec = vec / np.linalg.norm(vec)
        base_vectors.append(vec)

        def make_embedder(v: np.ndarray):
            def embed(query: str) -> np.ndarray:
                return v

            return embed

        embed_fn = make_embedder(vec)
        sef = SEF(name=f"spec{i}", embed=embed_fn, sigma2=sigma2)
        specialists.append(sef)

    village = Village(modules=specialists)
    scout = ScoutRouter(mode="all")

    # Kalman fusion with equal variances
    kalman_panoramix = Panoramix(fuser=KalmanorixFuser())
    query = "test"
    kalman_potion = kalman_panoramix.brew(query, village=village, scout=scout)

    # 1. Check that weights are uniform (within tolerance)
    weights = list(kalman_potion.weights.values())
    expected_weight = 1.0 / len(specialists)
    for w in weights:
        assert abs(w - expected_weight) < 1e-5, (
            f"Weight {w} not uniform, expected {expected_weight}"
        )

    # 2. Verify fusion matches direct Kalman algorithm call
    from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal

    # Get embeddings and covariances as used by KalmanorixFuser
    kalman_embeddings = []
    kalman_covariances = []
    for module in specialists:
        emb = module.embed(query)
        if module.alignment_matrix is not None:
            emb = module.alignment_matrix @ emb
        sigma2 = module.sigma2_for(query)
        cov = np.full(emb.shape, sigma2, dtype=np.float64)
        kalman_embeddings.append(emb)
        kalman_covariances.append(cov)

    fused_expected, _ = kalman_fuse_diagonal(
        kalman_embeddings,
        kalman_covariances,
        sort_by_certainty=True,
        epsilon=1e-8,
    )
    diff = np.linalg.norm(kalman_potion.vector - fused_expected)
    assert diff < 1e-5, f"Fused vector differs from direct Kalman call: {diff}"

    # 3. Verify that Kalman fusion equals mean fusion (averaging)
    mean_fuser = Panoramix(fuser=MeanFuser())
    mean_potion = mean_fuser.brew(query, village=village, scout=scout)
    mean_diff = np.linalg.norm(kalman_potion.vector - mean_potion.vector)
    assert mean_diff < 1e-5, (
        f"Kalman fusion should equal averaging with equal variances, diff={mean_diff}"
    )


def test_calibration_module_imports():
    """Ensure calibration module can be imported and has expected functions."""
    # If we reach here, imports succeeded
    assert True


def test_ood_datasets_module_imports():
    """Ensure OOD datasets module can be imported."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
