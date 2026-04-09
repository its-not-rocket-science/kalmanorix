"""Numerical stability tests for Kalman fusion under extreme covariance regimes."""

from __future__ import annotations

import time

import numpy as np

from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal


def _assert_stable_result(
    fused_embedding: np.ndarray,
    fused_covariance: np.ndarray,
    input_embeddings: list[np.ndarray],
) -> None:
    """Validate basic numerical stability invariants for fused outputs."""
    assert fused_embedding.ndim == 1
    assert fused_covariance.ndim == 1
    assert fused_embedding.shape == fused_covariance.shape

    # Finite outputs: no NaN or Inf allowed.
    assert np.all(np.isfinite(fused_embedding))
    assert np.all(np.isfinite(fused_covariance))

    # Positive-definite in the diagonal sense (all diagonal entries > 0).
    assert np.all(fused_covariance > 0.0)

    # Reasonable norm bound: fused state should not explode relative to inputs.
    input_norms = np.array([np.linalg.norm(emb) for emb in input_embeddings])
    fused_norm = np.linalg.norm(fused_embedding)
    assert fused_norm <= (np.max(input_norms) * 1.1 + 1e-6)


def test_extreme_covariance_scales_small_large_mixed() -> None:
    """Fusion remains stable for very small, very large, and mixed covariances."""
    rng = np.random.default_rng(42)
    d = 64
    n_models = 5
    embeddings = [rng.normal(size=d).astype(np.float64) for _ in range(n_models)]

    covariance_sets = [
        [np.full(d, 1e-6, dtype=np.float64) for _ in range(n_models)],
        [np.full(d, 1e6, dtype=np.float64) for _ in range(n_models)],
        [np.full(d, scale, dtype=np.float64) for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6)],
    ]

    for covariances in covariance_sets:
        fused_embedding, fused_covariance = kalman_fuse_diagonal(
            embeddings, covariances, sort_by_certainty=True, epsilon=1e-12
        )
        _assert_stable_result(fused_embedding, fused_covariance, embeddings)


def test_near_singular_diagonal_covariances() -> None:
    """Near-singular diagonal covariance entries should not destabilize updates."""
    rng = np.random.default_rng(7)
    d = 128
    embeddings = [rng.normal(size=d).astype(np.float64) for _ in range(4)]
    covariances = [
        np.full(d, 1e-14, dtype=np.float64),
        np.full(d, 1e-12, dtype=np.float64),
        np.full(d, 1e-10, dtype=np.float64),
        np.full(d, 1e-8, dtype=np.float64),
    ]

    fused_embedding, fused_covariance = kalman_fuse_diagonal(
        embeddings, covariances, sort_by_certainty=True, epsilon=1e-15
    )

    _assert_stable_result(fused_embedding, fused_covariance, embeddings)


def test_ill_conditioned_updates_dominant_and_ignored_model() -> None:
    """Very certain models dominate while very uncertain models are mostly ignored."""
    d = 96
    dominant = np.ones(d, dtype=np.float64)
    uncertain = -np.ones(d, dtype=np.float64)
    neutral = np.zeros(d, dtype=np.float64)

    embeddings = [dominant, uncertain, neutral]
    covariances = [
        np.full(d, 1e-9, dtype=np.float64),  # dominant certainty
        np.full(d, 1e9, dtype=np.float64),  # effectively ignored
        np.full(d, 1.0, dtype=np.float64),
    ]

    fused_embedding, fused_covariance = kalman_fuse_diagonal(
        embeddings, covariances, sort_by_certainty=True, epsilon=1e-12
    )

    _assert_stable_result(fused_embedding, fused_covariance, embeddings)

    # Dominant model should strongly influence output.
    assert np.linalg.norm(fused_embedding - dominant) < 0.2 * np.sqrt(d)


def test_edge_case_single_model_only() -> None:
    """Single-model fusion should return that model with stable covariance."""
    rng = np.random.default_rng(123)
    d = 80
    embedding = rng.normal(size=d).astype(np.float64)
    covariance = np.full(d, 0.5, dtype=np.float64)

    fused_embedding, fused_covariance = kalman_fuse_diagonal([embedding], [covariance])

    _assert_stable_result(fused_embedding, fused_covariance, [embedding])
    assert np.allclose(fused_embedding, embedding)
    assert np.allclose(fused_covariance, covariance)


def test_edge_case_identical_embeddings() -> None:
    """Identical embeddings should remain stable and close to the shared value."""
    d = 72
    embedding = np.linspace(-1.0, 1.0, d, dtype=np.float64)
    embeddings = [embedding.copy() for _ in range(5)]
    covariances = [
        np.full(d, val, dtype=np.float64) for val in (1e-4, 1e-2, 1, 10, 100)
    ]

    fused_embedding, fused_covariance = kalman_fuse_diagonal(
        embeddings, covariances, sort_by_certainty=True
    )

    _assert_stable_result(fused_embedding, fused_covariance, embeddings)
    assert np.allclose(fused_embedding, embedding, atol=1e-8)


def test_edge_case_conflicting_embeddings() -> None:
    """Opposing/orthogonal embeddings should still produce finite, bounded output."""
    d = 64
    e1 = np.zeros(d, dtype=np.float64)
    e2 = np.zeros(d, dtype=np.float64)
    e3 = np.zeros(d, dtype=np.float64)

    e1[0] = 1.0  # +x
    e2[0] = -1.0  # -x (opposite)
    e3[1] = 1.0  # +y (orthogonal)

    embeddings = [e1, e2, e3]
    covariances = [
        np.full(d, 0.1, dtype=np.float64),
        np.full(d, 0.1, dtype=np.float64),
        np.full(d, 0.1, dtype=np.float64),
    ]

    fused_embedding, fused_covariance = kalman_fuse_diagonal(embeddings, covariances)

    _assert_stable_result(fused_embedding, fused_covariance, embeddings)
    assert np.linalg.norm(fused_embedding) <= 1.0 + 1e-8


def test_performance_five_models_768d_under_50ms_cpu() -> None:
    """Fusion for 5 models at d=768 should complete in under 50 ms on CPU."""
    rng = np.random.default_rng(2026)
    d = 768
    n_models = 5

    embeddings = [rng.normal(size=d).astype(np.float64) for _ in range(n_models)]
    # Log-uniform scales for realistic spread and numeric stress.
    scales = np.exp(rng.uniform(np.log(1e-6), np.log(1e6), size=n_models))
    covariances = [np.full(d, s, dtype=np.float64) for s in scales]

    # Warmup to avoid first-call overhead affecting timing.
    kalman_fuse_diagonal(embeddings, covariances, sort_by_certainty=True)

    start = time.perf_counter()
    fused_embedding, fused_covariance = kalman_fuse_diagonal(
        embeddings, covariances, sort_by_certainty=True
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    _assert_stable_result(fused_embedding, fused_covariance, embeddings)
    assert elapsed_ms < 50.0, f"Expected < 50ms, got {elapsed_ms:.3f}ms"
