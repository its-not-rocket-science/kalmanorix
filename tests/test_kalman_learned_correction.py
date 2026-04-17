"""Tests for Kalman + learned correction benchmark."""

from __future__ import annotations

import numpy as np

from kalmanorix.benchmarks.kalman_learned_correction import (
    KalmanLearnedCorrection,
    LearnedCorrectionConfig,
    _build_features,
    _precision_weights,
    run_kalman_learned_correction,
)


def test_corrected_weights_are_normalized() -> None:
    rng = np.random.default_rng(0)
    n, k, d = 12, 4, 6
    specialists = rng.normal(size=(n, k, d))
    sigma2 = np.abs(rng.normal(size=(n, k))) + 0.05
    router_scores = rng.normal(size=(n, k))
    query_lengths = rng.uniform(0.1, 1.0, size=(n,))

    features = _build_features(specialists, sigma2, router_scores, query_lengths)
    kalman_w = _precision_weights(sigma2)
    oracle = np.full((n, k), 1.0 / k)

    model = KalmanLearnedCorrection(
        LearnedCorrectionConfig(model_type="linear", n_specialists=k)
    )
    model.fit(features, kalman_w, oracle)
    corrected = model.predict_weights(features, kalman_w)

    assert np.all(np.isfinite(corrected))
    assert np.allclose(np.sum(corrected, axis=1), 1.0, atol=1e-8)
    assert np.all(corrected >= 0.0)


def test_degenerate_equal_uncertainty_is_stable() -> None:
    rng = np.random.default_rng(1)
    n, k, d = 16, 3, 8
    specialists = rng.normal(size=(n, k, d))
    sigma2 = np.full((n, k), 0.2, dtype=np.float64)
    router_scores = np.zeros((n, k), dtype=np.float64)
    query_lengths = np.full(n, 0.5, dtype=np.float64)

    features = _build_features(specialists, sigma2, router_scores, query_lengths)
    kalman_w = _precision_weights(sigma2)
    oracle = np.full((n, k), 1.0 / k)

    model = KalmanLearnedCorrection(
        LearnedCorrectionConfig(model_type="linear", n_specialists=k, kalman_anchor=0.9)
    )
    model.fit(features, kalman_w, oracle)
    corrected = model.predict_weights(features, kalman_w)

    expected = np.full((n, k), 1.0 / k)
    assert np.allclose(kalman_w, expected)
    assert np.allclose(corrected, expected, atol=3e-2)


def test_checkpoint_roundtrip_is_deterministic() -> None:
    cfg = LearnedCorrectionConfig(model_type="linear", random_seed=7)
    summary = run_kalman_learned_correction(config=cfg)
    checkpoint_dict = summary["fit"]["checkpoint"]

    from kalmanorix.benchmarks.kalman_learned_correction import CorrectionCheckpoint

    cp = CorrectionCheckpoint(**checkpoint_dict)
    model_a = KalmanLearnedCorrection.from_checkpoint(cp)
    model_b = KalmanLearnedCorrection.from_checkpoint(cp)

    rng = np.random.default_rng(22)
    n, k, d = 10, cfg.n_specialists, cfg.dimension
    specialists = rng.normal(size=(n, k, d))
    sigma2 = np.abs(rng.normal(size=(n, k))) + 0.05
    router_scores = rng.normal(size=(n, k))
    ql = rng.uniform(size=(n,))

    features = _build_features(specialists, sigma2, router_scores, ql)
    kalman_w = _precision_weights(sigma2)

    w_a = model_a.predict_weights(features, kalman_w)
    w_b = model_b.predict_weights(features, kalman_w)
    assert np.allclose(w_a, w_b)
