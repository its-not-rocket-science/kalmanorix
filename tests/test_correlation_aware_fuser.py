"""Tests for correlation-aware Kalman fusion adjustments."""

from __future__ import annotations

import numpy as np

from kalmanorix import SEF, Village, ScoutRouter, Panoramix, KalmanorixFuser
from kalmanorix.panoramix import CorrelationAwareKalmanFuser
from kalmanorix.kalman_engine.correlation import ResidualCorrelationProfile


def _make_two_module_village() -> Village:
    def a_embed(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64)

    def b_embed(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float64)

    return Village([SEF("a", a_embed, sigma2=1.0), SEF("b", b_embed, sigma2=1.0)])


def test_identical_specialist_edge_case_inflates_posterior_variance() -> None:
    """Identical specialists with high correlation should not over-shrink variance."""
    village = _make_two_module_village()
    scout = ScoutRouter(mode="all")

    baseline = Panoramix(fuser=KalmanorixFuser()).brew("q", village=village, scout=scout)
    baseline_var = float(np.mean(baseline.meta["fused_covariance"]))  # type: ignore[index]

    profile = ResidualCorrelationProfile(
        module_names=["a", "b"],
        correlation_matrix=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    )
    aware = Panoramix(
        fuser=CorrelationAwareKalmanFuser(
            correlation_profile=profile, mode="covariance_inflation"
        )
    ).brew("q", village=village, scout=scout)
    aware_var = float(np.mean(aware.meta["fused_covariance"]))  # type: ignore[index]

    assert aware_var >= baseline_var


def test_near_perfect_correlation_precision_discounting() -> None:
    """Near-perfect correlation should impose strong ESS precision discount."""
    village = _make_two_module_village()
    scout = ScoutRouter(mode="all")
    profile = ResidualCorrelationProfile(
        module_names=["a", "b"],
        correlation_matrix=np.array([[1.0, 0.999], [0.999, 1.0]], dtype=np.float64),
    )
    potion = Panoramix(
        fuser=CorrelationAwareKalmanFuser(
            correlation_profile=profile, mode="effective_sample_size"
        )
    ).brew("q", village=village, scout=scout)

    discount = float(potion.meta["effective_sample_size_discount"])  # type: ignore[index]
    assert discount <= 0.501


def test_independence_reduces_to_baseline_behavior() -> None:
    """Zero off-diagonal correlation should recover baseline Kalman behavior."""
    village = _make_two_module_village()
    scout = ScoutRouter(mode="all")

    baseline = Panoramix(fuser=KalmanorixFuser()).brew("q", village=village, scout=scout)

    profile = ResidualCorrelationProfile(
        module_names=["a", "b"],
        correlation_matrix=np.eye(2, dtype=np.float64),
    )
    aware = Panoramix(
        fuser=CorrelationAwareKalmanFuser(
            correlation_profile=profile, mode="effective_sample_size"
        )
    ).brew("q", village=village, scout=scout)

    assert np.allclose(aware.vector, baseline.vector, atol=1e-12)
    assert np.allclose(
        aware.meta["fused_covariance"], baseline.meta["fused_covariance"], atol=1e-12  # type: ignore[index]
    )
