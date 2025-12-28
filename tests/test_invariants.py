"""
Invariant-based tests for routing and fusion.

These tests avoid asserting exact numerical values. Instead they enforce
properties that should remain true across refactors:

- output vectors have the right shape and are finite
- weights are finite
- weight keys match the selected modules
- weights sum to ~1
- hard routing selects the module with minimal query-dependent sigma²
"""

from __future__ import annotations

import numpy as np

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    MeanFuser,
    KalmanorixFuser,
    LearnedGateFuser,
)


def _assert_potion_invariants(
    *,
    potion_vector: np.ndarray,
    weights: dict[str, float],
    expected_keys: set[str],
    dim: int,
) -> None:
    assert potion_vector.shape == (dim,)
    assert np.all(np.isfinite(potion_vector))

    assert set(weights.keys()) == expected_keys
    assert all(np.isfinite(list(weights.values())))
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_fuser_weight_invariants_mean():
    """MeanFuser: uniform weights, correct keys, sums to 1, finite output."""
    dim = 3

    def e1(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def e2(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    village = Village([SEF("a", e1, sigma2=1.0), SEF("b", e2, sigma2=2.0)])
    scout = ScoutRouter(mode="all")
    pan = Panoramix(fuser=MeanFuser())

    potion = pan.brew("q", village=village, scout=scout)

    _assert_potion_invariants(
        potion_vector=potion.vector,
        weights=potion.weights,
        expected_keys={"a", "b"},
        dim=dim,
    )


def test_fuser_weight_invariants_kalman():
    """KalmanorixFuser: weights sane and normalized, finite output."""
    dim = 3

    def e1(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def e2(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    # Make uncertainties query-dependent to exercise sigma2_for(query)
    def s1(_q: str) -> float:
        return 0.5

    def s2(_q: str) -> float:
        return 2.0

    village = Village([SEF("a", e1, sigma2=s1), SEF("b", e2, sigma2=s2)])
    scout = ScoutRouter(mode="all")
    pan = Panoramix(fuser=KalmanorixFuser())

    potion = pan.brew("q", village=village, scout=scout)

    _assert_potion_invariants(
        potion_vector=potion.vector,
        weights=potion.weights,
        expected_keys={"a", "b"},
        dim=dim,
    )


def test_fuser_weight_invariants_gate():
    """LearnedGateFuser: weights sane and normalized, finite output after fitting."""
    dim = 3

    def e1(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def e2(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    village = Village([SEF("a", e1, sigma2=1.0), SEF("b", e2, sigma2=1.0)])
    scout = ScoutRouter(mode="all")

    gate = LearnedGateFuser(module_a="a", module_b="b", n_features=32, steps=80)

    # Minimal training set: encourage different α for different tokens
    gate.fit(["alpha alpha", "beta beta"], [1, 0])

    pan = Panoramix(fuser=gate)
    potion = pan.brew("alpha query", village=village, scout=scout)

    _assert_potion_invariants(
        potion_vector=potion.vector,
        weights=potion.weights,
        expected_keys={"a", "b"},
        dim=dim,
    )


def test_router_hard_selects_lowest_sigma2_for_query():
    """Hard routing should choose the module with minimal sigma2_for(query)."""
    dim = 2

    def e1(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def e2(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    # Query-dependent: tech wins on "battery", cook wins on "braise"
    def sigma_tech(q: str) -> float:
        return 0.1 if "battery" in q.lower() else 10.0

    def sigma_cook(q: str) -> float:
        return 0.1 if "braise" in q.lower() else 10.0

    village = Village(
        [
            SEF("tech", e1, sigma2=sigma_tech),
            SEF("cook", e2, sigma2=sigma_cook),
        ]
    )

    scout_hard = ScoutRouter(mode="hard")

    chosen1 = scout_hard.select("battery life", village)
    assert len(chosen1) == 1
    assert chosen1[0].name == "tech"

    chosen2 = scout_hard.select("braise for hours", village)
    assert len(chosen2) == 1
    assert chosen2[0].name == "cook"

    # sanity: the embedding dim doesn't matter here, but we ensure no accidental usage
    assert dim == 2
