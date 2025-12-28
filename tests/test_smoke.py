"""
Smoke tests for the Kalmanorix pipeline.

These tests intentionally avoid asserting numerical correctness or performance.
Their purpose is to verify that the *end-to-end orchestration* of Kalmanorix
components works:

    SEF → Village → ScoutRouter → Panoramix → Fuser → Potion

If these tests fail, it indicates a breaking change in the public API or
a wiring error between components, not a modeling issue.
"""

import numpy as np
from kalmanorix import SEF, Village, ScoutRouter, Panoramix, KalmanorixFuser


def test_brew_runs():
    """
    Verify that Panoramix.brew executes end-to-end with a minimal setup.

    This is a pure smoke test:
    - one trivial embedding module
    - a default router
    - KalmanorixFuser as the fusion strategy

    The test asserts only that:
    - no exceptions are raised
    - the output embedding has the expected shape

    It deliberately avoids checking numerical values, which are covered by
    more specific unit tests and examples.
    """

    def e(_q: str) -> np.ndarray:
        """Trivial embedder returning a fixed 2D vector."""
        return np.array([1.0, 0.0])

    village = Village([SEF("a", e, sigma2=1.0)])
    scout = ScoutRouter()
    panoramix = Panoramix(fuser=KalmanorixFuser())

    potion = panoramix.brew("q", village=village, scout=scout)

    assert potion.vector.shape == (2,)

def test_learned_gate_fuser_basic_properties():
    """
    Do-not-regress test for LearnedGateFuser.

    This test ensures that:
    - LearnedGateFuser can be trained on minimal data
    - Panoramix.brew runs end-to-end with the gate
    - Output vector shape is correct
    - Fusion weights are well-formed and sum to ~1

    This guards against accidental API or wiring regressions,
    not against model quality.
    """
    from kalmanorix import LearnedGateFuser

    def e1(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def e2(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    village = Village(
        [
            SEF("a", e1, sigma2=1.0),
            SEF("b", e2, sigma2=1.0),
        ]
    )

    gate = LearnedGateFuser(
        module_a="a",
        module_b="b",
        n_features=32,
        steps=50,  # keep test fast
    )

    texts = ["alpha alpha", "beta beta"]
    labels = [1, 0]
    gate.fit(texts, labels)

    panoramix = Panoramix(fuser=gate)
    scout = ScoutRouter(mode="all")

    potion = panoramix.brew("alpha query", village=village, scout=scout)

    # Shape invariant
    assert potion.vector.shape == (2,)

    # Weight invariants
    assert set(potion.weights.keys()) == {"a", "b"}
    total_weight = potion.weights["a"] + potion.weights["b"]
    assert abs(total_weight - 1.0) < 1e-6
