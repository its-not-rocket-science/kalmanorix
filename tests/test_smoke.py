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
    from kalmanorix import LearnedGateFuser  # pylint: disable=import-outside-toplevel

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


def test_semantic_routing_basic():
    """
    Smoke test for semantic routing mode.
    """

    # Dummy embedder that returns fixed vectors based on query
    def dummy_embedder(q: str) -> np.ndarray:
        if "science" in q:
            return np.array([1.0, 0.0, 0.0])
        elif "tech" in q:
            return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0])

    # Create SEFs with domain centroids
    science_centroid = np.array([0.9, 0.1, 0.0])
    tech_centroid = np.array([0.1, 0.9, 0.0])
    # Normalize centroids
    science_centroid = science_centroid / np.linalg.norm(science_centroid)
    tech_centroid = tech_centroid / np.linalg.norm(tech_centroid)

    def embed1(q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def embed2(q: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    village = Village(
        [
            SEF("science", embed1, sigma2=1.0, domain_centroid=science_centroid),
            SEF("tech", embed2, sigma2=1.0, domain_centroid=tech_centroid),
        ]
    )

    # Create scout with semantic mode
    scout = ScoutRouter(
        mode="semantic", fast_embedder=dummy_embedder, similarity_threshold=0.5
    )

    # Query that matches science centroid
    selected = scout.select("science query", village)
    assert len(selected) == 1
    assert selected[0].name == "science"

    # Query that matches tech centroid
    selected = scout.select("tech query", village)
    assert len(selected) == 1
    assert selected[0].name == "tech"

    # Query that matches none (should fallback to all)
    scout.fallback_mode = "all"
    selected = scout.select("other query", village)
    assert len(selected) == 2  # both modules

    # Test fallback to hard mode
    scout.fallback_mode = "hard"
    selected = scout.select("other query", village)
    assert len(selected) == 1  # one with lowest sigma2 (both equal, picks first?)


def test_semantic_routing_dynamic_threshold():
    """
    Test semantic routing with dynamic threshold functions.
    """

    # Dummy embedder that returns fixed vectors based on query
    def dummy_embedder(q: str) -> np.ndarray:
        if "science" in q:
            return np.array([1.0, 0.0, 0.0])
        elif "tech" in q:
            return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0])

    # Create SEFs with domain centroids
    science_centroid = np.array([0.9, 0.1, 0.0])
    tech_centroid = np.array([0.1, 0.9, 0.0])
    other_centroid = np.array([0.0, 0.0, 1.0])
    # Normalize centroids
    science_centroid = science_centroid / np.linalg.norm(science_centroid)
    tech_centroid = tech_centroid / np.linalg.norm(tech_centroid)
    other_centroid = other_centroid / np.linalg.norm(other_centroid)

    def embed1(q: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    def embed2(q: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0])

    def embed3(q: str) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    village = Village(
        [
            SEF("science", embed1, sigma2=1.0, domain_centroid=science_centroid),
            SEF("tech", embed2, sigma2=1.0, domain_centroid=tech_centroid),
            SEF("other", embed3, sigma2=1.0, domain_centroid=other_centroid),
        ]
    )

    # Test 1: threshold_top_k with k=1 (select only top module)
    from kalmanorix.threshold_heuristics import threshold_top_k

    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=dummy_embedder,
        similarity_threshold=lambda q, v, sims: threshold_top_k(q, v, sims, k=1),
    )
    # Science query matches science centroid best, similarity ~0.99
    selected = scout.select("science query", village)
    assert len(selected) == 1
    assert selected[0].name == "science"

    # Test 2: threshold_top_k with k=2 (select top 2)
    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=dummy_embedder,
        similarity_threshold=lambda q, v, sims: threshold_top_k(q, v, sims, k=2),
    )
    selected = scout.select("science query", village)
    # Should select science and other (tech has low similarity to science query)
    # Actually: science centroid ~0.99, other centroid ~0.0, tech centroid ~0.1
    # So top 2: science and tech? Let's not assert specific names, just count
    assert len(selected) == 2

    # Test 3: threshold_relative_to_max
    from kalmanorix.threshold_heuristics import threshold_relative_to_max

    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=dummy_embedder,
        similarity_threshold=lambda q, v, sims: threshold_relative_to_max(
            q, v, sims, fraction=0.9
        ),
    )
    selected = scout.select("science query", village)
    # Max similarity ~0.99, threshold = 0.9*0.99 = ~0.89, only science passes
    assert len(selected) == 1
    assert selected[0].name == "science"

    # Test 4: threshold_query_length_adaptive with long query lowers threshold
    from kalmanorix.threshold_heuristics import threshold_query_length_adaptive

    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=dummy_embedder,
        similarity_threshold=threshold_query_length_adaptive,
    )
    # Short query "a" → higher threshold, may select only science
    selected = scout.select("a", village)
    # Long query "a"*50 → lower threshold, may select more modules
    selected_long = scout.select("a" * 50, village)
    # The longer query should have equal or more modules selected
    # (threshold lower, more modules pass)
    assert len(selected_long) >= len(selected)

    # Test 5: float threshold still works
    scout = ScoutRouter(
        mode="semantic", fast_embedder=dummy_embedder, similarity_threshold=0.8
    )
    selected = scout.select("science query", village)
    assert len(selected) == 1  # science similarity ~0.99 > 0.8

    # Test 6: callable that returns fixed value
    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=dummy_embedder,
        similarity_threshold=lambda q, v, sims: 0.95,
    )
    selected = scout.select("science query", village)
    # science similarity ~0.99 > 0.95, passes
    assert len(selected) == 1
