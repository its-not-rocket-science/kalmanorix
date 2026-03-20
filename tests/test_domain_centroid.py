"""Tests for domain centroid computation."""

import numpy as np
from kalmanorix import SEF, compute_domain_centroid


def test_compute_domain_centroid_basic():
    """Test centroid computation from sample texts."""

    def embed(text: str) -> np.ndarray:
        # Simple embedder: tech -> [1,0], cook -> [0,1], mixed -> [0.5,0.5]
        t = text.lower()
        v = np.array([0.0, 0.0], dtype=np.float64)
        if "tech" in t:
            v += np.array([1.0, 0.0])
        if "cook" in t:
            v += np.array([0.0, 1.0])
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    calibration_texts = ["tech tech", "tech gadget", "tech innovation"]
    centroid = compute_domain_centroid(embed, calibration_texts)
    # All texts are tech-only, centroid should point toward [1,0]
    assert centroid.shape == (2,)
    assert np.allclose(np.linalg.norm(centroid), 1.0, atol=1e-10)
    assert centroid[0] > 0.9  # Mostly x-axis
    assert centroid[1] < 0.1  # Little y-axis


def test_compute_domain_centroid_empty():
    """Empty calibration texts should raise ValueError."""

    def embed(text: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    try:
        compute_domain_centroid(embed, [])
    except ValueError as e:
        assert "must not be empty" in str(e)
    else:
        assert False, "Expected ValueError for empty calibration_texts"


def test_compute_domain_centroid_normalization():
    """Centroid should be normalized."""

    def embed(text: str) -> np.ndarray:
        # Always return same vector
        return np.array([2.0, 3.0])

    calibration_texts = ["a", "b", "c"]
    centroid = compute_domain_centroid(embed, calibration_texts)
    assert np.allclose(np.linalg.norm(centroid), 1.0, atol=1e-10)
    # Direction should be [2,3] normalized
    expected = np.array([2.0, 3.0])
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(centroid, expected, atol=1e-10)


def test_sef_with_domain_centroid():
    """Test SEF.with_domain_centroid creates new SEF with centroid."""

    def embed(text: str) -> np.ndarray:
        if "science" in text:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])

    # Original SEF without centroid
    sef = SEF("science", embed, sigma2=1.0)
    assert sef.domain_centroid is None

    # Compute centroid from calibration texts
    calibration_texts = ["science paper", "science experiment", "science lab"]
    sef_with_centroid = sef.with_domain_centroid(calibration_texts)

    # Should be a new SEF (SEF is frozen)
    assert sef_with_centroid is not sef
    assert sef_with_centroid.name == "science"
    assert sef_with_centroid.domain_centroid is not None
    assert np.allclose(np.linalg.norm(sef_with_centroid.domain_centroid), 1.0)
    # Centroid should point toward science direction [1,0]
    assert sef_with_centroid.domain_centroid[0] > 0.9
    assert sef_with_centroid.domain_centroid[1] < 0.1

    # Original SEF unchanged
    assert sef.domain_centroid is None


def test_sef_with_domain_centroid_already_has_centroid():
    """If SEF already has centroid, with_domain_centroid should replace it."""

    def embed(text: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    # SEF with existing centroid
    old_centroid = np.array([0.0, 1.0])
    sef = SEF("test", embed, sigma2=1.0, domain_centroid=old_centroid)
    assert sef.domain_centroid is old_centroid

    # Compute new centroid
    calibration_texts = ["a", "b"]
    sef_new = sef.with_domain_centroid(calibration_texts)
    assert not np.allclose(sef_new.domain_centroid, old_centroid)
    # New centroid should point to [1,0] direction
    assert sef_new.domain_centroid[0] > 0.9


def test_semantic_routing_with_computed_centroid():
    """Integration test: compute centroids then use semantic routing."""
    from kalmanorix import ScoutRouter, Village

    # Dummy embedder that returns fixed vectors based on query
    def fast_embedder(q: str) -> np.ndarray:
        if "science" in q:
            return np.array([0.9, 0.1])
        elif "tech" in q:
            return np.array([0.1, 0.9])
        else:
            return np.array([0.0, 0.0])

    # Specialist embedders (simplified)
    def science_embed(q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def tech_embed(q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    # Create SEFs without centroids
    science_sef = SEF("science", science_embed, sigma2=1.0)
    tech_sef = SEF("tech", tech_embed, sigma2=1.0)

    # Compute centroids from calibration texts
    science_calibration = ["science paper", "science experiment", "physics research"]
    tech_calibration = ["tech gadget", "software code", "engineering design"]

    science_sef = science_sef.with_domain_centroid(science_calibration)
    tech_sef = tech_sef.with_domain_centroid(tech_calibration)

    # Verify centroids computed
    assert science_sef.domain_centroid is not None
    assert tech_sef.domain_centroid is not None

    # Create village
    village = Village([science_sef, tech_sef])

    # Create semantic router
    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=fast_embedder,
        similarity_threshold=0.5,
        fallback_mode="all",
    )

    # Test routing
    selected = scout.select("science query", village)
    assert len(selected) == 1
    assert selected[0].name == "science"

    selected = scout.select("tech query", village)
    assert len(selected) == 1
    assert selected[0].name == "tech"

    # Query with no clear domain (zero vector) should fallback to all
    selected = scout.select("random query", village)
    assert len(selected) == 2  # both modules


def test_semantic_routing_dynamic_threshold_with_centroids():
    """Test semantic routing with dynamic threshold and computed centroids."""
    from kalmanorix import ScoutRouter, Village
    from kalmanorix.threshold_heuristics import threshold_relative_to_max

    # Dummy fast embedder for routing
    def fast_embedder(q: str) -> np.ndarray:
        if "science" in q:
            return np.array([0.9, 0.1])
        elif "tech" in q:
            return np.array([0.1, 0.9])
        else:
            return np.array([0.0, 0.0])

    # Specialist embedders
    def science_embed(q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def tech_embed(q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    # Create SEFs without centroids
    science_sef = SEF("science", science_embed, sigma2=1.0)
    tech_sef = SEF("tech", tech_embed, sigma2=1.0)

    # Compute centroids from calibration texts
    science_calibration = ["science paper", "science experiment", "physics research"]
    tech_calibration = ["tech gadget", "software code", "engineering design"]

    science_sef = science_sef.with_domain_centroid(science_calibration)
    tech_sef = tech_sef.with_domain_centroid(tech_calibration)

    village = Village([science_sef, tech_sef])

    # Create scout with dynamic threshold (relative to max)
    def dynamic_threshold(
        query: str, query_vec: np.ndarray, similarities: list
    ) -> float:
        return threshold_relative_to_max(query, query_vec, similarities, fraction=0.8)

    scout = ScoutRouter(
        mode="semantic",
        fast_embedder=fast_embedder,
        similarity_threshold=dynamic_threshold,
        fallback_mode="all",
    )

    # Science query: max similarity ~0.99, threshold ~0.79, only science passes
    selected = scout.select("science query", village)
    assert len(selected) == 1
    assert selected[0].name == "science"

    # Tech query: similar result
    selected = scout.select("tech query", village)
    assert len(selected) == 1
    assert selected[0].name == "tech"

    # Query with low similarity to both: max similarity low, threshold may be below min_threshold (0.3)
    # With our dummy embedder returning [0,0] for other queries, similarity will be 0 to both centroids
    # max=0, threshold=0.3 (min_threshold), no modules pass → fallback to all
    selected = scout.select("random query", village)
    assert len(selected) == 2  # fallback to all

    # Test that float threshold still works
    scout_fixed = ScoutRouter(
        mode="semantic",
        fast_embedder=fast_embedder,
        similarity_threshold=0.7,
        fallback_mode="all",
    )
    selected = scout_fixed.select("science query", village)
    assert len(selected) == 1  # science similarity ~0.99 > 0.7
