"""Regression tests for production-readiness core API refactor."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from kalmanorix.scout import ScoutRouter
from kalmanorix.village import SEF, Village
from kalmanorix.kalman_engine.fuser import Panoramix



def test_sef_get_covariance_uses_declared_dimension_without_dummy_embed() -> None:
    """SEF.get_covariance should use explicit dimension metadata, not embed('dummy')."""

    def embed(text: str) -> np.ndarray:
        if text == "dummy":
            raise AssertionError("dummy probes are forbidden")
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)

    sef = SEF(name="core", embed=embed, sigma2=0.5, embedding_dimension=3)
    cov = sef.get_covariance("real query")

    assert cov.shape == (3,)
    assert np.allclose(cov, np.array([0.5, 0.5, 0.5]))



def test_sef_get_covariance_requires_dimension_metadata_when_unavailable() -> None:
    """SEF.get_covariance should fail fast when dimensionality is unknown."""

    def embed(_text: str) -> np.ndarray:
        return np.array([1.0, 2.0], dtype=np.float64)

    sef = SEF(name="unknown-dim", embed=embed, sigma2=1.0)

    with pytest.raises(ValueError, match="Cannot infer embedding dimension"):
        _ = sef.get_covariance("query")



def test_scout_router_hard_mode_defaults_to_sigma2_only() -> None:
    """Default hard routing should not include ad hoc domain-name shortcuts."""

    def embed_a(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def embed_b(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    village = Village(
        [
            SEF("charge", embed_a, sigma2=lambda _q: 10.0),
            SEF("tech", embed_b, sigma2=lambda _q: 0.1),
        ]
    )

    router = ScoutRouter(mode="hard")
    selected = router.select("usb-c power delivery query", village)

    assert len(selected) == 1
    assert selected[0].name == "tech"



def test_scout_router_hard_mode_accepts_optional_heuristic() -> None:
    """Hard-mode hacks can be isolated behind explicit opt-in heuristics."""

    def embed_a(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def embed_b(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    village = Village(
        [
            SEF("charge", embed_a, sigma2=lambda _q: 10.0),
            SEF("tech", embed_b, sigma2=lambda _q: 0.1),
        ]
    )

    def force_charge(_query: str, village_arg: Village) -> SEF | None:
        return next((m for m in village_arg.modules if m.name == "charge"), None)

    router = ScoutRouter(mode="hard", hard_routing_heuristic=force_charge)
    selected = router.select("usb-c power delivery query", village)

    assert len(selected) == 1
    assert selected[0].name == "charge"



def test_scout_router_emits_structured_logs_for_semantic_mode(caplog: pytest.LogCaptureFixture) -> None:
    """Semantic routing should emit logging records (no print debugging)."""

    def fast_embedder(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64)

    def embed(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64)

    village = Village(
        [
            SEF("science", embed, sigma2=1.0, domain_centroid=np.array([1.0, 0.0])),
            SEF("other", embed, sigma2=1.0, domain_centroid=np.array([0.0, 1.0])),
        ]
    )

    router = ScoutRouter(mode="semantic", fast_embedder=fast_embedder, similarity_threshold=0.5)

    with caplog.at_level(logging.DEBUG, logger="kalmanorix.scout"):
        selected = router.select("science query", village)

    assert len(selected) == 1
    assert selected[0].name == "science"
    assert any("ScoutRouter threshold=" in rec.message for rec in caplog.records)
    assert any("ScoutRouter selected modules" in rec.message for rec in caplog.records)



def test_kalman_engine_fuse_batch_failure_fallback_uses_declared_dimension() -> None:
    """Batch failure fallback should infer output dimension from SEF metadata."""

    def failing_embed(_q: str) -> np.ndarray:
        raise RuntimeError("embed failed")

    village = Village(
        [
            SEF(
                "broken",
                failing_embed,
                sigma2=1.0,
                embedding_dimension=5,
            )
        ]
    )
    router = ScoutRouter(mode="all")
    engine = Panoramix(router=router)

    results = engine.fuse_batch(["q1"], village)

    assert len(results) == 1
    emb, cov = results[0]
    assert emb.shape == (5,)
    assert cov.shape == (5,)
    assert np.allclose(emb, np.zeros(5))
    assert np.allclose(cov, np.ones(5) * 1e6)
