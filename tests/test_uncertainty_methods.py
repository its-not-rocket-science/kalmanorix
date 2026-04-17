"""Tests for practical query-dependent uncertainty methods and diagnostics."""

from __future__ import annotations

import numpy as np

from kalmanorix import SEF
from kalmanorix.uncertainty import (
    CentroidNormPeerSigma2,
    CentroidDistanceSigma2,
    ConstantSigma2,
    EmbeddingNormSigma2,
    KeywordSigma2,
    SimilarityToCentroidSigma2,
    StochasticForwardSigma2,
    UncertaintyMethodConfig,
    apply_uncertainty_baseline_to_specialists,
    build_uncertainty_method,
    create_uncertainty_method,
    summarize_uncertainty_distribution,
    uncertainty_histogram,
)


def _toy_embed(query: str) -> np.ndarray:
    t = query.lower()
    if "strong" in t:
        return np.array([3.0, 0.0], dtype=np.float64)
    if "weak" in t:
        return np.array([0.05, 0.0], dtype=np.float64)
    if "tech" in t:
        return np.array([1.0, 0.0], dtype=np.float64)
    if "cook" in t:
        return np.array([0.0, 1.0], dtype=np.float64)
    return np.array([0.5, 0.5], dtype=np.float64)


def test_embedding_norm_sigma2_is_query_dependent_and_positive() -> None:
    sigma2 = EmbeddingNormSigma2(embed=_toy_embed, base_sigma2=0.2)

    strong = sigma2("strong signal")
    weak = sigma2("weak signal")

    assert strong > 0.0
    assert weak > 0.0
    assert weak > strong


def test_similarity_to_centroid_sigma2_is_lower_near_centroid() -> None:
    sigma2 = SimilarityToCentroidSigma2.from_calibration(
        embed=_toy_embed,
        calibration_texts=["tech", "tech strong"],
        base_sigma2=0.2,
    )

    near = sigma2("tech question")
    far = sigma2("cook recipe")

    assert near > 0.0
    assert far > 0.0
    assert near < far


def test_stochastic_forward_sigma2_positive_and_bounded() -> None:
    rng = np.random.default_rng(42)

    def stochastic_embed(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64) + rng.normal(0.0, 0.1, size=2)

    sigma2 = StochasticForwardSigma2(
        embed_stochastic=stochastic_embed,
        base_sigma2=0.2,
        n_passes=5,
    )
    value = sigma2("query")

    assert value > 0.0
    assert np.isfinite(value)


def test_apply_uncertainty_baseline_to_specialists_sets_query_dependent_sigma2() -> (
    None
):
    sefs = [
        SEF(name="a", embed=_toy_embed, sigma2=1.0),
        SEF(name="b", embed=_toy_embed, sigma2=1.0),
    ]
    updated = apply_uncertainty_baseline_to_specialists(sefs, method="embedding_norm")

    assert len(updated) == 2
    assert callable(updated[0].sigma2)
    assert updated[0].sigma2_for("weak signal") > updated[0].sigma2_for("strong signal")


def test_build_method_requires_calibration_for_centroid() -> None:
    try:
        build_uncertainty_method(method="centroid_similarity", embed=_toy_embed)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "calibration_texts" in str(exc)


def test_uncertainty_diagnostics_report_well_behaved_distribution() -> None:
    sigma2 = EmbeddingNormSigma2(embed=_toy_embed, base_sigma2=0.2)
    queries = ["weak", "strong", "tech", "cook"]

    stats = summarize_uncertainty_distribution(sigma2, queries)
    hist = uncertainty_histogram(sigma2, queries, bins=3)

    assert stats.n_queries == 4
    assert stats.min_sigma2 > 0.0
    assert stats.nonpositive_count == 0
    assert stats.max_sigma2 >= stats.min_sigma2
    assert len(hist["counts"]) == 3
    assert len(hist["bin_edges"]) == 4


def test_common_uncertainty_factory_builds_expected_method_types() -> None:
    calibration_texts = ["battery charging", "gpu thermal", "simmer sauce"]

    constant = create_uncertainty_method(
        config=UncertaintyMethodConfig(method="constant_sigma2", constant_value=0.33),
        embed=_toy_embed,
        calibration_texts=calibration_texts,
    )
    keyword = create_uncertainty_method(
        config=UncertaintyMethodConfig(method="keyword_based_sigma2"),
        embed=_toy_embed,
        calibration_texts=calibration_texts,
    )
    centroid = create_uncertainty_method(
        config=UncertaintyMethodConfig(method="centroid_distance_sigma2"),
        embed=_toy_embed,
        calibration_texts=calibration_texts,
    )
    improved = create_uncertainty_method(
        config=UncertaintyMethodConfig(method="centroid_norm_peer_sigma2"),
        embed=_toy_embed,
        calibration_texts=calibration_texts,
    )

    assert isinstance(constant, ConstantSigma2)
    assert isinstance(keyword, KeywordSigma2)
    assert isinstance(centroid, CentroidDistanceSigma2)
    assert isinstance(improved, CentroidNormPeerSigma2)


def test_common_uncertainty_factory_requires_inputs_for_keyword_method() -> None:
    try:
        create_uncertainty_method(
            config=UncertaintyMethodConfig(method="keyword_based_sigma2"),
            embed=_toy_embed,
            calibration_texts=[],
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "keywords" in str(exc) or "calibration_texts" in str(exc)


def test_centroid_norm_peer_sigma2_tracks_peer_disagreement() -> None:
    tech_sigma = CentroidNormPeerSigma2.from_calibration(
        embed=_toy_embed,
        calibration_texts=["tech strong", "tech"],
        peer_centroids=[np.array([0.0, 1.0], dtype=np.float64)],
        base_sigma2=0.2,
    )
    in_domain = tech_sigma("tech cpu")
    cross_domain = tech_sigma("cook recipe")
    assert in_domain > 0.0
    assert cross_domain > in_domain


def test_sigma2_for_can_use_precomputed_embedding_without_extra_embed_call() -> None:
    calls = {"n": 0}

    def counted_embed(text: str) -> np.ndarray:
        calls["n"] += 1
        return _toy_embed(text)

    sigma2 = EmbeddingNormSigma2(embed=counted_embed, base_sigma2=0.2)
    specialist = SEF(name="counted", embed=counted_embed, sigma2=sigma2)

    precomputed = counted_embed("weak signal")
    before = calls["n"]
    _ = specialist.sigma2_for("weak signal", query_embedding=precomputed)
    assert calls["n"] == before
