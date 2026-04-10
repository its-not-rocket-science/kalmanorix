from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from kalmanorix.village import Village
from experiments.registry.fusion import (
    AdaptiveRouteOrFuseStrategy,
    BestSingleSpecialistStrategy,
    LearnedLinearCombinerStrategy,
    RouterTop1Strategy,
    RouterTopKMeanStrategy,
    build_retrieval_baselines,
    rank_query_with_baseline,
)


@dataclass
class _FakeModule:
    name: str
    sigma2_value: float
    vectors: dict[str, np.ndarray]
    alignment_matrix: np.ndarray | None = None

    def embed(self, text: str) -> np.ndarray:
        return self.vectors[text]

    def sigma2_for(self, query: str) -> float:
        _ = query
        return self.sigma2_value


def _build_village() -> Village:
    vectors_a = {
        "q": np.array([1.0, 0.0]),
        "d1": np.array([1.0, 0.0]),
        "d2": np.array([0.0, 1.0]),
    }
    vectors_b = {
        "q": np.array([0.0, 1.0]),
        "d1": np.array([1.0, 0.0]),
        "d2": np.array([0.0, 1.0]),
    }
    vectors_general = {
        "q": np.array([0.8, 0.2]),
        "d1": np.array([0.9, 0.1]),
        "d2": np.array([0.2, 0.8]),
    }

    return Village(
        modules=[
            _FakeModule("specialist_a", 0.2, vectors_a),
            _FakeModule("specialist_b", 0.4, vectors_b),
            _FakeModule("general_qa", 0.3, vectors_general),
        ]
    )


def _rows() -> list[dict]:
    return [
        {
            "query_id": "q1",
            "query_text": "q",
            "candidate_documents": [
                {"doc_id": "d1", "title": "", "text": "d1"},
                {"doc_id": "d2", "title": "", "text": "d2"},
            ],
            "ground_truth_relevant_ids": ["d1"],
            "domain_label": "toy",
        }
    ]


def test_build_retrieval_baselines_contains_required_suite() -> None:
    baselines = build_retrieval_baselines(
        {"generalist_model_name": "general_qa", "router_top_k": 2}
    )
    names = [baseline.name for baseline in baselines]

    assert names == [
        "best_single_specialist",
        "single_generalist_model",
        "uniform_mean_fusion",
        "fixed_weighted_mean_fusion",
        "router_only_top1",
        "router_only_topk_mean",
        "adaptive_route_or_fuse",
        "learned_linear_combiner",
    ]


def test_router_top1_matches_topk_one() -> None:
    village = _build_village()
    candidates = _rows()[0]["candidate_documents"]

    top1 = RouterTop1Strategy()
    topk1 = RouterTopKMeanStrategy(top_k=1)

    ranked_top1, _ = rank_query_with_baseline("q", candidates, village, top1)
    ranked_topk1, _ = rank_query_with_baseline("q", candidates, village, topk1)

    assert ranked_top1 == ranked_topk1


def test_best_single_specialist_fits_expected_model() -> None:
    village = _build_village()
    strategy = BestSingleSpecialistStrategy(train_fraction=1.0)
    strategy.fit(_rows(), village, {"train_fraction": 1.0})

    assert strategy.model_name == "specialist_a"


def test_learned_linear_combiner_weights_are_valid() -> None:
    village = _build_village()
    strategy = LearnedLinearCombinerStrategy(ridge_lambda=1e-3, train_fraction=1.0)
    strategy.fit(_rows(), village, {"train_fraction": 1.0, "ridge_lambda": 1e-3})

    assert strategy.weights.shape == (3,)
    assert np.all(strategy.weights >= 0.0)
    assert float(np.sum(strategy.weights)) == pytest.approx(1.0)


def test_adaptive_policy_chooses_hard_routing_for_high_confidence_query() -> None:
    village = Village(
        modules=[
            _FakeModule(
                "dominant",
                0.1,
                {"q": np.array([1.0, 0.0]), "d1": np.array([1.0, 0.0]), "d2": np.array([0.0, 1.0])},
            ),
            _FakeModule(
                "backup",
                0.5,
                {"q": np.array([0.95, 0.05]), "d1": np.array([1.0, 0.0]), "d2": np.array([0.0, 1.0])},
            ),
        ]
    )
    strategy = AdaptiveRouteOrFuseStrategy(top_k=2)

    diag = strategy.diagnostics_for_query("q", village.modules)

    assert diag is not None
    assert diag["mode"] == "hard_routing"


def test_adaptive_policy_chooses_fusion_for_ambiguous_query() -> None:
    village = Village(
        modules=[
            _FakeModule(
                "m1",
                1.0,
                {
                    "ambiguous": np.array([1.0, 0.0]),
                    "d1": np.array([1.0, 0.0]),
                    "d2": np.array([0.0, 1.0]),
                },
            ),
            _FakeModule(
                "m2",
                1.02,
                {
                    "ambiguous": np.array([0.0, 1.0]),
                    "d1": np.array([1.0, 0.0]),
                    "d2": np.array([0.0, 1.0]),
                },
            ),
            _FakeModule(
                "m3",
                1.05,
                {
                    "ambiguous": np.array([-1.0, 0.0]),
                    "d1": np.array([1.0, 0.0]),
                    "d2": np.array([0.0, 1.0]),
                },
            ),
        ]
    )
    strategy = AdaptiveRouteOrFuseStrategy(top_k=2)

    diag = strategy.diagnostics_for_query("ambiguous", village.modules)

    assert diag is not None
    assert diag["mode"] in {"kalman_fusion", "topk_mean_fusion"}


def test_adaptive_policy_fallback_stays_stable_with_degenerate_sigma2() -> None:
    village = Village(
        modules=[
            _FakeModule(
                "s1",
                0.0,
                {"q": np.array([1.0, 0.0]), "d1": np.array([1.0, 0.0]), "d2": np.array([0.0, 1.0])},
            ),
            _FakeModule(
                "s2",
                0.0,
                {"q": np.array([1.0, 0.0]), "d1": np.array([1.0, 0.0]), "d2": np.array([0.0, 1.0])},
            ),
        ]
    )
    strategy = AdaptiveRouteOrFuseStrategy(top_k=2)

    weights = strategy.weights_for_query("q", village.modules)

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert float(np.sum(weights)) == pytest.approx(1.0)
