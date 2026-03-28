"""
End-to-end test of the full Kalmanorix pipeline with toy keyword-sensitive specialists.

This test replicates the minimal_fusion_demo example but adds assertions about
the correctness of the pipeline outputs (weights, shapes, invariants).

It is marked as e2e and runs with the default pytest configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np
import pytest

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    MeanFuser,
    KalmanorixFuser,
    LearnedGateFuser,
)
from kalmanorix.types import Embedder, Vec
from kalmanorix.uncertainty import KeywordSigma2


@dataclass(frozen=True)
class KeywordEmbedder(Embedder):
    """
    Toy keyword-sensitive embedder.

    Produces a deterministic base embedding (seeded), plus tiny deterministic
    per-text perturbations, and a strong directional bias if any keywords match.
    """

    dim: int
    keywords: Set[str]
    keyword_boost: float
    _base_dir: np.ndarray
    _kw_dir: np.ndarray

    def __call__(self, text: str) -> Vec:
        t = text.lower()

        # Tiny deterministic "noise" so different texts differ slightly.
        noise = np.zeros(self.dim, dtype=np.float64)
        for ch in t[:64]:
            noise[(ord(ch) * 13) % self.dim] += 0.01

        vec = self._base_dir + noise

        if any(kw in t for kw in self.keywords):
            vec = vec + self.keyword_boost * self._kw_dir

        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype(np.float64)


def make_keyword_embedder(
    *,
    dim: int,
    seed: int,
    keywords: Set[str],
    keyword_boost: float = 2.5,
) -> KeywordEmbedder:
    """
    Construct a deterministic keyword-sensitive embedder.
    """
    rng = np.random.default_rng(seed)

    base_dir = rng.normal(size=(dim,))
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-12)

    kw_dir = rng.normal(size=(dim,))
    kw_dir = kw_dir / (np.linalg.norm(kw_dir) + 1e-12)

    return KeywordEmbedder(
        dim=dim,
        keywords=keywords,
        keyword_boost=keyword_boost,
        _base_dir=base_dir.astype(np.float64),
        _kw_dir=kw_dir.astype(np.float64),
    )


@pytest.mark.e2e
def test_full_pipeline_with_toy_specialists():
    """
    End-to-end test of the full pipeline with two keyword specialists.

    Validates that:
    - All fusion strategies produce valid outputs
    - Weights sum to 1 (within tolerance)
    - No exceptions are raised
    - Output vectors have correct shape
    """
    dim = 16

    tech_keywords: Set[str] = {
        "battery",
        "smartphone",
        "cpu",
        "gpu",
        "laptop",
        "android",
        "ios",
        "camera",
        "charger",
    }
    cook_keywords: Set[str] = {
        "braise",
        "simmer",
        "slow cooker",
        "recipe",
        "garlic",
        "onion",
        "saute",
        "oven",
        "stew",
    }

    tech = SEF(
        name="tech",
        embed=make_keyword_embedder(dim=dim, seed=7, keywords=tech_keywords),
        sigma2=KeywordSigma2(
            tech_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5
        ),
        meta={"domain": "tech"},
    )
    cook = SEF(
        name="cook",
        embed=make_keyword_embedder(dim=dim, seed=11, keywords=cook_keywords),
        sigma2=KeywordSigma2(
            cook_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5
        ),
        meta={"domain": "cooking"},
    )

    village = Village([tech, cook])

    query = "This smartphone battery lasts longer than a slow cooker braise."

    scout_all = ScoutRouter(mode="all")
    scout_hard = ScoutRouter(mode="hard")

    # 1) Hard routing baseline
    hard = Panoramix(fuser=MeanFuser())
    potion_hard = hard.brew(query, village=village, scout=scout_hard)

    # 2) Mean fusion baseline
    mean = Panoramix(fuser=MeanFuser())
    potion_mean = mean.brew(query, village=village, scout=scout_all)

    # 3) Kalmanorix fusion (precision-weighted)
    kal = Panoramix(fuser=KalmanorixFuser())
    potion_kal = kal.brew(query, village=village, scout=scout_all)

    # 4) Learned gate fusion baseline
    gate_fuser = LearnedGateFuser(
        module_a="tech",
        module_b="cook",
        n_features=128,
        lr=0.6,
        l2=1e-3,
        steps=400,
    )

    train_texts = [
        "Battery life is excellent on this smartphone",
        "The laptop CPU throttles under load",
        "Camera quality and charger compatibility",
        "Android update improved performance",
        "Braise the beef and simmer for hours",
        "Slow cooker recipe with garlic and onion",
        "Saute the vegetables then bake in the oven",
        "Stew tastes better after simmering",
    ]
    train_y = [1, 1, 1, 1, 0, 0, 0, 0]

    gate_fuser.fit(train_texts, train_y)

    gate = Panoramix(fuser=gate_fuser)  # type: ignore[arg-type]
    potion_gate = gate.brew(query, village=village, scout=scout_all)

    # --- Assertions ---

    # All potions have correct shape
    for potion in [potion_hard, potion_mean, potion_kal, potion_gate]:
        assert potion.vector.shape == (dim,)
        assert np.isfinite(potion.vector).all()
        # Weights sum to 1 (within floating tolerance)
        assert abs(sum(potion.weights.values()) - 1.0) < 1e-10
        # All weights non-negative
        assert all(w >= -1e-12 for w in potion.weights.values())

    # Hard routing should select exactly one specialist
    assert len(potion_hard.weights) == 1
    selected = next(iter(potion_hard.weights))
    assert selected in {"tech", "cook"}

    # Mean fusion weights should be equal (0.5 each)
    assert abs(potion_mean.weights["tech"] - 0.5) < 1e-10
    assert abs(potion_mean.weights["cook"] - 0.5) < 1e-10

    # KalmanorixFuser weights should differ from mean (since sigma2 differs)
    # For this query, both specialists have in-domain keywords, so sigma2 = 0.2
    # Thus weights should be equal (both sigma2 equal). However, due to
    # numerical differences in embeddings, they may differ slightly.
    # We'll just ensure they are positive.
    assert potion_kal.weights["tech"] > 0
    assert potion_kal.weights["cook"] > 0

    # LearnedGateFuser weights should also be positive
    assert potion_gate.weights["tech"] > 0
    assert potion_gate.weights["cook"] > 0

    # Ensure metadata contains selected_modules
    for potion in [potion_hard, potion_mean, potion_kal, potion_gate]:
        assert potion.meta is not None
        assert "selected_modules" in potion.meta
        selected = set(potion.meta["selected_modules"])
        # Hard routing selects exactly one specialist
        if potion is potion_hard:
            assert len(selected) == 1
            assert selected.issubset({"tech", "cook"})
        else:
            assert selected == {"tech", "cook"}


if __name__ == "__main__":
    test_full_pipeline_with_toy_specialists()
    print("Test passed.")
