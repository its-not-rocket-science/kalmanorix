"""
Minimal Kalmanorix fusion demo.

This example demonstrates how multiple specialist embedding modules can be
combined at query time using different fusion strategies:

1. Hard routing (choose a single specialist)
2. Mean fusion (uniform averaging)
3. KalmanorixFuser (uncertainty / precision-weighted fusion)
4. LearnedGateFuser (tiny learned gating baseline)

The goal is not performance, but *behavioral clarity*: you should be able to
read the printed weights and immediately understand how each strategy responds
to a mixed-domain query.

This file is intentionally dependency-light and fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

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

    Returns an object implementing kalmanorix.types.Embedder:
        embed(text) -> np.ndarray[dim] (unit-normalized float64)
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


# pylint: disable=too-many-locals
def main() -> None:
    """Run minimal fusion demo with toy keyword-sensitive specialists."""
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

    # Tiny labeled dataset: 1 => tech, 0 => cooking
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

    gate = Panoramix(fuser=gate_fuser)
    potion_gate = gate.brew(query, village=village, scout=scout_all)

    def pretty(weights: dict[str, float]) -> str:
        return "{ " + ", ".join(f"{k}: {v:.3f}" for k, v in weights.items()) + " }"

    print()
    print("Query:", query)
    print()
    print("Hard routing (sigma2 baseline)")
    print("  weights:", pretty(potion_hard.weights))
    print()
    print("Mean fusion")
    print("  weights:", pretty(potion_mean.weights))
    print()
    print("KalmanorixFuser (precision-weighted by sigma2)")
    print("  weights:", pretty(potion_kal.weights))
    print()
    print("LearnedGateFuser (learned text gate)")
    print("  weights:", pretty(potion_gate.weights))
    print()

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    print("Cosine similarities between fused vectors:")
    print("  mean vs kalman:", f"{cos(potion_mean.vector, potion_kal.vector):.3f}")
    print("  mean vs gate:  ", f"{cos(potion_mean.vector, potion_gate.vector):.3f}")
    print("  kalman vs gate:", f"{cos(potion_kal.vector, potion_gate.vector):.3f}")
    print()


if __name__ == "__main__":
    main()
