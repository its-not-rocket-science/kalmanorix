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


def make_keyword_embedder(
    *,
    dim: int,
    seed: int,
    keywords: set[str],
    keyword_boost: float = 2.5,
) -> callable:
    """
    Construct a toy keyword-sensitive embedder.

    This function returns a callable `embed(text) -> np.ndarray[dim]` that:
    - produces a deterministic base embedding (controlled by `seed`)
    - injects small, deterministic noise derived from the input text
    - adds a strong directional bias if any domain-specific keywords appear

    The result is a simple stand-in for a "specialist" embedding model whose
    confidence increases when it sees in-domain language.

    Parameters
    ----------
    dim:
        Dimensionality of the embedding space.
    seed:
        Random seed controlling the base and keyword directions.
    keywords:
        Set of substrings that indicate the embedder's domain.
    keyword_boost:
        Strength of the domain-specific directional bias.

    Returns
    -------
    callable
        A function mapping `text: str` to a normalized numpy vector of shape (dim,).
    """
    rng = np.random.default_rng(seed)
    base_dir = rng.normal(size=(dim,))
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-12)

    kw_dir = rng.normal(size=(dim,))
    kw_dir = kw_dir / (np.linalg.norm(kw_dir) + 1e-12)

    def embed(text: str) -> np.ndarray:
        """
        Embed text into a normalized vector.

        The embedding is mostly fixed per model, with:
        - small deterministic variation per input string
        - a strong boost if domain keywords are detected
        """
        t = text.lower()

        # Tiny deterministic "noise" so different texts differ slightly
        noise = np.zeros(dim, dtype=np.float64)
        for ch in t[:64]:
            noise[(ord(ch) * 13) % dim] += 0.01

        vec = base_dir + noise

        if any(kw in t for kw in keywords):
            vec = vec + keyword_boost * kw_dir

        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype(np.float64)

    return embed


def main() -> None:
    """
    Run the minimal fusion demonstration.

    This function:
    - defines two specialist embedding modules (tech, cooking)
    - constructs a mixed-domain query
    - applies four fusion strategies
    - prints the resulting fusion weights for comparison

    The printed output should make it immediately obvious:
    - which strategy prefers which specialist
    - how uncertainty-weighted fusion differs from learned gating
    """
    dim = 16

    tech_keywords = {
        "battery", "smartphone", "cpu", "gpu", "laptop", "android", "ios", "camera", "charger"
    }
    cook_keywords = {
        "braise", "simmer", "slow cooker", "recipe", "garlic", "onion", "saute", "oven", "stew"
    }

    tech = SEF(
        name="tech",
        embed=make_keyword_embedder(dim=dim, seed=7, keywords=tech_keywords),
        sigma2=0.2,  # lower variance => higher confidence
        meta={"domain": "tech"},
    )
    cook = SEF(
        name="cook",
        embed=make_keyword_embedder(dim=dim, seed=11, keywords=cook_keywords),
        sigma2=1.2,  # higher variance => lower confidence
        meta={"domain": "cooking"},
    )

    village = Village([tech, cook])

    query = "This smartphone battery lasts longer than a slow cooker braise."

    # Routing strategies
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
        """Pretty-print fusion weights."""
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
        """Cosine similarity between two vectors."""
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    print("Cosine similarities between fused vectors:")
    print("  mean vs kalman:", f"{cos(potion_mean.vector, potion_kal.vector):.3f}")
    print("  mean vs gate:  ", f"{cos(potion_mean.vector, potion_gate.vector):.3f}")
    print("  kalman vs gate:", f"{cos(potion_kal.vector, potion_gate.vector):.3f}")
    print()


if __name__ == "__main__":
    main()
