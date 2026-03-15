"""
Mixed-domain retrieval evaluation for Kalmanorix (SentenceTransformer specialists).

This script evaluates routing + fusion strategies on a tiny mixed-domain
retrieval benchmark. It is designed to be:

- reproducible (small fixed corpus)
- diagnostic (prints per-group metrics + fusion weights)
- architecture-faithful (docs and queries are embedded using the SAME strategy)

Important design choice
-----------------------
We build a strategy-specific document embedding matrix. In retrieval systems,
queries and documents must live in the same embedding space. If you embed
documents with one strategy (e.g., neutral mean) but embed queries with another,
you can suppress meaningful differences between strategies.

Specialists
-----------
We load two fine-tuned checkpoints (same embedding dimension) so that fusion
can actually change vectors:

- models/tech-minilm
- models/cook-minilm

Uncertainty
-----------
We use centroid-distance sigma² (CentroidDistanceSigma2) built from small
in-domain calibration text lists. This yields a smooth, query-dependent
confidence signal rather than a binary keyword trigger.

Strategies compared
-------------------
- hard  : hard routing by minimum sigma², no fusion (single specialist)
- mean  : uniform averaging of specialist embeddings
- kalman: precision-weighted fusion using sigma²(query)
- gate  : LearnedGateFuser (tiny hashed-feature logistic gate)

Metric
------
Recall@k over a fixed document pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    MeanFuser,
    KalmanorixFuser,
    LearnedGateFuser,
)
from kalmanorix.arena import eval_retrieval
from kalmanorix.types import Embedder, Vec
from kalmanorix.uncertainty import CentroidDistanceSigma2
from kalmanorix.toy_corpus import build_toy_corpus


@dataclass(frozen=True)
class STEmbedder(Embedder):
    """
    SentenceTransformer-backed embedder implementing kalmanorix.types.Embedder.

    The returned vector is unit-normalized (float64) to keep downstream behavior
    stable and comparable across specialists.
    """

    model: SentenceTransformer

    def __call__(self, text: str) -> Vec:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[
            0
        ]
        return v.astype(np.float64)


def make_st_embedder(model_name: str) -> STEmbedder:
    """Load a SentenceTransformer model (path or HF id) and wrap it as an Embedder."""
    model = SentenceTransformer(model_name)
    return STEmbedder(model=model)


def build_doc_matrix(
    docs: List[str],
    *,
    village: Village,
    scout: ScoutRouter,
    pan: Panoramix,
) -> np.ndarray:
    """
    Embed every document using the same routing+fusion strategy as queries.

    This is critical for fair retrieval evaluation: documents and queries must
    share the same embedding space induced by the strategy.
    """
    embs: List[np.ndarray] = []
    for d in docs:
        potion = pan.brew(d, village=village, scout=scout)
        embs.append(potion.vector)
    return np.stack(embs, axis=0)


def pretty(weights: dict[str, float]) -> str:
    """Format weights for stable, readable printing."""
    return "{ " + ", ".join(f"{k}: {v:.3f}" for k, v in weights.items()) + " }"


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


# pylint: disable=too-many-locals
def main() -> None:
    """Run mixed-domain retrieval evaluation with various routing+fusion strategies."""
    # Grab and use command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--tech-path", type=str, default="models/tech-minilm")
    parser.add_argument("--cook-path", type=str, default="models/cook-minilm")
    parser.add_argument("--base-sigma2", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=3.0)
    args = parser.parse_args()
    tech_embed = make_st_embedder(args.tech_path)
    cook_embed = make_st_embedder(args.cook_path)

    # Calibration sets for centroid-distance sigma².
    tech_cal = [
        "Smartphone battery life and fast charging",
        "Laptop CPU performance under sustained load",
        "GPU driver update improves frame rates in games",
        "Camera sensor size affects low-light performance",
        "Thermal management reduces overheating during heavy usage",
        "Background apps reduce battery performance over time",
    ]
    cook_cal = [
        "Braise beef slowly with garlic and onion until tender",
        "Simmer stew for hours in a slow cooker",
        "Saute vegetables before baking in the oven",
        "Reduce a sauce by simmering to concentrate flavour",
        "Taste and adjust seasoning as the sauce reduces",
        "Use a food processor to chop onions quickly",
    ]

    # If you want Kalman to "bite" harder, increase scale (e.g. 6-10).
    tech_sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=tech_embed,
        calibration_texts=tech_cal,
        base_sigma2=args.base_sigma2,
        scale=args.scale,
    )
    cook_sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=cook_embed,
        calibration_texts=cook_cal,
        base_sigma2=args.base_sigma2,
        scale=args.scale,
    )

    tech = SEF(
        name="tech", embed=tech_embed, sigma2=tech_sigma2, meta={"domain": "tech"}
    )
    cook = SEF(
        name="cook", embed=cook_embed, sigma2=cook_sigma2, meta={"domain": "cooking"}
    )
    village = Village([tech, cook])

    # IMPORTANT: corpus comes from the shared library module now
    corpus = build_toy_corpus(british_spelling=True)

    print()
    print("== Specialist disagreement (cosine tech vs cook) ==")
    for q, _ in corpus.queries:
        ct = cos(tech_embed(q), cook_embed(q))
        print(f"  {ct:.3f}  {q!r}")

    # Strategies: (name, router, fusion orchestrator)
    strategies: List[Tuple[str, ScoutRouter, Panoramix]] = [
        ("hard", ScoutRouter(mode="hard"), Panoramix(fuser=MeanFuser())),
        ("mean", ScoutRouter(mode="all"), Panoramix(fuser=MeanFuser())),
        ("kalman", ScoutRouter(mode="all"), Panoramix(fuser=KalmanorixFuser())),
    ]

    gate = LearnedGateFuser(module_a="tech", module_b="cook", n_features=128, steps=400)
    gate.fit(
        texts=[
            "battery life smartphone charger",
            "cpu gpu laptop drivers performance",
            "braise stew garlic onion simmer",
            "slow cooker recipe simmer oven",
        ],
        y=[1, 1, 0, 0],
    )
    strategies.append(
        ("gate", ScoutRouter(mode="all"), Panoramix(fuser=gate))  # type: ignore[arg-type]
    )

    dim = int(tech_embed("probe").shape[0])

    print()
    print("== Mixed-domain retrieval (SentenceTransformer specialists) ==")
    print("specialists: models/tech-minilm + models/cook-minilm")
    print(f"docs: {len(corpus.docs)}, queries: {len(corpus.queries)}, dim: {dim}")
    print()

    # Overall scores (strategy-specific document indexing)
    for name, scout, pan in strategies:
        doc_embs = build_doc_matrix(corpus.docs, village=village, scout=scout, pan=pan)
        r1 = eval_retrieval(corpus.queries, doc_embs, village, scout, pan, k=1)
        r3 = eval_retrieval(corpus.queries, doc_embs, village, scout, pan, k=3)
        print(f"{name:>6}  Recall@1={r1:.3f}  Recall@3={r3:.3f}")

    print()
    print("Mixed-query fusion weights + top-1 predictions:")

    # Precompute doc matrices per strategy once
    doc_mats = {
        name: build_doc_matrix(corpus.docs, village=village, scout=scout, pan=pan)
        for name, scout, pan in strategies
    }

    mixed_queries = [q for q, g in zip(corpus.queries, corpus.groups) if g == "mixed"]
    for q, true_id in mixed_queries:
        print(f'  query: "{q}" (true={true_id})')
        for name, scout, pan in strategies:
            potion = pan.brew(q, village=village, scout=scout)
            sims = doc_mats[name] @ potion.vector
            pred = int(np.argmax(sims))
            ok = "OK" if pred == true_id else "MISS"
            print(f"    {name:>6}: {pretty(potion.weights)}  top1={pred} {ok}")
        print()


if __name__ == "__main__":
    main()
