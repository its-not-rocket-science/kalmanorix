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
- kalman : precision-weighted fusion using sigma²(query)
- gate  : LearnedGateFuser (tiny hashed-feature logistic gate)

Metric
------
Recall@k over a fixed document pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

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
from kalmanorix.arena import eval_retrieval
from kalmanorix.uncertainty import CentroidDistanceSigma2

EmbedderFn = Callable[[str], np.ndarray]


def make_st_embedder(model_name: str) -> EmbedderFn:
    """
    Build a sentence-transformers embedder returning a unit-normalized vector.

    Parameters
    ----------
    model_name:
        HuggingFace model id or local path (e.g. "models/tech-minilm").

    Returns
    -------
    Callable[[str], np.ndarray]
        A function mapping text -> embedding vector (float64).
    """
    from sentence_transformers import (
        SentenceTransformer,
    )  # local import keeps core deps minimal

    model = SentenceTransformer(model_name)

    def embed(text: str) -> np.ndarray:
        v = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float64)

    return embed


@dataclass(frozen=True)
class ToyCorpus:
    """
    Tiny mixed-domain corpus for retrieval evaluation.

    docs:
        Document texts. Retrieval is performed over this fixed pool.
    doc_ids:
        Integer ids for docs (0..N-1).
    queries:
        List of (query_text, true_doc_id) pairs.
    groups:
        Group label per query: "tech" | "cook" | "mixed".
    """

    docs: List[str]
    doc_ids: List[int]
    queries: List[Tuple[str, int]]
    groups: List[str]


def build_toy_corpus() -> ToyCorpus:
    """
    Build a small mixed-domain retrieval dataset with confusers.

    The confuser docs intentionally share surface vocabulary across domains to
    force the retriever to rely on deeper semantics rather than keyword overlap.
    """
    tech_docs = [
        "Smartphone battery life and fast charging",
        "Laptop CPU performance under sustained load",
        "GPU driver update improves frame rates in games",
        "Camera sensor size affects low-light performance",
    ]
    cook_docs = [
        "Braise beef slowly with garlic and onion until tender",
        "Simmer stew for hours in a slow cooker",
        "Saute vegetables before baking in the oven",
        "Reduce a sauce by simmering to concentrate flavor",
    ]
    confusers = [
        "Battery optimization recipe: reduce background activity to improve performance",
        "Use a food processor to chop onions quickly for a stew recipe",
        "Camera-ready plating: improve presentation with garnish and sauce reduction",
        "Thermal load: avoid overheating by reducing power draw under performance spikes",
        "Slow cooker liner helps cleanup after braising and simmering",
        "GPU acceleration improves image processing in camera pipelines",
        "Recipe for faster charging: choose the right charger and cable",
        "Oven heat affects moisture: reduce temperature for slow cooking",
    ]

    docs = tech_docs + cook_docs + confusers
    doc_ids = list(range(len(docs)))

    queries_and_groups: List[Tuple[str, int, str]] = [
        ("battery lasts all day", 0, "tech"),
        ("fast charging with the right charger", 0, "tech"),
        ("cpu throttles when hot under sustained load", 1, "tech"),
        ("gpu driver update improves frame rates", 2, "tech"),
        ("camera low light performance sensor size", 3, "tech"),
        ("braise for hours until tender", 4, "cook"),
        ("slow cooker stew simmer for hours", 5, "cook"),
        ("saute vegetables then bake in the oven", 6, "cook"),
        ("reduce sauce by simmering", 7, "cook"),
        ("food processor chop onions for stew", 13, "cook"),
        ("reduce background activity like reducing a sauce", 8, "mixed"),
        ("thermal load feels like oven heat", 11, "mixed"),
        ("camera pipeline acceleration on gpu", 14, "mixed"),
        ("recipe for faster charging", 15, "mixed"),
        ("smartphone battery lasts longer than a slow cooker braise", 0, "mixed"),
    ]

    queries = [(q, did) for (q, did, _g) in queries_and_groups]
    groups = [_g for (_q, _did, _g) in queries_and_groups]
    return ToyCorpus(docs=docs, doc_ids=doc_ids, queries=queries, groups=groups)


def build_doc_matrix(
    docs: List[str], village: Village, scout: ScoutRouter, pan: Panoramix
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


def main() -> None:
    # Load fine-tuned specialists (same architecture/dim => fusion is valid).
    tech_embed = make_st_embedder("models/tech-minilm")
    cook_embed = make_st_embedder("models/cook-minilm")

    # Calibration sets for centroid-distance sigma².
    # These do not need to be large; they just need to be in-domain.
    TECH_CAL = [
        "Smartphone battery life and fast charging",
        "Laptop CPU performance under sustained load",
        "GPU driver update improves frame rates in games",
        "Camera sensor size affects low-light performance",
        "Thermal management reduces overheating during heavy usage",
        "Background apps reduce battery performance over time",
    ]
    COOK_CAL = [
        "Braise beef slowly with garlic and onion until tender",
        "Simmer stew for hours in a slow cooker",
        "Saute vegetables before baking in the oven",
        "Reduce a sauce by simmering to concentrate flavor",
        "Taste and adjust seasoning as the sauce reduces",
        "Use a food processor to chop onions quickly",
    ]

    # If you want Kalman to "bite" harder, increase scale (e.g. 6-10).
    tech_sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=tech_embed,
        calibration_texts=TECH_CAL,
        base_sigma2=0.2,
        scale=8.0,
    )
    cook_sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=cook_embed,
        calibration_texts=COOK_CAL,
        base_sigma2=0.2,
        scale=8.0,
    )

    tech = SEF(
        name="tech", embed=tech_embed, sigma2=tech_sigma2, meta={"domain": "tech"}
    )
    cook = SEF(
        name="cook", embed=cook_embed, sigma2=cook_sigma2, meta={"domain": "cooking"}
    )
    village = Village([tech, cook])

    corpus = build_toy_corpus()

    # divergence check
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

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
    strategies.append(("gate", ScoutRouter(mode="all"), Panoramix(fuser=gate)))

    # Determine embedding dimension from one embed call (stable across specialists).
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

    # Precompute doc matrices per strategy once for this section
    doc_mats = {
        name: build_doc_matrix(corpus.docs, village=village, scout=scout, pan=pan)
        for name, scout, pan in strategies
    }

    mixed_queries = [q for q, g in zip(corpus.queries, corpus.groups) if g == "mixed"]
    for q, true_id in mixed_queries:
        print(f'  query: "{q}" (true={true_id})')
        for name, scout, pan in strategies:
            potion = pan.brew(q, village=village, scout=scout)
            doc_embs = doc_mats[name]

            sims = doc_embs @ potion.vector
            pred = int(np.argmax(sims))
            ok = "✓" if pred == true_id else "✗"

            print(f"    {name:>6}: {pretty(potion.weights)}  top1={pred} {ok}")
        print()


if __name__ == "__main__":
    main()
