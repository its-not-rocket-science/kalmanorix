"""
Mixed-domain evaluation for Kalmanorix.

This script creates a small retrieval benchmark with:
- two specialist embedders (tech / cooking)
- mixed-domain queries
- four strategies:
    1) hard routing
    2) mean fusion
    3) kalmanorix fusion (precision-weighted)
    4) learned gate fusion

Metric: Recall@k over a document pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
from kalmanorix.uncertainty import KeywordSigma2


def make_keyword_embedder(*, dim: int, seed: int, keywords: set[str], keyword_boost: float = 2.5):
    rng = np.random.default_rng(seed)
    base_dir = rng.normal(size=(dim,))
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-12)

    kw_dir = rng.normal(size=(dim,))
    kw_dir = kw_dir / (np.linalg.norm(kw_dir) + 1e-12)

    def embed(text: str) -> np.ndarray:
        t = text.lower()
        noise = np.zeros(dim, dtype=np.float64)
        for ch in t[:64]:
            noise[(ord(ch) * 13) % dim] += 0.01
        vec = base_dir + noise
        if any(kw in t for kw in keywords):
            vec = vec + keyword_boost * kw_dir
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype(np.float64)

    return embed


@dataclass(frozen=True)
class ToyCorpus:
    docs: List[str]
    doc_ids: List[int]
    queries: List[Tuple[str, int]]  # (query, true_doc_id)


def build_toy_corpus() -> ToyCorpus:
    tech_docs = [
        "Smartphone battery life and fast charging",
        "Laptop CPU performance under load",
        "GPU drivers improve frame rates",
        "Camera sensor and image processing pipeline",
    ]
    cook_docs = [
        "Braise beef slowly with garlic and onion",
        "Simmer stew for hours in a slow cooker",
        "Saute vegetables before baking in the oven",
        "Recipe: simmer stock then reduce sauce",
    ]

    docs = tech_docs + cook_docs
    doc_ids = list(range(len(docs)))

    # Queries include in-domain and mixed-domain phrases
    queries = [
        ("battery lasts all day", 0),
        ("cpu throttles when hot", 1),
        ("braise for hours until tender", 4),
        ("slow cooker simmer recipe", 5),
        ("smartphone battery lasts longer than a slow cooker braise", 0),  # mixed
        ("camera pipeline like reducing a sauce", 3),                      # mixed metaphor
    ]
    return ToyCorpus(docs=docs, doc_ids=doc_ids, queries=queries)


def main() -> None:
    dim = 32

    tech_keywords = {"battery", "smartphone", "cpu", "gpu", "laptop", "camera", "charger", "drivers"}
    cook_keywords = {"braise", "simmer", "slow cooker", "recipe", "garlic", "onion", "saute", "oven", "stew"}

    tech_embed = make_keyword_embedder(dim=dim, seed=7, keywords=tech_keywords)
    cook_embed = make_keyword_embedder(dim=dim, seed=11, keywords=cook_keywords)

    tech = SEF(
        name="tech",
        embed=tech_embed,
        sigma2=KeywordSigma2(tech_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.0),
        meta={"domain": "tech"},
    )
    cook = SEF(
        name="cook",
        embed=cook_embed,
        sigma2=KeywordSigma2(cook_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.0),
        meta={"domain": "cooking"},
    )
    village = Village([tech, cook])

    corpus = build_toy_corpus()

    # Precompute document embeddings for each strategy’s “reference embedder”.
    # For retrieval we need a single doc embedding matrix; we’ll use a simple approach:
    # doc embedding = mean fusion of the doc text across modules (neutral indexing).
    doc_embs = np.stack(
        [np.mean(np.stack([tech.embed(d), cook.embed(d)], axis=0), axis=0) for d in corpus.docs],
        axis=0,
    )

    # Strategies
    strategies = []

    # 1) Hard routing baseline
    strategies.append(("hard", ScoutRouter(mode="hard"), Panoramix(fuser=MeanFuser())))

    # 2) Mean fusion
    strategies.append(("mean", ScoutRouter(mode="all"), Panoramix(fuser=MeanFuser())))

    # 3) Kalmanorix
    strategies.append(("kalman", ScoutRouter(mode="all"), Panoramix(fuser=KalmanorixFuser())))

    # 4) Learned gate (trained on tiny labels)
    gate = LearnedGateFuser(module_a="tech", module_b="cook", n_features=64, steps=200)
    gate.fit(
        texts=[
            "battery life smartphone charger",
            "cpu gpu laptop drivers",
            "braise stew garlic onion",
            "slow cooker simmer recipe",
        ],
        y=[1, 1, 0, 0],
    )
    strategies.append(("gate", ScoutRouter(mode="all"), Panoramix(fuser=gate)))

    print()
    print("== Mixed-domain retrieval ==")
    print(f"docs: {len(corpus.docs)}, queries: {len(corpus.queries)}, dim: {dim}")
    print()

    for name, scout, pan in strategies:
        r1 = eval_retrieval(corpus.queries, doc_embs, village, scout, pan, k=1)
        r3 = eval_retrieval(corpus.queries, doc_embs, village, scout, pan, k=3)
        print(f"{name:>6}  Recall@1={r1:.3f}  Recall@3={r3:.3f}")

    print()


if __name__ == "__main__":
    main()
