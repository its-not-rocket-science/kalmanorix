"""
Toy mixed-domain corpus used across Kalmanorix demos and experiments.

This is intentionally tiny and deterministic so that:
- example scripts remain reproducible
- changes to labels/docs happen in one place
- experiments and scripts evaluate the *same* dataset

Doc index is the list index in `docs` (0..N-1). Queries reference these indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# pylint: disable=line-too-long
@dataclass(frozen=True)
class ToyCorpus:
    """Tiny mixed-domain corpus for retrieval evaluation."""

    docs: List[str]
    doc_ids: List[int]
    queries: List[Tuple[str, int]]
    groups: List[str]


def build_toy_corpus(*, british_spelling: bool = True) -> ToyCorpus:
    """
    Build a small mixed-domain retrieval dataset with confusers.

    The confuser docs intentionally share surface vocabulary across domains to
    force the retriever to rely on deeper semantics rather than keyword overlap.

    Parameters
    ----------
    british_spelling:
        If True, uses "optimisation" / "flavour". If False, uses US spellings.
        This only affects the literal document strings, not the indices.
    """
    optimisation = "optimisation" if british_spelling else "optimization"
    flavour = "flavour" if british_spelling else "flavor"

    tech_docs = [
        "Smartphone battery life and fast charging",
        "Laptop CPU performance under sustained load",
        "GPU driver update improves frame rates in games",
        "Camera sensor size affects low-light performance (pixel pitch, noise, dynamic range, optics)",
    ]
    cook_docs = [
        "Braise beef slowly with garlic and onion until tender",
        "Simmer stew for hours in a slow cooker",
        "Saute vegetables before baking in the oven",
        f"Reduce a sauce by simmering to concentrate {flavour}",
    ]
    confusers = [
        f"Battery {optimisation} settings: OS background app limits, screen brightness, and power-saving modes to extend battery life",
        "Use a food processor to chop onions quickly for a stew recipe",
        "Camera-ready plating: improve presentation with garnish and sauce reduction",
        "Thermal load: prevent overheating and CPU/GPU throttling with heatsinks, airflow, and better thermal paste",
        "Slow cooker liner helps cleanup after braising and simmering",
        "GPU acceleration improves image processing in camera pipelines",
        "Fast charging via USB-C Power Delivery (PD): PD negotiation (PDO/RDO), PPS, charger wattage, cable e-marker limits, and compatibility/handshake",
        "Oven heat affects moisture: reduce temperature for slow cooking",
        "Battery optimisation: reduce background activity (like reducing a sauce) to improve performance",
    ]

    docs = tech_docs + cook_docs + confusers
    doc_ids = list(range(len(docs)))

    # IMPORTANT: true_doc_id refers to indices in `docs` (see docs list above).
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
        ("food processor chop onions for stew", 9, "cook"),
        (
            "reduce background activity to extend battery life — like reducing a sauce",
            16,
            "mixed",
        ),
        (
            "thermal load spikes cause overheating like an oven's heat (reduce power draw)",
            11,
            "mixed",
        ),
        ("camera pipeline acceleration on gpu", 13, "mixed"),
        (
            "USB-C PD fast charging: source/sink negotiation (PDO/RDO), PPS APDO, current limits, e-marked cables, and power contracts",
            14,
            "mixed",
        ),
        (
            "battery lasts all day on my smartphone (background apps drain it) — unlike a slow cooker braise",
            0,
            "mixed",
        ),
    ]

    queries = [(q, did) for (q, did, _g) in queries_and_groups]
    groups = [_g for (_q, _did, _g) in queries_and_groups]
    return ToyCorpus(docs=docs, doc_ids=doc_ids, queries=queries, groups=groups)


def print_doc_index(corpus: ToyCorpus) -> None:
    """Print a stable document index mapping (doc_id -> doc text)."""
    print("Document index:")
    for i, d in enumerate(corpus.docs):
        print(f"  {i:2d}: {d}")
