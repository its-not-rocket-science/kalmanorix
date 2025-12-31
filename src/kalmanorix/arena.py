"""Lightweight retrieval evaluation utilities for Kalmanorix experiments/tests."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .panoramix import Panoramix
from .scout import ScoutRouter
from .village import Village


def recall_at_k(ranked_doc_ids: List[int], true_id: int, k: int) -> float:
    """Return 1.0 if `true_id` appears in the top-k ranked ids, else 0.0."""
    return float(true_id in ranked_doc_ids[:k])


# pylint: disable=too-many-arguments,too-many-positional-arguments
def eval_retrieval(
    queries: List[Tuple[str, int]],
    doc_embs: np.ndarray,
    village: Village,
    scout: ScoutRouter,
    panoramix: Panoramix,
    k: int = 10,
) -> float:
    """Compute mean Recall@k over (query, true_doc_id) pairs against `doc_embs`."""
    hits = 0.0
    for q, true_id in queries:
        potion = panoramix.brew(q, village=village, scout=scout)
        scores = doc_embs @ potion.vector
        ranked = list(np.argsort(-scores))
        hits += recall_at_k(ranked, true_id, k)
    return hits / max(len(queries), 1)
