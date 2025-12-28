from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .village import Village
from .scout import ScoutRouter
from .panoramix import Panoramix

def recall_at_k(ranked_doc_ids: List[int], true_id: int, k: int) -> float:
    return float(true_id in ranked_doc_ids[:k])

def eval_retrieval(
    queries: List[Tuple[str, int]],
    doc_embs: np.ndarray,
    village: Village,
    scout: ScoutRouter,
    panoramix: Panoramix,
    k: int = 10,
) -> float:
    hits = 0.0
    for q, true_id in queries:
        potion = panoramix.brew(q, village=village, scout=scout)
        scores = doc_embs @ potion.vector
        ranked = list(np.argsort(-scores))
        hits += recall_at_k(ranked, true_id, k)
    return hits / max(len(queries), 1)
