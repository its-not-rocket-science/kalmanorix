"""Fusion strategy creation and ranking helpers."""

from __future__ import annotations

import time

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, Village


def build_strategy(name: str, routing_mode: str) -> tuple[ScoutRouter, Panoramix]:
    """Create (scout, panoramix) tuple for strategy name."""
    if name == "mean":
        fuser = MeanFuser()
    elif name == "kalman":
        fuser = KalmanorixFuser()
    else:
        raise ValueError(f"Unsupported fusion strategy: {name}")
    return ScoutRouter(mode=routing_mode), Panoramix(fuser=fuser)


def rank_query(query_text: str, candidates: list[dict], village: Village, scout: ScoutRouter, pan: Panoramix) -> tuple[list[str], float]:
    """Rank candidate docs and return doc_ids + latency (ms)."""
    start = time.perf_counter()
    qv = pan.brew(query_text, village=village, scout=scout).vector
    scored = []
    for cand in candidates:
        doc_text = f"{cand.get('title', '')} {cand.get('text', '')}".strip()
        dv = pan.brew(doc_text, village=village, scout=scout).vector
        scored.append((cand["doc_id"], float(qv @ dv)))
    ranked = [doc_id for doc_id, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
    return ranked, (time.perf_counter() - start) * 1000.0
