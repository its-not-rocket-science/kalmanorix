"""Pre-registered and locked benchmark evaluation protocol.

This module defines retrieval and efficiency metrics exactly once and exposes
`evaluate_locked_protocol`, a single entry point intended to be called by all
experiments. The protocol metadata includes a deterministic fingerprint that can
be logged with experiment outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from statistics import median
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

PROTOCOL_VERSION = "preregistered_eval_v1"
PRIMARY_K_VALUES: tuple[int, ...] = (1, 5, 10)
NDCG_K = 10

# This text is part of the contractual protocol and should be logged.
PROTOCOL_SPEC_TEXT = (
    "Primary metrics: Recall@1, Recall@5, Recall@10, MRR, nDCG@10. "
    "Secondary metrics: latency_ms, flops, memory_mb. "
    "Per-query Recall@k = |R_q intersect TopK_q| / |R_q| with Recall@k=0 when |R_q|=0. "
    "Per-query MRR = 1/rank_q where rank_q is first rank of any relevant result, else 0. "
    "Per-query DCG@10 = sum_{i=1..10} ((2^{rel_i}-1)/log2(i+1)); "
    "IDCG@10 uses ideal ordering of judged docs; nDCG@10 = DCG@10/IDCG@10 when IDCG>0 else 0. "
    "Aggregation: arithmetic mean and median over queries, plus per-domain mean/median. "
    "Tie-breaking: descending score, then ascending doc_id lexical order. "
    "Duplicate doc_ids in a ranking: only first occurrence kept. "
    "Missing ranking for a query is treated as empty ranking."
)
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_SPEC_TEXT.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryRanking:
    """Single-query retrieval output.

    Attributes:
        doc_ids: Candidate document ids.
        scores: Optional scores aligned with ``doc_ids``. If provided, ranking
            order is recomputed via protocol tie rules.
    """

    doc_ids: tuple[str, ...]
    scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AggregateStats:
    """Mean and median aggregates for one metric."""

    mean: float
    median: float


@dataclass(frozen=True)
class EvaluationReport:
    """Full protocol output with fingerprint and aggregates."""

    protocol_version: str
    protocol_sha256: str
    global_primary: Mapping[str, AggregateStats]
    global_secondary: Mapping[str, AggregateStats]
    per_domain_primary: Mapping[str, Mapping[str, AggregateStats]]
    per_domain_secondary: Mapping[str, Mapping[str, AggregateStats]]
    num_queries: int


def _deduplicate(doc_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            output.append(doc_id)
    return output


def _resolve_ranking(ranking: QueryRanking) -> list[str]:
    if ranking.scores is None:
        return _deduplicate(list(ranking.doc_ids))
    if len(ranking.doc_ids) != len(ranking.scores):
        raise ValueError("scores must have the same length as doc_ids")

    paired = sorted(
        zip(ranking.doc_ids, ranking.scores, strict=True),
        key=lambda pair: (-float(pair[1]), pair[0]),
    )
    return _deduplicate([doc_id for doc_id, _ in paired])


def _recall_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return float(len(top_k.intersection(relevant)) / len(relevant))


def _mrr(ranked_doc_ids: Sequence[str], relevant: set[str]) -> float:
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return float(1.0 / idx)
    return 0.0


def _ndcg_at_k(ranked_doc_ids: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = float(gains.get(doc_id, 0.0))
        if rel > 0:
            dcg += (2.0**rel - 1.0) / np.log2(idx + 1.0)

    ideal_rels = sorted((float(v) for v in gains.values() if float(v) > 0), reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal_rels[:k], start=1):
        idcg += (2.0**rel - 1.0) / np.log2(idx + 1.0)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _aggregate(values: Sequence[float]) -> AggregateStats:
    if not values:
        return AggregateStats(mean=0.0, median=0.0)
    return AggregateStats(mean=float(np.mean(values)), median=float(median(values)))


def evaluate_locked_protocol(
    *,
    rankings: Mapping[str, QueryRanking],
    qrels: Mapping[str, Mapping[str, float]],
    query_domains: Mapping[str, str],
    latency_ms: Mapping[str, float] | None = None,
    flops: Mapping[str, float] | None = None,
    memory_mb: Mapping[str, float] | None = None,
) -> EvaluationReport:
    """Evaluate predictions with a pre-registered immutable protocol.

    Contract notes:
    - The query universe is exactly ``qrels.keys()``.
    - Missing ranking for a query is interpreted as an empty ranking.
    - Missing secondary metric values are excluded from that metric's aggregate.
    """

    if set(qrels) != set(query_domains):
        raise ValueError("query_domains keys must match qrels keys exactly")

    primary: dict[str, list[float]] = {
        "recall@1": [],
        "recall@5": [],
        "recall@10": [],
        "mrr": [],
        "ndcg@10": [],
    }
    secondary: dict[str, list[float]] = {
        "latency_ms": [],
        "flops": [],
        "memory_mb": [],
    }

    per_domain_primary_raw: dict[str, dict[str, list[float]]] = {}
    per_domain_secondary_raw: dict[str, dict[str, list[float]]] = {}

    for query_id in sorted(qrels):
        gains = {doc_id: float(rel) for doc_id, rel in qrels[query_id].items() if float(rel) > 0.0}
        relevant = set(gains)

        resolved = _resolve_ranking(rankings.get(query_id, QueryRanking(doc_ids=tuple())))

        r1 = _recall_at_k(resolved, relevant, 1)
        r5 = _recall_at_k(resolved, relevant, 5)
        r10 = _recall_at_k(resolved, relevant, 10)
        mrr = _mrr(resolved, relevant)
        ndcg = _ndcg_at_k(resolved, gains, NDCG_K)

        domain = query_domains[query_id]
        per_domain_primary_raw.setdefault(
            domain,
            {"recall@1": [], "recall@5": [], "recall@10": [], "mrr": [], "ndcg@10": []},
        )

        primary["recall@1"].append(r1)
        primary["recall@5"].append(r5)
        primary["recall@10"].append(r10)
        primary["mrr"].append(mrr)
        primary["ndcg@10"].append(ndcg)

        per_domain_primary_raw[domain]["recall@1"].append(r1)
        per_domain_primary_raw[domain]["recall@5"].append(r5)
        per_domain_primary_raw[domain]["recall@10"].append(r10)
        per_domain_primary_raw[domain]["mrr"].append(mrr)
        per_domain_primary_raw[domain]["ndcg@10"].append(ndcg)

        per_domain_secondary_raw.setdefault(domain, {"latency_ms": [], "flops": [], "memory_mb": []})
        if latency_ms is not None and query_id in latency_ms:
            value = float(latency_ms[query_id])
            secondary["latency_ms"].append(value)
            per_domain_secondary_raw[domain]["latency_ms"].append(value)
        if flops is not None and query_id in flops:
            value = float(flops[query_id])
            secondary["flops"].append(value)
            per_domain_secondary_raw[domain]["flops"].append(value)
        if memory_mb is not None and query_id in memory_mb:
            value = float(memory_mb[query_id])
            secondary["memory_mb"].append(value)
            per_domain_secondary_raw[domain]["memory_mb"].append(value)

    global_primary = MappingProxyType({name: _aggregate(vals) for name, vals in primary.items()})
    global_secondary = MappingProxyType({name: _aggregate(vals) for name, vals in secondary.items()})

    per_domain_primary = MappingProxyType(
        {
            domain: MappingProxyType({name: _aggregate(vals) for name, vals in metrics.items()})
            for domain, metrics in per_domain_primary_raw.items()
        }
    )
    per_domain_secondary = MappingProxyType(
        {
            domain: MappingProxyType({name: _aggregate(vals) for name, vals in metrics.items()})
            for domain, metrics in per_domain_secondary_raw.items()
        }
    )

    return EvaluationReport(
        protocol_version=PROTOCOL_VERSION,
        protocol_sha256=PROTOCOL_SHA256,
        global_primary=global_primary,
        global_secondary=global_secondary,
        per_domain_primary=per_domain_primary,
        per_domain_secondary=per_domain_secondary,
        num_queries=len(qrels),
    )
