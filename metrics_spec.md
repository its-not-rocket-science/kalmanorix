# Kalmanorix Preregistered Evaluation Specification

This document preregisters the **only valid metric definitions** for retrieval experiments in Kalmanorix.

- Version: `preregistered_eval_v1`
- Canonical implementation: `src/kalmanorix/benchmarks/evaluation_protocol.py`
- Protocol fingerprint: `sha256(PROTOCOL_SPEC_TEXT)` emitted in reports

## Scope and data contract

For each query `q`:

- Input ranking: ordered `doc_id` list, optionally accompanied by scores.
- Ground truth (`qrels[q]`): mapping `doc_id -> relevance gain` (float).
- Domain label (`query_domains[q]`): string used for per-domain aggregation.

Protocol rules:

1. Query universe is exactly `qrels.keys()`.
2. `query_domains` keys must match `qrels` keys exactly.
3. If ranking scores are provided, ranking is recomputed by sorting `(score desc, doc_id asc)`.
4. Duplicate `doc_id` entries in ranking are deduplicated by keeping first occurrence.
5. Missing ranking for a query is treated as empty ranking.
6. Relevant set for recall/MRR is `{d | qrels[q][d] > 0}`.

## Primary metrics (required)

Primary metrics are always computed for every query and aggregated by arithmetic mean and median.

### Recall@1, Recall@5, Recall@10

For `k in {1, 5, 10}`:

- `TopK_q`: first `k` document IDs in resolved ranking.
- `R_q`: relevant set for query `q`.
- Per-query metric:

`Recall@k(q) = |R_q ∩ TopK_q| / |R_q|`, with `Recall@k(q) = 0` when `|R_q| = 0`.

### MRR

- `rank_q`: 1-indexed rank of first relevant retrieved document.
- Per-query metric:

`MRR(q) = 1 / rank_q` if any relevant document is retrieved, else `0`.

### nDCG@10

Uses graded relevance gains from qrels.

- `DCG@10(q) = Σ_{i=1..10} (2^{rel_i} - 1) / log2(i + 1)`.
- `IDCG@10(q)` is DCG@10 computed on ideal descending gain order.
- `nDCG@10(q) = DCG@10(q) / IDCG@10(q)` when `IDCG@10(q) > 0`, else `0`.

## Secondary metrics (optional but fixed definitions)

Secondary metrics are included when a per-query value is provided by the caller:

- `latency_ms`: end-to-end per-query latency in milliseconds.
- `flops_proxy`: per-query FLOPs proxy value.
- `peak_memory_mb`: peak memory usage in MB attributable to the query.
- `specialist_count_selected`: number of specialists selected for the query.

Missing values are excluded from that metric’s aggregates.

## Aggregation

For every metric:

1. Compute per-query values.
2. Report global arithmetic mean and median across all eligible queries.
3. Report per-domain arithmetic mean and median using `query_domains[q]`.

## Immutability and enforcement

- Metric definitions are centrally declared in immutable mappings (`MappingProxyType`) in `evaluation_protocol.py`.
- Experiment adapters must call `evaluate_locked_protocol(...)` and must not redefine formulas.
- Protocol text and SHA-256 fingerprint are returned in every report for auditability.

## Legacy scripts

Legacy scripts that emit ad-hoc metrics are explicitly marked deprecated and should not be used for preregistered reporting.
