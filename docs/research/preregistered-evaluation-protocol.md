# Pre-Registered Evaluation Protocol (Locked Contract)

This document is the **pre-experiment contract** for benchmark evaluation. It must be finalized before running experiments and must not be changed during experiment execution.

## 1) Scope and Locking

- **Protocol ID:** `preregistered_eval_v1`
- **Reference implementation:** `kalmanorix.benchmarks.evaluation_protocol.evaluate_locked_protocol`
- **Contract fingerprint:** `sha256(PROTOCOL_SPEC_TEXT)` recorded by the evaluator as `protocol_sha256`
- All experiment outputs MUST log `protocol_version` and `protocol_sha256`.

## 2) Inputs and Query Universe

For every query `q` in `Q`:

- `qrels[q]`: judged relevance labels for documents (`rel(d, q) > 0` means relevant).
- `rankings[q]`: retrieval output as ordered docs or score-doc pairs.
- `domain(q)`: one domain label for per-domain aggregation.

The evaluated query set is **exactly** `Q = keys(qrels)`.

## 3) Primary Metrics

Primary metrics are computed per query and then aggregated:

- Recall@1
- Recall@5
- Recall@10
- MRR
- nDCG@10

### 3.1 Recall@k (k in {1,5,10})

Let:

- `R_q = {d | rel(d,q) > 0}`
- `TopK_q` = first `k` documents of the resolved ranking for query `q`

Formula:

`Recall@k(q) = |R_q ∩ TopK_q| / |R_q|` if `|R_q| > 0`, else `0`.

### 3.2 MRR

Let `rank_q` be rank (1-indexed) of first relevant document in the resolved ranking.

Formula:

`MRR(q) = 1 / rank_q` if a relevant document appears, else `0`.

### 3.3 nDCG@10

Let graded relevance at rank `i` be `rel_i(q)`.

`DCG@10(q) = Σ_{i=1..10} (2^{rel_i(q)} - 1) / log2(i + 1)`

`IDCG@10(q)` is computed from the ideal ordering of judged relevant documents for query `q`.

`nDCG@10(q) = DCG@10(q) / IDCG@10(q)` if `IDCG@10(q) > 0`, else `0`.

## 4) Secondary Metrics

Secondary metrics are query-level runtime/resource signals:

- `latency_ms(q)`: end-to-end retrieval latency per query, in milliseconds
- `flops(q)`: floating-point operations per query
- `memory_mb(q)`: peak memory per query, in megabytes

If missing for a query, the query is excluded from that specific secondary aggregate (no imputation).

## 5) Aggregation Rules

For every metric `m`:

- Global mean: `mean_{q in S_m} m(q)`
- Global median: `median_{q in S_m} m(q)`

Where `S_m` is:

- all queries `Q` for primary metrics,
- queries with available values for secondary metric `m`.

Per-domain breakdown:

- For each domain `d`, compute mean and median over `Q_d = {q in Q | domain(q)=d}` (or available subset for secondary metrics).

## 6) Deterministic Edge-Case Handling

1. **Ties in scores**: sort by score descending, then by `doc_id` ascending (lexical).
2. **Duplicate doc IDs in one ranking**: keep first occurrence only; later duplicates are dropped.
3. **Missing ranking for a query**: treat as empty ranking.
4. **No relevant docs (`|R_q|=0`)**: Recall@k = 0, MRR = 0, nDCG@10 = 0.
5. **nDCG denominator zero (`IDCG@10=0`)**: nDCG@10 = 0.
6. **Invalid score length**: if score list length != doc list length, raise `ValueError`.
7. **Mismatched query keys**: `keys(qrels)` must equal `keys(query_domains)` exactly; else raise `ValueError`.

## 7) Anti-p-hacking Clauses

- Metric definitions, formulas, and aggregation rules are fixed by this document and the locked evaluator implementation.
- Experiments must not alter metric code, ranking tie policy, or query-universe rules mid-study.
- Any protocol update requires a **new protocol ID/version** before new experiments begin.

## 8) Required Deliverables in Experiment Reports

- Protocol version and SHA fingerprint
- Global mean and median for each primary and secondary metric
- Per-domain mean and median for each primary and secondary metric
- Count of evaluated queries (`num_queries`)
