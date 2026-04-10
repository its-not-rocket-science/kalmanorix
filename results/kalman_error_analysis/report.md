# Kalman Bucketed Error Analysis

This report is query-bucketed and descriptive. It does **not** promote exploratory subgroups into global claims.

- Total queries analyzed: 12
- Metric used for per-query comparison: MRR@10
- Compared methods in every bucket: mean fusion, Kalman fusion, hard routing

## Bucket metrics

| Bucket | n | Mean fusion | Kalman fusion | Hard routing | Kalman-Mean | Kalman-Hard | Pattern |
|---|---:|---:|---:|---:|---:|---:|---|
| single-domain | 6 | 0.8333 | 0.8333 | 0.6667 | 0.0000 | 0.1667 | mixed |
| multi-domain | 6 | 0.6667 | 0.9167 | 0.7500 | 0.2500 | 0.1667 | mixed |
| high specialist agreement | 12 | 0.7500 | 0.8750 | 0.7083 | 0.1250 | 0.1667 | mixed |
| specialist disagreement | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | insufficient |
| high uncertainty spread | 6 | 0.7500 | 0.9167 | 0.7500 | 0.1667 | 0.1667 | mixed |
| low uncertainty spread | 6 | 0.7500 | 0.8333 | 0.6667 | 0.0833 | 0.1667 | mixed |
| router confidence: low | 5 | 0.7000 | 0.9000 | 0.8000 | 0.2000 | 0.1000 | mixed |
| router confidence: mid | 3 | 0.8333 | 0.8333 | 0.6667 | 0.0000 | 0.1667 | mixed |
| router confidence: high | 4 | 0.7500 | 0.8750 | 0.6250 | 0.1250 | 0.2500 | mixed |
| in-domain (proxy) | 4 | 0.7500 | 0.8750 | 0.6250 | 0.1250 | 0.2500 | mixed |
| ambiguous (proxy) | 6 | 0.6667 | 0.9167 | 0.7500 | 0.2500 | 0.1667 | mixed |

## Empirical patterns

### Buckets where Kalman consistently helps
- None met the consistency and minimum-size filter in this run.

### Buckets where Kalman hurts
- None met the consistency and minimum-size filter in this run.

### Buckets where Kalman appears redundant
- None met the redundancy and minimum-size filter in this run.

## Actionable hypotheses for next fusion revision

- If gains cluster in ambiguous/disagreement buckets, make Kalman conditional on disagreement and keep mean fusion for high-agreement cases.
- If regressions cluster in low-confidence buckets, add a router-confidence gate that falls back to hard routing or conservative mean.
- If low-uncertainty-spread buckets are redundant, skip covariance-heavy updates there to cut latency.
- Re-run with larger held-out query counts per bucket before elevating any subgroup into product-level policy.
