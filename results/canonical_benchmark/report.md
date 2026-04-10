# Canonical Benchmark Report

## Current Evidence State
- **Canonical artifact path:** `results/canonical_benchmark/`
- **State:** **Real artifact committed** (`summary.json`, `report.md`).
- **Interpretation boundary:** this run is valid benchmark evidence for this setup, but it does not close the open quality hypotheses.

## Setup
- Benchmark: `benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet`
- Split evaluated: `test`
- Available split counts: {'train': 18, 'validation': 6, 'test': 6}
- Specialists: general_qa, biomedical, finance
- LearnedGateFuser included: `False`

## Aggregate Metrics (mean with 95% bootstrap CI)

| Method | nDCG@10 | Recall@10 | MRR@10 | Latency (ms) | FLOPs proxy |
|---|---|---|---|---|---|
| MeanFuser | 0.4537 [0.3942, 0.5321] | 1.0000 [1.0000, 1.0000] | 0.2806 [0.2083, 0.3722] | 11.925 [11.170, 12.660] | 3.000 [3.000, 3.000] |
| KalmanorixFuser | 0.5247 [0.4050, 0.7269] | 1.0000 [1.0000, 1.0000] | 0.3750 [0.2222, 0.6389] | 25.812 [24.416, 27.136] | 3.000 [3.000, 3.000] |
| hard routing baseline | 0.6414 [0.4392, 0.8436] | 1.0000 [1.0000, 1.0000] | 0.5278 [0.2639, 0.8333] | 13.150 [12.454, 13.933] | 1.000 [1.000, 1.000] |
| all-routing + mean baseline | 0.4537 [0.3942, 0.5321] | 1.0000 [1.0000, 1.0000] | 0.2806 [0.2083, 0.3722] | 11.508 [10.905, 12.188] | 3.000 [3.000, 3.000] |

## Decision Framework: KalmanorixFuser vs MeanFuser

| Rule | Threshold | Observed | Pass |
|---|---:|---:|---|
| Primary metric (nDCG@10 Δ mean) | >= 0.0200 | 0.071012 | yes |
| Adjusted p-value (Holm) | <= 0.0500 | 1.000000 | no |
| Latency ratio (Kalman/Mean) | <= 1.500 | 2.165 | no |
| FLOPs ratio (Kalman/Mean) | <= 1.100 | 1.000 | yes |

## Paired Statistical Test: KalmanorixFuser vs MeanFuser

| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |
|---|---:|---|---:|---:|
| ndcg@10 | 0.071012 | [-0.063269, 0.252196] | 0.843750 | 1.000000 |
| recall@10 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| mrr@10 | 0.094444 | [-0.080556, 0.336111] | 0.843750 | 1.000000 |

## Verdict

- **kalman_vs_mean:** `inconclusive`
- Rule logic: `supported` if all checks pass; `unsupported` if nDCG@10 Δ <= 0 and Holm-adjusted p <= threshold; otherwise `inconclusive`.
- LearnedGateFuser omitted: LearnedGateFuser requires a two-specialist setup; current run uses 3 specialists
- This report is descriptive for the configured setup and should not be generalized beyond it.

## Demonstrated findings

- No demonstrated directional effect is established by the current statistical evidence.

## Unresolved findings

- Inconclusive result: kalman versus mean on ndcg@10 shows ambiguous evidence (Δ=0.071012, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@10 (Δ=0.000000, Holm-adjusted p=1.000000).
- Inconclusive result: kalman versus mean on mrr@10 shows ambiguous evidence (Δ=0.094444, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.

## Threats to validity

- Query sets may under-represent long-tail or adversarial cases.
- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.
- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.

## Benchmark limitations

- Evaluation is restricted to the selected benchmark split and may not cover broader deployment distributions.
- Latency and FLOPs values are proxy measurements and should not be treated as universal throughput guarantees.

## Recommended next experiments

- Increase held-out query count and rebalance domains before promoting unresolved findings.
- Add stress tests for distribution shift and low-resource domains to challenge demonstrated effects.
- Replicate on an independent benchmark slice with pre-registered metrics and hypotheses.
