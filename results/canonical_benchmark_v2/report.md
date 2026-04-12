# Canonical Benchmark Report

## Setup
- Benchmark: `benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet`
- Split evaluated: `test`
- Available split counts: {'train': 18, 'validation': 6, 'test': 6}
- Specialists: general_qa, biomedical, finance
- LearnedGateFuser included: `False`

## Aggregate Metrics (mean with 95% bootstrap CI)

| Method | nDCG@5 | nDCG@10 | MRR@5 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Top-1 success | Latency (ms) | FLOPs proxy |
|---|---|---|---|---|---|---|---|---|---|
| MeanFuser | 0.4537 [0.3942, 0.5321] | 0.4537 [0.3942, 0.5321] | 0.2806 [0.2083, 0.3722] | 0.2806 [0.2083, 0.3806] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 19.741 [18.533, 20.962] | 3.000 [3.000, 3.000] |
| KalmanorixFuser | 0.5486 [0.4057, 0.7341] | 0.5486 [0.4057, 0.7341] | 0.4056 [0.2222, 0.6668] | 0.4056 [0.2222, 0.6504] | 0.1667 [0.0000, 0.5000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.1667 [0.0000, 0.5000] | 40.575 [38.593, 42.649] | 3.000 [3.000, 3.000] |
| hard routing baseline | 0.5820 [0.3102, 0.8436] | 0.6414 [0.4392, 0.8436] | 0.5000 [0.2083, 0.8333] | 0.5278 [0.2639, 0.7917] | 0.3333 [0.0000, 0.6667] | 0.8333 [0.5000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.3333 [0.0000, 0.6667] | 22.214 [20.645, 23.783] | 1.000 [1.000, 1.000] |
| all-routing + mean baseline | 0.4537 [0.3942, 0.5321] | 0.4537 [0.3942, 0.5321] | 0.2806 [0.2083, 0.3722] | 0.2806 [0.2083, 0.3806] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 19.139 [17.862, 20.404] | 3.000 [3.000, 3.000] |

## Method Ranking Snapshot

- Ranking by nDCG@10 (higher is better): `learned_linear_combiner` (0.6748) > `single_generalist_model` (0.6748) > `router_only_top1` (0.6414) > `kalman` (0.5486) > `best_single_specialist` (0.5183) > `adaptive_route_or_fuse` (0.4799) > `router_only_topk_mean` (0.4799) > `fixed_weighted_mean_fusion` (0.4537) > `mean` (0.4537) > `uniform_mean_fusion` (0.4537)

## Decision Framework: KalmanorixFuser vs MeanFuser

| Rule | Threshold | Observed | Pass |
|---|---:|---:|---|
| Primary metric (nDCG@10 Δ mean) | >= 0.0200 | 0.094887 | yes |
| Adjusted p-value (Holm) | <= 0.0500 | 1.000000 | no |
| Latency ratio (Kalman/Mean) | <= 1.500 | 2.055 | no |
| FLOPs ratio (Kalman/Mean) | <= 1.100 | 1.000 | yes |

## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)

- Number of evaluated test queries: **6**
- Per-domain evaluated test counts: `{'biomedical': 2, 'finance': 2, 'general_qa': 2}`
- Observed primary effect size (nDCG@10 Δ mean): `0.094887`
- Detectable effect threshold estimate (80% power, α=0.05, paired-normal approximation): `0.265684`
- Target effect for decision rule: `0.020000`
- Sufficiently powered for target effect: `False`

## Sample Size Adequacy Checks

| Use case | Available | Minimum | Adequate | Notes |
|---|---:|---:|---|---|
| Uncertainty calibration (validation split) | 6 | 100 | no | Validation split size governs stability of uncertainty calibration. |
| Paired significance testing (test split) | 6 | 50 | no | Test split paired query count governs inferential precision. |
| Per-domain analysis (min test queries in any domain) | 2 | 20 | no | Lowest-count domain determines whether per-domain inference is stable. |

## Paired Statistical Test: KalmanorixFuser vs MeanFuser

| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |
|---|---:|---|---:|---:|
| ndcg@5 | 0.094887 | [0.000000, 0.284662] | 1.000000 | 1.000000 |
| ndcg@10 | 0.094887 | [0.000000, 0.284662] | 1.000000 | 1.000000 |
| mrr@5 | 0.125000 | [0.000000, 0.375000] | 1.000000 | 1.000000 |
| mrr@10 | 0.125000 | [0.000000, 0.375000] | 1.000000 | 1.000000 |
| recall@1 | 0.166667 | [0.000000, 0.500000] | 1.000000 | 1.000000 |
| recall@5 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| recall@10 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| top1_success | 0.166667 | [0.000000, 0.500000] | 1.000000 | 1.000000 |

## Verdict

- **kalman_vs_mean:** `inconclusive_underpowered`
- Rule logic: `supported` if all checks pass; `unsupported` if nDCG@10 Δ <= 0 and Holm-adjusted p <= threshold; otherwise inconclusive is split into `inconclusive_underpowered` vs `inconclusive_sufficiently_powered` from the detectable-effect threshold estimate.

## Bucketed Analysis (Exploratory unless significance criteria are met)

| Bucket | n | Mean | Kalman | Hard routing | Top-k mean | Δ(K-M) nDCG@10 | Δ(K-Hard) | Δ(K-TopK) | Significance status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| high_specialist_disagreement | 3 | 0.4015 | 0.5912 | 0.6872 | 0.4828 | 0.1898 | -0.0960 | 0.1084 | exploratory_only |
| low_specialist_disagreement | 3 | 0.5059 | 0.5059 | 0.5956 | 0.4769 | 0.0000 | -0.0897 | 0.0290 | exploratory_only |
| high_uncertainty_spread | 3 | 0.4682 | 0.4682 | 0.6872 | 0.4538 | 0.0000 | -0.2190 | 0.0144 | exploratory_only |
| low_uncertainty_spread | 3 | 0.4392 | 0.6290 | 0.5956 | 0.5059 | 0.1898 | 0.0333 | 0.1230 | exploratory_only |
| single_domain_clear_winner | 2 | 0.4434 | 0.4434 | 0.3934 | 0.4653 | 0.0000 | 0.0500 | -0.0219 | exploratory_only |
| true_multi_domain_queries | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| router_high_confidence | 2 | 0.4434 | 0.4434 | 0.3934 | 0.4653 | 0.0000 | 0.0500 | -0.0219 | exploratory_only |
| router_low_confidence | 2 | 0.3869 | 0.3869 | 0.5308 | 0.4088 | 0.0000 | -0.1440 | -0.0219 | exploratory_only |

### Buckets with consistent Kalman gains
- None met the consistency + inferential significance criteria in this run.
- These subgroup findings are secondary and must not be promoted to headline claims without dedicated confirmatory evaluation.
- LearnedGateFuser omitted: LearnedGateFuser requires a two-specialist setup; current run uses 3 specialists
- This report is descriptive for the configured setup and should not be generalized beyond it.

## Demonstrated findings

- No demonstrated directional effect is established by the current statistical evidence.

## Unresolved findings

- Inconclusive result: kalman versus mean on ndcg@5 shows ambiguous evidence (Δ=0.094887, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on ndcg@10 shows ambiguous evidence (Δ=0.094887, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on mrr@5 shows ambiguous evidence (Δ=0.125000, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on mrr@10 shows ambiguous evidence (Δ=0.125000, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on recall@1 shows ambiguous evidence (Δ=0.166667, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@5 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@10 (Δ=0.000000, Holm-adjusted p=1.000000).
- Inconclusive result: kalman versus mean on top1_success shows ambiguous evidence (Δ=0.166667, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.

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


## v1 vs v2 Provenance Comparison

This v2 artifact **does not modify** `results/canonical_benchmark/`; v1 remains untouched for provenance.

### Shared-metric snapshot (v2 − v1)

| Method | Δ nDCG@10 | Δ MRR@10 | Δ Recall@10 | Δ Latency (ms) | Δ FLOPs proxy |
|---|---:|---:|---:|---:|---:|
| MeanFuser | +0.000000 | +0.000000 | +0.000000 | +7.816 | +0.000 |
| KalmanorixFuser | +0.023875 | +0.030556 | +0.000000 | +14.763 | +0.000 |
| hard routing baseline | +0.000000 | +0.000000 | +0.000000 | +9.064 | +0.000 |
| all-routing + mean baseline | +0.000000 | +0.000000 | +0.000000 | +7.630 | +0.000 |

### Metric-surface expansion in v2

- Added and reported in this artifact: **nDCG@5, MRR@5, Recall@1, top1_success**.
- These metrics are absent from v1 summary and therefore are not part of the v1 delta table.

### Kalman support status in v2

- **Kalman is supported in this v2 artifact.** `KalmanorixFuser` is present in aggregate metrics and in paired Kalman-vs-Mean tests.
