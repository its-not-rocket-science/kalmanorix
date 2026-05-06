# Canonical Benchmark Report

> ⚠️ **Do not interpret this artifact as proof of Kalman-vs-mean superiority.**

## Setup
- Benchmark: `benchmarks\mixed_beir_v1.2.0\mixed_benchmark.parquet`
- Split evaluated: `test`
- Available split counts: {'train': 5078, 'test': 1193, 'validation': 634}
- **Benchmark status:** `toy` — Sample is toy-scale for Kalman-vs-mean claims; treat outcomes as smoke-test signals only.
- Specialists: general_qa, biomedical, finance
- LearnedGateFuser included: `False`

## Aggregate Metrics (mean with 95% bootstrap CI)

| Method | nDCG@5 | nDCG@10 | MRR@5 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Top-1 success | Latency (ms) | FLOPs proxy |
|---|---|---|---|---|---|---|---|---|---|
| MeanFuser | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 138.521 [121.350, 156.532] | 3.000 [3.000, 3.000] |
| KalmanorixFuser | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 117.966 [112.409, 124.236] | 3.000 [3.000, 3.000] |
| hard routing baseline | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 10.400 [8.032, 13.518] | 1.000 [1.000, 1.000] |
| all-routing + mean baseline | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.124 [1.001, 1.269] | 3.000 [3.000, 3.000] |
| fixed weighted mean baseline | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.303 [1.046, 1.610] | 3.000 [3.000, 3.000] |
| learned linear combiner | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.210 [1.000, 1.467] | 3.000 [3.000, 3.000] |
| single generalist model | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.449 [1.160, 1.749] | 1.000 [1.000, 1.000] |

## Method Ranking Snapshot

- Ranking by nDCG@10 (higher is better): `adaptive_route_or_fuse` (0.0000) > `best_single_specialist` (0.0000) > `fixed_weighted_mean_fusion` (0.0000) > `kalman` (0.0000) > `learned_linear_combiner` (0.0000) > `mean` (0.0000) > `router_only_top1` (0.0000) > `router_only_topk_mean` (0.0000) > `single_generalist_model` (0.0000) > `uniform_mean_fusion` (0.0000)

## Decision Framework: KalmanorixFuser vs MeanFuser

| Rule | Threshold | Observed | Pass |
|---|---:|---:|---|
| Primary metric (nDCG@10 Δ mean) | >= 0.0200 | 0.000000 | no |
| Adjusted p-value (Holm) | <= 0.0500 | 1.000000 | no |
| Latency ratio (Kalman/Mean) | <= 1.500 | 0.852 | yes |
| FLOPs ratio (Kalman/Mean) | <= 1.100 | 1.000 | yes |

## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)

- Number of evaluated test queries: **10**
- Per-domain evaluated test counts: `{'argumentation': 10}`
- Observed primary effect size (nDCG@10 Δ mean): `0.000000`
- Detectable effect threshold estimate (80% power, α=0.05, paired-normal approximation): `0.000000`
- Target effect for decision rule: `0.020000`
- Sufficiently powered for target effect: `True`

## Sample Size Adequacy Checks

| Use case | Available | Minimum | Adequate | Notes |
|---|---:|---:|---|---|
| Uncertainty calibration (validation split) | 634 | 100 | yes | Validation split size governs stability of uncertainty calibration. |
| Paired significance testing (test split) | 10 | 50 | no | Test split paired query count governs inferential precision. |
| Per-domain analysis (min test queries in any domain) | 10 | 20 | no | Lowest-count domain determines whether per-domain inference is stable. |
- ⚠️ WARNING: test query count is too small for reliable paired significance claims.
- ⚠️ WARNING: per-domain test coverage is too thin for domain-level inference.
- ⚠️ benchmark_status guardrail failures:
  - test_query_count=10 < 25

## Paired Statistical Test: KalmanorixFuser vs MeanFuser

| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |
|---|---:|---|---:|---:|
| ndcg@5 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| ndcg@10 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| mrr@5 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| mrr@10 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| recall@1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| recall@5 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| recall@10 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| top1_success | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |

## Kalman vs simple and learned weighting baselines

| Comparison | Δ nDCG@10 (Kalman-baseline) | 95% CI | Holm-adjusted p | Decision |
|---|---:|---|---:|---|
| kalman_vs_mean | 0.000000 | [0.000000, 0.000000] | 1.000000 | inconclusive_sufficiently_powered |
| kalman_vs_fixed_weighted_mean_fusion | 0.000000 | [0.000000, 0.000000] | 1.000000 | inconclusive_sufficiently_powered |
| kalman_vs_router_only_top1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | inconclusive_sufficiently_powered |
| kalman_vs_learned_linear_combiner | 0.000000 | [0.000000, 0.000000] | 1.000000 | inconclusive_sufficiently_powered |

## Did Kalman beat the required deployment baselines?

| Decision | Verdict |
|---|---|
| kalman_vs_mean | inconclusive_sufficiently_powered |
| kalman_vs_weighted_mean | inconclusive_sufficiently_powered |
| kalman_vs_router_only_top1 | inconclusive_sufficiently_powered |

## Verdict

- **benchmark_status:** `toy`
- **kalman_vs_mean:** `inconclusive_sufficiently_powered`
- **kalman_vs_weighted_mean:** `inconclusive_sufficiently_powered`
- **kalman_vs_router_only_top1:** `inconclusive_sufficiently_powered`
- **kalman_vs_learned_linear_combiner:** `inconclusive_sufficiently_powered`
- Interpretation: `benchmark_status` grades evidence readiness (`toy`, `underpowered`, `minimally_powered`, `claim_ready`) while verdict preserves the existing Kalman-vs-baseline decision rule.
- Rule logic: `supported` if all checks pass; `unsupported` if nDCG@10 Δ <= 0 and Holm-adjusted p <= threshold; otherwise inconclusive is split into `inconclusive_underpowered` vs `inconclusive_sufficiently_powered` from the detectable-effect threshold estimate.

## Hard claim gate

- Question: **May we honestly claim 'Kalman fusion beats mean fusion'?**
- Hard rule: Allowed only if benchmark_status == claim_ready, confirmatory slice verdict == supported, and all required baseline decisions (mean, weighted_mean, router_only_top1) are supported.
- Decision: **`blocked`**

## Confirmatory Slice (Pre-declared)

- Slice name: `preregistered_high_disagreement_high_uncertainty_multi_domain_low_router_confidence`
- Slice definition: `specialist_disagreement >= median AND uncertainty_spread >= median AND router_confidence <= 33rd percentile AND is_multi_domain == True`
- Why this slice is pre-declared: Pre-declared confirmatory slice exactly follows the preregistered contract: high specialist disagreement, high uncertainty spread, low router confidence, and multi-domain queries only.
- Query count: `0`
- Minimum paired queries for inferential claims: `20`
- Warnings:
  - Confirmatory slice contains zero paired queries; inferential testing is skipped.
- ⚠️ WARNING: confirmatory slice is underpowered; do not use this slice for inferential superiority claims.

| Method | nDCG@10 |
|---|---:|
| mean | 0.0000 |
| kalman | 0.0000 |
| fixed_weighted_mean_fusion | 0.0000 |
| router_only_top1 | 0.0000 |

### Confirmatory verdicts

- **overall_confirmatory_slice_verdict:** `inconclusive_underpowered`
- **kalman_vs_mean:** `underpowered`
- **kalman_vs_weighted_mean:** `underpowered`
- **kalman_vs_router_only_top1:** `underpowered`
- Hard rule for the claim “Kalman fusion beats mean” in the confirmatory slice: require all three pairwise comparisons to pass all of these checks on nDCG@10: positive delta, Holm-adjusted p <= threshold, CI excludes 0, and practical effect-size floor met.
- Therefore the confirmatory slice is `supported` only when all required pairwise comparisons pass; otherwise it is `unsupported` (or `inconclusive_*` if underpowered or unresolved).
- Underpowered: inferential paired testing was not run for this pre-declared confirmatory slice because paired query count is below the minimum threshold.

## Hostile Falsification Slice (Credibility Layer; does not replace canonical benchmark)

- Slice name: `hostile_disagreement_calibration_routing_ambiguity`
- Slice definition: `high_specialist_disagreement AND low_router_confidence AND high_uncertainty_spread`
- Why this is hostile: Adversarial slice intersects high disagreement, low router confidence, and high uncertainty spread to stress disagreement, routing ambiguity, and calibration error proxies together.
- Query count: `0`
- Minimum paired queries for inferential claims: `20`
- Warnings:
  - Hostile slice contains zero paired queries; inferential testing is skipped.

| Method | nDCG@10 |
|---|---:|
| kalman | 0.0000 |
| mean | 0.0000 |
| fixed_weighted_mean_fusion | 0.0000 |
| router_only_top1 | 0.0000 |
| learned_linear_combiner | 0.0000 |

### Where Kalman wins

- None.

### Where Kalman loses

- `kalman_vs_mean (baseline=mean)` → `underpowered`
- `kalman_vs_weighted_mean (baseline=fixed_weighted_mean_fusion)` → `underpowered`
- `kalman_vs_router_only_top1 (baseline=router_only_top1)` → `underpowered`
- `kalman_vs_learned_linear_combiner (baseline=learned_linear_combiner)` → `underpowered`

### Confirmatory-slice survival under hostile conditions

- Confirmatory slice verdict: `inconclusive_underpowered`.
- Hostile slice verdict: `inconclusive_underpowered`.
- Survival criterion for this credibility layer: confirmatory slice should remain `supported` and hostile slice should not degrade to `unsupported` for required available baselines.

## Bucketed Analysis (Exploratory unless significance criteria are met)

| Bucket | n | Mean | Kalman | Hard routing | Top-k mean | Δ(K-M) nDCG@10 | Δ(K-Hard) | Δ(K-TopK) | Significance status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| high_specialist_disagreement | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| low_specialist_disagreement | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| high_uncertainty_spread | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| low_uncertainty_spread | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| single_domain_clear_winner | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| true_multi_domain_queries | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| router_high_confidence | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |
| router_low_confidence | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory_only |

### Buckets with consistent Kalman gains
- None met the consistency + inferential significance criteria in this run.
- These subgroup findings are secondary and must not be promoted to headline claims without dedicated confirmatory evaluation.
- LearnedGateFuser omitted: LearnedGateFuser requires a two-specialist setup; current run uses 3 specialists
- This report is descriptive for the configured setup and should not be generalized beyond it.

## Demonstrated findings

- No demonstrated directional effect is established by the current statistical evidence.

## Unresolved findings

- Null result: no statistically reliable difference is demonstrated for kalman versus mean on ndcg@5 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on ndcg@10 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on mrr@5 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on mrr@10 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@1 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@5 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@10 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on top1_success (Δ=0.000000, Holm-adjusted p=1.000000).

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
