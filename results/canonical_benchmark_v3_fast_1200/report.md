# Canonical Benchmark Report

## Setup
- Benchmark: `benchmarks\mixed_beir_v1.2.0\mixed_benchmark.parquet`
- Split evaluated: `test`
- Available split counts: {'train': 5078, 'test': 1193, 'validation': 634}
- **Benchmark status:** `claim_ready` — Counts and detectable-effect headroom satisfy stricter claim-readiness thresholds.
- Specialists: general_qa, biomedical, finance
- LearnedGateFuser included: `False`

## Aggregate Metrics (mean with 95% bootstrap CI)

| Method | nDCG@5 | nDCG@10 | MRR@5 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Top-1 success | Latency (ms) | FLOPs proxy |
|---|---|---|---|---|---|---|---|---|---|
| MeanFuser | 0.0709 [0.0605, 0.0814] | 0.0943 [0.0837, 0.1052] | 0.0878 [0.0746, 0.1016] | 0.1022 [0.0893, 0.1163] | 0.0123 [0.0072, 0.0179] | 0.0648 [0.0533, 0.0768] | 0.1362 [0.1201, 0.1533] | 0.0520 [0.0402, 0.0654] | 198.558 [196.527, 200.623] | 3.000 [3.000, 3.000] |
| KalmanorixFuser | 0.0711 [0.0607, 0.0819] | 0.0941 [0.0834, 0.1052] | 0.0883 [0.0749, 0.1020] | 0.1025 [0.0896, 0.1165] | 0.0123 [0.0072, 0.0180] | 0.0651 [0.0536, 0.0774] | 0.1358 [0.1196, 0.1529] | 0.0528 [0.0411, 0.0654] | 202.858 [200.644, 205.234] | 3.000 [3.000, 3.000] |
| hard routing baseline | 0.0795 [0.0682, 0.0913] | 0.0979 [0.0868, 0.1095] | 0.1028 [0.0877, 0.1182] | 0.1147 [0.1005, 0.1303] | 0.0157 [0.0101, 0.0217] | 0.0712 [0.0589, 0.0844] | 0.1299 [0.1139, 0.1464] | 0.0629 [0.0495, 0.0780] | 6.433 [6.300, 6.575] | 1.000 [1.000, 1.000] |
| all-routing + mean baseline | 0.0698 [0.0594, 0.0803] | 0.0940 [0.0835, 0.1050] | 0.0869 [0.0737, 0.1007] | 0.1018 [0.0891, 0.1157] | 0.0119 [0.0069, 0.0175] | 0.0635 [0.0522, 0.0755] | 0.1363 [0.1202, 0.1535] | 0.0520 [0.0402, 0.0654] | 2.043 [2.020, 2.069] | 3.000 [3.000, 3.000] |
| fixed weighted mean baseline | 0.0698 [0.0594, 0.0803] | 0.0940 [0.0835, 0.1050] | 0.0869 [0.0737, 0.1007] | 0.1018 [0.0891, 0.1157] | 0.0119 [0.0069, 0.0175] | 0.0635 [0.0522, 0.0755] | 0.1363 [0.1202, 0.1535] | 0.0520 [0.0402, 0.0654] | 2.004 [1.979, 2.029] | 3.000 [3.000, 3.000] |
| learned linear combiner | 0.0735 [0.0635, 0.0838] | 0.0934 [0.0828, 0.1040] | 0.0937 [0.0800, 0.1075] | 0.1058 [0.0926, 0.1195] | 0.0105 [0.0062, 0.0153] | 0.0711 [0.0592, 0.0840] | 0.1334 [0.1173, 0.1502] | 0.0511 [0.0394, 0.0637] | 1.942 [1.917, 1.967] | 2.000 [2.000, 2.000] |
| single generalist model | 0.0770 [0.0661, 0.0885] | 0.0952 [0.0842, 0.1068] | 0.0977 [0.0838, 0.1124] | 0.1090 [0.0954, 0.1234] | 0.0129 [0.0078, 0.0185] | 0.0684 [0.0565, 0.0807] | 0.1287 [0.1124, 0.1457] | 0.0570 [0.0444, 0.0704] | 2.108 [2.081, 2.136] | 1.000 [1.000, 1.000] |

## Method Ranking Snapshot

- Ranking by nDCG@10 (higher is better): `router_only_top1` (0.0979) > `best_single_specialist` (0.0956) > `single_generalist_model` (0.0952) > `mean` (0.0943) > `kalman` (0.0941) > `fixed_weighted_mean_fusion` (0.0940) > `uniform_mean_fusion` (0.0940) > `learned_linear_combiner` (0.0934) > `adaptive_route_or_fuse` (0.0917) > `router_only_topk_mean` (0.0916)

## Decision Framework: KalmanorixFuser vs MeanFuser

| Rule | Threshold | Observed | Pass |
|---|---:|---:|---|
| Primary metric (nDCG@10 Δ mean) | >= 0.0200 | -0.000203 | no |
| Adjusted p-value (Holm) | <= 0.0500 | 1.000000 | no |
| Latency ratio (Kalman/Mean) | <= 1.500 | 1.022 | yes |
| FLOPs ratio (Kalman/Mean) | <= 1.100 | 1.000 | yes |

## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)

- Number of evaluated test queries: **1193**
- Per-domain evaluated test counts: `{'argumentation': 241, 'biomedical': 129, 'encyclopedic': 126, 'fact_checking': 272, 'finance': 158, 'general_qa': 267}`
- Observed primary effect size (nDCG@10 Δ mean): `-0.000203`
- Detectable effect threshold estimate (80% power, α=0.05, paired-normal approximation): `0.001561`
- Target effect for decision rule: `0.020000`
- Sufficiently powered for target effect: `True`

## Sample Size Adequacy Checks

| Use case | Available | Minimum | Adequate | Notes |
|---|---:|---:|---|---|
| Uncertainty calibration (validation split) | 634 | 100 | yes | Validation split size governs stability of uncertainty calibration. |
| Paired significance testing (test split) | 1193 | 50 | yes | Test split paired query count governs inferential precision. |
| Per-domain analysis (min test queries in any domain) | 126 | 20 | yes | Lowest-count domain determines whether per-domain inference is stable. |

## Paired Statistical Test: KalmanorixFuser vs MeanFuser

| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |
|---|---:|---|---:|---:|
| ndcg@5 | 0.000204 | [-0.001197, 0.001629] | 0.721677 | 1.000000 |
| ndcg@10 | -0.000203 | [-0.001296, 0.000846] | 0.918294 | 1.000000 |
| mrr@5 | 0.000405 | [-0.000796, 0.001718] | 0.652838 | 1.000000 |
| mrr@10 | 0.000307 | [-0.000612, 0.001484] | 0.782467 | 1.000000 |
| recall@1 | 0.000017 | [0.000000, 0.000052] | 0.317311 | 1.000000 |
| recall@5 | 0.000349 | [-0.002746, 0.003746] | 0.952515 | 1.000000 |
| recall@10 | -0.000456 | [-0.003953, 0.003084] | 0.636440 | 1.000000 |
| top1_success | 0.000838 | [0.000000, 0.002515] | 0.317311 | 1.000000 |

## Kalman vs simple and learned weighting baselines

| Comparison | Δ nDCG@10 (Kalman-baseline) | 95% CI | Holm-adjusted p | Decision |
|---|---:|---|---:|---|
| kalman_vs_mean | -0.000203 | [-0.001296, 0.000846] | 1.000000 | inconclusive_sufficiently_powered |
| kalman_vs_fixed_weighted_mean_fusion | 0.000105 | [-0.001173, 0.001401] | 1.000000 | inconclusive_sufficiently_powered |
| kalman_vs_router_only_top1 | -0.003747 | [-0.014411, 0.007114] | 1.000000 | inconclusive_sufficiently_powered |
| kalman_vs_learned_linear_combiner | 0.000661 | [-0.007315, 0.008509] | 1.000000 | inconclusive_sufficiently_powered |

## Did Kalman beat the required deployment baselines?

| Decision | Verdict |
|---|---|
| kalman_vs_mean | inconclusive_sufficiently_powered |
| kalman_vs_weighted_mean | inconclusive_sufficiently_powered |
| kalman_vs_router_only_top1 | inconclusive_sufficiently_powered |

## Verdict

- **benchmark_status:** `claim_ready`
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
| high_specialist_disagreement | 597 | 0.1031 | 0.1027 | 0.1097 | 0.1021 | -0.0004 | -0.0070 | 0.0006 | inferential (adj p=1.0000) |
| low_specialist_disagreement | 596 | 0.0855 | 0.0855 | 0.0860 | 0.0810 | -0.0000 | -0.0005 | 0.0045 | inferential (adj p=1.0000) |
| high_uncertainty_spread | 597 | 0.0839 | 0.0838 | 0.0866 | 0.0809 | -0.0000 | -0.0028 | 0.0029 | inferential (adj p=1.0000) |
| low_uncertainty_spread | 596 | 0.1048 | 0.1044 | 0.1091 | 0.1023 | -0.0004 | -0.0047 | 0.0022 | inferential (adj p=1.0000) |
| single_domain_clear_winner | 247 | 0.0708 | 0.0710 | 0.0714 | 0.0789 | 0.0003 | -0.0004 | -0.0079 | inferential (adj p=1.0000) |
| true_multi_domain_queries | 480 | 0.1185 | 0.1174 | 0.1191 | 0.1107 | -0.0011 | -0.0018 | 0.0066 | inferential (adj p=1.0000) |
| router_high_confidence | 394 | 0.0835 | 0.0838 | 0.0825 | 0.0846 | 0.0004 | 0.0014 | -0.0008 | inferential (adj p=1.0000) |
| router_low_confidence | 394 | 0.1065 | 0.1060 | 0.1005 | 0.1012 | -0.0005 | 0.0054 | 0.0048 | inferential (adj p=1.0000) |

### Buckets with consistent Kalman gains
- None met the consistency + inferential significance criteria in this run.
- These subgroup findings are secondary and must not be promoted to headline claims without dedicated confirmatory evaluation.
- LearnedGateFuser omitted: LearnedGateFuser requires a two-specialist setup; current run uses 3 specialists
- This report is descriptive for the configured setup and should not be generalized beyond it.

## Demonstrated findings

- No demonstrated directional effect is established by the current statistical evidence.

## Unresolved findings

- Inconclusive result: kalman versus mean on ndcg@5 shows ambiguous evidence (Δ=0.000204, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on ndcg@10 shows ambiguous evidence (Δ=-0.000203, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on mrr@5 shows ambiguous evidence (Δ=0.000405, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on mrr@10 shows ambiguous evidence (Δ=0.000307, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on recall@1 shows ambiguous evidence (Δ=0.000017, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on recall@5 shows ambiguous evidence (Δ=0.000349, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on recall@10 shows ambiguous evidence (Δ=-0.000456, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on top1_success shows ambiguous evidence (Δ=0.000838, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.

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
