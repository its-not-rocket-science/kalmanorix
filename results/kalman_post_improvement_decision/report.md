# Post-Improvement Kalman Decision Report

## Scope and evidence posture

This artifact consolidates **post-improvement** Kalman evidence from:
- canonical benchmark rerun under the latency-optimized path,
- latency microbenchmark,
- learned-correction experiment,
- prior-strength ablation,
- covariance-family ablation,
- uncertainty/calibration ablation,
- bucketed error analysis.

All claims below are benchmark-conditional. No claim is promoted to product-wide truth without statistical and cost checks.

## Demonstrated findings

- **Best quality gain achieved (point estimate):** `+0.125` MRR@10 for Kalman vs mean in the canonical post-optimization run; `+0.0949` NDCG@10 in the same run.
- **Calibration improvement is real in controlled ablations:** Kalman variants reduce ECE vs mean in prior ablation (`0.1070 -> 0.0869`, and best prior `0.0807`).
- **Latency optimization helped but did not close the gap:** optimized Kalman is `2.06x` faster than legacy in microbench, but remains `2.97x` slower than mean in microbench and `2.11x` slower in the canonical run.
- **FLOPs cost in canonical is neutral vs mean:** both report `flops_proxy = 3.0` (ratio `1.0x`).

## Unresolved findings

- **Statistical significance status is negative:** canonical paired testing reports adjusted p-value `1.0` for Kalman-vs-mean on NDCG@10 and MRR@10 (inconclusive despite positive deltas).
- **Quality wins are not robust across variants:** covariance ablation shows no clear quality gain from richer covariance families; prior and learned-correction gains are benchmark-local and not yet independently replicated with powered significance tests.
- **Calibration improvements have limited retrieval lift:** uncertainty ablation shows calibration movement, but retrieval metrics remain largely flat in that setup.

## Query buckets where Kalman helps most (descriptive, non-promotional)

From bucketed MRR@10 analysis, largest Kalman-minus-mean deltas were:
- **multi-domain:** `+0.2500`
- **ambiguous (proxy):** `+0.2500`
- **router confidence: low:** `+0.2000`
- **high uncertainty spread:** `+0.1667`

Important caveat: the same analysis reports **no bucket passed the consistency + minimum-size filter**, so these are hypotheses, not deployment-ready subgroup policies.

## Explicit decision rules

Promotion policy for Kalman as first-class must satisfy **all**:
1. Primary quality delta (NDCG@10) >= `+0.02` vs mean.
2. Adjusted p-value <= `0.05` on primary metric.
3. Latency ratio vs mean <= `1.5x`.
4. FLOPs ratio vs mean <= `1.1x`.
5. At least one practically relevant bucket shows consistent benefit after minimum-size filtering.

If rules (1) passes but (2) fails, or if cost gates fail, Kalman remains experimental.

## Rule check against current evidence

- Rule 1 (effect size): **PASS** (`+0.0949` NDCG@10).
- Rule 2 (significance): **FAIL** (adjusted p=`1.0`).
- Rule 3 (latency): **FAIL** (`2.11x` in canonical; `2.97x` in microbench).
- Rule 4 (FLOPs): **PASS** (`1.0x`).
- Rule 5 (consistent bucket wins): **FAIL** (none passed consistency/min-size filters).

## Final recommendation

## **Keep Kalman experimental.**

Brutally honest read: we have encouraging directional quality and calibration signals, but **not decision-grade evidence**. Statistical support is absent, latency is still materially over budget, and subgroup wins are exploratory only. Promoting Kalman to first-class now would overstate certainty.

## Exit criteria to revisit this decision

Re-open promotion only after all are met in a pre-registered rerun:
- powered query count with adjusted p <= 0.05 on primary metric,
- end-to-end latency ratio <= 1.5x,
- at least one stable, minimum-size bucket with replicated Kalman lift,
- no regression on calibration.
