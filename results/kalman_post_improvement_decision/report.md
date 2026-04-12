# Kalman Evidence-Upgrade Report (Stronger Benchmark Regime)

## Scope and benchmark versions used

This guarded report aggregates only the requested evidence tracks:
1. uncertainty calibration,
2. uncertainty ablation,
3. covariance ablation,
4. correlation-aware fusion,
5. canonical benchmark rerun,
6. per-bucket analysis.

### Explicit benchmark/version provenance

- **Uncertainty calibration:** internal calibration split artifact (`results/uncertainty_calibration/summary.json`), status `sufficient`, with validation/train/test split sizes 10/1/4.
- **Uncertainty ablation:** `toy_mixed` + `synthetic_shifted_queries` ablation artifact (`results/uncertainty_ablation/summary.json`).
- **Covariance ablation:** `kalman_covariance_ablation_v2_enlarged` (`results/kalman_covariance_ablation_v2/report.md`).
- **Correlation-aware fusion:** strengthened synthetic correlated-expert split with `n_test=420`, 50/50 high/low-correlation buckets (`results/correlation_aware_fusion/summary.json`).
- **Canonical benchmark baseline:** `benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet` (`results/canonical_benchmark/report.md`).
- **Canonical benchmark rerun artifact actually containing metrics:** latency-optimized canonical rerun on the **same** `mixed_beir_v1.0.0` test split (`results/kalman_latency_optimization/canonical/summary.json`).
- **Canonical v1.1.0/v1.2.0 stronger benchmark folders:** present as README-only placeholders without `summary.json` metrics (`results/canonical_benchmark_v2/README.md`, `results/canonical_benchmark_v3/README.md`).
- **Per-bucket analysis:** query-bucketed report over 12 queries (`results/kalman_error_analysis/report.md`).

## Rule-based evidence checks

Decision rules for this report (strict):

1. **Technical upgrade rule:** PASS if a method change improves at least one target metric in its own benchmark and does not show immediate contradiction in that same artifact.
2. **Empirical upgrade rule:** PASS only if canonical rerun evidence is both (a) positive on primary ranking metric and (b) statistically supported (adjusted p <= 0.05).
3. **Null-preservation rule:** always record null/negative findings even when directional gains exist.
4. **Centrality rule:**
   - **central**: empirical upgrade PASS + cost gates pass + stable bucket evidence,
   - **experimental**: directional gains but no statistical support and/or cost/bucket gates fail,
   - **selective-only**: no global support, but at least one stable bucket passes minimum-size + consistency filters.

## what improved technically

- **Uncertainty calibration pipeline quality controls improved:** validation power checks are now explicitly satisfied and leakage checks are clean; calibration candidates are selected from validation-only evidence. **Rule 1: PASS (technical process).**
- **Correlation-aware Kalman variant improved over baseline Kalman on the strengthened synthetic split:** best variant improved MRR@10 by `+0.0037` versus baseline Kalman. **Rule 1: PASS (small technical gain in synthetic stress setup).**
- **Covariance-family expansion did not provide material additional gain over scalar Kalman:** structured/diagonal variants are near-tied with scalar Kalman, while latency is worse. **Rule 1: FAIL for “richer covariance is technically better.”**

## what improved empirically

- **Canonical rerun (available metric artifact) shows positive direction:** Kalman vs mean improved by `+0.0949` (nDCG@10) and `+0.1250` (MRR@10) on `mixed_beir_v1.0.0`.
- **But empirical-upgrade gate fails:** adjusted p-values remain `1.0` for both nDCG@10 and MRR@10, so this is not decision-grade evidence.
- **Therefore:** empirical evidence quality improved only in *directional signal*, not in *statistical confidence*.

## what remained null

- **Uncertainty calibration did not change benchmark delta:** pre/post Kalman-minus-mean MRR delta remains `0.0`.
- **Uncertainty ablation shows calibration movement without retrieval lift:** on both datasets, recall@1 and MRR@10 are largely flat across sigma² methods.
- **Canonical significance remains null/inconclusive:** Kalman-vs-mean adjusted p-values are still `1.0`.
- **Bucket consistency remains null:** no bucket passed consistency + minimum-size filters.
- **Covariance richness remains practically null for quality:** richer covariance does not clear practical gains over scalar Kalman while adding latency.

## where Kalman helps, if anywhere

- **Global (current evidence):** directional gain over mean appears in canonical v1.0.0 artifacts, but remains statistically unsupported.
- **Synthetic correlated-expert regime:** Kalman helps over mean; correlation-aware inflation gives a small extra uplift over baseline Kalman.
- **Exploratory buckets (non-promotional):** larger deltas appear in multi-domain, ambiguous-proxy, low router-confidence, and high uncertainty-spread buckets, but none are stable enough for policy.

## whether Kalman should remain central, experimental, or selective-only

### Verdict: **experimental**

Rule outcomes:
- Technical upgrade rule: **mixed PASS** (better calibration controls; small synthetic correlation-aware gain).
- Empirical upgrade rule: **FAIL** (directional improvement without statistical support).
- Cost/stability support for centrality: **FAIL** (canonical significance unresolved; bucket stability unresolved; covariance complexity unattractive).
- Selective-only gate: **FAIL** (no bucket passed stability/min-size filters).

**Final decision:** the stronger evaluation regime did **not materially upgrade** the case for Kalman to central status. It upgraded instrumentation and surfaced directional signals, but retained core nulls. Keep Kalman **experimental**, not central and not yet selective-only policy.
