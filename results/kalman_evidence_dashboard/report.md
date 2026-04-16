# Kalman Evidence Dashboard (Top-level)

This dashboard is the single decision artifact for the current Kalmanorix evidence state. It intentionally preserves negative and inconclusive outcomes.

## Scope and source artifacts

Primary sources used:
- `results/canonical_benchmark/summary.json` (canonical v1)
- `results/canonical_benchmark_v2/summary.json` (canonical v2 + bucket analysis + decision scaffold)
- `results/uncertainty_calibration/summary.json`
- `results/uncertainty_ablation/summary.json`
- `results/kalman_covariance_ablation/summary.json`
- `results/kalman_covariance_ablation_v2/summary.json`
- `results/correlation_aware_fusion/summary.json`
- `results/kalman_latency_optimization/summary.json`
- `results/kalman_latency_optimization/canonical/summary.json`
- `results/routing_eval/small_routing_eval_v1_report.json`

---

## 1) Canonical outcomes (v1 and v2)

### Canonical v1 (reference)
- `ndcg@10`: Kalman **0.5247** vs Mean **0.4537** vs Router-top1 **0.6414**.
- `mrr@10`: Kalman **0.3750** vs Mean **0.2806** vs Router-top1 **0.5278**.
- Latency: Kalman **25.81ms** vs Mean **11.92ms** (≈2.16× slower).
- Pairwise stats (Kalman vs Mean) are not significant (`adjusted_p_value=1.0` in v1 paired statistics).

### Canonical v2 (current baseline decision frame)
- `ndcg@10`: Kalman **0.5486** vs Mean **0.4537** vs Router-top1 **0.6414**.
- `mrr@10`: Kalman **0.4056** vs Mean **0.2806** vs Router-top1 **0.5278**.
- Latency: Kalman **40.58ms** vs Mean **19.74ms** (≈2.06× slower).
- Official v2 decision: **`inconclusive_underpowered`**.
- Explicit check failures in v2 decision block:
  - adjusted p-value check: failed (`adjusted_p_value=1.0` > 0.05 threshold)
  - latency ratio check: failed (`2.055` > `1.5` threshold)

Interpretation: Kalman shows directional quality gains over Mean, but current canonical evidence does **not** meet its own significance + latency gating policy.

---

## 2) Uncertainty calibration result

- Selected objective: **`distance_to_relevant_doc_centroid`**.
- Calibration power checks: **passed** (`powered_for_calibration=true`; support tech=43, cook=31; min threshold=24).
- Reliability diagnostics improved post-calibration (ECE and Spearman moved in the right direction per module diagnostics).
- But benchmark outcome did not move:
  - validation delta change = **0.0**
  - test delta change = **0.0**
- Reported observation explicitly states diagnostics improved **without downstream fusion gain**.

Interpretation: calibration seems technically valid for uncertainty quality, but did not translate to fused retrieval gains in this setup.

---

## 3) Uncertainty ablation result

- Summary answer: **"Partially"** — calibration differences are measurable, but retrieval gains are limited and constant uncertainty remains competitive.
- On `toy_mixed`, all methods have identical retrieval (`recall_at_1=0.7333`, `mrr_at_10=0.8556`) while calibration metrics vary.
- Best `toy_mixed` ECE among listed methods is **stochastic_forward_sigma2 (0.0909)**.

Interpretation: sigma² method choice affects calibration quality more than retrieval quality in the tested ablation regime.

---

## 4) Covariance ablation result

### Covariance ablation (v1)
- Answer: richer covariance **did not clearly beat** simpler baselines.
- MRR@10: scalar (0.3526) > diagonal (0.3493) > structured (0.3441) > mean (0.3294).

### Covariance ablation (v2, enlarged)
- Answer: richer covariance **not worth it in this setup**.
- MRR@10: scalar **0.2806**, diagonal **0.2801**, structured **0.2810** (all close).
- Efficiency cost is substantial for richer forms:
  - mean latency/query **0.0491ms**
  - scalar **0.2671ms**
  - diagonal **0.2336ms**
  - structured **0.4927ms**
- Per-bucket Recall@1 small differences do not clear practical value thresholds.

Interpretation: move from scalar to richer covariance has weak empirical upside and clear runtime cost.

---

## 5) Correlation-aware result

- Best variant: **CorrelationAwareKalmanFuser (covariance_inflation)**.
- Test MRR@10:
  - Kalman baseline: **0.5318**
  - correlation-aware best: **0.5356**
  - delta: **+0.0037**
- Bucket behavior:
  - high-correlation bucket MRR@10: 0.4637 → **0.4645** (small)
  - low-correlation bucket MRR@10: 0.6000 → **0.6066** (small)
- Notes state no optimistic retuning on test and explicit reporting of null/negative outcomes.

Interpretation: correlation-aware adjustment is promising but currently a **small incremental gain**, not a decisive shift.

---

## 6) Latency optimization result

- Single-query speedup (legacy Kalman → optimized): **2.06×**.
- Batch speedup: **1.23×**.
- Despite optimization, optimized Kalman remains slower than Mean:
  - optimized/mean latency ratio: **2.53×** (microbenchmark)
  - canonical rerun decision ratio: **2.008×**, failing threshold `<=1.5`
- Canonical rerun verdict remains **`inconclusive_underpowered`**.

Interpretation: optimization improved Kalman internals, but not enough to satisfy deployment latency gates relative to Mean.

---

## 7) Routing evidence

- Single run at semantic threshold 0.7:
  - routing F1 **0.6944**
  - avg FLOPs savings **0.4444**
  - avg latency delta **+6.08ms** (all - routed)
- Outcome mix:
  - quality-preserving wins: **4**
  - compute-only win: **1**
  - failure with quality loss: **1**
- Threshold sweep:
  - best F1 at threshold **0.5** (F1=0.7778)
  - robustness ranges are non-trivial (`f1_range=0.1667`, `recall_range=0.4167`)

Interpretation: routing provides meaningful efficiency leverage, but robustness and quality risk need explicit guardrails.

---

## 8) Per-bucket findings (cross-artifact)

### Canonical v2 bucket analysis
- `consistent_kalman_gain_buckets`: **none**.
- `low_uncertainty_spread` (n=3): only bucket marked `consistent_gain`, Kalman `ndcg@10` delta vs Mean = **+0.1898**.
- Most other buckets are `mixed_or_no_gain` and all are exploratory due to low `n_pairs`.

### Covariance ablation v2 bucket table
- Recall@1 gains from richer covariance are small and inconsistent across
  `high_disagreement`, `multi-domain`, `uncertainty-skewed` buckets.

### Correlation-aware buckets
- Small positive shifts in both high- and low-correlation buckets for covariance inflation variant; effect size remains modest.

Interpretation: bucket-level evidence suggests **narrow, context-dependent wins**, not broad consistency.

---

## Rule-based verdicts

Rules are intentionally conservative and tied to source artifact thresholds/findings.

1) **Implementation quality verdict: `good_but_not_fully_hardened`**
- Rule:
  - If optimized path exists and preserves numerical closeness, and specialized ablations are reproducible/reportable, quality is at least good.
  - Downgrade from "hardened" if latency gates still fail in canonical decision artifact.
- Evidence:
  - optimization speedups + numerical deviation tracking exist;
  - canonical latency gate failure persists.

2) **Empirical support verdict: `mixed_and_underpowered`**
- Rule:
  - If canonical decision is `inconclusive_underpowered`, bucket consistency list is empty, and multiple side studies show partial/small gains, classify as mixed + underpowered.
- Evidence:
  - canonical v2 decision,
  - no consistent gain buckets,
  - uncertainty calibration no benchmark delta,
  - correlation-aware only small uplift,
  - covariance richer variants not worth complexity.

3) **Deployment readiness verdict: `not_ready_for_default_rollout`**
- Rule:
  - Block default rollout if either significance gate or latency gate fails in canonical decision framework.
- Evidence:
  - adjusted p-value and latency checks fail in canonical v2/canonical rerun artifacts.

4) **Program status verdict (required choice): `selective-only`**
- Rule:
  - Choose `stable` only if canonical significance + latency checks pass and gains are consistent across buckets.
  - Choose `experimental` for broad ongoing exploration without operational guardrails.
  - Choose `selective-only` when there are credible localized gains but not enough for global default.
- Evidence synthesis:
  - localized positives (low-uncertainty bucket, slight correlation-aware gain, routing compute wins)
  - offset by underpowered canonical stats, latency failures, and mixed bucket behavior.

**Final recommendation:** Keep Kalman as **selective-only** behind explicit routing/feature flags and bucket-aware eligibility rules; do not promote to stable default yet.
