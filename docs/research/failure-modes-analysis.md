# Kalmanorix Failure Modes Analysis

This document identifies practical failure cases in Kalmanorix and maps each to:

1. detection mechanisms (online and offline)
2. mitigations (fallbacks, thresholds, and routing/fusion changes)

Scope focus: bad uncertainty estimates, conflicting specialists, out-of-domain (OOD) queries, and degenerate embeddings.

---

## 1) Failure cases

### A. Bad uncertainty estimates

#### A1. Overconfident specialist dominates fusion
- **Pattern**: one specialist reports very low `sigma2` on an actually bad embedding.
- **Consequence**: Kalman gain over-weights the wrong expert, pushing the fused vector toward error.
- **Why it can happen**:
  - `sigma2` is query-independent (`ConstantSigma2`) while quality is query-dependent.
  - poor calibration drift after model updates.
  - uncertainty proxy saturates in unexpected regions.

#### A2. Underconfident specialist is ignored
- **Pattern**: specialist with genuinely useful embedding reports too large `sigma2`.
- **Consequence**: fusion collapses toward other modules; specialist value is lost.

#### A3. Global mis-scaling of uncertainty
- **Pattern**: all uncertainties shifted by multiplicative factor (too large/small).
- **Consequence**: either near-mean behavior (all too large/similar) or unstable winner-take-most behavior (all too small + tiny relative differences).

---

### B. Conflicting specialists

#### B1. Strong semantic disagreement with similar uncertainties
- **Pattern**: top specialists produce far-apart embeddings but have close sigma².
- **Consequence**: fused vector lands between incompatible modes (semantic blur).

#### B2. Routing ambiguity near centroid boundaries
- **Pattern**: query similar to multiple domain centroids; confidence gap is small.
- **Consequence**: unstable selection (jitter), inconsistent results across minor query edits.

#### B3. Alignment drift between specialists
- **Pattern**: one specialist’s space drifts (or alignment becomes stale), creating apparent conflict.
- **Consequence**: fusion quality degrades even with “correct” routing.

---

### C. Out-of-domain queries

#### C1. Unknown domain routed as known
- **Pattern**: OOD query still has moderate cosine similarity to one centroid.
- **Consequence**: wrong specialist selected with high confidence.

#### C2. No specialist passes threshold
- **Pattern**: semantic routing returns empty set (or would, absent fallback).
- **Consequence**: brittle behavior if fallback is weakly chosen.

#### C3. Novel compositional query
- **Pattern**: mixed-domain query where none of the specialists is truly appropriate end-to-end.
- **Consequence**: either over-pruning (missing needed modules) or over-fusion of irrelevant ones.

---

### D. Degenerate embeddings

#### D1. Zero/near-zero query embedding from router embedder
- **Pattern**: `fast_embedder` returns zero vector.
- **Consequence**: cosine normalization and similarity become uninformative; router falls back.

#### D2. Non-finite vectors (NaN/Inf)
- **Pattern**: embedder/runtime issue emits invalid values.
- **Consequence**: Kalman update fails input validation; request fails hard.

#### D3. Collapsed specialist representation
- **Pattern**: one specialist outputs almost-constant vectors.
- **Consequence**: misleadingly stable uncertainty signals + low semantic discrimination.

#### D4. Degenerate centroid
- **Pattern**: centroid norm is zero (or invalid), so module centroid is unusable.
- **Consequence**: semantic/confidence routing effectively ignores that specialist’s centroid.

---

## 2) Detection mechanisms

### A. Uncertainty quality monitoring

- **Calibration dashboards**:
  - embedding calibration (ECE/Brier over uncertainty bins)
  - retrieval calibration by query variance bins
- **Drift checks**:
  - rolling correlation between predicted sigma² and observed retrieval error
  - percentile drift for sigma² distribution (`p10/p50/p90`)
- **Online anomaly flags**:
  - specialist weight spikes (single module > 0.9 repeatedly)
  - abrupt variance floor hits (many modules clamped near minimum)

### B. Conflict detection

- **Pairwise disagreement score**:
  - cosine distance matrix between selected specialist embeddings
  - alert when max disagreement > threshold and sigma² values are similar
- **Conflict-to-confidence ratio**:
  - high routing confidence + high embedding disagreement indicates likely misrouting/alignment issue
- **Alignment health checks**:
  - periodic alignment validation on anchor pairs
  - monitor post-alignment similarity uplift over baseline

### C. OOD detection

- **Centroid margin metrics**:
  - top similarity (`s1`), second similarity (`s2`), gap (`s1-s2`)
  - low `s1` and low gap => likely OOD/ambiguous
- **Rejection-region policy**:
  - if `s1 < tau_low`, mark as OOD candidate regardless of rank
- **Fallback audit logging**:
  - track fallback path (`all` vs `hard`) rate and success

### D. Degeneracy detection

- **Vector validity checks**:
  - finite-value guardrails for all specialist embeddings and covariances
  - norm checks (`||z|| < eps`)
- **Specialist collapse metric**:
  - variance of embeddings across random query probe set
  - low variance implies representation collapse
- **Centroid integrity checks**:
  - non-zero norm and finite centroid before enabling semantic routing

---

## 3) Mitigation strategies

### A. Mitigations for bad uncertainty estimates

1. **Sigma² guardrails**
   - clamp per-specialist sigma² into calibrated interval `[sigma_min, sigma_max]`
   - enforce request-level max weight cap (e.g., 0.8) unless confidence is very high

2. **Adaptive rescaling**
   - use `ScaledSigma2`-style per-specialist correction factors learned from validation windows
   - retrain/recalibrate when calibration ECE degrades beyond threshold

3. **Uncertainty ensemble**
   - blend multiple uncertainty estimators (centroid distance + embedding norm + stochastic variance)
   - use robust aggregation (median/log-space mean) to reduce single-estimator failure

4. **Fallback fusion mode switch**
   - when uncertainty diagnostics fail, route to `MeanFuser` or ensemble Kalman until recalibration completes

### B. Mitigations for conflicting specialists

1. **Conflict-aware gating**
   - if disagreement score > `delta_conflict`, do not fully fuse all modules
   - prefer top-k routing + confidence-conditioned single-specialist path

2. **Dynamic thresholding**
   - use `threshold_top_k` or `threshold_relative_to_max` for bounded module count
   - increase selectivity when confidence gap is high; decrease when gap is low

3. **Mixture-of-experts fallback**
   - return candidate-specific outputs (top specialist + fused option) for downstream reranker
   - avoid forcing a single averaged representation under severe conflict

4. **Alignment refresh**
   - trigger re-alignment jobs when conflict rises while domain-confidence remains high

### C. Mitigations for OOD queries

1. **Explicit abstain route**
   - add OOD decision state before specialist selection:
     - if `s1 < tau_ood` and `gap < tau_gap`, abstain or escalate to generalist

2. **Safer fallback policy**
   - prefer `fallback_mode='all'` for suspected OOD to avoid brittle hard misrouting
   - optionally include a generalist prior via `fuse_with_prior`

3. **Query reformulation loop**
   - if OOD detected, request clarification or split into sub-queries (for interactive systems)

4. **OOD benchmark in CI**
   - enforce minimum robustness metrics on mixed-domain and synthetic OOD sets before deployment

### D. Mitigations for degenerate embeddings

1. **Input sanitization**
   - reject/repair non-finite vectors (`np.nan_to_num`) before fusion where appropriate
   - enforce minimum norm; if violated, route through fallback path

2. **Router hardening**
   - if fast embedder is degenerate, bypass semantic/confidence mode for request and use `all`
   - keep per-request reason code in metadata for observability

3. **Specialist quarantine**
   - temporarily disable modules with persistent collapse/invalid outputs
   - continue serving with remaining healthy specialists

4. **Canary probes**
   - scheduled synthetic probe queries to detect silent regression (zero vectors, constant outputs, NaNs)

---

## Suggested operational thresholds (starting points)

- **OOD candidate**: `top_similarity < 0.35` and `similarity_gap < 0.05`
- **Conflict trigger**: max pairwise cosine distance among selected specialists `> 0.6`
- **Uncertainty drift trigger**: 7-day ECE increase `> 25%` relative to trailing 30-day baseline
- **Degenerate output**:
  - `||embedding|| < 1e-8` OR non-finite values present
  - specialist probe-set variance below very small floor

These should be tuned per deployment using held-out logs.

---

## Implementation priority (practical rollout)

1. Add online metrics/alerts (calibration, conflict, OOD, degeneracy).
2. Add OOD-abstain + safer fallback routing.
3. Add sigma² guardrails + per-specialist recalibration factors.
4. Add conflict-aware gating and periodic alignment refresh.
5. Automate canary probes and specialist quarantine.

