# Pre-Registered Slice Claim Spec: Kalman vs Mean

## 0) Narrow claim under test

> Kalman fusion beats mean fusion on a pre-declared high-disagreement, high-uncertainty benchmark slice.

This document is a locked analysis contract for that single claim. It is intentionally narrower than the full canonical benchmark claim.

## 1) Comparison unit and methods

- Unit of analysis: query-level paired outcomes on the canonical test split.
- Candidate method: `KalmanorixFuser` (`kalman`).
- Baseline method: `MeanFuser` (`mean`).
- No method substitutions are allowed in this claim test.

## 2) Primary metric

- **Primary metric:** `ndcg@10` (query-level, then mean over the slice).

## 3) Primary comparison and hypothesis

- **Primary estimand:**
  - \(\Delta_q = ndcg@10_q(kalman) - ndcg@10_q(mean)\)
  - \(\bar{\Delta} = mean_q(\Delta_q)\)
- **Primary hypothesis test:** two-sided paired Wilcoxon signed-rank test on \(\Delta_q\).
- **Multiplicity handling:** Holm-adjusted p-values over the pre-registered family for this slice claim.
  - Family members:
    1. Kalman vs Mean on slice `ndcg@10` (primary)
    2. Kalman vs Mean on full-test `ndcg@10` (gatekeeper context check; non-headline)

## 4) Decision thresholds (locked)

- **Minimum effect size:** \(\bar{\Delta} \ge 0.02\) absolute `ndcg@10`.
- **Adjusted p-value threshold:** Holm-adjusted \(p \le 0.05\).
- **Latency ratio ceiling:**
  - \(mean\_latency\_{slice}(kalman) / mean\_latency\_{slice}(mean) \le 1.50\).
- **Power adequacy threshold:** both must hold:
  - `n_pairs_slice >= 20`, and
  - detectable-effect estimate \( (1.96 + 0.84) * std(\Delta_q) / sqrt(n) \le 0.02 \).

## 5) Exact slice definition (metadata only)

Slice membership is defined only from per-query metadata available in benchmark artifacts (`query_metadata`) and deterministic derived thresholds.

Let thresholds be computed once on the evaluated test split:

- `D50` = median of `specialist_disagreement`
- `U50` = median of `uncertainty_spread`
- `C33` = 33rd percentile of `router_confidence`

A query is in the **pre-registered slice** iff all are true:

1. `specialist_disagreement >= D50` (high disagreement)
2. `uncertainty_spread >= U50` (high uncertainty)
3. `router_confidence <= C33` (low router confidence)
4. `is_multi_domain == True`

If `is_multi_domain` is missing, derive fallback as:
- `is_multi_domain := (specialist_count_selected[router_only_top1] > 1.0)`.

No post-hoc threshold tuning is allowed.

## 6) Exclusion rules

Exclude a query from the slice analysis if any of the following hold:

1. Missing `ndcg@10` for either `kalman` or `mean`.
2. Missing required metadata for slice assignment after applying the one allowed fallback above.
3. Query not present in both methods (paired requirement violation).
4. Duplicate `query_id` records after artifact loading (must be de-duplicated before analysis; unresolved duplicates are excluded).

All exclusions must be counted and reported with reasons.

## 7) Interpretation rules

### Supported
Declare the narrow claim **supported** only if all conditions hold on the pre-registered slice:

1. \(\bar{\Delta} \ge 0.02\)
2. Holm-adjusted \(p \le 0.05\)
3. 95% paired bootstrap CI for \(\bar{\Delta}\) excludes 0 on the positive side
4. Latency ratio ceiling satisfied (<= 1.50)
5. Power adequacy threshold satisfied

### Unsupported
Declare **unsupported** if power is adequate and both hold:

- \(\bar{\Delta} \le 0\), and
- Holm-adjusted \(p \le 0.05\).

### Inconclusive
Declare **inconclusive** otherwise, with mandatory subtype:

- `inconclusive_underpowered`: power adequacy threshold failed.
- `inconclusive_sufficiently_powered`: power adequate, but supported/unsupported criteria not met.

## 8) Exploratory bucket firewall

## Do not promote exploratory buckets to headline claims

- Any bucket not matching the exact slice definition in Section 5 is exploratory.
- Exploratory findings may be reported for diagnosis, but must not be used to revise this claim outcome.
- Exploratory buckets must be labeled `exploratory_only` and kept out of abstract/headline statements.
- No retrospective “best bucket” selection is permitted for Kalman-vs-Mean claim language.

## 9) Required artifacts

The final run package must include:

1. Slice membership list (`query_id` + inclusion/exclusion reason)
2. Threshold values (`D50`, `U50`, `C33`)
3. Paired stats table (`mean_difference`, CI, raw/adjusted p, `n_pairs`)
4. Latency ratio calculation for the slice
5. Power adequacy calculation and verdict subtype
