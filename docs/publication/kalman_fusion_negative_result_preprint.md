# When Uncertainty-Weighted Fusion Does Not Help: A Claim-Ready Mixed-Domain Retrieval Study of Kalman Fusion vs Mean Fusion

## 1. Abstract
We evaluate whether uncertainty-weighted Kalman fusion improves retrieval quality over a simple mean-fusion baseline in a claim-ready, mixed-domain benchmark setting. The original hypothesis was that Kalman fusion would outperform mean fusion on ranking quality. We run paired evaluation on *n* = 1193 test queries and compare methods using pre-declared statistical criteria and operational gates. On nDCG@10, Kalman fusion underperforms mean fusion by a small margin (delta = -0.000203 for Kalman minus mean), with Holm-adjusted *p* = 1.0. Mean fusion obtains nDCG@10 = 0.0943, while Kalman fusion obtains nDCG@10 = 0.0941. Hard routing is the highest-performing method in this comparison at nDCG@10 = 0.0979. Latency analysis shows a Kalman/mean ratio of approximately 1.022, which passes the pre-specified latency gate. The pre-declared confirmatory slice had zero matching queries and therefore cannot support the original claim. We interpret these results as negative evidence for the hypothesis under this benchmark and implementation, not as a universal theorem about uncertainty-weighted fusion.

## 2. Introduction
Hybrid retrieval pipelines often combine signals from multiple subsystems. A common expectation is that uncertainty-aware fusion should improve ranking by weighting stronger signals more heavily and down-weighting noisy ones. Kalman-style fusion offers a principled mechanism to combine estimates with uncertainty terms and has intuitive appeal in retrieval settings with heterogeneous sources.

This study tests that expectation under a claim-ready protocol intended for robust hypothesis adjudication. The target claim was directional: Kalman fusion should outperform mean fusion. We report that the benchmark evidence does not support that claim.

The objective of this preprint is to document the benchmark design, pre-registered decision rule, empirical outcomes, and interpretation boundaries in a form suitable for open scrutiny and replication.

## 3. Background and Motivation
Mean fusion is widely used because it is simple, stable, and inexpensive. Kalman-style fusion introduces additional structure through uncertainty weighting, which may improve robustness when sources have systematically different noise levels.

The motivating question is practical rather than theoretical: in a mixed-domain retrieval workload, does this additional complexity yield measurable gains over a strong, simple baseline?

Prior related literature should be reviewed and mapped to this setup in a camera-ready version. **TODO: add references on data-fusion methods in IR, uncertainty-aware ranking, and Kalman-inspired weighting in information systems.**

## 4. Hypothesis
**Primary hypothesis (pre-declared):**

- H1: Kalman fusion yields higher nDCG@10 than mean fusion on the claim-ready benchmark.

This hypothesis is directional and benchmark-scoped.

## 5. Benchmark Construction
The benchmark is a mixed-domain retrieval test set assembled for claim-readiness and paired statistical comparison across methods.

Key benchmark facts used for confirmatory testing:

- Paired test queries: *n* = 1193.
- Primary quality metric: nDCG@10.
- Methods evaluated include mean fusion, Kalman fusion, and hard routing.

Benchmark curation details, annotation procedures, and query-domain composition should be documented alongside released artifacts. **TODO: add dataset-construction and annotation protocol references/artifacts.**

## 6. Methods Compared
We compare three operationally relevant approaches:

1. **Mean fusion** (baseline): aggregates component signals by arithmetic averaging.
2. **Kalman fusion** (test method): uses uncertainty-weighted fusion to combine component signals.
3. **Hard routing**: routes queries to a single subsystem according to routing logic rather than soft combination.

Implementation details (feature scaling, uncertainty estimation, calibration, and fallback behaviour) should be specified in the final reproducibility package. **TODO: add implementation appendix and config references.**

## 7. Pre-registered Decision Rule
The confirmatory decision framework contains both quality and operational criteria.

- **Quality criterion:** evaluate paired nDCG@10 difference (Kalman minus mean) with multiplicity control; use Holm-adjusted significance for confirmatory inference.
- **Latency criterion:** ensure latency overhead remains within the pre-specified gate.
- **Slice criterion:** evaluate a pre-declared confirmatory slice when present.

Observed criterion outcomes:

- Holm-adjusted *p* = 1.0 for the Kalman-vs-mean quality contrast.
- Latency ratio (Kalman/mean) ≈ 1.022, which passed the latency gate.
- The pre-declared confirmatory slice had zero matching queries, so no confirmatory slice evidence is available.

## 8. Results
### 8.1 Main paired comparison (n = 1193)
- Mean fusion nDCG@10 = **0.0943**.
- Kalman fusion nDCG@10 = **0.0941**.
- Delta (Kalman minus mean) = **-0.000203**.
- Holm-adjusted *p* = **1.0**.

Interpretation: the observed effect is slightly negative and statistically non-supportive for H1 under the pre-registered rule.

### 8.2 Method ranking context
- Hard routing achieved the highest nDCG@10 among compared methods at **0.0979**.

### 8.3 Efficiency outcome
- Kalman/mean latency ratio ≈ **1.022**.
- This satisfied the latency gate, indicating acceptable overhead despite no quality gain.

## 9. Analysis
The central empirical outcome is a failure to confirm the original directional hypothesis. Kalman fusion did not exceed mean fusion on nDCG@10 in this claim-ready benchmark.

Several interpretations are plausible and non-exclusive:

- The uncertainty estimates used by the Kalman variant may not have been sufficiently informative.
- The benchmark’s domain mixture may favor robust averaging over uncertainty-sensitive weighting.
- Gains may exist only in specific regions of the query space that were not represented in the confirmatory slice.

Critically, because the pre-declared confirmatory slice had zero matching queries, it cannot contribute evidence for or against the original claim. This limits mechanistic interpretation while leaving the main paired result intact.

## 10. Limitations
- This is a single benchmark/implementation study; results may not transfer to other datasets, retrievers, or calibration schemes.
- The absence of matched queries in the pre-declared confirmatory slice reduces inferential granularity.
- We report endpoint metrics, but broader utility (e.g., downstream task impact) was not adjudicated here.
- External validity depends on how closely future workloads match this mixed-domain benchmark.

Accordingly, this should be interpreted as **negative evidence under this benchmark and implementation**, not a universal theorem that uncertainty-weighted fusion cannot help.

## 11. Reproducibility
To support independent verification, the following should accompany release:

- Exact benchmark split and query IDs for the *n* = 1193 paired set.
- Frozen method configurations for mean fusion, Kalman fusion, and hard routing.
- Evaluation scripts for nDCG@10, paired deltas, and multiplicity-adjusted tests.
- Latency measurement protocol and raw timing logs.
- Environment specification (library versions, hardware profile, random seeds).

**TODO: add repository paths, commit hashes, and artifact manifest once finalized.**

## 12. Conclusion
The original hypothesis—Kalman fusion > mean fusion on retrieval quality—was not supported on this claim-ready mixed-domain benchmark. Across *n* = 1193 paired queries, Kalman scored 0.0941 versus 0.0943 for mean (delta = -0.000203), with Holm-adjusted *p* = 1.0. Hard routing led the compared methods at 0.0979 nDCG@10. Although Kalman’s latency ratio (≈1.022) passed the operational gate, quality evidence did not support superiority. The pre-declared confirmatory slice contained zero matching queries and therefore cannot substantiate the claim. These findings are best treated as benchmark-specific negative evidence, not a universal impossibility result.
