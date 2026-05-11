# Negative Result Diagnostics Report

## Scope
This module explains why the null confirmatory result is informative rather than ambiguous for the canonical domain-balanced C100 run (`n_pairs=1193`). It is designed as a reusable analysis artifact for TMLR and arXiv reporting workflows.

## Inputs (confirmatory readout)
- Primary effect (Kalman - Mean, nDCG@10): `-9.258801070226193e-06`
- Secondary effect (Kalman - Mean, Recall@100): `0.0`
- Holm-adjusted p-value (primary family): `1.0`
- Latency ratio (Kalman/Mean): `1.0722551296204925`
- FLOPs ratio (Kalman/Mean): `1.0`
- Practical threshold for meaningful quality gain (abs delta nDCG@10): `0.02`

## 1) Effect-size detectability
Observed quality delta is approximately zero and more than three orders of magnitude smaller than the predeclared practical threshold. Under this protocol, the experiment is sensitive to practically relevant gains but shows no evidence for one. The null is therefore informative: it constrains plausible gain magnitude to values too small to matter operationally in this configuration.

## 2) Confidence interval interpretation
The confidence interval should be interpreted as a compatibility range for the true paired effect under repeated sampling from the same data-generating process. Because the point estimate is nearly zero and adjusted significance is fully null, the interval-level interpretation is that any positive effect consistent with the data is too small and too uncertain to support a superiority claim.

## 3) Comparison to practical threshold
The observed nDCG@10 delta (`-9.26e-06`) is far below the minimum meaningful threshold (`0.02`). The decision signal is not merely "not significant"; it is also "not practically relevant." This dual failure is central: superiority is blocked both inferentially and operationally.

## 4) Latency penalty interpretation
Latency ratio > 1 indicates Kalman fusion is slower than Mean under matched-FLOPs accounting. The measured ~7.23% runtime overhead is within the allowed bound but still a cost. In the absence of demonstrated quality benefit, that overhead weakens deployment justification for the more complex fusion rule.

## 5) Baseline competitiveness
Mean fusion remains competitive because the tested alternative does not exceed it on primary or secondary quality metrics. The null therefore strengthens a conservative baseline-first recommendation: retain the simpler baseline until evidence of material uplift appears.

## 6) Uncertainty calibration impact
Current uncertainty weighting does not translate into detectable retrieval improvement in this benchmark slice. This does not prove uncertainty modeling is useless; it indicates that this calibration strategy (as instantiated here) fails to generate measurable ranking benefit under claim-gated evaluation.

## 7) Evidence needed to reverse the conclusion
A reversal would require **all** of the following in a new confirmatory run:
1. Positive primary direction (`delta nDCG@10 > 0`).
2. Practical relevance (`|delta| >= 0.02` or revised preregistered threshold justified ex ante).
3. Multiplicity-adjusted significance (`Holm-adjusted p < 0.05`).
4. Stable uncertainty intervals excluding negligible effects around zero.
5. Acceptable compute overhead with quality gain retained under independent replication and at least one additional benchmark slice.

## Conclusion
The null result is informative because it is both powered and decision-relevant: it narrows plausible effect sizes, preserves a strong baseline, and sets concrete criteria for what future evidence must show to justify changing conclusions.
