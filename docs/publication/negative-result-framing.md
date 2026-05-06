# Negative Result Framing

## 1) Original hypothesis
The original hypothesis was that uncertainty-weighted Kalman fusion would deliver meaningfully better mixed-domain retrieval quality than simpler fusion strategies, and therefore justify its additional complexity and cost.

## 2) Falsification-oriented benchmark design
The benchmark was intentionally designed to challenge that hypothesis rather than accommodate it. It compared uncertainty-weighted Kalman fusion against strong, practical alternatives under the same mixed-domain conditions, with a pre-registered decision rule for declaring a meaningful win. This made it possible for the method to fail clearly if its advantage was not robust.

## 3) Why this result is informative
- **Claim-ready sample size:** The evaluation set is large enough to support a stable performance conclusion, reducing the chance that the outcome is a small-sample artifact.
- **Strong baselines:** Kalman fusion was not compared to weak controls; it was tested against credible baselines that practitioners would actually deploy.
- **Pre-registered decision rule:** Success criteria were specified in advance, limiting post hoc reinterpretation.
- **Routing beats fusion:** In this setup, hard routing outperformed fusion while also being operationally simpler and cheaper.

## 4) What this result does not prove
- It is **not** a universal rejection of Kalman fusion.
- It is **not** a rejection of uncertainty-aware retrieval in general.
- It is **not** a claim about the non-hash neural embedding mode unless that mode is evaluated separately.

## 5) Stronger revised claim
“In this mixed-domain retrieval setup, uncertainty-weighted Kalman fusion offers no practical advantage over mean fusion, while hard routing is more effective and cheaper.”
