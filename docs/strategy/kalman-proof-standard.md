# Internal Memo: Proof Standard for the Kalman>Mean Claim

## Question
What evidence would actually convince us that Kalman fusion beats mean fusion here?

## Bottom line
Right now we do **not** have that evidence. The canonical v2 artifact reports an *inconclusive underpowered* verdict for Kalman vs mean, with only six test queries, a non-significant adjusted p-value, and a latency-ratio failure against the project’s own acceptance rule. We should treat any apparent quality edge as a signal to test harder, not as proof. 

## Why current evidence is insufficient

1. **The canonical decision is explicitly unresolved.**
   - Current decision verdict: `inconclusive_underpowered`.
   - Rule checks are mixed: effect-size check passes, but adjusted p-value and latency-ratio checks fail.

2. **Sample size is too small to support a strong claim.**
   - In the committed canonical v2 summary, Kalman/mean comparisons are based on `num_queries = 6` for the methods shown.
   - At this scale, confidence intervals are wide and one query can move the conclusion.

3. **Observed positive delta is not statistically reliable yet.**
   - The decision block shows a positive `primary_metric_delta` on nDCG@10, but adjusted p-value is `1.0`.
   - By our own rule, that means we do not have evidence of a true quality lift.

4. **Latency constraint is currently violated.**
   - The same decision block reports `latency_ratio_vs_mean` above the max allowed threshold.
   - Even if quality improved, the canonical claim requires acceptable latency/FLOPs trade-off.

5. **Most supporting tracks are either null downstream or exploratory.**
   - Uncertainty ablation shows calibration differences but largely flat retrieval outcomes in this setup.
   - Correlation-aware fusion is explicitly synthetic/narrowed and non-headline by policy.

## What confounds remain

1. **Underpowered evaluation confound.**
   With so few queries, we cannot separate true model effect from sampling noise.

2. **Benchmark representativeness confound.**
   Repository docs repeatedly warn that current setups may not represent production distributions or broader domain/query diversity.

3. **Metric-tradeoff confound.**
   A quality uptick is insufficient if it requires unacceptable latency overhead under the stated decision rule.

4. **Uncertainty-quality confound.**
   Kalman’s theoretical advantage depends on reliable uncertainty estimates; current ablations indicate better calibration proxies do not automatically translate into better retrieval.

5. **Synthetic-to-real transfer confound.**
   Correlation-aware synthetic gains may identify mechanism, but they do not establish real-benchmark superiority.

## What a convincing win would look like

A convincing win must satisfy all of the following on a **real**, reproducible canonical artifact:

1. **Rule-complete pass on the project’s canonical criteria** (not partial):
   - nDCG@10 paired mean delta `>= 0.02` (Kalman - Mean),
   - Holm-adjusted `p <= 0.05`,
   - latency ratio (Kalman/Mean) `<= 1.5`,
   - FLOPs ratio (Kalman/Mean) `<= 1.1`.

2. **Adequate statistical power / sample size** such that:
   - the run is no longer tagged underpowered,
   - the CI around the delta excludes trivial or zero effects,
   - the outcome is robust to reasonable bootstrap/resample settings.

3. **Replication, not one-off luck.**
   - Repeatable across at least one additional real split/run with the same direction and practical size of effect.

4. **No hidden degradation pattern.**
   - Bucket/domain analysis should not reveal material regressions masked by aggregate means.

If those conditions hold, we can honestly say Kalman beats mean *for the benchmarked regime*.

## What would count as a failure of the hypothesis

Treat the hypothesis (“Kalman fusion beats mean fusion here”) as failed for this regime if we see one or more of:

1. **Powered nulls over repeated real runs.**
   - Sufficiently powered canonical reruns keep yielding non-significant or near-zero deltas.

2. **Trade-off failure persists.**
   - Quality signal exists but repeatedly fails latency/FLOPs acceptance constraints.

3. **Mechanism checks stay flat where they should matter.**
   - Improvements in uncertainty calibration/covariance modeling continue to produce little or no downstream retrieval gain.

4. **Competitor baselines dominate.**
   - Simpler or learned combiners repeatedly match or beat Kalman in the same real protocol.

At that point, “Kalman > mean” is not the right top-line claim for this benchmark context.

## When to pivot to a narrower or different claim

We should pivot if the next powered canonical cycle still does not clear the full decision rule.

### Pivot trigger conditions
- Canonical verdict remains inconclusive/unsupported after a meaningfully larger run.
- Latency-ratio noncompliance remains persistent despite optimization.
- Gains appear only in specific structures (e.g., correlated experts) and fail to generalize.

### Better fallback claims (if supported by artifacts)
1. **Routing efficiency claim** as primary value proposition (already supported).
2. **Conditional Kalman claim** limited to a clearly defined regime (for example, correlated-expert settings), labeled as scoped and non-general.
3. **Engineering claim** that Kalman path quality is comparable but with trade-offs, rather than superior.

## Operating standard going forward (strict)

Until a claim-ready canonical artifact clears all gates, all external and internal summaries should use this language:

- “Kalman quality superiority over mean is **not demonstrated yet**.”
- “Current evidence is **underpowered/inconclusive** for headline superiority.”
- “Any synthetic or narrowed-regime gains are **exploratory**.”

That is the honest standard consistent with current repository evidence.
