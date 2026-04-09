# Overall Metrics

| strategy | mrr_mean | recall@1_mean | recall@5_mean | ndcg@10_mean | latency_ms_mean | memory_proxy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| kalman | 1.0 | 1.0 | 1.0 | 1.0 | 5.5 | 2.0 |
| mean | 0.5 | 0.0 | 1.0 | 0.63093 | 4.25 | 2.0 |


# Statistical Significance (Holm-corrected)

| reference | candidate | metric | mean_diff | p_value | adjusted_p_value | cohen_dz | rank_biserial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kalman | mean | mrr | 0.5 | 0.5 | 1.0 | 0.0 | 1.0 |
| kalman | mean | ndcg@10 | 0.36907 | 0.5 | 1.0 | 0.0 | 1.0 |
| kalman | mean | recall@1 | 1.0 | 0.5 | 1.0 | 0.0 | 1.0 |
| kalman | mean | recall@10 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| kalman | mean | recall@5 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |


## Calibration Summary

| strategy | n_samples | ece | brier_score | mean_confidence | mean_accuracy | overconfidence_gap |
| --- | --- | --- | --- | --- | --- | --- |
| kalman | 2.0 | 0.15 | 0.025 | 0.85 | 1.0 | -0.15 |
| mean | 2.0 | 0.65 | 0.425 | 0.65 | 0.0 | 0.65 |

# Figures

## Latency/Memory tradeoff

![Latency vs memory](<OUT_DIR>/figures/latency_memory_tradeoff.png)

## Quality/Latency frontier

![MRR vs latency](<OUT_DIR>/figures/quality_latency_frontier.png)


## Demonstrated findings

- No demonstrated directional effect is established by the current statistical evidence.

## Unresolved findings

- Inconclusive result: kalman versus mean on mrr shows ambiguous evidence (Δ=0.500000, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on ndcg@10 shows ambiguous evidence (Δ=0.369070, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Inconclusive result: kalman versus mean on recall@1 shows ambiguous evidence (Δ=1.000000, Holm-adjusted p=1.000000). Additional power or tighter controls are required before drawing conclusions.
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@10 (Δ=0.000000, Holm-adjusted p=1.000000).
- Null result: no statistically reliable difference is demonstrated for kalman versus mean on recall@5 (Δ=0.000000, Holm-adjusted p=1.000000).

## Threats to validity

- Query sets may under-represent long-tail or adversarial cases.
- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.
- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.

## Benchmark limitations

- The benchmark evaluates retrieval behavior only; downstream task performance is not measured here.
- Calibration metrics use confidence proxies rather than calibrated probabilities from a dedicated model.

## Recommended next experiments

- Increase held-out query count and rebalance domains before promoting unresolved findings.
- Add stress tests for distribution shift and low-resource domains to challenge demonstrated effects.
- Replicate on an independent benchmark slice with pre-registered metrics and hypotheses.

