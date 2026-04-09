## Demonstrated findings

- Positive result: kalman exceeds mean on ndcg@10 (Δ=0.021000, Holm-adjusted p=0.010000). This is demonstrated only for this benchmark configuration.
- Regression: kalman underperforms mean on recall@10 (Δ=-0.031000, Holm-adjusted p=0.020000). Treat this as a demonstrated risk until mitigated by follow-up experiments.

## Unresolved findings

- Null result: no statistically reliable difference is demonstrated for kalman versus mean on mrr (Δ=0.000000, Holm-adjusted p=0.400000).
- Inconclusive result: kalman versus mean on recall@1 shows ambiguous evidence (Δ=0.002000, Holm-adjusted p=0.600000). Additional power or tighter controls are required before drawing conclusions.

## Threats to validity

- Query sets may under-represent long-tail or adversarial cases.
- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.
- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.

## Benchmark limitations

- Results depend on the provided benchmark artifacts and should be treated as conditional evidence.
- Latency and memory proxies are environment-sensitive and may shift under different hardware/runtime settings.

## Recommended next experiments

- Increase held-out query count and rebalance domains before promoting unresolved findings.
- Add stress tests for distribution shift and low-resource domains to challenge demonstrated effects.
- Replicate on an independent benchmark slice with pre-registered metrics and hypotheses.
