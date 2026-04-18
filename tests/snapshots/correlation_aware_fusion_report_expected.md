# Correlated Experts Benchmark Slice (Synthetic, Narrow Hypothesis)

**Label:** Synthetic-to-real bridge. Synthetic outputs are exploratory only and are not headline proof.

Question: under partially correlated experts with non-identical uncertainty quality, does correlation-aware Kalman improve over baseline Kalman?

**Answer:** In this synthetic narrowed regime, LearnedLinearCombiner outperformed baseline Kalman (ΔMRR@10=+0.0503). Treat as exploratory only.

## Test metrics

| Method | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| MeanFuser | 0.4778 | 0.9000 | 0.6498 |
| KalmanorixFuser | 0.5333 | 0.9111 | 0.6916 |
| CorrelationAwareKalmanFuser | 0.5333 | 0.9111 | 0.6920 |
| WeightedMeanFuser | 0.5333 | 0.9111 | 0.6916 |
| LearnedLinearCombiner | 0.6111 | 0.9556 | 0.7419 |

## Primary metric deltas vs baseline Kalman

| Method | ΔMRR@10 |
| --- | ---: |
| MeanFuser | -0.0418 |
| CorrelationAwareKalmanFuser | +0.0004 |
| WeightedMeanFuser | +0.0000 |
| LearnedLinearCombiner | +0.0503 |

## Paired statistics vs baseline Kalman (MRR@10 per query)

| Method | ΔMean | 95% bootstrap CI | Wilcoxon p | Cohen dz | Rank-biserial |
| --- | ---: | --- | ---: | ---: | ---: |
| MeanFuser | -0.0418 | [-0.0808, -0.0060] | 0.01439 | -0.231 | -0.520 |
| CorrelationAwareKalmanFuser | +0.0004 | [-0.0017, +0.0028] | 0.6547 | +0.036 | +0.333 |
| WeightedMeanFuser | +0.0000 | [+0.0000, +0.0000] | 1 | +0.000 | +0.000 |
| LearnedLinearCombiner | +0.0503 | [+0.0124, +0.0894] | 0.02299 | +0.266 | +0.483 |

## Fusion latency

- Latency metrics are recorded in `summary.json` under `latency_ms` (mean and p95 ms/query).

## Correlated slice definition

- Query-level correlated score: `||mean(residuals)|| / mean(||residual_i||)`.
- Correlated if score ≥ `0.70`.
- Correlated queries on test: `75` / `90`.

## Interpretation

- This is a narrowed hypothesis regime, not a broad claim about general retrieval settings.
- Any synthetic win is treated as directional evidence only; it does not close the headline Kalman-vs-mean claim.
