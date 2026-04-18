# Correlated Experts Benchmark Slice (Synthetic, Narrow Hypothesis)

**Label:** Synthetic-to-real bridge. Synthetic outputs are exploratory only and are not headline proof.

Question: under partially correlated experts with non-identical uncertainty quality, does correlation-aware Kalman improve over baseline Kalman?

**Answer:** In this synthetic narrowed regime, LearnedLinearCombiner outperformed baseline Kalman (ΔMRR@10=+0.0453). Treat as exploratory only.

## Test metrics

| Method | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| MeanFuser | 0.3381 | 0.6762 | 0.4818 |
| KalmanorixFuser | 0.4048 | 0.7476 | 0.5512 |
| CorrelationAwareKalmanFuser | 0.4048 | 0.7500 | 0.5516 |
| WeightedMeanFuser | 0.4048 | 0.7476 | 0.5512 |
| LearnedLinearCombiner | 0.4429 | 0.8095 | 0.5964 |

## Primary metric deltas vs baseline Kalman

| Method | ΔMRR@10 |
| --- | ---: |
| MeanFuser | -0.0694 |
| CorrelationAwareKalmanFuser | +0.0005 |
| WeightedMeanFuser | +0.0000 |
| LearnedLinearCombiner | +0.0453 |

## Paired statistics vs baseline Kalman (MRR@10 per query)

| Method | ΔMean | 95% bootstrap CI | Wilcoxon p | Cohen dz | Rank-biserial |
| --- | ---: | --- | ---: | ---: | ---: |
| MeanFuser | -0.0694 | [-0.0857, -0.0531] | 1.405e-18 | -0.407 | -0.747 |
| CorrelationAwareKalmanFuser | +0.0005 | [-0.0010, +0.0019] | 0.2923 | +0.031 | +0.330 |
| WeightedMeanFuser | +0.0000 | [+0.0000, +0.0000] | 1 | +0.000 | +0.000 |
| LearnedLinearCombiner | +0.0453 | [+0.0254, +0.0650] | 9.558e-08 | +0.220 | +0.476 |

## Fusion latency

- Latency metrics are recorded in `summary.json` under `latency_ms` (mean and p95 ms/query).

## Correlated slice definition

- Query-level correlated score: `||mean(residuals)|| / mean(||residual_i||)`.
- Correlated if score ≥ `0.70`.
- Correlated queries on test: `287` / `420`.

## Interpretation

- This is a narrowed hypothesis regime, not a broad claim about general retrieval settings.
- Any synthetic win is treated as directional evidence only; it does not close the headline Kalman-vs-mean claim.
