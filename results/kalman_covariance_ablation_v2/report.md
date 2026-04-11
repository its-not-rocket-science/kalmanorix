# Kalman Covariance Ablation

Benchmark version: `kalman_covariance_ablation_v2_enlarged`

Do richer uncertainty families justify complexity?

**Answer:** Richer covariance is not worth it in this setup: no global or bucket-level gain cleared practical thresholds.

## Retrieval Metrics

| Method | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| mean_fusion | 0.1487 | 0.3762 | 0.2467 |
| scalar_kalman | 0.1725 | 0.4244 | 0.2806 |
| diagonal_kalman | 0.1731 | 0.4256 | 0.2801 |
| structured_kalman | 0.1750 | 0.4213 | 0.2810 |

## Per-bucket Recall@1

| Method | All | High Disagreement | Multi-domain | Uncertainty-skewed |
| --- | ---: | ---: | ---: | ---: |
| mean_fusion | 0.1487 | 0.1429 | 0.1626 | 0.1406 |
| scalar_kalman | 0.1725 | 0.1649 | 0.1871 | 0.1375 |
| diagonal_kalman | 0.1731 | 0.1656 | 0.1888 | 0.1437 |
| structured_kalman | 0.1750 | 0.1675 | 0.1941 | 0.1469 |

## Efficiency Trade-offs

| Method | Total latency (ms) | Latency/query (ms) | Peak memory (KiB) |
| --- | ---: | ---: | ---: |
| mean_fusion | 78.57 | 0.0491 | 1043.0 |
| scalar_kalman | 427.37 | 0.2671 | 1054.4 |
| diagonal_kalman | 373.76 | 0.2336 | 1040.1 |
| structured_kalman | 788.29 | 0.4927 | 1049.8 |
