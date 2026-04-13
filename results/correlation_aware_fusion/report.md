# Correlation-Aware Fusion Benchmark

Question: does correlation adjustment improve Kalman fusion on a strengthened correlated-expert test split?

**Answer:** Correlation-aware Kalman improved over baseline Kalman (best: CorrelationAwareKalmanFuser (covariance_inflation), ΔMRR@10=0.0037).

## Test metrics (strengthened split)

| Method | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| MeanFuser | 0.3500 | 0.6833 | 0.4929 |
| KalmanorixFuser | 0.3929 | 0.7214 | 0.5318 |
| CorrelationAwareKalmanFuser (covariance_inflation) | 0.3952 | 0.7262 | 0.5356 |
| CorrelationAwareKalmanFuser (effective_sample_size) | 0.3929 | 0.7214 | 0.5318 |

## Per-bucket metrics

### MeanFuser

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.2810 | 0.6238 | 0.4205 |
| low_correlation | 0.4190 | 0.7429 | 0.5652 |

### KalmanorixFuser

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.3238 | 0.6476 | 0.4637 |
| low_correlation | 0.4619 | 0.7952 | 0.6000 |

### CorrelationAwareKalmanFuser (covariance_inflation)

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.3238 | 0.6476 | 0.4645 |
| low_correlation | 0.4667 | 0.8048 | 0.6066 |

### CorrelationAwareKalmanFuser (effective_sample_size)

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.3238 | 0.6476 | 0.4637 |
| low_correlation | 0.4619 | 0.7952 | 0.6000 |

## Notes

- Correlation profile was estimated only from validation residuals.
- Test split is strengthened by including a high-correlation half (ρ=0.85).
- Null/negative outcomes are reported directly; no optimistic retuning on test.
