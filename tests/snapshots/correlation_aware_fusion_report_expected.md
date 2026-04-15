# Correlation-Aware Fusion Benchmark

Question: does correlation adjustment improve Kalman fusion on a strengthened correlated-expert test split?

**Answer:** Null result: correlation-aware adjustments did not materially improve baseline Kalman (best ΔMRR@10=0.0011).

## Test metrics (strengthened split)

| Method | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| MeanFuser | 0.4778 | 0.9000 | 0.6512 |
| KalmanorixFuser | 0.5111 | 0.9111 | 0.6775 |
| CorrelationAwareKalmanFuser (covariance_inflation) | 0.5111 | 0.9111 | 0.6786 |
| CorrelationAwareKalmanFuser (effective_sample_size) | 0.5111 | 0.9111 | 0.6775 |

## Per-bucket metrics

### MeanFuser

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.5556 | 0.8667 | 0.6782 |
| low_correlation | 0.4000 | 0.9333 | 0.6241 |

### KalmanorixFuser

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.5556 | 0.9333 | 0.6875 |
| low_correlation | 0.4667 | 0.8889 | 0.6675 |

### CorrelationAwareKalmanFuser (covariance_inflation)

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.5556 | 0.9333 | 0.6875 |
| low_correlation | 0.4667 | 0.8889 | 0.6697 |

### CorrelationAwareKalmanFuser (effective_sample_size)

| Bucket | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| high_correlation | 0.5556 | 0.9333 | 0.6875 |
| low_correlation | 0.4667 | 0.8889 | 0.6675 |

## Notes

- Correlation profile was estimated only from validation residuals.
- Test split is strengthened by including a high-correlation half (ρ=0.85).
- Null/negative outcomes are reported directly; no optimistic retuning on test.
