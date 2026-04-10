# Kalman Covariance Ablation

Do richer uncertainty families justify complexity?

**Answer:** Richer covariance did not clearly beat simpler baselines in this benchmark setting.

## Retrieval Metrics

| Method | Recall@1 | Recall@5 | MRR@10 |
| --- | ---: | ---: | ---: |
| mean_fusion | 0.1933 | 0.5300 | 0.3294 |
| scalar_kalman | 0.2100 | 0.5633 | 0.3526 |
| diagonal_kalman | 0.2033 | 0.5667 | 0.3493 |
| structured_kalman | 0.1867 | 0.5733 | 0.3441 |
