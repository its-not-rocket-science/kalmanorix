# Kalman + Learned Correction Benchmark

Can task-aware learned correction recover Kalman fusion quality without abandoning precision weighting?

**Answer:** Yes in this benchmark: Kalman + learned correction improves MRR while retaining Kalman precision as the anchor.

## Retrieval metrics (test split)

| Method | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| mean | 0.4909 | 0.7864 | 0.6245 |
| kalman_precision | 0.5136 | 0.8136 | 0.6492 |
| learned_linear_combiner | 0.5364 | 0.8000 | 0.6565 |
| kalman_learned_correction | 0.6000 | 0.8364 | 0.7077 |
