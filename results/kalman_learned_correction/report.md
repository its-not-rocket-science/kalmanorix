# Kalman + Learned Correction Benchmark

Can task-aware learned correction recover Kalman fusion quality without abandoning precision weighting?

**Answer:** Yes in this benchmark: Kalman + learned correction improves MRR while retaining Kalman precision as the anchor.

## Retrieval metrics (test split)

| Method | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| mean | 0.6136 | 0.8182 | 0.7173 |
| kalman_precision | 0.6364 | 0.8318 | 0.7350 |
| learned_linear_combiner | 0.6091 | 0.8318 | 0.7131 |
| kalman_learned_correction | 0.6682 | 0.8500 | 0.7599 |
