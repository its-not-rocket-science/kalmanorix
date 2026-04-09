# Uncertainty Ablation Report

## Scope
- Methods evaluated uniformly: constant, keyword-based, centroid-distance, embedding-norm, stochastic-forward.
- Datasets include one real toy-mixed split and one clearly-labeled synthetic split.

## Results Snapshot
- dataset=toy_mixed , method=constant_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.333, brier=0.111
- dataset=toy_mixed , method=keyword_based_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.204, brier=0.074
- dataset=toy_mixed , method=centroid_distance_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.130, brier=0.017
- dataset=toy_mixed , method=embedding_norm_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.145, brier=0.021
- dataset=toy_mixed , method=stochastic_forward_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.091, brier=0.008
- dataset=synthetic_shifted_queries (synthetic), method=constant_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.333, brier=0.111
- dataset=synthetic_shifted_queries (synthetic), method=keyword_based_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.091, brier=0.008
- dataset=synthetic_shifted_queries (synthetic), method=centroid_distance_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.121, brier=0.015
- dataset=synthetic_shifted_queries (synthetic), method=embedding_norm_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.145, brier=0.021
- dataset=synthetic_shifted_queries (synthetic), method=stochastic_forward_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.091, brier=0.008

## Sensitivity to Mis-Specified Uncertainty Scaling
Each method was re-evaluated with sigma² scaled by {0.5, 1.0, 2.0, 4.0}.
- toy_mixed / constant_sigma2: recall@1 range=0.000, ECE range=0.467
- toy_mixed / keyword_based_sigma2: recall@1 range=0.000, ECE range=0.306
- toy_mixed / centroid_distance_sigma2: recall@1 range=0.000, ECE range=0.304
- toy_mixed / embedding_norm_sigma2: recall@1 range=0.000, ECE range=0.326
- toy_mixed / stochastic_forward_sigma2: recall@1 range=0.000, ECE range=0.238
- synthetic_shifted_queries / constant_sigma2: recall@1 range=0.250, ECE range=0.467
- synthetic_shifted_queries / keyword_based_sigma2: recall@1 range=0.000, ECE range=0.238
- synthetic_shifted_queries / centroid_distance_sigma2: recall@1 range=0.000, ECE range=0.291
- synthetic_shifted_queries / embedding_norm_sigma2: recall@1 range=0.000, ECE range=0.326
- synthetic_shifted_queries / stochastic_forward_sigma2: recall@1 range=0.000, ECE range=0.238

## Does better uncertainty estimation improve Kalman fusion enough to matter?
Partially: calibration differences are visible, but retrieval gains from better uncertainty estimation are limited in this setup; constant uncertainty remains competitive.

## Artifacts
- summary.json
- report.md
