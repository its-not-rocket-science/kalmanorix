# Uncertainty Ablation Report

## Scope
- Methods evaluated uniformly: constant, centroid-distance (current default), centroid+norm+peer (improved).
- Datasets include one real toy-mixed split and one clearly-labeled synthetic split.

## Results Snapshot
- dataset=toy_mixed , method=constant_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.333, brier=0.111
- dataset=toy_mixed , method=centroid_distance_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.126, brier=0.016
- dataset=toy_mixed , method=centroid_norm_peer_sigma2: recall@1=0.733, recall@5=1.000, mrr@10=0.856, ece=0.259, brier=0.068
- dataset=synthetic_shifted_queries (synthetic), method=constant_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.333, brier=0.111
- dataset=synthetic_shifted_queries (synthetic), method=centroid_distance_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.119, brier=0.014
- dataset=synthetic_shifted_queries (synthetic), method=centroid_norm_peer_sigma2: recall@1=0.250, recall@5=1.000, mrr@10=0.521, ece=0.238, brier=0.057

## Focused Delta vs Constant Baseline
- toy_mixed: default Δrecall@1=+0.000, default ΔECE=-0.208; improved Δrecall@1=+0.000, improved ΔECE=-0.075
- synthetic_shifted_queries: default Δrecall@1=+0.000, default ΔECE=-0.214; improved Δrecall@1=+0.000, improved ΔECE=-0.095

## Sensitivity to Mis-Specified Uncertainty Scaling
Each method was re-evaluated with sigma² scaled by {0.5, 1.0, 2.0, 4.0}.
- toy_mixed / constant_sigma2: recall@1 range=0.000, ECE range=0.467
- toy_mixed / centroid_distance_sigma2: recall@1 range=0.000, ECE range=0.298
- toy_mixed / centroid_norm_peer_sigma2: recall@1 range=0.000, ECE range=0.432
- synthetic_shifted_queries / constant_sigma2: recall@1 range=0.250, ECE range=0.467
- synthetic_shifted_queries / centroid_distance_sigma2: recall@1 range=0.000, ECE range=0.288
- synthetic_shifted_queries / centroid_norm_peer_sigma2: recall@1 range=0.000, ECE range=0.420

## Does better uncertainty estimation improve Kalman fusion enough to matter?
Partially: calibration differences are visible, but retrieval gains from better uncertainty estimation are limited in this setup; constant uncertainty remains competitive.

## Artifacts
- summary.json
- report.md
