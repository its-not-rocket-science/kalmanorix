# Kalman-vs-Mean Evidence Dashboard

**Overall status:** `yellow`

## Traffic-light legend
- `green` = supported
- `yellow` = unresolved
- `red` = unsupported in tested regime

## Compact summary

| Field | Traffic light | Evidence (artifact-sourced) |
|---|---|---|
| Canonical benchmark verdict | yellow | `inconclusive_underpowered` from `results/canonical_benchmark_v2/summary.json` |
| Confirmatory slice verdict | yellow | `not_run_in_committed_canonical_artifact` from `results/canonical_benchmark_v2/summary.json` |
| Baseline comparisons | yellow | Kalman nDCG@10 0.5486, Mean 0.4537, Δ 0.0949, adjusted p=1.0000 |
| Uncertainty ablation result | yellow | Partially: calibration differences are visible, but retrieval gains from better uncertainty estimation are limited in this setup; constant uncertainty remains competitive. |
| Latency ratio | red | Kalman/Mean=2.055 vs threshold=1.500 |
| Replication status | yellow | `replicated_same_verdict` (canonical verdict `inconclusive_underpowered`, rerun verdict `inconclusive_underpowered`) |

## Source artifacts
- `results/canonical_benchmark_v2/summary.json`
- `results/uncertainty_ablation/summary.json`
- `results/kalman_latency_optimization/canonical/summary.json`

All fields above are extracted from committed JSON artifacts via `scripts/build_kalman_evidence_dashboard.py`.
