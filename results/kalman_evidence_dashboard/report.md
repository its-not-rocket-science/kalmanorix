# Kalman-vs-Mean Evidence Dashboard

**Claim-ready support:** `no`

## Traffic-light legend
- `green` = supported
- `yellow` = unresolved
- `red` = unsupported in tested regime

## Compact summary

| Field | Traffic light | Evidence (artifact-sourced) |
|---|---|---|
| Canonical v3 benchmark status | yellow | `placeholder_pending_run` from `results/canonical_benchmark_v3/summary.json` |
| Canonical v3 verdict | yellow | `not_available_placeholder_pending_run` from `results/canonical_benchmark_v3/summary.json` |
| Confirmatory slice verdict | red | `missing_confirmatory_evidence` from `results/canonical_benchmark_v3/summary.json` |
| Kalman vs mean | yellow | `inconclusive_underpowered` from `results/kalman_latency_optimization/canonical/summary.json` |
| Kalman vs weighted mean | yellow | `inconclusive_underpowered` from `results/kalman_latency_optimization/canonical/summary.json` |
| Kalman vs router only top1 | yellow | `inconclusive_underpowered` from `results/kalman_latency_optimization/canonical/summary.json` |
| Uncertainty ablation result | yellow | Partially: calibration differences are visible, but retrieval gains from better uncertainty estimation are limited in this setup; constant uncertainty remains competitive. |
| Latency gate status | green | Kalman/Mean=0.924 vs threshold=1.500 |
| Replication status | yellow | `replicated_same_verdict` (canonical verdict=`inconclusive_underpowered`, replication verdicts=`['inconclusive_underpowered']`) |

## Why the repo can / cannot currently claim Kalman beats mean
- canonical v3 benchmark status is `placeholder_pending_run`.
- canonical v3 verdict is `not_available_placeholder_pending_run`.
- confirmatory slice verdict is `missing_confirmatory_evidence`.
- kalman_vs_mean verdict is `inconclusive_underpowered`.
- kalman_vs_weighted_mean verdict is `inconclusive_underpowered`.
- kalman_vs_router_only_top1 verdict is `inconclusive_underpowered`.
- replication status is `replicated_same_verdict`.

## Source artifacts
- `results/canonical_benchmark_v3/summary.json`
- `results/kalman_latency_optimization/canonical/summary.json`
- `results/uncertainty_ablation/summary.json`

All fields above are extracted from committed JSON artifacts via `scripts/build_kalman_evidence_dashboard.py`.
