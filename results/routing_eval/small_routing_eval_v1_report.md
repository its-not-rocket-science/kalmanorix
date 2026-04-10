# Routing Evaluation Report

## Single-run summary
- Routing precision: **0.639**
- Routing recall: **0.833**
- Routing F1: **0.694**
- Avg FLOPs savings fraction: **0.444**
- Avg latency delta (all - routed, ms): **6.083**

## Outcome split (wins and failures)
- Quality-preserving routing wins: **4** (q_tech_fix_bug, q_recipe_pasta, q_multi_doc_query, q_ambiguous_help)
- Compute-only wins (quality loss tolerated by config): **1** (q_chargeback_fraud)
- Failure modes: **1** (q_longtail_unknown)
- Failure breakdown: quality_loss=1, zero_recall=0

## Threshold robustness
- Best semantic threshold by F1: **0.5**
- F1 range across sweep: **0.167**
- Precision range across sweep: **0.056**
- Recall range across sweep: **0.417**
- FLOPs savings range across sweep: **0.333**

### Sweep table
| Threshold | Precision | Recall | F1 | FLOPs savings | Latency delta ms |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.694 | 1.000 | 0.778 | 0.333 | 4.417 |
| 0.60 | 0.694 | 1.000 | 0.778 | 0.333 | 4.417 |
| 0.70 | 0.639 | 0.833 | 0.694 | 0.444 | 6.083 |
| 0.80 | 0.667 | 0.583 | 0.611 | 0.667 | 9.417 |

### Per-query outcomes
| Query | Selected domains | Precision | Recall | F1 | FLOPs savings | Latency delta ms | Quality delta | Category |
|---|---|---:|---:|---:|---:|---:|---:|---|
| q_tech_fix_bug | tech | 1.000 | 1.000 | 1.000 | 0.667 | 9.800 | 0.010 | quality_preserving_win |
| q_recipe_pasta | cook | 1.000 | 1.000 | 1.000 | 0.667 | 9.800 | 0.000 | quality_preserving_win |
| q_chargeback_fraud | tech | 0.000 | 0.000 | 0.000 | 0.667 | 9.800 | -0.030 | compute_only_win |
| q_multi_doc_query | cook, tech | 1.000 | 1.000 | 1.000 | 0.333 | 4.800 | 0.000 | quality_preserving_win |
| q_longtail_unknown | charge, cook, tech | 0.333 | 1.000 | 0.500 | 0.000 | -2.500 | -0.010 | failure_quality_loss |
| q_ambiguous_help | cook, tech | 0.500 | 1.000 | 0.667 | 0.333 | 4.800 | 0.000 | quality_preserving_win |
