# Routing Evaluation

## Reproducibility command

```bash
kalmanorix-eval-routing --dataset datasets/routing_eval/small_routing_eval_v1.json --output results/routing_eval/small_routing_eval_v1_report.json --markdown-output results/routing_eval/small_routing_eval_v1_report.md --mode semantic --semantic-threshold 0.7 --semantic-thresholds 0.5,0.6,0.7,0.8 --quality-tolerance 0.0
```

This directory uses `summary.json` as the machine-readable index and `report.md` (when present) for narrative interpretation.
