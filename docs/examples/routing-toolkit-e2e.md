# Routing Toolkit End-to-End (CLI)

This example shows a complete routing-evaluation workflow using the committed toy dataset and the `kalmanorix-eval-routing` CLI.

## 1) Run routing evaluation on the small dataset

```bash
kalmanorix-eval-routing \
  --dataset datasets/routing_eval/small_routing_eval_v1.json \
  --output results/routing_eval/small_routing_eval_v1_report.json \
  --markdown-output results/routing_eval/small_routing_eval_v1_report.md \
  --mode semantic \
  --semantic-threshold 0.7 \
  --semantic-thresholds 0.5,0.6,0.7,0.8 \
  --quality-tolerance 0.0
```

This writes:

- `results/routing_eval/small_routing_eval_v1_report.json` (full machine-readable artifact)
- `results/routing_eval/small_routing_eval_v1_report.md` (compact human-readable summary)

## 2) What to inspect first

The markdown report is intended as the first stop for interpretation:

- **Single-run summary** for precision/recall/F1 plus efficiency deltas.
- **Outcome split** that lists quality-preserving wins, compute-only wins, and failure modes.
- **Threshold robustness** to see metric stability as thresholds move.
- **Per-query table** for transparent failure/win inspection.

## 3) Interpretation guardrails

Treat compute wins and quality failures symmetrically:

- A routing run can reduce FLOPs while still showing quality loss on some queries.
- Use `quality_preserving_routing_wins`, `compute_only_wins`, and `failure_modes` together.
- Avoid quality-improvement claims unless supported by committed quality artifacts.
