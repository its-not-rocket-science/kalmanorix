# Matched Compute Report

## Current Evidence State
- Artifact path: `results/matched_compute/`.
- State: **Scaffold only (pending run)**.
- Interpretation boundary: no outcome is demonstrated until real artifacts are generated.

## Objective
- Evaluate specialists vs monolith under explicit training/inference compute parity constraints with guarded interpretation.

## Outcome slots (to be completed with evidence)
- **positive:** [pending evidence]
- **null:** [pending evidence]
- **inconclusive:** [pending evidence]
- **regression:** [pending evidence]

## Guarded findings scaffold

## Demonstrated findings

- No demonstrated directional effect is established by the current statistical evidence.

## Unresolved findings

- No unresolved pairwise findings were detected in this output.

## Threats to validity

- Query sets may under-represent long-tail or adversarial cases.
- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.
- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.

## Benchmark limitations

- Results depend on the provided benchmark artifacts and should be treated as conditional evidence.
- Latency and memory proxies are environment-sensitive and may shift under different hardware/runtime settings.

## Recommended next experiments

- Increase held-out query count and rebalance domains before promoting unresolved findings.
- Add stress tests for distribution shift and low-resource domains to challenge demonstrated effects.
- Replicate on an independent benchmark slice with pre-registered metrics and hypotheses.

## Artifact checklist
- [ ] `summary.json` updated with real metrics and statistical outputs.
- [ ] `report.md` updated with benchmark-specific findings.
- [ ] Synthetic/debug runs labeled and separated from headline evidence.