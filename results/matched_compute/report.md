# Matched Compute Report

## Current Evidence State
- Artifact path: `results/matched_compute/`.
- State: **Completed artifact (real run summary committed)**.
- Interpretation boundary: fairness checks pass for training parity, but overall superiority claims remain guarded.

## Objective
- Evaluate specialists vs monolith under explicit training/inference compute parity constraints with guarded interpretation.

## Reproducible setup
- Benchmark: `matched_compute_specialists_vs_monolith`.
- Deterministic controls: `seed=7`, `samples_per_domain=1200`, `test_size=600`.
- Domain set: `finance`, `legal`, `medical`, `science`.
- Training parity rule: specialists/monolith training compute ratio must remain within `±0.01`.

## Outcome slots
- **positive:** training compute parity check passes (`ratio=1.0000`, tolerance `0.0100`).
- **null:** no quality delta is observed in this run (`Acc@1` and `MRR` are tied).
- **inconclusive:** no end-to-end superiority claim is justified because inference compute is materially asymmetric.
- **regression:** routed specialist variants show inference FLOPs proxy above monolith (2.20x to 4.12x).

## Rule-based verdicts
- `quality_delta_rule`: **null**
- `compute_parity_rule`: **positive**
- `end_to_end_efficiency_rule`: **regression**
- `overall`: **inconclusive**

Decision logic:
1. Mark quality as `null` when headline ranking metrics are tied within numerical tolerance.
2. Mark compute parity as `positive` when explicit parity checks pass.
3. Mark efficiency as `regression` when specialist inference proxy exceeds `2.0x` monolith.
4. Mark overall as `inconclusive` when mixed labels include both guardrail passes and regressions.

## Key metrics

| Strategy | Acc@1 | MRR | Train Budget Proxy | Inference FLOPs Proxy (mean) | Inference ratio vs monolith |
|---|---:|---:|---:|---:|---:|
| monolith_baseline | 1.0000 | 1.0000 | 4246732800 | 442368.0 | 1.000x |
| specialists_all_routing | 1.0000 | 1.0000 | 4246732800 | 1822556.0 | 4.120x |
| specialists_semantic_routing | 1.0000 | 1.0000 | 4246732800 | 973209.0 | 2.200x |
| specialists_kalman_fusion | 1.0000 | 1.0000 | 4246732800 | 1017445.0 | 2.300x |

## Guarded findings scaffold

## Demonstrated findings

- Positive result: training compute parity is demonstrated for specialists versus monolith (ratio=1.0000, tolerance=0.0100).
- Regression: specialist routing variants require materially higher inference FLOPs proxy than monolith in this benchmark slice.

## Unresolved findings

- Null result: no statistically reliable quality difference is demonstrated for specialists versus monolith on Acc@1 and MRR in this run (exact tie).
- Inconclusive result: end-to-end superiority is not demonstrated because quality is tied while inference cost regresses.

## Threats to validity

- Query sets may under-represent long-tail or adversarial cases.
- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.
- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.

## Benchmark limitations

- Results depend on the provided benchmark artifacts and should be treated as conditional evidence.
- Inference FLOPs proxies are model- and implementation-dependent, not wall-clock latency measurements.

## Recommended next experiments

- Add a compute-matched inference regime (e.g., constrained specialist invocation budget) before revisiting superiority claims.
- Expand benchmark difficulty beyond linearly separable synthetic-like domains to avoid ceiling effects.
- Replicate with stronger quality metrics and paired statistical tests over harder query slices.
