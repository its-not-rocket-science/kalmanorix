# OOD Robustness Report

## Current Evidence State
- Artifact path: `results/ood_robustness/`.
- State: **Completed artifact (guarded evidence synthesis committed)**.
- Interpretation boundary: reproducibility and failure modes are explicit, but directional superiority is not claimed.

## Objective
- Evaluate robustness under distribution shift and distinguish supported effects from null/inconclusive/regression outcomes.

## Explicit OOD definition (reproducible)
- OOD query rule: queries from an unseen domain are labeled with `true_doc_id=-1`.
- Corpus rule: the OOD domain is excluded from the candidate document corpus.
- Construction APIs: `create_ood_test_set` and `create_synthetic_ood_test_set` in `src/kalmanorix/ood_datasets.py`.
- Reference reproducibility controls: seen domains `medical, legal`; OOD domain `tech`; OOD proportion `0.5`; seed `42`.

## Outcome slots
- **positive:** OOD construction and reproducibility constraints are explicit and tested.
- **null:** no committed evidence currently shows a reliable OOD quality gain from uncertainty weighting.
- **inconclusive:** dedicated paired significance outputs for OOD head-to-head quality remain unavailable in this track.
- **regression:** strict OOD queries (`true_doc_id=-1`) still expose an overconfident retrieval risk because no abstention gate is committed here.

## Rule-based verdicts
- `ood_definition_reproducibility_rule`: **positive**
- `ood_quality_gain_rule`: **null**
- `ood_uncertainty_weighting_advantage_rule`: **inconclusive**
- `ood_failure_mode_rule`: **regression**
- `overall`: **inconclusive**

Decision logic:
1. `positive` when OOD split semantics are explicit, parameterized, and validated.
2. `null` when existing uncertainty artifacts show no downstream quality lift.
3. `inconclusive` when no direct paired OOD significance table is present.
4. `regression` when OOD queries require rejection behavior but no reject/abstain rule is committed.

## Guarded findings scaffold

## Demonstrated findings

- Positive result: OOD construction is reproducible and testable with explicit controls (seen domains, OOD domain, OOD ratio, and seed).
- Regression: strict OOD handling currently lacks a committed abstention decision rule, leaving an overconfident-retrieval failure mode.

## Unresolved findings

- Null result: current committed uncertainty-calibration and uncertainty-ablation outputs do not show a reliable downstream quality uplift attributable to uncertainty weighting.
- Inconclusive result: direct paired OOD significance evidence for Kalman-vs-mean remains pending in this track.

## Threats to validity

- Query sets may under-represent long-tail or adversarial cases.
- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.
- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.

## Benchmark limitations

- This artifact is a guarded synthesis anchored to committed OOD protocol and uncertainty artifacts; it is not a new large-scale OOD rerun.
- Robust OOD conclusions still require a dedicated paired benchmark run with significance testing and reject-option metrics.

## Recommended next experiments

- Add an explicit abstain/reject policy and evaluate AUROC/FPR@TPR for OOD detection.
- Run a paired OOD benchmark comparing Kalman-vs-mean under identical routing and compute constraints.
- Extend OOD domains beyond a single held-out domain and pre-register success criteria.
