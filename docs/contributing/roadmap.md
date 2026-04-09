# Kalmanorix Roadmap (Contributor View)

**Last updated:** April 9, 2026

This page is the contributor-facing mirror of `ROADMAP.md` and uses the same evidence policy: do not present planned outcomes as demonstrated results.

## Demonstrated vs Planned

### Demonstrated
- Routing efficiency benchmark artifacts are present and reproducible from repository scripts.
- Core infrastructure and stabilization fixes are implemented.

### Planned / Pending Demonstration
- Kalman quality advantage over mean fusion.
- Specialists-vs-monolith advantage at matched compute.
- OOD robustness advantage from uncertainty weighting.

## Evidence Status for Core Claims

### Claim: Routing reduces compute cost
**Evidence status:** **Supported.**
- Backed by committed efficiency artifacts in `results/` and benchmark scripts in `experiments/`.

### Claim: Kalman improves retrieval quality vs mean
**Evidence status:** **Unresolved.**
- No final statistical report in the repo currently supports this as a demonstrated claim.

### Claim: Specialists beat monolith at equal compute
**Evidence status:** **Unresolved.**
- Planned milestone; final reproducible comparison artifact is pending.

### Claim: Uncertainty weighting improves OOD robustness
**Evidence status:** **Unresolved.**
- Planned milestone; completed OOD benchmark artifact is pending.

## Synthetic Results Labeling Rule

If an experiment uses toy/debug/synthetic data, label results as **Synthetic (not headline evidence)**.

Examples:
- `experiments/benchmark_fusion_methods.py`
- `experiments/validate_fusion.py --debug-synthetic`
- `experiments/mixed_domain_eval.py --debug-synthetic`

## Threats to Validity Checklist (must be stated in reports)

- Dataset representativeness and domain coverage.
- Statistical power and confidence intervals.
- Compute-parity fairness (for specialists vs monolith).
- Sensitivity to routing thresholds/centroid construction.
- Potential leakage between train/validation/test resources.
