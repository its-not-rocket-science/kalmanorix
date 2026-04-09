# Results

This page reports only what is currently supported by committed artifacts.

## Demonstrated vs Planned

### Demonstrated
- **Routing efficiency:** semantic routing shows substantial FLOPs reductions in current benchmark artifacts.

### Planned / Not yet demonstrated
- Kalman fusion quality improvement over mean fusion.
- Specialists-vs-monolith quality advantage at matched compute.
- OOD robustness gains from uncertainty-weighted fusion.

---

## Core Claim Evidence Status

### 1) Semantic routing improves compute efficiency
**Evidence status:** **Supported (artifact-backed).**

Evidence sources:
- `results/efficiency_semantic_routing.json`
- `results/semantic_routing_efficiency_report.md`

Interpretation:
- Current artifacts report roughly 65% average FLOPs reduction in the tested setup.
- This supports the routing-efficiency claim for those benchmark conditions.

### 2) Kalman fusion improves retrieval quality over mean fusion
**Evidence status:** **Unresolved.**

- No final statistical artifact in the repository currently establishes a robust improvement claim.

### 3) Fused specialists outperform monolith at equal compute
**Evidence status:** **Unresolved.**

- No completed matched-compute benchmark report is currently committed.

### 4) Uncertainty weighting improves OOD robustness
**Evidence status:** **Unresolved.**

- No completed OOD benchmark artifact currently supports this claim.

---

## Synthetic Results (for debugging only)

Any outputs from toy/debug pipelines must be cited as **Synthetic (not headline evidence)**.

Examples include:
- `experiments/benchmark_fusion_methods.py`
- `experiments/validate_fusion.py --debug-synthetic`
- `experiments/mixed_domain_eval.py --debug-synthetic`

Synthetic results are useful for regression detection and implementation checks, but not for confirming the core hypotheses.

---

## Threats to Validity

1. **Limited domain/query diversity** in current demonstrated efficiency runs.
2. **Benchmark-to-production gap** in latency and system overhead behavior.
3. **Threshold/centroid dependence** in routing outcomes.
4. **Uncertainty estimation quality** may constrain Kalman fusion performance.
5. **Incomplete hypothesis coverage** until pending benchmarks are finalized.
