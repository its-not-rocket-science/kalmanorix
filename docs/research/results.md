# Results

This page reports only what is currently supported by committed artifacts.

## Current Evidence State

- **Canonical artifact paths:** `results/canonical_benchmark/` (v1) and `results/canonical_benchmark_v2/` (stronger protocol target).
- **State:** v1 and v2 artifacts are committed (`summary.json`, `report.md`, and runner-level JSONs for v2).
- **What this means:** canonical benchmark machinery exists and has produced evidence; benchmark closure has not happened for the unresolved quality claims.

## Demonstrated vs Planned

### Demonstrated
- **Routing efficiency:** semantic routing shows substantial FLOPs reductions in current benchmark artifacts.

### Planned / Not yet demonstrated
- Kalman fusion quality improvement over mean fusion (**not demonstrated in the latest canonical artifact**).
- Specialists-vs-monolith quality advantage at matched compute.
- OOD robustness gains from uncertainty-weighted fusion.

---

### Canonical Benchmark Artifact
- `results/canonical_benchmark/summary.json`
- `results/canonical_benchmark/report.md`
- `results/canonical_benchmark_v2/summary.json`
- `results/canonical_benchmark_v2/report.md`
- `results/canonical_benchmark_v2/runner_summary.json`
- `results/canonical_benchmark_v2/runner_details.json`
- `results/canonical_benchmark_v2/README.md` (regeneration command)

Canonical v2 is produced with a two-step but fully reproducible command sequence:

```bash
PYTHONPATH=src python scripts/build_mixed_benchmark.py \
  --output-dir benchmarks/mixed_beir_v1.1.0 \
  --seed 1337 \
  --max-candidates 80 \
  --cross-domain-negative-ratio 0.45 \
  --max-queries-per-domain 900 \
  --max-test-queries-per-domain 180

PYTHONPATH=src python experiments/run_canonical_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.1.0/mixed_benchmark.json \
  --split test \
  --max-queries 600 \
  --output-dir results/canonical_benchmark_v2
```

The canonical artifact includes MeanFuser, KalmanorixFuser, hard-routing, and all-routing+mean baselines (plus LearnedGateFuser only when a two-specialist setup is used), with paired Kalman-vs-mean testing and confidence intervals for quality/latency/FLOPs proxy metrics. v2 additionally tracks nDCG@5, MRR@5, Recall@1, and top-1 success.

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

- The latest canonical run (`results/canonical_benchmark/report.md`) reports **no statistically significant nDCG@10 improvement** for Kalman vs mean under the configured setup; do not generalize beyond that benchmark configuration.

### 3) Fused specialists outperform monolith at equal compute
**Evidence status:** **Unresolved.**

- No completed matched-compute benchmark report is currently committed.

### 4) Uncertainty weighting improves OOD robustness
**Evidence status:** **Unresolved.**

- No completed OOD benchmark artifact currently supports this claim.

## Next empirical phase scaffold (post-canonical closure)

The repository now includes stable scaffold directories for the next three unresolved tracks:
- `results/matched_compute/`
- `results/uncertainty_ablation/`
- `results/ood_robustness/`

Each scaffold includes `summary_template.json` and `report_template.md` with explicit outcome slots for:
- `positive`
- `null`
- `inconclusive`
- `regression`

These templates are structural only and must be replaced with real run artifacts before any claim status is upgraded.

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
