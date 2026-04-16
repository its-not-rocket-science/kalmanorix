# Experiments

This document defines the experiment policy after the April 2026 audit: **separate demonstrated evidence from planned work, and label synthetic results explicitly**.

## Current Evidence State

- **Canonical artifact paths:** `results/canonical_benchmark/` (v1 provenance) and `results/canonical_benchmark_v2/` (current decision artifact).
- **State:** **Real artifacts committed** (`summary.json`, `report.md`, and v2 runner summaries).
- **Status signal:** benchmark pipelines are active and reproducible; unresolved quality claims remain unresolved under current evidence thresholds.

## Kalman improvement experiment tracks (current map)

This is the compact map of implemented Kalman-improvement tracks and how to interpret their current outputs.

| Track | Artifacts | Current interpretation |
|---|---|---|
| Canonical benchmark v2 | `results/canonical_benchmark_v2/summary.json`, `results/canonical_benchmark_v2/report.md` | Decision remains `inconclusive_underpowered`; do not claim Kalman>Mean quality superiority from this run. |
| Uncertainty calibration | `results/uncertainty_calibration/summary.json`, `results/uncertainty_calibration/report.md` | Powered calibration setup now exists, but downstream delta change is currently null (`0.0` validation/test). |
| Uncertainty ablation | `results/uncertainty_ablation/summary.json`, `results/uncertainty_ablation/report.md` | Calibration proxies improve for some methods, while retrieval metrics are largely unchanged in this setup. |
| Covariance ablation | `results/kalman_covariance_ablation_v2/summary.json`, `results/kalman_covariance_ablation_v2/report.md` | Richer covariance families are implemented and tested but not justified by practical gains here. |
| Correlation-aware fusion | `results/correlation_aware_fusion/summary.json`, `results/correlation_aware_fusion/report.md` | Small positive signal vs baseline Kalman on correlated split; treat as exploratory pending replication. |
| Latency optimization | `results/kalman_latency_optimization/summary.json`, `results/kalman_latency_optimization/report.md` | Legacy Kalman path is measurably faster after optimization, but canonical latency-ratio acceptance is still not met. |

## 1) Demonstrated Evidence (as of April 9, 2026)

### Demonstrated claim
- **Claim:** Semantic routing can reduce compute by selecting fewer specialists.
- **Evidence status:** **Supported in current benchmark artifacts**.

### Evidence artifacts
- `results/efficiency_semantic_routing.json`
- `results/semantic_routing_efficiency_report.md`
- `experiments/benchmark_efficiency.py`

### Scope limitation
This evidence demonstrates compute behavior in the tested benchmark setup; it does **not** by itself prove quality gains or universal real-world latency improvements.

---

## Canonical benchmark decision criteria (Kalman vs Mean)

The canonical benchmark must end with a rule-based verdict for **KalmanorixFuser vs MeanFuser**:

- **primary metric:** `nDCG@10` paired mean difference (`Kalman - Mean`)
- **minimum effect size:** `>= 0.02`
- **adjusted p-value threshold:** Holm-adjusted `p <= 0.05`
- **latency/FLOPs trade-off constraint:** `latency_ratio(Kalman/Mean) <= 1.5` and `flops_ratio(Kalman/Mean) <= 1.1`

Verdict labels are defined as:

- **supported:** all four checks pass
- **unsupported:** `nDCG@10` delta is non-positive (`<= 0`) **and** Holm-adjusted p-value is significant (`<= 0.05`)
- **inconclusive:** any other outcome

This framework is intentionally conservative and prefers **inconclusive** over forced directional claims when evidence is weak or trade-offs fail.

---

## 2) Pending Core Claims (planned, not demonstrated)

### Claim: Kalman fusion > mean fusion on retrieval quality
**Evidence status:** **Unresolved** (no final statistical artifact proving improvement).

### Claim: fused specialists > monolith at equal compute
**Evidence status:** **Unresolved** (no completed matched-compute benchmark report).

### Claim: uncertainty weighting improves OOD robustness
**Evidence status:** **Unresolved** (no completed OOD report committed).

---

## 3) Synthetic Results (explicitly labeled)

The following are **Synthetic / Debug only** and are not valid as headline evidence:

- `experiments/benchmark_fusion_methods.py`
- `experiments/validate_fusion.py --debug-synthetic`
- `experiments/mixed_domain_eval.py --debug-synthetic`

Use these to debug pipelines and catch regressions; do not use them to support core scientific claims.

---

## 4) Primary Evaluation Path (real benchmark path)

### Benchmark objective
Evaluate Kalman fusion against mean fusion on a mixed-domain benchmark with explicit relevance labels.

### Build benchmark artifacts
```bash
PYTHONPATH=src python scripts/build_mixed_benchmark.py \
  --output-dir benchmarks/mixed_beir_v1.1.0 \
  --seed 1337 \
  --max-candidates 80 \
  --cross-domain-negative-ratio 0.45 \
  --max-queries-per-domain 900 \
  --max-test-queries-per-domain 180
```

### Run primary benchmark
```bash
PYTHONPATH=src python experiments/run_canonical_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.1.0/mixed_benchmark.json \
  --split test \
  --max-queries 600 \
  --output-dir results/canonical_benchmark_v2
```

### Compatibility entrypoints (real mode by default)
```bash
python experiments/validate_fusion.py
python experiments/mixed_domain_eval.py
```

---

## 5) Threats to Validity (must be reported with results)

1. **Benchmark representativeness:** selected domains and queries may not match production distributions.
2. **Statistical stability:** limited query counts can produce unstable estimates and p-values.
3. **Compute parity validity:** specialists-vs-monolith comparisons can be biased if training/inference budgets differ.
4. **Calibration quality:** uncertainty quality directly affects Kalman weighting behavior.
5. **Routing sensitivity:** fixed thresholds may under-select or over-select specialists depending on query mix.

---

## What Kalman improvements changed empirically (current)

- **Improved implementation and instrumentation:** uncertainty calibration now has explicit power checks; covariance variants, correlation-aware fusion, and latency optimizations all have committed artifact trails.
- **Unchanged/null outcomes where observed:** canonical Kalman-vs-Mean remains unresolved; uncertainty calibration and uncertainty ablation currently do not produce a robust downstream quality lift in committed runs.
- **Open questions:** can small correlation-aware gains replicate at larger sample sizes, and can additional latency optimization meet canonical latency thresholds while preserving quality metrics.
