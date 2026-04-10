# Experiments

This document defines the experiment policy after the April 2026 audit: **separate demonstrated evidence from planned work, and label synthetic results explicitly**.

## Current Evidence State

- **Canonical artifact path:** `results/canonical_benchmark/`
- **State:** **Real artifact committed** (`summary.json`, `report.md`).
- **Status signal:** this confirms the benchmark pipeline is active; it does not close the unresolved quality claims.

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
python scripts/build_mixed_benchmark.py
```

### Run primary benchmark
```bash
python experiments/run_real_mixed_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet \
  --split test \
  --max-queries 150 \
  --output results/real_mixed_benchmark/real_benchmark_summary.json
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
