# Kalmanorix: Specialist-Embedding Fusion Research Framework

> **Research status (as of April 9, 2026):**
> - ✅ Demonstrated: routing-related compute savings in controlled benchmark runs.
> - ⚠️ Not yet demonstrated: statistically significant quality gains of Kalman fusion over simple mean fusion.
> - ⚠️ Not yet demonstrated: specialists-vs-monolith quality advantage under matched compute.

Kalmanorix is an experimental framework for combining specialist embedding models with uncertainty-aware fusion and semantic routing.

## Demonstrated Results vs Planned Work

### Demonstrated (with committed artifacts)
- **Routing efficiency:** semantic routing can reduce FLOPs by selecting fewer specialists in benchmark runs (reported around 65% average reduction in current efficiency artifacts). See `results/efficiency_semantic_routing.json` and `results/semantic_routing_efficiency_report.md`.
- **Engineering stabilization:** alignment, covariance scaling, and fusion-path bug fixes are implemented and tested.

### Planned / In Progress
- **Kalman vs mean fusion quality:** pending reproducible statistical result showing improvement.
- **Specialists vs monolith (matched compute):** pending full benchmark artifact and analysis.
- **OOD robustness of uncertainty weighting:** pending completed benchmark report.

---

## Core Claims and Evidence Status

### Claim A: Semantic routing improves computational efficiency
**Evidence status:** **Supported (demonstrated in benchmark artifacts).**

- Evidence type: benchmark output JSON + report.
- Scope: tested benchmark configurations in repository artifacts.
- Caveat: current runs are limited in domain breadth and deployment realism.

### Claim B: Kalman fusion improves retrieval quality over mean fusion
**Evidence status:** **Unresolved (not demonstrated).**

- Current state: no published artifact in this repository that establishes a statistically significant improvement over mean fusion.
- Prior synthetic/debug experiments exist but are not sufficient for headline claims.

### Claim C: Fused specialists outperform a monolith at equal compute
**Evidence status:** **Unresolved (not demonstrated).**

- Current state: milestone is in progress; no final reproducible report is committed.

### Claim D: Uncertainty-weighted fusion is more robust OOD
**Evidence status:** **Unresolved (not demonstrated).**

- Current state: planned benchmark track exists; final supporting artifact not yet published.

---

## Synthetic Results Policy

Any numbers from toy/debug pipelines are **synthetic** and must be treated as development diagnostics only.

- Synthetic/debug entry points include:
  - `experiments/benchmark_fusion_methods.py`
  - `experiments/validate_fusion.py --debug-synthetic`
  - `experiments/mixed_domain_eval.py --debug-synthetic`
- Synthetic outputs are useful for regression checks, not for confirming the core scientific claims.

---

## Threats to Validity (Explicit)

1. **Benchmark representativeness:** current demonstrated efficiency evidence is from limited benchmark setups and may not generalize to production traffic.
2. **Routing–quality coupling:** compute savings do not by themselves establish quality preservation across all domains.
3. **Model heterogeneity:** many tests assume specific model families/embedding-space behavior.
4. **Threshold sensitivity:** routing outcomes depend strongly on threshold heuristics and centroid quality.
5. **Evaluation completeness:** core scientific claims remain open until full statistical reports are published.

---

## Validation Commands

```bash
# Core tests
python -m pytest

# Efficiency benchmark pipeline (current demonstrated claim)
python experiments/benchmark_efficiency.py

# Primary mixed-domain benchmark (quality validation path)
python experiments/run_real_mixed_benchmark.py
```

## Roadmap and Research Docs

- Top-level roadmap: [ROADMAP.md](ROADMAP.md)
- Contributor roadmap view: [docs/contributing/roadmap.md](docs/contributing/roadmap.md)
- Experiment protocol/status: [docs/research/experiments.md](docs/research/experiments.md)
- Results interpretation: [docs/research/results.md](docs/research/results.md)
