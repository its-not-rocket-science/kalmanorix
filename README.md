# Kalmanorix: Specialist-Embedding Fusion Research Framework

> **Research status (as of April 9, 2026):**
> - ✅ Demonstrated: routing-related compute savings in controlled benchmark runs.
> - ⚠️ Not yet demonstrated: statistically significant quality gains of Kalman fusion over simple mean fusion.
> - ⚠️ Specialists-vs-monolith track now has a completed matched-compute artifact, but current verdict is mixed (`null` quality delta + `regression` on inference compute), so superiority remains unproven.

## Current Evidence State

- **Canonical artifact paths:** `results/canonical_benchmark/` (historical v1), `results/canonical_benchmark_v2/` (current decision artifact), and `results/canonical_benchmark_v3/` (next stronger rerun target)
- **State:** **Real artifact committed** (`summary.json` and `report.md` are present).
- **Readout:** the committed canonical run is evidence, but it does **not** close the quality hypotheses (Kalman-vs-mean and specialists-vs-monolith remain unresolved).

Kalmanorix is an experimental framework for combining specialist embedding models with uncertainty-aware fusion and semantic routing. The project can also be used as a standalone **routing-and-efficiency toolkit** even if fusion-quality hypotheses remain unresolved.

## API Maturity Tiers

Kalmanorix now documents three API tiers to keep the top-level package easier to learn:

- **Stable API (`kalmanorix`)**: core workflow primitives (`SEF`, `Village`, `ScoutRouter`, `Panoramix`, `KalmanorixFuser`, etc.).
- **Experimental API (`kalmanorix.experimental`)**: advanced adapters, alignment helpers, threshold heuristics, and research fuser variants.
- **Internal utilities (`kalmanorix.internal`)**: maintainer-facing helpers with no compatibility guarantee.

For compatibility, experimental symbols remain temporarily importable from `kalmanorix` with deprecation warnings.


## Demonstrated Results vs Planned Work

### Demonstrated (with committed artifacts)
- **Routing efficiency:** semantic routing can reduce FLOPs by selecting fewer specialists in benchmark runs (reported around 65% average reduction in current efficiency artifacts). See `results/efficiency_semantic_routing.json` and `results/semantic_routing_efficiency_report.md`.
- **Routing evaluation toolkit:** a stable routing evaluator is available for semantic and confidence modes, including precision/recall against labeled domain relevance, FLOPs savings, latency trade-offs, and threshold robustness sweeps (`kalmanorix-eval-routing` CLI).
- **Engineering stabilization:** alignment, covariance scaling, and fusion-path bug fixes are implemented and tested.

### Planned / In Progress
- **Kalman vs mean fusion quality:** pending reproducible statistical result showing improvement.
- **Specialists vs monolith (matched compute):** completed artifact now committed in `results/matched_compute/`, currently labeled mixed/inconclusive (no quality win and higher inference cost).
- **OOD robustness of uncertainty weighting:** completed guarded artifact now committed in `results/ood_robustness/`, currently labeled inconclusive with explicit null and regression-risk outcomes.
- **Broader routing realism:** extend routing benchmarks with larger domain sets and production-like latency traces.

## Kalman improvement work: implemented, tested, and current empirical status

The Kalman-improvement line now has committed artifacts across implementation, ablations, and benchmark reruns. The evidence map below is the current repository source of truth.

| Workstream | Artifact(s) | Current empirical status |
|---|---|---|
| Uncertainty calibration | `results/uncertainty_calibration/report.md`, `results/uncertainty_calibration/summary.json` | **Implemented + power-audited** calibration split; non-fallback isotonic calibrators selected, but downstream Kalman-vs-Mean delta change is `0.0` on validation and test in this run. |
| Uncertainty ablation | `results/uncertainty_ablation/report.md`, `results/uncertainty_ablation/summary.json` | Multiple uncertainty estimators compared; calibration metrics differ, but retrieval metrics are largely unchanged in this setup (constant uncertainty remains competitive). |
| Covariance ablation | `results/kalman_covariance_ablation_v2/report.md`, `results/kalman_covariance_ablation_v2/summary.json` | Scalar/diagonal/structured Kalman families benchmarked; richer covariance did not clear practical gain thresholds, while latency rose substantially vs mean. |
| Correlation-aware fusion | `results/correlation_aware_fusion/report.md`, `results/correlation_aware_fusion/summary.json` | Correlation-aware variant shows a small positive delta vs baseline Kalman on strengthened correlated split (best reported ΔMRR@10 = `+0.0037`); currently exploratory and not yet a headline claim. |
| Latency optimization | `results/kalman_latency_optimization/report.md`, `results/kalman_latency_optimization/summary.json` | Kalman hot path is faster than legacy (reported `~2.06x` single-query speedup), but optimized Kalman is still materially slower than mean and still fails canonical latency-ratio decision threshold. |
| Canonical benchmark v2 | `results/canonical_benchmark_v2/report.md`, `results/canonical_benchmark_v2/summary.json` | Canonical decision remains `inconclusive_underpowered` for Kalman-vs-Mean; observed quality delta is positive but statistically non-significant with current sample size, and latency ratio check fails. |

### What Kalman improvements changed empirically

- **Implementation quality improved:** calibration is now power-audited; covariance and correlation-aware variants are benchmarked with explicit artifacted outcomes; latency-focused engineering reduced Kalman overhead vs legacy.
- **Headline empirical outcome remains unresolved:** current canonical v2 evidence still does not establish a statistically reliable Kalman quality win over mean under decision-rule thresholds.
- **Null/unchanged outcomes are explicit:** calibration and uncertainty-method improvements currently show limited or zero downstream retrieval gains in committed runs.
- **Open questions next:** increase benchmark power, test whether small correlation-aware gains replicate on larger real splits, and determine whether further latency work can satisfy the canonical latency constraint without hurting quality.

---

## Core Claims and Evidence Status

### Claim A: Semantic routing improves computational efficiency
**Evidence status:** **Supported (demonstrated in benchmark artifacts).**

- Evidence type: benchmark output JSON + report.
- Scope: tested benchmark configurations in repository artifacts.
- Caveat: current runs are limited in domain breadth and deployment realism.

### Claim B: Kalman fusion improves retrieval quality over mean fusion
**Evidence status:** **Unresolved (not demonstrated).**

- Current state: the committed canonical benchmark run does not establish a statistically significant improvement over mean fusion for the configured run.
- Prior synthetic/debug experiments exist but are not sufficient for headline claims.

### Claim C: Fused specialists outperform a monolith at equal compute
**Evidence status:** **Evaluated, but unresolved superiority.**

- Current state: a completed matched-compute artifact is now committed (`results/matched_compute/summary.json`, `results/matched_compute/report.md`), but outcome labels are mixed (`positive` parity, `null` quality delta, `regression` inference cost), so superiority is not established.

### Claim D: Uncertainty-weighted fusion is more robust OOD
**Evidence status:** **Evaluated, currently inconclusive.**

- Current state: a completed guarded OOD artifact is committed (`results/ood_robustness/summary.json`, `results/ood_robustness/report.md`) with explicit `positive/null/inconclusive/regression` labeling; no supported OOD quality advantage is demonstrated yet.

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

# Routing evaluation toolkit (semantic + confidence)
kalmanorix-eval-routing \
  --dataset path/to/routing_eval_dataset.json \
  --output results/routing_eval/report.json \
  --mode semantic \
  --semantic-threshold 0.7 \
  --semantic-thresholds 0.5,0.6,0.7,0.8

# Primary mixed-domain benchmark (quality validation path)
python experiments/run_real_mixed_benchmark.py
```

## Quickstart by Use Case

### 1) Routing toolkit use (quality + efficiency diagnostics)

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

### 2) Canonical benchmark use (claim-governing quality track)

```bash
PYTHONPATH=src python experiments/run_canonical_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.2.0/mixed_benchmark.json \
  --split test \
  --max-queries 1200 \
  --output-dir results/canonical_benchmark_v3
```

### 3) Kalman evidence inspection (artifact-first readout)

```bash
python -m json.tool results/canonical_benchmark_v2/summary.json | head -80
python -m json.tool results/kalman_evidence_dashboard/summary.json | head -120
```

### 4) Extending with new specialists (contributor flow)

```bash
python experiments/train_specialists_st.py --config experiments/configs/milestone_2_1.yaml
python experiments/run_real_mixed_benchmark.py --max-queries 150
```

Then document specialist behavior and expected uncertainty behavior in `docs/contributing/model-contributors.md`.

## Routing Toolkit Quickstart

Use the committed tiny dataset to run an end-to-end routing evaluation with both JSON and markdown outputs:

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

- Dataset: `datasets/routing_eval/small_routing_eval_v1.json`
- Real committed artifact: `results/routing_eval/small_routing_eval_v1_report.json`
- Human-readable report: `results/routing_eval/small_routing_eval_v1_report.md`

Interpretation note: evaluate quality-preserving wins, compute-only wins, and failure modes together; avoid extrapolating quality-improvement claims from routing-efficiency metrics alone.

## Roadmap and Research Docs

- Top-level roadmap: [ROADMAP.md](ROADMAP.md)
- Contributor roadmap view: [docs/contributing/roadmap.md](docs/contributing/roadmap.md)
- Experiment protocol/status: [docs/research/experiments.md](docs/research/experiments.md)
- Results interpretation: [docs/research/results.md](docs/research/results.md)
