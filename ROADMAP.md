# Kalmanorix Roadmap

**Last updated:** April 9, 2026  
**Current version:** v0.2.0 (research prototype)

This roadmap separates what has been **demonstrated** from what is still **planned**.

## Snapshot: Evidence-Grounded Status

| Core claim | Evidence status | Notes |
|---|---|---|
| Semantic routing reduces compute | **Supported** | Efficiency artifacts are committed in `results/`.
| Kalman fusion beats mean fusion on quality | **Unresolved** | No final statistical artifact proving improvement yet.
| Fused specialists beat monolith at matched compute | **Unresolved** | Benchmark track exists, final report pending.
| Uncertainty weighting improves OOD robustness | **Unresolved** | Planned evaluations exist, but no completed evidence artifact yet.

## Demonstrated Work (completed)

- Framework architecture, APIs, and test scaffolding are implemented.
- Stabilization fixes landed for alignment orientation, covariance scaling, and fusion-path consistency.
- Efficiency benchmark artifacts support the routing-compute claim in current benchmark conditions.
- Stable routing evaluation module now covers semantic + confidence routing with:
  - labeled-domain precision/recall,
  - FLOPs savings,
  - latency trade-off tracking,
  - threshold robustness sweeps,
  - report sections split into quality-preserving wins, compute-only wins, and failure modes.

## Planned Work (not yet demonstrated)

### Phase 1 — Quality Baseline Closure
- Publish a reproducible Kalman-vs-mean benchmark report with statistical testing.
- Establish explicit pass/fail criterion (e.g., p-value threshold and effect size).

### Phase 2 — Core Hypothesis Validation
- Specialists-vs-monolith comparison with compute parity constraints.
- OOD robustness tests for uncertainty weighting.
- Threat-model documentation for negative results.

### Phase 3 — Productionization (conditional)
- Proceed only if at least one core quality hypothesis is positively supported.
- Otherwise, re-scope project as a routing/efficiency toolkit rather than a quality-improvement method.

### Phase 4 — Routing Toolkit Track (can proceed independently)
- Package and maintain routing evaluation as a first-class toolkit surface (`kalmanorix-eval-routing`).
- Publish reproducible routing datasets with labeled domain relevance and realistic latency/FLOPs metadata.
- Document negative/failure cases where routing hurts recall or quality despite compute savings.

## Evidence Status Sections (Core Claims)

### 1) Routing efficiency
**Status:** Supported  
**Evidence:** `results/efficiency_semantic_routing.json`, `results/semantic_routing_efficiency_report.md`, `experiments/benchmark_efficiency.py`.  
**Threats to validity:** limited workload diversity, threshold dependence, and potential mismatch with production latency profiles.

### 2) Kalman quality gain vs mean
**Status:** Unresolved  
**Evidence:** no final committed artifact establishing statistically significant improvement.  
**Threats to validity:** benchmark composition, covariance quality, and alignment assumptions may mask or inflate effects.

### 3) Specialists vs monolith
**Status:** Unresolved  
**Evidence:** no final committed matched-compute comparison report.  
**Threats to validity:** fairness of compute accounting, domain mixture realism, and training-data leakage risk.

### 4) OOD robustness via uncertainty
**Status:** Unresolved  
**Evidence:** no completed OOD report committed yet.  
**Threats to validity:** OOD definition choice, calibration drift, and sensitivity to query distribution shifts.
