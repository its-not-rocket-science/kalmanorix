# Kalmanorix Roadmap

**Last updated: April 3, 2026**  
**Current version: v0.2.0 (Core Kalman)**

## Current Validation Status

⚠️ **CORE HYPOTHESIS UNPROVEN: No evidence yet that Kalman fusion outperforms simple averaging. Milestones 1.3, 2.1, and 2.2 are incomplete pending Q3 2026 benchmarks.**

### Status Legend
- `[ ]` = Not started
- `[~]` = In progress
- `[x]` = Complete **with evidence link** (must point to a specific file in `results/` or `experiments/`)

---

## 🎯 Phase 0: Foundation (Completed)
*What exists now*

- [x] Project scaffold with modular architecture
- [x] Basic CI/CD (GitHub Actions, pre-commit)
- [x] Type hints and packaging (`pyproject.toml`)
- [x] Mixed-domain retrieval benchmark runner
- [x] Test structure with pytest
- [x] Initial API design (Village, Panoramix, ScoutRouter)

**Deliverable:** v0.1.0 - Working scaffold with placeholder implementations

---

## 🚧 Phase 1: Core Algorithm Validation (Q2 2026)
*Prove the Kalman fusion works in principle*

### Milestone 1.1: Diagonal Covariance Estimation `[x]`
- [x] Implement covariance estimation from validation set errors
- [x] Add support for per-model diagonal covariance matrices
- [x] Create visualisation tools for embedding uncertainty
- **Success criteria:** Models can report `(embedding, covariance)` tuples
- **Evidence:** [experiments/validate_covariance.py](experiments/validate_covariance.py)

### Milestone 1.2: Procrustes Alignment `[x]`
- [x] Implement orthogonal Procrustes for embedding space alignment
- [x] Build reference anchor set for alignment
- [x] Validate alignment quality on STS tasks
- **Success criteria:** Embeddings from different specialists map to comparable spaces
- **Evidence:** [experiments/validate_alignment.py](experiments/validate_alignment.py)

### Milestone 1.3: Basic Kalman Fuser `[~]`
- [x] Implement Kalman update with diagonal covariance
- [x] Add batch fusion for multiple measurements
- [x] Benchmark against simple averaging with reproducible artifact generation
- **Success criteria:** Kalman fusion outperforms averaging on mixed-domain retrieval (p < 0.05)
- **Evidence:** [experiments/benchmark_fusion_methods.py](experiments/benchmark_fusion_methods.py), [experiments/visualize_fusion_benchmark.py](experiments/visualize_fusion_benchmark.py), [results/milestone_1_3_kalman_vs_mean.csv](results/milestone_1_3_kalman_vs_mean.csv), [results/milestone_1_3_summary.json](results/milestone_1_3_summary.json), [results/kalman_vs_mean_plot.png](results/kalman_vs_mean_plot.png)
- **Expected completion:** July 2026 (analysis outcome depends on benchmark summary conclusion)


### Stabilization Fixes (April 2026) `[x]`
- [x] Fix Procrustes alignment transpose orientation bug (negative centroid similarity)
  - **Evidence:** [experiments/validate_alignment.py](experiments/validate_alignment.py), [tests/test_validation_suite.py#L21-L67](tests/test_validation_suite.py#L21-L67)
- [x] Replace linear covariance distance scaling with exponential scaling to avoid specialist suppression
  - **Evidence:** [experiments/validate_covariance.py](experiments/validate_covariance.py), [tests/test_validation_suite.py#L71-L118](tests/test_validation_suite.py#L71-L118)
- [x] Switch to ensemble (parallel) Kalman fusion path for order-invariant stability
  - **Evidence:** [experiments/benchmark_fusion_methods.py](experiments/benchmark_fusion_methods.py), [tests/test_validation_suite.py#L121-L168](tests/test_validation_suite.py#L121-L168)
- [x] Remove fabricated benchmark outputs; require statistical significance reporting (`p_values`)
  - **Evidence:** [experiments/benchmark_fusion_methods.py](experiments/benchmark_fusion_methods.py), [tests/test_validation_suite.py#L170-L207](tests/test_validation_suite.py#L170-L207)

**Deliverable:** v0.2.0 - Working Kalman fusion on toy domains

---

## 🔬 Phase 2: Hypothesis Testing (Q3 2026)
*Validate the core claims from the original proposal*

### Milestone 2.1: H1 Test - Specialists vs Monolith `[~]`
- [~] Create two disjoint domain datasets (e.g., legal, medical)
- [~] Train two specialists (each on one domain)
- [~] Train monolithic model on combined data (same total compute)
- [ ] Compare fused specialists vs monolithic on mixed-domain test (**no final artifact yet**)
- **Target:** Fused specialists achieve higher accuracy with same training FLOPs
- **Expected completion:** July 2026

### Milestone 2.2: H2 Test - Uncertainty Robustness `[~]`
- [~] Create out-of-domain test queries
- [ ] Compare Kalman fusion (uncertainty-weighted) vs naive averaging (**no published OOD benchmark report yet**)
- **Target:** Kalman shows smaller performance drop on OOD queries
- **Expected completion:** July 2026

### Milestone 2.3: Efficiency Benchmarking `[x]`
- [x] Measure inference FLOPs for fusion vs single large model
- [x] Track memory usage for multiple loaded specialists
- [x] Profile fusion latency across model counts (1-20)
- **Target:** Demonstrate efficiency advantage of modular deployment
- **Results:** Semantic routing achieves ~65% average FLOPs reduction by selecting only relevant specialists; selection efficiency ~35% across 3-20 specialists.
- **Evidence:** [results/efficiency_semantic_routing.json](results/efficiency_semantic_routing.json), [results/semantic_routing_efficiency_report.md](results/semantic_routing_efficiency_report.md), [experiments/benchmark_efficiency.py](experiments/benchmark_efficiency.py)

**Deliverable:** v0.3.0 - Experimental validation of KEFF hypotheses

---

## 📦 Phase 3: Production Readiness (Q4 2026)
*Make it usable by real applications*

### Milestone 3.1: SEF Specification v0.1 `[x]`
- [x] Define Shareable Embedding Format serialisation
- [x] Implement `SEFModel.save_pretrained()` and `from_pretrained()`
- [x] Add metadata manifest (domain tags, benchmarks, licence)
- **Success criteria:** Models can be packaged and shared as single files (directory format with checksum verification)
- **Evidence:** [experiments/update_results.py](experiments/update_results.py)

### Milestone 3.2: Performance Optimisation `[~]`
- [ ] Implement low-rank covariance approximations
- [x] Add support for Ensemble Kalman Filter (parallel updates)
- [ ] Optimise router for minimal latency
- **Target:** Fusion adds <10ms latency for d=768, 5 specialists
- **Expected completion:** November 2026

### Milestone 3.3: Integration & Ecosystem `[~]`
- [x] Hugging Face integration (`AutoModel` wrapper)
- [x] ONNX runtime support
- [x] Basic model registry (local directory/index)
- [x] Create adapters for OpenAI, Cohere, Anthropic, Azure, and Vertex AI embedding models
- [ ] Implement API rate limiting, error handling, and caching
- [ ] Add uncertainty estimation strategies for proprietary models (distance-based fallback)
- [x] Create interactive Jupyter notebook for education and debugging
- **Expected completion:** December 2026

### Milestone 3.4: Intelligent Routing `[~]`
- [x] Implement semantic router using fast embedder for query encoding
- [x] Add dynamic thresholding based on query characteristics
- [x] Create domain centroid pre-computation for similarity matching
- [x] Add threshold heuristics module (top-k, relative-to-max, adaptive spread, query-length adaptive)
- [ ] Add confidence-based routing decisions (single specialist vs fusion)
- [ ] Create unified routing evaluation harness
- **Results so far:** Semantic routing shows substantial FLOPs reduction.
- **Evidence:** [results/efficiency_semantic_routing.json](results/efficiency_semantic_routing.json)
- **Expected completion:** December 2026

**Deliverable:** v1.0.0 - Production-ready framework

---

## 🔮 Phase 4: Advanced Research (2027)
*Push beyond the original vision*

### Research Track A: Learned Alignment `[ ]`
- [ ] Replace Procrustes with contrastive learning
- [ ] Investigate non-linear alignment networks
- [ ] Test on massively multilingual scenarios
- **Expected completion:** 2027

### Research Track B: Query-Dependent Uncertainty `[ ]`
- [ ] Implement Heteroscedastic Uncertainty Network (HUN) estimator
- [ ] Train sidecar neural networks to predict covariance from query features
- [ ] Compare neural vs statistical uncertainty estimation approaches
- [ ] Implement Monte Carlo dropout for on-the-fly uncertainty
- [ ] Compare against deep ensemble methods
- **Expected completion:** 2027

### Research Track C: Beyond Embeddings `[ ]`
- [ ] Fuse at different layer depths (not just final embeddings)
- [ ] Investigate attention map fusion
- [ ] Cross-modal applications (text+image specialists)
- **Expected completion:** 2027

---

## Known Limitations

- Covariance estimation is currently placeholder/naive in practical settings and needs stronger empirical calibration.
- Kalman fusion vs simple averaging has **not** yet been benchmarked with published, reproducible artifacts proving superiority.
- OOD robustness for uncertainty-weighted fusion has not yet been validated with a completed benchmark report.

---

## Early Stopping Criteria

- If by Month 3 there is no statistically significant improvement over simple averaging, pause KEFF hypothesis claims and redirect effort to baseline strengthening.

### April 2026 Checkpoint (Month 3)

- **Status:** Trigger condition is likely met or unresolved, because no published benchmark artifact currently demonstrates statistically significant improvement of Kalman fusion over averaging.
- **Action:** Treat Milestones 1.3, 2.1, and 2.2 as in progress; prioritize publishing reproducible benchmark artifacts before making further performance claims.
- **Next decision date:** July 2026 (Q3 benchmark review).

---

## ⚠️ Risk Register

| Risk | Mitigation | Status |
|------|------------|--------|
| Kalman O(d³) too slow | Prioritise diagonal + low-rank approximations | Active |
| Procrustes alignment insufficient | Fallback to learned alignment in Phase 4 | Monitor |
| Uncertainty estimation unreliable | Start with simple empirical covariance | Active |
| No performance gain over averaging | Fail fast - enforce Month 3/July 2026 benchmark gates | Critical |

## 📈 Success Metrics

1. **Scientific:** At least one hypothesis (H1/H2) validated with p < 0.05
2. **Performance:** Fusion adds <20% latency overhead vs single model
3. **Adoption:** 3+ external contributors, 100+ GitHub stars
4. **Community:** 2+ research papers using Kalmanorix

## 🤝 How to Help

See our [good first issues](https://github.com/its-not-rocket-science/kalmanorix/labels/good%20first%20issue) and [help wanted](https://github.com/its-not-rocket-science/kalmanorix/labels/help%20wanted) labels. We particularly need:

- Linear algebra optimisation for Kalman updates
- Benchmark design and evaluation
- Documentation and examples
