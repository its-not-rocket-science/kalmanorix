
# Kalmanorix: Efficient Specialist Fusion with Kalman Filtering

> **⚠️ Research Status**: Core hypothesis unproven. Efficiency demonstrated; accuracy validation pending (target Q3 2026).

Kalmanorix is a research framework for fusing embeddings from multiple domain-specialist models using Kalman filtering. The core hypothesis is that a fused ensemble of specialists can outperform monolithic models while being computationally efficient (KEFF: Kalman Ensemble of Fusion-Frugal specialists).

## 🧠 Core Hypothesis

Kalmanorix aims to prove three linked claims:
- Specialist models can retain stronger domain signal than a monolith at equal compute.
- Kalman fusion can combine specialist embeddings better than naive averaging by using uncertainty weighting.
- Semantic routing can reduce runtime by invoking only relevant specialists.

Only the routing efficiency claim is currently validated. The Kalman-fusion accuracy claim remains an active research question.

## 🚀 Key Result (Status-Separated)

- ✅ **Proven: Semantic routing efficiency (65% average FLOPs reduction)**
  - Routing selects only relevant specialists per query.
  - Benchmark details:
    - **Selection efficiency**: 35% (specialists selected/loaded) across 3-20 specialists
    - **Latency reduction**: Up to 34% when routing overhead < cost of extra specialists
    - **Dynamic thresholding**: Ensures at least one specialist selected even for ambiguous queries
- ❌ **Unproven: Kalman fusion accuracy improvement over averaging**
  - We have not yet demonstrated statistically significant improvement versus simple averaging.
- 🔬 **In Progress: Specialist vs monolith comparison**
  - Hypothesis testing is planned/ongoing in the Q3 2026 validation track.

> **Important:** The 65% FLOPs reduction comes from **semantic routing**, not from the Kalman fusion update itself.

## ⚠️ Current Limitations (April 2026)
- Kalman fusion has not yet been proven to outperform simple averaging
- Covariance estimation is currently naive/placeholder
- Out-of-domain robustness not yet validated
- See [ROADMAP.md](ROADMAP.md) for validation timeline


## 🔧 Recent Fixes (April 2026)

The following critical bugs have been fixed:

1. **Procrustes alignment transpose error** - Fixed incorrect matrix orientation that caused negative centroid similarities
2. **Covariance scaling** - Replaced linear scaling with exponential to prevent specialist suppression
3. **Sequential fusion** - Switched to ensemble (parallel) fusion for numerical stability
4. **Result fabrication** - Removed hardcoded benchmark results; added proper statistical testing

## 📊 Current Performance

After fixes, the system achieves:
- Kalman fusion vs averaging: [Run `experiments/run_real_mixed_benchmark.py` for primary real-data results; `experiments/benchmark_fusion_methods.py` is debug-only]
- Specialists vs monolith: [Awaiting validation]

## 🧪 Running Validation

```bash
# Test alignment fix
python -m pytest tests/test_validation_suite.py -k TestProcrustesFix

# Run proper benchmark
python experiments/run_real_mixed_benchmark.py

# Validate covariance calibration
python experiments/validate_covariance.py
```

## 📄 Paper-grade Results Pipeline

Use the benchmark registry for reproducible paper artifacts.

### 1) Run one benchmark config (experiment runner)

```bash
python -m experiments.registry.runner \
  --config experiments/configs/benchmark_registry/paper_grade_real_mixed.json
```

This writes:
- `results/paper_grade/benchmark_summary.json`
- `results/paper_grade/benchmark_details.json`

### 2) Generate tables and figures (reporting runner)

```bash
python -m experiments.registry.reporting_runner \
  --summary-json results/paper_grade/benchmark_summary.json \
  --details-json results/paper_grade/benchmark_details.json \
  --output-dir results/paper_grade/report
```

Generated outputs:
- CSV: overall, per-domain, calibration, and statistical-significance tables
- JSON: `results_bundle.json`
- Markdown summary: `summary.md`
- Publication-ready plots (PNG + PDF): latency/memory tradeoff and quality/latency frontier

## 📋 Roadmap

```markdown
# Kalmanorix Roadmap

**Last updated: March 30, 2026**
**Current version: v0.2.0 (Core Kalman)**
**Next release: v0.3.0 (Experimental validation) - September 2026**

This roadmap reflects the actual development status and prioritises the critical path to validating the KEFF hypotheses. It's a living document - updates occur after each milestone.

---

## 🎯 Critical Success Factors

For Kalmanorix to be considered successful, we must demonstrate:

| Factor | Target | Measurement | Deadline |
|--------|--------|-------------|----------|
| **Scientific Validity** | Kalman fusion > averaging (p < 0.05) | Mixed-domain retrieval benchmark | Q3 2026 |
| **Computational Efficiency** | <20% latency overhead vs single model | Profiling suite | Q4 2026 |
| **Practical Utility** | 3+ external users build applications | GitHub issues, citations | Q1 2027 |
| **Reproducibility** | All results replicable | `experiments/` with configs | Ongoing |

## 🚨 Early Stopping Criteria

We will reconsider the approach if:

| Checkpoint | Criterion | Action |
|------------|-----------|--------|
| **Month 3** (April 2026) | No improvement over averaging on toy domains | Document findings, pivot to hybrid approach |
| **Month 6** (July 2026) | O(d³) cannot be adequately approximated | Release as research prototype only |
| **Month 9** (Oct 2026) | No community interest / contributions | Focus on internal use cases |
| **Month 12** (Jan 2027) | No papers citing / using framework | Archive as completed research |

---

## ✅ Phase 0: Foundation (Completed - Dec 2025)

*What exists now*

- [x] Project scaffold with modular architecture (Village + Panoramix + Scout)
- [x] CI/CD pipeline (GitHub Actions, pre-commit hooks)
- [x] Type hints and packaging (pyproject.toml)
- [x] Mixed-domain retrieval benchmark runner
- [x] Test structure with pytest
- [x] Initial API design
- [x] Pre-commit hooks for code quality

**Deliverable:** v0.1.0 - Working scaffold with placeholder implementations
**Evidence:** Repository structure, CI passing, basic imports work

---

## 🚧 Phase 1: Core Algorithm Validation (Q2 2026 - Apr to Jun)

*Prove the Kalman fusion works in principle*

### Milestone 1.1: Diagonal Covariance Estimation (Apr 2026)

**Goal:** Models can report meaningful uncertainty estimates

- [x] Implement covariance estimation from validation set errors
- [x] Add support for per-model diagonal covariance matrices
- [x] Create visualisation tools for embedding uncertainty
- [x] Validate on synthetic data with known noise levels
- [x] Document covariance estimation methods

**Success criteria:** Models return `(embedding, covariance)` tuples where:
- Covariance correlates with empirical error (r > 0.5)
- Computation adds <10% to inference time
- All tests passing with >80% coverage

### Milestone 1.2: Procrustes Alignment (Apr 2026)

**Goal:** Embeddings from different models map to comparable spaces

- [x] Implement orthogonal Procrustes with SVD
- [x] Build reference anchor set for alignment (500 sentences)
- [x] Validate alignment quality on STS tasks
- [x] Add alignment visualisation tools
- [x] Benchmark against identity mapping

**Success criteria:**
- Alignment improves cross-model similarity by >20%
- Determinant correction ensures proper rotation (no reflection)
- Works for d=384, 768, 1024 dimensions

### Milestone 1.3: Basic Kalman Fuser (May 2026)

**Goal:** Working fusion that outperforms averaging

- [x] Implement Kalman update with diagonal covariance (O(d))
- [x] Add batch fusion for multiple measurements
- [x] Benchmark against simple averaging on toy domains
- [x] Numerical stability testing with extreme covariances
- [x] Document update equations and assumptions

**Success criteria:**
- Kalman fusion outperforms averaging on mixed-domain retrieval (p < 0.05)
- Numerical stability across 1e-6 to 1e6 covariance scales
- Processing 5 models in <50ms on CPU

**Deliverable:** v0.2.0 - Working Kalman fusion on toy domains

---

## 🔬 Phase 2: Hypothesis Testing (Q3 2026 - Jul to Sep)

*Validate the core claims from the original proposal*

### Milestone 2.1: H1 Test - Specialists vs Monolith (Jul 2026)

**Goal:** Prove fused specialists beat monolithic training

- [x] Create two disjoint domain datasets (legal, medical - 50k each)
- [x] Train two specialists (each on one domain, 1 epoch)
- [x] Train monolithic model on combined data (2 epochs = same compute)
- [x] Compare on mixed-domain test set (25% legal, 25% medical, 50% mixed)
- [x] Repeat with 3, 4, 5 domains
- [x] Document compute and energy usage

**Target:** Fused specialists achieve higher accuracy with same training FLOPs
**Expected improvement:** 5-15% on mixed queries

### Milestone 2.2: H2 Test - Uncertainty Robustness (Aug 2026)

**Goal:** Demonstrate uncertainty weighting improves OOD performance

- [x] Create out-of-domain test queries (unseen domains)
- [x] Compare Kalman fusion vs naive averaging
- [x] Test with mis-specified covariances (over/under-confident)
- [x] Measure calibration of uncertainty estimates
- [x] Ablation: fix all covariances equal (degenerates to averaging)

**Target:** Kalman shows smaller performance drop on OOD queries
**Expected:** 20% smaller drop than averaging

### Milestone 2.3: Efficiency Benchmarking (Sep 2026)

**Goal:** Quantify computational advantages

- [x] Measure inference FLOPs for fusion vs single large model
- [x] Track memory usage for multiple loaded specialists
- [ ] Compare total training FLOPs: N specialists vs one N× model
- [x] Profile fusion latency across model counts (1-20)
- [ ] Document energy consumption estimates

**Key Results:** Semantic routing achieves **65% average FLOPs reduction** by selecting only relevant specialists. Benchmark shows selection efficiency of 35% (specialists selected/loaded) across 3-20 specialists. Latency reductions up to 34% when routing overhead is less than cost of extra specialist invocations.

**Target:** Demonstrate efficiency advantage of modular deployment ✅
**Expected:** 2-5× reduction in training compute for same capability

**Deliverable:** v0.3.0 - Experimental validation of KEFF hypotheses
**Artifact:** Technical report/paper draft with results

---

## 📦 Phase 3: Production Readiness (Q4 2026 - Oct to Dec)

*Make it usable by real applications*

### Milestone 3.1: SEF Specification v0.1 (Oct 2026)

**Goal:** Standardised model format for sharing

- [x] Define Shareable Embedding Format serialisation (JSON + NPY)
- [x] Implement `SEFModel.save_pretrained()` and `from_pretrained()`
- [x] Add metadata manifest (domain tags, benchmarks, licence)
- [x] Create model card template
- [x] Checksum verification for integrity
- [x] Documentation for model contributors

**Success criteria:** Models can be packaged and shared as single directories
**Format:** Directory with:
- `metadata.json` - Human-readable info
- `model.pkl` - Pickled embed function (optional)
- `alignment.npy` - Procrustes matrix
- `covariance.npz` - Uncertainty data
- `checksum.txt` - SHA-256 verification

### Milestone 3.2: Performance Optimisation (Nov 2026)

**Goal:** Production-ready speed

- [ ] Implement low-rank covariance approximations (UUT + D)
- [x] Add support for Ensemble Kalman Filter (parallel updates)
- [ ] Optimise router for minimal latency (TF-IDF caching)
- [x] Add batch fusion for multiple queries
- [x] ONNX runtime support for embed functions
- [ ] Profile-guided optimisations

**Target:** Fusion adds <10ms latency for d=768, 5 specialists
**Memory:** <2GB for 10 loaded specialists

### Milestone 3.3: Integration & Ecosystem (Dec 2026)

**Goal:** Easy to use in real applications

- [x] Hugging Face integration (`AutoModel` wrapper)
- [x] FastAPI server for remote fusion
- [x] Basic model registry (local directory/index)
- [x] Documentation site with examples
- [x] Tutorials: 3 complete use cases
- [x] Add API usage examples (Python, JavaScript, curl) (Medium priority)
- [ ] PyPI release
- [x] Create adapters for OpenAI, Cohere, Anthropic, Azure, and Vertex AI embedding models
- [x] Implement API rate limiting, error handling, and caching (High priority)
- [x] Add uncertainty estimation strategies for proprietary models (distance-based fallback) (High priority)
- [x] Create interactive Jupyter notebook for education and debugging

**Deliverable:** v1.0.0 - Production-ready framework
**Metrics:**
- 100+ GitHub stars
- 5+ external contributors
- 3+ blog posts/tutorials

### Milestone 3.4: Intelligent Routing (Q1 2027) ✅ Completed Early

**Goal:** Adaptive routing based on query semantics ✅

- [x] Implement semantic router using fast embedder for query encoding
- [x] Add dynamic thresholding based on query characteristics
- [x] Create domain centroid pre-computation for similarity matching
- [x] Add threshold heuristics module (top-k, relative-to-max, adaptive spread, query-length adaptive)
- [ ] Add confidence-based routing decisions (single specialist vs fusion)
- [ ] Create unified routing evaluation harness

**Results:** Semantic routing achieves 65% average FLOPs reduction by selecting only relevant specialists. Dynamic threshold heuristics ensure at least one specialist selected even for ambiguous queries.

---

## 🔮 Phase 4: Advanced Research (2027)

*Push beyond the original vision*

### Research Track A: Learned Alignment

**Challenge:** Procrustes assumes linear relationship between spaces

- Replace Procrustes with contrastive learning
- Investigate non-linear alignment networks (MLP, small transformer)
- Test on massively multilingual scenarios (100+ languages)
- Compare against linear baseline

**Success metric:** Non-linear improves cross-lingual transfer by >10%

### Research Track B: Query-Dependent Uncertainty

**Challenge:** Fixed covariance per model ignores query difficulty

- Implement Heteroscedastic Uncertainty Network (HUN) estimator
- Train sidecar neural networks to predict covariance from query features
- Compare neural vs statistical uncertainty estimation approaches
- Implement Monte Carlo dropout for on-the-fly uncertainty
- Compare against deep ensemble methods
- Test on adversarial/hard queries

**Success metric:** Uncertainty correlates with actual error (r > 0.7)

### Research Track C: Beyond Embeddings

**Challenge:** Final embeddings may lose information

- Fuse at different layer depths (not just final embeddings)
- Investigate attention map fusion
- Cross-modal applications (text+image specialists)
- Hierarchical fusion (fuse within domains first)

**Success metric:** Layer fusion improves performance on probing tasks

### Research Track D: Theoretical Foundations

**Challenge:** Gaussian assumption may be too restrictive

- Derive non-linear variants (UKF, EnKF)
- Investigate particle filter alternatives
- Connect to Bayesian deep learning theory
- Formalise error bounds

**Deliverable:** Journal paper with theoretical guarantees

### Research Track E: Risk-Aware Fusion (Black-Scholes Inspired)

**Challenge:** Current Kalman filter assumes symmetric Gaussian uncertainty; real-world decisions often require risk-aware uncertainty quantification.

- **Implied volatility covariance (Medium priority):** Train a neural network to predict diagonal covariance directly from input text (analogous to implied volatility in options)
- **Risk-sensitive Kalman gain (Medium priority):** Introduce risk-aversion parameter λ for asymmetric uncertainty weighting (penalize underestimation more than overestimation)
- **Stochastic embedding dynamics (Low priority):** Model semantic state as geometric Brownian motion with drift term for streaming text
- **Monte Carlo fusion (Low priority):** Use Monte Carlo simulation to price fused embeddings under risk-neutral measures

**Success metric:** Risk-aware fusion improves performance on safety-critical tasks without sacrificing overall accuracy

---

## 📊 Experiment Tracking

All experiments follow this structure:
