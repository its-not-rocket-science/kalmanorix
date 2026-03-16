
## ROADMAP.md

```markdown
# Kalmanorix Roadmap

**Last updated: March 16, 2026**
**Current version: v0.1.0 (scaffold)**
**Next release: v0.2.0 (Core Kalman) - April 2026**

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
- [ ] Create visualisation tools for embedding uncertainty
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
- [ ] Add alignment visualisation tools
- [ ] Benchmark against identity mapping

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

- [ ] Create two disjoint domain datasets (legal, medical - 50k each)
- [ ] Train two specialists (each on one domain, 1 epoch)
- [ ] Train monolithic model on combined data (2 epochs = same compute)
- [ ] Compare on mixed-domain test set (25% legal, 25% medical, 50% mixed)
- [ ] Repeat with 3, 4, 5 domains
- [ ] Document compute and energy usage

**Target:** Fused specialists achieve higher accuracy with same training FLOPs
**Expected improvement:** 5-15% on mixed queries

### Milestone 2.2: H2 Test - Uncertainty Robustness (Aug 2026)

**Goal:** Demonstrate uncertainty weighting improves OOD performance

- [ ] Create out-of-domain test queries (unseen domains)
- [ ] Compare Kalman fusion vs naive averaging
- [ ] Test with mis-specified covariances (over/under-confident)
- [ ] Measure calibration of uncertainty estimates
- [ ] Ablation: fix all covariances equal (degenerates to averaging)

**Target:** Kalman shows smaller performance drop on OOD queries
**Expected:** 20% smaller drop than averaging

### Milestone 2.3: Efficiency Benchmarking (Sep 2026)

**Goal:** Quantify computational advantages

- [ ] Measure inference FLOPs for fusion vs single large model
- [ ] Track memory usage for multiple loaded specialists
- [ ] Compare total training FLOPs: N specialists vs one N× model
- [ ] Profile fusion latency across model counts (1-20)
- [ ] Document energy consumption estimates

**Target:** Demonstrate efficiency advantage of modular deployment
**Expected:** 2-5× reduction in training compute for same capability

**Deliverable:** v0.3.0 - Experimental validation of KEFF hypotheses
**Artifact:** Technical report/paper draft with results

---

## 📦 Phase 3: Production Readiness (Q4 2026 - Oct to Dec)

*Make it usable by real applications*

### Milestone 3.1: SEF Specification v0.1 (Oct 2026)

**Goal:** Standardised model format for sharing

- [ ] Define Shareable Embedding Format serialisation (JSON + NPY)
- [ ] Implement `SEFModel.save_pretrained()` and `from_pretrained()`
- [ ] Add metadata manifest (domain tags, benchmarks, licence)
- [ ] Create model card template
- [ ] Checksum verification for integrity
- [ ] Documentation for model contributors

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
- [ ] Add support for Ensemble Kalman Filter (parallel updates)
- [ ] Optimise router for minimal latency (TF-IDF caching)
- [ ] Add batch fusion for multiple queries
- [ ] ONNX runtime support for embed functions
- [ ] Profile-guided optimisations

**Target:** Fusion adds <10ms latency for d=768, 5 specialists
**Memory:** <2GB for 10 loaded specialists

### Milestone 3.3: Integration & Ecosystem (Dec 2026)

**Goal:** Easy to use in real applications

- [ ] Hugging Face integration (`AutoModel` wrapper)
- [ ] FastAPI server for remote fusion
- [ ] Basic model registry (local directory/index)
- [ ] Documentation site with examples
- [ ] Tutorials: 3 complete use cases
- [ ] PyPI release

**Deliverable:** v1.0.0 - Production-ready framework
**Metrics:**
- 100+ GitHub stars
- 5+ external contributors
- 3+ blog posts/tutorials

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

- Train sidecar models to predict uncertainty from query
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

---

## 📊 Experiment Tracking

All experiments follow this structure:
