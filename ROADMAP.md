## ROADMAP

```markdown
# Kalmanorix Roadmap

**Last updated: March 20, 2026**
**Current version: v0.2.0 (Core Kalman)**

This roadmap reflects the actual development status and prioritises the critical path to validating the KEFF hypotheses.

---

## 🎯 Phase 0: Foundation (Completed)
*What exists now*

- [x] Project scaffold with modular architecture
- [x] Basic CI/CD (GitHub Actions, pre-commit)
- [x] Type hints and packaging (pyproject.toml)
- [x] Mixed-domain retrieval benchmark runner
- [x] Test structure with pytest
- [x] Initial API design (Village, Panoramix, ScoutRouter)

**Deliverable:** v0.1.0 - Working scaffold with placeholder implementations

---

## 🚧 Phase 1: Core Algorithm Validation (Q2 2026)
*Prove the Kalman fusion works in principle*

### Milestone 1.1: Diagonal Covariance Estimation
- [x] Implement covariance estimation from validation set errors
- [x] Add support for per-model diagonal covariance matrices
- [x] Create visualisation tools for embedding uncertainty
- **Success criteria:** Models can report `(embedding, covariance)` tuples

### Milestone 1.2: Procrustes Alignment
- [x] Implement orthogonal Procrustes for embedding space alignment
- [x] Build reference anchor set for alignment
- [x] Validate alignment quality on STS tasks
- **Success criteria:** Embeddings from different specialists map to comparable spaces

### Milestone 1.3: Basic Kalman Fuser
- [x] Implement Kalman update with diagonal covariance
- [x] Add batch fusion for multiple measurements
- [x] Benchmark against simple averaging
- **Success criteria:** Kalman fusion outperforms averaging on mixed-domain retrieval (p < 0.05)

**Deliverable:** v0.2.0 - Working Kalman fusion on toy domains

---

## 🔬 Phase 2: Hypothesis Testing (Q3 2026)
*Validate the core claims from the original proposal*

### Milestone 2.1: H1 Test - Specialists vs Monolith
- [x] Create two disjoint domain datasets (e.g., legal, medical)
- [x] Train two specialists (each on one domain)
- [x] Train monolithic model on combined data (same total compute)
- [x] Compare: fused specialists vs monolithic on mixed-domain test
- **Target:** Fused specialists achieve higher accuracy with same training FLOPs

### Milestone 2.2: H2 Test - Uncertainty Robustness
- [x] Create out-of-domain test queries
- [x] Compare Kalman fusion (uncertainty-weighted) vs naive averaging
- **Target:** Kalman shows smaller performance drop on OOD queries

### Milestone 2.3: Efficiency Benchmarking
- [ ] Measure inference FLOPs for fusion vs single large model
- [ ] Track memory usage for multiple loaded specialists
- **Target:** Demonstrate efficiency advantage of modular deployment

**Deliverable:** v0.3.0 - Experimental validation of KEFF hypotheses

---

## 📦 Phase 3: Production Readiness (Q4 2026)
*Make it usable by real applications*

### Milestone 3.1: SEF Specification v0.1
- [ ] Define Shareable Embedding Format serialisation
- [ ] Implement `SEFModel.save_pretrained()` and `from_pretrained()`
- [ ] Add metadata manifest (domain tags, benchmarks, licence)
- **Success criteria:** Models can be packaged and shared as single files

### Milestone 3.2: Performance Optimisation
- [ ] Implement low-rank covariance approximations
- [x] Add support for Ensemble Kalman Filter (parallel updates)
- [ ] Optimise router for minimal latency
- **Target:** Fusion adds <10ms latency for d=768, 5 specialists

### Milestone 3.3: Integration & Ecosystem
- [ ] Hugging Face integration (`AutoModel` wrapper)
- [ ] ONNX runtime support
- [ ] Basic model registry (local directory/index)
- [x] Create adapters for OpenAI, Cohere, Anthropic, Azure, and Vertex AI embedding models
- [ ] Implement API rate limiting, error handling, and caching
- [ ] Add uncertainty estimation strategies for proprietary models (distance-based fallback)
- [x] Create interactive Jupyter notebook for education and debugging
- **Deliverable:** v1.0.0 - Production-ready framework

### Milestone 3.4: Intelligent Routing
- [x] Implement semantic router using fast embedder for query encoding
- [x] Add dynamic thresholding based on query characteristics
- [x] Create domain centroid pre-computation for similarity matching
- [x] Add threshold heuristics module (top-k, relative-to-max, adaptive spread, query-length adaptive)
- [ ] Add confidence-based routing decisions (single specialist vs fusion)
- [ ] Create unified routing evaluation harness

---

## 🔮 Phase 4: Advanced Research (2027)
*Push beyond the original vision*

### Research Track A: Learned Alignment
- Replace Procrustes with contrastive learning
- Investigate non-linear alignment networks
- Test on massively multilingual scenarios

### Research Track B: Query-Dependent Uncertainty
- Implement Heteroscedastic Uncertainty Network (HUN) estimator
- Train sidecar neural networks to predict covariance from query features
- Compare neural vs statistical uncertainty estimation approaches
- Implement Monte Carlo dropout for on-the-fly uncertainty
- Compare against deep ensemble methods

### Research Track C: Beyond Embeddings
- Fuse at different layer depths (not just final embeddings)
- Investigate attention map fusion
- Cross-modal applications (text+image specialists)

---

## ⚠️ Risk Register

| Risk | Mitigation | Status |
|------|------------|--------|
| Kalman O(d³) too slow | Prioritise diagonal + low-rank approximations | Active |
| Procrustes alignment insufficient | Fallback to learned alignment in Phase 4 | Monitor |
| Uncertainty estimation unreliable | Start with simple empirical covariance | Active |
| No performance gain over averaging | Fail fast - test H1 in Phase 2 | Critical |

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
