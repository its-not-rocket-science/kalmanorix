# Experiments

*TODO: Document the experimental methodology and benchmarks used to validate Kalmanorix hypotheses.*

This section describes the experimental setup used to test the core Kalmanorix hypotheses (H1, H2) and evaluate fusion efficiency.

## Mixed-domain benchmark specification

For the concrete real-data benchmark design (datasets, preprocessing, schema, versioning), see [Mixed-Domain Retrieval Benchmark](mixed-domain-retrieval-benchmark.md).

## Hypothesis 1 (H1): Specialists vs Monolith

**Claim**: A fused ensemble of domain specialists outperforms a monolithic model trained on the same total compute budget.

### Dataset
- **Medical domain**: PubMed abstracts (10 000 sentences)
- **Legal domain**: Court opinions from Caselaw Access Project (10 000 sentences)
- **Mixed test set**: 2 000 sentences with equal mix of medical, legal, and hybrid topics.

### Models
- **Specialists**: Two Sentence‑Transformer models fine‑tuned on medical and legal data separately.
- **Monolith**: One Sentence‑Transformer model fine‑tuned on the combined medical+legal dataset (same total training steps).

### Training Compute Equivalence
Total training FLOPs for specialists = FLOPs for monolith. This ensures fair comparison.

### Evaluation Metric
**Recall@k** on a mixed‑domain retrieval task: given a query, find the most semantically similar sentences in a corpus containing both medical and legal texts.

### Results
*See [Results](results.md) for detailed numbers.*

## Hypothesis 2 (H2): Uncertainty Robustness

**Claim**: Kalman fusion (uncertainty‑weighted) shows smaller performance drop on out‑of‑domain queries compared to naive averaging.

### Out‑of‑Domain Queries
- **Financial queries**: Sentences from SEC filings and earnings reports (domain unseen during training).
- **Scientific queries**: Abstracts from arXiv CS papers.

### Baselines
- `MeanFuser` (uniform averaging)
- `KalmanorixFuser` (diagonal covariance)
- `DiagonalKalmanFuser` (scalar variance)
- `LearnedGateFuser` (learned gating)

### Metric
**Relative performance drop**: `(accuracy_on_indomain - accuracy_on_ood) / accuracy_on_indomain`. Lower drop is better.

## Efficiency Benchmarking

### FLOPs Measurement
Count floating‑point operations for:
- Embedding a query with each specialist.
- Fusion computation (Kalman update).
- Routing overhead (centroid similarity computation).

### Memory Footprint
Measure peak GPU/CPU memory when loading 1–20 specialists.

### Latency Profiling
End‑to‑end latency (query → fused embedding) across specialist counts, with and without semantic routing.

### Semantic Routing Efficiency
**Selection ratio**: `(specialists selected) / (specialists loaded)`. Lower is better (more efficient).

**FLOPs reduction**: `1 - (FLOPs_with_routing / FLOPs_without_routing)`.

## Experimental Code

The experiments are implemented in the `experiments/` directory:

- `run_milestone_2_1.py` – H1 test (specialists vs monolith).
- `train_specialists_st.py` – Fine‑tune specialist Sentence‑Transformers.
- `train_monolith.py` – Fine‑tune monolithic model.
- `generate_test_set.py` – Create mixed‑domain test sets.
- `efficiency_benchmark.py` – FLOPs, memory, latency measurements.

## Reproducibility

All experiments use fixed random seeds (42). Dataset splits are provided as CSV files in `experiments/data/`. Training logs and model checkpoints are saved to `experiments/runs/`.

## Future Experiments

- **Low‑rank covariance approximation** vs full diagonal covariance.
- **Cross‑lingual specialists** (multilingual BERT) with alignment.
- **Query‑dependent uncertainty** using HUN (Heteroscedastic Uncertainty Network).

*TODO: Add exact command lines to reproduce each experiment, sample‑size justifications, and statistical‑test details.*
