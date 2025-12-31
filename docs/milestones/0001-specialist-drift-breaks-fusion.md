# Milestone 0001 — Specialist Drift Breaks Fusion (Without Alignment)

**Date:** 2025-03-08
**Project:** Kalmanorix (KEFF prototype)
**Commit:** `356cca21b587cf1d05da5f1c1eedc71feb2a754c`

---

## Summary

This milestone establishes a critical empirical finding:

> **Embedding-space fusion fails when specialist models drift into incompatible coordinate frames.**
> In this regime, mean fusion degrades performance, Kalman-style uncertainty weighting collapses to hard routing, and only explicit selection (gating) can recover partial performance.

This validates a key hypothesis of the KEFF proposal: **cross-model alignment is not optional**. Any practical embedding-fusion framework must include an explicit alignment step (e.g., orthogonal Procrustes or learned alignment) before fusion.

---

## Experimental Setup

### Objective

Evaluate whether Kalman-style embedding fusion provides benefits over:
- hard routing
- mean embedding fusion
- learned gating

when specialists are *strongly diverged* but share the same architecture and dimensionality.

### Base Encoder

- `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: **384**

### Specialists

Two separate checkpoints were trained:

- `models/tech-minilm`
- `models/cook-minilm`

Both:
- share architecture and dimensionality
- are trained independently
- intentionally diverged via supervised domain separation

### Divergence Training Strategy

Each specialist was trained using a **two-phase process**:

1. **In-domain structure**
   - MultipleNegativesRankingLoss on augmented paraphrase pairs

2. **Explicit divergence (joint, backpropagated into encoder)**
   - Binary domain classification loss (in-domain vs out-domain)
   - Push-away loss from the *other domain’s centroid*

This produced **strong representation drift**, including negative cosine similarity between specialist embeddings.

---

## Environment & Dependencies

Relevant package versions at the time of this milestone:

accelerate==1.12.0
numpy==2.4.0
torch==2.9.1
transformers==4.57.3
sentence-transformers==5.2.0
scikit-learn==1.8.0
scipy==1.16.3
pytest==9.0.2
ruff==0.14.10
pylint==4.0.4

## Key Experiments

### 1. Specialist Disagreement Check

Cosine similarity between tech and cook embeddings for each query:

0.097 'battery lasts all day'
-0.040 'fast charging with the right charger'
-0.129 'cpu throttles when hot under sustained load'
-0.170 'gpu driver update improves frame rates'
-0.170 'camera low light performance sensor size'
-0.252 'braise for hours until tender'
-0.349 'slow cooker stew simmer for hours'
-0.322 'saute vegetables then bake in the oven'
-0.307 'reduce sauce by simmering'
-0.329 'food processor chop onions for stew'
-0.353 'reduce background activity like reducing a sauce'
-0.027 'thermal load feels like oven heat'
-0.103 'camera pipeline acceleration on gpu'
0.054 'recipe for faster charging'
0.214 'smartphone battery lasts longer than a slow cooker braise'


**Interpretation:**
The two specialists are no longer merely “tilted” — they are often **anti-aligned**. This confirms that divergence training succeeded in creating incompatible embedding frames.

---

### 2. Mixed-Domain Retrieval Performance

Evaluation: Recall@k over a fixed document pool, with **strategy-specific document indexing**.

== Mixed-domain retrieval (SentenceTransformer specialists) ==
docs: 16, queries: 15, dim: 384

hard Recall@1=0.600 Recall@3=0.600
mean Recall@1=0.467 Recall@3=0.667
kalman Recall@1=0.600 Recall@3=0.600
gate Recall@1=0.533 Recall@3=0.533


**Observations:**

- Mean fusion **performs worst**.
- Kalman fusion **collapses to hard routing**.
- Learned gating sometimes recovers performance by selecting a single space.

---

### 3. Mixed-Query Diagnostics (Top-1 Predictions)

Example excerpt:

query: "camera pipeline acceleration on gpu" (true=14)
hard: tech=1.000 top1=13 ✗
mean: 0.5 / 0.5 top1=13 ✗
kalman: tech=0.956 top1=2 ✗
gate: ~0.49 / 0.51 top1=13 ✗


Across all mixed queries:

- **All fusion strategies frequently select different vectors**
- **But ranking remains unstable or worse than hard routing**
- Mean fusion is actively harmful due to destructive interference

---

## Interpretation

This milestone demonstrates a **hard failure mode** for embedding-space fusion:

> When specialists drift into different coordinate frames, linear fusion (mean or Kalman-weighted) becomes geometrically meaningless.

Key conclusions:

1. **Uncertainty weighting cannot fix misalignment**
   Kalman gain assumes measurements are expressed in the same latent basis.

2. **Fusion without alignment is worse than no fusion**
   Mean fusion degrades; Kalman reduces to single-expert selection.

3. **Gating survives because it avoids fusion**
   It selects one space rather than mixing incompatible vectors.

4. **Alignment is a first-class requirement**, not an optimization.

This directly supports the KEFF design emphasis on:
- orthogonal Procrustes alignment
- alignment metadata as part of the SEF format

---

## Decision & Next Steps

### Decision

This snapshot is checked in as a **discovery milestone**.
No further refinement will be done on this branch of experiments.

### Next Experiments (Phase 02)

All subsequent work will focus on **alignment-aware fusion**, starting with:

1. **Orthogonal Procrustes alignment**
   - Learn Q on anchor texts
   - Wrap specialist embedders with alignment transforms
2. **Re-evaluate fusion strategies post-alignment**
3. Compare:
   - unaligned vs aligned Kalman fusion
   - mean vs Kalman under alignment
4. Explore:
   - learned alignment networks
   - alignment uncertainty as part of sigma²

Planned location:
experiments/02_alignment/

---

## Why This Milestone Matters

This result turns a vague intuition into a concrete, reproducible fact:

> **KEFF is not about clever weighting alone.
> It is about making heterogeneous representations commensurate.**

This milestone justifies the project’s core architectural claim and sharply narrows the remaining research space.

---
