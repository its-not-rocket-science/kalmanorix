# Uncertainty Estimation

In Kalmanorix, each specialist provides a scalar measurement variance `sigma2(query)`.
Kalman fusion uses this variance to set how strongly each specialist influences the fused embedding.

## Current sigma² estimation paths (audited)

The code currently supports these scalar uncertainty paths.

1. **Constant** (`ConstantSigma2`)
   - Fixed variance for all queries.
   - Useful as baseline/ablation.

2. **Keyword based** (`KeywordSigma2`)
   - Query contains domain keywords → lower variance.
   - Otherwise higher variance.
   - Cheap, interpretable fallback.

3. **Centroid distance** (`CentroidDistanceSigma2`) **(current default query-dependent path in benchmarks)**
   - Compute cosine distance between query embedding and specialist centroid.
   - Distance is mapped with a softplus-style transform.
   - Includes a small query-length term.

4. **Embedding norm diagnostic** (`EmbeddingNormSigma2`)
   - Low embedding norm implies diffuse/weak representation.
   - Increases variance as norm drops.

5. **Centroid similarity (linearized)** (`SimilarityToCentroidSigma2`)
   - Simpler linear distance-to-centroid mapping.

6. **Stochastic forward variance** (`StochasticForwardSigma2`)
   - Monte-Carlo style dispersion over multiple stochastic passes.
   - Uses average per-dimension variance as scalar uncertainty proxy.

7. **Centroid + norm + peer disagreement** (`CentroidNormPeerSigma2`) **(improved query-dependent estimator)**
   - Combines:
     - distance to own centroid,
     - embedding norm deficit relative to calibration norms,
     - disagreement with peer specialists (peer centroid advantage),
   - then applies a sigmoid-calibrated transform to produce bounded, smooth sigma².

## Precomputed embedding support

`SEF.sigma2_for(query, query_embedding=...)` now supports embedding-aware estimators.
When fusion code already computed a specialist query embedding, sigma² can reuse it instead
of calling the embedder a second time. Estimators that support this expose
`estimate_with_embedding(query, embedding)`.

This is currently implemented for:
- `CentroidDistanceSigma2`
- `EmbeddingNormSigma2`
- `SimilarityToCentroidSigma2`
- `CentroidNormPeerSigma2`

## Why this matters for Kalman vs mean fusion

Kalman fusion only helps when relative uncertainty is informative.
If sigma² is too flat, noisy, or mis-ranked across specialists, Kalman reduces to near-mean behavior.
The improved estimator is designed to make confidence ranking more query-sensitive while remaining
lightweight and model-free.
