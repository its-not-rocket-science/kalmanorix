# Kalman Fusion

*TODO: Explain the mathematical foundations of Kalman filtering for embeddings*

## Overview

Kalman fusion in Kalmanorix adapts the classical Kalman filter—a recursive estimator for linear dynamic systems—to the task of combining embeddings from multiple domain‑specialist models. The key insight is that each specialist’s embedding can be treated as a noisy measurement of an underlying “true” semantic representation, with the specialist’s uncertainty (covariance) quantifying the noise.

## Diagonal Covariance Approximation

To avoid the O(d³) complexity of full covariance matrices, Kalmanorix uses a **diagonal covariance** approximation. This reduces the fusion cost to O(d) while still capturing per‑dimension uncertainty.

**Mathematical form:**
```
posterior = prior + Σ_i K_i (measurement_i - prior)
```
where the Kalman gain `K_i` depends on the prior covariance and the measurement covariance.

## Sequential Updates

Measurements are processed in order of increasing uncertainty (lowest variance first). This ordering improves numerical stability and gives more certain measurements greater influence.

## Batch Fusion

The same equations extend naturally to batch processing of multiple queries, enabling efficient fusion of embeddings for entire document collections.

## Comparison with Alternatives

- **Mean fusion**: Uniform averaging (equivalent to Kalman with infinite uncertainty).
- **Weighted averaging**: Static weights, not adaptive to query‑specific uncertainty.
- **Kalman fusion**: Dynamically adjusts weights based on per‑measurement uncertainty.

*TODO: Add equations, pseudocode, and visual examples.*
