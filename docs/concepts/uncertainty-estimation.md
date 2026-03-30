# Uncertainty Estimation

*TODO: Explain variance estimation and calibration for embedding models*

## Why Uncertainty Matters

In Kalman fusion, the uncertainty (variance) of each embedding determines its influence on the fused result. A specialist that is very certain about a query should contribute more than one that is uncertain.

## Types of Uncertainty

### 1. Aleatoric Uncertainty
- **Intrinsic noise** in the data (e.g., ambiguous queries).
- Cannot be reduced with more training data.
- Estimated via **empirical covariance** on a validation set.

### 2. Epistemic Uncertainty
- **Model uncertainty** due to limited training data.
- Can be reduced with more data or better architecture.
- Estimated via **ensemble methods** or Monte‑Carlo dropout.

Kalmanorix currently focuses on aleatoric uncertainty, treating epistemic uncertainty as future work.

## Estimation Strategies

### Constant Uncertainty (`sigma2 = 0.1`)
Simplest approach: assign a fixed variance to each specialist. Works when specialists are equally reliable across their domain.

### Keyword‑Based Uncertainty (`KeywordSigma2`)
Increase uncertainty when the query lacks domain‑specific keywords. Example: a medical specialist becomes uncertain for a query without medical terms.

### Centroid‑Distance Uncertainty (`CentroidDistanceSigma2`)
Compute the cosine distance between the query embedding and the specialist’s domain centroid. Higher distance → higher uncertainty.

### Empirical Covariance (`EmpiricalCovariance`)
Compute the per‑dimension variance of embedding errors on a held‑out validation set. Most accurate but requires labelled validation data.

### Heteroscedastic Uncertainty Network (HUN) *(planned)*
A small neural network that takes query features (e.g., bag‑of‑words, length) and predicts per‑dimension variance.

## Calibration

Uncertainty estimates should be **calibrated**: a variance of 0.1 should correspond to a 68% confidence interval (± 0.3 standard deviation) around the true embedding. Calibration can be checked by measuring the empirical coverage on a validation set.

*TODO: Add calibration plots, code examples for each strategy, and comparison table.*
