# Uncertainty Calibration

*TODO: Guide to calibrating uncertainty estimates for better fusion.*

Well‑calibrated uncertainty estimates are crucial for Kalman fusion. This guide explains how to measure and improve the calibration of your specialists’ variance estimates.

## What is Calibration?

A variance estimate `sigma²` is **calibrated** if, on average, the true embedding error falls within `± sqrt(sigma²)` about 68% of the time (for Gaussian errors). Poor calibration leads to over‑ or under‑confident fusion.

## Measuring Calibration

### 1. Collect Validation Data
You need a set of `(query, reference_embedding)` pairs where the reference embedding is considered “ground truth” (e.g., from a high‑quality monolithic model or human judgement).

### 2. Compute Errors
For each specialist, compute:
```
error = || embedding(query) - reference_embedding ||²
```

### 3. Check Coverage
For a given variance estimate `sigma²`, compute the proportion of errors that satisfy `error < sigma²`. This should be close to the expected coverage (e.g., 0.68 for 1‑sigma).

### 4. Plot Reliability Diagram
Plot expected vs observed coverage across multiple variance bins. Ideal calibration follows the diagonal.

## Calibration Techniques

### Scaling Variance
If a specialist is consistently over‑ or under‑confident, apply a scaling factor:
```
sigma²_calibrated = alpha * sigma²_original
```
Choose `alpha` to match observed coverage.

### Temperature Scaling
Similar to scaling but with a per‑dimension temperature vector (for diagonal covariance).

### Bayesian Linear Regression
Fit a linear model `error ~ beta0 + beta1 * sigma²` and adjust predictions.

## Calibration with Centroid‑Distance Uncertainty

For `CentroidDistanceSigma2`, calibration involves tuning the mapping from cosine distance to variance:

```python
from kalmanorix import CentroidDistanceSigma2

# Tune slope and intercept
uncertainty_fn = CentroidDistanceSigma2(
    centroid=centroid_embedding,
    slope=2.0,   # increase for more sensitivity
    intercept=0.1  # minimum variance
)
```

## Automatic Calibration Loop

1. Split validation data into `train_cal` and `test_cal`.
2. Use `train_cal` to fit scaling parameters.
3. Evaluate on `test_cal` to avoid overfitting.

## When Calibration is Impossible

If you lack ground‑truth embeddings, you can still use **relative calibration**: ensure that specialists’ relative certainty orders are correct (e.g., a medical specialist should be more certain about medical queries than a legal specialist).

*TODO: Add code for calibration loops, visualisation of reliability diagrams, and troubleshooting tips.*
