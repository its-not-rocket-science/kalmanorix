# Uncertainty Calibration

Well‑calibrated uncertainty estimates are crucial for Kalman fusion. This guide explains how to measure and improve the calibration of your specialists’ variance estimates (`sigma²`), ensuring optimal fusion weights.

## What is Calibration?

A variance estimate `sigma²` is **calibrated** if, on average, the true embedding error falls within `± sqrt(sigma²)` about 68% of the time (for Gaussian errors). Poor calibration leads to over‑ or under‑confident fusion:

- **Over‑confident**: `sigma²` too small → specialist receives excessive weight, dominates fusion
- **Under‑confident**: `sigma²` too large → specialist receives insufficient weight, contribution diminished

Kalman fusion is optimal only when uncertainty estimates are well‑calibrated.

## Measuring Calibration

### 1. Collect Validation Data
You need a set of `(query, reference_embedding)` pairs where the reference embedding is considered “ground truth”. Options:

- **High‑quality monolithic model** (e.g., `text‑embedding‑3‑large` for OpenAI specialists)
- **Human‑annotated similarity judgments** (convert to embeddings via triplet loss)
- **Synthetic data** with known true embeddings (for testing)

```python
validation_data = [
    ("patient diagnosis report", reference_embedding_1),
    ("court ruling on contract", reference_embedding_2),
    # ...
]
```

### 2. Compute Errors
For each specialist and each query, compute the squared L2 error:

```python
import numpy as np

def compute_errors(sef, validation_data):
    errors = []
    predicted_variances = []
    for query, ref_emb in validation_data:
        pred_emb = sef.embed(query)
        error = np.sum((pred_emb - ref_emb) ** 2)  # squared L2
        sigma2 = sef.sigma2_for(query)
        errors.append(error)
        predicted_variances.append(sigma2)
    return np.array(errors), np.array(predicted_variances)
```

### 3. Check Coverage
For a given confidence level `p` (e.g., `p=0.68` for 1‑sigma), compute the empirical coverage:

```python
def empirical_coverage(errors, variances, p=0.68):
    # For Gaussian: error < sigma² corresponds to |error| < sqrt(sigma²)
    # But we use squared error, so check error < sigma²
    within = errors < variances
    return np.mean(within)
```

The coverage should be close to `p`. For `p=0.68`, ideal coverage is 0.68.

### 4. Plot Reliability Diagram
Visualize calibration across multiple variance bins:

```python
import matplotlib.pyplot as plt

def plot_reliability_diagram(errors, variances, n_bins=10):
    # Bin by predicted variance
    bins = np.percentile(variances, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    empirical_coverages = []

    for i in range(n_bins):
        mask = (variances >= bins[i]) & (variances <= bins[i + 1])
        if np.sum(mask) > 0:
            bin_var = np.mean(variances[mask])
            bin_coverage = np.mean(errors[mask] < variances[mask])
            bin_centers.append(bin_var)
            empirical_coverages.append(bin_coverage)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, empirical_coverages, 'o-', label='Empirical')
    plt.plot([min(bin_centers), max(bin_centers)], [0.68, 0.68], 'k--', label='Ideal')
    plt.xlabel('Predicted variance (sigma²)')
    plt.ylabel('Empirical coverage')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Calibration Techniques

### Scaling Variance (Isotropic Calibration)
If a specialist is consistently over‑ or under‑confident, apply a global scaling factor:

```python
def calibrate_scaling(errors, variances, target_coverage=0.68):
    """Find alpha that minimizes |coverage(alpha*variances) - target_coverage|."""
    from scipy.optimize import minimize_scalar

    def loss(alpha):
        scaled = alpha * variances
        coverage = np.mean(errors < scaled)
        return (coverage - target_coverage) ** 2

    result = minimize_scalar(loss, bounds=(0.1, 10.0))
    return result.x

alpha = calibrate_scaling(errors, variances)
calibrated_variances = alpha * variances
```

### Temperature Scaling (Per‑Dimension)
For diagonal covariance models, scale each dimension independently:

```python
def calibrate_temperature(errors_per_dim, variances_per_dim):
    """errors_per_dim: (n_samples, d), variances_per_dim: (n_samples, d)"""
    # Solve for temperature vector t where errors ~ t * variances
    # Using least squares per dimension
    t = np.mean(errors_per_dim / variances_per_dim, axis=0)
    return t

# Assuming you have per‑dimension error estimates
temperature = calibrate_temperature(errors_per_dim, variances_per_dim)
calibrated_variances_per_dim = temperature * variances_per_dim
```

### Bayesian Linear Regression
Model the relationship between predicted variance and actual error:

```python
from sklearn.linear_model import BayesianRidge

def calibrate_bayesian(errors, variances):
    X = variances.reshape(-1, 1)
    y = errors
    model = BayesianRidge()
    model.fit(X, y)
    # Predict calibrated variance as expected error
    calibrated = model.predict(X)
    return calibrated, model

calibrated_variances, model = calibrate_bayesian(errors, variances)
```

## Calibration with Centroid‑Distance Uncertainty

`CentroidDistanceSigma2` maps cosine similarity to variance: `sigma² = base_sigma² + scale * (1 - similarity)`. Calibration involves tuning `base_sigma²` and `scale`:

```python
from kalmanorix.uncertainty import CentroidDistanceSigma2
import numpy as np

# 1. Compute centroid from calibration texts
calibration_texts = ["medical diagnosis", "patient treatment", ...]
embeddings = [embedder(t) for t in calibration_texts]
centroid = np.mean(embeddings, axis=0)
centroid = centroid / np.linalg.norm(centroid)

# 2. Tune parameters using validation data
def tune_centroid_params(errors, similarities, target_coverage=0.68):
    """Find optimal base_sigma2 and scale."""
    from scipy.optimize import minimize

    def loss(params):
        base, scale = params
        pred_var = base + scale * (1 - similarities)
        coverage = np.mean(errors < pred_var)
        return (coverage - target_coverage) ** 2

    initial = [0.2, 2.0]
    bounds = [(0.01, 1.0), (0.1, 10.0)]
    result = minimize(loss, initial, bounds=bounds)
    return result.x

# Compute similarities for validation queries
validation_similarities = []
for query, _ in validation_data:
    emb = embedder(query)
    emb = emb / np.linalg.norm(emb)
    sim = emb @ centroid
    validation_similarities.append(sim)

validation_similarities = np.array(validation_similarities)
base_sigma2, scale = tune_centroid_params(errors, validation_similarities)

# 3. Create calibrated uncertainty function
uncertainty_fn = CentroidDistanceSigma2(
    embed=embedder,
    centroid=centroid,
    base_sigma2=base_sigma2,
    scale=scale,
)
```

## Automatic Calibration Loop

A complete calibration pipeline:

```python
def calibrate_specialist(sef, validation_data, test_size=0.2):
    """Calibrate a specialist using train/test split."""
    from sklearn.model_selection import train_test_split

    # Split data
    queries, ref_embs = zip(*validation_data)
    train_data, test_data = train_test_split(
        validation_data, test_size=test_size, random_state=42
    )

    # Compute errors on training set
    train_errors, train_variances = compute_errors(sef, train_data)

    # Calibration method (choose one)
    alpha = calibrate_scaling(train_errors, train_variances)

    # Evaluate on test set
    test_errors, test_variances = compute_errors(sef, test_data)
    test_calibrated = alpha * test_variances
    test_coverage = empirical_coverage(test_errors, test_calibrated)

    print(f"Test coverage: {test_coverage:.3f} (target: 0.68)")
    print(f"Scaling factor alpha: {alpha:.3f}")

    # Return calibrated sigma2 function
    original_sigma2 = sef.sigma2
    calibrated_sigma2 = lambda q: alpha * original_sigma2(q)
    return calibrated_sigma2
```

## Visualization and Diagnostics

### Calibration Curve
Plot predicted vs actual error:

```python
def plot_calibration_curve(errors, variances):
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.scatter(variances, errors, alpha=0.5, s=10)
    plt.plot([0, max(variances)], [0, max(variances)], 'r--', label='Perfect')
    plt.xlabel('Predicted variance (sigma²)')
    plt.ylabel('Actual squared error')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plot_reliability_diagram(errors, variances)

    plt.tight_layout()
    plt.show()
```

### Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Coverage < 0.68 | Over‑confident (sigma² too small) | Increase scaling factor (alpha > 1) |
| Coverage > 0.68 | Under‑confident (sigma² too large) | Decrease scaling factor (alpha < 1) |
| Coverage varies by domain | Poor centroid selection | Improve calibration texts for centroid |
| Coverage inconsistent across queries | Non‑Gaussian errors | Consider robust loss or quantile calibration |
| Coverage improves but fusion quality drops | Scaling breaks relative ordering | Use isotonic regression instead of scaling |

## When Calibration is Impossible

If you lack ground‑truth embeddings, use **relative calibration**:

1. **Domain expertise**: Manually verify that specialists are more certain about their own domains.
2. **Cross‑specialist validation**: For query `q`, check that `sigma²_medical(q) < sigma²_legal(q)` when `q` is medical.
3. **Synthetic tests**: Create minimal pairs that should have clear uncertainty ordering.

```python
def verify_relative_calibration(medical_sef, legal_sef, medical_queries, legal_queries):
    """Check that specialists are more certain about their own domains."""
    medical_self = [medical_sef.sigma2_for(q) for q in medical_queries]
    medical_other = [medical_sef.sigma2_for(q) for q in legal_queries]
    legal_self = [legal_sef.sigma2_for(q) for q in legal_queries]
    legal_other = [legal_sef.sigma2_for(q) for q in medical_queries]

    print(f"Medical specialist: self={np.mean(medical_self):.3f}, other={np.mean(medical_other):.3f}")
    print(f"Legal specialist: self={np.mean(legal_self):.3f}, other={np.mean(legal_other):.3f}")

    # Self‑variance should be lower than other‑variance
    assert np.mean(medical_self) < np.mean(medical_other)
    assert np.mean(legal_self) < np.mean(legal_other)
```

## Best Practices

1. **Start simple**: Use constant `sigma²` for prototyping, then add query‑dependent uncertainty.
2. **Validate on held‑out data**: Always evaluate calibration on data not used for tuning.
3. **Monitor over time**: Re‑calibrate periodically as data distribution drifts.
4. **Combine with routing**: Well‑calibrated uncertainty improves both fusion and routing decisions.
5. **Document calibration process**: Record parameters, validation coverage, and any assumptions.

## Further Reading

- [Creating Specialists](creating-specialists.md) – How to build specialists with calibrated uncertainty
- [Fusion Strategies](fusion-strategies.md) – How calibration affects fusion performance
- [Research: Uncertainty Robustness](../research/experiments.md) – Experimental validation of uncertainty calibration
- [API Usage Examples](api-usage.md) – Python, JavaScript, and curl examples for both library and REST API
