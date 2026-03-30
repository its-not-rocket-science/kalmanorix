# Fusion Strategies

*TODO: Guide to choosing and configuring fusion strategies in Kalmanorix.*

Kalmanorix provides several fusion strategies (`Fuser` implementations) for combining specialist embeddings. This guide helps you select the right strategy for your use case.

## Comparison Table

| Fuser | Description | When to Use |
|-------|-------------|-------------|
| `MeanFuser` | Uniform average of all selected embeddings. | Baseline; when specialists have equal reliability. |
| `KalmanorixFuser` | True Kalman fusion with per‑dimension diagonal covariance. | Default for most applications; uses per‑specialist uncertainty. |
| `EnsembleKalmanFuser` | Parallel Kalman updates (no ordering). | When specialists are independent and uncertainty ordering is not critical. |
| `StructuredKalmanFuser` | Kalman with low‑rank + diagonal covariance. | When specialists share systematic errors (e.g., same architecture). |
| `DiagonalKalmanFuser` | Scalar Kalman update (shared variance across dimensions). | When per‑dimension variance estimates are unreliable. |
| `LearnedGateFuser` | Learned two‑way gating (logistic regression on bag‑of‑words). | When you have labelled fusion outcomes for training. |

## Choosing Based on Data

### High‑Quality Uncertainty Estimates
If you have validation data to compute empirical covariance, use `KalmanorixFuser`.

### No Uncertainty Estimates
If specialists provide only embeddings (no variance), start with `MeanFuser` or `DiagonalKalmanFuser` (with a heuristic variance).

### Many Similar Specialists
If specialists are fine‑tuned from the same base model, consider `StructuredKalmanFuser` to capture shared errors.

### Need for Interpretability
`LearnedGateFuser` produces human‑readable weights (based on keyword presence) but requires training data.

## Configuration Examples

```python
from kalmanorix import Panoramix, KalmanorixFuser, MeanFuser, DiagonalKalmanFuser

# Kalman fusion (default)
fuser = KalmanorixFuser()
panoramix = Panoramix(village=village, router=router, fuser=fuser)

# Mean fusion (simple baseline)
mean_fuser = MeanFuser()
mean_panoramix = Panoramix(village=village, router=router, fuser=mean_fuser)

# Diagonal Kalman fusion (scalar variance)
diag_fuser = DiagonalKalmanFuser(prior_variance=1.0, prior_covariance=0.1)
diag_panoramix = Panoramix(village=village, router=router, fuser=diag_fuser)
```

## Tuning Parameters

### KalmanorixFuser
- `prior_variance`: Initial uncertainty (default 1.0). Increase for more conservative fusion.
- `prior_covariance`: Initial covariance (default 0.0). Set to non‑zero if you expect correlations.

### DiagonalKalmanFuser
- `prior_variance`: Shared prior variance across all dimensions.
- `process_variance`: Variance added between updates (prevents over‑confidence).

### LearnedGateFuser
- `feature_extractor`: Function mapping text to feature vector (default bag‑of‑words).
- `regularization`: L2 regularization strength.

*TODO: Add performance benchmarks, ablation studies, and tuning guidelines.*
