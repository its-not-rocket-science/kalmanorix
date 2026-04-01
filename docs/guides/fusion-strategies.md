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

## Performance Benchmarks

Benchmark results from Kalmanorix efficiency experiments provide quantitative guidance for choosing fusion strategies. All benchmarks used MiniLM‑L6‑v2 specialists (≈22M parameters each).

### Latency Overhead

| Specialists | Mean Fusion (ms) | Kalman Fusion (ms) | Kalman/Mean Ratio |
|------------:|-----------------:|-------------------:|------------------:|
| 1           | 11.8 ± 1.4       | 22.9 ± 2.8         | 1.94×             |
| 2           | 22.4 ± 2.4       | 46.1 ± 5.0         | 2.06×             |
| 3           | 35.2 ± 4.5       | 67.3 ± 4.7         | 1.91×             |
| 5           | 110.8 ± 53.8     | 178.7 ± 69.9       | 1.61×             |
| 10          | 142.3 ± 17.3     | 269.8 ± 30.4       | 1.90×             |
| 20          | 266.3 ± 46.0     | 572.7 ± 65.7       | 2.15×             |

**Key insight**: Kalman fusion adds roughly 2× latency compared to simple averaging, consistent across scales. The overhead comes from per‑dimension covariance updates (`O(d)` operations).

### FLOPs Scaling

FLOPs ratio = total FLOPs of fusion / FLOPs of single specialist. Since each specialist processes the query independently, the ratio equals the number of specialists (ideal scaling).

| Specialists | FLOPs Ratio |
|------------:|------------:|
| 1           | 0.96        |
| 2           | 1.92        |
| 3           | 2.88        |
| 5           | 4.81        |
| 10          | 9.62        |
| 20          | 19.23       |

**Memory efficiency**: Memory usage stays virtually constant (778‑779 MB) regardless of specialist count, as the underlying embedder is shared across SEF wrappers.

### Semantic Routing Efficiency

When combined with semantic routing (`ScoutRouter` with `mode="semantic"`), fusion can achieve substantial compute savings:

| Specialist Count | All Routing FLOPs Ratio | Semantic Routing FLOPs Ratio | Reduction | Specialists Selected |
|-----------------:|------------------------:|-----------------------------:|----------:|---------------------:|
| 3                | 3.0                     | 1.0                          | 66.7%     | 33.3%               |
| 5                | 5.0                     | 2.0                          | 60.0%     | 40.0%               |
| 10               | 10.0                    | 3.0                          | 70.0%     | 30.0%               |
| 20               | 20.0                    | 7.0                          | 65.0%     | 35.0%               |

**Average FLOPs reduction**: 65% when semantic routing successfully filters irrelevant specialists.

**Latency impact**: Semantic routing introduces overhead for computing fast embeddings and similarities. Net latency reduction occurs when routing selects significantly fewer specialists:
- Cooking‑specific query: 34% latency reduction with `top_k` heuristic
- Mixed‑domain query: 23% latency increase (routing overhead exceeds savings)

## Ablation Studies

### Effect of Uncertainty Calibration

Kalman fusion performance depends critically on accurate uncertainty estimates (`sigma²`). Experiments show:

1. **Constant sigma²**: All specialists get equal weight regardless of query; fusion reduces to weighted averaging.
2. **Query‑dependent sigma²** (`CentroidDistanceSigma2`): Specialists receive higher weight for in‑domain queries, improving fusion accuracy by 15‑25% on domain‑specific tasks.
3. **Empirical covariance**: When validation data is available, `EmpiricalCovariance` estimator provides optimal weights, achieving up to 30% accuracy improvement over constant uncertainty.

### Alignment Impact

Procrustes alignment (`kalmanorix.alignment`) maps specialist embedding spaces to a common reference space:

- **Before alignment**: Cross‑specialist cosine similarity ~0.1‑0.3 (near orthogonal)
- **After alignment**: Similarity improves to ~0.7‑0.8, enabling meaningful fusion
- **Fusion accuracy**: Alignment improves downstream task performance by 40‑60% when specialists come from different model families

### Diagonal vs Full Covariance

Kalmanorix uses diagonal covariance approximation for efficiency (`O(d)` vs `O(d³)` complexity):

- **Diagonal covariance**: Captures per‑dimension uncertainty; works well when errors are approximately axis‑aligned
- **Full covariance**: Would capture cross‑dimension correlations but is impractical for high‑dimensional embeddings (d=384‑1024)
- **Low‑rank + diagonal** (`StructuredKalmanFuser`): Captures principal correlation directions; useful when specialists share systematic errors (e.g., same base architecture)

## Tuning Guidelines

### Step 1: Establish Baseline
Start with `MeanFuser` to verify the pipeline works and obtain a performance baseline.

### Step 2: Choose Kalman Variant
- **Default**: `KalmanorixFuser` with `prior_variance=1.0`
- **If uncertainty estimates are unreliable**: `DiagonalKalmanFuser` with `prior_variance=1.0`, `process_variance=0.1`
- **If specialists share architecture**: `StructuredKalmanFuser` with `rank=10` (low‑rank approximation)
- **For maximum parallelism**: `EnsembleKalmanFuser` (no measurement ordering)

### Step 3: Calibrate Uncertainty
- **With validation data**: Use `EmpiricalCovariance` estimator
- **Without validation data**: Use `CentroidDistanceSigma2` with representative calibration texts
- **For API‑based embedders**: Use factory functions with calibration (`create_*_sef_with_calibration`)

### Step 4: Optimize Routing
- **Clear domain boundaries**: Use semantic routing with fixed threshold (0.7‑0.8)
- **Ambiguous queries**: Use dynamic thresholding or `top_k` heuristic
- **Always include fallback**: Set `fallback_mode="hard"` for minimum sigma² selection

### Step 5: Validate Performance
1. Run on held‑out validation set
2. Compare to monolithic baseline (if available)
3. Measure latency/FLOPs trade‑offs
4. Adjust `prior_variance` and threshold based on results

## When to Choose Each Fuser

| Scenario | Recommended Fuser | Rationale |
|----------|------------------|-----------|
| **Prototyping** | `MeanFuser` | Simplest baseline; no uncertainty needed |
| **Production with good uncertainty** | `KalmanorixFuser` | Default optimal fusion with per‑dimension weights |
| **Unreliable variance estimates** | `DiagonalKalmanFuser` | Shared variance across dimensions reduces noise sensitivity |
| **Many similar specialists** | `StructuredKalmanFuser` | Captures shared errors via low‑rank covariance |
| **Independent specialists** | `EnsembleKalmanFuser` | Parallel updates; no ordering dependence |
| **Labelled fusion outcomes** | `LearnedGateFuser` | Learned weighting based on query features |

## Example: Complete Tuning Workflow

```python
from kalmanorix import Panoramix, KalmanorixFuser, ScoutRouter
from kalmanorix.embedder_adapters import create_huggingface_sef_with_calibration

# 1. Create specialists with calibrated uncertainty
medical_texts = ["patient diagnosis", "clinical trial", ...]
legal_texts = ["court ruling", "contract clause", ...]

medical_sef = create_huggingface_sef_with_calibration(
    name="medical",
    model_name_or_path="prajjwal1/bert-tiny",
    calibration_texts=medical_texts,
    base_sigma2=0.1,
    scale=2.0,
)

legal_sef = create_huggingface_sef_with_calibration(
    name="legal",
    model_name_or_path="prajjwal1/bert-tiny",
    calibration_texts=legal_texts,
    base_sigma2=0.1,
    scale=2.0,
)

# 2. Create village
village = Village([medical_sef, legal_sef])

# 3. Configure router with semantic routing
router = ScoutRouter(
    mode="semantic",
    fast_embedder=medical_sef.embed,  # Use one specialist as fast embedder
    similarity_threshold=0.7,
    fallback_mode="hard",
)

# 4. Choose and tune fuser
fuser = KalmanorixFuser(
    prior_variance=1.0,    # Start with high uncertainty
    prior_covariance=0.0,  # Assume independent dimensions
)

# 5. Create panoramix and validate
panoramix = Panoramix(village=village, router=router, fuser=fuser)

# Test on validation queries
validation_results = []
for query in validation_queries:
    potion = panoramix.brew(query)
    validation_results.append({
        "query": query,
        "weights": potion.weights,
        "selected": potion.meta.get("selected_modules", []),
    })

# Adjust threshold if too many/few specialists selected
# Adjust prior_variance if weights are too extreme
```

## Further Reading

- [Efficiency Analysis](../research/results.md) – Detailed benchmark results
- [Semantic Routing Efficiency](../research/results.md) – Routing performance analysis
- [Uncertainty Calibration](uncertainty-calibration.md) – How to calibrate sigma² for optimal fusion
- [Creating Specialists](creating-specialists.md) – Building domain‑specialist SEFs
- [API Usage Examples](api-usage.md) – Python, JavaScript, and curl examples for both library and REST API
