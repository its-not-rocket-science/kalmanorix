# Quickstart

Get started with Kalmanorix in 5 minutes. This guide walks you through creating your first fusion pipeline.

## Basic Concepts

Kalmanorix revolves around a few key concepts:

- **SEF (Specialist Embedding Format)**: A wrapper around an embedder function with uncertainty
- **Village**: Container for available SEFs
- **ScoutRouter**: Selects which specialists to consult for a query
- **Panoramix**: High-level fusion orchestrator
- **Fuser**: Strategy for combining embeddings (mean, Kalman, etc.)

## Step 1: Create Toy Specialists

Let's create simple keyword-based specialists for demonstration:

```python
import numpy as np
from kalmanorix import SEF, Village

# Simple keyword-sensitive embedders
def medical_embedder(text: str) -> np.ndarray:
    """Embeds medical text with bias toward medical dimension."""
    vec = np.random.randn(10)  # 10-dimensional embeddings
    if any(kw in text.lower() for kw in ["patient", "disease", "treatment"]):
        vec[0] += 2.0  # Boost medical dimension
    return vec / np.linalg.norm(vec)  # Normalize

def legal_embedder(text: str) -> np.ndarray:
    """Embeds legal text with bias toward legal dimension."""
    vec = np.random.randn(10)
    if any(kw in text.lower() for kw in ["court", "law", "contract"]):
        vec[1] += 2.0  # Boost legal dimension
    return vec / np.linalg.norm(vec)

# Create SEFs with constant uncertainty
medical_sef = SEF(name="medical", embed=medical_embedder, sigma2=0.1)
legal_sef = SEF(name="legal", embed=legal_embedder, sigma2=0.2)

# Create a village (container for specialists)
village = Village(sefs=[medical_sef, legal_sef])
```

## Step 2: Create a Fusion Pipeline

Now let's create a fusion pipeline using `Panoramix`:

```python
from kalmanorix import Panoramix, ScoutRouter, KalmanorixFuser

# Create a router that selects all specialists (for fusion)
router = ScoutRouter(village=village, mode="all")

# Create a Kalman fuser
fuser = KalmanorixFuser()

# Create the fusion orchestrator
panoramix = Panoramix(village=village, router=router, fuser=fuser)
```

## Step 3: Fuse Embeddings

Fuse embeddings for different types of queries:

```python
# Medical query
medical_query = "Patient with influenza needs treatment"
result = panoramix.brew(medical_query)
print(f"Medical query - fused embedding shape: {result.embedding.shape}")
print(f"Weights: {result.weights}")  # Higher weight for medical specialist

# Legal query
legal_query = "Court ruled on contract dispute"
result = panoramix.brew(legal_query)
print(f"Legal query - fused embedding shape: {result.embedding.shape}")
print(f"Weights: {result.weights}")  # Higher weight for legal specialist

# Mixed query
mixed_query = "Medical malpractice lawsuit settlement"
result = panoramix.brew(mixed_query)
print(f"Mixed query - fused embedding shape: {result.embedding.shape}")
print(f"Weights: {result.weights}")  # Both specialists contribute
```

## Step 4: Try Different Fusion Strategies

Compare different fusion strategies:

```python
from kalmanorix import MeanFuser, DiagonalKalmanFuser

# Mean fusion (simple averaging)
mean_fuser = MeanFuser()
mean_panoramix = Panoramix(village=village, router=router, fuser=mean_fuser)
mean_result = mean_panoramix.brew(medical_query)
print(f"Mean fusion weights: {mean_result.weights}")

# Diagonal Kalman fusion (scalar variance)
diagonal_fuser = DiagonalKalmanFuser()
diagonal_panoramix = Panoramix(village=village, router=router, fuser=diagonal_fuser)
diagonal_result = diagonal_panoramix.brew(medical_query)
print(f"Diagonal Kalman weights: {diagonal_result.weights}")
```

## Step 5: Batch Processing

Process multiple queries efficiently:

```python
queries = [
    "Patient diagnosis report",
    "Legal contract review",
    "Medical legal consultation"
]

# Batch fusion
batch_results = panoramix.brew_batch(queries)
for i, result in enumerate(batch_results):
    print(f"Query {i}: {queries[i][:30]}...")
    print(f"  Weights: {result.weights}")
```

## Step 6: Semantic Routing

Use semantic routing to select only relevant specialists:

```python
from kalmanorix import threshold_top_k

# Create router with semantic routing
semantic_router = ScoutRouter(
    village=village,
    mode="semantic",
    threshold_fn=threshold_top_k(k=1)  # Select top specialist
)

# Create fusion pipeline with routing
routing_panoramix = Panoramix(
    village=village,
    router=semantic_router,
    fuser=fuser
)

# Medical query will only use medical specialist
result = routing_panoramix.brew(medical_query)
print(f"Selected specialists: {result.selected_sef_names}")
print(f"Effective FLOPs reduction: Using {len(result.selected_sef_names)}/{len(village)} specialists")
```

## Next Steps

- Explore the [examples](examples.md) for more advanced use cases
- Learn about [creating specialists](../guides/creating-specialists.md) with real models
- Understand [uncertainty calibration](../guides/uncertainty-calibration.md) for better fusion
- Check out the [API reference](../api-reference/village.md) for detailed documentation
