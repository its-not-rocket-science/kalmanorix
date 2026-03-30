# Minimal Fusion Example

The `minimal_fusion_demo.py` script demonstrates the core Kalmanorix concepts using simple, deterministic toy specialists. It is designed to be dependency‑light and fully reproducible, making it an ideal starting point for understanding how fusion works.

## Overview

The demo creates two keyword‑sensitive specialists:

- **Tech specialist**: Boosts its embedding when the query contains technology keywords (`battery`, `smartphone`, `cpu`, `gpu`, `laptop`, `android`, `ios`, `camera`, `charger`).
- **Cooking specialist**: Boosts its embedding when the query contains cooking keywords (`braise`, `simmer`, `slow cooker`, `recipe`, `garlic`, `onion`, `saute`, `oven`, `stew`).

Each specialist returns a 16‑dimensional embedding with a deterministic base plus small perturbations. Uncertainty is implemented via `KeywordSigma2`: variance increases when the query lacks domain keywords, making the specialist less certain about out‑of‑domain queries.

## Running the Demo

```bash
python examples/minimal_fusion_demo.py
```

Output shows the fused embedding weights for different fusion strategies on a mixed‑domain query:

```
Query: This smartphone battery lasts longer than a slow cooker braise.

Hard routing (sigma2 baseline)
  weights: { tech: 1.000, cook: 0.000 }

Mean fusion
  weights: { tech: 0.500, cook: 0.500 }

KalmanorixFuser (precision-weighted by sigma2)
  weights: { tech: 0.920, cook: 0.080 }

LearnedGateFuser (learned text gate)
  weights: { tech: 0.876, cook: 0.124 }
```

The demo also prints cosine similarities between the fused vectors, showing how different strategies produce different embeddings.

## Fusion Strategies Compared

### 1. Hard Routing (`mode="hard"`)
Selects exactly one specialist (the one with lowest sigma² for the query). No fusion occurs—the chosen specialist’s embedding is returned as‑is.

### 2. Mean Fusion (`MeanFuser`)
Uniform averaging of all selected specialists’ embeddings. Both specialists receive equal weight regardless of query content.

### 3. KalmanorixFuser
Uncertainty‑weighted fusion using diagonal covariance. Specialists that are more certain (lower variance) receive higher weight. In the example, the tech specialist has lower uncertainty for the mixed query because it contains tech keywords, giving it ~92% weight.

### 4. LearnedGateFuser
A tiny logistic‑regression gating model trained on bag‑of‑words features. It learns to predict which specialist should be emphasized based on text content, without explicit uncertainty estimates.

## Code Walkthrough

### Creating Keyword‑Sensitive Embedders

```python
from kalmanorix import SEF, Village, ScoutRouter, Panoramix, MeanFuser, KalmanorixFuser, LearnedGateFuser
from kalmanorix.types import Embedder, Vec
from kalmanorix.uncertainty import KeywordSigma2

# Define keyword sets
tech_keywords = {"battery", "smartphone", "cpu", "gpu", "laptop", "android", "ios", "camera", "charger"}
cook_keywords = {"braise", "simmer", "slow cooker", "recipe", "garlic", "onion", "saute", "oven", "stew"}

# Create SEFs with KeywordSigma2 uncertainty
tech = SEF(
    name="tech",
    embed=make_keyword_embedder(dim=16, seed=7, keywords=tech_keywords),
    sigma2=KeywordSigma2(tech_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5),
    meta={"domain": "tech"},
)
cook = SEF(
    name="cook",
    embed=make_keyword_embedder(dim=16, seed=11, keywords=cook_keywords),
    sigma2=KeywordSigma2(cook_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5),
    meta={"domain": "cooking"},
)
```

### Building the Village

```python
village = Village([tech, cook])
```

### Configuring Routing and Fusion

```python
scout_all = ScoutRouter(mode="all")   # select all specialists for fusion
scout_hard = ScoutRouter(mode="hard") # select single specialist

# Hard routing baseline
hard = Panoramix(fuser=MeanFuser())
potion_hard = hard.brew(query, village=village, scout=scout_hard)

# Mean fusion baseline
mean = Panoramix(fuser=MeanFuser())
potion_mean = mean.brew(query, village=village, scout=scout_all)

# Kalmanorix fusion (precision‑weighted)
kal = Panoramix(fuser=KalmanorixFuser())
potion_kal = kal.brew(query, village=village, scout=scout_all)
```

### Training a Learned Gate

```python
# LearnedGateFuser trains a logistic regression on bag‑of‑words features
gate_fuser = LearnedGateFuser(
    module_a="tech",
    module_b="cook",
    n_features=128,
    lr=0.6,
    l2=1e-3,
    steps=400,
)

# Tiny labeled dataset: 1 => tech, 0 => cooking
train_texts = [
    "Battery life is excellent on this smartphone",
    "The laptop CPU throttles under load",
    "Camera quality and charger compatibility",
    "Android update improved performance",
    "Braise the beef and simmer for hours",
    "Slow cooker recipe with garlic and onion",
    "Saute the vegetables then bake in the oven",
    "Stew tastes better after simmering",
]
train_y = [1, 1, 1, 1, 0, 0, 0, 0]

gate_fuser.fit(train_texts, train_y)
gate = Panoramix(fuser=gate_fuser)
potion_gate = gate.brew(query, village=village, scout=scout_all)
```

## Key Observations

- **Kalman fusion** gives higher weight to the specialist whose domain matches the query, because that specialist reports lower uncertainty for domain‑relevant keywords.
- **Mean fusion** treats both specialists equally, regardless of query content.
- **Learned gating** can approximate the keyword‑based weighting without explicit uncertainty estimates, showing how learned routing can complement Kalman fusion.

## Extending the Example

Try modifying the demo to:

- Add a third specialist (e.g., `financial` with keywords `bank`, `stock`, `market`).
- Change the uncertainty function to `CentroidDistanceSigma2`.
- Implement a custom `Fuser` that blends mean and Kalman strategies.
- Experiment with different query types and observe how weights change.

The full source is at [`examples/minimal_fusion_demo.py`](https://github.com/its-not-rocket-science/kalmanorix/blob/main/examples/minimal_fusion_demo.py).
