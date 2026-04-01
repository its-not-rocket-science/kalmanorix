# Creating Specialists

A step‑by‑step guide to building domain‑specialist embedding models and wrapping them as SEFs for use in Kalmanorix.

## Overview

A **specialist** is an embedding model tailored to a specific domain (e.g., biomedical literature, legal contracts, academic papers). In Kalmanorix, specialists are wrapped as **SEFs** (Specialist Embedding Format) – a lightweight container that pairs an embedder function with an uncertainty estimate (`sigma2`).

This guide walks through:
1. Choosing a base model
2. Creating a SEF using factory functions
3. Adding query‑dependent uncertainty via calibration
4. Computing domain centroids for semantic routing
5. Saving and loading specialists

## Step 1: Choose a Base Model

You can create a specialist from any embedder that maps text to a fixed‑size vector. Kalmanorix provides adapters for several popular sources:

| Source | Example Models | Factory Function |
|--------|----------------|------------------|
| **Hugging Face Transformers** | `prajjwal1/bert‑tiny`, `bert‑base‑uncased`, `roberta‑large` | `create_huggingface_sef()` |
| **OpenAI Embedding API** | `text‑embedding‑3‑small`, `text‑embedding‑3‑large` | `create_openai_sef()` |
| **Cohere Embedding API** | `embed‑english‑v3.0`, `embed‑multilingual‑v3.0` | `create_cohere_sef()` |
| **Google Vertex AI** | `text‑embedding‑005`, `text‑multilingual‑embedding‑002` | `create_vertexai_sef()` |
| **Azure OpenAI** | `text‑embedding‑3‑small` (Azure deployment) | `create_azure_openai_sef()` |
| **TF‑IDF (fast local)** | Custom vocabulary from domain corpus | `create_tfidf_sef()` |

**Recommendations:**
- For research/prototyping: Use small Hugging Face models (e.g., `prajjwal1/bert‑tiny`, `sentence‑transformers/all‑MiniLM‑L6‑v2`).
- For production with high accuracy: Use API‑based embedders (OpenAI, Cohere, Vertex AI).
- For latency‑sensitive applications: Use TF‑IDF or tiny transformer specialists.

## Step 2: Create a SEF with Constant Uncertainty

The simplest way to create a specialist is with a constant uncertainty (`sigma2`). Higher `sigma2` means higher variance (less certainty).

### Hugging Face Model Example

```python
from kalmanorix import create_huggingface_sef

medical_sef = create_huggingface_sef(
    name="medical_bert",
    model_name_or_path="prajjwal1/bert‑tiny",
    sigma2=0.1,           # constant variance
    pooling="mean",       # or "cls"
    device="cpu",         # or "cuda"
    normalize=True,
)
```

### OpenAI API Example

Requires `openai` package and an API key.

```python
import os
from kalmanorix import create_openai_sef

# Set API key via environment variable
os.environ["OPENAI_API_KEY"] = "sk‑..."

openai_sef = create_openai_sef(
    name="openai_embedder",
    model="text‑embedding‑3‑small",
    sigma2=0.05,
    dimensions=256,       # optional output dimensionality
)
```

### Cohere API Example

```python
import os
from kalmanorix import create_cohere_sef

os.environ["CO_API_KEY"] = "..."

cohere_sef = create_cohere_sef(
    name="cohere_embedder",
    model="embed‑english‑v3.0",
    input_type="search_document",  # "search_query", "classification", "clustering"
    sigma2=0.1,
)
```

### TF‑IDF Example (Fast, No External Dependencies)

```python
from kalmanorix import create_tfidf_sef

# Calibration texts define the vocabulary
calibration_texts = [
    "patient diagnosis report",
    "medical treatment plan",
    "clinical trial results",
    # ... more domain‑specific sentences
]

tfidf_sef = create_tfidf_sef(
    name="medical_tfidf",
    calibration_texts=calibration_texts,
    sigma2=0.3,
    max_features=500,      # vocabulary size
    stop_words="english",  # remove common words
)
```

## Step 3: Add Query‑Dependent Uncertainty

Constant uncertainty treats all queries equally. For better fusion, you can use `CentroidDistanceSigma2`, which increases variance for queries that are semantically distant from the specialist's domain.

### Using Calibration Texts

Each factory function has a `_with_calibration` variant that builds a `CentroidDistanceSigma2` automatically:

```python
from kalmanorix import create_openai_sef_with_calibration

calibration_texts = [
    "Patient presents with fever and cough",
    "MRI shows abnormal lesion in left lung",
    "Prescribed antibiotics for bacterial infection",
    # ... 10‑50 representative sentences
]

medical_sef = create_openai_sef_with_calibration(
    name="medical_openai",
    model="text‑embedding‑3‑small",
    calibration_texts=calibration_texts,
    base_sigma2=0.1,   # variance when similarity = 1 (identical to centroid)
    scale=2.0,         # additional variance when similarity = 0
)
```

The resulting `sigma2` is a callable: `sigma2(query)` returns lower variance for queries similar to the calibration texts, higher variance for out‑of‑domain queries.

### Manual CentroidDistanceSigma2

You can also construct the uncertainty function directly:

```python
from kalmanorix import SEF, HuggingFaceEmbedder
from kalmanorix.uncertainty import CentroidDistanceSigma2

embedder = HuggingFaceEmbedder("prajjwal1/bert‑tiny")
calibration_texts = [...]  # domain‑representative sentences

# Compute centroid from calibration texts
centroid_embeddings = [embedder(t) for t in calibration_texts]
centroid = np.mean(centroid_embeddings, axis=0)
centroid = centroid / np.linalg.norm(centroid)  # normalize

sigma2_fn = CentroidDistanceSigma2(
    centroid=centroid,
    base_sigma2=0.1,
    scale=2.0,
)

medical_sef = SEF(name="medical", embed=embedder, sigma2=sigma2_fn)
```

## Step 4: Compute Domain Centroid for Semantic Routing

Semantic routing (ScoutRouter with `mode="semantic"`) selects specialists based on cosine similarity between the query embedding and each specialist's **domain centroid**. You can attach a centroid to an existing SEF:

```python
# Calibration texts that define the specialist's domain
domain_texts = [
    "medical journal article about cancer research",
    "clinical trial protocol for new drug",
    "patient electronic health record",
    # ...
]

# Compute and attach centroid
medical_sef_with_centroid = medical_sef.with_domain_centroid(domain_texts)
```

The centroid is stored in `medical_sef_with_centroid.domain_centroid` and will be used automatically by semantic routing.

**Tip:** Use the same calibration texts for uncertainty and centroid computation for consistency.

## Step 5: Save and Load Specialists

### Pickling (Simple)

Any SEF can be pickled:

```python
import pickle

# Save
with open("medical_specialist.pkl", "wb") as f:
    pickle.dump(medical_sef, f)

# Load
with open("medical_specialist.pkl", "rb") as f:
    loaded_sef = pickle.load(f)
```

### SEFModel Format (Recommended)

For sharing and versioning, use `SEFModel`, which stores metadata, alignment matrices, and covariance estimators:

```python
from kalmanorix import create_huggingface_sef_model

model = create_huggingface_sef_model(
    model_name_or_path="prajjwal1/bert‑tiny",
    name="medical_bert",
    sigma2=0.1,
    metadata={
        "domain": "medical",
        "author": "Your Name",
        "license": "Apache‑2.0",
    },
)

# Save to directory
model.save_pretrained("./medical_bert_sef")

# Load later
from kalmanorix.models.sef import SEFModel
loaded_model = SEFModel.from_pretrained("./medical_bert_sef")
sef = loaded_model.to_sef()  # Convert back to SEF
```

The directory contains:
- `metadata.json` – human‑readable information
- `embedder.pkl` – pickled embedder (optional)
- `alignment.npy` – Procrustes alignment matrix (if aligned)
- `covariance.npz` – covariance estimator parameters
- `checksum.txt` – SHA‑256 for integrity verification

## Step 6: Add to Village

Once you have one or more SEFs, create a `Village`:

```python
from kalmanorix import Village

village = Village(sefs=[medical_sef, legal_sef, tech_sef])
print(f"Village has {len(village)} specialists: {[s.name for s in village.modules]}")
```

The village is ready for use with `Panoramix` fusion.

## Best Practices

1. **Domain Representation**: Calibration texts should cover the breadth of your specialist's domain. Include 50‑200 representative sentences.
2. **Uncertainty Calibration**: Validate that `sigma2(query)` correlates with actual error on a held‑out validation set.
3. **Centroid Quality**: Domain centroids computed from too few texts may not generalize; use at least 20‑30 sentences.
4. **Model Size**: Balance accuracy vs. memory/latency. For fusion, smaller specialists can work well because errors average out.
5. **Testing**: Test each specialist individually before fusion to ensure embeddings are well‑behaved (norm ≈ 1, no NaN values).

## Next Steps

- Learn about [fusion strategies](fusion-strategies.md) to combine multiple specialists.
- Understand [uncertainty calibration](uncertainty-calibration.md) for improving fusion weights.
- Explore [semantic routing](../api-reference/scout-router.md) to select only relevant specialists per query.
- Try the [HuggingFace integration example](../examples/huggingface-integration.md) for a complete workflow.
- See [API Usage Examples](api-usage.md) for Python, JavaScript, and curl examples.
