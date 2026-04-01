# Guide for Model Contributors

This guide explains how to create, benchmark, package, and share specialist embedding models for the Kalmanorix framework. By contributing models, you help build a diverse ecosystem of domain specialists that can be fused to tackle complex retrieval and classification tasks.

## Overview

Kalmanorix uses the **Shareable Embedding Format (SEF)** to package specialist models. A SEF model contains:

1. **Core embedder**: A function that converts text to embedding vectors
2. **Metadata**: Human‑readable information about the model (domain, benchmarks, licence, etc.)
3. **Uncertainty data**: Diagonal covariance (or low‑rank structured covariance) that quantifies the model's uncertainty for different inputs
4. **Alignment matrix** (optional): Procrustes matrix that maps the model's embedding space into a common reference space

Models are saved as directories with a standard file layout and can be loaded with a single call to `SEFModel.from_pretrained()`.

## Creating a SEF Model

### 1. Start with an embedder function

Any callable `(str) → np.ndarray` can be wrapped as a specialist. If you already have a model from a popular framework, use one of the built‑in adapters:

```python
from kalmanorix.embedder_adapters import (
    create_sentence_transformer_sef,
    create_openai_sef,
    create_cohere_sef,
    create_anthropic_sef,
    create_vertex_ai_sef,
    create_azure_openai_sef,
    create_huggingface_sef,
)

# Example: wrap a Sentence‑Transformers model
import sentence_transformers
model = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
sef = create_sentence_transformer_sef(
    model=model,
    model_id="tech-minilm-v1",
    name="Technology MiniLM",
    version="1.0.0",
    domain_tags=["technology", "software", "hardware"],
    task_tags=["semantic_search", "retrieval"],
    description="MiniLM‑L6‑v2 fine‑tuned on tech documentation",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    training_data_description="200k tech‑domain sentences from Stack Overflow, GitHub READMEs, and API docs",
    training_date="2025-12-01",
    author="Your Name/Organization",
    licence="MIT",
)
```

For a custom embedder, create a `SEFModel` directly:

```python
from kalmanorix.models.sef import SEFModel, SEFMetadata
import numpy as np

def my_embedder(text: str) -> np.ndarray:
    # Your implementation
    return ...

metadata = SEFMetadata(
    model_id="my-model-v1",
    name="My Custom Model",
    version="1.0.0",
    description="Custom embedder for my domain",
    domain_tags=["my-domain"],
    task_tags=["classification", "clustering"],
    benchmarks={"sts-b": 0.82, "my-dataset@1": 0.91},
    training_data_description="Proprietary dataset of 50k sentences",
    base_model="custom",
    training_date="2025-12-15",
    author="My Team",
    licence="Apache-2.0",
    embedding_dimension=384,
    covariance_format="diagonal",
    alignment_method="identity",
    checksum="",  # Will be filled automatically
)

sef = SEFModel(
    embed_function=my_embedder,
    metadata=metadata,
    alignment_matrix=None,  # Optional
    covariance_data={"method": "fixed", "diagonal": np.ones(384) * 0.1},
)
```

### 2. Add uncertainty estimation

Every specialist should provide a covariance estimate for each query. The simplest is a **fixed diagonal covariance** (same uncertainty for all inputs). More sophisticated methods include:

- **Distance‑based scaling**: Uncertainty increases with distance from a reference set (see `CentroidDistanceSigma2`)
- **Model‑based uncertainty**: Train a separate uncertainty predictor (future work)

Use the built‑in factory functions to add centroid‑distance uncertainty:

```python
from kalmanorix.embedder_adapters import create_sentence_transformer_sef_with_calibration

# This will compute reference centroids and calibrate alpha automatically
sef = create_sentence_transformer_sef_with_calibration(
    model=model,
    calibration_texts=[...],  # List of representative texts from your domain
    # ... plus all the metadata fields from above
)
```

For custom covariance, pass a `covariance_data` dict to `SEFModel`. The dict must contain:

- `method`: `"fixed"` or `"distance_based"`
- `diagonal`: base diagonal covariance (vector of length d)
- For `"distance_based"`, also include:
  - `reference_embeddings`: matrix of reference embeddings (n × d)
  - `alpha`: scaling factor (default 1.0)

### 3. Align to a reference space (optional)

If you want your model to be fused with other specialists that use different embedding spaces, you can compute a Procrustes alignment matrix that maps your embeddings into a common reference space.

```python
from kalmanorix.models.sef import create_procrustes_alignment

# Suppose you have paired embeddings: yours and reference versions
source_embs = np.array([sef.embed(t) for t in anchor_texts])
target_embs = np.array([reference_embedder(t) for t in anchor_texts])

alignment_matrix = create_procrustes_alignment(source_embs, target_embs)

# Create a new SEF with alignment
sef_with_alignment = SEFModel(
    embed_function=sef.embed_function,
    metadata=metadata,
    alignment_matrix=alignment_matrix,
    covariance_data=sef.covariance_data,
)
```

The alignment matrix will be automatically applied when the model is used in fusion.

## Benchmarking Your Model

Before sharing, evaluate your model on standard benchmarks and record the scores in the metadata. Kalmanorix provides a retrieval evaluation harness in `kalmanorix.arena`:

```python
from kalmanorix.arena import evaluate_retrieval

# Prepare a test corpus: list of (id, text) pairs
corpus = [(f"doc{i}", text) for i, text in enumerate(corpus_texts)]
# Queries: list of (query_id, query_text, relevant_doc_ids)
queries = [("q1", "example query", ["doc42", "doc87"]), ...]

results = evaluate_retrieval(
    embedder=sef.embed,
    corpus=corpus,
    queries=queries,
    metrics=["recall@1", "recall@5", "mrr"],
)

print(results)  # e.g., {"recall@1": 0.76, "recall@5": 0.92, "mrr": 0.84}
```

Also consider evaluating on standard datasets like STS‑B, SQuAD, or domain‑specific benchmarks. Add all scores to `metadata.benchmarks`:

```python
metadata.benchmarks = {
    "sts-b": 0.85,
    "squad-f1": 0.92,
    "mixed-domain-retrieval@1": 0.76,
    "my-domain-test@5": 0.93,
}
```

## Saving and Packaging

Once your model is ready, save it to disk:

```python
sef.save_pretrained("./my-specialist-model")
```

This creates a directory with the following structure:

```
my-specialist-model/
├── metadata.json          # Human‑readable model card (JSON)
├── model.pkl              # Pickled embed_function (if pickleable)
├── alignment.npy          # Alignment matrix (if provided)
├── covariance.npz         # Covariance arrays (if provided)
├── covariance_config.json # Covariance non‑array config
├── checksum.txt           # SHA‑256 checksum of all files
└── LOAD_INSTRUCTIONS.txt  # Instructions if embed_function couldn't be pickled
```

If your embedder cannot be pickled (e.g., it contains a TensorFlow/Keras model), the `LOAD_INSTRUCTIONS.txt` file will be created instead of `model.pkl`. In that case, users will need to provide an `embed_loader` callback when loading the model.

## Creating a Model Card

Every SEF model should include a detailed model card. Use the template at `docs/templates/model_card_template.md` and fill it with your model's information. Save it as `MODEL_CARD.md` in your model directory for completeness (though the essential metadata is already in `metadata.json`).

Key sections to complete:

- **Model Overview**: Name, version, description
- **Domain & Capabilities**: Domain tags, intended use, out‑of‑scope warnings
- **Performance Benchmarks**: Quantitative results on standard datasets
- **Technical Details**: Base model, training data, covariance format, alignment method
- **Model Provenance**: Author, licence, checksum
- **Limitations & Bias**: Known limitations, demographic or domain biases
- **Environmental Impact**: Optional estimate of training/inference energy

## Sharing Your Model

### Option 1: Share a directory (ZIP)

Package your model directory as a `.zip` or `.tar.gz` file and share it via:

- GitHub releases
- Personal website / cloud storage
- Model sharing platforms (Hugging Face Hub, etc.)

### Option 2: Hugging Face Hub

If your model is based on a Hugging Face transformer, you can push it directly to the Hub:

```python
from huggingface_hub import HfApi
import shutil

# Create a temporary directory with your SEF model
sef.save_pretrained("./tmp-model")

# Upload to Hugging Face Hub
api = HfApi()
api.upload_folder(
    folder_path="./tmp-model",
    repo_id="your-username/your-model-name",
    repo_type="model",
)
```

Users can then load it with:

```python
from kalmanorix.models.sef import SEFModel

def load_hf_embedder(path):
    # Custom loader for Hugging Face models
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    def embed(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    return embed

model = SEFModel.from_pretrained(
    "your-username/your-model-name",
    embed_loader=load_hf_embedder,
)
```

### Option 3: Kalmanorix Model Registry (future)

We plan to maintain a central registry of high‑quality specialist models. Contact the maintainers if you have a model that meets the quality criteria:

- Well‑documented model card
- Benchmark scores on at least two standard datasets
- Clear domain definition
- Open‑source licence (MIT, Apache‑2.0, etc.)
- Reproducible training procedure

## Best Practices

### 1. Domain specificity

- Choose a clear, narrow domain (e.g., "biomedical abstracts", "legal contracts", "product reviews")
- Avoid overly broad domains like "general text" – those are better served by monolithic models
- Use precise domain tags that will help the semantic router select your model appropriately

### 2. Uncertainty calibration

- Provide meaningful covariance estimates, not just `np.ones(d)`
- Use distance‑based uncertainty if you have a representative reference set
- Document how the uncertainty was estimated (fixed, empirical, distance‑based, etc.)

### 3. Benchmark transparency

- Report scores on standard public datasets whenever possible
- Include both in‑domain and out‑of‑domain performance
- If using proprietary datasets, describe them at a high level (size, source, pre‑processing)

### 4. Licensing

- Use a standard open‑source licence (MIT, Apache‑2.0, BSD‑3)
- If your model is based on a pre‑existing model, respect its licence terms
- Clearly state any use restrictions (commercial use, attribution requirements)

### 5. Documentation

- Fill every field in the metadata – don't leave placeholders
- Write a thorough model card that explains strengths, weaknesses, and appropriate use cases
- Include a minimal working example in the model card

## Example: Full Workflow

Here's a complete example of creating, benchmarking, and saving a specialist model:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from kalmanorix.embedder_adapters import create_sentence_transformer_sef_with_calibration
from kalmanorix.arena import evaluate_retrieval

# 1. Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Prepare calibration texts (representative of your domain)
calibration_texts = [
    "Python function for sorting lists",
    "JavaScript async await pattern",
    "Docker container configuration",
    # ... more tech‑domain sentences
]

# 3. Create SEF with calibrated uncertainty
sef = create_sentence_transformer_sef_with_calibration(
    model=model,
    calibration_texts=calibration_texts,
    model_id="tech-minilm-v1",
    name="Technology MiniLM Specialist",
    version="1.0.0",
    domain_tags=["technology", "programming", "software"],
    task_tags=["semantic_search", "code_search", "documentation"],
    description="MiniLM‑L6‑v2 fine‑tuned on programming documentation and Q&A",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    training_data_description="150k sentences from Stack Overflow, GitHub READMEs, and API documentation",
    training_date="2025-11-30",
    author="TechNLP Lab",
    licence="MIT",
    benchmarks={},  # Will fill after evaluation
)

# 4. Benchmark on a tech‑domain retrieval task
corpus = [...]  # Your corpus
queries = [...]  # Your queries
results = evaluate_retrieval(sef.embed, corpus, queries, metrics=["recall@1", "mrr"])

# 5. Update metadata with benchmark scores
sef.metadata.benchmarks = {
    "tech-retrieval-recall@1": results["recall@1"],
    "tech-retrieval-mrr": results["mrr"],
}

# 6. Save
sef.save_pretrained("./tech-minilm-specialist")

print(f"Model saved to ./tech-minilm-specialist")
print(f"Benchmark scores: {sef.metadata.benchmarks}")
```

## Troubleshooting

### Embedder cannot be pickled

If your embedder contains non‑pickleable objects (e.g., TensorFlow graphs, database connections), the `save_pretrained` method will create `LOAD_INSTRUCTIONS.txt` instead of `model.pkl`. To load such a model, users must provide an `embed_loader` function:

```python
def my_embed_loader(path):
    # Reconstruct your embedder here
    return embed_function

model = SEFModel.from_pretrained("./my-model", embed_loader=my_embed_loader)
```

### Large covariance data

If your covariance data is large (e.g., many reference embeddings), consider using low‑rank approximation (`covariance_format: "low_rank"`) to reduce storage. The `StructuredCovariance` class supports the representation `R = D + UUᵀ` where `U` is `d × k` with `k ≪ d`.

### Alignment matrix issues

If alignment causes degenerate results (e.g., all embeddings map to zero), check that:
- Your anchor texts are the same for source and target
- Both embedding spaces have the same dimension
- The Procrustes computation succeeded (determinant should be +1)

## Getting Help

If you encounter issues or have questions about contributing models:

- Open an issue on the [GitHub repository](https://github.com/its-not-rocket-science/kalmanorix)
- Check existing examples in the `examples/` directory
- Review the API documentation for `SEFModel` and `SEFMetadata`

Thank you for contributing to the Kalmanorix specialist ecosystem! Your models enable more accurate and efficient fusion for diverse applications.
