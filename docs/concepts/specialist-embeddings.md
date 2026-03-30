# Specialist Embeddings

*TODO: Explain domain‑specialist models and the SEF (Specialist Embedding Format)*

## What is a Specialist?

A **specialist** is a machine‑learning model trained (or fine‑tuned) on a specific domain (e.g., medical literature, legal documents, scientific papers). Because it sees only one type of data during training, it develops a rich, domain‑specific representation space that can capture nuances a generalist model might miss.

## Specialist Embedding Format (SEF)

Kalmanorix wraps each specialist in a **SEF** – a lightweight container that holds:

1. **Embedder function** – A callable `(str) → np.ndarray` that maps text to an embedding vector.
2. **Uncertainty estimator** – A callable `(str) → float` (or constant) that returns the variance (`sigma²`) for the embedding.
3. **Metadata** – Name, domain tags, dimensionality, alignment matrix (optional).

## Creating SEFs

You can create a SEF from:
- A simple Python function (for prototyping)
- A Hugging Face transformer model (via `HuggingFaceEmbedder`)
- Proprietary embedding APIs (OpenAI, Cohere, Anthropic, Vertex AI, Azure OpenAI)
- Sentence‑Transformers models

Example:
```python
from kalmanorix import SEF

def my_embedder(text: str) -> np.ndarray:
    ...

my_sef = SEF(name="my_specialist", embed=my_embedder, sigma2=0.1)
```

## Uncertainty in SEFs

Every SEF carries an uncertainty value (`sigma²`) that can be:
- **Constant** – Same for all queries (good for well‑calibrated models).
- **Query‑dependent** – Computed on the fly (e.g., based on distance to domain centroid).
- **Learned** – Predicted by a side‑car network (future work).

The uncertainty drives the Kalman gain: lower variance → higher weight in fusion.

## Domain Centroids

For semantic routing, each specialist can compute a **domain centroid** – the average embedding of a representative set of domain sentences. Centroids allow fast similarity comparison between a query and each specialist’s domain.

*TODO: Add diagrams of specialist vs generalist embeddings, SEF structure, and centroid illustration.*
