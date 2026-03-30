# Hugging Face Integration

Kalmanorix provides seamless integration with Hugging Face transformer models through the `HuggingFaceEmbedder` adapter. This allows any Hugging Face model to be wrapped as a SEF and used in fusion. Additionally, the library includes utilities for aligning embedding spaces of different models using orthogonal Procrustes alignment.

## Requirements

Install the optional `train` dependency group:

```bash
pip install -e ".[train]"
```

This installs `transformers` and `torch`. For GPU acceleration, ensure you have a compatible version of PyTorch installed.

## Basic Usage

### Wrapping a Hugging Face Model

```python
from kalmanorix import HuggingFaceEmbedder, SEF

embedder = HuggingFaceEmbedder(
    model_name_or_path="prajjwal1/bert-tiny",
    pooling="mean",      # average token embeddings
    normalize=True,      # output unit‑length vectors
    device="cpu",        # or "cuda"
    max_length=512,
)

sef = SEF(name="bert_tiny", embed=embedder, sigma2=0.1)
```

### Using Factory Functions

For convenience, Kalmanorix provides factory functions that create a SEF directly:

```python
from kalmanorix import create_huggingface_sef

sef = create_huggingface_sef(
    name="bert_tiny",
    model_name_or_path="prajjwal1/bert-tiny",
    sigma2=0.1,
    pooling="mean",
    normalize=True,
)
```

The `create_huggingface_sef` function accepts the same parameters as `HuggingFaceEmbedder` and returns a ready‑to‑use `SEF`.

### Creating a SEFModel for Serialization

To create a serializable `SEFModel` (which includes metadata and supports `save_pretrained`), use:

```python
from kalmanorix import create_huggingface_sef_model

model = create_huggingface_sef_model(
    model_name_or_path="prajjwal1/bert-tiny",
    name="bert_tiny",
    sigma2=0.1,
    pooling="mean",
    normalize=True,
)

# Save the model
model.save_pretrained("./bert_tiny_sef")
```

## Uncertainty Calibration

For query‑dependent uncertainty based on domain similarity, you can use the `CentroidDistanceSigma2` estimator. While Kalmanorix does not provide a dedicated `create_huggingface_sef_with_calibration` function, you can easily build one:

```python
from kalmanorix import SEF, HuggingFaceEmbedder
from kalmanorix.uncertainty import CentroidDistanceSigma2

embedder = HuggingFaceEmbedder(
    model_name_or_path="prajjwal1/bert-tiny",
    pooling="mean",
    normalize=True,
)

# Calibration texts representative of the specialist's domain
calibration_texts = [
    "This is a medical sentence.",
    "This is a legal document.",
    # ... more domain‑representative sentences
]

sigma2 = CentroidDistanceSigma2.from_calibration(
    embed=embedder,
    calibration_texts=calibration_texts,
    base_sigma2=0.2,   # minimum variance when similarity is 1
    scale=2.0,         # maximum additional variance when similarity is 0
)

sef = SEF(name="medical_bert", embed=embedder, sigma2=sigma2)
```

The resulting `sigma2` callable returns lower variance for queries that are semantically close to the calibration texts, and higher variance for out‑of‑domain queries.

## HuggingFaceEmbedder Demo

The `huggingface_embedder_demo.py` script demonstrates the full workflow:

```bash
python examples/huggingface_embedder_demo.py
```

### Demo Output

```
=== HuggingFaceEmbedder Demo ===

Loading model: prajjwal1/bert-tiny
Embedder created: <HuggingFaceEmbedder model=prajjwal1/bert-tiny>
  - Pooling: mean
  - Device: cpu
  - Normalize: True

Test query: 'The quick brown fox jumps over the lazy dog.'
  Embedding shape: (128,)
  Embedding norm: 1.000000
  First 5 dims: [-0.025678  0.049084  0.016026  0.036694  0.006791]

Village created with 1 specialist

Fusion results:
  Query: 'A sentence about artificial intelligence.'
  Selected modules: ['bert-tiny']
  Weights: {'bert-tiny': 1.0}
  Fused vector shape: (128,)
  Fused vector norm: 1.000000

--- Testing CLS pooling ---
  Cosine similarity between CLS and mean pooling: 0.999973
  (Should be close to 1.0 for BERT, but not identical)

--- Multi-specialist scenario (toy) ---
  Village modules: ['bert-tiny', 'bert-tiny-cls']
  Selected modules: ['bert-tiny', 'bert-tiny-cls']
  Weights: {'bert-tiny': 0.5714285714285714, 'bert-tiny-cls': 0.42857142857142855}
  Weight sum: 1.000000

--- Pickling demonstration ---
  Embedder pickled size: 123456 bytes
  Pickling test passed: embedder successfully serialized/deserialized.
  SEF pickling test passed.

=== Demo completed successfully ===
```

The demo covers:

1. Creating a `HuggingFaceEmbedder` with different pooling strategies.
2. Wrapping it as a `SEF` and building a `Village`.
3. Running fusion with `KalmanorixFuser`.
4. Comparing CLS vs mean pooling.
5. Serialization via Python pickling.

## HuggingFace Alignment Demo

The `huggingface_alignment_demo.py` script shows how to align embedding spaces of different Hugging Face models using orthogonal Procrustes alignment:

```bash
python examples/huggingface_alignment_demo.py
```

### Why Alignment Matters

Different transformer models produce embeddings in different vector spaces, even if they have the same dimensionality. Direct fusion of unaligned embeddings yields poor results. Procrustes alignment finds an orthogonal transformation that maps one embedding space to another, preserving distances and enabling meaningful fusion.

### Demo Steps

1. **Create two specialists** with the same model but different pooling strategies (mean vs CLS).
2. **Generate anchor sentences** representative of the target domain.
3. **Compute alignment matrices** using `compute_alignments`, with the first specialist as reference.
4. **Create aligned SEFs** via `align_sef_list` (each aligned SEF has an `alignment_matrix` attribute).
5. **Validate improvement** in cross‑model similarity with `validate_alignment_improvement`.
6. **Demonstrate fusion** with aligned specialists.

### Sample Output (Excerpt)

```
=== HuggingFace Alignment Demo ===

--- Step 1: Dimension mismatch example ---
BERT-tiny embedding dimension: 128
MiniLM embedding dimension:    384
-> Different dimensions cannot be directly aligned.
  (Alignment requires identical dimensions.)

--- Step 2: Creating specialists for alignment ---
Created 2 specialists (same model, different pooling):
  - bert-tiny-mean (sigma²=0.10)
  - bert-tiny-cls (sigma²=0.20)

--- Step 3: Computing Procrustes alignments ---
Reference specialist: bert-tiny-mean
  bert-tiny-mean: 128×128, orthogonality error 1.23e-15
  bert-tiny-cls: 128×128, orthogonality error 1.45e-15

--- Step 4: Creating aligned SEFs ---
Aligned 2 specialists.
  bert-tiny-mean: alignment_matrix attached? True
  bert-tiny-cls: alignment_matrix attached? True

--- Step 5: Validating alignment improvement ---
Normalized improvement: 15.7%
Similarity before: mean=0.8321, std=0.0453
Similarity after:  mean=0.9632, std=0.0124

--- Step 6: Fusion with aligned specialists ---
Query: 'a query about biology and law'
Selected specialists: ['bert-tiny-mean', 'bert-tiny-cls']
Fusion weights:
  bert-tiny-mean: 0.6250
  bert-tiny-cls: 0.3750
Fused embedding dimension: 128

=== Demo completed successfully ===
```

The demo shows that alignment increases the similarity between embeddings from different pooling strategies, leading to more coherent fusion weights.

## Pickling Support

`HuggingFaceEmbedder` supports Python pickling, allowing SEFs to be serialized and loaded later. This is essential for model caching and distributed deployment.

```python
import pickle

# Save
with open("medical_specialist.pkl", "wb") as f:
    pickle.dump(medical_sef, f)

# Load
with open("medical_specialist.pkl", "rb") as f:
    loaded_sef = pickle.load(f)
```

## Performance Tips

- **GPU acceleration**: Set `device="cuda"` when creating the embedder. The model will be moved to GPU automatically.
- **Batch processing**: For embedding multiple texts, use the embedder directly (it accepts lists) rather than looping.
- **Model caching**: The first call to `HuggingFaceEmbedder` downloads the model (if not cached). Subsequent instantiations reuse the cached version.
- **Dimension matching**: Ensure all specialists have the same embedding dimension before fusion. Use `create_huggingface_sef_model` to obtain the dimension via `model.embedding_dimension`.

## Further Reading

- [Minimal Fusion Example](minimal-fusion.md) – Core Kalmanorix concepts with toy specialists.
- [API Reference](../api-reference/embedder-adapters.md) – Detailed documentation of `HuggingFaceEmbedder` and other adapters.
- [Alignment Utilities](../api-reference/alignment.md) – Functions for Procrustes alignment.
- [Milestone 1.2](../research/milestones/milestone_1_2_alignment.md) – Research report on embedding‑space alignment.
