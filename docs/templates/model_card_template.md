# SEF Model Card Template

This is a model card template for Kalmanorix Specialist Embedding Format (SEF) models. Fill in each section with information about your specialist model.

## Model Overview

**Model ID:** `[unique identifier, e.g., "tech-minilm-v1"]`

**Name:** `[human-readable name]`

**Version:** `[semantic version, e.g., "1.0.0"]`

**Description:** `[Brief description of the model's purpose and domain]`

**Embedding Dimension:** `[integer, e.g., 384]`

## Domain & Capabilities

**Domain Tags:** `[list of domains, e.g., ["technology", "software", "hardware"]]`

**Task Tags:** `[list of tasks, e.g., ["semantic_search", "retrieval", "classification"]]`

**Intended Use:** `[Describe the intended use cases]`

**Out-of-Scope Use:** `[Describe use cases where the model should not be used]`

## Performance Benchmarks

Report benchmark scores on standard datasets. Use the format `{"dataset": score}`.

```json
{
  "sts-b": 0.85,
  "squad-f1": 0.92,
  "mixed-domain-retrieval@1": 0.76
}
```

## Technical Details

**Base Model:** `[e.g., "sentence-transformers/all-MiniLM-L6-v2"]`

**Training Data Description:** `[High-level description of training data, no raw data]`

**Training Date:** `[YYYY-MM-DD]`

**Covariance Format:** `["diagonal", "low_rank", or "full"]`

**Alignment Method:** `["procrustes", "identity", "learned"]`

**Uncertainty Estimation Method:** `["fixed", "distance_based", "model_based"]`

## Model Provenance

**Author:** `[Name or organization]`

**License:** `[e.g., "MIT", "Apache-2.0"]`

**Checksum (SHA-256):** `[auto-generated when saving]`

## Model Files

When saved using `SEFModel.save_pretrained()`, the model directory contains:

- `metadata.json` - This model card information
- `model.pkl` - Pickled embed function (if pickleable)
- `alignment.npy` - Alignment matrix (if exists)
- `covariance.npz` - Covariance data (if exists)
- `covariance_config.json` - Covariance configuration (if exists)
- `checksum.txt` - SHA-256 checksum of all files
- `LOAD_INSTRUCTIONS.txt` - Instructions if embed function couldn't be pickled

## Usage Example

```python
from kalmanorix.models.sef import SEFModel

# Load the model
model = SEFModel.from_pretrained("./path/to/model")

# Get embedding
embedding = model.embed("example text")

# Get uncertainty covariance
covariance = model.get_covariance("example text")
```

## Limitations & Bias

`[Describe known limitations, biases, and ethical considerations]`

## Environmental Impact

`[Optional: Estimate of training/inference energy consumption]`

## Citation

If you use this model, please cite:

```bibtex
[Your citation here]
```

## Contact

`[Contact information for model maintainers]`

---

*This model card follows the Kalmanorix SEF specification v0.1.*
