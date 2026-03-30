# Examples

Kalmanorix comes with several example scripts that demonstrate different aspects of the framework. These examples are designed to be self-contained and easy to run.

## Available Examples

### 1. Minimal Fusion Demo
**File**: `examples/minimal_fusion_demo.py`

A lightweight, dependency-free demonstration of Kalmanorix core concepts. Creates toy keyword‑sensitive specialists and compares fusion strategies:

- Hard routing (choose a single specialist)
- Mean fusion (uniform averaging)
- KalmanorixFuser (uncertainty‑weighted fusion)
- LearnedGateFuser (tiny learned gating baseline)

**Run it**:
```bash
python examples/minimal_fusion_demo.py
```

**Key learning points**:
- How to create custom embedders with uncertainty
- How to set up a `Village` of specialists
- How different fusion strategies produce different weights for the same query

### 2. Procrustes Alignment Demo
**File**: `examples/procrustes_alignment_demo.py`

Demonstrates orthogonal Procrustes alignment between embedding spaces of different specialists. Shows how alignment improves cross‑model similarity.

**Run it**:
```bash
python examples/procrustes_alignment_demo.py
```

**Key learning points**:
- Why alignment is necessary when specialists come from different base models
- How to compute and apply alignment matrices
- How to measure alignment improvement

### 3. Hugging Face Embedder Demo
**File**: `examples/huggingface_embedder_demo.py`

Shows how to wrap a Hugging Face transformer model as a specialist using the `HuggingFaceEmbedder` adapter.

**Requirements**: Install the `train` optional dependency group:
```bash
pip install -e ".[train]"
```

**Run it**:
```bash
python examples/huggingface_embedder_demo.py
```

**Key learning points**:
- How to use pre‑trained transformer models as specialists
- How to configure embedding extraction (pooling, normalization)
- How to save/load SEF models with `SEFModel.save_pretrained()`

### 4. Hugging Face Alignment Demo
**File**: `examples/huggingface_alignment_demo.py`

Demonstrates Procrustes alignment between different Hugging Face transformer models.

**Requirements**: Install the `train` optional dependency group.

**Run it**:
```bash
python examples/huggingface_alignment_demo.py
```

**Key learning points**:
- Alignment between heterogeneous transformer architectures
- Practical considerations for real‑world specialist fusion

### 5. FastAPI Server
**File**: `examples/fastapi_server.py`

A production‑ready FastAPI server that exposes Kalmanorix fusion as a REST API. Supports batch queries, multiple fusion strategies, and health checks.

**Requirements**: Install the `api` optional dependency group:
```bash
pip install -e ".[api]"
```

**Run it**:
```bash
python examples/fastapi_server.py
```

Then visit `http://localhost:8000/docs` for the interactive API documentation.

**Key learning points**:
- How to deploy Kalmanorix as a microservice
- API design patterns for embedding fusion
- Production considerations (timeouts, error handling)

### 6. Interactive Demo
**File**: `examples/create_interactive_demo.py`

Generates an interactive Jupyter notebook that lets you experiment with different fusion strategies in real time.

**Requirements**: Install the `viz` optional dependency group:
```bash
pip install -e ".[viz]"
```

**Run it**:
```bash
python examples/create_interactive_demo.py
```

This creates a `demo.ipynb` notebook that you can open in Jupyter.

**Key learning points**:
- Interactive exploration of fusion behavior
- Visualizing embedding spaces and uncertainty
- Real‑time parameter tuning

## Running Examples

All examples are located in the `examples/` directory of the repository. Most examples have minimal dependencies beyond the base Kalmanorix installation.

### Dependency Groups

Some examples require optional dependencies:

| Example | Required Dependency Group | Purpose |
|---------|---------------------------|---------|
| Hugging Face demos | `train` | Transformer models, PyTorch |
| FastAPI Server | `api` | FastAPI, Uvicorn |
| Interactive Demo | `viz` | Matplotlib, Jupyter |

Install them with:
```bash
pip install -e ".[group1,group2]"
```

### Common Issues

**`ModuleNotFoundError` for `transformers` or `torch`**
- Install the `train` group: `pip install -e ".[train]"`

**`ModuleNotFoundError` for `fastapi` or `uvicorn`**
- Install the `api` group: `pip install -e ".[api]"`

**CUDA/GPU support**
- Install PyTorch with CUDA separately if needed

## Creating Your Own Examples

See the [Creating Specialists guide](../guides/creating-specialists.md) for instructions on building custom specialists and fusion pipelines.

## Next Steps

- [Quickstart](quickstart.md) – Run your first fusion pipeline
- [API Reference](../api-reference/village.md) – Explore the full API
- [Guides](../guides/creating-specialists.md) – Learn advanced techniques
