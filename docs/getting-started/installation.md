# Installation

Kalmanorix requires Python ≥3.11. We recommend using a virtual environment.

## Using pip

Install the latest stable release from PyPI:

```bash
pip install kalmanorix
```

## Development Installation

For development or to use the latest features:

1. Clone the repository:
   ```bash
   git clone https://github.com/its-not-rocket-science/kalmanorix.git
   cd kalmanorix
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

3. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Optional Dependencies

Kalmanorix has several optional dependency groups:

- **`dev`**: Development tools (pytest, ruff, mypy, etc.)
  ```bash
  pip install -e ".[dev]"
  ```

- **`viz`**: Visualization (matplotlib)
  ```bash
  pip install -e ".[viz]"
  ```

- **`train`**: Training dependencies (sentence-transformers, torch, transformers)
  ```bash
  pip install -e ".[train]"
  ```

- **`cloud`**: Cloud API clients (OpenAI, Cohere, Anthropic, Vertex AI)
  ```bash
  pip install -e ".[cloud]"
  ```

- **`api`**: FastAPI server dependencies
  ```bash
  pip install -e ".[api]"
  ```

- **`docs`**: Documentation tools (mkdocs, mkdocs-material)
  ```bash
  pip install -e ".[docs]"
  ```

You can combine multiple groups:
```bash
pip install -e ".[dev,train,cloud]"
```

## Verification

Verify your installation:

```python
import kalmanorix
print(f"Kalmanorix version: {kalmanorix.__version__}")
```

## Troubleshooting

### Common Issues

**`ImportError: No module named 'numpy'`**
- Ensure you have installed the package: `pip install -e "."`

**CUDA/GPU support for training**
- Install PyTorch with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Missing API dependencies**
- Install the appropriate optional dependency group: `pip install -e ".[cloud]"` or `pip install -e ".[api]"`

## Next Steps

- [Quickstart](quickstart.md) - Run your first fusion pipeline
- [Examples](examples.md) - Explore available examples
