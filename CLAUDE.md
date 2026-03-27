# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kalmanorix is a research framework for fusing embeddings from multiple domain-specialist models using Kalman filtering. The core hypothesis is that a fused ensemble of specialists can outperform monolithic models while being more computationally efficient (KEFF: Kalman Ensemble of Fusion-Frugal specialists).

Key concepts:
- **SEF (Specialist Embedding Format)**: A wrapper around an embedder function with associated uncertainty (sigma²). Can have constant or query-dependent uncertainty.
- **Village**: Container for available SEFs at runtime.
- **ScoutRouter**: Selects which specialists to consult for a given query (`mode="all"` for fusion, `mode="hard"` for single specialist).
- **Panoramix**: High‑level fusion orchestrator that combines routing with a `Fuser` strategy and returns a `Potion`.
- **Potion**: Result container holding the fused embedding, per‑module weights, and metadata.
- **Fuser**: Strategy for combining embeddings (e.g., `MeanFuser`, `KalmanorixFuser`, `DiagonalKalmanFuser`, `LearnedGateFuser`).
- **Kalman Fuser**: Core algorithm (`kalman_fuse_diagonal`) that performs sequential updates with diagonal covariance approximation (O(d) complexity).

Phase 1 (Core Algorithm Validation) is complete. Phase 2: validating the core hypothesis that fused specialists outperform monolithic models with compute equivalence. Milestone 2.1 (Specialists vs Monolith) and Milestone 2.2 (Uncertainty Robustness) are completed.

## Development Setup

The project uses Python ≥3.11 with dependencies managed via `pyproject.toml`. Development dependencies include pytest, pylint, ruff, black, mypy, and pre-commit.

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. Install the package in development mode with all dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

   For training experiments (Milestone 2+), also install the optional `train` group:
   ```bash
   pip install -e ".[dev,train]"
   ```

## Common Commands

### Running Tests
- Run all tests: `pytest`
- Run a specific test file: `pytest tests/test_smoke.py`
- Run with verbose output: `pytest -v`
- Run with coverage: `pytest --cov=src/kalmanorix`

### Linting & Code Quality
- Format code with ruff: `ruff format src tests`
- Lint with ruff: `ruff check src tests`
- Type checking with mypy: `mypy src tests`
- Lint with pylint: `pylint src/kalmanorix tests`

### Running Examples
- Run the minimal fusion demo: `python examples/minimal_fusion_demo.py`
  - Creates toy keyword‑sensitive specialists and compares fusion strategies.
- Run the Hugging Face embedder demo: `python examples/huggingface_embedder_demo.py`
  - Shows how to wrap a Hugging Face transformer model as a specialist.
  - Requires optional `train` dependencies (`pip install -e ".[train]"`).

### Running Milestone 2.1 Experiment (Specialists vs Monolith)
- Install training dependencies: `pip install -e ".[train]"`
- Generate example configurations: `python experiments/create_configs.py`
- Run full experiment with default config: `python experiments/run_milestone_2_1.py`
- Run with custom config: `python experiments/run_milestone_2_1.py --config experiments/configs/milestone_2_1.yaml`
- Train specialists only: `python experiments/train_specialists_st.py --config experiments/configs/milestone_2_1.yaml`
- Train monolith only: `python experiments/train_monolith.py --config experiments/configs/milestone_2_1.yaml`
- Generate test set: `python experiments/generate_test_set.py --config experiments/configs/milestone_2_1.yaml`

### Pre‑commit Hooks
Pre‑commit hooks run ruff (formatting and linting), mypy, pylint, and basic file checks. They are configured in `.pre-commit-config.yaml`. To install and run:
```bash
pre-commit install  # install hooks
pre-commit run --all-files  # run on all files
```
The hooks also include a local `pytest` hook that runs on `pre-push` for fast feedback.

### CI Pipeline
GitHub Actions CI (`.github/workflows/ci.yml`) runs on push/pull request and executes:
```bash
pip install -e ".[dev]"
pytest
```

## Architecture

### Package Structure
- `src/kalmanorix/` – main package
  - `__init__.py` – public API (Village, ScoutRouter, Panoramix, Potion, MeanFuser, KalmanorixFuser, LearnedGateFuser)
  - `village.py` – SEF and Village container
  - `scout.py` – ScoutRouter for model selection
  - `panoramix.py` – High‑level fusion orchestration with `Fuser` abstraction and `Potion` result type
  - `kalman_engine/` – Core Kalman fusion algorithms
    - `kalman_fuser.py` – Core Kalman update with per‑dimension diagonal covariance (`kalman_fuse_diagonal`)
    - `covariance.py` – Uncertainty estimation strategies (`CovarianceEstimator`)
    - `fuser.py` – New `Panoramix` orchestrator using the Kalman engine directly (not yet exported)
    - `__init__.py` – Exports core algorithms
  - `models/sef.py` – SEFModel (Shareable Embedding Format with save/load capabilities)
  - `sef_io.py` – I/O utilities for SEF (future)
  - `uncertainty.py` – Uncertainty helpers (`KeywordSigma2`, `CentroidDistanceSigma2`)
  - `embedder_adapters.py` – Adapters for third‑party embedders (Sentence‑Transformers, OpenAI, Cohere, Anthropic, Vertex AI, Azure OpenAI, Hugging Face Transformers)
  - `arena.py` – Evaluation utilities (retrieval benchmark)
  - `types.py` – Shared type definitions (`Embedder` protocol, `Vec`)
  - `toy_corpus.py` – Synthetic data for testing
- `tests/` – unit and integration tests
- `examples/` – demonstration scripts (`minimal_fusion_demo.py`)

### API Layers
- **High‑level API (Public)**: The `kalmanorix` module exports `Panoramix` (from `panoramix.py`) which uses the `Fuser` abstraction. Available fusers:
  - `MeanFuser` – uniform averaging
  - `KalmanorixFuser` – True Kalman fusion with diagonal covariance (uses `kalman_fuse_diagonal`)
  - `EnsembleKalmanFuser` – parallel Kalman updates for diagonal covariance (uses `kalman_fuse_diagonal_ensemble`)
  - `StructuredKalmanFuser` – Kalman fusion with low‑rank covariance (uses `kalman_fuse_structured`)
  - `DiagonalKalmanFuser` – scalar Kalman update (shared prior variance across dimensions)
  - `LearnedGateFuser` – learned two‑way gating (logistic regression on bag‑of‑words features)
  The `Fuser` base class provides both single-query (`fuse`) and batch (`fuse_batch`) methods. `Panoramix` provides corresponding `brew` and `brew_batch` methods. The API returns a `Potion` object (or list of `Potion`s for batch) containing fused embedding, per‑module weights, and optional metadata.
- **Low‑level Kalman engine (Internal)**: `kalman_engine.kalman_fuser` implements core algorithms:
  - `kalman_fuse_diagonal` – sequential updates with diagonal covariance
  - `kalman_fuse_diagonal_ensemble` – parallel updates for diagonal covariance
  - `kalman_fuse_structured` – sequential updates with structured (diagonal + low‑rank) covariance
  - `kalman_fuse_diagonal_batch`, `kalman_fuse_diagonal_ensemble_batch` – batch versions for multiple queries
  These are used by the corresponding fusers.
- **Covariance estimation**: `kalman_engine.covariance` provides `CovarianceEstimator` base class with implementations (`EmpiricalCovariance`, `DistanceBasedCovariance`).

### Key Design Patterns
- **Protocol‑based embedders**: Any callable `(str) -> np.ndarray` can be wrapped as an SEF.
- **Diagonal covariance**: The Kalman filter uses only diagonal covariance matrices, making fusion O(d) instead of O(d³).
- **Sequential updates**: Measurements are sorted by certainty (lowest variance first) for numerical stability.
- **Batch fusion**: Both low‑level algorithms and high‑level `Fuser` abstraction support batch processing of multiple queries for efficiency.
- **Pluggable components**: Uncertainty estimation (`CovarianceEstimator`), alignment methods, and routing strategies are configurable.
- **Fuser abstraction**: Fusion strategies are decoupled from routing and orchestration via the `Fuser` base class.

### Current Status
- **Phase 1 (Core Algorithm Validation)**: Completed
  - ✓ Diagonal covariance estimation framework (Milestone 1.1)
  - ✓ Procrustes alignment for embedding-space unification (Milestone 1.2)
  - ✓ Basic Kalman fusion against averaging baselines (Milestone 1.3)
- **Phase 2 (Specialists vs Monolith)**: In progress
  - Milestone 2.1: Specialists vs monolith test with compute equivalence
  - Real dataset integration (PubMed, legal case law)
  - Compute tracking (FLOPs, energy)
- **Remaining limitations**:
  - Serialization (`SEFModel.save_pretrained`) is implemented (Milestone 3.1)
  - No production-scale specialist models included

## Testing Philosophy

Tests are written with pytest and focus on:
- **Invariants**: Mathematical properties of the fusion algorithm (e.g., certainty never decreases).
- **Smoke tests**: Basic import and instantiation of all public classes.
- **Demo outputs**: Golden tests that ensure the example in the README produces the same output.
- **Edge cases**: Zero/negative variances, empty village, single specialist.

Run `pytest` after any change to ensure all invariants hold.

## Contributing

When adding new functionality:
1. Follow existing type annotations (mypy strict).
2. Add unit tests for both success and error cases.
3. Update the public API in `src/kalmanorix/__init__.py` if needed.
4. Ensure `pytest`, `ruff`, `mypy`, and `pylint` pass before pushing.

The roadmap (`README.md`) outlines the planned milestones; contributions should align with the current phase.

## Commit Messages

Use descriptive commit messages that explain *why* changes were made, not just *what* changed. Focus on the purpose and impact.

**Important**: Do NOT add "Co-Authored-By:" lines or any other automatic attribution tags to commit messages. Commit messages should only contain the actual change description.

Good patterns:
- **Feature additions**: "Add X for Y" (e.g., "Add EmpiricalCovariance estimator for diagonal covariance estimation")
- **Bug fixes**: "Fix X in Y" (e.g., "Fix return type in DiagonalKalmanFuser._kalman_updates")
- **Refactoring**: "Refactor X to improve Y" (e.g., "Refactor imports for better type checking")
- **Documentation**: "Update docs for X" (e.g., "Update README with latest roadmap")
- **Chores**: "chore: X" for maintenance tasks (e.g., "chore: update dependencies")

For milestone work, reference the milestone in the commit message (e.g., "[Milestone 1.1] Add covariance estimation framework").

Keep messages concise but informative. If a commit touches multiple related changes, summarize the theme rather than listing every file.
