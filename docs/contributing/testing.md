# Testing Guide

This guide covers the testing philosophy, test structure, and best practices for writing tests for Kalmanorix.

## Testing Philosophy

Tests in Kalmanorix are written with pytest and focus on four key areas:

1. **Invariants**: Mathematical properties of the fusion algorithm
   - Certainty never decreases after incorporating a new measurement
   - Fusion weights sum to 1 (for convex combination strategies)
   - Output embedding dimension matches input dimension

2. **Smoke tests**: Basic import and instantiation of all public classes
   - Every exported class can be instantiated with minimal configuration
   - Core workflows (create Village → add SEFs → fuse) execute without errors

3. **Demo outputs**: Golden tests that ensure examples produce consistent output
   - The minimal fusion demo produces the same weights across runs
   - Example scripts in the documentation work as shown

4. **Edge cases**: Boundary conditions and error handling
   - Zero or negative variances
   - Empty village (no specialists)
   - Single specialist (trivial fusion)
   - Invalid input queries

Run `pytest` after any change to ensure all invariants hold.

## Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── test_smoke.py           # Basic import and instantiation tests
├── test_village.py         # Village and SEF functionality
├── test_panoramix.py       # Panoramix orchestrator and Fusers
├── test_scout.py           # ScoutRouter routing logic
├── test_kalman_engine.py   # Core Kalman fusion algorithms
├── test_embedder_adapters.py  # Third-party embedder adapters
├── test_uncertainty.py     # Uncertainty estimators
├── test_alignment.py       # Procrustes alignment utilities
├── test_fastapi_server.py  # FastAPI server integration
└── test_structured_covariance.py  # Low-rank covariance tests
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_smoke.py

# Run tests matching a pattern
pytest -k "village"

# Run tests with coverage reporting
pytest --cov=src/kalmanorix
```

### Test Environment Setup
Tests are designed to run without external dependencies (no network calls, no GPU). Third‑party embedder adapters are mocked in tests to avoid requiring API keys or internet access.

## Writing New Tests

### Test Conventions
- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Use pytest fixtures for shared test data
- Mark integration tests with `@pytest.mark.integration`
- Mark slow tests with `@pytest.mark.slow`

### Example Test Structure
```python
import pytest
import numpy as np
from kalmanorix import Village, SEF

def test_village_add_sef():
    """Adding a SEF to a Village increases its size."""
    village = Village()
    embedder = lambda x: np.zeros(10)
    sef = SEF(embedder, sigma2=1.0, name="test")

    village.add_sef(sef)
    assert len(village) == 1
    assert "test" in village.sef_names
```

### Testing Mathematical Invariants
When testing Kalman fusion algorithms, verify properties like:

```python
def test_kalman_certainty_non_decreasing():
    """Certainty (inverse variance) never decreases after fusion."""
    # Arrange: create measurements with variances
    # Act: perform Kalman update
    # Assert: posterior certainty >= prior certainty
    pass
```

### Testing Edge Cases
Always test boundary conditions:

```python
def test_fusion_empty_village():
    """Fusion with empty village raises appropriate error."""
    village = Village()
    with pytest.raises(ValueError, match="No specialists"):
        # Attempt fusion
        pass
```

## Integration Tests

Integration tests verify that components work together correctly. These tests are marked with `@pytest.mark.integration` and may have additional dependencies.

### Running Integration Tests
```bash
# Run only integration tests
pytest -m integration

# Run all tests except integration
pytest -m "not integration"
```

## Mocking External Dependencies

When testing components that depend on external services (OpenAI, Cohere, etc.), use pytest mocking:

```python
from unittest.mock import Mock, patch

def test_openai_embedder_adapter():
    """OpenAIEmbedder adapter calls the OpenAI API with correct parameters."""
    with patch("openai.OpenAI") as mock_openai:
        # Set up mock response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 768)]
        )

        # Test the adapter
        adapter = OpenAIEmbedder(api_key="fake", model="text-embedding-3-small")
        result = adapter("test query")
        assert len(result) == 768
```

## Pre‑commit Hooks and CI

### Pre‑commit Hooks
The project includes a pre‑commit hook that runs tests before pushing:
```bash
pre-commit install  # Install hooks
pre-commit run --all-files  # Run hooks manually
```

The `pre-push` hook runs `pytest` to ensure tests pass before code is pushed to the repository.

### Continuous Integration
GitHub Actions CI runs on every push and pull request:
1. Installs dependencies
2. Runs `pytest` with coverage
3. Reports test results

Check `.github/workflows/ci.yml` for the complete CI configuration.

## Test Data

### Synthetic Test Data
The `kalmanorix.toy_corpus` module provides synthetic data for testing:
- `create_toy_corpus()`: Creates a small corpus of sentences with domain labels
- `create_toy_embedder()`: Creates a deterministic embedder for testing

### Real Data for Integration Tests
Integration tests may use small real datasets (e.g., PubMed abstracts, legal texts) stored in `tests/data/`. These files are not included in the repository but are downloaded on demand.

## Performance Testing

Performance tests measure:
- **Latency**: End‑to‑end fusion time for different specialist counts
- **Memory**: Peak memory usage during batch fusion
- **FLOPs**: Computational cost of fusion algorithms

Run performance tests with:
```bash
pytest tests/ -m "performance" --benchmark
```

## Debugging Test Failures

### Common Issues
1. **Numerical precision**: Use `pytest.approx()` for floating‑point comparisons
2. **Random seeds**: Tests should be deterministic; set `np.random.seed(42)`
3. **Import errors**: Ensure all dependencies are installed

### Debugging Tools
```bash
# Run tests with debug output
pytest -v --tb=short

# Run a single test with pdb
pytest tests/test_village.py::test_specific_function -xvs --pdb

# Generate HTML coverage report
pytest --cov=src/kalmanorix --cov-report=html
```

## Best Practices

1. **Test behavior, not implementation**: Focus on what the code does, not how it does it
2. **Keep tests fast**: Avoid unnecessary I/O, network calls, or heavy computation
3. **Make tests independent**: Each test should set up its own state
4. **Use descriptive test names**: Test function names should describe what they test
5. **Test error cases**: Verify that appropriate exceptions are raised
6. **Document test assumptions**: Use docstrings to explain what each test verifies

## Next Steps

- [Development Guide](development.md) – General development setup and workflow
- [API Reference](../api-reference/village.md) – Detailed API documentation
- [Roadmap](roadmap.md) – Project milestones and future work
