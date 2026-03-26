"""
Tests for model_registry module.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from kalmanorix.model_registry import (
    ModelRegistry,
    get_default_registry,
    set_default_registry,
)
from kalmanorix.models.sef import SEFModel, SEFMetadata


def dummy_embedder(text: str) -> np.ndarray:
    return np.ones(4, dtype=np.float64) * len(text)


def test_registry_init(tmp_path: Path):
    """Test ModelRegistry initialization."""
    registry = ModelRegistry(tmp_path)
    assert registry.base_dir == tmp_path
    assert isinstance(registry.embedder_registry.embedders, dict)
    assert len(registry.embedder_registry.embedders) == 0


def test_scan_empty_directory(tmp_path: Path):
    """Test scanning empty directory returns no models."""
    registry = ModelRegistry(tmp_path)
    model_ids = registry.scan()
    assert not model_ids
    assert registry.list_models() == {}


def test_scan_invalid_metadata(tmp_path: Path):
    """Test scanning directory with invalid metadata.json is skipped."""
    model_dir = tmp_path / "bad_model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text("{ invalid json", encoding="utf-8")

    registry = ModelRegistry(tmp_path)
    model_ids = registry.scan()
    assert not model_ids
    assert registry.list_models() == {}


def test_scan_valid_model(tmp_path: Path):
    """Test scanning directory with valid SEF artefact."""
    # Create a minimal SEF model directory
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    metadata = SEFMetadata(
        model_id="test_model",
        name="Test Model",
        version="1.0.0",
        description="Test description",
        domain_tags=["test"],
        task_tags=["embedding"],
        benchmarks={"test": 0.9},
        training_data_description="Synthetic",
        base_model="test",
        training_date="2026-01-01",
        author="Test",
        licence="MIT",
        embedding_dimension=4,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="dummy",
    )

    # Save metadata
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.to_json())

    # Create a dummy embedder pickle (optional)
    # Not required for scanning

    registry = ModelRegistry(tmp_path)
    model_ids = registry.scan()
    assert model_ids == ["test_model"]

    models = registry.list_models()
    assert "test_model" in models
    assert models["test_model"]["name"] == "Test Model"
    assert models["test_model"]["domain_tags"] == ["test"]


def test_load_model(tmp_path: Path):
    """Test loading a SEF model."""
    # Use module-level dummy embedder
    embedder = dummy_embedder

    # Create SEF model and save it
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()

    metadata = SEFMetadata(
        model_id="dummy_model",
        name="Dummy Model",
        version="1.0.0",
        description="Dummy model for testing",
        domain_tags=["dummy"],
        task_tags=["embedding"],
        benchmarks={},
        training_data_description="Synthetic",
        base_model="dummy",
        training_date="2026-01-01",
        author="Test",
        licence="MIT",
        embedding_dimension=4,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="dummy",
    )

    model = SEFModel(
        embed_function=dummy_embedder,
        metadata=metadata,
        alignment_matrix=None,
        covariance_data={"method": "fixed", "diagonal": np.ones(4)},
    )

    model.save_pretrained(model_dir)

    # Now test registry loading
    registry = ModelRegistry(tmp_path)
    registry.scan()

    loaded_model = registry.load_model("dummy_model")
    assert isinstance(loaded_model, SEFModel)
    assert loaded_model.metadata.name == "Dummy Model"

    # Test embedder retrieval
    embedder = registry.get_embedder("dummy_model")
    vec = embedder("hello")
    assert vec.shape == (4,)
    assert np.allclose(vec, np.ones(4) * 5)  # len("hello") = 5

    # Verify embedder is cached in registry
    assert "dummy_model" in registry.embedder_registry.embedders


def test_register_custom_embedder(tmp_path: Path):
    """Test registering custom embedder functions."""
    registry = ModelRegistry(tmp_path)

    def custom_embedder(text: str) -> np.ndarray:
        return np.zeros(8)

    registry.register_embedder("custom", custom_embedder)

    # Should be retrievable via get_embedder
    embedder = registry.get_embedder("custom")
    vec = embedder("test")
    assert vec.shape == (8,)
    assert np.all(vec == 0)

    # Should also be in embedder registry
    assert registry.embedder_registry.get("custom") is custom_embedder


def test_get_metadata_nonexistent(tmp_path: Path):
    """Test getting metadata for non-existent model raises KeyError."""
    registry = ModelRegistry(tmp_path)
    with pytest.raises(KeyError):
        registry.get_metadata("missing")


def test_default_registry():
    """Test default global registry functions."""
    # Get default registry (creates on first call)
    reg1 = get_default_registry()
    assert isinstance(reg1, ModelRegistry)

    # Should be same instance on second call
    reg2 = get_default_registry()
    assert reg2 is reg1

    # Can replace with set_default_registry
    new_reg = ModelRegistry(Path("/tmp/test"))
    set_default_registry(new_reg)
    assert get_default_registry() is new_reg

    # Restore original (for other tests)
    set_default_registry(reg1)


def test_lazy_loading(tmp_path: Path):
    """Test that models are loaded lazily, not during scan."""
    # Create model directory with metadata but no model.pkl
    # This simulates a model that cannot be pickled (requires custom loader)
    model_dir = tmp_path / "lazy_model"
    model_dir.mkdir()

    metadata = SEFMetadata(
        model_id="lazy_model",
        name="Lazy Model",
        version="1.0.0",
        description="Lazy model for testing",
        domain_tags=["lazy"],
        task_tags=["embedding"],
        benchmarks={},
        training_data_description="Synthetic",
        base_model="lazy",
        training_date="2026-01-01",
        author="Test",
        licence="MIT",
        embedding_dimension=4,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="dummy",
    )

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.to_json())

    # Create LOAD_INSTRUCTIONS.txt to simulate unpicklable embedder
    (model_dir / "LOAD_INSTRUCTIONS.txt").write_text(
        "This model requires custom loading.", encoding="utf-8"
    )

    registry = ModelRegistry(tmp_path)
    model_ids = registry.scan()
    assert "lazy_model" in model_ids

    # Loading should fail without embed_loader
    with pytest.raises(ValueError, match="No pickled model found"):
        registry.load_model("lazy_model")

    # But we can still get metadata
    meta = registry.get_metadata("lazy_model")
    assert meta["name"] == "Lazy Model"


@patch("kalmanorix.models.sef.SEFModel")
def test_scan_with_mock_model(MockSEFModel, tmp_path: Path):
    """Test scanning with mocked SEFModel to verify interaction."""
    model_dir = tmp_path / "mock_model"
    model_dir.mkdir()

    # Create minimal metadata
    metadata = {
        "model_id": "mock_model",
        "name": "Mock Model",
        "version": "1.0.0",
        "domain_tags": ["mock"],
        "task_tags": ["embedding"],
        "benchmarks": {},
        "training_data_description": "Mock",
        "base_model": "mock",
        "training_date": "2026-01-01",
        "author": "Mock",
        "licence": "MIT",
        "embedding_dimension": 4,
        "covariance_format": "diagonal",
        "alignment_method": "identity",
        "checksum": "mock",
    }

    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    registry = ModelRegistry(tmp_path)
    model_ids = registry.scan()
    assert model_ids == ["mock_model"]

    # Mock SEFModel.from_pretrained to return a mock model
    mock_model = Mock()
    mock_model.embed = Mock(return_value=np.ones(4))
    MockSEFModel.from_pretrained.return_value = mock_model

    # Load model (should call from_pretrained)
    loaded = registry.load_model("mock_model")
    assert loaded is mock_model
    MockSEFModel.from_pretrained.assert_called_once_with(model_dir)

    # Verify embedder registration
    embedder = registry.get_embedder("mock_model")
    assert embedder is mock_model.embed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
