"""
End-to-end test of Model Registry integration.

This test verifies that SEF models can be saved to disk, discovered by the registry,
loaded back, and used in the fusion pipeline.
"""

import numpy as np
import pytest
from pathlib import Path

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    KalmanorixFuser,
    ModelRegistry,
)
from kalmanorix.models.sef import SEFModel, SEFMetadata


def dummy_embedder(text: str) -> np.ndarray:
    """Simple deterministic embedder for testing."""
    # Return a vector based on text length, normalized
    vec = np.ones(4, dtype=np.float64) * len(text)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


@pytest.mark.e2e
def test_pipeline_with_model_registry(tmp_path: Path):
    """
    Full pipeline using ModelRegistry to load specialists from disk.

    Steps:
    1. Create a dummy SEF model and save it to a temporary directory.
    2. Create a ModelRegistry pointing to that directory and scan.
    3. Load the model from the registry, obtaining a SEF.
    4. Use the SEF in a village and run fusion.
    """
    # Create a model directory
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()

    # Metadata
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

    # Create SEF model with dummy embedder
    model = SEFModel(
        embed_function=dummy_embedder,
        metadata=metadata,
        alignment_matrix=None,
        covariance_data={"method": "fixed", "diagonal": np.ones(4)},
    )

    # Save model to disk
    model.save_pretrained(model_dir)

    # Create registry and scan
    registry = ModelRegistry(tmp_path)
    model_ids = registry.scan()
    assert model_ids == ["dummy_model"]

    # Load the model as a SEF
    sef = registry.load_model("dummy_model")
    assert sef.name == "dummy_model"
    assert sef.embed is not None
    assert sef.sigma2 == 1.0  # default constant uncertainty

    # Create a second specialist directly (not from registry) for fusion
    def dummy_embedder2(text: str) -> np.ndarray:
        vec = np.ones(4, dtype=np.float64) * (len(text) + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    sef2 = SEF(name="direct", embed=dummy_embedder2, sigma2=0.5)

    # Build village with both specialists
    village = Village([sef, sef2])

    # Scout router selects all modules
    scout = ScoutRouter(mode="all")

    # Panoramix with KalmanorixFuser
    panoramix = Panoramix(fuser=KalmanorixFuser())

    query = "Test query."
    potion = panoramix.brew(query, village=village, scout=scout)

    # Assertions
    assert potion.vector.shape == (4,)
    assert np.isfinite(potion.vector).all()
    assert len(potion.weights) == 2
    assert abs(sum(potion.weights.values()) - 1.0) < 1e-10
    assert all(w >= 0 for w in potion.weights.values())

    # Metadata should contain selected_modules
    assert potion.meta is not None
    assert "selected_modules" in potion.meta
    selected_names = set(potion.meta["selected_modules"])
    assert selected_names == {"dummy_model", "direct"}

    # Test batch fusion
    batch_queries = ["First query", "Second query"]
    potions = panoramix.brew_batch(batch_queries, village=village, scout=scout)
    assert len(potions) == 2
    for p in potions:
        assert p.vector.shape == (4,)
        assert np.isfinite(p.vector).all()

    # Additional registry checks
    models = registry.list_models()
    assert "dummy_model" in models
    assert models["dummy_model"]["name"] == "Dummy Model"
    assert models["dummy_model"]["embedding_dimension"] == 4


if __name__ == "__main__":
    # Run with a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_pipeline_with_model_registry(Path(tmpdir))
    print("Test passed.")
