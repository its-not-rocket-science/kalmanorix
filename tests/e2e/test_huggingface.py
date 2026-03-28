"""
End-to-end test of the pipeline with HuggingFaceEmbedder.

Requires optional dependencies: transformers and torch.
"""

import sys
import numpy as np
import pytest


# Skip if transformers or torch not installed
@pytest.mark.skipif(
    "transformers" not in sys.modules,
    reason="transformers library not installed",
)
@pytest.mark.skipif(
    "torch" not in sys.modules,
    reason="torch library not installed",
)
@pytest.mark.e2e
def test_pipeline_with_huggingface_embedder():
    """
    Test full pipeline with a tiny Hugging Face model.

    This test verifies that:
    - HuggingFaceEmbedder can be instantiated and used
    - SEF wrapping works
    - Village, ScoutRouter, Panoramix integrate correctly
    - Fusion produces valid outputs
    """
    from kalmanorix import (
        SEF,
        Village,
        ScoutRouter,
        Panoramix,
        KalmanorixFuser,
        HuggingFaceEmbedder,
    )

    # Use a tiny BERT model for testing (2 layers, 128 hidden)
    model_name = "prajjwal1/bert-tiny"

    # Create embedder with CPU device
    embedder = HuggingFaceEmbedder(
        model_name_or_path=model_name,
        pooling="mean",
        normalize=True,
        device="cpu",
        max_length=512,
    )

    # Wrap as SEF with constant uncertainty
    sef = SEF(
        name="bert-tiny",
        embed=embedder,
        sigma2=1.0,
        meta={"model": model_name, "pooling": "mean"},
    )

    # Create a village with this specialist
    village = Village([sef])

    # Scout router selects all modules
    scout = ScoutRouter(mode="all")

    # Panoramix with KalmanorixFuser
    panoramix = Panoramix(fuser=KalmanorixFuser())

    query = "A sentence about artificial intelligence."
    potion = panoramix.brew(query, village=village, scout=scout)

    # Assertions
    assert potion.vector.shape == (embedder.dim,)
    assert np.isfinite(potion.vector).all()
    assert len(potion.weights) == 1
    assert "bert-tiny" in potion.weights
    assert potion.weights["bert-tiny"] > 0
    assert abs(sum(potion.weights.values()) - 1.0) < 1e-10

    # Metadata should contain selected_modules
    assert potion.meta is not None
    assert "selected_modules" in potion.meta
    assert potion.meta["selected_modules"] == ["bert-tiny"]

    # Optional: test batch fusion
    queries = ["First query.", "Second query about technology."]
    potions = panoramix.brew_batch(queries, village=village, scout=scout)

    assert len(potions) == 2
    for potion in potions:
        assert potion.vector.shape == (embedder.dim,)
        assert np.isfinite(potion.vector).all()
        assert len(potion.weights) == 1
        assert abs(sum(potion.weights.values()) - 1.0) < 1e-10


if __name__ == "__main__":
    # If run directly, try to import dependencies
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        print(f"Skipping test: {e}")
        sys.exit(0)
    test_pipeline_with_huggingface_embedder()
    print("Test passed.")
