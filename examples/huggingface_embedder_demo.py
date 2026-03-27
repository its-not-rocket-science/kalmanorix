#!/usr/bin/env python3
"""
Demo of HuggingFaceEmbedder adapter for Kalmanorix.

This script shows how to wrap a Hugging Face transformer model as an SEF
and use it in fusion. It uses a tiny BERT model for demonstration.

Requirements (optional dependencies):
    pip install -e ".[train]"  # installs transformers and torch
"""

import sys
import importlib.util
import pickle
import numpy as np

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    KalmanorixFuser,
    HuggingFaceEmbedder,
)

# Check if transformers and torch are available
torch_spec = importlib.util.find_spec("torch")
transformers_spec = importlib.util.find_spec("transformers")
HAVE_HF = torch_spec is not None and transformers_spec is not None

if not HAVE_HF:
    print("Hugging Face dependencies not found. Install with:")
    print('  pip install -e ".[train]"')
    print("Skipping demo.")
    sys.exit(0)


def main():
    print("=== HuggingFaceEmbedder Demo ===\n")

    # 1. Create a HuggingFaceEmbedder
    # Using a tiny BERT model (2 layers, 128 hidden size)
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading model: {model_name}")
    embedder = HuggingFaceEmbedder(
        model_name_or_path=model_name,
        pooling="mean",  # average of token embeddings
        normalize=True,  # output unit-length vectors
        device="cpu",  # use "cuda" if GPU available
        max_length=512,
    )
    print(f"Embedder created: {embedder}")
    print(f"  - Pooling: {embedder.pooling}")
    print(f"  - Device: {embedder.device}")
    print(f"  - Normalize: {embedder.normalize}\n")

    # 2. Test embedding a single query
    test_query = "The quick brown fox jumps over the lazy dog."
    vec = embedder(test_query)
    print(f"Test query: '{test_query}'")
    print(f"  Embedding shape: {vec.shape}")
    print(f"  Embedding norm: {np.linalg.norm(vec):.6f}")
    print(f"  First 5 dims: {vec[:5].round(6)}\n")

    # 3. Wrap as SEF with uncertainty
    # For demo, we'll use constant sigma² = 1.0
    sef = SEF(
        name="bert-tiny",
        embed=embedder,
        sigma2=1.0,
        meta={"model": model_name, "pooling": "mean"},
    )

    # 4. Create a village with this specialist
    village = Village([sef])
    print(f"Village created with {len(village.modules)} specialist\n")

    # 5. Test fusion (trivial with one specialist)
    scout = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=KalmanorixFuser())

    potion = panoramix.brew(
        query="A sentence about artificial intelligence.",
        village=village,
        scout=scout,
    )

    print("Fusion results:")
    print(f"  Query: '{potion.meta.get('query', 'N/A')}'")
    print(f"  Selected modules: {potion.meta.get('selected_modules', [])}")
    print(f"  Weights: {potion.weights}")
    print(f"  Fused vector shape: {potion.vector.shape}")
    print(f"  Fused vector norm: {np.linalg.norm(potion.vector):.6f}\n")

    # 6. Compare with CLS pooling
    print("--- Testing CLS pooling ---")
    embedder_cls = HuggingFaceEmbedder(
        model_name_or_path=model_name,
        pooling="cls",
        normalize=True,
    )
    vec_cls = embedder_cls(test_query)
    vec_mean = embedder(test_query)  # same query, mean pooling

    # Compute similarity
    sim = np.dot(vec_cls, vec_mean)
    print(f"  Cosine similarity between CLS and mean pooling: {sim:.6f}")
    print("  (Should be close to 1.0 for BERT, but not identical)\n")

    # 7. Show integration with multiple specialists (toy example)
    print("--- Multi-specialist scenario (toy) ---")
    # Create a second embedder with same model but different pooling
    # In a real scenario you'd use different models or fine‑tuned variants.
    sef2 = SEF(
        name="bert-tiny-cls",
        embed=embedder_cls,
        sigma2=1.5,  # higher uncertainty
        meta={"model": model_name, "pooling": "cls"},
    )

    village2 = Village([sef, sef2])
    potion2 = panoramix.brew(
        query="Another test sentence.",
        village=village2,
        scout=scout,
    )

    print(f"  Village modules: {[m.name for m in village2.modules]}")
    print(f"  Selected modules: {potion2.meta.get('selected_modules', [])}")
    print(f"  Weights: {potion2.weights}")
    print(f"  Weight sum: {sum(potion2.weights.values()):.6f}\n")

    # 8. Pickling demonstration
    print("--- Pickling demonstration ---")
    # Pickle the embedder
    pickled = pickle.dumps(embedder)
    print(f"  Embedder pickled size: {len(pickled)} bytes")
    # Unpickle
    embedder_restored = pickle.loads(pickled)
    # Verify configuration preserved
    assert embedder_restored.model_name_or_path == embedder.model_name_or_path
    assert embedder_restored.pooling == embedder.pooling
    assert embedder_restored.device == embedder.device
    # Ensure it still works
    vec_restored = embedder_restored(test_query)
    np.testing.assert_allclose(vec_restored, vec, rtol=1e-6)
    print("  Pickling test passed: embedder successfully serialized/deserialized.")
    # Pickle SEF
    sef_pickled = pickle.dumps(sef)
    sef_restored = pickle.loads(sef_pickled)
    assert sef_restored.name == sef.name
    assert sef_restored.sigma2 == sef.sigma2
    vec_sef = sef_restored.embed(test_query)
    np.testing.assert_allclose(vec_sef, vec, rtol=1e-6)
    print("  SEF pickling test passed.\n")

    print("=== Demo completed successfully ===")


if __name__ == "__main__":
    main()
