#!/usr/bin/env python3
"""
Demo of Procrustes alignment for Hugging Face transformer models.

This script shows how to align embedding spaces of different Hugging Face models
using orthogonal Procrustes alignment. The alignment maps specialists into a
common reference space, enabling meaningful fusion even when models were
trained independently or have different architectures.

The demo uses two small transformer models:
  1. prajjwal1/bert-tiny (2 layers, 128-dim)
  2. sentence-transformers/all-MiniLM-L6-v2 (384-dim)

Requirements (optional dependencies):
    pip install -e ".[train]"  # installs transformers and torch
"""

import sys
import importlib.util
import numpy as np

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    KalmanorixFuser,
    HuggingFaceEmbedder,
    compute_alignments,
    align_sef_list,
    validate_alignment_improvement,
)

# Check for Hugging Face dependencies
torch_spec = importlib.util.find_spec("torch")
transformers_spec = importlib.util.find_spec("transformers")
HAVE_HF = torch_spec is not None and transformers_spec is not None

if not HAVE_HF:
    print("Required dependencies not found. Install with:")
    print('  pip install -e ".[train]"')
    print("Skipping demo.")
    sys.exit(0)


def main() -> None:
    print("=== HuggingFace Alignment Demo ===\n")

    # Seed for reproducibility
    np.random.seed(42)

    # 1. Show dimension mismatch issue
    print("--- Step 1: Dimension mismatch example ---")
    # Model 1: Tiny BERT (128 dimensions)
    bert_embedder = HuggingFaceEmbedder(
        model_name_or_path="prajjwal1/bert-tiny",
        pooling="mean",
        normalize=True,
        device="cpu",
        max_length=512,
    )
    # Model 2: MiniLM (384 dimensions) - also a Hugging Face transformer model
    # Note: This is a sentence-transformer model but can be loaded via transformers.
    minilm_embedder = HuggingFaceEmbedder(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        pooling="mean",
        normalize=True,
        device="cpu",
        max_length=512,
    )

    test_text = "A sample sentence."
    emb1 = bert_embedder(test_text)
    emb2 = minilm_embedder(test_text)
    print(f"BERT-tiny embedding dimension: {emb1.shape[0]}")
    print(f"MiniLM embedding dimension:    {emb2.shape[0]}")
    print("-> Different dimensions cannot be directly aligned.")
    print("  (Alignment requires identical dimensions.)")
    print()

    # 2. Create two specialists with same model but different pooling
    print("--- Step 2: Creating specialists for alignment ---")
    # Use BERT-tiny with two pooling strategies (same dimension, different embeddings)
    embedder_mean = HuggingFaceEmbedder(
        model_name_or_path="prajjwal1/bert-tiny",
        pooling="mean",
        normalize=True,
        device="cpu",
    )
    embedder_cls = HuggingFaceEmbedder(
        model_name_or_path="prajjwal1/bert-tiny",
        pooling="cls",
        normalize=True,
        device="cpu",
    )

    specialists = [
        SEF(
            name="bert-tiny-mean",
            embed=embedder_mean,
            sigma2=0.1,  # low uncertainty
            meta={"model": "bert-tiny", "pooling": "mean"},
        ),
        SEF(
            name="bert-tiny-cls",
            embed=embedder_cls,
            sigma2=0.2,  # slightly higher uncertainty
            meta={"model": "bert-tiny", "pooling": "cls"},
        ),
    ]

    print(f"Created {len(specialists)} specialists (same model, different pooling):")
    for sef in specialists:
        print(f"  - {sef.name} (sigma²={sef.sigma2:.2f})")

    # 3. Anchor sentences for alignment (should be representative of target domain)
    anchor_sentences = [
        "the cat sat on the mat",
        "quantum physics is fascinating",
        "legal documents require precise language",
        "medical diagnosis relies on accurate data",
        "financial markets exhibit complex behavior",
        "artificial intelligence transforms industries",
        "climate change impacts global ecosystems",
        "neural networks learn hierarchical representations",
    ]

    # 4. Compute alignment matrices (use first specialist as reference)
    print("\n--- Step 3: Computing Procrustes alignments ---")
    reference_name = specialists[0].name
    alignments = compute_alignments(
        sef_list=specialists,
        anchor_sentences=anchor_sentences,
        reference_sef_name=reference_name,
    )

    print(f"Reference specialist: {reference_name}")
    for name, matrix in alignments.items():
        ortho_error = np.linalg.norm(
            matrix.T @ matrix - np.eye(matrix.shape[0]), ord="fro"
        )
        print(
            f"  {name}: {matrix.shape[0]}×{matrix.shape[1]}, "
            f"orthogonality error {ortho_error:.2e}"
        )

    # 5. Create aligned SEFs (with alignment_matrix attribute)
    print("\n--- Step 4: Creating aligned SEFs ---")
    aligned_specialists = align_sef_list(specialists, alignments)
    print(f"Aligned {len(aligned_specialists)} specialists.")
    for sef in aligned_specialists:
        has_matrix = sef.alignment_matrix is not None
        print(f"  {sef.name}: alignment_matrix attached? {has_matrix}")

    # 6. Test sentences for validation (different from anchors)
    test_sentences = [
        "machine learning algorithms improve over time",
        "contract law governs business agreements",
        "immunology studies immune system responses",
        "stock prices reflect market sentiment",
        "natural language processing enables human-computer interaction",
        "deep learning requires large amounts of data",
        "copyright law protects creative works",
        "genetic engineering modifies organism DNA",
    ]

    print("\n--- Step 5: Validating alignment improvement ---")
    norm_improvement, sim_before, sim_after = validate_alignment_improvement(
        sef_list=specialists,
        alignments=alignments,
        test_sentences=test_sentences,
        reference_sef_name=reference_name,
    )

    print(f"\nNormalized improvement: {100 * norm_improvement:.1f}%")
    print(
        f"Similarity before: mean={np.mean(sim_before):.4f}, std={np.std(sim_before):.4f}"
    )
    print(
        f"Similarity after:  mean={np.mean(sim_after):.4f}, std={np.std(sim_after):.4f}"
    )

    # 7. Demonstrate fusion with aligned specialists
    print("\n--- Step 6: Fusion with aligned specialists ---")
    aligned_village = Village(aligned_specialists)
    router = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=KalmanorixFuser())

    query = "a query about biology and law"
    potion = panoramix.brew(query, village=aligned_village, scout=router)

    print(f"Query: '{query}'")
    print(f"Selected specialists: {list(potion.weights.keys())}")
    print("Fusion weights:")
    for name, weight in potion.weights.items():
        print(f"  {name}: {weight:.4f}")
    print(f"Fused embedding dimension: {potion.vector.shape[0]}")

    # 8. Show that alignment matrices affect embeddings
    print("\n--- Step 7: Alignment effect on single embedding ---")
    example_text = "example sentence for alignment comparison"
    spec_original = specialists[1]  # second specialist (not reference)
    spec_aligned = aligned_specialists[1]

    # Embedding before alignment (original specialist)
    emb_orig = spec_original.embed(example_text)
    # Embedding after alignment (using alignment_matrix attached to SEF)
    matrix = spec_aligned.alignment_matrix
    if matrix is not None:
        emb_aligned = matrix @ spec_original.embed(example_text)
    else:
        emb_aligned = emb_orig

    cos_sim = (
        emb_orig
        @ emb_aligned
        / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_aligned))
    )
    print(f"Original embedding norm: {np.linalg.norm(emb_orig):.4f}")
    print(f"Aligned embedding norm:  {np.linalg.norm(emb_aligned):.4f}")
    print(f"Cosine similarity between original and aligned: {cos_sim:.4f}")
    # Alignment is orthogonal, so norms should be identical (within numerical error)
    print(
        f"Norm difference: {abs(np.linalg.norm(emb_orig) - np.linalg.norm(emb_aligned)):.2e}"
    )

    print("\n=== Demo completed successfully ===")


if __name__ == "__main__":
    main()
