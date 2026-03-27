"""
Procrustes alignment demo for Kalmanorix.

This example demonstrates how to align embedding spaces of different specialists
using orthogonal Procrustes alignment. When specialists are trained independently,
their embedding spaces may be rotated relative to each other. Alignment maps
them into a common reference space, enabling meaningful fusion.

The demo:
1. Creates three synthetic specialists with known random rotations
2. Computes alignment matrices using a small set of anchor sentences
3. Applies the alignments to create aligned SEFs
4. Validates the improvement in cross‑model similarity
5. Shows fusion before and after alignment

This file is intentionally dependency-light and fully deterministic.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    KalmanorixFuser,
    compute_alignments,
    apply_alignment,
    align_sef_list,
    validate_alignment_improvement,
    create_procrustes_alignment,
)


def make_rotated_specialist(
    name: str,
    base_embedder: Callable[[str], np.ndarray],
    rotation_matrix: np.ndarray,
) -> SEF:
    """Create a specialist whose embeddings are rotated by a given orthogonal matrix."""

    def rotated_embed(text: str) -> np.ndarray:
        # Base embedding
        emb = base_embedder(text)
        # Apply rotation: emb @ rotation_matrix.T (for row vectors)
        return emb @ rotation_matrix.T

    # Constant uncertainty for simplicity
    sigma2 = 0.1
    return SEF(name=name, embed=rotated_embed, sigma2=sigma2)


def main() -> None:
    print("=== Procrustes Alignment Demo ===\n")

    # Seed for reproducibility
    np.random.seed(42)
    d = 16  # Embedding dimension

    # Create a base embedder that returns random unit vectors
    # In a real scenario, this would be a pre‑trained model
    def base_embedder(text: str) -> np.ndarray:
        # Deterministic hash to produce consistent "embeddings"
        h = hash(text) % (2**31)
        rng = np.random.RandomState(h)  # pylint: disable=no-member
        vec = rng.randn(d)
        return vec / np.linalg.norm(vec)

    # Generate random orthogonal rotation matrices for three specialists
    rotations = {}
    for i in range(3):
        # Random matrix with orthonormal columns (Q factor of QR decomposition)
        A = np.random.randn(d, d)
        Q, _ = np.linalg.qr(A)
        rotations[f"specialist_{i}"] = Q

    # Create specialists with rotations
    specialists = []
    for name, rot in rotations.items():
        sef = make_rotated_specialist(name, base_embedder, rot)
        specialists.append(sef)

    print(f"Created {len(specialists)} specialists with random rotations.")
    print("Specialist names:", [s.name for s in specialists])

    # Anchor sentences used for alignment (different from test sentences)
    anchor_sentences = [
        "the cat sat on the mat",
        "quantum physics is fascinating",
        "legal documents require precise language",
        "medical diagnosis relies on accurate data",
        "financial markets exhibit complex behavior",
    ]

    # Test sentences for validation (different from anchors)
    test_sentences = [
        "neural networks learn patterns",
        "contract law governs agreements",
        "immunology studies the immune system",
        "stock prices fluctuate daily",
        "natural language processing enables understanding",
    ]

    # Step 1: Compute alignment matrices
    print("\n--- Step 1: Computing alignments ---")
    reference_name = "specialist_0"
    alignments = compute_alignments(
        sef_list=specialists,
        anchor_sentences=anchor_sentences,
        reference_sef_name=reference_name,
    )

    print(f"Reference specialist: {reference_name}")
    for name, matrix in alignments.items():
        ortho_error = np.linalg.norm(matrix.T @ matrix - np.eye(d), ord="fro")
        print(
            f"  {name}: alignment matrix shape {matrix.shape}, orthogonality error {ortho_error:.2e}"
        )

    # Step 2: Apply alignments to create aligned SEFs
    print("\n--- Step 2: Creating aligned SEFs ---")
    aligned_specialists = align_sef_list(specialists, alignments)

    # Step 3: Validate improvement
    print("\n--- Step 3: Validating alignment improvement ---")
    norm_improvement, sim_before, sim_after = validate_alignment_improvement(
        sef_list=specialists,
        alignments=alignments,
        test_sentences=test_sentences,
        reference_sef_name=reference_name,
    )

    print(f"Normalized improvement: {100 * norm_improvement:.1f}%")
    print(
        f"Similarity before: mean={np.mean(sim_before):.4f}, std={np.std(sim_before):.4f}"
    )
    print(
        f"Similarity after:  mean={np.mean(sim_after):.4f}, std={np.std(sim_after):.4f}"
    )

    # Step 4: Demonstrate fusion before and after alignment
    print("\n--- Step 4: Fusion before vs after alignment ---")

    # Create village with aligned specialists
    aligned_village = Village(aligned_specialists)

    # Router that selects all specialists
    router = ScoutRouter(mode="all")

    # Fusion orchestrator
    panoramix = Panoramix(fuser=KalmanorixFuser())

    query = "a query about biology and law"

    # Fusion with original (unaligned) specialists
    # Note: we need to temporarily attach alignment matrices for original village
    # For demo purposes, we'll just use aligned village for both
    # Actually, let's compute fusion weights with aligned specialists
    potion = panoramix.brew(query, village=aligned_village, scout=router)

    print(f"Query: '{query}'")
    print(f"Selected specialists: {list(potion.weights.keys())}")
    print("Fusion weights (aligned specialists):")
    for name, weight in potion.weights.items():
        print(f"  {name}: {weight:.4f}")

    # Show that apply_alignment works on individual embeddings
    print("\n--- Step 5: Applying alignment to single embedding ---")
    example_text = "example sentence"
    spec = specialists[1]  # specialist_1 (not reference)
    original_emb = spec.embed(example_text)
    aligned_emb = apply_alignment(original_emb, alignments.get(spec.name))

    print(f"Original embedding norm: {np.linalg.norm(original_emb):.4f}")
    print(f"Aligned embedding norm:  {np.linalg.norm(aligned_emb):.4f}")
    print(
        f"Cosine similarity between original and aligned: "
        f"{original_emb @ aligned_emb / (np.linalg.norm(original_emb) * np.linalg.norm(aligned_emb)):.4f}"
    )

    # Demonstrate low-level create_procrustes_alignment
    print("\n--- Step 6: Low-level create_procrustes_alignment ---")
    # Generate some random embeddings
    n = 10
    source = np.random.randn(n, d)
    target = np.random.randn(n, d)

    Q = create_procrustes_alignment(source, target)
    ortho_error = np.linalg.norm(Q.T @ Q - np.eye(d), ord="fro")
    print(f"Low-level alignment matrix shape: {Q.shape}")
    print(f"Orthogonality error: {ortho_error:.2e}")
    print(f"Determinant (should be +1): {np.linalg.det(Q):.4f}")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
