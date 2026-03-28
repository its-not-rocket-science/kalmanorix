"""
End-to-end test of Procrustes alignment integration.

This test verifies that alignment functions work correctly within the pipeline:
- compute_alignments produces orthogonal matrices
- apply_alignment transforms embeddings
- align_sef_list creates aligned SEFs
- validate_alignment_improvement measures similarity gain
"""

import numpy as np
import pytest

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
)


@pytest.mark.e2e
def test_alignment_in_pipeline():
    """
    Test full pipeline with alignment of two randomly rotated embedders.

    Steps:
    1. Create two specialists with embeddings that are random linear
       transformations of a common latent space.
    2. Compute alignment matrices between their spaces.
    3. Apply alignment to one specialist's embeddings.
    4. Verify that after alignment, cosine similarity between embeddings
       of the same text increases.
    5. Use align_sef_list to create aligned SEFs and run fusion.
    """
    dim = 16
    num_samples = 20

    # Create a random orthogonal matrix to simulate different embedding spaces
    rng = np.random.default_rng(42)
    Q1 = rng.standard_normal((dim, dim))
    Q1, _ = np.linalg.qr(Q1)  # orthogonal matrix
    Q2 = rng.standard_normal((dim, dim))
    Q2, _ = np.linalg.qr(Q2)

    # Base embeddings (latent space)
    base_embeddings = rng.standard_normal((num_samples, dim))
    # Normalize rows
    base_embeddings /= np.linalg.norm(base_embeddings, axis=1, keepdims=True) + 1e-12

    # Specialist 1: apply Q1
    emb1 = base_embeddings @ Q1.T
    # Specialist 2: apply Q2
    emb2 = base_embeddings @ Q2.T

    # Create dummy embedders that return precomputed embeddings by index
    # For simplicity, we'll use a deterministic mapping from text to index.
    # We'll use a list of dummy texts.
    texts = [f"text_{i}" for i in range(num_samples)]

    def make_embedder(emb_matrix):
        """Return an embedder that looks up embedding by text index."""

        def embedder(text: str) -> np.ndarray:
            idx = int(text.split("_")[1])
            return emb_matrix[idx].astype(np.float64)

        return embedder

    embedder1 = make_embedder(emb1)
    embedder2 = make_embedder(emb2)

    # Create SEFs with constant uncertainty
    sef1 = SEF(name="spec1", embed=embedder1, sigma2=1.0)
    sef2 = SEF(name="spec2", embed=embedder2, sigma2=1.0)

    # Compute alignment from spec2 to spec1 (reference)
    # We need embeddings from both specialists for the same texts.
    # We'll use the same texts list.
    embeddings1 = np.array([embedder1(t) for t in texts])
    embeddings2 = np.array([embedder2(t) for t in texts])

    # Compute alignment matrices for all SEFs relative to reference (spec1)
    alignments = compute_alignments(
        sef_list=[sef1, sef2],
        anchor_sentences=texts,
        reference_sef_name="spec1",
    )
    # Alignment matrix for spec2 (spec1 gets identity)
    alignment_matrix = alignments["spec2"]

    # Apply alignment to spec2's embeddings
    aligned_emb2 = apply_alignment(embeddings2, alignment_matrix)

    # Compute average cosine similarity before and after alignment
    def avg_cosine_sim(a, b):
        dots = np.einsum("ij,ij->i", a, b)
        norms_a = np.linalg.norm(a, axis=1)
        norms_b = np.linalg.norm(b, axis=1)
        cos = dots / (norms_a * norms_b + 1e-12)
        return float(np.mean(cos))

    sim_before = avg_cosine_sim(embeddings1, embeddings2)
    sim_after = avg_cosine_sim(embeddings1, aligned_emb2)

    # Alignment should improve similarity (should be closer to 1)
    assert sim_after > sim_before, (
        f"Similarity did not improve after alignment: "
        f"before={sim_before:.4f}, after={sim_after:.4f}"
    )
    # After alignment, similarity should be high (close to 1)
    assert sim_after > 0.99, f"Similarity after alignment too low: {sim_after:.4f}"

    # Test align_sef_list
    aligned_sefs = align_sef_list(
        sef_list=[sef1, sef2],
        alignments=alignments,
    )
    assert len(aligned_sefs) == 2
    # First SEF should be unchanged (reference)
    assert aligned_sefs[0].name == sef1.name
    # Second SEF should have alignment matrix attached
    assert hasattr(aligned_sefs[1], "alignment_matrix")
    assert aligned_sefs[1].alignment_matrix.shape == (dim, dim)

    # Validate alignment improvement using helper
    improvement, sim_before_arr, sim_after_arr = validate_alignment_improvement(
        sef_list=[sef1, sef2],
        alignments=alignments,
        test_sentences=texts,
        reference_sef_name="spec1",
    )
    assert improvement > 0, f"Alignment improvement not positive: {improvement}"

    # Now integrate aligned SEFs into full pipeline
    village = Village(aligned_sefs)
    scout = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=KalmanorixFuser())

    # Test fusion on a query
    query = texts[0]  # "text_0"
    potion = panoramix.brew(query, village=village, scout=scout)

    # Basic checks
    assert potion.vector.shape == (dim,)
    assert np.isfinite(potion.vector).all()
    assert len(potion.weights) == 2
    assert abs(sum(potion.weights.values()) - 1.0) < 1e-10
    assert all(w >= 0 for w in potion.weights.values())

    # Metadata should contain selected_modules
    assert potion.meta is not None
    assert "selected_modules" in potion.meta
    assert set(potion.meta["selected_modules"]) == {"spec1", "spec2"}

    # Test batch fusion
    batch_queries = texts[:3]
    potions = panoramix.brew_batch(batch_queries, village=village, scout=scout)
    assert len(potions) == 3
    for p in potions:
        assert p.vector.shape == (dim,)
        assert np.isfinite(p.vector).all()


if __name__ == "__main__":
    test_alignment_in_pipeline()
    print("Test passed.")
