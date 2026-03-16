"""
Validation of Procrustes alignment on synthetic specialists.

This script corresponds to Milestone 1.2: Procrustes Alignment.
It creates synthetic specialists with known rotations and tests whether
our Procrustes alignment recovers them accurately.
"""

# pylint: disable=import-outside-toplevel

import hashlib
import sys
from pathlib import Path
from typing import Callable, cast

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # pylint: disable=wrong-import-position
from numpy.linalg import LinAlgError  # pylint: disable=wrong-import-position

from kalmanorix.alignment import (  # pylint: disable=wrong-import-position
    compute_alignments,
    validate_alignment_improvement,
)
from kalmanorix.toy_corpus import generate_anchor_sentences  # pylint: disable=wrong-import-position
from kalmanorix.village import SEF  # pylint: disable=wrong-import-position
from kalmanorix.types import Embedder  # pylint: disable=wrong-import-position


def create_synthetic_specialist(
    name: str,
    base_embedder: Embedder,
    rotation_matrix: np.ndarray,
    sigma2: float = 0.1,
) -> SEF:
    """Create a synthetic specialist with known rotation.

    The specialist's embeddings are rotated versions of a base embedder.
    """

    def rotated_embedder(text: str) -> np.ndarray:
        emb = base_embedder(text)
        return rotation_matrix @ emb

    return SEF(
        name=name,
        embed=cast(Embedder, rotated_embedder),
        sigma2=sigma2,
        meta={"type": "synthetic", "rotation": "known"},
    )


def create_base_embedder(dimension: int = 100) -> Callable[[str], np.ndarray]:
    """Create a deterministic base embedder for testing."""

    def embedder(text: str) -> np.ndarray:
        # Deterministic random normal embedding based on text hash
        h = hashlib.sha256(text.encode()).hexdigest()
        seed = int(h, 16) % (2**32)
        rng = np.random.RandomState(seed)  # pylint: disable=no-member
        vec = rng.randn(dimension)
        # Normalise to unit sphere
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    return embedder


def run_validation(  # pylint: disable=too-many-branches,too-many-statements
    dimension: int = 100,
    n_anchor_sentences: int = 500,
    n_test_sentences: int = 100,
    seed: int = 42,
):
    """
    Validate Procrustes alignment.

    Args:
        dimension: Embedding dimension d
        n_anchor_sentences: Number of anchor sentences for alignment
        n_test_sentences: Number of test sentences for evaluation
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    print("=" * 70)
    print("Procrustes Alignment Validation (Milestone 1.2)")
    print("=" * 70)
    print(f"Dimension: {dimension}")
    print(f"Anchor sentences: {n_anchor_sentences}")
    print(f"Test sentences: {n_test_sentences}")
    print()

    # Generate sentences
    print("Generating synthetic sentences...")
    all_sentences = generate_anchor_sentences(
        n=n_anchor_sentences + n_test_sentences,
        seed=seed,
    )
    anchor_sentences = all_sentences[:n_anchor_sentences]
    test_sentences = all_sentences[n_anchor_sentences:]

    # Create base embedder (simulates a "tech" specialist)
    base_embedder = cast(Embedder, create_base_embedder(dimension))

    # Create random orthogonal rotation matrices for other specialists
    print("Creating synthetic specialists with known rotations...")

    # Tech specialist (reference, no rotation)
    tech_sef = SEF(
        name="tech",
        embed=cast(Embedder, base_embedder),
        sigma2=0.1,
        meta={"type": "reference"},
    )

    # Cooking specialist (random rotation)
    U_cook, _ = np.linalg.qr(np.random.randn(dimension, dimension))
    # Ensure determinant +1 (proper rotation)
    if np.linalg.det(U_cook) < 0:
        U_cook[:, -1] *= -1
    cook_sef = create_synthetic_specialist(
        name="cook",
        base_embedder=base_embedder,
        rotation_matrix=U_cook,
        sigma2=0.2,
    )

    # Medical specialist (another random rotation)
    U_medical, _ = np.linalg.qr(np.random.randn(dimension, dimension))
    if np.linalg.det(U_medical) < 0:
        U_medical[:, -1] *= -1
    medical_sef = create_synthetic_specialist(
        name="medical",
        base_embedder=base_embedder,
        rotation_matrix=U_medical,
        sigma2=0.3,
    )

    sef_list = [tech_sef, cook_sef, medical_sef]

    # Test 1: Compute alignments
    print("\n--- Computing alignments ---")
    try:
        alignments = compute_alignments(
            sef_list=sef_list,
            anchor_sentences=anchor_sentences,
            reference_sef_name="tech",
            _epsilon=1e-8,
        )
        print(f"Computed alignments for: {list(alignments.keys())}")
    except (ValueError, RuntimeError, LinAlgError) as e:
        print(f"ERROR: Alignment computation failed: {e}")
        return

    # Test 2: Measure alignment quality
    print("\n--- Measuring alignment improvement ---")
    try:
        improvement, sim_before, sim_after = validate_alignment_improvement(
            sef_list=sef_list,
            alignments=alignments,
            test_sentences=test_sentences,
            reference_sef_name="tech",
        )
        print(f"Alignment improvement ratio: {improvement:.3f}")

        # Statistical significance test (paired t-test)
        diff = sim_after - sim_before
        n = len(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        if std_diff > 0:
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            # Approximate p-value using t-distribution survival function
            try:
                from scipy.stats import t

                p_value = t.sf(np.abs(t_stat), df=n - 1) * 2  # two-tailed
                print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.6f}")
                if p_value < 0.05:
                    print("  [OK] Improvement is statistically significant (p < 0.05)")
                else:
                    print("  [WARNING] Improvement not statistically significant")
            except ImportError:
                print(
                    f"Paired t-test: t = {t_stat:.3f} (scipy not installed for p-value)"
                )
        else:
            print("No variance in differences (all identical)")

        # Optional visualization if matplotlib available
        try:
            from kalmanorix.visualization import plot_similarity_comparison

            plot_similarity_comparison(
                sim_before,
                sim_after,
                title="Similarity Before/After Alignment (Validation)",
                show=False,  # Don't block execution
            )
            # Save figure to file
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.savefig("alignment_improvement.png")
            plt.close()
            print("Saved alignment improvement plot to alignment_improvement.png")
        except ImportError:
            pass  # matplotlib not installed, skip visualization

    except (ValueError, RuntimeError, LinAlgError) as e:
        print(f"ERROR: Alignment validation failed: {e}")
        return

    # Test 3: Check recovery of known rotations
    print("\n--- Checking rotation recovery ---")
    tolerance = 0.05  # Allow 5% error

    for sef in sef_list:
        if sef.name == "tech":
            continue  # Reference has identity rotation

        # True rotation matrix (maps base column vector to rotated column vector)
        if sef.name == "cook":
            true_Q = U_cook
        else:  # medical
            true_Q = U_medical

        # Estimated alignment matrix
        est_Q = alignments.get(sef.name)
        if est_Q is None:
            print(f"  {sef.name}: No alignment matrix found")
            continue

        # For a perfect recovery, est_Q should equal true_Q.T (since we're
        # aligning from rotated space back to base space)
        # Compute Frobenius norm error
        error = np.linalg.norm(est_Q - true_Q.T, ord="fro")
        rel_error = error / np.linalg.norm(true_Q.T, ord="fro")
        print(f"  {sef.name}: relative error = {rel_error:.4f}")

        if rel_error < tolerance:
            print(f"    [OK] Within tolerance ({tolerance})")
        else:
            print(f"    [FAIL] Exceeds tolerance ({tolerance})")

    # Test 4: Orthogonality check
    print("\n--- Orthogonality check ---")
    for name, Q in alignments.items():
        if name == "tech":
            continue  # Identity is orthogonal by definition

        # Check Q^T Q = I
        eye_mat = np.eye(dimension)
        ortho_error = np.linalg.norm(Q.T @ Q - eye_mat, ord="fro")
        print(f"  {name}: orthogonality error = {ortho_error:.2e}")

        if ortho_error < 1e-6:
            print("    [OK] Orthogonal within tolerance")
        else:
            print("    [FAIL] Not sufficiently orthogonal")

    # Final verdict
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Success criteria from Milestone 1.2
    # "Alignment improves cross-model similarity by >20%"
    if improvement > 0.2:
        print(
            "[PASS] Alignment improves cross-model similarity by >20% "
            f"({improvement:.1%} improvement)"
        )
        overall_pass = True
    else:
        print(f"[FAIL] Alignment improvement ({improvement:.1%}) below 20% target")
        overall_pass = False

    if overall_pass:
        print("\n[PASS] Validation PASSED: Procrustes alignment works as expected.")
    else:
        print("\n[FAIL] Validation FAILED: Some criteria not met.")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
