"""Procrustes alignment utilities for embedding spaces.

This module provides functions to compute orthogonal alignment matrices that map
specialist embedding spaces into a common reference space.
"""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import numpy as np

from .village import SEF
from .models.sef import create_procrustes_alignment


def compute_alignments(
    sef_list: List[SEF],
    anchor_sentences: List[str],
    reference_sef_name: str,
    *,
    _epsilon: float = 1e-8,  # unused, kept for future numerical stability
) -> Dict[str, np.ndarray]:
    """Compute Procrustes alignment matrices for a list of SEFs.

    For each SEF (excluding the reference), compute an orthogonal matrix that
    maps its embedding space to the reference SEF's space. The reference SEF
    gets identity alignment.

    Args:
        sef_list: List of SEF objects
        anchor_sentences: List of text sentences used for alignment
        reference_sef_name: Name of the SEF to use as reference space
        _epsilon (float): Small constant for numerical stability (unused)

    Returns:
        Dictionary mapping SEF name to alignment matrix (d×d orthogonal).
        The reference SEF's matrix is identity.

    Raises:
        ValueError: If reference_sef_name not found in sef_list,
                    or if embeddings have inconsistent dimensions.
    """
    # Find reference SEF
    ref_sef = None
    for sef in sef_list:
        if sef.name == reference_sef_name:
            ref_sef = sef
            break
    if ref_sef is None:
        raise ValueError(
            f"Reference SEF '{reference_sef_name}' not found in sef_list. "
            f"Available names: {[s.name for s in sef_list]}"
        )

    # Compute reference embeddings
    print(f"Computing reference embeddings for {reference_sef_name}...")
    ref_embeddings = np.stack([ref_sef.embed(s) for s in anchor_sentences], axis=0)

    d = ref_embeddings.shape[1]
    alignments: Dict[str, np.ndarray] = {reference_sef_name: np.eye(d)}

    # Compute alignments for other SEFs
    for sef in sef_list:
        if sef.name == reference_sef_name:
            continue

        print(f"Computing alignment for {sef.name}...")
        # Get embeddings for this SEF
        src_embeddings = np.stack([sef.embed(s) for s in anchor_sentences], axis=0)

        # Check dimension match
        if src_embeddings.shape[1] != d:
            raise ValueError(
                f"Dimension mismatch: {sef.name} has {src_embeddings.shape[1]} "
                f"dimensions, reference has {d}"
            )

        # Compute Procrustes alignment (Q maps source rows to target rows)
        Q = create_procrustes_alignment(src_embeddings, ref_embeddings)
        # For column-vector embeddings we need Q.T (maps source columns to target columns)
        align_matrix = Q.T
        alignments[sef.name] = align_matrix

        # Optional: verify orthogonality
        ortho_error = np.linalg.norm(
            align_matrix.T @ align_matrix - np.eye(d), ord="fro"
        )
        if ortho_error > 1e-6:
            print(
                f"  Warning: alignment matrix for {sef.name} is not orthogonal "
                f"(error={ortho_error:.2e})"
            )

    return alignments


def apply_alignment(
    embeddings: np.ndarray,
    alignment_matrix: Optional[np.ndarray],
) -> np.ndarray:
    """Apply alignment matrix to embeddings.

    Args:
        embeddings: Array of shape (n, d) or (d,)
        alignment_matrix: Optional orthogonal matrix (d×d). If None, returns unchanged.

    Returns:
        Aligned embeddings.
    """
    if alignment_matrix is None:
        return embeddings

    if embeddings.ndim == 1:
        return alignment_matrix @ embeddings
    # else
    return embeddings @ alignment_matrix.T  # (n, d) @ (d, d).T = (n, d)


def align_sef_list(
    sef_list: List[SEF],
    alignments: Dict[str, np.ndarray],
) -> List[SEF]:
    """Create new SEF list with alignment matrices attached.

    This returns a new list of SEF objects where each SEF's alignment_matrix
    attribute is set according to the provided dictionary.

    Args:
        sef_list: Original SEF objects
        alignments: Dictionary mapping SEF name to alignment matrix

    Returns:
        New SEF objects with alignment_matrix set.
    """
    aligned_sefs = []
    for sef in sef_list:
        matrix = alignments.get(sef.name)
        # Create a new SEF with the same fields plus alignment_matrix
        aligned_sef = replace(sef, alignment_matrix=matrix)
        aligned_sefs.append(aligned_sef)

    return aligned_sefs


def validate_alignment_improvement(
    sef_list: List[SEF],
    alignments: Dict[str, np.ndarray],
    test_sentences: List[str],
    reference_sef_name: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Measure how much alignment improves cross‑model similarity.

    Computes the average cosine similarity between each specialist and the
    reference, before and after alignment. Returns the relative improvement
    and the raw similarity scores.

    Args:
        sef_list: Original SEF objects
        alignments: Dictionary mapping SEF name to alignment matrix
        test_sentences: List of sentences for evaluation (different from anchors)
        reference_sef_name: Name of reference SEF

    Returns:
        Tuple of (normalized_improvement, similarities_before, similarities_after)
        normalized_improvement: (similarity_after - similarity_before) / (1 - similarity_before)
        similarities_before: Array of cosine similarities before alignment
        similarities_after: Array of cosine similarities after alignment
    """
    # Find reference SEF
    ref_sef = None
    for sef in sef_list:
        if sef.name == reference_sef_name:
            ref_sef = sef
            break
    if ref_sef is None:
        raise ValueError(f"Reference SEF '{reference_sef_name}' not found")

    similarities_before = []
    similarities_after = []

    for sef in sef_list:
        if sef.name == reference_sef_name:
            continue

        matrix = alignments.get(sef.name)

        for sentence in test_sentences:
            # Reference embedding
            ref_emb = ref_sef.embed(sentence)
            ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)

            # Specialist embedding before alignment
            emb_before = sef.embed(sentence)
            emb_before = emb_before / (np.linalg.norm(emb_before) + 1e-8)
            sim_before = float(ref_emb @ emb_before)
            similarities_before.append(sim_before)

            # After alignment
            if matrix is not None:
                emb_after = matrix @ sef.embed(sentence)
            else:
                emb_after = emb_before
            emb_after = emb_after / (np.linalg.norm(emb_after) + 1e-8)
            sim_after = float(ref_emb @ emb_after)
            similarities_after.append(sim_after)

    avg_before = np.mean(similarities_before)
    avg_after = np.mean(similarities_after)

    print(f"Average similarity before alignment: {avg_before:.4f}")
    print(f"Average similarity after alignment:  {avg_after:.4f}")
    # Normalized improvement (fraction of possible improvement achieved)
    # Denominator clipped to avoid division by zero
    denominator = 1 - avg_before
    if denominator < 1e-12:
        norm_improvement = 0.0
    else:
        norm_improvement = float((avg_after - avg_before) / denominator)
    print(f"Normalized improvement: {100 * norm_improvement:.1f}%")

    return (
        float(norm_improvement),
        np.array(similarities_before),
        np.array(similarities_after),
    )
