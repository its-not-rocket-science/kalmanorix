"""Heuristic functions for dynamic thresholding in semantic routing."""

import numpy as np
from typing import List
from .types import Vec


def threshold_top_k(
    query: str, query_vec: Vec, similarities: List[float], k: int = 1
) -> float:
    """Set threshold to select top k most similar modules.

    Args:
        query: Original query string.
        query_vec: Normalized query embedding vector.
        similarities: Cosine similarities between query and each module centroid.
        k: Number of top modules to select.

    Returns:
        Threshold value that selects at most k modules (or -1 to select all if k >= len).
    """
    if k >= len(similarities):
        return -1.0  # Select all modules
    sorted_sims = sorted(similarities, reverse=True)
    # Return similarity of (k-1)-th module (0-indexed) to select top k
    # For k=1, return sorted_sims[0] (top similarity)
    # For k=2, return sorted_sims[1] (second highest similarity)
    return sorted_sims[k - 1]


def threshold_relative_to_max(
    query: str,
    query_vec: Vec,
    similarities: List[float],
    fraction: float = 0.8,
    min_threshold: float = 0.3,
) -> float:
    """Threshold as fraction of maximum similarity.

    Args:
        query: Original query string.
        query_vec: Normalized query embedding vector.
        similarities: Cosine similarities between query and each module centroid.
        fraction: Fraction of max similarity to use as threshold (0-1).
        min_threshold: Minimum absolute threshold value.

    Returns:
        max(fraction * max_similarity, min_threshold)
    """
    if not similarities:
        return min_threshold
    max_sim = max(similarities)
    return max(fraction * max_sim, min_threshold)


def threshold_adaptive_spread(
    query: str,
    query_vec: Vec,
    similarities: List[float],
    spread_factor: float = 0.5,
    min_threshold: float = 0.4,
) -> float:
    """Threshold based on similarity spread (std dev).

    When similarities are tightly clustered, be more selective (higher threshold).
    When spread is large, be more permissive (lower threshold).

    Args:
        query: Original query string.
        query_vec: Normalized query embedding vector.
        similarities: Cosine similarities between query and each module centroid.
        spread_factor: Multiplier for spread influence (0-1).
        min_threshold: Minimum absolute threshold value.

    Returns:
        Threshold = max(1 - spread_factor * std, min_threshold)
    """
    if len(similarities) < 2:
        return min_threshold
    std = float(np.std(similarities))
    # Normalize std: cosine similarities range ~[-1, 1], std max ~1
    normalized_std = float(min(std, 1.0))
    threshold = max(1.0 - spread_factor * normalized_std, min_threshold)
    return float(threshold)


def threshold_query_length_adaptive(
    query: str,
    query_vec: Vec,
    similarities: List[float],
    length_factor: float = 0.01,
    base_threshold: float = 0.5,
) -> float:
    """Adjust threshold based on query length (longer queries → lower threshold).

    Longer queries are often more specific and may match fewer domains.

    Args:
        query: Original query string.
        query_vec: Normalized query embedding vector.
        similarities: Cosine similarities between query and each module centroid.
        length_factor: How much to reduce threshold per character.
        base_threshold: Starting threshold for zero-length query.

    Returns:
        threshold = max(base_threshold - length_factor * len(query), 0.1)
    """
    query_len = len(query)
    threshold = base_threshold - length_factor * query_len
    return max(threshold, 0.1)
