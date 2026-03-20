"""Tests for dynamic thresholding heuristics."""

import numpy as np
from kalmanorix.threshold_heuristics import (
    threshold_top_k,
    threshold_relative_to_max,
    threshold_adaptive_spread,
    threshold_query_length_adaptive,
)


def test_threshold_top_k():
    """Test top-k threshold heuristic."""
    query = "test"
    query_vec = np.array([1.0, 0.0])
    similarities = [0.9, 0.8, 0.7, 0.6]

    # k=1: select only top module, threshold = 0.9
    thresh = threshold_top_k(query, query_vec, similarities, k=1)
    assert abs(thresh - 0.9) < 1e-10

    # k=2: select top 2, threshold = 0.8 (2nd highest)
    thresh = threshold_top_k(query, query_vec, similarities, k=2)
    assert abs(thresh - 0.8) < 1e-10

    # k >= len: select all, threshold = -1
    thresh = threshold_top_k(query, query_vec, similarities, k=4)
    assert thresh == -1.0
    thresh = threshold_top_k(query, query_vec, similarities, k=5)
    assert thresh == -1.0

    # Edge case: empty similarities
    thresh = threshold_top_k(query, query_vec, [], k=1)
    assert thresh == -1.0


def test_threshold_relative_to_max():
    """Test threshold as fraction of maximum similarity."""
    query = "test"
    query_vec = np.array([1.0, 0.0])
    similarities = [0.9, 0.8, 0.7]

    # Default fraction=0.8 → 0.8 * 0.9 = 0.72, above min=0.3
    thresh = threshold_relative_to_max(query, query_vec, similarities)
    assert abs(thresh - 0.72) < 1e-10

    # fraction=0.5 → 0.5 * 0.9 = 0.45
    thresh = threshold_relative_to_max(query, query_vec, similarities, fraction=0.5)
    assert abs(thresh - 0.45) < 1e-10

    # fraction=0.8 but max=0.1 → 0.08, below min=0.3 → returns 0.3
    similarities_low = [0.1, 0.05]
    thresh = threshold_relative_to_max(query, query_vec, similarities_low)
    assert thresh == 0.3

    # Empty similarities returns min_threshold
    thresh = threshold_relative_to_max(query, query_vec, [])
    assert thresh == 0.3


def test_threshold_adaptive_spread():
    """Test threshold based on similarity spread."""
    query = "test"
    query_vec = np.array([1.0, 0.0])

    # Tight cluster (low spread) → high threshold
    similarities_tight = [0.9, 0.89, 0.91]
    thresh = threshold_adaptive_spread(query, query_vec, similarities_tight)
    # std ≈ 0.01, threshold ≈ 1 - 0.5*0.01 = 0.995 → min 0.4 → 0.995
    assert thresh > 0.99

    # Wide spread → lower threshold
    similarities_wide = [0.9, 0.5, 0.2]
    thresh = threshold_adaptive_spread(query, query_vec, similarities_wide)
    # std ≈ 0.29, threshold ≈ 1 - 0.5*0.29 = 0.855
    assert 0.85 < thresh < 0.86

    # Edge cases
    thresh = threshold_adaptive_spread(query, query_vec, [0.5])
    assert thresh == 0.4  # min_threshold
    thresh = threshold_adaptive_spread(query, query_vec, [])
    assert thresh == 0.4


def test_threshold_query_length_adaptive():
    """Test threshold adjustment based on query length."""
    query = "short"
    query_vec = np.array([1.0, 0.0])
    similarities = [0.9, 0.8]

    # Base threshold 0.5, length_factor 0.01, len=5 → 0.5 - 0.05 = 0.45
    thresh = threshold_query_length_adaptive(query, query_vec, similarities)
    assert thresh == 0.45

    # Longer query → lower threshold
    long_query = "a" * 50  # 50 chars
    thresh = threshold_query_length_adaptive(long_query, query_vec, similarities)
    # 0.5 - 0.01*50 = 0.0 → min 0.1
    assert thresh == 0.1

    # Very short query (len=1) → 0.5 - 0.01 = 0.49
    short_query = "a"
    thresh = threshold_query_length_adaptive(short_query, query_vec, similarities)
    assert thresh == 0.49

    # Edge case: zero length query (shouldn't happen but test)
    empty_query = ""
    thresh = threshold_query_length_adaptive(empty_query, query_vec, similarities)
    assert thresh == 0.5
