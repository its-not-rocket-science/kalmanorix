"""
Validation of Kalman fusion on synthetic specialists.

This script corresponds to Milestone 1.3: Basic Kalman Fuser.
It tests whether Kalman fusion outperforms simple averaging on mixed-domain
retrieval, meets latency targets, and handles extreme covariance values robustly.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    MeanFuser,
    KalmanorixFuser,
)
from kalmanorix.toy_corpus import build_toy_corpus, generate_anchor_sentences
from kalmanorix.types import Embedder, Vec
from kalmanorix.uncertainty import KeywordSigma2
from kalmanorix.alignment import compute_alignments, align_sef_list
from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal


class KeywordEmbedder(Embedder):
    """
    Toy keyword-sensitive embedder (copied from examples/minimal_fusion_demo.py).

    Produces a deterministic base embedding (seeded), plus tiny deterministic
    per-text perturbations, and a strong directional bias if any keywords match.
    """

    dim: int
    keywords: List[str]
    keyword_boost: float
    _base_dir: np.ndarray
    _kw_dir: np.ndarray

    def __init__(
        self,
        dim: int,
        keywords: List[str],
        keyword_boost: float,
        _base_dir: np.ndarray,
        _kw_dir: np.ndarray,
    ):
        self.dim = dim
        self.keywords = keywords
        self.keyword_boost = keyword_boost
        self._base_dir = _base_dir
        self._kw_dir = _kw_dir

    def __call__(self, text: str) -> Vec:
        t = text.lower()

        # Tiny deterministic "noise" so different texts differ slightly.
        noise = np.zeros(self.dim, dtype=np.float64)
        for ch in t[:64]:
            noise[(ord(ch) * 13) % self.dim] += 0.01

        vec = self._base_dir + noise

        if any(kw in t for kw in self.keywords):
            vec = vec + self.keyword_boost * self._kw_dir

        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype(np.float64)


def make_keyword_embedder(
    dim: int,
    seed: int,
    keywords: List[str],
    keyword_boost: float = 2.5,
) -> KeywordEmbedder:
    """
    Construct a deterministic keyword-sensitive embedder.

    Returns an object implementing kalmanorix.types.Embedder:
        embed(text) -> np.ndarray[dim] (unit-normalized float64)
    """
    rng = np.random.default_rng(seed)

    base_dir = rng.normal(size=(dim,))
    base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-12)

    kw_dir = rng.normal(size=(dim,))
    kw_dir = kw_dir / (np.linalg.norm(kw_dir) + 1e-12)

    return KeywordEmbedder(
        dim=dim,
        keywords=keywords,
        keyword_boost=keyword_boost,
        _base_dir=base_dir.astype(np.float64),
        _kw_dir=kw_dir.astype(np.float64),
    )


def build_doc_matrix(
    docs: List[str],
    *,
    village: Village,
    scout: ScoutRouter,
    pan: Panoramix,
) -> np.ndarray:
    """
    Embed every document using the same routing+fusion strategy as queries.

    This matches the evaluation methodology from mixed_domain_eval.py.
    """
    embs: List[np.ndarray] = []
    for d in docs:
        potion = pan.brew(d, village=village, scout=scout)
        embs.append(potion.vector)
    return np.stack(embs, axis=0)


def permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_perm: int = 10000,
) -> float:
    """Two-sided permutation test for difference in means."""
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    pooled = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)

    perm_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        perm_diffs.append(np.mean(perm_a) - np.mean(perm_b))

    # Two-sided p-value
    extreme_count = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff))
    return (extreme_count + 1) / (n_perm + 1)  # Add-1 smoothing


# pylint: disable=too-many-branches,too-many-statements
def run_retrieval_comparison(
    dimension: int = 100,
    seed: int = 42,
    n_trials: int = 100,
) -> Tuple[float, float, float]:
    """
    Compare Kalman vs Mean fusion on mixed-domain retrieval.

    Returns:
        kalman_score: Average Recall@1 for Kalman fusion
        mean_score: Average Recall@1 for Mean fusion
        p_value: Permutation test p-value for difference
    """
    np.random.seed(seed)
    corpus = build_toy_corpus(british_spelling=True)

    # Define keywords for each specialist
    tech_keywords = ["battery", "cpu", "gpu", "camera", "smartphone", "laptop"]
    cook_keywords = ["braise", "simmer", "sauté", "oven", "stew", "sauce"]

    # Create specialists with different uncertainty patterns
    # Tech specialist: certain on tech, uncertain on cooking
    tech_embed = make_keyword_embedder(dim=dimension, seed=7, keywords=tech_keywords)
    tech_sef = SEF(
        name="tech",
        embed=tech_embed,
        sigma2=KeywordSigma2(
            set(tech_keywords), in_domain_sigma2=0.1, out_domain_sigma2=0.5
        ),
        meta={"domain": "tech"},
    )

    # Cooking specialist: certain on cooking, uncertain on tech
    cook_embed = make_keyword_embedder(dim=dimension, seed=11, keywords=cook_keywords)
    cook_sef = SEF(
        name="cook",
        embed=cook_embed,
        sigma2=KeywordSigma2(
            set(cook_keywords), in_domain_sigma2=0.1, out_domain_sigma2=0.5
        ),
        meta={"domain": "cooking"},
    )

    village = Village([tech_sef, cook_sef])
    # Generate anchor sentences for alignment
    anchor_sentences = generate_anchor_sentences(n=500, seed=seed + 999)
    # Compute alignments to tech reference
    alignments = compute_alignments(
        sef_list=[tech_sef, cook_sef],
        anchor_sentences=anchor_sentences,
        reference_sef_name="tech",
        _epsilon=1e-8,
    )
    # Create aligned SEFs
    aligned_sefs = align_sef_list([tech_sef, cook_sef], alignments)
    village = Village(aligned_sefs)
    scout = ScoutRouter(mode="all")

    # Create fusers
    kalman_pan = Panoramix(fuser=KalmanorixFuser())
    mean_pan = Panoramix(fuser=MeanFuser())

    kalman_scores = []
    mean_scores = []

    # Identify mixed queries (where fusion matters most)
    mixed_indices = [i for i, g in enumerate(corpus.groups) if g == "mixed"]
    if not mixed_indices:
        raise ValueError("No mixed queries found in corpus")

    for trial in range(n_trials):
        # Shuffle mixed queries for each trial
        rng = np.random.default_rng(seed + trial)
        shuffled = rng.permutation(mixed_indices)
        # Use first half for warm-up, second half for eval
        split = len(shuffled) // 2
        eval_indices = shuffled[split:]

        # Build document matrices for each strategy
        kalman_docs = build_doc_matrix(
            corpus.docs, village=village, scout=scout, pan=kalman_pan
        )
        mean_docs = build_doc_matrix(
            corpus.docs, village=village, scout=scout, pan=mean_pan
        )

        # Evaluate on mixed queries subset
        for idx in eval_indices:
            query, true_id = corpus.queries[idx]
            # Kalman fusion
            kalman_potion = kalman_pan.brew(query, village=village, scout=scout)
            if trial == 0 and idx == eval_indices[0]:
                print(f"  Debug query: {query}")
                print(f"  Kalman weights: {kalman_potion.weights}")
                print(f"  Kalman vector norm: {np.linalg.norm(kalman_potion.vector)}")
                # Also check sigma2 values
                for module in village.modules:
                    sigma2 = module.sigma2_for(query)
                    print(f"    {module.name} sigma2: {sigma2}")
                # Compute similarities and top-3
                kalman_sim_local = kalman_docs @ kalman_potion.vector
                top_k = 5
                top_indices = np.argsort(kalman_sim_local)[-top_k:][::-1]
                print(f"  Kalman top-{top_k} docs:")
                for i, doc_idx in enumerate(top_indices):
                    sim = kalman_sim_local[doc_idx]
                    # Determine doc category
                    if doc_idx < 4:
                        cat = "tech"
                    elif doc_idx < 8:
                        cat = "cook"
                    else:
                        cat = "confuser"
                    print(f"    {i + 1}. doc {doc_idx} ({cat}): sim={sim:.4f}")
                # True doc info
                print(f"  True doc {true_id}: sim={kalman_sim_local[true_id]:.4f}")
            kalman_sim = kalman_docs @ kalman_potion.vector
            kalman_pred = np.argmax(kalman_sim)
            kalman_scores.append(1.0 if kalman_pred == true_id else 0.0)

            # Mean fusion
            mean_potion = mean_pan.brew(query, village=village, scout=scout)
            if trial == 0 and idx == eval_indices[0]:
                print(f"  Mean weights: {mean_potion.weights}")
                print(f"  Mean vector norm: {np.linalg.norm(mean_potion.vector)}")
                # Compute similarities and top-3 for mean fusion
                mean_sim_local = mean_docs @ mean_potion.vector
                top_k = 5
                top_indices = np.argsort(mean_sim_local)[-top_k:][::-1]
                print(f"  Mean top-{top_k} docs:")
                for i, doc_idx in enumerate(top_indices):
                    sim = mean_sim_local[doc_idx]
                    # Determine doc category
                    if doc_idx < 4:
                        cat = "tech"
                    elif doc_idx < 8:
                        cat = "cook"
                    else:
                        cat = "confuser"
                    print(f"    {i + 1}. doc {doc_idx} ({cat}): sim={sim:.4f}")
                # True doc info
                print(f"  True doc {true_id}: sim={mean_sim_local[true_id]:.4f}")
            mean_sim = mean_docs @ mean_potion.vector
            mean_pred = np.argmax(mean_sim)
            mean_scores.append(1.0 if mean_pred == true_id else 0.0)

    kalman_avg = np.mean(kalman_scores)
    mean_avg = np.mean(mean_scores)
    p_value = permutation_test(kalman_scores, mean_scores, n_perm=1000)

    return float(kalman_avg), float(mean_avg), float(p_value)


def test_numerical_stability(
    dimension: int = 100,
    n_trials: int = 50,
    seed: int = 42,
) -> Dict[str, bool]:
    """
    Test Kalman fusion with extreme covariance values.

    Returns dictionary of pass/fail results.
    """
    np.random.seed(seed)
    results = {}

    # Test covariance scales
    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    for scale in scales:
        stable = True
        for _ in range(n_trials):
            # Generate random embeddings and covariances
            emb1 = np.random.randn(dimension).astype(np.float64)
            emb2 = np.random.randn(dimension).astype(np.float64)
            cov1 = np.full(dimension, scale, dtype=np.float64)
            cov2 = np.full(dimension, scale * 0.1, dtype=np.float64)  # Different scale

            # Normalize embeddings for cosine similarity
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-12)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-12)

            # Run Kalman fusion
            try:
                fused, fused_cov = kalman_fuse_diagonal(
                    [emb1, emb2], [cov1, cov2], epsilon=1e-12
                )
                # Check for NaN/Inf
                if not np.all(np.isfinite(fused)) or not np.all(np.isfinite(fused_cov)):
                    stable = False
                    break
                # Check covariance remains positive
                if np.any(fused_cov < 0):
                    stable = False
                    break
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                print(f"  Scale {scale}: error {e}")
                stable = False
                break

        results[f"scale_{scale}"] = stable

    return results


# pylint: disable=too-many-statements
def profile_latency(
    dimension: int = 768,
    n_models: int = 5,
    n_queries: int = 100,
    seed: int = 42,
) -> Tuple[float, bool]:
    """
    Profile fusion latency for n_models.

    Returns average milliseconds per query and whether <50ms target is met.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Create synthetic specialists
    village = Village(
        [
            SEF(
                name=f"model_{i}",
                embed=lambda text: rng.normal(size=(dimension,)).astype(np.float64),
                sigma2=0.1 + i * 0.05,  # Different uncertainties
                meta={"synthetic": "true"},
            )
            for i in range(n_models)
        ]
    )

    scout = ScoutRouter(mode="all")
    kalman_pan = Panoramix(fuser=KalmanorixFuser())

    # Warm-up
    for _ in range(10):
        kalman_pan.brew("warm up", village=village, scout=scout)

    # Time queries
    query_texts = [f"query {i}" for i in range(n_queries)]
    start = time.perf_counter()
    for query in query_texts:
        kalman_pan.brew(query, village=village, scout=scout)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / n_queries
    target_met = avg_ms < 50.0

    return avg_ms, target_met


def run_validation(
    dimension: int = 100,
    retrieval_trials: int = 100,
    stability_trials: int = 50,
    latency_queries: int = 100,
    seed: int = 42,
):
    """
    Main validation function for Milestone 1.3.

    Args:
        dimension: Embedding dimension for synthetic specialists
        retrieval_trials: Number of trials for retrieval comparison
        stability_trials: Number of trials per covariance scale
        latency_queries: Number of queries for latency profiling
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    print("=" * 70)
    print("Kalman Fusion Validation (Milestone 1.3)")
    print("=" * 70)
    print(f"Dimension: {dimension}")
    print(f"Retrieval trials: {retrieval_trials}")
    print(f"Stability trials: {stability_trials}")
    print()

    # 1. Retrieval performance comparison
    print("--- Retrieval Performance Comparison ---")
    kalman_score, mean_score, p_value = run_retrieval_comparison(
        dimension=dimension,
        seed=seed,
        n_trials=retrieval_trials,
    )
    print(f"Kalman fusion Recall@1: {kalman_score:.4f}")
    print(f"Mean fusion Recall@1:   {mean_score:.4f}")
    print(f"Difference: {kalman_score - mean_score:.4f}")
    print(f"Permutation test p-value: {p_value:.4f}")

    if p_value < 0.05 and kalman_score > mean_score:
        print("[PASS] Kalman fusion significantly outperforms averaging (p < 0.05)")
        retrieval_pass = True
    elif p_value >= 0.05:
        print("[FAIL] No significant improvement over averaging (p >= 0.05)")
        retrieval_pass = False
    else:  # p_value < 0.05 but kalman_score <= mean_score
        print("[FAIL] Kalman fusion significantly worse than averaging (p < 0.05)")
        retrieval_pass = False

    # 2. Numerical stability
    print("\n--- Numerical Stability ---")
    stability_results = test_numerical_stability(
        dimension=dimension,
        n_trials=stability_trials,
        seed=seed,
    )
    all_stable = all(stability_results.values())
    for scale, stable in stability_results.items():
        status = "PASS" if stable else "FAIL"
        print(f"  {scale}: {status}")

    if all_stable:
        print("[PASS] All covariance scales handled stably")
        stability_pass = True
    else:
        print("[FAIL] Numerical issues with some covariance scales")
        stability_pass = False

    # 3. Latency profiling
    print("\n--- Latency Profiling ---")
    avg_ms, target_met = profile_latency(
        dimension=768,  # Realistic dimension
        n_models=5,
        n_queries=latency_queries,
        seed=seed,
    )
    print(f"Average fusion latency: {avg_ms:.2f} ms (5 models, d=768)")
    if target_met:
        print(f"[PASS] Latency < 50ms target ({avg_ms:.2f} ms)")
        latency_pass = True
    else:
        print(f"[FAIL] Latency exceeds 50ms target ({avg_ms:.2f} ms)")
        latency_pass = False

    # Overall verdict
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    criteria = {
        "Retrieval improvement (p < 0.05)": retrieval_pass,
        "Numerical stability": stability_pass,
        "Latency < 50ms": latency_pass,
    }

    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {criterion}")

    overall_pass = all(criteria.values())
    if overall_pass:
        print("\n[PASS] All Milestone 1.3 criteria met.")
    else:
        print("\n[FAIL] Some criteria not met.")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
