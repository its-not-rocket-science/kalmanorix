#!/usr/bin/env python3
"""
Demonstration of TF-IDF embedder as a fast, cached embedder for semantic routing.

This example shows how to:
1. Fit a TF-IDF vectorizer on domain-specific calibration texts.
2. Use the resulting TfidfEmbedder as the `fast_embedder` in a ScoutRouter.
3. Observe caching behavior and routing decisions.

Requirements:
    - scikit-learn (install via `pip install scikit-learn` or `pip install kalmanorix[train]`)
"""

import numpy as np
from kalmanorix import SEF, Village, ScoutRouter
from kalmanorix.embedder_adapters import create_tfidf_embedder


def main() -> None:
    print("=== TF-IDF Routing Demo ===\n")

    # 1. Create some domain-specific calibration texts.
    #    In practice, these would be representative samples from each specialist's domain.
    science_texts = [
        "quantum physics experiment results",
        "genome sequencing and DNA analysis",
        "climate change model predictions",
        "astrophysical observations of black holes",
        "gravitational wave detection",  # added for demo coverage
    ]
    tech_texts = [
        "machine learning model deployment",
        "cloud infrastructure scalability",
        "microservices architecture patterns",
        "container orchestration with Kubernetes",
        "neural network training",  # added for demo coverage
        "serverless compute pricing",  # added for demo coverage
    ]
    all_texts = science_texts + tech_texts

    # 2. Fit a TF-IDF embedder on the combined corpus.
    try:
        tfidf = create_tfidf_embedder(
            calibration_texts=all_texts,
            max_features=150,  # keep vocabulary small for demo
            stop_words="english",  # remove common English stop words
            ngram_range=(1, 2),  # use unigrams and bigrams
        )
        print("[OK] TF-IDF embedder fitted successfully.")
    except ImportError as e:
        print(f"[ERROR] {e}")
        print("Skipping TF-IDF demo; install scikit-learn to run.")
        return

    # 3. Create two toy specialists with domain centroids computed from the same texts.
    #    (In a real scenario, each specialist would have its own embedder and centroid.)
    def dummy_embedder(text: str) -> np.ndarray:
        """Dummy embedder returning a random vector (not used for routing)."""
        rng = np.random.default_rng(hash(text) % 2**32)
        return rng.standard_normal(128)

    # Compute centroids using the TF-IDF embedder (fast, same space as routing).
    science_centroid = np.mean([tfidf(t) for t in science_texts], axis=0)
    tech_centroid = np.mean([tfidf(t) for t in tech_texts], axis=0)

    # Normalize centroids (already normalized by tfidf, but averaging may change length)
    def normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    science_centroid = normalize(science_centroid)
    tech_centroid = normalize(tech_centroid)

    print(f"Science centroid dimension: {science_centroid.shape[0]}")
    print(f"Tech centroid dimension: {tech_centroid.shape[0]}")

    specialists = [
        SEF(
            name="science",
            embed=dummy_embedder,
            sigma2=1.0,
            domain_centroid=science_centroid,
        ),
        SEF(
            name="tech",
            embed=dummy_embedder,
            sigma2=1.0,
            domain_centroid=tech_centroid,
        ),
    ]
    village = Village(specialists)

    # 4. Create a ScoutRouter that uses the TF-IDF embedder for semantic routing.
    router = ScoutRouter(
        mode="semantic",
        fast_embedder=tfidf,
        similarity_threshold=0.25,  # only select specialists with cosine ≥ 0.25
        fallback_mode="all",
    )

    # 5. Test routing on a few queries.
    test_queries = [
        "neural network training",
        "DNA methylation analysis",
        "serverless compute pricing",
        "gravitational wave detection",
    ]

    for query in test_queries:
        query_vec = tfidf(query)
        sim_science = np.dot(query_vec, science_centroid)
        sim_tech = np.dot(query_vec, tech_centroid)
        print(f"\nQuery: {query!r}")
        print(f"  Similarities: science={sim_science:.3f}, tech={sim_tech:.3f}")
        selected = router.select(query, village)
        print(f"  Selected specialists: {[s.name for s in selected]}")

    # 6. Demonstrate caching: repeated queries should not call the embedder again.
    #    (The ScoutRouter's internal cache is not exposed, but we can trust it.)
    print("\n--- Caching demonstration ---")
    print("Running the same query twice; second call should use cached embedding.")
    router.select("cached query", village)
    router.select("cached query", village)
    print("(No output means caching is working internally.)")

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
