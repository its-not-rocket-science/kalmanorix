"""
Out-of-domain (OOD) test set creation for Milestone 2.2 (uncertainty robustness).

Creates test sets where documents come from seen domains but queries come from
an unseen domain, enabling evaluation of generalization under distribution shift.
"""

from __future__ import annotations

import logging
import random

import numpy as np

from .datasets import Domain, DomainDataset

logger = logging.getLogger(__name__)


def create_ood_test_set(
    domain_datasets: dict[Domain, DomainDataset],
    seen_domains: list[Domain],
    ood_domain: Domain,
    ood_proportion: float = 0.5,
    n_docs: int = 5000,
    n_queries: int = 500,
    seed: int = 42,
) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Create an out-of-domain test set for retrieval evaluation.

    Documents are sampled from seen domains (excluding the OOD domain).
    Queries are sampled from the OOD domain (truly unseen).

    Parameters
    ----------
    domain_datasets : dict[Domain, DomainDataset]
        Loaded domain datasets (must contain test splits).
    seen_domains : list[Domain]
        List of domains that have been seen during training.
        These domains will contribute documents to the corpus.
    ood_domain : Domain
        Domain that is completely unseen during training.
        Queries will be sampled from this domain.
    ood_proportion : float, optional
        Proportion of queries that should be from OOD domain.
        The rest are sampled from seen domains (for comparison).
        Default: 0.5 (half OOD, half seen).
    n_docs : int, optional
        Total number of documents in corpus. Default: 5000.
    n_queries : int, optional
        Total number of queries. Default: 500.
    seed : int, optional
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    tuple[list[str], list[tuple[str, int]]]
        (documents, queries) where queries are (text, true_doc_id).
        For OOD queries, true_doc_id = -1 (no matching document in corpus).
        For seen-domain queries, true_doc_id points to the matching document.

    Notes
    -----
    - The OOD domain is excluded from the document corpus entirely.
    - For seen-domain queries, we follow the standard practice of using a
      document as its own query (self-retrieval) with true_doc_id pointing
      to that document's position in the corpus.
    - For OOD queries, there is no matching document in the corpus, so
      true_doc_id = -1. This allows measuring recall@k only for seen-domain
      queries while still evaluating embedding quality on OOD queries via
      other metrics (e.g., calibration).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Validate inputs
    if not 0.0 <= ood_proportion <= 1.0:
        raise ValueError(f"ood_proportion must be in [0, 1], got {ood_proportion}")
    if ood_domain in seen_domains:
        raise ValueError(
            f"OOD domain {ood_domain} cannot be in seen_domains {seen_domains}"
        )
    for domain in seen_domains + [ood_domain]:
        if domain not in domain_datasets:
            raise ValueError(f"Domain {domain} not in provided domain_datasets")

    # 1. Sample documents from seen domains only
    docs_by_domain: dict[Domain, list[str]] = {}
    remaining_docs = n_docs
    n_seen = len(seen_domains)

    # Distribute documents evenly across seen domains
    for i, domain in enumerate(seen_domains):
        dataset = domain_datasets[domain]
        if not dataset.test:
            raise ValueError(f"Domain {domain} has empty test split")

        # Calculate number of documents for this domain
        if i == n_seen - 1:
            n_domain = remaining_docs  # last domain gets remaining
        else:
            n_domain = n_docs // n_seen
            remaining_docs -= n_domain

        # Sample without replacement if possible
        test_size = len(dataset.test)
        if n_domain > test_size:
            logger.warning(
                "Domain %s test split size (%d) smaller than requested (%d), "
                "sampling with replacement",
                domain,
                test_size,
                n_domain,
            )
            idx = np.random.choice(test_size, n_domain, replace=True)
        else:
            idx = np.random.choice(test_size, n_domain, replace=False)

        docs_by_domain[domain] = [dataset.test[j] for j in idx]

    # Flatten documents and keep track of domain for each document
    documents: list[str] = []
    doc_domain: list[Domain] = []
    for domain in seen_domains:
        docs = docs_by_domain[domain]
        documents.extend(docs)
        doc_domain.extend([domain] * len(docs))

    # 2. Create queries
    queries: list[tuple[str, int]] = []

    # Number of OOD queries
    n_ood = int(n_queries * ood_proportion)
    n_seen_queries = n_queries - n_ood

    # Seen-domain queries: sample from documents themselves (self-retrieval)
    if n_seen_queries > 0:
        if n_seen_queries > len(documents):
            logger.warning(
                "More seen-domain queries (%d) than documents (%d), "
                "sampling with replacement",
                n_seen_queries,
                len(documents),
            )
            query_indices = np.random.choice(
                len(documents), n_seen_queries, replace=True
            )
        else:
            query_indices = np.random.choice(
                len(documents), n_seen_queries, replace=False
            )

        for idx in query_indices:
            queries.append((documents[idx], idx))  # self-retrieval

    # OOD queries: sample from OOD domain test split
    if n_ood > 0:
        ood_dataset = domain_datasets[ood_domain]
        if not ood_dataset.test:
            raise ValueError(f"OOD domain {ood_domain} has empty test split")

        ood_test = ood_dataset.test
        if n_ood > len(ood_test):
            logger.warning(
                "OOD domain test split size (%d) smaller than requested (%d), "
                "sampling with replacement",
                len(ood_test),
                n_ood,
            )
            idx = np.random.choice(len(ood_test), n_ood, replace=True)
        else:
            idx = np.random.choice(len(ood_test), n_ood, replace=False)

        for j in idx:
            queries.append((ood_test[j], -1))  # -1 indicates no matching document

    # Shuffle queries to mix OOD and seen-domain queries
    random.shuffle(queries)

    logger.info(
        "Created OOD test set: %d documents (domains: %s), %d queries (%d OOD, %d seen)",
        len(documents),
        ", ".join(seen_domains),
        len(queries),
        n_ood,
        n_seen_queries,
    )

    return documents, queries


def create_synthetic_ood_test_set(
    seen_domains: list[Domain],
    ood_domain: Domain,
    n_docs: int = 1000,
    n_queries: int = 100,
    seed: int = 42,
) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Create a synthetic OOD test set using toy corpus when real data is unavailable.

    Useful for testing and development without downloading full datasets.

    Parameters
    ----------
    seen_domains : list[Domain]
        Domains considered "seen" during training.
    ood_domain : Domain
        Unseen domain for OOD queries.
    n_docs : int, optional
        Number of documents. Default: 1000.
    n_queries : int, optional
        Number of queries. Default: 100.
    seed : int, optional
        Random seed. Default: 42.

    Returns
    -------
    tuple[list[str], list[tuple[str, int]]]
        Same format as `create_ood_test_set`.
    """
    from .toy_corpus import generate_anchor_sentences

    random.seed(seed)
    np.random.seed(seed)

    # Generate synthetic documents from seen domains
    documents = []
    n_per_domain = n_docs // len(seen_domains)
    for domain in seen_domains:
        # Map domain to toy_corpus domain tag
        domain_map = {
            "medical": "medical",
            "legal": "general",  # No legal-specific vocabulary
            "tech": "tech",
            "cook": "cook",
            "general": "general",
        }
        tag = domain_map.get(domain, "general")
        sentences = generate_anchor_sentences(
            n=n_per_domain,
            domains=(tag,),
            seed=seed + hash(domain),
        )
        documents.extend(sentences)

    # Pad if needed
    if len(documents) < n_docs:
        extra = n_docs - len(documents)
        sentences = generate_anchor_sentences(
            n=extra,
            domains=("general",),
            seed=seed + 999,
        )
        documents.extend(sentences)

    # Generate OOD queries
    domain_map = {
        "medical": "medical",
        "legal": "general",
        "tech": "tech",
        "cook": "cook",
        "general": "general",
    }
    tag = domain_map.get(ood_domain, "general")
    ood_sentences = generate_anchor_sentences(
        n=n_queries,
        domains=(tag,),
        seed=seed + 12345,
    )

    # Create queries (all OOD, no matching documents)
    queries = [(sentence, -1) for sentence in ood_sentences]

    logger.info(
        "Created synthetic OOD test set: %d documents (domains: %s), %d OOD queries (%s)",
        len(documents),
        ", ".join(seen_domains),
        len(queries),
        ood_domain,
    )

    return documents, queries
