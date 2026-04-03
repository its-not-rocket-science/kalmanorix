"""
Generate mixed-domain test set for Milestone 2.1 evaluation.

Creates a retrieval corpus with specified proportions of pure and mixed documents.
Saves documents and queries to JSON for reproducibility.
"""

# pylint: disable=wrong-import-position,import-outside-toplevel,logging-fstring-interpolation

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple, Union

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from experiments.config import TrainingConfig, DomainEnum
from kalmanorix.datasets import load_multiple_domains

logger = logging.getLogger(__name__)


def generate_test_set(
    config: TrainingConfig,
    output_path: Union[str, Path],
    use_real_data: bool = True,
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Generate mixed-domain test set according to configuration.

    Parameters
    ----------
    config : TrainingConfig
        Experiment configuration specifying domains and proportions.
    output_path : Union[str, Path]
        Path to save test set JSON.
    use_real_data : bool
        Whether to load real datasets (requires internet).
        If False, uses synthetic data from toy_corpus.

    Returns
    -------
    Tuple[List[str], List[Tuple[str, int]]]
        (documents, queries) where queries are (text, doc_index).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Generating test set with proportions: {config.mixed_test_proportions}"
    )
    logger.info(f"Domains: {[d.value for d in config.domains]}")

    if use_real_data:
        # Load domain datasets
        domain_datasets = load_multiple_domains(
            domains=[d.value for d in config.domains],
            samples_per_domain=config.samples_per_domain // 10,  # Smaller for test
            cache=True,
            force_download=False,
        )

        # Convert to format expected by create_mixed_test_set
        from kalmanorix.datasets import create_mixed_test_set

        docs, queries = create_mixed_test_set(
            domain_datasets=domain_datasets,
            proportions=config.mixed_test_proportions,
            seed=config.seed,
        )

    else:
        # Use synthetic data
        from kalmanorix.toy_corpus import generate_anchor_sentences

        # Generate documents according to proportions
        total_docs = 1000  # Manageable size for testing
        docs = []
        queries = []

        # Map domain enum to toy_corpus domain tag
        domain_tags = {
            DomainEnum.MEDICAL: "medical",
            DomainEnum.LEGAL: "legal",
            DomainEnum.TECH: "tech",
            DomainEnum.COOK: "cook",
            DomainEnum.GENERAL: "general",
        }

        # Track where each domain's docs start
        domain_starts = {}

        for domain_type, proportion in config.mixed_test_proportions.items():
            n_docs = int(total_docs * proportion)
            if n_docs == 0:
                continue

            if domain_type == "mixed":
                # Create mixed documents by combining sentences from multiple domains
                for i in range(n_docs):
                    # Pick 2-3 random domains
                    n_mix = np.random.randint(2, min(4, len(config.domains) + 1))
                    mix_domains = np.random.choice(
                        [d.value for d in config.domains], n_mix, replace=False
                    )

                    parts = []
                    for mix_domain in mix_domains:
                        # Generate a sentence for this domain
                        tag = domain_tags.get(DomainEnum(mix_domain), "general")
                        sentence = generate_anchor_sentences(
                            n=1,
                            domains=(tag,),
                            seed=config.seed + hash(mix_domain) + i,
                        )[0]
                        parts.append(sentence)

                    mixed_doc = " ".join(parts)
                    docs.append(mixed_doc)

                # Create queries for mixed docs
                n_queries = max(1, int(n_docs * 0.1))
                query_idx = np.random.choice(n_docs, n_queries, replace=False)
                for q_idx in query_idx:
                    q_idx_int = int(q_idx)
                    doc_idx = len(docs) - n_docs + q_idx_int
                    queries.append((docs[doc_idx], doc_idx))

            else:
                # Pure domain document
                domain = DomainEnum(domain_type)
                tag = domain_tags.get(domain, "general")

                # Generate domain-specific documents
                domain_docs = generate_anchor_sentences(
                    n=n_docs,
                    domains=(tag,),
                    seed=config.seed + hash(domain_type),
                )
                domain_starts[domain_type] = len(docs)
                docs.extend(domain_docs)

                # Create queries (sample some documents as queries)
                n_queries = max(1, int(n_docs * 0.1))
                query_idx = np.random.choice(n_docs, n_queries, replace=False)
                for q_idx in query_idx:
                    q_idx_int = int(q_idx)
                    doc_idx = domain_starts[domain_type] + q_idx_int
                    queries.append((domain_docs[q_idx_int], doc_idx))

    logger.info(f"Generated {len(docs)} documents and {len(queries)} queries")

    # Save to JSON
    test_set = {
        "config": {
            "domains": [d.value for d in config.domains],
            "mixed_test_proportions": config.mixed_test_proportions,
            "seed": config.seed,
        },
        "documents": docs,
        "queries": [{"query": q, "true_doc_id": doc_id} for q, doc_id in queries],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    logger.info(f"Test set saved to {output_path}")
    return docs, queries


def load_test_set(path: Union[str, Path]) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Load test set from JSON file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to test set JSON.

    Returns
    -------
    Tuple[List[str], List[Tuple[str, int]]]
        (documents, queries) where queries are (text, doc_index).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data["documents"]
    queries = [(q["query"], q["true_doc_id"]) for q in data["queries"]]

    return docs, queries


def main() -> None:
    """Command-line entry point."""
    import argparse
    from experiments.config import load_config

    parser = argparse.ArgumentParser(
        description="Generate mixed-domain test set for retrieval evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/data/test_set.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real datasets",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print statistics about generated test set",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load configuration
    config = load_config(args.config)

    # Generate test set
    docs, queries = generate_test_set(
        config,
        output_path=args.output,
        use_real_data=not args.synthetic,
    )

    if args.print_stats:
        print("Test Set Statistics:")
        print(f"  Documents: {len(docs)}")
        print(f"  Queries: {len(queries)}")
        print(
            f"  Avg document length: {np.mean([len(d.split()) for d in docs]):.1f} words"
        )
        print(
            f"  Avg query length: {np.mean([len(q.split()) for q, _ in queries]):.1f} words"
        )

        # Domain distribution in documents (approximate)
        domain_keywords = {
            "medical": ["patient", "treatment", "medicine", "diagnosis"],
            "legal": ["case", "court", "law", "legal", "judge"],
            "tech": ["computer", "software", "algorithm", "network"],
            "cook": ["recipe", "ingredient", "cook", "food", "flavor"],
        }

        print("\nDocument domain analysis:")
        for domain, keywords in domain_keywords.items():
            count = sum(
                1 for doc in docs if any(keyword in doc.lower() for keyword in keywords)
            )
            print(f"  {domain}: {count} ({count / len(docs) * 100:.1f}%)")


if __name__ == "__main__":
    main()
