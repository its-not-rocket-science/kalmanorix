"""
Dataset loading utilities for Milestone 2.1 (specialists vs monolith experiment).

Provides loaders for public domain-specific datasets:
- Medical: PubMed abstracts (scientific_papers pubmed subset)
- Legal: Case law (lex_glue or casehold)
- Tech: Stack Overflow (optional)
- Cooking: Recipe datasets (optional)

All loaders include train/validation/test splits, text preprocessing,
and caching to avoid repeated downloads.
"""

# pylint: disable=too-many-branches,too-many-statements,import-outside-toplevel,broad-exception-caught

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, cast

import numpy as np

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

logger = logging.getLogger(__name__)

# Type aliases
Domain = Literal["medical", "legal", "tech", "cook", "general"]
Split = Literal["train", "validation", "test"]


class DatasetConfig(TypedDict, total=False):
    """Configuration for dataset loading."""

    name: str
    subset: Optional[str]
    text_column: str
    max_tokens: int
    samples_per_split: Optional[int]
    seed: int


# Default configurations for each domain
DEFAULT_CONFIGS: dict[Domain, DatasetConfig] = {
    "medical": {
        "name": "scientific_papers",
        "subset": "pubmed",
        "text_column": "article",
        "max_tokens": 512,
        "samples_per_split": 50_000,
        "seed": 42,
    },
    "legal": {
        "name": "lex_glue",
        "subset": "case_hold",
        "text_column": "context",
        "max_tokens": 512,
        "samples_per_split": 50_000,
        "seed": 42,
    },
    "tech": {
        "name": "stack_overflow",
        "subset": None,
        "text_column": "text",
        "max_tokens": 512,
        "samples_per_split": 50_000,
        "seed": 42,
    },
    "cook": {
        "name": "recipe_nlg",
        "subset": None,
        "text_column": "title",
        "max_tokens": 512,
        "samples_per_split": 50_000,
        "seed": 42,
    },
}


@dataclass(frozen=True)
class DomainDataset:
    """Container for domain-specific dataset splits."""

    domain: Domain
    train: list[str]
    validation: list[str]
    test: list[str]
    config: DatasetConfig


def _cache_key(config: DatasetConfig, split: Split) -> str:
    """Generate a deterministic cache key for a dataset config."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(f"{config_str}:{split}".encode()).hexdigest()[:16]


def _get_cache_path(config: DatasetConfig, split: Split) -> Path:
    """Return local cache path for a dataset split."""
    cache_dir = Path("experiments") / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(config, split)
    return cache_dir / f"{key}.jsonl"


def _load_from_cache(cache_path: Path, limit: Optional[int] = None) -> list[str]:
    """Load preprocessed texts from cache."""
    texts = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            if limit is not None and len(texts) >= limit:
                break
    return texts


def _save_to_cache(cache_path: Path, texts: list[str]) -> None:
    """Save preprocessed texts to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump({"text": text}, f)
            f.write("\n")


def _preprocess_text(text: str, tokenizer, max_tokens: int) -> str:
    """Clean and truncate text to max_tokens."""
    if not text or not isinstance(text, str):
        return ""

    # Basic cleaning
    text = " ".join(text.split())  # normalize whitespace

    # Tokenize and truncate if tokenizer available
    if tokenizer is not None:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        # Decode back to text (approximate)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        # Fallback: simple word-based truncation
        words = text.split()
        if len(words) > max_tokens:
            words = words[:max_tokens]
        text = " ".join(words)

    return text.strip()


def load_domain_dataset(
    domain: Domain,
    config_overrides: Optional[DatasetConfig] = None,
    cache: bool = True,
    force_download: bool = False,
) -> DomainDataset:
    """
    Load a domain-specific dataset with train/validation/test splits.

    Parameters
    ----------
    domain : Domain
        Domain identifier (medical, legal, tech, cook).
    config_overrides : Optional[DatasetConfig]
        Override default configuration.
    cache : bool
        Whether to cache processed texts locally.
    force_download : bool
        Whether to force re-download even if cache exists.

    Returns
    -------
    DomainDataset
        Container with train/validation/test text lists.

    Raises
    ------
    ImportError
        If datasets or transformers not installed.
    ValueError
        If domain not supported.
    """
    if not HAS_DATASETS:
        raise ImportError(
            "Datasets library required. Install with: pip install 'kalmanorix[train]'"
        )

    if domain not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Domain {domain} not supported. Choose from {list(DEFAULT_CONFIGS.keys())}"
        )

    # Merge config
    config = DEFAULT_CONFIGS[domain].copy()
    if config_overrides:
        config.update(config_overrides)

    # Load tokenizer for truncation
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        logger.warning("Could not load tokenizer, using simple truncation")
        tokenizer = None

    splits_texts: dict[Split, list[str]] = {}

    for split in ("train", "validation", "test"):
        cache_path = _get_cache_path(config, split) if cache else None

        # Try cache first
        if (
            cache
            and cache_path is not None
            and cache_path.exists()
            and not force_download
        ):
            assert cache_path is not None
            logger.info("Loading %s %s from cache: %s", domain, split, cache_path)
            texts = _load_from_cache(cache_path, config.get("samples_per_split"))
            splits_texts[split] = texts
            continue

        # Download from Hugging Face
        logger.info("Downloading %s %s dataset...", domain, split)
        dataset_name = config["name"]
        subset = config.get("subset")

        try:
            if subset:
                hf_dataset = load_dataset(dataset_name, subset, split=split)
            else:
                hf_dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            logger.warning("Failed to load %s %s: %s", dataset_name, split, e)
            # Fallback: generate synthetic data
            logger.info("Generating synthetic %s data for %s", domain, split)
            n_samples = config.get("samples_per_split") or 1000
            splits_texts[split] = _generate_synthetic_data(domain, n_samples)
            continue

        # Extract texts
        text_column = config["text_column"]
        raw_texts = []
        for example in hf_dataset:
            if text_column in example:
                raw_texts.append(example[text_column])
            # Try common fallback columns
            elif "text" in example:
                raw_texts.append(example["text"])
            elif "content" in example:
                raw_texts.append(example["content"])

        # Preprocess
        max_tokens = config.get("max_tokens", 512)
        texts = []
        for text in raw_texts:
            processed = _preprocess_text(text, tokenizer, max_tokens)
            if processed:  # Skip empty
                texts.append(processed)

        # Limit samples
        samples: Optional[int] = config.get("samples_per_split")
        if samples is not None:
            np.random.seed(config["seed"])
            if len(texts) > samples:
                idx = np.random.choice(len(texts), samples, replace=False)
                texts = [texts[i] for i in idx]

        splits_texts[split] = texts

        # Cache if requested
        if cache and cache_path:
            logger.info("Caching %s %s to %s", domain, split, cache_path)
            _save_to_cache(cache_path, texts)

    return DomainDataset(
        domain=domain,
        train=splits_texts["train"],
        validation=splits_texts["validation"],
        test=splits_texts["test"],
        config=config,
    )


def _generate_synthetic_data(domain: Domain, n: int) -> list[str]:
    """Generate synthetic domain-specific text when real data unavailable."""
    from .toy_corpus import generate_anchor_sentences

    # Map domain to toy_corpus domain tag
    domain_map = {
        "medical": "medical",
        "legal": "general",  # No legal-specific vocabulary in toy_corpus
        "tech": "tech",
        "cook": "cook",
    }
    tag = domain_map.get(domain, "general")

    # Generate sentences with domain vocabulary
    sentences = generate_anchor_sentences(
        n=n,
        domains=(tag,),
        seed=hash(domain) % 1000,
    )

    # Add domain prefix for realism
    prefix = f"[{domain.upper()}] "
    return [prefix + s for s in sentences]


def load_multiple_domains(
    domains: list[Domain],
    samples_per_domain: Optional[int] = None,
    **kwargs,
) -> dict[Domain, DomainDataset]:
    """
    Load multiple domain datasets.

    Parameters
    ----------
    domains : list[Domain]
        List of domains to load.
    samples_per_domain : Optional[int]
        Override default samples per split.
    **kwargs
        Passed to load_domain_dataset.

    Returns
    -------
    dict[Domain, DomainDataset]
        Mapping from domain to loaded dataset.
    """
    result = {}
    for domain in domains:
        config: DatasetConfig = {}
        if samples_per_domain is not None:
            config["samples_per_split"] = samples_per_domain

        result[domain] = load_domain_dataset(
            domain,
            config_overrides=config if config else None,
            **kwargs,
        )
    return result


def create_mixed_test_set(
    domain_datasets: dict[Domain, DomainDataset],
    proportions: dict[str, float],
    seed: int = 42,
) -> Tuple[list[str], list[Tuple[str, int]]]:
    """
    Create a mixed-domain test corpus for retrieval evaluation.

    Parameters
    ----------
    domain_datasets : dict[Domain, DomainDataset]
        Loaded domain datasets (must contain test splits).
    proportions : dict[str, float]
        Mapping from domain to proportion in corpus.
        Domain can be "legal", "medical", or "mixed" (requires multiple domains).
    seed : int
        Random seed for sampling.

    Returns
    -------
    Tuple[list[str], list[Tuple[str, int]]]
        (documents, queries) where queries are (text, doc_index).

    Notes
    -----
    For "mixed" documents, we create hybrid texts by combining terms from
    multiple domains.
    """
    np.random.seed(seed)

    # Validate proportions sum to 1.0
    total = sum(proportions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Proportions must sum to 1.0, got {total}")

    # Determine total size based on smallest domain test set
    min_test_size = min(len(ds.test) for ds in domain_datasets.values())
    total_docs = min(10_000, min_test_size * len(proportions))  # Reasonable limit

    docs: list[str] = []
    queries: list[tuple[str, int]] = []

    for domain_type, proportion in proportions.items():
        n_docs = int(total_docs * proportion)
        if n_docs == 0:
            continue

        if domain_type in domain_datasets:
            # Pure domain
            domain = cast(Domain, domain_type)
            dataset = domain_datasets[domain]
            # Sample from test split
            idx = np.random.choice(len(dataset.test), n_docs, replace=False)
            domain_docs = [dataset.test[i] for i in idx]

            # Create queries (sample some documents as queries)
            n_queries = max(1, int(n_docs * 0.1))  # 10% of docs as queries
            query_idx = np.random.choice(n_docs, n_queries, replace=False)
            for q_idx in query_idx:
                doc_idx = len(docs) + q_idx  # Position in final corpus
                queries.append((domain_docs[q_idx], doc_idx))

            docs.extend(domain_docs)

        elif domain_type == "mixed":
            # Mixed domain: combine texts from multiple domains
            all_domains = list(domain_datasets.keys())
            for _ in range(n_docs):
                # Sample 2-3 domains to mix
                n_mix = np.random.randint(2, min(4, len(all_domains) + 1))
                mix_domains = np.random.choice(all_domains, n_mix, replace=False)

                # Take a sentence from each domain
                parts = []
                for mix_domain in mix_domains:
                    dataset = domain_datasets[mix_domain]
                    if dataset.test:
                        sentence = np.random.choice(dataset.test)
                        parts.append(sentence)

                mixed_doc = " ".join(parts)
                docs.append(mixed_doc)

            # Create mixed queries (sample some mixed docs)
            n_queries = max(1, int(n_docs * 0.1))
            query_idx = np.random.choice(n_docs, n_queries, replace=False)
            for q_idx in query_idx:
                doc_idx = len(docs) - n_docs + q_idx  # Position of mixed docs
                queries.append((docs[doc_idx], doc_idx))

    return docs, queries
