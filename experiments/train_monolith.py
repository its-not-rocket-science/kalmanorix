"""
Train a monolithic model on combined domain data.

Ensures compute equivalence with specialist training:
- Specialists: 1 epoch per domain
- Monolith: 2 epochs on combined data (same total examples)
"""

# pylint: disable=wrong-import-position,import-outside-toplevel,logging-fstring-interpolation,unused-variable

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from experiments.config import TrainingConfig, DomainEnum
from kalmanorix.compute_tracker import track_compute

logger = logging.getLogger(__name__)


@dataclass
class MonolithTrainingResult:
    """Result of monolithic training."""

    model_path: Path
    compute_metrics_path: Path
    config: TrainingConfig


def prepare_monolith_data(
    domain_data: dict[DomainEnum, List[str]],
    config: TrainingConfig,
) -> List[InputExample]:
    """
    Prepare training data for monolithic model.

    Combines all domain data and creates positive pairs via augmentation.
    """
    from experiments.train_specialists_st import (
        AugmentConfig,
        build_positive_pairs,
    )

    all_sentences = []
    for domain, sentences in domain_data.items():
        # Take up to samples_per_domain from each domain
        if config.samples_per_domain and len(sentences) > config.samples_per_domain:
            np.random.seed(config.seed + hash(domain))
            idx = np.random.choice(
                len(sentences), config.samples_per_domain, replace=False
            )
            domain_samples = [sentences[i] for i in idx]
        else:
            domain_samples = sentences

        all_sentences.extend(domain_samples)

    logger.info(
        f"Combined {len(all_sentences)} sentences from {len(domain_data)} domains"
    )

    # Build positive pairs (same augmentation as specialist training)
    augment_cfg = AugmentConfig(seed=config.seed)
    pairs = build_positive_pairs(all_sentences, augment_cfg, k=config.augmentation_k)

    # Convert to InputExample format
    examples = [InputExample(texts=[a, b]) for (a, b) in pairs]
    return examples


def train_monolith(
    config: TrainingConfig,
    domain_data: dict[DomainEnum, List[str]],
    output_dir: Optional[Path] = None,
) -> MonolithTrainingResult:
    """
    Train a monolithic model on combined domain data.

    Parameters
    ----------
    config : TrainingConfig
        Experiment configuration.
    domain_data : dict[DomainEnum, List[str]]
        Mapping from domain to list of training sentences.
    output_dir : Optional[Path]
        Output directory (defaults to config.output_dir / "monolith").

    Returns
    -------
    MonolithTrainingResult
        Training result with model path and compute metrics.
    """
    if output_dir is None:
        output_dir = config.output_dir / "monolith"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output_path = output_dir / f"{config.base_model.replace('/', '-')}"
    metrics_output_path = output_dir / "compute_metrics.json"

    logger.info(f"Training monolithic model on {len(domain_data)} domains")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Output path: {model_output_path}")

    # Prepare training data
    train_examples = prepare_monolith_data(domain_data, config)
    logger.info(f"Created {len(train_examples)} training pairs")

    # Initialize model
    model = SentenceTransformer(config.base_model)

    # Create DataLoader
    from experiments.train_specialists_st import InputExampleDataset

    train_dataset = InputExampleDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.batch_size,
    )

    # Setup loss
    loss = losses.MultipleNegativesRankingLoss(model)

    # Calculate warmup steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs_monolith
    warmup_steps = max(10, int(0.1 * total_steps))

    logger.info(f"Training for {config.epochs_monolith} epochs")
    logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    # Train with compute tracking
    with track_compute(  # type: ignore
        model_name=config.base_model,
        output_path=metrics_output_path,
        track_gpu=config.track_energy,
    ) as tracker:
        # Record tokens processed (approximate)
        # Each example is a pair of sentences, each ~length tokens
        # We'll track per batch
        avg_sequence_length = 64  # Conservative estimate for sentence transformers

        def track_batch_callback(batch_size: int) -> None:
            """Callback to record tokens processed in a batch."""
            # Forward + backward = 2 passes through tokens
            tokens = batch_size * avg_sequence_length * 2
            tracker.add_tokens(tokens)

        # Wrap training to track tokens
        # We'll monkey-patch the model's fit method or use a custom loop
        # For simplicity, we'll estimate total tokens at the end
        # But we need to track during training. Let's use a custom training loop.
        # However, SentenceTransformer's fit doesn't expose per-batch callbacks.
        # We'll approximate by tracking total tokens after training.

        # Train using model.fit (standard)
        model.fit(
            train_objectives=[(train_loader, loss)],
            epochs=config.epochs_monolith,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": config.learning_rate},
            show_progress_bar=True,
        )

        # Estimate tokens processed
        # Each training example seen epochs_monolith times
        # Each example: 2 sentences * avg_sequence_length tokens * 2 (forward+backward)
        total_examples = len(train_examples) * config.epochs_monolith
        estimated_tokens = total_examples * avg_sequence_length * 2 * 2
        tracker.add_tokens(estimated_tokens)

    # Save model
    model.save(str(model_output_path))
    logger.info(f"Model saved to {model_output_path}")

    return MonolithTrainingResult(
        model_path=model_output_path,
        compute_metrics_path=metrics_output_path,
        config=config,
    )


def main() -> None:
    """Command-line entry point."""
    import argparse
    from experiments.config import load_config

    parser = argparse.ArgumentParser(
        description="Train monolithic model for compute equivalence experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (default: use default config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["medical", "legal", "tech", "cook", "general"],
        default=["legal", "medical"],
        help="Domains to include",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Samples per domain",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override domains if specified
    if args.domains:
        config = replace(config, domains=[DomainEnum(d) for d in args.domains])

    # Override samples if specified
    if args.samples != 50_000:
        config = replace(config, samples_per_domain=args.samples)

    # Override output directory if specified
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))

    # Load domain data
    # For now, use synthetic data. In full experiment, use datasets.py
    from experiments.train_specialists_st import TECH_SENTENCES, COOK_SENTENCES
    from kalmanorix.toy_corpus import generate_anchor_sentences

    domain_data = {}
    domain_sentences = {
        DomainEnum.TECH: TECH_SENTENCES * 10,  # Expand for testing
        DomainEnum.COOK: COOK_SENTENCES * 10,
        DomainEnum.MEDICAL: generate_anchor_sentences(
            n=1000, domains=("medical",), seed=config.seed
        ),
        DomainEnum.LEGAL: generate_anchor_sentences(
            n=1000, domains=("general",), seed=config.seed + 1
        ),
        DomainEnum.GENERAL: generate_anchor_sentences(
            n=1000, domains=("general",), seed=config.seed + 2
        ),
    }

    for domain in config.domains:
        if domain in domain_sentences:
            domain_data[domain] = domain_sentences[domain]
        else:
            logger.warning(f"No data for domain {domain}, using synthetic")
            domain_data[domain] = generate_anchor_sentences(
                n=1000, domains=(domain.value,), seed=config.seed + hash(domain)
            )

    # Train
    result = train_monolith(config, domain_data)
    print("Monolithic training complete:")
    print(f"  Model: {result.model_path}")
    print(f"  Metrics: {result.compute_metrics_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
