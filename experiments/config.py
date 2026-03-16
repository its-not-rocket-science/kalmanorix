"""
Configuration system for Kalmanorix training experiments.

Supports YAML-based configuration for reproducible experiments.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class DomainEnum(str, Enum):
    """Supported domains for specialist training."""

    MEDICAL = "medical"
    LEGAL = "legal"
    TECH = "tech"
    COOK = "cook"
    GENERAL = "general"


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class TrainingConfig:
    """
    Configuration for training specialists and monolithic models.

    This configuration ensures compute equivalence between:
    - Specialists: 1 epoch per domain
    - Monolith: 2 epochs on combined data
    """

    # Experiment metadata
    experiment_name: str = "milestone_2_1"
    output_dir: Path = field(default_factory=lambda: Path("experiments") / "outputs")
    seed: int = 42

    # Domains and data
    domains: List[DomainEnum] = field(
        default_factory=lambda: [DomainEnum.LEGAL, DomainEnum.MEDICAL]
    )
    samples_per_domain: int = 50_000
    mixed_test_proportions: Dict[str, float] = field(
        default_factory=lambda: {"legal": 0.25, "medical": 0.25, "mixed": 0.5}
    )

    # Model architecture
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # For MiniLM-L6

    # Training hyperparameters (must match between specialist and monolith)
    batch_size: int = 32
    learning_rate: float = 2e-5
    epochs_per_specialist: int = 1
    epochs_monolith: int = 2  # Total epochs = specialists * 1 = monolith * 2

    # Specialist divergence parameters
    lambda_cls: float = 0.8
    lambda_away: float = 0.8
    div_steps: int = 200
    div_batch_size: int = 16
    augmentation_k: int = 8  # Number of augmentations per anchor

    # Evaluation
    eval_batch_size: int = 64
    recall_k_values: List[int] = field(default_factory=lambda: [1, 5, 10])

    # Compute tracking
    track_compute: bool = True
    track_energy: bool = False  # Requires pynvml and GPU

    # Optional: dataset overrides
    dataset_configs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert output_dir to Path if it's a string
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # Ensure output_dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate mixed_test_proportions sum to 1.0
        total = sum(self.mixed_test_proportions.values())
        if not abs(total - 1.0) < 1e-9:
            raise ValueError(f"mixed_test_proportions must sum to 1.0, got {total}")

        # Validate compute equivalence
        total_specialist_epochs = len(self.domains) * self.epochs_per_specialist
        if total_specialist_epochs != self.epochs_monolith:
            raise ValueError(
                f"Compute equivalence violated: "
                f"specialists total epochs ({total_specialist_epochs}) "
                f"!= monolith epochs ({self.epochs_monolith})"
            )

        # Validate at least 2 domains for mixed proportion
        if "mixed" in self.mixed_test_proportions and len(self.domains) < 2:
            raise ValueError("Need at least 2 domains for mixed test proportion")

    @property
    def total_train_steps(self) -> int:
        """Total training steps for compute equivalence verification."""
        # Each domain sees samples_per_domain examples
        steps_per_epoch = self.samples_per_domain // self.batch_size
        return steps_per_epoch * self.epochs_monolith

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> TrainingConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Convert string domains to DomainEnum
        if "domains" in data:
            data["domains"] = [DomainEnum(d) for d in data["domains"]]

        # Convert output_dir string to Path
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict
        data = dataclasses.asdict(self)

        # Convert Path objects to strings
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, list) and all(
                isinstance(v, DomainEnum) for v in value
            ):
                data[key] = [v.value for v in value]
            elif isinstance(value, DomainEnum):
                data[key] = value.value

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_json(self) -> str:
        """Return JSON representation for logging."""
        data = dataclasses.asdict(self)

        # Convert Path and Enum for JSON serialization
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, DomainEnum):
                return obj.value
            if isinstance(obj, list) and obj and isinstance(obj[0], DomainEnum):
                return [v.value for v in obj]
            return obj

        serialized = {k: _serialize(v) for k, v in data.items()}
        return json.dumps(serialized, indent=2)


# Default configuration for Milestone 2.1
DEFAULT_CONFIG = TrainingConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> TrainingConfig:
    """
    Load configuration from file or use default.

    Parameters
    ----------
    config_path : Optional[Union[str, Path]]
        Path to YAML configuration file. If None, returns default.

    Returns
    -------
    TrainingConfig
        Loaded configuration.
    """
    if config_path is None:
        return DEFAULT_CONFIG

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return TrainingConfig.from_yaml(config_path)


def create_config_dir(config_dir: Path) -> None:
    """Create configuration directory with example configs."""
    config_dir.mkdir(parents=True, exist_ok=True)

    # Milestone 2.1 default
    config_2_1 = TrainingConfig()
    config_2_1.to_yaml(config_dir / "milestone_2_1.yaml")

    # 3-domain example
    config_3dom = TrainingConfig(
        experiment_name="milestone_2_1_3dom",
        domains=[DomainEnum.LEGAL, DomainEnum.MEDICAL, DomainEnum.TECH],
        mixed_test_proportions={
            "legal": 0.2,
            "medical": 0.2,
            "tech": 0.2,
            "mixed": 0.4,
        },
        epochs_monolith=3,
    )
    config_3dom.to_yaml(config_dir / "milestone_2_1_3dom.yaml")

    # 5-domain example
    config_5dom = TrainingConfig(
        experiment_name="milestone_2_1_5dom",
        domains=[
            DomainEnum.LEGAL,
            DomainEnum.MEDICAL,
            DomainEnum.TECH,
            DomainEnum.COOK,
            DomainEnum.GENERAL,
        ],
        mixed_test_proportions={
            "legal": 0.15,
            "medical": 0.15,
            "tech": 0.15,
            "cook": 0.15,
            "mixed": 0.4,
        },
        epochs_monolith=5,
    )
    config_5dom.to_yaml(config_dir / "milestone_2_1_5dom.yaml")
