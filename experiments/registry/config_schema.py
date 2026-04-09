"""Configuration schema for benchmark registry experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeedConfig:
    python: int = 42
    numpy: int = 42
    torch: int = 42


@dataclass(frozen=True)
class ArtifactConfig:
    summary_json: Path
    details_json: Path | None = None


@dataclass(frozen=True)
class DatasetConfig:
    kind: str
    path: Path | None = None
    split: str = "test"
    max_queries: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    device: str = "cpu"
    specialists: list[dict[str, Any]] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionConfig:
    strategies: list[str] = field(default_factory=lambda: ["mean", "kalman"])
    routing_mode: str = "all"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportingConfig:
    print_stdout: bool = True


@dataclass(frozen=True)
class BenchmarkExperimentConfig:
    name: str
    experiment_type: str
    seed: SeedConfig
    artifacts: ArtifactConfig
    dataset: DatasetConfig
    models: ModelConfig
    fusion: FusionConfig
    evaluation: EvaluationConfig
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Use JSON configs or install pyyaml."
        ) from exc
    return yaml.safe_load(text)


def load_experiment_config(path: Path) -> BenchmarkExperimentConfig:
    raw = _load_raw(path)
    seed = SeedConfig(**raw.get("seed", {}))

    artifacts_raw = raw["artifacts"]
    artifacts = ArtifactConfig(
        summary_json=Path(artifacts_raw["summary_json"]),
        details_json=_as_path(artifacts_raw.get("details_json")),
    )
    dataset_raw = raw["dataset"]
    dataset = DatasetConfig(
        kind=dataset_raw["kind"],
        path=_as_path(dataset_raw.get("path")),
        split=dataset_raw.get("split", "test"),
        max_queries=dataset_raw.get("max_queries"),
        options=dataset_raw.get("options", {}),
    )
    models_raw = raw.get("models", {})
    models = ModelConfig(
        kind=models_raw["kind"],
        device=models_raw.get("device", "cpu"),
        specialists=models_raw.get("specialists", []),
        options=models_raw.get("options", {}),
    )
    fusion_raw = raw.get("fusion", {})
    fusion = FusionConfig(
        strategies=fusion_raw.get("strategies", ["mean", "kalman"]),
        routing_mode=fusion_raw.get("routing_mode", "all"),
        options=fusion_raw.get("options", {}),
    )
    evaluation_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        kind=evaluation_raw["kind"],
        options=evaluation_raw.get("options", {}),
    )
    reporting = ReportingConfig(**raw.get("reporting", {}))

    return BenchmarkExperimentConfig(
        name=raw["name"],
        experiment_type=raw["experiment_type"],
        seed=seed,
        artifacts=artifacts,
        dataset=dataset,
        models=models,
        fusion=fusion,
        evaluation=evaluation,
        reporting=reporting,
    )
