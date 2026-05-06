from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.registry.config_schema import (
    ArtifactConfig,
    BenchmarkExperimentConfig,
    DatasetConfig,
    EvaluationConfig,
    FusionConfig,
    ModelConfig,
    ReportingConfig,
    SeedConfig,
)
import experiments.registry.runner as runner


class _FakeModule:
    def __init__(self, name: str) -> None:
        self.name = name

    def sigma2_for(self, _query: str) -> float:
        return 0.1


class _FakeVillage:
    def __init__(self) -> None:
        self.modules = [_FakeModule("m1"), _FakeModule("m2")]


class _FakeScout:
    def select(self, query: str, village: _FakeVillage):  # noqa: ARG002
        return village.modules[:1]


class _FakeBaseline:
    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, **kwargs):
        return None

    def weights_for_query(self, *, query_text: str, modules: list[_FakeModule]):  # noqa: ARG002
        return [1.0 / len(modules)] * len(modules)


@pytest.fixture
def base_config(tmp_path: Path) -> BenchmarkExperimentConfig:
    return BenchmarkExperimentConfig(
        name="t",
        experiment_type="real_mixed",
        seed=SeedConfig(1, 1, 1),
        artifacts=ArtifactConfig(
            summary_json=tmp_path / "s.json", details_json=tmp_path / "d.json"
        ),
        dataset=DatasetConfig(
            kind="mixed_parquet",
            path=tmp_path / "bench.parquet",
            split="test",
            max_queries=5,
            options={"max_candidates": 5},
        ),
        models=ModelConfig(
            kind="hf_specialists",
            device="cpu",
            specialists=[{"name": "a"}],
            options={"force_hash_embedder": False},
        ),
        fusion=FusionConfig(
            strategies=["mean", "kalman"], routing_mode="all", options={}
        ),
        evaluation=EvaluationConfig(
            kind="locked_protocol",
            options={"checkpoint_every": 1, "checkpoint_dir": str(tmp_path / "cp")},
        ),
        reporting=ReportingConfig(print_stdout=False),
    )


def _patch_core(monkeypatch: pytest.MonkeyPatch, interrupt_after: int | None = None):
    rows = [
        {
            "query_id": f"q{i}",
            "query_text": f"query {i}",
            "candidate_documents": ["a", "b"],
            "ground_truth_relevant_ids": ["a"],
            "domain_label": "d",
        }
        for i in range(5)
    ]
    monkeypatch.setattr(runner, "load_dataset", lambda **kwargs: rows)
    monkeypatch.setattr(runner, "build_village", lambda **kwargs: _FakeVillage())
    monkeypatch.setattr(
        runner, "build_strategy", lambda **kwargs: (_FakeScout(), object())
    )
    monkeypatch.setattr(
        runner, "build_retrieval_baselines", lambda options: [_FakeBaseline("baseline")]
    )
    monkeypatch.setattr(
        runner, "rank_query_with_baseline", lambda **kwargs: (["a", "b"], 1.0)
    )
    monkeypatch.setattr(
        runner,
        "evaluate_locked",
        lambda **kwargs: {
            "mean": {
                "global_primary": {
                    "mrr": {"mean": 0.5},
                    "recall@1": {"mean": 0.5},
                    "recall@5": {"mean": 1.0},
                }
            },
            "kalman": {
                "global_primary": {
                    "mrr": {"mean": 0.6},
                    "recall@1": {"mean": 0.6},
                    "recall@5": {"mean": 1.0},
                }
            },
            "baseline": {
                "global_primary": {
                    "mrr": {"mean": 0.4},
                    "recall@1": {"mean": 0.4},
                    "recall@5": {"mean": 1.0},
                }
            },
        },
    )
    calls = {"count": 0}

    def _rank_query(**kwargs):
        calls["count"] += 1
        if interrupt_after is not None and calls["count"] > interrupt_after:
            raise KeyboardInterrupt
        return ["a", "b"], 1.0

    monkeypatch.setattr(runner, "rank_query", _rank_query)
    return calls


def test_interrupted_run_resumes_and_matches_uninterrupted(
    monkeypatch: pytest.MonkeyPatch,
    base_config: BenchmarkExperimentConfig,
    tmp_path: Path,
) -> None:
    _patch_core(monkeypatch, interrupt_after=2)
    with pytest.raises(RuntimeError, match="Resume with exact command suffix"):
        runner.run_experiment(base_config)

    cp_dir = Path(base_config.evaluation.options["checkpoint_dir"])
    results_lines = (
        (cp_dir / "query_results.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(results_lines) == 1

    resumed = BenchmarkExperimentConfig(
        **{
            **base_config.__dict__,
            "evaluation": EvaluationConfig(
                kind="locked_protocol",
                options={**base_config.evaluation.options, "resume": True},
            ),
        }
    )
    _patch_core(monkeypatch)
    resumed_out = runner.run_experiment(resumed)

    fresh_cfg = BenchmarkExperimentConfig(
        **{
            **base_config.__dict__,
            "evaluation": EvaluationConfig(
                kind="locked_protocol",
                options={
                    "checkpoint_every": 0,
                    "checkpoint_dir": str(tmp_path / "fresh_cp"),
                },
            ),
        }
    )
    _patch_core(monkeypatch)
    fresh_out = runner.run_experiment(fresh_cfg)
    assert resumed_out["results"] == fresh_out["results"]


def test_resume_mismatch_refused(
    monkeypatch: pytest.MonkeyPatch, base_config: BenchmarkExperimentConfig
) -> None:
    _patch_core(monkeypatch)
    runner.run_experiment(base_config)
    bad = BenchmarkExperimentConfig(
        **{
            **base_config.__dict__,
            "dataset": DatasetConfig(
                kind="mixed_parquet",
                path=base_config.dataset.path,
                split="validation",
                max_queries=5,
                options={"max_candidates": 5},
            ),
            "evaluation": EvaluationConfig(
                kind="locked_protocol",
                options={**base_config.evaluation.options, "resume": True},
            ),
        }
    )
    _patch_core(monkeypatch)
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        runner.run_experiment(bad)


def test_checkpoint_files_written_during_run(
    monkeypatch: pytest.MonkeyPatch, base_config: BenchmarkExperimentConfig
) -> None:
    _patch_core(monkeypatch, interrupt_after=2)
    with pytest.raises(RuntimeError):
        runner.run_experiment(base_config)
    cp_dir = Path(base_config.evaluation.options["checkpoint_dir"])
    assert (cp_dir / "checkpoint_metadata.json").exists()
    assert (cp_dir / "query_results.jsonl").exists()
    assert (cp_dir / "failed_queries.jsonl").exists() is False
    meta = json.loads((cp_dir / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted"
