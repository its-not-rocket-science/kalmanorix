from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import run_canonical_benchmark as canonical


def _fake_details() -> dict[str, object]:
    return {
        "query_level": {
            "domains": {
                "q1": "finance",
                "q2": "biomedical",
                "q3": "general_qa",
            },
            "ground_truth": {
                "q1": ["d1"],
                "q2": ["d2"],
                "q3": ["d3"],
            },
            "rankings": {
                "mean": {"q1": ["d1", "d4"], "q2": ["d2", "d3"], "q3": ["d3", "d1"]},
                "kalman": {"q1": ["d1", "d4"], "q2": ["d2", "d3"], "q3": ["d3", "d1"]},
                "router_only_top1": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
                "uniform_mean_fusion": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
            },
            "latency_ms": {
                "mean": {"q1": 1.0, "q2": 1.2, "q3": 1.1},
                "kalman": {"q1": 1.1, "q2": 1.3, "q3": 1.2},
                "router_only_top1": {"q1": 0.8, "q2": 0.9, "q3": 0.85},
                "uniform_mean_fusion": {"q1": 1.0, "q2": 1.1, "q3": 1.0},
            },
            "specialist_count_selected": {
                "mean": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "kalman": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "router_only_top1": {"q1": 1.0, "q2": 1.0, "q3": 1.0},
                "uniform_mean_fusion": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
            },
        }
    }


def test_canonical_benchmark_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        canonical,
        "_load_split_counts",
        lambda _: {"train": 8, "validation": 4, "test": 3},
    )
    monkeypatch.setattr(
        canonical,
        "load_experiment_config",
        lambda cfg_path: {"cfg_path": str(cfg_path)},
    )
    monkeypatch.setattr(canonical, "run_experiment", lambda cfg: _fake_details())

    output_dir = tmp_path / "results"
    summary = canonical.run_canonical_benchmark(
        benchmark_path=tmp_path / "dummy.parquet",
        output_dir=output_dir,
        split="test",
        max_queries=3,
        device="cpu",
        seed=7,
        num_resamples=300,
    )

    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    assert summary_path.exists()
    assert report_path.exists()

    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk["benchmark"]["split_counts"] == {
        "train": 8,
        "validation": 4,
        "test": 3,
    }
    assert set(on_disk["methods"]) >= {
        "mean",
        "kalman",
        "router_only_top1",
        "uniform_mean_fusion",
    }
    assert "ndcg@10" in on_disk["paired_statistics"]["kalman_vs_mean"]["overall"]
    assert on_disk["decision"]["kalman_vs_mean"]["verdict"] in {
        "supported",
        "unsupported",
        "inconclusive",
    }
    assert summary["comparisons"]["LearnedGateFuser"]["included"] is False
    assert "two-specialist" in summary["comparisons"]["LearnedGateFuser"]["reason"]


def test_canonical_benchmark_requires_core_baselines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _fake_details()
    del payload["query_level"]["rankings"]["router_only_top1"]  # type: ignore[index]

    monkeypatch.setattr(
        canonical,
        "_load_split_counts",
        lambda _: {"train": 8, "validation": 4, "test": 3},
    )
    monkeypatch.setattr(
        canonical,
        "load_experiment_config",
        lambda cfg_path: {"cfg_path": str(cfg_path)},
    )
    monkeypatch.setattr(canonical, "run_experiment", lambda cfg: payload)

    with pytest.raises(ValueError, match="Missing strategies"):
        canonical.run_canonical_benchmark(
            benchmark_path=tmp_path / "dummy.parquet",
            output_dir=tmp_path / "results",
            split="test",
            max_queries=3,
            device="cpu",
            seed=7,
            num_resamples=100,
        )


def test_canonical_benchmark_fails_loudly_when_mean_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _fake_details()
    del payload["query_level"]["rankings"]["mean"]  # type: ignore[index]
    del payload["query_level"]["latency_ms"]["mean"]  # type: ignore[index]
    del payload["query_level"]["specialist_count_selected"]["mean"]  # type: ignore[index]

    monkeypatch.setattr(
        canonical,
        "_load_split_counts",
        lambda _: {"train": 8, "validation": 4, "test": 3},
    )
    monkeypatch.setattr(
        canonical,
        "load_experiment_config",
        lambda cfg_path: {"cfg_path": str(cfg_path)},
    )
    monkeypatch.setattr(canonical, "run_experiment", lambda cfg: payload)

    with pytest.raises(
        ValueError, match=r"Missing strategies: \['mean'\]"
    ) as exc_info:
        canonical.run_canonical_benchmark(
            benchmark_path=tmp_path / "dummy.parquet",
            output_dir=tmp_path / "results",
            split="test",
            max_queries=3,
            device="cpu",
            seed=7,
            num_resamples=100,
        )
    assert "Canonical benchmark requires MeanFuser" in str(exc_info.value)


def test_canonical_report_includes_paired_statistics_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        canonical,
        "_load_split_counts",
        lambda _: {"train": 8, "validation": 4, "test": 3},
    )
    monkeypatch.setattr(
        canonical,
        "load_experiment_config",
        lambda cfg_path: {"cfg_path": str(cfg_path)},
    )
    monkeypatch.setattr(canonical, "run_experiment", lambda cfg: _fake_details())

    output_dir = tmp_path / "results"
    canonical.run_canonical_benchmark(
        benchmark_path=tmp_path / "dummy.parquet",
        output_dir=output_dir,
        split="test",
        max_queries=3,
        device="cpu",
        seed=9,
        num_resamples=200,
    )

    report_text = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Decision Framework: KalmanorixFuser vs MeanFuser" in report_text
    assert "## Paired Statistical Test: KalmanorixFuser vs MeanFuser" in report_text
    assert "| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |" in report_text
    assert "top1_success" in report_text
    assert "## Method Ranking Snapshot" in report_text
    assert "## Verdict" in report_text
    assert "## Demonstrated findings" in report_text
