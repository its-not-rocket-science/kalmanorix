from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments import run_canonical_benchmark as canonical_cli
from experiments import run_correlation_aware_fusion as corr_cli
from experiments import run_kalman_latency_optimization as latency_cli
from experiments import run_uncertainty_calibration as unc_cli


@pytest.mark.e2e
def test_cli_correlation_aware_fusion_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corr"
    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", str(out), "--seed", "17"])
    corr_cli.main()

    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "test_metrics" in payload


@pytest.mark.e2e
def test_cli_uncertainty_calibration_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "unc"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(out),
            "--objective",
            "rank_error_proxy",
            "--min-validation-total",
            "8",
            "--min-validation-per-domain",
            "2",
            "--min-effective-support-per-specialist",
            "6",
            "--calibrator-min-samples",
            "8",
        ],
    )
    unc_cli.main()

    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["selection_is_validation_only"] is True


@pytest.mark.e2e
def test_cli_canonical_benchmark_v2_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "canonical"
    benchmark_path = Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--benchmark-path",
            str(benchmark_path),
            "--split",
            "test",
            "--max-queries",
            "16",
            "--num-resamples",
            "80",
            "--output-dir",
            str(out),
        ],
    )
    canonical_cli.main()

    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["decision"]["kalman_vs_mean"]["verdict"] in {
        "supported",
        "unsupported",
        "inconclusive_underpowered",
        "inconclusive_sufficiently_powered",
    }


@pytest.mark.e2e
def test_cli_canonical_benchmark_v3_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_canonical_benchmark(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "paired_statistics": {"kalman_vs_mean": {"overall": {}}},
            "decision": {"kalman_vs_mean": {"verdict": "inconclusive_underpowered"}},
        }

    monkeypatch.setattr(
        canonical_cli, "run_canonical_benchmark", _fake_run_canonical_benchmark
    )
    monkeypatch.setattr(
        sys, "argv", ["prog", "--output-dir", str(tmp_path / "override")]
    )
    canonical_cli.main()

    assert captured["benchmark_path"] == Path(
        "benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet"
    )
    assert captured["max_queries"] == 1800


@pytest.mark.e2e
def test_cli_latency_benchmark_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "latency"
    benchmark_path = Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(out),
            "--n-queries",
            "12",
            "--n-specialists",
            "3",
            "--dim",
            "32",
            "--batch-size",
            "4",
            "--canonical-benchmark-path",
            str(benchmark_path),
            "--canonical-max-queries",
            "12",
        ],
    )
    latency_cli.main()

    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (
        payload["numerical_deviation_vs_legacy"]["single_query"]["vector_max_abs"]
        >= 0.0
    )
