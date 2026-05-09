from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments import run_canonical_benchmark as canonical_cli
from experiments import run_correlation_aware_fusion as corr_cli
from experiments import run_kalman_latency_optimization as latency_cli
from experiments import run_uncertainty_calibration as unc_cli


def _write_tiny_benchmark_parquet(path: Path, *, n_test_queries: int = 16) -> Path:
    rows = []
    split_plan = (["train"] * 2) + (["validation"] * 2) + (["test"] * n_test_queries)
    for idx, split in enumerate(split_plan):
        qid = f"q{idx}"
        pos_id = f"d{idx}_pos"
        rows.append(
            {
                "query_id": qid,
                "query_text": f"query text {idx}",
                "candidate_documents": [
                    {
                        "doc_id": pos_id,
                        "title": "positive",
                        "text": f"document text positive {idx}",
                        "domain": "general_qa",
                        "source_dataset": "tiny_fixture",
                    },
                    {
                        "doc_id": f"d{idx}_neg1",
                        "title": "negative one",
                        "text": f"document text negative one {idx}",
                        "domain": "finance",
                        "source_dataset": "tiny_fixture",
                    },
                    {
                        "doc_id": f"d{idx}_neg2",
                        "title": "negative two",
                        "text": f"document text negative two {idx}",
                        "domain": "biomedical",
                        "source_dataset": "tiny_fixture",
                    },
                ],
                "ground_truth_relevant_ids": [pos_id],
                "domain_label": "general_qa",
                "source_dataset": "tiny_fixture",
                "split": split,
                "contains_cross_domain_hard_negatives": True,
                "dominant_domain": "general_qa",
                "secondary_domain": "finance",
                "query_category": "original",
                "ambiguity_category": "none",
                "ambiguity_score": 0.0,
                "fusion_usefulness_bucket": "low",
                "is_synthetic": False,
                "provenance_note": "tiny e2e fixture",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), path)
        return path
    except ModuleNotFoundError:
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps(rows), encoding="utf-8")
        return json_path


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
    benchmark_path = _write_tiny_benchmark_parquet(tmp_path / "tiny_benchmark.parquet")
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
    benchmark_path = _write_tiny_benchmark_parquet(tmp_path / "tiny_benchmark.parquet")
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
