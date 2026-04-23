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
                "router_only_topk_mean": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
                "uniform_mean_fusion": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
                "fixed_weighted_mean_fusion": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
                "learned_linear_combiner": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
                "single_generalist_model": {
                    "q1": ["d1", "d4"],
                    "q2": ["d2", "d3"],
                    "q3": ["d3", "d1"],
                },
            },
            "latency_ms": {
                "mean": {"q1": 1.0, "q2": 1.2, "q3": 1.1},
                "kalman": {"q1": 1.1, "q2": 1.3, "q3": 1.2},
                "router_only_top1": {"q1": 0.8, "q2": 0.9, "q3": 0.85},
                "router_only_topk_mean": {"q1": 0.9, "q2": 1.0, "q3": 0.95},
                "uniform_mean_fusion": {"q1": 1.0, "q2": 1.1, "q3": 1.0},
                "fixed_weighted_mean_fusion": {"q1": 1.0, "q2": 1.1, "q3": 1.0},
                "learned_linear_combiner": {"q1": 1.0, "q2": 1.1, "q3": 1.0},
                "single_generalist_model": {"q1": 1.0, "q2": 1.1, "q3": 1.0},
            },
            "confidence_proxy": {
                "router_only_top1": {"q1": 0.65, "q2": 0.55, "q3": 0.75}
            },
            "specialist_count_selected": {
                "mean": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "kalman": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "router_only_top1": {"q1": 1.0, "q2": 1.0, "q3": 1.0},
                "router_only_topk_mean": {"q1": 2.0, "q2": 2.0, "q3": 2.0},
                "uniform_mean_fusion": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "fixed_weighted_mean_fusion": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "learned_linear_combiner": {"q1": 3.0, "q2": 3.0, "q3": 3.0},
                "single_generalist_model": {"q1": 1.0, "q2": 1.0, "q3": 1.0},
            },
            "query_metadata": {
                "q1": {
                    "is_multi_domain": False,
                    "specialist_disagreement": 0.2,
                    "uncertainty_spread": 0.1,
                    "router_confidence": 0.8,
                },
                "q2": {
                    "is_multi_domain": True,
                    "specialist_disagreement": 0.8,
                    "uncertainty_spread": 0.3,
                    "router_confidence": 0.4,
                },
                "q3": {
                    "is_multi_domain": False,
                    "specialist_disagreement": 0.4,
                    "uncertainty_spread": 0.2,
                    "router_confidence": 0.6,
                },
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
        "router_only_topk_mean",
        "uniform_mean_fusion",
        "fixed_weighted_mean_fusion",
        "learned_linear_combiner",
        "single_generalist_model",
    }
    assert "ndcg@10" in on_disk["paired_statistics"]["kalman_vs_mean"]["overall"]
    assert (
        "ndcg@10"
        in on_disk["paired_statistics"]["kalman_vs_fixed_weighted_mean_fusion"][
            "overall"
        ]
    )
    assert (
        "ndcg@10"
        in on_disk["paired_statistics"]["kalman_vs_learned_linear_combiner"]["overall"]
    )
    assert "bucket_analysis" in on_disk
    assert on_disk["decision"]["kalman_vs_mean"]["verdict"] in {
        "supported",
        "unsupported",
        "inconclusive_underpowered",
        "inconclusive_sufficiently_powered",
    }
    assert on_disk["decision"]["kalman_vs_weighted_mean"]["verdict"] in {
        "supported",
        "unsupported",
        "inconclusive_underpowered",
        "inconclusive_sufficiently_powered",
    }
    assert on_disk["decision"]["kalman_vs_router_only_top1"]["verdict"] in {
        "supported",
        "unsupported",
        "inconclusive_underpowered",
        "inconclusive_sufficiently_powered",
    }
    assert on_disk["decision"]["kalman_vs_learned_linear_combiner"]["verdict"] in {
        "supported",
        "unsupported",
        "inconclusive_underpowered",
        "inconclusive_sufficiently_powered",
    }
    assert "power_diagnostics" in on_disk
    assert "sample_size_adequacy" in on_disk
    assert set(on_disk["sample_size_adequacy"]) == {
        "uncertainty_calibration",
        "paired_significance_testing",
        "per_domain_analysis",
    }
    assert "kalman_vs_mean" in on_disk["decision"]
    assert on_disk["benchmark_status"]["status"] == "toy"
    assert on_disk["claim_success_decision"]["status"] == "blocked"
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


def test_canonical_benchmark_requires_claim_ready_weighting_baselines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _fake_details()
    del payload["query_level"]["rankings"]["learned_linear_combiner"]  # type: ignore[index]
    del payload["query_level"]["latency_ms"]["learned_linear_combiner"]  # type: ignore[index]
    del payload["query_level"]["specialist_count_selected"]["learned_linear_combiner"]  # type: ignore[index]

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
        ValueError, match=r"Missing strategies: \['learned_linear_combiner'\]"
    ):
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

    with pytest.raises(ValueError, match=r"Missing strategies: \['mean'\]") as exc_info:
        canonical.run_canonical_benchmark(
            benchmark_path=tmp_path / "dummy.parquet",
            output_dir=tmp_path / "results",
            split="test",
            max_queries=3,
            device="cpu",
            seed=7,
            num_resamples=100,
        )
    assert "required deployment baselines" in str(exc_info.value)


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
    assert "## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)" in report_text
    assert "## Sample Size Adequacy Checks" in report_text
    assert (
        "| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |"
        in report_text
    )
    assert "## Kalman vs simple and learned weighting baselines" in report_text
    assert "## Did Kalman beat the required deployment baselines?" in report_text
    assert "top1_success" in report_text
    assert "## Method Ranking Snapshot" in report_text
    assert "## Verdict" in report_text
    assert "**Benchmark status:** `toy`" in report_text
    assert "**benchmark_status:** `toy`" in report_text
    assert "## Demonstrated findings" in report_text
    assert "## Bucketed Analysis" in report_text


def test_validate_summary_contract_requires_claim_ready_blocks() -> None:
    summary = {
        "methods": {
            "mean": {},
            "fixed_weighted_mean_fusion": {},
            "router_only_top1": {},
            "router_only_topk_mean": {},
            "learned_linear_combiner": {},
        },
        "decision": {
            "kalman_vs_mean": {},
            "kalman_vs_weighted_mean": {},
            "kalman_vs_router_only_top1": {},
            "kalman_vs_learned_linear_combiner": {},
        },
        "sample_size_adequacy": {
            "uncertainty_calibration": {},
            "paired_significance_testing": {},
            "per_domain_analysis": {},
        },
        "benchmark_status": {"status": "toy"},
    }
    canonical._validate_summary_contract(summary)

    missing_status = dict(summary)
    del missing_status["benchmark_status"]
    with pytest.raises(ValueError, match="benchmark_status"):
        canonical._validate_summary_contract(missing_status)

    missing_decision = dict(summary)
    missing_decision["decision"] = {"kalman_vs_weighted_mean": {}}
    with pytest.raises(ValueError, match="decision blocks"):
        canonical._validate_summary_contract(missing_decision)

    missing_baseline = dict(summary)
    missing_baseline["methods"] = {"mean": {}}
    missing_baseline["benchmark_status"] = {"status": "claim_ready"}
    with pytest.raises(ValueError, match="required deployment baselines"):
        canonical._validate_summary_contract(missing_baseline)

    missing_adequacy = dict(summary)
    missing_adequacy["sample_size_adequacy"] = {"uncertainty_calibration": {}}
    with pytest.raises(ValueError, match="sample size adequacy"):
        canonical._validate_summary_contract(missing_adequacy)


def test_validate_report_contract_requires_required_sections() -> None:
    valid_report = "\n".join(
        [
            "## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)",
            "## Sample Size Adequacy Checks",
            "## Kalman vs simple and learned weighting baselines",
            "## Did Kalman beat the required deployment baselines?",
            "## Verdict",
        ]
    )
    canonical._validate_report_contract(valid_report)

    with pytest.raises(ValueError, match="missing required sections"):
        canonical._validate_report_contract("## Verdict\n")


@pytest.mark.parametrize(
    ("sample_size_adequacy", "power_diag", "expected_status"),
    [
        (
            {
                "uncertainty_calibration": {"available_queries": 40},
                "per_domain_analysis": {"minimum_domain_count": 9},
            },
            {
                "num_test_queries": 24,
                "detectable_effect_threshold_estimate": 0.03,
                "target_effect_size": 0.02,
            },
            "toy",
        ),
        (
            {
                "uncertainty_calibration": {"available_queries": 120},
                "per_domain_analysis": {"minimum_domain_count": 20},
            },
            {
                "num_test_queries": 60,
                "detectable_effect_threshold_estimate": 0.03,
                "target_effect_size": 0.02,
            },
            "underpowered",
        ),
        (
            {
                "uncertainty_calibration": {"available_queries": 140},
                "per_domain_analysis": {"minimum_domain_count": 22},
            },
            {
                "num_test_queries": 70,
                "detectable_effect_threshold_estimate": 0.018,
                "target_effect_size": 0.02,
            },
            "minimally_powered",
        ),
        (
            {
                "uncertainty_calibration": {"available_queries": 240},
                "per_domain_analysis": {"minimum_domain_count": 45},
            },
            {
                "num_test_queries": 140,
                "detectable_effect_threshold_estimate": 0.012,
                "target_effect_size": 0.02,
            },
            "claim_ready",
        ),
    ],
)
def test_classify_benchmark_status_covers_all_statuses(
    sample_size_adequacy: dict[str, dict[str, int]],
    power_diag: dict[str, float],
    expected_status: str,
) -> None:
    summary = {
        "sample_size_adequacy": sample_size_adequacy,
        "power_diagnostics": {"kalman_vs_mean": power_diag},
    }

    status_payload = canonical._classify_benchmark_status(summary)
    assert status_payload["status"] == expected_status


def test_confirmatory_slice_selection_is_locked_to_predeclared_definition() -> None:
    details = _fake_details()
    query_level = details["query_level"]  # type: ignore[index]
    rankings = query_level["rankings"]  # type: ignore[index]
    query_ids = sorted(rankings["kalman"])  # type: ignore[index]
    bucket_to_qids, thresholds = canonical._bucket_query_ids(
        query_ids=query_ids,
        query_metadata=query_level["query_metadata"],  # type: ignore[index]
        specialist_counts=query_level["specialist_count_selected"]["router_only_top1"],  # type: ignore[index]
        confidence_proxy=query_level["confidence_proxy"]["router_only_top1"],  # type: ignore[index]
    )

    selected = canonical._resolve_confirmatory_slice_ids(
        slice_name=canonical.CONFIRMATORY_SLICE_NAME,
        query_ids=query_ids,
        query_metadata=query_level["query_metadata"],  # type: ignore[index]
        specialist_counts=query_level["specialist_count_selected"]["router_only_top1"],  # type: ignore[index]
        confidence_proxy=query_level["confidence_proxy"]["router_only_top1"],  # type: ignore[index]
        bucket_to_qids=bucket_to_qids,
        bucket_thresholds=thresholds,
    )

    assert selected == ["q2"]


def test_confirmatory_slice_empty_and_underpowered_emit_warnings() -> None:
    details = _fake_details()
    query_level = details["query_level"]  # type: ignore[index]
    methods = {}
    for method in query_level["rankings"]:  # type: ignore[index]
        methods[method] = canonical.aggregate_strategy_metrics(
            rankings=query_level["rankings"][method],  # type: ignore[index]
            ground_truth={  # type: ignore[index]
                qid: set(doc_ids)
                for qid, doc_ids in query_level["ground_truth"].items()  # type: ignore[index]
            },
            latency_ms=query_level["latency_ms"][method],  # type: ignore[index]
            flops_proxy=query_level["specialist_count_selected"][method],  # type: ignore[index]
            seed=5,
            num_resamples=10,
        )

    query_ids = sorted(query_level["rankings"]["kalman"])  # type: ignore[index]
    empty = canonical._build_confirmatory_slice_results(
        slice_name=canonical.CONFIRMATORY_SLICE_NAME,
        methods=methods,
        query_ids=query_ids,
        selected_qids=[],
        seed=11,
        num_resamples=50,
    )
    underpowered = canonical._build_confirmatory_slice_results(
        slice_name=canonical.CONFIRMATORY_SLICE_NAME,
        methods=methods,
        query_ids=query_ids,
        selected_qids=["q2"],
        seed=11,
        num_resamples=50,
    )

    assert empty["warning_count"] == 1
    assert "zero paired queries" in empty["warnings"][0]
    assert empty["paired_statistics"] is None
    assert empty["verdicts"]["kalman_vs_mean"]["status"] == "underpowered"
    assert underpowered["warning_count"] == 1
    assert "underpowered" in underpowered["warnings"][0]
    assert underpowered["paired_statistics"] is None
    assert (
        underpowered["verdicts"]["kalman_vs_router_only_top1"]["status"]
        == "underpowered"
    )
    assert empty["decision"]["verdict"] == "inconclusive_underpowered"
    assert underpowered["decision"]["verdict"] == "inconclusive_underpowered"


def test_confirmatory_slice_decision_supported_when_all_baselines_pass() -> None:
    verdicts = {
        "kalman_vs_mean": {"status": "supported"},
        "kalman_vs_weighted_mean": {"status": "supported"},
        "kalman_vs_router_only_top1": {"status": "supported"},
    }
    decision = canonical._classify_confirmatory_slice_decision(
        n_pairs=60,
        minimum_pairs=canonical.CONFIRMATORY_SLICE_MIN_PAIRS,
        verdicts=verdicts,
        required_comparisons=canonical.CONFIRMATORY_REQUIRED_BASELINE_COMPARISONS,
    )
    assert decision["verdict"] == "supported"


def test_confirmatory_slice_decision_unsupported_when_one_baseline_fails() -> None:
    verdicts = {
        "kalman_vs_mean": {"status": "supported"},
        "kalman_vs_weighted_mean": {"status": "not_supported"},
        "kalman_vs_router_only_top1": {"status": "supported"},
    }
    decision = canonical._classify_confirmatory_slice_decision(
        n_pairs=60,
        minimum_pairs=canonical.CONFIRMATORY_SLICE_MIN_PAIRS,
        verdicts=verdicts,
        required_comparisons=canonical.CONFIRMATORY_REQUIRED_BASELINE_COMPARISONS,
    )
    assert decision["verdict"] == "unsupported"


def test_canonical_benchmark_writes_confirmatory_slice_section(
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
        seed=7,
        num_resamples=100,
        confirmatory_slice=canonical.CONFIRMATORY_SLICE_NAME,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    confirmatory = summary["confirmatory_slice_results"]
    assert confirmatory["slice_name"] == canonical.CONFIRMATORY_SLICE_NAME
    assert "uncertainty_spread" in confirmatory["slice_definition"]["description"]
    assert confirmatory["warning_count"] == 1
    assert confirmatory["paired_statistics"] is None
    assert (
        confirmatory["verdicts"]["kalman_vs_weighted_mean"]["status"] == "underpowered"
    )
    report_text = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "## Confirmatory Slice (Pre-declared)" in report_text
    assert (
        "Confirmatory paired statistical tests (Kalman vs baselines)" not in report_text
    )
    assert "Underpowered: inferential paired testing was not run" in report_text
    assert (
        "## Bucketed Analysis (Exploratory unless significance criteria are met)"
        in report_text
    )
    assert "## Hard claim gate" in report_text
    assert summary["confirmatory_slice_artifacts"]["membership"]
    assert (
        "D50_specialist_disagreement_median"
        in summary["confirmatory_slice_artifacts"]["thresholds"]
    )


def test_build_replication_summary_aggregates_runs() -> None:
    run_summaries = [
        {
            "seed": 11,
            "paired_statistics": {
                "kalman_vs_mean": {
                    "overall": {
                        "ndcg@10": {
                            "mean_difference": 0.03,
                            "adjusted_p_value": 0.02,
                            "n_pairs": 100,
                        }
                    }
                }
            },
            "decision": {
                "kalman_vs_mean": {
                    "verdict": "supported",
                    "observed": {"latency_ratio_vs_mean": 1.2},
                }
            },
        },
        {
            "seed": 12,
            "paired_statistics": {
                "kalman_vs_mean": {
                    "overall": {
                        "ndcg@10": {
                            "mean_difference": -0.01,
                            "adjusted_p_value": 0.40,
                            "n_pairs": 80,
                        }
                    }
                }
            },
            "decision": {
                "kalman_vs_mean": {
                    "verdict": "inconclusive_underpowered",
                    "observed": {"latency_ratio_vs_mean": 1.1},
                }
            },
        },
    ]

    replication = canonical._build_replication_summary(run_summaries)
    assert replication["num_runs"] == 2
    assert replication["fraction_positive_deltas"] == 0.5
    assert replication["fraction_significant_runs"] == 0.5
    assert replication["direction_consistency"] == "mixed"
    assert replication["sign_consistency_delta_kalman_minus_mean"] == "mixed"
    assert replication["latency_ratio_consistency"] == "all_within_threshold"
    assert replication["latency_ratio_threshold_vs_mean"] == pytest.approx(1.5)
    assert replication["fraction_latency_within_threshold"] == pytest.approx(1.0)
    assert replication["median_latency_ratio"] == pytest.approx(1.15)
    assert replication["pooled_effect_summaries"][
        "weighted_mean_delta_ndcg10"
    ] == pytest.approx(0.0122222222)


def test_canonical_report_renders_replication_section() -> None:
    summary = {
        "benchmark": {
            "path": "dummy.parquet",
            "evaluated_split": "test",
            "split_counts": {"train": 8, "validation": 4, "test": 3},
        },
        "specialists": ["tech", "cook"],
        "comparisons": {"LearnedGateFuser": {"included": False, "reason": "omitted"}},
        "methods": {
            "mean": {
                "metrics": {
                    k: {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
                    for k in [
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@5",
                        "mrr@10",
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "top1_success",
                        "latency_ms",
                        "flops_proxy",
                    ]
                }
            },
            "kalman": {
                "metrics": {
                    k: {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
                    for k in [
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@5",
                        "mrr@10",
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "top1_success",
                        "latency_ms",
                        "flops_proxy",
                    ]
                }
            },
            "router_only_top1": {
                "metrics": {
                    k: {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
                    for k in [
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@5",
                        "mrr@10",
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "top1_success",
                        "latency_ms",
                        "flops_proxy",
                    ]
                }
            },
            "uniform_mean_fusion": {
                "metrics": {
                    k: {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
                    for k in [
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@5",
                        "mrr@10",
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "top1_success",
                        "latency_ms",
                        "flops_proxy",
                    ]
                }
            },
            "fixed_weighted_mean_fusion": {
                "metrics": {
                    k: {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
                    for k in [
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@5",
                        "mrr@10",
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "top1_success",
                        "latency_ms",
                        "flops_proxy",
                    ]
                }
            },
            "learned_linear_combiner": {
                "metrics": {
                    k: {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
                    for k in [
                        "ndcg@5",
                        "ndcg@10",
                        "mrr@5",
                        "mrr@10",
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "top1_success",
                        "latency_ms",
                        "flops_proxy",
                    ]
                }
            },
        },
        "paired_statistics": {
            "kalman_vs_mean": {
                "overall": {
                    k: {
                        "mean_difference": 0.0,
                        "ci95_low": 0.0,
                        "ci95_high": 0.0,
                        "p_value": 1.0,
                        "adjusted_p_value": 1.0,
                    }
                    for k in canonical.REPORT_METRICS
                }
            },
            "kalman_vs_fixed_weighted_mean_fusion": {
                "overall": {
                    "ndcg@10": {
                        "mean_difference": 0.0,
                        "ci95_low": 0.0,
                        "ci95_high": 0.0,
                        "adjusted_p_value": 1.0,
                    }
                }
            },
            "kalman_vs_router_only_top1": {
                "overall": {
                    "ndcg@10": {
                        "mean_difference": 0.0,
                        "ci95_low": 0.0,
                        "ci95_high": 0.0,
                        "adjusted_p_value": 1.0,
                    }
                }
            },
            "kalman_vs_learned_linear_combiner": {
                "overall": {
                    "ndcg@10": {
                        "mean_difference": 0.0,
                        "ci95_low": 0.0,
                        "ci95_high": 0.0,
                        "adjusted_p_value": 1.0,
                    }
                }
            },
        },
        "decision": {
            "kalman_vs_mean": {
                "verdict": "inconclusive_underpowered",
                "rules": canonical.CANONICAL_DECISION_RULES,
                "observed": {
                    "primary_metric_delta": 0.0,
                    "primary_metric_adjusted_p_value": 1.0,
                    "latency_ratio_vs_mean": 1.0,
                    "flops_ratio_vs_mean": 1.0,
                },
                "checks": {
                    "effect_size_ok": False,
                    "adjusted_p_value_ok": False,
                    "latency_ratio_ok": True,
                    "flops_ratio_ok": True,
                },
            },
            "kalman_vs_weighted_mean": {"verdict": "inconclusive_underpowered"},
            "kalman_vs_router_only_top1": {"verdict": "inconclusive_underpowered"},
            "kalman_vs_learned_linear_combiner": {
                "verdict": "inconclusive_underpowered"
            },
        },
        "benchmark_status": {"status": "toy", "status_note": "toy setup"},
        "power_diagnostics": {
            "kalman_vs_mean": {
                "num_test_queries": 3,
                "per_domain_test_counts": {"finance": 1},
                "observed_effect_size": 0.0,
                "detectable_effect_threshold_estimate": 1.0,
                "target_effect_size": 0.02,
                "is_sufficiently_powered_for_target_effect": False,
            }
        },
        "sample_size_adequacy": {
            "uncertainty_calibration": {
                "available_queries": 4,
                "minimum_required": 100,
                "adequate": False,
                "note": "n/a",
            },
            "paired_significance_testing": {
                "available_queries": 3,
                "minimum_required": 50,
                "adequate": False,
                "note": "n/a",
            },
            "per_domain_analysis": {
                "minimum_domain_count": 1,
                "minimum_required_per_domain": 20,
                "adequate": False,
                "note": "n/a",
            },
        },
        "bucket_analysis": {"buckets": {}, "consistent_kalman_gain_buckets": []},
        "replication": {
            "num_runs": 2,
            "fraction_positive_deltas": 0.5,
            "fraction_significant_runs": 0.5,
            "median_latency_ratio": 1.15,
            "sign_consistency_delta_kalman_minus_mean": "mixed",
            "direction_consistency": "mixed",
            "latency_ratio_consistency": "all_within_threshold",
            "fraction_latency_within_threshold": 1.0,
            "latency_ratio_threshold_vs_mean": 1.5,
            "per_run_verdicts": [
                {
                    "run_id": "run_001",
                    "seed": 11,
                    "verdict": "supported",
                    "primary_delta_ndcg10": 0.03,
                    "primary_adjusted_p_value_ndcg10": 0.02,
                    "latency_ratio_vs_mean": 1.2,
                },
                {
                    "run_id": "run_002",
                    "seed": 12,
                    "verdict": "inconclusive_underpowered",
                    "primary_delta_ndcg10": -0.01,
                    "primary_adjusted_p_value_ndcg10": 0.4,
                    "latency_ratio_vs_mean": 1.1,
                },
            ],
            "pooled_effect_summaries": {
                "weighted_mean_delta_ndcg10": 0.012,
                "median_delta_ndcg10": 0.01,
                "median_adjusted_p_value_ndcg10": 0.21,
            },
        },
    }

    report = canonical._render_report(summary)
    assert "## Replication status of Kalman-vs-mean conclusion" in report
    assert "Replication runs: `2`" in report


def test_build_replication_summary_flags_latency_inconsistency() -> None:
    run_summaries = [
        {
            "seed": 1,
            "paired_statistics": {
                "kalman_vs_mean": {
                    "overall": {
                        "ndcg@10": {
                            "mean_difference": 0.02,
                            "adjusted_p_value": 0.01,
                            "n_pairs": 40,
                        }
                    }
                }
            },
            "decision": {
                "kalman_vs_mean": {
                    "verdict": "supported",
                    "observed": {"latency_ratio_vs_mean": 1.2},
                }
            },
        },
        {
            "seed": 2,
            "paired_statistics": {
                "kalman_vs_mean": {
                    "overall": {
                        "ndcg@10": {
                            "mean_difference": 0.03,
                            "adjusted_p_value": 0.03,
                            "n_pairs": 35,
                        }
                    }
                }
            },
            "decision": {
                "kalman_vs_mean": {
                    "verdict": "supported",
                    "observed": {"latency_ratio_vs_mean": 1.8},
                }
            },
        },
    ]

    replication = canonical._build_replication_summary(run_summaries)
    assert replication["sign_consistency_delta_kalman_minus_mean"] == "all_positive"
    assert replication["latency_ratio_consistency"] == "mixed"
    assert replication["fraction_latency_within_threshold"] == pytest.approx(0.5)


def test_resolve_replication_build_specs() -> None:
    specs = canonical._resolve_replication_build_specs(
        replication_builds=(
            "benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet@7,"
            "benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet@8"
        )
    )
    assert specs == [
        (Path("benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet"), 7),
        (Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet"), 8),
    ]
