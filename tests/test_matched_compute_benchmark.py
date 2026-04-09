from __future__ import annotations

import json

import pytest

from kalmanorix.benchmarks.matched_compute import (
    ComputeBudgetAssumptions,
    _validate_assumptions,
    run_matched_compute_benchmark,
)


def test_validate_assumptions_rejects_inconsistent_top_k() -> None:
    assumptions = ComputeBudgetAssumptions(
        n_specialists=4,
        specialist_params_proxy=10,
        monolith_params_proxy=10,
        specialist_epochs=1,
        monolith_epochs=4,
        avg_tokens_per_sample=12,
        training_flop_multiplier=6,
        specialist_inference_flops_proxy=100,
        monolith_inference_flops_proxy=100,
        routing_overhead_all_proxy=10,
        routing_overhead_semantic_proxy=10,
        kalman_fusion_overhead_proxy=10,
        semantic_top_k=5,
    )

    with pytest.raises(ValueError, match="semantic_top_k"):
        _validate_assumptions(assumptions)


def test_validate_assumptions_rejects_missing_positive_budget() -> None:
    assumptions = ComputeBudgetAssumptions(
        n_specialists=4,
        specialist_params_proxy=10,
        monolith_params_proxy=10,
        specialist_epochs=1,
        monolith_epochs=4,
        avg_tokens_per_sample=12,
        training_flop_multiplier=0,
        specialist_inference_flops_proxy=100,
        monolith_inference_flops_proxy=100,
        routing_overhead_all_proxy=10,
        routing_overhead_semantic_proxy=10,
        kalman_fusion_overhead_proxy=10,
        semantic_top_k=2,
    )

    with pytest.raises(ValueError, match="training_flop_multiplier"):
        _validate_assumptions(assumptions)


def test_run_matched_compute_benchmark_writes_outputs(tmp_path) -> None:
    summary = run_matched_compute_benchmark(
        output_dir=tmp_path,
        seed=9,
        samples_per_domain=80,
        test_size=40,
        semantic_top_k=2,
    )

    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    assert summary_path.exists()
    assert report_path.exists()
    assert summary["fairness_checks"]["training_compute_parity_achieved"] is True

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["experiment"] == "matched_compute_specialists_vs_monolith"
    assert len(loaded["results"]) == 4

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Fairness checks" in report_text
    assert "missing/inconsistent assumptions raise errors" in report_text
