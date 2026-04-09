from __future__ import annotations

from kalmanorix.benchmarks.routing_evaluation import (
    RoutingRunConfig,
    RoutingSample,
    ThresholdSweepConfig,
    evaluate_routing,
    evaluate_threshold_robustness,
)


def _samples() -> list[RoutingSample]:
    return [
        RoutingSample(
            query_id="q1",
            relevant_domains=("tech",),
            semantic_scores={"tech": 0.92, "cook": 0.31, "charge": 0.4},
            confidence_scores={"tech": 0.91, "cook": 0.1, "charge": 0.15},
            routing_overhead_ms=0.2,
            domain_flops={"tech": 6.0, "cook": 6.0, "charge": 6.0},
            domain_latency_ms={"tech": 3.0, "cook": 3.0, "charge": 3.0},
            quality_delta=0.01,
        ),
        RoutingSample(
            query_id="q2",
            relevant_domains=("cook", "tech"),
            semantic_scores={"tech": 0.81, "cook": 0.78, "charge": 0.62},
            confidence_scores={"tech": 0.66, "cook": 0.64, "charge": 0.2},
            routing_overhead_ms=0.2,
            domain_flops={"tech": 6.0, "cook": 6.0, "charge": 6.0},
            domain_latency_ms={"tech": 3.0, "cook": 3.0, "charge": 3.0},
            quality_delta=0.0,
        ),
        RoutingSample(
            query_id="q3",
            relevant_domains=("charge",),
            semantic_scores={"tech": 0.89, "cook": 0.5, "charge": 0.49},
            confidence_scores={"tech": 0.93, "cook": 0.2, "charge": 0.25},
            routing_overhead_ms=0.2,
            domain_flops={"tech": 6.0, "cook": 6.0, "charge": 6.0},
            domain_latency_ms={"tech": 3.0, "cook": 3.0, "charge": 3.0},
            quality_delta=-0.04,
        ),
    ]


def test_semantic_routing_report_categories() -> None:
    report = evaluate_routing(
        _samples(),
        RoutingRunConfig(
            mode="semantic", semantic_threshold=0.75, quality_tolerance=0.01
        ),
    )

    assert report["summary"]["routing_precision"] > 0
    assert report["summary"]["routing_recall"] > 0
    assert report["summary"]["avg_flops_savings_fraction"] > 0

    split = report["report"]
    assert split["quality_preserving_routing_wins"]["count"] == 2
    assert split["compute_only_wins"]["count"] == 1
    assert split["failure_modes"]["count"] == 0


def test_confidence_routing_can_create_failure_modes() -> None:
    report = evaluate_routing(
        _samples(),
        RoutingRunConfig(
            mode="confidence",
            semantic_threshold=0.75,
            confidence_threshold=0.8,
            quality_tolerance=0.01,
        ),
    )

    assert report["report"]["compute_only_wins"]["count"] >= 1
    assert report["report"]["failure_modes"]["breakdown"]["quality_loss"] >= 0


def test_threshold_robustness_contains_ranges() -> None:
    robustness = evaluate_threshold_robustness(
        _samples(),
        ThresholdSweepConfig(
            mode="semantic",
            semantic_thresholds=(0.6, 0.75, 0.85),
            quality_tolerance=0.01,
        ),
    )

    assert len(robustness["threshold_runs"]) == 3
    assert robustness["robustness"]["best_semantic_threshold_by_f1"] in {
        0.6,
        0.75,
        0.85,
    }
    assert robustness["robustness"]["f1_range"] >= 0
    assert robustness["robustness"]["flops_savings_range"] >= 0
