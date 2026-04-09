from __future__ import annotations

import pytest

from kalmanorix.benchmarks.uncertainty_ablation import MethodMetrics, summarize_scale_sensitivity


def test_summarize_scale_sensitivity_computes_ranges() -> None:
    metrics = {
        "0.5": MethodMetrics(0.25, 0.75, 0.4, 0.2, 0.3, 0.5),
        "1.0": MethodMetrics(0.5, 0.75, 0.5, 0.1, 0.2, 0.4),
        "2.0": MethodMetrics(0.35, 0.75, 0.45, 0.35, 0.25, 0.6),
    }

    sensitivity = summarize_scale_sensitivity(metrics)

    assert sensitivity.recall1_range == 0.25
    assert sensitivity.ece_range == pytest.approx(0.25)
    assert set(sensitivity.scales.keys()) == {"0.5", "1.0", "2.0"}
