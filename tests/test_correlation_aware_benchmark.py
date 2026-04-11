from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kalmanorix.benchmarks.correlation_aware_fusion import (
    CorrelationAwareFusionConfig,
    run_correlation_aware_fusion_benchmark,
)


def test_correlation_aware_benchmark_emits_expected_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "correlation_aware_fusion"
    summary = run_correlation_aware_fusion_benchmark(
        output_dir=out,
        config=CorrelationAwareFusionConfig(
            random_seed=5,
            n_val=80,
            n_test=120,
            n_docs=100,
            n_specialists=3,
            dimension=16,
        ),
    )

    assert (out / "summary.json").exists()
    assert (out / "report.md").exists()

    on_disk = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(on_disk["test_metrics"]) == {
        "mean_fusion",
        "baseline_kalman",
        "corr_kalman_cov_inflation",
        "corr_kalman_effective_sample_size",
    }
    assert on_disk["validation_fit"]["residual_norm_shape"] == [80, 3]
    assert on_disk["test_split"]["buckets"]["low_correlation"] == 60
    assert on_disk["test_split"]["buckets"]["high_correlation"] == 60
    assert isinstance(summary["answer"], str)


def test_validation_correlation_matrix_is_stable_and_bounded(tmp_path: Path) -> None:
    out = tmp_path / "correlation_aware_fusion"
    summary = run_correlation_aware_fusion_benchmark(
        output_dir=out,
        config=CorrelationAwareFusionConfig(
            random_seed=9,
            n_val=64,
            n_test=64,
            n_docs=80,
            n_specialists=4,
            dimension=12,
        ),
    )

    corr = np.asarray(summary["validation_fit"]["correlation_matrix"], dtype=np.float64)
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(corr <= 1.0 + 1e-12)
    assert np.all(corr >= -1.0 - 1e-12)
