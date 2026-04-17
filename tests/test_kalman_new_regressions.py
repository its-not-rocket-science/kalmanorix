from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kalmanorix import KalmanorixFuser, SEF
from kalmanorix.benchmarks.correlation_aware_fusion import (
    CorrelationAwareFusionConfig,
    run_correlation_aware_fusion_benchmark,
)
from kalmanorix.benchmarks.kalman_covariance_ablation import (
    AblationConfig,
    CovarianceFitConfig,
    run_kalman_covariance_ablation,
)
from kalmanorix.benchmarks.uncertainty_calibration import (
    ValidationPowerConfig,
    run_uncertainty_calibration,
    run_uncertainty_calibration_objective_study,
)
from kalmanorix.kalman_engine.correlation import ResidualCorrelationProfile
from kalmanorix.panoramix import CorrelationAwareKalmanFuser
from kalmanorix.uncertainty import (
    EmbeddingNormSigma2,
    StochasticForwardSigma2,
    UncertaintyMethodConfig,
    create_uncertainty_method,
)


def _toy_embed(query: str) -> np.ndarray:
    scale = max(len(query.split()), 1)
    return np.array([1.0, float(scale), 0.5], dtype=np.float64)


def test_correlation_aware_fuser_rejects_unknown_mode() -> None:
    profile = ResidualCorrelationProfile(
        module_names=["a"], correlation_matrix=np.eye(1)
    )
    with pytest.raises(ValueError, match="Unsupported mode"):
        CorrelationAwareKalmanFuser(correlation_profile=profile, mode="invalid")


def test_correlation_aware_fuser_requires_profile_modules_to_exist() -> None:
    profile = ResidualCorrelationProfile(
        module_names=["a"], correlation_matrix=np.eye(1)
    )
    fuser = CorrelationAwareKalmanFuser(
        correlation_profile=profile, mode="effective_sample_size"
    )
    modules = [SEF(name="b", embed=_toy_embed, sigma2=0.2)]
    with pytest.raises(KeyError):
        fuser.fuse("query", modules)


def test_uncertainty_factory_whitespace_case_and_stochastic_pass_floor() -> None:
    method = create_uncertainty_method(
        config=UncertaintyMethodConfig(
            method="  EMBEDDING_NORM_SIGMA2  ", base_sigma2=0.7
        ),
        embed=_toy_embed,
    )
    assert isinstance(method, EmbeddingNormSigma2)
    assert method.base_sigma2 == pytest.approx(0.7)

    rng = np.random.default_rng(123)

    def stochastic_embed(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float64) + rng.normal(scale=0.01, size=2)

    sigma2 = StochasticForwardSigma2(
        embed_stochastic=stochastic_embed, n_passes=1, base_sigma2=0.3
    )
    value = sigma2("tiny")
    assert value > 0.0
    assert np.isfinite(value)


def test_uncertainty_factory_unknown_method_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown uncertainty method"):
        create_uncertainty_method(
            config=UncertaintyMethodConfig(method="not_a_method"),
            embed=_toy_embed,
        )


def test_uncertainty_calibration_status_contract_for_powered_vs_underpowered(
    tmp_path: Path,
) -> None:
    powered = run_uncertainty_calibration(tmp_path / "powered")
    assert powered["status"] == "sufficient"
    assert powered["powered_for_calibration"] is True
    assert powered["fallback_reason"] is None

    underpowered = run_uncertainty_calibration(
        tmp_path / "underpowered",
        power_config=ValidationPowerConfig(
            min_validation_total=500,
            min_validation_per_domain=220,
            min_effective_support_per_specialist=260,
            min_validation_per_query_bucket=260,
            calibrator_min_samples=260,
        ),
    )
    assert underpowered["status"] == "underpowered_validation"
    assert underpowered["powered_for_calibration"] is False
    assert underpowered["fallback_reason"] == "underpowered_validation"


@pytest.mark.parametrize(
    ("name", "summary_relpath", "required_keys"),
    [
        (
            "canonical_benchmark_v2",
            "canonical_benchmark_v2/summary.json",
            {
                "benchmark",
                "methods",
                "paired_statistics",
                "decision",
                "bucket_analysis",
            },
        ),
        (
            "uncertainty_calibration",
            "uncertainty_calibration/summary.json",
            {
                "selected_objective",
                "objective_reports",
                "selected_report",
                "selection_is_validation_only",
            },
        ),
        (
            "correlation_aware_fusion",
            "correlation_aware_fusion/summary.json",
            {"config", "validation_fit", "test_metrics", "bucket_metrics", "answer"},
        ),
        (
            "kalman_covariance_ablation_v2",
            "kalman_covariance_ablation_v2/summary.json",
            {"config", "covariance_fit", "metrics", "bucket_metrics", "answer"},
        ),
        (
            "kalman_latency_optimization",
            "kalman_latency_optimization/summary.json",
            {
                "assumptions",
                "single_query_strategies",
                "batch_strategies",
                "numerical_deviation_vs_legacy",
                "canonical_rerun",
            },
        ),
    ],
)
def test_major_results_artifact_schema_contracts(
    name: str, summary_relpath: str, required_keys: set[str]
) -> None:
    summary_path = Path("results") / summary_relpath
    assert summary_path.exists(), (
        f"Expected committed artifact for {name}: {summary_path}"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert required_keys.issubset(payload.keys())


def test_snapshot_uncertainty_calibration_markdown(tmp_path: Path) -> None:
    out = tmp_path / "uncertainty_calibration"
    run_uncertainty_calibration_objective_study(
        out,
        objectives=("rank_error_proxy",),
        power_config=ValidationPowerConfig(query_expansion_multiplier=6),
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    expected = Path(
        "tests/snapshots/uncertainty_calibration_report_expected.md"
    ).read_text(encoding="utf-8")
    assert report == expected


def test_snapshot_covariance_ablation_markdown(tmp_path: Path) -> None:
    out = tmp_path / "kalman_covariance_ablation"
    run_kalman_covariance_ablation(
        out,
        config=AblationConfig(
            random_seed=7,
            dimension=12,
            n_docs=96,
            n_val=120,
            n_test=160,
            n_specialists=3,
            n_domains=3,
            fit=CovarianceFitConfig(lowrank_rank=2),
        ),
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "# Kalman Covariance Ablation" in report
    assert "## Retrieval Metrics" in report
    assert "## Per-bucket Recall@1" in report
    assert "| mean_fusion |" in report
    assert "| scalar_kalman |" in report
    assert "| diagonal_kalman |" in report
    assert "| structured_kalman |" in report
    assert "## Efficiency Trade-offs" in report


def test_snapshot_correlation_aware_fusion_markdown(tmp_path: Path) -> None:
    out = tmp_path / "correlation_aware"
    run_correlation_aware_fusion_benchmark(
        out,
        config=CorrelationAwareFusionConfig(
            random_seed=5,
            dimension=12,
            n_docs=80,
            n_val=64,
            n_test=90,
            n_specialists=3,
        ),
    )
    report = (out / "report.md").read_text(encoding="utf-8")
    expected = Path(
        "tests/snapshots/correlation_aware_fusion_report_expected.md"
    ).read_text(encoding="utf-8")
    assert report == expected


def test_optimized_kalman_numerical_equivalence_tight_tolerance() -> None:
    modules = [
        SEF(
            name="m0", embed=lambda q: np.array([1.0, 0.1, len(q) * 0.001]), sigma2=1e-4
        ),
        SEF(
            name="m1",
            embed=lambda q: np.array([0.9, 0.2, len(q) * 0.0015]),
            sigma2=2e-4,
        ),
        SEF(
            name="m2",
            embed=lambda q: np.array([0.2, 0.95, len(q) * 0.0003]),
            sigma2=8e-4,
        ),
    ]
    queries = ["tiny query", "longer deterministic query here"]

    legacy = KalmanorixFuser(use_fast_scalar_path=False)
    optimized = KalmanorixFuser(use_fast_scalar_path=True)

    x_legacy, w_legacy, m_legacy = legacy.fuse_batch(queries, modules)
    x_opt, w_opt, m_opt = optimized.fuse_batch(queries, modules)

    for xl, xo, wl, wo, ml, mo in zip(
        x_legacy, x_opt, w_legacy, w_opt, m_legacy, m_opt
    ):
        assert np.allclose(xo, xl, rtol=1e-11, atol=1e-12)
        assert np.allclose(
            mo["fused_covariance"], ml["fused_covariance"], rtol=1e-11, atol=1e-12
        )
        for k in wl:
            assert wo[k] == pytest.approx(wl[k], rel=1e-12, abs=1e-12)
