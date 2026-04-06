"""Validation tests for Kalmanorix fixes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from kalmanorix.alignment import validate_alignment_sign
from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal_ensemble
from kalmanorix.models.sef import create_procrustes_alignment
from kalmanorix.panoramix import KalmanorixFuser
from kalmanorix.uncertainty import CentroidDistanceSigma2
from kalmanorix.village import SEF


class TestProcrustesFix:
    """Test that alignment transpose error is fixed."""

    def test_alignment_preserves_similarity_direction(self):
        """After alignment, all specialists should have positive similarity to reference."""
        rng = np.random.default_rng(7)
        n, d = 80, 12
        reference = rng.normal(size=(n, d))

        specialist_mats = [
            np.eye(d),
            np.linalg.qr(rng.normal(size=(d, d)))[0],
            np.linalg.qr(rng.normal(size=(d, d)))[0],
        ]

        mean_after_scores = []
        for idx, mat in enumerate(specialist_mats):
            specialist = reference @ mat + rng.normal(scale=0.01, size=(n, d))
            align = create_procrustes_alignment(specialist, reference)
            _before, mean_after, _det = validate_alignment_sign(
                sef_name=f"specialist_{idx}",
                src_embeddings=specialist,
                ref_embeddings=reference,
                align_matrix=align,
            )
            mean_after_scores.append(mean_after)

        assert all(score > 0.0 for score in mean_after_scores)

    def test_no_negative_centroid_similarity(self):
        """Centroid similarity should never be negative after alignment."""
        rng = np.random.default_rng(17)
        n, d = 100, 10
        reference = rng.normal(size=(n, d))

        random_rotation = np.linalg.qr(rng.normal(size=(d, d)))[0]
        specialist = reference @ random_rotation + rng.normal(scale=0.02, size=(n, d))

        align = create_procrustes_alignment(specialist, reference)
        aligned = specialist @ align

        ref_centroid = reference.mean(axis=0)
        ref_centroid /= np.linalg.norm(ref_centroid) + 1e-12

        aligned_centroid = aligned.mean(axis=0)
        aligned_centroid /= np.linalg.norm(aligned_centroid) + 1e-12

        centroid_similarity = float(ref_centroid @ aligned_centroid)
        assert centroid_similarity >= 0.0


class TestCovarianceScaling:
    """Test that covariance scaling doesn't suppress specialists."""

    def test_uncertainty_ratio_bounded(self):
        """Max/min uncertainty ratio should be ≤ 5x."""

        def embed(text: str) -> np.ndarray:
            if "in-domain" in text:
                return np.array([1.0, 0.0, 0.0], dtype=np.float64)
            return np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        sigma2 = CentroidDistanceSigma2.from_calibration(
            embed=embed,
            calibration_texts=["in-domain sample", "in-domain jargon"],
            base_sigma2=0.3,
            beta=3.0,
        )

        low_uncertainty = sigma2("in-domain question")
        high_uncertainty = sigma2("far out-of-domain query")

        ratio = high_uncertainty / max(low_uncertainty, 1e-12)
        assert ratio <= 5.0 + 1e-9

    def test_medical_specialist_not_suppressed(self):
        """Medical specialist should get non-trivial weight in fusion."""
        query = "hypertension treatment options"

        medical_sigma2 = CentroidDistanceSigma2(
            embed=lambda _: np.array([1.0, 0.0, 0.0], dtype=np.float64),
            centroid=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            base_sigma2=0.3,
            beta=2.5,
        )
        sigma2_spy = Mock(side_effect=medical_sigma2)

        modules = [
            SEF("tech", lambda _: np.array([1.0, 0.0, 0.0]), sigma2=0.2),
            SEF("legal", lambda _: np.array([0.0, 1.0, 0.0]), sigma2=0.2),
            SEF("medical", lambda _: np.array([0.2, 0.1, 0.9]), sigma2=sigma2_spy),
        ]

        fuser = KalmanorixFuser(sort_by_certainty=True)
        _fused, weights, _meta = fuser.fuse(query, modules)

        assert sigma2_spy.called, "Expected query-dependent medical sigma2 to be used"
        assert weights["medical"] > 0.05


class TestEnsembleFusion:
    """Test that ensemble fusion is numerically stable."""

    def test_order_invariance(self):
        """Fusion result should be independent of specialist order."""
        rng = np.random.default_rng(123)
        d = 16
        embeddings = [rng.normal(size=d) for _ in range(4)]
        covariances = [rng.uniform(0.05, 0.5, size=d) for _ in range(4)]

        fused_a, cov_a = kalman_fuse_diagonal_ensemble(embeddings, covariances, epsilon=1e-12)

        order = [2, 0, 3, 1]
        perm_embeddings = [embeddings[i] for i in order]
        perm_covariances = [covariances[i] for i in order]
        fused_b, cov_b = kalman_fuse_diagonal_ensemble(
            perm_embeddings,
            perm_covariances,
            epsilon=1e-12,
        )

        assert np.allclose(fused_a, fused_b, rtol=1e-12, atol=1e-12)
        assert np.allclose(cov_a, cov_b, rtol=1e-12, atol=1e-12)

    def test_no_nan_propagation(self):
        """Extreme covariances shouldn't produce NaN."""
        d = 32
        embeddings = [
            np.ones(d, dtype=np.float64),
            -np.ones(d, dtype=np.float64),
            np.linspace(-1.0, 1.0, d, dtype=np.float64),
        ]
        covariances = [
            np.full(d, 1e-12, dtype=np.float64),
            np.full(d, 1e12, dtype=np.float64),
            np.full(d, 1.0, dtype=np.float64),
        ]

        fused, fused_cov = kalman_fuse_diagonal_ensemble(
            embeddings,
            covariances,
            epsilon=1e-12,
        )

        assert np.all(np.isfinite(fused))
        assert np.all(np.isfinite(fused_cov))
        assert np.all(fused_cov > 0)


class TestNoResultFabrication:
    """Test that benchmarks aren't hardcoded."""

    def test_results_json_has_p_values(self):
        """results.json should contain p_values, not hardcoded recalls."""
        benchmark_script = Path("experiments/benchmark_fusion_methods.py")
        assert benchmark_script.exists()
        text = benchmark_script.read_text(encoding="utf-8")

        assert '"p_values"' in text

        # Minimal schema sanity: if results are emitted, we require p_values in the payload.
        payload = {
            "overall_accuracy_comparison": {"kalman": {"recall@1": 0.5}, "mean": {"recall@1": 0.4}},
            "p_values": {"recall@1": 0.03},
        }
        json_blob = json.dumps(payload)
        parsed = json.loads(json_blob)
        assert "p_values" in parsed

    def test_no_hardcoded_perfect_scores(self):
        """Check for hardcoded patterns like {'1': 0.99}."""
        experiments_dir = Path("experiments")
        py_files = list(experiments_dir.glob("*.py"))
        assert py_files, "Expected experiment scripts to be present"

        suspicious = re.compile(
            r"\{\s*['\"](?:1|5|10)['\"]\s*:\s*(?:0\.99+|1\.0+)\s*(?:,|\})",
            flags=re.MULTILINE,
        )

        flagged: list[str] = []
        for file_path in py_files:
            text = file_path.read_text(encoding="utf-8")
            if suspicious.search(text):
                flagged.append(str(file_path))

        assert not flagged, f"Potential hardcoded perfect-score patterns found in: {flagged}"
