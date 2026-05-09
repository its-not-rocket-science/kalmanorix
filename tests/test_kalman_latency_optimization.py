"""Regression tests for optimised Kalmanorix scalar-sigma² fast path."""

from __future__ import annotations

import numpy as np

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village


def _make_modules(n: int = 4, d: int = 64) -> list[SEF]:
    rng = np.random.default_rng(123)
    modules: list[SEF] = []
    for i in range(n):
        direction = rng.normal(size=(d,))
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        bias = rng.normal(scale=0.05, size=(d,))

        def make_embed(dir_vec: np.ndarray, b: np.ndarray):
            def _embed(query: str) -> np.ndarray:
                q_scale = (len(query.split()) + 1) / 10.0
                vec = dir_vec * q_scale + b
                return vec.astype(np.float64)

            return _embed

        embed = make_embed(direction, bias)
        sigma2 = 0.05 + 0.1 * (i + 1)
        modules.append(SEF(name=f"m{i}", embed=embed, sigma2=sigma2))
    return modules


def test_optimised_kalman_matches_legacy_single_query() -> None:
    modules = _make_modules(n=5, d=128)
    query = "optimise kalman latency without changing semantics"

    legacy = KalmanorixFuser(use_fast_scalar_path=False)
    fast = KalmanorixFuser(use_fast_scalar_path=True)

    x_legacy, w_legacy, m_legacy = legacy.fuse(query, modules)
    x_fast, w_fast, m_fast = fast.fuse(query, modules)

    assert np.allclose(x_fast, x_legacy, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        m_fast["fused_covariance"], m_legacy["fused_covariance"], rtol=1e-10, atol=1e-12
    )
    for key in w_legacy:
        assert np.isclose(w_fast[key], w_legacy[key], rtol=1e-12, atol=1e-12)


def test_optimised_kalman_matches_legacy_batch() -> None:
    modules = _make_modules(n=4, d=96)
    queries = [
        "finance portfolio hedging",
        "medical diagnosis treatment options",
        "distributed systems throughput latency",
        "sourdough hydration fermentation",
    ]

    legacy = KalmanorixFuser(use_fast_scalar_path=False)
    fast = KalmanorixFuser(use_fast_scalar_path=True)

    x_legacy, w_legacy, m_legacy = legacy.fuse_batch(queries, modules)
    x_fast, w_fast, m_fast = fast.fuse_batch(queries, modules)

    for xl, xf, ml, mf, wl, wf in zip(
        x_legacy, x_fast, m_legacy, m_fast, w_legacy, w_fast
    ):
        assert np.allclose(xf, xl, rtol=1e-10, atol=1e-12)
        assert np.allclose(
            mf["fused_covariance"], ml["fused_covariance"], rtol=1e-10, atol=1e-12
        )
        for key in wl:
            assert np.isclose(wf[key], wl[key], rtol=1e-12, atol=1e-12)


def test_shared_embedding_path_preserves_mean_and_kalman_outputs() -> None:
    modules = _make_modules(n=4, d=64)
    village = Village(modules=modules)
    scout = ScoutRouter(mode="all")
    query = "shared embedding path should preserve numerical outputs"

    old_mean = Panoramix(fuser=MeanFuser(), use_shared_embedding_path=False)
    new_mean = Panoramix(fuser=MeanFuser(), use_shared_embedding_path=True)
    old_kalman = Panoramix(
        fuser=KalmanorixFuser(use_fast_scalar_path=True),
        use_shared_embedding_path=False,
    )
    new_kalman = Panoramix(
        fuser=KalmanorixFuser(use_fast_scalar_path=True),
        use_shared_embedding_path=True,
    )

    old_mean_p = old_mean.brew(query, village=village, scout=scout)
    new_mean_p = new_mean.brew(query, village=village, scout=scout)
    old_kalman_p = old_kalman.brew(query, village=village, scout=scout)
    new_kalman_p = new_kalman.brew(query, village=village, scout=scout)

    assert np.allclose(new_mean_p.vector, old_mean_p.vector, rtol=1e-12, atol=1e-12)
    assert np.allclose(new_kalman_p.vector, old_kalman_p.vector, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        np.asarray(new_kalman_p.meta["fused_covariance"], dtype=np.float64),
        np.asarray(old_kalman_p.meta["fused_covariance"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_kalman_fuser_reuses_precomputed_embeddings_without_reembedding() -> None:
    class CountingEmbed:
        def __init__(self, seed: int):
            self.rng = np.random.default_rng(seed)
            self.calls = 0
            self.direction = self.rng.normal(size=(32,))

        def __call__(self, query: str) -> np.ndarray:
            self.calls += 1
            scale = max(len(query.split()), 1)
            vec = self.direction * (scale / 10.0)
            return vec.astype(np.float64)

    class EmbeddingAwareSigma2:
        def __call__(self, query: str) -> float:
            raise AssertionError("estimate_with_embedding should be used")

        def estimate_with_embedding(self, query: str, embedding: np.ndarray) -> float:
            return float(0.1 + np.linalg.norm(embedding) * 0.01 + len(query) * 1e-5)

    embeds = [CountingEmbed(1), CountingEmbed(2), CountingEmbed(3)]
    modules = [
        SEF(name=f"m{i}", embed=embeds[i], sigma2=EmbeddingAwareSigma2())
        for i in range(3)
    ]
    query = "uncertainty heavy query for duplicate embed regression"
    fuser = KalmanorixFuser(use_fast_scalar_path=True)

    _ = fuser.fuse(query, modules)
    baseline_calls = [e.calls for e in embeds]

    precomputed = [m.embed(query) for m in modules]

    for e in embeds:
        e.calls = 0

    _ = fuser.fuse(query, modules, precomputed_embeddings=precomputed)
    reused_calls = [e.calls for e in embeds]

    assert baseline_calls == [1, 1, 1]
    assert reused_calls == [0, 0, 0]


def test_panoramix_shared_path_computes_each_embedding_once_per_query() -> None:
    class CountingEmbed:
        def __init__(self, seed: int):
            self.rng = np.random.default_rng(seed)
            self.calls = 0
            self.direction = self.rng.normal(size=(24,))

        def __call__(self, query: str) -> np.ndarray:
            self.calls += 1
            scale = max(len(query.split()), 1)
            return (self.direction * (scale / 10.0)).astype(np.float64)

    class EmbeddingAwareSigma2:
        def __init__(self):
            self.calls = 0
            self.embedding_calls = 0

        def __call__(self, query: str) -> float:
            self.calls += 1
            return 0.5 + 1e-6 * len(query)

        def estimate_with_embedding(self, query: str, embedding: np.ndarray) -> float:
            self.embedding_calls += 1
            return float(0.1 + np.linalg.norm(embedding) * 0.001 + len(query) * 1e-6)

    embeds = [CountingEmbed(11), CountingEmbed(22), CountingEmbed(33)]
    sigma2 = [EmbeddingAwareSigma2(), EmbeddingAwareSigma2(), EmbeddingAwareSigma2()]
    village = Village(
        modules=[SEF(name=f"m{i}", embed=embeds[i], sigma2=sigma2[i]) for i in range(3)]
    )
    scout = ScoutRouter(mode="all")
    pan = Panoramix(fuser=KalmanorixFuser(use_fast_scalar_path=True))

    _ = pan.brew("query-level duplicate embedding audit", village=village, scout=scout)

    assert [embed.calls for embed in embeds] == [1, 1, 1]
    assert [s.calls for s in sigma2] == [0, 0, 0]
    assert [s.embedding_calls for s in sigma2] == [1, 1, 1]


def test_run_canonical_benchmark_uses_sys_executable_and_merges_pythonpath(
    tmp_path, monkeypatch
) -> None:
    import json
    import os
    import sys
    from pathlib import Path

    from experiments import run_kalman_latency_optimization as latency

    out_dir = tmp_path / "out"
    canonical_out = out_dir / "canonical"
    canonical_out.mkdir(parents=True)
    (canonical_out / "summary.json").write_text(
        json.dumps(
            {
                "decision": {
                    "kalman_vs_mean": {
                        "verdict": "supported",
                        "observed": {
                            "latency_ratio_vs_mean": 1.0,
                            "primary_metric_delta": 0.0,
                            "primary_metric_adjusted_p_value": 1.0,
                        },
                        "rules": {"max_latency_ratio_vs_mean": 1.05},
                        "checks": {"latency_ratio_ok": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PYTHONPATH", "existing_path")
    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    def _fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr(latency.subprocess, "run", _fake_run)

    result = latency._run_canonical_benchmark(
        out_dir=out_dir,
        benchmark_path=Path("tiny.parquet"),
        max_queries=5,
    )

    assert result["decision_verdict"] == "supported"
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1] == str(Path("experiments") / "run_canonical_benchmark.py")
    expected_pythonpath = os.pathsep.join(
        [str(tmp_path), str(tmp_path / "src"), "existing_path"]
    )
    assert captured["env"]["PYTHONPATH"] == expected_pythonpath
