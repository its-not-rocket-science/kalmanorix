"""Regression tests for optimized Kalmanorix scalar-sigma² fast path."""

from __future__ import annotations

import numpy as np

from kalmanorix import KalmanorixFuser, SEF


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


def test_optimized_kalman_matches_legacy_single_query() -> None:
    modules = _make_modules(n=5, d=128)
    query = "optimize kalman latency without changing semantics"

    legacy = KalmanorixFuser(use_fast_scalar_path=False)
    fast = KalmanorixFuser(use_fast_scalar_path=True)

    x_legacy, w_legacy, m_legacy = legacy.fuse(query, modules)
    x_fast, w_fast, m_fast = fast.fuse(query, modules)

    assert np.allclose(x_fast, x_legacy, rtol=1e-10, atol=1e-12)
    assert np.allclose(m_fast["fused_covariance"], m_legacy["fused_covariance"], rtol=1e-10, atol=1e-12)
    for key in w_legacy:
        assert np.isclose(w_fast[key], w_legacy[key], rtol=1e-12, atol=1e-12)


def test_optimized_kalman_matches_legacy_batch() -> None:
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

    for xl, xf, ml, mf, wl, wf in zip(x_legacy, x_fast, m_legacy, m_fast, w_legacy, w_fast):
        assert np.allclose(xf, xl, rtol=1e-10, atol=1e-12)
        assert np.allclose(mf["fused_covariance"], ml["fused_covariance"], rtol=1e-10, atol=1e-12)
        for key in wl:
            assert np.isclose(wf[key], wl[key], rtol=1e-12, atol=1e-12)
