"""Regression tests for public API tiering and compatibility shims."""

from __future__ import annotations

import importlib

import pytest

import kalmanorix


STABLE_PUBLIC_API = {
    "SEF",
    "Village",
    "compute_domain_centroid",
    "ScoutRouter",
    "Panoramix",
    "Potion",
    "MeanFuser",
    "KalmanorixFuser",
    "eval_retrieval",
}


@pytest.mark.parametrize("symbol", sorted(STABLE_PUBLIC_API))
def test_stable_api_symbol_is_importable(symbol: str) -> None:
    assert hasattr(kalmanorix, symbol), f"Missing stable API symbol: {symbol}"


def test_stable_all_matches_contract() -> None:
    assert set(kalmanorix.__all__) == STABLE_PUBLIC_API


def test_experimental_symbols_live_under_experimental_module() -> None:
    experimental = importlib.import_module("kalmanorix.experimental")

    assert hasattr(experimental, "LearnedGateFuser")
    assert hasattr(experimental, "create_openai_sef")
    assert hasattr(experimental, "compute_alignments")


def test_legacy_top_level_experimental_import_warns() -> None:
    with pytest.deprecated_call(match="deprecated"):
        symbol = getattr(kalmanorix, "LearnedGateFuser")

    assert symbol is not None


def test_internal_utility_moves_to_internal_module() -> None:
    internal = importlib.import_module("kalmanorix.internal")
    assert hasattr(internal, "create_procrustes_alignment")
