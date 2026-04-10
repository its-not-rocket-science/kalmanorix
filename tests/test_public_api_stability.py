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

    assert hasattr(experimental, "__path__")
    assert hasattr(experimental, "LearnedGateFuser")
    assert hasattr(experimental, "create_openai_sef")
    assert hasattr(experimental, "compute_alignments")


def test_legacy_top_level_experimental_import_warns() -> None:
    with pytest.deprecated_call(match="deprecated"):
        symbol = getattr(kalmanorix, "LearnedGateFuser")

    assert symbol is not None


def test_internal_utility_moves_to_internal_module() -> None:
    internal = importlib.import_module("kalmanorix.internal")
    assert hasattr(internal, "__path__")
    assert hasattr(internal, "create_procrustes_alignment")


def test_legacy_top_level_internal_import_warns() -> None:
    with pytest.deprecated_call(match="deprecated"):
        symbol = getattr(kalmanorix, "create_procrustes_alignment")

    assert symbol is not None


def test_internal_package_not_listed_in_public_api_docs_surface() -> None:
    import ast
    from pathlib import Path

    docs_generator = Path(__file__).resolve().parents[1] / "docs" / "generate_api.py"
    tree = ast.parse(docs_generator.read_text(encoding="utf-8"))
    api_modules_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "API_MODULES"
            for target in node.targets
        )
    )
    api_modules = ast.literal_eval(api_modules_node.value)
    documented_modules = {module for _, module, _, _ in api_modules}
    assert "kalmanorix.internal" not in documented_modules
