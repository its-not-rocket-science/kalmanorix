"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _PatchProxy:
    """Minimal proxy that mimics the pytest-mock patch API used in tests."""

    def __init__(self, patches: list[patch]) -> None:
        self._patches = patches

    def dict(self, target: str, values: dict[str, object]) -> None:
        patcher = patch.dict(target, values)
        patcher.start()
        self._patches.append(patcher)


class SimpleMocker:
    """Lightweight stand-in for the pytest-mock ``mocker`` fixture."""

    def __init__(self) -> None:
        self._patches: list[patch] = []
        self.patch = _PatchProxy(self._patches)
        self.MagicMock = MagicMock

    def stop(self) -> None:
        while self._patches:
            self._patches.pop().stop()


@pytest.fixture
def mocker() -> SimpleMocker:
    """Provide a minimal ``mocker`` fixture for tests that need patch.dict."""
    fixture = SimpleMocker()
    try:
        yield fixture
    finally:
        fixture.stop()
