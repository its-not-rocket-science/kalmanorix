"""Shared structural types used across Kalmanorix."""

from __future__ import annotations

from typing import Protocol

import numpy as np

Vec = np.ndarray


# pylint: disable=too-few-public-methods
class Embedder(Protocol):
    """Protocol for text -> vector embedders used by SEFs and fusers."""

    def __call__(self, text: str) -> Vec: ...
