"""Kalmanorix public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import numpy as np


from .types import Embedder

Vec = np.ndarray
Sigma2 = Union[float, Callable[[str], float]]


@dataclass(frozen=True)
class SEF:
    """Specialist Embedding Format (Phase 0/1).

    sigma2 can be:
      - float: constant uncertainty
      - Callable[[str], float]: query-dependent uncertainty
    """

    name: str
    embed: Embedder
    sigma2: Sigma2
    meta: Optional[Dict[str, str]] = None

    def sigma2_for(self, query: str) -> float:
        """Return uncertainty (variance) for a given query."""
        if callable(self.sigma2):
            val = float(self.sigma2(query))
        else:
            val = float(self.sigma2)

        # Safety: avoid zero/negative variances
        return max(val, 1e-12)


@dataclass
class Village:
    """A simple container for specialists available at runtime."""

    modules: List[SEF]

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("Village must contain at least one SEF")

    def list(self) -> List[str]:
        """List names of available modules."""
        return [m.name for m in self.modules]
