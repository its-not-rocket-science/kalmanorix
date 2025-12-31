"""
Embedder registry.

Artifacts reference embedders by an identifier. This registry maps embedder_id
to an actual embed(text)->vector callable at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

EmbedderFn = Callable[[str], np.ndarray]


@dataclass
class EmbedderRegistry:
    """Simple name -> object registry used for loading modules/components."""

    embedders: Dict[str, EmbedderFn]

    def get(self, embedder_id: str) -> EmbedderFn:
        """Retrieve embedder by ID."""
        if embedder_id not in self.embedders:
            raise KeyError(f"Unknown embedder_id: {embedder_id}")
        return self.embedders[embedder_id]
