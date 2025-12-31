"""SentenceTransformer embedder adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kalmanorix.types import Embedder, Vec

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class STEmbedder(Embedder):
    """SentenceTransformer-backed embedder implementing kalmanorix.types.Embedder."""

    model: "SentenceTransformer"

    def __call__(self, text: str) -> Vec:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[
            0
        ]
        return v.astype(np.float64)
