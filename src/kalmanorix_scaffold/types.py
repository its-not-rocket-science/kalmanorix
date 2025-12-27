from __future__ import annotations
from typing import Protocol
import numpy as np
Vec = np.ndarray

class Embedder(Protocol):
    def __call__(self, text: str) -> Vec: ...
