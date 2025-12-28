from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .types import Embedder

Vec = np.ndarray

@dataclass(frozen=True)
class SEF:
    name: str
    embed: Embedder
    sigma2: float
    meta: Optional[Dict[str, str]] = None

@dataclass
class Village:
    modules: List[SEF]

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("Village must contain at least one SEF")

    def list(self) -> List[str]:
        return [m.name for m in self.modules]
