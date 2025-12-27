from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .village import SEF, Village

@dataclass
class ScoutRouter:
    mode: str = "all"

    def select(self, query: str, village: Village) -> List[SEF]:
        _ = query
        if self.mode == "all":
            return village.modules
        if self.mode == "hard":
            return [min(village.modules, key=lambda m: m.sigma2)]
        raise ValueError("mode must be 'all' or 'hard'")
