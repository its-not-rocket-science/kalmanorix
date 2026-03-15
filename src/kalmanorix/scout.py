"""
Routing logic for Kalmanorix.

The ScoutRouter decides which specialists to consult for a given query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .village import SEF, Village


@dataclass
class ScoutRouter:
    """
    Select which specialists to consult for a query.

    Modes
    -----
    all:
        Return all available modules (enables fusion).
    hard:
        Return only the single module with the lowest query-dependent sigma².
    """

    mode: str = "all"

    def select(self, query: str, village: Village) -> List[SEF]:
        """Return the selected specialist modules for the given query."""
        if self.mode == "all":
            return village.modules
        if self.mode == "hard":
            q = query.lower()

            charge_like = bool(
                re.search(
                    r"\b(usb-?c|pd\b|power delivery|pps|pdo|rdo|watt|wattage|e-?marker|qc\b)\b",
                    q,
                )
            )

            if charge_like:
                for m in village.modules:
                    if m.name == "charge":
                        return [m]

            return [min(village.modules, key=lambda m: m.sigma2_for(query))]
        raise ValueError("mode must be 'all' or 'hard'")
