"""
SEF artifact I/O.

This module defines a minimal, versioned on-disk representation for a
Specialist Embedding Format (SEF).

An SEF artifact captures *configuration and metadata*, not model weights.
At runtime, the artifact is combined with:
- an embedder registry (mapping embedder_id -> embed function)
- the core Kalmanorix fusion pipeline

Design goals
------------
- Portability: artifacts are small JSON files.
- Composability: specialists can be mixed without retraining.
- Forward compatibility: explicit versioning.

Non-goals (for now)
-------------------
- Serializing model weights or LoRA adapters
- Storing alignment matrices
- Learned uncertainty models

These can be added in future artifact versions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .uncertainty import KeywordSigma2


SEF_ARTIFACT_VERSION = "0.1"


@dataclass(frozen=True)
class SEFArtifact:
    """
    Serializable description of a specialist embedding module.

    Attributes
    ----------
    name:
        Human-readable name of the specialist.
    embedder_id:
        Identifier used to resolve the embedder at runtime.
    meta:
        Arbitrary metadata (e.g. domain, license, provenance).
    sigma2_kind:
        Kind of uncertainty model ("keyword" in version 0.1).
    sigma2_params:
        Parameters for the uncertainty model.
    version:
        Artifact schema version.
    """

    name: str
    embedder_id: str
    meta: Dict[str, str]
    sigma2_kind: str
    sigma2_params: Dict[str, Any]
    version: str = SEF_ARTIFACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the artifact to a JSON-serializable dictionary.
        """
        return {
            "version": self.version,
            "name": self.name,
            "embedder_id": self.embedder_id,
            "meta": self.meta,
            "sigma2": {
                "kind": self.sigma2_kind,
                "params": self.sigma2_params,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SEFArtifact":
        """
        Construct an artifact from a dictionary.

        Raises
        ------
        ValueError
            If the artifact version is unsupported.
        """
        if d.get("version") != SEF_ARTIFACT_VERSION:
            raise ValueError(f"Unsupported SEF artifact version: {d.get('version')}")

        sigma2 = d.get("sigma2", {})
        return cls(
            name=str(d["name"]),
            embedder_id=str(d["embedder_id"]),
            meta=dict(d.get("meta", {})),
            sigma2_kind=str(sigma2.get("kind", "")),
            sigma2_params=dict(sigma2.get("params", {})),
            version=str(d["version"]),
        )

    def save(self, path: str | Path) -> None:
        """
        Write the artifact to disk as pretty-printed JSON.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SEFArtifact":
        """
        Load an artifact from disk.
        """
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def build_sigma2(self):
        """
        Instantiate a query-dependent sigma² callable from the artifact.

        Returns
        -------
        callable
            A function mapping query -> variance.

        Raises
        ------
        ValueError
            If the sigma² kind is unknown.
        """
        if self.sigma2_kind == "keyword":
            keywords = set(self.sigma2_params["keywords"])
            return KeywordSigma2(
                keywords=keywords,
                in_domain_sigma2=float(self.sigma2_params["in_domain_sigma2"]),
                out_domain_sigma2=float(self.sigma2_params["out_domain_sigma2"]),
            )

        raise ValueError(f"Unknown sigma2 kind: {self.sigma2_kind}")
