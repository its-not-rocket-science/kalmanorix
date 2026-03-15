"""
SEF artefact I/O.

This module defines a minimal, versioned on-disk representation for a
Specialist Embedding Format (SEF).

An SEF artefact captures *configuration and metadata*, not model weights.
At runtime, the artefact is combined with:
- an embedder registry (mapping embedder_id -> embed(text) -> vector)
- the core Kalmanorix fusion pipeline
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol

import numpy as np

from .uncertainty import CentroidDistanceSigma2, KeywordSigma2

SEF_ARTEFACT_VERSION = "0.1"
Sigma2Fn = Callable[[str], float]


def _read_lines(path: Path) -> List[str]:
    """
    Read non-empty, stripped lines from a UTF-8 text file.

    Lines beginning with '#' are treated as comments and ignored.
    """
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _coerce_float(d: Mapping[str, Any], key: str, default: float) -> float:
    val = d.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for '{key}': {val!r}") from exc


# pylint: disable=too-few-public-methods
class _RegistryLike(Protocol):
    """
    Minimal protocol for an embedder registry.

    Any object that provides `get(embedder_id) -> embed(text) -> np.ndarray`
    satisfies this protocol (duck typing).
    """

    # pylint: disable=unnecessary-ellipsis
    def get(self, embedder_id: str) -> Callable[[str], np.ndarray]:
        """Return an embedder callable for the given embedder_id."""
        ...


@dataclass(frozen=True)
class SEFArtefact:
    """
    Serializable description of a specialist embedding module.

    Attributes
    ----------
    name:
        Human-readable name of the specialist.
    embedder_id:
        Identifier used to resolve the embedder at runtime.
    meta:
        Arbitrary metadata (e.g. domain, licence, provenance).
    sigma2_kind:
        Kind of uncertainty model ("keyword", "centroid_distance").
    sigma2_params:
        Parameters for the uncertainty model.
    version:
        Artefact schema version.
    """

    name: str
    embedder_id: str
    meta: Dict[str, str]
    sigma2_kind: str
    sigma2_params: Dict[str, Any]
    version: str = SEF_ARTEFACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert the artefact to a JSON-serializable dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "embedder_id": self.embedder_id,
            "meta": self.meta,
            "sigma2": {"kind": self.sigma2_kind, "params": self.sigma2_params},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SEFArtefact":
        """Construct an artefact from a dictionary."""
        if d.get("version") != SEF_ARTEFACT_VERSION:
            raise ValueError(f"Unsupported SEF artefact version: {d.get('version')}")

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
        """Write the artefact to disk as pretty-printed JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SEFArtefact":
        """Load an artefact from disk."""
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def build_sigma2(
        self,
        *,
        registry: _RegistryLike | None = None,
        base_dir: str | Path | None = None,
    ) -> Sigma2Fn:
        """
        Instantiate a query-dependent sigma² callable from the artefact.

        Parameters
        ----------
        registry:
            Embedder registry used to resolve self.embedder_id.
        base_dir:
            Base directory used to resolve relative calibration_path entries
            for centroid_distance. If None, defaults to current working directory.

        Returns
        -------
        Sigma2Fn
            Callable mapping query -> variance.
        """
        kind = self.sigma2_kind.strip()
        params = self.sigma2_params

        if kind == "keyword":
            raw_keywords = params.get("keywords")
            if not isinstance(raw_keywords, list):
                raise ValueError("keyword sigma2 requires params.keywords: list[str]")
            keywords = {str(k) for k in raw_keywords}
            return KeywordSigma2(
                keywords=keywords,
                in_domain_sigma2=_coerce_float(params, "in_domain_sigma2", 0.2),
                out_domain_sigma2=_coerce_float(params, "out_domain_sigma2", 2.0),
            )

        if kind == "centroid_distance":
            if registry is None:
                raise ValueError("centroid_distance sigma2 requires a registry")
            embed = registry.get(self.embedder_id)
            base_sigma2 = _coerce_float(params, "base_sigma2", 0.2)
            scale = _coerce_float(params, "scale", 2.0)

            calibration_texts: Optional[Iterable[str]] = None

            if "calibration_texts" in params:
                raw = params["calibration_texts"]
                if not isinstance(raw, list):
                    raise ValueError(
                        "centroid_distance sigma2 requires params.calibration_texts "
                        "to be a list[str] when provided"
                    )
                calibration_texts = [str(s) for s in raw]

            if calibration_texts is None and "calibration_path" in params:
                raw_path = params["calibration_path"]
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(
                        "centroid_distance sigma2 requires params.calibration_path "
                        "to be a non-empty string when provided"
                    )
                root = Path(base_dir) if base_dir is not None else Path.cwd()
                cal_path = Path(raw_path)
                if not cal_path.is_absolute():
                    cal_path = root / cal_path
                if not cal_path.exists():
                    raise FileNotFoundError(f"Calibration file not found: {cal_path}")
                calibration_texts = _read_lines(cal_path)

            if calibration_texts is None:
                raise ValueError(
                    "centroid_distance sigma2 requires either "
                    "params.calibration_texts or params.calibration_path"
                )

            return CentroidDistanceSigma2.from_calibration(
                embed=embed,
                calibration_texts=calibration_texts,
                base_sigma2=base_sigma2,
                scale=scale,
            )

        raise ValueError(f"Unknown sigma2 kind: {kind}")
