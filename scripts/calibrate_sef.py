#!/usr/bin/env python
"""
SEF calibration script (Phase 1).

This script creates a minimal, portable SEF artifact describing a specialist
embedding module. The resulting artifact does NOT contain model weights.
Instead, it records:

- the specialist name
- an embedder identifier (resolved at runtime via a registry)
- domain metadata
- a query-dependent uncertainty (sigma²) configuration

This enables specialists to be shared, versioned, and composed without
retraining or distributing large models.

Current scope (Phase 1):
- keyword-based sigma² heuristics only
- JSON-based artifact format
- no alignment matrices or learned uncertainty models yet

Example
-------
python scripts/calibrate_sef.py \
  --name tech \
  --embedder-id toy-tech-v1 \
  --domain tech \
  --keywords battery smartphone cpu gpu laptop camera charger \
  --in-sigma2 0.2 \
  --out-sigma2 2.0 \
  --out artifacts/tech.sef.json
"""

from __future__ import annotations

import argparse

from kalmanorix.sef_io import SEFArtifact


def main() -> None:
    """
    Parse CLI arguments and write a SEF artifact to disk.

    The output artifact can later be loaded and combined with a runtime
    embedder registry to construct a full SEF object.
    """
    ap = argparse.ArgumentParser(description="Create a minimal SEF artifact.")
    ap.add_argument("--name", required=True, help="Name of the specialist module.")
    ap.add_argument(
        "--embedder-id",
        required=True,
        help="Identifier used to resolve the embedder at runtime.",
    )
    ap.add_argument(
        "--domain",
        required=True,
        help="High-level domain label (e.g. tech, cooking).",
    )
    ap.add_argument(
        "--keywords",
        nargs="+",
        required=True,
        help="Keywords indicating in-domain queries.",
    )
    ap.add_argument(
        "--in-sigma2",
        type=float,
        default=0.2,
        help="Uncertainty for in-domain queries (lower = more confident).",
    )
    ap.add_argument(
        "--out-sigma2",
        type=float,
        default=2.0,
        help="Uncertainty for out-of-domain queries (higher = less confident).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output path for the .sef.json artifact.",
    )
    args = ap.parse_args()

    art = SEFArtifact(
        name=args.name,
        embedder_id=args.embedder_id,
        meta={"domain": args.domain},
        sigma2_kind="keyword",
        sigma2_params={
            "keywords": list(args.keywords),
            "in_domain_sigma2": args.in_sigma2,
            "out_domain_sigma2": args.out_sigma2,
        },
    )

    art.save(args.out)
    print(f"Wrote SEF artifact to {args.out}")


if __name__ == "__main__":
    main()
