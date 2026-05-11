#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail if manuscript claims exceed claim_gate permissions."
    )
    parser.add_argument("--claim-gate", type=Path, required=True)
    parser.add_argument("manuscripts", nargs="+", type=Path)
    args = parser.parse_args()

    gate = json.loads(args.claim_gate.read_text(encoding="utf-8"))
    prohibited = [
        str(x).strip() for x in gate.get("prohibited_claims", []) if str(x).strip()
    ]
    allowed_sentence = str(gate.get("allowed_headline_sentence", "")).strip()
    final_verdict = str(gate.get("final_verdict", "")).strip().lower()

    violations: list[str] = []
    for manuscript in args.manuscripts:
        text = manuscript.read_text(encoding="utf-8").lower()
        for claim in prohibited:
            if claim.lower() in text:
                violations.append(f"{manuscript}: prohibited claim found -> {claim}")
        if (
            final_verdict != "allowed"
            and allowed_sentence
            and allowed_sentence.lower() in text
        ):
            violations.append(
                f"{manuscript}: contains allowed_headline_sentence while final_verdict={final_verdict}"
            )

    if violations:
        raise SystemExit(
            "\n".join(["Publication claim validation failed:", *violations])
        )

    print(
        json.dumps(
            {
                "status": "pass",
                "claim_gate": str(args.claim_gate),
                "manuscripts_checked": [str(p) for p in args.manuscripts],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
