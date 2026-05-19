#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

UNSUPPORTED_CLAIMS = {
    "Kalman improves retrieval": "kalman_vs_mean_quality",
    "outperforms mean": "kalman_vs_mean_quality",
    "superior to monolith": "matched_compute_specialists_vs_monolith",
    "robust OOD": "ood_uncertainty_weighting",
}

SUPPORTED_STATUSES = {"supported", "exploratory"}
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?%?)(?![A-Za-z0-9_])")
ALLOWED_DISCLAIMER_CONTEXT = (
    "do not",
    "unsupported",
    "prohibited",
    "not supported",
    "inconclusive",
    "negative result",
)


def collect_target_files(root: Path) -> list[Path]:
    files: list[Path] = []
    files.extend(
        [root / "README.md", root / "ROADMAP.md", root / "docs/research/results.md"]
    )
    files.extend(sorted((root / "docs/publication").glob("*.md")))
    files.extend(sorted((root / "paper/arxiv").glob("*.tex")))
    files.extend(sorted((root / "paper/tmlr").glob("*.tex")))
    files.extend(sorted((root / "paper/joss").glob("*.md")))
    return [p for p in files if p.exists()]


def load_evidence(root: Path) -> dict[str, str]:
    data = json.loads(
        (root / "results/evidence_registry.json").read_text(encoding="utf-8")
    )
    return {entry["claim_id"]: entry["evidence_status"] for entry in data["claims"]}


def check_unsupported_claim_phrases(
    files: Iterable[Path], statuses: dict[str, str], root: Path
) -> list[str]:
    errors: list[str] = []
    for f in files:
        if "claims-policy" in f.name or "claims_policy" in f.name:
            continue
        text = f.read_text(encoding="utf-8")
        for phrase, claim_id in UNSUPPORTED_CLAIMS.items():
            m = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
            if m:
                line = text[max(0, m.start() - 100) : m.end() + 100].lower()
                if any(c in line for c in ALLOWED_DISCLAIMER_CONTEXT):
                    continue
                if statuses.get(claim_id) not in SUPPORTED_STATUSES:
                    errors.append(
                        f"Unsupported phrase '{phrase}' appears in {f.relative_to(root)} but {claim_id}={statuses.get(claim_id)!r}."
                    )
    return errors


def collect_artifact_text(root: Path) -> str:
    registry = json.loads(
        (root / "results/evidence_registry.json").read_text(encoding="utf-8")
    )
    artifact_paths = {
        Path(p) for c in registry["claims"] for p in c.get("artifact_paths_used", [])
    }
    artifact_paths.update((root / "docs/publication/tables").glob("*.md"))

    parts: list[str] = []
    for p in artifact_paths:
        full = p if p.is_absolute() else root / p
        if full.exists() and full.is_file():
            parts.append(full.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def check_headline_numbers(
    files: Iterable[Path], artifact_text: str, root: Path
) -> list[str]:
    errors: list[str] = []
    for f in files:
        if "claims-policy" in f.name or "claims_policy" in f.name:
            continue
        text = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            # headline heuristic: metrics-bearing headings / emphasized claim lines
            ll = line.lower()
            if not (line.startswith("#") or "\\section" in line or "**" in line):
                continue
            if not any(
                k in ll for k in ("ndcg", "mrr", "recall", "latency", "flops", "%")
            ):
                continue
            for num in NUMBER_PATTERN.findall(line):
                if len(num.rstrip("%")) <= 1:
                    continue
                if num not in artifact_text and num.rstrip("%") not in artifact_text:
                    errors.append(
                        f"Headline number {num} in {f.relative_to(root)}:{line_no} not found in generated artifact files."
                    )
    return errors


def extract_positioning_paragraph(text: str) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    for p in paras:
        pl = p.lower()
        if "position" in pl or "this paper" in pl or "this work" in pl:
            return re.sub(r"\s+", " ", pl)
    return ""


def check_distinct_positioning(root: Path) -> list[str]:
    files = {
        "arxiv": root / "paper/arxiv/kalmanorix_negative_result.tex",
        "tmlr": root / "paper/tmlr/main.tex",
        "joss": root / "paper/joss/paper.md",
    }
    paragraphs = {}
    errors = []
    for venue, path in files.items():
        if not path.exists():
            errors.append(f"Missing expected {venue} file: {path.relative_to(root)}")
            continue
        paragraphs[venue] = extract_positioning_paragraph(
            path.read_text(encoding="utf-8")
        )
        if not paragraphs[venue]:
            errors.append(
                f"Could not find positioning paragraph in {path.relative_to(root)}"
            )
    if len(set(paragraphs.values())) != len(paragraphs):
        errors.append("TMLR, arXiv, and JOSS positioning paragraphs must be distinct.")
    return errors


def check_scope_limitations(root: Path) -> list[str]:
    errors = []
    for venue, path in {
        "arxiv": root / "paper/arxiv/kalmanorix_negative_result.tex",
        "tmlr": root / "paper/tmlr/main.tex",
        "joss": root / "paper/joss/paper.md",
    }.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        if "limitations" not in text and "scope" not in text:
            errors.append(
                f"{venue} paper lacks a scope/limitations paragraph ({path.relative_to(root)})."
            )
    return errors


def check_joss_overclaim(root: Path) -> list[str]:
    path = root / "paper/joss/paper.md"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").lower()
    bad = ["state-of-the-art", "proves", "outperforms", "superior"]
    allowed = (
        "not",
        "no evidence",
        "inconclusive",
        "do not",
        "unsupported",
        "not a claim of",
    )
    errors: list[str] = []
    for t in bad:
        if t not in text:
            continue
        for line in text.splitlines():
            if t in line and not any(c in line for c in allowed):
                errors.append(
                    f"JOSS overclaim term '{t}' found in {path.relative_to(root)}."
                )
                break
    statuses = load_evidence(root)
    for phrase, claim_id in UNSUPPORTED_CLAIMS.items():
        if phrase.lower() not in text:
            continue
        if statuses.get(claim_id) in SUPPORTED_STATUSES:
            continue
        for line in text.splitlines():
            ll = line.lower()
            if phrase.lower() in ll and not any(
                c in ll for c in ALLOWED_DISCLAIMER_CONTEXT
            ):
                errors.append(
                    "JOSS unsupported claim phrase "
                    f"'{phrase}' found in {path.relative_to(root)} while "
                    f"{claim_id}={statuses.get(claim_id)!r}."
                )
                break
    return errors


def check_arxiv_tmlr_consistency(root: Path) -> list[str]:
    a = root / "paper/arxiv/kalmanorix_negative_result.tex"
    t = root / "paper/tmlr/main.tex"
    if not a.exists() or not t.exists():
        return []
    at = a.read_text(encoding="utf-8").lower()
    tt = t.read_text(encoding="utf-8").lower()
    pos = ["improves retrieval", "outperforms mean", "superior to monolith"]
    neg = ["does not", "not supported", "inconclusive", "negative result"]
    errors = []
    for p in pos:
        a_pos, t_pos = p in at, p in tt
        if a_pos != t_pos:
            errors.append(
                f"Potential contradiction: phrase '{p}' appears in only one of arXiv/TMLR."
            )
    # coarse polarity contradiction
    if any(n in at for n in neg) and any(p in tt for p in pos):
        errors.append(
            "Potential contradiction: arXiv is negative while TMLR contains positive claim language."
        )
    if any(n in tt for n in neg) and any(p in at for p in pos):
        errors.append(
            "Potential contradiction: TMLR is negative while arXiv contains positive claim language."
        )
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = collect_target_files(root)
    statuses = load_evidence(root)
    artifact_text = collect_artifact_text(root)

    errors: list[str] = []
    errors += check_unsupported_claim_phrases(files, statuses, root)
    errors += check_headline_numbers(files, artifact_text, root)
    errors += check_distinct_positioning(root)
    errors += check_joss_overclaim(root)
    errors += check_arxiv_tmlr_consistency(root)
    errors += check_scope_limitations(root)

    if errors:
        print("Publication consistency checks failed:")
        for e in errors:
            print(f" - {e}")
        return 1
    print("Publication consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
