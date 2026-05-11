#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "publication_bundle"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def _needs_bibtex(aux_path: Path) -> bool:
    if not aux_path.exists():
        return False
    aux_text = aux_path.read_text(encoding="utf-8", errors="ignore")
    citation_markers = ("\\citation", "\\abx")
    return any(marker in aux_text for marker in citation_markers)


def run_tex_build(workdir: Path, stem: str) -> Path:
    run(["pdflatex", f"{stem}.tex"], cwd=workdir)
    aux_path = workdir / f"{stem}.aux"
    if _needs_bibtex(aux_path):
        run(["bibtex", stem], cwd=workdir)
    else:
        print(f"$ bibtex {stem} (skipped: no bibliography markers in {aux_path.name})")
    run(["pdflatex", f"{stem}.tex"], cwd=workdir)
    run(["pdflatex", f"{stem}.tex"], cwd=workdir)
    pdf = workdir / f"{stem}.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"Expected PDF not produced: {pdf}")
    return pdf


def validate_joss_metadata() -> None:
    import yaml

    paper = ROOT / "paper/joss/paper.md"
    text = paper.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("paper/joss/paper.md is missing YAML front matter.")

    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("paper/joss/paper.md front matter is not closed with '---'.")

    metadata = yaml.safe_load(text[4:end])
    required = ["title", "tags", "authors", "affiliations", "date", "bibliography"]
    missing = [
        key
        for key in required
        if key not in metadata or metadata[key] in (None, "", [])
    ]
    if missing:
        raise ValueError(f"Missing required JOSS metadata fields: {', '.join(missing)}")

    if not isinstance(metadata["authors"], list) or not metadata["authors"]:
        raise ValueError("JOSS metadata 'authors' must be a non-empty list.")

    bib_path = ROOT / "paper/joss" / str(metadata["bibliography"])
    if not bib_path.exists():
        raise FileNotFoundError(f"JOSS bibliography not found: {bib_path}")


def write_readme(fast: bool, tmlr_pdf: Path, arxiv_pdf: Path) -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""# Publication bundle

Mode: {"fast" if fast else "full"}

## Generated files

- `results/evidence_registry.json`
- `paper/tmlr/tables/main_results.tex`
- `paper/tmlr/tables/statistical_tests.tex`
- `paper/arxiv/tables/main_results.tex`
- `paper/arxiv/tables/statistical_tests.tex`
- `paper/joss/results_summary.md`
- `{tmlr_pdf.relative_to(ROOT)}`
- `{arxiv_pdf.relative_to(ROOT)}`

## Source artifacts

- `results/canonical_benchmark_v3_fast_1200/summary.json`
- `results/canonical_benchmark_v3_fast_1200/report.md`
- `results/canonical_benchmark_v2/summary.json`
- `results/canonical_benchmark_v3/summary.json`

## Known limitations

- TMLR builds require `paper/tmlr/tmlr.sty` and `paper/tmlr/tmlr.bst` to be available.
- PDF builds require `pdflatex` and `bibtex` in PATH.
- Fast mode uses committed artifacts and skips expensive benchmark regeneration.

## Submission readiness checklist

- [ ] Evidence registry rebuilt from current committed artifacts.
- [ ] Publication claim consistency check passes.
- [ ] LaTeX tables regenerated from canonical benchmark artifacts.
- [ ] TMLR PDF builds successfully.
- [ ] arXiv PDF builds successfully.
- [ ] JOSS metadata front matter validates.
- [ ] Venue-specific manual checks completed (anonymity, references, figures, package completeness).
"""
    (BUNDLE_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build publication bundle artifacts.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip expensive benchmark regeneration and use committed artifacts only.",
    )
    args = parser.parse_args()

    if not args.fast:
        run(["python", "scripts/build_mixed_benchmark.py"])

    run(["python", "scripts/build_evidence_registry.py"])
    run(["python", "scripts/check_publication_consistency.py"])
    run(["python", "scripts/export_paper_tables.py"])

    tmlr_pdf = run_tex_build(ROOT / "paper/tmlr", "main")
    arxiv_pdf = run_tex_build(ROOT / "paper/arxiv", "kalmanorix_negative_result")

    validate_joss_metadata()
    write_readme(args.fast, tmlr_pdf, arxiv_pdf)
    print(f"Wrote {BUNDLE_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
