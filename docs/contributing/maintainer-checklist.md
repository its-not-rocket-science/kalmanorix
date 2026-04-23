# Maintainer Release Checklist

Use this checklist before cutting a release or merging major benchmark/result updates.

## 1) Regenerate artifacts deterministically

- Re-run canonical benchmark artifacts:
  - `PYTHONPATH=src python experiments/run_canonical_benchmark.py --benchmark-path benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet --split test --max-queries 1800 --output-dir results/canonical_benchmark_v3`
- Re-run routing evaluation reference artifact:
  - `kalmanorix-eval-routing --dataset datasets/routing_eval/small_routing_eval_v1.json --output results/routing_eval/small_routing_eval_v1_report.json --markdown-output results/routing_eval/small_routing_eval_v1_report.md --mode semantic --semantic-threshold 0.7 --semantic-thresholds 0.5,0.6,0.7,0.8 --quality-tolerance 0.0`
- Confirm every touched `results/<track>/` directory contains:
  - `README.md` or `report.md`
  - a reproducibility command in documentation
  - `summary.json`

## 2) Validate evidence-state honesty

- Re-check top-level evidence statements in:
  - `README.md`
  - `ROADMAP.md`
  - `docs/research/results.md`
- Ensure claims match committed artifacts and not local-only runs.
- Preserve provenance: keep old artifact folders; add new versioned folders instead of overwriting historical claims.

## 3) Prevent overclaiming

- Keep wording claim-safe:
  - "supported in current artifacts" instead of "proven generally"
  - "inconclusive" when significance/power thresholds are not met
- Ensure synthetic/debug runs are labeled as non-claim evidence.
- Require explicit uncertainty language for mixed/null/regression outcomes.

## 4) Final verification commands

- `python -m pytest`
- `python scripts/check_no_hardcoded_results.py`
- `python scripts/check_packaging_metadata.py`
