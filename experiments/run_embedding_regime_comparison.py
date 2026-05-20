#!/usr/bin/env python3
"""Run canonical benchmark v3 across embedding regimes and synthesize comparison artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_REGIMES: dict[str, list[dict[str, str]]] = {
    "sentence_bert": [
        {
            "name": "general_qa",
            "domain": "general_qa",
            "model_name": "sentence-transformers/all-mpnet-base-v2",
        },
        {
            "name": "biomedical",
            "domain": "biomedical",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "name": "finance",
            "domain": "finance",
            "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        },
    ],
    "e5": [
        {
            "name": "general_qa",
            "domain": "general_qa",
            "model_name": "intfloat/e5-base-v2",
        },
        {
            "name": "biomedical",
            "domain": "biomedical",
            "model_name": "intfloat/e5-small-v2",
        },
        {
            "name": "finance",
            "domain": "finance",
            "model_name": "intfloat/multilingual-e5-base",
        },
    ],
    "bge": [
        {
            "name": "general_qa",
            "domain": "general_qa",
            "model_name": "BAAI/bge-base-en-v1.5",
        },
        {
            "name": "biomedical",
            "domain": "biomedical",
            "model_name": "BAAI/bge-small-en-v1.5",
        },
        {
            "name": "finance",
            "domain": "finance",
            "model_name": "BAAI/bge-large-en-v1.5",
        },
    ],
    "lightweight_local": [
        {
            "name": "general_qa",
            "domain": "general_qa",
            "model_name": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        },
        {
            "name": "biomedical",
            "domain": "biomedical",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "name": "finance",
            "domain": "finance",
            "model_name": "sentence-transformers/paraphrase-albert-small-v2",
        },
    ],
}


def _write_regime_config(path: Path, specialists: list[dict[str, str]]) -> None:
    payload = {"specialists": specialists}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_regime(
    args: argparse.Namespace, regime: str, specialists: list[dict[str, str]]
) -> Path:
    output_dir = args.output_root / regime
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "specialists.json"
    _write_regime_config(config_path, specialists)

    command = [
        "python",
        "experiments/run_canonical_benchmark.py",
        "--seed",
        str(args.seed),
        "--max-queries",
        str(args.max_queries),
        "--num-resamples",
        str(args.num_resamples),
        "--output-dir",
        str(output_dir),
    ]
    if args.max_candidates is not None:
        command.extend(["--max-candidates", str(args.max_candidates)])
    command.extend(["--specialists-config", str(config_path)])
    subprocess.run(command, check=True)
    return output_dir


def _effect(summary: dict[str, Any], baseline: str) -> dict[str, Any]:
    comp = summary["paired_statistics"].get(f"kalman_vs_{baseline}", {})
    return {
        "baseline": baseline,
        "mean_difference": comp.get("mean_difference"),
        "ci_lower": comp.get("bootstrap_ci", {}).get("lower"),
        "ci_upper": comp.get("bootstrap_ci", {}).get("upper"),
        "adjusted_p_value": comp.get("adjusted_p_value"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/embedding_regime_comparison")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=1800)
    parser.add_argument("--num-resamples", type=int, default=5000)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--include-api-regime", action="store_true")
    args = parser.parse_args()

    regimes = dict(DEFAULT_REGIMES)
    if args.include_api_regime:
        regimes["api_reproducible"] = [
            {
                "name": "general_qa",
                "domain": "general_qa",
                "model_name": "text-embedding-3-large",
            },
            {
                "name": "biomedical",
                "domain": "biomedical",
                "model_name": "text-embedding-3-small",
            },
            {
                "name": "finance",
                "domain": "finance",
                "model_name": "text-embedding-3-small",
            },
        ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for regime, specialists in regimes.items():
        out_dir = _run_regime(args, regime, specialists)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        claim_gate = json.loads(
            (out_dir / "claim_gate.json").read_text(encoding="utf-8")
        )
        results.append(
            {
                "regime": regime,
                "output_dir": str(out_dir),
                "seed": args.seed,
                "max_candidates": args.max_candidates,
                "paired_query_table": claim_gate.get("paired_query_table"),
                "kalman_decision": summary["decision"]["kalman_vs_mean"]["verdict"],
                "uncertainty_stability": summary["sample_size_adequacy"][
                    "uncertainty_calibration"
                ],
                "effects": [
                    _effect(summary, "mean"),
                    _effect(summary, "weighted_mean"),
                    _effect(summary, "router_only_top1"),
                    _effect(summary, "learned_linear_combiner"),
                ],
            }
        )

    comparison = {
        "seed": args.seed,
        "max_queries": args.max_queries,
        "regimes": results,
    }
    (args.output_root / "embedding_regime_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"Regime & $\Delta$NDCG@10 vs Mean & 95\% CI low & 95\% CI high & Kalman verdict \\",
        r"\hline",
    ]
    for row in results:
        eff = next(e for e in row["effects"] if e["baseline"] == "mean")
        lines.append(
            f"{row['regime']} & {eff['mean_difference']:.4f} & {eff['ci_lower']:.4f} & {eff['ci_upper']:.4f} & {row['kalman_decision']} \\"
        )
    lines.append(r"\end{tabular}")
    (args.output_root / "embedding_regime_comparison.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    summary_lines = ["# Cross-regime Summary", "", "## Synthesis", ""]
    consistent_failure = all(r["kalman_decision"] != "supported" for r in results)
    summary_lines.append(
        f"- Did Kalman fail consistently? {'Yes' if consistent_failure else 'No'}."
    )
    summary_lines.append(
        "- Were uncertainty estimates stable? See per-regime uncertainty calibration adequacy in JSON."
    )
    summary_lines.append(
        "- Did routing help more than fusion? Compare Kalman vs `router_only_top1` and vs mean effects per regime."
    )
    summary_lines.append(
        "- Were any gains practically meaningful? Treat >0.02 NDCG@10 and CI above zero as practically meaningful."
    )
    (args.output_root / "cross_regime_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
