"""Specialist diversity diagnostics for adaptive fusion analysis."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from math import log
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_score(ranking: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranking[:10], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def generate_specialist_diversity_artifacts(
    details: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    ql = details.get("query_level", {})
    query_meta = ql.get("query_metadata", {})
    query_ids = sorted(query_meta.keys())
    if not query_ids:
        return {"status": "no_query_metadata"}

    specialists = sorted(
        {
            s
            for qid in query_ids
            for s in query_meta.get(qid, {}).get("sigma2_by_specialist", {}).keys()
        }
    )

    sigma2_matrix = np.asarray(
        [
            [
                float(
                    query_meta.get(qid, {}).get("sigma2_by_specialist", {}).get(s, 0.0)
                )
                for s in specialists
            ]
            for qid in query_ids
        ],
        dtype=float,
    )
    inv_sigma = 1.0 / np.clip(sigma2_matrix, 1e-8, None)

    cos_pairs: dict[str, float] = {}
    corr_pairs: dict[str, float] = {}
    for i, j in combinations(range(len(specialists)), 2):
        a = inv_sigma[:, i]
        b = inv_sigma[:, j]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        corr = _safe_corr(a, b)
        key = f"{specialists[i]}__{specialists[j]}"
        cos_pairs[key] = cos
        corr_pairs[key] = corr

    disagreement = {
        qid: float(query_meta.get(qid, {}).get("specialist_disagreement", 0.0))
        for qid in query_ids
    }

    # Effective sample size from normalized precision weights
    ess_by_query: dict[str, float] = {}
    for row, qid in zip(inv_sigma, query_ids):
        p = row / max(float(np.sum(row)), 1e-12)
        ess_by_query[qid] = float(1.0 / max(float(np.sum(p**2)), 1e-12))

    rankings = ql.get("rankings", {})
    ground_truth = {qid: set(v) for qid, v in ql.get("ground_truth", {}).items()}
    mean_rank = rankings.get("mean", {})
    kalman_rank = rankings.get("kalman", {})
    hard_rank = rankings.get("router_only_top1", {})

    scatter_rows: list[dict[str, float | str]] = []
    delta_vals: list[float] = []
    diss_vals: list[float] = []
    for qid in sorted(
        set(mean_rank) & set(kalman_rank) & set(hard_rank) & set(ground_truth)
    ):
        s_mean = _rank_score(list(mean_rank[qid]), ground_truth[qid])
        s_kalman = _rank_score(list(kalman_rank[qid]), ground_truth[qid])
        s_hard = _rank_score(list(hard_rank[qid]), ground_truth[qid])
        delta = float(s_kalman - s_mean)
        routing_delta = float(s_hard - s_mean)
        d = float(disagreement.get(qid, 0.0))
        scatter_rows.append(
            {
                "query_id": qid,
                "specialist_disagreement": d,
                "kalman_minus_mean_mrr10": delta,
                "hard_minus_mean_mrr10": routing_delta,
                "ess": float(ess_by_query.get(qid, 1.0)),
                "uncertainty_spread": float(
                    query_meta.get(qid, {}).get("uncertainty_spread", 0.0)
                ),
            }
        )
        delta_vals.append(delta)
        diss_vals.append(d)

    mi_proxy = 0.0
    if delta_vals:
        d_bin = np.asarray(diss_vals, dtype=float) >= float(np.median(diss_vals))
        g_bin = np.asarray(delta_vals, dtype=float) > 0.0
        joint = np.zeros((2, 2), dtype=float)
        for a, b in zip(d_bin.astype(int), g_bin.astype(int)):
            joint[a, b] += 1.0
        joint /= max(float(np.sum(joint)), 1.0)
        pa = np.sum(joint, axis=1)
        pb = np.sum(joint, axis=0)
        for i in range(2):
            for j in range(2):
                if joint[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                    mi_proxy += float(
                        joint[i, j] * log(joint[i, j] / (pa[i] * pb[j] + 1e-12) + 1e-12)
                    )

    out_json = {
        "specialists": specialists,
        "cosine_similarity_precision_profiles": cos_pairs,
        "pairwise_residual_correlation": corr_pairs,
        "disagreement_distribution": {
            "mean": float(mean(diss_vals)) if diss_vals else 0.0,
            "median": float(np.median(diss_vals)) if diss_vals else 0.0,
            "min": float(min(diss_vals)) if diss_vals else 0.0,
            "max": float(max(diss_vals)) if diss_vals else 0.0,
        },
        "effective_sample_size": {
            "mean": float(mean(ess_by_query.values())) if ess_by_query else 0.0,
            "median": float(np.median(list(ess_by_query.values())))
            if ess_by_query
            else 0.0,
            "per_query": ess_by_query,
        },
        "mutual_information_proxy_disagreement_vs_gain": mi_proxy,
        "relationships": {
            "disagreement_vs_retrieval_improvement_corr": _safe_corr(
                np.asarray(diss_vals), np.asarray(delta_vals)
            )
            if delta_vals
            else 0.0,
            "disagreement_vs_uncertainty_usefulness_corr": _safe_corr(
                np.asarray(diss_vals),
                np.asarray(
                    [float(r["uncertainty_spread"]) for r in scatter_rows], dtype=float
                ),
            )
            if scatter_rows
            else 0.0,
            "routing_redundancy_reduction_vs_fusion_corr": _safe_corr(
                np.asarray(
                    [float(r["specialist_disagreement"]) for r in scatter_rows],
                    dtype=float,
                ),
                np.asarray(
                    [
                        float(r["hard_minus_mean_mrr10"])
                        - float(r["kalman_minus_mean_mrr10"])
                        for r in scatter_rows
                    ],
                    dtype=float,
                ),
            )
            if scatter_rows
            else 0.0,
        },
        "interpretation": "If specialists mostly agree, adaptive uncertainty weighting cannot contribute much beyond averaging.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "specialist_diversity.json").write_text(
        json.dumps(out_json, indent=2), encoding="utf-8"
    )

    # Heatmap table (correlation over precision profiles)
    corr_m = np.corrcoef(inv_sigma.T) if inv_sigma.shape[1] > 0 else np.zeros((0, 0))
    lines = [
        "% Specialist correlation heatmap matrix",
        "\\begin{tabular}{l" + "r" * len(specialists) + "}",
        "Specialist & " + " & ".join(specialists) + " \\\\",
        "\\hline",
    ]
    for i, name in enumerate(specialists):
        vals = " & ".join(f"{float(corr_m[i, j]):.3f}" for j in range(len(specialists)))
        lines.append(f"{name} & {vals} \\\\")
    lines.append("\\end{tabular}")
    (output_dir / "specialist_correlation_heatmap.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    with (output_dir / "disagreement_vs_gain_scatter.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(scatter_rows[0].keys())
            if scatter_rows
            else [
                "query_id",
                "specialist_disagreement",
                "kalman_minus_mean_mrr10",
                "hard_minus_mean_mrr10",
                "ess",
                "uncertainty_spread",
            ],
        )
        writer.writeheader()
        writer.writerows(scatter_rows)

    md = [
        "# Effective Sample Size Analysis",
        "",
        f"- Queries analyzed: {len(query_ids)}",
        f"- Specialists: {len(specialists)}",
        f"- Mean ESS: {out_json['effective_sample_size']['mean']:.3f}",
        f"- Median ESS: {out_json['effective_sample_size']['median']:.3f}",
        "",
        "## Interpretation",
        "If specialists mostly agree, adaptive uncertainty weighting cannot contribute much beyond averaging.",
    ]
    (output_dir / "effective_sample_size_analysis.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return out_json
