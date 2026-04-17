#!/usr/bin/env python3
"""Adaptive benchmark to decide whether Kalman should be default or conditional."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.registry.datasets import load_dataset
from experiments.registry.evaluation import evaluate_locked
from experiments.registry.fusion import (
    _fuse_and_rank,
    _normalize,
    _resolve_embedding,
)
from experiments.registry.models import build_village
from experiments.registry.runner import DEFAULT_REAL_SPECIALISTS
from kalmanorix.benchmarks import QueryRanking


MODES = ("hard_routing", "mean_fusion", "kalman_fusion")


@dataclass(frozen=True)
class PolicyConfig:
    high_gap: float
    low_gap: float
    agreement_high: float
    agreement_low: float
    spread_high: float
    spread_low: float
    kalman_ambiguity: str


def _ambiguity_bucket(raw: Any) -> str:
    value = "" if raw is None else str(raw).strip().lower()
    if not value:
        return "unknown"
    if "high" in value:
        return "high"
    if "medium" in value or "mid" in value:
        return "medium"
    if "low" in value:
        return "low"
    return value


def _signals_for_query(
    row: dict[str, Any], modules: list[Any]
) -> dict[str, float | int | str]:
    query_text = row["query_text"]
    sigma2 = np.asarray(
        [float(module.sigma2_for(query_text)) for module in modules], dtype=np.float64
    )
    router_scores = 1.0 / np.clip(sigma2, 1e-8, None)
    order = np.argsort(-router_scores)

    top = float(router_scores[order[0]])
    second = float(router_scores[order[1]]) if len(order) > 1 else 0.0
    gap = (top - second) / max(top, 1e-8)

    selected = np.where(router_scores >= (0.85 * top))[0]
    if selected.size == 0:
        selected = np.asarray([int(order[0])])

    qvecs = [_normalize(_resolve_embedding(module, query_text)) for module in modules]
    pair_sims: list[float] = []
    for i, left in enumerate(selected):
        for right in selected[i + 1 :]:
            pair_sims.append(float(np.dot(qvecs[int(left)], qvecs[int(right)])))
    agreement = float(np.mean(pair_sims)) if pair_sims else 1.0

    spread = float(np.std(sigma2[selected])) if selected.size else 0.0

    return {
        "confidence_gap": float(gap),
        "selected_count": int(selected.size),
        "specialist_agreement": agreement,
        "uncertainty_spread": spread,
        "domain_ambiguity_bucket": _ambiguity_bucket(
            row.get("ambiguity_category", row.get("fusion_usefulness_bucket"))
        ),
    }


def _mode_weights(
    mode: str, query_text: str, modules: list[Any], top_k: int
) -> np.ndarray:
    sigma2 = np.asarray(
        [float(module.sigma2_for(query_text)) for module in modules], dtype=np.float64
    )
    router_scores = 1.0 / np.clip(sigma2, 1e-8, None)
    order = np.argsort(-router_scores)
    w = np.zeros(len(modules), dtype=np.float64)

    if mode == "hard_routing":
        w[int(order[0])] = 1.0
        return w
    if mode == "mean_fusion":
        k = min(max(top_k, 1), len(modules))
        idx = order[:k]
        w[idx] = 1.0 / k
        return w

    k = min(max(top_k, 2), len(modules))
    idx = order[:k]
    precision = 1.0 / np.clip(sigma2[idx], 1e-8, None)
    denom = float(np.sum(precision))
    if denom <= 0:
        w[idx] = 1.0 / len(idx)
    else:
        w[idx] = precision / denom
    return w


def _rank_mode(
    *,
    mode: str,
    row: dict[str, Any],
    modules: list[Any],
    top_k: int,
) -> list[str]:
    query_text = row["query_text"]
    weights = _mode_weights(mode, query_text, modules, top_k)
    qvecs = [_resolve_embedding(module, query_text) for module in modules]
    doc_vecs: list[list[np.ndarray]] = [[] for _ in modules]
    for candidate in row["candidate_documents"]:
        doc_text = f"{candidate.get('title', '')} {candidate.get('text', '')}".strip()
        for idx, module in enumerate(modules):
            doc_vecs[idx].append(_resolve_embedding(module, doc_text))
    return _fuse_and_rank(
        query_vecs=qvecs,
        doc_vecs=doc_vecs,
        weights=weights,
        candidates=row["candidate_documents"],
    )


def _select_mode(signals: dict[str, Any], cfg: PolicyConfig) -> str:
    ambig = str(signals["domain_ambiguity_bucket"])
    if (
        signals["selected_count"] <= 1
        and signals["confidence_gap"] >= cfg.high_gap
        and signals["specialist_agreement"] >= cfg.agreement_high
        and signals["uncertainty_spread"] <= cfg.spread_low
        and ambig in {"low", "unknown"}
    ):
        return "hard_routing"

    if (
        signals["confidence_gap"] <= cfg.low_gap
        or signals["specialist_agreement"] <= cfg.agreement_low
        or signals["uncertainty_spread"] >= cfg.spread_high
        or ambig == cfg.kalman_ambiguity
    ):
        return "kalman_fusion"
    return "mean_fusion"


def _reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def _fit_policy(
    val_rows: list[dict[str, Any]],
    modules: list[Any],
    top_k: int,
    min_kalman_frac: float,
) -> tuple[PolicyConfig, dict[str, Any]]:
    feature_map = {
        row["query_id"]: _signals_for_query(row, modules) for row in val_rows
    }
    ranking_map: dict[str, dict[str, list[str]]] = {mode: {} for mode in MODES}
    rr_map: dict[str, dict[str, float]] = {mode: {} for mode in MODES}

    for row in val_rows:
        qid = row["query_id"]
        gt = set(row["ground_truth_relevant_ids"])
        for mode in MODES:
            ranked = _rank_mode(mode=mode, row=row, modules=modules, top_k=top_k)
            ranking_map[mode][qid] = ranked
            rr_map[mode][qid] = _reciprocal_rank(ranked, gt)

    grid: list[PolicyConfig] = []
    for high_gap in (0.35, 0.45, 0.55):
        for low_gap in (0.08, 0.15, 0.25):
            if low_gap >= high_gap:
                continue
            for agreement_high in (0.85, 0.90):
                for agreement_low in (0.55, 0.70):
                    if agreement_low >= agreement_high:
                        continue
                    for spread_high in (0.2, 0.35, 0.5):
                        for spread_low in (0.05, 0.12, 0.2):
                            if spread_low >= spread_high:
                                continue
                            for kalman_ambiguity in ("high", "medium"):
                                grid.append(
                                    PolicyConfig(
                                        high_gap=high_gap,
                                        low_gap=low_gap,
                                        agreement_high=agreement_high,
                                        agreement_low=agreement_low,
                                        spread_high=spread_high,
                                        spread_low=spread_low,
                                        kalman_ambiguity=kalman_ambiguity,
                                    )
                                )

    best_cfg = grid[0]
    best_mrr = -1.0
    best_usage: dict[str, int] = {}

    for cfg in grid:
        rr_values: list[float] = []
        usage = {mode: 0 for mode in MODES}
        for row in val_rows:
            qid = row["query_id"]
            mode = _select_mode(feature_map[qid], cfg)
            usage[mode] += 1
            rr_values.append(rr_map[mode][qid])
        kalman_frac = usage["kalman_fusion"] / max(len(val_rows), 1)
        if kalman_frac < min_kalman_frac:
            continue
        mrr = float(np.mean(rr_values)) if rr_values else 0.0
        if mrr > best_mrr:
            best_mrr = mrr
            best_cfg = cfg
            best_usage = usage

    return best_cfg, {
        "validation_mrr": best_mrr,
        "validation_usage": best_usage,
        "min_kalman_frac": min_kalman_frac,
        "grid_size": len(grid),
    }


def _evaluate(
    rows: list[dict[str, Any]],
    modules: list[Any],
    top_k: int,
    cfg: PolicyConfig,
) -> dict[str, Any]:
    rankings: dict[str, dict[str, QueryRanking]] = {
        "hard_routing": {},
        "mean_fusion": {},
        "kalman_fusion": {},
        "adaptive_fusion_selector": {},
    }
    latencies = {name: {} for name in rankings}
    specialist_counts = {name: {} for name in rankings}

    policy_usage: dict[str, Any] = {}
    bucket_outcomes: dict[str, dict[str, list[float] | dict[str, int]]] = {}

    for row in rows:
        qid = row["query_id"]
        gt = set(row["ground_truth_relevant_ids"])
        signals = _signals_for_query(row, modules)
        bucket = str(signals["domain_ambiguity_bucket"])

        for mode in MODES:
            ranked = _rank_mode(mode=mode, row=row, modules=modules, top_k=top_k)
            rankings[mode][qid] = QueryRanking(doc_ids=tuple(ranked))
            latencies[mode][qid] = 0.0
            weights = _mode_weights(mode, row["query_text"], modules, top_k)
            specialist_counts[mode][qid] = float(np.count_nonzero(weights > 0.0))

        adaptive_mode = _select_mode(signals, cfg)
        adaptive_ranked = rankings[adaptive_mode][qid].doc_ids
        rankings["adaptive_fusion_selector"][qid] = QueryRanking(
            doc_ids=tuple(adaptive_ranked)
        )
        latencies["adaptive_fusion_selector"][qid] = 0.0
        specialist_counts["adaptive_fusion_selector"][qid] = specialist_counts[
            adaptive_mode
        ][qid]

        rr = _reciprocal_rank(list(adaptive_ranked), gt)
        hit1 = 1.0 if adaptive_ranked and adaptive_ranked[0] in gt else 0.0
        bucket_payload = bucket_outcomes.setdefault(
            bucket,
            {"rr": [], "hit1": [], "mode_usage": {mode: 0 for mode in MODES}},
        )
        bucket_payload["rr"].append(rr)
        bucket_payload["hit1"].append(hit1)
        bucket_payload["mode_usage"][adaptive_mode] += 1

        policy_usage[qid] = {"mode": adaptive_mode, "signals": signals}

    report = evaluate_locked(
        rows=rows,
        rankings_by_strategy=rankings,
        latencies_by_strategy=latencies,
        specialist_count_by_strategy=specialist_counts,
    )

    freq: dict[str, int] = {mode: 0 for mode in MODES}
    for entry in policy_usage.values():
        freq[entry["mode"]] += 1

    bucket_summary = {}
    for bucket, payload in bucket_outcomes.items():
        rr_values = payload["rr"]
        hit_values = payload["hit1"]
        bucket_summary[bucket] = {
            "count": len(rr_values),
            "mrr": float(np.mean(rr_values)) if rr_values else 0.0,
            "recall_at_1": float(np.mean(hit_values)) if hit_values else 0.0,
            "mode_usage": payload["mode_usage"],
        }

    return {
        "policy_usage": policy_usage,
        "policy_usage_frequency": freq,
        "bucket_outcomes": bucket_summary,
        "strategy_metrics": {
            "always_mean": report["mean_fusion"]["global_primary"],
            "always_kalman": report["kalman_fusion"]["global_primary"],
            "adaptive": report["adaptive_fusion_selector"]["global_primary"],
            "always_hard_routing": report["hard_routing"]["global_primary"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive fusion-selection benchmark")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet"),
    )
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--max-validation-queries", type=int, default=400)
    parser.add_argument("--max-test-queries", type=int, default=600)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-kalman-fraction", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adaptive_fusion_selection"),
    )
    args = parser.parse_args()

    val_rows = load_dataset(
        kind="mixed_parquet",
        path=args.benchmark_path,
        split=args.validation_split,
        max_queries=args.max_validation_queries,
    )
    test_rows = load_dataset(
        kind="mixed_parquet",
        path=args.benchmark_path,
        split=args.test_split,
        max_queries=args.max_test_queries,
    )

    village = build_village(
        kind="hf_specialists",
        payload=test_rows,
        specialists=DEFAULT_REAL_SPECIALISTS,
        device=args.device,
    )

    cfg, val_fit = _fit_policy(
        val_rows=val_rows,
        modules=village.modules,
        top_k=args.top_k,
        min_kalman_frac=args.min_kalman_fraction,
    )
    test_eval = _evaluate(
        rows=test_rows, modules=village.modules, top_k=args.top_k, cfg=cfg
    )

    summary = {
        "benchmark": {
            "path": str(args.benchmark_path),
            "validation_split": args.validation_split,
            "test_split": args.test_split,
            "max_validation_queries": args.max_validation_queries,
            "max_test_queries": args.max_test_queries,
        },
        "policy": {
            "type": "interpretable_threshold_rules",
            "config": cfg.__dict__,
            "fit_validation": val_fit,
            "strict_split_note": "Threshold tuning uses validation only; final metrics are test-only.",
        },
        "test": test_eval,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Adaptive Fusion-Selection Benchmark",
        "",
        "Question: should Kalman be default or selective?",
        "",
        "## Policy",
        "- Type: interpretable threshold rules tuned on validation split.",
        f"- Tuned config: `{json.dumps(cfg.__dict__, sort_keys=True)}`",
        f"- Validation fit MRR: {val_fit['validation_mrr']:.4f}",
        f"- Validation mode usage: {val_fit['validation_usage']}",
        "- Strict split: validation used only for threshold selection; test used only for final evaluation.",
        "",
        "## Test strategy comparison",
    ]

    for name, metrics in summary["test"]["strategy_metrics"].items():
        lines.append(
            f"- {name}: mrr={metrics['mrr']['mean']:.4f}, recall@1={metrics['recall@1']['mean']:.4f}, recall@5={metrics['recall@5']['mean']:.4f}"
        )

    total = max(sum(test_eval["policy_usage_frequency"].values()), 1)
    lines.extend(["", "## Adaptive policy usage frequency (test)"])
    for mode, count in test_eval["policy_usage_frequency"].items():
        lines.append(f"- {mode}: {count} ({count / total:.1%})")

    lines.extend(["", "## Per-bucket outcomes (test)"])
    for bucket, payload in sorted(test_eval["bucket_outcomes"].items()):
        lines.append(
            f"- {bucket}: n={payload['count']}, mrr={payload['mrr']:.4f}, recall@1={payload['recall_at_1']:.4f}, mode_usage={payload['mode_usage']}"
        )

    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(test_eval["policy_usage_frequency"], indent=2))


if __name__ == "__main__":
    main()
