#!/usr/bin/env python3
"""Run the primary mixed-domain retrieval benchmark on real BEIR-derived data.

Primary path:
- Real datasets with qrels from ``benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet``
- Real domain specialists (general QA, biomedical, finance)
- Locked metrics from ``kalmanorix.benchmarks.evaluation_protocol``

This is the headline validation path for fusion quality.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village
from kalmanorix.benchmarks import QueryRanking, evaluate_locked_protocol
from kalmanorix.uncertainty import CentroidDistanceSigma2


@dataclass(frozen=True)
class SpecialistSpec:
    """Configuration for a real specialist encoder."""

    name: str
    domain: str
    model_name: str


DEFAULT_SPECIALISTS: tuple[SpecialistSpec, ...] = (
    SpecialistSpec(
        name="general_qa",
        domain="general_qa",
        model_name="sentence-transformers/all-mpnet-base-v2",
    ),
    SpecialistSpec(
        name="biomedical",
        domain="biomedical",
        model_name="emilyalsentzer/Bio_ClinicalBERT",
    ),
    SpecialistSpec(
        name="finance",
        domain="finance",
        model_name="ProsusAI/finbert",
    ),
)


class HFMeanPoolEmbedder:
    """HuggingFace encoder with masked mean pooling + L2 normalization."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        masked = hidden * mask
        denom = torch.clamp(mask.sum(dim=1), min=1)
        pooled = masked.sum(dim=1) / denom
        vec = pooled[0].detach().cpu().numpy().astype(np.float64)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec


def _read_mixed_benchmark(path: Path, split: str, max_queries: int | None) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    records = table.to_pylist()
    filtered = [row for row in records if row.get("split") == split]
    if max_queries is not None:
        filtered = filtered[:max_queries]
    if not filtered:
        raise ValueError(f"No rows found for split='{split}' in {path}")
    return filtered


def _domain_calibration_texts(rows: list[dict[str, Any]], domain: str, n: int = 128) -> list[str]:
    texts = [row["query_text"] for row in rows if row.get("domain_label") == domain]
    if len(texts) >= n:
        return texts[:n]
    if not texts:
        return [
            f"{domain} terminology and retrieval calibration",
            f"{domain} specialist embedding calibration text",
        ]
    repeats = (n // len(texts)) + 1
    return (texts * repeats)[:n]


def _build_village(rows: list[dict[str, Any]], device: str) -> Village:
    modules: list[SEF] = []
    for spec in DEFAULT_SPECIALISTS:
        embedder = HFMeanPoolEmbedder(spec.model_name, device=device)
        calibration = _domain_calibration_texts(rows, spec.domain)
        sigma2 = CentroidDistanceSigma2.from_calibration(
            embed=embedder,
            calibration_texts=calibration,
            base_sigma2=0.15,
            scale=2.5,
        )
        modules.append(
            SEF(
                name=spec.name,
                embed=embedder,
                sigma2=sigma2,
                meta={"domain": spec.domain, "model_name": spec.model_name},
            )
        )
    return Village(modules)


def _rank_query(
    query_text: str,
    candidates: list[dict[str, Any]],
    village: Village,
    scout: ScoutRouter,
    pan: Panoramix,
) -> tuple[list[str], float]:
    start = time.perf_counter()
    q_potion = pan.brew(query_text, village=village, scout=scout)
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        doc_text = f"{cand.get('title', '')} {cand.get('text', '')}".strip()
        d_potion = pan.brew(doc_text, village=village, scout=scout)
        score = float(q_potion.vector @ d_potion.vector)
        scored.append((cand["doc_id"], score))
    ranked = [doc_id for doc_id, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
    return ranked, (time.perf_counter() - start) * 1000.0


def run_real_benchmark(
    benchmark_path: Path,
    split: str,
    max_queries: int | None,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    rows = _read_mixed_benchmark(benchmark_path, split=split, max_queries=max_queries)
    village = _build_village(rows, device=device)

    strategies = {
        "mean": (ScoutRouter(mode="all"), Panoramix(fuser=MeanFuser())),
        "kalman": (ScoutRouter(mode="all"), Panoramix(fuser=KalmanorixFuser())),
    }

    qrels = {r["query_id"]: {doc_id: 1.0 for doc_id in r["ground_truth_relevant_ids"]} for r in rows}
    query_domains = {r["query_id"]: r["domain_label"] for r in rows}

    reports: dict[str, Any] = {}
    for name, (scout, pan) in strategies.items():
        rankings: dict[str, QueryRanking] = {}
        latency_ms: dict[str, float] = {}
        for row in rows:
            ranked_ids, elapsed_ms = _rank_query(
                query_text=row["query_text"],
                candidates=row["candidate_documents"],
                village=village,
                scout=scout,
                pan=pan,
            )
            rankings[row["query_id"]] = QueryRanking(doc_ids=tuple(ranked_ids))
            latency_ms[row["query_id"]] = elapsed_ms

        report = evaluate_locked_protocol(
            rankings=rankings,
            qrels=qrels,
            query_domains=query_domains,
            latency_ms=latency_ms,
        )
        reports[name] = {
            "protocol_version": report.protocol_version,
            "protocol_sha256": report.protocol_sha256,
            "num_queries": report.num_queries,
            "global_primary": {
                metric: {"mean": stats.mean, "median": stats.median}
                for metric, stats in report.global_primary.items()
            },
            "global_secondary": {
                metric: {"mean": stats.mean, "median": stats.median}
                for metric, stats in report.global_secondary.items()
            },
            "per_domain_primary": {
                domain: {
                    metric: {"mean": stats.mean, "median": stats.median}
                    for metric, stats in metrics.items()
                }
                for domain, metrics in report.per_domain_primary.items()
            },
        }

    summary = {
        "benchmark_path": str(benchmark_path),
        "split": split,
        "max_queries": max_queries,
        "specialists": [spec.__dict__ for spec in DEFAULT_SPECIALISTS],
        "results": reports,
        "delta_kalman_minus_mean": {
            metric: (
                reports["kalman"]["global_primary"][metric]["mean"]
                - reports["mean"]["global_primary"][metric]["mean"]
            )
            for metric in reports["mean"]["global_primary"]
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run real mixed-domain retrieval benchmark")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_mixed_benchmark/real_benchmark_summary.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = run_real_benchmark(
        benchmark_path=args.benchmark_path,
        split=args.split,
        max_queries=args.max_queries,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(summary["delta_kalman_minus_mean"], indent=2))


if __name__ == "__main__":
    main()
