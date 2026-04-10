"""Held-out uncertainty calibration pipeline for scalar sigma2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kalmanorix.panoramix import KalmanorixFuser, MeanFuser, Panoramix
from kalmanorix.scout import ScoutRouter
from kalmanorix.toy_corpus import build_toy_corpus
from kalmanorix.uncertainty import UncertaintyMethodConfig, create_uncertainty_method
from kalmanorix.uncertainty_calibration import (
    CalibratorName,
    CalibrationFit,
    apply_calibrator_to_sigma2_fn,
    fit_scalar_calibrator,
    reliability_summary,
    uncertainty_rank_correlation,
)
from kalmanorix.village import SEF, Village


@dataclass(frozen=True)
class QueryEvaluation:
    query: str
    target_doc: int
    rank: int
    reciprocal_rank: float
    hit_at_5: float
    relevant_distance: float
    relevant_score: float


def _token_embedder(vocab: list[str], boost: tuple[str, str]) -> Callable[[str], np.ndarray]:
    to_idx = {w: i for i, w in enumerate(vocab)}
    boost_tokens = set(boost)

    def _embed(text: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float64)
        lower = text.lower()
        for token, idx in to_idx.items():
            if token in lower:
                v[idx] += 1.0
        if any(tok in lower for tok in boost_tokens):
            v += 0.1
        n = np.linalg.norm(v)
        if n > 1e-12:
            v = v / n
        return v

    return _embed


def _build_village(method: str) -> tuple[Village, list[str], list[int]]:
    corpus = build_toy_corpus(british_spelling=False)
    docs = corpus.docs
    queries = [q for q, _ in corpus.queries]
    targets = [t for _, t in corpus.queries]

    vocab = [
        "battery", "charging", "cpu", "gpu", "camera", "usb", "thermal",
        "braise", "simmer", "sauce", "oven", "stew", "cook", "recipe",
    ]
    tech_embed = _token_embedder(vocab, ("battery", "gpu"))
    cook_embed = _token_embedder(vocab, ("braise", "simmer"))

    tech_cal = [d for d in docs if any(k in d.lower() for k in ("battery", "gpu", "cpu", "camera", "usb"))]
    cook_cal = [d for d in docs if any(k in d.lower() for k in ("braise", "simmer", "oven", "sauce", "stew"))]

    tech_sigma = create_uncertainty_method(
        config=UncertaintyMethodConfig(method=method), embed=tech_embed, calibration_texts=tech_cal
    )
    cook_sigma = create_uncertainty_method(
        config=UncertaintyMethodConfig(method=method), embed=cook_embed, calibration_texts=cook_cal
    )

    village = Village(
        [
            SEF(name="tech", embed=tech_embed, sigma2=tech_sigma, embedding_dimension=len(vocab)),
            SEF(name="cook", embed=cook_embed, sigma2=cook_sigma, embedding_dimension=len(vocab)),
        ]
    )
    return village, queries, targets


def _evaluate_specialist(module: SEF, queries: list[str], targets: list[int], docs: list[str]) -> list[QueryEvaluation]:
    doc_embs = np.stack([module.embed(d) for d in docs], axis=0)
    results: list[QueryEvaluation] = []
    for q, target in zip(queries, targets):
        q_emb = module.embed(q)
        scores = doc_embs @ q_emb
        ranked = np.argsort(-scores)
        rank = int(np.where(ranked == target)[0][0]) + 1
        rr = 1.0 / rank
        hit5 = 1.0 if target in ranked[:5] else 0.0
        rel_dist = float(np.linalg.norm(q_emb - doc_embs[target]))
        rel_score = float(scores[target])
        results.append(QueryEvaluation(q, int(target), rank, rr, hit5, rel_dist, rel_score))
    return results


def _target_from_eval(row: QueryEvaluation, objective: str, max_rank: int) -> float:
    if objective == "rank_error_proxy":
        return float((row.rank - 1) / max(max_rank - 1, 1))
    if objective == "topk_miss_indicator":
        return float(1.0 - row.hit_at_5)
    if objective == "distance_to_relevant_doc_centroid":
        return float(row.relevant_distance)
    if objective == "score_quality_residual":
        return float(abs(row.relevant_score - row.reciprocal_rank))
    raise ValueError(f"Unknown objective: {objective}")


def _split_indices(n: int) -> dict[str, list[int]]:
    train_end = max(1, int(0.5 * n))
    val_end = max(train_end + 1, int(0.75 * n))
    val_end = min(val_end, n)
    return {
        "train": list(range(0, train_end)),
        "validation": list(range(train_end, val_end)),
        "test": list(range(val_end, n)),
    }


def _eval_fusers(village: Village, queries: list[str], targets: list[int], docs: list[str], split: list[int]) -> dict[str, float]:
    scout = ScoutRouter(mode="all")
    kalman = Panoramix(fuser=KalmanorixFuser())
    mean = Panoramix(fuser=MeanFuser())
    doc_emb = np.stack([village.modules[0].embed(d) for d in docs], axis=0)

    def _mrr(pan: Panoramix) -> float:
        vals = []
        for i in split:
            potion = pan.brew(queries[i], village=village, scout=scout)
            ranked = np.argsort(-(doc_emb @ potion.vector))
            rank = int(np.where(ranked == targets[i])[0][0]) + 1
            vals.append(1.0 / rank)
        return float(np.mean(vals)) if vals else 0.0

    kalman_mrr = _mrr(kalman)
    mean_mrr = _mrr(mean)
    return {"kalman_mrr": kalman_mrr, "mean_mrr": mean_mrr, "delta": kalman_mrr - mean_mrr}


def _audit_sigma2_paths(repo_root: Path) -> dict[str, list[str]]:
    produced = [
        "src/kalmanorix/uncertainty.py",
        "src/kalmanorix/village.py",
        "src/kalmanorix/models/sef.py",
    ]
    consumed = [
        "src/kalmanorix/panoramix.py",
        "src/kalmanorix/kalman_engine/fuser.py",
        "src/kalmanorix/calibration.py",
        "src/kalmanorix/benchmarks/uncertainty_ablation.py",
    ]
    return {
        "produced_existing": [p for p in produced if (repo_root / p).exists()],
        "consumed_existing": [p for p in consumed if (repo_root / p).exists()],
    }


def run_uncertainty_calibration(
    output_dir: Path,
    sigma2_method: str = "centroid_distance_sigma2",
    objective: str = "rank_error_proxy",
    calibrators: tuple[CalibratorName, ...] = (
        "affine",
        "temperature",
        "isotonic",
        "piecewise_monotonic",
    ),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]

    village, queries, targets = _build_village(sigma2_method)
    docs = build_toy_corpus(british_spelling=False).docs
    split = _split_indices(len(queries))

    leakage_checks: list[dict[str, Any]] = []
    cal_records: list[dict[str, Any]] = []
    fitted_by_module: dict[str, CalibrationFit] = {}

    for module in village.modules:
        evals = _evaluate_specialist(module, queries, targets, docs)
        max_rank = len(docs)

        val_idx = split["validation"]
        val_sigma = np.array([module.sigma2_for(queries[i]) for i in val_idx], dtype=np.float64)
        val_target = np.array([_target_from_eval(evals[i], objective, max_rank) for i in val_idx], dtype=np.float64)

        leakage_checks.append(
            {
                "module": module.name,
                "train_indices": split["train"],
                "validation_indices": val_idx,
                "intersection": sorted(set(split["train"]).intersection(val_idx)),
            }
        )

        best_fit: CalibrationFit | None = None
        for method in calibrators:
            fit = fit_scalar_calibrator(val_sigma, val_target, method)
            cal_records.append(
                {
                    "module": module.name,
                    "calibrator": method,
                    "objective_mse": fit.objective_mse,
                    "used_fallback": fit.used_fallback,
                    "n_train": fit.n_train,
                }
            )
            if best_fit is None or fit.objective_mse < best_fit.objective_mse:
                best_fit = fit

        assert best_fit is not None
        fitted_by_module[module.name] = best_fit
        best_fit.calibrator.to_json(output_dir / f"calibrator_{module.name}.json")

    # Pre/Post diagnostics on test
    test_idx = split["test"]
    diagnostics = {}
    calibrated_modules = []
    for module in village.modules:
        fit = fitted_by_module[module.name]
        calibrated_modules.append(
            SEF(
                name=module.name,
                embed=module.embed,
                sigma2=apply_calibrator_to_sigma2_fn(module.sigma2_for, fit.calibrator),
                embedding_dimension=module.embedding_dimension,
            )
        )

        evals = _evaluate_specialist(module, queries, targets, docs)
        max_rank = len(docs)
        raw = np.array([module.sigma2_for(queries[i]) for i in test_idx], dtype=np.float64)
        target = np.array([_target_from_eval(evals[i], objective, max_rank) for i in test_idx], dtype=np.float64)
        post = fit.calibrator.transform(raw)
        diagnostics[module.name] = {
            "pre": {
                "reliability": asdict(reliability_summary(raw, target)),
                "spearman": uncertainty_rank_correlation(raw, target),
            },
            "post": {
                "reliability": asdict(reliability_summary(post, target)),
                "spearman": uncertainty_rank_correlation(post, target),
            },
        }

    pre_bench = _eval_fusers(village, queries, targets, docs, test_idx)
    post_bench = _eval_fusers(Village(calibrated_modules), queries, targets, docs, test_idx)

    summary = {
        "objective": objective,
        "sigma2_method": sigma2_method,
        "split": split,
        "leakage_checks": leakage_checks,
        "calibration_candidates": cal_records,
        "selected_calibrators": {
            m: {"name": fit.calibrator.name, "mse": fit.objective_mse, "fallback": fit.used_fallback}
            for m, fit in fitted_by_module.items()
        },
        "uncertainty_diagnostics": diagnostics,
        "benchmark_delta": {
            "pre": pre_bench,
            "post": post_bench,
            "delta_change": post_bench["delta"] - pre_bench["delta"],
        },
        "sigma2_path_audit": _audit_sigma2_paths(repo_root),
        "notes": "Calibration was fit on validation split only and applied to held-out test split.",
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "sigma2_audit.json").write_text(json.dumps(summary["sigma2_path_audit"], indent=2), encoding="utf-8")
    return summary
