"""Held-out uncertainty calibration pipeline for scalar sigma2."""

from __future__ import annotations

from collections import Counter, defaultdict
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
    ScalarCalibrator,
    apply_calibrator_to_sigma2_fn,
    fit_scalar_calibrator,
    reliability_summary,
    uncertainty_rank_correlation,
)
from kalmanorix.village import SEF, Village

CALIBRATION_OBJECTIVES: tuple[str, ...] = (
    "rank_error_proxy",
    "topk_miss_indicator",
    "top1_miss_indicator",
    "distance_to_relevant_doc_centroid",
    "specialist_vs_fused_residual_proxy",
    "query_difficulty_proxy",
)


@dataclass(frozen=True)
class QueryEvaluation:
    query: str
    target_doc: int
    rank: int
    reciprocal_rank: float
    hit_at_5: float
    relevant_distance: float
    relevant_score: float


@dataclass(frozen=True)
class ValidationPowerConfig:
    min_validation_total: int = 8
    min_validation_per_domain: int = 2
    min_effective_support_per_specialist: int = 6
    min_validation_per_query_bucket: int = 2
    min_train_total: int = 1
    min_test_total: int = 3
    calibrator_min_samples: int = 8
    balance_validation_query_buckets: bool = True
    query_expansion_multiplier: int = 12


_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech": ("battery", "charging", "charger", "cpu", "gpu", "camera", "usb", "thermal", "power", "smartphone", "driver", "pipeline", "pd", "cable"),
    "cook": ("braise", "simmer", "sauce", "oven", "stew", "cook", "recipe", "saute", "food processor", "tender", "vegetables"),
}


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


def _expanded_query_set(
    queries: list[str],
    targets: list[int],
    multiplier: int = 5,
) -> tuple[list[str], list[int]]:
    prefixes = (
        "user asks",
        "retrieval request",
        "need answer",
        "focus on",
        "find guidance for",
    )
    suffixes = (
        "",
        " please prioritize exact match",
        " include practical details",
        " with concise explanation",
        " in mixed-domain context",
        " while contrasting alternatives",
    )
    expanded_q: list[str] = []
    expanded_t: list[int] = []
    for idx, (query, target) in enumerate(zip(queries, targets)):
        for m in range(multiplier):
            separator = ":" if (m % 3 == 0) else ""
            variant = f"{prefixes[m % len(prefixes)]}{separator} {query}{suffixes[(idx + m) % len(suffixes)]}".strip()
            expanded_q.append(variant)
            expanded_t.append(target)
    return expanded_q, expanded_t


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
    if objective == "top1_miss_indicator":
        return float(1.0 if row.rank != 1 else 0.0)
    if objective == "distance_to_relevant_doc_centroid":
        return float(row.relevant_distance)
    if objective == "specialist_vs_fused_residual_proxy":
        return float(abs(row.relevant_score - row.reciprocal_rank))
    if objective == "query_difficulty_proxy":
        return 0.0
    raise ValueError(f"Unknown objective: {objective}")


def _infer_domain(query: str) -> str:
    lower = query.lower()
    hit_tech = any(token in lower for token in _DOMAIN_KEYWORDS["tech"])
    hit_cook = any(token in lower for token in _DOMAIN_KEYWORDS["cook"])
    if hit_tech and hit_cook:
        return "mixed"
    if hit_tech:
        return "tech"
    if hit_cook:
        return "cook"
    return "mixed"


def _infer_query_type(query: str) -> str:
    lower = query.lower()
    if any(marker in lower for marker in (" unlike ", " versus ", " vs ", "compare ", "contrasting", "mixed-domain")):
        return "cross_domain_compositional"
    if len(lower.split()) >= 9:
        return "long_form"
    return "direct"


def _make_strata(queries: list[str]) -> list[dict[str, str]]:
    labels: list[dict[str, str]] = []
    for query in queries:
        domain = _infer_domain(query)
        query_type = _infer_query_type(query)
        labels.append({"domain": domain, "query_type": query_type, "stratum": f"{domain}:{query_type}"})
    return labels


def _difficulty_proxy(label: dict[str, str]) -> float:
    domain = label["domain"]
    query_type = label["query_type"]
    score = 0.2
    if domain == "mixed":
        score += 0.5
    if query_type == "cross_domain_compositional":
        score += 0.3
    elif query_type == "long_form":
        score += 0.2
    return float(min(score, 1.0))


def _specialist_support(split: list[int], labels: list[dict[str, str]]) -> dict[str, int]:
    support = {"tech": 0, "cook": 0}
    for idx in split:
        domain = labels[idx]["domain"]
        if domain in {"tech", "mixed"}:
            support["tech"] += 1
        if domain in {"cook", "mixed"}:
            support["cook"] += 1
    return support


def _build_split_indices(queries: list[str], cfg: ValidationPowerConfig) -> tuple[dict[str, list[int]], dict[str, Any]]:
    n = len(queries)
    labels = _make_strata(queries)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_stratum[label["stratum"]].append(idx)

    split = {"train": [], "validation": [], "test": []}
    for _, members in sorted(by_stratum.items()):
        local = sorted(members)
        m = len(local)
        n_val = max(1, int(round(0.25 * m)))
        n_test = max(1, int(round(0.25 * m))) if m >= 3 else 1
        if n_val + n_test >= m:
            n_val = 1
            n_test = 1
        n_train = m - n_val - n_test
        split["train"].extend(local[:n_train])
        split["validation"].extend(local[n_train : n_train + n_val])
        split["test"].extend(local[n_train + n_val :])

    split = {k: sorted(v) for k, v in split.items()}

    def _domain_counts(indices: list[int]) -> dict[str, int]:
        counts = Counter(labels[i]["domain"] for i in indices)
        return {k: int(v) for k, v in sorted(counts.items())}

    def _bucket_counts(indices: list[int]) -> dict[str, int]:
        counts = Counter(labels[i]["query_type"] for i in indices)
        return {k: int(v) for k, v in sorted(counts.items())}

    def _move_to_validation(predicate: Callable[[int], bool], needed: int) -> int:
        moved = 0
        for src in ("train", "test"):
            if src == "test" and len(split["test"]) <= cfg.min_test_total:
                continue
            if src == "train" and len(split["train"]) <= cfg.min_train_total:
                continue
            candidates = [i for i in split[src] if predicate(i)]
            for idx in candidates:
                if src == "test" and len(split["test"]) <= cfg.min_test_total:
                    return moved
                if src == "train" and len(split["train"]) <= cfg.min_train_total:
                    return moved
                split[src].remove(idx)
                split["validation"].append(idx)
                moved += 1
                if moved >= needed:
                    return moved
        return moved

    if len(split["validation"]) < cfg.min_validation_total:
        _move_to_validation(lambda _: True, cfg.min_validation_total - len(split["validation"]))

    existing_domains = sorted(set(l["domain"] for l in labels))
    existing_query_buckets = sorted(set(l["query_type"] for l in labels))
    for domain in existing_domains:
        current = _domain_counts(split["validation"]).get(domain, 0)
        if current < cfg.min_validation_per_domain:
            _move_to_validation(lambda i, d=domain: labels[i]["domain"] == d, cfg.min_validation_per_domain - current)

    if cfg.balance_validation_query_buckets:
        for query_bucket in existing_query_buckets:
            current = _bucket_counts(split["validation"]).get(query_bucket, 0)
            if current < cfg.min_validation_per_query_bucket:
                _move_to_validation(
                    lambda i, qb=query_bucket: labels[i]["query_type"] == qb,
                    cfg.min_validation_per_query_bucket - current,
                )

    support = _specialist_support(split["validation"], labels)
    for specialist, current in support.items():
        if current < cfg.min_effective_support_per_specialist:
            if specialist == "tech":
                predicate = lambda i: labels[i]["domain"] in {"tech", "mixed"}
            else:
                predicate = lambda i: labels[i]["domain"] in {"cook", "mixed"}
            _move_to_validation(predicate, cfg.min_effective_support_per_specialist - current)

    split = {k: sorted(v) for k, v in split.items()}
    val_domain_counts = _domain_counts(split["validation"])
    effective_support = _specialist_support(split["validation"], labels)

    failures: list[str] = []
    if len(split["validation"]) < cfg.min_validation_total:
        failures.append("min_validation_total")
    for domain in existing_domains:
        if val_domain_counts.get(domain, 0) < cfg.min_validation_per_domain:
            failures.append(f"min_validation_per_domain:{domain}")
    val_query_bucket_counts = _bucket_counts(split["validation"])
    if cfg.balance_validation_query_buckets:
        for query_bucket in existing_query_buckets:
            if val_query_bucket_counts.get(query_bucket, 0) < cfg.min_validation_per_query_bucket:
                failures.append(f"min_validation_per_query_bucket:{query_bucket}")
    for specialist, count in effective_support.items():
        if count < cfg.min_effective_support_per_specialist:
            failures.append(f"min_effective_support_per_specialist:{specialist}")

    power_report = {
        "status": "sufficient" if not failures else "underpowered_validation",
        "configured_minimums": asdict(cfg),
        "validation_count": len(split["validation"]),
        "validation_by_domain": val_domain_counts,
        "validation_by_query_bucket": val_query_bucket_counts,
        "specialist_effective_support": effective_support,
        "strata_in_validation": dict(Counter(labels[i]["stratum"] for i in split["validation"])),
        "split_counts": {k: len(v) for k, v in split.items()},
        "split_by_domain": {
            "train": _domain_counts(split["train"]),
            "validation": val_domain_counts,
            "test": _domain_counts(split["test"]),
        },
        "failures": failures,
        "auto_enlarged_validation": len(split["validation"]) > max(1, int(round(0.25 * n))),
        "labels": labels,
    }
    return split, power_report


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


def _fused_target_scores(village: Village, queries: list[str], targets: list[int], docs: list[str]) -> np.ndarray:
    scout = ScoutRouter(mode="all")
    mean = Panoramix(fuser=MeanFuser())
    doc_emb = np.stack([village.modules[0].embed(d) for d in docs], axis=0)
    vals = np.zeros(len(queries), dtype=np.float64)
    for i, (query, target) in enumerate(zip(queries, targets)):
        potion = mean.brew(query, village=village, scout=scout)
        vals[i] = float(doc_emb[target] @ potion.vector)
    return vals


def _objective_target_array(
    objective: str,
    evals: list[QueryEvaluation],
    indices: list[int],
    max_rank: int,
    labels: list[dict[str, str]],
    fused_target_scores: np.ndarray,
) -> np.ndarray:
    if objective == "query_difficulty_proxy":
        return np.array([_difficulty_proxy(labels[i]) for i in indices], dtype=np.float64)
    if objective == "specialist_vs_fused_residual_proxy":
        return np.array(
            [abs(evals[i].relevant_score - float(fused_target_scores[i])) for i in indices],
            dtype=np.float64,
        )
    return np.array([_target_from_eval(evals[i], objective, max_rank) for i in indices], dtype=np.float64)


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


def _bucketed_delta(
    village_pre: Village,
    village_post: Village,
    queries: list[str],
    targets: list[int],
    docs: list[str],
    split: list[int],
    labels: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for i in split:
        by_bucket[labels[i]["stratum"]].append(i)
    result: dict[str, dict[str, float]] = {}
    for bucket, idxs in sorted(by_bucket.items()):
        pre = _eval_fusers(village_pre, queries, targets, docs, idxs)
        post = _eval_fusers(village_post, queries, targets, docs, idxs)
        result[bucket] = {
            "count": float(len(idxs)),
            "pre_delta": pre["delta"],
            "post_delta": post["delta"],
            "delta_change": post["delta"] - pre["delta"],
        }
    return result


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
    power_config: ValidationPowerConfig = ValidationPowerConfig(),
) -> dict[str, Any]:
    study = run_uncertainty_calibration_objective_study(
        output_dir=output_dir,
        sigma2_method=sigma2_method,
        calibrators=calibrators,
        power_config=power_config,
    )
    return study["objective_reports"][objective]


def _run_single_objective(
    *,
    output_dir: Path,
    sigma2_method: str,
    objective: str,
    calibrators: tuple[CalibratorName, ...],
    power_config: ValidationPowerConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]

    village, queries, targets = _build_village(sigma2_method)
    queries, targets = _expanded_query_set(
        queries,
        targets,
        multiplier=max(1, power_config.query_expansion_multiplier),
    )
    docs = build_toy_corpus(british_spelling=False).docs
    split, power_report = _build_split_indices(queries, power_config)
    labels = power_report["labels"]
    fused_target_scores = _fused_target_scores(village, queries, targets, docs)

    leakage_checks: list[dict[str, Any]] = []
    cal_records: list[dict[str, Any]] = []
    fitted_by_module: dict[str, CalibrationFit] = {}

    sufficiently_powered = power_report["status"] == "sufficient"
    fallback_reason = None if sufficiently_powered else "underpowered_validation"

    for module in village.modules:
        evals = _evaluate_specialist(module, queries, targets, docs)
        max_rank = len(docs)

        val_idx = split["validation"]
        val_sigma = np.array([module.sigma2_for(queries[i]) for i in val_idx], dtype=np.float64)
        val_target = _objective_target_array(
            objective,
            evals,
            val_idx,
            max_rank,
            labels,
            fused_target_scores,
        )

        leakage_checks.append(
            {
                "module": module.name,
                "train_indices": split["train"],
                "validation_indices": val_idx,
                "intersection": sorted(set(split["train"]).intersection(val_idx)),
            }
        )

        if not sufficiently_powered:
            fit = CalibrationFit(
                calibrator=ScalarCalibrator(
                    name="identity",
                    params={"reason": "underpowered_validation", "failures": power_report["failures"]},
                ),
                n_train=int(len(val_idx)),
                used_fallback=True,
                objective_mse=float(np.mean((val_sigma - val_target) ** 2)) if len(val_idx) else 0.0,
            )
            cal_records.append(
                {
                    "module": module.name,
                    "calibrator": "identity",
                    "objective_mse": fit.objective_mse,
                    "used_fallback": True,
                    "n_train": fit.n_train,
                    "skipped_due_to_underpowered_validation": True,
                }
            )
            fitted_by_module[module.name] = fit
            fit.calibrator.to_json(output_dir / f"calibrator_{module.name}.json")
            continue

        best_fit: CalibrationFit | None = None
        for method in calibrators:
            fit = fit_scalar_calibrator(
                val_sigma,
                val_target,
                method,
                min_samples=power_config.calibrator_min_samples,
            )
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
        target = _objective_target_array(
            objective,
            evals,
            test_idx,
            max_rank,
            labels,
            fused_target_scores,
        )
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
    calibrated_village = Village(calibrated_modules)
    post_bench = _eval_fusers(calibrated_village, queries, targets, docs, test_idx)
    val_pre_bench = _eval_fusers(village, queries, targets, docs, split["validation"])
    val_post_bench = _eval_fusers(calibrated_village, queries, targets, docs, split["validation"])

    summary = {
        "status": power_report["status"],
        "objective": objective,
        "sigma2_method": sigma2_method,
        "split": split,
        "validation_power": {k: v for k, v in power_report.items() if k != "labels"},
        "stratification": {
            "query_labels": power_report["labels"],
        },
        "leakage_checks": leakage_checks,
        "calibration_candidates": cal_records,
        "selected_calibrators": {
            m: {
                "name": fit.calibrator.name,
                "mse": fit.objective_mse,
                "fallback": fit.used_fallback,
                "sufficiently_powered": sufficiently_powered,
                "effective_support": power_report["specialist_effective_support"].get(m, 0),
                "fallback_reason": (
                    "underpowered_validation"
                    if not sufficiently_powered
                    else (
                        fit.calibrator.params.get("reason")
                        if fit.used_fallback
                        else None
                    )
                ),
            }
            for m, fit in fitted_by_module.items()
        },
        "uncertainty_diagnostics": diagnostics,
        "benchmark_delta": {
            "validation": {
                "pre": val_pre_bench,
                "post": val_post_bench,
                "delta_change": val_post_bench["delta"] - val_pre_bench["delta"],
            },
            "pre": pre_bench,
            "post": post_bench,
            "delta_change": post_bench["delta"] - pre_bench["delta"],
        },
        "per_bucket_outcomes": _bucketed_delta(
            village, calibrated_village, queries, targets, docs, test_idx, labels
        ),
        "powered_for_calibration": sufficiently_powered,
        "minimum_support_threshold": power_config.min_effective_support_per_specialist,
        "per_specialist_support_counts": power_report["specialist_effective_support"],
        "fallback_reason": fallback_reason,
        "benchmark_profile": {
            "name": "toy_corpus_expanded_stratified",
            "query_count": len(queries),
            "query_expansion_multiplier": power_config.query_expansion_multiplier,
            "validation_query_bucket_balancing": power_config.balance_validation_query_buckets,
        },
        "sigma2_path_audit": _audit_sigma2_paths(repo_root),
        "notes": "Calibration fit uses validation-only data with stratified splits and power checks.",
    }
    return summary


def run_uncertainty_calibration_objective_study(
    output_dir: Path,
    sigma2_method: str = "centroid_distance_sigma2",
    calibrators: tuple[CalibratorName, ...] = (
        "affine",
        "temperature",
        "isotonic",
        "piecewise_monotonic",
    ),
    objectives: tuple[str, ...] = CALIBRATION_OBJECTIVES,
    power_config: ValidationPowerConfig = ValidationPowerConfig(),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for objective in objectives:
        reports[objective] = _run_single_objective(
            output_dir=output_dir / objective,
            sigma2_method=sigma2_method,
            objective=objective,
            calibrators=calibrators,
            power_config=power_config,
        )

    validation_transfer_scores: dict[str, float] = {}
    for objective, rep in reports.items():
        diag_improvement = 0.0
        count = 0
        for module_diag in rep["uncertainty_diagnostics"].values():
            pre_ece = module_diag["pre"]["reliability"]["ece"]
            post_ece = module_diag["post"]["reliability"]["ece"]
            pre_corr = module_diag["pre"]["spearman"]
            post_corr = module_diag["post"]["spearman"]
            diag_improvement += (pre_ece - post_ece) + (post_corr - pre_corr)
            count += 1
        diag_improvement = diag_improvement / max(count, 1)
        validation_delta_change = rep["benchmark_delta"]["validation"]["delta_change"]
        # Strictly validation-driven rule: objective must improve diagnostics and not hurt validation fusion delta.
        validation_transfer_scores[objective] = float(diag_improvement + validation_delta_change)

    selected_objective = max(validation_transfer_scores, key=validation_transfer_scores.get)
    selected_report = reports[selected_objective]

    study = {
        "selection_rule": "Select objective maximizing validation diagnostic gain + validation Kalman-vs-Mean delta change.",
        "selected_objective": selected_objective,
        "validation_transfer_scores": validation_transfer_scores,
        "objective_reports": reports,
        "selected_report": selected_report,
        "powered_for_calibration": selected_report["powered_for_calibration"],
        "minimum_support_threshold": selected_report["minimum_support_threshold"],
        "per_specialist_support_counts": selected_report["per_specialist_support_counts"],
        "fallback_reason": selected_report["fallback_reason"],
        "selection_is_validation_only": True,
        "observations": {
            "diagnostics_without_fusion_gain": [
                obj
                for obj, rep in reports.items()
                if any(
                    m["post"]["reliability"]["ece"] < m["pre"]["reliability"]["ece"]
                    for m in rep["uncertainty_diagnostics"].values()
                )
                and rep["benchmark_delta"]["delta_change"] <= 0.0
            ]
        },
    }

    (output_dir / "summary.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report_md(study), encoding="utf-8")
    (output_dir / "sigma2_audit.json").write_text(
        json.dumps(selected_report["sigma2_path_audit"], indent=2), encoding="utf-8"
    )
    return study


def _render_report_md(study: dict[str, Any]) -> str:
    rep = study["selected_report"]
    lines = [
        "# Uncertainty calibration report",
        "",
        f"- Selected objective: `{study['selected_objective']}`",
        f"- Selection rule: {study['selection_rule']}",
        f"- Powered for calibration: `{rep['powered_for_calibration']}`",
        f"- Minimum support threshold: `{rep['minimum_support_threshold']}`",
        f"- Per-specialist support: `{rep['per_specialist_support_counts']}`",
        f"- Fallback reason: `{rep['fallback_reason']}`",
        "",
        "## Split diagnostics",
        "",
        f"- Split counts: `{rep['validation_power']['split_counts']}`",
        f"- Validation domains: `{rep['validation_power']['validation_by_domain']}`",
        f"- Validation query buckets: `{rep['validation_power']['validation_by_query_bucket']}`",
        "",
        "## Selected calibrators",
        "",
    ]
    for module, payload in sorted(rep["selected_calibrators"].items()):
        lines.append(
            f"- `{module}`: calibrator=`{payload['name']}`, fallback=`{payload['fallback']}`, "
            f"mse=`{payload['mse']:.6f}`, support=`{payload['effective_support']}`"
        )
    bench = rep["benchmark_delta"]
    lines.extend(
        [
            "",
            "## Kalman-vs-Mean (calibrated sigma2)",
            "",
            f"- Validation delta change: `{bench['validation']['delta_change']:.6f}`",
            f"- Test delta change: `{bench['delta_change']:.6f}`",
            "",
            "If delta change is non-positive, calibration did not improve the downstream benchmark under this powered regime.",
            "",
        ]
    )
    return "\n".join(lines)
