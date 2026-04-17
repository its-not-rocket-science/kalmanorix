"""Uncertainty estimation ablation benchmark.

This benchmark compares sigma² methods on the same query/doc datasets and reports:
- retrieval quality
- calibration quality
- sensitivity to uncertainty scaling mis-specification
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from kalmanorix.calibration import compute_retrieval_calibration
from kalmanorix.panoramix import KalmanorixFuser, Panoramix
from kalmanorix.scout import ScoutRouter
from kalmanorix.toy_corpus import build_toy_corpus
from kalmanorix.uncertainty import (
    CentroidNormPeerSigma2,
    SimilarityToCentroidSigma2,
    ScaledSigma2,
    UncertaintyMethodConfig,
    create_uncertainty_method,
)
from kalmanorix.village import SEF, Village


@dataclass(frozen=True)
class AblationDataset:
    name: str
    synthetic: bool
    docs: list[str]
    queries: list[str]
    true_indices: list[int]
    groups: list[str]


@dataclass(frozen=True)
class MethodMetrics:
    recall_at_1: float
    recall_at_5: float
    mrr_at_10: float
    ece: float
    brier_score: float
    mean_variance: float


@dataclass(frozen=True)
class ScaleSensitivity:
    scales: dict[str, MethodMetrics]
    recall1_range: float
    ece_range: float


def _token_embedder(
    vocabulary: Sequence[str], domain_boost: tuple[str, str]
) -> Callable[[str], np.ndarray]:
    vocab_to_idx = {w: i for i, w in enumerate(vocabulary)}
    domain_tokens = set(domain_boost)

    def _embed(text: str) -> np.ndarray:
        vec = np.zeros(len(vocabulary), dtype=np.float64)
        lower = text.lower()
        for token, idx in vocab_to_idx.items():
            if token in lower:
                vec[idx] += 1.0
        if any(tok in lower for tok in domain_tokens):
            vec += 0.15
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    return _embed


def build_ablation_datasets() -> list[AblationDataset]:
    corpus = build_toy_corpus(british_spelling=False)
    q_texts = [q for q, _ in corpus.queries]
    q_targets = [idx for _, idx in corpus.queries]

    synthetic_shift_queries = [
        "gpu heat with oven-like thermal spikes",
        "slow braise and battery saving analogy",
        "camera pipeline like sauce reduction",
        "usb-c charger and simmer timing confusion",
    ]
    synthetic_shift_targets = [11, 16, 13, 14]

    return [
        AblationDataset(
            name="toy_mixed",
            synthetic=False,
            docs=corpus.docs,
            queries=q_texts,
            true_indices=q_targets,
            groups=corpus.groups,
        ),
        AblationDataset(
            name="synthetic_shifted_queries",
            synthetic=True,
            docs=corpus.docs,
            queries=synthetic_shift_queries,
            true_indices=synthetic_shift_targets,
            groups=["synthetic"] * len(synthetic_shift_queries),
        ),
    ]


def _build_specialists(docs: Sequence[str], method: str) -> Village:
    vocab = [
        "battery",
        "charging",
        "cpu",
        "gpu",
        "camera",
        "usb",
        "thermal",
        "braise",
        "simmer",
        "sauce",
        "oven",
        "stew",
        "cook",
        "recipe",
    ]
    tech_embed = _token_embedder(vocab, ("battery", "gpu"))
    cook_embed = _token_embedder(vocab, ("braise", "simmer"))

    tech_cal = [
        d
        for d in docs
        if any(k in d.lower() for k in ("battery", "gpu", "cpu", "camera", "usb"))
    ]
    cook_cal = [
        d
        for d in docs
        if any(k in d.lower() for k in ("braise", "simmer", "oven", "sauce", "stew"))
    ]

    if method == "centroid_norm_peer_sigma2":
        tech_centroid = SimilarityToCentroidSigma2.from_calibration(
            embed=tech_embed, calibration_texts=tech_cal
        ).centroid
        cook_centroid = SimilarityToCentroidSigma2.from_calibration(
            embed=cook_embed, calibration_texts=cook_cal
        ).centroid
        tech_sigma = CentroidNormPeerSigma2.from_calibration(
            embed=tech_embed,
            calibration_texts=tech_cal,
            peer_centroids=[cook_centroid],
        )
        cook_sigma = CentroidNormPeerSigma2.from_calibration(
            embed=cook_embed,
            calibration_texts=cook_cal,
            peer_centroids=[tech_centroid],
        )
    else:
        tech_sigma = create_uncertainty_method(
            config=UncertaintyMethodConfig(method=method),
            embed=tech_embed,
            calibration_texts=tech_cal,
        )
        cook_sigma = create_uncertainty_method(
            config=UncertaintyMethodConfig(method=method),
            embed=cook_embed,
            calibration_texts=cook_cal,
        )

    modules = [
        SEF(
            name="tech",
            embed=tech_embed,
            sigma2=tech_sigma,
            embedding_dimension=len(vocab),
        ),
        SEF(
            name="cook",
            embed=cook_embed,
            sigma2=cook_sigma,
            embedding_dimension=len(vocab),
        ),
    ]
    return Village(modules)


def _rank_docs(vec: np.ndarray, doc_embeddings: np.ndarray) -> list[int]:
    sims = doc_embeddings @ vec
    return list(np.argsort(-sims))


def _compute_metrics(
    dataset: AblationDataset, village: Village
) -> tuple[MethodMetrics, list[float]]:
    pan = Panoramix(fuser=KalmanorixFuser())
    scout = ScoutRouter(mode="all")
    doc_embeddings = np.stack(
        [village.modules[0].embed(d) for d in dataset.docs], axis=0
    )

    rankings: list[list[int]] = []
    fused_variances: list[float] = []
    query_embeddings: list[np.ndarray] = []
    for query in dataset.queries:
        potion = pan.brew(query, village=village, scout=scout)
        ranking = _rank_docs(potion.vector, doc_embeddings)
        rankings.append(ranking)
        query_embeddings.append(potion.vector)
        var = 1.0
        if potion.meta is not None and "variance" in potion.meta:
            var = float(potion.meta["variance"])
        fused_variances.append(var)

    success1 = []
    success5 = []
    rr10 = []
    for ranked, tgt in zip(rankings, dataset.true_indices):
        success1.append(1.0 if ranked[:1] == [tgt] else 0.0)
        success5.append(1.0 if tgt in ranked[:5] else 0.0)
        rr = 0.0
        for idx, doc_idx in enumerate(ranked[:10], start=1):
            if doc_idx == tgt:
                rr = 1.0 / idx
                break
        rr10.append(rr)

    cal = compute_retrieval_calibration(
        query_embeddings=np.stack(query_embeddings, axis=0),
        doc_embeddings=doc_embeddings,
        query_variances=np.array(fused_variances, dtype=np.float64),
        true_indices=dataset.true_indices,
        k=5,
        n_bins=10,
        distance_metric="cosine",
    )

    return (
        MethodMetrics(
            recall_at_1=float(np.mean(success1)),
            recall_at_5=float(np.mean(success5)),
            mrr_at_10=float(np.mean(rr10)),
            ece=float(cal.ece),
            brier_score=float(cal.brier_score),
            mean_variance=float(np.mean(fused_variances)),
        ),
        fused_variances,
    )


def evaluate_uncertainty_method_on_dataset(
    method: str, dataset: AblationDataset
) -> dict[str, Any]:
    village = _build_specialists(dataset.docs, method)
    base_metrics, _ = _compute_metrics(dataset, village)

    scales = [0.5, 1.0, 2.0, 4.0]
    scaled_metrics: dict[str, MethodMetrics] = {}
    for scale in scales:
        scaled_modules = []
        for module in village.modules:
            sigma2 = module.sigma2
            assert callable(sigma2)
            scaled_modules.append(
                SEF(
                    name=module.name,
                    embed=module.embed,
                    sigma2=ScaledSigma2(base_sigma2=sigma2, scale=scale),
                    embedding_dimension=module.embedding_dimension,
                )
            )
        scaled_village = Village(scaled_modules)
        metrics, _ = _compute_metrics(dataset, scaled_village)
        scaled_metrics[str(scale)] = metrics

    sensitivity = summarize_scale_sensitivity(scaled_metrics)
    return {
        "dataset": dataset.name,
        "synthetic": dataset.synthetic,
        "method": method,
        "metrics": asdict(base_metrics),
        "scale_sensitivity": {
            "scales": {k: asdict(v) for k, v in sensitivity.scales.items()},
            "recall1_range": sensitivity.recall1_range,
            "ece_range": sensitivity.ece_range,
        },
    }


def summarize_scale_sensitivity(
    metrics_by_scale: dict[str, MethodMetrics],
) -> ScaleSensitivity:
    recalls = [m.recall_at_1 for m in metrics_by_scale.values()]
    eces = [m.ece for m in metrics_by_scale.values()]
    return ScaleSensitivity(
        scales=metrics_by_scale,
        recall1_range=float(max(recalls) - min(recalls)) if recalls else 0.0,
        ece_range=float(max(eces) - min(eces)) if eces else 0.0,
    )


def run_uncertainty_ablation(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Focused comparison requested for Kalman-fusion usefulness:
    # constant baseline vs current default query-dependent method vs improved method.
    methods = [
        "constant_sigma2",
        "centroid_distance_sigma2",
        "centroid_norm_peer_sigma2",
    ]
    datasets = build_ablation_datasets()

    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            results.append(evaluate_uncertainty_method_on_dataset(method, dataset))

    answered = _answer_core_question(results)
    summary = {
        "methods": methods,
        "datasets": [{"name": d.name, "synthetic": d.synthetic} for d in datasets],
        "results": results,
        "answer": answered,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = _render_report(summary)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    return summary


def _answer_core_question(results: Iterable[dict[str, Any]]) -> str:
    rows = [r for r in results if not r["synthetic"]]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    def _avg(key: str, method: str) -> float:
        vals = [float(r["metrics"][key]) for r in by_method.get(method, [])]
        return float(np.mean(vals)) if vals else 0.0

    best_uncertainty = min(
        by_method.keys(),
        key=lambda m: _avg("ece", m),
        default="constant_sigma2",
    )
    best_retrieval = max(
        by_method.keys(),
        key=lambda m: _avg("recall_at_1", m),
        default="constant_sigma2",
    )

    if best_uncertainty == best_retrieval and best_retrieval != "constant_sigma2":
        return (
            "Yes: in non-synthetic evaluation, the method with the best calibration "
            "also achieved the strongest retrieval under Kalman fusion, indicating "
            "uncertainty quality materially changes fusion usefulness."
        )

    if best_retrieval == "constant_sigma2":
        return (
            "Partially: calibration differences are visible, but retrieval gains from "
            "better uncertainty estimation are limited in this setup; constant uncertainty "
            "remains competitive."
        )

    return (
        "Mixed: better uncertainty estimation improves calibration quality, but the best "
        "retrieval method is not always the best-calibrated one, so impact depends on "
        "dataset and operating regime."
    )


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Uncertainty Ablation Report",
        "",
        "## Scope",
        "- Methods evaluated uniformly: constant, centroid-distance (current default), centroid+norm+peer (improved).",
        "- Datasets include one real toy-mixed split and one clearly-labeled synthetic split.",
        "",
        "## Results Snapshot",
    ]

    for row in summary["results"]:
        m = row["metrics"]
        label = "(synthetic)" if row["synthetic"] else ""
        lines.append(
            f"- dataset={row['dataset']} {label}, method={row['method']}: "
            f"recall@1={m['recall_at_1']:.3f}, recall@5={m['recall_at_5']:.3f}, "
            f"mrr@10={m['mrr_at_10']:.3f}, ece={m['ece']:.3f}, brier={m['brier_score']:.3f}"
        )

    lines.extend(
        [
            "",
            "## Focused Delta vs Constant Baseline",
        ]
    )
    by_dataset_and_method = {
        (row["dataset"], row["method"]): row for row in summary["results"]
    }
    for dataset in summary["datasets"]:
        dataset_name = dataset["name"]
        base = by_dataset_and_method.get((dataset_name, "constant_sigma2"))
        default = by_dataset_and_method.get((dataset_name, "centroid_distance_sigma2"))
        improved = by_dataset_and_method.get(
            (dataset_name, "centroid_norm_peer_sigma2")
        )
        if not base or not default or not improved:
            continue
        base_r1 = float(base["metrics"]["recall_at_1"])
        base_ece = float(base["metrics"]["ece"])
        lines.append(
            f"- {dataset_name}: "
            f"default Δrecall@1={float(default['metrics']['recall_at_1']) - base_r1:+.3f}, "
            f"default ΔECE={float(default['metrics']['ece']) - base_ece:+.3f}; "
            f"improved Δrecall@1={float(improved['metrics']['recall_at_1']) - base_r1:+.3f}, "
            f"improved ΔECE={float(improved['metrics']['ece']) - base_ece:+.3f}"
        )

    lines.extend(
        [
            "",
            "## Sensitivity to Mis-Specified Uncertainty Scaling",
            "Each method was re-evaluated with sigma² scaled by {0.5, 1.0, 2.0, 4.0}.",
        ]
    )
    for row in summary["results"]:
        sens = row["scale_sensitivity"]
        lines.append(
            f"- {row['dataset']} / {row['method']}: recall@1 range={sens['recall1_range']:.3f}, "
            f"ECE range={sens['ece_range']:.3f}"
        )

    lines.extend(
        [
            "",
            "## Does better uncertainty estimation improve Kalman fusion enough to matter?",
            summary["answer"],
            "",
            "## Artifacts",
            "- summary.json",
            "- report.md",
        ]
    )
    return "\n".join(lines) + "\n"
