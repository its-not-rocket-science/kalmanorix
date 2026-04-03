"""Validate Procrustes alignment across real sentence embedding models.

Milestone 1.2 success criterion:
    alignment improves cross-model similarity by >20%.

This script:
1) Loads three Sentence-Transformers-compatible models.
2) Builds a diverse anchor set of 500 sentences.
3) Computes cosine similarity before/after Procrustes alignment for every model pair.
4) Validates determinant correction so alignments are proper rotations (+1 det).
5) Repeats evaluation under d in {384, 768, 1024} via pad/truncate.
6) Saves JSON metrics + PNG visualisation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en",
]
TARGET_DIMS = (384, 768, 1024)


@dataclass
class PairDimensionResult:
    """Metrics for one (model_a, model_b, target_dim) evaluation."""

    model_a: str
    model_b: str
    target_dim: int
    similarity_before: float
    similarity_after: float
    improvement_pct: float
    determinant: float
    determinant_passed: bool
    sample_improvements: np.ndarray


def generate_diverse_sentences(n: int, *, seed: int = 1234) -> List[str]:
    """Generate deterministic diverse sentences across many topics/registers."""
    rng = np.random.default_rng(seed)
    topics = {
        "technology": [
            "distributed databases",
            "neural network pruning",
            "zero-trust security",
            "edge inference",
            "compiler optimization",
            "stream processing",
        ],
        "medicine": [
            "cardiac rehabilitation",
            "type-2 diabetes prevention",
            "telemedicine workflow",
            "clinical trial enrollment",
            "radiology triage",
            "vaccine cold chain",
        ],
        "finance": [
            "yield curve inversion",
            "credit risk modeling",
            "portfolio rebalancing",
            "fraud detection",
            "retail banking churn",
            "mortgage refinancing",
        ],
        "law": [
            "contract interpretation",
            "intellectual property dispute",
            "e-discovery protocol",
            "class action certification",
            "administrative review",
            "arbitration clause",
        ],
        "science": [
            "protein folding",
            "ocean acidification",
            "battery cathode chemistry",
            "exoplanet transit",
            "climate attribution",
            "quantum error correction",
        ],
        "daily_life": [
            "grocery budgeting",
            "public transit planning",
            "sleep hygiene routine",
            "home internet troubleshooting",
            "pet nutrition schedule",
            "community volunteering",
        ],
    }
    styles = [
        "Explain {topic} to a curious high-school student.",
        "Write a concise summary about {topic} for a manager.",
        "List two practical risks associated with {topic}.",
        "Give a neutral definition of {topic} in one sentence.",
        "Describe a recent challenge in {topic} and a possible fix.",
        "Compare traditional and modern approaches to {topic}.",
        "State one misconception people have about {topic}.",
    ]

    all_sentences: List[str] = []
    topic_values = [item for values in topics.values() for item in values]
    while len(all_sentences) < n:
        topic = rng.choice(topic_values)
        style = rng.choice(styles)
        suffix = rng.choice(
            [
                "Use plain language.",
                "Include one concrete example.",
                "Avoid jargon.",
                "Keep the tone objective.",
                "Mention trade-offs briefly.",
            ]
        )
        all_sentences.append(f"{style.format(topic=topic)} {suffix}")

    return all_sentences[:n]


def adapt_dimension(emb: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or truncate embedding matrix to target dimension."""
    current_dim = emb.shape[1]
    if current_dim == target_dim:
        return emb
    if current_dim > target_dim:
        return emb[:, :target_dim]

    padded = np.zeros((emb.shape[0], target_dim), dtype=emb.dtype)
    padded[:, :current_dim] = emb
    return padded


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return np.sum(a_n * b_n, axis=1)


def procrustes_rotation(
    source: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, float, bool]:
    """Compute orthogonal Procrustes map source->target with det correction."""
    cross_cov = source.T @ target
    u_mat, _, vt_mat = np.linalg.svd(cross_cov, full_matrices=False)

    rotation = u_mat @ vt_mat
    det_val = float(np.linalg.det(rotation))

    if det_val < 0:
        correction = np.eye(u_mat.shape[1])
        correction[-1, -1] = -1.0
        rotation = u_mat @ correction @ vt_mat
        det_val = float(np.linalg.det(rotation))

    passed = bool(np.isclose(det_val, 1.0, atol=1e-5))
    return rotation, det_val, passed


def load_model_embeddings(
    model_name: str, texts: List[str], batch_size: int = 64
) -> np.ndarray:
    """Encode text with sentence-transformers model."""
    from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float64)


def build_visualization(
    results: List[PairDimensionResult],
    out_path: Path,
) -> None:
    """Create composite figure with heatmap, distribution, and determinant bars."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    pair_labels = sorted(
        {f"{r.model_a.split('/')[-1]} vs {r.model_b.split('/')[-1]}" for r in results}
    )
    dims = sorted({r.target_dim for r in results})

    before_grid = np.zeros((len(pair_labels), len(dims)))
    after_grid = np.zeros((len(pair_labels), len(dims)))

    pair_to_idx = {p: i for i, p in enumerate(pair_labels)}
    dim_to_idx = {d: j for j, d in enumerate(dims)}

    all_deltas: List[np.ndarray] = []
    det_labels: List[str] = []
    det_values: List[float] = []

    for row in results:
        p_label = f"{row.model_a.split('/')[-1]} vs {row.model_b.split('/')[-1]}"
        i, j = pair_to_idx[p_label], dim_to_idx[row.target_dim]
        before_grid[i, j] = row.similarity_before
        after_grid[i, j] = row.similarity_after
        all_deltas.append(row.sample_improvements)
        det_labels.append(f"{p_label}\n d={row.target_dim}")
        det_values.append(row.determinant)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_before, ax_after, ax_hist, ax_det = axes.flatten()

    im1 = ax_before.imshow(before_grid, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    ax_before.set_title("Mean Cosine Similarity (Before Alignment)")
    ax_before.set_xticks(range(len(dims)), labels=[str(d) for d in dims])
    ax_before.set_yticks(range(len(pair_labels)), labels=pair_labels)
    fig.colorbar(im1, ax=ax_before, fraction=0.046, pad=0.04)

    im2 = ax_after.imshow(after_grid, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    ax_after.set_title("Mean Cosine Similarity (After Alignment)")
    ax_after.set_xticks(range(len(dims)), labels=[str(d) for d in dims])
    ax_after.set_yticks(range(len(pair_labels)), labels=pair_labels)
    fig.colorbar(im2, ax=ax_after, fraction=0.046, pad=0.04)

    if all_deltas:
        ax_hist.hist(np.concatenate(all_deltas), bins=40, color="#3B82F6", alpha=0.8)
    ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax_hist.set_title("Distribution of Per-sentence Similarity Improvements")
    ax_hist.set_xlabel("After - Before cosine")
    ax_hist.set_ylabel("Count")

    x = np.arange(len(det_values))
    colors = [
        "#10B981" if np.isclose(v, 1.0, atol=1e-5) else "#EF4444" for v in det_values
    ]
    ax_det.bar(x, det_values, color=colors)
    ax_det.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax_det.axhline(-1.0, color="gray", linestyle=":", linewidth=1)
    ax_det.set_title("Determinant Values After Correction")
    ax_det.set_xticks(x, labels=det_labels, rotation=35, ha="right")
    ax_det.set_ylabel("det(R)")

    fig.suptitle("Procrustes Alignment Validation", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_validation(
    models: List[str],
    n_anchor: int,
    n_eval: int,
    batch_size: int,
    out_json: Path,
    out_png: Path,
) -> Dict[str, object]:
    """Run end-to-end validation and return serializable result dictionary."""
    anchor_sentences = generate_diverse_sentences(n_anchor, seed=7)
    eval_sentences = generate_diverse_sentences(n_eval, seed=99)

    print(f"Loading models: {models}")
    anchor_emb: Dict[str, np.ndarray] = {}
    eval_emb: Dict[str, np.ndarray] = {}
    for model in models:
        print(f"  Encoding anchor/eval with {model} ...")
        anchor_emb[model] = load_model_embeddings(
            model, anchor_sentences, batch_size=batch_size
        )
        eval_emb[model] = load_model_embeddings(
            model, eval_sentences, batch_size=batch_size
        )

    pair_results: List[PairDimensionResult] = []
    pairs = list(itertools.combinations(models, 2))

    for model_a, model_b in pairs:
        for dim in TARGET_DIMS:
            src_anchor = adapt_dimension(anchor_emb[model_b], dim)
            tgt_anchor = adapt_dimension(anchor_emb[model_a], dim)
            src_eval = adapt_dimension(eval_emb[model_b], dim)
            tgt_eval = adapt_dimension(eval_emb[model_a], dim)

            similarity_before_samples = cosine_per_row(tgt_eval, src_eval)
            similarity_before = float(np.mean(similarity_before_samples))

            rotation, det_value, det_passed = procrustes_rotation(
                src_anchor, tgt_anchor
            )
            aligned_eval = src_eval @ rotation
            similarity_after_samples = cosine_per_row(tgt_eval, aligned_eval)
            similarity_after = float(np.mean(similarity_after_samples))

            denom = similarity_before if abs(similarity_before) > 1e-12 else 1e-12
            improvement_pct = float((similarity_after - similarity_before) / denom)

            pair_results.append(
                PairDimensionResult(
                    model_a=model_a,
                    model_b=model_b,
                    target_dim=dim,
                    similarity_before=similarity_before,
                    similarity_after=similarity_after,
                    improvement_pct=improvement_pct,
                    determinant=det_value,
                    determinant_passed=det_passed,
                    sample_improvements=similarity_after_samples
                    - similarity_before_samples,
                )
            )

            print(
                f"[{dim}] {model_a.split('/')[-1]} vs {model_b.split('/')[-1]} | "
                f"before={similarity_before:.4f} after={similarity_after:.4f} "
                f"impr={improvement_pct * 100:.2f}% det={det_value:.6f}"
            )

    avg_improvement = (
        float(np.mean([r.improvement_pct for r in pair_results]))
        if pair_results
        else 0.0
    )
    determinant_ok = all(r.determinant_passed for r in pair_results)
    meets_criteria = avg_improvement > 0.20

    per_pair_improvements: Dict[str, Dict[str, object]] = {}
    for r in pair_results:
        key = f"{r.model_a}__{r.model_b}__d{r.target_dim}"
        per_pair_improvements[key] = {
            "model_a": r.model_a,
            "model_b": r.model_b,
            "target_dim": r.target_dim,
            "similarity_before": r.similarity_before,
            "similarity_after": r.similarity_after,
            "improvement_pct": r.improvement_pct,
        }

    determinant_checks = {
        "status": "passed" if determinant_ok else "failed",
        "details": {
            f"{r.model_a}__{r.model_b}__d{r.target_dim}": {
                "determinant": r.determinant,
                "passed": r.determinant_passed,
            }
            for r in pair_results
        },
    }

    output: Dict[str, object] = {
        "models": models,
        "anchor_count": n_anchor,
        "eval_count": n_eval,
        "dimensions_tested": list(TARGET_DIMS),
        "per_pair_improvements": per_pair_improvements,
        "average_improvement": avg_improvement,
        "meets_criteria": meets_criteria,
        "determinant_checks": determinant_checks,
    }

    if not meets_criteria:
        output["alternatives"] = [
            "Consider non-linear alignment using a contrastive projection head trained on paired anchor embeddings.",
            "Use Research Track A: contrastive learning objective (InfoNCE) across model pairs with hard negatives.",
            "Try CCA/whitening preprocessing before Procrustes to reduce anisotropy mismatch.",
        ]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    build_visualization(pair_results, out_png)
    print(f"Saved JSON results to {out_json}")
    print(f"Saved plot to {out_png}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Procrustes alignment across real models"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--anchor-count", type=int, default=500)
    parser.add_argument("--eval-count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/alignment_validation.json"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("results/alignment_improvement.png"),
    )

    args = parser.parse_args()
    run_validation(
        models=args.models,
        n_anchor=args.anchor_count,
        n_eval=args.eval_count,
        batch_size=args.batch_size,
        out_json=args.out_json,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
