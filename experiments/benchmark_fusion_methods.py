#!/usr/bin/env python3
"""Benchmark Kalman fusion vs simple mean fusion on mixed-domain retrieval.

This script creates (or loads) a synthetic mixed-domain retrieval dataset,
embeds texts with lightweight sentence-transformer specialists, evaluates
retrieval quality for Kalman and arithmetic-mean fusion, runs a paired
bootstrap significance test, and saves detailed + summary artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "sentence-transformers is required. Install with `pip install sentence-transformers`."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal


DOMAINS: Dict[str, Dict[str, List[str]]] = {
    "legal": {
        "docs": [
            "Breach of contract claims require proving duty, breach, causation, and damages.",
            "A preliminary injunction needs likelihood of success and irreparable harm.",
            "Consideration is a bargained-for exchange in contract formation.",
            "Hearsay is generally inadmissible unless a recognized exception applies.",
            "Summary judgment is proper when no genuine issue of material fact exists.",
            "The statute of limitations sets filing deadlines for civil claims.",
            "Negligence per se can arise from violating a safety statute.",
            "An enforceable non-compete must be reasonable in scope and duration.",
            "Miranda warnings protect against compelled self-incrimination during custodial interrogation.",
            "Corporate veil piercing requires misuse of the corporate form and inequitable conduct.",
            "Res judicata bars relitigation of claims resolved by a final judgment.",
            "Arbitration clauses are generally enforceable under the Federal Arbitration Act.",
        ],
        "queries": [
            "What elements are needed for breach of contract?",
            "When will a court grant preliminary injunction relief?",
            "Define legal consideration in a contract.",
            "Is hearsay allowed in court testimony?",
            "When can summary judgment be granted?",
            "How long do I have to file a civil lawsuit?",
            "What is negligence per se?",
            "Are non-compete agreements enforceable?",
            "Why are Miranda rights required?",
            "When do courts pierce the corporate veil?",
            "What does res judicata prevent?",
            "Can arbitration clauses override litigation?",
        ],
    },
    "medical": {
        "docs": [
            "Type 2 diabetes management includes lifestyle change and glucose-lowering medications.",
            "Hypertension diagnosis usually requires elevated blood pressure on repeated measurements.",
            "Broad-spectrum antibiotics should be narrowed once culture results are available.",
            "Asthma exacerbations are treated with bronchodilators and systemic corticosteroids.",
            "Myocardial infarction often presents with chest pain, diaphoresis, and elevated troponin.",
            "Vaccination induces adaptive immune memory against targeted pathogens.",
            "Chronic kidney disease staging uses estimated glomerular filtration rate thresholds.",
            "Iron deficiency anemia commonly causes fatigue and microcytic red blood cells.",
            "MRI is sensitive for soft tissue evaluation without ionizing radiation.",
            "Sepsis requires early recognition, fluids, and prompt empiric antimicrobials.",
            "Insulin corrects hyperglycemia by increasing cellular glucose uptake.",
            "Stroke warning signs include facial droop, arm weakness, and speech difficulty.",
        ],
        "queries": [
            "How is type 2 diabetes usually treated?",
            "How do clinicians diagnose hypertension?",
            "When should broad-spectrum antibiotics be de-escalated?",
            "What helps during an asthma flare?",
            "What symptoms suggest myocardial infarction?",
            "How do vaccines provide protection?",
            "How is chronic kidney disease staged?",
            "What findings are common in iron deficiency anemia?",
            "Why choose MRI for soft tissue imaging?",
            "What are first steps in sepsis treatment?",
            "How does insulin lower blood sugar?",
            "What are common stroke warning signs?",
        ],
    },
    "technical": {
        "docs": [
            "Binary search runs in logarithmic time on sorted arrays.",
            "A hash table offers average O(1) lookup with good hashing and load factors.",
            "TCP provides reliable ordered byte-stream delivery over IP.",
            "Gradient descent updates parameters opposite the loss gradient direction.",
            "Docker containers package applications with dependencies for reproducible deployment.",
            "REST APIs typically use stateless HTTP methods and resource-oriented URIs.",
            "Vector databases support semantic search using nearest-neighbor indexing.",
            "Transformer attention computes weighted token interactions in parallel.",
            "A/B testing compares variants with randomized traffic allocation.",
            "Caching reduces latency by storing frequently requested computation results.",
            "Public-key cryptography separates encryption and decryption keys.",
            "Unit tests validate behavior of small isolated code components.",
        ],
        "queries": [
            "What is the complexity of binary search?",
            "When do hash tables give O(1) lookups?",
            "What does TCP guarantee compared to IP?",
            "How does gradient descent update parameters?",
            "Why use Docker containers in deployment?",
            "What defines a REST API design?",
            "Why are vector databases useful for semantic retrieval?",
            "How does transformer attention work?",
            "What is the purpose of A/B testing?",
            "How does caching improve performance?",
            "How does public-key encryption differ from symmetric keys?",
            "Why write unit tests?",
        ],
    },
}


@dataclass
class Specialist:
    """Lightweight specialist wrapper around a sentence-transformer model."""

    name: str
    domain: str
    model: SentenceTransformer
    sigma2_base: float

    def embed(self, text: str) -> np.ndarray:
        prefixed = f"[{self.domain} specialist] {text}"
        return self.model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True)

    def sigma2_for(self, text: str) -> float:
        text_l = text.lower()
        domain_bonus = 0.6 if self.domain in text_l else 1.0
        keyword_bonus = 0.7 if any(k in text_l for k in self.domain_keywords()) else 1.0
        return max(self.sigma2_base * domain_bonus * keyword_bonus, 1e-6)

    def domain_keywords(self) -> Sequence[str]:
        if self.domain == "legal":
            return ("court", "contract", "law", "claim", "judge")
        if self.domain == "medical":
            return ("patient", "treatment", "disease", "clinical", "diagnosis")
        return ("algorithm", "api", "model", "system", "code")


class KalmanFuser:
    """Kalman fuser backed by `kalman_fuse_diagonal`."""

    def __init__(self, sort_by_certainty: bool = True, epsilon: float = 1e-8):
        self.sort_by_certainty = sort_by_certainty
        self.epsilon = epsilon

    def fuse(self, text: str, specialists: Sequence[Specialist]) -> np.ndarray:
        embeddings = [sp.embed(text).astype(np.float64) for sp in specialists]
        covariances = [
            np.full(emb.shape, sp.sigma2_for(text), dtype=np.float64)
            for emb, sp in zip(embeddings, specialists)
        ]
        fused, _ = kalman_fuse_diagonal(
            embeddings,
            covariances,
            sort_by_certainty=self.sort_by_certainty,
            epsilon=self.epsilon,
        )
        return fused


class SimpleMeanFuser:
    """Simple arithmetic mean over specialist embeddings."""

    def fuse(self, text: str, specialists: Sequence[Specialist]) -> np.ndarray:
        embeddings = np.stack([sp.embed(text) for sp in specialists], axis=0)
        return embeddings.mean(axis=0)


def build_or_load_dataset(dataset_path: Path) -> Dict[str, object]:
    """Load cached mixed-domain data or create it deterministically."""
    if dataset_path.exists():
        return json.loads(dataset_path.read_text(encoding="utf-8"))

    rng = random.Random(1337)
    documents = []
    queries = []

    for domain, payload in DOMAINS.items():
        docs = payload["docs"]
        domain_queries = payload["queries"]
        for doc_text in docs:
            documents.append({"domain": domain, "text": doc_text})

        start_idx = len(documents) - len(docs)
        for i, q in enumerate(domain_queries):
            doc_id = start_idx + (i % len(docs))
            paraphrase = q
            if rng.random() < 0.3:
                paraphrase = f"{q} Please answer briefly."
            queries.append(
                {
                    "query_id": f"{domain}_{i:03d}",
                    "domain": domain,
                    "text": paraphrase,
                    "relevant_doc_id": doc_id,
                }
            )

    dataset = {"documents": documents, "queries": queries}
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    return dataset


def cosine_scores(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    return d @ q


def query_metrics(ranked_ids: Sequence[int], true_id: int) -> Tuple[int, int, float]:
    hit1 = int(true_id in ranked_ids[:1])
    hit5 = int(true_id in ranked_ids[:5])
    if true_id in ranked_ids:
        rank = ranked_ids.index(true_id) + 1
        mrr = 1.0 / rank
    else:
        mrr = 0.0
    return hit1, hit5, mrr


def paired_bootstrap(
    kalman_values: Sequence[float],
    mean_values: Sequence[float],
    n_bootstrap: int = 4000,
    seed: int = 7,
) -> Tuple[float, Tuple[float, float], float]:
    """Return observed diff, 95% CI, and two-sided bootstrap p-value."""
    kalman = np.asarray(kalman_values, dtype=np.float64)
    mean = np.asarray(mean_values, dtype=np.float64)
    diffs = kalman - mean
    observed = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diffs[idx])

    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    p_two_sided = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
    p_two_sided = min(p_two_sided, 1.0)
    return observed, (float(ci_low), float(ci_high)), float(p_two_sided)


def load_specialists(model_names: Sequence[str]) -> List[Specialist]:
    """Create 3-5 specialists using lightweight sentence-transformers."""
    domains = ["legal", "medical", "technical", "legal", "technical"]
    sigma_bases = [0.8, 0.75, 0.78, 0.83, 0.81]

    specialists: List[Specialist] = []
    dim = None
    for idx, model_name in enumerate(model_names):
        domain = domains[idx]
        model = SentenceTransformer(model_name)
        cur_dim = model.get_sentence_embedding_dimension()
        if dim is None:
            dim = cur_dim
        elif cur_dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {dim}, got {cur_dim} for {model_name}"
            )
        specialists.append(
            Specialist(
                name=f"{domain}_{Path(model_name).name}_{idx}",
                domain=domain,
                model=model,
                sigma2_base=sigma_bases[idx],
            )
        )
    return specialists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=ROOT / "results" / "milestone_1_3_mixed_domain_dataset.json",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=ROOT / "results" / "milestone_1_3_kalman_vs_mean.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "results" / "milestone_1_3_summary.json",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="SentenceTransformer model name (repeat 3-5 times).",
    )
    args = parser.parse_args()

    model_names = args.model or [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ]

    if not 3 <= len(model_names) <= 5:
        raise ValueError("Please provide between 3 and 5 specialist models.")

    dataset = build_or_load_dataset(args.dataset_path)
    specialists = load_specialists(model_names)

    docs = dataset["documents"]
    queries = dataset["queries"]

    kalman_fuser = KalmanFuser()
    mean_fuser = SimpleMeanFuser()

    kalman_doc_embs = np.stack([kalman_fuser.fuse(d["text"], specialists) for d in docs], axis=0)
    mean_doc_embs = np.stack([mean_fuser.fuse(d["text"], specialists) for d in docs], axis=0)

    rows = []
    kalman_hit1s, mean_hit1s = [], []
    kalman_hit5s, mean_hit5s = [], []
    kalman_mrrs, mean_mrrs = [], []

    for q in queries:
        q_text = q["text"]
        true_id = q["relevant_doc_id"]

        q_kalman = kalman_fuser.fuse(q_text, specialists)
        q_mean = mean_fuser.fuse(q_text, specialists)

        ranked_kalman = list(np.argsort(-cosine_scores(q_kalman, kalman_doc_embs)))
        ranked_mean = list(np.argsort(-cosine_scores(q_mean, mean_doc_embs)))

        k_h1, k_h5, k_mrr = query_metrics(ranked_kalman, true_id)
        m_h1, m_h5, m_mrr = query_metrics(ranked_mean, true_id)

        kalman_hit1s.append(k_h1)
        mean_hit1s.append(m_h1)
        kalman_hit5s.append(k_h5)
        mean_hit5s.append(m_h5)
        kalman_mrrs.append(k_mrr)
        mean_mrrs.append(m_mrr)

        rows.append(
            {
                "domain": q["domain"],
                "query_id": q["query_id"],
                "kalman_hit1": k_h1,
                "mean_hit1": m_h1,
                "kalman_mrr": round(k_mrr, 6),
                "mean_mrr": round(m_mrr, 6),
            }
        )

    observed_diff, confidence_interval, p_value = paired_bootstrap(kalman_mrrs, mean_mrrs)

    kalman_summary = {
        "hit@1": float(np.mean(kalman_hit1s)),
        "hit@5": float(np.mean(kalman_hit5s)),
        "mrr": float(np.mean(kalman_mrrs)),
    }
    mean_summary = {
        "hit@1": float(np.mean(mean_hit1s)),
        "hit@5": float(np.mean(mean_hit5s)),
        "mrr": float(np.mean(mean_mrrs)),
    }
    deltas = {
        "hit@1": kalman_summary["hit@1"] - mean_summary["hit@1"],
        "hit@5": kalman_summary["hit@5"] - mean_summary["hit@5"],
        "mrr": observed_diff,
    }

    if p_value < 0.05 and observed_diff > 0:
        conclusion = "kalman_wins"
    elif p_value < 0.05 and observed_diff < 0:
        conclusion = "mean_wins"
    else:
        conclusion = "tie"

    summary = {
        "overall_accuracy_comparison": {
            "kalman": kalman_summary,
            "mean": mean_summary,
            "delta_kalman_minus_mean": deltas,
        },
        "p_value": p_value,
        "conclusion": conclusion,
        "confidence_interval": {
            "metric": "delta_mrr",
            "lower": confidence_interval[0],
            "upper": confidence_interval[1],
            "level": 0.95,
        },
        "threshold": "p < 0.05",
    }

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "query_id",
                "kalman_hit1",
                "mean_hit1",
                "kalman_mrr",
                "mean_mrr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved detailed metrics: {args.results_csv}")
    print(f"Saved summary: {args.summary_json}")
    print(
        "Conclusion:",
        conclusion,
        f"(delta_mrr={observed_diff:.4f}, p={p_value:.4f}, "
        f"95% CI=[{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])",
    )


if __name__ == "__main__":
    main()
