"""Matched-compute benchmark for specialists-vs-monolith comparisons.

The benchmark enforces explicit compute assumptions and fails fast when those
assumptions are missing or inconsistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DOMAIN_DEFS: dict[str, dict[str, Any]] = {
    "legal": {
        "core": ["contract", "statute", "liability", "court", "clause", "appeal"],
        "intents": {
            "contract_review": ["agreement", "breach", "signature", "term"],
            "case_law": ["precedent", "judgment", "holding", "opinion"],
            "compliance": ["regulation", "filing", "audit", "obligation"],
        },
    },
    "medical": {
        "core": ["patient", "diagnosis", "therapy", "symptom", "clinical", "dose"],
        "intents": {
            "diagnosis": ["differential", "exam", "finding", "indicator"],
            "treatment": ["medication", "protocol", "intervention", "response"],
            "radiology": ["imaging", "lesion", "contrast", "scan"],
        },
    },
    "finance": {
        "core": ["market", "asset", "equity", "yield", "portfolio", "risk"],
        "intents": {
            "valuation": ["discount", "cashflow", "multiple", "intrinsic"],
            "macro": ["inflation", "gdp", "policy", "cycle"],
            "credit": ["default", "rating", "covenant", "leverage"],
        },
    },
    "science": {
        "core": ["experiment", "hypothesis", "method", "sample", "analysis", "result"],
        "intents": {
            "biology": ["cell", "genome", "protein", "pathway"],
            "physics": ["quantum", "field", "particle", "energy"],
            "chemistry": ["reaction", "catalyst", "compound", "solvent"],
        },
    },
}

COMMON_TOKENS = ["question", "explain", "impact", "trend", "risk", "evidence"]


@dataclass(frozen=True)
class Sample:
    text: str
    label: str
    domain: str


@dataclass(frozen=True)
class ComputeBudgetAssumptions:
    """Explicit assumptions used for fairness validation and compute accounting."""

    n_specialists: int
    specialist_params_proxy: int
    monolith_params_proxy: int
    specialist_epochs: int
    monolith_epochs: int
    avg_tokens_per_sample: int
    training_flop_multiplier: int
    specialist_inference_flops_proxy: int
    monolith_inference_flops_proxy: int
    routing_overhead_all_proxy: int
    routing_overhead_semantic_proxy: int
    kalman_fusion_overhead_proxy: int
    semantic_top_k: int
    parity_tolerance: float = 0.01


class SimpleMultinomialNB:
    def __init__(self, alpha: float = 1.0, max_vocab: int | None = None) -> None:
        self.alpha = alpha
        self.max_vocab = max_vocab
        self.class_log_prior_: dict[str, float] = {}
        self.feature_log_prob_: dict[str, dict[str, float]] = {}
        self.vocab_: set[str] = set()
        self.classes_: list[str] = []

    def fit(self, samples: Sequence[Sample], epochs: int = 1) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        class_counts = Counter()
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_token_counts = Counter()
        for _ in range(epochs):
            for sample in samples:
                class_counts[sample.label] += 1
                tokens = sample.text.split()
                token_counts[sample.label].update(tokens)
                total_token_counts.update(tokens)
                self.vocab_.update(tokens)

        if self.max_vocab is not None and len(self.vocab_) > self.max_vocab:
            self.vocab_ = {
                token for token, _ in total_token_counts.most_common(self.max_vocab)
            }

        self.classes_ = sorted(class_counts.keys())
        total = sum(class_counts.values())
        vocab_size = max(len(self.vocab_), 1)

        for class_name in self.classes_:
            self.class_log_prior_[class_name] = math.log(
                class_counts[class_name] / total
            )
            in_vocab_count = sum(
                token_counts[class_name][token] for token in self.vocab_
            )
            denom = in_vocab_count + self.alpha * vocab_size
            self.feature_log_prob_[class_name] = {}
            for token in self.vocab_:
                self.feature_log_prob_[class_name][token] = math.log(
                    (token_counts[class_name][token] + self.alpha) / denom
                )

    def predict_proba(self, text: str, all_classes: Sequence[str]) -> np.ndarray:
        if not self.classes_:
            raise RuntimeError("Model must be fitted before prediction")
        token_counts = Counter(text.split())
        scores: dict[str, float] = {}
        for class_name in self.classes_:
            score = self.class_log_prior_[class_name]
            log_probs = self.feature_log_prob_[class_name]
            for token, count in token_counts.items():
                if token in log_probs:
                    score += count * log_probs[token]
            scores[class_name] = score

        max_score = max(scores.values())
        exps = {name: math.exp(value - max_score) for name, value in scores.items()}
        norm = sum(exps.values()) + 1e-12
        probs = {name: exps[name] / norm for name in exps}

        output = np.zeros(len(all_classes), dtype=np.float64)
        class_to_index = {name: idx for idx, name in enumerate(all_classes)}
        for class_name, prob in probs.items():
            output[class_to_index[class_name]] = prob
        return output


def _build_label(domain: str, intent: str) -> str:
    return f"{domain}:{intent}"


def _generate_domain_samples(
    domain: str,
    n_samples: int,
    rng: random.Random,
    all_domains: Sequence[str],
) -> list[Sample]:
    domain_info = DOMAIN_DEFS[domain]
    intents = list(domain_info["intents"].keys())
    out: list[Sample] = []
    for _ in range(n_samples):
        intent = rng.choice(intents)
        tokens: list[str] = []
        tokens += rng.choices(domain_info["core"], k=4)
        tokens += rng.choices(domain_info["intents"][intent], k=4)
        tokens += rng.choices(COMMON_TOKENS, k=2)
        distractor_domain = rng.choice([d for d in all_domains if d != domain])
        tokens += rng.choices(DOMAIN_DEFS[distractor_domain]["core"], k=2)
        rng.shuffle(tokens)
        out.append(
            Sample(
                text=" ".join(tokens), label=_build_label(domain, intent), domain=domain
            )
        )
    return out


def _generate_mixed_test_set(
    domains: Sequence[str], n_total: int, rng: random.Random
) -> list[Sample]:
    samples: list[Sample] = []
    for _ in range(n_total):
        domain = rng.choice(list(domains))
        intent = rng.choice(list(DOMAIN_DEFS[domain]["intents"].keys()))
        tokens = []
        tokens += rng.choices(DOMAIN_DEFS[domain]["core"], k=3)
        tokens += rng.choices(DOMAIN_DEFS[domain]["intents"][intent], k=3)
        tokens += rng.choices(COMMON_TOKENS, k=2)
        other = rng.choice([d for d in domains if d != domain])
        tokens += rng.choices(DOMAIN_DEFS[other]["core"], k=4)
        rng.shuffle(tokens)
        samples.append(
            Sample(
                text=" ".join(tokens), label=_build_label(domain, intent), domain=domain
            )
        )
    return samples


def _estimate_training_flops(
    *,
    n_params: int,
    n_tokens: int,
    epochs: int,
    multiplier: int,
) -> int:
    return int(multiplier * n_params * n_tokens * epochs)


def _entropy_uncertainty(prob: np.ndarray) -> float:
    nz = prob[prob > 0]
    entropy = -float(np.sum(nz * np.log(nz + 1e-12)))
    return entropy / math.log(len(prob) + 1e-12)


def _domain_centroid(samples: Sequence[Sample]) -> Counter[str]:
    centroid: Counter[str] = Counter()
    for sample in samples:
        centroid.update(sample.text.split())
    return centroid


def _centroid_distance(query: str, centroid: Mapping[str, int]) -> float:
    query_counts = Counter(query.split())
    keys = set(query_counts.keys()) | set(centroid.keys())
    if not keys:
        return 1.0
    query_vector = np.array([query_counts[key] for key in keys], dtype=np.float64)
    centroid_vector = np.array([centroid[key] for key in keys], dtype=np.float64)
    query_norm = np.linalg.norm(query_vector) + 1e-12
    centroid_norm = np.linalg.norm(centroid_vector) + 1e-12
    cosine = float(np.dot(query_vector, centroid_vector) / (query_norm * centroid_norm))
    return 1.0 - max(min(cosine, 1.0), -1.0)


def _accuracy_and_mrr(
    probs: Sequence[np.ndarray], labels: Sequence[str], classes: Sequence[str]
) -> tuple[float, float]:
    class_to_index = {name: idx for idx, name in enumerate(classes)}
    correct = 0
    reciprocal_rank_sum = 0.0
    for probability, label in zip(probs, labels):
        label_index = class_to_index[label]
        ranked = np.argsort(-probability)
        if int(ranked[0]) == label_index:
            correct += 1
        rank = int(np.where(ranked == label_index)[0][0]) + 1
        reciprocal_rank_sum += 1.0 / rank
    total = max(len(labels), 1)
    return correct / total, reciprocal_rank_sum / total


def _kalman_fuse_probabilities(
    specialist_probs: Sequence[np.ndarray],
    sigma2s: Sequence[float],
) -> np.ndarray:
    precisions = np.array(
        [1.0 / max(sigma2, 1e-6) for sigma2 in sigma2s], dtype=np.float64
    )
    normalized = precisions / (float(np.sum(precisions)) + 1e-12)
    fused = np.sum(np.stack(specialist_probs, axis=0) * normalized[:, None], axis=0)
    fused = np.maximum(fused, 0.0)
    return fused / (float(np.sum(fused)) + 1e-12)


def _validate_assumptions(assumptions: ComputeBudgetAssumptions) -> None:
    values = asdict(assumptions)
    required_positive = [
        "n_specialists",
        "specialist_params_proxy",
        "monolith_params_proxy",
        "specialist_epochs",
        "monolith_epochs",
        "avg_tokens_per_sample",
        "training_flop_multiplier",
        "specialist_inference_flops_proxy",
        "monolith_inference_flops_proxy",
    ]
    for key in required_positive:
        if values[key] <= 0:
            raise ValueError(f"Compute budget assumption '{key}' must be > 0")

    required_non_negative = [
        "routing_overhead_all_proxy",
        "routing_overhead_semantic_proxy",
        "kalman_fusion_overhead_proxy",
    ]
    for key in required_non_negative:
        if values[key] < 0:
            raise ValueError(f"Compute budget assumption '{key}' must be >= 0")

    if assumptions.semantic_top_k <= 0:
        raise ValueError("semantic_top_k must be positive")
    if assumptions.semantic_top_k > assumptions.n_specialists:
        raise ValueError("semantic_top_k cannot exceed n_specialists")
    if assumptions.parity_tolerance < 0:
        raise ValueError("parity_tolerance must be non-negative")


def _build_default_assumptions(
    *,
    n_specialists: int,
    specialist_epochs: int,
    params_proxy: int,
    avg_tokens_per_sample: int,
    semantic_top_k: int,
) -> ComputeBudgetAssumptions:
    monolith_epochs = specialist_epochs
    specialist_inference = 3 * params_proxy * avg_tokens_per_sample
    monolith_inference = specialist_inference
    return ComputeBudgetAssumptions(
        n_specialists=n_specialists,
        specialist_params_proxy=params_proxy,
        monolith_params_proxy=params_proxy,
        specialist_epochs=specialist_epochs,
        monolith_epochs=monolith_epochs,
        avg_tokens_per_sample=avg_tokens_per_sample,
        training_flop_multiplier=6,
        specialist_inference_flops_proxy=specialist_inference,
        monolith_inference_flops_proxy=monolith_inference,
        routing_overhead_all_proxy=int(0.12 * specialist_inference),
        routing_overhead_semantic_proxy=int(0.20 * specialist_inference),
        kalman_fusion_overhead_proxy=int(0.10 * specialist_inference),
        semantic_top_k=semantic_top_k,
        parity_tolerance=0.01,
    )


def run_matched_compute_benchmark(
    output_dir: Path,
    *,
    seed: int = 7,
    samples_per_domain: int = 1200,
    test_size: int = 600,
    semantic_top_k: int = 2,
) -> dict[str, Any]:
    """Run specialists-vs-monolith benchmark with explicit compute parity checks."""

    domains = sorted(DOMAIN_DEFS.keys())
    rng = random.Random(seed)

    all_classes = [
        _build_label(domain, intent)
        for domain in domains
        for intent in DOMAIN_DEFS[domain]["intents"].keys()
    ]
    params_proxy = len(all_classes) * 1024
    assumptions = _build_default_assumptions(
        n_specialists=len(domains),
        specialist_epochs=1,
        params_proxy=params_proxy,
        avg_tokens_per_sample=12,
        semantic_top_k=semantic_top_k,
    )
    _validate_assumptions(assumptions)

    train_by_domain: dict[str, list[Sample]] = {
        domain: _generate_domain_samples(domain, samples_per_domain, rng, domains)
        for domain in domains
    }
    test_samples = _generate_mixed_test_set(domains, test_size, rng)

    specialists: dict[str, SimpleMultinomialNB] = {}
    centroids: dict[str, Counter[str]] = {}
    for domain in domains:
        model = SimpleMultinomialNB(alpha=0.5)
        model.fit(train_by_domain[domain], epochs=assumptions.specialist_epochs)
        specialists[domain] = model
        centroids[domain] = _domain_centroid(train_by_domain[domain])

    monolith = SimpleMultinomialNB(alpha=0.5, max_vocab=80)
    monolith.fit(
        [sample for domain in domains for sample in train_by_domain[domain]],
        epochs=assumptions.monolith_epochs,
    )

    train_tokens_per_specialist = samples_per_domain * assumptions.avg_tokens_per_sample
    train_tokens_monolith = (
        assumptions.n_specialists
        * samples_per_domain
        * assumptions.avg_tokens_per_sample
    )
    specialist_training_budget = assumptions.n_specialists * _estimate_training_flops(
        n_params=assumptions.specialist_params_proxy,
        n_tokens=train_tokens_per_specialist,
        epochs=assumptions.specialist_epochs,
        multiplier=assumptions.training_flop_multiplier,
    )
    monolith_training_budget = _estimate_training_flops(
        n_params=assumptions.monolith_params_proxy,
        n_tokens=train_tokens_monolith,
        epochs=assumptions.monolith_epochs,
        multiplier=assumptions.training_flop_multiplier,
    )

    train_ratio = specialist_training_budget / max(monolith_training_budget, 1)
    if abs(1.0 - train_ratio) > assumptions.parity_tolerance:
        raise ValueError(
            "Training compute parity check failed: "
            f"ratio={train_ratio:.4f}, tolerance={assumptions.parity_tolerance:.4f}"
        )

    labels = [sample.label for sample in test_samples]
    strategies: dict[str, dict[str, Any]] = {
        "monolith_baseline": {"probs": [], "inference_flops": [], "active": []},
        "specialists_all_routing": {"probs": [], "inference_flops": [], "active": []},
        "specialists_semantic_routing": {
            "probs": [],
            "inference_flops": [],
            "active": [],
        },
        "specialists_kalman_fusion": {"probs": [], "inference_flops": [], "active": []},
    }

    for sample in test_samples:
        mono_prob = monolith.predict_proba(sample.text, all_classes)
        strategies["monolith_baseline"]["probs"].append(mono_prob)
        strategies["monolith_baseline"]["inference_flops"].append(
            assumptions.monolith_inference_flops_proxy
        )
        strategies["monolith_baseline"]["active"].append(1)

        specialist_outputs: dict[str, np.ndarray] = {
            domain: model.predict_proba(sample.text, all_classes)
            for domain, model in specialists.items()
        }

        all_probs = list(specialist_outputs.values())
        mean_prob = np.mean(np.stack(all_probs, axis=0), axis=0)
        strategies["specialists_all_routing"]["probs"].append(mean_prob)
        strategies["specialists_all_routing"]["inference_flops"].append(
            assumptions.routing_overhead_all_proxy
            + assumptions.n_specialists * assumptions.specialist_inference_flops_proxy
        )
        strategies["specialists_all_routing"]["active"].append(
            assumptions.n_specialists
        )

        ranked_domains = sorted(
            domains,
            key=lambda domain: _centroid_distance(sample.text, centroids[domain]),
        )
        selected_domains = ranked_domains[: assumptions.semantic_top_k]
        selected_probs = [specialist_outputs[domain] for domain in selected_domains]
        semantic_prob = np.mean(np.stack(selected_probs, axis=0), axis=0)
        strategies["specialists_semantic_routing"]["probs"].append(semantic_prob)
        strategies["specialists_semantic_routing"]["inference_flops"].append(
            assumptions.routing_overhead_semantic_proxy
            + assumptions.semantic_top_k * assumptions.specialist_inference_flops_proxy
        )
        strategies["specialists_semantic_routing"]["active"].append(
            assumptions.semantic_top_k
        )

        sigma2s = []
        for domain in selected_domains:
            probability = specialist_outputs[domain]
            entropy = _entropy_uncertainty(probability)
            ood = _centroid_distance(sample.text, centroids[domain])
            sigma2s.append(0.05 + 0.7 * entropy + 0.8 * max(0.0, ood))
        kalman_prob = _kalman_fuse_probabilities(selected_probs, sigma2s)
        strategies["specialists_kalman_fusion"]["probs"].append(kalman_prob)
        strategies["specialists_kalman_fusion"]["inference_flops"].append(
            assumptions.routing_overhead_semantic_proxy
            + assumptions.kalman_fusion_overhead_proxy
            + assumptions.semantic_top_k * assumptions.specialist_inference_flops_proxy
        )
        strategies["specialists_kalman_fusion"]["active"].append(
            assumptions.semantic_top_k
        )

    rows: list[dict[str, Any]] = []
    for strategy_name, payload in strategies.items():
        accuracy, mrr = _accuracy_and_mrr(payload["probs"], labels, all_classes)
        mean_inference_flops = float(np.mean(payload["inference_flops"]))
        mean_active = float(np.mean(payload["active"]))
        rows.append(
            {
                "strategy": strategy_name,
                "accuracy_at_1": accuracy,
                "mrr": mrr,
                "training_budget_proxy": (
                    monolith_training_budget
                    if strategy_name == "monolith_baseline"
                    else specialist_training_budget
                ),
                "inference_flops_proxy_mean": mean_inference_flops,
                "active_specialists_mean": mean_active,
                "routing_overhead_proxy": (
                    0
                    if strategy_name == "monolith_baseline"
                    else (
                        assumptions.routing_overhead_all_proxy
                        if strategy_name == "specialists_all_routing"
                        else assumptions.routing_overhead_semantic_proxy
                    )
                ),
            }
        )

    monolith_infer = next(
        row["inference_flops_proxy_mean"]
        for row in rows
        if row["strategy"] == "monolith_baseline"
    )
    fairness_checks = {
        "training_compute_parity_achieved": abs(1.0 - train_ratio)
        <= assumptions.parity_tolerance,
        "training_compute_ratio_specialists_over_monolith": train_ratio,
        "parity_tolerance": assumptions.parity_tolerance,
        "assumption_validation_passed": True,
        "inference_ratio_vs_monolith": {
            row["strategy"]: row["inference_flops_proxy_mean"]
            / max(monolith_infer, 1.0)
            for row in rows
        },
    }

    supports_strong_conclusion = bool(
        fairness_checks["training_compute_parity_achieved"]
        and all(
            ratio <= 2.0
            for strategy, ratio in fairness_checks[
                "inference_ratio_vs_monolith"
            ].items()
            if strategy != "specialists_all_routing"
        )
    )

    summary = {
        "experiment": "matched_compute_specialists_vs_monolith",
        "domains": domains,
        "samples_per_domain": samples_per_domain,
        "test_size": test_size,
        "assumptions": asdict(assumptions),
        "results": rows,
        "fairness_checks": fairness_checks,
        "supports_strong_conclusion": supports_strong_conclusion,
        "conclusion_note": (
            "Training compute parity was enforced exactly by construction. Inference "
            "costs differ by routing policy; interpret quality deltas together with "
            "inference FLOPs ratios."
            if supports_strong_conclusion
            else "Training parity was enforced, but inference-cost asymmetry remains "
            "material; this run does not support a strong conclusion about overall "
            "efficiency superiority."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Matched Compute: Specialists vs Monolith",
        "",
        "## Assumptions",
        "",
        "- Training budget proxy uses `6 * params_proxy * tokens * epochs`.",
        f"- Specialists: {assumptions.n_specialists} models, {assumptions.specialist_epochs} epoch each.",
        f"- Monolith: {assumptions.monolith_epochs} epochs (chosen to match training compute).",
        f"- Inference FLOPs proxy includes specialist invocation cost and routing overhead.",
        "",
        "## Results",
        "",
        "| Strategy | Acc@1 | MRR | Train Budget Proxy | Inference FLOPs Proxy (mean) | Active Specialists (mean) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {strategy} | {accuracy_at_1:.4f} | {mrr:.4f} | {training_budget_proxy} | "
            "{inference_flops_proxy_mean:.1f} | {active_specialists_mean:.2f} |".format(
                **row
            )
        )

    report_lines.extend(
        [
            "",
            "## Fairness checks",
            "",
            f"- Training compute parity achieved: **{fairness_checks['training_compute_parity_achieved']}** ",
            f"(ratio={fairness_checks['training_compute_ratio_specialists_over_monolith']:.4f}, "
            f"tolerance={fairness_checks['parity_tolerance']:.4f}).",
            "- Validation checks passed: **True** (missing/inconsistent assumptions raise errors).",
            "- Inference FLOPs ratios vs monolith:",
        ]
    )
    for strategy, ratio in fairness_checks["inference_ratio_vs_monolith"].items():
        report_lines.append(f"  - `{strategy}`: {ratio:.3f}x")

    report_lines.extend(
        [
            "",
            "## Conclusion quality",
            "",
            summary["conclusion_note"],
        ]
    )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary
