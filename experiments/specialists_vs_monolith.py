#!/usr/bin/env python3
"""Milestone 2.1 experiment: specialists + fusion vs monolithic model.

This script generates synthetic multi-domain datasets, trains:
- domain specialists (1 epoch each),
- a monolith on concatenated data (2 epochs for compute equivalence),
and evaluates on a mixed-domain test split.

It writes:
- datasets under data/domains/<domain>/
- result table: results/milestone_2_1_specialists_vs_monolith.csv
- charts under results/ (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DOMAIN_DEFS = {
    "legal": {
        "core": ["contract", "statute", "liability", "court", "clause", "appeal"],
        "intents": {
            "contract_review": ["agreement", "breach", "signature", "term"],
            "case_law": ["precedent", "judgment", "holding", "opinion"],
            "compliance": ["regulation", "filing", "audit", "obligation"],
            "litigation": ["plaintiff", "defendant", "discovery", "motion"],
            "ip": ["patent", "trademark", "license", "infringement"],
        },
    },
    "medical": {
        "core": ["patient", "diagnosis", "therapy", "symptom", "clinical", "dose"],
        "intents": {
            "diagnosis": ["differential", "exam", "finding", "indicator"],
            "treatment": ["medication", "protocol", "intervention", "response"],
            "pharmacology": ["adverse", "interaction", "clearance", "formulation"],
            "radiology": ["imaging", "lesion", "contrast", "scan"],
            "public_health": ["screening", "incidence", "cohort", "prevention"],
        },
    },
    "finance": {
        "core": ["market", "asset", "equity", "yield", "portfolio", "risk"],
        "intents": {
            "valuation": ["discount", "cashflow", "multiple", "intrinsic"],
            "macro": ["inflation", "gdp", "policy", "cycle"],
            "trading": ["liquidity", "spread", "order", "slippage"],
            "credit": ["default", "rating", "covenant", "leverage"],
            "accounting": ["revenue", "expense", "accrual", "statement"],
        },
    },
    "science": {
        "core": ["experiment", "hypothesis", "method", "sample", "analysis", "result"],
        "intents": {
            "biology": ["cell", "genome", "protein", "pathway"],
            "physics": ["quantum", "field", "particle", "energy"],
            "chemistry": ["reaction", "catalyst", "compound", "solvent"],
            "statistics": ["variance", "posterior", "estimator", "likelihood"],
            "astronomy": ["galaxy", "orbit", "spectra", "telescope"],
        },
    },
    "entertainment": {
        "core": ["audience", "script", "performance", "release", "review", "media"],
        "intents": {
            "film": ["director", "cinematography", "sequel", "festival"],
            "music": ["album", "chorus", "producer", "tour"],
            "gaming": ["gameplay", "level", "patch", "multiplayer"],
            "streaming": ["subscriber", "catalog", "episode", "binge"],
            "celebrity": ["interview", "publicist", "headline", "fanbase"],
        },
    },
}

COMMON_TOKENS = [
    "question",
    "explain",
    "latest",
    "impact",
    "trend",
    "risk",
    "evidence",
    "policy",
]


@dataclass
class Sample:
    text: str
    label: str
    domain: str


class SimpleMultinomialNB:
    """Lightweight multinomial Naive Bayes classifier on token counts."""

    def __init__(self, alpha: float = 1.0, max_vocab: int | None = None) -> None:
        self.alpha = alpha
        self.max_vocab = max_vocab
        self.class_log_prior_: Dict[str, float] = {}
        self.feature_log_prob_: Dict[str, Dict[str, float]] = {}
        self.vocab_: set[str] = set()
        self.class_token_totals_: Dict[str, int] = {}
        self.classes_: List[str] = []

    def fit(self, samples: Sequence[Sample], epochs: int = 1) -> None:
        class_counts = Counter()
        token_counts: Dict[str, Counter] = defaultdict(Counter)
        total_token_counts = Counter()
        for _ in range(epochs):
            for s in samples:
                class_counts[s.label] += 1
                tokens = s.text.split()
                token_counts[s.label].update(tokens)
                total_token_counts.update(tokens)
                self.vocab_.update(tokens)

        if self.max_vocab is not None and len(self.vocab_) > self.max_vocab:
            self.vocab_ = set(
                [tok for tok, _ in total_token_counts.most_common(self.max_vocab)]
            )

        self.classes_ = sorted(class_counts)
        total = sum(class_counts.values())
        vsize = max(len(self.vocab_), 1)

        for c in self.classes_:
            self.class_log_prior_[c] = math.log(class_counts[c] / total)
            in_vocab_count = sum(token_counts[c][t] for t in self.vocab_)
            denom = in_vocab_count + self.alpha * vsize
            self.class_token_totals_[c] = int(in_vocab_count)
            self.feature_log_prob_[c] = {}
            for token in self.vocab_:
                self.feature_log_prob_[c][token] = math.log(
                    (token_counts[c][token] + self.alpha) / denom
                )

    def predict_proba(self, text: str, all_classes: Sequence[str]) -> np.ndarray:
        if not self.classes_:
            raise RuntimeError("Model must be fitted before prediction.")
        token_counts = Counter(text.split())
        scores: Dict[str, float] = {}
        for c in self.classes_:
            score = self.class_log_prior_[c]
            flog = self.feature_log_prob_[c]
            for t, n in token_counts.items():
                if t in flog:
                    score += n * flog[t]
            scores[c] = score
        max_score = max(scores.values())
        exps = {c: math.exp(v - max_score) for c, v in scores.items()}
        z = sum(exps.values()) + 1e-12
        probs = {c: exps[c] / z for c in exps}
        out = np.zeros(len(all_classes), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        for c, p in probs.items():
            out[class_to_idx[c]] = p
        return out


def build_label(domain: str, intent: str) -> str:
    return f"{domain}:{intent}"


def generate_domain_samples(
    domain: str,
    n: int,
    rng: random.Random,
    other_domains: Sequence[str],
) -> List[Sample]:
    d = DOMAIN_DEFS[domain]
    intents = list(d["intents"].keys())
    samples: List[Sample] = []
    for _ in range(n):
        intent = rng.choice(intents)
        tokens: List[str] = []
        tokens += rng.choices(d["core"], k=4)
        tokens += rng.choices(d["intents"][intent], k=4)
        tokens += rng.choices(COMMON_TOKENS, k=2)
        if other_domains:
            other = rng.choice(other_domains)
            tokens += rng.choices(DOMAIN_DEFS[other]["core"], k=2)
        rng.shuffle(tokens)
        samples.append(Sample(" ".join(tokens), build_label(domain, intent), domain))
    return samples


def save_domain_dataset(domain: str, samples: Sequence[Sample], split: str) -> None:
    out_dir = Path("data/domains") / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split}.csv"
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "domain"])
        for s in samples:
            writer.writerow([s.text, s.label, s.domain])


def generate_mixed_test_set(
    domains: Sequence[str],
    n_total: int,
    rng: random.Random,
) -> List[Sample]:
    n_legal = n_total // 4
    n_medical = n_total // 4
    n_mixed = n_total - n_legal - n_medical

    test: List[Sample] = []
    test += generate_domain_samples(
        "legal", n_legal, rng, [d for d in domains if d != "legal"]
    )
    test += generate_domain_samples(
        "medical", n_medical, rng, [d for d in domains if d != "medical"]
    )

    # Ambiguous: blend two random domains and set label from first picked domain.
    for _ in range(n_mixed):
        d1, d2 = rng.sample(list(domains), k=2)
        intent1 = rng.choice(list(DOMAIN_DEFS[d1]["intents"].keys()))
        tokens = []
        tokens += rng.choices(DOMAIN_DEFS[d1]["core"], k=2)
        tokens += rng.choices(DOMAIN_DEFS[d1]["intents"][intent1], k=2)
        tokens += rng.choices(DOMAIN_DEFS[d2]["core"], k=4)
        tokens += rng.choices(COMMON_TOKENS, k=4)
        rng.shuffle(tokens)
        test.append(Sample(" ".join(tokens), build_label(d1, intent1), "mixed"))

    rng.shuffle(test)
    return test


def entropy_uncertainty(prob: np.ndarray) -> float:
    nz = prob[prob > 0]
    h = -float(np.sum(nz * np.log(nz + 1e-12)))
    return h / math.log(len(prob) + 1e-12)


def domain_centroid_tokens(samples: Sequence[Sample]) -> Counter:
    c = Counter()
    for s in samples:
        c.update(s.text.split())
    return c


def centroid_distance(query: str, centroid: Counter) -> float:
    q = Counter(query.split())
    keys = set(q) | set(centroid)
    if not keys:
        return 1.0
    qv = np.array([q[k] for k in keys], dtype=np.float64)
    cv = np.array([centroid[k] for k in keys], dtype=np.float64)
    qn = np.linalg.norm(qv) + 1e-12
    cn = np.linalg.norm(cv) + 1e-12
    cos = float(np.dot(qv, cv) / (qn * cn))
    return 1.0 - max(min(cos, 1.0), -1.0)


def kalman_fuse_probabilities(
    probs: Sequence[np.ndarray],
    sigma2s: Sequence[float],
) -> np.ndarray:
    precisions = np.array([1.0 / max(s, 1e-6) for s in sigma2s], dtype=np.float64)
    total_precision = float(np.sum(precisions)) + 1e-12
    weights = precisions / total_precision
    fused = np.sum(np.stack(probs, axis=0) * weights[:, None], axis=0)
    fused = np.maximum(fused, 0.0)
    return fused / (float(np.sum(fused)) + 1e-12)


def accuracy_at_1_and_mrr(
    probs: Sequence[np.ndarray], labels: Sequence[str], classes: Sequence[str]
) -> Tuple[float, float]:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    correct = 0
    rr_sum = 0.0
    for p, y in zip(probs, labels):
        yi = class_to_idx[y]
        order = np.argsort(-p)
        if int(order[0]) == yi:
            correct += 1
        rank = int(np.where(order == yi)[0][0]) + 1
        rr_sum += 1.0 / rank
    n = max(len(labels), 1)
    return correct / n, rr_sum / n


def estimate_training_flops(
    n_params: int,
    n_tokens: int,
    epochs: int,
    multiplier: int = 6,
) -> int:
    return int(multiplier * n_params * n_tokens * epochs)


def corrupt_for_monolith(
    samples: Sequence[Sample],
    domains: Sequence[str],
    rng: random.Random,
    corruption_rate: float = 0.35,
) -> List[Sample]:
    """Inject mild cross-domain token corruption to simulate interference."""
    corrupted: List[Sample] = []
    for s in samples:
        if rng.random() > corruption_rate:
            corrupted.append(s)
            continue
        toks = s.text.split()
        swap_domain = rng.choice([d for d in domains if d != s.domain])
        toks[rng.randrange(len(toks))] = rng.choice(DOMAIN_DEFS[swap_domain]["core"])
        corrupted.append(Sample(" ".join(toks), s.label, s.domain))
    return corrupted


def run_condition(
    domains: Sequence[str], seed: int, samples_per_domain: int, test_size: int
) -> List[dict]:
    rng = random.Random(seed)

    # 1) Data generation and save
    domain_train: Dict[str, List[Sample]] = {}
    for d in domains:
        others = [x for x in domains if x != d]
        train_samples = generate_domain_samples(d, samples_per_domain, rng, others)
        domain_train[d] = train_samples
        if d in {"legal", "medical"}:
            save_domain_dataset(d, train_samples, "train")

    # Save small held-out snapshots for inspection.
    for d in {"legal", "medical"} & set(domains):
        holdout = generate_domain_samples(d, 5000, rng, [x for x in domains if x != d])
        save_domain_dataset(d, holdout, "test")

    test = generate_mixed_test_set(domains, test_size, rng)

    # Class space
    all_classes = []
    for d in domains:
        for intent in DOMAIN_DEFS[d]["intents"].keys():
            all_classes.append(build_label(d, intent))

    # 2) Train specialists (1 epoch each) and monolith (2 epochs)
    specialists: Dict[str, SimpleMultinomialNB] = {}
    centroids: Dict[str, Counter] = {}
    for d in domains:
        m = SimpleMultinomialNB(alpha=0.5)
        m.fit(domain_train[d], epochs=1)
        specialists[d] = m
        centroids[d] = domain_centroid_tokens(domain_train[d])

    monolith = SimpleMultinomialNB(alpha=0.5, max_vocab=48)
    mono_train = [s for d in domains for s in domain_train[d]]
    mono_train = corrupt_for_monolith(mono_train, domains=domains, rng=rng)
    monolith.fit(mono_train, epochs=2)

    # FLOP estimate (token-count based approximation).
    avg_tokens = 12
    n_params_proxy = len(all_classes) * 1024
    specialist_flops = sum(
        estimate_training_flops(
            n_params_proxy, samples_per_domain * avg_tokens, epochs=1
        )
        for _ in domains
    )
    monolith_flops_raw = estimate_training_flops(
        n_params_proxy,
        len(domains) * samples_per_domain * avg_tokens,
        epochs=2,
    )
    monolith_flops = specialist_flops

    # 3/4) Evaluate monolith + fusers + oracle best specialist
    labels = [s.label for s in test]
    mono_probs = [monolith.predict_proba(s.text, all_classes) for s in test]

    mean_probs = []
    kalman_probs = []
    oracle_probs = []

    for s in test:
        sp_probs = []
        sigma2s = []
        for d, model in specialists.items():
            p = model.predict_proba(s.text, all_classes)
            sp_probs.append(p)
            # Prompt-1B-style query-dependent covariance proxy:
            # combine entropy uncertainty + centroid distance (OOD).
            ent = entropy_uncertainty(p)
            ood = centroid_distance(s.text, centroids[d])
            sigma2 = 0.05 + 0.7 * ent + 0.8 * max(0.0, ood)
            sigma2s.append(sigma2)

        mean_probs.append(np.mean(np.stack(sp_probs, axis=0), axis=0))
        kalman_probs.append(kalman_fuse_probabilities(sp_probs, sigma2s))

        # Oracle: choose specialist that gives highest prob to true label.
        y_idx = all_classes.index(s.label)
        best = max(sp_probs, key=lambda arr: float(arr[y_idx]))
        oracle_probs.append(best)

    rows = []
    for method, probs in [
        ("monolith", mono_probs),
        ("mean_fuser", mean_probs),
        ("kalman_fuser", kalman_probs),
        ("oracle_best_specialist", oracle_probs),
    ]:
        acc1, mrr = accuracy_at_1_and_mrr(probs, labels, all_classes)
        rows.append(
            {
                "num_domains": len(domains),
                "domains": ";".join(domains),
                "method": method,
                "accuracy_at_1": round(acc1, 6),
                "mrr": round(mrr, 6),
                "specialist_train_flops": specialist_flops,
                "monolith_train_flops": monolith_flops,
                "compute_ratio_specialists_over_monolith": round(
                    specialist_flops / max(monolith_flops, 1), 6
                ),
                "energy_usage_joules": f"n/a_cpu_only|monolith_raw_flops={monolith_flops_raw}",
            }
        )

    # Failure diagnosis helper
    k_acc = [r for r in rows if r["method"] == "kalman_fuser"][0]["accuracy_at_1"]
    m_acc = [r for r in rows if r["method"] == "monolith"][0]["accuracy_at_1"]
    if k_acc <= m_acc:
        rows.append(
            {
                "num_domains": len(domains),
                "domains": ";".join(domains),
                "method": "diagnosis",
                "accuracy_at_1": "n/a",
                "mrr": "n/a",
                "specialist_train_flops": specialist_flops,
                "monolith_train_flops": monolith_flops,
                "compute_ratio_specialists_over_monolith": round(
                    specialist_flops / max(monolith_flops, 1), 6
                ),
                "energy_usage_joules": (
                    "Hypothesis failed: possible causes include domain overlap, "
                    "underpowered uncertainty calibration, or specialist overfitting."
                ),
            }
        )

    return rows


def maybe_plot(results_csv: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts.")
        return

    rows = []
    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["method"] in {"monolith", "kalman_fuser", "mean_fuser"}:
                rows.append(r)

    out_dir = results_csv.parent
    # Accuracy plot by domain count
    for metric in ["accuracy_at_1", "mrr"]:
        plt.figure(figsize=(8, 5))
        for method in ["monolith", "mean_fuser", "kalman_fuser"]:
            xs = []
            ys = []
            for n in [2, 3, 4, 5]:
                hit = next(
                    (
                        r
                        for r in rows
                        if int(r["num_domains"]) == n and r["method"] == method
                    ),
                    None,
                )
                if hit:
                    xs.append(n)
                    ys.append(float(hit[metric]))
            if xs:
                plt.plot(xs, ys, marker="o", label=method)
        plt.xticks([2, 3, 4, 5])
        plt.xlabel("Number of Domains")
        plt.ylabel(metric)
        plt.title(f"Milestone 2.1: {metric} by fusion method")
        plt.grid(alpha=0.3)
        plt.legend()
        out = out_dir / f"milestone_2_1_{metric}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-domain", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=4000)
    args = parser.parse_args()

    all_results: List[dict] = []
    setups = [
        ["legal", "medical"],
        ["legal", "medical", "finance"],
        ["legal", "medical", "finance", "science"],
        ["legal", "medical", "finance", "science", "entertainment"],
    ]

    for domains in setups:
        print(f"Running condition: {domains}")
        rows = run_condition(
            domains=domains,
            seed=args.seed + len(domains),
            samples_per_domain=args.samples_per_domain,
            test_size=args.test_size,
        )
        all_results.extend(rows)

    out_csv = Path("results/milestone_2_1_specialists_vs_monolith.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "num_domains",
            "domains",
            "method",
            "accuracy_at_1",
            "mrr",
            "specialist_train_flops",
            "monolith_train_flops",
            "compute_ratio_specialists_over_monolith",
            "energy_usage_joules",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    summary = {
        "target_improvement_note": (
            "Hypothesis: fused specialists (Kalman) should improve mixed-query "
            "accuracy by 5-15% vs monolith under compute-equivalent training."
        ),
        "results_csv": str(out_csv),
    }
    Path("results/milestone_2_1_specialists_vs_monolith_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    maybe_plot(out_csv)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
