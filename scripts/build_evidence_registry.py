#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(*parts: str) -> str:
    return str(Path(*parts))


def build_registry() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    claims = []

    cb2 = _load_json(REPO / "results/canonical_benchmark_v2/summary.json") or {}
    cb3 = _load_json(REPO / "results/canonical_benchmark_v3/summary.json")
    matched = _load_json(REPO / "results/matched_compute/summary.json") or {}
    ood = _load_json(REPO / "results/ood_robustness/summary.json") or {}
    ucal = _load_json(REPO / "results/uncertainty_calibration/summary.json") or {}
    uabl = _load_json(REPO / "results/uncertainty_ablation/summary.json") or {}
    cov = _load_json(REPO / "results/kalman_covariance_ablation_v2/summary.json") or {}
    corr = _load_json(REPO / "results/correlation_aware_fusion/summary.json") or {}
    lat = (
        _load_json(REPO / "results/kalman_latency_optimisation/summary.json")
        or _load_json(REPO / "results/kalman_latency_optimization/summary.json")
        or {}
    )
    routing = _load_json(REPO / "results/routing_eval/summary.json") or {}

    claims.append(
        {
            "claim_id": "routing_efficiency",
            "claim_text": "Semantic routing improves compute efficiency in benchmarked settings.",
            "evidence_status": "supported",
            "artifact_paths_used": [
                _artifact("results/routing_eval/summary.json"),
                _artifact("results/efficiency_semantic_routing.json"),
            ],
            "headline_safe_sentence": "Routing efficiency is supported for current benchmark artifacts (about 65% average FLOPs reduction in current setup).",
            "prohibited_overclaims": [
                "Guaranteed production savings across all workloads."
            ],
            "venue_relevance": ["arxiv", "tmlr", "joss"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )

    cb_status = (
        cb3.get("status")
        if isinstance(cb3, dict)
        else cb2.get("verdict", "inconclusive")
    )
    claims.append(
        {
            "claim_id": "kalman_vs_mean_quality",
            "claim_text": "Kalman fusion improves retrieval quality over mean fusion.",
            "evidence_status": "unresolved" if cb_status else "inconclusive",
            "artifact_paths_used": [
                _artifact("results/canonical_benchmark_v2/summary.json")
            ]
            + (
                [_artifact("results/canonical_benchmark_v3/summary.json")]
                if cb3
                else []
            ),
            "headline_safe_sentence": "Current canonical artifacts do not close a Kalman-over-mean quality claim.",
            "prohibited_overclaims": ["Kalman beats mean as a general result."],
            "venue_relevance": ["arxiv", "tmlr"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )

    claims.append(
        {
            "claim_id": "matched_compute_specialists_vs_monolith",
            "claim_text": "Fused specialists outperform monolith at matched compute.",
            "evidence_status": matched.get("rule_based_verdict", {}).get(
                "overall", "inconclusive"
            ),
            "artifact_paths_used": [_artifact("results/matched_compute/summary.json")],
            "headline_safe_sentence": "Matched-compute artifact is mixed and does not establish superiority.",
            "prohibited_overclaims": [
                "Specialists are better than monolith at equal compute."
            ],
            "venue_relevance": ["arxiv", "tmlr"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )

    claims.append(
        {
            "claim_id": "ood_uncertainty_weighting",
            "claim_text": "Uncertainty weighting improves OOD robustness.",
            "evidence_status": ood.get("rule_based_verdict", {}).get(
                "overall", "inconclusive"
            ),
            "artifact_paths_used": [_artifact("results/ood_robustness/summary.json")],
            "headline_safe_sentence": "OOD robustness evidence is currently inconclusive with explicit null/regression slots.",
            "prohibited_overclaims": ["OOD robustness is proven improved."],
            "venue_relevance": ["arxiv", "tmlr"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )

    claims.append(
        {
            "claim_id": "uncertainty_calibration_downstream",
            "claim_text": "Powered uncertainty calibration improves downstream retrieval.",
            "evidence_status": "null",
            "artifact_paths_used": [
                _artifact("results/uncertainty_calibration/summary.json")
            ],
            "headline_safe_sentence": "Calibration is powered but currently shows null downstream quality delta.",
            "prohibited_overclaims": [
                "Calibration improves retrieval quality in this artifact."
            ],
            "venue_relevance": ["arxiv", "tmlr", "joss"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )
    claims.append(
        {
            "claim_id": "uncertainty_ablation_downstream",
            "claim_text": "Alternative uncertainty estimators improve downstream retrieval.",
            "evidence_status": "inconclusive",
            "artifact_paths_used": [
                _artifact("results/uncertainty_ablation/summary.json")
            ],
            "headline_safe_sentence": "Uncertainty ablation shows limited downstream gains in this setup.",
            "prohibited_overclaims": [
                "Uncertainty ablation proves quality improvement."
            ],
            "venue_relevance": ["arxiv", "tmlr"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )
    claims.append(
        {
            "claim_id": "covariance_ablation_value",
            "claim_text": "Richer covariance models are justified by practical gains.",
            "evidence_status": "null",
            "artifact_paths_used": [
                _artifact("results/kalman_covariance_ablation_v2/summary.json")
            ],
            "headline_safe_sentence": "Richer covariance variants are not currently justified by practical gain thresholds.",
            "prohibited_overclaims": [
                "Structured covariance should replace simpler variants."
            ],
            "venue_relevance": ["arxiv", "tmlr"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )
    claims.append(
        {
            "claim_id": "correlation_aware_fusion",
            "claim_text": "Correlation-aware fusion improves over baseline Kalman.",
            "evidence_status": "exploratory",
            "artifact_paths_used": [
                _artifact("results/correlation_aware_fusion/summary.json")
            ],
            "headline_safe_sentence": "Correlation-aware gains are exploratory (best reported ΔMRR@10 = +0.0037) and not headline-safe claim closure.",
            "prohibited_overclaims": [
                "Correlation-aware fusion is proven superior in general."
            ],
            "venue_relevance": ["arxiv", "tmlr"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )
    claims.append(
        {
            "claim_id": "kalman_latency_optimization",
            "claim_text": "Kalman latency optimization satisfies canonical latency constraints.",
            "evidence_status": "regression" if lat else "unresolved",
            "artifact_paths_used": [
                _artifact("results/kalman_latency_optimization/summary.json")
                if (REPO / "results/kalman_latency_optimization/summary.json").exists()
                else _artifact("results/kalman_latency_optimisation/summary.json")
            ],
            "headline_safe_sentence": "Latency improved versus legacy (~2.06x single-query speedup) but canonical latency constraint remains unmet.",
            "prohibited_overclaims": [
                "Kalman is now latency-competitive with mean under canonical guardrails."
            ],
            "venue_relevance": ["arxiv", "tmlr", "joss"],
            "last_generated": now,
            "regenerate_command": "PYTHONPATH=src python scripts/build_evidence_registry.py",
        }
    )

    return {"schema_version": 1, "generated_at": now, "claims": claims}


def main() -> int:
    out = REPO / "results/evidence_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_registry(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
