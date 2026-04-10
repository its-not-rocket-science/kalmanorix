from __future__ import annotations

import json
from pathlib import Path

from experiments.run_adaptive_route_or_fuse import run_adaptive_route_or_fuse


def test_adaptive_route_or_fuse_writes_policy_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    details = {
        "results": {
            "router_only_top1": {"global_primary": {"mrr": {"mean": 0.4}, "recall@1": {"mean": 0.2}, "recall@5": {"mean": 0.6}}},
            "uniform_mean_fusion": {"global_primary": {"mrr": {"mean": 0.5}, "recall@1": {"mean": 0.3}, "recall@5": {"mean": 0.7}}},
            "kalman": {"global_primary": {"mrr": {"mean": 0.52}, "recall@1": {"mean": 0.32}, "recall@5": {"mean": 0.72}}},
            "adaptive_route_or_fuse": {"global_primary": {"mrr": {"mean": 0.55}, "recall@1": {"mean": 0.4}, "recall@5": {"mean": 0.75}}},
        },
        "query_level": {
            "ground_truth": {"q1": ["d1"], "q2": ["d2"]},
            "rankings": {
                "adaptive_route_or_fuse": {"q1": ["d1", "d2"], "q2": ["d1", "d2"]}
            },
            "policy_usage": {
                "adaptive_route_or_fuse": {
                    "q1": {"mode": "hard_routing", "signals": {}},
                    "q2": {"mode": "kalman_fusion", "signals": {}},
                }
            },
        },
    }

    monkeypatch.setattr(
        "experiments.run_adaptive_route_or_fuse.run_experiment", lambda cfg: details
    )

    output_dir = tmp_path / "adaptive"
    summary = run_adaptive_route_or_fuse(
        benchmark_path=tmp_path / "bench.json",
        output_dir=output_dir,
        split="test",
        max_queries=2,
        device="cpu",
        seed=1,
        selector_type="rule",
    )

    assert summary["selection_frequencies"] == {"hard_routing": 1, "kalman_fusion": 1}
    assert output_dir.joinpath("summary.json").exists()
    assert output_dir.joinpath("report.md").exists()

    on_disk = json.loads(output_dir.joinpath("summary.json").read_text(encoding="utf-8"))
    assert "per_mode_outcomes" in on_disk
