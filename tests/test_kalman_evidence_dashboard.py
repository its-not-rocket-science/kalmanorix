from scripts import build_kalman_evidence_dashboard as dashboard


def test_classify_replication_status_not_replicated() -> None:
    status = dashboard._classify_replication_status(
        canonical_verdict="inconclusive_underpowered",
        replication=None,
    )
    assert status == "not_replicated"


def test_classify_replication_status_same_verdict() -> None:
    status = dashboard._classify_replication_status(
        canonical_verdict="inconclusive_underpowered",
        replication={
            "per_run_verdicts": [{"verdict": "inconclusive_underpowered"}],
        },
    )
    assert status == "replicated_same_verdict"


def test_classify_replication_status_mixed_verdict() -> None:
    status = dashboard._classify_replication_status(
        canonical_verdict="inconclusive_underpowered",
        replication={
            "per_run_verdicts": [
                {"verdict": "inconclusive_underpowered"},
                {"verdict": "supported"},
            ],
        },
    )
    assert status == "replicated_mixed_verdict"


def test_classify_replication_status_supported() -> None:
    status = dashboard._classify_replication_status(
        canonical_verdict="supported",
        replication={
            "per_run_verdicts": [{"verdict": "supported"}, {"verdict": "supported"}],
            "sign_consistency_delta_kalman_minus_mean": "all_positive",
            "latency_ratio_consistency": "all_within_threshold",
        },
    )
    assert status == "replicated_supported"


def test_claim_ready_support_is_no_without_confirmatory_evidence() -> None:
    status = dashboard._derive_claim_ready_support(
        canonical_v3_status="placeholder_pending_run",
        canonical_v3_verdict="not_available",
        confirmatory_verdict="missing_confirmatory_evidence",
        kalman_vs_mean_verdict="supported",
        kalman_vs_weighted_mean_verdict="supported",
        kalman_vs_router_only_top1_verdict="supported",
        latency_gate_ok=True,
        replication_status="replicated_supported",
    )
    assert status == "no"


def test_claim_ready_support_is_yes_only_when_all_gates_are_supported() -> None:
    status = dashboard._derive_claim_ready_support(
        canonical_v3_status="claim_ready",
        canonical_v3_verdict="supported",
        confirmatory_verdict="supported",
        kalman_vs_mean_verdict="supported",
        kalman_vs_weighted_mean_verdict="supported",
        kalman_vs_router_only_top1_verdict="supported",
        latency_gate_ok=True,
        replication_status="replicated_supported",
    )
    assert status == "yes"
