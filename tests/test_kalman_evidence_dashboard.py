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
