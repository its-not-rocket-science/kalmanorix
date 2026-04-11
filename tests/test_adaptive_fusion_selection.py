from experiments.run_adaptive_fusion_selection import PolicyConfig, _select_mode


def test_select_mode_prefers_hard_routing_on_clear_case() -> None:
    cfg = PolicyConfig(
        high_gap=0.45,
        low_gap=0.15,
        agreement_high=0.90,
        agreement_low=0.60,
        spread_high=0.35,
        spread_low=0.12,
        kalman_ambiguity="high",
    )
    signals = {
        "selected_count": 1,
        "confidence_gap": 0.70,
        "specialist_agreement": 0.95,
        "uncertainty_spread": 0.05,
        "domain_ambiguity_bucket": "low",
    }
    assert _select_mode(signals, cfg) == "hard_routing"


def test_select_mode_can_force_kalman_on_ambiguity() -> None:
    cfg = PolicyConfig(
        high_gap=0.45,
        low_gap=0.15,
        agreement_high=0.90,
        agreement_low=0.60,
        spread_high=0.35,
        spread_low=0.12,
        kalman_ambiguity="high",
    )
    signals = {
        "selected_count": 2,
        "confidence_gap": 0.40,
        "specialist_agreement": 0.82,
        "uncertainty_spread": 0.10,
        "domain_ambiguity_bucket": "high",
    }
    assert _select_mode(signals, cfg) == "kalman_fusion"


def test_select_mode_defaults_to_mean_for_middle_case() -> None:
    cfg = PolicyConfig(
        high_gap=0.45,
        low_gap=0.15,
        agreement_high=0.90,
        agreement_low=0.60,
        spread_high=0.35,
        spread_low=0.12,
        kalman_ambiguity="high",
    )
    signals = {
        "selected_count": 2,
        "confidence_gap": 0.30,
        "specialist_agreement": 0.80,
        "uncertainty_spread": 0.20,
        "domain_ambiguity_bucket": "medium",
    }
    assert _select_mode(signals, cfg) == "mean_fusion"
