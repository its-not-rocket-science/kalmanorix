# Kalman Assumption Stress-Test Slice

This artifact is isolated from canonical benchmarking and is intended for mechanism-level hypothesis testing.

## Guardrails

- Do not use this slice for headline claims unless findings are replicated on broader canonical benchmarks.
- Use this slice to falsify/support assumptions about reliability heterogeneity and expert redundancy.

## Assumption type: strong_specialist_asymmetry

**Stress rationale:** One specialist is clearly better per-query while others are much noisier; tests if uncertainty-aware fusion and routing exploit heterogeneity.

| Method | Recall@1 | Recall@5 | MRR | ΔMRR vs mean |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.3000 | 0.8500 | 0.4958 | +0.0000 |
| hard_routing | 0.5500 | 0.8500 | 0.7051 | +0.2092 |
| scalar_kalman | 0.3000 | 0.7500 | 0.4574 | -0.0385 |
| correlation_aware_kalman | 0.3000 | 0.7500 | 0.4574 | -0.0384 |
| structured_kalman | 0.3000 | 0.8000 | 0.4457 | -0.0501 |

## Assumption type: conflicting_specialists

**Stress rationale:** Specialists are similarly noisy, but one specialist may carry an adversarial bias; tests robustness when experts disagree with structured conflict.

| Method | Recall@1 | Recall@5 | MRR | ΔMRR vs mean |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.2500 | 0.6500 | 0.4435 | +0.0000 |
| hard_routing | 0.1500 | 0.3500 | 0.2762 | -0.1673 |
| scalar_kalman | 0.2500 | 0.6500 | 0.4274 | -0.0161 |
| correlation_aware_kalman | 0.2500 | 0.6500 | 0.4274 | -0.0161 |
| structured_kalman | 0.2500 | 0.5500 | 0.3860 | -0.0576 |

## Assumption type: high_redundancy

**Stress rationale:** Experts are near-identical and share uncertainty; Kalman should offer little gain over mean when assumptions of complementary information are weak.

| Method | Recall@1 | Recall@5 | MRR | ΔMRR vs mean |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.2000 | 0.4500 | 0.3214 | +0.0000 |
| hard_routing | 0.2000 | 0.4500 | 0.3251 | +0.0037 |
| scalar_kalman | 0.2000 | 0.4500 | 0.3214 | +0.0000 |
| correlation_aware_kalman | 0.2000 | 0.4500 | 0.3214 | +0.0000 |
| structured_kalman | 0.2000 | 0.4500 | 0.3169 | -0.0045 |

## Assumption type: low_redundancy_complementary

**Stress rationale:** No single specialist dominates all dimensions of query space; signals are complementary with lower redundancy, where Kalman-style weighting should help.

| Method | Recall@1 | Recall@5 | MRR | ΔMRR vs mean |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.3500 | 0.8500 | 0.5561 | +0.0000 |
| hard_routing | 0.3000 | 0.6500 | 0.4638 | -0.0923 |
| scalar_kalman | 0.3500 | 0.8500 | 0.5518 | -0.0043 |
| correlation_aware_kalman | 0.3500 | 0.8500 | 0.5560 | -0.0001 |
| structured_kalman | 0.4000 | 0.8000 | 0.5685 | +0.0124 |

## Overall slice aggregate

| Method | Recall@1 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| mean | 0.2750 | 0.7000 | 0.4542 |
| hard_routing | 0.3000 | 0.5750 | 0.4425 |
| scalar_kalman | 0.2750 | 0.6750 | 0.4395 |
| correlation_aware_kalman | 0.2750 | 0.6750 | 0.4405 |
| structured_kalman | 0.2875 | 0.6500 | 0.4293 |
