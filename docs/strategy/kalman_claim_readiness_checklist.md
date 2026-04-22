# Internal Evidence Memo: Kalman Claim Readiness Checklist

This is the gate for one sentence only:

> "Kalman fusion beats mean."

If any required artifact is missing or any gate fails, that sentence is not allowed.

## Required committed artifacts

### 1) Canonical v3 (must exist and be claim-ready)

- [ ] `results/canonical_benchmark_v3/summary.json` exists and is not a placeholder.
- [ ] `results/canonical_benchmark_v3/report.md` exists.
- [ ] `results/canonical_benchmark_v3/runner_summary.json` exists.
- [ ] `results/canonical_benchmark_v3/runner_details.json` exists.
- [ ] Canonical `benchmark_status` is `claim_ready`.
- [ ] Canonical Kalman-vs-Mean verdict is not `inconclusive_*`.

### 2) Confirmatory slice (pre-registered)

- [ ] Slice verdict is present in committed artifacts (not `missing_confirmatory_evidence`).
- [ ] Slice membership artifact is committed (`query_id` + inclusion/exclusion reason).
- [ ] Slice threshold artifact is committed (`D50`, `U50`, `C33`).
- [ ] Paired slice stats are committed (mean delta, CI, raw/adjusted p, `n_pairs`).
- [ ] Slice latency-ratio calculation is committed.
- [ ] Slice power-adequacy calculation is committed.
- [ ] Slice verdict is `supported` under the locked pre-registered rules.

### 3) Required baseline comparisons (same run family)

- [ ] `kalman_vs_mean` passes all decision checks.
- [ ] `kalman_vs_weighted_mean` passes all decision checks.
- [ ] `kalman_vs_router_only_top1` passes all decision checks.
- [ ] Holm-adjusted p-values are reported for the full required comparison family.
- [ ] No required baseline comparison is `unsupported` or `inconclusive_*`.

### 4) Latency gate

- [ ] Canonical Kalman/Mean latency ratio is committed.
- [ ] Ratio is `<= 1.50` for the claim run.
- [ ] If quality passes but latency fails, the claim remains blocked.

### 5) Replication

- [ ] At least one independent replication artifact is committed under `results/`.
- [ ] Replication uses the same locked protocol and reports protocol fingerprint.
- [ ] Replication reproduces direction and practical magnitude of the Kalman-vs-Mean effect.
- [ ] Replication does not downgrade any required baseline comparison.
- [ ] Replication is not underpowered.

## Claim allowed only if all boxes are checked

- [ ] Every box in sections 1-5 is checked.
- [ ] No box is satisfied by exploratory or synthetic-only evidence.
- [ ] No unresolved placeholder status remains in the evidence dashboard inputs.

If any box above is unchecked, the allowed language is:

- "Kalman quality superiority over mean is not demonstrated yet."

## Common failure modes that do not count as proof

- Positive delta with underpowered verdict.
- Beats mean but loses to `router_only_top1`.
- Wins only in exploratory buckets.
- Wins on quality but fails latency gate badly.
- One-off win that does not replicate.
- Placeholder canonical v3 artifacts treated as final evidence.

## Required references (docs)

- Proof standard: [docs/strategy/kalman-proof-standard.md](kalman-proof-standard.md)
- Pre-registered evaluation protocol: [docs/research/preregistered-evaluation-protocol.md](../research/preregistered-evaluation-protocol.md)
- Pre-registered slice claim spec: [docs/research/preregistered-kalman-vs-mean-slice.md](../research/preregistered-kalman-vs-mean-slice.md)
- Statistical comparison framework: [docs/research/statistical-comparison-framework.md](../research/statistical-comparison-framework.md)
- Results index and interpretation rules: [docs/research/results.md](../research/results.md)

## Required references (artifact directories)

- Canonical claim track: [`results/canonical_benchmark_v3/`](../../results/canonical_benchmark_v3/)
- Evidence dashboard: [`results/kalman_evidence_dashboard/`](../../results/kalman_evidence_dashboard/)
- Latency optimization evidence: [`results/kalman_latency_optimization/`](../../results/kalman_latency_optimization/)
- Prior canonical artifacts for provenance: [`results/canonical_benchmark/`](../../results/canonical_benchmark/), [`results/canonical_benchmark_v2/`](../../results/canonical_benchmark_v2/)
- Exploratory-only synthetic narrowed regime (not claim-closing): [`results/correlation_aware_fusion/`](../../results/correlation_aware_fusion/)
