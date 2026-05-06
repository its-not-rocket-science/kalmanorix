# Publication Plan: Mixed-Domain Retrieval Negative Result

## Working Claim (Scoped and Falsifiable)
In this claim-ready mixed-domain retrieval benchmark, Kalman fusion does not outperform mean fusion, and a hard routing strategy is stronger on the primary ranking metric.

This claim is intentionally benchmark-scoped. It does **not** imply Kalman-style fusion is universally ineffective across all retrieval settings.

## Recommended Publication Path

### Primary path
1. **arXiv preprint first**
   - Publish a transparent, timestamped empirical report with full methods, artifacts, and limitations.
   - Use the preprint to establish the negative result clearly and invite community scrutiny.

2. **TMLR submission next (conditional on strengthening the empirical paper)**
   - Proceed if the manuscript is improved on reproducibility quality, analysis depth, and limitations framing.
   - Position the contribution as careful empirical science: strong baselines, statistical discipline, and practical implications.

### Non-primary path
- **JOSS only as a separate software paper**
  - If pursued, scope JOSS around software and reproducibility infrastructure.
  - Do **not** position JOSS as the main venue for the empirical negative-result claim.

## Current Evidence Summary
- `benchmark_status = claim_ready`
- Paired test size: `n = 1193` queries
- Kalman vs mean at nDCG@10: delta approximately `-0.000203`
- Holm-adjusted p-value: `1.0`
- `router_only_top1` currently ranks first by nDCG@10
- Confirmatory slice is empty / underpowered and cannot support additional confirmatory claims

Interpretation: for this benchmark and metric, observed Kalman-vs-mean differences are effectively null-to-negative, while hard routing shows stronger point performance.

## Central Contribution Framing

### 1) Negative empirical result
A well-powered paired comparison on a claim-ready benchmark finds no measurable nDCG@10 gain from Kalman fusion over mean fusion.

### 2) Strong baseline discipline
The study emphasizes rigorous baseline selection (including simple and hard-routing alternatives), avoiding novelty bias.

### 3) Benchmark and artifact governance
The work foregrounds claim-readiness criteria, transparent reporting, and reproducible evaluation artifacts.

### 4) Practical lesson (scope-limited)
In this setting, model **selection/routing** appears to matter more than fusion weight adaptation.

## Blockers Before Submission
1. **Fix encoding mojibake in the report** so tables/text render correctly and are citable.
2. **Ensure reproducible artifact commands** (single-source, copy-paste executable paths/CLI invocations).
3. **Add exact hardware/runtime notes** (CPU/GPU model, memory, software versions, wall-clock conventions).
4. **Preserve timing JSON outputs** in versioned artifacts for auditability and replication.
5. **Add a dedicated limitations section**, including:
   - benchmark scope,
   - confirmatory slice underpowering,
   - metric sensitivity,
   - and explicit non-universality of conclusions.

## Suggested Paper Positioning Language
- "We report a benchmark-scoped negative result: Kalman fusion does not improve over mean fusion on our mixed-domain claim-ready benchmark."
- "Our results indicate that hard routing is stronger than both fusion variants for nDCG@10 in this setting."
- "We do not claim Kalman fusion is generally ineffective; conclusions are limited to the tested benchmark, protocol, and metrics."

## Submission Readiness Checklist
- [ ] All figures/tables free of encoding artifacts
- [ ] End-to-end reproduction commands validated from a clean environment
- [ ] Hardware/runtime and seed policies fully documented
- [ ] Timing JSON and statistical outputs archived with immutable references
- [ ] Limitations and external validity section finalized
- [ ] Abstract and conclusion wording audited to avoid overclaiming
