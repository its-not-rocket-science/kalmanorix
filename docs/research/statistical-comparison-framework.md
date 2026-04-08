# Statistical Comparison Framework for Fusion Methods

This framework defines a **pre-registered**, paired, and multiplicity-aware process for comparing Kalman fusion against baseline fusion methods on the same query set.

## 1) Exact Statistical Tests (and Why)

All significance tests are run on **query-level paired outcomes** (same queries for all methods) to maximize power and eliminate between-query confounding.

### 1.1 Primary estimand

For each metric \(m\) and method pair \((A, B)\), define per-query difference:

\[
\Delta_q^{(m)} = m_q(A) - m_q(B)
\]

The primary effect size is:

\[
\bar{\Delta}^{(m)} = \frac{1}{|Q|}\sum_{q \in Q}\Delta_q^{(m)}
\]

### 1.2 Paired non-parametric hypothesis test

Use a **two-sided Wilcoxon signed-rank test** on \(\Delta_q^{(m)}\) for each pair \((A, B)\), metric \(m\):

- Null: median\((\Delta_q^{(m)}) = 0\)
- Alternative: median\((\Delta_q^{(m)}) \neq 0\)

Why Wilcoxon:

- Query-level deltas for retrieval metrics are often non-normal and skewed.
- It preserves paired structure.
- It is robust to outliers versus paired \(t\)-tests.

If all non-zero paired differences collapse to ties (degenerate case), report test as "not estimable" and rely on bootstrap interval + descriptive statistics.

### 1.3 Bootstrap confidence intervals

Use **paired bootstrap** over queries:

1. Sample \(|Q|\) queries with replacement from \(Q\).
2. Recompute \(\bar{\Delta}^{(m)}\) on sampled pairs.
3. Repeat for \(B=10{,}000\) replicates.
4. Report **BCa 95% CI** (fallback: percentile CI if BCa fails numerically).

Why paired bootstrap:

- Respects dependence induced by common queries.
- Produces interpretable uncertainty intervals for absolute improvement.
- Avoids distributional assumptions.

### 1.4 Multiple-comparison correction

If comparing Kalman against \(k>1\) baselines and/or across multiple primary metrics:

- Define one confirmed family of hypotheses **before running**:
  - Recommended family: all Kalman-vs-baseline tests on primary metrics.
- Control FWER with **Holm-Bonferroni** (strong control, more power than Bonferroni).
- Report both raw and adjusted p-values.

If secondary/exploratory metrics are tested, control FDR separately with Benjamini-Hochberg and explicitly mark as exploratory.

## 2) Implementation Plan

## 2.1 Data contract

For each run, persist a long-format table:

- `query_id`
- `domain`
- `method`
- primary metrics: `recall@1`, `recall@5`, `recall@10`, `mrr`, `ndcg@10`
- optional secondary metrics: `latency_ms`, `flops`, `memory_mb`

The comparison module only accepts datasets where all compared methods share the same `query_id` set (strict paired requirement).

## 2.2 Analysis pipeline (deterministic)

1. **Validate pairing**
   - Assert identical query universe across methods.
   - Abort if mismatch is detected.
2. **Compute per-query deltas**
   - For each metric and baseline, compute \(\Delta_q\) = Kalman - baseline.
3. **Run Wilcoxon**
   - Two-sided Wilcoxon signed-rank.
4. **Run paired bootstrap**
   - 10,000 replicates, fixed seed.
   - Estimate mean delta and BCa 95% CI.
5. **Apply multiplicity correction**
   - Holm across pre-registered primary family.
6. **Emit report artifacts**
   - machine-readable `stats_summary.json`
   - publication table `stats_summary.csv`
   - optional markdown report for docs.

## 2.3 Reproducibility controls

- Pre-register:
  - metric set,
  - query inclusion/exclusion,
  - primary hypothesis family,
  - alpha level (default 0.05),
  - bootstrap settings (`B`, seed).
- Freeze analysis code version and log commit SHA.
- One locked command for final analysis; no manual spreadsheet calculations.

## 3) Clear Criteria for "Kalman Wins"

Kalman is declared to "win" against a baseline for a primary metric only if **all** conditions hold:

1. **Directionality**: observed mean delta \( \bar{\Delta}^{(m)} > 0 \) (higher-is-better metric).
2. **Uncertainty**: 95% paired-bootstrap CI for \( \bar{\Delta}^{(m)} \) excludes 0.
3. **Significance**: Holm-adjusted Wilcoxon p-value < 0.05.
4. **Practical relevance floor**: improvement exceeds pre-registered minimum effect size (example: `ndcg@10` absolute +0.01).

For lower-is-better metrics (latency/FLOPs/memory), invert sign convention or equivalent criteria accordingly.

Overall claim ("Kalman wins study") should be pre-registered as:

- wins on the primary metric (`ndcg@10`) **and**
- does not lose catastrophically on efficiency guardrails (predefined thresholds).

## 4) Reporting Format (tables + CIs)

Do not report means alone. Every comparison table must include:

- method means
- paired mean delta (Kalman - baseline)
- 95% bootstrap CI for delta
- Wilcoxon statistic and raw p-value
- adjusted p-value (Holm)
- decision flag (`win` / `no-win`)

### 4.1 Example primary table schema

| Metric | Baseline | Kalman mean | Baseline mean | Delta | 95% CI (paired bootstrap) | Wilcoxon p | Holm p | Decision |
|---|---|---:|---:|---:|---|---:|---:|---|
| nDCG@10 | MeanFuser | 0.421 | 0.405 | +0.016 | [+0.006, +0.026] | 0.0012 | 0.0036 | win |

### 4.2 Domain-stratified table schema

Report the same columns by domain (medical/legal/finance/...) for diagnosis only. Mark these as secondary unless pre-registered as co-primary.

## 5) Explicit Guardrails Against p-Hacking and Overfitting

1. **Single preregistered protocol** before seeing test outcomes.
2. **Fixed test set lock**: no query edits after first evaluation.
3. **No metric shopping**: primary metric(s) declared in advance.
4. **No optional stopping**: fixed sample size and one final analysis pass.
5. **Family definition locked** before inference; no retroactive regrouping.
6. **Separation of tuning vs evaluation**:
   - hyperparameters tuned only on train/validation,
   - final significance tests only on untouched test split.
7. **Audit trail**:
   - log seeds, commit SHA, protocol SHA, and run timestamp.
8. **Full disclosure**:
   - report all pre-registered comparisons, including null/negative findings.

This protocol makes outcome switching, selective reporting, and repeated re-testing observable and non-compliant.
