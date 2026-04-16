# Kalmanorix Falsification Experiment Design (Adversarial Protocol)

## Objective

Design an experiment that is **maximally capable of disproving** the Kalmanorix claim that uncertainty-weighted specialist fusion is reliably better than simpler alternatives.

Core falsifiable claim under test:

> Under realistic domain shift, calibration error, and specialist disagreement, Kalman fusion still improves retrieval quality over strong simple baselines.

This protocol is intentionally hostile to that claim. If Kalman fusion wins here, the result is unusually credible. If it loses, the failure is informative and actionable.

---

## 1) Experimental Design

### 1.1 Design principles (anti-hypothesis-protection)

- **Adversarial conditions first**: prioritize settings where Kalman assumptions are likely to break.
- **Strong baselines, not strawmen**: include methods that often win in practice despite simplicity.
- **Locked analysis**: pre-register primary metrics, primary contrasts, and stop criteria before running.
- **Negative-result symmetry**: write interpretation rules before seeing outcomes.

### 1.2 Data domains with weak alignment

Use at least 5 domains with intentionally weak semantic overlap and divergent style:

1. Biomedical evidence retrieval (e.g., scientific abstracts)
2. Financial QA/news
3. Legal case retrieval
4. Consumer forum/support text
5. Code/documentation snippets

Build query sets with three strata:

- **In-domain (ID)**: clearly belongs to one specialist domain.
- **Cross-domain ambiguous (XD)**: blends concepts from multiple domains.
- **Out-of-domain (OOD)**: domains absent from specialist training (e.g., recipes, gaming slang, poetry).

Target mix per run:

- 40% ID
- 30% XD
- 30% OOD

Rationale: this forces frequent router uncertainty, representation mismatch, and specialist conflict.

### 1.3 Specialist panel and disagreement construction

Instantiate at least 6 specialists:

- 4 domain specialists (one per major domain except one held-out domain to induce OOD pressure)
- 1 broad generalist encoder
- 1 intentionally brittle specialist (small model or mismatched fine-tuning)

Create controlled disagreement slices:

- **Natural disagreement**: queries where top-1 documents from specialists are disjoint.
- **Engineered disagreement**: prompts containing overloaded terms ("python", "charge", "appeal", "bond").

For each query, log a disagreement index:

- pairwise cosine variance across specialist embeddings
- entropy of top-k specialist retrieval overlap
- spread of specialist-assigned confidence values

### 1.4 Uncertainty corruption matrix (miscalibration stress)

For each specialist, evaluate under five uncertainty regimes:

1. **Calibrated** (best available)
2. **Overconfident x4** (reported sigma^2 divided by 4)
3. **Underconfident x4** (reported sigma^2 multiplied by 4)
4. **Rank-inverted** (higher quality predictions assigned larger sigma^2)
5. **Heteroscedastic noise injection** (query-dependent random multiplicative noise)

Cross all specialists with independent corruption draws to produce hard mixed-calibration scenarios.

Primary adversarial settings:

- one wrong-but-overconfident specialist
- two mutually contradictory overconfident specialists
- all specialists mildly miscalibrated in different directions

### 1.5 Baseline suite (strong simple alternatives)

Evaluate Kalman fusion against:

- **Best-single-specialist oracle** (upper bound, ex-post)
- **Best-single-specialist deployable** (selected on validation)
- **Uniform mean fusion**
- **Trimmed mean fusion** (drop highest-uncertainty specialist)
- **Median embedding fusion** (coordinate-wise robust aggregation)
- **Hard router winner-take-all**
- **Learned gate (lightweight logistic/MLP)** using query features only
- **Uncertainty-agnostic rank fusion** for retrieval outputs (e.g., reciprocal rank fusion)

Any claim that Kalman is robust must exceed at least the best deployable simple baseline, not only naive mean.

### 1.6 Evaluation structure

Use a fully crossed design:

- Fusion method × uncertainty regime × query stratum (ID/XD/OOD) × disagreement quartile

Primary metrics:

- nDCG@10
- Recall@10
- MRR

Secondary metrics:

- calibration quality (ECE-like retrieval calibration proxy)
- instability (variance across random seeds and corruption draws)
- latency/query

Statistical protocol:

- paired bootstrap CIs per contrast
- stratified permutation test for primary metric deltas
- Holm-Bonferroni correction for multiple primary contrasts
- minimum practically important effect (MPIE) threshold declared in advance

### 1.7 Kill criteria (falsification thresholds)

Kalman claim is considered falsified for robustness if either holds:

1. In adversarial aggregate (all OOD + top disagreement quartile + miscalibration), Kalman is significantly below the best simple deployable baseline on nDCG@10 by more than MPIE.
2. Kalman wins in easy ID conditions but loses in at least 2 of 3 hostile strata (XD, OOD, high disagreement) with consistent sign across seeds.

This makes it easy to fail and hard to "win by averaging".

---

## 2) Implementation Plan

### 2.1 Protocol registration and config

Add an explicit adversarial protocol config in the benchmark registry:

- `experiments/configs/benchmark_registry/real_mixed_domain.yaml` (copy and extend into a dedicated adversarial config before running falsification experiments)

Include:

- domain dataset mapping
- query stratum quotas
- uncertainty corruption grid
- fixed seeds and bootstrap counts
- predefined primary contrasts and MPIE

### 2.2 Dataset assembly and tagging

Extend dataset pipeline to emit per-query metadata:

- `stratum`: ID / XD / OOD
- `heldout_domain_flag`
- `disagreement_index`

Likely touchpoints:

- `experiments/registry/datasets.py`
- `src/kalmanorix/benchmarks/mixed_domain.py`

### 2.3 Uncertainty corruption harness

Add a deterministic corruption wrapper for sigma^2 outputs:

- multiplicative scaling
- rank inversion
- stochastic heteroscedastic perturbation

Likely touchpoints:

- `experiments/registry/models.py`
- `experiments/registry/fusion.py`
- `src/kalmanorix/uncertainty.py`

### 2.4 Baseline expansion

Ensure registry runner includes robust/simple baselines in one run graph.

Likely touchpoints:

- `src/kalmanorix/benchmarks/fusion_baselines.py`
- `experiments/registry/evaluation.py`
- `experiments/registry/runner.py`

### 2.5 Reporting templates for negative results

Add mandatory tables:

1. "Where Kalman loses" (largest negative deltas)
2. "Win/loss by hostility stratum"
3. "Sensitivity to uncertainty corruption"

Likely touchpoints:

- `experiments/registry/reporting.py`
- `experiments/registry/templates/table_overall.md`
- `experiments/registry/templates/table_significance.md`

### 2.6 CI and reproducibility guardrails

- Add smoke test for adversarial config parsing.
- Add invariant test to verify corruption modes are applied as specified.
- Save full run manifest: commit SHA, model versions, seed list, and config hash.

Likely touchpoints:

- `tests/test_registry_baseline_strategies.py`
- `tests/test_validation_suite.py`
- `tests/test_uncertainty_methods.py`

---

## 3) Expected Failure Modes

1. **Overconfident wrong specialist collapse**
   - Kalman over-weights a miscalibrated specialist and suppresses better signals.

2. **Disagreement instability**
   - In high-conflict queries, small uncertainty perturbations produce large rank swings.

3. **OOD confidence illusion**
   - Specialists emit deceptively low sigma^2 on OOD inputs, creating confident but incorrect fusion.

4. **Alignment-fragility amplification**
   - Weakly aligned spaces create systematic directional errors that uncertainty weighting cannot fix.

5. **Router-fusion compounding error**
   - If routing picks the wrong subset, Kalman optimally combines the wrong experts.

6. **Method overfitting to ID**
   - Kalman improves ID averages while harming XD/OOD tails.

---

## 4) Interpretation Guide for Negative Results

### 4.1 If Kalman loses broadly

Interpretation:

- The uncertainty model is not reliable enough for weighting-based fusion in hostile conditions.
- The claim should be narrowed from "robustly better" to "better when calibration/alignment assumptions hold".

Action:

- treat uncertainty estimation and OOD detection as blockers, not polish tasks.

### 4.2 If Kalman wins ID but loses XD/OOD

Interpretation:

- Gains are conditional on domain familiarity; robustness claim is not supported.

Action:

- report split metrics prominently; avoid pooled headline claims.
- add deployment policy that falls back to robust simple baselines in OOD/high-disagreement regimes.

### 4.3 If Kalman is high variance across seeds/corruption draws

Interpretation:

- method is brittle; expected deployment performance is uncertain.

Action:

- prefer baselines with smaller variance unless mean gain exceeds MPIE with tighter intervals.

### 4.4 If a simple baseline matches or beats Kalman

Interpretation:

- Kalman complexity is not currently justified by empirical value.

Action:

- shift research objective from "prove Kalman wins" to "identify the smallest conditions under which Kalman materially helps".

### 4.5 Credibility rules for reporting

- Publish full per-stratum results, not only aggregate means.
- Keep failed settings in the main results table.
- Preserve pre-registered thresholds and corrections.
- Explicitly state where the hypothesis fails.

---

## 5) Decision Outcomes (Pre-committed)

- **Promote** Kalman only if it beats best deployable simple baseline in hostile aggregate and does not collapse on OOD/high-disagreement strata.
- **Conditional deploy** if Kalman wins only in ID or low-disagreement segments; use automatic fallback policy elsewhere.
- **Reject current formulation** if Kalman underperforms in adversarial aggregate or shows unacceptable instability.

This ensures the experiment is maximally informative regardless of outcome and does not protect the hypothesis.
