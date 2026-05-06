# TMLR Submission Outline: Negative Empirical Result on Uncertainty-Weighted Fusion

## Working title
**When Uncertainty-Weighted Fusion Fails in Mixed-Domain Retrieval: A Strong-Baseline Negative Result**

## Core claim (scope-limited)
This paper is an **experimental study** of conditions under which uncertainty-weighted specialist fusion does **not** improve retrieval quality over strong alternatives. The current evidence suggests that in mixed-domain settings with uneven specialist quality, uncertainty-aware fusion can yield negligible gains, while hard routing or max-selection policies can outperform fusion.

We position this as a **negative empirical result with diagnostic value**, not as a universal rejection of fusion.

## 1) Why this fits TMLR (with caveats)
TMLR is a good venue for rigorous, well-documented empirical findings, including negative results, when they are methodologically strong, reproducible, and informative for future system design. This submission could fit that scope if we strengthen external validity and diagnostics before submission. In its current state, the story is promising but incomplete: the conclusions are plausible for the tested setup, yet not broad enough to justify a strong general claim.

## Proposed paper structure

### A. Problem setting and motivation
- Mixed-domain retrieval with specialist retrievers and a fusion layer.
- Hypothesis under test: uncertainty-weighted fusion should improve aggregate retrieval by down-weighting uncertain specialists.
- Main question: when does this intuition fail in practice?

### B. Experimental setup
- Datasets/splits, domain composition, and retrieval metrics.
- Specialist retriever training and calibration assumptions.
- Fusion mechanisms:
  - uncertainty-weighted fusion,
  - unweighted or score-based fusion baselines,
  - hard routing / winner-take-most routing.
- Evaluation protocols including confirmatory slices.

### C. Main empirical result (negative)
- Uncertainty-weighted fusion does not produce meaningful aggregate improvements versus strong baselines.
- Observed “Kalman effect” (incremental benefit attributable to uncertainty weighting) is near zero in current runs.
- In mixed-domain scenarios, hard routing can dominate fusion on key metrics.

### D. Why this may happen (analysis)
- Specialist score correlation limits fusion gains.
- Uncertainty estimates may be weakly informative or misaligned with per-query specialist correctness.
- Fusion overhead may not translate into error-correction when specialists fail on overlapping queries.

### E. Practical takeaways
- Do not assume uncertainty weighting improves retrieval by default.
- Benchmark against strong routing and max-selection baselines, not only naive averaging.
- Validate uncertainty quality directly before adopting uncertainty-weighted fusion in production pipelines.

## 2) Required additional experiments before submission

These are required to make the paper submission-ready and reviewer-resilient:

1. **Non-fast-local neural embedding replication**
   - Re-run headline experiments with at least one non-fast-local embedding configuration (higher-fidelity neural encoder setting).
   - Confirm whether the near-zero Kalman effect persists under less artificial compute-constrained settings.

2. **At least one external dataset or alternate split**
   - Add one external benchmark or a materially different split protocol.
   - Goal: test whether observed failure modes generalize beyond the current in-house/data-local setup.

3. **Ablation of uncertainty estimates**
   - Compare calibrated uncertainty, shuffled uncertainty, constant uncertainty, and no-uncertainty variants.
   - Quantify how much of performance is due to uncertainty signal versus incidental reweighting.

4. **Analysis of specialist correlation**
   - Measure inter-specialist agreement/correlation on retrieval success and error overlap.
   - Relate correlation regimes to fusion gains/losses to explain why hard routing can win.

5. **Relaxed confirmatory-slice diagnostics**
   - Redefine or relax confirmatory-slice criteria to avoid empty-slice artifacts.
   - Report slice coverage, stability, and effect-size uncertainty under the revised diagnostic.

## 3) Key risks to acknowledge explicitly

1. **Fast-local setting may appear too artificial**
   - Reviewers may argue conclusions are driven by a constrained runtime mode rather than retrieval fundamentals.

2. **Confirmatory slice is currently empty**
   - This weakens inferential confidence and may be interpreted as a protocol failure.

3. **Kalman effect is effectively zero in current evidence**
   - Without stronger diagnostics and replication, reviewers may see this as either noise-level null or underpowered experimentation.

## 4) Likely reviewer objections and response strategy

### Objection A: “This is a setup-specific null result.”
**Response:**
- Agree partially; narrow the claim.
- Add external dataset/alternate split and non-fast-local replication.
- Present conclusions as conditional: “under these identifiable conditions, fusion fails.”

### Objection B: “Uncertainty is poorly estimated; this does not invalidate fusion.”
**Response:**
- Include explicit uncertainty ablations and calibration diagnostics.
- Separate two claims: (i) current uncertainty estimator has limited utility; (ii) end-to-end uncertainty-weighted fusion is not robustly better in tested regimes.

### Objection C: “Hard routing wins because specialists are highly separable; fusion is unnecessary.”
**Response:**
- Provide specialist-correlation and domain-overlap analyses.
- Show where routing dominates and where it does not; avoid overgeneralization.

### Objection D: “Empty confirmatory slice undermines the paper.”
**Response:**
- Treat this as a methodological issue, not a hidden failure.
- Introduce relaxed slice diagnostics, report coverage transparently, and include uncertainty intervals.

### Objection E: “No positive contribution beyond saying ‘it does not work.’”
**Response:**
- Emphasize contribution as actionable evaluation guidance:
  - stronger baseline discipline,
  - uncertainty ablation protocol,
  - correlation-aware decision rule for choosing routing vs fusion.

## Submission-readiness checklist (must be true before claiming TMLR readiness)
- [ ] Non-fast-local replication completed and consistent with headline claim.
- [ ] External dataset or alternate split added.
- [ ] Uncertainty ablations completed with clear interpretation.
- [ ] Specialist-correlation analysis integrated into main text.
- [ ] Confirmatory-slice diagnostics revised; no silent empty-slice reporting.
- [ ] Claims narrowed to evidence-supported scope.

## Recommended claim language for abstract/conclusion
- “We report a negative empirical result: in our mixed-domain retrieval settings, uncertainty-weighted fusion does not consistently outperform strong routing and max-selection baselines.”
- “Our analysis suggests that high specialist error correlation and weakly informative uncertainty estimates can erase expected fusion gains.”
- “These findings motivate correlation-aware baseline selection and stricter uncertainty ablations before deploying fusion-based retrieval.”

## What we should *not* claim yet
- Do **not** claim general superiority of hard routing across all retrieval regimes.
- Do **not** claim that uncertainty-weighted fusion is broadly ineffective.
- Do **not** imply TMLR acceptance suitability without the additional experiments above.
