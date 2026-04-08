# Draft Results Section (Evidence-Constrained)

## 4. Results

### 4.1 Primary question: does Kalman fusion outperform baselines?

Based on the currently committed benchmark artifacts, **there is no statistically validated evidence that Kalman fusion consistently outperforms baselines on retrieval quality**. The project registry explicitly marks Milestones 1.3 (Kalman vs averaging), 2.1 (specialists vs monolith), and 2.2 (OOD robustness) as pending final statistical reports, while only Milestone 2.3 (efficiency) is marked completed.

In the one fully documented retrieval comparison available (Milestone 0001, unaligned specialists), Kalman fusion does **not** improve over hard routing and does not dominate simple alternatives.

### 4.2 Where Kalman helps vs where it does not

#### Setting where Kalman does *not* help (current strongest evidence)

In the drifted, unaligned setting (Milestone 0001), reported mixed-domain retrieval was:

| Method | Recall@1 | Recall@3 | Interpretation |
|---|---:|---:|---|
| Hard routing | 0.600 | 0.600 | Best/tying baseline |
| Mean fusion | 0.467 | 0.667 | Worse at R@1, better at R@3 |
| Kalman fusion | 0.600 | 0.600 | Ties hard routing, no gain |
| Learned gate | 0.533 | 0.533 | Below hard routing |

Interpretation: with embedding-space misalignment, Kalman weighting appears to collapse to near single-expert selection, yielding parity with hard routing rather than improvements.

#### Setting where Kalman *may* help

The repository includes positive narrative claims in research docs (e.g., improved mixed-domain recall and OOD robustness), but those sections still contain TODO markers and are not backed by committed final statistical artifacts in the milestone registry. Therefore these should be treated as **provisional** and not confirmatory.

### 4.3 Effect sizes and significance

Because finalized paired query-level analysis outputs (e.g., Wilcoxon p-values with multiplicity correction, paired bootstrap CIs) are not committed for Milestones 1.3/2.1/2.2, inferential significance cannot be established from the available artifacts.

For transparency, we report descriptive effect sizes from the documented Milestone 0001 table only:

| Comparison | Metric | Absolute Δ (Kalman - Baseline) | Relative change |
|---|---|---:|---:|
| Kalman vs Hard routing | Recall@1 | +0.000 | 0.0% |
| Kalman vs Hard routing | Recall@3 | +0.000 | 0.0% |
| Kalman vs Mean fusion | Recall@1 | +0.133 | +28.5% vs mean |
| Kalman vs Mean fusion | Recall@3 | -0.067 | -10.0% vs mean |
| Kalman vs Learned gate | Recall@1 | +0.067 | +12.6% vs gate |
| Kalman vs Learned gate | Recall@3 | +0.067 | +12.6% vs gate |

These descriptive deltas are mixed and do not support a general superiority claim.

### 4.4 Efficiency evidence (completed milestone)

The efficiency milestone is the only one marked completed in the results registry. Existing efficiency analyses indicate non-trivial compute trade-offs (e.g., higher Kalman latency than mean fusion in raw fusion timing), but these do not by themselves establish retrieval-quality superiority.

### 4.5 Limitations and threats to validity

1. **Incomplete confirmatory reporting**: key quality milestones remain pending final statistical reports, limiting inferential conclusions.
2. **Missing paired test artifacts**: no committed final tables with adjusted p-values and paired CIs for the core Kalman-vs-baseline retrieval claims.
3. **Potential reporting inconsistency**: some docs contain strong positive claims while other project artifacts explicitly mark those milestones as incomplete.
4. **Small and stress-test-like evidence base**: the strongest concrete retrieval evidence comes from a single drift-focused scenario (15 queries), which is useful diagnostically but weak for broad generalization.
5. **Alignment dependence**: the documented failure mode shows that without alignment, fusion quality can degrade; conclusions from unaligned settings should not be extrapolated to aligned production pipelines.

## 5. Conservative interpretation

Current evidence supports a narrow conclusion: **Kalman fusion is not yet proven to outperform baselines on core retrieval quality under completed, statistically validated evaluation in this repository state**. The best-supported finding is a failure mode (misalignment) plus completed efficiency characterization. Stronger claims require finalized paired statistical reports for Milestones 1.3, 2.1, and 2.2.
