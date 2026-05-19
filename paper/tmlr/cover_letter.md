# Cover letter — TMLR submission

**Subject: Submission — "Claim-Gated Evaluation for Small Retrieval Deltas: A Confirmatory Mixed-Domain Study"**

Dear TMLR Editors,

I am submitting the above manuscript for consideration in TMLR.

**What the paper is.** This is a methodology paper, not a positive empirical result. It presents a claim-gated evaluation protocol for settings where retrieval metric deltas are small and overclaim risk is high. The protocol combines predeclared endpoints, paired Wilcoxon inference with Holm correction, a practical-effect-size threshold, a baseline matrix, and artifact-backed reproducibility. The protocol is instantiated on a concrete experiment — confirmatory paired evaluation of Kalman-style uncertainty fusion versus mean fusion on a domain-balanced mixed-domain benchmark — and the result is negative: no reliable benefit is demonstrated under the tested configuration.

**Why TMLR.** TMLR explicitly welcomes negative results and reproducibility contributions. This submission is both: a constrained, artifact-backed null finding reported with statistical rigour, and a reusable evaluation template that other researchers can apply to similar small-delta retrieval claims.

**What the contribution is not.** The paper does not claim Kalman fusion is generally ineffective. The result is protocol-local: under n=1193 test pairs, max_candidates=100, and the mixed-beir-v1.2.0 benchmark family, no reliable advantage was found. The preregistered confirmatory slice was underpowered (n=1 query pair), so its verdict is inconclusive rather than negative; this limitation is disclosed in §4 and §5.

**Reproducibility.** All numerical claims are sourced from committed artifacts in the public repository. Manuscript constants are generated programmatically from benchmark summary JSON to avoid manual transcription error. The repository includes a full test suite and pre-push statistical consistency checks.

This submission is non-anonymous per TMLR policy.

Paul Schleifer
