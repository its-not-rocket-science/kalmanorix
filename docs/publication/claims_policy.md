# Publication Claims Policy

This repository adopts a conservative claims standard for all publication materials.

## Scope

Apply this policy to:

- `paper/arxiv/`
- `paper/tmlr/`
- `paper/joss/`
- `docs/publication/`

## Prohibited overclaiming language

Avoid unsupported superiority language such as:

- "Kalman beats mean"
- "Kalman improves retrieval"
- "Kalman outperforms"
- "proved superior"
- "state of the art" (unsupported claim example)
- "statistically significant improvement" (unsupported claim example)
- "robustly improves" (unsupported claim example)

These phrases are only acceptable when clearly negated or explicitly described as unsupported.

## Preferred wording

Use language that accurately reflects negative or bounded findings:

- "The hypothesis was not supported."
- "No practical or statistically significant advantage was observed."
- "The result is benchmark-specific."
- "Hard routing was the strongest observed baseline in this configuration."
- "The confirmatory slice was empty and cannot support a superiority claim."

## Enforcement

Run `scripts/audit_paper_claims.py` before release or submission. The audit exits nonzero when risky, unsupported claim language is detected.
