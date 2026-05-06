# Claims Policy for Publications

This repository reports a negative-result finding. Publication language must avoid unsupported superiority claims.

## Scope

Apply this policy to:
- `paper/arxiv/`
- `paper/tmlr/`
- `paper/joss/`
- `docs/publication/`

## Disallowed claim patterns (unless explicitly negated or marked unsupported)

The following phrases are risky and should be treated as disallowed by default:
- “Kalman beats mean”
- “Kalman improves retrieval”
- “Kalman outperforms”
- “proved superior”
- “state of the art”
- “statistically significant improvement”
- “robustly improves”

These phrases are only acceptable when they are clearly negated or directly described as unsupported.

## Preferred wording

Use wording aligned with observed evidence:
- “The hypothesis was not supported.”
- “No practical or statistically significant advantage was observed.”
- “The result is benchmark-specific.”
- “Hard routing was the strongest observed baseline in this configuration.”
- “The confirmatory slice was empty and cannot support a superiority claim.”

## Enforcement

Run the audit script before submission:

```bash
python scripts/audit_paper_claims.py
```

The script exits nonzero when risky claims are found.
