# Publication Claims Policy

To avoid overclaiming in publication drafts, avoid absolute performance language unless it is clearly framed as unsupported by evidence.

## Avoid direct superiority claims

Do **not** use unqualified phrases such as:

- "Kalman beats mean"
- "Kalman improves retrieval"
- "Kalman outperforms"
- "proved superior"
- "state-of-the-art"

## Preferred framing

When discussing hypotheses or outcomes, use cautious wording tied to observed evidence, for example:

- "does not beat"
- "hypothesis was not supported"
- "we found no evidence"

This repository includes `scripts/audit_publication_claims.py` to flag risky phrasing in `docs/publication/**/*.md` and `paper/**/*.md`.
