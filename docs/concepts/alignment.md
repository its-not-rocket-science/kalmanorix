# Alignment

*TODO: Explain Procrustes alignment for unifying embedding spaces*

## The Alignment Problem

Different specialists produce embeddings in **different vector spaces**. Even if both are 768‑dimensional, their axes have arbitrary orientation and scale. Directly averaging or fusing such embeddings is meaningless.

## Orthogonal Procrustes

Kalmanorix uses **orthogonal Procrustes analysis** to align each specialist’s space to a common reference space (usually the space of a chosen reference specialist).

Given two sets of matched anchor embeddings `X` (reference) and `Y` (specialist), Procrustes finds an orthogonal matrix `Q` that minimizes:
```
|| X - Y Q ||_F
```
where `Q` is orthogonal (`QᵀQ = I`). This preserves distances (rigid rotation/reflection) and ensures the aligned embeddings are comparable.

## Anchor Set

Alignment requires a small set of **anchor sentences** that are embedded by all specialists. These can be:
- General‑purpose sentences (e.g., from Wikipedia)
- Domain‑neutral phrases
- A mix of domain‑specific sentences (if overlapping vocabulary exists)

## Implementation Steps

1. **Choose a reference specialist** (e.g., the one with the largest domain coverage).
2. **Embed anchor sentences** with all specialists.
3. **Compute alignment matrices** `Q_i` for each non‑reference specialist.
4. **Apply alignment** on‑the‑fly: `embedding_aligned = embedding_raw @ Q_i`.

## Quality Validation

Alignment success can be measured by:
- **Cross‑model similarity**: Average cosine similarity between aligned embeddings of the same sentence.
- **Retrieval accuracy**: Whether nearest‑neighbour retrieval works across aligned spaces.

## Limitations

- Procrustes assumes a **linear** relationship between spaces; non‑linear distortions are not corrected.
- Requires anchor sentences that are meaningful to all specialists (challenging for highly disjoint domains).
- Alignment quality degrades far from the anchor distribution.

## Future: Learned Alignment

Planned research includes replacing Procrustes with contrastive learning or non‑linear alignment networks, especially for massively multilingual or cross‑modal scenarios.

*TODO: Add visualisation of before/after alignment, code example, and anchor‑set recommendations.*
