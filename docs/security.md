# Security: Safe SEF model loading

SEF model directories may contain `model.pkl`, which is Python pickle data. Unpickling can execute arbitrary code.

## Default policy (safe)

`SEFModel.from_pretrained(...)` now defaults to **safe loading**:
- loads metadata, covariance, and alignment files
- does **not** execute `model.pkl`
- requires `embed_loader=...` to provide model code explicitly

## Trusted legacy pickle path

For legacy artefacts, pickle loading must be explicitly opted into:

```python
model = SEFModel.from_pretrained("./my_sef", allow_pickle=True)
```

This path emits a warning explaining the security risk.

## Recommended practice

- Treat SEF artefacts as untrusted by default.
- Prefer `embed_loader`-based loading in production.
- Enable `allow_pickle=True` only for trusted, integrity-checked sources.
- Keep checksum verification enabled to detect tampering.
