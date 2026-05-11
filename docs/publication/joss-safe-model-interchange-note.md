# JOSS Note: Defensible safe model interchange

To support publication-grade reproducibility and security claims, Kalmanorix now separates
**data interchange** from **code execution** in SEF loading.

- Default `SEFModel.from_pretrained` behavior is non-executable: metadata, covariance, and alignment load safely.
- Executable model recovery requires explicit `embed_loader` injection.
- Legacy pickle recovery requires explicit `allow_pickle=True` and emits a security warning.
- Checksum verification remains mandatory and fails on mismatch.

This design improves defensibility for JOSS by making unsafe behavior opt-in, auditable, and test-covered.
