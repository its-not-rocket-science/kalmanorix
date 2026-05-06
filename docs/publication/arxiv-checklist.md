# arXiv / Preprint Submission Checklist

Use this checklist as a final operational gate before uploading a preprint.

## 1) Manuscript

- [ ] Title, abstract, and introduction use the same problem framing.
- [ ] Contributions are listed explicitly and match what is actually evaluated.
- [ ] Every major claim in the abstract appears with supporting evidence in Results.
- [ ] The conclusion does not introduce new claims.
- [ ] All key terms are defined once and used consistently (e.g., routing, confirmatory slice, fast-local hash embedding).
- [ ] The text explicitly states that **the hypothesis was not supported**.

## 2) Figures/tables

- [ ] Every figure/table is referenced in the main text before it appears.
- [ ] Captions are self-contained (dataset/split, metric, and comparison scope included).
- [ ] Axis labels, units, and legends are readable and unambiguous.
- [ ] Error bars/confidence intervals are described in Methods.
- [ ] Figure/table numbers, names, and values are consistent with the manuscript narrative.
- [ ] Any figure that could imply global superiority is explicitly scoped to the observed benchmark.

## 3) References

- [ ] All in-text citations have bibliography entries and vice versa.
- [ ] Reference formatting is consistent with the target venue/preprint style.
- [ ] URLs/DOIs are present where appropriate.
- [ ] Claims about prior work are faithful to cited sources.
- [ ] Software/datasets used are cited with version or access date when possible.

## 4) Artifact availability

- [ ] Repository URL is included (or a blinded placeholder if required).
- [ ] Commit hash/tag corresponding to reported results is recorded.
- [ ] Environment requirements are documented (OS, Python, package manager, hardware assumptions).
- [ ] Scripts/commands to reproduce headline tables are included.
- [ ] Non-public dependencies or data restrictions are explicitly stated.

## 5) Reproducibility

- [ ] Random seeds and split definitions are documented.
- [ ] Hyperparameters and selection criteria are fully reported.
- [ ] Number of runs/replicates per result is specified.
- [ ] Any post-hoc analysis is labeled as exploratory.
- [ ] Confirmatory analyses are clearly separated from exploratory analyses.
- [ ] The confirmatory slice is documented as having **zero queries** and therefore cannot support superiority claims.

## 6) Claims audit

- [ ] The manuscript does **not** contain the phrase or claim: “Kalman beats mean.”
- [ ] The manuscript explicitly says the hypothesis was not supported.
- [ ] Any positive routing claim is tied to the observed benchmark conditions (dataset, split, metric, and scope).
- [ ] Fast-local hash embedding limitations are explicit, concrete, and easy to find.
- [ ] The confirmatory slice had zero queries and is not used to justify superiority claims.
- [ ] Abstract, conclusion, and press-ready summary language are all aligned with the same constrained claims.

## 7) Licensing

- [ ] Code license is present and compatible with included dependencies.
- [ ] Data licenses and usage restrictions are documented.
- [ ] Third-party assets (figures/icons/text) have clear reuse permissions.
- [ ] Model/artifact redistribution terms are stated.

## 8) Ethics / limitations

- [ ] Limitations section is explicit about failure modes and external validity limits.
- [ ] Potential harms/misuse risks are identified.
- [ ] Evaluation blind spots (coverage gaps, demographic/domain gaps) are acknowledged.
- [ ] Resource/computational footprint is stated where relevant.
- [ ] Fast-local hash embedding limitations are described in practical terms for deployment decisions.

## Final pre-upload gate

- [ ] A coauthor (or internal reviewer) completed an independent claims-and-evidence pass.
- [ ] PDF was rebuilt from the final source and spot-checked page-by-page.
- [ ] arXiv metadata (title, authors, abstract, categories) exactly matches the final manuscript.
