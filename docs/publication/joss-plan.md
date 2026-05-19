# JOSS Software-Paper Plan for Kalmanorix

## Purpose and scope
This document defines a **separate** publication path for a Journal of Open Source Software (JOSS) software paper focused on Kalmanorix as research software infrastructure.

It is intentionally distinct from the empirical negative-results narrative (Kalman vs mean), which should remain in a preprint/TMLR-style empirical manuscript.

---

## 1) Preliminary JOSS-eligibility assessment (conservative)

### Current status: **potentially eligible in principle, not submission-ready yet**

Against core JOSS expectations (open-source software, research application, public browsable repository, `paper.md`, documentation/tests, and software-paper focus rather than new research claims):

- **Open-source software**: appears satisfiable if repository licensing is explicit and approved.
- **Research application**: appears satisfiable; project targets research workflows around retrieval/fusion benchmarking.
- **Browsable public repository**: **not yet confirmed in this internal plan** (must be public at submission time).
- **`paper.md` in repository**: currently a gap for a JOSS submission package.
- **Documentation/tests**: partially/mostly present in repository structure, but must be validated for JOSS reviewer usability.
- **Paper scope discipline**: must explicitly frame as software contribution, not as a venue for primary empirical novelty.

### Conservative conclusion
Do **not** assert current full JOSS eligibility yet. Treat Kalmanorix as a plausible JOSS candidate **after** the concrete gaps below are closed and verified.

---

## 2) Gaps to fix before JOSS submission

Below are required gates, mapped to the requested checklist.

### A. Public development history
- Ensure active, inspectable commit history in a public VCS host (e.g., GitHub/GitLab).
- Ensure contribution process is visible (`CONTRIBUTING`, issue/PR norms, changelog/release notes).
- Avoid "single dump" appearance; preserve iterative development evidence.

**Status in this plan:** unverified externally; must be confirmed.

### B. Open-source licence
- Keep an OSI-approved licence at repository root.
- Confirm licence compatibility with dependencies and bundled assets/data.
- Make licence prominent in README and docs.

**Status in this plan:** licence file exists in-repo, but compliance details still need explicit audit.

### C. Install instructions
- Provide short "happy path" install for common users (pip/uv/conda/poetry, whichever is canonical).
- Include Python/version requirements and optional extras.
- Include quick verification command after install.

**Status in this plan:** docs exist, but JOSS-grade install path should be simplified and smoke-tested from clean env.

### D. Tests
- Maintain automated test suite and document how reviewers run it.
- Distinguish quick tests from heavy/integration/stress tests.
- Ensure CI configuration is visible and passing on default branch.

**Status in this plan:** strong signs of tests, but reviewer-facing subset and pass evidence should be curated.

### E. API docs
- Provide clear user-facing API reference for core modules.
- Keep examples synchronized with current API signatures.
- Ensure docs build reproducibly.

**Status in this plan:** API docs appear present; must verify completeness and freshness.

### F. Examples
- Provide runnable, minimal examples that demonstrate typical research workflows.
- Include expected outputs and runtime notes where possible.

**Status in this plan:** examples appear present; must identify 2-3 canonical examples for JOSS reviewers.

### G. Issue tracker
- Public issue tracker must be enabled and discoverable.
- Use labels/templates for bug reports and feature requests where possible.

**Status in this plan:** external/public state not yet verified here.

### H. Archived release DOI
- Create an archived release (e.g., Zenodo) linked to a tagged software version.
- Mint DOI and reference it in `paper.md` and README.
- Ensure archived artefact corresponds exactly to reviewed version.

**Status in this plan:** required and commonly missing; assume gap until DOI is minted.

---

## 3) Draft JOSS statement of need

> Modern retrieval and embedding research often requires combining multiple specialist models, comparing fusion/routing strategies, and reproducing benchmark decisions under controlled protocols. These workflows are frequently reimplemented ad hoc, reducing reproducibility and increasing methodological drift across studies.
>
> Kalmanorix provides a reusable open-source toolkit for research-oriented retrieval fusion experiments, including configurable fusion/routing pipelines, evaluation utilities, and reproducibility-focused testing/documentation scaffolding. By packaging these capabilities into a documented software project, Kalmanorix lowers setup cost for researchers, supports transparent comparison workflows, and enables more consistent replication of retrieval-system experiments.
>
> The primary contribution of the JOSS submission is the software infrastructure itself (design, usability, and reproducibility support), rather than claiming novel empirical state-of-the-art results.

---

## 4) Draft `paper.md` outline (target: 250-1000 words)

Suggested JOSS paper structure and approximate word budget:

1. **Summary (40-90 words)**
   - What Kalmanorix is.
   - Problem domain: reproducible retrieval-fusion experimentation.

2. **Statement of need (80-180 words)**
   - Pain points in current practice (fragmented scripts, weak reproducibility).
   - Who needs this software (IR/ML researchers, benchmark builders).

3. **Core functionality (80-220 words)**
   - Main components (fusion/routing primitives, evaluation/reporting utilities, adapters/examples).
   - High-level architecture and extensibility.

4. **Quality control (40-150 words)**
   - Test strategy, CI checks, and numerical/regression safeguards.
   - Documentation and developer workflows.

5. **Usage in research workflows (40-150 words)**
   - Typical reproducible pipeline from configuration to report artefacts.
   - Brief mention of example notebooks/scripts.

6. **Related software (30-120 words)**
   - Briefly position relative to adjacent retrieval/evaluation tooling.
   - Clarify differentiator: integrated fusion-focused experimentation stack.

7. **Acknowledgements and references (as needed)**
   - Funding, collaborators, and cited tools/libraries.

### Drafting constraints
- Keep language software-centric and non-hype.
- Avoid claiming broad scientific conclusions.
- Keep empirical claims minimal and only as usage context.

---

## 5) Warning on scope separation (required)

**Important:** The JOSS paper should **not** center on the negative "Kalman vs mean" empirical result.

That result belongs in a separate empirical manuscript (preprint/TMLR-style), where experimental design, statistical testing, limitations, and claim scope can be treated as the primary scientific contribution.

For JOSS, the evaluative focus should be:
- software quality,
- documentation and tests,
- reproducibility support,
- and practical utility to research users.

---

## Practical next steps checklist

- [ ] Confirm repository is public and browsable.
- [ ] Audit and confirm OSI-approved licensing details.
- [ ] Create/verify concise install + quickstart path.
- [ ] Curate reviewer-friendly test command subset.
- [ ] Verify API docs build and are up to date.
- [ ] Select 2-3 canonical runnable examples.
- [ ] Confirm public issue tracker setup.
- [ ] Mint archived release DOI for tagged version.
- [ ] Draft `paper.md` (250-1000 words) from outline above.
