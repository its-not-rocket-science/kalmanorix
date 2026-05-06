# Reference Audit (shared across arXiv, TMLR, and JOSS)

This audit records references that are actually cited in current drafts and normalizes metadata for a shared bibliography.

## Key-stability notes

- Preferred stable key for nDCG paper: `jarvelin2002` (alias currently cited in JOSS: `jarvelin2002cumulated`).
- Preferred stable key for Sentence-BERT paper: `reimers2019` (alias currently cited in JOSS: `reimers2019sentencebert`).
- Both bootstrap citations are currently used in drafts (`efron1979bootstrap` and `efron1994`). Keep one in camera-ready if possible.

## Verified references

| Citation key | Title | Authors | Year | Venue | DOI / arXiv URL | Used by paper(s) | Notes |
|---|---|---|---:|---|---|---|---|
| `kalman1960` | A New Approach to Linear Filtering and Prediction Problems | Rudolf E. Kalman | 1960 | Journal of Basic Engineering | DOI: 10.1115/1.3662552 | arXiv, TMLR, JOSS | Kalman filtering foundation |
| `thakur2021beir` | BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models | Nandan Thakur; Nils Reimers; Andreas Rücklé; Abhishek Srivastava; Iryna Gurevych | 2021 | NeurIPS Datasets and Benchmarks Track | URL: https://openreview.net/forum?id=wCu6T5xFjeJ | arXiv, TMLR, JOSS | BEIR benchmark |
| `jarvelin2002` | Cumulated Gain-Based Evaluation of IR Techniques | Kalervo Järvelin; Jaana Kekäläinen | 2002 | ACM Transactions on Information Systems | DOI: 10.1145/582415.582418 | arXiv, TMLR | nDCG / IR evaluation |
| `jarvelin2002cumulated` | Cumulated Gain-Based Evaluation of IR Techniques | Kalervo Järvelin; Jaana Kekäläinen | 2002 | ACM Transactions on Information Systems | DOI: 10.1145/582415.582418 | JOSS | Alias key in current JOSS draft |
| `holm1979` | A Simple Sequentially Rejective Multiple Test Procedure | Sture Holm | 1979 | Scandinavian Journal of Statistics | URL: https://www.jstor.org/stable/4615733 | arXiv, TMLR | Holm correction |
| `wilcoxon1945` | Individual Comparisons by Ranking Methods | Frank Wilcoxon | 1945 | Biometrics Bulletin | DOI: 10.2307/3001968 | arXiv, TMLR | Wilcoxon signed-rank test |
| `efron1979bootstrap` | Bootstrap Methods: Another Look at the Jackknife | Bradley Efron | 1979 | The Annals of Statistics | DOI: 10.1214/aos/1176344552 | TMLR | Bootstrap confidence intervals |
| `efron1994` | An Introduction to the Bootstrap | Bradley Efron; Robert J. Tibshirani | 1994 | Chapman & Hall/CRC (book) | verification needed | arXiv | Bootstrap confidence intervals |
| `reimers2019` | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Nils Reimers; Iryna Gurevych | 2019 | EMNLP-IJCNLP | DOI: 10.18653/v1/D19-1410; arXiv: https://arxiv.org/abs/1908.10084 | arXiv, TMLR | Sentence-BERT |
| `reimers2019sentencebert` | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Nils Reimers; Iryna Gurevych | 2019 | EMNLP-IJCNLP | DOI: 10.18653/v1/D19-1410; arXiv: https://arxiv.org/abs/1908.10084 | JOSS | Alias key in current JOSS draft |
| `harris2020array` | Array programming with NumPy | Charles R. Harris et al. | 2020 | Nature | DOI: 10.1038/s41586-020-2649-2 | JOSS | NumPy citation |
| `virtanen2020scipy` | SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python | Pauli Virtanen et al. | 2020 | Nature Methods | DOI: 10.1038/s41592-019-0686-2 | JOSS | SciPy citation |
| `pedregosa2011scikit` | Scikit-learn: Machine Learning in Python | F. Pedregosa et al. | 2011 | Journal of Machine Learning Research | URL: https://jmlr.org/papers/v12/pedregosa11a.html | JOSS | scikit-learn citation |
| `voorhees2016` | The TREC Robust Retrieval Track | Ellen M. Voorhees | 2016 | SIGIR Forum | DOI: 10.1145/3055401.3055421 | TMLR | IR evaluation context |

## Exclusions

- JOSS meta-citation (e.g., `@article{smith2018joss}`) is not included because it is not cited in the current drafts.
- Placeholder TODO citations in `paper/paper.md` are excluded because they are unresolved placeholders, not verifiable references.
