# Creating Specialists

*TODO: Step‑by‑step guide to building domain‑specialist embedding models.*

## Overview

A specialist is an embedding model tailored to a specific domain (e.g., biomedical literature, legal contracts, academic papers). This guide walks through the process of creating a specialist and wrapping it as a SEF for use in Kalmanorix.

## Step 1: Choose a Base Model

Start with a pre‑trained transformer model that has strong general‑language understanding:

- **Hugging Face models**: `bert‑base‑uncased`, `roberta‑large`, `all‑MiniLM‑L6‑v2`
- **Sentence‑Transformers**: `all‑mpnet‑base‑v2`, `all‑MiniLM‑L12‑v2`
- **Domain‑specific pre‑training**: BioBERT, Legal‑BERT, SciBERT

## Step 2: Collect Domain Data

Gather a corpus of text from your target domain:

- **Medical**: PubMed abstracts, clinical notes, medical textbooks
- **Legal**: Court opinions, statutes, contract templates
- **Scientific**: ArXiv papers, conference proceedings

Aim for at least 10 000–50 000 sentences for fine‑tuning.

## Step 3: Fine‑Tune for Embedding

Use contrastive learning (e.g., Multiple‑Negatives Ranking Loss) to adapt the model to your domain. The goal is to make semantically similar sentences close in embedding space.

**Recommended framework:** Sentence‑Transformers training API.

```python
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer('all‑mpnet‑base‑v2')
train_examples = [
    InputExample(texts=[anchor, positive]),
    # ...
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path='./my_specialist'
)
```

## Step 4: Wrap as a SEF

Use the appropriate adapter factory:

```python
from kalmanorix import create_sentence_transformer_sef

my_sef = create_sentence_transformer_sef(
    name="medical_specialist",
    model_name_or_path="./my_specialist",
    sigma2=0.1  # constant uncertainty
)
```

## Step 5: Calibrate Uncertainty (Optional)

If you have a small validation set with ground‑truth similarity pairs, you can compute empirical covariance or set up centroid‑distance uncertainty:

```python
from kalmanorix import create_sentence_transformer_sef_with_calibration

my_sef = create_sentence_transformer_sef_with_calibration(
    name="medical_specialist",
    model_name_or_path="./my_specialist",
    calibration_corpus=["sentence1", "sentence2", ...]
)
```

## Step 6: Add to Village

```python
from kalmanorix import Village

village = Village(sefs=[my_sef])
# ... add more specialists
```

*TODO: Add data‑collection tips, fine‑tuning hyperparameters, and evaluation metrics.*
