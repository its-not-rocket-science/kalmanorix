"""
Train two specialist SentenceTransformer checkpoints (tech / cooking).

This script fine-tunes the same base encoder twice:

    sentence-transformers/all-MiniLM-L6-v2

and writes two local specialist checkpoints:

    models/tech-minilm/
    models/cook-minilm/

Why this exists
---------------
Kalmanorix-style fusion is only testable when specialists can *disagree*.
If both SEFs point at the same embedder, fusion weights can change but the
embedding vectors are identical, so retrieval rankings won’t move.

By training two separate checkpoints (same architecture and embedding
dimension), we get:

- identical vector dimensionality (no projection/alignment required yet)
- different embedding geometries (specialists can disagree)
- a clean setup to compare fusion strategies (hard/mean/kalman/gate)

Training approach
-----------------
We create positive pairs from a small domain corpus using cheap, deterministic
text augmentations and train using:

- MultipleNegativesRankingLoss (in-batch negatives)

This is not meant to produce state-of-the-art specialists; it is designed to
produce *controlled divergence* cheaply.

Notes
-----
- Requires: sentence-transformers, torch, accelerate (CPU wheels are fine).
- Output directories are experiment artifacts and usually should be gitignored.
- For stronger divergence, increase:
    - number of domain sentences
    - epochs
    - k (augmentations per sentence)
    - batch size (more in-batch negatives)


Key changes vs the earlier script
---------------------------------
1) Add a small domain classifier head (linear) trained on mean-pooled embeddings.
   This explicitly pulls tech vs cook representations apart.

2) Add a "domain contrast" term:
   - tech examples are trained to be dissimilar to the cooking centroid
   - cook examples are trained to be dissimilar to the tech centroid

3) Keep the original intra-domain MultipleNegativesRankingLoss as a stabilizer.

This is still lightweight and CPU-friendly for tiny corpora, but creates a much
stronger incentive for separation than paraphrase-only contrastive training.

Outputs:
  models/tech-minilm/
  models/cook-minilm/


Usage
-----
    python experiments/train_specialists_st.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader, Dataset

import torch
from torch import nn
import torch.nn.functional as F

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = Path("models")

# --- Small, deterministic domain corpora ---
TECH_SENTENCES = [
    "Smartphone battery life improved after the update.",
    "Fast charging requires a compatible charger and cable.",
    "The laptop CPU throttles under sustained load.",
    "GPU driver updates can improve frame rates in games.",
    "Camera sensor size affects low-light performance.",
    "Thermal management reduces overheating during heavy usage.",
    "Power consumption rises when performance mode is enabled.",
    "Background apps reduce battery performance over time.",
    "The device supports USB-C Power Delivery fast charging.",
    "High refresh rate displays can impact battery life.",
    "Benchmark results show improved CPU performance per watt.",
    "Image processing pipelines run faster with GPU acceleration.",
]

COOK_SENTENCES = [
    "Braise beef slowly with garlic and onion until tender.",
    "Simmer stew for hours in a slow cooker.",
    "Saute vegetables before baking in the oven.",
    "Reduce a sauce by simmering to concentrate flavor.",
    "A food processor helps chop onions quickly.",
    "Low heat cooking prevents the sauce from breaking.",
    "Season the stew and let it simmer gently.",
    "Deglaze the pan to build flavor for the sauce.",
    "Oven temperature affects moisture and browning.",
    "Let the braise rest before serving for better texture.",
    "Use a slow cooker to keep the simmer steady.",
    "Taste and adjust seasoning as the sauce reduces.",
]


class InputExampleDataset(Dataset[InputExample]):
    """Torch Dataset wrapper for a list of SentenceTransformers InputExample."""

    def __init__(self, items: list[InputExample]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> InputExample:
        return self._items[idx]


@dataclass(frozen=True)
class AugmentConfig:
    """Configuration for simple token-drop + swap augmentation."""

    drop_prob: float = 0.15
    swap_prob: float = 0.10
    seed: int = 123


def _tokenize_simple(text: str) -> List[str]:
    """Minimal tokenizer: alphanumeric chunks only."""
    toks: List[str] = []
    buff: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buff.append(ch)
        else:
            if buff:
                toks.append("".join(buff))
                buff.clear()
    if buff:
        toks.append("".join(buff))
    return toks


def _augment(text: str, cfg: AugmentConfig, rng: np.random.Generator) -> str:
    """Cheap augmentation: token dropout + occasional adjacent swaps."""
    toks = _tokenize_simple(text)
    if not toks:
        return text

    kept = [t for t in toks if rng.random() > cfg.drop_prob]
    if len(kept) < 3:
        kept = toks[:]

    i = 0
    while i < len(kept) - 1:
        if rng.random() < cfg.swap_prob:
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            i += 2
        else:
            i += 1

    return " ".join(kept)


def build_positive_pairs(
    sentences: List[str], cfg: AugmentConfig, k: int = 8
) -> List[Tuple[str, str]]:
    """Build (anchor, positive) pairs with k augmented positives per anchor."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(cfg.seed)
    pairs: List[Tuple[str, str]] = []
    for s in sentences:
        a = s.strip()
        for _ in range(k):
            b = _augment(a, cfg, rng)
            pairs.append((a, b))
    return pairs


def _compute_centroid(model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    """
    Compute a unit-normalized centroid embedding for a set of texts.

    Returns a torch tensor on CPU with shape (dim,).
    """
    with torch.no_grad():
        embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    c = np.mean(np.asarray(embs, dtype=np.float64), axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return torch.tensor(c, dtype=torch.float32)


# pylint: disable=too-few-public-methods
class DomainHead(nn.Module):
    """
    Tiny linear domain classifier.

    Takes normalized embeddings and predicts domain label:
      1 => in-domain (for the current specialist)
      0 => out-of-domain
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits for domain classification."""
        return self.lin(x).squeeze(-1)


# pylint: disable=too-many-arguments,too-many-locals
def train_specialist(
    *,
    domain_name: str,
    in_domain: List[str],
    out_domain: List[str],
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 123,
    k: int = 8,
    # Divergence knobs
    lambda_cls: float = 0.8,
    lambda_away: float = 0.8,
    # Manual divergence phase size
    div_steps: int = 200,
    div_batch_size: int = 16,
) -> Path:
    """
    Train one specialist with *joint* divergence objectives.

    This function intentionally creates representation divergence between two
    specialists by combining:

    1) In-domain structure (SentenceTransformers contrastive objective):
         MN_rank  (MultipleNegativesRankingLoss on in-domain positive pairs)

    2) Domain separation (supervised, backpropagates into the encoder):
         domain_BCE (a tiny linear head trained to classify in-domain vs out-domain)

    3) "Push-away" from the opposite domain (backpropagates into the encoder):
         away_loss  (penalize cosine similarity to the out-domain centroid)

    Composite divergence objective (manual phase)
    ---------------------------------------------
        L = lambda_cls * domain_BCE + lambda_away * away_loss

    Notes
    -----
    - The initial MN_rank phase uses SentenceTransformers' built-in trainer.
    - The divergence phase uses a manual loop that *does* backprop through the encoder.
    - CPU-friendly: batches are small and div_steps is configurable.
    """
    out_path = out_dir / f"{domain_name}-minilm"
    out_path.mkdir(parents=True, exist_ok=True)

    # Best-effort determinism
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = SentenceTransformer(BASE_MODEL)

    # --- Phase 1: MN ranking on in-domain positive pairs (stabilizer) ---
    pairs = build_positive_pairs(in_domain, AugmentConfig(seed=seed), k=k)

    # train_examples = [InputExample(texts=[a, b]) for (a, b) in pairs]
    # train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_examples: list[InputExample] = [
        InputExample(texts=[a, b]) for (a, b) in pairs
    ]
    train_ds = InputExampleDataset(train_examples)
    train_loader: DataLoader[InputExample] = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size
    )

    mn_rank = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(10, int(0.1 * len(train_loader) * max(1, epochs // 2)))

    model.fit(
        train_objectives=[(train_loader, mn_rank)],
        epochs=max(1, epochs // 2),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    # --- Phase 2: Joint divergence (BCE + away) WITH encoder gradients ---
    dim_opt = model.get_sentence_embedding_dimension()
    assert dim_opt is not None
    dim = dim_opt
    head = DomainHead(dim)

    # Encoder optimizer (for manual phase) and head optimizer
    enc_optim = torch.optim.AdamW(model.parameters(), lr=lr)
    head_optim = torch.optim.AdamW(head.parameters(), lr=2e-3)

    # Precompute out-domain centroid with current encoder (no grads)
    out_centroid = _compute_centroid(model, out_domain)  # (dim,)

    # Build a balanced pool for BCE
    cls_texts = in_domain + out_domain
    cls_labels = torch.tensor(
        [1.0] * len(in_domain) + [0.0] * len(out_domain),
        dtype=torch.float32,
    )

    # For simple batching without extra deps
    rng = np.random.default_rng(seed + 999)

    def _sample_batch() -> Tuple[List[str], torch.Tensor]:
        idx = rng.choice(
            len(cls_texts), size=min(div_batch_size, len(cls_texts)), replace=False
        )
        texts = [cls_texts[i] for i in idx]
        y = cls_labels[idx]
        return texts, y

    def _encode_with_grad(texts: List[str]) -> torch.Tensor:
        """
        Encode texts with gradients through the SentenceTransformer.
        Returns normalized embeddings (B, dim).
        """
        features = model.tokenize(texts)
        # model.forward expects tensors; tokenize already returns tensors on CPU
        out = model.forward(features)
        z = out["sentence_embedding"]
        z = F.normalize(z, p=2, dim=1)
        return z

    head.train()
    for _ in range(div_steps):
        # --- BCE batch (encoder + head get grads) ---
        b_texts, b_y = _sample_batch()

        enc_optim.zero_grad()
        head_optim.zero_grad()

        z_tensor = _encode_with_grad(b_texts)  # (B, dim)
        logits = head(z_tensor)  # (B,)
        bce = F.binary_cross_entropy_with_logits(logits, b_y)

        # --- Away batch (use in-domain texts; encoder gets grads) ---
        # Sample a small in-domain batch to push away from the out centroid
        in_idx = rng.choice(
            len(in_domain), size=min(div_batch_size, len(in_domain)), replace=False
        )
        in_batch = [in_domain[i] for i in in_idx]
        z_in = _encode_with_grad(in_batch)  # (B, dim)
        sim = z_in @ out_centroid  # (B,)
        away = sim.mean()

        loss = (lambda_cls * bce) + (lambda_away * away)
        loss.backward()

        enc_optim.step()
        head_optim.step()

    model.save(str(out_path))
    return out_path


def main() -> None:
    """
    Train both specialists with explicit divergence.

    Tech specialist:
      in_domain = TECH_SENTENCES
      out_domain = COOK_SENTENCES

    Cook specialist:
      in_domain = COOK_SENTENCES
      out_domain = TECH_SENTENCES
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {BASE_MODEL}")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print()

    tech_path = train_specialist(
        domain_name="tech",
        in_domain=TECH_SENTENCES,
        out_domain=COOK_SENTENCES,
        out_dir=OUT_DIR,
        epochs=20,
        batch_size=16,
        lr=2e-5,
        seed=123,
        k=16,
        lambda_cls=0.8,
        lambda_away=0.8,
    )
    print(f"Saved tech specialist to: {tech_path}")

    cook_path = train_specialist(
        domain_name="cook",
        in_domain=COOK_SENTENCES,
        out_domain=TECH_SENTENCES,
        out_dir=OUT_DIR,
        epochs=20,
        batch_size=16,
        lr=2e-5,
        seed=456,
        k=16,
        lambda_cls=0.8,
        lambda_away=0.8,
    )
    print(f"Saved cook specialist to: {cook_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
