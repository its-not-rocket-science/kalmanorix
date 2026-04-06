"""Non-linear alignment using contrastive learning.

For specialists fine-tuned from the same base model, Procrustes often
over-rotates because the relationship is non-linear. This module provides
a learned alignment network that preserves semantic similarity while
correcting non-linear distortions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class AlignmentNetwork(nn.Module):
    """Small MLP that learns to map specialist embeddings to reference space."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: Optional[int] = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 normalize output to preserve cosine similarity structure
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


def _info_nce_loss(
    projected: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    projected = F.normalize(projected, p=2, dim=-1)
    targets = F.normalize(targets, p=2, dim=-1)
    logits = torch.matmul(projected, targets.t()) / temperature
    labels = torch.arange(projected.size(0), device=projected.device)
    return F.cross_entropy(logits, labels)


def train_alignment(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> AlignmentNetwork:
    """Train alignment network using cosine similarity + InfoNCE loss."""
    if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("source_embeddings and target_embeddings must be shape (n, d)")
    if source_embeddings.shape != target_embeddings.shape:
        raise ValueError("source_embeddings and target_embeddings must have the same shape")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.from_numpy(source_embeddings.astype(np.float32))
    target = torch.from_numpy(target_embeddings.astype(np.float32))

    model = AlignmentNetwork(input_dim=source.shape[1], output_dim=target.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    dataset = TensorDataset(source, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            pred = model(src_batch)
            info_nce = _info_nce_loss(pred, tgt_batch)
            cosine = 1.0 - F.cosine_similarity(pred, F.normalize(tgt_batch, p=2, dim=-1), dim=-1).mean()
            loss = 0.7 * info_nce + 0.3 * cosine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def apply_nonlinear_alignment(
    embeddings: np.ndarray,
    network: AlignmentNetwork,
) -> np.ndarray:
    """Apply trained alignment network."""
    with torch.no_grad():
        x = torch.from_numpy(embeddings).float()
        aligned = network(x)
    return aligned.cpu().numpy()
