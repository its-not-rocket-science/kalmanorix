"""Embedder adapters for various model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from kalmanorix.types import Embedder, Vec

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI as OpenAIClient
    from cohere import Client as CohereClient
    from anthropic import Anthropic as AnthropicClient
    from google.cloud.aiplatform import VertexAIEmbeddingModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _normalize(vec: Vec) -> Vec:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


@dataclass(frozen=True)
class STEmbedder(Embedder):
    """SentenceTransformer-backed embedder implementing kalmanorix.types.Embedder."""

    model: "SentenceTransformer"

    def __call__(self, text: str) -> Vec:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[
            0
        ]
        return v.astype(np.float64)


@dataclass(frozen=True)
class OpenAIEmbedder(Embedder):
    """OpenAI embedding API adapter.

    Args:
        client: OpenAI client instance (from `openai.OpenAI()`)
        model: Model name, e.g., "text-embedding-3-small"
        dimensions: Optional output dimensions (if model supports it)
    """

    client: "OpenAIClient"
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    normalize: bool = True

    def __call__(self, text: str) -> Vec:
        try:
            from openai import OpenAIError
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK not installed. Install with: pip install openai"
            ) from e

        try:
            kwargs: dict[str, Any] = {"model": self.model, "input": text}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            embedding = response.data[0].embedding
            vec = np.array(embedding, dtype=np.float64)
            if self.normalize:
                vec = _normalize(vec)
            return vec
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


@dataclass(frozen=True)
class CohereEmbedder(Embedder):
    """Cohere embedding API adapter.

    Args:
        client: Cohere client instance (from `cohere.Client()`)
        model: Model name, e.g., "embed-english-v3.0"
        input_type: One of "search_document", "search_query", "classification", "clustering"
    """

    client: "CohereClient"
    model: str = "embed-english-v3.0"
    input_type: str = "search_document"
    normalize: bool = True

    def __call__(self, text: str) -> Vec:
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "Cohere SDK not installed. Install with: pip install cohere"
            ) from e

        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type,
            )
            embedding = response.embeddings[0]
            vec = np.array(embedding, dtype=np.float64)
            if self.normalize:
                vec = _normalize(vec)
            return vec
        except cohere.CohereError as e:
            raise RuntimeError(f"Cohere API error: {e}") from e


@dataclass(frozen=True)
class AnthropicEmbedder(Embedder):
    """Anthropic embedding API adapter.

    Note: As of March 2026, Anthropic does not offer a dedicated embedding API.
    This adapter uses the Claude model to generate embeddings via the messages API
    if/when available. Currently raises NotImplementedError.
    """

    client: "AnthropicClient"
    model: str = "claude-3-5-sonnet-20241022"

    def __call__(self, text: str) -> Vec:
        raise NotImplementedError(
            "Anthropic does not currently offer an embedding API. "
            "Use OpenAI, Cohere, or VertexAI adapters instead."
        )


@dataclass(frozen=True)
class VertexAIEmbedder(Embedder):
    """Google Vertex AI embedding adapter.

    Args:
        model: VertexAIEmbeddingModel instance
        task_type: Task type for embeddings, e.g., "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"
    """

    model: "VertexAIEmbeddingModel"
    task_type: str = "RETRIEVAL_QUERY"
    normalize: bool = True

    def __call__(self, text: str) -> Vec:
        try:
            from google.api_core.exceptions import GoogleAPIError
        except ImportError as e:
            raise ImportError(
                "Google Cloud AI Platform not installed. Install with: pip install google-cloud-aiplatform"
            ) from e

        try:
            embeddings = self.model.get_embeddings([text], task_type=self.task_type)
            embedding = embeddings[0].values
            vec = np.array(embedding, dtype=np.float64)
            if self.normalize:
                vec = _normalize(vec)
            return vec
        except GoogleAPIError as e:
            raise RuntimeError(f"Vertex AI API error: {e}") from e


@dataclass(frozen=True)
class AzureOpenAIEmbedder(Embedder):
    """Azure OpenAI embedding adapter.

    This uses the same OpenAI SDK but with Azure-specific configuration.

    Args:
        client: OpenAI client configured for Azure (with azure_endpoint, api_version, etc.)
        model: Deployment name, e.g., "my-embedding-deployment"
        dimensions: Optional output dimensions
    """

    client: "OpenAIClient"
    model: str = field(default_factory=lambda: "embedding-deployment")
    dimensions: int | None = None
    normalize: bool = True

    def __call__(self, text: str) -> Vec:
        try:
            from openai import OpenAIError
        except ImportError as e:
            raise ImportError(
                "OpenAI SDK not installed. Install with: pip install openai"
            ) from e

        try:
            kwargs: dict[str, Any] = {"model": self.model, "input": text}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            embedding = response.data[0].embedding
            vec = np.array(embedding, dtype=np.float64)
            if self.normalize:
                vec = _normalize(vec)
            return vec
        except OpenAIError as e:
            raise RuntimeError(f"Azure OpenAI API error: {e}") from e


@dataclass(frozen=True)
class HuggingFaceEmbedder(Embedder):
    """Hugging Face transformer model embedder.

    Args:
        model_name_or_path: Model identifier or local path
        tokenizer_name_or_path: Optional tokenizer identifier (defaults to same as model)
        pooling: Pooling strategy: "mean" (average of token embeddings) or "cls" (use [CLS] token)
        device: Device to run model on ("cpu" or "cuda")
        max_length: Maximum token length (default 512)
        normalize: Whether to normalize output embeddings to unit length
    """

    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    pooling: Literal["mean", "cls"] = "mean"
    device: str = "cpu"
    max_length: int = 512
    normalize: bool = True
    _model: Any = field(init=False, default=None, repr=False)
    _tokenizer: Any = field(init=False, default=None, repr=False)

    @property
    def model(self) -> "PreTrainedModel":
        """Lazy load the model."""
        if self._model is None:
            try:
                from transformers import AutoModel
            except ImportError as e:
                raise ImportError(
                    "Transformers library not installed. Install with: pip install transformers"
                ) from e

            model = AutoModel.from_pretrained(self.model_name_or_path)
            model.to(self.device)
            model.eval()
            object.__setattr__(self, "_model", model)
        return self._model

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "Transformers library not installed. Install with: pip install transformers"
                ) from e

            tokenizer_name = self.tokenizer_name_or_path or self.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            object.__setattr__(self, "_tokenizer", tokenizer)
        return self._tokenizer

    def __call__(self, text: str) -> Vec:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch"
            ) from e

        # Tokenize
        inputs = self.tokenizer(  # pylint: disable=not-callable
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass (no gradient)
        with torch.no_grad():
            outputs = self.model(**inputs)  # pylint: disable=not-callable
            last_hidden_state = (
                outputs.last_hidden_state
            )  # (batch, seq_len, hidden_dim)

        # Pooling
        if self.pooling == "cls":
            # Use [CLS] token embedding (first token)
            embeddings = last_hidden_state[:, 0, :]
        elif self.pooling == "mean":
            # Mean pooling, ignoring padding tokens
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                # Expand mask to hidden dimension
                mask = (
                    attention_mask.unsqueeze(-1)
                    .expand(last_hidden_state.size())
                    .float()
                )
                # Sum embeddings, divide by sum of mask
                sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                # No attention mask, simple mean across sequence dimension
                embeddings = torch.mean(last_hidden_state, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Convert to numpy
        vec = embeddings.cpu().numpy().astype(np.float64).flatten()

        # Normalize if requested
        if self.normalize:
            vec = _normalize(vec)

        return vec


__all__ = [
    "STEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "AnthropicEmbedder",
    "VertexAIEmbedder",
    "AzureOpenAIEmbedder",
    "HuggingFaceEmbedder",
]
