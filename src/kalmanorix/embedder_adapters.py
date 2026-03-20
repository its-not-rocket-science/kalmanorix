"""Embedder adapters for various model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from kalmanorix.types import Embedder, Vec

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI as OpenAIClient
    from cohere import Client as CohereClient
    from anthropic import Anthropic as AnthropicClient
    from google.cloud.aiplatform import VertexAIEmbeddingModel


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


__all__ = [
    "STEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "AnthropicEmbedder",
    "VertexAIEmbedder",
    "AzureOpenAIEmbedder",
]
