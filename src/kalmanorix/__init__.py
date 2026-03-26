"""Kalmanorix public API."""

from .village import SEF, Village, compute_domain_centroid
from .scout import ScoutRouter
from .panoramix import (
    Panoramix,
    Potion,
    MeanFuser,
    KalmanorixFuser,
    EnsembleKalmanFuser,
    StructuredKalmanFuser,
    DiagonalKalmanFuser,
    LearnedGateFuser,
)
from .arena import eval_retrieval
from .embedder_adapters import (
    STEmbedder,
    OpenAIEmbedder,
    CohereEmbedder,
    AnthropicEmbedder,
    VertexAIEmbedder,
    AzureOpenAIEmbedder,
    HuggingFaceEmbedder,
)
from .model_registry import ModelRegistry, get_default_registry
from .threshold_heuristics import (
    threshold_top_k,
    threshold_relative_to_max,
    threshold_adaptive_spread,
    threshold_query_length_adaptive,
)

__all__ = [
    "SEF",
    "Village",
    "compute_domain_centroid",
    "ScoutRouter",
    "Panoramix",
    "Potion",
    "MeanFuser",
    "KalmanorixFuser",
    "EnsembleKalmanFuser",
    "StructuredKalmanFuser",
    "DiagonalKalmanFuser",
    "LearnedGateFuser",
    "eval_retrieval",
    "STEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "AnthropicEmbedder",
    "VertexAIEmbedder",
    "AzureOpenAIEmbedder",
    "HuggingFaceEmbedder",
    "threshold_top_k",
    "threshold_relative_to_max",
    "threshold_adaptive_spread",
    "threshold_query_length_adaptive",
    "ModelRegistry",
    "get_default_registry",
]
