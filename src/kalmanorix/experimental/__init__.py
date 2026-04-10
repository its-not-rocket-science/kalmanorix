"""Experimental public API for :mod:`kalmanorix`.

This package contains advanced and still-evolving features that remain
importable for early adopters, but are not part of the stable top-level
``kalmanorix`` namespace contract.
"""

from ..alignment import (
    align_sef_list,
    apply_alignment,
    compute_alignments,
    validate_alignment_improvement,
    validate_alignment_sign,
)
from ..embedder_adapters import (
    AnthropicEmbedder,
    AzureOpenAIEmbedder,
    CohereEmbedder,
    HuggingFaceEmbedder,
    OnnxEmbedder,
    OpenAIEmbedder,
    STEmbedder,
    VertexAIEmbedder,
    create_azure_openai_sef,
    create_azure_openai_sef_with_calibration,
    create_cohere_sef,
    create_cohere_sef_with_calibration,
    create_huggingface_sef,
    create_huggingface_sef_model,
    create_onnx_sef,
    create_onnx_sef_model,
    create_onnx_sef_with_calibration,
    create_openai_sef,
    create_openai_sef_with_calibration,
    create_vertexai_sef,
    create_vertexai_sef_with_calibration,
)
from ..model_registry import ModelRegistry, get_default_registry
from ..panoramix import (
    CorrelationAwareKalmanFuser,
    DiagonalKalmanFuser,
    EnsembleKalmanFuser,
    LearnedGateFuser,
    StructuredKalmanFuser,
)
from ..threshold_heuristics import (
    threshold_adaptive_spread,
    threshold_query_length_adaptive,
    threshold_relative_to_max,
    threshold_top_k,
)

__all__ = [
    "STEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "AnthropicEmbedder",
    "VertexAIEmbedder",
    "AzureOpenAIEmbedder",
    "HuggingFaceEmbedder",
    "OnnxEmbedder",
    "create_huggingface_sef",
    "create_huggingface_sef_model",
    "create_onnx_sef",
    "create_onnx_sef_with_calibration",
    "create_onnx_sef_model",
    "create_openai_sef",
    "create_openai_sef_with_calibration",
    "create_cohere_sef",
    "create_cohere_sef_with_calibration",
    "create_vertexai_sef",
    "create_vertexai_sef_with_calibration",
    "create_azure_openai_sef",
    "create_azure_openai_sef_with_calibration",
    "threshold_top_k",
    "threshold_relative_to_max",
    "threshold_adaptive_spread",
    "threshold_query_length_adaptive",
    "ModelRegistry",
    "get_default_registry",
    "compute_alignments",
    "apply_alignment",
    "align_sef_list",
    "validate_alignment_improvement",
    "validate_alignment_sign",
    "EnsembleKalmanFuser",
    "StructuredKalmanFuser",
    "DiagonalKalmanFuser",
    "CorrelationAwareKalmanFuser",
    "LearnedGateFuser",
]
