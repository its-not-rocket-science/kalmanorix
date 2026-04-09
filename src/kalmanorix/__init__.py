"""Kalmanorix public API.

Intended audience and maturity tiers:

1. **Stable public API**
   - Recommended for most users and examples.
   - Backward-compatibility focused.
2. **Experimental API** (``kalmanorix.experimental``)
   - Advanced adapters, alignment helpers, and research fusers.
   - Still supported, but may evolve faster.
3. **Internal utilities** (``kalmanorix.internal`` and private modules)
   - Maintainer-focused internals.
   - No stability guarantees.

Compatibility note:
Experimental symbols that were previously available from the top-level module
are still temporarily accessible via deprecation shims.
"""

from __future__ import annotations

from importlib import import_module
import warnings

from .arena import eval_retrieval
from .panoramix import KalmanorixFuser, MeanFuser, Panoramix, Potion
from .scout import ScoutRouter
from .village import SEF, Village, compute_domain_centroid

__all__ = [
    "SEF",
    "Village",
    "compute_domain_centroid",
    "ScoutRouter",
    "Panoramix",
    "Potion",
    "MeanFuser",
    "KalmanorixFuser",
    "eval_retrieval",
]

_EXPERIMENTAL_SHIMS: dict[str, tuple[str, str]] = {
    "STEmbedder": ("kalmanorix.experimental", "STEmbedder"),
    "OpenAIEmbedder": ("kalmanorix.experimental", "OpenAIEmbedder"),
    "CohereEmbedder": ("kalmanorix.experimental", "CohereEmbedder"),
    "AnthropicEmbedder": ("kalmanorix.experimental", "AnthropicEmbedder"),
    "VertexAIEmbedder": ("kalmanorix.experimental", "VertexAIEmbedder"),
    "AzureOpenAIEmbedder": ("kalmanorix.experimental", "AzureOpenAIEmbedder"),
    "HuggingFaceEmbedder": ("kalmanorix.experimental", "HuggingFaceEmbedder"),
    "OnnxEmbedder": ("kalmanorix.experimental", "OnnxEmbedder"),
    "create_huggingface_sef": ("kalmanorix.experimental", "create_huggingface_sef"),
    "create_huggingface_sef_model": (
        "kalmanorix.experimental",
        "create_huggingface_sef_model",
    ),
    "create_onnx_sef": ("kalmanorix.experimental", "create_onnx_sef"),
    "create_onnx_sef_with_calibration": (
        "kalmanorix.experimental",
        "create_onnx_sef_with_calibration",
    ),
    "create_onnx_sef_model": ("kalmanorix.experimental", "create_onnx_sef_model"),
    "create_openai_sef": ("kalmanorix.experimental", "create_openai_sef"),
    "create_openai_sef_with_calibration": (
        "kalmanorix.experimental",
        "create_openai_sef_with_calibration",
    ),
    "create_cohere_sef": ("kalmanorix.experimental", "create_cohere_sef"),
    "create_cohere_sef_with_calibration": (
        "kalmanorix.experimental",
        "create_cohere_sef_with_calibration",
    ),
    "create_vertexai_sef": ("kalmanorix.experimental", "create_vertexai_sef"),
    "create_vertexai_sef_with_calibration": (
        "kalmanorix.experimental",
        "create_vertexai_sef_with_calibration",
    ),
    "create_azure_openai_sef": (
        "kalmanorix.experimental",
        "create_azure_openai_sef",
    ),
    "create_azure_openai_sef_with_calibration": (
        "kalmanorix.experimental",
        "create_azure_openai_sef_with_calibration",
    ),
    "threshold_top_k": ("kalmanorix.experimental", "threshold_top_k"),
    "threshold_relative_to_max": ("kalmanorix.experimental", "threshold_relative_to_max"),
    "threshold_adaptive_spread": ("kalmanorix.experimental", "threshold_adaptive_spread"),
    "threshold_query_length_adaptive": (
        "kalmanorix.experimental",
        "threshold_query_length_adaptive",
    ),
    "ModelRegistry": ("kalmanorix.experimental", "ModelRegistry"),
    "get_default_registry": ("kalmanorix.experimental", "get_default_registry"),
    "compute_alignments": ("kalmanorix.experimental", "compute_alignments"),
    "apply_alignment": ("kalmanorix.experimental", "apply_alignment"),
    "align_sef_list": ("kalmanorix.experimental", "align_sef_list"),
    "validate_alignment_improvement": (
        "kalmanorix.experimental",
        "validate_alignment_improvement",
    ),
    "validate_alignment_sign": ("kalmanorix.experimental", "validate_alignment_sign"),
    "EnsembleKalmanFuser": ("kalmanorix.experimental", "EnsembleKalmanFuser"),
    "StructuredKalmanFuser": ("kalmanorix.experimental", "StructuredKalmanFuser"),
    "DiagonalKalmanFuser": ("kalmanorix.experimental", "DiagonalKalmanFuser"),
    "LearnedGateFuser": ("kalmanorix.experimental", "LearnedGateFuser"),
    "create_procrustes_alignment": ("kalmanorix.internal", "create_procrustes_alignment"),
}


def __getattr__(name: str):
    """Resolve legacy top-level exports while warning about new import paths."""

    if name in _EXPERIMENTAL_SHIMS:
        module_name, symbol_name = _EXPERIMENTAL_SHIMS[name]
        warnings.warn(
            (
                f"'kalmanorix.{name}' is deprecated and will be removed from the "
                f"top-level namespace in a future release. Import it from "
                f"'{module_name}' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'kalmanorix' has no attribute '{name}'")
