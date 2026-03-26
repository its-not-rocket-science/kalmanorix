"""
Model registry for SEF artefacts.

Provides directory scanning, metadata indexing, and lazy loading of SEF models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Any


from .registry import EmbedderRegistry, EmbedderFn

logger = logging.getLogger(__name__)


class ModelMetadata(TypedDict):
    """Metadata for a discovered SEF model."""

    model_id: str
    name: str
    version: str
    domain_tags: List[str]
    description: str
    path: str


@dataclass
class ModelRegistry:
    """Registry for SEF models stored in a local directory.

    Scans a base directory for SEF artefacts (subdirectories containing
    metadata.json) and provides lazy loading of embedders.

    Args:
        base_dir: Root directory to scan for SEF models
        embedder_registry: Optional embedder registry for sigma2 resolution
            (if None, creates a new one)
    """

    base_dir: Path
    embedder_registry: EmbedderRegistry = field(default_factory=EmbedderRegistry)
    _metadata: Dict[str, ModelMetadata] = field(default_factory=dict, init=False)
    _loaded_models: Dict[str, Any] = field(
        default_factory=dict, init=False
    )  # SEFModel instances

    def __post_init__(self) -> None:
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> List[str]:
        """Scan base directory for SEF models and index metadata.

        Returns:
            List of model IDs (directory names) found.
        """
        self._metadata.clear()

        for model_dir in self.base_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metadata_file = model_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                model_id = model_dir.name
                self._metadata[model_id] = ModelMetadata(
                    model_id=model_id,
                    name=metadata.get("name", model_id),
                    version=metadata.get("version", "unknown"),
                    domain_tags=metadata.get("domain_tags", []),
                    description=metadata.get("description", ""),
                    path=str(model_dir),
                )
                logger.debug("Found SEF model: %s (%s)", model_id, metadata.get("name"))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Invalid metadata in %s: %s", model_dir, e)
                continue

        logger.info("Scanned %d SEF models from %s", len(self._metadata), self.base_dir)
        return list(self._metadata.keys())

    def list_models(self) -> Dict[str, ModelMetadata]:
        """Return metadata for all discovered models.

        Returns:
            Dictionary mapping model ID to metadata.
        """
        if not self._metadata:
            self.scan()
        return self._metadata.copy()

    def load_model(self, model_id: str) -> Any:
        """Load SEFModel instance for given model ID.

        Args:
            model_id: Model identifier (directory name)

        Returns:
            SEFModel instance

        Raises:
            KeyError: If model_id not found
            ImportError: If required dependencies missing
        """
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        if model_id not in self._metadata:
            # Try to scan first
            self.scan()
            if model_id not in self._metadata:
                raise KeyError(f"Model not found: {model_id}")

        model_dir = Path(self._metadata[model_id]["path"])

        try:
            from .models.sef import SEFModel
        except ImportError as e:
            raise ImportError(
                "SEFModel not available. Make sure kalmanorix is properly installed."
            ) from e

        # Load with default embedder loader (pickle)
        model = SEFModel.from_pretrained(model_dir)
        self._loaded_models[model_id] = model

        # Register embedder in embedder registry
        self.embedder_registry.embedders[model_id] = model.embed

        logger.debug("Loaded model: %s", model_id)
        return model

    def get_embedder(self, model_id: str) -> EmbedderFn:
        """Get embedder function for given model ID (cached).

        Args:
            model_id: Model identifier

        Returns:
            Embedder function (str -> np.ndarray)
        """
        if model_id in self.embedder_registry.embedders:
            return self.embedder_registry.embedders[model_id]

        model = self.load_model(model_id)
        return model.embed

    def register_embedder(self, name: str, embedder: EmbedderFn) -> None:
        """Register a custom embedder function.

        Args:
            name: Embedder identifier
            embedder: Callable that takes text and returns embedding vector
        """
        self.embedder_registry.embedders[name] = embedder
        logger.debug("Registered embedder: %s", name)

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get metadata for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model metadata

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in self._metadata:
            self.scan()
            if model_id not in self._metadata:
                raise KeyError(f"Model not found: {model_id}")
        return self._metadata[model_id].copy()


# Default global registry (can be configured by users)
_DEFAULT_BASE_DIR = Path.home() / ".kalmanorix" / "models"
_default_registry: Optional[ModelRegistry] = None


# pylint: disable=global-statement
def get_default_registry() -> ModelRegistry:
    """Get or create the default global model registry.

    The default registry uses `~/.kalmanorix/models` as base directory.

    Returns:
        ModelRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry(_DEFAULT_BASE_DIR)
    return _default_registry


# pylint: disable=global-statement
def set_default_registry(registry: ModelRegistry) -> None:
    """Set the default global model registry.

    Args:
        registry: ModelRegistry instance to use as default
    """
    global _default_registry
    _default_registry = registry
