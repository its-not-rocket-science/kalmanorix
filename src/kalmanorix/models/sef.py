"""Shareable Embedding Format (SEF) implementation.

This module defines the standard for packaging and sharing specialist models.
A SEF model contains everything needed for Kalman fusion:

1. Core embedding vectors for a reference anchor set
2. Uncertainty covariance matrix (diagonal approximation)
3. Alignment matrix (Procrustes) to map into reference space
4. Metadata manifest (domain, benchmarks, licence)

The format is designed to be:
- Self-contained: All needed data in one file
- Lightweight: Diagonal covariance, not full matrices
- Interoperable: Can be loaded by any KEFF-compliant framework
- Verifiable: Includes benchmark results for quality assessment
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, asdict
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SEFMetadata:
    """Metadata manifest for a SEF model.

    This contains all the human-readable information about the model,
    enabling discovery and appropriate selection.
    """

    # Basic identification
    model_id: str
    name: str
    version: str
    description: str

    # Domain and capabilities
    domain_tags: List[str]  # e.g., ["biomedical", "entity_recognition"]
    task_tags: List[str]  # e.g., ["ner", "classification"]

    # Performance metrics
    benchmarks: Dict[str, float]  # e.g., {"sts-b": 0.85, "squad-f1": 0.92}

    # Provenance
    training_data_description: str  # High-level only, no raw data
    base_model: str  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    training_date: str
    author: str
    licence: str

    # Technical
    embedding_dimension: int
    covariance_format: str  # "diagonal", "low_rank", "full"
    alignment_method: str  # "procrustes", "identity", "learned"

    # Verification
    checksum: str  # SHA-256 of the binary data

    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SEFMetadata":
        """Create metadata from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class SEFModel:
    """Shareable Embedding Format model.

    This class represents a specialist model packaged for KEFF. It can be
    saved to disk, loaded, and used in fusion operations.

    The model provides two key capabilities:
    1. embed(text): Convert text to embedding vector
    2. get_covariance(text): Estimate uncertainty for this text
    """

    def __init__(
        self,
        embed_function: Callable[[str], np.ndarray],
        metadata: SEFMetadata,
        alignment_matrix: Optional[np.ndarray] = None,
        covariance_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialise a SEF model.

        Args:
            embed_function: Function that takes string and returns embedding
            metadata: Model metadata
            alignment_matrix: Optional (d x d) orthogonal matrix for space alignment
            covariance_data: Optional uncertainty estimation data
        """
        self.embed_function = embed_function
        self.metadata = metadata
        self.alignment_matrix = alignment_matrix
        self.covariance_data = covariance_data or {}

        # Validate dimension
        test_emb = self.embed_function("test")
        if test_emb.shape[0] != metadata.embedding_dimension:
            raise ValueError(
                f"Embedding dimension {test_emb.shape[0]} does not match "
                f"metadata {metadata.embedding_dimension}"
            )

        self.dimension = metadata.embedding_dimension
        logger.info(
            "Loaded SEF model: %s v%s, domain: %s",
            metadata.name,
            metadata.version,
            metadata.domain_tags,
        )

    def embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        return self.embed_function(text)

    def get_covariance(self, text: str) -> np.ndarray:
        """Get diagonal covariance for this text.

        Returns a vector of length d representing uncertainty in each dimension.
        Higher values = more uncertain.

        The implementation depends on what covariance_data contains:
        - If 'fixed': return stored covariance for all inputs
        - If 'distance_based': scale stored covariance by distance to reference
        - If 'model_based': use a separate uncertainty model
        """
        method = self.covariance_data.get("method", "fixed")

        if method == "fixed":
            # Return pre-computed diagonal
            return self.covariance_data["diagonal"].copy()

        if method == "distance_based":
            # Scale by distance to reference set
            base_cov = self.covariance_data["diagonal"]
            alpha = self.covariance_data.get("alpha", 1.0)
            ref_embeddings = self.covariance_data["reference_embeddings"]

            # Compute distance
            emb = self.embed(text)
            # Normalise for cosine distance
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            ref_norm = ref_embeddings / (
                np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8
            )
            similarities = ref_norm @ emb_norm
            distance = 1.0 - np.max(similarities)

            return base_cov * (1.0 + alpha * distance)

        else:
            # Unknown method, return ones (max uncertainty)
            logger.warning(
                "Unknown covariance method '%s', returning unit covariance", method
            )
            return np.ones(self.dimension)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model to disk in SEF format.

        Creates a directory containing:
        - metadata.json: Model metadata
        - model.pkl: Pickled embed_function (if pickleable)
        - alignment.npy: Alignment matrix (if exists)
        - covariance.npz: Covariance data (if exists)

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            f.write(self.metadata.to_json())

        # Save alignment matrix
        if self.alignment_matrix is not None:
            np.save(path / "alignment.npy", self.alignment_matrix)

        # Save covariance data
        if self.covariance_data:
            cov_path = path / "covariance.npz"
            # Convert any arrays to numpy
            arrays = {}
            for k, v in self.covariance_data.items():
                if isinstance(v, np.ndarray):
                    arrays[k] = v
            if arrays:
                np.savez(cov_path, **arrays)  # type: ignore
            # Save non-array data as JSON
            non_arrays = {
                k: v
                for k, v in self.covariance_data.items()
                if not isinstance(v, np.ndarray)
            }
            if non_arrays:
                with open(path / "covariance_config.json", "w", encoding="utf-8") as f:
                    json.dump(non_arrays, f, indent=2)

        # Try to pickle the embed function
        try:
            with open(path / "model.pkl", "wb") as f:
                pickle.dump(self.embed_function, f)
        except (pickle.PicklingError, AttributeError) as e:
            logger.warning("Could not pickle embed_function: %s", e)
            # Create a wrapper that explains how to load
            with open(path / "LOAD_INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
                f.write(
                    "This model's embed_function could not be pickled.\n"
                    "You need to provide your own loading function.\n"
                    f"Base model: {self.metadata.base_model}\n"
                    f"Description: {self.metadata.description}\n"
                )

        # Compute and verify checksum
        self._write_checksum(path)

        logger.info("Saved SEF model to %s", path)

    @classmethod
    def from_pretrained(
        cls, path: Union[str, Path], embed_loader: Optional[Callable] = None
    ) -> "SEFModel":
        """Load model from disk.

        Args:
            path: Directory containing SEF files
            embed_loader: Optional function to load embed_function if not pickled

        Returns:
            Loaded SEFModel
        """
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = SEFMetadata.from_json(f.read())

        # Load alignment matrix
        alignment_matrix = None
        if (path / "alignment.npy").exists():
            alignment_matrix = np.load(path / "alignment.npy")

        # Load covariance data
        covariance_data = {}
        if (path / "covariance.npz").exists():
            with np.load(path / "covariance.npz") as data:
                covariance_data = dict(data)
        if (path / "covariance_config.json").exists():
            with open(path / "covariance_config.json", "r", encoding="utf-8") as f:
                covariance_data.update(json.load(f))

        # Load embed function
        embed_function = None
        if (path / "model.pkl").exists():
            with open(path / "model.pkl", "rb") as f:
                embed_function = pickle.load(f)
        elif embed_loader is not None:
            embed_function = embed_loader(path)
        else:
            raise ValueError(
                f"No pickled model found at {path} and no embed_loader provided.\n"
                f"See {path}/LOAD_INSTRUCTIONS.txt if it exists."
            )

        # Verify checksum
        cls._verify_checksum(path)

        return cls(embed_function, metadata, alignment_matrix, covariance_data)

    def _write_checksum(self, path: Path) -> None:
        """Compute and save SHA-256 checksum of all model files."""
        hasher = hashlib.sha256()
        for file in sorted(path.glob("*")):
            if file.name == "checksum.txt":
                continue
            with open(file, "rb") as f:
                hasher.update(f.read())
        checksum = hasher.hexdigest()

        with open(path / "checksum.txt", "w", encoding="utf-8") as f:
            f.write(checksum)

    @classmethod
    def _verify_checksum(cls, path: Path) -> bool:
        """Verify checksum of loaded model."""
        if not (path / "checksum.txt").exists():
            logger.warning("No checksum found for %s, skipping verification", path)
            return True

        with open(path / "checksum.txt", "r", encoding="utf-8") as f:
            expected = f.read().strip()

        hasher = hashlib.sha256()
        for file in sorted(path.glob("*")):
            if file.name == "checksum.txt":
                continue
            with open(file, "rb") as f:
                hasher.update(f.read())
        actual = hasher.hexdigest()

        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected}, got {actual}"
            )

        logger.info("Checksum verified for %s", path)
        return True


def create_procrustes_alignment(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute orthogonal Procrustes alignment matrix.

    Finds orthogonal matrix Q that minimises ||source_embeddings @ Q - target_embeddings||_F

    This is used to align embeddings from different specialist spaces into
    a common reference space.

    Args:
        source_embeddings: Embeddings from source model, shape (n, d)
        target_embeddings: Corresponding embeddings in reference space, shape (n, d)

    Returns:
        Q: Orthogonal matrix (d, d) that maps source to target space

    Mathematical derivation:
        We want Q* = argmin_Q ||A - BQ||_F subject to Q^T Q = I
        Solution: Q* = UV^T where A^T B = UΣV^T (SVD)
    """
    # Centre the data
    source_centered = source_embeddings - np.mean(source_embeddings, axis=0)
    target_centered = target_embeddings - np.mean(target_embeddings, axis=0)

    # Compute cross-covariance matrix
    C = source_centered.T @ target_centered

    # SVD
    U, _, Vt = np.linalg.svd(C, full_matrices=False)

    # Orthogonal Procrustes solution
    Q = U @ Vt

    # Ensure determinant is +1 (proper rotation, not reflection)
    if np.linalg.det(Q) < 0:
        Vt[-1, :] *= -1
        Q = U @ Vt

    return Q
