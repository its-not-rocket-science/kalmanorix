#!/usr/bin/env python3
"""Unit tests for SEFModel (Shareable Embedding Format)."""

import json

import numpy as np
import pytest

from kalmanorix.models.sef import SEFModel, SEFMetadata, create_procrustes_alignment


def dummy_embedder(text: str) -> np.ndarray:
    """Dummy embedder for testing."""
    # Deterministic based on text hash
    seed = hash(text) % 1000
    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    vec = rng.randn(384).astype(np.float64)
    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def test_sefmetadata_serialization():
    """Test SEFMetadata JSON serialization round-trip."""
    meta = SEFMetadata(
        model_id="test-123",
        name="Test Model",
        version="1.0.0",
        description="A test embedding model",
        domain_tags=["biomedical", "entity_recognition"],
        task_tags=["ner", "classification"],
        benchmarks={"sts-b": 0.85, "squad-f1": 0.92},
        training_data_description="PubMed abstracts",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        training_date="2026-03-21",
        author="Test Author",
        licence="MIT",
        embedding_dimension=384,
        covariance_format="diagonal",
        alignment_method="procrustes",
        checksum="",
    )

    # Round-trip through JSON
    json_str = meta.to_json()
    loaded = SEFMetadata.from_json(json_str)

    assert loaded.model_id == meta.model_id
    assert loaded.name == meta.name
    assert loaded.version == meta.version
    assert loaded.domain_tags == meta.domain_tags
    assert loaded.task_tags == meta.task_tags
    assert loaded.benchmarks == meta.benchmarks
    assert loaded.training_data_description == meta.training_data_description
    assert loaded.base_model == meta.base_model
    assert loaded.training_date == meta.training_date
    assert loaded.author == meta.author
    assert loaded.licence == meta.licence
    assert loaded.embedding_dimension == meta.embedding_dimension
    assert loaded.covariance_format == meta.covariance_format
    assert loaded.alignment_method == meta.alignment_method
    # checksum may be updated later


def test_sefmodel_creation():
    """Test basic SEFModel creation and validation."""
    metadata = SEFMetadata(
        model_id="test-1",
        name="Test",
        version="1.0",
        description="Test",
        domain_tags=["test"],
        task_tags=["test"],
        benchmarks={"test": 0.9},
        training_data_description="Test",
        base_model="dummy",
        training_date="2026-03-21",
        author="Test",
        licence="MIT",
        embedding_dimension=384,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="",
    )

    model = SEFModel(
        embed_function=dummy_embedder,
        metadata=metadata,
        alignment_matrix=None,
        covariance_data={"method": "fixed", "diagonal": np.ones(384) * 0.1},
    )

    assert model.dimension == 384
    assert model.metadata.model_id == "test-1"

    # Test embedding
    emb = model.embed("hello world")
    assert emb.shape == (384,)
    assert emb.dtype == np.float64

    # Test covariance
    cov = model.get_covariance("hello world")
    assert cov.shape == (384,)
    assert np.allclose(cov, 0.1)


def test_sefmodel_save_load(tmp_path):
    """Test save_pretrained and from_pretrained round-trip."""
    metadata = SEFMetadata(
        model_id="save-test",
        name="Save Test",
        version="1.0",
        description="Test saving",
        domain_tags=["test"],
        task_tags=["test"],
        benchmarks={"test": 0.95},
        training_data_description="Synthetic",
        base_model="dummy",
        training_date="2026-03-21",
        author="Test",
        licence="MIT",
        embedding_dimension=384,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="",
    )

    # Create model with alignment matrix and covariance
    alignment = np.eye(384)
    covariance_data = {
        "method": "fixed",
        "diagonal": np.ones(384) * 0.05,
    }

    model = SEFModel(
        embed_function=dummy_embedder,
        metadata=metadata,
        alignment_matrix=alignment,
        covariance_data=covariance_data,
    )

    # Save to temporary directory
    save_dir = tmp_path / "test_model"
    model.save_pretrained(save_dir)

    # Verify files were created
    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "alignment.npy").exists()
    assert (save_dir / "covariance.npz").exists()
    assert (save_dir / "covariance_config.json").exists()
    assert (save_dir / "model.pkl").exists()  # Embedder is pickleable
    assert (save_dir / "checksum.txt").exists()

    # Load back
    loaded = SEFModel.from_pretrained(save_dir, embed_loader=lambda p: dummy_embedder)

    # Verify properties match
    assert loaded.dimension == model.dimension
    assert loaded.metadata.model_id == model.metadata.model_id
    assert loaded.metadata.name == model.metadata.name
    assert loaded.metadata.domain_tags == model.metadata.domain_tags
    assert loaded.metadata.benchmarks == model.metadata.benchmarks
    assert loaded.metadata.licence == model.metadata.licence

    # Verify alignment matrix
    assert loaded.alignment_matrix is not None
    assert np.allclose(loaded.alignment_matrix, alignment)

    # Verify covariance
    cov = loaded.get_covariance("test")
    assert np.allclose(cov, 0.05)

    # Verify embedding function works
    emb1 = model.embed("test query")
    emb2 = loaded.embed("test query")
    assert np.allclose(emb1, emb2)


def test_sefmodel_with_distance_based_covariance(tmp_path):
    """Test SEFModel with distance-based covariance estimation."""
    metadata = SEFMetadata(
        model_id="dist-cov",
        name="Distance Covariance",
        version="1.0",
        description="Test distance-based covariance",
        domain_tags=["test"],
        task_tags=["test"],
        benchmarks={},
        training_data_description="Synthetic",
        base_model="dummy",
        training_date="2026-03-21",
        author="Test",
        licence="MIT",
        embedding_dimension=384,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="",
    )

    # Create reference embeddings (10 samples, 384 dim)
    rng = np.random.RandomState(42)  # pylint: disable=no-member
    reference_embeddings = rng.randn(10, 384).astype(np.float64)
    # Normalize each row
    norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    reference_embeddings = reference_embeddings / (norms + 1e-8)

    covariance_data = {
        "method": "distance_based",
        "diagonal": np.ones(384) * 0.1,
        "alpha": 2.0,
        "reference_embeddings": reference_embeddings,
    }

    model = SEFModel(
        embed_function=dummy_embedder,
        metadata=metadata,
        alignment_matrix=None,
        covariance_data=covariance_data,
    )

    # Test covariance varies with distance
    query1 = "similar query"  # Should be somewhat similar to references (random)
    query2 = "different query"  # Different
    cov1 = model.get_covariance(query1)
    cov2 = model.get_covariance(query2)

    # Covariance should be at least base value
    assert np.all(cov1 >= 0.1)
    assert np.all(cov2 >= 0.1)
    # May differ due to distance scaling
    # (not asserting equality since random)

    # Save and load
    save_dir = tmp_path / "dist_model"
    model.save_pretrained(save_dir)

    # Load back (need embed_loader since reference_embeddings is numpy array)
    loaded = SEFModel.from_pretrained(save_dir, embed_loader=lambda p: dummy_embedder)

    # Verify covariance similar
    cov1_loaded = loaded.get_covariance(query1)
    cov2_loaded = loaded.get_covariance(query2)
    assert np.allclose(cov1, cov1_loaded, rtol=1e-5)
    assert np.allclose(cov2, cov2_loaded, rtol=1e-5)


def test_sefmodel_checksum_verification(tmp_path):
    """Test checksum computation and verification."""
    metadata = SEFMetadata(
        model_id="checksum-test",
        name="Checksum Test",
        version="1.0",
        description="Test checksum",
        domain_tags=["test"],
        task_tags=["test"],
        benchmarks={},
        training_data_description="Synthetic",
        base_model="dummy",
        training_date="2026-03-21",
        author="Test",
        licence="MIT",
        embedding_dimension=384,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="",
    )

    model = SEFModel(
        embed_function=dummy_embedder,
        metadata=metadata,
        alignment_matrix=None,
        covariance_data={"method": "fixed", "diagonal": np.ones(384) * 0.01},
    )

    save_dir = tmp_path / "checksum_model"
    model.save_pretrained(save_dir)

    # Should load successfully (checksum verified internally)
    loaded = SEFModel.from_pretrained(save_dir, embed_loader=lambda p: dummy_embedder)
    assert loaded.dimension == 384

    # Tamper with a file (in a way that still produces valid JSON but different content)
    with open(save_dir / "metadata.json", "r") as f:
        meta_data = json.load(f)
    meta_data["description"] = "Tampered description"
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta_data, f, indent=2)

    # Should raise ValueError due to checksum mismatch
    with pytest.raises(ValueError, match="Checksum mismatch"):
        SEFModel.from_pretrained(save_dir, embed_loader=lambda p: dummy_embedder)


def test_create_procrustes_alignment():
    """Test Procrustes alignment matrix computation."""
    n = 100
    d = 384

    # Generate random source and target embeddings
    rng = np.random.RandomState(123)  # pylint: disable=no-member
    source = rng.randn(n, d)
    target = rng.randn(n, d)

    # Compute alignment matrix
    Q = create_procrustes_alignment(source, target)

    # Verify shape and orthogonality
    assert Q.shape == (d, d)
    assert np.allclose(Q.T @ Q, np.eye(d), atol=1e-10)
    assert np.allclose(Q @ Q.T, np.eye(d), atol=1e-10)

    # Verify determinant is +1 (proper rotation)
    assert np.linalg.det(Q) > 0.99  # Should be close to 1

    # Verify alignment reduces distance
    aligned = source @ Q
    dist_before = np.mean(np.linalg.norm(source - target, axis=1))
    dist_after = np.mean(np.linalg.norm(aligned - target, axis=1))
    assert dist_after < dist_before  # Alignment should reduce distance


def test_sefmodel_dimension_mismatch():
    """Test that dimension mismatch raises error."""
    metadata = SEFMetadata(
        model_id="mismatch",
        name="Mismatch",
        version="1.0",
        description="Test",
        domain_tags=["test"],
        task_tags=["test"],
        benchmarks={},
        training_data_description="Test",
        base_model="dummy",
        training_date="2026-03-21",
        author="Test",
        licence="MIT",
        embedding_dimension=512,  # Different from embedder!
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="",
    )

    # Embedder returns 384-dim vectors
    def embedder_384(text: str) -> np.ndarray:
        return np.ones(384, dtype=np.float64)

    with pytest.raises(ValueError, match="does not match"):
        SEFModel(
            embed_function=embedder_384,
            metadata=metadata,
            alignment_matrix=None,
            covariance_data={},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
