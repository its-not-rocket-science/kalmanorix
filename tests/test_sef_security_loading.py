import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from kalmanorix.models.sef import (
    SEFMetadata,
    SEFModel,
    SafeSEFLoader,
    TrustedPickleSEFLoader,
)


def _dummy_embedder(text: str) -> np.ndarray:
    return np.ones(4, dtype=np.float64) * len(text)


def _write_minimal_sef(path: Path) -> None:
    metadata = SEFMetadata(
        model_id="secure_model",
        name="Secure Model",
        version="1.0.0",
        description="security test model",
        domain_tags=["test"],
        task_tags=["embedding"],
        benchmarks={},
        training_data_description="Synthetic",
        base_model="dummy",
        training_date="2026-01-01",
        author="Test",
        licence="MIT",
        embedding_dimension=4,
        covariance_format="diagonal",
        alignment_method="identity",
        checksum="dummy",
    )
    model = SEFModel(
        embed_function=_dummy_embedder,
        metadata=metadata,
        covariance_data={"method": "fixed", "diagonal": np.ones(4)},
    )
    model.save_pretrained(path)


def test_safe_loader_rejects_pickle_without_embed_loader(tmp_path: Path):
    model_dir = tmp_path / "secure"
    _write_minimal_sef(model_dir)

    with pytest.raises(ValueError, match="blocks pickle execution"):
        SEFModel.from_pretrained(model_dir)


def test_safe_loader_uses_explicit_embed_loader(tmp_path: Path):
    model_dir = tmp_path / "secure"
    _write_minimal_sef(model_dir)

    loaded = SEFModel.from_pretrained(
        model_dir, embed_loader=lambda _p: _dummy_embedder
    )
    vec = loaded.embed("abc")
    assert vec.shape == (4,)
    assert np.allclose(vec, np.ones(4) * 3)


def test_trusted_pickle_loader_requires_explicit_opt_in(tmp_path: Path):
    model_dir = tmp_path / "legacy"
    _write_minimal_sef(model_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = SEFModel.from_pretrained(model_dir, allow_pickle=True)

    assert loaded.embed("hi").shape == (4,)
    assert any("execute arbitrary code" in str(w.message) for w in caught)


def test_checksum_mismatch_raises(tmp_path: Path):
    model_dir = tmp_path / "corrupt"
    _write_minimal_sef(model_dir)

    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["name"] = "tampered"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        SEFModel.from_pretrained(model_dir, embed_loader=lambda _p: _dummy_embedder)


def test_explicit_loader_classes_are_supported(tmp_path: Path):
    model_dir = tmp_path / "custom_loader"
    _write_minimal_sef(model_dir)

    safe_loaded = SEFModel.from_pretrained(
        model_dir,
        embed_loader=lambda _p: _dummy_embedder,
        loader=SafeSEFLoader(),
    )
    assert safe_loaded.embed("x").shape == (4,)

    trusted_loaded = SEFModel.from_pretrained(
        model_dir, loader=TrustedPickleSEFLoader()
    )
    assert trusted_loaded.embed("xyz").shape == (4,)
