"""Tests for embedder adapters."""

import importlib.util
import pickle
import sys
import numpy as np
import pytest

from kalmanorix import SEF
from kalmanorix.embedder_adapters import (
    STEmbedder,
    OpenAIEmbedder,
    CohereEmbedder,
    AnthropicEmbedder,
    VertexAIEmbedder,
    AzureOpenAIEmbedder,
    HuggingFaceEmbedder,
)


def test_st_embedder_smoke():
    """Smoke test for STEmbedder (requires sentence-transformers)."""
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")

    # Use a tiny model if available, otherwise skip
    # We'll mock with a dummy model? For now skip.
    pytest.skip("STEmbedder test requires actual model - skipping for now")


def test_huggingface_embedder_import():
    """Test that HuggingFaceEmbedder can be imported."""
    # Already imported above
    assert HuggingFaceEmbedder is not None


@pytest.mark.skipif(
    "transformers" not in sys.modules,
    reason="transformers library not installed",
)
@pytest.mark.skipif(
    "torch" not in sys.modules,
    reason="torch library not installed",
)
class TestHuggingFaceEmbedder:
    """Tests for HuggingFaceEmbedder with a tiny BERT model."""

    @pytest.fixture(scope="class")
    def tiny_model_name(self):
        # A tiny BERT model for testing (2 layers, 128 hidden)
        return "prajjwal1/bert-tiny"

    @pytest.fixture(scope="class")
    def embedder(self, tiny_model_name):
        # Create embedder with CPU device
        return HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            device="cpu",
            pooling="mean",
            normalize=True,
        )

    def test_embedder_call(self, embedder):
        """Test that embedder returns a vector."""
        vec = embedder("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float64
        # BERT-tiny hidden size is 128
        assert vec.shape == (128,)

    def test_embedder_normalization(self, embedder):
        """Test that normalized vectors have unit length."""
        vec = embedder("Test normalization")
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, rtol=1e-6)

    def test_embedder_pooling_cls(self, tiny_model_name):
        """Test CLS pooling."""
        embedder = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            pooling="cls",
            normalize=False,
        )
        vec = embedder("CLS pooling test")
        assert vec.shape == (128,)
        # With CLS pooling, output should be different from mean pooling
        embedder_mean = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            pooling="mean",
            normalize=False,
        )
        vec_mean = embedder_mean("CLS pooling test")
        # Not equal (but could be accidentally similar)
        # Just ensure they're both valid vectors
        assert not np.allclose(vec, vec_mean, atol=1e-6)

    def test_embedder_max_length(self, tiny_model_name):
        """Test that max_length parameter works."""
        embedder = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            max_length=10,
            normalize=False,
        )
        # Long text should be truncated
        long_text = " ".join(["word"] * 20)
        vec = embedder(long_text)
        assert vec.shape == (128,)
        # Should not crash

    def test_embedder_with_sef(self, embedder):
        """Test HuggingFaceEmbedder integrated with SEF."""
        sef = SEF(name="bert", embed=embedder, sigma2=1.0)
        # Should be able to get embedding via SEF
        vec = sef.embed("test query")
        assert vec.shape == (128,)
        assert vec.dtype == np.float64

    def test_embedder_device_cpu(self, tiny_model_name):
        """Test explicit CPU device."""
        embedder = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            device="cpu",
        )
        vec = embedder("test")
        assert vec.shape == (128,)

    @pytest.mark.skipif(
        not ("torch" in sys.modules and hasattr(sys.modules["torch"], "cuda")),
        reason="CUDA not available",
    )
    def test_embedder_device_cuda(self, tiny_model_name):
        """Test CUDA device (skip if CUDA not available)."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        embedder = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            device="cuda",
        )
        vec = embedder("test")
        assert vec.shape == (128,)

    def test_embedder_custom_tokenizer(self, tiny_model_name):
        """Test with custom tokenizer path."""
        # Use same model for tokenizer
        embedder = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            tokenizer_name_or_path=tiny_model_name,
        )
        vec = embedder("custom tokenizer test")
        assert vec.shape == (128,)

    def test_embedder_no_normalize(self, tiny_model_name):
        """Test with normalization disabled."""
        embedder = HuggingFaceEmbedder(
            model_name_or_path=tiny_model_name,
            normalize=False,
        )
        vec = embedder("no normalize")
        norm = np.linalg.norm(vec)
        # Not necessarily unit length
        assert norm > 0

    def test_embedder_pickling(self, embedder):
        """Test that HuggingFaceEmbedder can be pickled and unpickled."""
        # Pickle the embedder
        pickled = pickle.dumps(embedder)
        # Unpickle
        unpickled = pickle.loads(pickled)
        # Verify configuration is preserved
        assert unpickled.model_name_or_path == embedder.model_name_or_path
        assert unpickled.pooling == embedder.pooling
        assert unpickled.device == embedder.device
        assert unpickled.max_length == embedder.max_length
        assert unpickled.normalize == embedder.normalize
        # Lazy attributes should be None
        assert unpickled._model is None  # pylint: disable=protected-access
        assert unpickled._tokenizer is None  # pylint: disable=protected-access
        # Should still be able to compute embeddings
        vec = unpickled("test pickling")
        assert vec.shape == (128,)
        assert vec.dtype == np.float64
        # Compare with original embedder (should be close)
        vec_original = embedder("test pickling")
        np.testing.assert_allclose(vec, vec_original, rtol=1e-6)


def test_other_adapters_import():
    """Smoke test that other adapters can be imported."""
    # They're already imported, just verify they exist
    assert STEmbedder is not None
    assert OpenAIEmbedder is not None
    assert CohereEmbedder is not None
    assert AnthropicEmbedder is not None
    assert VertexAIEmbedder is not None
    assert AzureOpenAIEmbedder is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
