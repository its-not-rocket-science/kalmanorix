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
    OnnxEmbedder,
    create_openai_sef,
    create_openai_sef_with_calibration,
    create_cohere_sef,
    create_cohere_sef_with_calibration,
    create_vertexai_sef,
    create_vertexai_sef_with_calibration,
    create_azure_openai_sef,
    create_azure_openai_sef_with_calibration,
    create_onnx_sef,
    create_onnx_sef_model,
)


def _dummy_onnx_preprocessor(text: str):
    """Simple preprocessor that creates deterministic input array for ONNX tests."""
    # Use hash of text to generate deterministic but varied inputs
    h = hash(text) & 0xFFFFFFFF  # Get positive 32-bit hash
    # Create three floats based on hash
    return {
        "input": np.array(
            [[float((h >> i) & 0xFF) for i in range(3)]], dtype=np.float32
        )
    }


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


def test_factory_functions_with_mock(mocker):
    """Test factory functions with mocked clients."""
    # Mock the SDK imports to avoid requiring actual installations
    # Mock openai import
    mock_openai = mocker.MagicMock()
    mock_openai.OpenAIError = Exception
    mocker.patch.dict("sys.modules", {"openai": mock_openai})

    # Mock cohere import
    mock_cohere = mocker.MagicMock()
    mock_cohere.CohereError = Exception
    mocker.patch.dict("sys.modules", {"cohere": mock_cohere})

    # Mock google.cloud.aiplatform import
    mock_aiplatform = mocker.MagicMock()
    mock_aiplatform.VertexAIEmbeddingModel = mocker.MagicMock
    mock_google_api_error = Exception
    mocker.patch.dict(
        "sys.modules",
        {
            "google.cloud.aiplatform": mock_aiplatform,
            "google.api_core.exceptions": mocker.MagicMock(
                GoogleAPIError=mock_google_api_error
            ),
        },
    )

    # Mock clients that return a dummy embedding
    class MockOpenAIClient:
        class embeddings:
            @staticmethod
            def create(*, model, input, dimensions=None):
                class Response:
                    class Data:
                        embedding = [0.1, 0.2, 0.3]

                    data = [Data()]

                return Response()

    class MockCohereClient:
        def embed(self, *, texts, model, input_type):
            class Response:
                embeddings = [[0.1, 0.2, 0.3]]

            return Response()

    # Mock VertexAIEmbeddingModel
    class MockVertexAIEmbeddingModel:
        def get_embeddings(self, texts, task_type):
            class Embedding:
                values = [0.1, 0.2, 0.3]

            return [Embedding()]

    # Test OpenAI factory functions
    mock_openai_client = MockOpenAIClient()
    sef1 = create_openai_sef(
        client=mock_openai_client,
        name="test-openai",
        sigma2=1.0,
        model="text-embedding-3-small",
    )
    assert sef1.name == "test-openai"
    assert sef1.sigma2 == 1.0
    # Embed should work (returns normalized vector)
    vec = sef1.embed("test")
    assert vec.shape == (3,)
    assert vec.dtype == np.float64

    # Test with calibration
    calibration_texts = ["calib1", "calib2"]
    sef2 = create_openai_sef_with_calibration(
        client=mock_openai_client,
        name="test-openai-calib",
        calibration_texts=calibration_texts,
        base_sigma2=0.2,
        scale=2.0,
    )
    assert sef2.name == "test-openai-calib"
    assert callable(sef2.sigma2)  # CentroidDistanceSigma2 instance

    # Test Cohere factory functions
    mock_cohere_client = MockCohereClient()
    sef3 = create_cohere_sef(
        client=mock_cohere_client,
        name="test-cohere",
        sigma2=1.5,
    )
    assert sef3.name == "test-cohere"
    assert sef3.sigma2 == 1.5

    sef4 = create_cohere_sef_with_calibration(
        client=mock_cohere_client,
        name="test-cohere-calib",
        calibration_texts=calibration_texts,
    )
    assert sef4.name == "test-cohere-calib"
    assert callable(sef4.sigma2)

    # Test Vertex AI factory functions
    mock_vertexai_model = MockVertexAIEmbeddingModel()
    sef5 = create_vertexai_sef(
        model=mock_vertexai_model,
        name="test-vertexai",
        sigma2=0.8,
    )
    assert sef5.name == "test-vertexai"
    assert sef5.sigma2 == 0.8

    sef6 = create_vertexai_sef_with_calibration(
        model=mock_vertexai_model,
        name="test-vertexai-calib",
        calibration_texts=calibration_texts,
    )
    assert sef6.name == "test-vertexai-calib"
    assert callable(sef6.sigma2)

    # Test Azure OpenAI factory functions (same as OpenAI but different default model)
    sef7 = create_azure_openai_sef(
        client=mock_openai_client,
        name="test-azure-openai",
        sigma2=1.2,
    )
    assert sef7.name == "test-azure-openai"
    assert sef7.sigma2 == 1.2

    sef8 = create_azure_openai_sef_with_calibration(
        client=mock_openai_client,
        name="test-azure-openai-calib",
        calibration_texts=calibration_texts,
    )
    assert sef8.name == "test-azure-openai-calib"
    assert callable(sef8.sigma2)


def _has_onnx_deps():
    return (
        importlib.util.find_spec("onnxruntime") is not None
        and importlib.util.find_spec("onnx") is not None
    )


@pytest.mark.skipif(
    not _has_onnx_deps(),
    reason="onnxruntime or onnx library not installed",
)
class TestOnnxEmbedder:
    """Tests for OnnxEmbedder with a dummy identity model."""

    @pytest.fixture(scope="class")
    def dummy_onnx_model_path(self, tmp_path_factory):
        """Create a dummy ONNX model that performs identity transformation (input -> output)."""
        try:
            import onnx
            from onnx import helper
            import onnx.checker
        except ImportError:
            pytest.skip("ONNX library required for creating test model")

        # Create a simple graph: input -> Identity -> output
        # Input shape: (1, 3) batch size 1, dimension 3 (small for testing)
        input_name = "input"
        output_name = "output"

        # Create tensor type
        tensor_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [1, 3])

        # Create input/value info
        input_value_info = helper.make_value_info(input_name, tensor_type)
        output_value_info = helper.make_value_info(output_name, tensor_type)

        # Create Identity node
        identity_node = helper.make_node(
            "Identity", inputs=[input_name], outputs=[output_name], name="identity_node"
        )

        # Create graph
        graph = helper.make_graph(
            nodes=[identity_node],
            name="identity_graph",
            inputs=[input_value_info],
            outputs=[output_value_info],
        )

        # Create model
        model = helper.make_model(
            graph,
            producer_name="kalmanorix-test",
            opset_imports=[helper.make_opsetid("", 18)],
        )
        onnx.checker.check_model(model)

        # Save to temporary file
        tmpdir = tmp_path_factory.mktemp("onnx_models")
        model_path = tmpdir / "identity.onnx"
        with open(model_path, "wb") as f:
            f.write(model.SerializeToString())
        return str(model_path)

    @pytest.fixture(scope="class")
    def preprocessor(self):
        """Simple preprocessor that creates random input array."""
        return _dummy_onnx_preprocessor

    @pytest.fixture(scope="class")
    def embedder(self, dummy_onnx_model_path, preprocessor):
        """Create OnnxEmbedder instance."""
        return OnnxEmbedder(
            model_path=dummy_onnx_model_path,
            preprocessor=preprocessor,
            normalize=True,
        )

    def test_embedder_call(self, embedder):
        """Test that embedder returns a vector."""
        vec = embedder("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float64
        # Output dimension is 3 (flattened from (1, 3))
        assert vec.shape == (3,)

    def test_embedder_normalization(self, embedder):
        """Test that normalized vectors have unit length."""
        vec = embedder("Test normalization")
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, rtol=1e-6)

    def test_embedder_no_normalize(self, dummy_onnx_model_path, preprocessor):
        """Test with normalization disabled."""
        embedder = OnnxEmbedder(
            model_path=dummy_onnx_model_path,
            preprocessor=preprocessor,
            normalize=False,
        )
        vec = embedder("no normalize")
        norm = np.linalg.norm(vec)
        # Not necessarily unit length (depends on random input)
        assert norm > 0

    def test_embedder_input_names(self, embedder):
        """Test input_names property."""
        input_names = embedder.input_names
        assert isinstance(input_names, list)
        assert len(input_names) == 1
        assert input_names[0] == "input"

    def test_embedder_output_names(self, embedder):
        """Test output_names property."""
        output_names = embedder.output_names
        assert isinstance(output_names, list)
        assert len(output_names) == 1
        assert output_names[0] == "output"

    def test_embedder_pickling(self, embedder):
        """Test that OnnxEmbedder can be pickled and unpickled."""
        # Pickle the embedder
        pickled = pickle.dumps(embedder)
        # Unpickle
        unpickled = pickle.loads(pickled)
        # Verify configuration is preserved
        assert unpickled.model_path == embedder.model_path
        assert unpickled.normalize == embedder.normalize
        assert unpickled.providers == embedder.providers
        # Lazy session should be None
        assert unpickled._session is None  # pylint: disable=protected-access
        # Should still be able to compute embeddings
        vec = unpickled("test pickling")
        assert vec.shape == (3,)
        assert vec.dtype == np.float64
        # Compare with original embedder (should be close)
        vec_original = embedder("test pickling")
        np.testing.assert_allclose(vec, vec_original, rtol=1e-6)

    def test_embedder_with_sef(self, embedder):
        """Test OnnxEmbedder integrated with SEF."""
        sef = SEF(name="onnx", embed=embedder, sigma2=1.0)
        vec = sef.embed("test query")
        assert vec.shape == (3,)
        assert vec.dtype == np.float64

    def test_factory_functions(self, dummy_onnx_model_path, preprocessor):
        """Test factory functions create_onnx_sef and create_onnx_sef_model."""
        # Test create_onnx_sef
        sef = create_onnx_sef(
            model_path=dummy_onnx_model_path,
            preprocessor=preprocessor,
            name="onnx-test",
            sigma2=0.5,
        )
        assert sef.name == "onnx-test"
        assert sef.sigma2 == 0.5
        vec = sef.embed("factory test")
        assert vec.shape == (3,)

        # Test create_onnx_sef_model
        model = create_onnx_sef_model(
            model_path=dummy_onnx_model_path,
            preprocessor=preprocessor,
            name="onnx-model-test",
            sigma2=0.7,
        )
        assert model.metadata.model_id.startswith("onnx:")
        assert model.metadata.name == "onnx-model-test"
        assert model.metadata.embedding_dimension == 3
        vec = model.embed("model test")
        assert vec.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
