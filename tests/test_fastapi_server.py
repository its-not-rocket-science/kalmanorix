"""
Tests for FastAPI server (Milestone 3.3 integration).

These tests verify that the REST API endpoints work correctly and return
the expected data structures. They use the TestClient from fastapi.testclient.
"""

import sys
import importlib.util

import numpy as np
import pytest

# Skip all tests if fastapi is not installed
if importlib.util.find_spec("fastapi") is None:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from fastapi.testclient import TestClient

# Import the FastAPI app from examples
sys.path.insert(0, "examples")
from fastapi_server import app  # pylint: disable=wrong-import-position

client = TestClient(app)


def test_root_endpoint():
    """GET / returns server info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "Kalmanorix Fusion Server"
    assert "version" in data
    assert "endpoints" in data


def test_health_and_readiness_endpoints():
    """Health and readiness endpoints should be available."""
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    ready_data = ready.json()
    assert ready_data["status"] in {"ready", "not_ready"}
    assert "modules_loaded" in ready_data


def test_modules_endpoint():
    """GET /modules lists available specialist modules."""
    response = client.get("/modules")
    assert response.status_code == 200
    modules = response.json()
    assert isinstance(modules, list)
    assert len(modules) == 2  # tech and cook
    names = {m["name"] for m in modules}
    assert names == {"tech", "cook"}
    for module in modules:
        assert "name" in module
        assert "sigma2_type" in module
        assert module["sigma2_type"] == "KeywordSigma2"


@pytest.mark.parametrize(
    "strategy",
    [
        "mean",
        "kalmanorix",
        "ensemble_kalman",
        "structured_kalman",
        "diagonal_kalman",
        "learned_gate",
    ],
)
def test_fuse_endpoint_all_routing(strategy: str):
    """POST /fuse works with all fusion strategies (routing='all')."""
    payload = {
        "query": "This smartphone battery lasts longer than a slow cooker braise.",
        "strategy": strategy,
        "routing": "all",
    }
    response = client.post("/fuse", json=payload)
    assert response.status_code == 200, (
        f"Failed for strategy {strategy}: {response.text}"
    )
    data = response.json()

    # Validate response structure
    assert data["query"] == payload["query"]
    assert data["strategy"] == strategy
    assert data["routing"] == "all"
    assert "selected_modules" in data
    assert isinstance(data["selected_modules"], list)
    assert set(data["selected_modules"]) == {"tech", "cook"}
    assert "fused_vector" in data
    assert isinstance(data["fused_vector"], list)
    assert len(data["fused_vector"]) == 16  # DIM from server
    assert "weights" in data
    assert isinstance(data["weights"], dict)
    assert set(data["weights"].keys()) == {"tech", "cook"}
    # Weights should sum to approximately 1
    weight_sum = sum(data["weights"].values())
    assert abs(weight_sum - 1.0) < 1e-6


def test_fuse_endpoint_hard_routing():
    """POST /fuse with routing='hard' selects only one module."""
    payload = {
        "query": "This smartphone battery lasts longer than a slow cooker braise.",
        "strategy": "mean",  # strategy irrelevant for hard routing
        "routing": "hard",
    }
    response = client.post("/fuse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routing"] == "hard"
    assert len(data["selected_modules"]) == 1
    assert data["selected_modules"][0] in {"tech", "cook"}
    # With hard routing, the selected module should have weight 1.0
    assert abs(data["weights"][data["selected_modules"][0]] - 1.0) < 1e-6


def test_fuse_endpoint_invalid_strategy():
    """POST /fuse with unknown strategy returns 422 validation error."""
    payload = {
        "query": "test",
        "strategy": "invalid_strategy",
        "routing": "all",
    }
    response = client.post("/fuse", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"


def test_fuse_endpoint_missing_field():
    """POST /fuse with missing required field returns 422 validation error."""
    payload = {
        "strategy": "mean",
        "routing": "all",
        # missing "query"
    }
    response = client.post("/fuse", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"


def test_fuse_metadata_structure():
    """Check that meta field is present and serializable."""
    payload = {
        "query": "test query",
        "strategy": "kalmanorix",
        "routing": "all",
    }
    response = client.post("/fuse", json=payload)
    assert response.status_code == 200
    data = response.json()
    # meta may be None or a dict
    if data["meta"] is not None:
        assert isinstance(data["meta"], dict)
        # Check for some expected keys in meta
        if "selected_modules" in data["meta"]:
            assert isinstance(data["meta"]["selected_modules"], list)


def test_numpy_serialization():
    """Ensure numpy arrays in metadata are properly converted to lists."""
    # This test uses the server's internal functions to verify numpy_to_list
    from fastapi_server import numpy_to_list  # pylint: disable=import-error

    # Test with numpy array
    arr = np.array([1.0, 2.0, 3.0])
    assert numpy_to_list(arr) == [1.0, 2.0, 3.0]

    # Test with dict containing numpy arrays
    d = {"a": arr, "b": 5}
    result = numpy_to_list(d)
    assert result == {"a": [1.0, 2.0, 3.0], "b": 5}

    # Test with nested list
    lst = [arr, arr]
    result = numpy_to_list(lst)
    assert result == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


def test_batch_embedding_endpoint():
    """POST /embed/batch returns per-item embeddings."""
    payload = {
        "texts": ["smartphone battery", "slow cooker recipe"],
        "modules": ["tech", "cook"],
    }
    response = client.post("/embed/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 2
    for item in data["items"]:
        assert "embeddings" in item
        assert len(item["embeddings"]) >= 1


def test_sentence_difficulty_scoring_components():
    """Difficulty score should include unknown ratio, grammar, and length."""
    from fastapi_server import (  # pylint: disable=import-error
        UserKnowledge,
        sentence_difficulty_score,
    )

    user = UserKnowledge(
        user_id="u1", known_vocabulary={"the", "battery", "is", "good"}
    )
    sentence = "Although the battery is good, the processor throttles unexpectedly."
    score, components, unknown = sentence_difficulty_score(sentence, user)

    assert 0.0 <= score <= 1.0
    assert set(components.keys()) == {
        "unknown_vocabulary_ratio",
        "grammar_complexity",
        "sentence_length",
    }
    assert components["grammar_complexity"] > 0.0
    assert components["unknown_vocabulary_ratio"] > 0.0
    assert "processor" in unknown


def test_recommend_text_endpoint_progression_and_shape():
    """GET /recommend-text returns scored recommendations and progresses sessions."""
    user_id = "curriculum-test-user"

    first = client.get("/recommend-text", params={"user_id": user_id, "limit": 4})
    assert first.status_code == 200
    first_data = first.json()
    assert first_data["user_id"] == user_id
    assert first_data["sessions_completed"] == 1
    assert 0.0 <= first_data["target_difficulty"] <= 1.0
    assert len(first_data["recommendations"]) == 4

    for rec in first_data["recommendations"]:
        assert "sentence" in rec
        assert "score" in rec
        assert "components" in rec
        assert "unknown_tokens" in rec
        assert 0.0 <= rec["score"] <= 1.0
        assert set(rec["components"].keys()) == {
            "unknown_vocabulary_ratio",
            "grammar_complexity",
            "sentence_length",
        }

    second = client.get("/recommend-text", params={"user_id": user_id, "limit": 4})
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["sessions_completed"] == 2
    assert second_data["target_difficulty"] > first_data["target_difficulty"]


def test_recommend_text_prioritizes_approximately_80_20_known_new():
    """Top recommendations should be near 20% unknown vocabulary when possible."""
    response = client.get(
        "/recommend-text", params={"user_id": "ratio-check-user", "limit": 3}
    )
    assert response.status_code == 200
    data = response.json()
    unknown_ratios = [
        rec["components"]["unknown_vocabulary_ratio"] for rec in data["recommendations"]
    ]
    # Heuristic ranking should keep recommendations reasonably close to 0.2.
    assert all(abs(ratio - 0.2) <= 0.5 for ratio in unknown_ratios)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
