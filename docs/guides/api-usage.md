# API Usage Examples

This guide provides concrete examples of using Kalmanorix through both the **Python library API** and the **REST API**. Choose the approach that fits your application:

- **Python library**: Direct import for maximum control and lowest latency
- **REST API**: Language‑agnostic access via HTTP, suitable for microservices and web applications

All examples assume you have installed Kalmanorix with the required dependencies:

```bash
# For Python library usage
pip install -e "."

# For REST API client usage (Python requests)
pip install requests

# For running the REST server
pip install -e ".[api]"
```

## Python Library API

### 1. Creating Specialists

```python
from kalmanorix import create_huggingface_sef, create_openai_sef, create_tfidf_sef
import os

# Hugging Face specialist (local model)
medical_sef = create_huggingface_sef(
    name="medical",
    model_name_or_path="prajjwal1/bert-tiny",
    sigma2=0.1,
    pooling="mean",
    normalize=True,
)

# OpenAI specialist (API-based)
os.environ["OPENAI_API_KEY"] = "sk-..."
openai_sef = create_openai_sef(
    name="openai_general",
    model="text-embedding-3-small",
    sigma2=0.05,
    dimensions=256,  # optional dimensionality reduction
)

# TF-IDF specialist (fast, no external dependencies)
calibration_texts = ["patient diagnosis", "clinical trial", "medical treatment"]
tfidf_sef = create_tfidf_sef(
    name="medical_tfidf",
    calibration_texts=calibration_texts,
    sigma2=0.3,
    max_features=500,
)
```

### 2. Building a Village and Fusing Embeddings

```python
from kalmanorix import Village, Panoramix, ScoutRouter, KalmanorixFuser

# Create village with multiple specialists
village = Village([medical_sef, openai_sef, tfidf_sef])

# Configure router (select which specialists to consult)
router = ScoutRouter(mode="semantic", similarity_threshold=0.6)

# Choose fusion strategy
fuser = KalmanorixFuser(prior_variance=1.0)

# Create the fusion orchestrator
panoramix = Panoramix(village=village, router=router, fuser=fuser)

# Fuse embeddings for a query
query = "Patient with pneumonia shows improvement after antibiotics"
potion = panoramix.brew(query)

print(f"Fused embedding shape: {potion.embedding.shape}")
print(f"Weights: {potion.weights}")
print(f"Selected specialists: {[s.name for s in potion.meta['selected_modules']]}")
```

### 3. Batch Processing

```python
# Process multiple queries efficiently
queries = [
    "Medical diagnosis report",
    "Legal contract dispute",
    "Kubernetes cluster autoscaling",
]

potions = panoramix.brew_batch(queries)

for i, potion in enumerate(potions):
    print(f"Query {i}: {queries[i][:50]}...")
    print(f"  Selected {len(potion.meta['selected_modules'])} specialists")
    print(f"  Weight distribution: {potion.weights}")
```

### 4. Different Fusion Strategies

```python
from kalmanorix import MeanFuser, DiagonalKalmanFuser, EnsembleKalmanFuser

# Mean fusion (uniform averaging)
mean_fuser = MeanFuser()

# Diagonal Kalman fusion (scalar variance per dimension)
diagonal_fuser = DiagonalKalmanFuser(prior_variance=1.0)

# Ensemble Kalman fusion (parallel updates)
ensemble_fuser = EnsembleKalmanFuser(prior_variance=1.0)

# Use with Panoramix
panoramix_mean = Panoramix(village=village, router=router, fuser=mean_fuser)
panoramix_kalman = Panoramix(village=village, router=router, fuser=diagonal_fuser)
```

## REST API Usage

Kalmanorix includes a production‑ready FastAPI server (`examples/fastapi_server.py`). Start it with:

```bash
uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
```

The server provides endpoints for fusion with rate limiting, CORS support, and caching.

### 1. Python Client (requests)

```python
import requests
import json

# Server information
response = requests.get("http://localhost:8000/")
print(json.dumps(response.json(), indent=2))

# List available specialists
response = requests.get("http://localhost:8000/modules")
specialists = response.json()
print(f"Available specialists: {[s['name'] for s in specialists]}")

# Fuse embeddings for a query
payload = {
    "query": "Patient confidentiality in cloud storage",
    "strategy": "kalmanorix",
    "routing": "semantic",
}

response = requests.post("http://localhost:8000/fuse", json=payload)
result = response.json()

print(f"Selected modules: {result['selected_modules']}")
print(f"Weights: {result['weights']}")
print(f"Fused vector length: {len(result['fused_vector'])}")
```

### 2. JavaScript Client (fetch)

```javascript
// Browser or Node.js (with node-fetch)
async function fuseQuery(query) {
    const response = await fetch('http://localhost:8000/fuse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            strategy: 'kalmanorix',
            routing: 'semantic'
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    console.log('Selected modules:', data.selected_modules);
    console.log('Weights:', data.weights);
    return data.fused_vector;
}

// Usage
fuseQuery('GDPR compliance for healthcare apps')
    .then(vector => console.log('Got embedding of length', vector.length))
    .catch(err => console.error('Error:', err));
```

### 3. cURL Commands

```bash
# Get server information
curl http://localhost:8000/

# List specialists
curl http://localhost:8000/modules

# Fuse embeddings (single query)
curl -X POST http://localhost:8000/fuse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Kubernetes HIPAA compliance",
    "strategy": "kalmanorix",
    "routing": "semantic"
  }'

# With pretty-printed JSON output (jq required)
curl -s -X POST http://localhost:8000/fuse \
  -H "Content-Type: application/json" \
  -d '{"query": "Medical legal consultation"}' | jq .
```

### 4. Batch Fusion via REST

While the example server doesn't include a batch endpoint, you can implement one by extending the server:

```python
# In your custom server implementation
@app.post("/fuse_batch")
async def fuse_batch(request: BatchFusionRequest):
    queries = request.queries
    potions = panoramix.brew_batch(queries)
    return [{"query": q, "embedding": p.embedding.tolist()} for q, p in zip(queries, potions)]
```

Then call it from clients:

```python
# Python batch request
payload = {
    "queries": ["query1", "query2", "query3"],
    "strategy": "kalmanorix"
}
response = requests.post("http://localhost:8000/fuse_batch", json=payload)
```

## Common Use Case Examples

### Multi‑Domain Search System

```python
# Python library approach
from kalmanorix import create_huggingface_sef_with_calibration, Village, Panoramix

# Create domain specialists with calibrated uncertainty
medical_sef = create_huggingface_sef_with_calibration(
    name="medical",
    model_name_or_path="prajjwal1/bert-tiny",
    calibration_texts=medical_calibration_texts,
    base_sigma2=0.1,
)

legal_sef = create_huggingface_sef_with_calibration(
    name="legal",
    model_name_or_path="prajjwal1/bert-tiny",
    calibration_texts=legal_calibration_texts,
    base_sigma2=0.1,
)

# Build and query
village = Village([medical_sef, legal_sef])
panoramix = Panoramix(village=village)
result = panoramix.brew("HIPAA compliance requirements")
```

```bash
# REST API approach
curl -X POST http://localhost:8000/fuse \
  -H "Content-Type: application/json" \
  -d '{"query": "HIPAA compliance requirements", "strategy": "kalmanorix"}'
```

### Real‑Time Semantic Search

For latency‑sensitive applications, use fast embedders and caching:

```python
# Python library with caching
from cachetools import TTLCache
from kalmanorix import create_tfidf_sef, Village

# Fast TF‑IDF specialists
tech_sef = create_tfidf_sef(
    name="tech",
    calibration_texts=tech_texts,
    sigma2=0.2,
    max_features=1000,
)

# Cache fusion results (5-minute TTL)
cache = TTLCache(maxsize=1000, ttl=300)

def cached_fuse(query):
    if query in cache:
        return cache[query]
    result = panoramix.brew(query)
    cache[query] = result
    return result
```

```javascript
// JavaScript client with local caching
const cache = new Map();

async function cachedFuse(query) {
    if (cache.has(query)) {
        return cache.get(query);
    }
    const result = await fuseQuery(query);
    cache.set(query, result);
    // Optional: limit cache size
    if (cache.size > 1000) {
        const firstKey = cache.keys().next().value;
        cache.delete(firstKey);
    }
    return result;
}
```

### Cross‑Lingual Fusion

```python
# Python library with language detection
from kalmanorix import create_huggingface_sef_with_calibration
from langdetect import detect

# Language‑specific specialists
english_sef = create_huggingface_sef_with_calibration(
    name="english",
    model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    calibration_texts=english_texts,
)

french_sef = create_huggingface_sef_with_calibration(
    name="french",
    model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    calibration_texts=french_texts,
)

def fuse_with_language_routing(query):
    lang = detect(query)
    if lang == "fr":
        selected = [french_sef]
    else:
        selected = [english_sef]
    # ... fuse with selected specialists
```

## Error Handling

### Python Library Errors

```python
import numpy as np
from kalmanorix import Village

try:
    # Empty village error
    village = Village([])
    result = panoramix.brew("test")
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    # Invalid query
    result = panoramix.brew("")
except ValueError as e:
    print(f"Input error: {e}")

try:
    # Specialist embedding failure
    result = panoramix.brew("test" * 1000)  # Very long query
except Exception as e:
    print(f"Embedding error: {e}")
```

### REST API Error Responses

The server returns appropriate HTTP status codes:

```bash
# 400 Bad Request (missing query)
curl -X POST http://localhost:8000/fuse \
  -H "Content-Type: application/json" \
  -d '{"strategy": "kalmanorix"}'

# Response:
# {"detail":"Field required: query"}

# 429 Too Many Requests (rate limit exceeded)
# Response:
# {"detail":"Too Many Requests"}

# 500 Internal Server Error (specialist failure)
# Response:
# {"detail":"Embedding failed: ..."}
```

Handle errors in clients:

```python
# Python error handling
try:
    response = requests.post("http://localhost:8000/fuse", json=payload, timeout=10)
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx
    result = response.json()
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code} - {e.response.text}")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

```javascript
// JavaScript error handling
async function safeFuse(query) {
    try {
        const response = await fetch('http://localhost:8000/fuse', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query: query}),
            timeout: 10000  // 10 seconds
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Fusion request failed:', error);
        // Fallback to local embedding or default
        return {fused_vector: Array(384).fill(0), weights: {}};
    }
}
```

## Performance Tips

### Python Library
- **Batch processing**: Use `brew_batch()` for multiple queries (30‑50% faster)
- **Cache embeddings**: Use `functools.lru_cache` on embedder functions
- **Async embedding**: For API‑based specialists, use async HTTP clients
- **GPU acceleration**: Set `device="cuda"` for Hugging Face specialists

### REST API
- **Keep‑alive connections**: Reuse HTTP connections for multiple requests
- **Client‑side caching**: Cache results locally when queries repeat
- **Batch requests**: Implement batch endpoint for multiple queries
- **Load balancing**: Distribute requests across multiple server instances

## Further Reading

- [Creating Specialists](creating-specialists.md) – Detailed guide to building specialists
- [Fusion Strategies](fusion-strategies.md) – Choosing and tuning fusion algorithms
- [Uncertainty Calibration](uncertainty-calibration.md) – Calibrating variance estimates
- [Deployment Guide](deployment.md) – Production deployment best practices
- [API Server Example](../examples/api-server.md) – Complete FastAPI server reference
- [Multi‑Domain Search Tutorial](../examples/multi-domain-search-tutorial.md) – End‑to‑end use case
- [Real‑Time Semantic Search](../examples/real-time-semantic-search.md) – Performance‑optimized system
- [Cross‑Lingual Fusion](../examples/cross-lingual-fusion.md) – Multilingual specialist fusion
