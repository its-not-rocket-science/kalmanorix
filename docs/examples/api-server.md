# API Server Example

Kalmanorix includes a production‑ready FastAPI server that exposes fusion capabilities via a REST API. This allows non‑Python clients to leverage specialist fusion and enables easy integration into microservice architectures.

The example server (`examples/fastapi_server.py`) demonstrates a fully functional API with rate limiting, CORS support, response caching, and health checks. It uses the same toy specialists as the minimal fusion demo, making it easy to run without external model dependencies.

## Features

- **REST endpoints** for single‑query fusion.
- **CORS support** for web‑based clients.
- **Rate limiting** (200 requests/minute for root and modules, 100/minute for fusion).
- **Embedding cache** (TTL‑based) for repeated queries.
- **Health checks** via root endpoint.
- **Multiple fusion strategies**: `mean`, `kalmanorix`, `ensemble_kalman`, `structured_kalman`, `diagonal_kalman`, `learned_gate`.
- **Two routing modes**: `all` (fusion) and `hard` (single specialist).

## Requirements

Install the optional `api` dependency group:

```bash
pip install -e ".[api]"
```

This installs `fastapi`, `uvicorn`, `slowapi`, `cachetools`, and `pydantic`.

## Starting the Server

Run the example server with:

```bash
uvicorn examples.fastapi_server:app --reload --host 0.0.0.0 --port 8000
```

Or run the module directly:

```bash
python -m examples.fastapi_server
```

The server will start on `http://localhost:8000`. The `--reload` flag enables automatic reloading during development.

## API Endpoints

### `GET /`
Server information and available endpoints.

**Example request:**
```bash
curl http://localhost:8000/
```

**Example response:**
```json
{
  "name": "Kalmanorix Fusion Server",
  "version": "0.1.0",
  "description": "REST API for fusing embeddings from multiple specialist models",
  "endpoints": {
    "GET /": "This info",
    "GET /modules": "List available specialist modules",
    "POST /fuse": "Fuse embeddings for a query"
  }
}
```

### `GET /modules`
List all loaded specialists (SEFs) with their names, domains, and uncertainty settings.

**Example request:**
```bash
curl http://localhost:8000/modules
```

**Example response:**
```json
[
  {
    "name": "tech",
    "domain": "tech",
    "sigma2_type": "KeywordSigma2",
    "sigma2_in_domain": 0.2,
    "sigma2_out_domain": 2.5
  },
  {
    "name": "cook",
    "domain": "cooking",
    "sigma2_type": "KeywordSigma2",
    "sigma2_in_domain": 0.2,
    "sigma2_out_domain": 2.5
  }
]
```

### `POST /fuse`
Fuse embeddings for a single query.

**Request body:**
```json
{
  "query": "This smartphone battery lasts longer than a slow cooker braise",
  "strategy": "kalmanorix",
  "routing": "all"
}
```

**Response:**
```json
{
  "query": "This smartphone battery lasts longer than a slow cooker braise",
  "strategy": "kalmanorix",
  "routing": "all",
  "selected_modules": ["tech", "cook"],
  "fused_vector": [0.123, -0.456, ...],
  "weights": {"tech": 0.876, "cook": 0.124},
  "meta": {
    "selected_modules": ["tech", "cook"],
    "sigma2": {"tech": 0.2, "cook": 2.5}
  }
}
```

## Client Examples

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/fuse",
    json={
        "query": "Patient diagnosis report",
        "strategy": "kalmanorix",
        "routing": "all"
    }
)
result = response.json()
print("Fusion weights:", result["weights"])
print("Embedding length:", len(result["fused_vector"]))
```

### cURL

```bash
curl -X POST http://localhost:8000/fuse \
  -H "Content-Type: application/json" \
  -d '{"query": "Court ruling on contract", "strategy": "mean", "routing": "hard"}'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/fuse', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: 'Medical legal consultation',
    strategy: 'kalmanorix',
    routing: 'all'
  })
});
const data = await response.json();
console.log('Weights:', data.weights);
```

## Server Architecture

The example server is structured as follows:

1. **Toy specialists**: The server creates a `Village` with two keyword‑sensitive specialists (tech and cooking) using `KeywordSigma2` uncertainty.
2. **Global state**: The village, scout routers, and fusers are created at startup and reused across requests.
3. **Caching**: A `TTLCache` stores fusion results for 5 minutes, keyed by `(query, strategy, routing)`.
4. **Rate limiting**: Implemented via `slowapi` with separate limits for different endpoints.
5. **Error handling**: Custom exception handlers log errors and return appropriate HTTP status codes.

## Extending for Production Use

The example server is designed for demonstration. For production deployment, consider the following modifications:

### Loading Real Specialists

Replace `create_toy_village()` with a function that loads pre‑trained SEFs from disk (pickle files or `SEFModel` directories). For example:

```python
import pickle

def load_production_village() -> Village:
    sefs = []
    for path in ["models/medical.pkl", "models/legal.pkl"]:
        with open(path, "rb") as f:
            sef = pickle.load(f)
            sefs.append(sef)
    return Village(sefs)
```

### Configuration via Environment Variables

Add support for environment variables to control village path, fusion strategy defaults, and cache settings:

```python
import os

VILLAGE_PATH = os.getenv("KALMANORIX_VILLAGE_PATH", "./models")
DEFAULT_STRATEGY = os.getenv("KALMANORIX_DEFAULT_STRATEGY", "kalmanorix")
CACHE_TTL = int(os.getenv("KALMANORIX_CACHE_TTL_SECONDS", "300"))
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e ".[api]"

ENV KALMANORIX_VILLAGE_PATH=/app/models
ENV PORT=8000

EXPOSE ${PORT}
CMD ["uvicorn", "examples.fastapi_server:app", "--host", "0.0.0.0", "--port", "${PORT}"]
```

Build and run:

```bash
docker build -t kalmanorix-server .
docker run -p 8000:8000 kalmanorix-server
```

### Kubernetes Deployment

For Kubernetes, create a `Deployment` and `Service`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kalmanorix-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kalmanorix
  template:
    metadata:
      labels:
        app: kalmanorix
    spec:
      containers:
      - name: server
        image: kalmanorix-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: KALMANORIX_VILLAGE_PATH
          value: "/app/models"
---
apiVersion: v1
kind: Service
metadata:
  name: kalmanorix-service
spec:
  selector:
    app: kalmanorix
  ports:
  - port: 80
    targetPort: 8000
```

## Monitoring and Observability

- **Logging**: The server uses Python's `logging` module with timestamps and log levels.
- **Metrics**: Consider adding Prometheus metrics for request latency, cache hit rate, and error counts.
- **Health checks**: The root endpoint (`GET /`) can be used as a health check.
- **Tracing**: Integrate OpenTelemetry for distributed tracing in microservice environments.

## Testing the Server

A smoke test script (`examples/test_server_import.py`) verifies that all imports work and basic fusion succeeds:

```bash
python examples/test_server_import.py
```

This is useful for CI/CD pipelines to ensure the server can start correctly.

## Limitations and Future Enhancements

The current example server has the following limitations:

1. **Batch fusion not implemented**: The `/fuse_batch` endpoint is not included but can be added using `Panoramix.brew_batch`.
2. **Static village configuration**: The village is fixed at startup; dynamic addition/removal of specialists requires server restart.
3. **No authentication/authorization**: The server is open to all clients; add API keys or OAuth for production.
4. **No persistent cache**: The cache is in‑memory and lost on restart; consider Redis or similar for distributed caching.

These can be addressed by extending the server with additional endpoints and configuration options.

## Further Reading

- [Minimal Fusion Example](minimal-fusion.md) – Details on the toy specialists used in the server.
- [HuggingFace Integration](huggingface-integration.md) – How to wrap real transformer models as SEFs.
- [API Reference](../api-reference/panoramix.md) – Documentation of the `Panoramix` fusion orchestrator.
- [Milestone 3.3](../contributing/roadmap.md#milestone-33-integration-ecosystem) – Project roadmap section on production‑ready API features.
