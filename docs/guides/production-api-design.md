# Production-Grade API Design for Kalmanorix

This guide defines a **production-oriented FastAPI API surface** for Kalmanorix with:

- explicit request/response contracts,
- strong validation,
- rate limiting,
- caching,
- standardized error handling,
- observability,
- containerized deployment.

---

## 1) API Spec

### 1.1 Base conventions

- **Base path:** `/v1`
- **Content type:** `application/json`
- **Auth:** `Authorization: Bearer <token>` (pluggable dependency)
- **Idempotency:** for expensive `POST` calls, support `Idempotency-Key` header
- **Trace propagation:** accept `X-Request-ID` and W3C `traceparent`
- **Time budget:** per-request timeout enforced at service layer (e.g., 2-5s default)

### 1.2 Endpoint summary

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/embedding` | `POST` | Generate one or more embeddings from a configured specialist or provider adapter |
| `/v1/retrieval` | `POST` | Retrieve top-k records from configured index using query text/vector |
| `/v1/fusion` | `POST` | Fuse vectors from selected specialists using Kalmanorix strategy |
| `/v1/health/live` | `GET` | Liveness probe |
| `/v1/health/ready` | `GET` | Readiness probe (checks model/index/cache availability) |
| `/v1/metrics` | `GET` | Prometheus metrics endpoint |

### 1.3 Schemas (Pydantic v2 style)

```python
from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, conlist


class FusionStrategy(str, Enum):
    kalmanorix = "kalmanorix"
    mean = "mean"
    diagonal_kalman = "diagonal_kalman"
    structured_kalman = "structured_kalman"


class RoutingMode(str, Enum):
    all = "all"
    hard = "hard"
    semantic = "semantic"


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: conlist(str, min_length=1, max_length=128)
    specialist: str = Field(min_length=1, max_length=64)
    normalize: bool = True


class EmbeddingItem(BaseModel):
    index: int
    vector: list[float]


class EmbeddingResponse(BaseModel):
    vectors: list[EmbeddingItem]
    dimension: int
    elapsed_ms: float


class RetrievalFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str | None = None
    metadata: dict[str, Any] | None = None


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str | None = Field(default=None, min_length=1, max_length=4096)
    query_vector: list[float] | None = None
    top_k: int = Field(default=10, ge=1, le=200)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    filter: RetrievalFilter | None = None


class RetrievalHit(BaseModel):
    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    hits: list[RetrievalHit]
    total: int
    elapsed_ms: float


class FusionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4096)
    strategy: FusionStrategy = FusionStrategy.kalmanorix
    routing: RoutingMode = RoutingMode.semantic
    modules: list[str] | None = None
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class FusionResponse(BaseModel):
    fused_vector: list[float]
    selected_modules: list[str]
    weights: list[float]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float
```

### 1.4 Endpoint behavior

#### `POST /v1/embedding`

- Validates input size and text length.
- Rejects unknown specialists.
- Returns vector dimension and timing.
- Cache key: `sha256(specialist + normalize + texts)`.

#### `POST /v1/retrieval`

- Requires **exactly one** of `query` or `query_vector`.
- Uses vector backend abstraction (FAISS/Qdrant/pgvector).
- Returns rank-ordered hits.
- Cache key includes query payload + index generation/version.

#### `POST /v1/fusion`

- Routes to specialists, obtains candidate vectors, fuses with selected strategy.
- Includes diagnostics (sigma2 estimates, selected route mode, strategy).
- Optional fallback: if one specialist fails, continue with healthy subset when policy allows.

### 1.5 Standard error model

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "top_k must be <= 200",
    "details": {"field": "top_k"},
    "request_id": "2f6c...",
    "retryable": false
  }
}
```

Error codes to standardize:

- `VALIDATION_ERROR` (`400`)
- `UNAUTHORIZED` (`401`)
- `FORBIDDEN` (`403`)
- `NOT_FOUND` (`404`)
- `RATE_LIMITED` (`429`)
- `UPSTREAM_TIMEOUT` (`504`)
- `DEPENDENCY_UNAVAILABLE` (`503`)
- `INTERNAL_ERROR` (`500`)

---

## 2) Rate limiting strategy

Use `slowapi` at HTTP layer + optional distributed counters.

- Default policy: `60/minute` per API key, burst `10`.
- Anonymous fallback: `20/minute` per IP.
- Apply stricter limits for heavy endpoints (`/fusion`, `/embedding` batch).
- Return `Retry-After` header for `429`.
- For multi-instance deployments, prefer Redis-backed shared limiter.

Example:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

@app.post("/v1/fusion")
@limiter.limit("30/minute")
async def fuse(...):
    ...
```

---

## 3) Caching layer design

Use a two-tier approach:

1. **L1 local cache** (`cachetools.TTLCache`) for microsecond lookups.
2. **L2 Redis** for cross-pod sharing and invalidation.

Recommendations:

- TTLs:
  - Embedding: 5-30 minutes (provider/model dependent)
  - Retrieval: 30-120 seconds (data freshness dependent)
  - Fusion: 2-10 minutes (if deterministic)
- Canonicalize request JSON before hashing.
- Include model/index/version in cache keys to avoid stale responses.
- Expose cache metrics: hits, misses, evictions, stale serves.

Key format example:

```text
kalmanorix:{env}:{endpoint}:{version}:{sha256_payload}
```

---

## 4) Error handling strategy

Implement layered exception handling:

1. **Validation errors** (Pydantic/FastAPI) mapped to `VALIDATION_ERROR`.
2. **Domain errors** (unknown specialist, incompatible dimensions) mapped to `400/422`.
3. **Dependency errors** (vector DB / embedder API timeout) mapped to `503/504`.
4. **Catch-all** mapped to `500` with redacted messages.

Rules:

- Always return structured error payload.
- Always include request id in logs and error response.
- Never leak secrets or stack traces in response bodies.
- Mark retriable errors (`429`, `503`, `504`) with `retryable: true`.

---

## 5) Logging and observability hooks

### 5.1 Logging

- Use structured JSON logs (e.g., `structlog` or stdlib JSON formatter).
- Required fields: `timestamp`, `level`, `service`, `endpoint`, `request_id`, `latency_ms`, `status_code`.
- Redact user text by policy if handling sensitive data.

### 5.2 Metrics

Prometheus counters/histograms:

- `http_requests_total{endpoint,method,status}`
- `http_request_duration_seconds_bucket{endpoint}`
- `kalmanorix_fusion_duration_seconds`
- `kalmanorix_cache_hits_total{endpoint,tier}`
- `kalmanorix_rate_limited_total{endpoint}`

### 5.3 Tracing

- Instrument FastAPI with OpenTelemetry (`opentelemetry-instrumentation-fastapi`).
- Propagate trace context to external dependencies.
- Add spans for:
  - routing,
  - specialist embedding calls,
  - fusion computation,
  - retrieval backend call,
  - cache get/set.

### 5.4 Health/readiness

- `/health/live`: process heartbeat only.
- `/health/ready`: validate model registry loaded, vector backend reachable, Redis reachable (or degraded mode indicator).

---

## 6) Example Docker deployment setup

### 6.1 `Dockerfile`

```dockerfile
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY examples /app/examples

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .[api]

EXPOSE 8000

CMD ["uvicorn", "examples.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 6.2 `docker-compose.yml` (API + Redis)

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - KALMANORIX_ENV=prod
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes"]
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### 6.3 Production hardening checklist

- Run as non-root user.
- Pin dependencies with lock file.
- Configure resource limits/requests.
- Use managed secret store for API keys.
- Enable TLS at ingress and enforce authn/authz.
- Configure graceful shutdown and readiness gates.

---

## 7) Implementation plan

### Phase 1 — Foundation (Day 1-2)

1. Create `kalmanorix/api/` package with:
   - `main.py` (app factory),
   - `schemas.py`,
   - `routes/embedding.py`, `routes/retrieval.py`, `routes/fusion.py`,
   - `deps.py` (auth, request id, clients),
   - `errors.py` (domain + exception mapping).
2. Add centralized settings (`pydantic-settings`) from environment variables.
3. Wire OpenAPI tags and response models for all endpoints.

### Phase 2 — Reliability controls (Day 3-4)

1. Add rate limiting middleware and endpoint-level policies.
2. Add cache abstraction with in-memory + Redis implementations.
3. Add timeout/retry policy for external embedders and vector DB.
4. Implement structured error envelope and global handlers.

### Phase 3 — Observability (Day 4-5)

1. Integrate structured logging + request correlation IDs.
2. Add Prometheus metrics and `/v1/metrics` endpoint.
3. Add OpenTelemetry tracing and export pipeline.
4. Implement liveness/readiness endpoints.

### Phase 4 — Validation, testing, and delivery (Day 6-7)

1. Add tests:
   - schema validation/unit tests,
   - route tests with dependency overrides,
   - cache behavior tests,
   - rate-limit behavior tests,
   - failure-mode tests (dependency timeouts).
2. Add load test baseline (k6/Locust) for p95/p99 latency and throughput.
3. Build and run containerized stack; verify startup/readiness.
4. Publish runbook and SLO draft.

### Definition of done

- All required endpoints implemented with typed contracts.
- Error envelope consistent across all failures.
- Rate limiting and caching active in production profile.
- Metrics, logs, and traces visible in staging.
- Docker deployment reproducible with health checks.

---

## 8) Suggested next step

Start by extracting the existing `examples/fastapi_server.py` into a reusable `kalmanorix/api/` module and keep the example file as a thin compatibility wrapper. This minimizes migration risk while moving to a production layout.
