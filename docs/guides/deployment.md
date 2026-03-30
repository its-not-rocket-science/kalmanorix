# Deployment

*TODO: Guide to deploying Kalmanorix in production environments.*

Kalmanorix can be deployed as a library embedded in your application, as a standalone REST API, or as part of a larger ML‑serving pipeline.

## Deployment Options

### 1. Embedded Library
Import Kalmanorix directly into your Python application. Suitable for batch processing or when you control the runtime environment.

**Pros:** Minimal latency, full control.
**Cons:** Requires Python environment, model loading overhead per process.

### 2. FastAPI REST Server
Use the built‑in FastAPI server (`kalmanorix.api.server`) to expose fusion as a web service.

**Pros:** Language‑agnostic clients, easy scaling, built‑in monitoring.
**Cons:** Network latency, additional infrastructure.

### 3. Docker Container
Package Kalmanorix with pre‑loaded specialists in a Docker image for consistent deployment.

**Pros:** Reproducible, cloud‑ready, isolation.
**Cons:** Image size, startup time.

### 4. Serverless (AWS Lambda, Google Cloud Functions)
Wrap fusion as a stateless function for sporadic workloads.

**Pros:** Cost‑effective for low volume, automatic scaling.
**Cons:** Cold‑start latency, limited memory for large specialists.

## FastAPI Server Setup

```bash
# Install API dependencies
pip install -e ".[api]"

# Start server (default port 8000)
python -m kalmanorix.api.server \
    --village_path ./my_village.json \
    --router_mode semantic \
    --fuser kalman
```

### API Endpoints
- `POST /brew` – Fuse embeddings for a single query.
- `POST /brew_batch` – Fuse embeddings for multiple queries.
- `GET /health` – Health check.
- `GET /metadata` – List loaded specialists and configuration.

### Configuration
The server can be configured via environment variables:
- `KALMANORIX_VILLAGE_PATH`: Path to village configuration or directory of SEFs.
- `KALMANORIX_ROUTER_MODE`: `all`, `hard`, or `semantic`.
- `KALMANORIX_FUSER`: `mean`, `kalman`, `diagonal_kalman`, `ensemble_kalman`, `structured_kalman`, `learned_gate`.

## Monitoring and Observability

### Logging
Kalmanorix uses Python’s standard logging at level `INFO`. Enable `DEBUG` for detailed fusion steps.

### Metrics
Key metrics to track:
- **Fusion latency** (p50, p95, p99)
- **Specialist selection count** (for semantic routing)
- **Cache hit rate** (if using embedding cache)
- **Error rates** (timeouts, rate‑limit exceedances)

### Health Checks
Implement readiness and liveness probes that verify:
- All specialists are loaded and responsive.
- Routing cache is warmed (if applicable).
- GPU memory available (if using GPU acceleration).

## Scaling Considerations

### Memory
Each specialist model resides in memory. Estimate:
```
total_memory ≈ sum(specialist_sizes) + (batch_size * embedding_dim * 4)
```

### Latency
Fusion adds overhead proportional to the number of selected specialists. Profile with realistic queries to set expectations.

### Caching
Cache frequent queries’ fused embeddings (or at least their routing decisions) to reduce compute.

*TODO: Add Kubernetes manifests, Terraform examples, load‑testing results, and cost‑estimation spreadsheet.*
