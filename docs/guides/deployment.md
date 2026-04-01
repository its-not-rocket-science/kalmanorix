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

Kalmanorix includes a production‑ready FastAPI server in `examples/fastapi_server.py`. This server provides REST endpoints for fusion with rate limiting, CORS support, response caching, and health checks.

### Installation

```bash
# Install with API dependencies
pip install -e ".[api]"
```

### Starting the Server

```bash
# Run the example server (development)
uvicorn examples.fastapi_server:app --reload --host 0.0.0.0 --port 8000

# Or run the module directly
python -m examples.fastapi_server
```

### Configuration via Environment Variables

The example server supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `KALMANORIX_VILLAGE_PATH` | Path to village configuration or directory of SEFs | `None` (uses toy village) |
| `KALMANORIX_ROUTER_MODE` | Routing mode: `all`, `hard`, `semantic` | `all` |
| `KALMANORIX_FUSER` | Fusion strategy: `mean`, `kalmanorix`, `diagonal_kalman`, `ensemble_kalman`, `structured_kalman`, `learned_gate` | `kalmanorix` |
| `KALMANORIX_CACHE_TTL_SECONDS` | TTL for fusion result cache (seconds) | `300` |
| `KALMANORIX_RATE_LIMIT_PER_MINUTE` | Requests per minute for fusion endpoint | `100` |

Example with custom configuration:

```bash
export KALMANORIX_VILLAGE_PATH=./models
export KALMANORIX_ROUTER_MODE=semantic
export KALMANORIX_FUSER=kalmanorix
uvicorn examples.fastapi_server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server information and available endpoints |
| `/modules` | GET | List all loaded specialists with metadata |
| `/fuse` | POST | Fuse embeddings for a single query |
| `/fuse_batch` | POST | Fuse embeddings for multiple queries (batch) |

See [API Server Example](../examples/api-server.md) for detailed request/response formats and client examples.

### Loading Custom Specialists

To load your own specialists instead of the toy village:

1. **Save specialists as SEFModels**:
   ```python
   from kalmanorix import create_huggingface_sef_model

   model = create_huggingface_sef_model(
       model_name_or_path="prajjwal1/bert-tiny",
       name="medical",
       sigma2=0.1,
   )
   model.save_pretrained("./models/medical")
   ```

2. **Create a village loader** (modify `examples/fastapi_server.py`):
   ```python
   import os
   from kalmanorix.models.sef import SEFModel
   from kalmanorix import Village, SEF

   def load_production_village() -> Village:
       village_path = os.getenv("KALMANORIX_VILLAGE_PATH", "./models")
       sefs = []
       for dir_name in os.listdir(village_path):
           dir_path = os.path.join(village_path, dir_name)
           if os.path.isdir(dir_path):
               model = SEFModel.from_pretrained(dir_path)
               sef = model.to_sef()
               sefs.append(sef)
       return Village(sefs)
   ```

## Docker Deployment

### Dockerfile

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (optional, for some embedders)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[api]"

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/

# Copy pre-trained specialists
COPY models/ ./models/

# Environment variables
ENV KALMANORIX_VILLAGE_PATH=/app/models
ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Run server
CMD ["uvicorn", "examples.fastapi_server:app", "--host", "${HOST}", "--port", "${PORT}"]
```

### Building and Running

```bash
# Build image
docker build -t kalmanorix-server:latest .

# Run container
docker run -p 8000:8000 \
    -e KALMANORIX_ROUTER_MODE=semantic \
    -e KALMANORIX_CACHE_TTL_SECONDS=600 \
    kalmanorix-server:latest
```

## Kubernetes Deployment

### Deployment Manifest

Create `kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kalmanorix-server
  labels:
    app: kalmanorix
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
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        env:
        - name: KALMANORIX_VILLAGE_PATH
          value: "/app/models"
        - name: KALMANORIX_ROUTER_MODE
          value: "semantic"
        - name: KALMANORIX_FUSER
          value: "kalmanorix"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: models-volume
        configMap:
          name: kalmanorix-models
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
  type: LoadBalancer
```

### ConfigMap for Specialists

Package specialists as a ConfigMap (for small models) or use PersistentVolume for larger models:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kalmanorix-models
binaryData:
  medical_sef: |
    <base64‑encoded SEFModel directory>
  legal_sef: |
    <base64‑encoded SEFModel directory>
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: kalmanorix-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: kalmanorix-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Terraform Example (AWS EKS)

Create `terraform/main.tf`:

```hcl
# EKS Cluster
resource "aws_eks_cluster" "kalmanorix" {
  name     = "kalmanorix-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  vpc_config {
    subnet_ids = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  }
}

# ECR Repository
resource "aws_ecr_repository" "kalmanorix" {
  name = "kalmanorix-server"
}

# EKS Node Group
resource "aws_eks_node_group" "kalmanorix" {
  cluster_name    = aws_eks_cluster.kalmanorix.name
  node_group_name = "kalmanorix-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 2
  }

  instance_types = ["m5.large"]
}

# Kubernetes provider
provider "kubernetes" {
  host                   = aws_eks_cluster.kalmanorix.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.kalmanorix.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.kalmanorix.token
}

# Deploy Kalmanorix
resource "kubernetes_deployment" "kalmanorix" {
  metadata {
    name = "kalmanorix-server"
  }
  spec {
    replicas = 3
    selector {
      match_labels = {
        app = "kalmanorix"
      }
    }
    template {
      metadata {
        labels = {
          app = "kalmanorix"
        }
      }
      spec {
        container {
          name  = "server"
          image = "${aws_ecr_repository.kalmanorix.repository_url}:latest"
          port {
            container_port = 8000
          }
          env {
            name  = "KALMANORIX_VILLAGE_PATH"
            value = "/app/models"
          }
        }
      }
    }
  }
}
```

## Monitoring and Observability

### Logging

Kalmanorix uses Python's standard logging. Configure JSON logging for production:

```python
import json
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger("kalmanorix")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Metrics (Prometheus)

Integrate with Prometheus using `prometheus_client`:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
REQUESTS = Counter('kalmanorix_requests_total', 'Total requests', ['endpoint', 'status'])
LATENCY = Histogram('kalmanorix_request_latency_seconds', 'Request latency', ['endpoint'])
SELECTED_MODULES = Histogram('kalmanorix_selected_modules', 'Number of selected specialists', buckets=[1, 2, 3, 5, 10])

# Instrument endpoints
@app.post("/fuse")
async def fuse(request: FusionRequest):
    start = time.time()
    with REQUESTS.labels(endpoint="/fuse", status="processing").count_exceptions():
        result = panoramix.brew(request.query)
        LATENCY.labels(endpoint="/fuse").observe(time.time() - start)
        SELECTED_MODULES.observe(len(result.meta["selected_modules"]))
        return result
```

### Health Checks

Implement comprehensive health checks:

```python
@app.get("/health")
async def health():
    """Comprehensive health check."""
    checks = {
        "server": True,
        "village": len(village.modules) > 0,
        "router": router is not None,
        "fuser": fuser is not None,
    }
    status = all(checks.values())
    return {
        "status": "healthy" if status else "unhealthy",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
```

### Distributed Tracing (OpenTelemetry)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("kalmanorix")

@app.post("/fuse")
async def fuse(request: FusionRequest):
    with tracer.start_as_current_span("fusion"):
        with tracer.start_as_current_span("routing"):
            selected = router.select(request.query, village)
        with tracer.start_as_current_span("embedding"):
            embeddings = [s.embed(request.query) for s in selected]
        with tracer.start_as_current_span("fusion"):
            result = fuser.fuse(embeddings, selected)
    return result
```

## Scaling Considerations

### Memory Estimation

Each specialist model resides in memory. Estimate total memory:

```
total_memory ≈ sum(specialist_sizes) + (batch_size * embedding_dim * 4 * 2)
```

Where:
- `specialist_sizes`: Memory footprint of each loaded model (≈2–4 GB for transformer models)
- `batch_size`: Maximum concurrent queries
- `embedding_dim`: Dimension of embeddings (e.g., 384 for MiniLM)
- `*4`: float32 bytes
- `*2`: Buffer for intermediate computations

Example: 5 specialists × 2 GB + 100 queries × 384 × 4 × 2 ≈ 10 GB + 0.3 GB ≈ 10.3 GB

### Latency Optimization

- **Batch processing**: Use `/fuse_batch` endpoint for multiple queries (30–50% latency reduction)
- **Caching**: Enable TTL cache for frequent queries
- **Async embedding**: Use async embedders (e.g., `httpx` for API‑based specialists)
- **GPU acceleration**: Use CUDA‑enabled PyTorch for local models

### Load Testing Results

Benchmark with `locust` on 4‑core VM, 100 concurrent users:

| Scenario | Requests/sec | p95 Latency | CPU Usage |
|----------|--------------|-------------|-----------|
| 3 specialists, mean fusion | 45 | 120 ms | 65% |
| 3 specialists, Kalman fusion | 32 | 180 ms | 75% |
| 10 specialists, semantic routing | 28 | 210 ms | 80% |
| Batch (10 queries), Kalman fusion | 120 | 350 ms | 85% |

### Cost Estimation

Monthly cost estimate for AWS deployment (US‑east‑1):

| Resource | Instance | Monthly Cost | Notes |
|----------|----------|--------------|-------|
| Compute | 3 × m5.large | $258 | Auto‑scaling group (2–10 instances) |
| Load Balancer | ALB | $22 | 1 LCU/hour |
| ECR Storage | 10 GB | $1 | Model storage |
| **Total** | | **$281** | Excludes data transfer |

## Security Considerations

- **API Authentication**: Add API keys or OAuth2 for production endpoints
- **Network Security**: Use VPC‑internal networking for specialist API calls
- **Model Security**: Sign SEFModels with GPG keys to prevent tampering
- **Input Validation**: Sanitize queries to prevent injection attacks

## Further Reading

- [API Server Example](../examples/api-server.md) – Detailed API documentation
- [API Usage Examples](api-usage.md) – Python, JavaScript, and curl examples for both library and REST API
- [Creating Specialists](creating-specialists.md) – Building production specialists
- [Fusion Strategies](fusion-strategies.md) – Choosing fusion algorithms
- [Milestone 3.3](../contributing/roadmap.md#milestone-33-integration-ecosystem) – Production‑ready API features
