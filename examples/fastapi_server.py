"""
FastAPI server for Kalmanorix remote fusion.

This server exposes Kalmanorix fusion capabilities via REST API, allowing
remote clients to fuse embeddings from multiple specialist models.

Endpoints:
- GET /: Server info and available fusion strategies
- GET /modules: List available specialist modules
- POST /fuse: Fuse embeddings for a query with specified strategy

Run with:
    uvicorn examples.fastapi_server:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    MeanFuser,
    KalmanorixFuser,
    EnsembleKalmanFuser,
    StructuredKalmanFuser,
    DiagonalKalmanFuser,
    LearnedGateFuser,
)
from kalmanorix.types import Vec
from kalmanorix.uncertainty import KeywordSigma2

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def log_event(level: int, event: str, **kwargs: object) -> None:
    """Emit structured JSON log events."""
    payload = {"event": event, **kwargs}
    logger.log(level, json.dumps(payload, default=str))

# -----------------------------------------------------------------------------
# Rate limiting configuration
# -----------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# -----------------------------------------------------------------------------
# Response caching configuration
# -----------------------------------------------------------------------------

# Cache for fusion results: max 1000 entries, 5 minute TTL
fusion_cache: TTLCache[Tuple[str, str, str], Dict[str, Any]] = TTLCache(
    maxsize=1000, ttl=300
)

# -----------------------------------------------------------------------------
# Toy specialist setup (same as minimal_fusion_demo.py)
# -----------------------------------------------------------------------------

DIM = 16


class KeywordEmbedder:
    """Toy keyword-sensitive embedder (deterministic)."""

    def __init__(
        self,
        dim: int,
        seed: int,
        keywords: set[str],
        keyword_boost: float = 2.5,
    ):
        self.dim = dim
        self.keywords = keywords
        self.keyword_boost = keyword_boost

        rng = np.random.default_rng(seed)
        self._base_dir = rng.normal(size=(dim,))
        self._base_dir = self._base_dir / (np.linalg.norm(self._base_dir) + 1e-12)
        self._kw_dir = rng.normal(size=(dim,))
        self._kw_dir = self._kw_dir / (np.linalg.norm(self._kw_dir) + 1e-12)

    def __call__(self, text: str) -> Vec:
        t = text.lower()

        # Tiny deterministic "noise"
        noise = np.zeros(self.dim, dtype=np.float64)
        for ch in t[:64]:
            noise[(ord(ch) * 13) % self.dim] += 0.01

        vec = self._base_dir + noise

        if any(kw in t for kw in self.keywords):
            vec = vec + self.keyword_boost * self._kw_dir

        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype(np.float64)


def create_toy_village() -> Village:
    """Create the same toy specialists as minimal_fusion_demo."""
    tech_keywords = {
        "battery",
        "smartphone",
        "cpu",
        "gpu",
        "laptop",
        "android",
        "ios",
        "camera",
        "charger",
    }
    cook_keywords = {
        "braise",
        "simmer",
        "slow cooker",
        "recipe",
        "garlic",
        "onion",
        "saute",
        "oven",
        "stew",
    }

    tech = SEF(
        name="tech",
        embed=KeywordEmbedder(dim=DIM, seed=7, keywords=tech_keywords),
        sigma2=KeywordSigma2(
            tech_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5
        ),
        meta={"domain": "tech"},
    )
    cook = SEF(
        name="cook",
        embed=KeywordEmbedder(dim=DIM, seed=11, keywords=cook_keywords),
        sigma2=KeywordSigma2(
            cook_keywords, in_domain_sigma2=0.2, out_domain_sigma2=2.5
        ),
        meta={"domain": "cooking"},
    )

    return Village([tech, cook])


# -----------------------------------------------------------------------------
# Global server state
# -----------------------------------------------------------------------------

@dataclass
class SpecialistRuntime:
    village: Village
    scout_all: ScoutRouter
    scout_hard: ScoutRouter
    fusers: Dict[str, Any]


_runtime: Optional[SpecialistRuntime] = None
_runtime_lock = threading.RLock()

# -----------------------------------------------------------------------------
# FastAPI app and models
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Kalmanorix Fusion Server",
    description="REST API for fusing embeddings from multiple specialist models",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting state and error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


class ErrorResponse(BaseModel):
    """Consistent error envelope."""

    error: Dict[str, object]
    request_id: Optional[str] = None


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle expected HTTP errors with a consistent schema."""
    request_id = getattr(request.state, "request_id", None)
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={
                "code": "http_error",
                "message": message,
                "details": {"status_code": exc.status_code},
            },
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalize validation errors."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error={
                "code": "validation_error",
                "message": "Invalid request",
                "details": exc.errors(),
            },
            request_id=request_id,
        ).model_dump(),
    )


# Custom exception handler for general errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions with logging and JSON response."""
    request_id = getattr(request.state, "request_id", None)
    log_event(
        logging.ERROR,
        "request.unhandled_exception",
        request_id=request_id,
        path=request.url.path,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error={
                "code": "internal_error",
                "message": "Internal server error",
                "details": {"type": type(exc).__name__},
            },
            request_id=request_id,
        ).model_dump(),
    )


class FusionRequest(BaseModel):
    """Request body for fusion endpoint."""

    query: str = Field(..., description="Input text query to fuse")
    strategy: Literal[
        "mean",
        "kalmanorix",
        "ensemble_kalman",
        "structured_kalman",
        "diagonal_kalman",
        "learned_gate",
    ] = Field("kalmanorix", description="Fusion strategy to use")
    routing: Literal["all", "hard"] = Field(
        "all", description="Routing mode: 'all' (fusion) or 'hard' (single module)"
    )


class ModuleInfo(BaseModel):
    """Information about a specialist module."""

    name: str
    domain: Optional[str] = None
    sigma2_type: str
    sigma2_in_domain: Optional[float] = None
    sigma2_out_domain: Optional[float] = None


class FusionResponse(BaseModel):
    """Response from fusion endpoint."""

    query: str
    strategy: str
    routing: str
    selected_modules: List[str]
    fused_vector: List[float]
    weights: Dict[str, float]
    meta: Optional[Dict[str, object]] = None


class HealthResponse(BaseModel):
    """Liveness endpoint schema."""

    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    """Readiness endpoint schema."""

    status: Literal["ready", "not_ready"]
    modules_loaded: int
    cache_backend: str


class BatchEmbedRequest(BaseModel):
    """Request schema for optional batch embeddings."""

    texts: List[str] = Field(..., min_length=1, max_length=256)
    modules: Optional[List[str]] = None
    timeout_ms: int = Field(1500, ge=100, le=20000)


class ModuleEmbedding(BaseModel):
    module: str
    vector: List[float]


class BatchEmbedItem(BaseModel):
    text: str
    embeddings: List[ModuleEmbedding]
    errors: List[str] = Field(default_factory=list)


class BatchEmbedResponse(BaseModel):
    items: List[BatchEmbedItem]
    duration_ms: float


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_cache_key(query: str, strategy: str, routing: str) -> Tuple[str, str, str]:
    """Generate cache key for fusion request."""
    return (query, strategy, routing)


def cached_fusion(query: str, strategy: str, routing: str) -> Optional[Dict[str, Any]]:
    """Check cache for existing fusion result."""
    key = get_cache_key(query, strategy, routing)
    return fusion_cache.get(key)


def store_fusion_result(
    query: str, strategy: str, routing: str, result: Dict[str, Any]
) -> None:
    """Store fusion result in cache."""
    key = get_cache_key(query, strategy, routing)
    fusion_cache[key] = result


def get_runtime() -> SpecialistRuntime:
    """Concurrency-safe lazy initialization for specialists and fusers."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        village = create_toy_village()
        scout_all = ScoutRouter(mode="all")
        scout_hard = ScoutRouter(mode="hard")
        fusers = {
            "mean": MeanFuser(),
            "kalmanorix": KalmanorixFuser(),
            "ensemble_kalman": EnsembleKalmanFuser(),
            "structured_kalman": StructuredKalmanFuser(),
            "diagonal_kalman": DiagonalKalmanFuser(),
        }
        gate_fuser = LearnedGateFuser(
            module_a="tech",
            module_b="cook",
            n_features=128,
            lr=0.6,
            l2=1e-3,
            steps=400,
        )
        train_texts = [
            "Battery life is excellent on this smartphone",
            "The laptop CPU throttles under load",
            "Camera quality and charger compatibility",
            "Android update improved performance",
            "Braise the beef and simmer for hours",
            "Slow cooker recipe with garlic and onion",
            "Saute the vegetables then bake in the oven",
            "Stew tastes better after simmering",
        ]
        train_y = [1, 1, 1, 1, 0, 0, 0, 0]
        gate_fuser.fit(train_texts, train_y)
        fusers["learned_gate"] = gate_fuser
        _runtime = SpecialistRuntime(
            village=village,
            scout_all=scout_all,
            scout_hard=scout_hard,
            fusers=fusers,
        )
        log_event(logging.INFO, "runtime.initialized", modules=len(village.modules))
        return _runtime


def numpy_to_list(obj):
    """Recursively convert numpy arrays to Python lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    return obj


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------


@limiter.limit("200/minute")
@app.get("/", response_model=Dict[str, Any])
async def root(request: Request):
    """Server info and available endpoints."""
    return {
        "name": "Kalmanorix Fusion Server",
        "version": "0.1.0",
        "description": "REST API for fusing embeddings from multiple specialist models",
        "endpoints": {
            "GET /": "This info",
            "GET /modules": "List available specialist modules",
            "POST /fuse": "Fuse embeddings for a query",
        },
    }


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach request-id and structured request logs."""
    request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        raise
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["x-request-id"] = request.state.request_id
    log_event(
        logging.INFO,
        "request.completed",
        request_id=request.state.request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Liveness probe endpoint."""
    return HealthResponse(status="ok", service="kalmanorix-fusion-server", version="0.2.0")


@app.get("/readyz", response_model=ReadinessResponse)
async def readyz():
    """Readiness probe endpoint."""
    try:
        runtime = get_runtime()
        return ReadinessResponse(
            status="ready",
            modules_loaded=len(runtime.village.modules),
            cache_backend="in_memory_ttl",
        )
    except Exception:
        return ReadinessResponse(
            status="not_ready",
            modules_loaded=0,
            cache_backend="in_memory_ttl",
        )


@limiter.limit("200/minute")
@app.get("/modules", response_model=List[ModuleInfo])
async def list_modules(request: Request):
    """List available specialist modules in the village."""
    runtime = get_runtime()
    modules = []
    for sef in runtime.village.modules:
        sigma2 = sef.sigma2
        sigma2_type = type(sigma2).__name__
        sigma2_in_domain = None
        sigma2_out_domain = None

        # Extract sigma2 values for KeywordSigma2
        if hasattr(sigma2, "in_domain_sigma2") and hasattr(sigma2, "out_domain_sigma2"):
            sigma2_in_domain = sigma2.in_domain_sigma2
            sigma2_out_domain = sigma2.out_domain_sigma2

        modules.append(
            ModuleInfo(
                name=sef.name,
                domain=sef.meta.get("domain") if sef.meta else None,
                sigma2_type=sigma2_type,
                sigma2_in_domain=sigma2_in_domain,
                sigma2_out_domain=sigma2_out_domain,
            )
        )
    return modules


@limiter.limit("100/minute")
@app.post("/fuse", response_model=FusionResponse)
async def fuse(request: Request, body: FusionRequest):
    """Fuse embeddings for a query using the specified strategy."""
    request_id = getattr(request.state, "request_id", None)
    start_time = time.perf_counter()
    runtime = get_runtime()
    timeout_s = float(os.getenv("KALMANORIX_FUSE_TIMEOUT_SEC", "2.5"))

    # Check cache first
    cached_result = cached_fusion(body.query, body.strategy, body.routing)
    if cached_result is not None:
        log_event(
            logging.INFO,
            "fusion.cache_hit",
            request_id=request_id,
            query=body.query[:80],
            strategy=body.strategy,
            routing=body.routing,
        )
        return FusionResponse(**cached_result)

    log_event(
        logging.INFO,
        "fusion.request",
        request_id=request_id,
        query=body.query[:80],
        strategy=body.strategy,
        routing=body.routing,
    )

    # Select routing
    scout = runtime.scout_all if body.routing == "all" else runtime.scout_hard

    # Select fuser
    if body.strategy not in runtime.fusers:
        error_msg = f"Unknown fusion strategy: {body.strategy}. Available: {list(runtime.fusers.keys())}"
        raise HTTPException(status_code=400, detail=error_msg)
    fuser = runtime.fusers[body.strategy]

    # Perform fusion
    try:
        panoramix = Panoramix(fuser=fuser)
        potion = await asyncio.wait_for(
            asyncio.to_thread(
                panoramix.brew,
                body.query,
                runtime.village,
                scout,
            ),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        log_event(
            logging.WARNING,
            "fusion.timeout",
            request_id=request_id,
            timeout_sec=timeout_s,
            strategy=body.strategy,
            routing=body.routing,
        )
        raise HTTPException(
            status_code=504,
            detail="Fusion request timed out",
        ) from exc
    except Exception as exc:
        log_event(
            logging.ERROR,
            "fusion.failed",
            request_id=request_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail="Fusion failed") from exc

    # Convert numpy arrays to lists for JSON serialization
    fused_vector_list = potion.vector.tolist()
    weights_dict = {k: float(v) for k, v in potion.weights.items()}

    # Convert metadata if present
    meta_dict = None
    if potion.meta:
        meta_dict = numpy_to_list(potion.meta)

    # Build response
    response = FusionResponse(
        query=body.query,
        strategy=body.strategy,
        routing=body.routing,
        selected_modules=potion.meta.get("selected_modules", []) if potion.meta else [],
        fused_vector=fused_vector_list,
        weights=weights_dict,
        meta=meta_dict,
    )

    # Store in cache
    store_fusion_result(
        body.query,
        body.strategy,
        body.routing,
        response.model_dump(),
    )

    elapsed_time = round((time.perf_counter() - start_time) * 1000, 2)
    log_event(
        logging.INFO,
        "fusion.completed",
        request_id=request_id,
        duration_ms=elapsed_time,
        strategy=body.strategy,
        routing=body.routing,
    )

    return response


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: Request, body: BatchEmbedRequest):
    """Optional endpoint for batch specialist embeddings."""
    runtime = get_runtime()
    request_id = getattr(request.state, "request_id", None)
    start_time = time.perf_counter()
    target_modules = body.modules or [sef.name for sef in runtime.village.modules]
    module_map = {sef.name: sef for sef in runtime.village.modules}

    items: List[BatchEmbedItem] = []
    for text in body.texts:
        entry = BatchEmbedItem(text=text, embeddings=[], errors=[])
        for module_name in target_modules:
            sef = module_map.get(module_name)
            if sef is None:
                entry.errors.append(f"unknown_module:{module_name}")
                continue
            try:
                vector = await asyncio.wait_for(
                    asyncio.to_thread(sef.embed, text),
                    timeout=body.timeout_ms / 1000.0,
                )
                entry.embeddings.append(
                    ModuleEmbedding(module=module_name, vector=vector.tolist())
                )
            except TimeoutError:
                entry.errors.append(f"timeout:{module_name}")
            except Exception as exc:
                entry.errors.append(f"error:{module_name}:{type(exc).__name__}")
        items.append(entry)

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    log_event(
        logging.INFO,
        "embed.batch.completed",
        request_id=request_id,
        texts=len(body.texts),
        modules=len(target_modules),
        duration_ms=duration_ms,
    )
    return BatchEmbedResponse(items=items, duration_ms=duration_ms)


# -----------------------------------------------------------------------------
# Main entry point (for running directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
