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

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException, Request, status
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

VILLAGE = create_toy_village()
SCOUT_ALL = ScoutRouter(mode="all")
SCOUT_HARD = ScoutRouter(mode="hard")

# Available fusion strategies
FUSERS = {
    "mean": MeanFuser(),
    "kalmanorix": KalmanorixFuser(),
    "ensemble_kalman": EnsembleKalmanFuser(),
    "structured_kalman": StructuredKalmanFuser(),
    "diagonal_kalman": DiagonalKalmanFuser(),
}

# Learned gate fuser needs training data - create and train it
GATE_FUSER = LearnedGateFuser(
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
GATE_FUSER.fit(train_texts, train_y)
FUSERS["learned_gate"] = GATE_FUSER

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


# Custom exception handler for general errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions with logging and JSON response."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc),
        },
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


@limiter.limit("200/minute")
@app.get("/modules", response_model=List[ModuleInfo])
async def list_modules(request: Request):
    """List available specialist modules in the village."""
    modules = []
    for sef in VILLAGE.modules:
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
    start_time = time.time()

    # Check cache first
    cached_result = cached_fusion(body.query, body.strategy, body.routing)
    if cached_result is not None:
        logger.info(
            "Cache hit for query='%s', strategy=%s, routing=%s",
            body.query[:50] + "..." if len(body.query) > 50 else body.query,
            body.strategy,
            body.routing,
        )
        return FusionResponse(**cached_result)

    logger.info(
        "Processing fusion request: query='%s', strategy=%s, routing=%s",
        body.query[:50] + "..." if len(body.query) > 50 else body.query,
        body.strategy,
        body.routing,
    )

    # Select routing
    scout = SCOUT_ALL if body.routing == "all" else SCOUT_HARD

    # Select fuser
    if body.strategy not in FUSERS:
        error_msg = f"Unknown fusion strategy: {body.strategy}. Available: {list(FUSERS.keys())}"
        logger.warning(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)
    fuser = FUSERS[body.strategy]

    # Perform fusion
    try:
        panoramix = Panoramix(fuser=fuser)
        potion = panoramix.brew(body.query, village=VILLAGE, scout=scout)
    except Exception as e:
        error_msg = f"Fusion error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg) from e

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

    elapsed_time = time.time() - start_time
    logger.info(
        "Fusion completed in %.3f seconds (strategy=%s, routing=%s)",
        elapsed_time,
        body.strategy,
        body.routing,
    )

    return response


# -----------------------------------------------------------------------------
# Main entry point (for running directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
