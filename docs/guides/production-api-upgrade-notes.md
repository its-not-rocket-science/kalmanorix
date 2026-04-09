# FastAPI Server Production Upgrade Notes (April 2026)

This note documents a **minimal-viable** upgrade of `examples/fastapi_server.py` from demo-grade to production-oriented behavior without rewriting the core fusion engine.

## What was not production-safe before

1. **Runtime initialization at import-time**
   - Specialists and learned gate training were executed during module import, which is brittle in multi-worker startup and hard to control operationally.
2. **Inconsistent error payloads**
   - Different paths returned different shapes (`detail`, `type`, raw exception text).
3. **Unstructured logging**
   - Logging used free-text messages, making metrics extraction and filtering harder.
4. **No explicit timeout behavior**
   - A slow embedder/fuser could hold request workers indefinitely.
5. **No probe endpoints for orchestration**
   - Missing liveness/readiness endpoints for Kubernetes/containers.
6. **Cache was hard-coded and opaque**
   - In-memory TTL existed but there were no explicit extension points for external caches.
7. **No optional batch embedding API**
   - API only exposed single-query fusion.

## Minimal viable code changes applied

1. **Concurrency-safe specialist loading**
   - Added lazy singleton runtime initialization via lock (`get_runtime`).
   - Prevents race-prone cold-start state creation under concurrent traffic.
2. **Clear request/response schemas**
   - Added explicit Pydantic models for health/readiness and batch embedding paths.
3. **Consistent error handling**
   - Added unified `ErrorResponse` envelope.
   - Added handlers for `HTTPException`, validation errors, and generic errors.
4. **Structured logging**
   - Added `log_event(...)` helper that emits JSON event logs.
   - Added per-request middleware generating request IDs and completion logs.
5. **Timeouts and graceful failures**
   - Wrapped fusion and embedding execution in `asyncio.wait_for` + thread offload.
   - Timeout returns controlled error instead of hung workers.
6. **Health/readiness endpoints**
   - Added `/healthz` and `/readyz`.
7. **Caching hooks (minimal)**
   - Preserved current TTL cache, documented as `in_memory_ttl` in readiness response.
   - Existing cache helpers remain isolated for future Redis/memcached swap.
8. **Optional batch embedding endpoint**
   - Added `/embed/batch` with per-item/per-module partial error capture.

## Deployment notes

- Configure timeout with:
  - `KALMANORIX_FUSE_TIMEOUT_SEC` (default: `2.5`).
- Keep the current server behind a process manager (e.g., Gunicorn/Uvicorn workers).
- Use `/healthz` for liveness checks and `/readyz` for readiness checks.
- Preserve request IDs by forwarding `x-request-id` from ingress.
- For production cache evolution, replace cache helper internals with a shared backend while preserving existing helper signatures.

## Load-testing checklist

Use this checklist before production cutover.

1. **Baseline latency**
   - Measure p50/p95/p99 for `/fuse` by strategy and routing mode.
2. **Timeout behavior**
   - Force slow specialists; verify timeout returns quickly with controlled error payload.
3. **Error schema contract**
   - Confirm 4xx/5xx responses share `{"error": ..., "request_id": ...}` format.
4. **Probe reliability**
   - Verify `/healthz` and `/readyz` under startup, steady-state, and degraded conditions.
5. **Cache effectiveness**
   - Replay repeated query sets; capture hit ratio and latency deltas.
6. **Batch endpoint**
   - Test `/embed/batch` with mixed valid/invalid module inputs and large text lists.
7. **Concurrency safety**
   - Load-test cold startup with parallel requests to ensure no duplicate initialization failures.
8. **Rate limits and observability**
   - Trigger rate limits and verify logging pipeline captures structured events for alerting.
