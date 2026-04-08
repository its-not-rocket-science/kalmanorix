# Kalmanorix Production Deployment Audit (April 2026)

## Scope

This audit evaluates the current Kalmanorix inference pipeline for production deployment, with focus on:

1. latency bottlenecks,
2. memory usage,
3. scalability as specialist count grows.

It is based on current implementation and benchmark artefacts in-repo.

---

## Current Architecture (inference path)

`query -> ScoutRouter.select() -> Panoramix.brew() -> Fuser.fuse() -> N specialist embed() calls (+ sigma2) -> fusion`

Key characteristics today:
- Routing loops over all specialists in Python (`O(N*d)` similarity checks).
- Fusion loops over all selected specialists and calls `module.embed(query)` sequentially.
- Query-dependent uncertainty methods may call `embed(query)` again, duplicating work.
- In the FastAPI example, cache is process-local TTL and key is exact query string.

---

## Bottleneck Analysis

## 1) Latency bottlenecks

### A. Sequential specialist inference is dominant
- `Panoramix.brew()` calls fuser once per request; fuser implementations iterate specialists and call `module.embed(query)` one-by-one.
- This creates near-linear scaling in wall time with specialist count.
- Existing benchmark shows mean-fusion latency increasing from ~11.8 ms (1 specialist) to ~266.3 ms (20), and Kalman from ~22.9 ms to ~572.7 ms.

**Impact:** Primary p95/p99 latency driver as N grows.

### B. Double work from uncertainty functions
- Query-dependent sigma2 implementations (e.g., centroid distance) can execute their own embedding pass.
- In the current fusion loop, embeddings are computed once for fusion and potentially again for sigma2, per specialist.

**Impact:** Up to ~2x specialist compute for sigma2-heavy deployments.

### C. Router overhead can erase gains when thresholding is poor
- Semantic routing adds fast embedding + similarity computation overhead.
- With fixed threshold 0.7, mixed queries can fall back to all specialists, causing extra overhead without fewer specialist invocations.

**Impact:** Negative latency gains in ambiguous-query traffic.

### D. Python object churn and repeated tiny allocations
- For each query, arrays/lists/dicts for embeddings, covariances, weights, metadata are rebuilt.
- `np.full(...)` covariance vectors are created per specialist per query.

**Impact:** GC pressure and avoidable CPU overhead at high QPS.

## 2) Memory usage

### A. Baseline memory is stable in benchmark setup, but not representative of true multi-model serving
- Current benchmark indicates ~778-779 MB across 1-20 specialists because wrappers share one embedder instance.
- In real deployment with distinct specialist checkpoints loaded simultaneously, memory should scale with number and size of loaded models.

**Impact:** Potential OOM risk once specialists are truly independent.

### B. Router/query caches are in-process only
- Router LRU (`_embedding_cache`) and FastAPI TTL cache are per-process.
- Multi-worker deployments duplicate cache footprint and lose warm-state sharing.

**Impact:** Higher aggregate memory and lower effective hit rates.

### C. No explicit memory budgeting / eviction by model weight
- No active specialist residency policy (e.g., keep-hot, unload-cold).

**Impact:** Unbounded memory growth if many specialists are loaded naively.

## 3) Scalability with number of specialists

### A. Core scaling behavior today
- FLOPs ratio tracks specialist count in all-routing mode (~N).
- Throughput drops roughly linearly as N increases.
- Routing compute cost itself is also O(N*d) and becomes material with many specialists.

### B. Duplicate-centroid behavior in scaled experiments
- When specialists are duplicated, semantic routing can select many near-identical modules.

**Impact:** Selection cardinality grows with duplication; efficiency decays.

---

## What must be parallelized (required for production)

## Must-parallelize #1: specialist embedding calls
Parallelize `module.embed(query)` across selected specialists.

**Implementation options:**
- Thread pool for CPU-bound native kernels (often effective if backend releases GIL).
- Async micro-batching executor for GPU inference.
- External model server fan-out (Ray Serve / Triton / vLLM-style workers).

**Why mandatory:** This is the dominant latency term and scales with selected specialist count.

## Must-parallelize #2: batched multi-query execution
Use `Panoramix.brew_batch()` path by grouping incoming queries (small dynamic micro-batches).

**Why mandatory:** Increases hardware utilization and amortizes Python overhead.

## Must-parallelize #3: routing similarity over centroids
Vectorize centroid similarity into one matrix operation (centroid matrix @ query vector), optionally SIMD/BLAS-backed.

**Why mandatory:** Needed to keep router overhead sub-millisecond as specialist count grows.

## Should-parallelize #4: sigma2 computation
Avoid separate embed pass; compute uncertainty from already-produced specialist embedding where possible.

**Why:** Removes duplicate compute and improves tail latency.

---

## Caching strategy (query + embedding + routing)

## A. Query-result cache (final fused output)

**Current:** Exact `(query, strategy, routing)` TTL cache in-process.

**Production strategy:**
- Two-tier cache: local LRU + shared Redis/Memcached.
- Normalize key: lowercase/trim/whitespace-folding + versioned config hash (model versions, routing config).
- Use short TTL for dynamic corpora; longer TTL for static retrieval workloads.
- Cache both response and selected_modules metadata.

**Estimated impact:**
- 20-60% latency reduction on repeat-heavy workloads (depends on hit rate).

## B. Embedding cache

**Goal:** Avoid repeated specialist and router embedding work.

**Strategy:**
- Cache fast-embedder query vectors (router stage).
- Cache per-specialist query embeddings: key `(specialist_id, normalized_query, model_revision)`.
- Pre-warm with top historical queries at startup.
- Use quantized vector cache (e.g., float16) to reduce RAM.

**Estimated impact:**
- 25-50% CPU/GPU compute reduction for repeated-query traffic.

## C. Routing cache

**Goal:** Avoid recomputing specialist selection when query intent repeats.

**Strategy:**
- Cache `selected_modules` for normalized query.
- Store top-k similarities as well for debugging and adaptive threshold tuning.
- Invalidate when centroid versions change.

**Estimated impact:**
- 5-20% p50 latency reduction; larger in high-N systems.

---

## Concrete optimization plan (prioritized)

## P0 (do first: 1-2 sprints)

1. **Parallel specialist inference fan-out**
   - Replace per-specialist sequential embed calls with bounded worker pool or model-serving fan-out.
   - **Estimated impact:** 35-70% p95 latency reduction for N>=5 (hardware dependent).

2. **Eliminate duplicate embedding for sigma2**
   - Refactor sigma2 APIs to accept precomputed embedding.
   - **Estimated impact:** 15-35% latency reduction for query-dependent uncertainty methods.

3. **Adaptive routing threshold defaults (top-k / relative-to-max)**
   - Avoid fallback-to-all on ambiguous queries.
   - **Estimated impact:** preserve current ~65% FLOPs reduction and avoid regressions where semantic routing increases latency.

4. **Vectorized centroid scoring**
   - Pre-stack centroids and compute one batched dot-product per query.
   - **Estimated impact:** 2-10 ms saved at high specialist counts; more stable router latency.

## P1 (next: 2-4 sprints)

5. **Two-tier distributed cache (query + embedding + routing)**
   - Add Redis-backed shared cache with model-versioned keys.
   - **Estimated impact:** 20-60% latency reduction on warm traffic; better multi-replica efficiency.

6. **Micro-batching at API layer + `brew_batch` usage**
   - Batch concurrent requests by short queue window (e.g., 5-20 ms).
   - **Estimated impact:** 1.5-3x throughput increase under load.

7. **Specialist residency / lazy loading policy**
   - Keep hot specialists loaded, evict cold ones by memory budget.
   - **Estimated impact:** prevents OOM and stabilizes memory for large specialist fleets.

## P2 (medium term)

8. **Approximate routing index for large specialist pools**
   - ANN over centroids or hierarchical routing (coarse -> fine).
   - **Estimated impact:** router complexity from O(N*d) toward sublinear for very large N (100+ specialists).

9. **Structured covariance fast path / lower precision fusion**
   - Use float32 or low-rank approximations where quality allows.
   - **Estimated impact:** 10-30% fusion compute reduction with minimal quality loss (to be validated).

10. **Production observability + autoscaling policies**
    - Add p50/p95/p99 latency by route/fuser/specialist-count, cache hit ratios, queue depth, memory-watermark alerts.
    - **Estimated impact:** indirect but essential for safe scaling and cost control.

---

## Deployment guardrails (minimum production checklist)

- Parallel specialist execution enabled.
- Adaptive routing threshold enabled (not fixed static threshold only).
- Shared cache configured with versioned keys.
- Per-request tracing captures:
  - selected specialists,
  - routing time,
  - embed time per specialist,
  - fusion time,
  - cache hit/miss.
- Memory budget and specialist residency policy defined.

---

## Summary: highest-leverage improvements

1. **Parallelize specialist inference first** (largest latency win).
2. **Remove duplicate embedding work for sigma2** (easy, high ROI).
3. **Deploy robust routing + routing cache** (protect FLOPs and tail latency).
4. **Adopt shared multi-tier caches** (major benefit in replicated production environments).
5. **Add micro-batching and memory residency controls** (scalability and cost stability).

Together, these changes should move the system from research-grade sequential inference toward production-grade low-latency serving while preserving Kalmanorix’s routing-driven efficiency advantage.
