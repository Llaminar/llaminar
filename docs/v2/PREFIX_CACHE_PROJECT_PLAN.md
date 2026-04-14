# Prefix Cache Project Plan

**Status**: Planning  
**Priority**: High (server throughput multiplier)  
**Scope**: New subsystem — additive, no existing API changes

## Motivation

In server mode, Llaminar clears the KV cache between every HTTP request. Many requests share identical prefixes — system prompts, few-shot examples, tool definitions — often 500–2000 tokens. Today every request recomputes the full prefix KV from scratch.

With prefix caching, a 2000-token system prompt is computed **once** and reused across all subsequent requests. For a typical chat workload where the system prompt is 80% of the input, this eliminates ~80% of prefill computation.

### Prior Art

| System | Approach | Granularity | Detection | Eviction |
|--------|----------|-------------|-----------|----------|
| **llama.cpp** | Seq bitmask sharing | Per-token | Manual (`seq_cp`) | Implicit ref-count + SWA |
| **vLLM** | Hash-chain block pool | Per-block (16 tok) | Automatic (content hash) | LRU doubly-linked list |
| **Llaminar (proposed)** | Hash-chain block cache on top of ring buffer | Per-block (64 tok) | Automatic (content hash) | LRU doubly-linked list |

We adopt vLLM's automatic hash-chain approach but layer it on top of our existing ring buffer rather than replacing the cache architecture. This preserves all existing kernel, stage, and graph capture compatibility.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    PrefixBlockCache                       │
│  (persistent across requests, lives in VRAM / host RAM)   │
│                                                           │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ BlockHasher  │  │  BlockPool       │  │ LRU Evictor  │ │
│  │ (xxHash-128  │  │  (pre-allocated  │  │ (doubly-     │ │
│  │  hash-chain) │  │   K/V storage)   │  │  linked list)│ │
│  └─────────────┘  └──────────────────┘  └──────────────┘ │
│                                                           │
│  hash_map: BlockHash → PrefixBlock*                       │
│  Operations: lookup / store / evict                       │
└─────────────────────┬─────────────────────────────────────┘
                      │
             populate / harvest
                      │
┌─────────────────────▼─────────────────────────────────────┐
│               IKVCache (ring buffer)                       │
│  (per-request, cleared between requests)                   │
│  Unchanged API — stages and kernels see no difference      │
└───────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Ring buffer stays unchanged** — attention kernels, append stages, and graph capture are unaffected
2. **Additive, not invasive** — new files, minimal modifications to existing code
3. **Device-aware** — prefix blocks live on the same device as the KV cache (VRAM for GPU, host RAM for CPU)
4. **Precision-agnostic** — stores raw bytes; works with FP16, BF16, Q8_1, TQ4, etc.
5. **Hybrid-cache compatible** — only caches FA layers for Qwen 3.5-style models (GDN layers have recurrent state, not cacheable)

## Key Concepts

### Hash-Chain Block Hashing

Each block's hash depends on the **previous block's hash**, creating a chain that ensures only true prefixes match:

```
tokens:  [t₀ t₁ ... t₆₃] [t₆₄ t₆₅ ... t₁₂₇] [t₁₂₈ ...]
           Block 0              Block 1            Block 2

hash₀ = H(SEED,  tokens[0:64])
hash₁ = H(hash₀, tokens[64:128])    ← depends on hash₀
hash₂ = H(hash₁, tokens[128:192])   ← depends on hash₁
```

Identical tokens at different positions produce different hashes — only true shared prefixes match. This is the same approach as vLLM.

### Block Size: 64 Tokens

We use 64-token blocks (vs vLLM's 16) because our populate/harvest operations are memcpy-based rather than pointer-based. Larger blocks mean:
- Fewer hash computations and dict lookups
- Fewer memcpy calls (each copy is larger but there are fewer of them)
- Slightly coarser granularity (worst case: recompute up to 63 tokens that could have been cached)
- Better amortization of per-block metadata overhead

Block size is configurable for tuning.

### Block Lifecycle

```
 FREE (in LRU eviction queue)
   │
   │ allocate_block() during harvest
   ▼
 ALLOCATED (ref_cnt=1, KV data being copied in)
   │
   │ insert(hash, block) — added to hash map
   ▼
 CACHED + IN-USE (ref_cnt≥1, in hash map, removed from LRU queue)
   │
   │ release() — request finishes
   ▼
 CACHED + EVICTABLE (ref_cnt=0, in hash map AND LRU queue)
   │                    ↑
   │ touch() on hit     │ release() again
   ├────────────────────┘
   │
   │ evict() — memory pressure
   ▼
 FREE (hash cleared, back in LRU queue)
```

## Data Structures

### PrefixBlock

```cpp
struct PrefixBlock {
    int block_id;                          // Index in block pool
    int ref_cnt = 0;                       // Active references
    BlockHash block_hash;                  // Content hash (0 if free)

    // Per-layer K/V storage (device pointers for GPU, host pointers for CPU)
    // Layout: [block_size × kv_dim] per layer, contiguous
    void* k_data;                          // Points into block pool allocation
    void* v_data;                          // Points into block pool allocation

    // LRU doubly-linked list pointers
    PrefixBlock* lru_prev = nullptr;
    PrefixBlock* lru_next = nullptr;
};
```

### BlockHash

```cpp
// 128-bit hash (xxHash-128 output)
struct BlockHash {
    uint64_t low;
    uint64_t high;

    bool operator==(const BlockHash& o) const;
    struct Hasher { size_t operator()(const BlockHash& h) const; };

    static constexpr BlockHash SEED = {0, 0};
};
```

### PrefixBlockCache

```cpp
class PrefixBlockCache {
public:
    PrefixBlockCache(const PrefixCacheConfig& config,
                     int n_layers, int kv_dim, int kv_element_size,
                     DeviceId device);

    // --- Core operations ---

    // Find longest prefix match. Returns number of matched tokens.
    // On hit, matched blocks have ref_cnt incremented.
    int lookup(const BlockHash* block_hashes, int num_blocks,
               std::vector<PrefixBlock*>& matched_blocks);

    // Store completed blocks from ring buffer into cache.
    void store(const BlockHash* block_hashes, int num_blocks,
               const IKVCache& source_cache, int seq_idx);

    // Release references (called when request completes).
    void release(const std::vector<PrefixBlock*>& blocks);

    // --- Memory management ---
    int num_cached_blocks() const;
    int num_free_blocks() const;
    size_t memory_usage_bytes() const;

    // --- Stats ---
    struct Stats {
        uint64_t lookups = 0;
        uint64_t hits = 0;          // Block-level hits
        uint64_t tokens_saved = 0;  // Tokens not recomputed
        uint64_t evictions = 0;
        uint64_t stores = 0;
    };
    const Stats& stats() const;

private:
    std::unordered_map<BlockHash, PrefixBlock*, BlockHash::Hasher> hash_map_;
    LRUQueue lru_queue_;                   // Doubly-linked eviction list
    std::vector<PrefixBlock> block_pool_;  // Pre-allocated block metadata
    void* block_storage_;                  // Contiguous K/V memory allocation

    PrefixCacheConfig config_;
    int n_layers_, kv_dim_, kv_element_size_;
    DeviceId device_;
    Stats stats_;

    PrefixBlock* allocate_block();         // Pop from LRU, may evict
    void evict(PrefixBlock* block);        // Remove from hash map, reset
};
```

### PrefixBlockHasher

```cpp
class PrefixBlockHasher {
public:
    explicit PrefixBlockHasher(int block_size = 64);

    // Hash a token sequence into block hashes.
    // Only complete blocks are hashed (trailing tokens ignored).
    // Returns number of complete block hashes written.
    int hash_blocks(const int* token_ids, int num_tokens,
                    BlockHash* out_hashes, int max_blocks) const;

    int block_size() const;

private:
    int block_size_;
};
```

### PrefixCacheConfig

```cpp
struct PrefixCacheConfig {
    bool enabled = false;
    size_t memory_budget_mb = 512;     // Max memory for prefix cache
    int block_size = 64;               // Tokens per block
    // Computed at init:
    // num_blocks = memory_budget / (n_layers * block_size * kv_dim * elem_size * 2)
};
```

## Integration Points

### Modified Files

| File | Change | LOC |
|------|--------|-----|
| `KernelFactory.h` | Add `PrefixCacheConfig` to `KVCacheConfig` | ~10 |
| `DeviceGraphOrchestrator.h` | Add `PrefixBlockCache` member, `prefix_hasher_` member | ~10 |
| `DeviceGraphOrchestrator.cpp` | Create prefix cache during init; wire populate/harvest into `forward()` and `clear_cache()` | ~80 |
| `IInferenceRunner.h` | Add `prefix_cache_stats()` query method (optional) | ~5 |
| `ChatCompletionHandler.cpp` | Log prefix cache hit rate in response headers (optional) | ~10 |
| `OrchestrationConfig.h/cpp` | Add `--prefix-cache`, `--prefix-cache-budget` CLI flags | ~20 |
| `DebugEnv.h` | Add `LLAMINAR_PREFIX_CACHE`, `LLAMINAR_PREFIX_CACHE_BUDGET_MB` env vars | ~5 |

### New Files

| File | Purpose | LOC |
|------|---------|-----|
| `src/v2/memory/prefix_cache/BlockHash.h` | 128-bit hash type + xxHash-128 wrapper | ~80 |
| `src/v2/memory/prefix_cache/PrefixBlockHasher.h/cpp` | Hash-chain block hasher | ~100 |
| `src/v2/memory/prefix_cache/PrefixBlock.h` | Block metadata struct | ~40 |
| `src/v2/memory/prefix_cache/LRUQueue.h` | Doubly-linked eviction queue | ~120 |
| `src/v2/memory/prefix_cache/PrefixBlockCache.h/cpp` | Main cache: hash map + pool + LRU + populate/harvest | ~500 |
| `src/v2/memory/prefix_cache/PrefixCacheConfig.h` | Configuration struct | ~30 |
| `tests/v2/unit/memory/Test__PrefixBlockHasher.cpp` | Hasher unit tests | ~150 |
| `tests/v2/unit/memory/Test__LRUQueue.cpp` | LRU queue unit tests | ~100 |
| `tests/v2/unit/memory/Test__PrefixBlockCache.cpp` | Cache logic unit tests (CPU) | ~300 |
| `tests/v2/integration/Test__PrefixCacheE2E.cpp` | End-to-end server-style test | ~200 |

### Unchanged (No Modifications Required)

- `IKVCache.h` — interface unchanged
- `CPURingKVCache.h` — ring buffer unchanged
- `CUDARingKVCache.h/.cu` — ring buffer unchanged
- `ROCmRingKVCache.h/.cpp` — ring buffer unchanged
- `KVCacheAppendStage` — appends after pre-populated region, no change
- `AttentionComputeStage` — reads from ring buffer as before
- All attention kernels — see contiguous ring buffer data
- CUDA/ROCm graph capture — ring buffer state populated before capture
- Hybrid KV cache — prefix cache operates at ring buffer level

## Populate and Harvest Operations

### Populate (Cache → Ring Buffer)

Called at the start of `forward()` before prefill. Copies cached KV blocks into the ring buffer so the model skips recomputing them.

```
populate(tokens, seq_len, kv_cache):
    block_hashes = hasher.hash_blocks(tokens, seq_len)
    matched = []
    for hash in block_hashes:
        block = cache.lookup(hash)
        if block == null:
            break                          // Chain broken — stop
        matched.push(block)

    if matched.empty():
        return 0                           // Full miss

    num_cached_tokens = matched.size() * block_size

    for each matched block:
        for layer in 0..n_layers:
            offset = block_index * block_size * kv_dim * elem_size
            // CPU: memcpy
            // CUDA: cudaMemcpyAsync (D2D if both in VRAM)
            // ROCm: hipMemcpyAsync
            copy(kv_cache.K[layer] + offset, block.k_data + layer_offset, ...)
            copy(kv_cache.V[layer] + offset, block.v_data + layer_offset, ...)
        kv_cache.advance_head(block_size)  // Update ring buffer state

    return num_cached_tokens
```

**Performance**: For a 2000-token prefix (31 blocks at size 64) on a 7B model with FP16 KV:
- Per-block copy: `32 layers × 2 (K+V) × 64 tokens × 1024 dim × 2 bytes = 8 MB`
- Total: `31 × 8 MB = 248 MB` copy, at ~50 GB/s PCIe → ~5 ms
- vs recomputing 2000 tokens at ~300 tok/s → ~6700 ms
- **Speedup: ~1300×** for the prefix portion

### Harvest (Ring Buffer → Cache)

Called after prefill completes (or at `clear_cache()` time). Copies computed KV blocks from the ring buffer into the prefix cache for future reuse.

```
harvest(tokens, seq_len, kv_cache):
    block_hashes = hasher.hash_blocks(tokens, seq_len)
    for i, hash in enumerate(block_hashes):
        if cache.contains(hash):
            continue                       // Already cached
        block = cache.allocate_block()     // May evict LRU
        if block == null:
            break                          // Cache full, no evictable blocks

        for layer in 0..n_layers:
            offset = i * block_size * kv_dim * elem_size
            copy(block.k_data + layer_offset, kv_cache.K[layer] + offset, ...)
            copy(block.v_data + layer_offset, kv_cache.V[layer] + offset, ...)
        cache.insert(hash, block)
```

### Orchestrator Integration

```cpp
// In DeviceGraphOrchestrator::forward():
bool forward(const int* tokens, int seq_len) {
    int cached_tokens = 0;

    if (prefix_cache_ && prefix_cache_->config().enabled && is_prefill) {
        cached_tokens = prefix_cache_->populate(tokens, seq_len, *state_.kv_cache);
    }

    if (cached_tokens < seq_len) {
        // Run prefill for remaining tokens only
        run_prefill(tokens + cached_tokens, seq_len - cached_tokens,
                    /*position_offset=*/cached_tokens);
    }

    // ... decode loop ...
}

// In DeviceGraphOrchestrator::clear_cache():
void clear_cache() {
    if (prefix_cache_ && prefix_cache_->config().enabled) {
        // Harvest current ring buffer contents before clearing
        prefix_cache_->harvest(last_tokens_, last_seq_len_, *state_.kv_cache);
        prefix_cache_->release(current_matched_blocks_);
    }
    state_.clear();  // Clear ring buffer (not prefix cache)
    ++session_epoch_;
}
```

## Memory Budget Analysis

Per-token KV storage cost depends on model size:

| Model | Layers | KV Heads | Head Dim | KV Dim | FP16 Per-Token | Per 64-Token Block |
|-------|--------|----------|----------|--------|----------------|-------------------|
| Qwen2.5 0.5B | 24 | 2 | 64 | 128 | 12 KB | 768 KB |
| Qwen2.5 7B | 28 | 4 | 128 | 512 | 56 KB | 3.5 MB |
| Llama 3.1 8B | 32 | 8 | 128 | 1024 | 128 KB | 8 MB |
| Llama 3.1 70B | 80 | 8 | 128 | 1024 | 320 KB | 20 MB |

With a 512 MB budget:

| Model | Blocks Available | Tokens Cacheable | Typical System Prompts |
|-------|------------------|------------------|-----------------------|
| Qwen2.5 0.5B | 682 | 43,648 | ~20 different prompts |
| Qwen2.5 7B | 149 | 9,536 | ~5 different prompts |
| Llama 3.1 8B | 65 | 4,160 | ~2 different prompts |
| Llama 3.1 70B | 26 | 1,664 | ~1 system prompt |

For 70B+ models, increase budget to 2–4 GB for practical multi-prompt caching.

## Hybrid Cache Considerations (Qwen 3.5)

Qwen 3.5 uses a mixed FA (full attention) + GDN (gated delta net) architecture. GDN layers have recurrent state that depends on the full sequence — not just the prefix — so they cannot be prefix-cached.

**Approach**: The prefix cache only stores blocks for FA layers. The `n_layers` parameter passed to `PrefixBlockCache` is `config.countKVLayers()` (FA layers only), not `total_layers`.

During populate, only FA layer slots in the ring buffer are filled. GDN layers are always recomputed. This works because `HybridRingKVCache` already separates FA and GDN state.

During harvest, only FA layer data is copied from the ring buffer to the prefix cache.

## Configuration

### CLI Flags

```
--prefix-cache                     Enable prefix caching (default: off)
--prefix-cache-budget <MB>         Memory budget in MB (default: 512)
--prefix-cache-block-size <N>      Tokens per block (default: 64)
```

### Environment Variables

```
LLAMINAR_PREFIX_CACHE=1                   Enable prefix caching
LLAMINAR_PREFIX_CACHE_BUDGET_MB=512       Memory budget
LLAMINAR_PREFIX_CACHE_BLOCK_SIZE=64       Block size
```

### Dry-Run Output

```
$ llaminar2 --prefix-cache --prefix-cache-budget 1024 --dry-run -m model.gguf

Prefix Cache Configuration:
  Enabled:      Yes
  Budget:       1024 MB
  Block Size:   64 tokens
  Blocks:       131 blocks (per KV dim 1024, 32 layers, FP16)
  Capacity:     8384 tokens
```

## Phased Implementation Plan

### Phase 1: Core Data Structures (CPU-only)

**Goal**: Build and unit-test the foundational components in isolation.

**Deliverables**:
- `BlockHash` — 128-bit hash type with xxHash-128 integration
- `PrefixBlockHasher` — hash-chain computation over token arrays
- `LRUQueue` — doubly-linked eviction queue with O(1) insert/remove/touch
- `PrefixBlock` — block metadata struct
- Unit tests for hasher (determinism, chain dependency, partial blocks)
- Unit tests for LRU queue (eviction order, touch promotion, empty/full)

**Dependencies**: xxHash library (header-only, add to `external/`)

**Tests**: ~250 LOC across `Test__PrefixBlockHasher.cpp`, `Test__LRUQueue.cpp`

### Phase 2: PrefixBlockCache (CPU, Host Memory)

**Goal**: Complete cache with hash map + LRU + block pool, tested on CPU.

**Deliverables**:
- `PrefixBlockCache` — full cache implementation with host memory block pool
- `PrefixCacheConfig` — configuration struct
- CPU populate operation (memcpy from cache → CPURingKVCache)
- CPU harvest operation (memcpy from CPURingKVCache → cache)
- Unit tests: lookup miss, lookup hit, chain-break, eviction, store+lookup round-trip, ref counting, stats tracking

**Tests**: ~300 LOC in `Test__PrefixBlockCache.cpp`

### Phase 3: Orchestrator Integration (CPU Path)

**Goal**: Wire prefix cache into the inference pipeline for CPU execution.

**Deliverables**:
- Add `PrefixCacheConfig` to `KVCacheConfig` and `OrchestrationConfig`
- Create `PrefixBlockCache` in `DeviceGraphOrchestrator` constructor
- Call `populate()` at start of `forward()` for prefill
- Call `harvest()` + `release()` in `clear_cache()`
- Skip recomputing cached prefix tokens (adjust `run_prefill` offset)
- Add CLI flags and environment variables
- Add `--dry-run` prefix cache info output

**Tests**: Integration test simulating two requests with shared prefix

### Phase 4: GPU Support (CUDA + ROCm)

**Goal**: Device-memory prefix cache for GPU execution paths.

**Deliverables**:
- Device-memory block pool allocation (`cudaMalloc` / `hipMalloc`)
- GPU populate: `cudaMemcpyAsync(D2D)` / `hipMemcpyAsync(D2D)` from cache blocks to ring buffer
- GPU harvest: reverse direction copy
- Handle stream synchronization (harvest must wait for prefill compute to complete)
- Device-aware `PrefixBlockCache` constructor (CPU vs CUDA vs ROCm allocation)

**Tests**: GPU integration test with CUDA and ROCm parity checks

### Phase 5: Hybrid Cache Support (Qwen 3.5)

**Goal**: Prefix caching for hybrid FA+GDN models.

**Deliverables**:
- Filter to FA-only layers during populate and harvest
- Use `HybridLayerMap` to translate layer indices
- Skip GDN layers (always recomputed)
- Correct block size calculation (FA layers only)

**Tests**: Qwen 3.5 prefix cache integration test

### Phase 6: Telemetry and Observability

**Goal**: Production-ready monitoring.

**Deliverables**:
- `PrefixCacheStats` — hit rate, tokens saved, eviction count, memory usage
- Log prefix cache stats at INFO level on each request
- Optional: expose stats via REST API response headers
- Optional: `--show-prefix-cache-stats` flag for periodic reporting
- Benchmark: measure prefill speedup with/without prefix cache

### Phase 7: Advanced Features (Future)

**Goal**: Optimize for production workloads.

**Potential features** (not committed):
- **Multi-tenant isolation**: Per-user cache salt to prevent cross-user cache sharing
- **Disk offload**: Evicted blocks written to SSD for larger effective cache
- **Warm-up mode**: Pre-populate cache from a list of known system prompts at startup
- **Concurrent request support**: Thread-safe cache for parallel HTTP requests (requires mutex on hash map)
- **Adaptive block size**: Smaller blocks for short prefixes, larger for long ones
- **LoRA-aware hashing**: Include LoRA adapter ID in hash (like vLLM's extra keys)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| memcpy overhead dominates small prefixes | Low | Low | Only activate for prefixes ≥ 1 block (64 tokens); below that, recompute is fast |
| Hash collision produces wrong KV | Negligible | High | xxHash-128: collision probability $2^{-128}$ per lookup; effectively impossible |
| Memory fragmentation from block pool | Low | Medium | Pre-allocate contiguous pool at init; no dynamic allocation on hot path |
| Quantized formats (Q8_1, TQ4) break | Low | Medium | Copy raw bytes; format is preserved. Unit test each precision mode |
| Graph capture incompatibility | None | — | Ring buffer populated *before* graph capture; graphs see normal state |
| Hybrid cache layer mismatch | Low | High | Unit test FA-only layer filtering with actual Qwen 3.5 config |
| Thread safety for concurrent requests | Medium | Medium | Phase 7; initially serialize requests (current server behavior) |

## Success Criteria

1. **Functional**: Two sequential requests with identical 1000-token system prompt — second request skips 960 tokens of prefill (15 blocks × 64 tokens)
2. **Performance**: Prefix cache populate is ≥100× faster than recomputing the same prefix
3. **Correctness**: Output tokens are bit-identical with and without prefix cache enabled
4. **Memory**: Cache stays within configured budget; no leaks after 1000 requests
5. **Stability**: All existing parity tests pass unchanged with prefix cache disabled (default)
