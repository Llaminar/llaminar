# Prefix Cache Project Plan

**Status**: New implementation plan, revised for dense, hybrid attention, and Qwen 3.5 MoE models  
**Date**: 2026-04-26  
**Priority**: High (server throughput multiplier)  
**Scope**: New persistent prefix-state cache for cross-request reuse, with RAM-backed capacity as an early feature

## Summary

Llaminar currently clears request-local inference state before every server request. This is correct for isolation, but expensive for chat workloads where many requests share the same system prompt, tool definitions, few-shot examples, or conversation prefix.

The prefix cache should make repeated prefixes cheap by storing reusable **prefix state snapshots** at fixed token block boundaries. For dense transformer models, a snapshot is mostly full-attention KV. For hybrid attention models such as Qwen 3.5, a snapshot must also include GDN recurrence and short-convolution state. For Qwen 3.5 MoE variants, the cache must additionally be keyed by MoE execution placement/version so rebalanced expert layouts do not silently change numerics.

Because the intended deployment target includes consumer GPUs with 24-32 GB of VRAM, the first practical implementation cannot rely on device memory as the main cache capacity. The early design should use **RAM as the primary capacity tier** and treat VRAM/HBM as a small optional hot tier for the most frequently reused blocks. Disk-backed persistence can come later once the snapshot format and promotion path are stable.

The core design remains vLLM-style hash-chain prefix detection, but the payload is generalized from "KV blocks" to "model prefix state".

## Current Code Anchors

The current V2 codebase already gives us most of the places to integrate:

| Area | Current State | Implication |
|------|---------------|-------------|
| Server lifecycle | `ServerMode` serializes requests with a mutex; `ChatCompletionHandler::setupInference()` calls `runner_.clearCache()` before every request | Cross-request prefix reuse can be added without concurrent mutation in the first version |
| Runner lifecycle | `OrchestrationRunner::prefill()` calls `runner_->forward(prompt_tokens.data(), prompt_tokens.size())`; first `decodeStep()` samples from prefill logits instead of re-feeding the last prompt token | Prefix cache must preserve prefill-logits semantics |
| Local execution | `DeviceGraphOrchestrator::forward()` builds position IDs from `state_.positions`; `clearInferenceState()` clears KV and request-local graph caches | Prefix cache storage must live outside request-local graph caches |
| KV cache | `IKVCache` supports `get_kv()`, `append()`, `clear()`, graph-capture head helpers, multiple precisions, sharding, and layouts | Add a narrow logical block import/export API instead of copying assumed tensor rows directly |
| Hybrid state | `IHybridKVCache` exposes GDN state and kernels; `HybridGDNLayerState` owns recurrence and conv host vectors; CUDA/ROCm kernels also keep device state | Add snapshot/restore APIs that include both host and device-resident GDN state |
| Qwen 3.5 graph | `Qwen35Graph` dispatches FA vs GDN layers from `layer_types`; FA uses ring KV, GDN uses hybrid cache state | Prefix cache must restore both FA and GDN state before suffix execution |
| MoE | `MoERoutingStage`, `MoEExpertComputeStage`, `MoERebalanceController`, expert masks and replicas can change runtime expert placement | Prefix cache key must include a MoE placement/version fingerprint, or cache must invalidate on rebalance |
| Config | `OrchestrationConfigParser` uses `CliSpec`; dry-run exits before model load | Prefix flags go through `CliSpec`; exact capacity is reported after metadata is available |

## Goals

1. Reuse shared request prefixes across HTTP requests without changing model outputs.
2. Support dense full-attention models, hybrid FA+GDN models, and Qwen 3.5 MoE variants.
3. Keep attention kernels, compute stages, and graph execution semantics intact.
4. Make the hot path predictable: hash lookup plus RAM-to-device restore or device-local block copies, no dynamic allocation during suffix execution.
5. Preserve request isolation: `clearCache()` resets request-local state but not the persistent prefix cache.
6. Provide a RAM-backed cache early enough to make prefix caching useful on 24-32 GB consumer GPUs.
7. Keep prefix caching disabled by default until parity and stress tests pass for each supported model class.

## Non-Goals

- Cross-process shared prefix cache in the first version.
- Disk-backed cache in the first version. Disk should be a later backend after RAM and device promotion are correct.
- Concurrent request mutation in the first version. Server inference is currently serialized.
- Pointer-swapping the request ring buffer. We copy into the existing ring/cache state so graph capture and kernels continue to see normal request-local storage.
- Caching arbitrary partial blocks. Only complete blocks are cached; trailing tokens are recomputed.

## Model Support Strategy

| Model Class | Prefix Payload | Initial Support Target |
|-------------|----------------|------------------------|
| Dense full attention | FA KV blocks, optional terminal logits, stored primarily in RAM | Phase 3 |
| Hybrid FA+GDN, such as Qwen 3.5 | FA KV blocks plus GDN recurrence and conv state snapshot, stored primarily in RAM | Phase 6 |
| Qwen 3.5 MoE | Hybrid payload plus MoE placement/version fingerprint | Phase 7 |

Dense and hybrid models use the same lookup and LRU mechanics. The difference is what gets snapshotted at each matched block boundary.

## Core Concept: Prefix State Snapshot

A prefix block is identified by a hash-chain key and stores the exact model state after processing that block.

For a 64-token block size:

```text
tokens:    [0..63]       [64..127]       [128..191]
hash:      H(seed,b0) ->  H(h0,b1)  ->   H(h1,b2)
snapshot:  state@64       state@128       state@192
```

When a future request has the same first 192 tokens, Llaminar can restore `state@192`, set positions to 192, and run prefill only for the suffix.

For dense models, `state@N` is full-attention KV for tokens `0..N-1`. For Qwen 3.5, `state@N` is full-attention KV for FA layers plus all GDN recurrence and convolution state after token `N-1`. For MoE variants, expert routing is stateless, but the cache key must ensure the same expert placement/version is active when the snapshot is reused.

## Cache Key

The hash-chain token hash alone is not enough. A prefix state is reusable only if the model, execution layout, and state semantics match.

```cpp
struct PrefixCacheKey {
    BlockHash content_hash;        // Hash-chain token content
    uint64_t model_fingerprint;    // Model path, architecture, GGUF metadata, vocab/tokenizer version
    uint64_t runtime_fingerprint;  // KV precision, layout, rope_on_read, activation mode, backend family
    uint64_t topology_fingerprint; // TP/PP degree, rank shard, local device, first layer, local KV heads
    uint64_t hybrid_fingerprint;   // Layer types and GDN dimensions; zero for dense
    uint64_t moe_fingerprint;      // Expert placement/version/replica mask; zero for non-MoE
    uint32_t block_index;          // Defensive; hash chain should imply this, but store explicitly
};
```

### Fingerprint Rules

- `model_fingerprint` must include tokenizer/template identity. A different chat template can produce different token IDs, but including it prevents confusing cache diagnostics.
- `runtime_fingerprint` must include `ActivationPrecision`, K/V cache precision, `TensorLayout`, RoPE-on-read, and quantization mode such as TQ4 or split TQ.
- `topology_fingerprint` must differ per TP shard and PP slice. Each shard stores only its local payload.
- `hybrid_fingerprint` must include `layer_types`, `gdn.state_size`, `gdn.inner_size`, `gdn.group_count`, `gdn.time_step_rank`, and `gdn.conv_kernel_size`.
- `moe_fingerprint` must change when dynamic expert masks, ownership, replicas, or placement epoch change. Safest first behavior: invalidate or bypass prefix cache when `MoERebalanceMode::DYNAMIC` changes placement.

## Payload Layout

```cpp
enum class PrefixStorageTier {
    DEVICE_HOT,  // Optional VRAM/HBM resident copy for very hot blocks
    RAM,         // Primary capacity tier for the early implementation
    DISK         // Later extension, not in the first implementation
};

struct PrefixStateBlock {
    PrefixCacheKey key;
    int block_id = -1;
    int token_count = 0;           // Complete block size, normally 64
    int ref_count = 0;
    PrefixStorageTier resident_tier = PrefixStorageTier::RAM;

    PrefixPayloadLayout layout;

    // Dense and FA portion of hybrid models.
    void* kv_storage = nullptr;
    size_t kv_storage_bytes = 0;

    // Hybrid-only state after this block.
    void* hybrid_state_storage = nullptr;
    size_t hybrid_state_bytes = 0;

    // Optional. Used only when this block is a terminal complete prompt.
    void* terminal_logits = nullptr;
    size_t terminal_logits_bytes = 0;

    PrefixStateBlock* lru_prev = nullptr;
    PrefixStateBlock* lru_next = nullptr;
};
```

## Storage Architecture

Prefix-cache storage should be tiered from the beginning, but only the first two tiers are required for the initial production feature:

| Tier | Role | Backing | Initial Status |
|------|------|---------|----------------|
| L1 device hot cache | Fastest restore for the hottest blocks | VRAM/HBM on the local device | Optional early optimization, small budget |
| L2 RAM cache | Primary capacity tier for useful prefix reuse on consumer GPUs | Host memory, preferably pinned for GPU restore paths | Required in Phase 3 |
| L3 disk cache | Large persistent cache across process restarts | Local NVMe directory | Future extension |

The RAM tier is the source of truth for early prefix caching. A device-resident block is only a promotion of a RAM block, not the only copy. This keeps capacity useful on 24-32 GB devices and avoids using scarce VRAM for cold prefixes.

Recommended early policy:

1. Store new blocks in RAM after harvest.
2. Promote a block to the device hot tier after repeated hits or when a request is actively restoring it.
3. Keep the device hot tier small and evictable under pressure from KV cache, weights, workspace, or graph capture needs.
4. Never require device-tier residency for correctness; every cache hit must be restorable from RAM.
5. Use pinned host allocations for RAM blocks when the active backend is CUDA/ROCm and the configured RAM budget is not too large. Fall back to pageable RAM if pinned allocation fails.

This yields a simple restore path:

```text
lookup block
if block is in device hot tier:
    import from device block into request-local KV/GDN state
else:
    import from RAM block into request-local KV/GDN state
    optionally promote to device hot tier
run suffix prefill
```

Disk should reuse the same abstraction later, but with an explicit deserialize-and-promote step into RAM before device restore.

### Storage Backend Interface

The cache manager should depend on a narrow storage interface rather than embedding allocation policy in `PrefixStateCache`:

```cpp
struct PrefixBlockHandle {
    PrefixCacheKey key;
    PrefixStorageTier tier;
    void* kv_payload = nullptr;
    void* hybrid_payload = nullptr;
    void* logits_payload = nullptr;
    size_t total_bytes = 0;
};

class IPrefixStorageBackend {
public:
    virtual ~IPrefixStorageBackend() = default;
    virtual bool canStore(size_t bytes) const = 0;
    virtual PrefixBlockHandle allocate(const PrefixCacheKey& key,
                                       const PrefixPayloadLayout& layout) = 0;
    virtual bool release(const PrefixBlockHandle& handle) = 0;
    virtual bool copyToRequestState(const PrefixBlockHandle& handle,
                                    IKVCache& kv_cache,
                                    IHybridKVCache* hybrid_cache,
                                    int cached_tokens) = 0;
};
```

Initial backends:

- `RamPrefixStorageBackend`: required; owns the main LRU capacity.
- `DeviceHotPrefixStorageBackend`: optional; owns a small hot LRU and can be disabled by setting budget to zero.
- `DiskPrefixStorageBackend`: future extension.

### PrefixPayloadLayout

```cpp
struct PrefixPayloadLayout {
    DeviceId device;
    ActivationPrecision k_precision;
    ActivationPrecision v_precision;
    TensorLayout kv_layout;
    int first_layer_index = 0;
    int total_layers = 0;
    int fa_kv_layers = 0;
    int gdn_layers = 0;
    int local_kv_heads = 0;
    int kv_head_start = 0;
    int head_dim = 0;
    int block_size = 64;
    size_t bytes_per_fa_layer_k = 0;
    size_t bytes_per_fa_layer_v = 0;
    size_t bytes_per_gdn_snapshot = 0;
};
```

## Required APIs

### IKVCache Logical Block I/O

`PrefixStateCache` should not know how each KV cache arranges rows, heads, quantized blocks, ring wrap, or device streams. Add backend-neutral block I/O to `IKVCache`:

```cpp
struct KVCacheBlockDescriptor {
    int layer = 0;                 // Global layer index for hybrid caches
    int seq_idx = 0;
    int logical_token_start = 0;   // Oldest-to-newest logical index
    int num_tokens = 0;
    void* stream = nullptr;        // cudaStream_t / hipStream_t / nullptr
};

struct KVCacheBlockMetadata {
    ActivationPrecision k_precision;
    ActivationPrecision v_precision;
    TensorLayout layout;
    int local_kv_heads = 0;
    int kv_head_start = 0;
    int head_dim = 0;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
};

virtual KVCacheBlockMetadata blockMetadata(int global_layer, int num_tokens) const;
virtual bool exportLogicalBlock(const KVCacheBlockDescriptor& desc,
                                void* dst_k, void* dst_v) const;
virtual bool importLogicalBlock(const KVCacheBlockDescriptor& desc,
                                const void* src_k, const void* src_v);
```

Dense caches implement this directly. Hybrid caches remap FA global layer indices through `HybridLayerMap`; GDN layer export/import is handled by the hybrid-state API below, not KV block I/O.

### IHybridKVCache Prefix State I/O

Hybrid support should be explicit, not inferred from raw `HybridGDNLayerState` vectors. Extend `IHybridKVCache` with snapshot APIs:

```cpp
struct HybridPrefixStateMetadata {
    int total_layers = 0;
    int gdn_layers = 0;
    size_t host_bytes = 0;
    size_t device_bytes = 0;
    bool has_device_kernel_state = false;
};

struct HybridPrefixStateDescriptor {
    int seq_idx = 0;
    int logical_token_count = 0;   // State after this many prefix tokens
    void* stream = nullptr;
};

virtual HybridPrefixStateMetadata hybridPrefixStateMetadata() const = 0;
virtual bool exportHybridPrefixState(const HybridPrefixStateDescriptor& desc,
                                     void* dst_host, void* dst_device) const = 0;
virtual bool importHybridPrefixState(const HybridPrefixStateDescriptor& desc,
                                     const void* src_host, const void* src_device) = 0;
```

CPU hybrid caches can copy `HybridGDNLayerState::recurrence_state` and `conv_state` directly. CUDA and ROCm hybrid caches must also snapshot the device-resident state owned by `ITensorShortConvolution` and `ITensorGatedDeltaNet` implementations. If those kernels only expose reset today, add kernel-level export/import methods so D2D copies can be used when the prefix cache lives on the same GPU.

### Optional Terminal Logits

The current decode flow samples the first generated token from prefill logits. If a request is a full cache hit and no suffix is computed, there are no fresh logits.

Use this policy:

1. When harvesting a prompt whose length ends on a complete block boundary, store terminal logits for the final block if affordable.
2. On lookup, if the request would be a full hit and terminal logits are available, restore them into the runner's logits buffer and set `prefill_logits_ready_ = true`.
3. If terminal logits are unavailable, reduce the match by one block and recompute the final block. This preserves correctness without a special no-append graph path.

Terminal logits are optional because they can be large for vocab-heavy models. The fallback costs one block of prefill, which is acceptable and much simpler than duplicating a logits-only graph.

## Lookup, Populate, Harvest

### Lookup

```text
hash complete prompt blocks
for each block in order:
    find block by full PrefixCacheKey
    stop at first miss
if full hit and no terminal logits:
    reduce match by one block
coordinate matched block count across ranks/devices
return matched blocks and cached token count
```

### Populate

```text
clear request-local state
import FA KV blocks into IKVCache
if hybrid payload exists:
    import GDN recurrence/conv state from last matched block
set positions and sequence_lengths to cached_tokens
if terminal logits used:
    restore logits buffer and mark prefill logits ready
run suffix prefill if suffix_tokens > 0
```

For dense models, populate is only KV import. For hybrid models, populate is KV import plus hybrid-state import. For MoE models, populate also validates the MoE fingerprint.

### Harvest

Harvest immediately after successful prompt prefill, before decode adds generated tokens.

```text
hash complete prompt blocks
for each uncached complete block:
    allocate/eject PrefixStateBlock
    export FA KV blocks for all local FA layers
    if hybrid cache:
        export hybrid GDN state after this block
    if this is the final complete block of the prompt:
        optionally store terminal logits
    insert into hash map
```

For hybrid models, exporting state after every block can be expensive if it copies all GDN state every 64 tokens. Start with correctness, then optimize with larger block sizes or sparse checkpointing if needed.

## Runner Integration

`OrchestrationRunner::prefill()` should own the high-level flow because it already has the full prompt token vector and coordinates MPI workers.

```cpp
bool OrchestrationRunner::prefill(const std::vector<int32_t>& prompt_tokens) {
    runner_->clear_cache();

    PrefixLookupResult hit;
    if (prefix_cache_enabled_) {
        hit = runner_->lookupAndPopulatePrefix(prompt_tokens);
        hit.cached_tokens = coordinateMatchedTokensAcrossRanks(hit.cached_tokens);
    }

    const int suffix_start = hit.cached_tokens;
    const int suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;

    if (suffix_len > 0) {
        runner_->forward(prompt_tokens.data() + suffix_start, suffix_len);
        prefill_logits_ready_ = true;
    } else if (hit.has_terminal_logits) {
        runner_->restorePrefixTerminalLogits(hit);
        prefill_logits_ready_ = true;
    } else {
        // Defensive fallback: do not allow a no-logits full hit.
        // Reduce match by one block and recompute that final block.
    }

    runner_->harvestPrefix(prompt_tokens);
    return true;
}
```

`DeviceGraphOrchestrator` should expose lower-level operations:

- `lookupAndPopulatePrefix(tokens)`
- `harvestPrefix(tokens)`
- `restorePrefixTerminalLogits(hit)`
- `prefixCacheStats()`
- `setPrefixCache(std::shared_ptr<PrefixStateCache>)` or construction-time injection

`clear_cache()` continues clearing request-local ring state, positions, sequence lengths, and forward graph cache. It should release current prefix-cache references, but it must not evict the persistent prefix cache.

## Parallelism Integration

### Single Device

One `DeviceGraphOrchestrator` owns one local prefix cache for its KV/state payload. The RAM tier is the durable in-process source of truth; the device hot tier, if enabled, is an accelerator for recently reused blocks.

### LOCAL TP (`RankOrchestrator`)

Each child `DeviceGraphOrchestrator` stores the local shard for its device. All children use the same token hashes but different `topology_fingerprint` values. In LOCAL TP, each child can have its own device hot tier, while the RAM budget can either be per child or shared by the parent `RankOrchestrator` depending on the final allocator design.

Lookup must choose the minimum matched block count across children. If device 0 has 20 blocks and device 1 has 18, the rank may restore only 18 blocks.

### GLOBAL / NODE_LOCAL TP

Every rank receives the same prompt tokens through existing server MPI coordination. Each rank performs local lookup, then ranks run `MPI_Allreduce(MIN)` over matched block counts. All ranks restore and run the same suffix positions.

If a rank lacks a block that another rank has, the minimum rule prevents divergence.

### Pipeline Parallelism

Each PP stage stores only its local layer payload. Prefix lookup is still based on global tokens and a stage-specific topology fingerprint. The matched count must be coordinated across all PP stages before any suffix execution.

### MoE Expert Parallelism and Rebalancing

MoE does not add recurrent prefix state. Router outputs and expert FFN results are deterministic functions of hidden state and weights. However, runtime expert placement can affect execution path and floating-point reduction order.

Rules:

- Include expert mask, ownership, replica set, and placement epoch in `moe_fingerprint`.
- When `MoERebalanceController` changes placement, increment a prefix-cache MoE epoch or invalidate MoE-prefixed entries for that runner.
- In `OBSERVE` mode, placement does not change, so reuse is safe under the same fingerprint.
- In `DYNAMIC` mode, reuse is safe only while the placement epoch matches.
- Do not replay or synthesize decode histograms from prefix hits. The histogram is runtime telemetry, not model state.

## Configuration

### CLI

```text
--prefix-cache                         Enable prefix caching (default: off)
--prefix-cache-storage <mode>          ram|device|tiered (default: tiered)
--prefix-cache-ram-budget-mb <MB>      RAM capacity budget per local runner/rank (default: 4096)
--prefix-cache-device-budget-mb <MB>   Optional device hot-tier budget (default: 256, 0 disables)
--prefix-cache-block-size <N>          Complete tokens per block (default: 64)
--prefix-cache-terminal-logits <mode>  off|auto|always (default: auto)
--prefix-cache-hybrid <mode>           off|on|auto (default: auto)
--prefix-cache-moe-policy <mode>       disabled|placement-fingerprint|invalidate-on-rebalance
```

Recommended defaults after implementation:

- `--prefix-cache` remains opt-in.
- `--prefix-cache-storage tiered` means RAM primary plus optional device hot tier.
- `--prefix-cache-ram-budget-mb` is the main capacity knob and should be large enough to hold useful shared prompts on consumer GPUs.
- `--prefix-cache-device-budget-mb 0` is valid and should still preserve full prefix-cache functionality through RAM restore.
- `--prefix-cache-hybrid auto` enables hybrid only when hybrid snapshot APIs are implemented for the active backend.
- `--prefix-cache-terminal-logits auto` stores terminal logits only when they fit within a small fraction of the cache budget.
- `--prefix-cache-moe-policy placement-fingerprint` is the default for Qwen 3.5 MoE.

### Environment Variables

Environment variables are optional overrides, parsed through `debugEnv()` or config setup rather than hot-path `std::getenv` calls.

```text
LLAMINAR_PREFIX_CACHE=1
LLAMINAR_PREFIX_CACHE_STORAGE=tiered
LLAMINAR_PREFIX_CACHE_RAM_BUDGET_MB=4096
LLAMINAR_PREFIX_CACHE_DEVICE_BUDGET_MB=256
LLAMINAR_PREFIX_CACHE_BLOCK_SIZE=64
LLAMINAR_PREFIX_CACHE_TERMINAL_LOGITS=auto
LLAMINAR_PREFIX_CACHE_HYBRID=auto
LLAMINAR_PREFIX_CACHE_MOE_POLICY=placement-fingerprint
```

### Reporting

`--dry-run` currently exits before model load. It can report requested prefix-cache settings, but exact capacity requires model metadata and parallelism resolution.

Accurate reporting should happen after model metadata is available:

```text
Prefix Cache Configuration:
  Enabled:              yes
  Model Class:          qwen35_moe_hybrid
  Device:               cuda:0
    Storage:              tiered
    RAM Budget:           8192 MB
    Device Hot Budget:    512 MB
  Block Size:           64 tokens
  FA KV Layers:         9
  GDN Layers:           27
  KV Payload/Block:     144 MB
  GDN Payload/Block:    18 MB
  Terminal Logits:      auto
    RAM Blocks Available: 50
    Device Hot Blocks:    3
    RAM Capacity:         3200 tokens
```

The numbers above are illustrative; final values should come from actual metadata and selected precision.

## Memory Budgeting

The dense KV formula is:

```text
bytes_per_block = fa_kv_layers * block_size * local_kv_heads * head_dim *
                  (bytes_per_k + bytes_per_v)
```

For hybrid models:

```text
bytes_per_block = fa_kv_bytes_per_block + gdn_snapshot_bytes_per_block
```

GDN snapshot bytes are approximately:

```text
sum over GDN layers:
    recurrence_state_floats * 4 + conv_state_floats * 4 + device_kernel_state_bytes
```

For MoE variants, expert weights are not stored in prefix blocks. Only the MoE placement/version fingerprint is stored. Terminal logits, if enabled, add `vocab_size * sizeof(float)` per terminal block, or local vocab size per TP shard.

RAM and device memory must be budgeted separately:

- The RAM budget is the primary capacity budget. If it cannot hold at least one complete block for the active model/backend, initialization should warn and disable prefix caching rather than fail inference.
- The device hot-tier budget is optional. If it cannot hold one complete block, disable only the device tier and continue with RAM-backed caching.
- `MemoryPlanner` should subtract only the device hot-tier budget from available VRAM/HBM. The RAM tier should be tracked by prefix-cache allocation itself and reported separately from KV cache, workspace, and graph-capture budgets.
- For CUDA/ROCm, prefer pinned host RAM for small and medium RAM budgets because restores become H2D transfers on the active stream. For very large budgets, use a capped pinned pool plus pageable backing storage to avoid starving the OS and driver.

The critical production rule is: **a valid prefix-cache hit must not require spare VRAM beyond the request-local KV/GDN state that inference already needs**. VRAM improves hit latency; RAM provides useful capacity.

## Implementation Phases

### Phase 1: Hashing, Keys, LRU, Storage Interfaces, and Layout Metadata

Goal: Build cache metadata without touching inference.

Deliverables:

- `BlockHash` and hash-chain hasher.
- `PrefixCacheKey` with model/runtime/topology/hybrid/MoE fingerprints.
- `PrefixPayloadLayout` and size estimator.
- `PrefixStateBlock` metadata.
- `PrefixStorageTier`, `PrefixBlockHandle`, and `IPrefixStorageBackend` interfaces.
- `LRUQueue` with O(1) insert/remove/touch.
- Unit tests for hash determinism, chain dependency, partial-block behavior, key equality, LRU ordering, and backend allocation failure behavior.

### Phase 2: Dense KV Logical Block I/O

Goal: Add safe import/export for request-local ring KV caches.

Deliverables:

- `IKVCache::blockMetadata`, `exportLogicalBlock`, `importLogicalBlock`.
- CPU implementation for `POSITION_MAJOR` and `HEAD_MAJOR`.
- CUDA and ROCm implementations using async device copies where possible.
- Tests for ring wrap, sharded KV heads, Q16_1 head-major layout, TQ formats, and stream synchronization.

### Phase 3: Dense RAM-Backed Prefix Cache End to End

Goal: Reuse prefixes for dense full-attention models with RAM as the primary cache capacity tier.

Deliverables:

- `RamPrefixStorageBackend` with hard memory budget, LRU, stats, populate, and harvest.
- `PrefixStateCache` manager that indexes blocks by key and stores payloads through `IPrefixStorageBackend`.
- `OrchestrationConfig` / `CliSpec` flags.
- `OrchestrationRunner` lookup, rank coordination, RAM-to-request-state restore, suffix prefill, harvest.
- `DeviceGraphOrchestrator` populate/harvest helpers.
- Terminal logits policy with final-block fallback.
- Dense integration test: same prompt twice, second request restores complete blocks from RAM and produces identical tokens.
- Stress test: RAM budget holds multiple shared prompts without VRAM growth.

### Phase 4: Optional Device Hot Tier

Goal: Reduce hit latency for very hot prefixes without making VRAM the main capacity tier.

Deliverables:

- `DeviceHotPrefixStorageBackend` with a small independent budget and LRU.
- Promotion policy from RAM to device hot tier after hit-count or recency threshold.
- Demotion/eviction policy that leaves the RAM copy intact.
- Restore path that prefers device-hot blocks and falls back to RAM blocks.
- Tests proving `--prefix-cache-device-budget-mb 0` still gives correct RAM-backed cache hits.
- Tests proving device hot-tier eviction does not invalidate the RAM source-of-truth entry.

### Phase 5: Hybrid GDN Snapshot API

Goal: Make Qwen 3.5 hybrid state cacheable.

Deliverables:

- `IHybridKVCache` prefix-state metadata/export/import APIs.
- CPU implementation copying `HybridGDNLayerState` recurrence and conv vectors.
- CUDA and ROCm implementation copying device-resident short-conv and recurrence kernel state.
- Kernel-level state export/import where needed for `ITensorShortConvolution` and `ITensorGatedDeltaNet`.
- Tests that restore state after N tokens and produce identical GDN outputs for suffix tokens.

### Phase 6: Hybrid Prefix Cache End to End

Goal: Reuse prefixes for Qwen 3.5 FA+GDN models.

Deliverables:

- Hybrid payload support in RAM-backed `PrefixStateCache`.
- FA-layer-only KV export/import using `HybridLayerMap`.
- GDN state snapshot per cached block.
- Hybrid integration test: shared system prompt, differing user suffixes, identical outputs versus no-cache.
- Full-hit test with terminal logits or final-block recompute fallback.

### Phase 7: Qwen 3.5 MoE Safety and Rebalance Integration

Goal: Support Qwen 3.5 MoE variants without stale expert-placement assumptions.

Deliverables:

- `moe_fingerprint` derived from expert masks, placement, replica set, and rebalance epoch.
- Prefix-cache invalidation or epoch bump when `MoERebalanceController` applies new placement.
- Tests for `OFF`, `OBSERVE`, and `DYNAMIC` rebalance modes.
- MoE integration test proving cache hit results match no-cache under stable placement.
- Rebalance test proving old prefix entries are not reused after placement changes unless fingerprint matches.

### Phase 8: Multi-Device and MPI Hardening

Goal: Make all supported parallelism modes deterministic and safe.

Deliverables:

- LOCAL TP minimum matched-block coordination across child device runners.
- GLOBAL/NODE_LOCAL TP `MPI_Allreduce(MIN)` matched-block coordination.
- PP stage coordination so every stage restores the same prefix length.
- MPI worker loop integration for lookup, populate, harvest, and cache stats.
- Multi-rank tests for cache hit/miss asymmetry and fallback to common prefix length.

### Phase 9: Observability and Production Controls

Goal: Make behavior visible and tunable in server deployments.

Deliverables:

- `PrefixCacheStats`: lookups, hits, matched blocks, tokens saved, stores, RAM evictions, device-hot promotions, device-hot evictions, memory usage by tier, terminal-logit hits, hybrid-state bytes.
- INFO-level per-request summary when enabled.
- Optional response metadata or headers in server mode.
- `--show-prefix-cache-stats` or periodic stats logging.
- Benchmark comparing repeated system prompts with and without prefix cache.

## Test Matrix

| Test Area | Required Cases |
|-----------|----------------|
| Hashing | Same tokens same chain, same block tokens at different positions different key, trailing partial block ignored |
| LRU | Free block allocation, touch promotion, eviction skips referenced blocks, no leaks |
| Dense KV I/O | CPU/CUDA/ROCm, ring wrap, head-major Q16_1, position-major FP16/Q8_1/TQ, sharded KV |
| RAM storage | Budget enforcement, LRU eviction, pinned/pageable fallback, RAM-to-device restore, no VRAM growth when device tier disabled |
| Device hot tier | Promotion, hit preference, hot-tier eviction, RAM source-of-truth preservation, zero-budget disabled behavior |
| Dense E2E | Identical prompt, shared prefix with differing suffix, full-hit terminal logits, final-block recompute fallback, RAM-backed second-request hit |
| Hybrid state | Export/import GDN state after block N, suffix output matches no-cache, reset still clears request-local state |
| Qwen 3.5 E2E | FA+GDN cache hit with suffix, full hit, cache miss after runtime fingerprint change |
| Qwen 3.5 MoE | Stable placement hit, rebalance invalidation, replica fingerprint change, no histogram replay |
| Parallelism | LOCAL TP min match, GLOBAL TP allreduce min match, PP stage local payloads, MPI worker loop |
| Memory | RAM budget holds at least one block, insufficient RAM disables cache with warning, insufficient device budget disables only hot tier, no unbounded growth after many requests |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hybrid cache restores KV but not device GDN kernel state | Wrong outputs | Make hybrid support depend on explicit `IHybridKVCache` and kernel state export/import tests |
| Full-prefix hit has no logits | Decode starts from stale/missing logits | Store terminal logits when practical; otherwise recompute final block |
| MoE rebalance changes execution path | Subtle numerical drift or wrong expert ownership | Include placement epoch/fingerprint; invalidate on rebalance |
| Async GPU copy races suffix execution | Corrupted or incomplete state | Use active backend stream and synchronize before suffix graph execution |
| Ring wrap breaks logical block ordering | Wrong attention history | Implement logical import/export in each KV cache and test wrapped cases |
| Cache memory overwhelms large hybrid models | Startup OOM or tiny useful capacity | Treat RAM and device budgets as hard caps; disable cache only if RAM cannot hold one block; disable hot tier independently if VRAM cannot hold one block |
| Pinned RAM budget starves OS or GPU driver | System instability or allocation failures | Cap pinned pool separately, fall back to pageable RAM, report pinned/pageable split |
| MPI ranks match different prefix lengths | Rank divergence or collective mismatch | Coordinate common matched block count with min reduction before populate |
| Cache key misses important runtime feature | Incorrect reuse | Centralize fingerprint construction and test changes in precision/layout/topology/hybrid/MoE placement |

## Success Criteria

1. Dense full-attention repeated-prefix requests skip complete cached blocks and produce identical tokens to no-cache.
2. Qwen 3.5 hybrid repeated-prefix requests restore FA KV plus GDN state and produce identical tokens to no-cache.
3. Qwen 3.5 MoE repeated-prefix requests are safe under stable expert placement and do not reuse stale entries after rebalance.
4. Full-prefix hits either restore terminal logits or deliberately recompute the final block.
5. Prefix-cache disabled remains the default and leaves existing parity tests unchanged.
6. RAM-backed cache hits work with `--prefix-cache-device-budget-mb 0`, proving useful capacity does not depend on spare VRAM.
7. Cache memory stays within configured RAM and device budgets after long server runs.
8. LOCAL TP, GLOBAL/NODE_LOCAL TP, and PP modes restore a common prefix length across all participants.

## Future Extensions

- Multi-tenant cache salting.
- Disk-backed cache for large persistent prefix stores.
- Warm-up API for known system prompts.
- Adaptive block size per model or prompt distribution.
- Cross-process shared cache for multi-worker server mode.
- LoRA/adaptor-aware cache keys.
