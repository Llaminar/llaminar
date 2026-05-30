# Unified KV Prefix Cache And MTP Phased Implementation Plan

This plan implements one shared prefix-state subsystem for two features:

1. Cross-request KV prefix caching.
2. In-request MTP speculative decoding and rollback.

The central design decision is that both features use the same state contract. Full-attention KV, hybrid GDN recurrence/conv state, MTP's shifted KV cache, terminal hidden/logit rows, and MoE placement fingerprints all flow through typed snapshot, import, restore, and invalidation APIs. The live request state is still cleared at prompt boundaries; persistent prefix blocks live outside request-local `InferenceState` and can repopulate it when a request hits the cache.

The phases below are ordered to keep correctness ahead of performance. Each phase includes goal, implementation details, files, tests, and exit criteria inline.

## Guiding Constraints

- Graphs remain per-device and symmetric. Do not introduce nested multi-device subgraphs for prefix caching or MTP.
- Prefix blocks are runtime state, not model weights and not activation arena state. Persistent prefix storage lives outside `BufferArena`; `BufferArena` owns only request-local staging buffers.
- RAM is the primary correctness tier. Device memory is a hot promotion tier, and disk is durable backing. Device-tier eviction must never invalidate RAM/disk source blocks.
- GDN rollback is snapshot/restore/replay first. GDN recurrence and short-conv state mutate in place and cannot be treated like append-only KV rows.
- MoE decode histograms are telemetry, not prefix payload. MoE placement, masks, replicas, and rebalance epoch are key material.
- MTP support starts with Qwen3.5/Qwen3.6 D=1 shared-embedding models and stays config-gated until greedy parity is stable.
- All config parsing follows the V2 pipeline: `OrchestrationConfig` -> `RuntimeConfig` / `RankExecutionPlan` -> `GraphConfig` -> graph builders and runners.

## Phase 0: State Probe And Scope Lock

### Goal

Build a test-backed inventory of every state item that changes during prefill/decode and is reset by `clearCache()`. This phase proves what must be persisted, what must be rolled back, and what should remain transient.

### Implementation Details

Add an integration probe target under:

```text
tests/v2/integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp
```

The probe should run representative flows without implementing the full prefix-cache manager yet:

1. `prefill(prompt_prefix)`.
2. Optional `decodeStep()`.
3. Capture state inventory.
4. `clearCache()`.
5. Confirm cleared state.
6. Prototype restore with whatever low-level hooks exist in the current phase.
7. Run suffix and compare logits/tokens against a no-clear baseline.

Classify state with this contract:

| State | Owner Today | Prefix Payload | MTP Rollback Payload | Notes |
|------|-------------|----------------|----------------------|-------|
| Full-attention K/V rows | `IKVCache` implementations | Yes | Yes | Export/import in logical token order only. |
| Ring head/count | `IKVCache` implementations | Metadata | Metadata | Restore must make `get_cached_tokens()` and graph replay head buffers consistent. |
| GDN recurrence state | `IHybridKVCache`, `HybridGDNLayerState`, GPU GDN kernel | Yes | Yes | FP32, in-place, not append-only. |
| GDN short-conv state | `IHybridKVCache`, `HybridGDNLayerState`, GPU conv kernel | Yes | Yes | Same restore contract as recurrence state. |
| MTP depth KV | New `mtp_kv_caches` | Yes when MTP enabled | Yes | Separate FA cache per MTP depth. Current target depth is 1. |
| Terminal hidden row | `InferenceState::hidden` | Yes when MTP enabled | Useful | Required to continue MTP after a full prefix hit. |
| Terminal logits row | `InferenceState::logits` | Optional for non-MTP, required for full MTP hit | Useful | Preserves first-token semantics. |
| `positions` / `sequence_lengths` | `DeviceGraphOrchestrator::InferenceState` | Metadata | Metadata | Populate sets these to the restored token count. |
| `prefill_logits_ready_` | `OrchestrationRunner` | Derived | Derived | Set only when terminal logits are restored or recomputed. |
| MoE placement | `MoERuntimeTable`, rebalance controller, stage masks | Key only | Key only | Placement changes must miss or invalidate. |
| MoE decode histogram | `MoERuntimeTable`, `DecodeExpertHistogram` | No | No | Telemetry only. |
| MoE route/grouped scratch | MoE runtime table and stages | No | No | Transient capacity and scratch. |
| Graph cache entries | `ForwardExecutionEngine`, `LayerGraphCache` | No | No | Rebuilt or reset around request boundaries. |

Dense probe:

- Use a small Qwen2.5 model.
- Record FA KV token counts, ring head/count, position values, sequence lengths, prefill logits readiness, and graph-cache reset behavior before and after `clearCache()`.
- Prototype FA KV restore if the low-level API is available in the branch.

Hybrid probe:

- Use Qwen3.5/Qwen3.6 hybrid models when available.
- Record FA KV plus GDN recurrence/conv hashes.
- Verify `clearCache()` zeroes host-side GDN vectors and GPU kernel state.

MoE probe:

- Record placement bank, active epoch, expert masks, local compute masks, and replica role values.
- Confirm decode histogram changes do not need to be restored.
- Confirm placement epoch changes alter the MoE fingerprint or force cache bypass.

MTP probe:

- Inspect `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf` and `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` when present.
- Enumerate `mtp.*` metadata and tensors.
- Confirm D=1 and shared embeddings for current target files.
- Define the shifted-cache invariant used in Phase 5.

### Files

- `tests/v2/integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp`
- `tests/v2/integration/prefix_cache/CMakeLists.txt` or the relevant test registration file.
- Existing runner/cache files are read-only in this phase unless a tiny probe-only accessor is unavoidable.

### Tests

Add these tests, model-gated where needed:

| Test | Purpose |
|------|---------|
| `DenseQwen25_ResetStateInventory` | Capture dense KV, positions, logits readiness, and graph-cache reset behavior. |
| `DenseQwen25_PrototypeKVRestoreMatchesSuffix` | Export a dense prefix, clear, import, run suffix, compare to baseline. |
| `HybridQwen35_GDNStateInventory` | Verify GDN state changes during execution and zeros after clear. |
| `HybridQwen35_PrototypeHybridRestoreMatchesSuffix` | Restore FA KV plus GDN state and compare suffix logits. |
| `MoE_RuntimeFingerprintChangesOnPlacementEpoch` | Confirm MoE placement changes alter fingerprint material. |
| `MTP_Qwen36_MetadataAndTensorInventory` | Enumerate Qwen3.6 MTP metadata/tensors, skipping when files are absent. |

### Exit Criteria

- A test log or table identifies all request-state payload and transient state.
- Dense and hybrid restore blockers are known before broad API work begins.
- MTP tensor names and metadata expectations are recorded against actual GGUF files.
- MoE histogram state is explicitly excluded from prefix payload.

## Phase 1: Config, Feature Gates, And Fingerprints

### Goal

Add disabled-by-default configuration and stable key material so prefix cache and MTP behavior can be enabled without changing normal inference.

### Implementation Details

Add explicit structs instead of sprinkling feature flags across call sites:

```cpp
enum class PrefixCacheStorageMode
{
    Disabled,
    Ram,
    Device,
    Tiered,
};

enum class PrefixCacheTerminalStateMode
{
    Off,
    Auto,
    Always,
};

enum class PrefixCacheMoEPolicy
{
    Disabled,
    PlacementFingerprint,
    InvalidateOnRebalance,
};

struct PrefixCacheRuntimeConfig
{
    bool enabled = false;
    PrefixCacheStorageMode storage_mode = PrefixCacheStorageMode::Tiered;
    int block_size = 64;
    size_t ram_budget_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    size_t device_budget_bytes = 256ull * 1024ull * 1024ull;
    size_t disk_budget_bytes = 0;
    std::string disk_dir;
    PrefixCacheTerminalStateMode terminal_state = PrefixCacheTerminalStateMode::Auto;
    PrefixCacheMoEPolicy moe_policy = PrefixCacheMoEPolicy::PlacementFingerprint;
};

enum class MTPVerifyMode
{
    Greedy,
    SpeculativeSampling,
};

struct MTPRuntimeConfig
{
    bool enabled = false;
    int draft_tokens = 1;
    MTPVerifyMode verify_mode = MTPVerifyMode::Greedy;
    bool require_terminal_hidden_for_full_hit = true;
};
```

Wire these through the V2 configuration pipeline:

1. `OrchestrationConfig`: raw CLI/YAML config.
2. `RuntimeConfig`: parsed values copied through plan building.
3. `RankExecutionPlan`: per-rank runtime contract.
4. `GraphConfig`: model-specific graph config and cache layout decisions.
5. `InferenceRunnerFactory`: runner construction and feature enablement.

Add CLI flags in `OrchestrationConfigParser::buildSpec()` under `Prefix Cache` and `MTP` help categories:

```text
--prefix-cache
--prefix-cache-storage ram|device|tiered
--prefix-cache-block-size <n>
--prefix-cache-vram-budget-mb <mb>
--prefix-cache-ram-budget-mb <mb>
--prefix-cache-disk-budget-mb <mb>
--prefix-cache-disk-dir <path>
--prefix-cache-terminal-state off|auto|always
--prefix-cache-moe-policy disabled|placement-fingerprint|invalidate-on-rebalance
--mtp
--mtp-draft-tokens <n>
--mtp-verify-mode greedy|speculative-sampling
```

Implement fingerprints centrally:

```text
src/v2/execution/prefix_cache/PrefixCacheFingerprint.h
src/v2/execution/prefix_cache/PrefixCacheFingerprint.cpp
```

The fingerprint should be composed from named parts:

```cpp
struct PrefixFingerprintParts
{
    uint64_t model = 0;
    uint64_t tokenizer = 0;
    uint64_t runtime = 0;
    uint64_t topology = 0;
    uint64_t hybrid = 0;
    uint64_t moe = 0;
    uint64_t mtp = 0;
};
```

Fingerprint material:

- `model`: architecture, GGUF metadata fingerprint, tensor directory fingerprint, vocab size, tied/separate LM head mode.
- `tokenizer`: tokenizer model, added tokens, chat-template identity, template override identity.
- `runtime`: activation precision, KV precision, KV layout, RoPE-on-read, TurboQuant mode, Q16 scales, fused attention backend, partial RoPE factor.
- `topology`: TP/PP degree, rank/participant id, device id, local KV-head start/count, layer range, vocab shard.
- `hybrid`: layer types, FA/GDN counts, GDN state size, group count, time-step rank, inner size, conv kernel, local head assignment.
- `moe`: expert count, top-k, expert mode, placement epoch, active bank, local compute masks, replica roles, rebalance domain id.
- `mtp`: enabled flag, depth, draft token count, shared/dedicated embedding mode, MTP tensor-name fingerprint, MTP weight shapes.

If `PrefixCacheMoEPolicy::Disabled` is active for a MoE model, bypass prefix caching rather than hashing a zero MoE part.

Environment overrides must go through `DebugEnv`. Do not add local `getenv()` calls in prefix-cache or MTP runtime code.

### Files

- `src/v2/config/OrchestrationConfig.h`
- `src/v2/config/OrchestrationConfigParser.cpp`
- `src/v2/execution/config/RuntimeConfig.h`
- `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`
- `src/v2/models/GraphTypes.h`
- `src/v2/execution/factory/InferenceRunnerFactory.*`
- `src/v2/execution/prefix_cache/PrefixCacheFingerprint.h/.cpp`
- `src/v2/utils/DebugEnv.h` if env overrides are added.

### Tests

- Parser tests for every new CLI flag, including invalid enum values.
- Runtime config copy tests that prove values survive `OrchestrationConfig` -> `RuntimeConfig` -> `GraphConfig`.
- Fingerprint tests that alter one field at a time and confirm the final key changes.
- MoE policy tests: disabled bypasses, placement-fingerprint keys, rebalance epoch changes key material.

### Exit Criteria

- Normal inference behavior is unchanged when all new flags are off.
- `--dry-run` / placement logging can report resolved prefix/MTP settings after model metadata is known.
- Fingerprints are deterministic across runs for identical config and model inputs.

## Phase 2: IKVCache Logical Snapshot Contract

### Goal

Expose logical KV export/import/truncate through `IKVCache` so prefix storage and MTP rollback never need to know raw backend ring layout.

### Implementation Details

Add narrow default-false APIs to `IKVCache`:

```cpp
struct KVCacheLogicalBlockDescriptor
{
    int layer = 0;               // Global layer index as graph stages see it.
    int seq_idx = 0;
    int logical_token_start = 0; // Oldest-to-newest logical index.
    int token_count = 0;
    void *stream = nullptr;      // cudaStream_t, hipStream_t, or nullptr.
};

struct KVCacheLogicalBlockLayout
{
    ActivationPrecision k_precision = ActivationPrecision::FP32;
    ActivationPrecision v_precision = ActivationPrecision::FP32;
    TensorLayout layout = TensorLayout::KV_POS_HEAD_DIM;
    int local_kv_heads = 0;
    int kv_head_start = 0;
    int head_dim = 0;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    bool device_resident = false;
};

struct KVCacheSequenceState
{
    int cached_tokens = 0;
    int implementation_head = 0;
    bool wrapped = false;
};

virtual KVCacheLogicalBlockLayout logicalBlockLayout(
    int global_layer,
    int token_count) const;

virtual KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const;

virtual bool exportLogicalBlock(
    const KVCacheLogicalBlockDescriptor &desc,
    void *dst_k,
    void *dst_v) const;

virtual bool importLogicalBlock(
    const KVCacheLogicalBlockDescriptor &desc,
    const void *src_k,
    const void *src_v);

virtual bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr);
```

Semantics:

- `logical_token_start` and `token_count` are logical sequence positions, not raw ring rows.
- `PrefixStateCache` never calculates raw ring offsets.
- `global_layer` uses graph-stage numbering. Hybrid caches do internal FA-index remapping.
- Export to RAM may do D2H; export to device-hot tier may do D2D. GPU paths should honor `stream`.
- Import must set head/count metadata before suffix execution.
- Import/truncate/clear must invalidate converted KV and RoPE-on-read shadow state.
- `truncateSequence()` is logical. Dense caches can update counts; hybrid rollback still uses full snapshot restore.

Backend implementation:

- `CPURingKVCache`: copy logical rows oldest-to-newest. CPU documentation treats `head` as oldest valid token.
- CUDA and ROCm bases: use existing hooks `entryHead`, `entryCount`, `setEntryHead`, `setEntryCount`, `resetEntry`, `onClearSequence`, `onEviction`, and `onAdvanceComplete`.
- Add `onRestoreSequence(layer, seq_idx)` if needed for shadow invalidation.
- CUDA/ROCm import should initially restore into an unwrapped layout starting at row 0 and set next-write head to `cached_tokens % max_seq_len`.
- TQ and Q16_1 paths must preserve native cache precision and layout. Do not dequantize for persistent payload unless a backend lacks native I/O in the first CPU-only slice.

The GPU RoPE shadow hazard is a blocker: converted/RoPE shadow caches can track count while missing content/head changes. Import, truncate, clear, and eviction must invalidate shadows by layer/sequence.

### Files

- `src/v2/kernels/IKVCache.h`
- `src/v2/kernels/cpu/CPURingKVCache.h/.cpp`
- `src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h/.cpp`
- `src/v2/kernels/cuda/kvcache/CUDARingKVCache.h/.cu`
- `src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.h/.cu`
- `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h/.cpp`
- `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.h/.cpp`
- `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.h/.hip`

### Tests

Add `tests/v2/unit/kernels/Test__IKVCacheLogicalBlockIO.cpp` with backend coverage where available:

- Empty export/import contract.
- Non-wrapped export.
- Wrapped export.
- Import into empty cache.
- Import over non-empty cache after clear.
- Truncate to zero, current length, shorter length, and invalid longer length.
- Q16_1 head-major layout.
- TQ asymmetric K/V layout.
- Sharded KV heads with nonzero `kv_head_start`.
- GPU stream import/export and synchronization.

### Exit Criteria

- CPU logical export/import/truncate passes for the formats used in unit tests.
- CUDA/ROCm either pass equivalent tests or cleanly return unsupported while feature gates keep prefix caching disabled on those backends.
- Import restores `get_cached_tokens()` and graph-captured append head behavior correctly.

## Phase 3: Prefix Store, Payload Layout, And Arena Staging

### Goal

Create the storage manager and request-local staging needed for dense prefix caching without wiring it into generation yet.

### Implementation Details

Add module layout:

```text
src/v2/execution/prefix_cache/
  BlockHash.h/.cpp
  PrefixCacheConfig.h
  PrefixCacheKey.h/.cpp
  PrefixPayloadLayout.h/.cpp
  PrefixStateBlock.h
  PrefixStateCache.h/.cpp
  PrefixStorageBackend.h
  RamPrefixStorageBackend.h/.cpp
  DeviceHotPrefixStorageBackend.h/.cpp
  DiskPrefixStorageBackend.h/.cpp
  PrefixCacheStats.h
  PrefixCacheFingerprint.h/.cpp
  PrefixStateSnapshot.h
```

Derive `PrefixPayloadLayout` once after `GraphConfig` and KV cache creation:

```cpp
struct PrefixPayloadLayout
{
    DeviceId device;
    int block_size = 64;
    int first_layer_index = 0;
    int total_layers = 0;
    int fa_layers = 0;
    int gdn_layers = 0;
    int local_kv_heads = 0;
    int kv_head_start = 0;
    int head_dim = 0;
    ActivationPrecision k_precision = ActivationPrecision::FP32;
    ActivationPrecision v_precision = ActivationPrecision::FP32;
    TensorLayout kv_layout = TensorLayout::KV_POS_HEAD_DIM;
    size_t bytes_per_fa_layer_k = 0;
    size_t bytes_per_fa_layer_v = 0;
    size_t hybrid_state_bytes = 0;
    size_t mtp_kv_bytes = 0;
    size_t terminal_hidden_bytes = 0;
    size_t terminal_logits_bytes = 0;
    bool includes_hybrid_state = false;
    bool includes_mtp_state = false;
    bool includes_terminal_hidden = false;
    bool includes_terminal_logits = false;
};
```

Dense KV bytes are:

```text
sum over local FA layers:
  block_size * local_kv_heads * head_dim * (bytes_per_k + bytes_per_v)
```

Add tier backend interfaces:

```cpp
enum class PrefixStorageTier
{
    DeviceHot,
    Ram,
    Disk,
};

struct PrefixBlockHandle
{
    PrefixCacheKey key;
    PrefixStorageTier tier = PrefixStorageTier::Ram;
    PrefixPayloadLayout layout;
    void *kv_payload = nullptr;
    void *hybrid_payload = nullptr;
    void *mtp_payload = nullptr;
    void *terminal_hidden = nullptr;
    void *terminal_logits = nullptr;
    size_t total_bytes = 0;
};

class IPrefixStorageBackend
{
public:
    virtual ~IPrefixStorageBackend() = default;
    virtual bool canStore(size_t bytes) const = 0;
    virtual PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                       const PrefixPayloadLayout &layout) = 0;
    virtual bool release(const PrefixBlockHandle &handle) = 0;
    virtual bool hydrateToRam(const PrefixBlockHandle &handle,
                              PrefixBlockHandle *ram_handle) = 0;
};
```

`PrefixStateCache` owns the hash map, block chains, LRU, ref counts, and tier promotion policy. Storage backends own memory/files only.

Tier rules:

- RAM is source of truth for early correctness.
- Device-hot blocks are promotions of RAM blocks.
- Disk blocks hydrate into RAM before request-state import.
- Device-hot eviction drops only the hot handle.
- If RAM cannot hold one complete block, disable prefix caching with a warning.
- If device-hot cannot hold one block, disable only the device tier.

Add `BufferArena` staging ids for request-local restore/harvest:

```cpp
PREFIX_K_STAGING,
PREFIX_V_STAGING,
PREFIX_HYBRID_STATE_STAGING,
PREFIX_MTP_K_STAGING,
PREFIX_MTP_V_STAGING,
PREFIX_TERMINAL_HIDDEN,
PREFIX_TERMINAL_LOGITS,
```

Update `bufferIdName()` and `BufferArena::bufferNameToId()`. Persistent blocks remain outside arena storage.

Disk format:

```text
<disk_dir>/
  manifest.json
  blocks/
    <key-hex>.meta.json
    <key-hex>.kv.bin
    <key-hex>.hybrid.bin
    <key-hex>.mtp.bin
    <key-hex>.terminal_hidden.bin
    <key-hex>.terminal_logits.bin
```

Disk metadata includes format version, key fields, payload layout, token count, block index, precision/layout, byte lengths, checksums, and all fingerprints. Disk write failure should record stats and keep the RAM block; it should not fail inference.

### Files

- `src/v2/execution/prefix_cache/*`
- `src/v2/memory/BufferId.h`
- `src/v2/memory/BufferArena.h/.cpp`
- `src/v2/CMakeLists.txt`

### Tests

- `tests/v2/unit/prefix_cache/Test__PrefixBlockHash.cpp`
- `tests/v2/unit/prefix_cache/Test__PrefixCacheKey.cpp`
- `tests/v2/unit/prefix_cache/Test__PrefixStateCacheLRU.cpp`
- `tests/v2/unit/prefix_cache/Test__RamPrefixStorageBackend.cpp`
- `tests/v2/unit/prefix_cache/Test__DiskPrefixStorageBackend.cpp`

### Exit Criteria

- Prefix blocks can be allocated, inserted, found, evicted, and hydrated in unit tests without runner integration.
- RAM LRU accounting is deterministic.
- Disk round-trip validates checksums and rejects malformed metadata.
- Arena registration supports the new staging ids without changing existing graph behavior.

## Phase 4: Dense Prefix Cache End To End

### Goal

Enable RAM-backed dense prefix caching for non-MTP, non-hybrid models first. This proves the prefix lookup, populate, suffix forward, terminal logits, and harvest control flow.

### Implementation Details

Add persistent prefix-cache ownership outside `InferenceState`:

```cpp
std::shared_ptr<PrefixStateCache> prefix_cache_;
PrefixPayloadLayout prefix_layout_;
PrefixCacheStats prefix_cache_stats_;
```

Add concrete `DeviceGraphOrchestrator` helpers before promoting them to `IInferenceRunner`:

```cpp
PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) const;
bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0);
bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count);
bool restorePrefixTerminalState(const PrefixLookupResult &hit);
PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const;
bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0);
```

`clear_cache()` continues to call `state_.clear()` and reset graph/stage dynamic state, but must not evict `prefix_cache_`.

Import order for a prefix hit:

1. Clear request-local state.
2. Import FA KV into `state_.kv_cache` or `state_.pp_kv_caches`.
3. Restore terminal hidden/logits when present.
4. Set `positions[seq_idx]` and `sequence_lengths[seq_idx]` to `hit.cached_tokens`.
5. Invalidate or refresh graph dynamic params before replay.

Update `OrchestrationRunner::prefill()` flow:

```cpp
bool OrchestrationRunner::prefill(const std::vector<int32_t> &prompt_tokens)
{
    if (!initialized_ || prompt_tokens.empty())
        return setError("Runner not initialized or prompt tokens are empty");

    broadcastPrefillToWorkers(prompt_tokens);

    PrefixLookupResult local_hit;
    if (prefix_cache_enabled_)
        local_hit = runner_->lookupPrefix(prompt_tokens);

    int matched_tokens = coordinateMinimumMatchedTokens(local_hit.cached_tokens);

    runner_->clear_cache();

    PrefixLookupResult common_hit = local_hit.clampedTo(matched_tokens);
    if (prefix_cache_enabled_ && matched_tokens > 0)
    {
        if (!runner_->populatePrefix(common_hit))
            matched_tokens = 0;
    }

    int suffix_start = matched_tokens;
    int suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;

    if (suffix_len > 0)
    {
        runner_->forward(prompt_tokens.data() + suffix_start, suffix_len);
        prefill_logits_ready_ = true;
    }
    else if (common_hit.has_terminal_logits)
    {
        runner_->restorePrefixTerminalState(common_hit);
        prefill_logits_ready_ = true;
    }
    else
    {
        matched_tokens = std::max(0, matched_tokens - prefix_block_size_);
        runner_->clear_cache();
        runner_->populatePrefix(common_hit.clampedTo(matched_tokens));
        suffix_start = matched_tokens;
        suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;
        runner_->forward(prompt_tokens.data() + suffix_start, suffix_len);
        prefill_logits_ready_ = true;
    }

    if (prefix_cache_enabled_)
        runner_->harvestPrefix(prompt_tokens, static_cast<int>(prompt_tokens.size()));

    return true;
}
```

Full-hit first-token semantics:

- If terminal logits are present, set `prefill_logits_ready_ = true` and skip refeeding the final prompt token.
- If terminal logits are absent, recompute the final block or final token before decode.
- Do not duplicate the final prompt token; doing so corrupts KV and GDN state.

### Files

- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h` after the concrete DGO helpers stabilize.
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp` only for passthrough/min coordination once single-device works.

### Tests

- Dense repeated prompt: no-cache vs RAM prefix cache logits/tokens match under greedy sampling.
- Shared prefix, different suffix: suffix logits match baseline.
- Full-hit with terminal logits: first generated token matches baseline and does not refeed prompt tail.
- Full-hit without terminal logits: final block recompute path matches baseline.
- Budget eviction: evicted block misses, non-evicted block hits.

### Exit Criteria

- Dense CPU path works end-to-end with RAM prefix cache.
- Dense GPU path is either supported or cleanly bypassed by backend support checks.
- Normal generation with prefix cache disabled remains unchanged.

## Phase 5: MTP Loading, Sidecar Graph, And Shifted Cache

### Goal

Load MTP weights and build the MTP sidecar execution path without changing decode behavior yet.

### Implementation Details

Add model-facing MTP weights:

```cpp
struct MTPDepthWeights
{
    TensorBase *fc = nullptr;
    TensorBase *pre_fc_norm_hidden = nullptr;
    TensorBase *pre_fc_norm_embedding = nullptr;
    TensorBase *final_norm = nullptr;
    LayerWeights fa_block;
};

struct MTPWeights
{
    int depth = 0;
    bool use_dedicated_embeddings = false;
    std::vector<MTPDepthWeights> depths;
};
```

Add `MTPWeights mtp;` to `ModelWeights`, plus prepared-weight bindings if graph builders need `PreparedWeightStore` refs.

Probe and support these Qwen3.5/Qwen3.6 tensor names first:

```text
mtp.fc.weight
mtp.pre_fc_norm_hidden.weight
mtp.pre_fc_norm_embedding.weight
mtp.norm.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.mlp.down_proj.weight
```

If `mtp.num_hidden_layers` metadata is absent but `mtp.fc.weight` exists, infer depth 1 and log the inference. If `--mtp` is explicit and required tensors are missing, fail loudly. If MTP is not requested, disable MTP and keep normal inference.

Add MTP buffers to `BufferId` and arena mapping:

```cpp
MTP_EMBEDDING,
MTP_NORM_HIDDEN,
MTP_NORM_EMBEDDING,
MTP_CONCAT,
MTP_PROJECTED,
MTP_HIDDEN,
MTP_Q_PROJ,
MTP_K_PROJ,
MTP_V_PROJ,
MTP_Q_ROPE,
MTP_K_ROPE,
MTP_ATTN_OUTPUT,
MTP_ATTN_PROJ,
MTP_GATE_PROJ,
MTP_UP_PROJ,
MTP_FFN_OUTPUT,
MTP_LOGITS,
```

Add request-local MTP cache state:

```cpp
std::vector<std::unique_ptr<IKVCache>> mtp_kv_caches;
std::shared_ptr<TensorBase> prefix_terminal_hidden;
std::shared_ptr<TensorBase> prefix_terminal_logits;
```

Build a sidecar graph method in Qwen35/Qwen36 graph code:

```cpp
ComputeGraph Qwen35Graph::buildMTPGraph(
    int depth_idx,
    const MTPDepthWeights &weights,
    const MTPForwardInput &input,
    MTPForwardOutput &output);
```

MTP graph sequence:

1. Embed draft token using shared embedding table unless future metadata requests dedicated embeddings.
2. RMSNorm main hidden with `mtp.pre_fc_norm_hidden.weight`.
3. RMSNorm draft embedding with `mtp.pre_fc_norm_embedding.weight`.
4. Concatenate `[norm_hidden ; norm_embedding]` into `MTP_CONCAT`.
5. Project with `mtp.fc.weight` into `MTP_PROJECTED`.
6. Run one full-attention block using existing Qwen35 FA stages: input norm, gated Q projection, Q/K norms, partial RoPE, MTP KV append/read, attention, attention output gate, Wo, residual, FFN norm, SwiGLU FFN, residual.
7. Apply `mtp.norm.weight`.
8. Run shared LM head into `MTP_LOGITS`.

Shifted-cache invariant after prompt length `N`:

```text
main KV/GDN state: state after tokens [0, N)
MTP depth-0 KV:    state after MTP inputs for pairs (hidden[i], token[i+1]) where i in [0, N-1)
terminal hidden:   main hidden row for token N-1
terminal logits:   logits predicting token N
```

Full prefix hits with MTP require terminal hidden and terminal logits. If terminal hidden is unavailable, reduce the hit by one block and recompute. Do not synthesize terminal hidden from logits.

### Files

- `src/v2/models/GraphTypes.h`
- `src/v2/models/qwen35/Qwen35GraphConfigBuilder.cpp`
- `src/v2/models/qwen35/Qwen35Graph.h/.cpp`
- Qwen36 graph/config files if separate from Qwen35.
- `src/v2/memory/BufferId.h`
- `src/v2/memory/BufferArena.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`

### Tests

- Model-gated MTP metadata and tensor inventory for Qwen3.6 dense and MoE files.
- MTP graph construction test with mocked or small fixture weights.
- MTP shifted-cache count probe: after prefill of `N`, MTP cache depth 0 has `max(0, N - 1)` logical tokens.
- MTP sidecar one-step logits smoke test against a Python/reference capture once available.

### Exit Criteria

- MTP weights load or explicitly disable with clear diagnostics.
- MTP sidecar graph can build without affecting normal graphs.
- MTP prefill cache invariant is test-backed.

## Phase 6: MTP Decode, Verification, And Rollback

### Goal

Enable greedy MTP speculative decoding behind `--mtp`, using the same snapshot/restore contract as prefix cache.

### Implementation Details

Extend `LMHeadStage::Params`:

```cpp
bool compute_all_positions = false;
```

Update LM-head row count:

```cpp
const int lm_m = params_.compute_all_positions ? params_.seq_len
               : (params_.seq_len > 1) ? 1
               : params_.seq_len;
const int lm_activation_offset = params_.compute_all_positions ? 0
                               : activationRowOffsetForLogits();
```

Update `estimatedFlops()`, `estimatedMemoryBytes()`, and `buildDumpInfoImpl()` to use the same effective row count.

Add runner hooks:

```cpp
bool forwardMTP(int32_t draft_condition_token);
const float *mtpLogits() const;
bool setComputeAllPositionLogits(bool enabled);
const float *getAllPositionLogits() const;
PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const;
bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
```

Start with conservative restore-then-replay:

```cpp
GenerationResult OrchestrationRunner::decodeStepMTP()
{
    PrefixStateSnapshot checkpoint = runner_->captureLivePrefixState();

    std::vector<int32_t> draft_tokens;

    if (prefill_logits_ready_)
    {
        draft_tokens.push_back(sampleFromCurrentLogits());
        prefill_logits_ready_ = false;
    }
    else
    {
        runner_->forward(&last_token_, 1);
        draft_tokens.push_back(sampleFromCurrentLogits());
    }

    runner_->forwardMTP(draft_tokens.back());
    draft_tokens.push_back(sampleFromMTPLogits());

    runner_->setComputeAllPositionLogits(true);
    runner_->forward(draft_tokens.data(), static_cast<int>(draft_tokens.size()));
    runner_->setComputeAllPositionLogits(false);

    AcceptanceResult accepted = verifyGreedy(draft_tokens, runner_->getAllPositionLogits());

    runner_->restoreLivePrefixState(checkpoint);
    runner_->forward(accepted.tokens.data(), static_cast<int>(accepted.tokens.size()));

    last_token_ = accepted.tokens.back();
    return accepted.toGenerationResult();
}
```

Dense-only truncate optimization can come later. Do not optimize around GDN until forced reject tests pass, because GDN state is in-place.

### Files

- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp`
- `src/v2/execution/compute_stages/stages/LMHeadStage.h/.cpp`

### Tests

- LMHead multi-position row count and dump info tests.
- MTP greedy acceptance with deterministic sampling: MTP-enabled output equals main-model greedy output.
- Forced mismatch test: rollback and replay match no-MTP state.
- Prefix + MTP full-hit path: terminal hidden/logits restored, no final-token duplication.

### Exit Criteria

- MTP greedy decode is correct but not necessarily faster.
- MTP can be enabled/disabled without changing non-MTP outputs.
- Rollback restores dense and hybrid state in tests.

## Phase 7: Hybrid GDN Prefix State

### Goal

Extend prefix cache and MTP rollback to Qwen3.5/Qwen3.6 hybrid FA+GDN models.

### Implementation Details

Add explicit hybrid state APIs to `IHybridKVCache`:

```cpp
struct HybridPrefixStateMetadata
{
    int total_layers = 0;
    int gdn_layers = 0;
    size_t host_bytes = 0;
    size_t device_bytes = 0;
    bool has_device_kernel_state = false;
};

struct HybridPrefixStateDescriptor
{
    int seq_idx = 0;
    int logical_token_count = 0;
    void *stream = nullptr;
};

virtual HybridPrefixStateMetadata hybridPrefixStateMetadata() const = 0;

virtual bool exportHybridPrefixState(
    const HybridPrefixStateDescriptor &desc,
    void *dst_host,
    void *dst_device) const = 0;

virtual bool importHybridPrefixState(
    const HybridPrefixStateDescriptor &desc,
    const void *src_host,
    const void *src_device) = 0;
```

Add kernel state methods where missing:

```cpp
class ITensorShortConvolution
{
public:
    virtual size_t stateBytes() const { return 0; }
    virtual bool exportState(void *dst_host, void *dst_device, void *stream) const { return false; }
    virtual bool importState(const void *src_host, const void *src_device, void *stream) { return false; }
};

class ITensorGatedDeltaNet
{
public:
    virtual size_t stateBytes() const { return 0; }
    virtual bool exportState(void *dst_host, void *dst_device, void *stream) const { return false; }
    virtual bool importState(const void *src_host, const void *src_device, void *stream) { return false; }
};
```

Rules:

- CPU hybrid export copies `HybridGDNLayerState::recurrence_state` and `conv_state` in global-layer order.
- GPU hybrid export/import includes device-resident short-conv and GDN recurrence kernel state.
- FA KV still goes through `IKVCache::exportLogicalBlock()` using global FA layer ids; hybrid cache implementation handles compressed FA-index remap.
- `clear_layer(global_gdn_layer)` must reset host vectors and GPU kernel state. Prefix restore uses import API, not raw vector mutation.

Hybrid payload layout adds one complete GDN-state snapshot per prefix block. This is redundant but simple and correct. Optimize delta storage later.

### Files

- `src/v2/kernels/IHybridKVCache.h`
- `src/v2/kernels/HybridKVCacheConfig.h/.cpp`
- `src/v2/kernels/cpu/CPUHybridRingKVCache.h`
- `src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h`
- `src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h`
- `src/v2/kernels/*/short_convolution*`
- `src/v2/kernels/*/gated_delta_net*`
- `src/v2/execution/prefix_cache/PrefixPayloadLayout.*`

### Tests

- Hybrid state metadata byte count tests.
- CPU/CUDA/ROCm hybrid export/import round-trip tests.
- Clear/import does not lose GDN kernel object pointers.
- Qwen3.5/Qwen3.6 suffix parity with prefix cache enabled/disabled.
- MTP forced rejection on hybrid model restores/replays to no-MTP state.

### Exit Criteria

- Hybrid prefix restore reproduces no-cache suffix logits.
- Hybrid MTP rollback is correct under forced mismatch.
- Existing CUDA/ROCm hybrid reset tests still pass.

## Phase 8: MoE Safety, Rebalance, And Parallelism

### Goal

Make prefix cache and MTP safe across MoE placement changes and multi-device execution.

### Implementation Details

Build MoE fingerprint material from `MoERuntimeTable`:

- `DeviceMoELayerRuntime::active_bank`
- `DeviceMoELayerRuntime::active_epoch`
- `DeviceMoEPlacementBank::expert_count`
- `DeviceMoEPlacementBank::experts[].logical_expert_id`
- `DeviceMoEPlacementBank::experts[].owner_participant`
- `DeviceMoEPlacementBank::experts[].local_slot`
- `DeviceMoEPlacementBank::experts[].flags`
- `DeviceMoEPlacementBank::local_compute_mask[]`
- `DeviceMoEPlacementBank::replica_role[]`

Expose fingerprint material with either:

```cpp
virtual uint64_t placementFingerprint() const = 0;
```

or a helper over public `hostLayerState(layer_idx)` if a virtual addition is too invasive.

When `MoERebalanceController` applies a placement update or replica mask change, it must either:

1. Increment the placement epoch used in `moe_fingerprint`, or
2. Call `PrefixStateCache::invalidateWhere(predicate)` for entries matching the old MoE fingerprint.

Parallelism coordination:

- LOCAL TP: each child `DeviceGraphOrchestrator` stores local shard payload; matched token count is minimum across child runners.
- PP: each stage stores local layer range only; matched token count remains global and is minimum across stages.
- GLOBAL/NODE_LOCAL TP: use `MPI_Allreduce(MIN)` over matched token counts before populate.
- Root rank must not populate a longer prefix than any non-root rank can restore.
- Harvest can be local after successful prefill; future lookups remain safe because min coordination handles partial availability.

Graph capture:

- Prefill graph capture keys or invalidation must include MoE placement epoch.
- Sparse host/collective overlay stages stay non-capturable until graph-native collectives exist.
- If a MoE sparse collective has not completed, follow-on transfers must wait for `collective_complete` before continuing.

### Files

- `src/v2/execution/moe/MoERuntimeTable.h/.cpp`
- `src/v2/execution/moe/MoERebalanceController.*`
- `src/v2/execution/moe/MoEExpertOverlayRuntimePlan.*`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h/.cpp`
- `src/v2/execution/runner/OrchestrationRunner.h/.cpp`
- `src/v2/execution/prefix_cache/PrefixCacheFingerprint.*`

### Tests

- Stable MoE placement hits cache.
- Rebalance epoch change misses or invalidates.
- Decode histogram is not restored as payload.
- LOCAL TP child miss clamps all children to common min prefix length.
- GLOBAL TP rank miss clamps all ranks via MPI min reduction.
- PP stage miss clamps the full pipeline to common min prefix length.
- Large Qwen3.6 MoE tests are opt-in/model-gated.

### Exit Criteria

- MoE prefix cache never reuses state across incompatible expert placement.
- Dynamic rebalance and prefix cache can coexist without silent wrong outputs.
- Multi-device prefix hits are all-or-common-prefix, never partially divergent.

## Phase 9: Observability, Performance, And Rollout

### Goal

Add production controls, diagnostics, and benchmark coverage after correctness has landed.

### Implementation Details

Add stats:

```cpp
struct PrefixCacheStats
{
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t partial_hits = 0;
    uint64_t matched_blocks = 0;
    uint64_t matched_tokens = 0;
    uint64_t stores = 0;
    uint64_t ram_bytes = 0;
    uint64_t device_bytes = 0;
    uint64_t disk_bytes = 0;
    uint64_t promotions = 0;
    uint64_t evictions = 0;
    uint64_t disk_hydrations = 0;
    uint64_t terminal_state_hits = 0;
    uint64_t hybrid_state_bytes = 0;
    uint64_t mtp_state_bytes = 0;
};

struct MTPStats
{
    uint64_t draft_steps = 0;
    uint64_t accepted_tokens = 0;
    uint64_t rejected_tokens = 0;
    uint64_t rollbacks = 0;
};
```

Expose stats through:

- INFO-level per-request summary when prefix cache or MTP is enabled.
- Existing profiling output where appropriate.
- Optional future server response headers/metadata.

Benchmark scenarios:

- Prefix disabled baseline.
- RAM-only prefix cache.
- RAM plus device-hot tier.
- Disk warm/hydrated path.
- MTP-only greedy decode.
- Prefix plus MTP.
- Hybrid Qwen3.5/Qwen3.6 prefix cache.
- MoE Qwen3.6 model-gated run.

Rollout controls:

- Keep prefix cache and MTP off by default.
- Allow dense RAM prefix cache to graduate first.
- Enable GPU/device-hot tier only after GPU logical I/O tests pass.
- Enable MTP only for model families with verified MTP tensor inventory and greedy parity.
- Keep stochastic speculative sampling deferred until greedy correctness and acceptance telemetry are stable.

### Files

- `src/v2/execution/prefix_cache/PrefixCacheStats.h`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/utils/BenchmarkRunner.cpp`
- `docs/v2/` follow-up benchmark notes or changelog files.

### Tests

- Stats counters increment deterministically in unit tests.
- Disabled feature paths produce zero/no-op stats.
- Benchmark smoke tests do not require large model files unless explicitly enabled.
- Regression filters before considering defaults:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_|^V2_Integration_" --output-on-failure --parallel
```

### Exit Criteria

- Users can tell whether a request hit, partially hit, missed, or bypassed prefix cache and why.
- MTP acceptance and rollback rates are visible.
- Benchmark data supports any future default enablement decision.

## Recommended First Implementation Slice

Start with the smallest slice that proves the contract without risking GPU/hybrid complexity:

1. Phase 1 config structs and CLI parsing, disabled by default.
2. Phase 2 `IKVCache` logical block API with default false implementations.
3. CPU `CPURingKVCache` logical export/import/truncate for FP32 and the Q8/Q16 formats already used in unit tests.
4. Phase 3 `PrefixStateCache`, hash-chain keys, RAM backend, and LRU without runner integration.
5. Dense CPU unit tests for export/import and RAM cache.
6. Phase 0 probe scaffolding that records state inventory even before full restore support exists.

Then proceed in this order:

1. Dense single-runner RAM prefix cache.
2. CUDA/ROCm logical I/O and shadow invalidation.
3. Hybrid GDN state import/export.
4. MTP sidecar loading and shifted prefill cache.
5. MTP greedy verification and rollback.
6. MoE placement fingerprint and multi-device min coordination.
7. Device-hot and disk tiers.
