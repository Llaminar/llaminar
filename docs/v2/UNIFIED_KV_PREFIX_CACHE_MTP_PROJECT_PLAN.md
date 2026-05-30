## Plan: Unified Prefix Cache And MTP

Build one "prefix state" subsystem that serves both cross-request prefix caching and in-request MTP rollback. The central contract should live at `IKVCache` / `IHybridKVCache`: dense FA KV, hybrid GDN recurrence/conv state, MoE placement fingerprints, and MTP's own shifted KV cache are exported/imported through typed snapshot APIs. Storage is tiered from the start (VRAM hot, RAM primary, disk durable), while request-local reset continues to clear live state and then repopulates from cached prefix state when a hit exists.

**Steps**

### Phase 0: Tier-0 State Probing And Scope Lock
1. Add a focused integration probing suite, `tests/v2/integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp`, that runs prefill/decode/reset/checkpoint/restore against representative runners without implementing the full cache manager yet.
2. Probe dense GQA state with a small Qwen2.5 model: capture logical FA KV hashes, ring head/count, positions, sequence lengths, prefill-logit readiness, and graph-cache reset behavior before and after `clearCache()`.
3. Probe hybrid Qwen3.5/Qwen3.6 state: capture FA KV plus `IHybridKVCache` GDN recurrence and short-conv hashes; verify reset zeroes both host and GPU kernel state and that a prototype restore reproduces suffix logits.
4. Probe MoE state: capture `MoERuntimeTable` active bank/epoch/fingerprint inputs, but classify decode histograms and routing scratch as telemetry/transient, not prefix payload. Verify placement epoch changes force a cache-key change or bypass.
5. Probe MTP metadata and state inventory using `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf` and `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`: enumerate `mtp.*` tensors, confirm `mtp_num_hidden_layers == 1`/shared embeddings naming, and define the MTP shifted-cache invariant.
6. Acceptance for Phase 0: a test report/table listing every reset-cleared state item, whether it is persistent prefix payload, speculative rollback payload, transient telemetry, or derived/recomputable.

### Phase 1: Prefix/MTP Configuration And Fingerprints
7. Add `PrefixCacheConfig` and `MTPConfig` through `OrchestrationConfig`, `RuntimeConfig`, `RankExecutionPlan`, and `GraphConfig`; keep both disabled by default.
8. Add CLI flags through `OrchestrationConfigParser::buildSpec()`: `--prefix-cache`, `--prefix-cache-block-size`, `--prefix-cache-vram-budget-mb`, `--prefix-cache-ram-budget-mb`, `--prefix-cache-disk-budget-mb`, `--prefix-cache-disk-dir`, `--prefix-cache-terminal-state`, `--prefix-cache-moe-policy`, `--mtp`, `--mtp-draft-tokens`, and `--mtp-verify-mode`.
9. Add centralized debug/env overrides only through `DebugEnv`, not ad hoc `getenv` calls.
10. Implement fingerprint builders for model/tokenizer, runtime precision/layout/RoPE-on-read, topology/TP/PP shard, hybrid layer/GDN dimensions, MoE placement epoch/masks/replicas, and MTP depth/weight set.
11. Report resolved cache sizing in dry-run/placement output after model metadata is loaded.

### Phase 2: Snapshot Contracts At IKVCache/IHybridKVCache
12. Extend `IKVCache` with logical block metadata/export/import APIs plus sequence truncate/checkpoint helpers. Implement against logical token order, not raw row assumptions, so ring wrap, head-major Q16_1, position-major FP16/Q8_1, TQ, sharded KV heads, and RoPE-on-read all work.
13. Update CUDA/ROCm cache bases so import/truncate invalidates converted/RoPE shadow buffers when `head` changes or content is overwritten; this follows the known GPU RoPE shadow wrap hazard.
14. Extend `IHybridKVCache` with hybrid prefix-state metadata/export/import covering host recurrence/conv vectors and device-resident short-conv/GDN kernel state. Hybrid FA layer export/import must use compressed FA indices, not global GDN indices.
15. Add kernel-level export/import on `ITensorShortConvolution` and `ITensorGatedDeltaNet` where GPU state is currently reset-only.
16. Reuse the same APIs for speculative rollback snapshots and persistent prefix blocks.

### Phase 3: Tiered Prefix Store And Arena Staging
17. Add `PrefixStateCache`, `PrefixCacheKey`, `PrefixPayloadLayout`, `PrefixStateBlock`, `IPrefixStorageBackend`, and three backends under a new prefix-cache module.
18. Treat RAM as the primary source-of-truth tier, VRAM/HBM as an optional hot promotion tier, and disk as a durable serialized tier. A VRAM eviction must never invalidate the RAM/disk source block.
19. Add dedicated `BufferArena` staging buffers for restore/harvest paths, not for long-lived persistent storage: prefix K/V staging, hybrid-state staging, MTP K/V staging, terminal hidden/logits staging, and disk IO staging. Persistent blocks live outside the arena like weights.
20. Preallocate staging by block layout during runner initialization so suffix execution performs no dynamic allocations.
21. Disk tier uses a versioned manifest plus raw binary payloads with key/fingerprint/layout/checksum, shaped after the stage dump metadata format; hydrate disk blocks into RAM before restoring to request state.

### Phase 4: Dense Prefix Cache End To End
22. Wire `OrchestrationRunner::prefill()` to clear request-local state, compute hash-chain block keys, coordinate common hit length across ranks/devices, populate request state, run only suffix prefill, and harvest complete prompt blocks.
23. Add `DeviceGraphOrchestrator` low-level helpers: lookup/populate prefix, harvest prefix, restore terminal state, expose cache stats, and preserve persistent cache across `clearInferenceState()`.
24. Preserve first-token semantics: if a full prefix hit does not include terminal logits and terminal hidden, reduce the hit by one block and recompute the final block.
25. Unit/integration tests: hash/LRU/storage, dense logical KV round-trip, repeated prompt exact-token parity, shared-prefix/different-suffix parity, full-hit terminal-state fallback, and budget eviction.

### Phase 5: MTP Loading, Graphs, And Shifted Cache
26. Extend `GraphConfig`/`ModelWeights`/weight bindings for MTP D=1, shared embeddings, `mtp.fc`, MTP norms, the FA block weights, q/k norms, FFN weights, and shared LM head. Support both llama.cpp GGUF `mtp.*` names and any close HF-compatible variants discovered in Phase 0.
27. Build an MTP sidecar graph in Qwen35/Qwen36 graph code using existing full-attention FA stages where possible. The MTP block is FA with gated Q, QK norm, partial RoPE, FFN, final `mtp.norm`, and shared LM head.
28. Add MTP activation/staging buffers to `BufferId` and graph buffer registration: MTP embedding, normalized hidden, normalized embedding, concat/projection, hidden, Q/K/V, attention output/proj, FFN intermediates, logits, and MTP cache staging.
29. Add `mtp_kv_cache` to `DeviceGraphOrchestrator::InferenceState`, one FA KV cache per MTP depth. For current Qwen3.5/3.6, depth is 1.
30. Define the shifted-cache invariant: after N prompt tokens, main KV/GDN state is at N, while MTP depth-0 KV is at max(0, N-1), built from main hidden rows 0..N-2 and token embeddings 1..N-1. Prefix blocks must store or recompute enough terminal hidden state to continue this invariant.
31. During normal prefill with MTP enabled, compute all main hidden rows needed for MTP prefill, run the shifted MTP prefill sidecar, and harvest MTP KV state at the same prefix block boundaries.

### Phase 6: MTP Decode, Verification, And Rollback
32. Add a config-gated MTP branch in `OrchestrationRunner::generate()`/`decodeStep()` while preserving the current one-token path when MTP is off.
33. Extend `IInferenceRunner`/`DeviceGraphOrchestrator` with access to last/all hidden rows, MTP forward, multi-position logits mode, terminal hidden restore, and state snapshot/restore/truncate helpers.
34. Extend `LMHeadStage` with a `compute_all_positions`/verification flag. Normal prefill remains last-row-only; verification computes `[seq_len, vocab]` logits.
35. Use the prefix-state snapshot API for rollback: checkpoint main KV/GDN + MTP KV before draft/verify, run MTP draft, run main verification over draft tokens, decide accept/correction, restore checkpoint, then replay only accepted/correction tokens. Dense-only optimization can later truncate instead of full restore/replay.
36. Add greedy verification first; add stochastic speculative sampling once deterministic parity is stable.
37. MTP prefix hits must restore MTP KV plus terminal hidden/logits. If terminal hidden is unavailable, reduce the hit by one block and recompute.

### Phase 7: Hybrid GDN Prefix And MTP End To End
38. Enable hybrid payloads once `IHybridKVCache` import/export passes CPU/CUDA/ROCm tests. Prefix block payload includes FA KV, all GDN recurrence/conv state, MTP KV, terminal hidden/logits when available.
39. Validate Qwen3.5/Qwen3.6 hybrid suffix parity with and without prefix cache, then with MTP enabled and greedy decode.
40. Add explicit tests for GDN rollback after rejected MTP draft tokens; this is the main correctness risk because GDN state is in-place and not append-only.

### Phase 8: MoE Safety, Rebalance, And Parallelism
41. Add `moe_fingerprint` from runtime table placement bank/epoch, expert masks, ownership, replica role, and rebalance domain. Do not store decode histograms as prefix model state.
42. In `MoERebalanceController`, bump/invalidate prefix-cache MoE epoch after placement changes. OFF/OBSERVE modes can hit stable fingerprints; DYNAMIC mode only hits while the placement epoch matches.
43. Ensure prefill graph capture keys or invalidation include MoE placement epoch. Sparse host/collective overlay stages remain non-capturable until graph-native collectives exist.
44. Coordinate hit length with MIN across LOCAL TP child DGOs, MPI ranks, and PP stages before any populate. Each participant restores only its local shard/payload.
45. Add Qwen3.6-35B-A3B tests behind an opt-in model availability guard because it is large.

### Phase 9: Observability, Production Controls, And Performance
46. Add stats: lookups, hits, matched blocks, tokens saved, stores, VRAM/RAM/disk bytes, promotions, evictions, disk hydrations, terminal-state hits, MTP acceptance rate, rollback count, and hybrid-state bytes.
47. Add INFO-level per-request summary when prefix cache or MTP is enabled; add server response metadata/headers later if desired.
48. Benchmark repeated system prompts with prefix cache disabled, RAM-only, VRAM+RAM, disk warm/hydrated, MTP-only, and prefix+MTP.

**Relevant files**
- `src/v2/kernels/IKVCache.h` — add logical block export/import, truncate, and snapshot helpers.
- `src/v2/kernels/IHybridKVCache.h` and `src/v2/kernels/HybridKVCacheConfig.h` — add hybrid GDN state metadata/export/import.
- `src/v2/kernels/cpu/CPURingKVCache.h`, `src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h`, `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h` — implement logical I/O and rollback semantics per backend.
- `src/v2/memory/BufferId.h`, `src/v2/memory/BufferArena.h`, `src/v2/memory/BufferArena.cpp` — add prefix/MTP staging buffers and arena registration/name mapping.
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h` and `.cpp` — own main/MTP caches, populate/harvest helpers, terminal hidden/logits, rollback snapshot helpers.
- `src/v2/execution/runner/OrchestrationRunner.h` and `.cpp` — high-level prefix lookup/harvest, MTP branch, MPI coordination.
- `src/v2/execution/compute_stages/stages/LMHeadStage.h` and `.cpp` — multi-position logits mode.
- `src/v2/models/GraphTypes.h` — add MTP config/weights/bindings and buffer fields.
- `src/v2/models/qwen35/Qwen35GraphConfigBuilder.cpp`, `src/v2/models/qwen35/Qwen35Graph.cpp`, and Qwen35MoE graph files — load MTP metadata/weights and build MTP sidecar graphs.
- `src/v2/execution/moe/MoERuntimeTable.h` and `.cpp` plus `MoERebalanceController` — expose safe placement fingerprint inputs and invalidation hooks.
- `src/v2/config/OrchestrationConfig.h`, `src/v2/config/OrchestrationConfigParser.cpp`, `src/v2/execution/config/RuntimeConfig.h` — config propagation.
- `tests/v2/unit/kernels/`, `tests/v2/integration/prefix_cache/`, `tests/v2/integration/parity/`, and model-gated tests using `/opt/llaminar-models`.

**Verification**
1. Configure/build integration with Ninja and full parallelism: `cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Integration -DHAVE_CUDA=ON -DHAVE_ROCM=ON` then `cmake --build build_v2_integration --parallel`.
2. Run focused unit tests after each contract phase: `ctest --test-dir build_v2_integration -R "V2_Unit_.*(KVCache|Prefix|MTP|Hybrid)" --output-on-failure --parallel`.
3. Run tier-0 probe: `ctest --test-dir build_v2_integration -R "V2_Integration_KVPrefixMTP_Tier0" --output-on-failure --parallel`.
4. Run dense prefix parity: repeated Qwen2.5 prompt with prefix disabled/enabled, compare logits/tokens exactly under greedy sampling.
5. Run hybrid prefix parity: Qwen3.5/Qwen3.6 hybrid prompt with shared prefix and differing suffixes; compare no-cache vs prefix cache outputs.
6. Run MTP metadata/weight tests against `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf` and `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`; skip gracefully if files are absent.
7. Run MTP greedy acceptance tests with MTP enabled and disabled; outputs must match the main model for greedy verification, while stats show accepted drafts.
8. Run GDN rejection rollback tests that force an MTP mismatch and verify restored/replayed GDN state matches no-MTP state.
9. Run MoE fingerprint tests: stable placement hits, rebalance epoch misses/invalidates, decode histogram not restored as model state.
10. Run LOCAL TP/GLOBAL TP/PP coordination tests where one participant intentionally misses a block; all participants must restore the minimum common prefix length.
11. Run memory-budget stress: RAM-only with VRAM budget 0, small VRAM hot tier eviction, disk hydration, and budget-too-small disable-with-warning behavior.
12. Run full regression filters before enabling defaults: `ctest --test-dir build_v2_integration -R "^V2_Unit_|^V2_Integration_" --output-on-failure --parallel`.

**Decisions**
- Prefix caching and MTP share one snapshot/restore contract instead of separate mechanisms.
- RAM is the primary capacity tier; VRAM is an accelerator and disk is durable backing.
- Persistent prefix blocks live outside BufferArena; BufferArena owns only request-local staging buffers required to restore/harvest without hot-path allocation.
- Full prefix hits require terminal logits and terminal hidden for MTP. Without both, recompute the final block.
- GDN rollback is snapshot/restore/replay first, not append-only truncate, because GDN recurrence and conv state mutate in place.
- MoE decode histograms are telemetry for rebalance decisions, not prefix payload. MoE placement/version is part of the key.
- MTP support starts with Qwen3.5/Qwen3.6 D=1 shared-embedding models and is config-gated.

**Further Considerations**
1. Disk tier compression can wait until raw manifest/payload correctness is proven; add compression as a later backend option.
2. Stochastic speculative sampling should follow greedy parity; it has distribution-correctness requirements beyond token equality.
3. Large Qwen3.6 MoE tests should be opt-in/model-gated to keep default CI practical.

## Concrete Implementation Details

This section turns the epic into a codebase-specific implementation contract. It is intentionally more prescriptive than the phase list above: the goal is to make the first implementation pass mechanical and to avoid rediscovering cache, graph replay, and hybrid-state edge cases during coding.

### New Module Layout

Add prefix-state code under a dedicated module so it does not leak into model graph builders or cache backends:

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

Keep the module below `execution/` because it coordinates runner state, model/runtime fingerprints, and KV cache import/export. It should not live under `kernels/`, because the storage tiers and hash-chain policy are runtime orchestration concerns.

Backend-specific logical KV copying belongs with the cache implementations:

```text
src/v2/kernels/IKVCache.h
src/v2/kernels/IHybridKVCache.h
src/v2/kernels/cpu/CPURingKVCache.h
src/v2/kernels/cpu/CPURingKVCache.cpp
src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.h/.cpp
src/v2/kernels/cuda/kvcache/CUDARingKVCache.h/.cu
src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.h/.cu
src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.h/.cpp
src/v2/kernels/rocm/kvcache/ROCmRingKVCache.h/.cpp
src/v2/kernels/rocm/kvcache/ROCmRingKVCacheTQ.h/.hip
```

### State Classification Contract

Phase 0 should produce a test-backed version of this table. The first implementation should assume these classifications until a probe disproves them.

| State | Owner Today | Prefix Payload | MTP Rollback Payload | Notes |
|------|-------------|----------------|----------------------|-------|
| Full-attention K/V rows | `IKVCache` implementations | Yes | Yes | Export/import in logical token order only. Never let `PrefixStateCache` infer raw ring layout. |
| Ring head/count | `IKVCache` implementations | Metadata | Metadata | Restore must make `get_cached_tokens()` and graph replay head buffers consistent before suffix execution. |
| GDN recurrence state | `IHybridKVCache` / `HybridGDNLayerState` / GPU GDN kernel | Yes | Yes | FP32, in-place, not append-only. Snapshot/restore is required for correctness. |
| GDN short-conv state | `IHybridKVCache` / `HybridGDNLayerState` / GPU conv kernel | Yes | Yes | Same restore contract as recurrence state. |
| MTP depth KV | New `mtp_kv_cache` in `DeviceGraphOrchestrator::InferenceState` | Yes when MTP enabled | Yes | Separate FA cache per MTP depth. Current target depth is 1. |
| Terminal hidden row | `InferenceState::hidden` | Yes when MTP enabled | Useful | Required for full prefix hits to continue MTP without recomputing the final block. |
| Terminal logits row | `InferenceState::logits` | Optional for non-MTP, required with full MTP hit | Useful | Preserves first decode token semantics after a full prompt hit. |
| `state_.positions` / `sequence_lengths` | `DeviceGraphOrchestrator::InferenceState` | Metadata | Metadata | Populate sets these to the restored token count. |
| `prefill_logits_ready_` | `OrchestrationRunner` | Derived | Derived | Set true only when terminal logits were restored or suffix/final block was recomputed. |
| MoE placement bank/epoch/masks/replicas | `MoERuntimeTable`, rebalance controller, stage masks | Key only | Key only | Changes invalidate or miss; not copied as payload. |
| MoE decode histogram | `MoERuntimeTable` / `DecodeExpertHistogram` | No | No | Runtime telemetry for rebalance decisions, not model state. |
| MoE route scratch and grouped scratch | `MoERuntimeTable` and MoE stages | No | No | Transient buffers; reset or capacity-preserve only. |
| Graph cache entries | `ForwardExecutionEngine`, `LayerGraphCache` | No | No | Rebuilt/reset at request boundary; prefix restore must work with cache miss and replay. |
| Stage snapshots/debug dumps | `SnapshotCapture`, stage dump framework | No | No | Useful for probes only. |
| Prepared GEMM engines | `PreparedWeightStore`, stage caches | No | No | Kernel lifetime/cache optimization, not request state. |

### Configuration Shape

Add explicit config structs rather than scattering booleans through existing config objects.

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

Wire these through:

- `OrchestrationConfig` for CLI/YAML values.
- `RuntimeConfig` for parsed values used after plan building.
- `RankExecutionPlan::runtime` for per-rank execution.
- `GraphConfig` for graph/model-specific behavior.

Add CLI flags in `OrchestrationConfigParser::buildSpec()` under two new help categories, `Prefix Cache` and `MTP`:

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

Environment overrides, if added, must be parsed through `DebugEnv` and copied into these structs during configuration setup. Do not read prefix/MTP environment variables directly from hot paths.

### Fingerprint Construction

Implement fingerprint construction in `PrefixCacheFingerprint.h/.cpp`. Keep it centralized so future features such as LoRA/adapters can add key material without hunting call sites.

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

Rules:

- `model` includes architecture, GGUF metadata fingerprint, tensor directory fingerprint, vocab size, and tied/separate LM-head mode.
- `tokenizer` includes tokenizer model, added tokens, chat-template identity, and template override identity.
- `runtime` includes activation precision, KV precision, KV layout, RoPE-on-read, TurboQuant/TQ mode, Q16 scales, fused attention backend, and partial RoPE factor.
- `topology` includes TP/PP degree, rank id within the participating domain, local device id, local KV-head start/count, layer range, and vocab shard.
- `hybrid` includes `layer_types`, FA/GDN counts, GDN state size, group count, time-step rank, inner size, conv kernel, and local head assignment.
- `moe` includes expert count, top-k, expert mode, static/dynamic placement epoch, active bank, local compute masks, replica roles, and rebalance domain id. If the active policy is `Disabled`, bypass prefix cache for MoE models instead of producing a zero fingerprint.
- `mtp` includes enabled flag, depth, draft token count, shared/dedicated embedding mode, MTP tensor-name fingerprint, and MTP weight shapes.

### IKVCache Logical Block API

Add a narrow logical API to `IKVCache`. The default implementation should return false or an empty layout so the interface can land before every backend is complete.

```cpp
struct KVCacheLogicalBlockDescriptor
{
	int layer = 0;                 // Global layer index as seen by graph stages.
	int seq_idx = 0;
	int logical_token_start = 0;   // Oldest-to-newest logical index.
	int token_count = 0;
	void *stream = nullptr;        // cudaStream_t/hipStream_t/nullptr.
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
	int implementation_head = 0;   // Backend-defined; diagnostic only.
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

Important semantics:

1. `logical_token_start` and `token_count` describe logical sequence order, not raw ring rows.
2. `PrefixStateCache` must never calculate raw cache row offsets. CPU/CUDA/ROCm implementations own that mapping.
3. `global_layer` follows graph-stage numbering. Hybrid cache implementations remap full-attention layers internally through `HybridLayerMap`.
4. Export to RAM may perform D2H. Export to device hot tier may perform D2D. The descriptor stream is mandatory for async GPU paths.
5. Import must set the backend's head/count metadata so `get_cached_tokens()`, `setDynamicHead()`, and `advanceHead()` remain consistent.
6. Import/truncate/clear must invalidate converted KV and RoPE-on-read shadow state.
7. `truncateSequence()` is a logical operation. Dense caches can usually update counts only; hybrid GDN still needs the higher-level snapshot restore path.

### Backend Implementation Notes

`CPURingKVCache` should implement logical export/import by copying row spans out of each entry in oldest-to-newest order. The CPU class documents `head` as oldest valid token, while CUDA/ROCm base classes treat `entryHead()` as the next write position. Do not rely on shared naming across backends; implement logical order per backend.

CUDA and ROCm base classes already expose these hooks:

- `entryHead(layer, seq_idx)`
- `entryCount(layer, seq_idx)`
- `setEntryHead(layer, seq_idx, value)`
- `setEntryCount(layer, seq_idx, value)`
- `resetEntry(layer, seq_idx)`
- `onClearSequence(layer, seq_idx)`
- `onEviction(layer, seq_idx, num_evicted)`
- `onAdvanceComplete(layer, seq_idx)`

Add one more hook for restore/truncate invalidation if needed:

```cpp
virtual void onRestoreSequence(int layer, int seq_idx) {}
```

For CUDA/ROCm, import should normally lay out restored tokens as an unwrapped logical prefix starting at raw row 0 and set the next-write head to `cached_tokens % max_seq_len`. That keeps subsequent append and graph-captured `setDynamicHead()` simple. If a future optimization imports directly into wrapped form, it must be tested against graph replay and `get_kv_converted()`.

The converted/RoPE shadow hazard is real: GPU converted KV shadow state may track count but not content/head changes. Any import, truncate, clear, or eviction path must invalidate the shadow for that layer/sequence. CPU already guards converted reads with head changes; CUDA/ROCm need equivalent invalidation.

### IHybridKVCache Prefix-State API

Add explicit hybrid state export/import rather than letting prefix-cache code reach into `HybridGDNLayerState` vectors.

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

Rules:

- CPU hybrid export copies `HybridGDNLayerState::recurrence_state` and `conv_state` in global-layer order.
- GPU hybrid export/import must include device-resident state owned by `ITensorShortConvolution` and `ITensorGatedDeltaNet`, not just host mirror vectors.
- Hybrid full-attention KV export/import still goes through `IKVCache::exportLogicalBlock()` and uses global FA layer ids. The hybrid cache implementation does the compressed FA-index remap.
- `clear_layer(global_gdn_layer)` must continue to reset both host GDN vectors and GPU kernel state. Prefix restore should use the new import API, not `clear_layer()` plus raw copies.

Add kernel state methods where they do not already exist:

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

### Prefix Payload Layout

`PrefixPayloadLayout` should be derived once after graph config and cache creation, then reused for budget checks and allocation.

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

Dense payload bytes:

```text
sum over local FA layers:
  block_size * local_kv_heads * head_dim * (bytes_per_k + bytes_per_v)
```

Hybrid payload bytes add one complete GDN-state snapshot per prefix block. This is intentionally redundant at block boundaries; optimize later only after correctness is proven.

MTP payload bytes add MTP-depth FA KV for the shifted MTP sidecar. For Qwen3.5/Qwen3.6 D=1, this is one local FA cache layer.

### Storage Tier Semantics

The storage stack should have one manager and independent tier backends:

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

`PrefixStateCache` owns the hash map, block chains, LRU policy, ref counts, and tier promotion/demotion policy. The storage backends own memory/files only.

Tier rules:

- RAM is the source of truth for early correctness. A device-hot block is a promotion of a RAM block.
- Disk blocks hydrate into RAM before request-state import. Do not import directly from disk into GPU in the first version.
- Device-hot eviction must drop only the hot handle; the RAM handle remains valid.
- If RAM budget cannot hold one complete block, prefix caching disables itself with a warning. It must not fail model initialization.
- If device-hot budget cannot hold one block, disable only the device tier and continue with RAM.

### BufferArena Additions

Persistent prefix blocks stay outside `BufferArena`. The arena should provide request-local staging tensors so restore/harvest does not allocate on the hot path.

Add these `BufferId` values before `_COUNT` in `BufferId.h`:

```cpp
PREFIX_K_STAGING,
PREFIX_V_STAGING,
PREFIX_HYBRID_STATE_STAGING,
PREFIX_MTP_K_STAGING,
PREFIX_MTP_V_STAGING,
PREFIX_TERMINAL_HIDDEN,
PREFIX_TERMINAL_LOGITS,

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

Update `bufferIdName()` and `BufferArena::bufferNameToId()` for every new buffer name. The MTP graph should refer to stable string names such as `mtp_embedding`, `mtp_norm_hidden`, and `mtp_logits` so schema descriptors map cleanly.

Sizing rules:

- `PREFIX_K_STAGING` / `PREFIX_V_STAGING`: enough for one block of one local FA layer in native cache precision and layout. If native precision has no TensorBase representation for raw bytes, add a small typed staging helper in the backend and keep only terminal state/MTP activations in the arena for the first pass.
- `PREFIX_HYBRID_STATE_STAGING`: FP32 rows sized to `ceil(hybrid_state_bytes / sizeof(float))` by 1.
- `PREFIX_MTP_K_STAGING` / `PREFIX_MTP_V_STAGING`: same as prefix K/V staging but for one MTP depth layer.
- `PREFIX_TERMINAL_HIDDEN`: `[1, d_model]` using activation precision or FP32 if terminal snapshots are always stored dequantized.
- `PREFIX_TERMINAL_LOGITS`: `[1, vocab_local]` for column-parallel LM head, `[1, vocab_size]` otherwise.
- `MTP_*` activation buffers follow existing Qwen35 FA buffer sizing, but scoped to the MTP sidecar graph.

`BufferArena::allocate()` is called once during runner initialization. Do not resize prefix/MTP arena staging on cache hits. If a model/layout needs more staging than predicted, disable prefix cache or MTP for that runner and log a hard warning.

### DeviceGraphOrchestrator Integration

Add request-local MTP/prefix fields to `InferenceState`:

```cpp
std::vector<std::unique_ptr<IKVCache>> mtp_kv_caches;
std::shared_ptr<TensorBase> prefix_terminal_hidden;
std::shared_ptr<TensorBase> prefix_terminal_logits;
```

Add persistent prefix-cache ownership to `DeviceGraphOrchestrator`, outside `InferenceState`, because `InferenceState::clear()` is request-local:

```cpp
std::shared_ptr<PrefixStateCache> prefix_cache_;
PrefixPayloadLayout prefix_layout_;
PrefixCacheStats prefix_cache_stats_;
```

Add low-level runner methods. These can start as concrete `DeviceGraphOrchestrator` methods, then move into `IInferenceRunner` once `RankOrchestrator` forwarding is ready.

```cpp
PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) const;
bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0);
bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count);
bool restorePrefixTerminalState(const PrefixLookupResult &hit);
PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const;
bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0);
bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0);
```

`clear_cache()` must continue to clear request-local KV/GDN/positions/graph state, but it must not evict or clear `prefix_cache_`. The existing `state_.clear()` call is correct for live caches as long as persistent prefix storage is outside `InferenceState`.

Import order for a prefix hit:

1. Clear request-local state.
2. Import FA KV blocks into `state_.kv_cache` or `state_.pp_kv_caches`.
3. Import hybrid GDN state through `IHybridKVCache`, if present.
4. Import MTP KV through `state_.mtp_kv_caches`, if MTP is enabled.
5. Restore terminal hidden/logits when present.
6. Set `state_.positions[seq_idx]` and `state_.sequence_lengths[seq_idx]` to `hit.cached_tokens`.
7. Invalidate graph-cache dynamic state that depends on cache head/count, or ensure dynamic params are refreshed before replay.

### OrchestrationRunner Prefill Flow

`OrchestrationRunner::prefill()` already has the prompt token vector and MPI coordination, so it should own high-level prefix lookup and harvest.

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
			matched_tokens = 0; // fallback to normal full prefill after clearing.
	}

	int suffix_start = matched_tokens;
	int suffix_len = static_cast<int>(prompt_tokens.size()) - suffix_start;

	if (suffix_len > 0)
	{
		runner_->forward(prompt_tokens.data() + suffix_start, suffix_len);
		prefill_logits_ready_ = true;
	}
	else if (common_hit.has_terminal_logits && (!mtp_enabled_ || common_hit.has_terminal_hidden))
	{
		runner_->restorePrefixTerminalState(common_hit);
		prefill_logits_ready_ = true;
	}
	else
	{
		// Recompute final complete block to recover logits and terminal hidden.
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

For worker ranks, add MPI commands or extend the existing `PREFILL` command payload so all ranks perform lookup, min-coordinate, populate, suffix forward, and harvest in the same order. Rank 0 should not independently decide a longer prefix than a non-root rank can restore.

### RankOrchestrator And Parallelism

`RankOrchestrator` already exposes `deviceRunner(int)` and owns `device_runners_` / `pp_stage_runners_`. Prefix integration should be a coordinator layer over those children.

Rules:

- LOCAL TP: every child `DeviceGraphOrchestrator` hashes the same token blocks but stores different local payloads because topology fingerprints differ. The matched token count is the minimum across children.
- PP: every stage runner stores only its local layer range. Prefix length is still global token count and must be the minimum across stages.
- GLOBAL/NODE_LOCAL TP: use `MPI_Allreduce(MIN)` over local matched tokens before any populate. Never populate one rank with a longer prefix.
- Harvest can be local after a successful prompt prefill. If one child cannot allocate a block, that child simply misses on future lookups and min coordination keeps execution safe.

Add `IInferenceRunner` virtual hooks only after `DeviceGraphOrchestrator` works:

```cpp
virtual PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) { return {}; }
virtual bool populatePrefix(const PrefixLookupResult &hit) { return false; }
virtual bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) { return false; }
virtual PrefixCacheStats prefixCacheStats() const { return {}; }
```

`RankOrchestrator` overrides these by dispatching to its child runners and applying the min-match rule.

### MTP Model Loading

Add a model-facing MTP structure rather than folding MTP weights into ordinary transformer layer weights.

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

Add `MTPWeights mtp;` to `ModelWeights` and matching prepared-weight bindings if the graph builder uses `PreparedWeightStore` directly.

For Qwen35/Qwen36 target files, probe and support these names first:

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

If `mtp.num_hidden_layers` metadata is absent but `mtp.fc.weight` exists, infer depth 1 and log that metadata was inferred from tensor presence. If metadata says MTP exists but required tensors are missing, disable MTP and keep normal inference working unless `--mtp` was explicitly requested; explicit request should fail loudly.

### MTP Sidecar Graph

Add a Qwen35/Qwen36 graph method that builds a sidecar graph for one MTP depth:

```cpp
ComputeGraph Qwen35Graph::buildMTPGraph(
	int depth_idx,
	const MTPDepthWeights &weights,
	const MTPForwardInput &input,
	MTPForwardOutput &output);
```

The MTP graph sequence is:

1. Embed the draft token using shared `embedding_table`, unless `mtp_use_dedicated_embeddings` becomes true for a future model.
2. RMSNorm main hidden with `mtp.pre_fc_norm_hidden.weight`.
3. RMSNorm draft embedding with `mtp.pre_fc_norm_embedding.weight`.
4. Concatenate `[norm_hidden ; norm_embedding]` into `MTP_CONCAT`.
5. Project with `mtp.fc.weight` into `MTP_PROJECTED`.
6. Run one FA block using existing Qwen35 FA stages where possible: input norm, gated Q projection, Q/K norms, partial RoPE, MTP KV append/read, attention, attention output gate if required, Wo, residual, FFN norm, SwiGLU FFN, residual.
7. Apply `mtp.norm.weight`.
8. Run shared LM head into `MTP_LOGITS`.

Avoid a separate model operator layer. This remains a graph of the same compute stages already used by Qwen35 full-attention layers, plus small MTP-specific concat/projection/norm stages if no existing stage fits cleanly.

### MTP Shifted-Cache Invariant

For a prompt length `N` after prefill:

```text
main KV/GDN state: state after tokens [0, N)
MTP depth-0 KV:    state after MTP inputs for pairs (hidden[i], token[i+1]) where i in [0, N-1)
terminal hidden:   main hidden row for token N-1
terminal logits:   logits predicting token N
```

This means a full prefix hit for MTP must restore terminal hidden as well as terminal logits. Terminal logits alone can produce the first generated token, but MTP cannot draft the following token without the hidden row that conditioned the MTP module.

If a prefix block lacks terminal hidden, reduce the hit by one block and recompute the final block. Do not try to synthesize terminal hidden from logits.

### Multi-Position LM Head

`LMHeadStage` currently optimizes prefill by computing only one row:

```cpp
const int lm_m = (params_.seq_len > 1) ? 1 : params_.seq_len;
const int lm_activation_offset = activationRowOffsetForLogits();
```

Add a flag to `LMHeadStage::Params`:

```cpp
bool compute_all_positions = false;
```

Then use:

```cpp
const int lm_m = params_.compute_all_positions ? params_.seq_len
			   : (params_.seq_len > 1) ? 1
			   : params_.seq_len;
const int lm_activation_offset = params_.compute_all_positions ? 0
							   : activationRowOffsetForLogits();
```

`estimatedFlops()`, `estimatedMemoryBytes()`, and `buildDumpInfoImpl()` must use the same effective row count. Normal prefill keeps the one-row optimization. Verification passes set `compute_all_positions = true` and require logits storage sized for `seq_len * vocab_size` or local vocab shard.

### MTP Decode Algorithm

Start with greedy verification only. Use the same live-state snapshot API as prefix restore so dense, hybrid, and MTP state rollback follow one path.

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

	MTPDraftResult mtp = runner_->forwardMTP(draft_tokens.back());
	draft_tokens.push_back(sampleFromMTPLogits(mtp));

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

The restore-then-replay step is deliberately conservative. Dense models can later optimize by keeping verification KV and truncating rejected tails, but hybrid GDN cannot rely on append-only truncate because recurrence and conv state are mutated in place.

### MoE Fingerprint And Rebalance Hook

`MoERuntimeTable` has the fields needed to build a stable placement fingerprint:

- `DeviceMoELayerRuntime::active_bank`
- `DeviceMoELayerRuntime::active_epoch`
- `DeviceMoEPlacementBank::expert_count`
- `DeviceMoEPlacementBank::experts[].logical_expert_id`
- `DeviceMoEPlacementBank::experts[].owner_participant`
- `DeviceMoEPlacementBank::experts[].local_slot`
- `DeviceMoEPlacementBank::experts[].flags`
- `DeviceMoEPlacementBank::local_compute_mask[]`
- `DeviceMoEPlacementBank::replica_role[]`

Add an accessor rather than letting prefix-cache code walk private internals:

```cpp
virtual uint64_t placementFingerprint() const = 0;
```

or, if avoiding a virtual addition initially, add a free helper over the public `hostLayerState(layer_idx)` API.

When `MoERebalanceController` applies a placement update or replica mask change, it must either:

1. Increment the placement epoch used in `moe_fingerprint`, or
2. Call `PrefixStateCache::invalidateWhere(predicate)` for entries matching the old MoE fingerprint.

Do not copy `decode_histogram` into prefix state. Histogram accumulation can continue after a prefix hit from the actual decoded suffix only.

### Disk Format

Use a simple directory layout first:

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

Metadata should include:

- Format version.
- `PrefixCacheKey` fields.
- `PrefixPayloadLayout` fields.
- Token count and block index.
- Tensor precisions/layout.
- Byte lengths and checksums for every payload file.
- Model/runtime/topology/hybrid/MoE/MTP fingerprints repeated for diagnostics.

Disk load path is `disk -> RAM -> request state`. Disk writes happen after RAM insertion succeeds. If disk write fails, keep the RAM block and record a stats failure; do not fail inference.

### Tier-0 Probe Tests

Add a new integration target under `tests/v2/integration/prefix_cache/`:

```text
Test__KVPrefixMTPStateProbe.cpp
```

Initial tests:

| Test | Purpose |
|------|---------|
| `DenseQwen25_ResetStateInventory` | Run prefill/decode/clear and record FA KV counts, positions, logits readiness, and graph-cache state. |
| `DenseQwen25_PrototypeKVRestoreMatchesSuffix` | Export live FA KV after a prefix, clear, import, run suffix, compare logits to no-clear baseline. |
| `HybridQwen35_GDNStateInventory` | Verify GDN recurrence/conv states change after prefill/decode and zero after clear. |
| `HybridQwen35_PrototypeHybridRestoreMatchesSuffix` | Restore FA KV plus GDN state and compare suffix logits. |
| `MoE_RuntimeFingerprintChangesOnPlacementEpoch` | Flip/update placement and verify MoE fingerprint changes while histograms remain non-payload. |
| `MTP_Qwen36_MetadataAndTensorInventory` | Model-gated inspection of `/opt/llaminar-models/Qwen3.6-27B-Q4_K_S.gguf` and `/opt/llaminar-models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf`. |
| `MTP_ShiftedCacheInvariantProbe` | Once MTP sidecar exists, verify MTP cache token count is `max(0, main_tokens - 1)` after prefill. |
| `MTP_GDNRollbackForcedReject` | Force a draft mismatch and prove restore/replay state matches no-MTP decode state. |

The model-gated tests should skip, not fail, when the large GGUFs are absent. The two files are present on the current development machine, but CI should not assume them.

### Unit Test Matrix

Add focused unit tests before enabling end-to-end behavior:

```text
tests/v2/unit/prefix_cache/Test__PrefixBlockHash.cpp
tests/v2/unit/prefix_cache/Test__PrefixCacheKey.cpp
tests/v2/unit/prefix_cache/Test__PrefixStateCacheLRU.cpp
tests/v2/unit/prefix_cache/Test__RamPrefixStorageBackend.cpp
tests/v2/unit/prefix_cache/Test__DiskPrefixStorageBackend.cpp
tests/v2/unit/kernels/Test__IKVCacheLogicalBlockIO.cpp
tests/v2/unit/kernels/Test__HybridPrefixStateIO.cpp
tests/v2/unit/models/Test__MTPWeightInventory.cpp
```

Backend-specific KV I/O tests should cover:

- Empty cache export/import fails cleanly or returns zero bytes according to API contract.
- Non-wrapped logical block export.
- Wrapped logical block export.
- Import into empty cache.
- Import over non-empty cache after clear.
- Truncate to zero, current length, shorter length, and invalid longer length.
- Q16_1 head-major layout.
- TQ asymmetric K/V layout.
- Sharded KV heads with nonzero `kv_head_start`.
- GPU stream import/export and synchronization.

### First Implementation Slice

The safest first slice is small and testable:

1. Add config structs and CLI parsing with prefix/MTP disabled by default.
2. Add `IKVCache` logical block API with default false implementations.
3. Implement CPU `CPURingKVCache` logical export/import/truncate for FP32 and Q8/Q16 formats already used in unit tests.
4. Add `PrefixStateCache` metadata, hash-chain, RAM backend, and LRU without runner integration.
5. Add dense CPU unit tests for export/import and RAM cache.
6. Add tier-0 probe scaffolding that can record state inventory even before full restore support exists.

After that slice passes, add CUDA/ROCm logical I/O, then runner dense RAM-backed prefix cache, then hybrid GDN, then MTP.