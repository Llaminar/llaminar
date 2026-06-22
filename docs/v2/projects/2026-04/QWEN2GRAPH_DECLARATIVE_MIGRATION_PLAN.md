# Qwen2Graph Declarative Migration Plan

**Status**: PLANNING  
**Created**: April 2026  
**Goal**: Transform `Qwen2Graph.cpp` from 52% imperative to >90% declarative by extracting runtime logic into the components that own it — executor, stages, config builders, and post-build passes.

---

## Audit Summary

**File**: `src/v2/models/qwen/Qwen2Graph.cpp` — 2,467 lines, 29 methods  
**Classification**: 5 declarative (17%) · 9 mixed (31%) · 15 imperative (52%)

| Anti-Pattern | Count | Worst Offender |
|---|---|---|
| `env.execution.exec_*` checks | 24 | `buildAttentionGraph` (13) |
| `InferenceMode` / `isHybridQ16()` | 9 | `buildPartialForwardGraph` (3), `buildUnifiedPipelineGraph` (3) |
| KV cache runtime queries | 6 | `buildAttentionGraph` L1801-1835 (5) |
| Mutable `config_` mutation | 7 | `buildUnifiedPipelineGraph` L960-1006 (2) |
| Device-type branching | 4 | `buildAttentionGraph` L1530 (2), `buildFFNGraph` L2026 (1) |
| Manual memory ops | 3 | `buildPartialForwardGraph` L617-628 |
| Boilerplate config wiring | 73 lines | `buildForwardGraphFromSchema` L1201-1290 |
| Dead code | 1 | `buildFFNGraph` L2030 (`backend` computed but never used) |

### Root Cause

The graph builder is doing work that belongs in 4 other components:

| Logic in graph builder | Should live in |
|---|---|
| Execution skip flags (`exec_rmsnorm`, `exec_gemm`, etc.) | `DeviceGraphExecutor` via `StageRunPolicy` |
| Device/backend selection | Stage constructors (via `DeviceId`) |
| KV cache mode detection (prefill/decode/batched) | Pre-computed `AttentionMode` in `ForwardInput` |
| TP collective injection | Post-build graph pass |
| Buffer layout selection (`isHybridQ16`) | Pre-computed in `GraphConfig` at config time |
| Config translation boilerplate | `GraphResolverConfig::from()` factory |
| PP tensor copying | `TensorCopyStage` |

---

## Phase 1: Kill execution-flag branching (24 sites)

**Severity**: HIGH — largest single source of imperative logic  
**Risk**: LOW — `StageRunPolicy` already exists and is used by the executor  
**Net LOC**: -60 to -80 (remove 24 `if (env.execution.exec_*)` blocks)

### Problem

Every stage creation is wrapped in `if (env.execution.exec_rmsnorm)` / `if (env.execution.exec_gemm)` / etc. These debug flags exist to selectively disable stages during development, but they currently make the graph non-deterministic — the DAG topology itself changes based on environment variables.

**Sites** (24 total):

In `buildAttentionGraph` (13):
- L1540: `if (env.execution.exec_rmsnorm)` — attn_norm
- L1579: `if (env.execution.exec_gemm && layer.wq && layer.wk && layer.wv)` — QKV proj
- L1618: `if (env.execution.exec_gemm)` — qkv→norm dependency
- L1665: `if (env.execution.exec_gemm && layer.wq)` — q_norm→qkv dependency
- L1682: `if (env.execution.exec_gemm && layer.wk)` — k_norm→qkv dependency
- L1687: `if (env.execution.exec_rope)` — RoPE
- L1710-1714: 3× `if (env.execution.exec_gemm)` — RoPE dependency wiring
- L1724: `if (env.execution.exec_attention)` — attention stage
- L1901: `if (env.execution.exec_attention)` — redundant nested check
- L1913-1916: 4-way conditional dependency chain
- L1924: `if (env.execution.exec_gemm && layer.wo)` — Wo projection
- L1953: `if (env.execution.exec_attention)` — Wo→attn dependency
- L1958: `if (env.execution.exec_gemm && layer.wo && ...)` — TP allreduce

In `buildFFNGraph` (10):
- L2038: `if (env.execution.exec_rmsnorm)` — FFN norm
- L2065: `if (env.execution.exec_gemm && layer.gate_proj && layer.up_proj)` — gate/up
- L2093: `if (env.execution.exec_rmsnorm)` — gate_up→norm dependency
- L2102: `if (env.execution.exec_gemm && layer.down_proj)` — down_proj
- L2127-2133: 2-way swiglu exec check
- L2143: TP allreduce
- L2161: `if (env.execution.exec_residual && !skip_ffn_residual)` — residual add
- L2177-2183: 2× nested env flag checks for residual dependency

In `buildForwardGraphFromSchema` (1):
- L1245: `config.exec_policy = ExecutionPolicyFlags::fromDebugEnv()`

### Solution

**Always build the complete graph.** Move skip logic to the executor.

1. Remove all `if (env.execution.exec_*)` guards from `buildAttentionGraph` and `buildFFNGraph`. Always create every stage and wire every dependency.
2. The executor's `StageRunPolicy` (already exists) inspects `ExecutionPolicyFlags` and skips stages at execution time. This is already how the schema/resolver path works.
3. Keep the `layer.wq && layer.wk && layer.wv` null-weight guards — those are legitimate (a weight genuinely missing from the model file). Factor them to early validation.

### Changes

| File | Change |
|---|---|
| `src/v2/models/qwen/Qwen2Graph.cpp` | Remove 24 `env.execution.exec_*` conditionals. Always create nodes/dependencies. |
| (none — executor already has `StageRunPolicy`) | Verify `StageRunPolicy` handles all stage types currently gated. |

### Acceptance

- `buildAttentionGraph` and `buildFFNGraph` have zero `env.execution.exec_*` references.
- `grep -c 'env.execution.exec_' src/v2/models/qwen/Qwen2Graph.cpp` returns 0 (excluding the `fromDebugEnv()` call in `buildForwardGraphFromSchema` which is the correct location for that).
- All unit tests pass. Parity unchanged.
- `LLAMINAR_EXEC_ATTENTION=0` still skips attention execution (via executor, not graph structure).

---

## Phase 2: Pre-compute buffer layout selection (9 sites)

**Severity**: MEDIUM — scattered repetitive branching  
**Risk**: LOW — pure config-time computation  
**Net LOC**: -30 to -40

### Problem

Nine sites construct `InferenceMode(config_.activation_precision)` and call `.isHybridQ16()` to choose between `residual` vs `current_hidden` buffers or compute byte sizes for Q16_1 blocks. Two redundant constructions shadow each other at L1542 and L1639 in `buildAttentionGraph`.

**Sites** (9 total):

In `buildPartialForwardGraph` (3):
- L590-593: Selects working buffer
- L604-610: Computes copy_bytes for Q16_1 vs FP32

In `buildUnifiedPipelineGraph` (3):
- L931-933: Selects embed output buffer
- L1040: Selects PP transfer buffer
- L1107: Selects final_norm input buffer

In `buildAttentionGraph` (3):
- L1542: First `InferenceMode` construction + `isHybridQ16` for FusedResidualNorm
- L1639: Second redundant `InferenceMode` construction (shadows L1542)
- (1 more in attention buffer selection)

### Solution

Pre-compute `isHybridQ16` and derived buffer selections once in `GraphConfig` at construction time.

1. Add `bool is_hybrid_q16` field to `GraphConfig` (set by `IGraphConfigBuilder`).
2. Add `BufferId primary_hidden_buffer` and `BufferId residual_buffer` to `GraphConfig` — pre-resolved at config time.
3. Replace all 9 `InferenceMode` construction + `isHybridQ16()` call sites with `config_.is_hybrid_q16` or `config_.primary_hidden_buffer`.
4. Delete the `#include "../../execution/config/InferenceMode.h"` from `Qwen2Graph.cpp` if no references remain.

### Changes

| File | Change |
|---|---|
| `src/v2/models/GraphTypes.h` | Add `bool is_hybrid_q16` and buffer selection fields to `GraphConfig`. |
| `src/v2/models/qwen/Qwen2GraphConfigBuilder.cpp` | Set `is_hybrid_q16` from `activation_precision`. |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Replace 9 `InferenceMode(...).isHybridQ16()` with `config_.is_hybrid_q16`. Remove `InferenceMode.h` include. |

### Acceptance

- `grep -c 'isHybridQ16\|InferenceMode' src/v2/models/qwen/Qwen2Graph.cpp` returns 0.
- All unit tests + parity pass.

---

## Phase 3: Replace mutable config mutation with parameters (2 sites)

**Severity**: HIGH — exception-unsafe, non-reentrant  
**Risk**: LOW — method signature change  
**Net LOC**: -5 (net neutral, but eliminates hazard)

### Problem

`buildUnifiedPipelineGraph()` at L960-1006 temporarily mutates `config_.default_device` and `config_.local_tp_ctx`, calls sub-methods, then restores the originals. If any sub-method throws, the config is left in a corrupted state. The restoration at L1006 is also easy to miss when adding new early-returns.

### Solution

Pass `DeviceId device` and `ILocalTPContext* tp_ctx` as explicit parameters to all sub-methods called from the PP stage loop, instead of writing them to `config_` and reading them back.

1. Add `DeviceId device` and `ILocalTPContext* tp_ctx` parameters to `buildLayerGraph()`, `buildAttentionGraph()`, and `buildFFNGraph()`.
2. These methods read the device from the parameter, not from `config_.default_device`.
3. Remove the save/restore pattern from `buildUnifiedPipelineGraph()`.

### Changes

| File | Change |
|---|---|
| `src/v2/models/qwen/Qwen2Graph.h` | Add optional `DeviceId` / `ILocalTPContext*` params to layer/attention/FFN builder signatures (defaulting to `config_` values for backward compat). |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Thread device/tp_ctx through `buildUnifiedPipelineGraph` → `buildLayerGraph` → `buildAttentionGraph` / `buildFFNGraph`. Remove L960-1006 save/restore. |

### Acceptance

- `grep -c 'config_.default_device =' src/v2/models/qwen/Qwen2Graph.cpp` returns 0 (no writes to `config_` fields in method bodies).
- `buildUnifiedPipelineGraph` has no save/restore pattern.
- PP parity tests pass.

---

## Phase 4: Extract device-type branching to stages (4 sites + 1 dead code)

**Severity**: MEDIUM — wrong abstraction layer  
**Risk**: LOW — stages already have device context  
**Net LOC**: -20

### Problem

Four sites compute `ComputeBackend backend` from `device.is_cuda()` / `device.is_rocm()`:
- `buildAttentionGraph` L1530-1536
- `buildAttentionGraph` L1870 (GPU-specific KV cache read policy)
- `buildFFNGraph` L2026-2032

Plus one dead-code instance: `buildFFNGraph` L2030 computes `backend` but never uses it.

### Solution

1. Delete the `backend` derivation from `buildAttentionGraph` and `buildFFNGraph`. Stages already receive `DeviceId` and determine their own backend internally.
2. For the KV cache read policy at L1870 (`read_kv_from_cache = device.is_gpu() && ...`), move this into the attention stage constructor — it should decide its own read strategy based on its device.
3. Delete the dead `backend` variable in `buildFFNGraph`.

### Changes

| File | Change |
|---|---|
| `src/v2/models/qwen/Qwen2Graph.cpp` | Remove backend derivation blocks. Remove `ComputeBackend.h` include if no longer needed. Kill dead `backend` in `buildFFNGraph`. |
| `src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp` (or equivalent) | Move `read_kv_from_cache` device-type decision into stage constructor. |

### Acceptance

- `grep -c 'is_cuda\|is_rocm\|is_gpu' src/v2/models/qwen/Qwen2Graph.cpp` returns 0.
- All unit tests + parity pass.

---

## Phase 5: Pre-compute attention mode (6 KV cache query sites)

**Severity**: HIGH — runtime queries make graph non-deterministic  
**Risk**: MEDIUM — touches the attention stage's core branching logic  
**Net LOC**: -30 to -40

### Problem

`buildAttentionGraph` queries `kv_cache->get_cached_tokens()` at build time to determine whether the model is in prefill, single-decode, or batched-decode mode (L1801-1835). This makes the graph structure depend on runtime KV cache state.

**Sites** (6 total):
- L1238-1242 in `buildForwardGraphFromSchema`: `kv_cache->get_cached_tokens(0,0)` for config
- L1730: `if (kv_cache)` gates KVCacheAppendStage
- L1801-1806: `kv_cache->get_cached_tokens(kv_local_layer, 0)` — prefill vs decode
- L1807-1812: `if (cached_tokens > 0 && batch_size == 1)` — single decode K/V from cache
- L1815-1826: `if (cached_tokens > 0 && batch_size > 1)` — batched decode with gather fallback
- L1870: KV cache precision check for `read_kv_from_cache`

### Solution

Pre-compute an `AttentionMode` enum in `ForwardInput` (the per-call input struct) rather than querying cache state.

1. Add `AttentionMode { PREFILL, DECODE, BATCHED_DECODE }` to `ForwardInput` or `GraphConfig`.
2. The **orchestrator** (DGO) computes this from `seq_len`, `batch_size`, and `kv_cache->get_cached_tokens()` before calling the graph builder.
3. `buildAttentionGraph` switches on a clean enum instead of raw cache queries.
4. KV cache pointer is still passed for stage parameter setup — but graph **topology** doesn't change based on cache state.

### Changes

| File | Change |
|---|---|
| `src/v2/models/GraphTypes.h` | Add `enum class AttentionMode` to `ForwardInput`. |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | Compute `AttentionMode` from seq_len/batch/cache before graph build. |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Replace 6 `kv_cache->get_cached_tokens()` queries with `forward_input.attention_mode` switch. |

### Acceptance

- `grep -c 'get_cached_tokens' src/v2/models/qwen/Qwen2Graph.cpp` returns 0.
- Graph structure is deterministic for a given `AttentionMode` (same enum → same DAG).
- All unit tests + parity pass.

---

## Phase 6: Replace PP tensor copy with TensorCopyStage (3 manual memory ops)

**Severity**: MEDIUM — inline eager memory operations in graph builder  
**Risk**: MEDIUM — new stage type  
**Net LOC**: -20 (replace ~40 lines with stage creation)

### Problem

`buildPartialForwardGraph` at L617-628 performs `std::memcpy` + `ensureOnDevice` + `transitionTo(DEVICE_AUTHORITATIVE)` directly in the graph builder. This is an imperative eager copy that should be a compute stage.

### Solution

Create a `TensorCopyStage` that copies tensors between devices as part of the graph, replacing the inline memory operations.

1. Create `src/v2/execution/compute_stages/stages/TensorCopyStage.h/cpp` — a simple stage that copies `src` → `dst` with proper coherence.
2. Replace the L617-628 block in `buildPartialForwardGraph` with a `TensorCopyStage` node in the graph.
3. The existing `LocalPPTransferStage` may already cover this — evaluate whether to reuse it or create a simpler variant.

### Changes

| File | Change |
|---|---|
| `src/v2/execution/compute_stages/stages/TensorCopyStage.h/cpp` | New stage (or verify `LocalPPTransferStage` suffices). |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Replace L617-628 memcpy block with stage node creation. |

### Acceptance

- `grep -c 'std::memcpy\|mutable_data\|transitionTo' src/v2/models/qwen/Qwen2Graph.cpp` returns 0 (in graph-building methods).
- PP tests pass.

---

## Phase 7: Collapse GraphResolverConfig boilerplate (73 lines)

**Severity**: LOW — purely cosmetic  
**Risk**: LOW — mechanical refactor  
**Net LOC**: -60

### Problem

`buildForwardGraphFromSchema` at L1201-1290 has 73 lines of `config.X = config_.X` and `tensors.buffers["name"] = buffers_.field`. This is mechanical wiring that should be a factory method.

### Solution

1. Add `static GraphResolverConfig GraphResolverConfig::from(const GraphConfig&, const ForwardInput&)` factory.
2. Add `static TensorContext TensorContext::from(const Buffers&, const Weights&, int layer_idx)` factory.
3. Replace the 73-line block with two factory calls.

### Changes

| File | Change |
|---|---|
| `src/v2/execution/local_execution/graph/GraphResolver.h` | Add `GraphResolverConfig::from()` static factory. |
| `src/v2/execution/local_execution/graph/GraphResolver.cpp` | Implement factory. |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Replace L1201-1290 with `auto config = GraphResolverConfig::from(config_, forward_input);` + `auto tensors = TensorContext::from(buffers_, weights_, layer_idx);`. |

### Acceptance

- `buildForwardGraphFromSchema` is ≤30 lines.
- All unit tests pass (including `Test__GraphResolver.cpp`).

---

## Phase 8: Extract TP AllReduce injection as post-build pass

**Severity**: MEDIUM — interleaved collective logic  
**Risk**: MEDIUM — requires graph mutation after construction  
**Net LOC**: -40 to -50

### Problem

`buildAttentionGraph` (L1958+) and `buildFFNGraph` (L2143+) both conditionally inject `TPAllreduceStage` nodes with backend selection logic (`needsTPAllreduce()`, BAR tensor registration). This collective injection is interleaved with stage creation, making the graph builder responsible for TP topology decisions.

### Solution

Create a `CollectiveStageInjector` post-build pass that:
1. Walks the completed `ComputeGraph` looking for stages annotated as `ROW_PARALLEL` or `INPUT_PARALLEL`.
2. Inserts `TPAllreduceStage` nodes after them, wired to the correct collective backend.
3. The graph builder just creates the compute stages and annotates their parallelism mode.

### Changes

| File | Change |
|---|---|
| `src/v2/execution/local_execution/graph/CollectiveStageInjector.h/cpp` | New: post-build pass that inserts AllReduce stages. |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Remove inline AllReduce injection from `buildAttentionGraph` and `buildFFNGraph`. Add parallelism annotations to Wo/down_proj stages. |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | Call `CollectiveStageInjector::inject(graph)` after graph build, before execution. |

### Acceptance

- `grep -c 'TPAllreduceStage\|needsTPAllreduce' src/v2/models/qwen/Qwen2Graph.cpp` returns 0.
- TP parity tests pass.

---

## Phase 9: Eliminate fragile node-name string coupling

**Severity**: LOW — localized but fragile  
**Risk**: LOW — API addition  
**Net LOC**: -10

### Problem

`buildUnifiedPipelineGraph` at L1010-1020 searches for named leaf nodes like `"attn_residual"` / `"ffn_residual"` using `std::find` over `getLeafNodes()`, with a silent fallback to `.front()` if not found. This pattern is repeated 4+ times.

### Solution

`ComputeGraph::merge()` should return the merged graph's terminal node(s) directly, eliminating the need for post-hoc name search.

1. Change `ComputeGraph::merge(subgraph)` to return a `MergeResult { std::string terminal_node; }`.
2. Callers use the returned terminal node instead of searching by name.
3. As a fallback for cases where merge isn't the API entry, add `ComputeGraph::lastAddedNode()` which returns the most recently added leaf.

### Changes

| File | Change |
|---|---|
| `src/v2/execution/local_execution/graph/ComputeGraph.h/cpp` | Add `MergeResult` return from `merge()`. Add `lastAddedNode()`. |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Replace name-search + `.front()` fallback with `merge_result.terminal_node`. |

### Acceptance

- No `getLeafNodes()` + `std::find` + `.front()` fallback patterns remain.
- All unit tests pass.

---

## Phase Order and Dependencies

```
Phase 1 ─── Kill exec flags (24 sites)
             │
Phase 2 ─── Pre-compute buffer layout (9 sites)
             │
Phase 3 ─── Parameter threading (mutable config)
             │
Phase 4 ─── Device-type branching to stages
             │
Phase 5 ─── Pre-compute AttentionMode ──── Phase 8: TP AllReduce post-pass
             │                                        │
Phase 6 ─── TensorCopyStage (PP)                     │
             │                                        │
Phase 7 ─── GraphResolverConfig boilerplate           │
             │                                        │
Phase 9 ─── Node-name coupling                        │
             │                                        │
             └──────── ALL DONE ──────────────────────┘
```

Phases 1-4 are independent and can be done in any order. Phase 5 should precede Phase 8 (both touch the attention builder). Phases 6, 7, 9 are independent.

| Phase | Description | Risk | Est. Lines Changed | Dependency |
|---|---|---|---|---|
| 1 | Kill execution-flag branching | LOW | -60 to -80 | None |
| 2 | Pre-compute buffer layout | LOW | -30 to -40 | None |
| 3 | Parameter threading (mutable config) | LOW | ~0 net | None |
| 4 | Device branching to stages | LOW | -20 | None |
| 5 | Pre-compute AttentionMode | MEDIUM | -30 to -40 | None |
| 6 | TensorCopyStage | MEDIUM | -20 | None |
| 7 | GraphResolverConfig boilerplate | LOW | -60 | None |
| 8 | TP AllReduce post-pass | MEDIUM | -40 to -50 | Phase 5 |
| 9 | Node-name coupling | LOW | -10 | None |

**Estimated total reduction**: 270-320 lines from 2,467 → ~2,150-2,200 lines  
**Estimated declarative coverage**: 52% imperative → <15% imperative

---

## Inter-Phase Acceptance Gate

Every phase must pass the following gate **before** starting the next phase. This ensures incremental correctness — no phase is allowed to leave the tree red.

### Gate Steps

**Step 1: Unit tests (parallel)**

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
```

All unit tests must pass. No exceptions.

**Step 2: Integration tests (sequential)**

Run each integration parity suite one at a time. These are heavyweight tests that load models and compare against PyTorch ground truth.

```bash
# 2a. Qwen2 SingleDevice parity (CPU + CUDA)
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_SingleDevice" --output-on-failure --verbose

# 2b. Qwen3 SingleDevice parity (CPU + CUDA)
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen3_SingleDevice" --output-on-failure --verbose

# 2c. Qwen2 LocalTP parity (multi-device tensor parallelism)
#     NOTE: Pre-existing PCIeBAR test failures may be ignored.
#     All other LocalTP tests must pass.
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_LocalTP" --output-on-failure --verbose

# 2d. Qwen2 LocalPP parity (pipeline parallelism)
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_LocalPP" --output-on-failure --verbose
```

### Pass Criteria

| Suite | Requirement |
|---|---|
| `V2_Unit_*` | 100% pass |
| `V2_Integration_Parity_Qwen2_SingleDevice` | 100% pass |
| `V2_Integration_Parity_Qwen3_SingleDevice` | 100% pass |
| `V2_Integration_Parity_Qwen2_LocalTP` | All pass except pre-existing PCIeBAR failures |
| `V2_Integration_Parity_Qwen2_LocalPP` | 100% pass |

If any suite regresses (new failures not present before the phase), the phase must be fixed before proceeding.

---

## Final Acceptance Gate (all phases complete)

```bash
# Zero imperative patterns remaining:
grep -c 'env.execution.exec_' src/v2/models/qwen/Qwen2Graph.cpp         # → 0
grep -c 'isHybridQ16\|InferenceMode' src/v2/models/qwen/Qwen2Graph.cpp  # → 0
grep -c 'get_cached_tokens' src/v2/models/qwen/Qwen2Graph.cpp           # → 0
grep -c 'is_cuda\|is_rocm\|is_gpu' src/v2/models/qwen/Qwen2Graph.cpp   # → 0
grep -c 'std::memcpy' src/v2/models/qwen/Qwen2Graph.cpp                 # → 0
grep -c 'config_.default_device =' src/v2/models/qwen/Qwen2Graph.cpp    # → 0

# Full inter-phase gate (must also pass):
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_SingleDevice" --output-on-failure --verbose
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen3_SingleDevice" --output-on-failure --verbose
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_LocalTP" --output-on-failure --verbose
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_LocalPP" --output-on-failure --verbose
```
