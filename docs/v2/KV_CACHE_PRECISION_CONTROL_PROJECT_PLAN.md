# KV Cache Precision Control Project Plan (FP16 + Q8_1)

## Objective
Implement end-to-end KV cache precision control in V2 with **two selectable modes**:
- `FP16`
- `Q8_1`

Primary goals:
1. Reduce decode-time bandwidth pressure and KV memory footprint.
2. Preserve model quality/parity within acceptable thresholds.
3. Keep implementation generic and backend-consistent (CPU/CUDA/ROCm).

---

## Scope
### In scope
- Precision mode API and config plumbing.
- KV cache storage/allocation for FP16 and Q8_1.
- KV append path for both modes.
- Gather/attention consumption path updates for both modes.
- Memory estimation updates for orchestration/placement.
- Integration + parity + perf validation.

### Out of scope (initial pass)
- Introducing BF16 KV mode.
- New attention algorithm redesign.
- Non-Qwen graph schema redesign.

---

## Design Requirements
1. **Single precision control surface**: one mode selection propagated from runtime config to graph and kernels.
2. **No shape-specific hacks**: behavior must depend on tensor metadata and precision mode only.
3. **Backend parity**: CPU/CUDA/ROCm must expose consistent IKVCache behavior.
4. **Attention compute stability**: compute remains in existing accumulation precision; storage precision only affects cache representation.
5. **Minimal graph disruption**: retain existing stage structure (`KVCacheAppendStage`, `KVCacheGatherStage`, attention stages).

---

## Proposed Implementation Phases

## Phase 1 — Precision Mode API + Plumbing
- Add `KVCachePrecision` enum (e.g. `FP16`, `Q8_1`) in runtime/graph config types.
- Thread the selected mode through:
  - config parser/factory
  - runner initialization
  - graph config
  - stage params where needed
- Preserve current behavior as default mode (to be decided explicitly, likely `FP16` for first rollout).

Deliverable:
- End-to-end mode visible in logs/config and available at stage creation.

## Phase 2 — KV Storage and Append
- Extend KV cache implementations (CPU/CUDA/ROCm ring caches) to allocate/store by selected mode.
- Implement append conversion paths:
  - `FP16`: FP32 -> FP16 pack on append.
  - `Q8_1`: FP32 -> Q8_1 block quantization on append.
- Ensure wrap-around/eviction semantics are unchanged.

Deliverable:
- KV append/retrieve correctness for both modes.

## Phase 3 — Gather and Attention Consumption
- Ensure `get_kv_for_attention` / gather paths present tensors in expected representation for attention.
- Add/adjust conversion/dequant where required:
  - `FP16` path: vectorized half->float load where attention expects float math.
  - `Q8_1` path: dequant path with bounded overhead (prefer batched/decode-friendly access pattern).
- Keep stage interfaces stable where possible.

Deliverable:
- Prefill/decode attention works correctly for both storage modes.

## Phase 4 — Memory Model and Placement
- Update memory estimates (currently FP32-based assumptions in placement/orchestration) to precision-aware KV bytes.
- Validate placement decisions under different KV precision modes.

Deliverable:
- Placement estimates track selected KV precision.

## Phase 5 — Validation and Rollout
- Integration/parity/perf matrix execution.
- Threshold tuning and fallback behavior if needed.
- Documentation updates (runtime flags/config docs and expected tradeoffs).

Deliverable:
- Feature ready for guarded default or opt-in rollout.

---

## Integration Test Coverage Plan (Prefer Existing Files)
The following existing integration tests should be extended first; avoid creating new files unless a true coverage gap remains.

### 1) KV cache backend correctness
- `tests/v2/integration/kernels/rocm/Test__ROCmRingKVCache.cpp`
  - Extend/verify FP16 + add explicit Q8_1 append/retrieve, wrap-around, eviction checks.
- `tests/v2/integration/kernels/cuda/Test__CUDARingKVCacheParity.cpp`
  - Mirror precision-mode coverage (FP16/Q8_1 parity + ring behavior).
- `tests/v2/integration/kernels/rocm/Test__ROCmRingKVCache_Sharding.cpp`
  - Ensure sharded cache metadata and behavior remain correct for FP16 and Q8_1.

### 2) Stage/graph-level KV correctness
- `tests/v2/integration/execution/graph/Test__GraphBatchedKVCache.cpp`
  - Add precision-parameterized batched append/gather tests for FP16 and Q8_1.

### 3) End-to-end parity impact
- `tests/v2/integration/parity/qwen2/Test__Qwen2_SingleDevice_Parity.cpp`
  - Add mode-scoped parity runs for decode/prefill with KV precision set to FP16 and Q8_1.
- `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_Parity.cpp`
  - Add at least one TP parity case per KV mode to catch distributed sensitivity.

### 4) Attention-kernel interaction checks
- `tests/v2/integration/kernels/rocm/Test__ROCmFlashAttentionParity.cpp`
  - Add decode coverage where K/V inputs come from FP16 and Q8_1-backed cache paths.

### 5) MPI sharded behavior sanity
- `tests/v2/integration/utils/mpi/Test__MPI_ShardedKVCache.cpp`
  - Extend to verify per-rank sharded append/gather correctness for FP16 and Q8_1.

---

## Acceptance Criteria
1. **Functional**
- Both modes (`FP16`, `Q8_1`) run prefill+decode without crashes across CPU/CUDA/ROCm paths that support them.
- KV append/gather/wrap-around behavior matches existing semantics.

2. **Quality**
- Parity tests pass with agreed thresholds for decode/prefill in single-device and at least one TP scenario.

3. **Performance**
- Decode throughput shows expected bandwidth-driven improvement or at minimum no significant regression for FP16 mode.
- Q8_1 mode demonstrates net memory benefit with acceptable quality/perf tradeoff.

4. **Estimation/placement**
- KV memory estimation reflects selected precision mode in orchestration/placement logic.

---

## Risks and Mitigations
- **Q8_1 decode overhead** (dequant can erase bandwidth gains)
  - Mitigation: profile gather/dequant placement; prefer contiguous/vectorized dequant strategy.
- **Parity drift at long context**
  - Mitigation: dedicated long-context decode parity runs and per-layer diagnostics.
- **Backend divergence**
  - Mitigation: common enum/plumbing and mirrored integration tests across ROCm/CUDA/CPU.

---

## Suggested Execution Order
1. API/plumbing
2. FP16 storage+append+consume path
3. Q8_1 storage+append+consume path
4. Memory-estimation updates
5. Integration + parity + perf matrix

This sequencing gets a lower-risk high-value mode (FP16) landed early while building full dual-mode support in the same project arc.
