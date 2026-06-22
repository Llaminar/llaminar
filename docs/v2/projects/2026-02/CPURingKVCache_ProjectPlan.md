# CPU Ring KV Cache Project Plan

## Objective
Replace the existing CPU KV cache implementation with a ring-buffer-based `CPURingKVCache` that matches GPU ring-cache behavior, improves decode-time locality, and enables better performance for `q8_1` KV precision without regressions in correctness.

## Scope
- In scope:
  - New CPU KV cache implementation (`CPURingKVCache`) for `FP32`, `BF16`, `FP16`, `Q8_1`, `Q16_1`.
  - Integration through existing `IKVCache` and factory paths.
  - Migration of CPU execution paths to ring semantics.
  - Validation via unit, integration/parity, and benchmark testing.
- Out of scope (for this project):
  - GPU ring KV cache redesign.
  - Model architecture changes.
  - Unrelated kernel rewrites outside KV-cache/attention decode interaction.

## Design Principles
1. Preserve external behavior (`IKVCache` contract) while replacing internals.
2. Eliminate memmove-on-evict on CPU (ring pointer + logical indexing only).
3. Keep layout decode-friendly (head-major first-class path).
4. Favor zero-copy views when contiguous, two-segment copy only when wrapped.
5. Keep correctness and parity as hard gates before default switch.

---

## Phase 0 — Requirements and Contract Freeze
### Goals
- Lock the functional contract before implementation.

### Tasks
- Document required semantics for:
  - `append`, `get_k`, `get_v`, `get_kv`, `gather_kv_batched`, `clear`, `evict_oldest`.
  - Cache length accounting and per-sequence state.
  - Sharded TP behavior (`local_n_kv_heads`, `kv_head_start`).
- Define compatibility requirements for current stage/orchestrator call sites.

### Deliverables
- Contract section in this plan treated as source-of-truth.

### Exit Criteria
- No ambiguous behaviors for wrap-around, overflow, or gather ordering.

---

## Phase 1 — Core `CPURingKVCache` Skeleton (Buildable)
### Goals
- Introduce a compile-ready ring cache type with invariants and basic operations.

### Tasks
- Add new files:
  - `src/v2/kernels/cpu/CPURingKVCache.h`
  - `src/v2/kernels/cpu/CPURingKVCache.cpp`
- Implement per-entry metadata:
  - `head` (oldest logical token physical index)
  - `size` (valid token count)
  - `capacity` (`max_seq_len`)
- Implement constructors/factories for all supported precisions and sharded modes.
- Implement `append` with ring writes (no memmove).
- Implement `clear` / `clear_sequence` / `clear_layer` using metadata reset only.

### Deliverables
- New class compiles and links.
- Basic unit coverage for metadata transitions.

### Exit Criteria
- All cache state transitions pass targeted tests.

---

## Phase 2 — Data Layout and Access Semantics
### Goals
- Ensure decode-optimal layout and deterministic logical ordering.

### Tasks
- Standardize head-major ring physical layout:
  - Logical token `t` maps to `phys = (head + t) % capacity`.
- Implement typed row-address helpers for each precision.
- Provide contiguous-range detection:
  - If logical range does not wrap, return direct view where possible.
  - If wrapped, expose two-segment handling for gather path.

### Deliverables
- Internal layout helpers + invariants.

### Exit Criteria
- Deterministic logical ordering validated across wrap boundaries.

---

## Phase 3 — `get_*` and Batched Gather Behavior
### Goals
- Match existing external semantics with ring internals.

### Tasks
- Implement `get_k/get_v/get_kv` with logical-length aware views.
- Implement `gather_kv_batched` in logical order:
  - Fast contiguous memcpy/block-copy path.
  - Wrapped two-segment copy path.
  - Precision-specific copy strategy (`Q8_1`/`Q16_1` block-native).
- Preserve output tensor shape expectations and error handling.

### Deliverables
- Full read-path parity with legacy CPU KV cache behavior.

### Exit Criteria
- Existing stage-level call sites pass without semantic changes.

---

## Phase 4 — Integration Wiring
### Goals
- Route CPU orchestration to `CPURingKVCache` through existing factory pathways.

### Tasks
- Integrate into:
  - `KernelFactory` KV-cache creation path.
  - CPU orchestrator and runner config flow.
  - Any CPU-only append/gather stage assumptions.
- Add temporary feature gate:
  - Env/config switch to toggle legacy vs ring implementation during rollout.

### Deliverables
- End-to-end CPU inference works with ring cache under feature gate.

### Exit Criteria
- No functional regression in basic prefill/decode flows.

---

## Phase 5 — Correctness Validation
### Goals
- Prove parity and stability before perf tuning/default switch.

### Unit Tests
- Ring state transitions:
  - append within capacity
  - append across wrap
  - append overflow with evict behavior
- Logical ordering checks at wrap boundaries.
- `gather_kv_batched` correctness for all precisions.
- Sharded TP correctness (`local_n_kv_heads`, `kv_head_start`).

### Integration Tests
- Decode parity for representative prompts/models.
- Causal/window/GQA/MQA scenarios.
- TP local/global modes where CPU path is active.

### Exit Criteria
- All relevant unit/integration tests pass.

---

## Phase 6 — Performance Tuning (Primary Focus: `q8_1` KV)
### Goals
- Identify and reduce decode-time overhead caused by KV precision path.

### Tasks
- Add/enable timing points around:
  - append
  - gather
  - attention input preparation
  - quant/dequant boundaries
- Optimize hot spots discovered in profiling:
  - reduce redundant copies in wrapped gather
  - reduce/defer dequantization where possible
  - keep block-native operations in `q8_1` paths when feasible
- Validate against baseline benchmarks.

### Benchmarks
- Single socket: `-d cpu:0`
- Two-socket TP: `-d cpu`
- Model set:
  - `qwen2.5-3b-instruct-q8_0.gguf`
  - `Qwen2.5-14B-Instruct.Q8_0.gguf`
- Compare KV precision:
  - `auto/fp16`
  - `q8_1`

### Exit Criteria
- No regression in fp16/auto path.
- Measurable decode improvement for `q8_1` vs current CPU baseline.

---

## Phase 7 — Default Switch and Legacy Removal
### Goals
- Make ring cache the default CPU KV cache implementation.

### Tasks
- Flip default to `CPURingKVCache`.
- Keep fallback toggle for one stabilization window.
- Remove legacy `CPUKVCache` implementation after stabilization.
- Update docs and tests to reflect final architecture.

### Exit Criteria
- Stable CI and benchmark deltas accepted.
- Legacy code path removed.

---

## Risks and Mitigations
- Risk: subtle ordering bugs at wrap boundary.
  - Mitigation: exhaustive wrap boundary unit tests with deterministic fixtures.
- Risk: hidden assumptions in stages expecting contiguous memory.
  - Mitigation: explicit gather semantics and staged rollout via feature gate.
- Risk: optimization attempts may change numerics.
  - Mitigation: parity checks and tolerance-gated validation.

## Suggested File Touch List (Expected)
- `src/v2/kernels/cpu/CPURingKVCache.h`
- `src/v2/kernels/cpu/CPURingKVCache.cpp`
- `src/v2/kernels/KernelFactory.h`
- `src/v2/kernels/KernelFactory.cpp`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp`
- `src/v2/execution/compute_stages/stages/KVCacheGatherStage.cpp`
- `src/v2/execution/compute_stages/stages/AttentionWithKVCacheStage.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `tests/v2/unit/...` (new ring KV cache tests)
- `tests/v2/integration/...` (parity/integration coverage)

## Phase 1 Kickoff Checklist
- [ ] Add `CPURingKVCache` class skeleton and precision template instantiations.
- [ ] Add basic constructors and per-entry metadata state.
- [ ] Implement `append`/`clear` operations with ring pointer updates.
- [ ] Add initial unit tests for state transitions.
- [ ] Ensure integration build compiles with feature-gated selection.
