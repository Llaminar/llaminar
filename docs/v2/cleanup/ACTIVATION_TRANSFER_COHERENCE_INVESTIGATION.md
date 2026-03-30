# Activation Transfer & Tensor Coherence Investigation Brief

**Date**: March 30, 2026  
**Context**: HybridPPTP test debugging session on `tensor-parallel` branch  
**Hardware**: 2x AMD Mi50 (ROCm:0/1) + 1x NVIDIA 3090 (CUDA:0)  
**Model**: Qwen2.5-0.5B — 24 layers, PP split at layer 12, TP(2xROCm) + CUDA

---

## Bugs Found and Fixed

### 1. Same-Device Copy Crash in NCCL/RCCL Backends

- **Symptom**: Crash when `copy()` called with source and destination on the same device
- **Root Cause**: Both `NCCLBackend::copy()` and `RCCLBackend::copy()` unconditionally required a coordinator communicator, even for trivial same-device copies
- **Fix**: Added fast path using `cudaMemcpySameDevice()`/`hipMemcpySameDevice()` when source == destination device
- **Brittleness signal**: The collective backends are used for point-to-point transfers they weren't designed for. No single "transfer" abstraction decides how to move data.

### 2. MDO `kv_cache_scale` Default Mismatch

- **Symptom**: Decode parity ~0.77 cosine similarity (threshold 0.80) for PP configurations
- **Root Cause**: `MultiDeviceOrchestrator::Config::kv_cache_scale` defaulted to `1.0f` while every other path in the codebase uses `256.0f`
- **Fix**: Changed default in `MultiDeviceOrchestrator.h` to `256.0f`
- **Brittleness signal**: Critical numerical parameters are duplicated across config structs with no shared constant or validation. A new orchestration path silently picked up the wrong default.

### 3. BAR IOMEMORY `cudaMemcpy(D2D)` Produces NaN

- **Symptom**: NaN at `layer12_attn_norm` — the first layer of PP stage 1 on CUDA:0. The PP-transferred hidden state was all NaN.
- **Root Cause**: `cudaMemcpy(DeviceToDevice)` with IOMEMORY-registered BAR pointers as source silently produces NaN rather than returning an error. The PP copy path in both `Qwen2Graph.cpp` (fresh-build) and `DeviceGraphOrchestrator.cpp` (cached) used D2D memcpy from BAR source to CUDA destination.
- **Fix**: Host-bounce pattern for BAR sources: `data()` (D2H via BAR read) → `memcpy` (host→host) → `ensureOnDevice()` (H2D)
- **Brittleness signal**: PP transfer logic is duplicated in two completely separate code paths (fresh graph build vs. cached decode graph) with no shared implementation. The BAR tensor type has three different pointer types (`hip_staging_ptr_`, `bar_rocm_ptr_`, `bar_cuda_device_ptr_`) and the correct one to use depends on which operation you're doing, with no type-level enforcement.

### 4. BAR `hipMemcpy(D2D)` Staging→BAR Produces Corrupt Data Under Concurrent Access

- **Symptom**: `ensureOnHost()` for BAR-backed tensors returned wrong values — one ROCm device's tensor contained the other device's layer 11 output instead of its own embedding output. Direct D2H from staging VRAM showed correct values.
- **Root Cause**: `TensorBase::ensureOnHost()` BAR path did staging→BAR (hipMemcpy D2D) → BAR→host (memcpy). Under concurrent multi-device TP access (two ROCm devices running embedding in parallel via TPWorkerPool), the `hipMemcpy(D2D)` to BAR-registered memory produced stale/corrupt data.
- **Fix**: Bypass BAR intermediate entirely — direct `hipMemcpy(DeviceToHost)` from staging VRAM to host memory in `TensorBase.cpp`
- **Brittleness signal**: `ensureOnHost()` has five distinct code paths (mapped memory w/ event sync, BAR-backed ROCm, BAR-backed non-ROCm, standard GPU D2H, CPU no-op) selected by runtime state bits. Adding BAR support created a path that was never tested under concurrent multi-device access.

### 5. LocalPPTestRunner Bypassing Production Code

- **Symptom**: Tests passed with the hand-rolled `LocalPPTestRunner` but failed with production MDO TP_PP path
- **Root Cause**: `LocalPPTestRunner` was a test-only PP wrapper that reimplemented PP handoffs differently from MDO's production `forwardPP()`. It masked real bugs in the production path.
- **Fix**: Replaced `LocalPPTestRunner` path in `Qwen2ParityTestBase.h` with unified MDO(AUTO) for both PP and TP_PP
- **Brittleness signal**: The existence of a separate test-only PP implementation indicates the production path wasn't being tested by integration tests.

---

## Systemic Observations

### No Unified Transfer Abstraction

PP activation transfer is implemented in at least four places:

1. `Qwen2Graph::buildPartialForwardGraph()` — inline CPU memcpy or GPU BAR host-bounce (fresh graph build)
2. `DeviceGraphOrchestrator` cached decode path — BAR host-bounce for cached graphs
3. `MultiDeviceOrchestrator::forwardPP()` → `PPContext::transfer()` — the "official" PP transfer
4. Collective backends (`NCCLBackend::copy()`, `RCCLBackend::copy()`) — repurposed for point-to-point

Each path has its own logic for handling BAR vs. non-BAR, GPU vs. CPU, with no shared primitives.

### Coherence State is Implicit and Fragile

`TensorBase` tracks coherence via `host_valid_`, `authoritative_device_`, `is_bar_backed_`, `mapped_needs_sync_`, `device_completion_event_`, `hip_staging_ptr_`, `bar_rocm_ptr_`, `bar_cuda_device_ptr_`, and `gpu_data_ptr_`. The correct behavior of any operation depends on the combination of these 8+ state variables. There's no state machine or enum that captures the valid states — it's all `if/else` chains.

### BAR Tensor Has Too Many Roles

A single BAR-backed `FP32Tensor` simultaneously serves as:

- ROCm compute target (via `hip_staging_ptr_`)
- Cross-device transfer medium (via `bar_rocm_ptr_` / `bar_cuda_device_ptr_`)
- Host-readable buffer (via `bar_rocm_ptr_` mmap)
- Regular tensor with `data()` / `mutable_data()` coherence contract

The rules for which pointer to use when are entirely in programmer knowledge, not in the type system.

### PP Copy Path Duplication

The fresh-build path (Qwen2Graph) and cached-decode path (DGO) each have independent PP copy logic. Bug #3 had to be fixed in both places independently, and a fix in one would not have fixed the other.

### Config Propagation is Copy-Based

MDO creates nested MDO(TP) configs by copying and modifying fields (`nested_config.kv_cache_scale`, `nested_config.use_bar_backed_hidden`, etc.). Defaults in nested config structs can silently diverge from the parent, as happened with `kv_cache_scale = 1.0f`.

### No Integration Test Coverage for Production TP_PP Path

The `LocalPPTestRunner` test harness had been passing while the production MDO TP_PP path had multiple fatal bugs (NaN from BAR D2D, wrong kv_cache_scale). The tests were testing the wrong thing.

---

## Files Modified

| File | Change |
|------|--------|
| `src/v2/collective/backends/NCCLBackend.cpp` | Same-device copy fast path |
| `src/v2/collective/backends/RCCLBackend.cpp` | Same-device copy fast path |
| `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h` | `kv_cache_scale` default 1.0→256.0 |
| `src/v2/models/qwen/Qwen2Graph.cpp` | BAR host-bounce for PP copy |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | BAR host-bounce for cached PP copy |
| `src/v2/tensors/TensorBase.cpp` | Direct D2H for BAR-backed `ensureOnHost()` |
| `tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h` | Unified MDO(AUTO) for PP/TP_PP tests |
