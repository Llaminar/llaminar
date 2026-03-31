# Coherence State Machine + TransferEngine Migration Plan

**Status**: ✅ COMPLETE  
**Goal**: Unify tensor coherence bookkeeping under an explicit state machine, then make `TransferEngine` the single authority for all data movement. `TensorBase::ensureOnDevice()` and `ensureOnHost()` become thin facades.

### Completion Summary

All 8 phases implemented and verified:
- **Phase 1**: `transitionToWithEvent()` API — build verified (1017/1017)
- **Phase 2**: All ~27 production `mark_device_dirty*()` callers migrated — build verified (553/553)
- **Phase 3**: GpuCoherence RAII wrappers updated — build verified (12/12)
- **Phase 4**: CPUTensors.h bypass fixed (mark_device_dirty delegates, is_on_device uses accessors)
- **Phase 5**: All boolean reads replaced with state queries — build verified (545/545)
- **Phase 6-7**: TransferEngine `uploadFull()`/`downloadFull()` implemented; `ensureOnDevice()` (420→40 LOC) and `ensureOnHost()` (280→12 LOC) now delegate — build 544/544, tests 334/334
- **Phase 8**: `host_valid_`/`device_valid_` fields removed from TensorBase, `setCoherenceState_()` simplified, all test helpers updated — build clean, 334/334 unit tests pass, 11/11 coherence tests pass

**Note**: `CPUTensorBase` (separate class hierarchy) retains its own `host_valid_`/`device_valid_` fields. These are vestigial (no production reads) and independent from TensorBase's coherence system. Left as-is for now.

---

## Architecture Before

```
                     ┌──────────────────────────┐
                     │       TensorBase          │
                     │  (god-object coherence)   │
                     ├──────────────────────────┤
                     │ ensureOnDevice()  420 LOC │  ← allocates VRAM, memcpy, BAR, mapped,
                     │ ensureOnHost()    200 LOC │    secondary buffers, events, profiling
                     │ mark_device_dirty*()      │  ← 3 variants, event recording
                     │ host_valid_, device_valid_ │  ← raw booleans, no state machine
                     └──────────────────────────┘
                              ↑
            50+ callers of mark_device_dirty*()
            55+ callers of ensureOnDevice()
            7   callers of ensureOnHost()

    TransferEngine exists but has 1 production caller (cross-vendor PP only)
```

## Architecture After

```
    ┌───────────────────────────────────────────────────────┐
    │                   TransferEngine                       │
    │  (owns ALL data movement: H2D, D2H, D2D, BAR, mapped)│
    ├───────────────────────────────────────────────────────┤
    │ upload(tensor, device)    ← replaces ensureOnDevice    │
    │ download(tensor)          ← replaces ensureOnHost      │
    │ transferActivation(t, d)  ← PP cross-device            │
    │ planTransfer(src, dst)    ← pure logic                 │
    │ execute(request)          ← low-level dispatch         │
    └───────────────────────────────────────────────────────┘
                              ↑
    ┌───────────────────────────────────────────────────────┐
    │                   TensorBase                           │
    │  (coherence bookkeeping only, no data movement)        │
    ├───────────────────────────────────────────────────────┤
    │ coherence_state_          ← TensorCoherenceState enum  │
    │ transitionTo(state, dev)  ← metadata-only              │
    │ transitionToWithEvent()   ← metadata + GPU event       │
    │ ensureOnDevice(dev)       ← thin: TransferEngine call  │
    │ ensureOnHost()            ← thin: TransferEngine call  │
    │ hostValid() / deviceValid()  ← derived from state      │
    └───────────────────────────────────────────────────────┘
```

---

## Inventory Summary

| API                          | Production Sites | Notes                              |
|------------------------------|------------------|------------------------------------|
| `mark_device_dirty*()`       | ~50              | 3 variants, needs consolidation    |
| `ensureOnDevice()`           | ~55              | God method, delegates to TE        |
| `ensureOnHost()`             | 7                | God method, delegates to TE        |
| `transitionTo()`             | 17               | Already migrated (Phase 2 complete)|
| `TransferEngine`             | 1                | Only cross-vendor PP uses it       |
| `host_valid_`/`device_valid_`| 10 reads, 2 writes | Raw booleans, bypass-prone      |

---

## Phase Plan

### Phase 1: `transitionToWithEvent()` API

**What**: Add a new method that combines `transitionTo()` state management with GPU event recording from `mark_device_dirty_with_event()`.

**Files changed**:
- `src/v2/tensors/TensorClasses.h` — declare `transitionToWithEvent()`
- `src/v2/tensors/TensorBase.cpp` — implement; refactor `mark_device_dirty_with_event()` to call it

**Signature**:
```cpp
void transitionToWithEvent(TensorCoherenceState new_state,
                           std::optional<DeviceId> authoritative_dev = std::nullopt,
                           void* stream = nullptr);
```

**Behavior**: Acquires `coherence_mutex_`, calls `setCoherenceState_()`, sets `authoritative_device_`, handles `mapped_needs_sync_`, resolves backend, creates/records GPU event on stream. Then `mark_device_dirty_with_event(stream)` becomes:
```cpp
transitionToWithEvent(is_mapped_ ? MAPPED : DEVICE_AUTHORITATIVE, gpu_device_, stream);
```

### Phase 2: Migrate `mark_device_dirty*()` callers to `transitionTo*()` 

**What**: Replace all ~50 production `mark_device_dirty*()` calls with either `transitionTo()` (no event) or `transitionToWithEvent()` (GPU stream).

**Key caller groups**:
| Group                    | Sites | Migration Target                |
|--------------------------|-------|---------------------------------|
| StageCoherence.cpp       | 2     | `transitionToWithEvent` / `transitionTo` |
| LocalTPContext.cpp        | 12    | `transitionToWithEvent`         |
| CUDARoPEKernelT.h        | 6     | `transitionToWithEvent`         |
| FusedResidualNormStage    | 2     | `transitionToWithEvent`         |
| EmbeddingStage            | 1     | `transitionToWithEvent`         |
| CoherenceTracker.cpp      | 2     | `transitionToWithEvent` / `transitionTo` |
| MPIStager.cpp             | 1     | `transitionToWithEvent`         |
| CrossDomainTransfer.cpp   | 1     | `transitionToWithEvent`         |
| GpuCoherence.h            | 5     | Phase 3 (separate)              |
| TensorSlice.h             | 1     | Delegate wrapper (keep)         |

### Phase 3: GpuCoherence RAII Migration

**What**: Update the `GpuCoherence.h` RAII wrappers to use `transitionTo()` instead of `mark_device_dirty()`.

**Changes**:
- `CoherableTensor` concept: require `transitionTo()` + `ensureOnDevice()`
- `GpuOutput<T>` destructor: call `transitionTo(DEVICE_AUTHORITATIVE, device_)`
- `with_gpu_coherence()` success paths: same
- `GpuCoherenceScope::mark_success()` path: same

### Phase 4: Deprecate Old APIs + Fix CPUTensors Bug

**What**: 
1. Add `[[deprecated]]` to `mark_device_dirty()`, `mark_device_dirty_with_event()`, `mark_device_dirty_flags_only()`
2. Fix `CPUTensors.h` bug where `mark_device_dirty()` override directly writes `device_valid_=true; host_valid_=is_mapped_` bypassing the state machine
3. Fix `CPUTensors.h` `is_on_device()` to use `hostValid()`/`deviceValid()`

### Phase 5: Boolean Reads → State Queries

**What**: Replace all 10 direct reads of `host_valid_`/`device_valid_` in production code with `coherence_state_` queries or `hostValid()`/`deviceValid()` accessors.

**Locations**:
- `TensorBase.cpp`: 8 reads in `ensureOnDevice()` and `ensureOnHost()`
- `CPUTensors.h`: 2 reads in `is_on_device()`
- `FP32Tensor.cpp`: 1 read in `data()` trace logging

### Phase 6: `ensureOnDevice()` Delegates to TransferEngine

**What**: Gut the 420-line `ensureOnDevice()` god method and make it delegate to `TransferEngine`.

**Strategy**: Move logic into TransferEngine in stages:
1. Graph capture fast path → stays in TensorBase (it's a bailout, not a transfer)
2. CPU device fast path → stays in TensorBase (no-op)
3. BAR fast path → `TransferEngine::upload()` handles via `planTransfer()` returning `MAPPED_NOOP` or `BAR_HOST_BOUNCE`
4. Mapped memory fast path → `TransferEngine::upload()` handles via `MAPPED_NOOP`
5. Already-on-device check with event wait → **new** `TransferEngine::ensureReadable()` method
6. Device migration (secondary buffers) → **new** `TransferEngine::migrateDevice()` or stays on tensor
7. VRAM allocation → `TransferEngine::upload()` via `tensor->getOrAllocateDeviceBuffer()`
8. H2D memcpy → `TransferEngine::upload()` already does this
9. Host pinning → stays on tensor (memory management, not transfer)
10. Transfer profiling/tracing → `TransferEngine` already traces

**New TransferEngine methods needed**:
```cpp
/// Full upload with event wait, device migration, allocation, and memcpy.
/// This replaces TensorBase::ensureOnDevice().
TransferResult uploadFull(TensorBase* tensor, DeviceId target_device);
```

**Result**: `ensureOnDevice()` becomes ~20 lines:
```cpp
bool TensorBase::ensureOnDevice(DeviceId target_device) {
    std::lock_guard<std::mutex> lock(coherence_mutex_);
    if (isGraphCaptureActive()) { return handleGraphCapture(target_device); }
    if (!target_device.is_gpu()) { return true; }
    auto result = TransferEngine::instance().uploadFull(this, target_device);
    return result.success;
}
```

### Phase 7: `ensureOnHost()` Delegates to TransferEngine

**What**: Same approach for `ensureOnHost()`. Move mapped sync, BAR D2H, event wait, standard D2H into `TransferEngine::downloadFull()`.

**New TransferEngine method**:
```cpp
/// Full download with event wait, mapped sync, BAR path, and D2H.
/// This replaces TensorBase::ensureOnHost().
TransferResult downloadFull(TensorBase* tensor);
```

### Phase 8: Remove Deprecated Booleans

**What**: Remove `host_valid_` and `device_valid_` member fields entirely. All code now uses `coherence_state_` or `hostValid()`/`deviceValid()` accessors which derive from it.

**Prerequisites**: All reads migrated in Phase 5, all writes go through `setCoherenceState_()`.

---

## Test Gates

After each phase, verify:

| Test Suite | Command | Pass Criteria |
|------------|---------|---------------|
| Unit tests (334+) | `ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel` | All pass |
| Coherence tests | `ctest --test-dir build_v2_integration -R "Coherence" --output-on-failure --parallel` | All pass |
| Integration tests | `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel` | All pass |
| Parity tests | `ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure` | All pass |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Phases 6-7 touch the hot path | TransferEngine already has the logic; we're reorganizing, not rewriting |
| Event recording correctness | transitionToWithEvent reuses exact same logic from mark_device_dirty_with_event |
| CPUTensors bypass bug | Phase 4 fixes it; tests will catch regressions |
| Secondary buffer migration is complex | Keep in TensorBase initially (it's buffer management, not transfer) |
| 50+ callsite migration | Mechanical: `mark_device_dirty()` → `transitionTo(DEVICE_AUTHORITATIVE, dev)` |
