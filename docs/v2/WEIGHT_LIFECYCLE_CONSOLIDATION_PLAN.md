# Weight Lifecycle Consolidation Plan

**Date**: 2025-04-11
**Status**: Proposed
**Author**: Copilot + dbsanfte

---

## Problem Statement

Weight management is a fragile mess of spaghetti. No single class owns the full
load → shard → pack → upload → release lifecycle. The logic is smeared across
5+ files, with duplicated routines, dead code, and code paths that silently skip
critical steps (GEMM packing, host release) depending on which factory function
created the runner.

### Symptoms

1. **24.9 GB host RSS** for a 4.2 GB model — triple weight copies persisted
   because the multi-device path never called `releaseAllHostWeightData()`.

2. **Three copies of packing logic**: `WeightManager::packWeight()`,
   `WeightPreloader::packWeight()`, and the `upload_job` lambda inside
   `WeightManager::packGemmWeights()`. All three are nearly identical.

3. **Two factory paths with different weight behavior**:
   - `createInferenceRunner()` → calls `preloadAndUploadWeights()` → packs,
     uploads, releases. Works.
   - `createTestableInferenceRunner()` → calls `buildWeights()` only → no
     packing, no upload, no release. Broken for production multi-device.

4. **RankOrchestrator** had to manually call `releaseAllHostWeightData()`
   because the factory it delegates to (`createTestableInferenceRunner`) doesn't.
   This was a bug until last session.

5. **`packGemmWeights()` only iterates `cache_`**, not `per_device_cache_`. For
   LOCAL TP, the sharded tensors live in `per_device_cache_`. The single-device
   path works because `cache_` IS the right place. Multi-device works by
   accident — GEMM kernels are created lazily on first inference from whichever
   tensor `getWeightForDevice()` returns.

---

## Current Architecture (What's Wrong)

```
                            WHO DOES WHAT?
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   ModelLoader          WeightManager        InferenceRunner  │
    │   ───────────          ─────────────        ───────────────  │
    │   Loads GGUF           Stores in cache_     Calls preload    │
    │   Creates tensors      Shards on demand     Calls pack       │
    │                        Packs GEMM           Calls upload     │
    │                        Uploads non-GEMM     Calls release    │
    │                        Clones for devices                    │
    │                        Releases host data                    │
    │                                                              │
    │   WeightPreloader      InferenceRunnerFactory                │
    │   ───────────────      ──────────────────────                │
    │   Dead code            preloadAndUploadWeights() (static)    │
    │   Has own packWeight   Orchestrates the 4-step sequence      │
    │                        BUT only for single-device path       │
    │                                                              │
    │   RankOrchestrator                                    │
    │   ───────────────────────                                    │
    │   Calls preloadForDevices (clone + upload)                   │
    │   Creates runners via createTestableInferenceRunner          │
    │   Has to manually call releaseAllHostWeightData (patched)    │
    │   Does NOT call packGemmWeights (lazy packing by accident)   │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

**Key issues**:
- `preloadAndUploadWeights()` is a free function in `InferenceRunnerFactory.cpp`
  that should be a method on `WeightManager`.
- The 4-step sequence (pack GEMM → upload non-GEMM → sync → release) is
  implicit knowledge baked into a factory helper, not enforced by the type
  system.
- `RankOrchestrator` reimplements part of this sequence ad-hoc.
- Two dead classes (`WeightPreloader`, `WeightManager::packWeight()`) add
  confusion.

---

## Proposed Architecture

### Core Idea: WeightManager Owns the Entire Lifecycle

One class. One method per phase. One `finalizeForDevice()` method that runs the
complete sequence and is impossible to get wrong.

```
                         WeightManager
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   Phase 1: LOAD                                  │
    │   ──────────                                     │
    │   loadWeight(name) → cache_[name]                │
    │   (From ModelLoader/GGUF, one copy)              │
    │                                                  │
    │   Phase 2: SHARD (optional)                      │
    │   ────────────────                               │
    │   configureTPSharding(tp_config)                 │
    │   getWeightForDevice(name, device)               │
    │     → shards on demand, stores in per_device     │
    │                                                  │
    │   Phase 3: FINALIZE                              │
    │   ─────────────────                              │
    │   finalizeForDevice(device)                      │
    │     1. packGemmWeights(device)                   │
    │     2. uploadNonGemmWeights(device)              │
    │     3. releaseHostWeightData()                   │
    │                                                  │
    │   OR for multi-device:                           │
    │   finalizeForDevices(devices)                    │
    │     1. preloadForDevices(devices) [clone+upload] │
    │     2. for each: packGemmWeights(device)         │  
    │     3. releaseHostWeightData()  [all caches]     │
    │                                                  │
    └──────────────────────────────────────────────────┘
```

### What Changes

| Component | Before | After |
|-----------|--------|-------|
| **WeightManager** | Has `packGemmWeights`, `uploadNonGemmWeights`, `releaseAllHostWeightData` as separate public methods | Adds `finalizeForDevice(device)` and `finalizeForDevices(devices)` that call the sequence internally |
| **InferenceRunnerFactory** | `preloadAndUploadWeights()` static helper orchestrates the 4-step sequence | Calls `weight_mgr->finalizeForDevice(device)` — one line |
| **RankOrchestrator** | Manually calls `preloadForDevices()` then patches `releaseAllHostWeightData()` | Calls `weight_mgr->finalizeForDevices(device_ids)` — one line |
| **WeightPreloader** | Dead class, never instantiated | Delete |
| **WeightManager::packWeight()** | Dead method, never called | Delete |
| **createTestableInferenceRunner** | Skips all weight lifecycle | Calls `finalizeForDevice()` like everyone else |
| **IWeightManager** | Has individual lifecycle methods with no-op defaults | Adds `finalizeForDevice()` / `finalizeForDevices()` with sane defaults |

### New Methods on IWeightManager/WeightManager

```cpp
class IWeightManager {
public:
    // ...existing interface...

    /**
     * Complete weight lifecycle for a single device:
     * 1. Pack GEMM weights (prepareWeights + GPU upload)
     * 2. Upload non-GEMM weights (norms, embeddings)
     * 3. Release host weight data (if GPU device)
     *
     * Call ONCE per device, AFTER all weights are loaded and sharding
     * is configured. Idempotent — second call is a no-op.
     */
    virtual bool finalizeForDevice(DeviceId device) { return true; }

    /**
     * Complete weight lifecycle for multiple LOCAL TP devices:
     * 1. Clone and upload weights to all devices (preloadForDevices)
     * 2. Pack GEMM weights per device
     * 3. Release all host weight data (cache_ + per_device_cache_)
     *
     * Call ONCE during RankOrchestrator init.
     */
    virtual bool finalizeForDevices(const std::vector<DeviceId>& devices) { return true; }
};
```

### What Gets Deleted

| File | What | Lines | Why Dead |
|------|------|-------|----------|
| `loaders/WeightPreloader.h` | Entire file | ~200 | Never included outside own .cpp |
| `loaders/WeightPreloader.cpp` | Entire file | ~350 | Never instantiated by any production code |
| `loaders/WeightManager.cpp` | `packWeight()` method | ~100 | Never called — `packGemmWeights()` has its own inline lambda |
| `CMakeLists.txt` | `WeightPreloader.cpp` entry | 1 | Build system reference |

### What Gets Simplified

**Before** (InferenceRunnerFactory.cpp — `preloadAndUploadWeights`, 70 lines):
```cpp
static void preloadAndUploadWeights(weight_mgr, device, log_prefix) {
    if (overlap_enabled) {
        gemm_pack_future = std::async([weight_mgr, device]() {
            return weight_mgr->packGemmWeights(device, nullptr, true);
        });
    }
    non_gemm_upload_ok = weight_mgr->uploadNonGemmWeights(device);
    gemm_pack_ok = overlap_enabled ? gemm_pack_future.get()
                                   : weight_mgr->packGemmWeights(device, nullptr, true);
    // error handling...
    if (device.is_gpu() && gemm_pack_ok && non_gemm_upload_ok) {
        weight_mgr->releaseAllHostWeightData();
    }
}
```

**After** (one line at call site):
```cpp
weight_mgr->finalizeForDevice(device);
```

The async overlap, error handling, and release logic moves *into* WeightManager
where it belongs. The factory doesn't need to know the sequencing.

**Before** (RankOrchestrator.cpp — init, ~30 lines of weight plumbing):
```cpp
weight_mgr->preloadForDevices(device_ids);
// ... create runners via createTestableInferenceRunner ...
// ... manually patched release call ...
weight_mgr->releaseAllHostWeightData();
// ... duplicate release call (oops) ...
weight_mgr->releaseAllHostWeightData();
```

**After** (one call before runner creation, or integrated into runner creation):
```cpp
weight_mgr->finalizeForDevices(device_ids);
```

---

## Implementation Order

### Phase 1: Consolidate (Low Risk)

1. **Add `finalizeForDevice()` to `WeightManager`** — wraps the existing
   pack/upload/release sequence. Existing methods stay public for now.

2. **Add `finalizeForDevices()` to `WeightManager`** — wraps
   `preloadForDevices()` + per-device `packGemmWeights()` + `releaseAllHostWeightData()`.

3. **Wire in callers** — replace `preloadAndUploadWeights()` call sites and
   `RankOrchestrator` manual sequence with new methods.

4. **Remove duplicate release block** in RankOrchestrator (the
   `concreteWeightManager()` copy-paste).

5. **Fix `packGemmWeights` to handle `per_device_cache_`** — when called with a
   device that has entries in `per_device_cache_`, iterate those instead of
   `cache_`. This makes GEMM packing explicit instead of relying on lazy
   creation.

### Phase 2: Clean Up (Low Risk)

6. **Delete `WeightPreloader`** (`.h`, `.cpp`, CMakeLists entry). Dead code.

7. **Delete `WeightManager::packWeight()`**. Dead code.

8. **Remove diagnostic logging** added during the memory investigation
   (`try_release` key parameter, `LOG_INFO` for SKIP cases in
   `releaseAllHostWeightData`).

9. **Make `packGemmWeights`, `uploadNonGemmWeights`, `releaseAllHostWeightData`
   private** (or protected for tests). Callers should use `finalizeForDevice()`.

### Phase 3: Structural (Medium Risk, Optional)

10. **Move `preloadAndUploadWeights()` body into `WeightManager`** and delete it
    from `InferenceRunnerFactory.cpp`. This is the "one owner" principle.

11. **Audit `createTestableInferenceRunner`** — it should call
    `finalizeForDevice()` so test runners get the same weight lifecycle as
    production runners. The "testable" part is the injected `IModelContext`, not
    skipping weight setup.

12. **Consider merging `cache_` and `per_device_cache_`** into a single map with
    composite keys (e.g., `DeviceId + weight_name`). The base/unsharded entry
    would use a sentinel `DeviceId::none()` key. This eliminates the
    dual-iteration pattern in `releaseAllHostWeightData()` and removes the
    question of "which cache does this method iterate?"

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Changing weight lifecycle breaks inference | Phase 1 is purely additive — new methods call existing ones. Existing paths unchanged until Phase 2. |
| GEMM packing order matters for GPU memory | `finalizeForDevice` preserves the existing async overlap pattern internally. |
| Tests rely on lazy GEMM packing | `finalizeForDevice()` default is no-op on `IWeightManager`, so mocks keep working. Tests using real `WeightManager` call finalize explicitly. |
| `releaseAllHostWeightData` called while transfers in-flight | `finalizeForDevice` internally waits for async pack to complete before releasing. Same as current `preloadAndUploadWeights`. |
| CPU-only path shouldn't release host data | `finalizeForDevice(cpu_device)` skips release (same check: `device.is_gpu()`). |

---

## Validation

- Run all unit tests: `ctest -R "^V2_Unit_" --output-on-failure --parallel`
- Run integration tests: `ctest -R "^V2_Integration_" --output-on-failure`
- Run parity tests: `ctest -R "^V2_Integration_Parity_" --output-on-failure`
- Multi-device memory test: launch server with `--tp-devices "rocm:0,rocm:1"`,
  verify RSS after startup is ~model_size + activation buffers (not 3× model)
- Single-device regression: launch with `-d rocm:0`, verify identical output
- CPU fallback: launch with `-d cpu`, verify weights NOT released from host

---

## Appendix: Current Weight Data Flow (Detailed)

```
GGUF File
  │
  ├─[mmap or ifstream]──→ ModelLoader::loadTensors()
  │                         │
  │                         ├─ Creates Q8_0Tensor / Q4_0Tensor / etc.
  │                         ├─ Sets mmap_owner_ (zero-copy from file)
  │                         └─ Returns map<string, shared_ptr<TensorBase>>
  │
  ├─[getWeight(name)]────→ WeightManager::cache_[name]
  │                         │
  │                         ├─ First access: loads from ModelLoader
  │                         └─ Subsequent: returns cached
  │
  ├─[getWeightForDevice]─→ Sharding decision
  │                         │
  │                         ├─ REPLICATE: return cache_[name] as-is
  │                         ├─ COLUMN_PARALLEL: slice columns → new tensor
  │                         └─ ROW_PARALLEL: slice rows → new tensor
  │                              │
  │                              └─ Stored in per_device_cache_["rocm:0:name"]
  │                                 (Heap-allocated, is_view()=false)
  │
  ├─[packGemmWeights]────→ For each GEMM weight in cache_:
  │                         │
  │                         ├─ KernelFactory::getOrCreatePreparedGemmWeights()
  │                         │   └─ Allocates GPU buffer, copies quantized blocks
  │                         ├─ KernelFactory::getOrCreateGemmEngine()
  │                         │   └─ Creates kernel with packed GPU data
  │                         └─ Optionally release_raw_data() on source tensor
  │
  ├─[uploadNonGemmWeights]→ For each non-GEMM weight (norms, embeddings):
  │                         │
  │                         └─ tensor->ensureOnDevice(device)
  │                             └─ Allocates GPU buffer, H2D copy
  │
  └─[releaseAllHostWeightData]→ Sweep cache_ + per_device_cache_:
                                │
                                ├─ Check: deviceValid() || hasCachedDeviceData()
                                ├─ Skip: is_view() (parent owns data)
                                ├─ unpinHostMemory() (CUDA/HIP tracking)
                                └─ release_raw_data() (free vector storage)

After release:
  - GPU has all weight data (GEMM packed or direct upload)
  - Host tensors exist as metadata shells (shape, dtype, device ptr)
  - mmap region may still be mapped but pages are reclaimable by OS
  - Accessing tensor->data() triggers D2H sync (coherence system)
```
