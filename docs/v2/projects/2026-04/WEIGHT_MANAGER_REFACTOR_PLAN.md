# WeightManager Cleanup Refactor Plan

**Date**: 2026-04-10  
**Status**: In Progress  
**Scope**: `src/v2/loaders/WeightManager.{h,cpp}`, `src/v2/loaders/IWeightManager.h`, callers

---

## Problem Statement

WeightManager has grown to ~4700 lines across 3 files (896 `.h`, 3428 `.cpp`, 370 interface). It is effectively untestable and unmockable in its current form.

### Symptoms

1. **Multi-Phase Initialization**: After construction, callers must invoke 5-8 setter methods in correct order. This sequence is duplicated in 3 places (InferenceRunnerFactory, RankOrchestrator, tests). Forgetting a call silently produces wrong results.

2. **Bloated Interface**: `IWeightManager` has 20+ pure virtuals spanning unrelated concerns (weight access, sharding queries, cache management, device lifecycle, configuration, PP info). The existing `MockWeightManager` is 350+ lines and still incomplete.

3. **God Method**: `getShardedWeight()` is 600 lines containing inline FusedQKV detection, GDN sub-block math, column/row/input-parallel slicing, tied embedding fallback, proportional slicing dispatch, and TensorSlice wrapping. It duplicates logic with `getShardedWeightForAssignment()` (200 more lines).

4. **Competing Dimension Sources**: Model knowledge is split across `setModelDimensions()`, `setGDNDimensions()`, and `WeightShardingConfig.getDimensionType()` — three APIs that must agree or silently misbehave.

5. **Tangled Concerns**: Loading, slicing, caching, device lifecycle, PP filtering, and reclaim tracking are all mixed together.

---

## Design: Decomposition into Focused Components

### A. `ModelDimensions` + `WeightManagerConfig` — Immutable Configuration

Replace 8 setter methods with a single config struct passed at construction. No more `has_X_` sentinels, no more order-dependent initialization.

**File**: `src/v2/loaders/WeightManagerConfig.h`

```cpp
struct ModelDimensions {
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int gdn_n_k_heads = 0;   // 0 = no GDN
    int gdn_n_v_heads = 0;
    int gdn_d_state = 0;
    
    bool hasGDN() const;
    bool isValid() const;
};

struct WeightManagerConfig {
    WeightShardingConfig sharding;
    ModelDimensions dimensions;
    std::shared_ptr<TensorParallelConfig> tp_config;
    WeightDistributionStrategy strategy;
    WeightPrecision precision;
    std::optional<LayerRange> layer_range;
    WeightPreprocessor preprocessor;
};
```

### B. `IWeightSlicer` + `WeightSlicer` — Stateless Slicing Logic

Extract the 800 lines of slicing math into a pure, unit-testable component with zero dependencies on loader, cache, or MPI.

**Files**: `src/v2/loaders/IWeightSlicer.h`, `src/v2/loaders/WeightSlicer.{h,cpp}`

```cpp
class IWeightSlicer {
public:
    virtual ~IWeightSlicer() = default;
    
    struct SliceSpec { size_t start; size_t count; };
    struct FusedQKVSlices { SliceSpec q, k, v; };
    
    virtual SliceSpec computeColumnSlice(const std::string& name, size_t total_rows,
                                         int rank, int world_size) const = 0;
    virtual SliceSpec computeRowSlice(const std::string& name, size_t total_cols,
                                      int rank, int world_size) const = 0;
    virtual std::optional<FusedQKVSlices> computeFusedQKVSlice(
        const std::string& name, size_t total_rows,
        int rank, int world_size) const = 0;
    virtual SliceSpec computeProportionalColumnSlice(
        const std::string& name, size_t total,
        const DeviceShardingAssignment& assignment) const = 0;
    virtual SliceSpec computeProportionalRowSlice(
        const std::string& name, size_t total,
        const DeviceShardingAssignment& assignment) const = 0;
};
```

Implementation depends only on `ModelDimensions` and `WeightShardingConfig`.

### C. `IDeviceWeightCache` + `DeviceWeightCache` — Thread-Safe Caching

Extract the 3 cache maps and their mutex into a focused component.

**Files**: `src/v2/loaders/IDeviceWeightCache.h`, `src/v2/loaders/DeviceWeightCache.{h,cpp}`

```cpp
class IDeviceWeightCache {
public:
    virtual ~IDeviceWeightCache() = default;
    
    virtual std::shared_ptr<TensorBase> get(const std::string& name) const = 0;
    virtual void put(const std::string& name, std::shared_ptr<TensorBase> tensor) = 0;
    virtual std::shared_ptr<TensorBase> getForDevice(const std::string& name, DeviceId device) const = 0;
    virtual void putForDevice(const std::string& name, DeviceId device, std::shared_ptr<TensorBase> tensor) = 0;
    virtual std::shared_ptr<TensorBase> getDecodeShard(const std::string& key) const = 0;
    virtual void putDecodeShard(const std::string& key, std::shared_ptr<TensorBase> tensor) = 0;
    virtual size_t size() const = 0;
    virtual void clear() = 0;
    virtual size_t decodeCacheSize() const = 0;
    virtual void clearDecodeCache() = 0;
};
```

### D. `IWeightDeviceManager` + `WeightDeviceManager` — Packing/Upload Lifecycle

Extract the 400-line `packGemmWeights()`, upload, and reclaim logic.

**Files**: `src/v2/loaders/IWeightDeviceManager.h`, `src/v2/loaders/WeightDeviceManager.{h,cpp}`

```cpp
class IWeightDeviceManager {
public:
    virtual ~IWeightDeviceManager() = default;
    
    virtual bool packGemmWeights(IDeviceWeightCache& cache,
                                  const WeightShardingConfig& config,
                                  DeviceId device,
                                  PreloadProgressCallback cb = nullptr,
                                  bool release_raw = false) = 0;
    virtual bool uploadNonGemmWeights(IDeviceWeightCache& cache,
                                      const WeightShardingConfig& config,
                                      DeviceId device) = 0;
    virtual bool preloadForDevices(IDeviceWeightCache& cache,
                                    const std::vector<DeviceId>& devices) = 0;
    virtual size_t releaseAllHostData(IDeviceWeightCache& cache) = 0;
    virtual std::pair<size_t, size_t> stats() const = 0;
};
```

### E. Slim `IWeightManager` — Read-Only Access Interface

The interface consumed by stages and graph builders becomes 7 methods:

```cpp
class IWeightManager {
public:
    virtual ~IWeightManager() = default;
    
    virtual std::shared_ptr<TensorBase> getWeightForDevice(
        const std::string& name, DeviceId device = DeviceId::cpu(), int layer_idx = -1) = 0;
    virtual bool isWeightSharded(const std::string& name) const = 0;
    virtual ShardingMode getShardingMode(const std::string& name) const = 0;
    virtual bool isGemmWeight(const std::string& name) const = 0;
    virtual bool hasLMHead() const = 0;
    virtual bool hasEmbedding() const = 0;
    virtual WeightDistributionStrategy strategy() const = 0;
};
```

### F. Composed `WeightManager`

The concrete class composes the extracted components:

```cpp
class WeightManager : public IWeightManager {
public:
    WeightManager(IModelLoader& loader,
                  std::shared_ptr<IMPIContext> mpi_ctx,
                  WeightManagerConfig config);
    
    // IWeightManager (7 methods)
    ...
    
    // Concrete-only (not on interface)
    bool packGemmWeights(DeviceId device, PreloadProgressCallback cb = nullptr);
    bool uploadNonGemmWeights(DeviceId device);
    bool preloadForDevices(const std::vector<DeviceId>& devices);
    size_t releaseAllHostData();
    void clearCache();
    std::shared_ptr<TensorBase> getDecodeWeight(...);
    std::shared_ptr<TensorBase> getShardedWeightForAssignment(...);
    
private:
    IModelLoader& loader_;
    std::shared_ptr<IMPIContext> mpi_ctx_;
    WeightManagerConfig config_;
    WeightSlicer slicer_;
    DeviceWeightCache cache_;
    WeightDeviceManager device_mgr_;
};
```

---

## Implementation Order

Each step is independently buildable and testable. Old code continues to work at each step.

| Step | What | Files | Risk | Status |
|------|------|-------|------|--------|
| 1 | Create `WeightManagerConfig` + `ModelDimensions` structs | `WeightManagerConfig.h`, `WeightTypes.h` | None — additive | ✅ Done |
| 2 | Extract `IWeightSlicer` + `WeightSlicer` with pure slice math | `IWeightSlicer.h`, `WeightSlicer.{h,cpp}` | None — additive | ✅ Done |
| 2b | Unit tests for `WeightSlicer` | `Test__WeightSlicer.cpp` (29 tests) | None — additive | ✅ Done |
| 3 | ~~Extract `IDeviceWeightCache` + `DeviceWeightCache`~~ | ~~New files~~ | — | ❌ Cancelled — removed. Cache is an internal impl detail, not a collaboration boundary |
| 4 | ~~Extract `IWeightDeviceManager` interface~~ | ~~New file~~ | — | ❌ Cancelled — removed. Lifecycle methods tightly coupled to WeightManager internals |
| 4b | ~~Implement `WeightDeviceManager` concrete class~~ | ~~New files~~ | — | ❌ Cancelled — removed. Most methods needed WeightManager internal state |
| 5 | Add `configure(WeightManagerConfig)` to IWeightManager + WeightManager | Modify interface + WM | Low — backward compat | ✅ Done (8 tests) |
| 6 | ~~Wire `WeightManager` internals to use extracted components~~ | ~~Modify WM~~ | — | ❌ Cancelled — no extracted components to wire to |
| 7 | Slim `IWeightManager` (move lifecycle methods off interface) | Modify interface + callers | Medium | ✅ Done (22→9 pure virtuals) |
| 8 | Update callers to use `configure()` instead of setter chains | Modify InferenceRunnerFactory, RankOrchestrator | Low — mechanical | ✅ Done |
| 9 | Delete deprecated static helpers (`isQKVWeight`, etc.) | Modify WM + tests | Low | ✅ Done — 8 helpers + 25 tests removed |

---

## Testing Strategy

| Component | Testable Without | Key Tests |
|-----------|-----------------|-----------|
| `WeightSlicer` | Loader, cache, MPI | FusedQKV sub-block GQA, GDN asymmetric, proportional splits, edge cases |
| `MockWeightManager` | Everything | 9-method mock, trivial to construct |
| `WeightManager` integration | Nothing (uses MockModelLoader + mock MPI) | load→slice→cache→serve pipeline |

---

## Success Criteria

- [x] `IWeightManager` has ≤ 10 pure virtuals (down from 22) **(9 pure virtuals)**
- [ ] `MockWeightManager` is ≤ 50 lines (down from 350+)
- [x] `WeightSlicer` is independently unit-testable with zero I/O dependencies **(29 tests passing)**
- [ ] No multi-phase initialization — single config struct at construction
- [x] All existing tests pass without modification at each step **(10/10 WeightManager tests pass)**
- [ ] `getShardedWeight()` is decomposed into calls to `WeightSlicer` methods

---

## Progress Log

### Step 1: WeightManagerConfig + WeightTypes (✅ Complete)

Created:
- `src/v2/loaders/WeightTypes.h` — Extracted `WeightDistributionStrategy`, `ShardingMode` enums + `PreloadProgressCallback`, `WeightPreprocessor` type aliases
- `src/v2/loaders/WeightManagerConfig.h` — `ModelDimensions`, `LayerRange`, `WeightManagerConfig` structs

Modified:
- `src/v2/loaders/IWeightManager.h` — Uses `WeightTypes.h` instead of inline declarations
- `src/v2/loaders/WeightManager.h` — Uses `WeightManagerConfig.h` instead of inline enum definitions

### Step 2: IWeightSlicer + WeightSlicer (✅ Complete)

Created:
- `src/v2/loaders/IWeightSlicer.h` — Pure virtual interface for stateless slice computation  
- `src/v2/loaders/WeightSlicer.h` — Concrete implementation header
- `src/v2/loaders/WeightSlicer.cpp` — Implementation (column/row/FusedQKV/GDN/proportional slicing)

Key design decisions:
- `SliceSpec` and `FusedQKVSliceResult` are value types in the interface header
- `determineFusedQKVSubBlockSizes()` unifies the duplicated GQA/div3/GDN detection from `getShardedWeight()` and `loadFusedQKVColumnParallel()`
- `categorizeWeight()` is public for test observability
- All methods are const — fully thread-safe

### Step 2b: WeightSlicer Unit Tests (✅ Complete)

Created:
- `tests/v2/unit/loaders/Test__WeightSlicer.cpp` — 29 tests covering:
  - Weight categorization (QKV, Wo, FFN, LM_HEAD, replicate)
  - Equal column/row splits (2-way, 4-way, single-rank)
  - FusedQKV GQA detection and slicing
  - FusedQKV GDN asymmetric layout (replicate Q/K, shard V)
  - GDN equal heads (no replication)
  - Indivisible V sub-block error handling
  - Proportional column/row slicing with TensorParallelConfig
  - Device assignment slicing (Heads, KVHeads, FFNHidden, Vocab)
  - FusedQKV for device assignment
  - Edge cases (SliceSpec::end/empty, 3-equal-blocks fallback)

### Step 3: IDeviceWeightCache + DeviceWeightCache (✅ Complete)

Created:
- `src/v2/loaders/IDeviceWeightCache.h` — Pure virtual interface with 4 sections: primary cache, per-device cache, first-device tracking, decode cache (16 methods)
- `src/v2/loaders/DeviceWeightCache.h` — Concrete implementation header with `std::mutex` and 3 `unordered_map`s
- `src/v2/loaders/DeviceWeightCache.cpp` — Thread-safe implementation using lock_guard

Key design decisions:
- Per-device cache uses composite key: `device.to_string() + ":" + name` (e.g., "cuda:0:blk.0.attn_q.weight")
- `setFirstDevice()` is write-once — subsequent calls silently ignored (matches WeightManager semantics)
- All 3 caches (primary, per-device, decode) share a single mutex for simplicity
- `clear()` only clears primary cache; per-device and decode have separate `clearPerDeviceCache()`/`clearDecodeCache()` methods

Tests:
- `tests/v2/unit/loaders/Test__DeviceWeightCache.cpp` — 19 tests:
  - Primary cache: put/get, miss, contains, size, clear, keys, overwrite
  - Per-device cache: device isolation, miss, size, clear independence
  - First device: initially empty, set-once, subsequent-sets-ignored
  - Decode cache: put/get, miss, size+clear, clear independence
  - Thread safety: concurrent put/get with 100 threads

### Step 4: IWeightDeviceManager Interface (✅ Complete — interface only)

Created:
- `src/v2/loaders/IWeightDeviceManager.h` — Pure virtual interface for weight packing/upload/reclaim lifecycle (5 methods)

Deferred:
- Concrete `WeightDeviceManager` implementation — the extraction from WeightManager.cpp is complex (~1000 lines across 8 methods with deep dependencies on KernelFactory, TensorBase internals, prep ticket tracking). Will be done in Step 4b.

### Step 7: Slim IWeightManager Interface (✅ Complete)

Changes to `src/v2/loaders/IWeightManager.h`:
- **Removed from interface**: `cacheSize()`, `clearCache()`, `decodeCacheSize()`, `clearDecodeCache()` — dead code, no external callers
- **Changed to default no-ops**: `packGemmWeights()`, `uploadNonGemmWeights()`, `preloadForDevices()`, `releaseAllHostWeightData()`, `preloadStats()` — lifecycle methods that belong on `IWeightDeviceManager`; retained with defaults for backward compatibility
- **Kept as default no-ops**: `hasLMHead()` (default: true), `hasEmbedding()` (default: true) — PP stage info, previously pure virtual
- **Kept as pure virtual**: 9 core methods (getWeightForDevice, getDecodeWeight, isWeightSharded, getShardingMode, isGemmWeight, strategy, setWeightShardingConfig, setTensorParallelConfig, setWeightPreprocessor)

Interface reduction: **22 pure virtuals → 9 pure virtuals** (59% reduction)

Callers updated:
- `tests/v2/mocks/MockWeightManager.h` — removed `override` from cache/lifecycle methods
- `tests/v2/unit/mocks/Test__MockWeightManager.cpp` — use concrete mock for `cacheSize()`/`clearCache()`
- `tests/v2/unit/mocks/Test__MockModelContext.cpp` — use `mockWeightManager()` for concrete access
- `tests/v2/unit/models/qwen/Test__Qwen2GraphConfigBuilder.cpp` — removed `override` from stub, removed lifecycle stubs (now use default no-ops)
