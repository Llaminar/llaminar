# RankOrchestrator Decoupling & Test-Coverage Plan

**Date**: 2026-04-01
**Status**: P0 Complete ✅, P1 Deferred, P2 Complete ✅
**Scope**: Break concrete `DeviceGraphOrchestrator` coupling in `RankOrchestrator`, add unit test coverage.

## Current State (Post-Refactor)

- **Header**: `RankOrchestrator.h` — 900 lines
- **Implementation**: `RankOrchestrator.cpp` — 3,256 lines (4,156 total)
- **Model coupling**: Zero — already model-agnostic via SchemaFactoryRegistry
- **Interface usage**: ✅ `device_runners_` now typed as `vector<unique_ptr<IInferenceRunner>>`
- **DGO coupling**: ✅ Zero — all `inferenceState()` / `hasInferenceState()` calls eliminated
- **Test coverage**: 342/342 unit tests, including 20 new interface decoupling tests

### Problem (Solved): Concrete DGO Coupling

`device_runners_` was typed as `vector<unique_ptr<DeviceGraphOrchestrator>>` (concrete). MDO accessed DGO-specific APIs not on `IInferenceRunner`:

| API | Sites | Purpose | Resolution |
|-----|-------|---------|------------|
| `inferenceState().device_id` | 3 | Backend lookup for pinning/profiling | → `primaryDeviceId()` |
| `inferenceState().logits_local` | 8 | GPU pointer + shape for logits gather / GPU sampling | → `hasLogitsLocal()` + `getLogitsLocalInfo()` |
| `hasInferenceState()` | 3 | Guard checks | → `hasLogitsLocal()` |
| `setPPStageConfig()` | 1 | Configure partial graph (construction-time only) | → optional `dynamic_cast` at construction |

---

## Phase 0: Interface Widening ✅ COMPLETE

**Added to `IInferenceRunner`**:

```cpp
struct LogitsLocalInfo {
    const void* gpu_ptr = nullptr;
    std::optional<DeviceId> device;
    size_t vocab_local = 0;
    TensorBase* tensor = nullptr;
    explicit operator bool() const { return tensor != nullptr; }
};

virtual DeviceId primaryDeviceId() const { return DeviceId::cpu(); }
virtual bool hasLogitsLocal() const { return false; }
virtual LogitsLocalInfo getLogitsLocalInfo() const { return {}; }
```

**Added to `DeviceGraphOrchestrator`** (inline overrides):

```cpp
DeviceId primaryDeviceId() const override { return state_.device_id; }
bool hasLogitsLocal() const override { return state_.logits_local != nullptr; }
LogitsLocalInfo getLogitsLocalInfo() const override;
```

**Changed `device_runners_` type**: `vector<unique_ptr<DeviceGraphOrchestrator>>` → `vector<unique_ptr<IInferenceRunner>>`

**Handled `setPPStageConfig()`**: Optional `dynamic_cast` at construction time only.

**All 11 `inferenceState()` / `hasInferenceState()` call sites migrated.**

---

## Phase 1: Extract LogitsGatherer + DeviceSampler ✅ COMPLETE

Extracted two helper classes from RankOrchestrator:

### LogitsGatherer
- **File**: `src/v2/execution/local_execution/orchestrators/LogitsGatherer.h/cpp`
- Owns the combined logits buffer (`FP32Tensor`)
- Handles D2H gather from TP device runners (fast decode + general prefill paths)
- Handles PP stage logits copy
- Manages buffer pinning/unpinning lifecycle
- Skip-gather control (`setSkipDecode/Prefill`, `needsGather()`)
- ~280 lines of implementation moved from MDO

### DeviceSampler
- **File**: `src/v2/execution/local_execution/orchestrators/DeviceSampler.h/cpp`
- Stateless static methods for GPU-side sampling
- `sampleGreedy()`: per-device `argmaxF32()` + cross-device max on host
- `sample()`: per-device `topKF32()` + merge + temperature/softmax/top-p/multinomial
- ~265 lines of implementation moved from MDO

### MDO Changes
- Removed 5 member variables (`combined_logits_`, `combined_logits_pinned_`, `skip_logits_gather_decode_`, `skip_logits_gather_prefill_`, `last_gathered_logits_size_`)
- Added 1 member (`std::unique_ptr<LogitsGatherer> logits_gatherer_`)
- Removed 4 method bodies (`gatherLogits`, `copyLogitsFromStage`, `sampleGreedyOnDevice`, `sampleOnDevice`)
- `sampleGreedyOnDevice/sampleOnDevice` now thin 3-line delegations to `DeviceSampler`
- `setSkipLogitsGatherDecode/Prefill` delegate to `logits_gatherer_`
- `logits()`, `getSnapshot()`, `getTPSnapshot()` reference `logits_gatherer_` instead of raw members
- ~590 lines moved out of MDO.cpp

### Tests
- `Test__LogitsGatherer.cpp` — 17 tests: buffer alloc, skip control, needsGather logic, single-device gather, copyFromStage, move semantics
- `Test__DeviceSampler.cpp` — 7 tests: edge cases for empty/single runner, missing logits_local, greedy delegation
- All 345/345 unit tests pass

---

## Phase 2: Unit Tests ✅ COMPLETE

**New test file**: `Test__MDO_InterfaceDecoupling.cpp` (20 test cases)

| Test | Purpose |
|------|---------|
| `PrimaryDeviceIdDefaultsCPU` | Interface contract: default is CPU |
| `PrimaryDeviceIdReturnsCUDA` | Interface contract: CUDA device |
| `PrimaryDeviceIdReturnsROCm` | Interface contract: ROCm device |
| `HasLogitsLocalFalseByDefault` | Interface contract: default false |
| `HasLogitsLocalTrueWhenConfigured` | Interface contract: true with column-parallel |
| `LogitsLocalInfoEmptyWhenNoLocal` | LogitsLocalInfo: empty/falsy |
| `LogitsLocalInfoValidWhenConfigured` | LogitsLocalInfo: tensor + vocab_local |
| `LogitsLocalInfoBoolConversion` | LogitsLocalInfo operator bool |
| `CreateForTestAcceptsIInferenceRunners` | MDO construction with 2 mock runners |
| `CreateForTestWithSingleRunner` | MDO construction with 1 mock runner |
| `ForwardDelegatesToRunners` | forward() calls runner's forward() |
| `ForwardFailsWhenRunnerFails` | Failure propagation |
| `LogitsFromPrimaryRunner` | logits() returns primary runner's data |
| `VocabSizeFromModelContext` | vocab_size() reads from IModelContext |
| `ClearCacheDelegatesToAllRunners` | clear_cache() resets all runners |
| `MixedDeviceTypeRunnersConstruct` | CUDA+ROCm runners through IInferenceRunner |
| `ArchitectureFromPrimaryRunner` | architecture() delegation |
| `DeviceRunnerReturnsIInferenceRunner` | deviceRunner(i) returns interface type |
| `RunnersWithoutLogitsLocalAreReplicated` | Replicated LM head detection |
| `RunnersWithLogitsLocalAreColumnParallel` | Column-parallel LM head detection |

---

## Files Changed

### Modified
- `IInferenceRunner.h` — added LogitsLocalInfo struct + 3 virtual methods with defaults
- `DeviceGraphOrchestrator.h` — 3 inline override implementations
- `RankOrchestrator.h` — device_runners_ type changed, forward decl removed
- `RankOrchestrator.cpp` — all 11 inferenceState()/hasInferenceState() sites migrated
- `InferenceRunnerFactory.h/cpp` — createTestableRankOrchestrator signature updated
- `Test__InferenceRunnerFactory_MultiDevice.cpp` — vector type updated

### New
- `Test__MDO_InterfaceDecoupling.cpp` — 20 unit tests for interface contract + MDO behavior
- `docs/v2/cleanup/MDO_DECOUPLING_PLAN.md` — this plan document
