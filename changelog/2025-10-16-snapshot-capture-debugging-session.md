# Snapshot Capture Debugging Session - October 16, 2025

## Executive Summary

Investigated and partially fixed snapshot capture infrastructure for batch vs sequential pipeline parity testing. Discovered critical issues with snapshot key formatting and callback execution in the sequential pipeline path.

## Initial Problem

Test `BatchCorrectnessTest.FindFirstDivergenceStage` showed:
- **Sequential snapshots: 0**  
- **Batch snapshots: 120**  
- Root cause: Multiple compilation errors preventing test binary from rebuilding

## Issues Discovered and Fixed

### 1. **Compilation Errors (CRITICAL)**

**Problem**: Code had been failing to compile for multiple commits, causing test to run stale binary.

**Errors**:
```cpp
// Error 1: Ambiguous method call
legacy_->captureStageSnapshot(...);  
// Both AbstractPipeline and PipelineBase define this method

// Error 2: Private member access
snapshot_source_  // Declared private in PipelineBase
```

**Fix**:
```cpp
// src/PipelineBase.h - Make snapshot_source_ protected
protected:
    std::string snapshot_source_ = "llaminar";

// src/QwenPipelineAdapter.h - Use explicit cast
static_cast<AbstractPipeline*>(legacy_.get())->captureStageSnapshot(...);
```

**Impact**: After fixing, snapshots jumped from **0 → 73** for sequential pipeline!

### 2. **Missing Source Parameter in BatchQwenPipeline**

**Problem**: Embedding snapshot wasn't passing source parameter:
```cpp
PipelineSnapshotManager::instance().capture(
    PipelineStage::EMBEDDING, -1, data, seq, dim
    // MISSING: source parameter!
);
```

**Fix**:
```cpp
PipelineSnapshotManager::instance().capture(
    PipelineStage::EMBEDDING, -1, data, seq, dim,
    getSnapshotSource()  // "batch" for BatchQwenPipeline
);
```

**Impact**: batch_EMBEDDING now captured correctly (120 → 121 snapshots)

### 3. **Snapshot Key Format Inconsistency (ACTIVE ISSUE)**

**Problem**: Different key formats across pipelines:

| Pipeline    | Format Example | Captured? |
|-------------|----------------|-----------|
| Sequential  | `llaminar_layer_0_ATTENTION_NORM` | ✅ Yes |
| Batch       | `batch_layer_0_Q_PROJECTION` | ✅ Yes |
| Test Expects | `llaminar_ATTENTION_NORM_layer0` | ❌ No |

**Evidence**:
```
First 10 sequential keys:
  llaminar_layer_22_ATTENTION_RESIDUAL
  llaminar_layer_22_ATTENTION_NORM
  llaminar_layer_21_ATTENTION_RESIDUAL
  ...

Test searches for:
  llaminar_EMBEDDING
  llaminar_ATTENTION_NORM_layer0  ← Wrong format!
  llaminar_Q_PROJECTION_layer0
```

**Root Cause**: PipelineSnapshotManager constructs keys as `source_layer_X_stage`, but test expects `source_stage_layerX`.

**Status**: 🔴 **NOT YET FIXED** - Need to standardize key format

### 4. **Missing MPIAttentionOperator Snapshots (ACTIVE ISSUE)**

**Problem**: Sequential pipeline missing intermediate attention stages:

| Stage | Sequential | Batch |
|-------|-----------|-------|
| EMBEDDING | ❌ Wrong key | ✅ Captured |
| ATTENTION_NORM | ✅ Captured | ✅ Captured |
| Q_PROJECTION | ❌ Missing | ✅ Captured |
| K_PROJECTION | ❌ Missing | ✅ Captured |
| V_PROJECTION | ❌ Missing | ✅ Captured |
| ROPE_APPLICATION | ❌ Missing | ✅ Captured |
| ATTENTION_CONTEXT | ❌ Missing | ✅ Captured |
| ATTENTION_OUTPUT | ✅ Captured | ✅ Captured |

**Root Cause**: OpenBLASPrefillProvider sets MPIAttentionOperator callback at line 228-233:
```cpp
attention_kernel->setSnapshotCallback(
    [this, layer_idx](PipelineStage stage, int layer, 
                     const float *data, int seq_len, int feature_dim) {
        this->captureSnapshot(stage, layer, data, seq_len, feature_dim);
    });
```

But MPIAttentionOperator's `gatherAndSnapshotPreRoPE()` method (lines 1273-1412) is **never called** because:
1. Callback is set ✅
2. `world_size == 2` ✅  
3. **BUT**: Function isn't being invoked in execute() path for prefill

**Status**: 🔴 **NOT YET FIXED** - Need to investigate MPIAttentionOperator execution flow

## Current Snapshot Counts

```
Total snapshots: 580
Sequential: 72 / 291 expected (24.7%)
Batch: 121 / 291 expected (41.6%)
Missing comparisons: 8 critical stages
```

## Test Validation Added

Enhanced `test_batch_correctness.cpp` with:

1. **Expected snapshot count calculation**:
   - EMBEDDING (1)
   - Per layer (24): 12 stages = 288
   - FINAL_NORM + LM_HEAD (2)
   - **Total: 291 per pipeline**

2. **Test assertions**:
```cpp
ASSERT_EQ(missing, 0) << "Missing " << missing << " critical snapshots";
```

3. **Detailed reporting**:
```
Expected per pipeline: 291
Sequential captured: 72 / 291
Batch captured: 121 / 291
```

## Next Steps

### Priority 1: Fix Snapshot Key Format
- **Task**: Standardize to `source_STAGE_layerX` format
- **Files**: 
  - `src/PipelineSnapshotManager.cpp` (key construction)
  - `tests/test_batch_correctness.cpp` (key comparison)
- **Expected Impact**: Unlock all currently captured snapshots for comparison

### Priority 2: Enable MPIAttentionOperator Callbacks
- **Task**: Investigate why `gatherAndSnapshotPreRoPE()` isn't called
- **Debug approach**:
  1. Add EXECUTE_TRACE logs in MPIAttentionOperator::execute()
  2. Verify prefill vs decode path selection
  3. Check if `snapshot_callback_` is lost during provider setup
- **Expected Impact**: Add 120+ missing sequential snapshots (Q/K/V, RoPE, Context)

### Priority 3: Complete Snapshot Coverage
- **Current**: 72/291 (sequential), 121/291 (batch)
- **Target**: 291/291 for both pipelines
- **Missing categories**:
  - FFN stages (GATE, UP, SWIGLU, DOWN)
  - Residual connections
  - Final normalization

## Files Modified

```
src/PipelineBase.h                      - Made snapshot_source_ protected
src/QwenPipelineAdapter.h               - Fixed ambiguous method call
src/BatchQwenPipeline.cpp               - Added source parameter to embedding snapshot
tests/test_batch_correctness.cpp        - Added validation, debug output, assertions
src/operators/MPIAttentionOperator.cpp  - Added extensive debug logging (not yet effective)
```

## Lessons Learned

1. **Always check compilation errors first** - Stale binaries led to hours of confusion
2. **Snapshot key format matters** - Even 1 character difference breaks comparisons
3. **Multiple inheritance is dangerous** - AbstractPipeline + PipelineBase diamond pattern caused ambiguity
4. **Callback architecture is complex** - Setting callback ≠ callback being invoked
5. **Debug logging is essential** - Added TRACE logs revealed callback execution gaps

## Performance Notes

Test runtime: ~24 seconds for 4-token prefill on 2 MPI ranks (Debug build)

## References

- Original issue: Sequential snapshots = 0, batch snapshots = 120
- Key discovery: Compilation errors preventing test binary rebuild
- Architecture: Diamond inheritance (AbstractPipeline + PipelineBase → QwenPipeline)
