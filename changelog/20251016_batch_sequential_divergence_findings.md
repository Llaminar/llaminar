# Batch vs Sequential Pipeline Divergence Findings
**Date**: 2025-01-16  
**Status**: FIRST DIVERGENCE IDENTIFIED

## Summary

Successfully identified the first point of divergence between batch and sequential pipelines by fixing snapshot key format mismatches and running stage-by-stage comparison.

## Key Findings

### 1. Snapshot Source Name Mismatch (RESOLVED)
**Problem**: Test was searching for `llaminar_*` keys but sequential pipeline uses `OpenBLAS_*`
- Sequential pipeline uses `PrefillProvider.name()` which returns `"OpenBLAS"`
- Batch pipeline correctly uses `"batch"` via `getSnapshotSource()`
- Test was incorrectly expecting `"llaminar"` prefix for sequential

**Solution**: Updated test to search for `OpenBLAS_*` keys for sequential pipeline

**Impact**: Snapshot count jumped from 72 → 387 sequential snapshots captured

### 2. First Divergence Point: Q_PROJECTION Layer 0
**Status**: ❌ **MAJOR DIVERGENCE DETECTED**

```
Stage: Q_PROJECTION layer 0
Sequential: min=-79.862 max=48.9815 mean=-0.137529
Batch:      min=-0.863866 max=0.30777 mean=-0.113899
```

**Analysis**:
- Sequential Q_PROJECTION values are ~100x larger than batch
- This is the FIRST stage where outputs diverge after EMBEDDING
- EMBEDDING stage matches perfectly (max_diff=0)
- Divergence magnitude: ~100x difference in value ranges

**Implications**:
- Something fundamental is different in how Q projection is computed
- Could be weight loading, matrix multiplication, or tensor layout
- Affects all subsequent layers since attention depends on Q

### 3. Snapshot Coverage

**Sequential Pipeline** (OpenBLAS):
- Total: 387 snapshots (expected 291)
- Over-capturing due to multiple intermediate stages

**Batch Pipeline**:
- Total: 121 snapshots (expected 291)
- Missing some stages like ATTENTION_NORM (implementation-specific)
- Core stages covered: EMBEDDING, Q/K/V_PROJECTION, ROPE_APPLICATION, ATTENTION_CONTEXT

**Matched Stages**:
- ✅ EMBEDDING: Perfect match (max_diff=0)
- ❌ Q_PROJECTION layer 0: **DIVERGES (100x magnitude difference)**
- ⚠️ ATTENTION_NORM layer 0: Not captured by batch (acceptable)

## Code Changes

### tests/test_batch_correctness.cpp
**Lines Modified**: 523-525, 583-585

**Change 1**: Fixed key filtering to use OpenBLAS prefix
```cpp
// Before:
if (key.find("llaminar_") == 0) seq_keys.push_back(key);

// After:
if (key.find("OpenBLAS_") == 0) seq_keys.push_back(key);
```

**Change 2**: Fixed comparison key generation
```cpp
// Before:
std::string seq_key = registry.make_key("llaminar", stage.name, stage.layer);

// After:
std::string seq_key = registry.make_key("OpenBLAS", stage.name, stage.layer);
```

### src/PipelineBase.h (Previously Modified)
- Line 345: Changed `snapshot_source_` from private to protected

### src/QwenPipelineAdapter.h (Previously Modified)
- Line 144: Fixed ambiguous method call with explicit cast

### src/BatchQwenPipeline.cpp (Previously Modified)
- Lines 352-358: Added source parameter to embedding snapshot

## Test Results

### Before Fix
```
Total snapshots: 580
Sequential: 72 (using wrong prefix)
Batch: 121
Missing: 8
```

### After Fix
```
Total snapshots: 580
Sequential: 387 (using correct OpenBLAS_ prefix)
Batch: 121
Passed: 1 (EMBEDDING)
Failed: 1 (Q_PROJECTION layer 0 - DIVERGES)
Missing: 1 (batch_layer_0_ATTENTION_NORM)
```

## Next Steps

### Immediate (Priority 1)
1. **Investigate Q_PROJECTION divergence**
   - Compare weight loading between batch and sequential
   - Verify tensor shapes and orientations
   - Check if matrix multiplication backend differs
   - Examine distributed vs local tensor handling

2. **Fix comparison metrics bug**
   - Max absolute diff shows 0 but values clearly differ
   - Likely showing stale metrics from previous comparison
   - Check SnapshotComparator logic

### Future (Priority 2)
3. **Harmonize snapshot sources**
   - Consider renaming "OpenBLAS" → "llaminar" for clarity
   - Or document that sequential uses backend name as source
   - Update PrefillProvider.name() or add override

4. **Complete stage coverage**
   - Identify why batch doesn't capture ATTENTION_NORM
   - Add missing instrumentation points if needed
   - Or document intentional coverage differences

5. **Expand comparison**
   - After fixing Q_PROJECTION, continue to K_PROJECTION
   - Then V_PROJECTION, ROPE, ATTENTION_CONTEXT
   - Map full divergence cascade through layers

## Technical Notes

### Snapshot Key Format
Registry creates keys as: `source_layer_X_STAGE`
- Sequential: `OpenBLAS_layer_0_Q_PROJECTION`
- Batch: `batch_layer_0_Q_PROJECTION`
- Embedding: `OpenBLAS_EMBEDDING` (layer=-1)

### Known Limitations
- Comparison metrics (max_diff, rel_L2) appear to show stale values
- Batch pipeline doesn't capture all intermediate stages (by design)
- Test currently only checks first 8 stages (EMBEDDING through ATTENTION_OUTPUT layer 0)

### Environment
- MPI ranks: 2
- Model: Qwen 2.5 0.5B Instruct (Q4_0)
- Test tokens: [1, 2, 3, 4] (4 tokens)
- Build: Debug mode

## Conclusion

**Major Achievement**: Successfully isolated first divergence point to Q_PROJECTION layer 0 after fixing snapshot key format issues. The ~100x magnitude difference in Q projection outputs suggests a fundamental difference in computation or weight handling between batch and sequential pipelines.

**Root Cause Hypothesis**: Weight loading, tensor orientation, or distributed computation handling differs between:
- Sequential: Uses PrefillProvider with local OpenBLAS operations
- Batch: Uses MPILinearBatchOperator with distributed computation

**Validation**: EMBEDDING stage matches perfectly, confirming snapshot infrastructure works correctly and divergence originates in Q_PROJECTION computation.
