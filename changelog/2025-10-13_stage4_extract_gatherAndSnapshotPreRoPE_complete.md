# Stage 4: Extract gatherAndSnapshotPreRoPE() Helper Method

**Date:** 2025-10-13  
**Author:** David Sanftenberg  
**Branch:** feature/refactor-execute-orchestration  
**Stage:** 4/9 (Phase 8: Execute Method Refactoring)

## Summary

Successfully extracted STEP 4 (MPI gathering and snapshot callback invocation) from `MPIAttentionKernel::execute()` into a dedicated helper method `gatherAndSnapshotPreRoPE()`. This stage handles gathering Q/K/V projections across MPI ranks for snapshot comparison with PyTorch ground truth, with critical timing BEFORE RoPE application.

**Key Achievement:** All 3 parity tests passed with identical results (1,959 total comparisons).

## Changes Made

### Files Modified

1. **src/kernels/MPIAttentionKernel.h**
   - Added `GatherResult` struct (12 lines)
     - 3 nullable tensor pointers: `global_q`, `global_k`, `global_v`
     - `snapshot_performed` flag
   - Added `gatherAndSnapshotPreRoPE()` declaration (21 lines with Doxygen)

2. **src/kernels/MPIAttentionKernel.cpp**
   - Implemented `gatherAndSnapshotPreRoPE()` helper (166 lines)
     - Multi-rank path: MPI_Allgather + layout rearrangement + snapshot
     - Single-rank path: Direct snapshot of local tensors
     - No callback path: Return empty result
   - Replaced STEP 4 in `execute()` (~155 lines → ~7 lines)

### Code Structure

**GatherResult Struct:**
```cpp
struct GatherResult
{
    // Gathered global tensors (only allocated for multi-rank with callback)
    std::shared_ptr<TensorBase> global_q;  // [seq_len, n_head * head_dim] or nullptr
    std::shared_ptr<TensorBase> global_k;  // [seq_len, n_head_kv * head_dim] or nullptr
    std::shared_ptr<TensorBase> global_v;  // [seq_len, n_head_kv * head_dim] or nullptr
    
    // Flag indicating if snapshot was performed
    bool snapshot_performed = false;
};
```

**Method Signature:**
```cpp
GatherResult gatherAndSnapshotPreRoPE(
    const InputSetupResult &setup,
    const QKVProjectionResult &projections);
```

**Execute() Integration:**
```cpp
// STEP 4: Gather Q/K/V for snapshotting (BEFORE RoPE!) (REFACTORED)
auto gather_result = gatherAndSnapshotPreRoPE(setup, projections);
```

## Implementation Details

### Multi-Rank Gathering Path

**When:** `snapshot_callback_ && world_size > 1`

**Steps:**
1. Allocate global tensors for Q/K/V
2. Debug logging: Local Q stats, head boundaries
3. **MPI_Allgather Q:** Bulk gather into temp buffer (rank-major layout)
4. **Rearrange Q:** Double-loop transformation (rank-major → row-interleaved)
   ```cpp
   for (int t = 0; t < seq_len; ++t)
       for (int r = 0; r < world_size; ++r)
           memcpy(global + t*total_dim + r*local_dim,
                  temp + r*seq_len*local_dim + t*local_dim,
                  local_dim * sizeof(float));
   ```
5. Debug logging: Global Q stats, per-head verification, rank boundary check
6. **MPI_Allgather K/V:** Same bulk pattern
7. **Rearrange K/V:** Row-interleaved layout transformation
8. **Snapshot callback:** Invoke on rank 0 for Q_PROJECTION, K_PROJECTION, V_PROJECTION
9. Return `GatherResult` with populated tensors

### Single-Rank Path

**When:** `snapshot_callback_ && world_size == 1`

**Steps:**
1. Direct snapshot of local tensors (no MPI needed)
2. Invoke callback 3 times (Q/K/V projections)
3. Return `GatherResult` with snapshot_performed = true

### No Callback Path

**When:** `!snapshot_callback_` (production mode)

**Steps:**
1. Skip all gathering (expensive MPI overhead)
2. Return empty `GatherResult`

## Technical Challenges

### Layout Transformation

**Challenge:** MPI_Allgather produces rank-major layout `[rank0_all | rank1_all]`, but PyTorch expects row-interleaved layout `[token0: r0+r1 | token1: r0+r1]`.

**Solution:** Double-loop rearrangement with memcpy per (token, rank) pair:
- Outer loop: Iterate over tokens
- Inner loop: Iterate over ranks
- Copy: `local_dim` elements from temp to correct global position

**Performance:** O(seq_len × world_size × head_dim) complexity, but only executed in validation mode.

### Snapshot Timing

**Challenge:** PyTorch captures Q/K/V projections BEFORE RoPE application, but MPI implementation needs gathered tensors for comparison.

**Solution:** Explicit placement in execution flow as STEP 4 (after projections, before RoPE), with clear documentation and naming (`...PreRoPE`).

### Conditional Execution

**Challenge:** Gathering is expensive (all-to-all communication), should only run when needed.

**Solution:** Three-way conditional:
1. Multi-rank + callback: Full gather + snapshot
2. Single-rank + callback: Direct snapshot
3. No callback: Skip entirely (production fast path)

## Debug Instrumentation

### Q_GATHER_DEBUG Logging

**Before Gather (16 lines):**
- Local Q shape and dimensions
- First 10 values of token 0
- First 5 dims of each local head (up to 3 heads)

**After Gather (19 lines):**
- Global Q shape and first 10 values
- Per-head contribution (first 5 dims of up to 10 heads)
- Critical rank boundary check (heads 7 vs 8 for 2-rank split)

**Purpose:** Verify correct gathering and row-interleaved layout.

## Test Results

### Test 1: OpenBLAS Prefill vs PyTorch
- **Duration:** 90.7 seconds
- **Result:** ✅ **387/387 passed** (0 failed)
- **Q_PROJECTION validation:**
  - Layer 22: max_abs=5.29e-05, rel_l2=4.76e-06 ✓
  - Layer 23: max_abs=5.27e-05, rel_l2=3.76e-06 ✓
- **K_PROJECTION validation:**
  - Layer 22: max_abs=4.20e-05, rel_l2=4.16e-06 ✓
  - Layer 23: max_abs=4.65e-05, rel_l2=3.33e-06 ✓
- **V_PROJECTION validation:**
  - Layer 22: max_abs=6.68e-05, rel_l2=5.70e-06 ✓
  - Layer 23: max_abs=5.63e-05, rel_l2=5.32e-06 ✓

### Test 2: COSMA Prefill vs PyTorch
- **Duration:** 106.0 seconds
- **Result:** ✅ **387/387 passed** (0 failed)
- **Q_PROJECTION validation:**
  - Layer 22: max_abs=3.93e-05, rel_l2=1.86e-06 ✓
  - Layer 23: max_abs=3.81e-05, rel_l2=1.72e-06 ✓
- **K_PROJECTION validation:**
  - Layer 22: max_abs=3.58e-05, rel_l2=1.62e-06 ✓
  - Layer 23: max_abs=3.17e-05, rel_l2=1.40e-06 ✓
- **V_PROJECTION validation:**
  - Layer 22: max_abs=9.25e-05, rel_l2=2.37e-06 ✓
  - Layer 23: max_abs=6.56e-05, rel_l2=2.74e-06 ✓

### Test 3: Incremental Decode vs PyTorch
- **Duration:** 34.8 seconds
- **Result:** ✅ **585 stages passed, 1170 comparisons** (0 failed)
- **3 tokens decoded:** 6 → 25010 → 10
- **V_PROJECTION validation (decode mode):**
  - Layer 22, Token 2: max_abs=1.65e-05, rel_l2=6.26e-06 ✓
  - Layer 23, Token 2: max_abs=1.81e-05, rel_l2=6.26e-06 ✓

### Overall Results
- **Total comparisons:** 1,959 (387 + 387 + 1,170)
- **Pass rate:** 100% (1,959/1,959)
- **No behavior changes:** Gathering and snapshot logic identical to baseline

## Metrics

### Code Size Changes

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Execute() size | ~1,557 lines | ~1,410 lines | **-147 lines (-9%)** |
| Helper method | N/A | 166 lines | +166 lines |
| Struct definition | N/A | 12 lines | +12 lines |
| Method declaration | N/A | 21 lines | +21 lines |
| STEP 4 in execute() | ~155 lines | ~7 lines | **-148 lines (-95%)** |

### Cumulative Progress (Stages 1-4)

| Metric | Value | Progress |
|--------|-------|----------|
| Stages completed | 4/9 | **44%** |
| Lines extracted | 1,011 lines | **51% of ~2,000 target** |
| Execute() reduction | -878 lines | **59% of -1,500 target** |
| Helper code added | 981 lines | **65% of ~1,500 target** |
| Struct code | 98 lines | **49% of ~200 target** |
| Method declarations | 61 lines | **61% of ~100 target** |

### Execute() Size Progression

- **Original:** 2,287 lines
- After Stage 1: ~2,050 lines (-237)
- After Stage 2: ~1,786 lines (-264)
- After Stage 3: ~1,557 lines (-229)
- **After Stage 4:** ~1,410 lines (-147)
- **Total reduction:** -877 lines (-38%)

## Performance Considerations

### MPI Communication Cost
- **Operation:** MPI_Allgather (all-to-all communication)
- **Frequency:** Once per attention layer in validation mode
- **Cost:** O(seq_len × n_head × head_dim) data transfer
- **Mitigation:** Only execute when `snapshot_callback_` is set

### Layout Transformation Cost
- **Operation:** Double-loop rearrangement with memcpy
- **Complexity:** O(seq_len × world_size × head_dim)
- **Cost:** Minimal compared to MPI communication
- **Optimization:** Contiguous memcpy per (token, rank) pair

### Production Fast Path
- **Condition:** `snapshot_callback_ == nullptr` (inference mode)
- **Cost:** Single null check, immediate return
- **Impact:** Zero overhead in production

## Lessons Learned

### 1. Critical Timing Documentation
**Observation:** Snapshot must happen BEFORE RoPE, but this wasn't immediately obvious from code structure.

**Solution:** Explicit naming (`...PreRoPE`) and comprehensive Doxygen documentation emphasizing timing constraint.

**Impact:** Future maintainers will immediately understand ordering requirement.

### 2. Layout Transformation Complexity
**Observation:** MPI_Allgather layout (rank-major) differs from PyTorch layout (row-interleaved).

**Solution:** Double-loop rearrangement with clear documentation and debug logging to verify correctness.

**Impact:** Gathering logic is now testable and debuggable in isolation.

### 3. Conditional Execution Patterns
**Observation:** Three distinct execution paths (multi-rank+callback, single-rank+callback, no callback).

**Solution:** Early returns for simple cases, full implementation for complex path, clear guard conditions.

**Impact:** Minimal overhead for common cases, full functionality when needed.

### 4. Debug Instrumentation Value
**Observation:** Q_GATHER_DEBUG logging was invaluable during initial Phase 2A development.

**Solution:** Preserved all debug instrumentation in helper method.

**Impact:** Future debugging of gathering issues will be much easier.

## Future Work

### Remaining Refactoring Stages

**Stage 5 (NEXT): applyRotaryPositionEmbeddings()**
- **Lines to extract:** ~481 lines (LARGEST stage)
- **Logic:** RoPE frequency computation and rotation of Q/K tensors
- **Complexity:** High (extensive debug tracing, optional validation)

**Stage 6: handleGQAExpansion()**
- **Lines to extract:** ~208 lines
- **Logic:** Grouped Query Attention K/V expansion

**Stage 7: computeAttentionScores()**
- **Lines to extract:** ~389 lines
- **Logic:** Q@K^T computation, masking, softmax

**Stage 8: projectAndGatherOutput()**
- **Lines to extract:** ~174 lines
- **Logic:** Output projection and MPI gathering

**Stage 9: Final cleanup**
- Documentation updates
- Performance validation
- Code review

### Total Remaining Work
- **Stages:** 5/9 (56% to go)
- **Lines:** ~1,252 lines to extract
- **Target execute() size:** ~300 lines (orchestration only)

## Conclusion

Stage 4 successfully extracted MPI gathering and snapshot logic into a dedicated helper method, reducing `execute()` by 147 lines while maintaining 100% test parity. The three-way conditional execution pattern (multi-rank/single-rank/no callback) provides flexibility for validation vs production use cases with minimal overhead.

**Critical achievement:** All Q_PROJECTION, K_PROJECTION, and V_PROJECTION stages validated at correct timing (BEFORE RoPE), confirming proper MPI gathering and layout transformation.

**Next step:** Stage 5 will extract RoPE application (~481 lines), the largest remaining stage. This will reduce `execute()` to ~930 lines and reach 74% of total extraction goal.

---

**Related Changelogs:**
- 2025-10-12_stage1_extract_validateAndSetupInputs_complete.md
- 2025-10-12_stage2_extract_distributeWeightsByHead_complete.md
- 2025-10-13_stage3_extract_computeQKVProjections_complete.md
- 2025-10-13_stage4_extract_gatherAndSnapshotPreRoPE_complete.md (this file)
