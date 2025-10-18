# Snapshot Comparison Investigation - Size Mismatch Root Cause
**Date**: 2025-01-16  
**Issue**: Comparison metrics show 0 despite obvious data differences

## Root Cause Identified

### Issue #1: Incorrect 3D Tensor Dimension Extraction (FIXED)

**Location**: `src/PipelineBase.cpp::captureIfEnabled()`

**Problem**: The dimension extraction logic didn't handle 3D tensors properly:
```cpp
// BEFORE (WRONG):
int seq_len = shape.size() >= 1 ? shape[0] : 0;    // Gets B=1
int feature_dim = shape.size() >= 2 ? shape[1] : 1; // Gets T=4
// For [1, 4, 896]: captured 1×4 = 4 elements instead of 1×4×896 = 3584!
```

**Fix**: Added proper 3D tensor handling:
```cpp
// AFTER (CORRECT):
if (shape.size() == 3) {
    // 3D tensor: [batch, seq_len, features]
    // Flatten batch*seq_len for snapshot capture
    seq_len = shape[0] * shape[1];  // B * T = 1 * 4 = 4
    feature_dim = shape[2];          // features = 896
}
// Now captures 4×896 = 3584 elements ✓
```

**Impact**: Fixed batch Q_PROJECTION size from 4 → 1792 elements

### Issue #2: Distributed vs Global Tensor Mismatch (ACTIVE)

**Comparison Issue**:
- Sequential Q_PROJECTION: **3584 elements** (4 tokens × 896 features)
  - 896 features = 14 heads × 64 head_dim (FULL global tensor)
- Batch Q_PROJECTION: **1792 elements** (4 tokens × 448 features)  
  - 448 features = 7 heads × 64 head_dim (LOCAL per-rank tensor)

**Architecture Difference**:
```
Sequential Pipeline (OpenBLASPrefillProvider):
├─ Captures AFTER MPI gather → full global tensor [seq_len, 896]
└─ Snapshot: 4 tokens × 896 features = 3584 elements

Batch Pipeline (MPIAttentionBatchOperator):
├─ Captures BEFORE MPI gather → local per-rank tensor [B, T, n_heads_local * head_dim]
└─ Snapshot: 1 × 4 × (7×64) = 1792 elements (only half the heads!)
```

**Why Metrics Showed Zero**:
When `SnapshotComparator::compare()` detects size mismatch:
```cpp
if (expected.size() != actual.size()) {
    result.error_message = "Size mismatch: expected 3584 but got 1792";
    result.metrics.passed = false;
    return result;  // Returns default-initialized metrics (all zeros)
}
```

### Issue #3: Local vs Global Statistics Mismatch

The test computes min/max/mean from the raw snapshot data:
```
Sequential (global): min=-79.862 max=48.9815 mean=-0.137529
Batch (local rank 0): min=-35.4329 max=47.6146 mean=0.187047
```

These values differ because:
1. Batch has only **7 of 14 heads** (rank 0's partition)
2. Different heads may have different activation distributions
3. Can't directly compare local partition stats to global stats

## Solutions

### Option A: Gather Before Snapshot in Batch Pipeline
Modify `MPIAttentionBatchOperator` to gather Q/K/V across ranks before snapshot:
```cpp
// After projection, gather to rank 0
if (snapshot_callback_ && rank == 0) {
    auto q_global = gatherAcrossRanks(q_local);  // Combine all 14 heads
    snapshot_callback_(PipelineStage::Q_PROJECTION, current_layer_idx_, q_global);
}
```
**Pros**: Direct apples-to-apples comparison  
**Cons**: Extra MPI communication just for snapshots

### Option B: Compare Local Partitions
Change sequential to also capture local partitions (per-rank):
```cpp
// Sequential: Capture BEFORE gather, same as batch
snapshot_per_rank(Q_local);  // Each rank captures its partition
```
**Pros**: No extra communication overhead  
**Cons**: More complex comparison logic, need to handle per-rank snapshots

### Option C: Document and Accept (Current State)
Document that batch snapshots are local partitions and adjust test expectations:
```cpp
// Test: Skip Q/K/V comparison or compare only AFTER gather
if (stage.name.find("PROJECTION") != std::string::npos) {
    // Skip - distributed tensor partition
    continue;
}
```
**Pros**: No code changes, acknowledge architectural difference  
**Cons**: Can't validate correctness of projections

## Recommended Approach

**Short-term**: Option C - Update test to skip local partition comparisons and focus on post-gather stages (ATTENTION_OUTPUT, FFN outputs).

**Long-term**: Option A - Add gather-before-snapshot for critical stages in batch pipeline for full parity validation.

## Files Modified

### src/PipelineBase.cpp
- Lines 612-637: Fixed `captureIfEnabled` 3D tensor handling
- Lines 666-691: Fixed second `captureIfEnabled` overload

### tests/test_batch_correctness.cpp
- Line 446: Added `SnapshotRegistry::clear()` call
- Lines 606-612: Added size mismatch debug output
- Lines 624-627: Added error message and size reporting

## Lessons Learned

1. **Always handle all tensor dimensionalities**: 1D, 2D, AND 3D
2. **Document distributed vs global tensors**: Make it clear in snapshot keys
3. **Size mismatch is not the same as value mismatch**: Return proper error codes
4. **Test infrastructure should match production architecture**: Don't force incompatible comparisons

## Next Steps

1. Update test to skip distributed tensor comparisons (Q/K/V_PROJECTION pre-gather)
2. Focus comparison on globally-consistent stages (ATTENTION_OUTPUT, FFN_RESIDUAL)
3. Add `_local` vs `_global` suffix to snapshot keys for clarity
4. Consider adding gather-before-snapshot for full parity in future
