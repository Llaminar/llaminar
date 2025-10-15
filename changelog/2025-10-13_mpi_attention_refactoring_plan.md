# MPIAttentionKernel::execute() Refactoring Plan
**Date**: 2025-10-13  
**Author**: David Sanftenberg  
**Status**: Planning Phase

## Problem Statement

The `MPIAttentionKernel::execute()` method has grown to ~2300 lines, making it difficult to:
- Understand the high-level flow
- Debug individual stages
- Make targeted modifications
- Review code changes

## Objective

Refactor `execute()` into a clean orchestration method that delegates to stage-specific helper methods, while:
- Maintaining all existing functionality
- Preserving performance characteristics
- Keeping helpers inline-eligible where beneficial
- Ensuring all tests pass after each stage

## Current Structure Analysis

```
execute() method: Lines 389-2676 (~2287 lines)

├── STEP 1: Validate inputs and extract parameters (332 lines)
│   ├── Input validation
│   ├── Tensor extraction
│   ├── Mode detection (prefill vs decode)
│   ├── Head distribution calculation
│   └── Early exit for ranks with no work
│
├── STEP 2: Distribute weights by head dimension (282 lines)
│   ├── Weight sharding/slicing by head
│   └── Bias handling
│
├── STEP 3: Compute Q, K, V projections (242 lines)
│   ├── Linear projections (matmul)
│   └── KV cache management
│
├── STEP 4: Gather Q/K/V for snapshotting (154 lines)
│   ├── MPI_Allgatherv for snapshots
│   └── Pre-RoPE validation
│
├── STEP 5: Apply RoPE to Q and K (481 lines)
│   ├── Rotary position embeddings
│   └── Post-RoPE validation
│
├── STEP 6: Handle GQA - replicate K/V heads (208 lines)
│   ├── Group query attention expansion
│   └── KV cache gathering and replication
│
├── STEP 7: Compute attention scores and apply softmax (389 lines)
│   ├── Score computation
│   ├── Softmax
│   └── Context aggregation
│
└── STEP 8: Output projection + MPI gather (174 lines)
    ├── Output projection
    ├── Final MPI gather
    └── Return KV caches
```

## Refactoring Strategy

### Phase 1: Extract Step-by-Step (One at a Time)

Each step will be extracted in sequence with full testing between each extraction:

#### Stage 1: Input Validation & Setup
**Helper Method**: `validateAndSetupInputs()`
- **Lines**: 412-743
- **Returns**: `struct InputSetupResult { tensors, params, head_distribution, early_exit_flag }`
- **Complexity**: Medium (many locals, but self-contained)
- **Test After**: All 3 parity tests

#### Stage 2: Weight Distribution
**Helper Method**: `distributeWeightsByHead()`
- **Lines**: 744-1025
- **Returns**: `struct WeightSlices { wq_local, wk_local, wv_local, wo_local, biases }`
- **Complexity**: Medium (tensor slicing logic)
- **Test After**: All 3 parity tests

#### Stage 3: QKV Projections
**Helper Method**: `computeQKVProjections()`
- **Lines**: 1026-1267
- **Returns**: `struct ProjectionResult { local_q, local_k, local_v, k_cache, v_cache }`
- **Complexity**: High (matmul backend selection, cache management)
- **Test After**: All 3 parity tests

#### Stage 4: Pre-RoPE Snapshotting
**Helper Method**: `gatherAndSnapshotPreRoPE()`
- **Lines**: 1268-1421
- **Returns**: `void` (modifies snapshots via callback)
- **Complexity**: Low (mostly MPI + snapshot calls)
- **Test After**: All 3 parity tests

#### Stage 5: RoPE Application
**Helper Method**: `applyRotaryPositionEmbeddings()`
- **Lines**: 1422-1902
- **Returns**: `void` (modifies local_q, local_k in-place)
- **Complexity**: Medium (RoPE logic, validation)
- **Test After**: All 3 parity tests

#### Stage 6: GQA Head Replication
**Helper Method**: `handleGQAExpansion()`
- **Lines**: 1903-2110
- **Returns**: `struct GQAResult { k_expanded, v_expanded, attn_seq_len }`
- **Complexity**: High (KV gathering, phase 4/5 optimizations, GQA logic)
- **Test After**: All 3 parity tests

#### Stage 7: Attention Computation
**Helper Method**: `computeAttentionScores()`
- **Lines**: 2111-2499
- **Returns**: `struct AttentionResult { attended_values }`
- **Complexity**: Very High (scoring, softmax, context aggregation, snapshotting)
- **Test After**: All 3 parity tests

#### Stage 8: Output Projection & Finalization
**Helper Method**: `projectAndGatherOutput()`
- **Lines**: 2500-2673
- **Returns**: `bool` (success/failure)
- **Complexity**: Medium (output projection, final gather, cache return)
- **Test After**: All 3 parity tests

### Phase 2: Final Cleanup

After all stages extracted:
- Review for common patterns
- Extract further sub-helpers if beneficial
- Add inline hints where appropriate
- Update documentation

## Testing Protocol

After each stage extraction:

```bash
# 1. Build
cmake --build build --target test_parity_framework --parallel

# 2. Run all 3 parity tests
timeout 120 mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch"
timeout 120 mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.COSMAPrefillVsPyTorch"
timeout 120 mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch"

# Expected: 387/387, 387/387, 585/585 (all passing)
```

## Design Principles

1. **Minimal State Transfer**: Pass only required data between stages
2. **Clear Ownership**: Use move semantics where appropriate
3. **Preserve Debug Guards**: Keep all `debugEnv()` checks intact
4. **Inline Eligibility**: Keep helpers under ~200 lines where possible
5. **Self-Documenting**: Method names should clearly indicate purpose
6. **Error Handling**: Preserve all validation and error paths

## Expected Final Structure

```cpp
bool MPIAttentionKernel::execute(inputs, outputs) {
    // Setup and validation
    auto setup = validateAndSetupInputs(inputs, outputs);
    if (setup.early_exit) return setup.success;
    
    // Weight distribution
    auto weights = distributeWeightsByHead(setup);
    
    // QKV projections
    auto projections = computeQKVProjections(setup, weights);
    
    // Pre-RoPE snapshotting
    gatherAndSnapshotPreRoPE(projections, setup);
    
    // RoPE application
    applyRotaryPositionEmbeddings(projections, setup);
    
    // GQA expansion
    auto gqa = handleGQAExpansion(projections, setup);
    
    // Attention computation
    auto attention = computeAttentionScores(gqa, setup);
    
    // Output projection and finalization
    return projectAndGatherOutput(attention, setup, outputs);
}
```

## Success Criteria

- ✅ All parity tests passing after each stage
- ✅ No performance regression (verify with benchmarks if needed)
- ✅ Execute() method under 100 lines
- ✅ Each helper under 250 lines
- ✅ All debug guards preserved
- ✅ All error handling preserved

## Timeline

Estimated: 8 refactoring stages × 15-30 minutes each = 2-4 hours total
(Including build and test time for each stage)

## Notes

- The user requested one stage at a time with testing between each
- Must preserve all existing functionality including debug guards
- Phase 4/5 MPI optimizations in STEP 6 are critical - be careful
- Snapshot callbacks are used for testing - must preserve exact behavior

---

**Next Action**: Begin with Stage 1 (validateAndSetupInputs)
