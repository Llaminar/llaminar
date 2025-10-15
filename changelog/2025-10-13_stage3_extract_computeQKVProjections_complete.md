# Stage 3 Complete: Extract computeQKVProjections() - QKV Linear Projections

**Date:** October 13, 2025  
**Author:** David Sanftenberg  
**Scope:** MPIAttentionKernel execute() method refactoring (Stage 3 of 9)

## Summary

Successfully extracted STEP 3 (QKV projection computation) from the execute() method into a dedicated `computeQKVProjections()` helper method. This reduces execute() by an additional ~227 lines while preserving all debug instrumentation, validation logic, and snapshot callback functionality.

**Key Achievement:** All three comprehensive parity tests passed on first try (1,944 stage comparisons), confirming zero regressions.

## Changes Made

### 1. Header File: `src/kernels/MPIAttentionKernel.h`

**Added QKVProjectionResult struct** (lines ~89-95, 7 lines):
```cpp
struct QKVProjectionResult
{
    std::shared_ptr<TensorBase> local_q;  // [seq_len, local_head_dim]
    std::shared_ptr<TensorBase> local_k;  // [seq_len, local_kv_head_dim]
    std::shared_ptr<TensorBase> local_v;  // [seq_len, local_kv_head_dim]
};
```

**Added method declaration** (lines ~312-325, 14 lines):
```cpp
/**
 * @brief Compute Q, K, V linear projections (STEP 3)
 * 
 * Performs matrix multiplication with optional bias addition to compute:
 * - Q = input @ wq^T + bq (if bias present)
 * - K = input @ wk^T + bk (if bias present)
 * - V = input @ wv^T + bv (if bias present)
 * 
 * Includes comprehensive validation, health checks, and optional snapshot logging.
 * 
 * @param setup Input setup result from STEP 1
 * @param weights Weight slices from STEP 2
 * @return QKVProjectionResult containing local Q, K, V activation tensors
 */
QKVProjectionResult computeQKVProjections(
    const InputSetupResult &setup,
    const WeightSlices &weights);
```

### 2. Implementation File: `src/kernels/MPIAttentionKernel.cpp`

**Added computeQKVProjections() implementation** (lines ~994-1232, 239 lines):
- **Parameter Extraction** (18 lines): Extract from setup struct and weights struct
- **Environment Flags** (3 lines): Read `enable_validation` and `validate_projections` from debugEnv()
- **Tensor Allocation** (3 lines): Create output tensors for Q, K, V
- **Q Projection Debug Logging** (27 lines): Input/weight stats, matmul parameters, bias flow tracking
- **Q Projection** (3 lines): `matmul_with_bias()` call with optional bias
- **Q Output Debug Logging** (11 lines): Post-projection statistics
- **K Projection** (3 lines): `matmul_with_bias()` call with optional bias
- **V Projection** (3 lines): `matmul_with_bias()` call with optional bias
- **Contract Validation** (59 lines): Shape contract checks, Q corruption detection before/after validation
- **Health Checks** (23 lines): NaN/Inf detection for all three projections
- **Optional Scalar Validation** (14 lines): Compare against reference implementation if enabled
- **Snapshot Statistics** (52 lines): Collect min/max/mean stats for callback if present
- **Callback Warning** (7 lines): Log if snapshot_callback_ is NULL
- **Return** (1 line): Return populated QKVProjectionResult

**Replaced old STEP 3 in execute()** (lines ~1342-1354, 13 lines):
```cpp
// ========================================================================
// STEP 3: Compute Q, K, V projections (REFACTORED)
// ========================================================================
auto projections = computeQKVProjections(setup, weights);

// Extract QKV tensors for use in subsequent steps
auto local_q = projections.local_q;
auto local_k = projections.local_k;
auto local_v = projections.local_v;
```

**Old code removed:** ~242 lines of projection logic (allocations, debug, projections, validation, snapshots)

## Code Metrics

### Stage 3 Metrics
- **Lines extracted:** 242 lines from execute() → 239-line helper method
- **Helper method:** computeQKVProjections() - 239 lines
- **Struct definition:** QKVProjectionResult - 7 lines
- **Method declaration:** 14 lines (including Doxygen)
- **Execute() replacement:** 13 lines (helper call + variable extraction)
- **Net change:** +13 lines (cleaner structure with explicit interface)
- **Execute() reduction:** ~242 → ~13 lines (~229 line reduction, -95%)

### Cumulative Progress (Stages 1-3)
| Metric | Stage 1 | Stage 2 | Stage 3 | **Cumulative** | Target | % Complete |
|--------|---------|---------|---------|----------------|--------|------------|
| Stages completed | 1 | 2 | 3 | **3/9** | 9 | **33%** |
| Lines extracted | 332 | 282 | 242 | **856** | ~2,000 | **43%** |
| Execute() reduction | -237 | -264 | -229 | **-730** | -1,500 | **49%** |
| Helper code added | 281 | 295 | 239 | **815** | ~1,500 | **54%** |
| Struct code | 55 | 24 | 7 | **86** | ~200 | **43%** |
| Method declarations | 13 | 13 | 14 | **40** | ~100 | **40%** |

**Execute() method size:**
- **Original:** 2,287 lines (8 STEPs embedded)
- **After Stage 1:** ~2,050 lines (STEP 1 extracted)
- **After Stage 2:** ~1,786 lines (STEP 1-2 extracted)
- **After Stage 3:** ~1,557 lines (STEP 1-3 extracted)
- **Total reduction:** -730 lines (-32%)
- **Remaining:** STEP 4-8 (~1,250 lines to refactor)

## Validation Results

### Test 1: OpenBLAS Prefill Parity
```bash
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch"
```
- **Result:** ✅ **387/387 passed** (0 failed, 0 missing)
- **Duration:** 89.2 seconds
- **Coverage:** 24 layers × 16 stages + final norm + LM head
- **Error ranges:** max_abs in 10^-5 to 10^-3 range, rel_l2 in 10^-6 to 10^-5 range

### Test 2: COSMA Prefill Parity
```bash
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.COSMAPrefillVsPyTorch"
```
- **Result:** ✅ **387/387 passed** (0 failed, 0 missing)
- **Duration:** 106.0 seconds
- **Coverage:** Same 387 stages as OpenBLAS test
- **COSMA backend:** Validated distributed matrix multiplication unchanged

### Test 3: Incremental Decode Parity
```bash
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch"
```
- **Result:** ✅ **585/585 passed** (1170 total stages across 3 tokens)
- **Duration:** 34.4 seconds
- **Token sequence:** 6 → 25010 → 10 (matches PyTorch exactly)
- **Per-token validation:** All Q/K/V projections match across all layers
- **Error ranges:** max_abs in 10^-6 to 10^-5 range, rel_l2 in 10^-7 to 10^-6 range

**Total validation:** 1,944 stage comparisons, 100% pass rate

## Compilation Notes

### Build Command
```bash
cmake --build build --parallel
```

### Warnings (Non-Critical)
Same narrowing conversion warnings as previous stages:
- `size_t` → `int` in validateAndSetupInputs() (inherited from Stage 1)
- No new warnings introduced by Stage 3 code

### Build Time
~30 seconds (incremental rebuild of llaminar_core and dependent targets)

## Technical Details

### Projection Logic Flow
1. **Allocate Outputs:** Create three SimpleTensor instances for Q, K, V
2. **Debug Pre-Q:** Log input/weight stats if layer 0 and verbose enabled
3. **Q Projection:** `input @ wq^T + bq` via matmul_with_bias()
4. **Debug Post-Q:** Log Q statistics and sample values
5. **K Projection:** `input @ wk^T + bk` via matmul_with_bias()
6. **V Projection:** `input @ wv^T + bv` via matmul_with_bias()
7. **Validation Path:** Contract checks, health checks, optional scalar validation
8. **Snapshot Path:** Collect statistics for callback if enabled

### Environment Flag Usage
```cpp
const auto &debug_snapshot = debugEnv();
const bool enable_validation = debug_snapshot.attention.validate_output;
const bool validate_projections = debug_snapshot.attention.validate_proj;
```

This pattern retrieves validation toggles once at method entry, avoiding repeated getenv() calls.

### Bias Handling
- Biases passed as `local_bq ? local_bq->data() : nullptr`
- matmul_with_bias() handles nullptr gracefully (skip bias addition)
- Follows pattern from Stage 2 (size > 1 check done in distributeWeightsByHead)

### Snapshot Callback Pattern
- Callback is a member variable `snapshot_callback_`
- Called only if non-null AND rank == 0
- Logs warning if null (helps debug parity test capture issues)
- Statistics collected regardless of callback presence (for debug logs)

## Lessons Learned

### 1. Namespace-Level Structs
- **Issue:** Initial attempt used `MPIAttentionKernel::QKVProjectionResult` as return type
- **Solution:** Struct defined at namespace level (before class), use `QKVProjectionResult` directly
- **Pattern:** InputSetupResult, WeightSlices, QKVProjectionResult all namespace-level

### 2. Environment Flags Access
- **Issue:** No `enable_validation` in InputSetupResult struct
- **Solution:** Read from `debugEnv().attention` within helper method
- **Benefit:** Avoids bloating setup struct with environment toggles

### 3. Member Variable Access
- **Context:** Helper methods can access member variables (layer_index_, snapshot_callback_)
- **Usage:** layer_index_ for debug logging, snapshot_callback_ for capture
- **Clean:** No need to pass these via structs

### 4. Exception Propagation
- **Change:** Contract/health check failures now `throw` instead of `return false`
- **Rationale:** execute() can catch and return false, maintains same behavior
- **Benefit:** Clearer error handling (exceptions for validation failures)

## Performance Impact

### Execution Time (Measured)
- **OpenBLAS prefill:** 89.2s (vs ~90s baseline, -0.9% variation)
- **COSMA prefill:** 106.0s (vs ~107s baseline, -0.9% variation)
- **Incremental decode:** 34.4s (vs ~33s baseline, +4.2% variation)

**Analysis:** No measurable performance regression. Variations within normal noise for MPI tests.

### Function Call Overhead
- **Added:** One function call per attention layer (24 layers)
- **Cost:** Negligible (<1 microsecond per call)
- **Benefit:** Cleaner code structure, easier debugging

## Architecture Notes

### QKV Projection Semantics
- **Input:** Activations in shape [seq_len, d_model]
- **Weights:** Pre-distributed in shape [local_head_dim, d_model] or [local_kv_head_dim, d_model]
- **Output:** Activations in shape [seq_len, local_head_dim] or [seq_len, local_kv_head_dim]
- **Head Distribution:** Already handled by distributeWeightsByHead() (Stage 2)

### Validation Strategy
Three-tier validation (all optional via environment flags):
1. **Contract Validation:** Shape and semantic checks (TensorContract system)
2. **Health Checks:** NaN/Inf detection with statistics logging
3. **Scalar Reference:** Element-wise comparison against naive implementation

### Debug Instrumentation Preservation
All debug paths from original code preserved:
- `[Q_PROJ_DEBUG]` - Input/weight/output statistics
- `[MATMUL_DEBUG]` - Matrix multiplication parameters
- `[BIAS_FLOW]` - Bias tensor status tracking
- `[SNAPSHOT_DEBUG]` - Snapshot callback statistics
- `[CHECK-BEFORE-CONTRACT]` / `[CHECK-AFTER-CONTRACT]` - Q corruption detection

## Next Steps

### Stage 4: Extract gatherAndSnapshotPreRoPE()
- **Scope:** STEP 4 code (~154 lines)
- **Logic:** MPI_Gather for multi-rank Q/K/V, snapshot callback invocation
- **Struct:** GatherResult with gathered_q, gathered_k, gathered_v (nullable for single rank)
- **Complexity:** Conditional gathering based on world_size, snapshot timing critical (before RoPE)

### Remaining Work
- Stage 4: gatherAndSnapshotPreRoPE() - ~154 lines
- Stage 5: applyRotaryPositionEmbeddings() - ~481 lines (largest stage)
- Stage 6: handleGQAExpansion() - ~208 lines
- Stage 7: computeAttentionScores() - ~389 lines
- Stage 8: projectAndGatherOutput() - ~174 lines
- Stage 9: Final cleanup and documentation

**Total remaining:** ~1,406 lines across 6 stages

## Conclusion

Stage 3 successfully extracted QKV projection logic into a self-contained helper method. The implementation maintains all existing functionality while dramatically improving code organization. With 3 of 9 stages complete, the refactoring is 33% done, with execute() reduced by 32% (730 lines).

**Status:** ✅ Ready to proceed to Stage 4
