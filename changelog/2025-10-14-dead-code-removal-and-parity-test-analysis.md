# Dead Code Removal & Obsolete Test Cleanup
**Date:** October 14, 2025  
**Author:** GitHub Copilot (with David Sanftenberg)

## Summary

Successfully identified and removed dead code from the repository, analyzed the failing `IncrementalDecodeVsPyTorch` parity test to determine root cause, and removed obsolete parity tests from the test suite.

## Dead Code Removal

### Removed Files
The following kernel files were identified as dead code and removed from the build system:

1. **MPIMLPKernel.cpp / MPIMLPKernel.h**
   - **Status**: Fully implemented MPI kernel, compiled but never instantiated
   - **Reason**: QwenPipeline uses modular approach with separate `MPISwiGLUKernel` + `MPILinearKernel` instead of unified MLP kernel
   - **Size**: 458 lines in .cpp + substantial header
   - **Note**: Files were already deleted from filesystem; only CMakeLists.txt references remained

2. **RMSNormKernel.h**
   - **Status**: Header-only, never included by any .cpp file  
   - **Reason**: Replaced by `MPIRMSNormKernel` (MPI-aware version)
   - **Note**: Legacy pre-MPI artifact

3. **test_mlp_shard_correctness.cpp**
4. **test_mlp_tp_correctness.cpp**  
5. **test_end2end_shard_correctness.cpp**
   - **Status**: Test files already deleted; removed from CMakeLists.txt test registration
   - **Test count removed**: ~15 test cases

### Changes Made

**CMakeLists.txt:**
- Removed `src/kernels/MPIMLPKernel.cpp` from core library sources (line 146)
- Removed all MLP test executable definitions and test registrations
- Removed end2end shard test definitions

### Build Verification

✅ Clean build successful after removal:
```bash
cmake --build build --parallel
# 100% success - all 100+ targets built
```

✅ Smoke tests passing:
```bash
ctest -R "^(BasicTest|NumaTest|PipelineFactoryTest)$"
# 3/3 tests passed in 1.23s
```

---

## Active Kernel Inventory

### Used by QwenPipeline (7 MPI kernels)
1. `MPIEmbeddingKernel` - Token embeddings
2. `MPIRMSNormKernel` - Layer normalization  
3. `MPIAttentionKernel` - Multi-head attention
4. `MPILinearKernel` - Linear projections
5. `MPISwiGLUKernel` - SwiGLU activation
6. `MPIRoPEKernel` - Rotary position embeddings
7. `MPIResidualKernel` - Residual connections

### Active Primitive Files (5 files)

#### RMSNorm Primitives (both actively used)
- **`rmsnorm_core.cpp/h`** - Used by:
  - `qwen_pipeline.cpp` line 1661: `rmsnorm_row_major_fused()`
  - `MPIRMSNormKernel.cpp`: `rmsnorm_compute_row_sumsq()`, `rmsnorm_apply()`
  
- **`rmsnorm_t5.cpp/h`** - Used by:
  - `cosma_prefill_manager.cpp` line 983: `rmsnorm_t5_forward()`
  - `MPIRMSNormKernel.cpp` line 508: `rmsnorm_t5_forward_double_acc()`

#### Other Primitives
- `attention_primitives.cpp/h` - Included by `qwen_pipeline.cpp`
- `softmax_core.cpp/h` - Core softmax operations
- `attention/AttentionValidator.cpp/h` - Validation infrastructure

**Note:** Both `rmsnorm_core` and `rmsnorm_t5` serve different purposes:
- `rmsnorm_core`: Flexible fused implementation with options for sharding, parallelism
- `rmsnorm_t5`: HuggingFace-matching T5-style implementation for parity testing

---

## Parity Test Analysis

### Test Results

**ParityFrameworkTest** (total 8 tests):
- ✅ **7 tests PASSED** (87.5%)
- ❌ **1 test FAILED**: `ParityFramework.IncrementalDecodeVsPyTorch`
- ⏭️ **1 test SKIPPED**: `ParityFramework.DistributedPipelineVsPyTorchReference`

### Passing Tests Include:
- ✅ `TrueIncrementalDecodeVsPyTorch` - **100% pass rate (1170/1170 stages passing)**
- ✅ `COSMAPrefillVsPyTorch`
- ✅ `OpenBLASPrefillVsPyTorch`  
- And 4 more...

### Failing Test Analysis: `IncrementalDecodeVsPyTorch`

**Root Cause Identified:**

The test is failing due to **ATTENTION_SCORES size mismatch** between PyTorch snapshots and Llaminar output:

```
Decode step 0: PyTorch expects 504 elements, Llaminar has 84
Decode step 1: PyTorch expects 686 elements, Llaminar has 98
Decode step 2: PyTorch expects 868 elements, Llaminar has 112
```

**Technical Analysis:**

1. **PyTorch Snapshot Issue:**
   - The test uses `generate_pytorch_decode_snapshots()` which calls the **old** `generate_variance_thresholds.py` script
   - This script performs **full forward passes with ALL tokens** (prefill + decode), not true incremental decode
   - For decode step 0 with 6 tokens total: PyTorch generates ATTENTION_SCORES for ALL 6 query tokens
   - Shape: `[n_heads=14, all_q_tokens=6, k_len=6]` = 504 elements

2. **Llaminar Behavior (Correct):**
   - Llaminar performs **true incremental decode**: processes only the NEW token
   - For decode step 0: ATTENTION_SCORES for 1 query token only
   - Shape: `[n_heads=14, q_len=1, k_len=6]` = 84 elements

3. **Why TrueIncrementalDecodeVsPyTorch Works:**
   - Uses newer `generate_incremental_decode_snapshots.py` script
   - Properly implements per-token incremental decode via KV cache
   - Generates snapshots that match Llaminar's true incremental behavior
   - **Result: 100% pass rate (1170/1170 stages)**

**Comparison Failure Details:**
- 363/387 stages passing per decode step (94% pass rate)
- ALL 24 failures are ATTENTION_SCORES_layer* (one per layer)
- All other stages (Q_PROJECTION, K_PROJECTION, ATTENTION_SOFTMAX, etc.) **pass correctly**

### Recommended Fix

**Option 1: Disable Legacy Test (RECOMMENDED)**
```cpp
TEST(ParityFramework, DISABLED_IncrementalDecodeVsPyTorch)
```

**Rationale:**
- `TrueIncrementalDecodeVsPyTorch` provides superior coverage (1170 stages vs 387)
- Uses correct incremental decode methodology
- Already passing at 100%
- The old test uses deprecated snapshot generation approach

**Option 2: Fix Snapshot Generation**
- Update `generate_variance_thresholds.py` to use proper incremental decode
- This would require significant refactoring of the variance threshold system
- Not recommended since `TrueIncrementalDecodeVsPyTorch` already provides this

**Option 3: Skip ATTENTION_SCORES in Old Test**
- Add exclusion filter for ATTENTION_SCORES stages in `IncrementalDecodeVsPyTorch`
- Still leaves test using deprecated methodology
- Not recommended

---

## Verification Summary

### Build Status
- ✅ Clean build with no errors or warnings
- ✅ All 100+ build targets successful
- ✅ Removed dead code files from compilation
- ✅ Removed obsolete test registrations

### Test Status  
- ✅ Core functionality tests passing (BasicTest, NumaTest, PipelineFactoryTest)
- ✅ Primary parity test `TrueIncrementalDecodeVsPyTorch` passing at 100%
- ⚠️ Legacy parity test `IncrementalDecodeVsPyTorch` has known ATTENTION_SCORES issue (deprecated)

### Code Quality
- ✅ No dead MPI kernel code in build
- ✅ No obsolete test executables
- ✅ Clean separation of RMSNorm implementations (core vs T5)
- ✅ Proper kernel usage patterns verified

---

## Next Steps

1. **Disable the legacy test** to clean up test suite:
   ```cpp
   // tests/test_parity_framework.cpp line 2290
   TEST(ParityFramework, DISABLED_IncrementalDecodeVsPyTorch)
   ```

## Obsolete Test Removal (Completed)

### Tests Removed

1. **IncrementalDecodeVsPyTorch**
   - **Reason**: Uses deprecated snapshot generation methodology
   - **Replacement**: `TrueIncrementalDecodeVsPyTorch` (100% pass rate)
   - **Action**: Replaced test body with `GTEST_SKIP()` message directing users to replacement test

2. **DistributedPipelineVsPyTorchReference**
   - **Reason**: Deprecated; replaced by backend-specific tests
   - **Replacements**: `COSMAPrefillVsPyTorch` and `OpenBLASPrefillVsPyTorch`
   - **Action**: Removed test body and replaced with comment explaining removal

### Verification

✅ Build successful after test removal:
```bash
cmake --build build --parallel
# All targets built successfully
```

✅ Parity test suite verification:
```bash
mpirun -np 2 ./build/test_parity_framework --gtest_filter="ParityFramework.IncrementalDecodeVsPyTorch"
# Test properly skipped with message:
# "Removed: This test used deprecated snapshot generation. 
#  Use ParityFramework.TrueIncrementalDecodeVsPyTorch instead (100% pass rate)."
```

✅ Active parity tests:
- `ParityFramework.BasicSnapshotCapture` - ✓ Pass
- `ParityFramework.SnapshotComparison` - ✓ Pass
- `ParityFramework.OpenBLASPrefillVsPyTorch` - Active (requires model)
- `ParityFramework.COSMAPrefillVsPyTorch` - Active (requires model)
- `ParityFramework.CosmaModeValidation` - ✓ Pass
- `ParityFramework.TrueIncrementalDecodeVsPyTorch` - Active (100% pass rate, requires model)
- `ParityFramework.IncrementalDecodeVsPyTorch` - ⏭️ Skipped (obsolete)

---

## Files Modified

- `CMakeLists.txt` - Removed MPIMLPKernel.cpp and obsolete test definitions
- `tests/test_parity_framework.cpp` - Removed obsolete parity tests (IncrementalDecodeVsPyTorch, DistributedPipelineVsPyTorchReference)
- **No kernel source code changes** - dead code files were already deleted from filesystem

## Documentation Notes

The test suite now has a clear canonical structure:
- **Unit Tests**: `BasicSnapshotCapture`, `SnapshotComparison`, `CosmaModeValidation`
- **Prefill Validation**: `OpenBLASPrefillVsPyTorch`, `COSMAPrefillVsPyTorch`
- **Incremental Decode Validation**: `TrueIncrementalDecodeVsPyTorch` (primary)

Old tests using deprecated snapshot generation methodology have been removed from active testing.
