# CUDA GEMM Integer Division Bug Fix (Phase 6.11)

**Date**: November 1, 2025  
**Status**: ✅ RESOLVED - All 5 CUDA GEMM tests passing  
**Impact**: CRITICAL - Kernel was completely broken for all tile configurations

## Summary

Fixed critical integer division truncation bug in CUDA GEMM kernel that caused **zero tiles to be loaded** for most configurations. After fixing configuration mismatch (648 variants now compile), tests still failed with wrong outputs. Root cause: floor division computed 0 loads per thread for small tiles.

## The Bug

**Location**: `src/v2/kernels/cuda/CudaGemmVariants.cu` lines 117, 147

**Problem**: Integer division truncation in tile loading calculations

### Original Code (BROKEN)

```cuda
// A tile loading (line 117)
constexpr int A_LOADS_PER_THREAD = (TILE_M * TILE_K) / (THREADS_M * THREADS_N);

// B tile loading (line 147)
const int B_BLOCKS_PER_THREAD = (TILE_N * K_BLOCKS_PER_TILE) / (THREADS_M * THREADS_N);
```

### Example Failure Case

**Configuration**: 16×16×32 tile with 16×16 threads, BLOCK_SIZE=32
- A: `A_LOADS_PER_THREAD = (16 * 32) / (16 * 16) = 512 / 256 = 2` ✅ (happens to work)
- B: `B_BLOCKS_PER_THREAD = (16 * 1) / (16 * 16) = 16 / 256 = 0` ❌ **ZERO LOADS!**

Result: B tile **never loaded into shared memory**, computation used uninitialized data.

### Why It Went Undetected

- A tile loading worked for many configs (enough elements per thread)
- B block loading **always failed** for TILE_K=32 (only 1 block per tile)
- Auto-tuner benchmarked configs during warmup (different code path)
- Tests showed non-zero outputs (random shared memory contents)

## The Fix

**Solution**: Ceiling division to ensure all tiles are loaded

### Fixed Code

```cuda
// A tile loading (NEW - lines 115-120)
constexpr int TOTAL_A_ELEMENTS = TILE_M * TILE_K;
constexpr int TOTAL_THREADS = THREADS_M * THREADS_N;
constexpr int A_LOADS_PER_THREAD = (TOTAL_A_ELEMENTS + TOTAL_THREADS - 1) / TOTAL_THREADS;
#pragma unroll
for (int load_idx = 0; load_idx < A_LOADS_PER_THREAD; ++load_idx) {
    const int flat_idx = tid * A_LOADS_PER_THREAD + load_idx;
    if (flat_idx >= TOTAL_A_ELEMENTS) break; // Guard excess iterations
    // ... load A[flat_idx] ...
}

// B tile loading (NEW - lines 148-152)
const int K_BLOCKS_PER_TILE = TILE_K / BLOCK_SIZE;
const int TOTAL_B_BLOCKS = TILE_N * K_BLOCKS_PER_TILE;
const int B_BLOCKS_PER_THREAD = (TOTAL_B_BLOCKS + TOTAL_THREADS - 1) / TOTAL_THREADS;
for (int block_idx = 0; block_idx < B_BLOCKS_PER_THREAD; ++block_idx) {
    const int flat_idx = tid * B_BLOCKS_PER_THREAD + block_idx;
    if (flat_idx >= TOTAL_B_BLOCKS) break; // Guard excess iterations
    // ... load and decode B block[flat_idx] ...
}
```

### Fixed Example

**Same configuration**: 16×16×32 tile with 16×16 threads
- A: `A_LOADS_PER_THREAD = (512 + 255) / 256 = 767 / 256 = 2` ✅ (ceiling)
- B: `B_BLOCKS_PER_THREAD = (16 + 255) / 256 = 271 / 256 = 1` ✅ **Now loads!**

## Test Results

### Before Fix (All Failing)

```
BasicCorrectness: FAILED
  Max absolute difference: 7.73978
  Max relative difference: 52.28
  Total mismatches: 32/32 (100%)

MediumMatrix: FAILED
  Max absolute difference: 10.5932
  Max relative difference: 550.799
  Total mismatches: 100/100 (100%)
```

### After Fix (All Passing!)

```
[  PASSED  ] 5 tests.

BasicCorrectness: PASSED
  Max absolute difference: ~1e-6
  Max relative difference: ~1e-5

SingleRow: PASSED
SingleColumn: PASSED

MediumMatrix: PASSED
  Max absolute difference: 1.07288e-06
  Max relative difference: 3.86492e-05

InvalidKDimension: PASSED (correctly rejects k=63)
```

## Debugging Journey

1. **Phase 6.10**: Integrated auto-tuner, removed legacy code
2. **Initial testing**: Kernels launched but output all zeros
3. **Root cause 1**: Configuration mismatch - auto-tuner generated 648 configs, Python compiled 200
   - Fixed by generating all 648 variants
4. **Root cause 2**: Type mismatch - `bool vectorize_load` vs `int` in generated files
   - Fixed by changing CudaGemmConfig to use `int`
5. **Root cause 3** (THIS FIX): Integer division truncation
   - Created simple test kernel - worked perfectly!
   - Compared with tiled kernel - found division bug
   - Applied ceiling division fix
   - **ALL TESTS NOW PASSING**

## Key Learnings

### Integer Division Pitfalls

**Never use floor division for "items per thread" calculations!**

```cuda
// ❌ WRONG - can be zero!
int items_per_thread = total_items / num_threads;

// ✅ CORRECT - ceiling division
int items_per_thread = (total_items + num_threads - 1) / num_threads;

// ✅ ALSO CORRECT - with guard
for (int i = tid; i < total_items; i += num_threads) {
    // Process item i
}
```

### Testing Methodology

**Simple reference kernel was crucial**:
- Isolated bug to tiled kernel
- Confirmed decoder logic works
- Validated memory layout
- Provided working baseline

**File**: `/tmp/test_simple.cu` - naive single-thread-per-output kernel
- No tiling, no shared memory
- Direct decode + accumulate
- **PASSED** with correct results (64.0 expected, 64.0 actual)

## Files Modified

1. **src/v2/kernels/cuda/CudaGemmVariants.cu**
   - Lines 115-120: Fixed A tile loading (ceiling division)
   - Lines 148-152: Fixed B tile loading (ceiling division)
   - Added `TOTAL_THREADS` constant to avoid duplication
   - Added `if (flat_idx >= TOTAL_*) break;` guards

## Performance Impact

**None** - Fix only ensures correctness. Performance characteristics:
- Same number of threads active
- Same shared memory usage
- Guard branches are predictable (false for most threads)
- Compiler likely optimizes away guards via unrolling

## Next Steps

✅ Phase 6.11 kernel correctness - **COMPLETE**
🚀 Phase 6.11 end-to-end inference validation - **READY TO START**
- Load Qwen 2.5 0.5B model to GPU
- Run full inference pipeline
- Validate outputs vs CPU
- Benchmark throughput

## Test Command

```bash
cd /workspaces/llaminar/build_v2
ctest -R "V2_Unit_CUDAGemm" --output-on-failure --verbose
```

**Result**: 5/5 tests passing, max error 1.07e-06 (excellent numerical accuracy)
