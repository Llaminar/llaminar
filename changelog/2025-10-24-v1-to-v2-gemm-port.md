# V1 to V2 GEMM Implementation Port - October 24, 2025

## Objective

Port the complete V1 IQ4_NL quantized GEMM implementation to V2 to achieve performance parity (target: 314 GFLOPS on Q-Proj 1024 workload).

## Changes Made

### 1. Complete Algorithm Port

Ported the full V1 row-wise GEMM implementation from `src/v1/tensors/IQ4_NLTensor.h` (lines 750-1100) to `src/v2/kernels/cpu/QuantizedGemm.cpp`:

**Adaptive Tiling Logic** (Lines 105-196):
- ✅ Identical aspect ratio detection (wide output, square, tall matrix)
- ✅ Memory-bound path (FFN): 64×32 tiles
- ✅ Compute-bound path (Q-proj): Size-dependent tiles
  - m >= 4096: 64×32  
  - m >= 2048: 64×32
  - **m >= 1024: 32×32** (empirically optimal from V1 tile sweep)
  - m >= 512: 96×96
  - m < 512: 128×128
- ✅ Tall matrix path: 64×24, 96×32, 128×48
- ✅ Manual override support via `LLAMINAR_IQ4_M_TILE` / `LLAMINAR_IQ4_N_TILE`

**4-Column Vectorized Microkernel** (Lines 120-158):
```cpp
if (env.dequant.iq4_gemm_microkernel && n_block >= 4) {
    // Decode 4 columns simultaneously
    for (; j_vec + 4 <= n_block; j_vec += 4) {
        for (int kb = 0; kb < num_k_blocks; ++kb) {
            for (int jv = 0; jv < 4; ++jv) {
                decoder_->decode_block_at(j, kb, B_col + k_start);
            }
        }
    }
}
```
- Reduces loop overhead
- Improves instruction pipelining
- Matches V1's microkernel pattern exactly

**Software Prefetching** (Lines 168-175):
```cpp
if (ii + M_TILE < m) {
    const float *next_A = A + (ii + M_TILE) * k;
    for (int pf = 0; pf < std::min(M_TILE, m - ii - M_TILE); pf += 8) {
        __builtin_prefetch(next_A + pf * k, 0, 1);
    }
}
```
- Prefetches next M-tile of activations
- Low temporal locality hint (0, 1)
- Improves cache behavior

### 2. Tile Size Correction

**Before:**
- m >= 1024: 64×32 tiles (V1's "adaptive" default, suboptimal)

**After:**
- m >= 1024: 32×32 tiles (V1's empirically optimal from tile sweep)

**Rationale:**
From `changelog/2025-10-22-tile-sweep-analysis.md`:
- V1 with 32×32: **314.34 GFLOPS**
- V1 with 64×64 (adaptive): 233.77 GFLOPS  
- **Improvement: +34%**

## Performance Results

### Test Configuration
- Workload: Q-Proj 1024 (m=1024, n=896, k=896)
- Model: Qwen 2.5 0.5B IQ4_NL (`qwen2.5-0.5b-instruct-q4_0.gguf`)
- Build: Release with `-march=native -mtune=native`
- CPU: AVX512F, AVX2, FMA enabled
- Ranks: Single rank (V2 doesn't support MPI yet)

### Baseline (Before Port)
```
V2 with 64×32 tiles: 33.58 GFLOPS
V2 with microkernel:  26.94 GFLOPS (21% SLOWER - broken!)
```

### After Complete V1 Port
```
V2 with 32×32 tiles: 32.62 GFLOPS
V2 with microkernel:  (not tested - known broken)
```

**Result:** ❌ **NO IMPROVEMENT** despite identical algorithm

### V1 Baseline (Historical)
```
V1 with 32×32 tiles + microkernel: 314.34 GFLOPS
V1 with 64×64 tiles (adaptive):    233.77 GFLOPS
```

## Performance Gap Analysis

### Current State
- **V2 Performance:** 32.62 GFLOPS
- **V1 Performance:** 314.34 GFLOPS  
- **Gap:** **9.6× slower**

### Confirmed Working Components
1. ✅ AVX512/AVX2 SIMD code paths (verified in previous session)
2. ✅ `dot_product_simd()` implementation identical to V1
3. ✅ Adaptive tiling logic matches V1 exactly
4. ✅ Tile sizes corrected to empirical optimum (32×32)
5. ✅ 4-column microkernel pattern ported
6. ✅ Software prefetching added

### Suspected Issues

#### 1. Microkernel Not Functional
- Enabling `LLAMINAR_IQ4_GEMM_MICROKERNEL=1` **reduces** performance by 21%
- V1's 314 GFLOPS was achieved **with microkernel enabled**
- V2's microkernel implementation may be incomplete or buggy

#### 2. Architectural Differences
**V1 Architecture:**
- Uses `MPILinearOperator_v2` wrapper
- Quantized weights accessed via global reference with `row_offset`/`row_count`
- Tested with 2 MPI ranks (though local GEMM should be same)
- May have additional optimizations in operator layer

**V2 Architecture:**
- Direct kernel invocation (operator-free design)
- Generic `IBlockDecoder` interface (virtual function overhead?)
- Single-rank execution only
- Simpler call path but potentially missing optimizations

#### 3. Decode Performance
V2 uses `IBlockDecoder` interface with virtual function calls:
```cpp
decoder_->decode_block_at(j, kb, B_col + k_start);  // Virtual dispatch
```

V1 uses direct method calls on concrete `IQ4_NLTensor`:
```cpp
tensor_->decode_block_at(row_offset + j, kb, B_col + k_start);  // Direct call
```

**Impact:** Virtual dispatch overhead on hot path (called ~800k times for m=1024 workload)

#### 4. OpenMP Configuration
V1 may be using different OpenMP settings optimized for 2-rank MPI execution.
V2 test runs with default OpenMP configuration.

#### 5. Compiler Optimizations
V1 and V2 may have different CMake optimization flags beyond `-march=native`.

## Recommendations

### Immediate Next Steps

1. **Fix V2 Microkernel** (CRITICAL)
   - Current implementation makes performance worse
   - V1 achieves +35% gain with microkernel
   - Investigate why V2's microkernel degrades performance

2. **Eliminate Virtual Dispatch Overhead**
   - Replace `IBlockDecoder*` with template parameter or inline dispatch
   - Benchmark impact: virtual call overhead on 800k+ iterations

3. **Profile V1 vs V2 Side-by-Side**
   - Use `perf` to identify where time is spent
   - Compare instruction counts, cache misses, branch mispredictions
   - Verify actual CPU utilization (OpenMP threads)

4. **Verify Build Configuration Match**
   - Compare V1 and V2 CMake flags
   - Check OpenMP settings (`OMP_NUM_THREADS`, `OMP_PLACES`, etc.)
   - Ensure same optimization level (`-O3 -DNDEBUG`)

5. **Create Minimal Reproduction**
   - Build standalone V1 single-rank benchmark
   - Compare apples-to-apples: same model, same rank count, same OpenMP config

### Long-Term Considerations

**Option A: Accept V2 Performance for Now**
- Document that V2 is experimental
- Focus on correctness over performance
- Revisit optimization when V2 architecture stabilizes

**Option B: Deep Performance Investigation**
- Allocate 1-2 days for profiling and optimization
- Goal: Achieve 90% of V1 performance (280+ GFLOPS)
- May require architectural changes to V2

**Option C: Hybrid Approach**
- Use V1 GEMM kernels from V2 (link V1 library)
- Treat V1 GEMM as "production" backend
- Focus V2 development on new features (multi-GPU, etc.)

## Files Modified

### `/workspaces/llaminar/src/v2/kernels/cpu/QuantizedGemm.cpp`
**Function:** `QuantizedGemmKernel::multiply_row_wise()`
**Changes:**
- Lines 105-196: Complete adaptive tiling logic from V1
- Lines 120-158: 4-column vectorized microkernel pattern
- Lines 168-175: Software prefetching
- Lines 47-73: Corrected tile sizes (32×32 for m=1024)

**Before (Lines of Code):** ~80 lines
**After (Lines of Code):** ~200 lines
**Diff:** +120 lines (complete V1 algorithm)

### `/workspaces/llaminar/tests/v2/performance/Perf__IQ4_NL_GEMM.cpp`
**Added:** `QProjComparison_1024` test case (lines 320-370)
**Purpose:** Direct V1 comparison benchmark
**Expected:** >150 GFLOPS (half of V1's 314)
**Actual:** 32.62 GFLOPS (**FAILED**)

## Test Results

```bash
# Test without microkernel
./build_v2_release/performance/v2_perf_iq4nl_gemm --gtest_filter="*.QProjComparison_1024"

╔════════════════════════════════════════════════════════════════╗
║ Q-Proj 1024 (V1 Comparison, 1024x896x896)                      ║
╠════════════════════════════════════════════════════════════════╣
║ FP32 Activation Path:                                          ║
║   Time per iter:    50.400     ms                               ║
║   Throughput:       32.62      GFLOPS                           ║
║   Bandwidth:        0.15       GB/s                             ║
╚════════════════════════════════════════════════════════════════╝

Expected: (gflops) > (150.0), actual: 32.62 vs 150
[  FAILED  ] IQ4_NL_GEMM_Perf.QProjComparison_1024
```

## Conclusion

**Status:** ❌ **Performance parity NOT achieved**

Despite porting the complete V1 GEMM implementation to V2 (identical algorithm, tile sizes, microkernel pattern, prefetching), V2 achieves only **32.62 GFLOPS** compared to V1's **314.34 GFLOPS**.

**Root cause:** Likely architectural differences between V1 and V2:
- Virtual dispatch overhead (IBlockDecoder interface)
- Missing microkernel functionality (broken implementation)
- Different OpenMP/MPI configuration
- Possible operator-layer optimizations in V1

**Next steps:** Requires deep profiling to identify bottleneck. Consider using V1 GEMM as production backend until V2 architecture matures.

## Related Documentation

- `changelog/2025-10-22-tile-sweep-analysis.md` - V1 tile sweep results (314 GFLOPS achieved)
- `changelog/2025-10-24-v2-avx-simd-optimization.md` - AVX512 optimization (4-12× speedup)
- `.github/copilot-instructions.md` - V1 vs V2 architecture comparison
- `.github/instructions/llaminar-v2-architecture.instructions.md` - V2 design philosophy

## Environment

- Date: October 24, 2025
- V2 Build: Release (`-march=native -mtune=native -O3 -DNDEBUG`)
- CPU Features: AVX512F, AVX2, FMA, AVX512_VNNI
- Model: Qwen 2.5 0.5B IQ4_NL (638 MB)
- Test: Single-rank execution (V2 doesn't support MPI yet)
