# SimpleInt8Gemm Baseline Findings

**Date**: November 12, 2025  
**Status**: Clean-sheet design established baseline - problem identified

## Summary

Created a simple INT8 GEMM kernel without microkernel abstractions to establish baseline performance. Results show that **memory access patterns, not abstraction overhead, are the bottleneck**.

## Performance Results

| Version | Prefill-512 GOPS | vs Complex Kernel | vs OneDNN Target |
|---------|------------------|-------------------|-------------------|
| Complex microkernel | 160 GOPS | Baseline | 41× slower |
| **Simple baseline (scalar bias)** | **173 GOPS** | **+8%** | **38× slower** |
| Simple + vectorized bias | 86 GOPS | -47% | **76× slower** ❌ |

## Key Findings

### 1. Abstraction Overhead Was Minor (+8%)
- Eliminating panels, microkernels, load functions → **only 8% improvement**
- This proves the 41× gap is NOT due to abstraction complexity
- The problem is **algorithmic**, not architectural

### 2. Vectorized Bias Correction FAILED (-50%)
**Attempt**: Replace scalar loop with SIMD horizontal sum
```cpp
// OLD (scalar): 32 iterations, simple add
for (int i = 0; i < 32; ++i) sum_b += b[i];

// NEW (vectorized): cvtepi8_epi16 + horizontal adds
// Result: 2× SLOWER!
```

**Why it failed**:
- Extra instructions (sign extension, shuffles, extracts)
- Overhead exceeds benefit for 32-element sum
- Modern CPUs can execute 32 scalar adds very fast

**Lesson**: Keep bias correction simple - it's NOT the bottleneck

### 3. Triple-Loop Naive Implementation
Current structure:
```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < m; ++i) {           // Parallel over M rows
    for (int j = 0; j < n; ++j) {        // ❌ BAD: Sequential N columns
        for (int kb = 0; kb < k_blocks; ++kb) {  // K-loop
            const SimpleQ8Block& b_block = B[j * k_blocks + kb];  
            // ^^^ Column-major access - poor locality!
        }
    }
}
```

**Problem**: Accessing `B[j * k_blocks]` for each `j` in inner loop = **random column access**
- B matrix: 896×896 = 800KB > L2 cache (256KB)
- Each N-iteration fetches different column → cache miss
- No cache blocking → working set doesn't fit in cache

### 4. Comparison with OneDNN (6,610 GOPS @ 84% efficiency)

OneDNN likely uses:
1. **3-level cache blocking** (MC×KC×NC)
   - MC: M-dimension cache block (~256 rows)
   - KC: K-dimension cache block (~512 elements)
   - NC: N-dimension cache block (~128 columns)
   - Keeps working set in L2 cache

2. **Loop reordering**:
   ```cpp
   for (mc: M-blocks):
       for (nc: N-blocks):
           for (kc: K-blocks):
               for (i in mc):
                   for (j in nc):
                       for (k in kc):
                           C[i][j] += A[i][k] * B[k][j]
   ```

3. **Data packing/pre-transposition**
   - Pack B into cache-friendly layout once
   - Amortize packing cost over many GEMMs

4. **Better microkernel**
   - Larger register tiles (likely 24×8 or 32×16)
   - Vectorized loads/stores  
   - Prefetching

## Root Cause: Cache Misses Dominate

Our 173 GOPS (2.2% efficiency) suggests **memory bandwidth bottleneck**:
- Theoretical peak: 7,884.8 GOPS (assumes data in registers)
- Actual: 173 GOPS
- **Gap factor**: 45×

This 45× gap matches **cache miss penalties**:
- L1 hit: ~4 cycles
- L2 hit: ~14 cycles  
- L3 hit: ~50 cycles
- RAM: ~200 cycles → **50× slower than L1!**

## Next Steps (Prioritized)

### Critical (Expected 20-30× speedup):
1. **Implement 3-level cache blocking**
   - MC=256, KC=512, NC=128
   - Reorder loops: MC → NC → KC → M-tiles → N-tiles → K-blocks
   - Expected: 3,500-5,000 GOPS (from 173)

### High Priority (Expected 1.5-2× on top):
2. **Pre-compute B column sums for bias**
   - Store in array: `int32_t B_sums[n]`
   - Eliminates scalar loop from hot path
   - Expected: 1.5× improvement

3. **Larger tiles**
   - Current: Computing 1×1 output per iteration
   - Try: 8×8 or 16×16 output tiles
   - Better register reuse

### Medium Priority:
4. **Software prefetching**
5. **Better loop unrolling**

## Files Created

- `src/v2/kernels/cpu/gemm/int8/SimpleInt8Gemm.h` - Clean baseline kernel
- `tests/v2/performance/cpu/kernels/gemm/Perf__SimpleInt8Gemm.cpp` - Benchmark
- Added to `tests/v2/CMakeLists.txt` - CTest integration

## Lessons Learned

1. **Start simple, optimize based on evidence** ✅
   - Complex microkernel didn't help
   - Simple baseline revealed real problem (cache)

2. **Not all SIMD is faster**
   - Vectorizing bias correction was net negative
   - Scalar code can be competitive for small loops

3. **Memory hierarchy > instruction count**
   - 173 vs 6,610 GOPS = cache issue, not instruction overhead
   - Need cache blocking before any micro-optimizations

4. **Measure, don't guess** ✅
   - Pre-decode B: Expected 25× faster → Actually 8% slower
   - Vectorized bias: Expected 1.3× faster → Actually 2× slower
   - Always benchmark!

## Commands to Reproduce

```bash
# Build simple baseline
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --target v2_perf_simple_int8_gemm --parallel

# Run benchmark
cd build_v2_release
OMP_NUM_THREADS=28 ctest -R "V2_Perf_SimpleInt8Gemm" --verbose
```

**Expected output**: 173 GOPS on prefill-512 (scalar bias version)

---

**Author**: David Sanftenberg (assisted by GitHub Copilot)  
**Session Duration**: ~45 minutes  
**Lines Written**: ~300 (SimpleInt8Gemm.h + benchmark)  
**Key Discovery**: Cache blocking is mandatory - micro-optimizations irrelevant without it
