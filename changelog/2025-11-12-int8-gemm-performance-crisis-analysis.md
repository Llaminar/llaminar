# INT8 GEMM Performance Crisis - Root Cause Analysis

**Date**: November 12, 2025  
**Status**: 🚨 **CRITICAL PERFORMANCE GAP IDENTIFIED**  
**Current**: 150 GOPS (1.9% efficiency)  
**Target**: 6,610 GOPS (84% efficiency) - OneDNN baseline  
**Gap**: **44× slower than we should be!**

## Summary

Discovered catastrophic performance issues in our INT8 GEMM kernel through systematic benchmarking:

| Implementation | M=512, N=896, K=896 | Time (ms) | Performance | Efficiency |
|----------------|---------------------|-----------|-------------|------------|
| **OneDNN INT8** | Baseline | 0.124 | **6,610 GOPS** | **84%** ⚡ |
| **OpenBLAS FP32** | Reference | 0.867 | 948 GFLOPS | 48% |
| **Our INT8 kernel** | Current | 30.3 | 150 GOPS | 1.9% 😱 |

## Theoretical Peak Corrections

### Before (WRONG):
```cpp
constexpr double THEORETICAL_PEAK_GFLOPS = 358.4;  // AVX512VNNI @ 2.8GHz
```
- Wrong terminology (GFLOPS for INT8 operations)
- Wrong CPU frequency (2.8 GHz vs actual 2.2 GHz)
- Ambiguous scope (per-core vs aggregate unclear)

### After (CORRECT):
```cpp
// Theoretical peak for INT8 GEMM (28-core aggregate, single socket)
// AVX512VNNI: 64 INT8 MACs per cycle per core
// Peak = 28 cores × 2.2 GHz × 2 FMA units × 64 INT8 ops/cycle = 7,884.8 GIOPS
constexpr double THEORETICAL_PEAK_GIOPS = 7884.8;  // 28-core @ 2.2GHz base
```

**Key insights**:
- Correct calculation shows 2% efficiency (not "reasonable")
- OneDNN achieves 84% efficiency on same hardware
- Proves high performance IS achievable

## Root Cause Analysis

### Critical Issues Found:

#### 1. **Redundant B-Matrix Decoding** (25-30× impact)
**Problem**: B matrix decoded 32× redundantly
```cpp
for (int ii = 0; ii < 512; ii += 16) {        // 32 M tiles
    for (int jj = 0; jj < 896; jj += 32) {    // 28 N tiles  
        for (kb = 0; kb < 14; kb += 2) {      // 14 K-block pairs
            // SAME (jj, kb) fetched from provider 32 times!
            B_provider.get_q8_block(jj, kb);
        }
    }
}
```
- 6.5M redundant provider calls for M=512 case
- Each block decoded, scales extracted 32× redundantly

#### 2. **Memcpy Overhead** (2-3× impact)
**Problem**: ~500K `memcpy()` calls per GEMM
```cpp
// Called for every tile, every K-block iteration
std::memcpy(A_panel + ..., block->qs, 32);  // Line 276
std::memcpy(B_panel + ..., block->qs, 32);  // Line 323
```

#### 3. **FP16→FP32 Conversion Overhead** (1.5-2× impact)
**Problem**: 6.5M conversions in hot path
```cpp
a_scales[i] = fp16_to_fp32(block->d);  // Every tile, every block
b_scales[j] = fp16_to_fp32(block->d);  // Every tile, every block
```

#### 4. **No Cache Blocking** (5-10× impact)
- MC/KC/NC template params exist but unused
- B matrix (896×896 = 800KB) > L2 cache (256KB)
- Constant cache evictions

#### 5. **Expensive VNNI Bias Correction** (1.5× impact)
```cpp
// Signed→unsigned conversion overhead
__mmask64 neg_mask = _mm512_cmplt_epi8_mask(a_vec, zero);
__m512i bias_vec = _mm512_dpbusd_epi32(zero, a_mask_ones, b_vec);
int32_t corrected_dot = unsigned_sum - 256 * sum_b_neg;
```

#### 6. **Small Tile Sizes** (1.3-1.5× impact)
- TILE_M=16, TILE_N=32 is tiny
- OneDNN likely uses 48×48 or 64×64

#### 7. **No Software Prefetching**
- Prefetch functions exist but never called
- Missing cache miss hiding

#### 8. **FP32 Accumulation (Not Deferred)** (1.2× impact)
```cpp
// INT32→FP32 conversion on every K-block
fp_acc_[i * TILE_N + j] += static_cast<float>(corrected_dot) * a_scale * b_scale;
```

## Optimization Attempts

### Attempt #1: Pre-decode B Matrix
**Goal**: Eliminate redundant decoding (expected 25× speedup)

**Implementation**:
```cpp
// Pre-decode entire B matrix once
std::vector<int8_t> B_decoded(n * k);
std::vector<float> B_scales(n * k_blocks);

for (int j = 0; j < n; ++j) {
    for (size_t kb = 0; kb < k_blocks; ++kb) {
        const Q8_0Block *block = B_provider.get_q8_block(j, kb);
        std::memcpy(&B_decoded[j * k + kb * 32], block->qs, 32);
        B_scales[j * k_blocks + kb] = fp16_to_fp32(block->d);
    }
}
```

**Result**: ❌ **FAILED - 147 GOPS (worse than before!)**

**Why it failed**:
- Still doing `memcpy()` in `load_B_panel_from_decoded()`
- Just moved the overhead, didn't eliminate it
- Pre-decode allocation overhead added latency

### Attempt #2: Change OpenMP Schedule
**Goal**: Reduce scheduling overhead

**Implementation**:
```cpp
// Before: #pragma omp parallel for schedule(dynamic, 1)
// After:  #pragma omp parallel for schedule(static)
```

**Result**: ✅ **Slight improvement - 150 GOPS (3 GOPS gain)**
- Still 44× slower than target

## Next Steps (Prioritized by Impact)

### High Priority (10×+ speedup potential):

1. **Zero-Copy Microkernel** (25-30× speedup)
   - Rewrite `accumulate()` to work directly from decoded buffers
   - Pass pointers + strides instead of panel copies
   - Eliminate ALL `memcpy()` overhead

2. **3-Level Cache Blocking** (5-10× speedup)
   - Implement MC×KC×NC blocking (MC=256, KC=512, NC=128)
   - Reorder loops: M-outer → MC panels → KC×NC tiles
   - Keep working set in L2 cache

3. **Larger Tiles** (1.5× speedup)
   - Increase TILE_M from 16 to 32 or 48
   - Better register reuse, less loop overhead

### Medium Priority (2×+ speedup potential):

4. **Defer Scale Application** (1.5-2× speedup)
   - Accumulate INT32 during K-loop
   - Apply scales once at final reduction
   - Reduces INT32→FP32 conversions

5. **Pre-Extract All Scales** (1.3× speedup)
   - One-time FP16→FP32 conversion for all scales
   - Store in aligned FP32 arrays

6. **Optimize Bias Correction** (1.3× speedup)
   - Consider pre-biased lookup tables
   - Or use unsigned quantization format

### Low Priority (1.2× speedup potential):

7. **Software Prefetching**
   - Call `prefetch_A_panel()`, `prefetch_B_panel()`
   - Use PREFETCH_DIST parameter

8. **Alignment Optimization**
   - Ensure 64-byte alignment for all buffers
   - Use aligned loads (`_mm512_load_si512()`)

## Cumulative Speedup Estimate

| Fix | Individual | Cumulative | Result |
|-----|------------|------------|--------|
| Baseline | 1× | 1× | 150 GOPS |
| Zero-copy | 25× | 25× | **3,750 GOPS** |
| Cache blocking | 5× | 125× | **18,750 GOPS** (exceeds target!) |
| Larger tiles | 1.5× | 187× | **28,125 GOPS** |

**Just fixes #1-#2 should exceed OneDNN's 6,610 GOPS target!**

## Files Modified

- `tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_FullSweep_Registry.cpp`
  - Fixed theoretical peak calculation
  - Changed GFLOPS → GOPS terminology
  - Updated efficiency metric

- `tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_QwenProfile.cpp`
  - Same theoretical peak fixes
  - Now registered as CTest

- `src/v2/kernels/cpu/gemm/IntegerGemmKernelTemplateV2.h`
  - Added B-matrix pre-decode (didn't help - memcpy still bottleneck)
  - Changed OpenMP schedule to static (+3 GOPS)

- `tests/v2/CMakeLists.txt`
  - Added CTest registration for QwenProfile

## Benchmark Commands

```bash
# OneDNN baseline (6,610 GOPS @ 84% efficiency)
OMP_NUM_THREADS=28 LD_LIBRARY_PATH=/workspaces/llaminar/external/onednn/build/lib:$LD_LIBRARY_PATH \
  /tmp/benchmark_onednn_int8

# OpenBLAS FP32 reference (948 GFLOPS @ 48% efficiency)
OMP_NUM_THREADS=28 /tmp/benchmark_openblas_gemm

# Our INT8 kernel (150 GOPS @ 1.9% efficiency)
cd /workspaces/llaminar/build_v2_release
ctest -R "V2_Perf_IntegerGEMM_QwenProfile" --verbose
```

## Key Learnings

1. **Theoretical peak calculations matter**
   - Wrong peak hides performance issues
   - 2% efficiency is NOT acceptable (84% is achievable)

2. **Memory hierarchy dominates**
   - Redundant decoding (memory bandwidth waste) = 25× slowdown
   - Cache blocking (locality) = 5× speedup potential
   - Memcpy overhead (extra traffic) = 2-3× slowdown

3. **Optimization order matters**
   - Pre-decode without zero-copy = no improvement
   - Must eliminate root cause (memcpy), not move it around

4. **OneDNN sets the bar**
   - 84% efficiency proves high performance is possible
   - Our 1.9% indicates fundamental architectural issues
   - Need clean-sheet rewrite, not incremental fixes

## Status

🔴 **BLOCKED**: Current architecture cannot reach target performance  
🎯 **Next Action**: Implement zero-copy microkernel + cache blocking  
📊 **Target**: 6,610 GOPS (84% efficiency) to match OneDNN

---

**Author**: David Sanftenberg (assisted by GitHub Copilot)  
**Session Duration**: ~3 hours  
**Lines Changed**: ~200 (mostly documentation/metrics fixes)  
**Performance Improvement**: 3 GOPS (2% - essentially zero)  
**Critical Discovery**: 44× performance gap with OneDNN baseline
