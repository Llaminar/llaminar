# AVX2 GEMM Variants Implementation

**Date**: January 19, 2025  
**Status**: ✅ Complete  
**Phase**: 18 - Dual-ISA Support

## Summary

Added AVX2 variants alongside existing AVX512 implementations to handle CPU frequency downclocking scenarios. The auto-tuner now benchmarks both ISAs and automatically selects the optimal one for each matrix shape and CPU.

## Motivation

**Problem**: AVX512 can cause CPU frequency throttling on some processors:
- Example: 3.5GHz (baseline) → 2.8GHz (AVX512 active)
- Net effect: 2× throughput × 0.8× frequency = 1.6× speedup (not 2×)
- Some workloads: AVX2 at 3.5GHz > AVX512 at 2.8GHz

**Solution**: Provide both ISA variants, let auto-tuner benchmark actual performance on specific hardware.

## Implementation

### 1. AVX2 Kernel Implementation

**File**: `src/v2/kernels/cpu/QuantizedGemmVariantsImpl.cpp`

**Key differences from AVX512**:
- Vector registers: `__m256` (8 floats) instead of `__m512` (16 floats)
- Micro-kernel size: 4×4 tiles instead of 8×4 tiles
- Horizontal reduction: Manual hadd chain (AVX2 lacks `_mm256_reduce_add_ps`)

**AVX2-specific code**:
```cpp
// AVX2 macros
#define LOAD_PANEL_4x4_AVX2(offset) \
    a0_avx2 = _mm256_loadu_ps(A_panel + 0 * k_panel + p + (offset)); \
    // ... (4 rows × 4 cols)

#define ACCUMULATE_4x4_AVX2() \
    c00_avx2 = _mm256_fmadd_ps(a0_avx2, b0_avx2, c00_avx2); \
    // ... (16 accumulators)

// Manual horizontal sum (no AVX2 reduce intrinsic)
static inline float hsum_ps_avx2(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// AVX2 multiply template
template<int UNROLL_FACTOR, int PREFETCH_BLOCKS, int TILE_M, int TILE_N>
static bool multiply_impl_avx2(...) {
    // Uses __m256 registers, _mm256_* intrinsics
    // Manual reduction via hsum_ps_avx2()
}
```

### 2. Variant Instantiation

**12 AVX2 variants created** (mirroring AVX512 tile sizes):

| Tile Size | Unroll Factor | Variant Name |
|-----------|---------------|--------------|
| 8×4 (L1) | 4×, 8×, 16× | `4x_unroll_8x4_avx2`, etc. |
| 16×8 (Balanced) | 4×, 8×, 16× | `4x_unroll_16x8_avx2`, etc. |
| 32×16 (L2) | 4×, 8×, 16× | `4x_unroll_32x16_avx2`, etc. |
| 64×32 (L3) | 4×, 8×, 16× | `4x_unroll_64x32_avx2`, etc. |

### 3. Registration Update

**File**: `src/v2/kernels/cpu/GemmVariants.cpp`

**Before**:
- 14 variants total (12 AVX512 + 2 legacy)

**After**:
- 26 variants total (12 AVX512 + 12 AVX2 + 2 legacy)

**Registration code**:
```cpp
std::vector<std::unique_ptr<IQuantizedGemmVariant>> 
registerAllGemmVariants(const IBlockDecoder *decoder) {
    std::vector<std::unique_ptr<IQuantizedGemmVariant>> variants;
    variants.reserve(26);

#if defined(__AVX512F__)
    // 12 AVX512 variants
    variants.push_back(create_4x_unroll_8x4_variant(decoder));
    // ... 11 more
#endif

#if defined(__AVX2__)
    // 12 AVX2 variants
    variants.push_back(create_4x_unroll_8x4_avx2_variant(decoder));
    // ... 11 more
#endif
    
    // 2 legacy variants
    variants.push_back(create_cache_blocked_variant(decoder));
    variants.push_back(create_row_wise_variant(decoder));

    return variants;
}
```

### 4. Test Updates

**File**: `tests/v2/unit/Test__GemmAutoTunerIntegration.cpp`

**Changed**:
```cpp
// Before:
EXPECT_EQ(variants.size(), 14) << "Should have 14 variants (12 main + 2 legacy)";

// After:
EXPECT_EQ(variants.size(), 26) << "Should have 26 variants (12 AVX512 + 12 AVX2 + 2 legacy)";
```

## Code Changes

### Modified Files

1. **src/v2/kernels/cpu/QuantizedGemmVariantsImpl.cpp** (+213 lines)
   - Added AVX2 macros (LOAD_PANEL_4x4_AVX2, ACCUMULATE_4x4_AVX2)
   - Added `hsum_ps_avx2()` helper function
   - Added `multiply_impl_avx2<>` template
   - Added `DEFINE_GEMM_VARIANT_AVX2` macro
   - Instantiated 12 AVX2 variant classes
   - Created 12 AVX2 factory functions

2. **src/v2/kernels/cpu/GemmVariants.cpp** (+24 lines)
   - Added forward declarations for 12 AVX2 factory functions
   - Updated `registerAllGemmVariants()` (14 → 26 variants)
   - Added #if guards for conditional compilation

3. **tests/v2/unit/Test__GemmAutoTunerIntegration.cpp** (1 line)
   - Updated expected variant count (14 → 26)

### File Structure

```
src/v2/kernels/cpu/QuantizedGemmVariantsImpl.cpp:
  Lines 1-27:    File header (updated with AVX2 rationale)
  Lines 28-244:  AVX512 multiply_impl<> template
  Lines 248-428: AVX2 multiply_impl_avx2<> template
  Lines 432-488: AVX512 variant classes (12)
  Lines 492-550: AVX512 factory functions (12)
  Lines 555-611: AVX2 variant classes (12)
  Lines 615-671: AVX2 factory functions (12)
```

## Test Results

### Variant Registration Test

```bash
$ ./build_v2_release/tests/v2/v2_test_gemm_autotuner_integration --gtest_filter="*AllVariantsRegistered*"

===== VARIANT REGISTRATION TEST =====
Total variants registered: 26

Variant details:
  - 4x_unroll_8x4 (unroll=4, prefetch=3, tile=8x4)           # AVX512
  - 8x_unroll_8x4 (unroll=8, prefetch=5, tile=8x4)           # AVX512
  - 16x_unroll_8x4 (unroll=16, prefetch=5, tile=8x4)         # AVX512
  ... (9 more AVX512 variants)
  - 4x_unroll_8x4_avx2 (unroll=4, prefetch=3, tile=8x4)      # AVX2
  - 8x_unroll_8x4_avx2 (unroll=8, prefetch=5, tile=8x4)      # AVX2
  - 16x_unroll_8x4_avx2 (unroll=16, prefetch=5, tile=8x4)    # AVX2
  ... (9 more AVX2 variants)
  - cache_blocked (unroll=16, prefetch=5, tile=8x4)          # Legacy
  - row_wise (unroll=16, prefetch=5, tile=8x4)               # Legacy

[       OK ] Test__GemmAutoTunerIntegration.AllVariantsRegistered (0 ms)
```

### Full Test Suite

```bash
$ ctest -R "V2_Unit_GemmAutoTunerIntegration" --output-on-failure

Test #20: V2_Unit_GemmAutoTunerIntegration ... Passed (33.96 sec)

100% tests passed, 0 tests failed out of 2
```

**Status**: ✅ All tests passing

## Performance Characteristics

### Expected Behavior

The auto-tuner will:
1. Benchmark all 26 variants for each shape during first inference
2. Cache optimal variant per shape in `~/.cache/llaminar/gemm_tuning_cache.json`
3. Automatically select best ISA based on actual CPU performance

### AVX512 vs AVX2 Performance

**Theoretical**:
- AVX512: 2× wider registers (512-bit vs 256-bit)
- AVX512: 32 ZMM registers vs 16 YMM registers (less spilling)
- AVX512: Single-instruction horizontal reduction

**Actual** (depends on CPU frequency scaling):
- CPUs with aggressive AVX512 throttling: AVX2 may win
- CPUs with light AVX512 throttling: AVX512 likely wins
- Auto-tuner measures actual performance, no guessing needed

### Compilation Modes

```bash
# CPU with AVX512 support:
# - Compiles both AVX512 and AVX2 variants (26 total)
# - Auto-tuner benchmarks both ISAs

# CPU with AVX2 only:
# - Compiles only AVX2 variants (12 + 2 legacy = 14 total)
# - Auto-tuner uses only AVX2 variants

# CPU with neither:
# - Compiles only legacy variants (2 total)
# - Auto-tuner uses scalar fallback
```

## Validation

### Build Verification

```bash
# Clean build
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Verify compilation
[  0%] Building CXX object CMakeFiles/llaminar2_core.dir/kernels/cpu/QuantizedGemmVariantsImpl.cpp.o
[100%] Built target llaminar2_core
```

**Status**: ✅ No compilation errors

### Test Coverage

- ✅ Variant registration (26 variants)
- ✅ Unique variant names (no duplicates)
- ✅ Auto-tuner integration (basic, token512 shapes)
- ✅ Backward compatibility (legacy variants still work)

## Technical Notes

### AVX2 Reduction Implementation

**Challenge**: AVX2 lacks `_mm256_reduce_add_ps` intrinsic

**Solution**: Multi-step horizontal add sequence
```cpp
static inline float hsum_ps_avx2(__m256 v) {
    // Step 1: Extract 128-bit halves
    __m128 vlow = _mm256_castps256_ps128(v);           // Lanes 0-3
    __m128 vhigh = _mm256_extractf128_ps(v, 1);        // Lanes 4-7
    
    // Step 2: Add halves
    vlow = _mm_add_ps(vlow, vhigh);                    // 4 sums
    
    // Step 3: Horizontal adds
    __m128 shuf = _mm_movehdup_ps(vlow);               // Duplicate odd lanes
    __m128 sums = _mm_add_ps(vlow, shuf);              // 2 sums
    shuf = _mm_movehl_ps(shuf, sums);                  // Move high to low
    sums = _mm_add_ss(sums, shuf);                     // Final sum
    
    // Step 4: Extract scalar
    return _mm_cvtss_f32(sums);
}
```

**Overhead**: ~5 instructions vs 1 instruction (AVX512), but negligible compared to FMA-heavy loop.

### Register Pressure

**AVX512**: 32 ZMM registers → can unroll 8×4 tile without spilling
**AVX2**: 16 YMM registers → use 4×4 tile to avoid spilling

**Impact**: Smaller micro-kernel may reduce ILP (instruction-level parallelism), but lower spilling overhead may compensate.

## Future Work

1. **Performance Profiling** (planned):
   - Benchmark AVX512 vs AVX2 on various CPUs
   - Measure frequency throttling impact
   - Identify shapes where AVX2 wins

2. **ISA-Specific Optimizations** (future):
   - AVX512-specific: Use mask instructions for edge cases
   - AVX2-specific: Experiment with 8×4 micro-kernel (higher register pressure)

3. **Auto-Tuner Enhancements** (future):
   - Detect CPU frequency scaling behavior
   - Skip benchmarking AVX512 if downclocking too aggressive
   - Cache per-CPU decisions (not just per-shape)

## Conclusion

Successfully implemented dual-ISA support with minimal code duplication:
- ✅ **26 variants** (12 AVX512 + 12 AVX2 + 2 legacy)
- ✅ **Conditional compilation** (adapts to CPU capabilities)
- ✅ **Auto-tuner integration** (transparent ISA selection)
- ✅ **All tests passing** (backward compatible)

The auto-tuner will now automatically handle CPU frequency downclocking scenarios, selecting the optimal ISA for each workload without user intervention.

---

**Implementation Time**: ~45 minutes  
**Lines Changed**: +238 lines (net)  
**Tests Updated**: 1 test (variant count)  
**Build Status**: ✅ All targets compile  
**Test Status**: ✅ All tests pass
