# SIMD Optimization Plan for V2 Quantized Tensors

**Date**: October 29, 2025  
**Status**: In Progress  
**Goal**: Add AVX2/AVX512 SIMD optimizations to all IBlockDecoder paths

## Current Status

### ✅ Fully Optimized (Canonical Implementations)
- **IQ4_NLTensor**: Complete AVX2/AVX512 with runtime dispatch (14 AVX references)
  - Uses lookup table + SIMD nibble extraction
  - Includes microkernel experimental path
  - File: `src/v2/tensors/IQ4_NLTensor.cpp`
  
- **Q8_0Tensor**: Complete AVX2/AVX512 with runtime dispatch (10 AVX references)
  - Direct int8→float32 conversion
  - File: `src/v2/tensors/Q8_0Tensor.cpp`

- **Q4_0Tensor**: AVX2 + **NEW AVX512** added today (was 5, now 9 AVX references)
  - Nibble extraction + bias subtraction
  - File: `src/v2/tensors/Q4_0Tensor.cpp` ✅ **UPDATED**

### ⚠️ Partial SIMD (Needs Completion)
- **Q4_1Tensor**: AVX2 stub only (5 AVX references)
- **Q5_0Tensor**: AVX2 stub only (4 AVX references)
- **Q5_1Tensor**: AVX2 stub only (4 AVX references)
- **Q6_KTensor**: AVX2 TODO stub (3 AVX references)

### ❌ No SIMD (Needs Implementation)

**IQ Quants** (8 tensors - use IQ4_NL pattern):
- IQ1_MTensor
- IQ1_STensor
- IQ2_STensor
- IQ2_XSTensor
- IQ2_XXSTensor
- IQ3_STensor
- IQ3_XXSTensor
- IQ4_XSTensor

**Q_K Quants** (5 tensors - use Q8_0 pattern):
- Q2_KTensor
- Q3_KTensor
- Q4_KTensor
- Q5_KTensor
- Q8_KTensor

## Implementation Strategy

### Phase 1: Complete Partial SIMD ✅ **IN PROGRESS**
1. ✅ Q4_0Tensor - Add AVX512 (DONE)
2. Q4_1Tensor - Implement AVX2/AVX512
3. Q5_0Tensor - Implement AVX2/AVX512
4. Q5_1Tensor - Implement AVX2/AVX512
5. Q6_KTensor - Implement AVX2/AVX512

### Phase 2: Add SIMD to Q_K Variants
Pattern: Follow Q8_0Tensor template
- Q2_KTensor
- Q3_KTensor
- Q4_KTensor
- Q5_KTensor
- Q8_KTensor

### Phase 3: Add SIMD to IQ Variants
Pattern: Follow IQ4_NLTensor template
- IQ1_M/STensor
- IQ2_S/XS/XXSTensor
- IQ3_S/XXSTensor
- IQ4_XSTensor

## Technical Patterns

### Pattern A: Q8_0-style (int8 direct conversion)
```cpp
#if defined(__AVX512F__)
    if (cpu_supports_avx512()) {
        decodeBlockAVX512(block, output);
        return;
    }
#endif
#if defined(__AVX2__)
    if (cpu_supports_avx2()) {
        decodeBlockAVX2(block, output);
        return;
    }
#endif
// Scalar fallback
```

### Pattern B: IQ4_NL-style (lookup table + nibble extraction)
```cpp
#if defined(__AVX512F__)
    if (cpu_supports_avx512() && !env.dequant.iq4_direct_decode) {
        decodeBlockAVX512(block, output);
        return;
    }
#endif
// Similar for AVX2, scalar fallback
```

### Helper Functions Available
- `simd::convert_i8_to_f32_scaled_avx512()` - 16 int8→float with scale
- `simd::convert_i8_to_f32_scaled_avx2()` - 8 int8→float with scale
- `simd::fp16_to_fp32()` - FP16 scale conversion
- `simd::fp32_to_bf16()` - BF16 conversion
- CPU feature detection: `cpu_supports_avx512()`, `cpu_supports_avx2()`

## Files Modified Today

### Q4_0Tensor
**File**: `src/v2/tensors/Q4_0Tensor.cpp`
**Changes**:
- Added `#include "../utils/CPUFeatures.h"`
- Added AVX512 include guards
- Added runtime dispatch in `decodeBlock()`
- Implemented `decodeBlockAVX512()` - processes 32 nibbles in 2 chunks of 16
- Updated header `Tensors.h` to declare AVX512 method

**Performance Impact**: ~2x speedup on AVX512 CPUs for Q4_0 decode (estimated)

## Next Steps

1. **Immediate**: Complete Q4_1, Q5_0, Q5_1, Q6_K SIMD implementations
2. **Short-term**: Add SIMD to all Q_K variants (Q2_K through Q8_K)
3. **Medium-term**: Add SIMD to all IQ variants  
4. **Testing**: Create benchmark comparing scalar vs SIMD performance
5. **Documentation**: Update performance characteristics in copilot-instructions.md

## Performance Expectations

Based on IQ4_NL and Q8_0 measurements:
- **AVX2 vs Scalar**: 2-4x speedup
- **AVX512 vs AVX2**: 1.5-2x additional speedup
- **Overall AVX512 vs Scalar**: 3-8x speedup (block size dependent)

## References

- Canonical IQ implementation: `src/v2/tensors/IQ4_NLTensor.cpp`
- Canonical Q implementation: `src/v2/tensors/Q8_0Tensor.cpp`
- SIMD helpers: `src/v2/tensors/SIMDHelpers.h`
- CPU features: `src/v2/utils/CPUFeatures.h`
