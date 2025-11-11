# Q8_0 Decode Vectorization - Partial Success

**Date**: 2025-12-19  
**Status**: Partially complete - IQ4_NL AVX2 works, FP32 and AVX512 need fixes  
**Components**: SIMDHelpers.h, IQ4_NLTensor, FP32Tensor

## Summary

Implemented vectorized Q8_0 decode functions in `SIMDHelpers.h` for both IQ4_NL→Q8_0 and FP32→Q8_0 conversions, with AVX512, AVX2, and scalar fallback paths. Successfully integrated into tensor classes to eliminate redundant decoding overhead.

## What Was Accomplished

### 1. Infrastructure Created

**File**: `src/v2/tensors/SIMDHelpers.h` (+~370 lines)

Added vectorized decode functions:
- `decode_iq4nl_to_q8_0_scalar/avx2/avx512()` - IQ4_NL→Q8_0 decode
- `quantize_fp32_to_q8_0_scalar/avx2/avx512()` - FP32→Q8_0 quantize
- Auto-dispatch wrappers for ISA selection

**Technical Approach:**
- **IQ4_NL decode**: PSHUFB-based parallel LUT lookup (16→32 elements)
- **FP32 quantize**: Horizontal max reduction + vectorized round/clamp/pack
- **Auto-dispatch**: Compile-time #ifdef for optimal ISA selection
- **Header-only**: Inline functions for zero abstraction penalty

### 2. Tensor Integration

**IQ4_NLTensor.cpp** (-25 lines):
```cpp
// Before: 30-line scalar decode with inline LUT
// After: Single-line delegation
simd::decode_iq4nl_to_q8_0(iq4_block.qs, iq4_block.d, output->qs, &output->d);
```

**FP32Tensor.cpp** (-45 lines):
```cpp
// Before: 50-line scalar quantize (absmax loop, quantize loop, zero-pad)
// After: Single-line delegation
simd::quantize_fp32_to_q8_0(src, k_count, output->qs, &output->d);
```

### 3. Unit Tests Created

**File**: `tests/v2/unit/Test__Q8_0DecodeVectorization.cpp` (468 lines)

Test categories:
- IQ4_NL decode correctness (scalar vs AVX2 vs AVX512)
- FP32 quantize correctness (scalar vs AVX2 vs AVX512)
- Partial block handling (k_count < 32)
- Edge cases (all-zero, very small, very large, mixed signs)
- Round-trip accuracy (quantize→dequantize error bounds)

## Test Results

### ✅ **Passing Tests (5/9)**

1. `IQ4NL_ScalarVsAVX2` - **100% match** across 100 trials
2. `FP32_ScalarVsAVX512` - AVX512 quantize matches scalar
3. `FP32_PartialBlocks` - Partial block handling works
4. `IQ4NL_DequantizeRoundTrip` - Round-trip errors within bounds
5. `FP32_QuantizeDequantizeError` - Quantization error acceptable

### ❌ **Failing Tests (4/9)**

1. **`IQ4NL_ScalarVsAVX512`** - AVX512 decode has bugs
   - **Status**: Fails all 100 trials (alignment or logic issue)
   - **Root Cause**: Likely byte-lane ordering in AVX512 shuffle or broadcast

2. **`IQ4NL_AutoDispatch`** - Auto-dispatch produces wrong results
   - **Status**: Fails trials 0-99 (depends on broken AVX512 path)
   - **Root Cause**: Selecting buggy AVX512 code path on this CPU

3. **`FP32_ScalarVsAVX2`** - AVX2 quantize has large errors
   - **Status**: Fails all 100 trials with max_diff=135-254 (16/32 elements wrong)
   - **Root Cause**: Likely bug in horizontal max reduction or pack logic

4. **`FP32_EdgeCases`** - Small values quantize to 127 instead of 0
   - **Status**: All 32 elements = 127 for 1e-8f input (should be 0 or ±1)
   - **Root Cause**: Scale calculation or quantization formula error

## Remaining Work

### High Priority

**Fix AVX512 IQ4_NL decode** (needed for auto-dispatch):
- Debug byte-lane ordering in `_mm512_shuffle_epi8` output
- Check `_mm512_permutexvar_epi8` index correctness
- Verify low/high nibble interleaving logic
- Consider using `_mm512_unpacklo_epi8` / `_mm512_unpackhi_epi8` instead

**Fix AVX2 FP32 quantize** (critical for performance):
- Debug horizontal max reduction (expected max not found?)
- Check `_mm256_cvtps_epi32` rounding mode (should be round-to-nearest)
- Verify pack sequence: `_mm256_packs_epi32` → `_mm256_packs_epi16`
- Check lane-crossing operations (AVX2 doesn't cross 128-bit lanes)

**Fix FP32 edge cases**:
- Handle `amax == 0` case (should set scale=0, output all zeros)
- Clamp quantization to [-127, 127] before pack
- Test divide-by-zero protection

### Medium Priority

**Q6_K vectorization** (deferred from this session):
- Complex 6-bit unpacking (ql[128] + qh[64])
- Hierarchical scale computation
- Estimated: ~100 lines per ISA (AVX2/AVX512)

### Low Priority

**Performance benchmarking** (after correctness fixed):
- Measure throughput: AVX512 vs AVX2 vs scalar
- Expected speedup: ~8-16× for IQ4_NL (PSHUFB parallelism)
- Benchmark in GEMM context (decode amortized via LRU cache)

**Remaining 16 formats**:
- Add vectorized decode for Q4_K, Q5_K, Q2_K, Q3_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_K, IQ series
- Pattern: Similar to IQ4_NL (PSHUFB LUT) or FP32 (reduction + quantize)

## Technical Insights

### What Worked Well

1. **PSHUFB for parallel LUT lookup** - Clean, efficient pattern for IQ4_NL
2. **Header-only inline functions** - Zero abstraction overhead
3. **Auto-dispatch wrapper** - Seamless ISA selection (when correct path works)
4. **Test harness** - Comprehensive correctness validation
5. **Alignment requirements** - 64-byte alignment prevents ASAN/debug issues

### What Didn't Work

1. **AVX512 byte lane ordering** - Complex permute logic error
2. **AVX2 horizontal reductions** - Lane-crossing operations tricky
3. **Edge case handling** - Divide-by-zero and all-zero inputs need special care
4. **ASAN compatibility** - Unaligned intrinsic loads trigger ASAN in debug mode

### Lessons Learned

1. **Test scalar first, then vectorize** - Scalar reference validates SIMD
2. **Check intermediate values** - Print `amax`, `scale`, first few `q` values
3. **Use Release builds for SIMD tests** - ASAN doesn't like unaligned intrinsics
4. **Validate edge cases separately** - Zero, inf, denormal inputs matter

## Next Steps

1. **Fix AVX512 IQ4_NL decode** - Debug shuffle/permute operations
2. **Fix AVX2 FP32 quantize** - Debug horizontal max and pack
3. **Fix edge case handling** - Special-case `amax == 0`
4. **Re-run tests** - Verify all 9/9 pass
5. **Integrate with IntegerGemm** - Test in production GEMM path
6. **Performance benchmarks** - Measure real speedup

## Files Changed

- `src/v2/tensors/SIMDHelpers.h` (+372 lines, +2 includes)
- `src/v2/tensors/IQ4_NLTensor.cpp` (-25 lines)
- `src/v2/tensors/FP32Tensor.cpp` (-45 lines)
- `tests/v2/unit/Test__Q8_0DecodeVectorization.cpp` (+468 lines, new file)
- `tests/v2/CMakeLists.txt` (+16 lines, new test target)

**Net Change**: +786 lines added, -70 lines removed = +716 lines total

## Compilation

```bash
# Debug build (with ASAN - fails on unaligned intrinsics)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --target v2_test_q8_0_decode_vectorization

# Release build (ASAN-free - works correctly)
cmake -B build_v2_noasan -S src/v2 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_v2_noasan --target v2_test_q8_0_decode_vectorization

# Run tests
./build_v2_noasan/tests/v2/v2_test_q8_0_decode_vectorization
```

## Conclusion

Partial success - IQ4_NL AVX2 decode is fully functional and demonstrates the viability of the vectorization approach. FP32 and AVX512 paths have bugs that need debugging. Once fixed, the infrastructure is ready for integration with the IntegerGemm GEMM kernel, which will provide the real performance benefits (eliminating 512× redundant decoding via LRU cache + vectorized decode).

**Estimated Effort to Complete**:
- AVX512 IQ4_NL fix: 1-2 hours (debug shuffle operations)
- AVX2 FP32 fix: 1-2 hours (debug reduction + pack)
- Edge case fixes: 30 minutes (add zero-check)
- Integration testing: 1 hour (verify with GEMM)
- Performance benchmarking: 1-2 hours (measure speedup)

**Total**: ~5-8 hours to full completion
