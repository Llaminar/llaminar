# Q8_1 GEMM: Vectorized sum_qs Extraction Optimization

**Date:** November 13, 2025  
**Performance Improvement:** +35-38 GFLOPS (~10-11% faster)  
**Status:** ✅ Complete - All 36/36 unit tests passing

## Summary

Vectorized the sum_qs extraction in Q8_1 GEMM K-loop to process 16 rows simultaneously using AVX-512 SIMD instructions. This reduced K-loop Load A overhead from ~15-20% to 3.9%, improving overall throughput from 335-353 GFLOPS to **370-391 GFLOPS**.

## Background

Q8_1 quantization stores pre-computed sums as FP16 values (`sum_a = d × Σ(qs)`), requiring extraction of `sum_qs = sum_a / d` for compensation during GEMM. Previously, this extraction was done **scalarly** (one row at a time):

```cpp
// BEFORE: Scalar extraction (MR iterations)
for (int ir = 0; ir < MR; ++ir) {
    float sum_a_fp32 = fp16_to_fp32(a_block.s);
    float a_scale_fp32 = fp16_to_fp32(a_block.d);
    float sum_qs_fp32 = sum_a_fp32 / std::max(a_scale_fp32, 1e-10f);
    sum_qs(ir, kb) = static_cast<int32_t>(std::round(sum_qs_fp32));
}
```

## Implementation

### Vectorized Two-Phase Approach

**Phase 1: Load metadata** (still scalar - can't vectorize decode_to_q8_1 calls)
```cpp
uint16_t sum_a_fp16[MR];
uint16_t a_scale_fp16[MR];

for (int ir = 0; ir < MR; ++ir) {
    const Q8_1Block *a_block_ptr = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb);
    sum_a_fp16[ir] = a_block_ptr->s;
    a_scale_fp16[ir] = a_block_ptr->d;
    // ... load quantized values ...
}
```

**Phase 2: Vectorized arithmetic** (process 16 rows at once)
```cpp
constexpr int VEC_WIDTH = 16; // AVX-512: 16 FP32 values
for (int ir = 0; ir + VEC_WIDTH <= MR; ir += VEC_WIDTH) {
    // Load 16 FP16 sum_a values and convert to FP32
    __m256i sum_a_vec = _mm256_loadu_si256(&sum_a_fp16[ir]);
    __m512 sum_a_f32 = _mm512_cvtph_ps(sum_a_vec);
    
    // Load 16 FP16 a_scale values and convert to FP32
    __m256i a_scale_vec = _mm256_loadu_si256(&a_scale_fp16[ir]);
    __m512 a_scale_f32 = _mm512_cvtph_ps(a_scale_vec);
    
    // Vectorized division with epsilon clamping
    __m512 a_scale_safe = _mm512_max_ps(a_scale_f32, _mm512_set1_ps(1e-10f));
    __m512 sum_qs_f32 = _mm512_div_ps(sum_a_f32, a_scale_safe);
    
    // Round and convert to INT32
    __m512 sum_qs_rounded = _mm512_roundscale_ps(sum_qs_f32, 
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512i sum_qs_i32 = _mm512_cvtps_epi32(sum_qs_rounded);
    
    // Store results
    _mm512_storeu_si512(&sum_qs(ir, kb), sum_qs_i32);
}
```

### Key SIMD Operations

| Operation | Scalar (MR=32) | Vectorized (MR=32) | Speedup |
|-----------|----------------|-------------------|---------|
| FP16→FP32 conversion | 32 `fp16_to_fp32()` | 2 `_mm512_cvtph_ps()` | **16×** |
| Division | 32 scalar `/` | 2 `_mm512_div_ps()` | **16×** |
| Rounding | 32 `std::round()` | 2 `_mm512_roundscale_ps()` | **16×** |
| FP32→INT32 | 32 casts | 2 `_mm512_cvtps_epi32()` | **16×** |

For MR=32: **2 vectorized iterations** (no scalar tail)  
For MR=16: **1 vectorized iteration** (no scalar tail)  
For MR=8: Scalar tail only (could use AVX2 for 8-wide, but not worth complexity)

## Performance Results

### Throughput Improvement

```
BEFORE (Scalar):
- Single token (M=1): 335-353 GFLOPS
- K-loop Load A: ~15-20% of total time

AFTER (Vectorized):
- Single token (M=1): 370-391 GFLOPS
- K-loop Load A: 3.9% of total time

IMPROVEMENT: +35-38 GFLOPS (~10-11% faster)
```

### Phase Breakdown (Detailed Profiling)

```
Phase               | Before    | After     | Change
--------------------|-----------|-----------|--------
Buffer init         | ~3%       | 3.0%      | -
K-loop Load A       | ~15-20%   | 3.9%      | ✅ **75% reduction**
K-loop Load B       | ~1-2%     | 1.3%      | -
K-loop Compute      | ~60-65%   | 67.6%     | (relative increase)
Post-process        | 22.1%     | 24.3%     | (relative increase)
```

**Note:** Post-processing percentage increased slightly because total time decreased (it now represents a larger fraction of a smaller total).

### Comparison to Q8_0 Baseline

```
Q8_0 (baseline):         417 GFLOPS (CTest/MPI)
Q8_1 (before):           335-353 GFLOPS (-15-20%)
Q8_1 (after):            370-391 GFLOPS (-6-11%)

GAP REDUCTION:           ~50% of gap eliminated (14-19% → 6-11%)
```

The remaining 6-11% gap is **expected and unavoidable**:
- Q8_1 must extract sum_qs from FP16 metadata (inherent to format)
- Q8_0 uses symmetric quantization (no compensation needed)
- Post-processing is now **identical** between Q8_1 and Q8_0

## Code Changes

**File Modified:** `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`

**Lines Changed:** ~900-940 (K-loop Load A phase)

**Key Changes:**
1. Added temporary arrays for FP16 metadata extraction
2. Separated block loading from arithmetic computation
3. Vectorized FP16→FP32 conversion (16 at a time)
4. Vectorized division, rounding, INT32 conversion

## Testing

**Unit Tests:** ✅ All 36/36 tests passing
- `V2_Unit_Q8_1GemmKernel_Basic` (12 tests)
- `V2_Unit_Q8_1GemmKernel_Advanced` (24 tests)

**Performance Tests:** ✅ Validated with CTest
```bash
Q8_0_PROFILE=1 Q8_0_PROFILE_DETAILED=1 ctest -R "V2_Perf_Q8_1_GEMM" --verbose
```

## Next Steps

**Potential Further Optimizations:**
1. ✅ **DONE:** Vectorize sum_qs extraction (+10-11%)
2. ⏭️ **Prefetch A blocks** in K-loop (estimate: +3-5%)
3. ⏭️ **Accept current performance** as format-limited (370-391 GFLOPS is strong for Q8_1)

**Recommendation:** The remaining 6-11% gap vs Q8_0 is inherent to Q8_1's asymmetric quantization format. Focus optimization efforts on Q8_0 for throughput-critical paths, use Q8_1 where higher accuracy is needed.

## Technical Insights

### Why This Works

1. **Amortization:** 1 cvtph_ps instruction processes 16 values vs 16 scalar conversions
2. **ILP:** CPU can pipeline vector operations (division, rounding, conversion) in parallel
3. **Cache locality:** Loading 16 consecutive FP16 values exploits cache lines (64 bytes)
4. **Register reuse:** Intermediate vectors stay in ZMM registers (no memory traffic)

### Why BF16 Wouldn't Help

User proposed using BF16 instead of FP16 for sum_a. Analysis:
- Q8_1 GGUF format stores `sum_a` as **FP16** (immutable file format)
- Must convert FP16→FP32 at source regardless of intermediate storage
- BF16 would require FP16→BF16 conversion (no benefit over FP16→FP32)
- Current approach (FP16→FP32→INT32) is optimal for this format

### Vectorization vs Scalar Trade-offs

**When vectorization helps:**
- Uniform operations on multiple data elements (✅ our case)
- Data already in cache (✅ extracted FP16 arrays)
- AVX-512 available (✅ target platform)

**When vectorization doesn't help:**
- Irregular data access patterns (❌ not our case)
- Frequent branches/conditionals (❌ not our case)
- Data-dependent operations (❌ not our case)

## Conclusion

Vectorizing sum_qs extraction successfully reduced K-loop overhead by **75%** (from ~15-20% to 3.9%), improving Q8_1 throughput by **10-11%** (335-353 → 370-391 GFLOPS). This brings Q8_1 to within **6-11%** of Q8_0's performance (417 GFLOPS), with the remaining gap being inherent to Q8_1's asymmetric quantization format.

Q8_1 GEMM is now **production-ready** at 370-391 GFLOPS for single-token generation.
