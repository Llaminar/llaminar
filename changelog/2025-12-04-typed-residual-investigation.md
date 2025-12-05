# Typed Residual Project Investigation - 2025-12-04

## Summary

Resumed work on the Typed Residual project to investigate and optimize `CPURMSNormTypedKernel` performance. Key discoveries:

1. **Fixed catastrophic single-token performance** via `want_parallel()` threshold
2. **Q8_1 integer-space RMSNorm** is 5x slower than existing SIMD (exploration complete, not viable)
3. **Typed kernel integration blocked** by architectural mismatch (mutable tensor requirement)

## 1. want_parallel() Threshold Optimization

### Problem
Single-token inference showed 7000%+ overhead for typed kernels vs baseline due to OpenMP parallelization overhead for small workloads.

### Solution
Added intelligent parallelization thresholds in `CPURMSNormTypedKernel`:

```cpp
static constexpr int MIN_ROWS_FOR_PARALLEL = 8;
static constexpr int MIN_ELEMENTS_FOR_PARALLEL = 65536;

static bool want_parallel(int rows, size_t cols) {
    if (rows < MIN_ROWS_FOR_PARALLEL) return false;
    if (static_cast<size_t>(rows) * cols < MIN_ELEMENTS_FOR_PARALLEL) return false;
    return true;
}
```

### Results
| Precision | Rows=1 Before | Rows=1 After | Improvement |
|-----------|---------------|--------------|-------------|
| BF16      | +7252%        | -64%         | **Massive** |
| FP16      | +6890%        | -68%         | **Massive** |
| Q8_1      | +4520%        | +120%        | Significant |

Single-token BF16/FP16 now **faster** than FP32 baseline.

### Files Changed
- `src/v2/kernels/cpu/ops/CPURMSNormTypedKernel.cpp`
- `src/v2/kernels/cpu/ops/CPURMSNormTypedKernel.h`

## 2. Q8_1 Integer-Space RMSNorm Exploration

### Hypothesis
Computing sum-of-squares in integer space without FP32 dequantization could reduce Q8_1 RMSNorm overhead.

### Implementation
Created `rmsnorm_q8_1_integer_row()` in `RMSNormPrimitives.cpp`:

```cpp
// Algorithm:
// Phase 1: sum_sq = Σ(d² × Σqs²) using integer SIMD
// Phase 2: inv_rms = 1/sqrt(mean_sq + eps)
// Phase 3: Dequant → scale → requant per block
```

AVX512 SIMD for integer sum-of-squares:
- `_mm256_maddubs_epi16` for int8×int8→int16 products
- `_mm256_madd_epi16` for horizontal sum to int32

### Results
| Approach | Single Row (d_model=896) |
|----------|--------------------------|
| Integer-Space Q8_1 | 4.42 μs |
| Existing 3-Pass SIMD | 0.86 μs |
| **Speedup** | **0.19x (5x SLOWER)** |

### Analysis
The existing SIMD implementation is highly optimized:
- `dequantize_q8_1_to_fp32()`: ~0.21 μs (SIMD)
- `rmsnorm_fused_row_avx512()`: ~0.26 μs (SIMD)
- `quantize_fp32_to_q8_1_blocks()`: ~0.38 μs (SIMD)

My integer approach still requires dequant/requant in Phase 3 and has a scalar quantization bottleneck.

### Conclusion
Integer-space approach is **not viable**. The FP32 intermediate fits in L1 cache and 3-pass SIMD is near-optimal.

### Files Added
- `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp` (170 lines added)
- `src/v2/kernels/cpu/primitives/RMSNormPrimitives.h` (declarations)
- `tests/v2/unit/Test__Q8_1IntegerRMSNorm.cpp` (712 lines)

## 3. Typed Kernel Integration Blockers

### Issue
`CPURMSNormTypedKernel` exists but is NOT connected to the pipeline.

### Root Cause
Architectural mismatch between:
1. **RMSNormOp interface**: Uses `TensorBase::data()` / `mutable_data()` returning `float*`
2. **Typed kernels**: Take `StorageType*` (e.g., `Q8_1Block*`, `uint16_t*`)
3. **Typed tensors**: Have `mutable_data()` that **throws exception**

```cpp
// BF16Tensor.cpp
float *BF16Tensor::mutable_data() {
    throw std::runtime_error("BF16Tensor::mutable_data: BF16 tensors are immutable");
}
```

Same for FP16Tensor and Q8_1Tensor.

### Current Architecture
```
RMSNormOp::operator()
  → activation->createRMSNorm()        // Returns ITensorRMSNorm*
  → kernel->apply(input->data(), ..., output->mutable_data())  // ❌ Throws for non-FP32
```

### Required Changes
To enable BF16/FP16/Q8_1 activations:

**Option A**: Mutable Activation Tensors
- Add `mutable_blocks()` to Q8_1Tensor
- Add `mutable_bf16_data()` to BF16Tensor
- Modify RMSNormOp to detect type and use appropriate accessor

**Option B**: Separate Activation Tensor Classes
- `FP32ActivationTensor` (mutable) vs `FP32WeightTensor` (immutable)
- Same for BF16, FP16, Q8_1

**Option C**: New Op Interface
- Add `ITensorRMSNorm::apply_q8_1(Q8_1Block*, ...)` etc.
- Modify RMSNormOp to dispatch based on tensor type

### Recommendation
Option A is simplest. The key insight is that **activations need to be mutable** while **weights are immutable**. The current design conflates these concerns.

## Files Changed Summary

### Modified
- `src/v2/kernels/cpu/ops/CPURMSNormTypedKernel.cpp` - Added want_parallel()
- `src/v2/kernels/cpu/ops/CPURMSNormTypedKernel.h` - Added threshold constants
- `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp` - Integer Q8_1 RMSNorm (170 lines)
- `src/v2/kernels/cpu/primitives/RMSNormPrimitives.h` - Declarations

### Added
- `tests/v2/unit/Test__Q8_1IntegerRMSNorm.cpp` - Unit tests (712 lines)
- `tests/v2/CMakeLists.txt` - Test registration

## Next Steps

1. **Short-term**: Use FP32 activations (current working path)
2. **Medium-term**: Add mutable accessors to typed tensors for activation use
3. **Long-term**: Consider separating activation vs weight tensor hierarchies

## Performance Summary

| Scenario | Before | After | Status |
|----------|--------|-------|--------|
| Single-token BF16 RMSNorm | +7252% overhead | -64% overhead | ✅ Fixed |
| Single-token FP16 RMSNorm | +6890% overhead | -68% overhead | ✅ Fixed |
| Single-token Q8_1 RMSNorm | +4520% overhead | +120% overhead | ⚠️ Still slow |
| Batch (128 rows) all types | Faster than baseline | Same | ✅ Working |
| Typed kernel integration | Not connected | Blocked | ⏳ Architectural |
