# V2 Buffer Validation Fix

**Date**: October 26, 2025  
**Component**: V2 TensorDimensions validation  
**Issue**: Pre-allocated buffers rejected by exact shape matching  
**Status**: ✅ **FIXED**

## Problem

E2E test was failing with:
```
[TENSOR VALIDATION ERROR] after_attn_norm: Shape mismatch
  Expected: hidden[1,896] [1, 896]
  Actual:   [2048, 896]
```

Root cause: V2 pipeline pre-allocates buffers with `max_seq_len` (2048) but validation macros required **exact** shape matching. For a sequence of length 1, buffers are `[2048, 896]` but validation expected `[1, 896]`.

This is correct behavior for pre-allocated buffers - they should be sized for `max_seq_len` but only use the first `seq_len` rows.

## Solution

### 1. Added `matches_prefix()` method to `TensorSpec`

Allows first dimension (typically seq_len) to be >= expected:

```cpp
bool matches_prefix(const std::vector<size_t> &actual_shape) const {
    if (actual_shape.size() != expected_shape.size())
        return false;

    for (size_t i = 0; i < actual_shape.size(); ++i) {
        if (i == 0) {
            // First dimension: allow actual >= expected (buffer can be larger)
            if (actual_shape[i] < expected_shape[i])
                return false;
        } else {
            // Other dimensions: require exact match
            if (actual_shape[i] != expected_shape[i])
                return false;
        }
    }
    return true;
}
```

### 2. Added `VALIDATE_TENSOR_BUFFER` macro

New macro for validating pre-allocated buffers:

```cpp
#define VALIDATE_TENSOR_BUFFER(tensor, spec, stage) ...
    if (!(spec).matches_prefix(_actual_shape)) {
        std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Buffer too small\n";
        std::cerr << "  Required: " << (spec).description << " " << (spec).shape_str() << "\n";
        std::cerr << "  Actual:   " << TensorSpec::shape_str(_actual_shape) << "\n";
        std::cerr << "  Hint: Buffer first dimension must be >= required seq_len\n";
        std::abort();
    }
```

### 3. Updated Qwen2Pipeline to use new macro

Changed validation for pre-allocated buffers from `VALIDATE_TENSOR` to `VALIDATE_TENSOR_BUFFER`:

```cpp
// Before (exact match - fails for max_seq_len buffers):
VALIDATE_TENSOR(normalized_hidden, spec_hidden(seq_len), "after_attn_norm");

// After (allows buffer >= required):
VALIDATE_TENSOR_BUFFER(normalized_hidden, spec_hidden(seq_len), "after_attn_norm");
```

## Result

✅ **Validation error fixed** - buffers can now be larger than required  
🔄 **New issue discovered** - Segmentation fault in `q_gemm->multiply()`

The segfault is a **different bug** in the quantized GEMM kernel, unrelated to buffer validation. Progress: we've moved from "can't validate buffers" to "executing GEMM but crashing inside".

## Files Modified

- `src/v2/pipelines/TensorDimensions.h` - Added `matches_prefix()` and `VALIDATE_TENSOR_BUFFER`
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Use `VALIDATE_TENSOR_BUFFER` for pre-allocated buffers

## Next Issue

**Segmentation fault** in `QuantizedGemmKernel::multiply()` during OpenMP parallel execution:

```
[09:29:46.817] [INFO] About to call Q GEMM: seq_len=1, n_heads*head_dim=896, d_model=896
[09:29:46.817] [INFO] normalized_hidden data pointer: 0x7102b3a33010
[09:29:46.817] [INFO] buffers.Q data pointer: 0x7102b3332010
[09:29:46.817] [INFO] buffers.Q shape: [2048, 896]
[11681926e14f:762873] *** Process received signal ***
[11681926e14f:762873] Signal: Segmentation fault (11)
```

Likely causes:
1. Invalid `decoder_` state in `QuantizedGemmKernel`
2. Bad pointer in `Q4_0Tensor::raw_data_`
3. Buffer overrun in `decode_block_at()` or `dot_product_simd()`
4. Uninitialized weight tensor data

Pointers are valid (non-null), so the issue is likely inside the GEMM kernel's parallel loop.

## Commit Message

```
fix(v2): Add relaxed buffer validation for pre-allocated tensors

Pipelines pre-allocate buffers with max_seq_len but use them with
varying seq_len. Validation should allow buffer first dimension to be
>= required size, not require exact match.

- Add TensorSpec::matches_prefix() for relaxed shape matching
- Add VALIDATE_TENSOR_BUFFER macro for pre-allocated buffers
- Update Qwen2Pipeline to use buffer validation for normalized/residual tensors
- Fixes validation error: "Expected [1,896], Actual [2048,896]"

Unblocks E2E test progress (now hits different bug in GEMM kernel).
```
