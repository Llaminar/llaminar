# OneDNN for Q8_0×Q8_0 GEMM Investigation

**Date**: November 12, 2025  
**Question**: Can we use OneDNN's s8s8s32 GEMM for Q8_0×Q8_0 matrix multiplication?  
**Answer**: **No** - fundamental mismatch between uniform and block quantization.

---

## Background

### Q8_0 Quantization Format
```c
struct block_q8_0 {
    fp16 d;         // scale (per 32-element block)
    int8_t qs[32];  // quantized values
};

// Dequantization: x[i] = d * qs[i]
// GEMM: C[i,j] = Σ_k (A_d[i,k/32] * A_q[i,k]) * (B_d[j,k/32] * B_q[j,k])
```

### OneDNN s8s8s32 GEMM API
```c
dnnl_gemm_s8s8s32(
    char transa, char transb, char offsetc,
    int M, int N, int K,
    float alpha,              // Single global scale
    const int8_t *A, int lda, int8_t ao,  // A matrix + offset
    const int8_t *B, int ldb, int8_t bo,  // B matrix + offset
    float beta,
    int32_t *C, int ldc,
    const int32_t *co         // Column/row/fixed offsets
);

// Computes: C = alpha * (A - ao) * (B - bo) + beta * C + co
```

**Key Constraint**: OneDNN uses **single `alpha` scale** for entire GEMM operation.

---

## The Fundamental Problem

### What OneDNN Computes
```
C_out[i,j] = alpha * Σ_k (A[i,k] - ao) * (B[k,j] - bo) + beta * C_old[i,j] + co
```
- **Uniform quantization**: Same `alpha` applies to all elements
- Designed for: Neural network activations with per-tensor quantization

### What Q8_0 Needs
```
C_out[i,j] = Σ_k (A_scale[i, k/32] * A_quant[i,k]) * (B_scale[j, k/32] * B_quant[k,j])
           = Σ_k (A_scale[i, k/32] * B_scale[j, k/32]) * (A_quant[i,k] * B_quant[k,j])
```
- **Block quantization**: Different scale products for different k ranges!
- Each 32-element block along K has different `A_scale × B_scale` product
- Cannot factor out a single global scale

---

## Attempted Approaches

### Approach 1: Direct OneDNN GEMM ❌
**Idea**: Flatten Q8_0 blocks to dense INT8, call OneDNN, apply scales after.

**Problem**:
```python
# OneDNN gives us:
C_int32[i,j] = Σ_k A_quant[i,k] * B_quant[k,j]

# But we need:
C_fp32[i,j] = Σ_k (scale_A[i,k/32] * scale_B[j,k/32]) * (A_quant[i,k] * B_quant[k,j])

# Can't apply scales post-hoc because:
# - Different k blocks use different scale products
# - Can't factor out scales from the summation
```

**Conclusion**: Mathematically impossible to post-apply per-block scales.

---

### Approach 2: Per-Block GEMM ❌
**Idea**: Call OneDNN GEMM for each K-block (32 elements) with appropriate scale.

**Problem**:
```
# For K=896 with block_size=32:
num_blocks = 896 / 32 = 28 blocks

# Would need 28 separate GEMMs:
for kb in range(28):
    alpha_kb = mean(A_scales[:, kb] * B_scales[:, kb])  # Still wrong!
    C += OneDNN_GEMM(A[:, kb*32:(kb+1)*32], B[:, kb*32:(kb+1)*32], alpha=alpha_kb)
```

**Issues**:
1. 28× GEMM overhead (setup, dispatch, cleanup)
2. Still can't handle per-output scale variation (each C[i,j] needs different scales)
3. Defeats purpose of fast GEMM

**Conclusion**: Too slow and still mathematically incorrect.

---

### Approach 3: Dequantize to FP32 ⚠️
**Idea**: Dequantize Q8_0 → FP32, use standard BLAS GEMM.

**Problem**:
- Defeats entire purpose of INT8 quantization
- 4× memory bandwidth (FP32 vs INT8)
- Loses INT8 VNNI acceleration
- Would be slower than our custom microkernel

**Conclusion**: Defeats the purpose.

---

### Approach 4: OneDNN Matmul Primitive with Custom Post-Ops ⚠️
**Idea**: Use OneDNN's more flexible matmul primitive with custom post-processing.

**Investigation**:
- OneDNN matmul supports per-channel/per-tensor scales, but not per-block
- Custom post-ops would require:
  - Per-output scale computation (complex)
  - Separate kernel invocation for scale application
- Likely not faster than fused custom kernel

**Conclusion**: Complex, unproven benefit.

---

## Why OneDNN Achieves 6610 GOPS

OneDNN's impressive performance is for **uniform quantization** scenarios:

```c
// Typical neural network use case:
// - Activations: Per-tensor quantized (single scale)
// - Weights: Per-channel quantized (one scale per output channel)

// Example: INT8 conv2d
activation_scale = 0.05f;  // Single scale for entire activation tensor
weight_scales[] = {0.02f, 0.03f, ..., 0.04f};  // One per output channel

// OneDNN can factor out scales:
C[i,j] = activation_scale * weight_scale[j] * Σ_k A_quant[i,k] * W_quant[j,k]
       = (activation_scale * weight_scale[j]) * OneDNN_GEMM(...)
```

**Key difference**: Scales factor out of the K-summation!

---

## Q8_0 Characteristics

**Why Q8_0 uses block quantization**:
- LLM weights have non-uniform distributions
- Per-tensor quantization loses too much precision
- Block size 32 balances:
  - Quantization accuracy (small blocks capture local variation)
  - Memory overhead (2 bytes scale per 32 bytes data = 6.25% overhead)

**Trade-off**:
- ✅ Better quantization quality
- ❌ Can't use off-the-shelf uniform-quant GEMM kernels

---

## Recommendation

### ✅ Keep Custom INT8 Microkernel

**Advantages**:
1. **Fused scale handling**: Apply scales in K-loop as blocks are processed
2. **No separate passes**: Single kernel does quantized GEMM + scale application
3. **Correct semantics**: Handles per-block scales naturally
4. **Optimizable**: Can still use dpbusd, K-unrolling, prefetching, etc.

**Current performance**: 102 GOPS (M=128, Release build)

**Improvement opportunities**:
1. ✅ K-loop unrolling (already doing 4× unroll = 16 elements)
2. ⏳ Prefetching (not yet implemented)
3. ⏳ Compensation optimization (current bottleneck)
4. ⏳ Better register blocking (validated that 16×16 is optimal)

---

## Alternative: IQ4_NL → Q8_0 → GEMM Path

**Current workflow**:
```
IQ4_NL weights → decode to Q8_0 → Q8_0×Q8_0 INT8 GEMM → FP32 output
```

**Could we use OneDNN here?**

**Option A**: Decode IQ4_NL → INT8 (dense), use OneDNN
- Still block-quantized (scales per 32 elements)
- Same problem as Q8_0×Q8_0

**Option B**: Decode IQ4_NL → FP32, use FP32 GEMM
- Defeats purpose of quantization
- Much slower

**Conclusion**: No, same fundamental issue.

---

## Summary Table

| Approach | OneDNN Compatible? | Performance | Correctness | Recommendation |
|----------|-------------------|-------------|-------------|----------------|
| **Direct s8s8s32 GEMM** | ❌ No | N/A | ❌ Wrong scales | Don't use |
| **Per-block GEMM** | ⚠️ Partial | ❌ Very slow | ⚠️ Approximate | Don't use |
| **Dequant to FP32** | ✅ Yes | ❌ Slow | ✅ Correct | Defeats purpose |
| **Custom microkernel** | N/A | ✅ 102 GOPS | ✅ Correct | **Use this** |

---

## Key Insight

> **OneDNN's INT8 GEMM is optimized for uniform quantization (neural networks).  
> Q8_0 uses block quantization (LLM weights).  
> These are fundamentally incompatible without dequantization.**

Our custom microkernel with fused scale handling is the correct approach.

---

## Test Results

Test program: `test_onednn_q8_gemm.cpp`

```
=== OneDNN Q8_0×Q8_0 GEMM Investigation ===

Problem: M=64, N=128, K=896
Q8_0 block size: 32 elements

✅ OneDNN s8s8s32 GEMM succeeded!
   Result: INT32 accumulation of quantized values (no scales applied)

❌ BUT: Applying Q8_0 scales is complex:
   - Each output C[i,j] = sum_k (A[i,k] * B[j,k])
   - In Q8_0: C[i,j] = sum_k (A_scale[i,k/32] * A_q[i,k] * B_scale[j,k/32] * B_q[j,k])
   - Different k indices use different scale products!
   - Can't apply single alpha/beta to entire result

✅ CONCLUSION: Stick with custom INT8 microkernel!
```

---

## Next Steps for Custom Kernel Optimization

Based on this investigation, focus on improving our custom microkernel:

1. **Compensation optimization** (current bottleneck):
   - Per-row store/load cycles expensive
   - Investigate SIMD shuffle/permute for in-register compensation
   - Or precompute compensation offsets

2. **K-loop unrolling** (already partially done):
   - Current: 4× unroll (16 elements per iteration)
   - OneDNN: Likely similar or more aggressive

3. **Prefetching**:
   - Prefetch next B block (64 bytes ahead)
   - Prefetch next A rows

4. **Profile-guided optimization**:
   - Use perf/VTune to identify actual bottleneck
   - May be memory bandwidth limited, not compute

**Target**: Close gap from 102 GOPS to theoretical peak (~500-1000 GOPS for INT8 on AVX512).
