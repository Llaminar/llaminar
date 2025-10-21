# BF16 Architecture - Corrected Understanding
**Date**: October 20, 2025  
**Context**: After reviewing llama.cpp and current weight handling

---

## Critical Insight: We're Confusing Activations and Weights

### Current Broken Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Q8_0 Weights (GGUF file) - 640MB                           │
└────────────────────┬────────────────────────────────────────┘
                     │ decodeBlock() → FP32 → BF16
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ QuantSlabCache: 4GB of BF16 weight copies                  │
│ (defeats memory savings!)                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ BF16 Activations (also uses same cache)                    │
│ (frequent access → cache churn → 3× slowdown)              │
└─────────────────────────────────────────────────────────────┘
```

### What llama.cpp Does (Correct)

```
┌─────────────────────────────────────────────────────────────┐
│ Q8_0 Weights (GGUF file) - 640MB                           │
│ STAYS COMPRESSED                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ During GEMM only:
                     │ decodeBlock(panel) → small FP16 buffer
                     │ (16-32KB working buffer, not 4GB!)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ vec_dot(Q8_0 panel, FP16 activation panel)                 │
│ Fused kernel: decode + compute                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Two Separate Problems

### Problem 1: Weight Storage (SOLVED - just use llama.cpp pattern)

**What we're doing wrong:**
- Decode entire Q8_0 weight matrix → BF16 slab
- Cache 4GB of BF16 weights
- This defeats quantization memory savings!

**What we should do:**
- Keep weights in Q8_0 (640MB compressed)
- During GEMM: stream decode Q8_0 → FP32 in small panels
- Panel size: 256-512 elements (~1-2KB buffer)
- Never materialize full FP32/BF16 weight matrix

**Implementation:**
```cpp
// Streaming weight dequant during GEMM
void matmul_with_quant_weights(
    const float* input,           // FP32 activation
    const QuantizedTensor& weight, // Q8_0 compressed
    float* output,
    int m, int n, int k)
{
    constexpr int PANEL_K = 256;
    float weight_panel[PANEL_K * n];  // Small buffer
    
    for (int ki = 0; ki < k; ki += PANEL_K) {
        // Decode Q8_0 panel on-the-fly
        weight.decodePanel(ki, PANEL_K, weight_panel);
        
        // Compute with this panel
        cblas_sgemm(..., input + ki, weight_panel, output, ...);
    }
}
```

### Problem 2: Activation Storage (THIS is what BF16 is for!)

**What we're doing wrong:**
- BF16 activations also use QuantSlabCache
- Frequent access → cache decode churn → 3× slowdown
- Activations don't need expensive quantization, just lower precision

**What we should do:**
- Store activations directly as BF16 (no cache!)
- Decode BF16→FP32 on-the-fly during GEMM (cheap!)
- Use `bf16_data()` API for zero-copy access

**Implementation:**
```cpp
class BF16Tensor {
    std::vector<bfloat16> data_;  // Direct storage, no cache
    
public:
    // Primary interface (zero-copy, fast)
    const bfloat16* bf16_data() const { return data_.data(); }
    
    // Legacy interface (allocates temp buffer)
    float* data() override {
        // Warn and allocate temporary
        thread_local std::vector<float> temp;
        temp.resize(size());
        bf16_to_fp32(data_.data(), temp.data(), size());
        return temp.data();
    }
};

// Usage in operators
void MPILinearOperator::execute(...) {
    auto bf16_input = as_bf16(input);
    const bfloat16* input_bf16 = bf16_input->bf16_data();
    
    // Stream dequant BF16 input during GEMM
    matmul_with_bf16_input(input_bf16, quant_weight, output);
}
```

---

## Corrected Design: Hybrid Approach

### Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│ MODEL WEIGHTS (persistent, large ~640MB)                 │
│ Storage: Q8_0/Q4_0 (GGUF compressed)                    │
│ Dequant: Stream to FP32 panels during GEMM              │
│ Pattern: llama.cpp vec_dot style                        │
└──────────────────────────────────────────────────────────┘
                              ↓ GEMM with streaming dequant
┌──────────────────────────────────────────────────────────┐
│ ACTIVATIONS (temporary, small ~2GB)                      │
│ Storage: BF16 (direct, no cache)                        │
│ Dequant: Cheap BF16→FP32 on-the-fly                    │
│ Access: bf16_data() zero-copy API                       │
└──────────────────────────────────────────────────────────┘
```

### Memory Budget

| Component | Current | Corrected | Savings |
|-----------|---------|-----------|---------|
| Weights (Q8_0) | 640MB | 640MB | 0 |
| Weight Cache (BF16) | 4GB | **0** | -4GB ✅ |
| Activations (BF16) | 2GB | 2GB | 0 |
| Activation Cache | 4GB | **0** | -4GB ✅ |
| **TOTAL** | **10.6GB** | **2.6GB** | **-8GB** ✅ |

---

## Implementation Plan (Revised)

### Phase 1: Remove Weight Caching (Week 1)

**Goal**: Stream dequant Q8_0 weights during GEMM (like llama.cpp)

```cpp
// Replace QuantSlab approach
// BEFORE:
QuantSlab slab;
QuantSlabCache::instance().getOrDecode(quant_tensor, col_start, col_count, slab);
adaptiveMatMulBF16(input, slab.data.data(), output, ...);

// AFTER:
matmul_stream_quant_weight(input, quant_tensor, output, m, n, k);

void matmul_stream_quant_weight(...) {
    constexpr int PANEL_K = 256;
    float weight_panel[PANEL_K * n];
    
    for (int ki = 0; ki < k; ki += PANEL_K) {
        // Decode panel from Q8_0
        decode_quant_panel(quant_tensor, ki, PANEL_K, weight_panel);
        
        // Compute immediately
        cblas_sgemm(...);
    }
}
```

**Expected Impact**:
- Memory: -4GB (no more weight cache)
- Speed: Similar or faster (no cache overhead)
- Complexity: Medium (need panel decode API)

### Phase 2: Remove Activation Caching (Week 2)

**Goal**: BF16 activations without QuantSlabCache

```cpp
// BF16Tensor refactor
class BF16Tensor {
    std::vector<bfloat16> data_;  // Direct storage
    
    // NO QuantSlabCache dependency!
    
    const bfloat16* bf16_data() const { return data_.data(); }
    
    // Decode on-demand to caller buffer
    void decode_to_fp32(float* dst) const {
        bf16_to_fp32_vectorized(data_.data(), dst, size());
    }
};
```

**Expected Impact**:
- Memory: -4GB (no more activation cache)
- Speed: 3× faster (no cache decode churn)
- Complexity: Low (just remove cache dependency)

### Phase 3: Optimize Streaming Dequant (Week 3)

**Optimizations**:
1. **Vectorized BF16→FP32**: AVX2 can do 8 conversions per instruction
2. **Cache blocking**: Reuse decoded panels across multiple output rows
3. **Fused kernels**: Combine decode + GEMM in single operation
4. **Hardware BF16**: Use Intel MKL `cblas_gemm_bf16bf16f32` when available

---

## Key Differences from Original Design

| Aspect | Original Design | Corrected Design |
|--------|-----------------|------------------|
| **Weight storage** | BF16 (cached) | Q8_0 (compressed) ✅ |
| **Weight dequant** | Eager (cache fill) | Lazy (streaming) ✅ |
| **Activation storage** | BF16 (cached) | BF16 (direct) ✅ |
| **Activation dequant** | Through cache | On-the-fly ✅ |
| **QuantSlabCache** | Used for both | **Deprecated** ✅ |
| **Memory savings** | 0 (cache overhead) | 8GB ✅ |
| **Performance** | 3× slower | ~1× FP32 ✅ |

---

## Why This Matters

### We Already Have Most of This Working!

```cpp
// From MPILinearOperator.cpp:193
bool ok = adaptiveMatMulBF16(input_data, slab.data.data(), local_output->data(),
                             (int)seq_len, (int)local_output_size, (int)input_size,
                             /*is_prefill*/ false,
                             /*distributed_partition*/ true);
```

**What we have:**
- ✅ Q8_0 weight decoding (QuantizedTensor::decodeBlock)
- ✅ BF16 GEMM backend (adaptiveMatMulBF16)
- ✅ BF16 tensor storage (BF16Tensor)

**What we need to change:**
- ❌ Remove QuantSlabCache from weight path (stream decode instead)
- ❌ Remove QuantSlabCache from BF16Tensor (direct storage)
- ❌ Add panel-based decode API (QuantizedTensor::decodePanel)

**Estimated effort**: 2-3 weeks, not 4 weeks!

---

## Success Criteria (Unchanged)

1. ✅ Keep weights in Q8_0 (640MB compressed)
2. ✅ No 4GB weight cache (streaming dequant)
3. ✅ BF16 activations without cache (direct storage)
4. ✅ 8GB memory reduction (10.6GB → 2.6GB)
5. ✅ 3× performance improvement (3× slower → ~1× FP32)
6. ✅ Maintain numerical accuracy (parity tests pass)

---

## Next Steps

1. **Audit current weight path**: Understand decodeBlock() performance
2. **Prototype panel decode**: Add QuantizedTensor::decodePanel(offset, count)
3. **Benchmark streaming vs cache**: Compare Q8_0 streaming vs BF16 cache
4. **Remove BF16Tensor cache dependency**: Make activations cache-free
5. **Integration testing**: Validate parity and performance

---

## Open Questions

1. **Is decodeBlock() already optimized?** May be faster than BF16 cache hit
2. **What's optimal panel size?** 256? 512? 1024? (benchmark needed)
3. **Should we vectorize Q8_0 decode?** AVX2 can help here too
4. **Do we keep QuantSlabCache at all?** Maybe delete entirely?

---

## Conclusion

The original design document confused **weight storage** with **activation storage**. The corrected approach:

- **Weights**: Keep compressed (Q8_0), stream dequant during GEMM (llama.cpp pattern)
- **Activations**: Store as BF16 directly, no cache needed (cheap decode)

This is **much simpler** than the original 4-week plan and leverages existing infrastructure!
