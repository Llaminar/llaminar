# BF16 Native Operator Implementation - Design Document
**Date**: October 20, 2025  
**Status**: DRAFT - Architecture Design  
**Goal**: Production-ready BF16 activations with streaming dequant and native compute

---

## Executive Summary

Current BF16 implementation uses a global cache with decode churn, resulting in 3× performance penalty and 4GB memory overhead. This document outlines a path to **native BF16 operators** inspired by llama.cpp's streaming dequant approach, but adapted for BF16 instead of FP16.

**Key Insight from llama.cpp**: They never fully dequantize tensors. Instead, they:
1. **Stream dequant in panels** during GEMM operations
2. **Use specialized vec_dot kernels** that operate on quantized data directly
3. **Only materialize FP16 in small working buffers** (16-32 elements at a time)
4. **Fuse dequant into the compute kernel** (no separate dequant pass)

---

## Current Architecture Problems

### Problem 1: Cache-Based Decode (Root Cause)
```cpp
// Current broken pattern
float* data = bf16_tensor->data();  // Goes through QuantSlabCache
// Cache decodes entire tensor BF16→FP32
// Stores 4GB of decoded copies
// Evicts/reallocates constantly
cblas_sgemm(..., data, ...);  // Compute in FP32
```

**Issues**:
- 3× slower (decode overhead)
- 4GB cache memory
- Pointer invalidation bugs
- Defeats memory savings

### Problem 2: Wrong Abstraction
- `QuantSlabCache` designed for **expensive rare access** (quantized weights)
- BF16 activations need **cheap frequent access**
- Cache is the wrong tool for this job

---

## llama.cpp's Approach (Streaming Dequant)

### Key Pattern: Fused Vec_Dot Kernels

```c
// From ggml-cpu.c:1191
vec_dot(ne00, &tmp[ir0 - iir0], stride, 
        src0_row + ir0 * nb01, stride0,    // Quantized source A
        src1_col, stride1,                  // Quantized source B  
        num_rows_per_vec_dot);              // Process multiple rows

// vec_dot implementation (pseudo-code):
void ggml_vec_dot_q4_0_q8_0(int n, float * s, const block_q4_0 * x, const block_q8_0 * y) {
    float sum = 0.0f;
    for (int i = 0; i < n/block_size; i++) {
        // Dequant small block on-the-fly
        float scale_x = decode_scale(x[i]);
        float scale_y = decode_scale(y[i]);
        
        // Compute dot product of this block
        int block_sum = 0;
        for (int j = 0; j < block_size; j++) {
            int8_t val_x = decode_element(x[i], j);
            int8_t val_y = decode_element(y[i], j);
            block_sum += val_x * val_y;
        }
        
        sum += scale_x * scale_y * block_sum;
    }
    *s = sum;
}
```

### Key Concepts

1. **No full dequant**: Never materialize entire FP32 tensor
2. **Block-wise processing**: Dequant 32-256 elements at a time
3. **Fused compute**: Dequant + dot product in same kernel
4. **Small working buffers**: Only 16-32 floats in flight
5. **Type-specialized kernels**: One kernel per (srcA_type, srcB_type) pair

---

## Our BF16 Adaptation Strategy

### Phase 1: Streaming BF16 Matmul (Week 1)

Implement llama.cpp-style streaming dequant for BF16:

```cpp
// New approach: Stream BF16→FP32 during GEMM
void bf16_stream_gemm(
    int m, int n, int k,
    const bfloat16* A,  // BF16 input (no decode yet!)
    const bfloat16* B,  // BF16 weights (no decode yet!)
    float* C)           // FP32 output
{
    constexpr int PANEL_K = 256;  // Dequant panel size
    
    float A_panel[PANEL_K];  // Small working buffer
    float B_panel[PANEL_K];
    
    for (int ki = 0; ki < k; ki += PANEL_K) {
        int panel_size = std::min(PANEL_K, k - ki);
        
        for (int mi = 0; mi < m; mi++) {
            // Dequant A panel on-the-fly
            #pragma omp simd
            for (int kj = 0; kj < panel_size; kj++) {
                A_panel[kj] = A[mi * k + ki + kj].to_float();
            }
            
            for (int ni = 0; ni < n; ni++) {
                // Dequant B panel on-the-fly
                #pragma omp simd
                for (int kj = 0; kj < panel_size; kj++) {
                    B_panel[kj] = B[(ki + kj) * n + ni].to_float();
                }
                
                // Dot product on panels
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int kj = 0; kj < panel_size; kj++) {
                    sum += A_panel[kj] * B_panel[kj];
                }
                C[mi * n + ni] += sum;
            }
        }
    }
}
```

**Benefits**:
- Only 512-1024 bytes working memory (vs 4GB cache!)
- No pointer invalidation (no cache)
- Vectorizable dequant (SIMD-friendly)
- Cache-friendly access patterns

### Phase 2: BLAS Integration (Week 1-2)

Leverage existing BLAS for FP32 compute:

```cpp
void bf16_gemm_via_blas(
    int m, int n, int k,
    const bfloat16* A,
    const bfloat16* B,
    float* C)
{
    // Strategy: Dequant to FP32, use optimized BLAS
    
    // Option A: Dequant both to temporary buffers (simple but copies)
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(k * n);
    bf16_to_fp32(A, A_fp32.data(), m * k);
    bf16_to_fp32(B, B_fp32.data(), k * n);
    cblas_sgemm(..., A_fp32.data(), B_fp32.data(), C, ...);
    
    // Option B: Dequant A, stream-dequant B panels (better memory)
    std::vector<float> A_fp32(m * k);
    bf16_to_fp32(A, A_fp32.data(), m * k);
    
    for (int panel = 0; panel < n; panel += PANEL_SIZE) {
        float B_panel[k * PANEL_SIZE];
        bf16_to_fp32(B + panel * k, B_panel, k * PANEL_SIZE);
        cblas_sgemm(..., A_fp32.data(), B_panel, C + panel, ...);
    }
}
```

### Phase 3: Native BF16 GEMM (Week 2-3)

Use hardware BF16 GEMM when available:

```cpp
bool bf16_gemm_native(
    int m, int n, int k,
    const bfloat16* A,
    const bfloat16* B,
    float* C)
{
#ifdef HAVE_MKL
    // Intel MKL cblas_gemm_bf16bf16f32 (Ice Lake+, Sapphire Rapids)
    if (has_amx_bf16()) {
        cblas_gemm_bf16bf16f32(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
        return true;
    }
#endif

#ifdef HAVE_OPENBLAS_BF16
    // OpenBLAS 0.3.26+ experimental BF16 support
    if (openblas_has_bf16()) {
        cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
        return true;
    }
#endif

    // Fallback: streaming dequant
    return false;
}
```

### Phase 4: Operator Integration (Week 3-4)

Refactor operators to use BF16 natively:

```cpp
class MPILinearOperator {
    bool execute(...) {
        auto bf16_input = std::dynamic_pointer_cast<BF16Tensor>(input);
        auto bf16_weight = std::dynamic_pointer_cast<BF16Tensor>(weight);
        
        if (bf16_input && bf16_weight) {
            // Native BF16 path
            const bfloat16* input_bf16 = bf16_input->bf16_data();
            const bfloat16* weight_bf16 = bf16_weight->bf16_data();
            float* output_fp32 = output->data();
            
            // Try hardware BF16 GEMM first
            if (bf16_gemm_native(m, n, k, input_bf16, weight_bf16, output_fp32)) {
                return true;
            }
            
            // Fallback to streaming dequant
            bf16_gemm_via_streaming(m, n, k, input_bf16, weight_bf16, output_fp32);
            return true;
        }
        
        // Legacy FP32 path
        return execute_fp32(...);
    }
};
```

---

## Tensor Interface Changes

### New BF16Tensor API

```cpp
class BF16Tensor : public TensorBase {
private:
    std::vector<bfloat16> data_;  // Only BF16 storage!
    
public:
    // ===== Primary Interface (BF16 Native) =====
    
    /** Get raw BF16 data pointer (zero-copy, fast) */
    const bfloat16* bf16_data() const { return data_.data(); }
    bfloat16* bf16_data() { return data_.data(); }
    
    // ===== Legacy Interface (FP32 via temporary) =====
    
    /** Get FP32 data - ALLOCATES TEMPORARY BUFFER */
    float* data() override {
        LOG_WARN_ONCE("BF16Tensor::data() allocates FP32 copy - use bf16_data() for performance");
        
        // Allocate temporary buffer (caller responsible for lifetime)
        thread_local std::vector<float> fp32_buffer;
        fp32_buffer.resize(element_count());
        
        // Dequant to temporary
        bf16_to_fp32(data_.data(), fp32_buffer.data(), element_count());
        
        return fp32_buffer.data();
    }
    
    /** Decode to caller-provided buffer (explicit) */
    void decode_to_fp32(float* dst) const override {
        bf16_to_fp32(data_.data(), dst, element_count());
    }
};
```

**Key Changes**:
1. **Remove QuantSlabCache dependency** - no more global cache!
2. **Primary API is `bf16_data()`** - encourages native BF16 usage
3. **Legacy `data()` issues warning** - discourages FP32 conversion
4. **Explicit `decode_to_fp32()`** - caller controls buffer

---

## Operator Refactoring Checklist

### MPILinearOperator ✅ (Priority 1 - 90% of compute)

```cpp
// BEFORE (current - uses cache)
const float* input_data = input->data();
const float* weight_data = weight->data();
float* output_data = output->data();
adaptiveMatMul(input_data, weight_data, output_data, ...);

// AFTER (native BF16)
if (auto bf16_input = as_bf16(input)) {
    if (auto bf16_weight = as_bf16(weight)) {
        // Native BF16 path
        bf16_gemm(m, n, k,
                  bf16_input->bf16_data(),
                  bf16_weight->bf16_data(),
                  output->data());
        return true;
    }
}
// Fallback to FP32 path...
```

**Status**: We already have `adaptiveMatMulBF16` for quantized slabs! Just extend it.

### MPIAttentionOperator (Priority 2 - Numerically sensitive)

**Strategy**: Hybrid approach
- Q/K/V projections: BF16 (just matmuls)
- Attention scores: **FP32** (numerical stability)
- Softmax: **FP32** (critical for stability)
- Context aggregation: BF16 (just matmul)

```cpp
// Projections in BF16
bf16_gemm(seq_len, d_head, d_model, input_bf16, Wq_bf16, Q_fp32);

// Scores in FP32 (convert Q/K to FP32)
cblas_sgemm(..., Q_fp32, K_fp32, scores_fp32, ...);

// Softmax in FP32
softmax_fp32(scores_fp32, attn_weights_fp32, seq_len);

// Context in BF16 (can quantize attn_weights back to BF16)
bf16_gemm(seq_len, d_head, seq_len, attn_weights_bf16, V_bf16, context_fp32);
```

### MPIRMSNormOperator (Priority 3 - Stability critical)

**Strategy**: Keep in FP32
- RMSNorm is numerically sensitive (sqrt, division)
- Small operations (not performance bottleneck)
- Accept BF16 input, compute in FP32, output FP32

```cpp
void execute(...) {
    // Dequant to FP32, compute, output FP32
    std::vector<float> input_fp32(seq_len * hidden_size);
    if (auto bf16_input = as_bf16(input)) {
        bf16_input->decode_to_fp32(input_fp32.data());
    }
    
    // RMSNorm in FP32
    rms_norm_fp32(input_fp32.data(), output->data(), ...);
}
```

### MPIRoPEOperator (Priority 4 - Can be BF16)

**Strategy**: Native BF16
- RoPE is multiply + add (simple)
- Can operate directly on BF16
- Small numerical error acceptable

```cpp
void apply_rope_bf16(bfloat16* data, int seq_len, int d_head, int pos_offset) {
    for (int seq = 0; seq < seq_len; seq++) {
        int pos = pos_offset + seq;
        for (int d = 0; d < d_head / 2; d++) {
            float theta = pos / pow(10000.0f, 2.0f * d / d_head);
            float cos_val = cos(theta);
            float sin_val = sin(theta);
            
            // Apply rotation in BF16
            bfloat16 x0 = data[seq * d_head + 2*d];
            bfloat16 x1 = data[seq * d_head + 2*d + 1];
            
            data[seq * d_head + 2*d]     = bfloat16(cos_val * x0.to_float() - sin_val * x1.to_float());
            data[seq * d_head + 2*d + 1] = bfloat16(sin_val * x0.to_float() + cos_val * x1.to_float());
        }
    }
}
```

---

## Performance Optimization Strategies

### 1. Vectorized Dequant (SIMD)

```cpp
// Scalar dequant (slow)
for (int i = 0; i < n; i++) {
    dst[i] = src[i].to_float();
}

// AVX2 vectorized dequant (8× faster)
void bf16_to_fp32_avx2(const bfloat16* src, float* dst, size_t n) {
    #ifdef __AVX2__
    for (size_t i = 0; i < n; i += 8) {
        __m128i bf16_data = _mm_loadu_si128((__m128i*)&src[i]);
        __m256i expanded = _mm256_cvtepu16_epi32(bf16_data);
        __m256i shifted = _mm256_slli_epi32(expanded, 16);  // BF16 → FP32
        __m256 fp32_data = _mm256_castsi256_ps(shifted);
        _mm256_storeu_ps(&dst[i], fp32_data);
    }
    #endif
}
```

### 2. Cache Blocking

```cpp
// Block GEMM for better cache reuse
constexpr int BM = 64;   // M-dimension block
constexpr int BN = 256;  // N-dimension block
constexpr int BK = 256;  // K-dimension block

for (int bi = 0; bi < m; bi += BM) {
    for (int bj = 0; bj < n; bj += BN) {
        // Dequant A block once
        float A_block[BM * BK];
        bf16_to_fp32(&A[bi * k], A_block, BM * BK);
        
        for (int bk = 0; bk < k; bk += BK) {
            // Dequant B block, compute immediately
            float B_block[BK * BN];
            bf16_to_fp32(&B[bk * n + bj], B_block, BK * BN);
            
            // Compute on FP32 blocks
            sgemm_micro_kernel(BM, BN, BK, A_block, B_block, &C[bi * n + bj]);
        }
    }
}
```

### 3. OpenMP Parallelization

```cpp
void bf16_gemm_parallel(int m, int n, int k, const bfloat16* A, const bfloat16* B, float* C) {
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < m; bi += BM) {
        for (int bj = 0; bj < n; bj += BN) {
            // Thread-local buffers (no false sharing)
            float A_local[BM * BK];
            float B_local[BK * BN];
            
            // Each thread dequants and computes its blocks
            // ...
        }
    }
}
```

---

## Migration Path

### Week 1: Foundation
- [x] Fix cache bugs (4GB capacity) ✅ DONE
- [ ] Implement streaming BF16 GEMM (prototype)
- [ ] Add `bf16_data()` to BF16Tensor
- [ ] Benchmark streaming vs cache approach

### Week 2: Operator Migration
- [ ] Refactor MPILinearOperator for native BF16
- [ ] Add BF16 path to adaptiveMatMul
- [ ] Test with quantized slab path (already BF16-aware!)
- [ ] Validate numerical accuracy

### Week 3: Production Hardening
- [ ] Vectorize dequant with AVX2/NEON
- [ ] Add cache blocking optimization
- [ ] Implement fallback paths (FP32, mixed precision)
- [ ] Comprehensive benchmarking

### Week 4: Remaining Operators
- [ ] MPIAttentionOperator (hybrid BF16/FP32)
- [ ] MPIRoPEOperator (native BF16)
- [ ] Keep MPIRMSNormOperator in FP32
- [ ] Integration testing

---

## Expected Performance

### Memory Savings
- **Current**: 4GB cache + 2GB activations = 6GB total
- **Target**: 2GB activations (BF16 only) = **67% reduction**

### Speed Improvement
- **Current**: 3× slower than FP32 (cache decode overhead)
- **Target**: 0.9-1.1× FP32 speed (streaming dequant overhead)
- **Best Case**: 0.8× FP32 (hardware BF16 GEMM on Ice Lake+)

### Numerical Accuracy
- **Observed**: ~5e-05 relative error (parity tests)
- **Expected**: Same or better (no change in precision)

---

## Success Criteria

1. **No global cache** - BF16Tensor self-contained
2. **True 2× memory savings** - No FP32 copies in flight
3. **≤1.2× FP32 speed** - Acceptable performance overhead
4. **Passes parity tests** - Maintains numerical accuracy
5. **Production viable** - Can run 7B+ models efficiently

---

## Open Questions

1. **How much does vectorized dequant help?** Need AVX2 benchmarks
2. **Is panel size 256 optimal?** May vary by cache size
3. **Should we expose BF16 compute to Python?** API considerations
4. **Do we need mixed precision config?** (Some ops BF16, some FP32)

---

## Related Files

- `src/tensors/BF16Tensor.h` - Tensor interface (to be refactored)
- `src/operators/MPILinearOperator.cpp` - Primary target for BF16 (has `adaptiveMatMulBF16` already!)
- `src/AdaptiveMatmul.h` - Backend selection logic
- `src/operators/QuantSlabCache.h` - To be deprecated for BF16

---

## References

- llama.cpp ggml-cpu.c:1115 - `ggml_compute_forward_mul_mat_one_chunk` (streaming pattern)
- llama.cpp ggml-quants.c - Quantized vec_dot kernels (inspiration)
- Intel MKL: `cblas_gemm_bf16bf16f32` (hardware BF16 GEMM)
- OpenBLAS 0.3.26: `cblas_sbgemm` (experimental BF16 support)

---

**Next Step**: Implement streaming BF16 GEMM prototype and benchmark against current cache approach.
