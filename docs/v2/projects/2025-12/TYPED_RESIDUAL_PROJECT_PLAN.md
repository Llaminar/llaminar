# Typed Residual Optimization Project Plan

**Author**: David Sanftenberg  
**Date**: December 2, 2025  
**Last Updated**: December 3, 2025  
**Status**: In Progress (Phase 1 Complete, Phase 2 Partial - RMSNorm Done)

## Executive Summary

This document outlines the implementation plan for **typed residuals** and **typed KV cache** in Llaminar V2 - memory bandwidth optimizations that store intermediate tensors in reduced-precision formats (Q8_1, FP16, BF16) instead of FP32.

**Key insight**: The optimization only yields benefits if we **fuse** the output quantization directly into kernel epilogues, so we never write FP32 to DRAM and then read it back for quantization. The data must go `Registers → Quantize → DRAM` in one pass.

### Scope

| Component | Current | Target Formats | Memory Savings | Config |
|-----------|---------|----------------|----------------|--------|
| **Residual Buffers** | FP32 | FP32, BF16, FP16, Q8_1 | Up to 3.5× | `ActivationPrecision` enum |
| **KV Cache** | FP32 | FP32, BF16, FP16, Q8_1 | Up to 3.5× | `ActivationPrecision` enum |

**Design Simplification**: All buffer formats (residuals, activations, KV cache) use a single `ActivationPrecision` enum. Kernels are treated as black boxes - they accept inputs in the configured precision and produce outputs in the same precision. This unified approach simplifies configuration and eliminates the need for separate format enums.

### Canonical Implementation Pattern

The `CPURMSNormTypedKernel<ActivationPrecision>` serves as the reference implementation for typed kernels:

```cpp
// Template on ActivationPrecision (not TensorT)
template <ActivationPrecision Precision>
class CPURMSNormTypedKernel : public CPUKernelBase {
public:
    using Metadata = detail::PrecisionMetadata<Precision>;
    using StorageType = typename Metadata::StorageType;
    
    // Typed API: input/output in StorageType, gamma always FP32
    bool apply_typed(const StorageType* input, const float* gamma,
                     StorageType* output, int rows, int cols, float epsilon);
    
    // Fused residual + RMSNorm: dequant(residual) + fp32_input → RMSNorm → output
    bool apply_with_residual_add(const StorageType* residual, const float* fp32_input,
                                  const float* gamma, StorageType* output,
                                  int rows, int cols, float epsilon);
};
```

**Key characteristics:**
- Template on `ActivationPrecision` enum (not tensor types)
- Use `detail::PrecisionMetadata<Precision>` for storage type mapping
- Call `simd::` functions from `SIMDHelpers.h` for SIMD-optimized conversions
- Process row-by-row to keep scratch buffers in L1/L2 cache
- Stack-allocate scratch for d_model ≤ 4096, heap for larger

### Memory Impact (Qwen2.5-7B, 4K context)

| Component | FP32 Size | Q8_1 Size | BF16/FP16 Size |
|-----------|-----------|-----------|----------------|
| KV Cache (32 layers × 4K × 4096) | ~2 GB | ~600 MB | ~1 GB |
| Residuals (per forward) | ~168 MB | ~48 MB | ~84 MB |

## Current Architecture

### Existing Infrastructure

1. **ActivationPrecision Enum** (`PipelineConfig.h`)
   ```cpp
   enum class ActivationPrecision { FP32, BF16, FP16, Q8_1 };
   ```
   - Already plumbed through pipeline configuration
   - Used to create activation tensors via `createActivationTensor()`
   - Q8_1 replaces unused INT32 enum value
   - Single enum controls residuals, activations, and KV cache formats

2. **PrecisionMetadata Traits** (`CPURMSNormTypedKernel.h:detail`)
   ```cpp
   template <ActivationPrecision Precision>
   struct PrecisionMetadata;
   
   template<> struct PrecisionMetadata<ActivationPrecision::FP32> {
       using StorageType = float;
       static constexpr const char* name = "FP32";
       static constexpr float compression_ratio = 1.0f;
   };
   template<> struct PrecisionMetadata<ActivationPrecision::BF16> {
       using StorageType = uint16_t;  // BF16 as raw bits
       static constexpr const char* name = "BF16";
       static constexpr float compression_ratio = 2.0f;
   };
   template<> struct PrecisionMetadata<ActivationPrecision::FP16> {
       using StorageType = uint16_t;  // FP16 as raw bits
       static constexpr const char* name = "FP16";
       static constexpr float compression_ratio = 2.0f;
   };
   template<> struct PrecisionMetadata<ActivationPrecision::Q8_1> {
       using StorageType = Q8_1Block;  // 36 bytes per 32 elements
       static constexpr const char* name = "Q8_1";
       static constexpr float compression_ratio = 3.5f;
   };
   ```

3. **SIMD Conversion Functions** (`SIMDHelpers.h`)
   All precision conversion primitives live in `simd::` namespace with AVX512/AVX2/scalar fallbacks:
   ```cpp
   // BF16 ↔ FP32
   simd::convert_bf16_to_fp32(src, dst, count);
   simd::convert_fp32_to_bf16(src, dst, count);
   
   // FP16 ↔ FP32  
   simd::convert_fp16_to_fp32(src, dst, count);
   simd::convert_fp32_to_fp16(src, dst, count);
   
   // Q8_1 ↔ FP32
   simd::dequantize_q8_1_to_fp32(src, dst, count);
   simd::quantize_fp32_to_q8_1_blocks(src, dst, count);
   
   // Fused residual operations (dequant + add in single pass)
   simd::fused_fp32_residual_add(residual, input, output, count);
   simd::fused_bf16_residual_add(residual, input, output, count);
   simd::fused_fp16_residual_add(residual, input, output, count);
   simd::fused_q8_1_residual_add(residual, input, output, count);
   ```

4. **Templated Kernels** (existing tensor-typed kernels)
   - `CpuAttentionKernelT<TensorT>` - FP32, BF16, FP16, Q8_1 specializations
   - `CPURMSNormKernelT<TensorT>` - templated on tensor type
   - `CPUSwiGLUKernelT<TensorT>` - templated
   - `CPURoPEKernelT<TensorT>` - templated
   - `CPUSoftmaxKernelT<TensorT>` - templated

5. **Q8_1Block Structure** (`BlockStructures.h:68`)
   ```cpp
   struct Q8_1Block {
       uint16_t d;      // FP16 scale factor
       int16_t sum_qs;  // Pre-computed sum for VNNI compensation
       int8_t qs[32];   // 32 quantized int8 values
   };
   ```
   - Already used for activation quantization in GEMM kernels
   - 36 bytes per 32 elements = 1.125 bytes/element vs 4 bytes/element for FP32 (3.5x compression)

### Current Data Flow (FP32 Residuals)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Per-Layer Flow (Attention Block)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  current_hidden (FP32)                                                      │
│         │                                                                   │
│         ├──────────────────────────────────┐                                │
│         │                                  │                                │
│         ▼                                  ▼                                │
│  save_residual()                      RMSNorm                               │
│  (memcpy FP32→FP32)                       │                                 │
│         │                                  ▼                                │
│         │                             Q/K/V GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│  residual buffer (FP32)               Attention                             │
│         │                                  │                                │
│         │                                  ▼                                │
│         │                            Output GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│         └────────────────────────────► add_residual()                       │
│                                       (FP32 + FP32)                         │
│                                            │                                │
│                                            ▼                                │
│                                    current_hidden (FP32)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Memory Traffic per layer (d_model=896, seq_len=512):
- save_residual: 896 × 512 × 4 = 1.75 MB write
- add_residual:  896 × 512 × 4 = 1.75 MB read (residual)
                 896 × 512 × 4 = 1.75 MB read (input)
                 896 × 512 × 4 = 1.75 MB write (output)
Total: 7 MB per layer × 24 layers = 168 MB per forward pass (just residuals!)
```

### Proposed Data Flow (Q8_1 Residuals)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Per-Layer Flow (Attention Block) - Q8_1 Residuals                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  current_hidden (FP32)                                                      │
│         │                                                                   │
│         ├──────────────────────────────────┐                                │
│         │                                  │                                │
│         ▼                                  ▼                                │
│  quantize_to_residual()               RMSNorm                               │
│  (FP32→Q8_1 in registers)                 │                                 │
│         │                                  ▼                                │
│         │                             Q/K/V GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│  residual buffer (Q8_1)               Attention                             │
│         │                                  │                                │
│         │                                  ▼                                │
│         │                            Output GEMM                            │
│         │                                  │                                │
│         │                                  ▼                                │
│         └────────────────────────────► add_residual_q8()                    │
│                                       (dequant Q8_1 + FP32)                 │
│                                            │                                │
│                                            ▼                                │
│                                    current_hidden (FP32)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Memory Traffic per layer (d_model=896, seq_len=512):
- save_residual: 896 × 512 × 1.125 = 0.5 MB write (Q8_1)
- add_residual:  896 × 512 × 1.125 = 0.5 MB read (residual Q8_1)
                 896 × 512 × 4 = 1.75 MB read (input FP32)
                 896 × 512 × 4 = 1.75 MB write (output FP32)
Total: 4.5 MB per layer × 24 layers = 108 MB per forward pass

Bandwidth Reduction: 168 MB → 108 MB = 36% reduction in residual traffic
```

## Implementation Progress

### Phase 1: Infrastructure ✅ COMPLETE (Dec 2-3, 2025)

- [x] Updated `ActivationPrecision` enum: replaced INT32 with Q8_1
- [x] Created `detail::PrecisionMetadata<ActivationPrecision>` traits in `CPURMSNormTypedKernel.h`
- [x] SIMD-optimized conversion functions in `SIMDHelpers.h`:
  - `convert_bf16_to_fp32()`, `convert_fp32_to_bf16()`
  - `convert_fp16_to_fp32()`, `convert_fp32_to_fp16()`
  - `dequantize_q8_1_to_fp32()`, `quantize_fp32_to_q8_1_blocks()`
  - `fused_bf16_residual_add()`, `fused_fp16_residual_add()`, `fused_q8_1_residual_add()`
- [x] All conversions have AVX512, AVX2, and scalar fallback paths

**Design Decision**: ResidualTraits template class was **removed** from the design. Instead:
- Storage types come from `detail::PrecisionMetadata<Precision>::StorageType`
- Conversion primitives live directly in `simd::` namespace in `SIMDHelpers.h`
- This is simpler and avoids an extra abstraction layer

### Phase 2: Typed Kernel Implementations 🔄 IN PROGRESS

**Completed:**
- [x] `CPURMSNormTypedKernel<ActivationPrecision>` ✅ (18 unit tests passing)
  - `apply_typed()` - accepts typed input/output, computes in FP32, fused dequant/requant
  - `apply_with_residual_add()` - fused: dequant(residual) + fp32_input → RMSNorm → quantize(output)
  - Specializations: FP32, BF16, FP16, Q8_1
  - Row-by-row processing keeps scratch in L1/L2 cache
  - Stack allocation for d_model ≤ 4096, heap fallback for larger

**Remaining kernels to implement with typed pattern:**
- [ ] `CPUSwiGLUTypedKernel<ActivationPrecision>` - SwiGLU activation with fused precision conversion
- [ ] `CPURoPETypedKernel<ActivationPrecision>` - RoPE with fused precision conversion
- [ ] `CPUSoftmaxTypedKernel<ActivationPrecision>` - Softmax (attention scores) with typed I/O
- [ ] `CpuAttentionTypedKernel<ActivationPrecision>` - Full attention block with typed K/V cache

### Phase 3: Pipeline Integration 📋 NOT STARTED

- [ ] Update `ActivationBuffers` to allocate typed residual buffers
- [ ] Modify pipeline ops (RMSNormOp, SwiGLUOp, etc.) to use typed kernels
- [ ] Config-driven kernel instantiation based on `ActivationPrecision`

### Phase 4: Typed KV Cache 📋 NOT STARTED

- [ ] KV cache precision follows `ActivationPrecision` config
- [ ] Modified K/V storage: store in configured precision after projection
- [ ] Modified K/V retrieval: dequant before attention computation
- [ ] Performance validation: measure cache hit latency impact

---

## Implementation Phases (Detailed Design)

### Phase 1: Infrastructure ✅ COMPLETE

All infrastructure is in place. The key components are:

#### 1.1 PrecisionMetadata Traits (in CPURMSNormTypedKernel.h)

```cpp
namespace detail {
    template <ActivationPrecision Precision>
    struct PrecisionMetadata;

    template<> struct PrecisionMetadata<ActivationPrecision::FP32> {
        using StorageType = float;
        static constexpr const char* name = "FP32";
        static constexpr float compression_ratio = 1.0f;
    };
    
    template<> struct PrecisionMetadata<ActivationPrecision::BF16> {
        using StorageType = uint16_t;
        static constexpr const char* name = "BF16";
        static constexpr float compression_ratio = 2.0f;
    };
    
    template<> struct PrecisionMetadata<ActivationPrecision::FP16> {
        using StorageType = uint16_t;
        static constexpr const char* name = "FP16";
        static constexpr float compression_ratio = 2.0f;
    };
    
    template<> struct PrecisionMetadata<ActivationPrecision::Q8_1> {
        using StorageType = Q8_1Block;
        static constexpr const char* name = "Q8_1";
        static constexpr float compression_ratio = 3.5f;
    };
}
```

#### 1.2 SIMD Conversion Functions (in SIMDHelpers.h)

All conversion functions are implemented with AVX512 → AVX2 → scalar tiered fallback:

```cpp
namespace simd {
    // BF16 ↔ FP32
    void convert_bf16_to_fp32(const uint16_t* src, float* dst, size_t count);
    void convert_fp32_to_bf16(const float* src, uint16_t* dst, size_t count);
    
    // FP16 ↔ FP32
    void convert_fp16_to_fp32(const uint16_t* src, float* dst, size_t count);
    void convert_fp32_to_fp16(const float* src, uint16_t* dst, size_t count);
    
    // Q8_1 ↔ FP32
    void dequantize_q8_1_to_fp32(const Q8_1Block* src, float* dst, size_t count);
    void quantize_fp32_to_q8_1_blocks(const float* src, Q8_1Block* dst, size_t count);
    
    // Fused residual operations (dequant + add in single SIMD pass)
    void fused_fp32_residual_add(const float* res, const float* in, float* out, size_t n);
    void fused_bf16_residual_add(const uint16_t* res, const float* in, float* out, size_t n);
    void fused_fp16_residual_add(const uint16_t* res, const float* in, float* out, size_t n);
    void fused_q8_1_residual_add(const Q8_1Block* res, const float* in, float* out, size_t n);
}
```

### Phase 2: Typed Kernel Implementations (Template Pattern)

#### 2.1 Canonical Pattern: CPURMSNormTypedKernel

This is the reference implementation. All other typed kernels should follow this pattern:

```cpp
template <ActivationPrecision Precision>
class CPURMSNormTypedKernel : public CPUKernelBase {
public:
    using Metadata = detail::PrecisionMetadata<Precision>;
    using StorageType = typename Metadata::StorageType;

    // Primary typed operation
    bool apply_typed(
        const StorageType* input,    // Typed input buffer
        const float* gamma,          // Weights always FP32
        StorageType* output,         // Typed output buffer
        int rows, int cols,
        float epsilon = 1e-6f,
        int device_idx = -1);

    // Fused residual variant
    bool apply_with_residual_add(
        const StorageType* residual, // Typed residual from previous layer
        const float* fp32_input,     // FP32 input to add (e.g., projection output)
        const float* gamma,          // Weights always FP32
        StorageType* output,         // Typed output buffer
        int rows, int cols,
        float epsilon = 1e-6f,
        int device_idx = -1);

    static constexpr ActivationPrecision precision() { return Precision; }
    static const char* precision_name() { return Metadata::name; }
    static constexpr float compression_ratio() { return Metadata::compression_ratio; }
};
```

**Implementation pattern (BF16 example):**

```cpp
bool CPURMSNormTypedKernel<ActivationPrecision::BF16>::apply_typed(
    const uint16_t* input, const float* gamma, uint16_t* output,
    int rows, int cols, float epsilon, int device_idx)
{
    const size_t ucols = static_cast<size_t>(cols);

    #pragma omp parallel
    {
        // Thread-local scratch (stack for small, heap for large)
        std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
        std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
        float* fp32_in = (ucols <= MAX_STACK_ROW_SIZE) ? stack_fp32_in.data() : heap_alloc();
        float* fp32_out = (ucols <= MAX_STACK_ROW_SIZE) ? stack_fp32_out.data() : heap_alloc();

        #pragma omp for
        for (int row = 0; row < rows; ++row) {
            const uint16_t* in_row = input + row * ucols;
            uint16_t* out_row = output + row * ucols;

            // 1. Dequantize: BF16 → FP32 (one row, stays in L1)
            simd::convert_bf16_to_fp32(in_row, fp32_in, ucols);

            // 2. Compute: RMSNorm in FP32
            primitives::rmsnorm_fused_row_avx512(fp32_in, gamma, fp32_out, ucols, epsilon);

            // 3. Quantize: FP32 → BF16 (one row)
            simd::convert_fp32_to_bf16(fp32_out, out_row, ucols);
        }
    }
    return true;
}
```

#### 2.2 Remaining Kernels to Implement

Apply the same pattern to these kernels:

| Kernel | Input | Output | Notes |
|--------|-------|--------|-------|
| `CPUSwiGLUTypedKernel` | gate (typed), up (typed) | output (typed) | SwiGLU: gate * silu(up) |
| `CPURoPETypedKernel` | Q/K (typed) | Q/K (typed, in-place) | Rotary position embedding |
| `CPUSoftmaxTypedKernel` | scores (typed) | probs (typed) | Attention softmax |
| `CpuAttentionTypedKernel` | Q (typed), K_cache (typed), V_cache (typed) | output (typed) | Full GQA attention |

### Phase 6: Typed KV Cache (4-6 days)

The KV cache stores K/V tensors for all past tokens across all layers. For long-context inference, this dominates memory usage:

**Memory Formula**: `2 × n_layers × max_seq_len × n_kv_heads × head_dim × bytes_per_element`

| Model | Layers | KV Heads | Head Dim | 4K FP32 | 4K Q8_1 | 32K FP32 | 32K Q8_1 |
|-------|--------|----------|----------|---------|---------|----------|----------|
| Qwen2.5-0.5B | 24 | 2 | 64 | 24 MB | 7 MB | 192 MB | 55 MB |
| Qwen2.5-7B | 28 | 4 | 128 | 224 MB | 64 MB | 1.8 GB | 512 MB |
| Qwen2.5-14B | 40 | 8 | 128 | 640 MB | 183 MB | 5 GB | 1.4 GB |
| Qwen2.5-72B | 80 | 8 | 128 | 1.3 GB | 366 MB | 10 GB | 2.9 GB |

#### 6.1 KV Cache Uses ActivationPrecision (No Separate Enum)

**Design Decision**: KV cache format follows `ActivationPrecision` - no separate `KVCacheFormat` enum needed.

**Rationale**: K and V tensors are outputs from projection GEMMs, which produce activation tensors. If `ActivationPrecision::BF16`, then K/V projections produce BF16 tensors, which get stored in the KV cache as BF16. There's no sensible use case for "FP32 activations but Q8_1 KV cache" - the K/V values come from activation tensors!

Similarly, residuals now also use `ActivationPrecision` for the same reason - they are activation values being stored, and the kernel black-box model means we only need to specify the DRAM storage format.

```cpp
// KV cache format is derived from activation_precision
// NO new enum needed - just use ActivationPrecision

// In KVCache constructor:
KVCache(const MPIContext& mpi_ctx, int n_layers, int max_seq_len,
        int n_kv_heads, int head_dim, int device_idx,
        ActivationPrecision precision = ActivationPrecision::FP32);
```

#### 6.2 Create Typed KV Cache Entry

```cpp
// src/v2/tensors/KVCache.h

/**
 * @brief Storage format traits for KV cache (reuses ActivationPrecision)
 */
template<ActivationPrecision Precision>
struct KVCacheTraits;

template<>
struct KVCacheTraits<ActivationPrecision::FP32> {
    using StorageType = float;
    static constexpr size_t bytes_per_element = 4;
    static constexpr bool needs_conversion = false;
};

template<>
struct KVCacheTraits<ActivationPrecision::BF16> {
    using StorageType = uint16_t;
    static constexpr size_t bytes_per_element = 2;
    static constexpr bool needs_conversion = true;
};

template<>
struct KVCacheTraits<ActivationPrecision::FP16> {
    using StorageType = uint16_t;
    static constexpr size_t bytes_per_element = 2;
    static constexpr bool needs_conversion = true;
};

template<>
struct KVCacheTraits<ActivationPrecision::Q8_1> {
    using StorageType = Q8_1Block;
    static constexpr size_t bytes_per_element = 36.0f / 32.0f;  // 1.125
    static constexpr bool needs_conversion = true;
    static constexpr size_t block_size = 32;
};

/**
 * @brief Typed per-layer KV cache entry
 *
 * Stores K/V in the same format as activations.
 */
struct TypedKVCacheEntry
{
    void* K_storage = nullptr;  // Raw storage (format-dependent)
    void* V_storage = nullptr;
    size_t storage_bytes_k = 0;
    size_t storage_bytes_v = 0;
    int cached_tokens = 0;
    int device_idx = -1;
    ActivationPrecision precision = ActivationPrecision::FP32;
    
    ~TypedKVCacheEntry();
};
```

#### 6.3 KV Cache Interface Updates

```cpp
// KVCache.h - Updated interface

class KVCache {
public:
    /**
     * @brief Construct KV cache with specified storage precision
     *
     * @param precision Storage format matches activation precision
     */
    KVCache(const MPIContext& mpi_ctx, int n_layers, int max_seq_len,
            int n_kv_heads, int head_dim, int device_idx,
            ActivationPrecision precision = ActivationPrecision::FP32);
    
    /**
     * @brief Append K/V tensors (format matches activation precision)
     *
     * Input format depends on activation_precision setting.
     * - FP32: float* input
     * - BF16: uint16_t* input  
     * - Q8_1: Q8_1Block* input
     */
    bool append_kv(int layer, const void* new_k, const void* new_v, int new_seq_len);
    
    /**
     * @brief Get raw K/V storage for direct consumption by typed kernels
     */
    const void* get_k_raw(int layer) const;
    const void* get_v_raw(int layer) const;
    
    /**
     * @brief Get storage precision
     */
    ActivationPrecision precision() const { return precision_; }
    
    /**
     * @brief Get memory usage in bytes
     */
    size_t memory_usage_bytes() const;

private:
    ActivationPrecision precision_;
    // ...
};
```

#### 6.4 Fused K/V Projection with KV Cache Write

The key optimization is fusing K/V projection output with cache append:

```cpp
// In attention kernel or orchestrator

// K/V projection outputs are already in the correct activation format
// For BF16 activations, K_proj is BF16, so append directly:
kv_cache_->append_kv(layer_idx, K_proj->data(), V_proj->data(), effective_seq_len);

// No format conversion needed - K/V tensors match KV cache format
```

#### 6.5 Attention Kernel with Typed KV

Attention kernels need to handle typed K/V inputs:

```cpp
// GQAAttention or similar

class IAttentionKernel {
public:
    // Attention with typed K/V cache (format matches activation precision)
    virtual bool compute_with_typed_kv(
        const void* Q,               // Query (activation format)
        const void* K_cache,         // K cache (same format)
        const void* V_cache,         // V cache (same format)
        ActivationPrecision precision,
        int cached_seq_len,
        void* output,                // Output (activation format)
        const AttentionConfig& config
    ) = 0;
};
```

For Q8_1 activations, the attention kernel can either:
1. **Dequantize on-the-fly** during Q·K^T computation (bandwidth-efficient)
2. **Pre-dequantize** to temporary FP32 buffer (simpler implementation)

Option 1 is preferred for long contexts:

```cpp
// Q·K^T with on-the-fly Q8_1 dequantization
// K is stored as Q8_1 blocks: [cached_seq, n_kv_heads * head_dim]
// Need to compute: Q[q, :] · K[k, :]^T for all (q, k) pairs

void qk_dot_q8_cache(
    const float* Q,          // [seq_q, head_dim]
    const Q8_1Block* K_q8,   // [cached_seq, kv_dim / 32] blocks
    float* scores,           // [seq_q, cached_seq]
    int seq_q, int cached_seq, int head_dim)
{
    // For each query position
    for (int q = 0; q < seq_q; ++q) {
        const float* Q_row = Q + q * head_dim;
        
        // For each cached key position
        for (int k = 0; k < cached_seq; ++k) {
            float dot = 0.0f;
            
            // Accumulate over head_dim (32 elements at a time)
            int n_blocks = (head_dim + 31) / 32;
            for (int blk = 0; blk < n_blocks; ++blk) {
                const Q8_1Block& block = K_q8[k * n_blocks + blk];
                float d = fp16_to_fp32(block.d);
                
                // Dequant + dot in one pass
                for (int j = 0; j < 32 && blk * 32 + j < head_dim; ++j) {
                    float K_val = block.qs[j] * d;
                    dot += Q_row[blk * 32 + j] * K_val;
                }
            }
            
            scores[q * cached_seq + k] = dot;
        }
    }
}
```

#### 6.6 Memory Layout Considerations

For Q8_1 format, the layout must be attention-friendly:

```
Option A: Per-token blocks (contiguous K/V per position)
K[token=0]: [block_0, block_1, ..., block_N]  (N = head_dim / 32)
K[token=1]: [block_0, block_1, ..., block_N]
...

Option B: Per-dimension blocks (contiguous dimension across tokens)
K[dim_block=0]: [token_0, token_1, ..., token_T]
K[dim_block=1]: [token_0, token_1, ..., token_T]
...
```

Option A (per-token) is preferred because:
- Cache-friendly for sequential key access during Q·K^T
- Append is simple (just add blocks at end)
- Natural for position-based eviction

### Phase 7: Testing & Validation (2-3 days)

#### 6.1 Unit Tests

```cpp
// tests/v2/unit/Test__ResidualBuffer.cpp

TEST(ResidualBuffer, Q8_1_RoundTrip) {
    // Test quantize → dequant preserves values within tolerance
    std::vector<float> original(1024);
    // Fill with random values
    
    auto buffer = createResidualBuffer(ActivationPrecision::Q8_1, 1024);
    buffer->save(original.data(), 1024);
    
    std::vector<float> zeros(1024, 0.0f);
    std::vector<float> recovered(1024);
    buffer->add_to(zeros.data(), recovered.data(), 1024);
    
    // Check max error
    float max_err = 0.0f;
    for (size_t i = 0; i < 1024; ++i) {
        max_err = std::max(max_err, std::abs(original[i] - recovered[i]));
    }
    
    // Q8_1 should have <1% error for typical activation ranges
    float max_abs = *std::max_element(original.begin(), original.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); });
    EXPECT_LT(max_err / max_abs, 0.01f);
}
```

#### 6.2 E2E Parity Tests

```cpp
// tests/v2/e2e/Test__TypedResidualParity.cpp

TEST(TypedResidualParity, Q8_1_vs_FP32_Logits) {
    // Load same model twice with different activation precisions
    PipelineConfig config_fp32;
    config_fp32.activation_precision = ActivationPrecision::FP32;
    
    PipelineConfig config_q8;
    config_q8.activation_precision = ActivationPrecision::Q8_1;
    
    auto pipeline_fp32 = createPipeline(config_fp32);
    auto pipeline_q8 = createPipeline(config_q8);
    
    // Run same input through both
    std::vector<int> tokens = {785, 3974, 13876};  // "The quick brown"
    
    pipeline_fp32->forward(tokens.data(), tokens.size());
    pipeline_q8->forward(tokens.data(), tokens.size());
    
    // Compare logits
    const float* logits_fp32 = pipeline_fp32->logits();
    const float* logits_q8 = pipeline_q8->logits();
    
    // Top-1 should match
    int top1_fp32 = std::max_element(logits_fp32, logits_fp32 + vocab_size) - logits_fp32;
    int top1_q8 = std::max_element(logits_q8, logits_q8 + vocab_size) - logits_q8;
    EXPECT_EQ(top1_fp32, top1_q8);
    
    // Cosine similarity should be high
    float cos_sim = cosine_similarity(logits_fp32, logits_q8, vocab_size);
    EXPECT_GT(cos_sim, 0.999f);
}
```

#### 6.3 Performance Benchmarks

```cpp
// tests/v2/performance/Perf__TypedResidual.cpp

TEST(TypedResidualPerf, BandwidthComparison) {
    const int seq_len = 512;
    const int d_model = 896;
    const int n_iterations = 1000;
    
    // Benchmark FP32 residual
    auto buffer_fp32 = createResidualBuffer(ActivationPrecision::FP32, seq_len * d_model);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iterations; ++i) {
        buffer_fp32->save(input.data(), seq_len * d_model);
        buffer_fp32->add_to(input.data(), output.data(), seq_len * d_model);
    }
    auto fp32_time = elapsed(start);
    
    // Benchmark Q8_1 residual
    auto buffer_q8 = createResidualBuffer(ActivationPrecision::Q8_1, seq_len * d_model);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iterations; ++i) {
        buffer_q8->save(input.data(), seq_len * d_model);
        buffer_q8->add_to(input.data(), output.data(), seq_len * d_model);
    }
    auto q8_time = elapsed(start);
    
    // Report
    std::cout << "FP32 residual: " << fp32_time << " ms\n";
    std::cout << "Q8_1 residual: " << q8_time << " ms\n";
    std::cout << "Speedup: " << fp32_time / q8_time << "x\n";
}
```

### Phase 8: Documentation & CLI (1 day)

#### 8.1 CLI Arguments

```bash
# ArgParser.h
--activation-precision <prec> Activation/residual/KV cache precision: fp32, bf16, fp16, q8_1 (default: fp32)
                              Controls all intermediate buffer formats (unified design)

# Example usage
./run_llaminar.sh -- -m model.gguf --activation-precision bf16 --benchmark
# This gives: BF16 activations, BF16 residuals, BF16 KV cache (all unified)
```

#### 8.2 Documentation Update

Update `.github/copilot-instructions.md` with:
- Updated `--activation-precision` CLI option (now affects KV cache too)
- New `--residual-format` CLI option  
- Performance characteristics of each format
- Accuracy/performance tradeoffs
- Memory usage calculator for different configurations

## File List

### Implemented Files ✅

| File | Purpose |
|------|---------|
| `src/v2/kernels/cpu/ops/CPURMSNormTypedKernel.h` | Typed RMSNorm kernel with `detail::PrecisionMetadata` traits |
| `src/v2/kernels/cpu/ops/CPURMSNormTypedKernel.cpp` | Implementation for FP32, BF16, FP16, Q8_1 specializations |
| `src/v2/tensors/SIMDHelpers.h` | SIMD conversion functions (AVX512/AVX2/scalar) |
| `src/v2/pipelines/PipelineConfig.h` | `ActivationPrecision` enum with Q8_1 |
| `tests/v2/unit/Test__CPURMSNormTypedKernel.cpp` | Unit tests (18 passing) |

### Files to Create (Phase 2 Remaining Kernels)

| File | Purpose |
|------|---------|
| `src/v2/kernels/cpu/ops/CPUSwiGLUTypedKernel.h` | Typed SwiGLU kernel |
| `src/v2/kernels/cpu/ops/CPUSwiGLUTypedKernel.cpp` | Implementation |
| `src/v2/kernels/cpu/ops/CPURoPETypedKernel.h` | Typed RoPE kernel |
| `src/v2/kernels/cpu/ops/CPURoPETypedKernel.cpp` | Implementation |
| `src/v2/kernels/cpu/ops/CPUSoftmaxTypedKernel.h` | Typed Softmax kernel |
| `src/v2/kernels/cpu/ops/CPUSoftmaxTypedKernel.cpp` | Implementation |
| `tests/v2/unit/Test__CPUSwiGLUTypedKernel.cpp` | Unit tests |
| `tests/v2/unit/Test__CPURoPETypedKernel.cpp` | Unit tests |
| `tests/v2/unit/Test__CPUSoftmaxTypedKernel.cpp` | Unit tests |

### Files to Modify (Pipeline Integration)

| File | Changes |
|------|---------|
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Use typed kernels based on config |
| `src/v2/tensors/KVCache.h` | Add typed storage support |
| `src/v2/tensors/KVCache.cpp` | Implement typed K/V storage |
| `src/v2/kernels/cpu/attention/GQAAttention.cpp` | Support typed K/V cache |

### Removed from Design

| File | Reason |
|------|--------|
| `src/v2/kernels/cpu/primitives/ResidualTraits.h` | Replaced by `detail::PrecisionMetadata` in kernel headers |
| `src/v2/kernels/cpu/primitives/ResidualTraits.cpp` | Not needed - SIMD functions in `SIMDHelpers.h` |
| `src/v2/tensors/ResidualBuffer.h` | Not needed - typed kernels handle conversion directly |

## Timeline Estimate

| Phase | Duration | Dependencies | Status |
|-------|----------|--------------|--------|
| Phase 1: Infrastructure | 2-3 days | None | ✅ COMPLETE |
| Phase 2: Typed Kernels | 4-6 days | Phase 1 | 🔄 IN PROGRESS (RMSNorm done) |
| Phase 3: Pipeline Integration | 2-3 days | Phase 2 | 📋 NOT STARTED |
| Phase 4: Typed KV Cache | 3-4 days | Phase 3 | 📋 NOT STARTED |
| Phase 5: Testing & Validation | 2-3 days | Phase 4 | 📋 NOT STARTED |
| Phase 6: Documentation & CLI | 1 day | Phase 5 | 📋 NOT STARTED |

**Total**: ~14-20 days (3-4 weeks)

## Risk Assessment

### High Risk
- **Accuracy degradation (residuals)**: Q8_1 has limited precision. May cause accuracy issues in deeper layers where residuals accumulate quantization noise.
  - Mitigation: E2E parity tests, configurable per-layer residual format

- **Accuracy degradation (KV cache)**: Quantized K/V affects attention scores directly.
  - Mitigation: BF16/FP16 may be safer than Q8_1 for KV cache; test thoroughly

### Medium Risk
- **Performance not as expected**: Memory bandwidth savings may be offset by quantization compute overhead
  - Mitigation: Profile both paths, ensure quantization is SIMD-optimized, fuse operations

- **KV cache append overhead**: Q8_1 quantization during decode adds latency
  - Mitigation: Use BF16/FP16 for KV if Q8_1 decode latency is unacceptable

### Low Risk
- **Configuration simplicity**: Single `ActivationPrecision` enum controls all buffer formats
  - Benefit: Eliminates confusion about separate format enums

## Success Criteria

### Residual Typed Storage
1. **Correctness**: Q8_1 residuals produce same top-1 token as FP32 on test prompts
2. **Performance**: ≥20% reduction in residual memory traffic on benchmark
3. **Stability**: No accuracy degradation on 1000-token generation tests

### KV Cache Typed Storage
4. **Correctness**: Typed KV cache produces same top-1 token as FP32 on test prompts
5. **Performance**: ≥30% reduction in KV cache memory usage for long contexts
6. **Scalability**: Enable 32K+ context on memory-constrained systems

### General
7. **Compatibility**: Existing flags continue to work unchanged
8. **Documentation**: Clear guidance on when to use each format

## Appendix A: Memory Bandwidth Calculations

### Per-Layer Memory Traffic (Qwen2.5-0.5B, d_model=896, seq_len=512)

| Operation | FP32 Traffic | Q8_1 Traffic | Savings |
|-----------|-------------|--------------|---------|
| save_residual (attn) | 1.75 MB | 0.5 MB | 71% |
| add_residual (attn) | 3.5 MB | 2.25 MB | 36% |
| save_residual (ffn) | 1.75 MB | 0.5 MB | 71% |
| add_residual (ffn) | 3.5 MB | 2.25 MB | 36% |
| **Total per layer** | **10.5 MB** | **5.5 MB** | **48%** |

### Full Forward Pass (24 layers)

| Format | Total Residual Traffic | vs FP32 |
|--------|----------------------|---------|
| FP32 | 252 MB | baseline |
| Q8_1 | 132 MB | -48% |
| BF16/FP16 | 126 MB | -50% |

## Appendix B: Quantization Error Analysis

### Q8_1 Block Properties
- Scale: FP16 (16 bits)
- Values: INT8 × 32
- Range: -127 to +127 scaled by FP16 scale

### Expected Error
- Relative error: ≤ 1/127 ≈ 0.8% per quantization
- Accumulated over 24 layers: ~10-20% drift in residual stream
- Acceptable because: attention mechanism re-normalizes at each layer (RMSNorm)

### When to Avoid Q8_1 Residuals
- Very deep models (>100 layers)
- Tasks requiring high numerical precision
- When using INT8 weight quantization (compounds error)

## Appendix C: KV Cache Memory Calculations

### KV Cache Formula

```
KV Cache Size = 2 × n_layers × max_seq_len × n_kv_heads × head_dim × bytes_per_element
              = 2 × n_layers × max_seq_len × kv_dim × bytes_per_element

where kv_dim = n_kv_heads × head_dim
```

### KV Cache Size by Model and Format

| Model | Layers | KV Dim | 4K FP32 | 4K BF16 | 4K Q8_1 | 32K FP32 | 32K Q8_1 |
|-------|--------|--------|---------|---------|---------|----------|----------|
| Qwen2.5-0.5B | 24 | 128 | 24 MB | 12 MB | 7 MB | 192 MB | 55 MB |
| Qwen2.5-3B | 36 | 512 | 144 MB | 72 MB | 41 MB | 1.15 GB | 330 MB |
| Qwen2.5-7B | 28 | 512 | 112 MB | 56 MB | 32 MB | 896 MB | 256 MB |
| Qwen2.5-14B | 40 | 1024 | 320 MB | 160 MB | 91 MB | 2.5 GB | 730 MB |
| Qwen2.5-32B | 64 | 1024 | 512 MB | 256 MB | 146 MB | 4 GB | 1.1 GB |
| Qwen2.5-72B | 80 | 1024 | 640 MB | 320 MB | 183 MB | 5 GB | 1.4 GB |

### Per-Token KV Memory

| Format | Bytes per Token per Layer | Compression |
|--------|--------------------------|-------------|
| FP32 | 8 × kv_dim | 1× (baseline) |
| BF16/FP16 | 4 × kv_dim | 2× |
| Q8_1 | ~2.25 × kv_dim | ~3.5× |

### KV Cache Bandwidth During Attention

For each decode step, the full KV cache is read during Q·K^T and Score·V:

```
Bandwidth per decode step = 2 × cached_seq × kv_dim × bytes_per_element × n_layers
```

| Context | Layers | KV Dim | FP32 Read | Q8_1 Read | Savings |
|---------|--------|--------|-----------|-----------|---------|
| 4K | 28 | 512 | 112 MB | 32 MB | 71% |
| 16K | 28 | 512 | 448 MB | 128 MB | 71% |
| 32K | 28 | 512 | 896 MB | 256 MB | 71% |

This bandwidth is **per decode step**, making typed KV cache critical for long-context inference.

### Recommended Formats by Use Case

| Use Case | Residual Format | KV Cache Format | Notes |
|----------|-----------------|-----------------|-------|
| Maximum accuracy | FP32 | FP32 | Baseline, most memory |
| Balanced (recommended) | BF16 | BF16 | Good accuracy, 2× compression |
| Long context (16K+) | BF16 | Q8_1 | Memory-efficient for large contexts |
| Extreme compression | Q8_1 | Q8_1 | Test accuracy carefully |
| Intel AMX optimized | BF16 | BF16 | Native BF16 compute |
