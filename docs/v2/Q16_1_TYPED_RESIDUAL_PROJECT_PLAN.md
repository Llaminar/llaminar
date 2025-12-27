# Q16_1 Typed Residual Stream Project Plan

**Author**: David Sanftenberg  
**Date**: December 26, 2025  
**Branch**: `feature/typed-residuals`  
**Status**: In Progress

## Progress Summary

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Foundation | ✅ Complete | Enum, TensorFactory, HybridBufferType, config |
| Phase 2: Buffer Allocation | ✅ Complete | GraphOrchestrator Q16_1/Q8_1 buffer allocation |
| Phase 3: Stage Implementations | ✅ Complete | ResidualAddStage, EmbeddingStage Q16_1 support |
| Phase 3.5: Q16_1 RoPE | ✅ Complete | In-place Q16_1 RoPE primitives and kernel |
| Phase 4: RMSNorm Q16_1 Support | ✅ Complete | Q16_1 input → FP32 output normalization |
| Phase 5: JIT Fusion | ✅ Complete | Register-tiled fused Wo + Q16_1 residual kernel |
| Phase 6: Integration Wiring | ✅ Complete | CLI flag, stage params, graph wiring |

## Executive Summary

This project introduces a **Q16_1 typed residual stream** for the Q8_1/Hybrid activation precision modes. The residual stream is the numerical backbone of transformer inference—accumulated across all layers. Using Q16_1 (int16 quantization with FP32 scale) provides **266× better precision than Q8_1** while using only **56% more storage** (2.25 bytes/elem vs 1.125 bytes/elem).

The final phase fuses all operations from attention output through residual addition into the JIT kernel, performing the entire sequence in registers without intermediate memory traffic.

---

## Table of Contents

1. [Background and Motivation](#background-and-motivation)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Foundation](#phase-1-foundation) ✅
4. [Phase 2: Buffer Allocation](#phase-2-buffer-allocation) ✅
5. [Phase 3: Stage Implementations](#phase-3-stage-implementations) ✅
6. [Phase 3.5: Q16_1 RoPE](#phase-35-q16_1-rope) ✅
7. [Phase 4: RMSNorm Q16_1 Support](#phase-4-rmsnorm-q16_1-support) ✅
8. [Phase 5: JIT Fusion](#phase-5-jit-fusion) ✅
9. [Phase 6: Integration Wiring](#phase-6-integration-wiring-next) ✅
10. [Validation Plan](#validation-plan)
11. [Performance Targets](#performance-targets)
12. [Risk Assessment](#risk-assessment)
13. [Timeline](#timeline)

---

## Background and Motivation

### The Residual Stream Problem

In transformer inference, the residual stream accumulates contributions from every attention and FFN layer:

```
residual₀ = embedding(tokens)
for layer in layers:
    residual = residual + attention(residual)
    residual = residual + ffn(residual)
output = lm_head(residual)
```

With Q8_1 activations, quantization error accumulates across layers. For a 24-layer model, this means ~48 quantization operations on the residual stream, leading to significant numerical drift.

### Why Q16_1?

| Format | Storage | Precision | Relative Error |
|--------|---------|-----------|----------------|
| FP32 | 4.0 bytes/elem | Full | Baseline |
| Q8_1 | 1.125 bytes/elem | ±127 levels | ~0.4% |
| Q16_1 | 2.25 bytes/elem | ±32767 levels | ~0.0015% |

Q16_1 provides **256× finer quantization** than Q8_1 with only 2× the storage. The FP32 scale (vs FP16 in Q8_1) eliminates scale quantization error entirely.

### Measured Precision (from unit tests)

```
[Precision Comparison] Q16_1 relative L2 error: 0.00143964%
[Precision Comparison] Q8_1 relative L2 error: 0.383245%
[Precision Comparison] Q16_1 is 266.209× better than Q8_1
```

---

## Architecture Overview

### Current Flow (Hybrid Mode)

```
Embedding ──► FP32 residual
              │
    ┌─────────▼─────────┐
    │    RMSNorm        │ FP32 → FP32
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   QKV GEMM        │ FP32 × weights → Q8_1
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   RoPE            │ Q8_1 → FP32 (avoid requant)
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Attention       │ FP32 → FP32 context
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Wo GEMM         │ FP32 × weights → FP32
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  Residual Add     │ FP32 += FP32
    └─────────┬─────────┘
              │
           (repeat for FFN)
```

### Proposed Flow (Q16_1 Residual)

```
Embedding ──► FP32 ──► Q16_1 residual
                        │
    ┌───────────────────▼───────────────────┐
    │    RMSNorm (Q16_1 input)              │ Q16_1 → FP32
    └───────────────────┬───────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │   QKV GEMM                            │ FP32 × weights → Q8_1
    └───────────────────┬───────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │   RoPE                                │ Q8_1 → FP32
    └───────────────────┬───────────────────┘
                        │
    ┌───────────────────▼───────────────────┐
    │   Fused Attention + Wo + Residual     │ FP32 → Q16_1 residual
    │   (JIT kernel, all in registers)      │    (Phase 5)
    └───────────────────┬───────────────────┘
                        │
                   (FFN path similar)
```

### Memory Layout Comparison

| Buffer | Current (Hybrid) | Proposed (Q16_1 Residual) |
|--------|------------------|---------------------------|
| residual | FP32 (4 B/elem) | Q16_1 (2.25 B/elem) |
| attn_proj | FP32 (4 B/elem) | Q8_1 (1.125 B/elem) |
| ffn_output | FP32 (4 B/elem) | Q8_1 (1.125 B/elem) |

**Qwen2-0.5B (d_model=896), batch=1, seq=2048:**
- Current: 7.3 MB residual + 14.6 MB proj buffers = 21.9 MB
- Proposed: 4.1 MB residual + 4.1 MB proj buffers = 8.2 MB
- **Savings: 62%**

---

## Phase 1: Foundation

### 1.1 Add Q16_1 to ActivationPrecision Enum

**File**: `src/v2/execution/RuntimeConfig.h`

```cpp
// BEFORE (line ~140)
enum class ActivationPrecision
{
    FP32,
    BF16,
    FP16,
    Q8_1,
    Hybrid
};

// AFTER
enum class ActivationPrecision
{
    FP32,
    BF16,
    FP16,
    Q8_1,
    Q16_1,    // NEW: High-precision quantized (for residual stream)
    Hybrid,
    HybridQ16 // NEW: Hybrid with Q16_1 residual stream
};
```

**Also update**: `activationPrecisionToString()` function.

### 1.2 Add Q16_1 to TensorFactory

**File**: `src/v2/tensors/TensorFactory.h` and `TensorFactory.cpp`

```cpp
// In TensorFactory.h - add declaration:
std::unique_ptr<Q16_1Tensor> createQ16_1(const std::vector<size_t>& shape, int device_idx = -1);

// In TensorFactory.cpp - add implementation:
std::unique_ptr<Q16_1Tensor> TensorFactory::createQ16_1(const std::vector<size_t> &shape, int device_idx)
{
    if (numa_node_ >= 0)
    {
        bindToNumaNode();
    }
    return std::make_unique<Q16_1Tensor>(shape, device_idx);
}

// Update createActivation() switch:
case ActivationPrecision::Q16_1:
    return createQ16_1(shape, device_idx);
case ActivationPrecision::HybridQ16:
    // For HybridQ16 mode, createActivation still returns the appropriate type
    // Buffer allocation logic in GraphOrchestrator handles residual specially
    return createQ16_1(shape, device_idx);
```

### 1.3 Update HybridBufferType Enum

**File**: `src/v2/execution/HybridPrecisionConfig.h`

```cpp
enum class HybridBufferType
{
    // Core buffers
    Residual,           // Q16_1 in HybridQ16 mode
    Normalized,
    Hidden,
    Logits,
    
    // ... existing types ...
    
    // NEW: Explicit residual stream type
    ResidualStream,     // The accumulator buffer
};
```

### 1.4 Add HybridQ16 Configuration

**File**: `src/v2/execution/HybridPrecisionConfig.h`

```cpp
struct HybridQ16PrecisionConfig
{
    // Residual stream (the key change)
    ActivationPrecision residual_stream = ActivationPrecision::Q16_1;
    
    // Layer outputs that add to residual
    ActivationPrecision attention_output = ActivationPrecision::Q8_1;  // Changed from FP32
    ActivationPrecision ffn_down = ActivationPrecision::Q8_1;          // Changed from FP32
    
    // Everything else same as Hybrid
    ActivationPrecision qkv_gemm_output = ActivationPrecision::Q8_1;
    ActivationPrecision q_after_rope = ActivationPrecision::FP32;
    ActivationPrecision k_after_rope = ActivationPrecision::FP32;
    ActivationPrecision kv_cache = ActivationPrecision::FP32;
    ActivationPrecision attention_context = ActivationPrecision::FP32;
    ActivationPrecision ffn_gate = ActivationPrecision::Q8_1;
    ActivationPrecision ffn_up = ActivationPrecision::Q8_1;
    
    static HybridQ16PrecisionConfig defaultConfig();
};
```

---

## Phase 2: Buffer Allocation

### 2.1 Update GraphOrchestrator Buffer Allocation

**File**: `src/v2/execution/GraphOrchestrator.cpp`

**Change in `allocateModelBuffers()`** (around line 805):

```cpp
// BEFORE
// Allocate norm/residual buffers (always FP32 - high precision needed for residual stream)
state_.normalized = factory.createFP32(
    {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
    device_idx);
state_.residual = factory.createFP32(
    {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
    device_idx);

// AFTER
// Allocate norm buffer (FP32 - output of RMSNorm for GEMM input)
state_.normalized = factory.createFP32(
    {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
    device_idx);

// Allocate residual buffer based on activation precision mode
if (act_prec == ActivationPrecision::HybridQ16)
{
    LOG_INFO("[GraphOrchestrator] Using Q16_1 residual stream for HybridQ16 mode");
    state_.residual = factory.createQ16_1(
        {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
        device_idx);
}
else
{
    state_.residual = factory.createFP32(
        {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
        device_idx);
}
```

### 2.2 Update attn_proj and ffn_output Allocation

**File**: `src/v2/execution/GraphOrchestrator.cpp`

```cpp
// BEFORE (around line 858)
// attn_proj is the output of Wo projection which feeds into the residual stream
// Keep as FP32 for numerical stability in residual connections
state_.attn_proj = factory.createFP32(
    {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
    device_idx);

// AFTER
// attn_proj: Output of Wo projection, feeds into residual stream
// HybridQ16 mode: Q8_1 (native add to Q16_1 residual)
// Other modes: FP32 (numerical stability)
if (act_prec == ActivationPrecision::HybridQ16)
{
    state_.attn_proj = factory.createQ8_1(
        {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
        device_idx);
}
else
{
    state_.attn_proj = factory.createFP32(
        {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
        device_idx);
}
```

Similar change for `state_.ffn_output`.

---

## Phase 3: Stage Implementations

### 3.1 Add Q16_1 += Q8_1 to ResidualAddStage

**File**: `src/v2/execution/ComputeStage.cpp`

**Add new execute method** (after `executeQ8_1`):

```cpp
bool ResidualAddStage::executeQ16_1_Q8_1(IDeviceContext *ctx, size_t num_elements)
{
    // Mixed-type: Q8_1 delta + Q16_1 residual → Q16_1 output (in-place)
    // This is THE key operation for typed residual connections
    //
    // Note: params_.input is the delta (Q8_1), params_.residual is the accumulator (Q16_1)
    // The output is written in-place to params_.residual
    
    const auto *delta_q8 = dynamic_cast<const Q8_1Tensor *>(params_.input);
    auto *residual_q16 = dynamic_cast<Q16_1Tensor *>(params_.residual);
    
    if (!delta_q8 || !residual_q16)
    {
        LOG_ERROR("[ResidualAddStage::Q16_1_Q8_1] Failed to cast tensors");
        return false;
    }
    
    if (num_elements % 32 != 0)
    {
        LOG_ERROR("[ResidualAddStage::Q16_1_Q8_1] Element count not multiple of 32");
        return false;
    }
    
    const Q8_1Block *delta_blocks = delta_q8->q8_1_blocks();
    Q16_1Block *residual_blocks = residual_q16->mutable_q16_1_blocks();
    
    LOG_DEBUG("[ResidualAddStage::Q16_1_Q8_1] Adding " << num_elements 
              << " elements (" << (num_elements / 32) << " blocks)");
    
    // Use SIMD-optimized Q16_1 += Q8_1 addition
    // Function signature: q16_1_add_q8_1(Q16_1Block* residual, const Q8_1Block* delta, size_t count)
    simd::q16_1_add_q8_1(residual_blocks, delta_blocks, num_elements);
    
    return true;
}
```

**Update execute() dispatch** (around line 2020):

```cpp
// Add after FP32_Q8_1_to_Q8_1 check
// Handle Q8_1 input + Q16_1 residual → Q16_1 output
if (input_type == TensorType::Q8_1 && residual_type == TensorType::Q16_1)
{
    return executeQ16_1_Q8_1(ctx, num_elements);
}
```

### 3.2 Update EmbeddingStage for Q16_1 Output

**File**: `src/v2/execution/ComputeStage.cpp`

The embedding lookup outputs FP32, which then needs to be converted to Q16_1:

```cpp
bool EmbeddingStage::execute(IDeviceContext *ctx)
{
    // ... existing FP32 embedding lookup ...
    
    // If output is Q16_1, quantize the FP32 result
    if (params_.output->native_type() == TensorType::Q16_1)
    {
        auto *output_q16 = dynamic_cast<Q16_1Tensor *>(params_.output);
        if (!output_q16)
        {
            LOG_ERROR("[EmbeddingStage] Failed to cast output to Q16_1Tensor");
            return false;
        }
        
        // Quantize from FP32 embedding table
        output_q16->copyFrom_fp32(embedding_output_fp32, num_elements);
    }
    
    return true;
}
```

**Alternative**: Add a separate `QuantizeStage` that converts FP32→Q16_1 after embedding.

### 3.3 Add Header Declaration

**File**: `src/v2/execution/ComputeStage.h`

```cpp
class ResidualAddStage : public IComputeStage
{
    // ... existing methods ...
    
private:
    bool executeFP32(IDeviceContext *ctx, size_t num_elements);
    bool executeBF16(IDeviceContext *ctx, size_t num_elements);
    bool executeFP16(IDeviceContext *ctx, size_t num_elements);
    bool executeQ8_1(IDeviceContext *ctx, size_t num_elements);
    bool executeFP32_Q8_1_to_Q8_1(IDeviceContext *ctx, size_t num_elements);
    bool executeQ16_1_Q8_1(IDeviceContext *ctx, size_t num_elements);  // NEW
};
```

---

## Phase 3.5: Q16_1 RoPE ✅

**Status**: Complete (December 26, 2025)

### Motivation

Instead of the expensive path:
```
Q16_1 → dequant to FP32 → RoPE → requant to Q16_1
```

We implemented **in-place Q16_1 RoPE** that operates directly on quantized blocks, similar to the existing Q8_1 RoPE but with 256× finer precision.

### Implementation

#### 3.5.1 RoPE Primitives

**File**: `src/v2/kernels/cpu/primitives/RoPEPrimitives.h` / `.cpp`

```cpp
// Per-head Q16_1 RoPE rotation
void apply_rope_q16_1_integer_head(
    Q16_1Block* head_blocks,
    int blocks_per_head,
    const int16_t* cos_q15,
    const int16_t* sin_q15);

// Full tensor Q16_1 RoPE with OpenMP parallelization
void apply_rope_q16_1_integer(
    Q16_1Block* Q,
    Q16_1Block* K,
    const int* position_ids,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta,
    RoPEPersistentState* persistent_state);
```

**Algorithm**:
1. Precompute sin/cos tables in Q15 format (±32767 range)
2. For each block pair (first half / second half of head):
   - Dequantize to FP32 using FP32 scale (no FP16 conversion!)
   - Rotate: x' = x×cos - y×sin, y' = x×sin + y×cos
   - Requantize to Q16_1 with new scale

#### 3.5.2 CPURoPEKernelT Specialization

**File**: `src/v2/kernels/cpu/ops/CPURoPEKernelT.h` / `.cpp`

```cpp
template<>
class CPURoPEKernelT<ActivationPrecision::Q16_1> : public ITensorRoPE
{
    bool apply_q16_1(void* Q, void* K, const int* pos_ids,
                     int seq_len, int n_heads, int n_kv_heads,
                     int head_dim, float theta_base, int device_idx) override;
    
    bool apply_tensor(TensorBase* Q, TensorBase* K, ...) override;
};
```

#### 3.5.3 ITensorRoPE Interface

**File**: `src/v2/tensors/TensorKernels.h`

```cpp
// Added to ITensorRoPE:
virtual bool apply_q16_1(
    void* Q_data, void* K_data,
    const int* pos_ids,
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    float theta_base, int device_idx) { return false; }
```

#### 3.5.4 Q16_1Tensor Integration

**File**: `src/v2/tensors/Q16_1Tensor.cpp`

```cpp
std::unique_ptr<ITensorRoPE> Q16_1Tensor::createRoPE()
{
    return std::make_unique<CPURoPEKernelT<ActivationPrecision::Q16_1>>();
}

bool Q16_1Tensor::applyRoPE(float* K, const int* position_ids, ...)
{
    auto kernel = createRoPE();
    return kernel->apply_q16_1(mutable_q16_1_blocks(), (void*)K, ...);
}
```

### Measured Accuracy (vs Pure FP32 RoPE)

| Metric | Measured | Threshold |
|--------|----------|-----------|
| Max absolute diff | 9.07×10⁻⁵ | < 1×10⁻³ |
| Cosine similarity | 0.99999946 | > 0.99999 |
| Relative RMSE | 2.24×10⁻⁵ | < 2.5×10⁻⁴ |

**Conclusion**: Q16_1 RoPE is virtually indistinguishable from FP32 RoPE.

### Unit Tests Added

- `CPURoPEKernelTTest.Q16_1_apply_typed_with_conversion`
- `CPURoPEKernelTTest.Q16_1_precision_metadata`
- `CPURoPEKernelTTest.Q16_1_apply_tensor`
- `CPURoPEKernelTTest.Q16_1_vs_Q8_1_precision_improvement`
- `CPURoPEKernelTTest.Q16_1_vs_FP32_RoPE_accuracy`

---

## Phase 4: RMSNorm Q16_1 Support

### Design Decision: Option A

RMSNorm reads Q16_1 input directly, dequantizes on-the-fly during the normalization pass, and outputs FP32. This is a **single memory pass** approach.

### 4.1 RMSNorm Kernel with Q16_1 Input

**File**: `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp`

```cpp
/**
 * @brief RMSNorm with Q16_1 input, FP32 output
 *
 * Algorithm:
 * 1. Dequantize Q16_1 block → FP32 in registers
 * 2. Compute sum of squares (for RMS)
 * 3. Apply normalization and gamma scaling
 * 4. Write FP32 output
 *
 * Memory access: 1 read (Q16_1) + 1 write (FP32)
 */
void rms_norm_q16_1_fp32(
    const Q16_1Block* input,
    const float* gamma,
    float* output,
    size_t hidden_dim,
    float eps)
{
    const size_t n_blocks = hidden_dim / 32;
    
    // Pass 1: Compute sum of squares (dequant + accumulate)
    double sum_sq = 0.0;
    for (size_t blk = 0; blk < n_blocks; ++blk)
    {
        const Q16_1Block& block = input[blk];
        float scale = block.d;
        
        for (int i = 0; i < 32; ++i)
        {
            float val = scale * static_cast<float>(block.qs[i]);
            sum_sq += static_cast<double>(val * val);
        }
    }
    
    // Compute RMS and inverse
    float rms = std::sqrt(static_cast<float>(sum_sq / hidden_dim) + eps);
    float inv_rms = 1.0f / rms;
    
    // Pass 2: Dequant + normalize + gamma + write
    for (size_t blk = 0; blk < n_blocks; ++blk)
    {
        const Q16_1Block& block = input[blk];
        float scale = block.d;
        
        for (int i = 0; i < 32; ++i)
        {
            size_t idx = blk * 32 + i;
            float val = scale * static_cast<float>(block.qs[i]);
            output[idx] = val * inv_rms * gamma[idx];
        }
    }
}
```

### 4.2 SIMD-Optimized Version (AVX512)

```cpp
#ifdef __AVX512F__
void rms_norm_q16_1_fp32_avx512(
    const Q16_1Block* input,
    const float* gamma,
    float* output,
    size_t hidden_dim,
    float eps)
{
    const size_t n_blocks = hidden_dim / 32;
    
    // Pass 1: Vectorized sum of squares
    __m512 sum_sq_vec = _mm512_setzero_ps();
    
    for (size_t blk = 0; blk < n_blocks; ++blk)
    {
        const Q16_1Block& block = input[blk];
        __m512 scale_vec = _mm512_set1_ps(block.d);
        
        // Load 16 int16 → 16 float, compute squares
        __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.qs));
        __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.qs + 16));
        
        __m512 f_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_lo)), scale_vec);
        __m512 f_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_hi)), scale_vec);
        
        sum_sq_vec = _mm512_fmadd_ps(f_lo, f_lo, sum_sq_vec);
        sum_sq_vec = _mm512_fmadd_ps(f_hi, f_hi, sum_sq_vec);
    }
    
    float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);
    float rms = std::sqrt(sum_sq / hidden_dim + eps);
    __m512 inv_rms_vec = _mm512_set1_ps(1.0f / rms);
    
    // Pass 2: Vectorized dequant + normalize + gamma + write
    for (size_t blk = 0; blk < n_blocks; ++blk)
    {
        const Q16_1Block& block = input[blk];
        __m512 scale_vec = _mm512_set1_ps(block.d);
        
        __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.qs));
        __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.qs + 16));
        
        __m512 f_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_lo)), scale_vec);
        __m512 f_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_hi)), scale_vec);
        
        __m512 g_lo = _mm512_loadu_ps(gamma + blk * 32);
        __m512 g_hi = _mm512_loadu_ps(gamma + blk * 32 + 16);
        
        __m512 out_lo = _mm512_mul_ps(_mm512_mul_ps(f_lo, inv_rms_vec), g_lo);
        __m512 out_hi = _mm512_mul_ps(_mm512_mul_ps(f_hi, inv_rms_vec), g_hi);
        
        _mm512_storeu_ps(output + blk * 32, out_lo);
        _mm512_storeu_ps(output + blk * 32 + 16, out_hi);
    }
}
#endif
```

### 4.3 Update RMSNormStage

**File**: `src/v2/execution/ComputeStage.cpp`

Add dispatch for Q16_1 input:

```cpp
bool RMSNormStage::execute(IDeviceContext *ctx)
{
    // ... existing checks ...
    
    TensorType input_type = params_.input->native_type();
    
    // Q16_1 input → FP32 output (typed residual mode)
    if (input_type == TensorType::Q16_1)
    {
        return executeQ16_1_to_FP32(ctx);
    }
    
    // ... existing FP32/BF16/FP16 dispatch ...
}

bool RMSNormStage::executeQ16_1_to_FP32(IDeviceContext *ctx)
{
    const auto *input_q16 = dynamic_cast<const Q16_1Tensor *>(params_.input);
    auto *output_fp32 = dynamic_cast<FP32Tensor *>(params_.output);
    const auto *gamma_fp32 = dynamic_cast<const FP32Tensor *>(params_.gamma);
    
    if (!input_q16 || !output_fp32 || !gamma_fp32)
    {
        LOG_ERROR("[RMSNormStage] Failed to cast tensors for Q16_1→FP32");
        return false;
    }
    
    const size_t hidden_dim = params_.input->shape().back();
    const size_t num_rows = params_.num_elements > 0 
        ? params_.num_elements / hidden_dim 
        : params_.input->shape()[0];
    
    const Q16_1Block *input_blocks = input_q16->q16_1_blocks();
    float *output = output_fp32->mutable_data();
    const float *gamma = gamma_fp32->data();
    
    // Process each row
    const size_t blocks_per_row = hidden_dim / 32;
    for (size_t row = 0; row < num_rows; ++row)
    {
        rms_norm_q16_1_fp32_avx512(
            input_blocks + row * blocks_per_row,
            gamma,
            output + row * hidden_dim,
            hidden_dim,
            params_.eps);
    }
    
    return true;
}
```

### 4.4 Implementation Summary (Completed June 2025)

**Status**: ✅ Complete

#### Files Modified

1. **`src/v2/kernels/cpu/primitives/RMSNormPrimitives.h`**
   - Added `rmsnorm_q16_1_fp32_row_scalar()` - scalar per-row implementation
   - Added `rmsnorm_q16_1_fp32_row_avx512()` - AVX512 vectorized per-row
   - Added `rmsnorm_q16_1_fp32_fused()` - multi-row wrapper with OpenMP

2. **`src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp`**
   - ~100 lines of Q16_1→FP32 RMSNorm implementation
   - Double-precision accumulation for sum of squares
   - AVX512: Uses `_mm512_cvtepi16_epi32` → `_mm512_cvtepi32_ps` for dequant

3. **`src/v2/tensors/TensorKernels.h`**
   - Added `apply_q16_1_to_fp32()` virtual method to `ITensorRMSNorm`

4. **`src/v2/kernels/cpu/ops/CPURMSNormKernelT.h`**
   - Added `PrecisionMetadata<Q16_1>` specialization
   - Added full `CPURMSNormKernelT<Q16_1>` specialization (Q16_1→FP32)

5. **`src/v2/kernels/cpu/ops/CPURMSNormKernelT.cpp`**
   - Added `apply_typed()` implementation for Q16_1 specialization

6. **`src/v2/kernels/KernelFactory.h` / `KernelFactory.cpp`**
   - Added `Q16_1Tensor` forward declaration
   - Added `createRMSNorm(Q16_1Tensor*)` overload
   - Added Q16_1 case to generic TensorBase dispatch

7. **`tests/v2/unit/Test__Q16_1RMSNorm.cpp`**
   - 8 unit tests covering: basic functionality, scalar/AVX512 parity, multi-row, accuracy vs FP32, edge cases

#### Measured Accuracy (Q16_1 RMSNorm vs FP32 RMSNorm)

```
Max diff:    3.70×10⁻⁵
Mean diff:   1.25×10⁻⁵  
Rel RMSE:    1.46×10⁻⁵
Cosine sim:  1.0 (effectively perfect)
```

The Q16_1 quantization introduces minimal error (<4e-5 max) in RMSNorm output.

#### Test Results

All 177 unit tests pass (176 existing + 1 new Q16_1RMSNorm suite with 8 tests).

---

## Phase 5: JIT Fusion

### Goal

Fuse the entire attention tail into a single JIT kernel:
```
Attention Context (FP32) → Wo GEMM → FP32 + Dequant(Q16_1 residual) → Q16_1 output
```

**Key insight**: We do NOT need an intermediate Q8_1 quantization step. The Wo projection output stays as FP32 in ZMM registers, and we directly add it to the dequantized Q16_1 residual. This eliminates one quantization/dequantization round-trip, reducing error and compute.

All intermediate values stay in **ZMM registers**. No memory traffic between operations.

### 5.1 Current JitFusedAttentionWo Architecture

The existing `JitFusedAttentionWo` kernel already fuses:
- FlashAttention-2 tiled attention (Q×K^T, softmax, ×V)
- Wo projection (context × Wo weights)

**Current output**: FP32 projection result written to memory.

### 5.2 Proposed Extension: FusedAttentionWoResidual

**New kernel output flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    JitFusedAttentionWoResidual                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Q, K, V in registers]                                         │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │   FA2 Core   │  Q×K^T, softmax, ×V                          │
│  └──────┬───────┘                                               │
│         │ context (ZMM accumulators)                            │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  Wo Tiling   │  context × Wo_weights (VNNI)                 │
│  └──────┬───────┘                                               │
│         │ proj (ZMM registers, FP32)                            │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ Load Q16_1   │  Load residual block from memory              │
│  │ Dequant      │  scale × int16 → FP32 (in ZMM)               │
│  └──────┬───────┘                                               │
│         │ residual_fp32 (ZMM registers)                         │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │   vaddps     │  proj_fp32 + residual_fp32                    │
│  └──────┬───────┘  (FP32 arithmetic in ZMM)                     │
│         │ sum (ZMM registers, FP32)                             │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │ Quant Q16_1  │  max_abs → scale → round → int16              │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  [Store Q16_1 residual to memory]                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 JIT Code Generation Changes

**File**: `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h`

Add new config fields to `JitAttentionConfig`:

```cpp
struct JitAttentionConfig
{
    // ... existing fields (head_dim, num_heads, d_model, etc.) ...
    
    // NEW: Residual fusion config
    bool fuse_residual_add = false;                    // Enable fused residual addition
    TensorType residual_type = TensorType::FP32;       // Residual tensor format (FP32 or Q16_1)
    // When fuse_residual_add=true and residual_type=Q16_1:
    //   - Wo projection output stays in ZMM as FP32 (no intermediate quantization!)
    //   - Load Q16_1 residual, dequant to FP32
    //   - vaddps: proj_fp32 + residual_fp32
    //   - Quantize sum → Q16_1
    //   - Store Q16_1 residual
};
```

**File**: `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h`

New emission functions in `JitFusedAttentionWoGenerator`:

```cpp
class JitFusedAttentionWoGenerator : public JitMicrokernelBase
{
    // ... existing members ...
    
protected:
    // After Wo projection completes (proj in ZMM accumulators as FP32)
    void emit_residual_add_q16_1()
    {
        // Wo projection result is already in accumulators as FP32 - NO quantization needed!
        
        // 1. Load Q16_1 residual block
        emit_load_q16_1_residual();
        
        // 2. Dequant Q16_1 → FP32 (in registers)
        emit_dequant_q16_1_to_fp32();
        
        // 3. Add: proj_fp32 (accumulators) + residual_fp32 (scratch)
        emit_fp32_add();
        
        // 4. Quantize sum → Q16_1
        emit_quantize_to_q16_1();
        
        // 5. Store Q16_1 residual
        emit_store_q16_1_residual();
    }
    
    void emit_load_q16_1_residual()
    {
        // residual_ptr points to Q16_1Block for current position
        // Q16_1Block layout: float d (4B) | int32 sum_qs (4B) | int16 qs[32] (64B)
        
        // Load scale (FP32) - offset 0
        vbroadcastss(scratch2().zmm(), ptr[reg_residual_ptr]);
        
        // Load int16 values (32 elements = 64 bytes) - offset 8
        vmovdqu16(scratch3().zmm(), ptr[reg_residual_ptr + 8]);        // qs[0:15]
        vmovdqu16(scratch4().zmm(), ptr[reg_residual_ptr + 8 + 32]);   // qs[16:31]
    }
    
    void emit_dequant_q16_1_to_fp32()
    {
        // int16 → int32 → FP32 × scale
        vpmovsxwd(scratch3().zmm(), scratch3().ymm());  // Sign-extend int16 → int32
        vcvtdq2ps(scratch3().zmm(), scratch3().zmm());  // int32 → FP32
        vmulps(scratch3().zmm(), scratch3().zmm(), scratch2().zmm());  // × scale
        
        vpmovsxwd(scratch4().zmm(), scratch4().ymm());
        vcvtdq2ps(scratch4().zmm(), scratch4().zmm());
        vmulps(scratch4().zmm(), scratch4().zmm(), scratch2().zmm());
    }
    
    void emit_fp32_add()
    {
        // Add FP32 projection (in accumulators) to dequantized residual (in scratch3/4)
        // proj is in accum0-7 as FP32, residual_fp32 is in scratch3/4
        vaddps(scratch3().zmm(), scratch3().zmm(), accum0().zmm());
        vaddps(scratch4().zmm(), scratch4().zmm(), accum1().zmm());
        // Continue for all accumulator registers as needed...
    }
    
    void emit_quantize_to_q16_1()
    {
        // Find max_abs of sum
        vandps(scratch0().zmm(), scratch3().zmm(), const_abs_mask().zmm());
        vandps(scratch1().zmm(), scratch4().zmm(), const_abs_mask().zmm());
        vmaxps(scratch0().zmm(), scratch0().zmm(), scratch1().zmm());
        // ... horizontal reduction to get max_abs scalar ...
        
        // Compute scale = max_abs / 32767.0f
        // Compute inv_scale = 32767.0f / max_abs
        vbroadcastss(scratch1().zmm(), xmm_inv_scale);
        
        // Scale, round, convert to int16
        vmulps(scratch3().zmm(), scratch3().zmm(), scratch1().zmm());
        vmulps(scratch4().zmm(), scratch4().zmm(), scratch1().zmm());
        vroundps(scratch3().zmm(), scratch3().zmm(), _MM_FROUND_TO_NEAREST_INT);
        vroundps(scratch4().zmm(), scratch4().zmm(), _MM_FROUND_TO_NEAREST_INT);
        vcvtps_epi32(scratch3().zmm(), scratch3().zmm());
        vcvtps_epi32(scratch4().zmm(), scratch4().zmm());
        vpmovdw(scratch3().ymm(), scratch3().zmm());  // 32-bit → 16-bit pack
        vpmovdw(scratch4().ymm(), scratch4().zmm());
    }
    
    void emit_store_q16_1_residual()
    {
        // Store scale (FP32) - offset 0
        vmovss(ptr[reg_residual_ptr], xmm_new_scale);
        
        // Store sum_qs (int32) - offset 4, computed during quantization
        mov(dword[reg_residual_ptr + 4], reg_sum_qs);
        
        // Store int16 values - offset 8
        vmovdqu16(ptr[reg_residual_ptr + 8], scratch3().ymm());
        vmovdqu16(ptr[reg_residual_ptr + 8 + 32], scratch4().ymm());
    }
};
```

### 5.4 Register Allocation for Fusion

The JIT kernel needs additional registers for the residual fusion path:

| Zone | Registers | Purpose |
|------|-----------|---------|
| **Proj Accum** | zmm0-7 | Wo projection output (existing) |
| **Residual** | zmm8-11 | Q16_1 residual (load, dequant, sum) |
| **Scratch** | zmm12-15 | Intermediate computations |
| **Constants** | zmm28-31 | abs_mask, 32767.0f, etc. |

This fits within the existing register budget with careful scheduling.

### 5.5 Performance Impact

**Before fusion** (separate stages):
1. Store FP32 Wo output (memory write)
2. Residual Add: Load Wo, Load Q16_1 residual, Add, Store Q16_1 (4 memory ops)

**After fusion**:
1. Load Q16_1 residual (memory read)
2. Store Q16_1 result (memory write)

**Savings**: 3 memory operations per attention block × batch_size × seq_len × num_layers

For Qwen2-0.5B (24 layers), batch=1, seq=2048, d_model=896:
- Before: 24 × 2048 × 896 × 4 bytes × 4 ops = 700 MB memory traffic
- After: 24 × 2048 × 896 × 2.25 bytes × 2 ops = 197 MB memory traffic
- **3.5× reduction in memory bandwidth**

### 5.6 Phase 5 Implementation Progress (December 26, 2025)

#### Completed Work

1. **JitAttentionConfig Extensions**
   - Added `fuse_residual_add` flag (default: false)
   - Added `ResidualType` enum (FP32, Q16_1)
   - Updated equality operator and hash function for kernel caching

2. **JIT Emit Method: `emit_q16_1_residual_fusion()`**
   - Loads Q16_1 residual block from memory (scale + int16 values)
   - Dequantizes int16 → FP32 via `vpmovsxwd` + `vcvtdq2ps` + `vmulps`
   - Adds FP32 Wo output to dequantized residual via `vaddps`
   - Computes max_abs via horizontal reduction for new scale
   - Quantizes sum to int16 via `vcvtps2dq` + `vpmovsdw`
   - Stores Q16_1 block (scale, sum_qs, int16 values)

3. **emit_wo_projection() Integration**
   - Allocates temporary FP32 buffer on stack when fused
   - Adjusts output pointer stride for Q16_1 layout (72 bytes per 32 elements)
   - Calls residual fusion after Wo projection completes
   - Restores stack after fusion

4. **Unit Tests** (`Test__JitFusedAttentionWo_Q16_1Residual.cpp`)
   - Config validation tests
   - Q16_1 roundtrip accuracy (cosine: 1.0, rel error: 1.5e-5)
   - Reference residual addition parity
   - JIT kernel generation verification
   - Q16_1Block layout validation

#### Files Modified

| File | Changes |
|------|---------|
| `JitFusedAttentionWo.h` | +200 lines: config, emit methods, integration |
| `tests/v2/CMakeLists.txt` | Added test target |
| `Test__JitFusedAttentionWo_Q16_1Residual.cpp` | New file, 6 tests |

#### Test Results

All 8 JIT fused attention tests pass (6 new + 2 existing):
- `JitKernelGenerationWithFusedResidual`: Kernel generates successfully
- `Q16_1RoundtripAccuracy`: Cosine 1.0, Rel L2 Error 1.5e-5
- `ReferenceResidualAddition`: Cosine 1.0, Rel L2 Error 1.8e-5

#### Remaining Work

- Integration test: Full attention pipeline with Q16_1 residual fusion
- End-to-end parity test vs FP32 reference
- Performance benchmarking (decode latency, memory bandwidth)

---

## Phase 6: Integration Wiring (NEXT)

**Status**: Not Started  
**Estimated Effort**: 4-6 hours  
**Prerequisites**: Phase 5 (JIT Fusion kernel) complete

### Overview

Phase 6 wires the JIT fusion kernel into the production pipeline. The JIT code exists and passes unit tests, but the production code path doesn't enable it. This phase connects:

1. CLI argument `--activation-precision hybridq16`
2. `FusedAttentionWoStage` passing residual buffer to kernel
3. `FusedAttentionWoKernel` setting JIT fusion flags
4. `Qwen2Graph` wiring residual buffer when HybridQ16 mode

### 6.1 CLI Argument: Add `hybridq16` Value

**File**: `src/v2/utils/ArgParser.cpp` (line 30)

```cpp
// BEFORE (line 30-32):
{"--activation-precision",
 {"--activation-prec", "--act-prec"},
 {"fp32", "bf16", "fp16", "q8_1", "hybrid"},
 "hybrid",
 false,
 false},

// AFTER:
{"--activation-precision",
 {"--activation-prec", "--act-prec"},
 {"fp32", "bf16", "fp16", "q8_1", "hybrid", "hybridq16"},
 "hybrid",
 false,
 false},
```

**File**: `src/v2/Main.cpp` (lines 396-410)

```cpp
// BEFORE (line 396-410):
else if (args.activation_precision == "hybrid")
{
    runtime_config.activation_precision = ActivationPrecision::Hybrid;
}
else
{
    // This branch should never be reached...
    runtime_config.activation_precision = ActivationPrecision::Hybrid;
}

// AFTER:
else if (args.activation_precision == "hybrid")
{
    runtime_config.activation_precision = ActivationPrecision::Hybrid;
}
else if (args.activation_precision == "hybridq16")
{
    runtime_config.activation_precision = ActivationPrecision::HybridQ16;
}
else
{
    // This branch should never be reached...
    runtime_config.activation_precision = ActivationPrecision::Hybrid;
}
```

### 6.2 FusedAttentionWoStage: Add Residual Buffer to Params

**File**: `src/v2/execution/ComputeStage.h` (lines 1350-1355)

```cpp
// ADD to FusedAttentionWoStage::Params struct (after use_hybrid_wo field):

// Q16_1 Residual fusion (HybridQ16 mode)
// When enabled, the kernel fuses residual addition after Wo projection:
// - output stays in registers as FP32 after Wo GEMM
// - residual_buffer is loaded (Q16_1), dequantized, added
// - result is quantized to Q16_1 and stored to residual_buffer
// This eliminates FP32 intermediate memory traffic.
bool fuse_residual_add = false;
TensorBase *residual_buffer = nullptr;  ///< Q16_1 residual (input/output for fusion)
```

**File**: `src/v2/execution/ComputeStage.cpp` (lines 4288-4302)

```cpp
// MODIFY FusedAttentionWoStage constructor to pass fusion config to kernel:

FusedAttentionWoStage::FusedAttentionWoStage(Params params)
    : params_(std::move(params))
{
    // Create the kernel with configuration
    FusedAttentionWoKernel::Config kernel_config;
    kernel_config.num_heads = params_.n_heads;
    kernel_config.num_kv_heads = params_.n_kv_heads;
    kernel_config.head_dim = params_.head_dim;
    kernel_config.d_model = params_.d_model;
    kernel_config.backend = params_.backend;
    kernel_config.use_hybrid_wo = params_.use_hybrid_wo;
    // NEW: Residual fusion config
    kernel_config.fuse_residual_add = params_.fuse_residual_add;
    kernel_config.residual_buffer = params_.residual_buffer;

    kernel_ = std::make_unique<FusedAttentionWoKernel>(kernel_config);
}
```

### 6.3 FusedAttentionWoKernel: Wire JIT Config Flags

**File**: `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h` (lines 63-70)

```cpp
// ADD to FusedAttentionWoKernel::Config struct:

// Q16_1 Residual fusion (HybridQ16 mode)
bool fuse_residual_add = false;
TensorBase *residual_buffer = nullptr;
```

**File**: `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h` (lines 455-470)

In `execute_jit()` method, after creating `JitAttentionConfig jit_config`:

```cpp
// BEFORE (around line 455-470):
jit_config.d_model = params.d_model;
jit_config.wo_format = wo_format;
jit_config.batch_size = params.batch_size;
jit_config.causal = params.causal;
jit_config.use_fa2_tiling = true;

// AFTER:
jit_config.d_model = params.d_model;
jit_config.wo_format = wo_format;
jit_config.batch_size = params.batch_size;
jit_config.causal = params.causal;
jit_config.use_fa2_tiling = true;

// NEW: Wire residual fusion config
jit_config.fuse_residual_add = config_.fuse_residual_add;
if (config_.fuse_residual_add && config_.residual_buffer)
{
    // Validate residual buffer is Q16_1
    auto *residual_q16 = dynamic_cast<Q16_1Tensor *>(config_.residual_buffer);
    if (residual_q16)
    {
        jit_config.residual_type = JitAttentionConfig::ResidualType::Q16_1;
    }
    else
    {
        LOG_WARN("FusedAttentionWoKernel: fuse_residual_add=true but residual_buffer is not Q16_1, disabling fusion");
        jit_config.fuse_residual_add = false;
    }
}
```

Also need to pass residual pointer to JitFusedAttentionWo::compute() call (line ~570):

```cpp
// BEFORE:
(*slot_ptr)->compute(
    params.Q,
    params.K,
    params.V,
    wo_ptr,
    params.output,
    params.seq_len,
    params.kv_seq_len,
    params.scale,
    params.position_offset,
    params.context_snapshot);

// AFTER:
void *residual_ptr = nullptr;
if (jit_config.fuse_residual_add && config_.residual_buffer)
{
    auto *residual_q16 = dynamic_cast<Q16_1Tensor *>(config_.residual_buffer);
    if (residual_q16)
    {
        residual_ptr = residual_q16->mutable_q16_1_blocks();
    }
}

(*slot_ptr)->compute(
    params.Q,
    params.K,
    params.V,
    wo_ptr,
    params.output,
    params.seq_len,
    params.kv_seq_len,
    params.scale,
    params.position_offset,
    params.context_snapshot,
    residual_ptr);  // NEW: residual buffer for fusion
```

### 6.4 Qwen2Graph: Wire Residual Buffer in HybridQ16 Mode

**File**: `src/v2/models/qwen/Qwen2Graph.cpp` (lines 1036-1045)

After setting `fused_params.use_hybrid_wo`:

```cpp
// BEFORE (around line 1041):
fused_params.use_hybrid_wo = inference_mode.isHybrid();

// Optional context snapshot for debugging
fused_params.context_snapshot = buffers.context_snapshot;

// AFTER:
fused_params.use_hybrid_wo = inference_mode.isHybrid() || inference_mode.isHybridQ16();

// NEW: HybridQ16 residual fusion
// When activation_precision == HybridQ16, enable fused residual addition
// This keeps Wo output in registers and writes directly to Q16_1 residual
if (inference_mode.isHybridQ16())
{
    fused_params.fuse_residual_add = true;
    fused_params.residual_buffer = buffers.residual;
    LOG_DEBUG("[Qwen2Graph] Layer " << layer_idx << " HybridQ16 mode: fusing residual add");
}

// Optional context snapshot for debugging
fused_params.context_snapshot = buffers.context_snapshot;
```

### 6.5 InferenceMode: Add isHybridQ16() Helper

**File**: `src/v2/execution/InferenceMode.h`

```cpp
// ADD method to InferenceMode class:
bool isHybridQ16() const
{
    return precision_ == ActivationPrecision::HybridQ16;
}
```

### 6.6 JitFusedAttentionWo::compute(): Add Residual Parameter

**File**: `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h`

The compute() method signature needs to accept residual_ptr:

```cpp
// BEFORE:
void compute(
    const void *Q,
    const void *K,
    const void *V,
    const void *Wo,
    void *output,
    int seq_len,
    int kv_seq_len,
    float scale,
    int position_offset,
    void *context_snapshot = nullptr);

// AFTER:
void compute(
    const void *Q,
    const void *K,
    const void *V,
    const void *Wo,
    void *output,
    int seq_len,
    int kv_seq_len,
    float scale,
    int position_offset,
    void *context_snapshot = nullptr,
    void *residual_ptr = nullptr);  // NEW: Q16_1 residual for fusion
```

The JIT-generated code already handles residual fusion when `config_.fuse_residual_add` is true; we just need to pass the pointer.

### 6.7 Test: Verify Integration

**File**: `tests/v2/integration/Test__HybridQ16_Integration.cpp` (new file)

```cpp
TEST(Test__HybridQ16_Integration, FusedAttentionWo_WithResidual)
{
    // Setup model with HybridQ16 mode
    RuntimeConfig config;
    config.activation_precision = ActivationPrecision::HybridQ16;
    config.use_fused_attention = true;
    
    // Run single layer attention
    // Verify:
    // 1. No FP32 intermediate buffer allocated
    // 2. Residual buffer is Q16_1
    // 3. Output matches reference within Q16_1 precision
}
```

### 6.8 Files Summary

| File | Line Range | Changes |
|------|------------|---------|
| `src/v2/utils/ArgParser.cpp` | 30-32 | Add `"hybridq16"` to valid values |
| `src/v2/Main.cpp` | 396-410 | Add `hybridq16` → `HybridQ16` case |
| `src/v2/execution/ComputeStage.h` | 1350-1355 | Add `fuse_residual_add` to Params |
| `src/v2/execution/ComputeStage.cpp` | 4288-4302 | Pass fusion config to kernel |
| `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h` | 63-82 | Add `fuse_residual_add` to Config |
| `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h` | 245-270 | Handle Q16_1 output validation |
| `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h` | 488-505 | Set JitAttentionConfig fusion flags |
| `src/v2/models/qwen/Qwen2Graph.cpp` | 1036-1055 | Wire residual buffer in HybridQ16 |
| `src/v2/models/qwen/Qwen2Graph.cpp` | 1168-1185 | Update allreduce for Q16_1 buffer |
| `src/v2/models/qwen/Qwen2Graph.cpp` | 1186-1200 | Skip ResidualAddStage when fused |
| `src/v2/execution/InferenceMode.h` | 95-100 | Add `isHybridQ16()`, `isAnyHybrid()` |

### 6.9 Phase 6 Implementation Progress (December 26, 2025)

#### Completed Work

1. **CLI Support** (`ArgParser.cpp`, `Main.cpp`)
   - Added `"hybridq16"` to valid `--activation-precision` values
   - Added case for `hybridq16` → `ActivationPrecision::HybridQ16` in Main.cpp

2. **InferenceMode Helpers** (`InferenceMode.h`)
   - Added `isHybridQ16()` for HybridQ16-specific checks
   - Added `isAnyHybrid()` for Hybrid OR HybridQ16 checks
   - Updated `needsQRope()`, `needsKRope()`, `needsVDequant()` to use `isAnyHybrid()`

3. **FusedAttentionWoStage** (`ComputeStage.h`, `ComputeStage.cpp`)
   - Added `fuse_residual_add` to `Params` struct
   - Constructor passes fusion config to kernel

4. **FusedAttentionWoKernel** (`FusedAttentionWoKernel.h`)
   - Added `fuse_residual_add` to `Config` struct
   - `compute()`: Accept Q16_1 output tensor when fusion enabled
   - `execute_jit()`: Set `jit_config.fuse_residual_add` and `residual_type`

5. **Qwen2Graph** (`Qwen2Graph.cpp`)
   - Set `fuse_residual_add=true` and `output=buffers.residual` when HybridQ16
   - Update allreduce to use Q16_1 residual buffer when fused
   - Skip separate `ResidualAddStage` when fusion is enabled

#### Test Results

All tests pass:
- 178/178 unit tests pass
- 6/6 JIT Q16_1 residual fusion tests pass
- 10/10 fused attention Wo kernel tests pass
- 21/21 InferenceMode tests pass
- 33/33 ArgParser tests pass

---

## Validation Plan

### Unit Tests

#### 6.1 Q16_1 Tensor Tests (DONE ✓)

```cpp
TEST(Test__Q16_1Tensor, EncodeDecodeRoundTrip_SmallError)
TEST(Test__Q16_1Tensor, Q16_1_vs_Q8_1_PrecisionComparison)
TEST(Test__Q16_1Tensor, NativeQ16_1Addition)
TEST(Test__Q16_1Tensor, NativeQ16_1AddFP32)
TEST(Test__Q16_1Tensor, NativeQ16_1AddQ8_1)
TEST(Test__Q16_1Tensor, NativeQ16_1AddQ8_1_vs_TwoStep)
TEST(Test__Q16_1Tensor, NativeQ16_1AddQ8_1_RealisticMagnitudes)
TEST(Test__Q16_1Tensor, Q16_1ToQ8_1Packed)
```

#### 6.2 RMSNorm Q16_1 Tests (Phase 4)

```cpp
TEST(Test__RMSNormKernel, Q16_1_Input_FP32_Output)
{
    // Verify RMSNorm(Q16_1) matches RMSNorm(FP32) within Q16_1 precision
}

TEST(Test__RMSNormKernel, Q16_1_SIMD_vs_Scalar_Parity)
{
    // Verify AVX512 implementation matches scalar reference
}

TEST(Test__RMSNormKernel, Q16_1_Numerical_Stability)
{
    // Test with extreme values, verify no overflow/underflow
}
```

#### 6.3 ResidualAddStage Tests (Phase 3)

```cpp
TEST(Test__ResidualAddStage, Q16_1_Q8_1_Dispatch)
{
    // Verify correct type dispatch for Q8_1 + Q16_1
}

TEST(Test__ResidualAddStage, Q16_1_Q8_1_Precision)
{
    // Verify precision matches q16_1_add_q8_1() SIMD function
}

TEST(Test__ResidualAddStage, Q16_1_Q8_1_InPlace)
{
    // Verify in-place update works correctly
}
```

### Integration Tests

#### 6.4 Full Pipeline Tests (Phase 2-3)

```cpp
TEST(Test__GraphOrchestrator, HybridQ16_BufferAllocation)
{
    // Verify correct buffer types allocated for HybridQ16 mode
}

TEST(Test__LayerExecution, HybridQ16_AttentionResidual)
{
    // Single layer attention with Q16_1 residual
    // Compare output to FP32 reference
}

TEST(Test__LayerExecution, HybridQ16_FFNResidual)
{
    // Single layer FFN with Q16_1 residual
    // Compare output to FP32 reference
}
```

#### 6.5 JIT Fusion Tests (Phase 5)

```cpp
TEST(Test__JitFusedAttentionWoResidual, Correctness)
{
    // Compare fused kernel output to staged execution
}

TEST(Test__JitFusedAttentionWoResidual, NumericalParity)
{
    // Verify fused path matches non-fused path within precision bounds
}

TEST(Test__JitFusedAttentionWoResidual, Performance)
{
    // Benchmark fused vs non-fused, expect >2x speedup
}
```

### E2E Parity Tests

#### 6.6 Model-Level Validation

```cpp
TEST(Test__E2E_HybridQ16, Qwen2_0_5B_TokenParity)
{
    // Full inference with HybridQ16 mode
    // Compare generated tokens to Hybrid mode
    // Allow for minor divergence (1-2 tokens over 100)
}

TEST(Test__E2E_HybridQ16, Qwen2_0_5B_LogitsParity)
{
    // Compare top-5 logits between HybridQ16 and Hybrid
    // Expected: same ranking, minor probability differences
}

TEST(Test__E2E_HybridQ16, AccumulationError)
{
    // Track residual stream error accumulation across layers
    // Verify error stays bounded (< threshold)
}
```

---

## Performance Targets

| Metric | Baseline (Hybrid) | Target (HybridQ16) | Target (Phase 5 Fused) |
|--------|-------------------|--------------------|-----------------------|
| Memory (residual) | 4 B/elem | 2.25 B/elem | 2.25 B/elem |
| Memory (proj buffers) | 4 B/elem | 1.125 B/elem | N/A (in registers) |
| ResidualAdd bandwidth | ~30 GB/s | ~17 GB/s | ~8 GB/s |
| Throughput (tok/s) | Baseline | ≥ 95% | ≥ 105% |
| Precision (vs FP32) | ~0.5% err | ~0.3% err | ~0.3% err |

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Precision degradation over many layers | High | Medium | Monitor residual error per layer; add error bounds |
| JIT register pressure | Medium | Medium | Careful register allocation; use stack spills if needed |
| Performance regression from requantization | Medium | Low | Fuse operations to minimize requant overhead |
| Numerical instability in RMSNorm | Medium | Low | Use double accumulator for sum-of-squares |
| Compatibility with existing Hybrid mode | Low | Low | Keep Hybrid mode unchanged; HybridQ16 is new mode |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|------------------|--------------|
| Phase 1: Foundation | 2-3 hours | None |
| Phase 2: Buffer Allocation | 2-3 hours | Phase 1 |
| Phase 3: Stage Implementations | 4-6 hours | Phase 1, 2 |
| Phase 4: RMSNorm Q16_1 | 4-6 hours | Phase 1 |
| Phase 5: JIT Fusion | 8-12 hours | Phase 1-4 |
| Phase 6: Integration Wiring | 4-6 hours | Phase 5 |
| Validation & Testing | 4-6 hours | All phases |
| **Total** | **28-42 hours** | |

---

## Appendix: Q16_1Block Memory Layout

```
Offset  Size  Field       Description
------  ----  ----------  ----------------------------------
0       4     d           FP32 scale factor
4       4     sum_qs      INT32 sum of quantized values
8       64    qs[32]      INT16 quantized values (32 elements)
------  ----  ----------  ----------------------------------
Total:  72 bytes per block (32 elements)
        2.25 bytes per element average
```

---

## Appendix: SIMD Intrinsics Reference

### Q16_1 Dequantization (AVX512)
```cpp
// Load int16[16] → int32[16] → FP32[16]
__m256i q16 = _mm256_loadu_si256((__m256i*)block.qs);
__m512i q32 = _mm512_cvtepi16_epi32(q16);
__m512 fp32 = _mm512_cvtepi32_ps(q32);
__m512 result = _mm512_mul_ps(fp32, _mm512_set1_ps(block.d));
```

### Q16_1 Quantization (AVX512)
```cpp
// FP32[16] → round → int32[16] → int16[16]
__m512 scaled = _mm512_mul_ps(values, _mm512_set1_ps(inv_scale));
__m512 rounded = _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT);
__m512i i32 = _mm512_cvtps_epi32(rounded);
__m256i i16 = _mm512_cvtepi32_epi16(i32);
```

### Horizontal Max Reduction (AVX512)
```cpp
float max_abs = _mm512_reduce_max_ps(_mm512_abs_ps(values));
```
