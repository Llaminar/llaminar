# Phase 8: Attention Kernel k_head_scales Usage

## Overview

**Goal**: Modify the attention kernel to use the `k_head_scales` array from the pipeline instead of extracting scales from K block headers.

**Why This Matters**: In HybridQ16 mode with K precision fix:
- GEMM outputs K as Q16_1 (256× finer than Q8_1)
- RoPE uses **dynamic per-head scale** for K (data-adaptive quantization)
- Each K position can have a different scale based on its value range
- The attention kernel needs these scales to compute correct Q×K^T scores

**Current Behavior (Bug)**:
```cpp
// Q16FusedAttentionKernel::compute() extracts scales from K block headers:
for (int kv_h = 0; kv_h < params.n_kv_heads; ++kv_h) {
    const size_t head_start_block = kv_h * effective_kv_stride * blocks_per_head;
    kv_scales[kv_h] = extract_d(params.K, head_start_block);  // ← Only first position!
}
```

This only gets the scale from the **first position** of each KV head, but with dynamic-scale RoPE, different positions have different scales. The `k_head_scales` array contains the correct per-position scales.

---

## Current Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           HybridQ16 K Precision Pipeline                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FusedQKVGEMMStage                                                           │
│  ├─ Q → Q8_1 (fixed kv_cache_scale)                                          │
│  ├─ K → Q16_1 (fixed kv_cache_scale, 256× finer precision)                   │
│  └─ V → Q8_1 (fixed kv_cache_scale)                                          │
│                                                                              │
│  RoPEStage (apply_mixed_q8_k16_to_q16)                                       │
│  ├─ Q: Q8_1 → Q16_1 (fixed scale, convert for kernel input)                  │
│  ├─ K: Q16_1 → Q16_1 (DYNAMIC per-head scale, outputs K_head_scales[])      │
│  └─ K_head_scales: [seq_len * n_kv_heads] per-position scale factors         │
│                                                                              │
│  FusedAttentionWoStage                                                       │
│  ├─ Q: Q16_1 blocks with fixed scale in block headers                        │
│  ├─ K: Q16_1 blocks with DYNAMIC scale in block headers (per-position)       │
│  ├─ k_head_scales: [seq_len * n_kv_heads] from RoPE (CURRENTLY UNUSED!)      │
│  └─ Need to use k_head_scales in Q×K^T computation                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Role |
|------|------|
| [Q16FusedAttentionKernel.cpp](../../src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp) | Converts `FusedAttentionWoParams` → `Q16IntegerAttentionParams`, dispatches to ref kernel |
| [Q16IntegerAttentionRef.h](../../src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h) | `Q16IntegerAttentionParams` struct with `kv_head_scales` |
| [Q16IntegerAttentionRef.cpp](../../src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp) | `q16_integer_attention_decode/prefill` using `get_qk_scale()` |
| [TensorKernels.h](../../../../src/v2/tensors/TensorKernels.h) | `FusedAttentionWoParams` with `k_head_scales` field |

---

## Implementation Plan

### Step 1: Extend Q16IntegerAttentionParams for Per-Position Scales

**Problem**: Current `kv_head_scales` is `[n_kv_heads]` (per-head). Need `[kv_len * n_kv_heads]` (per-position).

**File**: [Q16IntegerAttentionRef.h](../../src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h)

```cpp
struct Q16IntegerAttentionParams {
    // ... existing fields ...
    
    // === Per-head scale factors (from block headers - fallback) ===
    const float *q_head_scales = nullptr;   // [num_heads] 
    const float *kv_head_scales = nullptr;  // [num_kv_heads]
    
    // === NEW: Per-position K scale factors (from RoPE dynamic quantization) ===
    // When non-null, these override block-header scales for K.
    // Shape: [kv_len * num_kv_heads] (position-major: scales[pos * num_kv_heads + kv_h])
    const float *k_position_scales = nullptr;
    
    // === Helper methods ===
    
    /**
     * @brief Get K scale for specific position and KV head.
     * 
     * Uses k_position_scales if available (dynamic-scale from RoPE),
     * otherwise falls back to kv_head_scales (from block header).
     */
    float get_k_scale(int kv_pos, int kv_head) const {
        if (k_position_scales) {
            return k_position_scales[kv_pos * num_kv_heads + kv_head];
        }
        return kv_head_scales ? kv_head_scales[kv_head] : 1.0f;
    }
    
    /**
     * @brief Get QK scale for specific Q position, K position, and heads.
     * 
     * NEW: Per-position K scale support for HybridQ16 K precision fix.
     */
    float get_qk_scale_position(int q_head, int kv_head, int kv_pos) const {
        float s_q = q_head_scales ? q_head_scales[q_head] : 1.0f;
        float s_k = get_k_scale(kv_pos, kv_head);
        return s_q * s_k / std::sqrt(static_cast<float>(head_dim));
    }
};
```

### Step 2: Wire k_head_scales in Q16FusedAttentionKernel::compute()

**File**: [Q16FusedAttentionKernel.cpp](../../src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp)

Currently, the kernel extracts scales from block headers. We need to:
1. Check if `params.k_head_scales` is provided (HybridQ16 K precision mode)
2. If provided, pass it to `ref_params.k_position_scales`
3. If not provided, fall back to extracting from block headers (existing behavior)

```cpp
bool Q16FusedAttentionKernel::compute(
    const FusedAttentionWoParams& params,
    const MPIContext* mpi_ctx,
    int device_idx)
{
    // ... existing validation ...
    
    Q16IntegerAttentionParams ref_params;
    ref_params.Q = params.Q;
    ref_params.K = params.K;
    ref_params.V = params.V;
    
    // ================================================================
    // HEAD SCALES: Two modes depending on HybridQ16 K precision fix
    // ================================================================
    
    if (params.k_head_scales) {
        // Mode 1: HybridQ16 K precision fix - use per-position scales from RoPE
        // k_head_scales shape: [kv_len * n_kv_heads] (position-major)
        
        // Q uses fixed scale from block headers (unchanged)
        std::vector<float> q_scales(params.n_heads);
        for (int h = 0; h < params.n_heads; ++h) {
            const size_t head_start_block = static_cast<size_t>(h) * params.seq_len_q * blocks_per_head;
            q_scales[h] = extract_d(params.Q, head_start_block);
        }
        ref_params.q_head_scales = q_scales.data();
        
        // K uses dynamic per-position scales from RoPE
        ref_params.k_position_scales = params.k_head_scales;
        
        // Still need fallback kv_head_scales for V (uses same scale as K block headers)
        std::vector<float> kv_scales(params.n_kv_heads);
        for (int kv_h = 0; kv_h < params.n_kv_heads; ++kv_h) {
            const size_t head_start_block = static_cast<size_t>(kv_h) * effective_kv_stride * blocks_per_head;
            kv_scales[kv_h] = extract_d(params.K, head_start_block);  // For V scale
        }
        ref_params.kv_head_scales = kv_scales.data();
        
        LOG_DEBUG("Q16FusedAttentionKernel: Using per-position K scales from RoPE");
    } else {
        // Mode 2: Standard - extract scales from block headers (existing behavior)
        // ... existing code ...
    }
    
    // ... rest of function ...
}
```

### Step 3: Update Q×K^T Computation to Use Per-Position Scales (Option C: Pass Scale to Softmax)

**File**: [Q16IntegerAttentionRef.cpp](../../src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp)

The tricky part: Q×K^T dot products are computed in **pure integer**, then the scale is applied. With per-position K scales, we use **Option C: Pass scales to softmax**.

#### Why Option C?

The OnlineSoftmax state already incorporates `qk_scale` into its alpha computation. We can:
1. Compute raw integer scores (no scale applied yet)
2. Pass per-position K scale to softmax as `k_scales[k]`
3. Softmax applies: `scaled_score = score[k] * q_scale * k_scales[k] / sqrt(head_dim)`
4. Then: `exp(scaled_score - running_max)`

**Benefits**:
- Keeps Q×K^T dot product **pure integer** (no per-position float ops)
- Scale application happens at the boundary (score → softmax), natural place for FP32
- OnlineSoftmax already does FP32 math for exp/max tracking
- Minimal changes to existing microkernel structure

#### Implementation Details

**Current flow** (uniform scale):
```cpp
// OnlineSoftmaxState::init() pre-computes:
//   alpha = qk_scale * log2(e) * 2^frac_bits
// Then process_block applies:
//   scaled_score = score * alpha  (in fixed-point)
//   exp_result = exp2_lut(scaled_score - running_max)
```

**New flow** (per-position K scale):
```cpp
// OnlineSoftmaxState::init() now takes q_scale only:
//   base_alpha = q_scale / sqrt(head_dim) * log2(e) * 2^frac_bits
// Then process_block_with_k_scales applies:
//   per_position_alpha = base_alpha * k_scales[k]
//   scaled_score = score * per_position_alpha
//   exp_result = exp2_lut(scaled_score - running_max)
```

### Step 4: Update OnlineSoftmax for Per-Position K Scales

**File**: [OnlineSoftmax.h](../../src/v2/kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.h)

#### 4a. Modify OnlineSoftmaxState::init()

```cpp
struct OnlineSoftmaxState {
    // ... existing fields ...
    
    // NEW: Store components separately for per-position scale support
    float base_alpha_fp32;  // q_scale / sqrt(head_dim) * log2(e)
    
    /**
     * @brief Initialize for per-position K scale mode
     * 
     * @param q_scale Scale factor for Q head (from q_head_scales)
     * @param head_dim Head dimension (for 1/sqrt normalization)
     * @param frac_bits Fixed-point fractional bits
     * @param lut_value_bits LUT precision bits
     */
    void init_per_position(float q_scale, int head_dim, int frac_bits, int lut_value_bits) {
        this->frac_bits = frac_bits;
        this->lut_value_bits = lut_value_bits;
        
        // Pre-compute base alpha (without K scale)
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
        base_alpha_fp32 = q_scale * inv_sqrt_d * 1.4426950408889634f;  // log2(e)
        
        // Alpha will be computed per-position as: base_alpha_fp32 * k_scale * 2^frac_bits
        
        // Initialize running state
        running_max = std::numeric_limits<int32_t>::min();
        running_sum = 0;
        // ... other init ...
    }
};
```

#### 4b. Add process_block Overload with K Scales

```cpp
/**
 * @brief Process a block of KV positions with per-position K scales.
 * 
 * @param scores Raw INT32 scores from Q×K^T [block_count]
 * @param k_scales Per-position K scale factors [block_count]
 * @param block_count Number of KV positions in this block
 * @param kv_start Starting KV position index (for masking)
 * @param weights_out Output INT16 weights [block_count]
 * @param context INT32 context accumulator [head_dim]
 * @param V_blocks V tensor blocks
 * @param v_row_stride Stride between V rows
 * @param head_dim Head dimension
 * @param blocks_per_row Blocks per head
 */
template<typename BlockType>
void process_block_with_k_scales(
    const int32_t* scores,
    const float* k_scales,
    int block_count,
    int kv_start,
    int32_t* weights_out,
    int32_t* context,
    const BlockType* V_blocks,
    int v_row_stride,
    int head_dim,
    int blocks_per_row)
{
    // For each position in block:
    for (int i = 0; i < block_count; ++i) {
        // Compute per-position alpha
        float alpha_fp32 = base_alpha_fp32 * k_scales[i];
        int32_t alpha_fixed = static_cast<int32_t>(alpha_fp32 * (1 << frac_bits));
        
        // Scale score
        int64_t scaled = static_cast<int64_t>(scores[i]) * alpha_fixed;
        int32_t scaled_score = static_cast<int32_t>(scaled >> frac_bits);
        
        // Update running max and compute weight
        // ... (same as existing online softmax logic)
    }
}
```

#### 4c. Optimization: Batch K Scales with Similar Values

Most K scales within a head will be similar (dynamic quantization typically produces scales within 2× of each other). We can optimize:

```cpp
// If all k_scales in block are within 1% of each other, use uniform path
float min_scale = *std::min_element(k_scales, k_scales + block_count);
float max_scale = *std::max_element(k_scales, k_scales + block_count);

if (max_scale / min_scale < 1.01f) {
    // Use uniform scale path (existing fast path)
    float avg_scale = (min_scale + max_scale) * 0.5f;
    process_block_uniform(scores, avg_scale, ...);
} else {
    // Use per-position scale path
    process_block_with_k_scales(scores, k_scales, ...);
}
```

### Step 5: Wire K Scales Through Decode/Prefill Paths

**File**: [Q16IntegerAttentionRef.cpp](../../src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp)

The prefill path (`q16_integer_attention_prefill`) uses the same `get_qk_scale()` call and needs the same update for per-position K scales.

---

## Test Plan

### Unit Tests

1. **Test__Q16IntegerAttentionRef_PerPositionScales**
   - Create K tensor where different positions have different scales
   - Verify `get_k_scale()` returns correct per-position values
   - Verify `get_qk_scale_position()` computes correct product

2. **Test__Q16FusedAttentionKernel_KHeadScalesPassthrough**
   - Set `FusedAttentionWoParams.k_head_scales` 
   - Verify it's passed to `ref_params.k_position_scales`

### Integration Tests

1. **HybridQ16 vs FP32 Parity** (existing test)
   - Should show improvement from Hybrid "winning" to roughly equal
   - K_ROPE cosine should improve from 0.878 to >0.99

2. **End-to-End Token Match**
   - Run HybridQ16 inference with K precision fix
   - Compare token outputs to FP32 baseline
   - Target: >95% token match rate

---

## Files to Modify (Option C: Pass Scale to Softmax)

| File | Change | Lines (Est.) |
|------|--------|--------------|
| `Q16IntegerAttentionRef.h` | Add `k_position_scales`, `get_k_scale()`, helper methods | +35 |
| `Q16FusedAttentionKernel.cpp` | Branch on `params.k_head_scales`, wire to `ref_params.k_position_scales` | +45 |
| `OnlineSoftmax.h` | Add `init_per_position()`, `base_alpha_fp32` field, `process_block_with_k_scales()` | +80 |
| `Q16IntegerAttentionRef.cpp` | Update decode/prefill to call new softmax methods with k_scales | +50 |

**Total Estimate**: ~200-210 lines of changes

---

## Implementation Order (Option C)

```
Step 1: Q16IntegerAttentionParams changes
   ↓
Step 2: Q16FusedAttentionKernel::compute() branching
   ↓
Step 4a: OnlineSoftmaxState::init_per_position()
   ↓
Step 4b: process_block_with_k_scales()
   ↓
Step 4c: (Optional) Fast-path optimization for similar scales
   ↓
Step 5: Wire k_scales through decode path
   ↓
Step 5: Wire k_scales through prefill path
   ↓
Unit tests → Integration tests
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Performance regression from per-position scale lookups | Most scales are similar; can use branch to skip if correction ≈ 1.0 |
| Numerical drift from scale correction | Use FP32 for correction math, only final result goes to INT32 |
| KV cache doesn't store per-position scales for decode | K_head_scales buffer is pre-allocated for max_seq_len; cache write appends new scales |
| JIT kernel needs update too | Start with reference kernel; JIT can use same approach later |

---

## Verification

After implementation:

```bash
# Build
cmake --build build_v2 --parallel

# Unit tests
ctest --test-dir build_v2 -R "Q16.*Attention" --output-on-failure

# Integration test - should show improvement
ctest --test-dir build_v2 -R "V2_Integration_HybridQ16_FP32_Parity" --verbose

# Manual verification
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/llaminar2 \
    -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "The capital of France is" \
    -n 5 -t 0
```

Expected output: Same or very similar tokens as FP32 baseline.
