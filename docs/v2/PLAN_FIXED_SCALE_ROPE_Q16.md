# Plan: Fixed-Scale Q16 Quantization for RoPE (Q and K)

## Problem Statement

The current Q8→Q16 RoPE implementation uses **data-adaptive quantization**:
- Computes `max_abs = max(|dequant(Q8)|)` across all blocks in a head
- Sets `common_scale = max_abs * 1.415 / 127` (with sqrt(2) headroom for rotation)
- Output `block.d = common_scale / 256`

This creates **variable per-head scales** that don't match the attention kernel's expectations:

| Tensor | Expected Scale | Actual Scale (Observed) |
|--------|---------------|------------------------|
| Q | `8.0 / 32767` ≈ 2.44e-4 | 1.07e-3 to 2.45e-3 (varies per head) |
| K | `8.0 / 32767` ≈ 2.44e-4 | 0 to 4.01e-3 (varies per position) |
| V | `8.0 / 32767` ≈ 2.44e-4 | ≈ 2.44e-4 ✓ (fixed-scale from KV cache) |

The Q16IntegerAttentionRef kernel assumes `q_head_scales[h]` and `kv_head_scales[kv_h]` represent the block `d` values. When we pass a fixed `BLOCK_SCALE` but blocks have different actual scales, the FP32 conversion becomes incorrect.

---

## Solution: Fixed-Scale RoPE Q16 Quantization

### Design

Use a **fixed KV_CACHE_SCALE** (8.0) for Q16 quantization, matching V's fixed-scale path:

```cpp
const float KV_CACHE_SCALE = 8.0f;
const float d = KV_CACHE_SCALE / 32767.0f;  // ≈ 2.44e-4

// For each output Q16 block:
block.d = d;  // FIXED, not data-dependent
block.qs[i] = clamp(round(fp32_val / d), -max_safe_int16, max_safe_int16);
```

### Benefits

1. **Consistent scales** - All heads have the same known scale factor
2. **True integer attention** - No FP32 fallbacks needed for scale handling
3. **Pipeline coherence** - Q, K, V all use the same fixed scale
4. **Simpler kernel** - Attention kernel can use a single `BLOCK_SCALE` constant

### Tradeoffs

1. **Potential clipping** - Values outside [-8.0, +8.0] will clip
   - Mitigation: RoPE rotation can increase magnitude by up to sqrt(2), so we may need headroom
   - Analysis: With `max_val/127` adaptive scale, the Q8 range is typically ±1.0 to ±2.0
   - After rotation and upcasting to Q16, values should fit in ±8.0 with room to spare

2. **Loss of precision for small values** - Fixed scale may under-utilize int16 range
   - Mitigation: KV_CACHE_SCALE=8.0 is empirically tuned for typical activation magnitudes
   - Analysis: Qwen2 activations typically stay in ±2.0 range, giving ~25% utilization

---

## Implementation Plan

### Phase 1: Add Fixed-Scale RoPE Primitives

**File**: `src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp`

Add new functions:
```cpp
// Fixed-scale variant: no per-head max_abs computation
template <typename OutBlockType>
void apply_rope_q8_1_to_q16_head_fixed_scale(
    const Q8_1Block *q8_in,
    OutBlockType *q16_out,
    int head_dim,
    const int16_t *cos_q15,
    const int16_t *sin_q15,
    float fixed_kv_cache_scale);  // e.g., 8.0f

// High-level wrapper
template <typename OutBlockType>
void apply_rope_q8_1_to_q16_fixed_scale(
    const Q8_1Block *Q_in,
    const Q8_1Block *K_in,
    OutBlockType *Q_out,
    OutBlockType *K_out,
    const int *position_ids,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta,
    float kv_cache_scale);  // e.g., 8.0f
```

### Phase 2: Update Kernel Interface

**File**: `src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp`

Add method to `ITensorRoPE` interface:
```cpp
bool apply_q8_1_to_q16_1_fixed_scale(
    TensorBase *Q_in,
    TensorBase *K_in,
    TensorBase *Q_out,
    TensorBase *K_out,
    const int *position_ids,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta,
    float kv_cache_scale,  // NEW: fixed scale parameter
    const MPIContext *mpi_ctx,
    int device_idx);
```

### Phase 3: Update RoPE Stage

**File**: `src/v2/execution/compute_stages/stages/RoPEStage.cpp`

Add `kv_cache_scale` parameter to `RoPEStage::Params`:
```cpp
struct Params {
    // ... existing fields ...
    float kv_cache_scale = 8.0f;  // For fixed-scale Q16 quantization
};
```

Update HybridQ16 path:
```cpp
if (hybrid_q16_mode)
{
    LOG_DEBUG("[RoPEStage] Using HybridQ16 mode with fixed scale: " << params_.kv_cache_scale);
    return kernel->apply_q8_1_to_q16_1_fixed_scale(
        params_.Q, params_.K,
        params_.Q_out, params_.K_out,
        position_ids.data(),
        seq_len, params_.n_heads, n_kv_heads, params_.head_dim,
        params_.theta_base,
        params_.kv_cache_scale,  // Fixed scale
        params_.mpi_ctx, params_.device_idx);
}
```

### Phase 4: Update GraphOrchestrator

**File**: `src/v2/execution/GraphOrchestrator.cpp`

Pass `kv_cache_scale` to RoPEStage when creating HybridQ16 pipeline:
```cpp
RoPEStage::Params rope_params;
// ... existing setup ...
rope_params.kv_cache_scale = config_.kv_cache_scale;  // e.g., 8.0f
```

### Phase 5: Simplify Attention Kernel

**File**: `src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp`

With fixed scales, the kernel becomes simpler:
```cpp
// All tensors use the same fixed scale
constexpr float KV_CACHE_SCALE = 8.0f;
constexpr float BLOCK_SCALE = KV_CACHE_SCALE / 32767.0f;

std::vector<float> q_scales(params.n_heads, BLOCK_SCALE);
std::vector<float> kv_scales(params.n_kv_heads, BLOCK_SCALE);
ref_params.q_head_scales = q_scales.data();
ref_params.kv_head_scales = kv_scales.data();
```

---

## Algorithm: Pure Integer Fixed-Scale Q8→Q16 RoPE

The key insight is that we can compute a **per-block scale ratio** in fixed-point, then perform all per-element operations in pure integer.

### Scale Ratio Fixed-Point Representation

For each Q8_1 input block:
- `d_block` = FP16 scale (variable per block)
- `d_fixed` = KV_CACHE_SCALE / 32767 ≈ 2.44e-4 (constant)
- `scale_ratio = d_block / d_fixed` (typically 0.4 to 4.0)

We represent the scale ratio in Q16 fixed-point:
```
scale_ratio_q16 = round(d_block / d_fixed * 65536)
```

This allows ~16 bits of precision for the ratio, which is more than enough.

### Per-Element Integer Rescaling

To rescale Q8 value to the fixed scale:
```
val_q8 = qs[i]                           // int8 from input block
val_scaled = (val_q8 * scale_ratio_q16) >> 16  // int32 arithmetic
```

This produces a value in the fixed-scale representation without per-element FP32 ops.

### Integer RoPE Rotation

With cos/sin in Q15 format (int16 / 32767):
```
// All int32 arithmetic:
x_rotated = (x_scaled * cos_q15 - y_scaled * sin_q15 + 16384) >> 15
y_rotated = (x_scaled * sin_q15 + y_scaled * cos_q15 + 16384) >> 15
```

The `+ 16384` provides rounding before the right shift.

### Full Algorithm

```cpp
template <typename OutBlockType>
void apply_rope_q8_1_to_q16_head_fixed_scale_integer(
    const Q8_1Block *q8_in,
    OutBlockType *q16_out,
    int head_dim,
    const int16_t *cos_q15,
    const int16_t *sin_q15,
    float kv_cache_scale)  // e.g., 8.0f
{
    constexpr int Q8_BLOCK_SIZE = 32;
    constexpr int OUT_BLOCK_SIZE = OutBlockType::BLOCK_SIZE;
    const int q8_blocks_per_head = head_dim / Q8_BLOCK_SIZE;
    const int half_dim = head_dim / 2;
    
    // Fixed output scale
    const float d_fixed = kv_cache_scale / 32767.0f;
    const float inv_d_fixed = 32767.0f / kv_cache_scale;
    
    // Step 1: Compute per-block scale ratios (Q16 fixed-point)
    // This is O(blocks_per_head) FP32 ops, NOT per-element
    std::vector<int32_t> scale_ratios(q8_blocks_per_head);
    for (int b = 0; b < q8_blocks_per_head; ++b)
    {
        float d_block = fp16_to_fp32_rope(q8_in[b].d);
        // scale_ratio_q16 = d_block / d_fixed * 65536
        float ratio = d_block * inv_d_fixed;  // d_block / d_fixed
        scale_ratios[b] = static_cast<int32_t>(ratio * 65536.0f + 0.5f);
    }
    
    // Step 2: Rescale all Q8 values to fixed scale (pure integer per-element)
    std::vector<int16_t> scaled(head_dim);
    for (int b = 0; b < q8_blocks_per_head; ++b)
    {
        const int32_t ratio = scale_ratios[b];
        for (int i = 0; i < Q8_BLOCK_SIZE; ++i)
        {
            int32_t val = q8_in[b].qs[i];  // int8
            // Rescale: val * ratio >> 16, with rounding
            int32_t rescaled = (val * ratio + 32768) >> 16;
            // Clamp to safe int16 range for subsequent multiply
            rescaled = std::clamp(rescaled, -16384, 16383);
            scaled[b * Q8_BLOCK_SIZE + i] = static_cast<int16_t>(rescaled);
        }
    }
    
    // Step 3: Apply RoPE rotation in pure integer (Q15 sin/cos)
    std::vector<int16_t> rotated(head_dim);
    for (int i = 0; i < half_dim; ++i)
    {
        int32_t x = scaled[i];
        int32_t y = scaled[i + half_dim];
        int32_t c = cos_q15[i];  // Q15
        int32_t s = sin_q15[i];  // Q15
        
        // x' = x*cos - y*sin, scaled by 2^15
        // y' = x*sin + y*cos, scaled by 2^15
        int32_t x_rot = (x * c - y * s + 16384) >> 15;  // Round and descale
        int32_t y_rot = (x * s + y * c + 16384) >> 15;
        
        // Clamp to safe Q16 range (for VNNI accumulation)
        rotated[i] = static_cast<int16_t>(std::clamp(x_rot, -16384, 16383));
        rotated[i + half_dim] = static_cast<int16_t>(std::clamp(y_rot, -16384, 16383));
    }
    
    // Step 4: Pack into Q16_1 output blocks with FIXED scale
    const int q16_blocks_per_head = head_dim / OUT_BLOCK_SIZE;
    for (int b = 0; b < q16_blocks_per_head; ++b)
    {
        OutBlockType &out = q16_out[b];
        out.d = d_fixed;  // FIXED scale for ALL blocks
        
        int32_t sum_qs = 0;
        for (int i = 0; i < OUT_BLOCK_SIZE; ++i)
        {
            out.qs[i] = rotated[b * OUT_BLOCK_SIZE + i];
            sum_qs += out.qs[i];
        }
        out.sum_qs = sum_qs;
    }
}
```

### Operation Count Analysis

| Operation | Count | Type |
|-----------|-------|------|
| FP16→FP32 conversion | `head_dim/32` (~4 for 128-dim) | Per-block |
| FP32 ratio computation | `head_dim/32` | Per-block |
| Integer rescale | `head_dim` | Per-element (int32 mul + shift) |
| Integer rotation | `head_dim` | Per-element (int32 mul/add + shift) |
| Output packing | `head_dim` | Per-element (int16 store) |

**Total FP32 ops**: O(head_dim/32) = ~4 ops per head
**Total integer ops**: O(3 * head_dim) = ~384 ops per head

This is essentially **pure integer** - the FP32 work is amortized over 32 elements per block.

### SIMD Optimization Opportunities

The integer operations are highly vectorizable:
- **AVX-512**: 16x int32 rescale in parallel
- **AVX-512**: 16x int32 rotation in parallel
- **VNNI**: Could fuse rescale + rotation with `vpdpwssd`

---

## Testing Plan

### Unit Tests
1. **Clipping test**: Verify no clipping for typical Qwen2 activation ranges
2. **Round-trip test**: Q8→Q16 RoPE→dequant matches FP32 RoPE within tolerance
3. **Scale verification**: All output blocks have `d == kv_cache_scale / 32767`

### Integration Tests
1. **HybridQ16 parity**: ATTENTION_CONTEXT cosine similarity vs FP32 should improve dramatically
2. **Layer-by-layer**: Compare all 24 layers for consistency

### E2E Tests
1. **Token prediction**: Greedy sampling should produce same tokens as FP32 reference
2. **Perplexity**: Measure perplexity degradation (expected: <1% increase)

---

## Estimated Effort

| Phase | Effort | Files Modified |
|-------|--------|----------------|
| Phase 1: RoPE primitives | 2-3 hours | RoPEPrimitives.cpp, RoPEPrimitives.h |
| Phase 2: Kernel interface | 1 hour | ITensorRoPE.h, CPURoPEKernelT.cpp |
| Phase 3: RoPE stage | 30 min | RoPEStage.h, RoPEStage.cpp |
| Phase 4: GraphOrchestrator | 30 min | GraphOrchestrator.cpp |
| Phase 5: Attention kernel | 15 min | Q16FusedAttentionKernel.cpp |
| Unit tests | 1 hour | Test__Q8_1_to_Q16_RoPE_FixedScale.cpp |
| Integration tests | 30 min | Test__HybridQ16Pipeline_vs_FP32.cpp |

**Total: ~6 hours**

---

## Alternative: Fused VNNI Rescale+Rotate

For maximum performance, we could fuse the rescale and rotation into a single VNNI-accelerated pass:

```cpp
// Using vpdpwssd for dot-product-like operations:
// 1. Pack [x, y] pairs into int16 vectors
// 2. Pack [cos, -sin] and [sin, cos] as "weight" vectors
// 3. Use VNNI to compute rotated values in one instruction

// This requires careful layout but can achieve 2x throughput
```

This is an optimization for Phase 2 once the basic integer path is verified correct.

---

## Conclusion

Pure integer fixed-scale Q16 quantization for RoPE is the cleanest solution to the scale mismatch problem. It:
1. Aligns Q and K with V's fixed-scale quantization
2. Keeps all per-element operations in integer domain
3. Only uses O(head_dim/32) FP32 ops for per-block scale ratio computation
4. Enables true integer attention without scale lookups
5. Is highly vectorizable with AVX-512 and VNNI

The main risk is potential clipping for outlier activations, but the ±16384 safe range (with kv_cache_scale=8.0) covers values up to ±4.0, which is sufficient for typical transformer activations.
