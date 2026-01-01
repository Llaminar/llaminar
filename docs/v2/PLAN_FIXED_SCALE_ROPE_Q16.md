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

---

# ADDENDUM: KV Cache Layout Refactor for Q16_1

## Problem Statement (2025-01-01)

After implementing fixed-scale RoPE quantization (Phases 1-3 above), the `V2_Integration_HybridQ16Pipeline_vs_FP32` test still fails with **~0.0 cosine similarity** for `ATTENTION_CONTEXT`.

**Root Cause**: Tensor layout mismatch between KV cache and attention kernel.

### Layout Analysis

**UnifiedKVCache** stores K/V in **position-major** layout:
```
Storage: [max_seq_len, n_kv_heads * head_dim]  → each row is one position

Memory layout (example: kv_len=4, n_kv_heads=2, head_dim=64):
Row 0: [head_0_pos_0, head_1_pos_0]  → 2 Q16 blocks
Row 1: [head_0_pos_1, head_1_pos_1]  → 2 Q16 blocks
Row 2: [head_0_pos_2, head_1_pos_2]  → 2 Q16 blocks
Row 3: [head_0_pos_3, head_1_pos_3]  → 2 Q16 blocks

Block indexing: block[p][h] = blocks[p * n_kv_heads + h]
```

**Q16IntegerAttentionRef** expects **head-major** layout:
```cpp
// Line 259 in Q16IntegerAttentionRef.cpp:
const Q16_1Block_64 *K_head = K + kv_h * kv_len * blocks_per_row;

Memory layout expected:
Block 0-3:   head_0, positions 0-3
Block 4-7:   head_1, positions 0-3

Block indexing: block[h][p] = blocks[h * kv_len + p]
```

**Result**: The kernel reads the wrong K/V data, producing garbage attention scores.

### Why Unit Tests Pass But Integration Fails

The unit tests explicitly transpose K/V before passing to the kernel:
```cpp
// In Test__Q16IntegerAttentionParity.cpp (line 382):
// Transpose K/V from [kv_len, num_kv_heads, head_dim] to [num_kv_heads, kv_len, head_dim]
auto K_transposed = transposeKV(K_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
auto V_transposed = transposeKV(V_fp32, KV_LEN, NUM_KV_HEADS, HEAD_DIM);
```

The integration test uses the KV cache directly (without transpose), exposing the mismatch.

---

## Solution Options

### Option A: Fix Kernel with Strided Access (Quick Fix)

Modify `Q16IntegerAttentionRef` to use strided indexing:
```cpp
// Current (WRONG):
const Q16_1Block_64 *K_head = K + kv_h * kv_len * blocks_per_row;
K_block_p = K_head[p * blocks_per_row];

// Fixed (correct strided access):
// Layout: blocks[p * n_kv_heads + kv_h]
K_block_p = K[(p * num_kv_heads + kv_h) * blocks_per_row];
```

**Pros**: Quick fix, minimal code changes
**Cons**: 
- Non-contiguous access for each head → cache inefficiency
- ~2× memory bandwidth for n_kv_heads=2
- Does not scale well for larger n_kv_heads

### Option B: Head-Major KV Cache Layout (Recommended)

Add a **separate head-major KV cache** for Q16_1 mode, optimized for the attention kernel's access pattern.

**New layout**: `[n_kv_heads, max_seq_len, head_dim]` stored as separate per-head tensors.

**Architecture**:
```cpp
// New specialized cache for Q16_INTEGER backend
class HeadMajorKVCache : public IUnifiedKVCache {
    // Storage: n_kv_heads separate tensors per layer
    // entries_[layer][kv_head] = { K: [max_seq_len, head_dim], V: [max_seq_len, head_dim] }
    std::vector<std::vector<HeadMajorKVCacheEntry>> entries_;
    
    // Interface: return contiguous K/V for a single head
    Q16_1Tensor* get_k_head(int layer, int kv_head, int seq_idx = 0);
    Q16_1Tensor* get_v_head(int layer, int kv_head, int seq_idx = 0);
};
```

**Pros**:
- Optimal memory access pattern (contiguous per-head)
- Kernel code stays simple (no stride calculations)
- Memory layout matches compute pattern
- Better cache locality during attention computation

**Cons**:
- New class to implement and maintain
- Need to update FusedAttentionWoStage to use new interface
- Append operation more complex (multiple per-head appends)

---

## Recommended Approach: Option B (Head-Major Layout) via Layout Mode

For production, head-major layout is the right design:
1. **Performance**: Contiguous access per head is critical for vectorization
2. **Simplicity**: Kernel code remains straightforward
3. **Scalability**: Works well for any n_kv_heads value
4. **Unified Implementation**: Single `UnifiedKVCache` class with selectable layout mode

---

## Implementation Plan: UnifiedKVCache Layout Mode

**Key Change**: Instead of a separate `HeadMajorKVCache` class, we add a **layout mode** parameter to the existing `UnifiedKVCache` class. This keeps the codebase unified and avoids duplication.

### Phase 6: Add Layout Mode to UnifiedKVCache

**Modified File**: `src/v2/tensors/UnifiedKVCache.h`

```cpp
/**
 * @brief KV cache storage layout mode
 */
enum class KVCacheLayoutMode {
    POSITION_MAJOR,  ///< [position, n_kv_heads, head_dim] - cache-friendly for append
    HEAD_MAJOR       ///< [n_kv_heads, position, head_dim] - attention-friendly for Q16
};

/**
 * @brief Unified KV cache with selectable storage layout.
 *
 * Layout modes:
 * - POSITION_MAJOR: [max_seq_len, n_kv_heads * head_dim] - optimal for sequential append
 * - HEAD_MAJOR: n_kv_heads separate [max_seq_len, head_dim] tensors - optimal for Q16 attention
 *
 * The layout mode is selected at construction time based on the attention backend.
 */
template <ActivationPrecision Precision>
class UnifiedKVCacheTensor : public IUnifiedKVCache {
public:
    // Constructor with layout mode selection
    UnifiedKVCacheTensor(
        const MPIContext& mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_idx,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR
    );
    
    // Layout query
    TensorLayout kv_layout() const override;
    KVCacheLayoutMode layout_mode() const { return layout_mode_; }
    
    // ================================================================
    // Head-specific accessors (HEAD_MAJOR mode)
    // ================================================================
    
    /**
     * @brief Get K tensor for a specific KV head (HEAD_MAJOR mode only)
     * @return Tensor view of shape [cached_tokens, head_dim], or nullptr if POSITION_MAJOR
     */
    TensorT* get_k_head(int layer, int kv_head, int seq_idx = 0);
    TensorT* get_v_head(int layer, int kv_head, int seq_idx = 0);
    
    /// Stride in bytes between consecutive heads (HEAD_MAJOR mode)
    size_t head_stride_bytes() const;
    
private:
    KVCacheLayoutMode layout_mode_;
    
    // For HEAD_MAJOR: separate per-head tensors
    struct HeadEntry {
        std::shared_ptr<TensorT> K;  // [max_seq_len, head_dim]
        std::shared_ptr<TensorT> V;
    };
    std::vector<std::vector<HeadEntry>> head_entries_;  // [layer][kv_head]
    
    // For POSITION_MAJOR: existing flat storage (unchanged)
};
```

**Key Implementation Details**:
1. **Constructor**: Check `layout_mode_` and allocate storage accordingly
   - POSITION_MAJOR: Single flat tensor `[max_seq_len, n_kv_heads * head_dim]` per layer
   - HEAD_MAJOR: `n_kv_heads` separate tensors `[max_seq_len, head_dim]` per layer
2. **append_kv**: Dispatch to appropriate append logic based on mode
3. **get_k_head/get_v_head**: Return per-head tensor (HEAD_MAJOR) or nullptr (POSITION_MAJOR)
4. **get_k_base/get_v_base**: Return correct view based on layout mode

### Phase 7: Update FusedAttentionWoStage

**File**: `src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp`

For Q16_INTEGER backend, check cache layout and use head-specific accessors:
```cpp
if (params_.backend == FusedAttentionBackend::Q16_INTEGER)
{
    // Check if cache supports head-major layout (no transpose needed)
    if (params_.kv_cache && 
        params_.kv_cache->kv_layout() == TensorLayout::KV_HEAD_POS_DIM)
    {
        auto* cache = dynamic_cast<UnifiedKVCacheTensor<ActivationPrecision::Q16_1>*>(
            params_.kv_cache);
        
        // Direct access to head-major K/V - no transpose!
        q16_params.K = cache->get_k_head(layer, 0)->typed_data();
        q16_params.V = cache->get_v_head(layer, 0)->typed_data();
        q16_params.kv_head_stride = cache->head_stride_bytes();
        
        LOG_DEBUG("[FusedAttentionWoStage] Using head-major KV cache (no transpose)");
    }
    else
    {
        // Fallback: transpose workaround for position-major cache
        LOG_WARN("[FusedAttentionWoStage] KV cache is position-major, applying transpose");
        // ... existing transpose code ...
    }
}
```

### Phase 8: Update Q16IntegerAttentionRef (Optional Simplification)

With head-major layout, the kernel can directly iterate through contiguous head data:
```cpp
// Old (wrong with position-major):
const Q16_1Block_64 *K_head = K + kv_h * kv_len * blocks_per_row;

// New (correct with head-major):
const Q16_1Block_64 *K_head = K + kv_h * head_stride_blocks;
// where head_stride_blocks = kv_len * blocks_per_head
```

### Phase 9: Update GraphOrchestrator

**File**: `src/v2/pipelines/qwen/GraphOrchestrator.cpp`

Select layout mode based on attention backend:
```cpp
// Select KV cache layout based on attention backend
KVCacheLayoutMode kv_layout_mode = 
    (config_.attention_backend == FusedAttentionBackend::Q16_INTEGER)
        ? KVCacheLayoutMode::HEAD_MAJOR    // Q16 kernel needs head-major
        : KVCacheLayoutMode::POSITION_MAJOR;  // Other backends use position-major

kv_cache_ = createUnifiedKVCache(
    config_.kv_cache_precision,
    mpi_ctx_, n_layers_, batch_size_, max_seq_len_,
    n_kv_heads_, head_dim_, device_idx_,
    kv_layout_mode);  // Pass layout mode

LOG_INFO("[GraphOrchestrator] Created KV cache with layout mode: " 
         << (kv_layout_mode == KVCacheLayoutMode::HEAD_MAJOR 
             ? "HEAD_MAJOR" : "POSITION_MAJOR"));
```

**Benefits of Unified Approach** (vs. separate HeadMajorKVCache class):
- **Single class to maintain**: No code duplication between cache implementations
- **Existing tests work**: All `UnifiedKVCache` tests remain valid
- **Clear intent**: Layout mode is explicit in constructor
- **Backward compatible**: Default is POSITION_MAJOR (existing behavior)
- **Flexible**: Can switch modes at runtime based on attention backend

---

## Memory Layout Comparison

### Position-Major (Current UnifiedKVCache)

```
Logical: K[position][kv_head][element_in_head]
Memory:  K[p * kv_dim + h * head_dim + e]

For kv_len=4, n_kv_heads=2, head_dim=64, blocks_per_head=1 (Q16_1Block_64):
Block layout:
  Block[0]: pos=0, head=0
  Block[1]: pos=0, head=1  
  Block[2]: pos=1, head=0
  Block[3]: pos=1, head=1
  Block[4]: pos=2, head=0
  Block[5]: pos=2, head=1
  Block[6]: pos=3, head=0
  Block[7]: pos=3, head=1

Accessing head 0 across positions: Block[0], Block[2], Block[4], Block[6]  → STRIDED
```

### Head-Major (Proposed HeadMajorKVCache)

```
Logical: K[kv_head][position][element_in_head]
Memory:  K[h * (kv_len * head_dim) + p * head_dim + e]

For kv_len=4, n_kv_heads=2, head_dim=64:
Head 0 tensor: [pos_0, pos_1, pos_2, pos_3]  → CONTIGUOUS
Head 1 tensor: [pos_0, pos_1, pos_2, pos_3]  → CONTIGUOUS

Accessing head 0 across positions: head_0_tensor[0..3]  → CONTIGUOUS
```

---

## Effort Estimate

| Phase | Effort | Files |
|-------|--------|-------|
| Phase 6: HeadMajorKVCache class | 3-4 hours | HeadMajorKVCache.h/.cpp |
| Phase 7: FusedAttentionWoStage update | 1 hour | FusedAttentionWoStage.cpp |
| Phase 8: Q16IntegerAttentionRef (optional) | 30 min | Q16IntegerAttentionRef.cpp |
| Phase 9: GraphOrchestrator update | 30 min | GraphOrchestrator.cpp |
| Unit tests | 1 hour | Test__HeadMajorKVCache.cpp |
| Integration test fix | 30 min | Test__HybridQ16Pipeline_vs_FP32.cpp |

**Total: ~7 hours**

---

## Alternative: Quick Fix for Validation

Before implementing the full HeadMajorKVCache, we can validate the approach with a **transpose-on-read** workaround:

**In FusedAttentionWoStage.cpp** (Q16_INTEGER path):
```cpp
// Allocate temporary head-major buffers
std::vector<Q16_1Block_64> K_transposed(kv_len * n_kv_heads);
std::vector<Q16_1Block_64> V_transposed(kv_len * n_kv_heads);

// Transpose from position-major to head-major
for (int h = 0; h < n_kv_heads; ++h) {
    for (int p = 0; p < kv_len; ++p) {
        // Source: position-major [p][h]
        // Dest: head-major [h][p]
        K_transposed[h * kv_len + p] = K_cache[(p * n_kv_heads + h)];
        V_transposed[h * kv_len + p] = V_cache[(p * n_kv_heads + h)];
    }
}

q16_params.K = K_transposed.data();
q16_params.V = V_transposed.data();
```

**Pros**: Quick validation, no new classes
**Cons**: O(kv_len * n_kv_heads) copy per layer per step, not viable for production

This is useful to **confirm the layout is the root cause** before investing in HeadMajorKVCache.

---

## Conclusion

The KV cache layout mismatch is the root cause of the HybridQ16 integration test failure. The recommended solution is:

1. **Short term**: Use transpose-on-read workaround to validate the fix
2. **Long term**: Implement `HeadMajorKVCache` class for optimal Q16_1 attention performance

This layout refactor complements the fixed-scale RoPE work (Phases 1-5) to complete the HybridQ16 integer attention pipeline.

---

# ADDENDUM 2: Wo Projection and Residual Add Implementation (2025-01-01)

## Problem Statement

Investigation of `V2_Integration_HybridQ16Pipeline_vs_FP32` test failure revealed:

| Stage | HybridQ16 Cosine | Hybrid Cosine | Status |
|-------|------------------|---------------|--------|
| `ATTENTION_CONTEXT` | ~0.05 | ~0.90+ | Wrong (Q layout issue) |
| `ATTENTION_OUTPUT` | **0.000000** | ~0.93+ | **NOT IMPLEMENTED** |
| `ATTENTION_RESIDUAL` | Variable | ~0.95+ | Depends on above |

**Root Cause**: The `Q16IntegerAttentionRef` kernel has **TODO placeholders** for Steps 4 and 5:

```cpp
// In Q16IntegerAttentionRef.cpp, lines ~444-460 (decode) and ~780-790 (prefill):

// ================================================================
// Step 4: Wo projection → Q16_1 output
// ================================================================

// TODO(v2): Implement Wo projection with VPDPWSSD
// For now, this is a placeholder
// The Wo projection should:
// 1. Take INT32 context [head_dim]
// 2. Multiply by packed Wo weights
// 3. Produce Q16_1 output [d_model]

// ================================================================
// Step 5: Residual add (integer)
// ================================================================

// TODO(v2): Implement integer residual add
// Should handle scale alignment between attention output and residual
```

**Key Discovery**: The `WoProjection.h` microkernel header declares all necessary functions, and `WoProjection.cpp` implements them (802 lines), but they are **never called** from `Q16IntegerAttentionRef.cpp`!

---

## What's Already Implemented (Available Microkernels)

### WoProjection.h/cpp (802 lines) - COMPLETE

| Function | Purpose | Status |
|----------|---------|--------|
| `q16_context_normalize_to_int16()` | INT32→INT16 normalization | ✅ Implemented |
| `q16_wo_row_gemv_tiled<BlockType>()` | Single-row GEMV (tiled) | ✅ Implemented |
| `q16_wo_projection<BlockType>()` | Full Wo projection (GEMV) | ✅ Implemented |
| `q16_wo_projection_batched<BlockType>()` | Batched Wo (GEMM for prefill) | ✅ Implemented |
| `q16_quantize_to_q16_1<BlockType>()` | INT32→Q16_1 output | ✅ Implemented |
| `wo_projection_vnni_int16()` | VNNI-accelerated Wo | ✅ Implemented |
| `wo_projection_vnni_int16_batched()` | Batched VNNI Wo | ✅ Implemented |
| `wo_gemv_row_vnni_int16()` | Single-row VNNI GEMV | ✅ Implemented |

### What's Missing: Wiring in Q16IntegerAttentionRef

The reference kernel computes attention context but **stops before Wo projection**:

1. ❌ Call to `wo_projection_vnni_int16()` after context computation
2. ❌ Integer residual add (`output_q16 + residual_in → residual_out`)
3. ❌ Snapshot capture for `wo_output_snapshot` and `residual_out_snapshot`

---

## Implementation Plan: Wire Wo Projection + Residual Add

### Phase 7: Complete Flash Decode Path (seq_len_q=1)

**File**: `src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp`

Replace TODO at ~line 444-460 with:

```cpp
// ================================================================
// Step 4: Wo projection → Q16_1 output
// ================================================================

// After processing all heads, concatenate contexts for Wo projection.
// Input: per-head INT32 context [num_heads][head_dim]
// Output: Q16_1 blocks [d_model / block_size]

// Allocate concatenated context buffer (one-time per decode step)
std::vector<int32_t> concat_context(params.d_model);

// Concatenate all head contexts into d_model vector
// (This happens outside the head loop - after all heads processed)
```

Then after the head loop:

```cpp
// Concatenate head contexts (done inside head loop by writing to concat_context)
// Each head writes context[head_dim] to concat_context[h * head_dim]

// Call VNNI Wo projection microkernel
float context_scale = /* computed from PV accumulation */;

microkernels::wo_projection_vnni_int16(
    concat_context.data(),       // INT32 context [d_model]
    context_scale,               // Scale factor
    params.Wo_packed,            // VNNI-packed Wo weights
    params.output,               // Q16_1 output [d_model / block_size]
    params.d_model,              // Output dimension
    params.d_model,              // Input dimension (num_heads * head_dim = d_model)
    params.block_size,           // Q16 block size
    params.snapshot_wo_output    // Optional FP32 snapshot buffer
);

// ================================================================
// Step 5: Residual add (integer)
// ================================================================

// Add Wo output to residual_in, write to residual_out
// All three are Q16_1 format with the same fixed scale (8.0 / 32767)

auto *res_in = reinterpret_cast<const Q16_1Block_64*>(params.residual_in);
auto *wo_out = reinterpret_cast<const Q16_1Block_64*>(params.output);
auto *res_out = reinterpret_cast<Q16_1Block_64*>(params.residual_out);

const int blocks_per_output = (params.d_model + 63) / 64;
for (int b = 0; b < blocks_per_output; ++b) {
    // Same scale → simple integer add
    res_out[b].d = res_in[b].d;  // Preserve scale
    for (int i = 0; i < 64; ++i) {
        // Saturating add to prevent overflow
        int32_t sum = static_cast<int32_t>(res_in[b].qs[i]) + 
                      static_cast<int32_t>(wo_out[b].qs[i]);
        res_out[b].qs[i] = static_cast<int16_t>(
            std::clamp(sum, -32768, 32767));
    }
}

// Snapshot: residual output
if (params.snapshot_residual_out) {
    // Dequantize to FP32 for snapshot
    for (int b = 0; b < blocks_per_output; ++b) {
        for (int i = 0; i < 64; ++i) {
            params.snapshot_residual_out[b * 64 + i] = 
                res_out[b].qs[i] * res_out[b].d;
        }
    }
}
```

### Phase 8: Complete FA2 Prefill Path (seq_len_q > 1)

Similar changes needed in `q16_integer_attention_prefill()` at ~line 780-790.

Key differences:
- Use `wo_projection_vnni_int16_batched()` for batch efficiency
- Context array is `[batch_size, d_model]`
- Residual add loops over `batch_size` rows

### Phase 9: Context Accumulation Fix

Currently context is computed **per-head** but Wo expects **concatenated** context.
Need to:
1. Allocate `concat_context[d_model]` outside head loop
2. Inside head loop: write to `concat_context[h * head_dim : (h+1) * head_dim]`
3. After head loop: call Wo projection on full `concat_context`

### Phase 10: Scale Alignment Verification

The residual add assumes matching scales:
- **residual_in**: Q16_1 with d = `kv_cache_scale / 32767` (from previous layer)
- **Wo output**: Q16_1 with d = ??? (from Wo projection)

**Critical Question**: Does `wo_projection_vnni_int16()` output Q16_1 with matching scale?

Need to verify or add parameter for output scale:
```cpp
// In wo_projection_vnni_int16():
// Option 1: Use fixed output scale matching residual
constexpr float OUTPUT_SCALE = 8.0f / 32767.0f;

// Option 2: Compute data-adaptive scale (may need rescaling in residual add)
```

---

## Testing Plan

### Unit Test: Wo Projection Wiring

```cpp
TEST(Test__Q16IntegerAttention, WoProjectionProducesNonZeroOutput) {
    // Setup: Create Q16 attention params with known inputs
    // Execute: q16_integer_attention_reference(params)
    // Assert: params.output is not all zeros
    // Assert: Wo output has reasonable FP32 magnitude
}
```

### Unit Test: Residual Add

```cpp
TEST(Test__Q16IntegerAttention, ResidualAddCorrectness) {
    // Setup: Known Wo output Q16_1 + known residual_in Q16_1
    // Execute: q16_integer_attention_reference(params)
    // Assert: residual_out[i] ≈ wo_out[i] + res_in[i] (within quantization error)
}
```

### Integration Test: Full Pipeline

```cpp
TEST(Test__HybridQ16Pipeline, AttentionOutputNonZero) {
    // Run HybridQ16 pipeline prefill
    // Check ATTENTION_OUTPUT snapshot is NOT all zeros
    // Check ATTENTION_OUTPUT cosine vs FP32 > 0.9
}
```

---

## Estimated Effort

| Phase | Effort | Description |
|-------|--------|-------------|
| Phase 7: Flash Decode | 2 hours | Wire Wo + residual add in decode path |
| Phase 8: FA2 Prefill | 1.5 hours | Wire batched Wo + residual add |
| Phase 9: Context concat | 30 min | Restructure context accumulation |
| Phase 10: Scale alignment | 1 hour | Verify/fix scale matching |
| Unit tests | 1 hour | Wo projection + residual add tests |
| Integration test fix | 30 min | Update test expectations |

**Total: ~6.5 hours**

---

## Implementation Order (Recommended)

1. **Phase 7 (Decode)**: Wire Wo projection + residual add in decode path
   - Simplest case (single query)
   - Validates Wo microkernel integration
   - Quick feedback loop

2. **Phase 9 (Context concat)**: Fix context accumulation structure
   - May be needed before Phase 7 can work
   - Currently context is per-head, need concatenated

3. **Phase 10 (Scale alignment)**: Verify output scales match
   - Critical for residual add correctness
   - May expose bugs in Wo projection

4. **Phase 8 (Prefill)**: Wire batched Wo + residual add
   - After decode works, generalize to batched
   - Uses same microkernels, different calling convention

5. **Unit tests**: Add targeted tests
   - Wo output non-zero
   - Residual add correctness
   - Scale preservation

6. **Integration test**: Run full pipeline
   - ATTENTION_OUTPUT should no longer be 0.000000
   - Cosine vs FP32 should improve dramatically

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `Q16IntegerAttentionRef.cpp` | Wire Wo projection + residual add (both paths) |
| `Q16IntegerAttentionRef.h` | May need helper methods for context concatenation |
| `Test__Q16IntegerAttentionParity.cpp` | Add Wo output + residual tests |
| `Test__HybridQ16Pipeline_vs_FP32.cpp` | Update expected behavior |

---

## Success Criteria

After implementation:

1. ✅ `ATTENTION_OUTPUT` cosine > 0.90 (vs current 0.000000)
2. ✅ `ATTENTION_RESIDUAL` cosine > 0.90
3. ✅ `FFN_RESIDUAL` cosine improves (currently ~0.5 due to bad attention)
4. ✅ Integration test `HybridQ16Pipeline_vs_FP32` passes
5. ✅ Unit tests for Wo projection and residual add pass

---

## Conclusion

The HybridQ16 pipeline failure has **three root causes**:

1. **Q layout mismatch** (FIXED): RoPE now outputs head-major Q, uses `mutable_typed_data()`
2. **KV cache layout** (Addendum 1): Position-major vs head-major - transpose workaround or HeadMajorKVCache
3. **Missing Wo + residual** (THIS ADDENDUM): Microkernels exist but aren't called

This addendum provides the plan to complete root cause #3. After this work, the Q16 integer attention kernel will produce actual output (not zeros) and the full HybridQ16 pipeline can be validated.
