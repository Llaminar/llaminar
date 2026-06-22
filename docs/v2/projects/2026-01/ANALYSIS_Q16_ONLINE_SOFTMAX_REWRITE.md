# Q16 Online Softmax Kernel: Corrected Redesign Specification

## Executive Summary

The current `OnlineSoftmax.cpp` kernel uses floating-point operations in the hot path, making it unsuitable for production integer attention. This document provides a **corrected specification** for deferred normalization that addresses all identified mathematical issues.

**Status**: Corrected specification addressing review holes (v2, 2026-01-05)

### Review Holes Addressed

1. ✅ **Denominator mismatch**: Tracked explicitly as `sum_w_unscaled` (see §4.2)
2. ✅ **128-bit rescale multiply**: Using `__int128` for `context * scale_num` (see §5.3)
3. ✅ **Weight precision policy**: 10-bit weights (1024 max) for VNNI safety (see §4.3)
4. ✅ **Type/scale transitions to Wo**: Complete pipeline documented (see §6)
5. ✅ **VNNI chunk sizing**: 64 positions per chunk with 10-bit weights (see §4.4)

---

## 1. Problem Statement

### Current State (Broken)

| Component | Intended Type | Actual Type | Problem |
|-----------|---------------|-------------|---------|
| `m` (running max) | INT32 | INT32 | ✓ OK |
| `l` (weight sum) | INT64 | INT64 | ✓ OK |
| `l_processed` | INT64 | **double** | ✗ FP leak |
| Context merge | INT division | **double division + std::round()** | ✗ FP leak (HOT PATH!) |

**Cost per prefill tile**:
```
head_dim × (2 FP64 multiplies + 1 FP64 division + 1 round)
```

For 596-token prefill: **~725K FP divisions** → 10+ minute integration tests.

### Root Cause

The "running average" approach stores normalized context:
```cpp
context[d] = Σ(w[k] × V[k][d]) / Σ(w[k])  // Normalized after EVERY block
```

This requires O(N × head_dim) divisions to maintain the invariant.

---

## 2. Solution: Deferred Normalization

### Core Insight

Store **unnormalized** accumulated sums; divide **once** at finalization:

```cpp
// During accumulation (PURE INTEGER):
context_accum[d] += w_scaled[k] × V[k][d]   // int64_t accumulator
sum_w += w_scaled[k]                         // int64_t sum (SAME SCALE!)

// At finalization (ONCE per head):
output[d] = context_accum[d] / sum_w         // Single division pass
```

### Critical Invariant: Numerator/Denominator Same Scale

**Review hole #1 was**: `Σ(w>>s) ≠ (Σw)>>s` due to truncation.

**Solution**: Scale BEFORE accumulating:
```cpp
int32_t w_scaled = w_raw >> weight_shift;    // Scale once
sum_w_scaled += w_scaled;                    // Accumulate scaled weights
context[d] += w_scaled * V[d];               // Same w_scaled in both!
```

Now `context[d] / sum_w_scaled` is mathematically exact (modulo integer truncation).

---

## 3. Complete Data Flow Specification

### 3.1 Type Transition Chain

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Q×K^T DOT PRODUCT                                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Input:   Q[h][pos][d]  → INT16 (Q16_1Block, scale=s_q[h])                      │
│          K[kv_h][k][d] → INT16 (Q16_1Block, scale=s_k[k] or s_k[kv_h])         │
│ Output:  score[k]      → INT32 (raw dot product, ~20 bits used)                │
│ Scale:   qk_scale = s_q × s_k / √(head_dim)  [FP32, for alpha computation]     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 2: ONLINE SOFTMAX (MAX TRACKING + WEIGHT COMPUTATION)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ State:   m (running max)     → INT32                                           │
│          sum_w_unscaled      → INT64 (before weight_shift) [for rescale only] │
│          sum_w_scaled        → INT64 (after weight_shift) [for final divide]  │
│ Output:  w_raw[k]            → INT32 (exp2 LUT output, ~30 bits)               │
│          w_scaled[k]         → INT32 (>> weight_shift, 10 bits for VNNI)       │
│ Config:  weight_shift = 20   [30-bit LUT → 10-bit VNNI-safe]                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 3: RESCALE ON MAX CHANGE                                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Trigger: new_max > old_max                                                      │
│ Compute: scale_factor = exp2(old_max - new_max) as (scale_num, scale_shift)    │
│ Apply:                                                                          │
│   sum_w_unscaled = ((__int128)sum_w_unscaled × scale_num) >> scale_shift       │
│   sum_w_scaled   = ((__int128)sum_w_scaled × scale_num) >> scale_shift         │
│   context[d]     = ((__int128)context[d] × scale_num) >> scale_shift           │
│ NOTE: __int128 required for 64-bit × 32-bit with sufficient precision          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 4: P×V ACCUMULATION (VNNI-SAFE CHUNKING)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Inner loop (INT32, VPDPWSSD-friendly):                                         │
│   for k in chunk (64 positions max):                                            │
│     chunk_accum[d] += w_scaled[k] × V[k][d]   // INT32                         │
│     chunk_sum_w += w_scaled[k]                 // INT32                         │
│                                                                                 │
│ Dump to INT64 (after each chunk):                                              │
│   context_accum[d] += chunk_accum[d]          // INT64                         │
│   sum_w_scaled += chunk_sum_w                  // INT64                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 5: FINALIZATION (ONCE PER HEAD)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Output:  context_int32[d] = context_accum[d] / sum_w_scaled                    │
│ Scale:   pv_scale = s_v[kv_h]  [V block scale, = s_k for tied KV]              │
│ Semantic: context_int32[d] represents: (weighted_avg_of_V) in Q16 units        │
│           FP32 value = context_int32[d] × pv_scale                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 6: Wo PROJECTION                                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Input:   context_int32[d_model] (concatenated heads)                           │
│          context_scale = avg(pv_scale) across heads                            │
│ Process:                                                                        │
│   1. Normalize INT32 → INT16: ctx_int16[d], ctx_norm_scale                     │
│   2. GEMV: ctx_int16 × Wo_int8 → acc_int32 (via VPDPWSSD)                      │
│   3. Apply scales: fp32 = acc × ctx_norm_scale × context_scale × wo_scale      │
│   4. Quantize: fp32 → Q16_1Block_32                                            │
│ Output:  Q16_1Block_32[d_model/32]                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ STAGE 7: RESIDUAL ADD                                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Input:   Wo output (Q16_1Block_32) + residual_in (Q16_1Block_32)               │
│ Output:  residual_out (Q16_1Block_32)                                          │
│ Process: simd::q16_1_add_q16_1() with scale alignment                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. VNNI Safety Analysis

### 4.1 VNNI Constraint

AVX-512 VNNI (`VPDPWSSD`) performs: `INT16 × INT16 → INT32` with accumulation.

Overflow when: `Σ(w[k] × V[k][d]) > INT32_MAX (2^31 - 1)`

### 4.2 Weight Precision Policy

From `VNNISafetyConstants.h`, we have established bounds for Q×K. For P×V:

| Weight bits | w_max | w×V max | Safe chunk size |
|-------------|-------|---------|-----------------|
| 15 bits | 32767 | 2^30 | 2 (impractical) |
| 12 bits | 4095 | 2^27 | 16 |
| **10 bits** | **1023** | **2^25** | **64** ✓ |
| 8 bits | 255 | 2^23 | 128 |

**Decision**: Use **10-bit weights** (`weight_shift = 20`):
- Max weight: 1023
- Max per-product: 1023 × 32767 ≈ 33.5M ≈ 2^25
- 64 positions: 64 × 2^25 = 2^31 (exactly at limit)
- **Use chunk size = 60** for safety margin

### 4.3 Precision Impact

LUT outputs 30-bit weights. After `>> 20`:
- Top positions (high attention): Lose 20 bits → 10 bits remain
- Low positions (near zero): Round to 0 or 1

**Mitigation**: Dominant attention patterns have large weights that preserve relative ordering. The weighted average is dominated by high-weight positions anyway.

### 4.4 Safe Chunk Configuration

```cpp
constexpr int WEIGHT_SHIFT = 20;        // 30-bit LUT → 10-bit VNNI-safe
constexpr int MAX_WEIGHT = 1023;        // After shift
constexpr int CHUNK_SIZE = 60;          // Slightly under 64 for margin
constexpr int MAX_V = 32767;            // INT16 max
// Verification: 60 × 1023 × 32767 = 2.0×10^9 < 2.1×10^9 (INT32_MAX) ✓
```

---

## 5. Rescale Mathematics (128-bit)

### 5.1 When Max Changes

When `new_max > old_max`, all previous weights must be scaled down:
```
scale_factor = exp2(old_max - new_max) = exp2(-delta) where delta > 0
```

We compute this as integer `(scale_num, scale_shift)` where:
```
scale_factor ≈ scale_num / 2^scale_shift
```

### 5.2 What Must Be Rescaled

1. `sum_w_unscaled`: For tracking total weight at original precision
2. `sum_w_scaled`: For final division (CRITICAL: same scale as numerator!)
3. `context_accum[d]`: All accumulated P×V products

### 5.3 128-bit Requirement

**Review hole #2**: `int64_t × int32_t` can overflow 64 bits.

Example:
- `context_accum[d]` after 1000 positions: ~2^50
- `scale_num`: up to 2^30
- Product: 2^80 → **requires 128-bit intermediate**

**Solution using compiler intrinsic**:
```cpp
inline int64_t rescale_int64(int64_t value, int32_t scale_num, int scale_shift) {
    __int128 product = static_cast<__int128>(value) * static_cast<__int128>(scale_num);
    return static_cast<int64_t>(product >> scale_shift);
}
```

**Portability**: `__int128` is supported by GCC/Clang on x86-64. For MSVC, use `_mul128` intrinsic.

### 5.4 Rescale Code

```cpp
if (needs_rescale) {
    // Rescale weight sums
    state.sum_w_unscaled = rescale_int64(state.sum_w_unscaled, scale_num, scale_shift);
    state.sum_w_scaled = rescale_int64(state.sum_w_scaled, scale_num, scale_shift);
    
    // Rescale context (all dimensions)
    for (int d = 0; d < head_dim; ++d) {
        context_accum[d] = rescale_int64(context_accum[d], scale_num, scale_shift);
    }
}
```

---

## 6. Updated State Structures

### 6.1 Decode State

```cpp
struct OnlineSoftmaxStateV2 {
    // === Max tracking ===
    int32_t m = std::numeric_limits<int32_t>::min();  // Running max score
    
    // === Weight sums (BOTH tracked for correctness) ===
    int64_t sum_w_unscaled = 0;  // Sum before weight_shift (for debugging/verification)
    int64_t sum_w_scaled = 0;    // Sum after weight_shift (for final division)
    
    // === Configuration ===
    int frac_bits = 11;          // Fractional bits in exp2 LUT
    int lut_value_bits = 30;     // Total bits in LUT output
    int weight_shift = 20;       // Shift for VNNI-safe weights (30→10 bits)
    int chunk_size = 60;         // P×V accumulation chunk
    AdaptiveAlphaConfig alpha_config;
    
    // === Position tracking ===
    int count = 0;               // Unmasked positions seen
    
    // REMOVED: double l_processed (no longer needed!)
};
```

### 6.2 Prefill Batch State

```cpp
struct OnlineSoftmaxStateBatchV2 {
    int Br = 0;  // Number of rows in batch
    
    // Per-row state
    std::vector<int32_t> m;              // [Br] running max
    std::vector<int64_t> sum_w_unscaled; // [Br] 
    std::vector<int64_t> sum_w_scaled;   // [Br]
    std::vector<int> count;              // [Br]
    
    // Shared configuration
    int frac_bits = 11;
    int lut_value_bits = 30;
    int weight_shift = 20;
    int chunk_size = 60;
    float base_alpha_fp32 = 0.0f;
    std::vector<AdaptiveAlphaConfig> alpha_configs;  // [Br] or [1] if uniform
    
    void init(int br, float qk_scale);
    void init_per_position(int br, float q_scale, int head_dim, int frac, int lut);
};
```

---

## 7. Algorithm Pseudocode

### 7.1 Decode: Process KV Block

```cpp
template <typename BlockType>
void flash_decode_process_kv_block_v2(
    const BlockType *Q,
    const BlockType *K,
    const BlockType *V,
    int64_t *context_accum,       // [head_dim] INT64 accumulator
    OnlineSoftmaxStateV2 &state,
    int32_t *scores_scratch,      // [block_size] 
    int32_t *weights_scratch,     // [block_size]
    int kv_block_start,
    int kv_block_size,
    int head_dim,
    int blocks_per_row)
{
    // ─────────────────────────────────────────────────────────────────
    // Step 1: Compute Q×K^T scores
    // ─────────────────────────────────────────────────────────────────
    for (int k = 0; k < kv_block_size; ++k) {
        scores_scratch[k] = q16_dot_single<BlockType>(Q, K + k * blocks_per_row, 
                                                       head_dim, blocks_per_row);
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Step 2: Online softmax - find max, compute weights, check rescale
    // ─────────────────────────────────────────────────────────────────
    int32_t block_max = *std::max_element(scores_scratch, scores_scratch + kv_block_size);
    int32_t scale_num = 1;
    int scale_shift = 0;
    bool needs_rescale = false;
    
    if (block_max > state.m) {
        needs_rescale = state.count > 0;  // Only if we have prior accumulation
        compute_rescale_factor(state.m, block_max, state.alpha_config,
                               scale_num, scale_shift);
        state.m = block_max;
    }
    
    // Compute weights via exp2 LUT
    for (int k = 0; k < kv_block_size; ++k) {
        weights_scratch[k] = exp2_lut_lookup(state.m - scores_scratch[k], 
                                              state.alpha_config, state.frac_bits);
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Step 3: Rescale prior accumulation if max changed (128-bit math)
    // ─────────────────────────────────────────────────────────────────
    if (needs_rescale) {
        state.sum_w_unscaled = rescale_int64(state.sum_w_unscaled, scale_num, scale_shift);
        state.sum_w_scaled = rescale_int64(state.sum_w_scaled, scale_num, scale_shift);
        for (int d = 0; d < head_dim; ++d) {
            context_accum[d] = rescale_int64(context_accum[d], scale_num, scale_shift);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Step 4: P×V accumulation with VNNI-safe chunking
    // ─────────────────────────────────────────────────────────────────
    alignas(64) int32_t chunk_accum[256];  // Max head_dim, stack-allocated
    int32_t chunk_sum_w = 0;
    
    for (int chunk_start = 0; chunk_start < kv_block_size; chunk_start += state.chunk_size) {
        int chunk_end = std::min(chunk_start + state.chunk_size, kv_block_size);
        
        // Zero chunk accumulator
        std::memset(chunk_accum, 0, head_dim * sizeof(int32_t));
        chunk_sum_w = 0;
        
        // Inner loop: VNNI-friendly INT32 accumulation
        for (int k = chunk_start; k < chunk_end; ++k) {
            int32_t w_raw = weights_scratch[k];
            int32_t w_scaled = w_raw >> state.weight_shift;
            
            // Track sums
            state.sum_w_unscaled += w_raw;
            chunk_sum_w += w_scaled;
            
            if (w_scaled == 0) continue;
            
            const BlockType *V_row = V + k * blocks_per_row;
            pv_accumulate_row<BlockType>(chunk_accum, V_row, w_scaled, 
                                          head_dim, blocks_per_row);
        }
        
        // Dump chunk to INT64
        for (int d = 0; d < head_dim; ++d) {
            context_accum[d] += static_cast<int64_t>(chunk_accum[d]);
        }
        state.sum_w_scaled += static_cast<int64_t>(chunk_sum_w);
    }
    
    state.count += kv_block_size;
    // NO DIVISION HERE! Context remains unnormalized.
}
```

### 7.2 Decode: Finalization

```cpp
void flash_decode_finalize_v2(
    const int64_t *context_accum,
    int32_t *context_out,
    const OnlineSoftmaxStateV2 &state,
    int head_dim)
{
    // Single division pass at the end
    if (state.sum_w_scaled > 0) {
        for (int d = 0; d < head_dim; ++d) {
            context_out[d] = static_cast<int32_t>(context_accum[d] / state.sum_w_scaled);
        }
    } else {
        std::memset(context_out, 0, head_dim * sizeof(int32_t));
    }
}
```

### 7.3 Full Decode Head Processing

```cpp
// In Q16IntegerAttentionRef.cpp

for (int h = 0; h < num_heads; ++h) {
    // Allocate INT64 context accumulator (per-head)
    alignas(64) int64_t context_accum[256];  // Max head_dim
    std::memset(context_accum, 0, head_dim * sizeof(int64_t));
    
    OnlineSoftmaxStateV2 state;
    state.init(qk_scale);
    
    // Process all KV blocks
    for (int kv_start = 0; kv_start < kv_len; kv_start += KV_BLOCK_SIZE) {
        flash_decode_process_kv_block_v2<BlockType>(
            Q_head, K_head + kv_start * blocks_per_row, V_head + kv_start * blocks_per_row,
            context_accum, state,
            scores_scratch.data(), weights_scratch.data(),
            kv_start, std::min(KV_BLOCK_SIZE, kv_len - kv_start),
            head_dim, blocks_per_row);
    }
    
    // Finalize: single division pass
    std::vector<int32_t> context_int32(head_dim);
    flash_decode_finalize_v2(context_accum, context_int32.data(), state, head_dim);
    
    // Copy to full context for Wo projection
    for (int d = 0; d < head_dim; ++d) {
        full_context_int32[h * head_dim + d] = context_int32[d];
    }
    head_pv_scales[h] = pv_scale;
}

// Wo projection (unchanged interface)
wo_projection_vnni_int16(full_context_int32.data(), avg_pv_scale, Wo_packed, output, ...);
```

---

## 8. Overflow Safety Proofs

### 8.1 INT32 Chunk Accumulation

With `weight_shift = 20` and `chunk_size = 60`:
```
max_weight_scaled = 2^30 >> 20 = 1024
max_V = 32767 (INT16)
max_per_product = 1024 × 32767 = 33,553,408 ≈ 2^25
max_per_chunk = 60 × 2^25 = 2.01 × 10^9 < 2.15 × 10^9 (INT32_MAX) ✓
```

### 8.2 INT64 Total Accumulation

For 32K sequence length:
```
chunks_per_sequence = 32768 / 60 ≈ 546 chunks
max_per_chunk = 2^25 × 60 = 2.01 × 10^9
max_total = 546 × 2.01 × 10^9 = 1.1 × 10^12 ≈ 2^40
INT64_MAX = 2^63 → 8 million × headroom ✓
```

### 8.3 Rescale Safety

After rescale with `scale_num < 2^30` and `scale_shift ≤ 30`:
```
context_accum max before rescale: 2^50 (conservative)
scale_num max: 2^30
product: 2^80 → requires __int128 ✓
after >> 30: back to ≤ 2^50 ✓
```

---

## 9. Migration Plan

### Phase 1: Add V2 State Structures (Non-breaking)
- Add `OnlineSoftmaxStateV2` and `OnlineSoftmaxStateBatchV2`
- Add `rescale_int64()` helper with `__int128`
- Add new function signatures with `_v2` suffix

### Phase 2: Implement Decode V2
- Implement `flash_decode_process_kv_block_v2()`
- Implement `flash_decode_finalize_v2()`
- Add unit tests comparing V1 vs V2 output

### Phase 3: Implement Prefill V2
- Implement `fa2_prefill_process_kv_tile_v2()`
- Implement batched finalization
- Add prefill unit tests

### Phase 4: Update Kernel Wrappers
- Update `Q16IntegerAttentionRef` to use V2 functions
- Update buffer allocations (INT64 context)
- Verify Wo projection interface unchanged

### Phase 5: Deprecate V1
- Mark V1 functions as deprecated
- Run full integration/E2E test suite
- Remove V1 after validation

---

## 10. Expected Performance

| Metric | Current (V1) | After (V2) | Improvement |
|--------|--------------|------------|-------------|
| FP ops per head | O(N × head_dim) | O(head_dim) | **N× reduction** |
| Hot loop | FP64 division | INT32 multiply-add | **~10× faster** |
| Vectorization | Blocked | Full SIMD | Better ILP |
| Integration test | 10+ minutes | < 1 minute | **10×+ faster** |

---

## 11. Files to Modify

| File | Changes |
|------|---------|
| `OnlineSoftmax.h` | Add V2 state structs, `rescale_int64()`, V2 function decls |
| `OnlineSoftmax.cpp` | Implement V2 functions, keep V1 for comparison |
| `Q16IntegerAttentionRef.cpp` | Switch to V2 functions, INT64 buffers |
| `Q16IntegerAttentionRef.h` | Update internal buffer types if needed |
| `tests/v2/unit/Test__OnlineSoftmax.cpp` | Add V2 tests, parity checks |

---

## 12. Open Items

1. **Per-position K scales**: V2 compatible - alpha varies per position but accumulation is same.
2. **Snapshot support**: May need to recompute weights at finalization for snapshot_weights.
3. **MPI tensor parallelism**: Context accumulation is per-rank; Wo allreduce unchanged.

---

## Appendix A: Denominator Tracking Proof

Let `w_i` be raw LUT weights, `s = weight_shift`.

**Claim**: `Σ(w_i >> s)` tracked explicitly equals the denominator for `Σ((w_i >> s) × V_i)`.

**Proof**:
- Numerator: `N = Σ_i ((w_i >> s) × V_i[d])`
- Denominator: `D = Σ_i (w_i >> s)` ← **we track this explicitly as `sum_w_scaled`**
- Result: `context[d] = N / D`

The key insight is that we accumulate `w_i >> s` into `sum_w_scaled` at the **same time** we accumulate `(w_i >> s) × V_i[d]` into `context_accum[d]`. This ensures N and D use identical scaled weights.

**Contrast with incorrect approach**:
```cpp
// WRONG: shift the sum instead of summing the shifts
sum_w_unscaled += w_i;
context[d] = context_accum[d] / (sum_w_unscaled >> s);  // Σ(w_i >> s) ≠ (Σw_i) >> s!
```

---

## Appendix B: 128-bit Multiply Implementation

```cpp
#include <cstdint>

// Cross-platform 128-bit rescale
inline int64_t rescale_int64(int64_t value, int32_t scale_num, int scale_shift) {
#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: native __int128
    __int128 product = static_cast<__int128>(value) * static_cast<__int128>(scale_num);
    return static_cast<int64_t>(product >> scale_shift);
#elif defined(_MSC_VER)
    // MSVC: use intrinsic
    int64_t high;
    int64_t low = _mul128(value, static_cast<int64_t>(scale_num), &high);
    // Combine and shift
    if (scale_shift < 64) {
        return (high << (64 - scale_shift)) | (static_cast<uint64_t>(low) >> scale_shift);
    } else {
        return high >> (scale_shift - 64);
    }
#else
    #error "128-bit multiply not implemented for this compiler"
#endif
}
```

---

## Appendix C: VNNI-safe P×V Inner Loop

```cpp
// Vectorizable inner loop for chunk accumulation
template <typename BlockType>
inline void pv_accumulate_row(
    int32_t *chunk_accum,
    const BlockType *V_row,
    int32_t w_scaled,
    int head_dim,
    int blocks_per_row)
{
    constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
    
    for (int b = 0; b < blocks_per_row; ++b) {
        const int16_t *v_data = V_row[b].qs;
        int base = b * BLOCK_SIZE;
        
        // This loop auto-vectorizes to VPMADDWD + VPADDD (or VPDPWSSD if available)
        #pragma omp simd
        for (int i = 0; i < BLOCK_SIZE && base + i < head_dim; ++i) {
            chunk_accum[base + i] += w_scaled * static_cast<int32_t>(v_data[i]);
        }
    }
}
```

---

*Document version: 2.0 (Corrected)*
*Last updated: 2026-01-05*
*Status: Ready for implementation*

