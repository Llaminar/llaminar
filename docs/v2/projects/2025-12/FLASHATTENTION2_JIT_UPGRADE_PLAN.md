# FlashAttention-2 JIT Kernel Upgrade Plan

**Author:** David Sanftenberg  
**Date:** December 2025  
**Status:** In Progress (Phase 1, 2, 3 & 4 Complete; Phase 5 & 6 Planned)  
**Target:** `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h`

---

## Current Test Coverage

The fused attention kernel has extensive test coverage across unit, integration, and performance tests.

### Test Inventory

| Test Name | Type | Status | Description |
|-----------|------|--------|-------------|
| `V2_Unit_Microkernel_Q8DotProduct` | Unit | ✅ Pass | Q8_1 dot product microkernel |
| `V2_Unit_Microkernel_OnlineSoftmax` | Unit | ✅ Pass | Online softmax state management |
| `V2_Unit_Microkernel_VWeightedAccum` | Unit | ✅ Pass | Weighted V accumulation |
| `V2_Unit_Microkernel_WoProjection` | Unit | ✅ Pass | Wo output projection |
| `V2_Unit_Microkernel_FastExp` | Unit | ✅ Pass | Fast exp polynomial approximation |
| `V2_Unit_FusedAttentionWo_Tiled` | Unit | ✅ Pass | Cache-blocked tiled implementation |
| `V2_Unit_JitMicrokernels` | Unit | ✅ Pass | JIT microkernel code generation |
| `V2_Unit_JitFusedAttentionWo` | Unit | ✅ Pass | JIT fused attention kernel |
| `V2_Unit_JitFusedAttentionWo_Correctness` | Unit | ✅ Pass | JIT vs reference parity |
| `V2_Unit_JitFusedAttentionWo_Prefill` | Unit | ✅ Pass | Multi-token prefill mode |
| `V2_Unit_JitFusedAttentionWo_Debug` | Unit | ✅ Pass | Debug instrumentation |
| `V2_Unit_JitWoProjection` | Unit | ✅ Pass | JIT Wo projection isolated |
| `V2_Unit_JitWoProjectionIsolated` | Unit | ✅ Pass | Wo projection edge cases |
| `V2_Unit_Q8_1_OnlineSoftmax` | Unit | ✅ Pass | Q8_1 online softmax |
| `V2_Unit_Q8_1_FusedAttention` | Unit | ✅ Pass | Q8_1 fused attention |
| `V2_Integration_FusedAttentionWoRef` | Integration | ✅ Pass | Reference implementation |
| `V2_Integration_FusedAttentionWo_Batch` | Integration | ✅ Pass | Batched inference |
| `V2_Integration_FusedAttentionWo_Robustness` | Integration | ✅ Pass | Edge cases and robustness |
| `V2_Integration_FusedAttentionWoKernel` | Integration | ✅ Pass | Pipeline wrapper |
| `V2_Integration_Q8_1_OnlineSoftmax` | Integration | ✅ Pass | Q8_1 online softmax integration |
| `V2_Integration_Q8_1_FusedAttention` | Integration | ✅ Pass | Real model weights + causal mask |
| `V2_Perf_FusedAttentionWo` | Performance | ✅ Pass | Throughput benchmarks |
| `V2_Perf_FusedAttentionWo_Tuning` | Performance | ✅ Pass | Tile size tuning |
| `V2_Perf_JitMicrokernels` | Performance | ✅ Pass | JIT microkernel perf |

**Summary:** 24/24 fused attention tests passing (100%). 

> **Note:** `V2_Perf_QuantisedGemmKernel_Q8_1_OnlineSoftmax` is a separate GEMM perf test with an unrelated test bug (incorrect expected values in `DEBUG_MinimalCase`). It does not affect fused attention correctness.

### Test Files Location

```
tests/v2/
├── unit/
│   ├── attention/
│   │   ├── Test__FusedAttentionWoTiled.cpp
│   │   ├── Test__JitFusedAttentionWo.cpp
│   │   ├── Test__JitFusedAttentionWo_Correctness.cpp
│   │   ├── Test__JitFusedAttentionWo_Debug.cpp
│   │   ├── Test__JitFusedAttentionWo_Prefill.cpp
│   │   ├── Test__JitMicrokernels.cpp
│   │   ├── Test__JitWoProjection.cpp
│   │   └── Test__JitWoProjectionIsolated.cpp
│   ├── microkernels/
│   │   ├── Test__FastExp.cpp
│   │   ├── Test__OnlineSoftmax.cpp
│   │   ├── Test__Q8DotProduct.cpp
│   │   ├── Test__VWeightedAccum.cpp
│   │   └── Test__WoProjection.cpp
│   └── kernels/cpu/gemm_v4/
│       └── Test__Q8_1_OnlineSoftmax_Unit.cpp
├── integration/
│   ├── Test__FusedAttentionWoRef.cpp
│   ├── Test__FusedAttentionWoRef_Batch.cpp
│   ├── Test__FusedAttentionWoRef_Robustness.cpp
│   ├── Test__Q8_1_FusedAttention.cpp
│   ├── Test__Q8_1_OnlineSoftmax.cpp
│   └── attention/
│       └── Test__FusedAttentionWoKernel.cpp
└── performance/
    └── kernels/cpu/
        ├── attention/
        │   ├── Perf__FusedAttentionWo.cpp
        │   ├── Perf__FusedAttentionWo_Tuning.cpp
        │   └── Perf__JitMicrokernels.cpp
        └── gemm/gemm_v4/
            ├── Perf__QuantisedGemmJit_Q8_1_OnlineSoftmax.cpp
            └── Perf__QuantisedGemmKernel__Q8_1_OnlineSoftmax.cpp
```

### Running Tests

```bash
# All fused attention tests (release build, parallel)
ctest --test-dir build_v2_release \
  -R "FusedAttention|JitFused|OnlineSoftmax|Q8DotProduct|VWeightedAccum|WoProjection|FastExp|JitMicrokernel|JitWo" \
  --output-on-failure --parallel

# Unit tests only
ctest --test-dir build_v2_release -R "V2_Unit.*Fused|V2_Unit.*Jit|V2_Unit.*Microkernel" --parallel

# Integration tests only
ctest --test-dir build_v2_release -R "V2_Integration.*Fused|V2_Integration.*Q8_1" --parallel

# Performance benchmarks
ctest --test-dir build_v2_release -R "V2_Perf.*Fused|V2_Perf.*Jit" --verbose
```

---

## Executive Summary

This document outlines the modifications required to upgrade the existing JIT fused attention kernel from FlashAttention-1 style to FlashAttention-2 (FA2) performance characteristics. The current implementation already has the core FA1 algorithm (online softmax, no N×N materialization), but lacks FA2's key optimizations: **tile-wise softmax batching** and **head-level parallelism**.

### Current State Assessment

| Feature | Status | Notes |
|---------|--------|-------|
| Online softmax (streaming max/sum) | ✅ Complete | `OnlineSoftmax.h` |
| No attention matrix materialization | ✅ Complete | Scores computed one-at-a-time |
| KV tiling with L2-awareness | ✅ Complete | `FusedAttentionWoTiled.cpp` |
| Wo projection fusion | ✅ Complete | Unique optimization beyond FA |
| FA2 loop order (Q outer, KV inner) | ✅ Complete | Already correct |
| Head-level parallelism | ✅ Complete | OpenMP with `OMP_WORKSHARE_REGION` (Phase 1) |
| Tile-wise softmax batching | ✅ Complete | `process_kv_tile_batched()` (Phase 2) |
| Vectorized score tile computation | ❌ Missing | Sequential dot products |
| **High-performance Wo projection** | 🔴 **Bottleneck** | **6× slower than OpenBLAS (Phase 6)** |

### Performance Gap Estimate

Based on FlashAttention-2 paper benchmarks and our current profiling:

| Metric | Current | Target | Expected Gain |
|--------|---------|--------|---------------|
| Prefill throughput | ~300 tok/s | ~450 tok/s | 1.5× |
| Decode latency | ~18ms/tok | ~12ms/tok | 1.5× |
| Non-matmul FLOP% | ~40% | ~15% | 2.7× reduction |
| **Wo projection GFLOP/s** | **~8 GFLOP/s** | **~50 GFLOP/s** | **6× (Phase 6)** |

> **⚠️ Critical Finding (Dec 23, 2025):** Performance analysis revealed that **93% of Qwen 7B attention FLOPs** are in the Wo projection, not attention itself. The naive JIT Wo implementation achieves only 8 GFLOP/s vs OpenBLAS's 54 GFLOP/s. See **Phase 6** for the fix.

---

## Phase 1: Head-Level OpenMP Parallelism ✅ COMPLETE

**Effort:** Low  
**Impact:** Medium (1.2-1.4× speedup on multi-core)  
**Risk:** Low  
**Status:** ✅ Implemented December 23, 2025

### Implementation Summary

Added OpenMP head-level parallelism using `OMP_WORKSHARE_REGION` macro for nested-safe execution:

- **Per-head output buffers**: Each head writes Wo projection to its own buffer, then reduce
- **Thread-local context buffers**: Allocated inside `OMP_WORKSHARE_REGION` lambda
- **Two-phase execution**: Phase 1 processes heads in parallel, Phase 2 reduces outputs
- **All 12 fused attention tests pass** (excluding perf tests)

Key code pattern:
```cpp
auto do_attention_work = [&]() {
    // Thread-local buffers inside lambda
    alignas(64) std::vector<float> context_buffers(q_count * head_dim);
    
    #pragma omp for schedule(static)
    for (int h = 0; h < num_heads; ++h) { /* process head */ }
    
    // Implicit barrier
    
    #pragma omp for schedule(static)
    for (int qi = 0; qi < q_count; ++qi) { /* reduce outputs */ }
};
OMP_WORKSHARE_REGION(do_attention_work);
```

### Original Problem

Current code processes heads sequentially:

```cpp
// FusedAttentionWoTiled.cpp:175
for (int h = 0; h < num_heads; ++h) {
    // ... process head h
}
```

This leaves cores idle. On a 56-core Xeon, processing 28 heads sequentially wastes half the available compute.

### Solution

Add OpenMP parallelization at the head level with thread-local buffers:

```cpp
// MODIFIED: FusedAttentionWoTiled.cpp
void FusedAttentionWoTiled::process_batch_item_tiled(
    const FusedAttentionWoParams &params,
    int batch_idx,
    const TileConfig &config)
{
    const int seq_len = params.seq_len;
    const int num_heads = params.num_heads;
    const int head_dim = params.head_dim;
    const int d_model = params.d_model;
    const int q_tile = config.q_tile;

    const int kv_len = params.get_kv_len(batch_idx);
    const int pos_offset = params.get_position_offset(batch_idx);

    const int num_blocks = head_dim / 32;
    const size_t output_batch_stride = static_cast<size_t>(seq_len) * d_model;
    float *output_base = params.output + batch_idx * output_batch_stride;

    // Process query positions in tiles
    for (int q_start = 0; q_start < seq_len; q_start += q_tile)
    {
        const int q_end = std::min(q_start + q_tile, seq_len);
        const int q_count = q_end - q_start;

        // === NEW: Parallel head processing ===
        #pragma omp parallel
        {
            // Thread-local buffers (avoid false sharing)
            alignas(64) std::vector<float> context_buffers(q_count * head_dim);
            std::vector<OnlineSoftmaxStateTiled> softmax_states(q_count);

            #pragma omp for schedule(static)
            for (int h = 0; h < num_heads; ++h)
            {
                // Reset buffers for this head
                std::memset(context_buffers.data(), 0, q_count * head_dim * sizeof(float));
                for (int i = 0; i < q_count; ++i) {
                    softmax_states[i] = OnlineSoftmaxStateTiled();
                }

                // Process this head with tiling
                process_head_tiled(
                    params, batch_idx, h,
                    q_start, q_end,
                    kv_len, pos_offset,
                    config,
                    context_buffers.data(),
                    softmax_states.data());

                // Finalize and project through Wo
                // NOTE: Wo projection accumulates into output, needs atomic or reduction
                for (int qi = 0; qi < q_count; ++qi)
                {
                    const int m = q_start + qi;
                    float *context = context_buffers.data() + qi * head_dim;
                    const OnlineSoftmaxStateTiled &state = softmax_states[qi];

                    // Normalize
                    if (state.sum_exp > 0.0f) {
                        const float inv_sum = 1.0f / state.sum_exp;
                        for (int d = 0; d < head_dim; ++d) {
                            context[d] *= inv_sum;
                        }
                    }

                    // Project through Wo (thread-safe: each head writes to disjoint d_model slice)
                    float *output_row = output_base + m * d_model;

                    microkernels::WoProjectionParams wo_params;
                    wo_params.context = context;
                    wo_params.wo_weights = params.Wo;
                    wo_params.wo_type = params.wo_type;
                    wo_params.head_dim = head_dim;
                    wo_params.d_model = d_model;
                    wo_params.head_idx = h;
                    wo_params.n_heads = num_heads;
                    wo_params.output = output_row;
                    wo_params.accumulate = true;  // Uses atomic adds internally

                    microkernels::wo_projection_ref(wo_params);
                }
            }
        } // end omp parallel
    }
}
```

### Wo Projection Thread Safety

The Wo projection accumulates all heads into the same output row. Options:

**Option A: Atomic accumulation (simple, slight overhead)**
```cpp
// WoProjection.cpp
void wo_projection_ref(const WoProjectionParams& params) {
    // ...
    if (params.accumulate) {
        #pragma omp atomic
        params.output[d] += result;
    }
}
```

**Option B: Per-head output buffers + final reduction (more memory, faster)**
```cpp
// Allocate [num_heads][seq_len][d_model] output buffer
// Each head writes to its slice
// Final reduction: output[m][d] = sum over h of head_outputs[h][m][d]
```

**Recommendation:** Start with Option A for simplicity, profile, then consider Option B if atomic overhead is significant (>5% runtime).

---

## Phase 2: Tile-Wise Softmax Batching ✅ COMPLETE

**Effort:** Medium  
**Impact:** High (1.3-1.5× speedup)  
**Risk:** Medium (numerical precision changes)  
**Status:** ✅ Implemented December 23, 2025

### Implementation Summary

Added `process_kv_tile_batched()` function that implements FA2-style batched softmax:

1. **Phase 1**: Compute ALL Q·K scores for the KV tile first into scratch buffer
2. **Phase 2**: Find tile maximum in single pass  
3. **Phase 3**: Update softmax state ONCE per tile (not per position!)
4. **Phase 4**: Apply correction once, then accumulate all V values

Benefits:
- Context correction applied once per tile instead of per position (~64× reduction)
- Single softmax state update per tile (fewer branches)
- Better cache behavior (sequential score buffer access)

Key code:
```cpp
void FusedAttentionWoTiled::process_kv_tile_batched(
    /* params */, float *tile_scores)
{
    // Phase 1: Compute all scores
    for (int t = 0; t < tile_len; ++t)
        tile_scores[t] = q8_dot_product_ref(dot_params).score;
    
    // Phase 2: Find tile max
    float tile_max = tile_scores[0];
    for (int t = 1; t < tile_len; ++t)
        if (tile_scores[t] > tile_max) tile_max = tile_scores[t];
    
    // Phase 3: Update softmax state ONCE
    if (tile_max > state.max_score) {
        correction = fast_exp_poly(state.max_score - tile_max);
        state.sum_exp *= correction;
        state.max_score = tile_max;
        for (int d = 0; d < head_dim; ++d)
            context[d] *= correction;  // Once per tile!
    }
    
    // Phase 4: Accumulate V
    for (int t = 0; t < tile_len; ++t) {
        float weight = fast_exp_poly(tile_scores[t] - state.max_score);
        state.sum_exp += weight;
        v_weighted_accum_ref(/* V[t], weight */);
    }
}
```

**All 12 fused attention tests pass** (excluding perf tests).

### Original Problem

Current implementation updates softmax state per KV position:

```cpp
// FusedAttentionWoTiled.cpp:320
for (int n = effective_kv_start; n < effective_kv_end; ++n) {
    float score = q8_dot_product_ref(dot_params).score;  // One at a time
    
    float old_max = state.max_score;
    float weight;
    
    if (!state.initialized) {
        // ... first score handling
    } else if (score > state.max_score) {
        float correction = fast_exp_poly(old_max - score);  // Correction per position
        state.sum_exp *= correction;
        // Apply correction to context (head_dim iterations!)
        for (int d = 0; d < head_dim; ++d) {
            context[d] *= correction;
        }
    }
    // ...
}
```

**Issues:**
1. Correction applied per-position → many redundant `context[d] *= correction` loops
2. Dot products computed sequentially → no SIMD parallelism across KV positions
3. Softmax state updated after each score → branch mispredictions

### Solution: Tile-Wise Score Computation + Deferred Correction

FA2's insight: Compute all scores in a tile first, find tile max, then update state once per tile.

```cpp
// NEW: TileSoftmaxState for batched processing
struct TileSoftmaxState {
    float running_max;      // Max across all tiles seen so far
    float running_sum_exp;  // Sum of exp(score - running_max) across all tiles
    bool initialized;
    
    TileSoftmaxState() : running_max(-INFINITY), running_sum_exp(0.0f), initialized(false) {}
};

// NEW: Process entire KV tile, update softmax once
void process_kv_tile_batched(
    const FusedAttentionWoParams& params,
    const Q8_1Block* Q_row,       // [num_blocks] - single query
    const Q8_1Block* K_tile,      // [tile_size][num_blocks] - KV tile
    const Q8_1Block* V_tile,      // [tile_size][num_blocks] - KV tile
    int tile_size,                // Actual positions in this tile
    int num_blocks,
    float scale,
    float* context,               // [head_dim] - accumulator for this query
    TileSoftmaxState& state)      // Softmax state for this query
{
    // Scratch buffers (should be thread-local or on stack)
    alignas(64) float scores[MAX_KV_TILE];   // e.g., MAX_KV_TILE = 512
    alignas(64) float weights[MAX_KV_TILE];
    
    // ========================================
    // STEP 1: Compute all scores in tile (vectorizable)
    // ========================================
    for (int i = 0; i < tile_size; ++i) {
        const Q8_1Block* K_row = K_tile + i * num_blocks;
        
        Q8DotProductParams dot_params;
        dot_params.q_blocks = Q_row;
        dot_params.k_blocks = K_row;
        dot_params.num_blocks = num_blocks;
        dot_params.global_scale = scale;
        
        scores[i] = q8_dot_product_ref(dot_params).score;
    }
    
    // ========================================
    // STEP 2: Find tile max (SIMD reducible)
    // ========================================
    float tile_max = -INFINITY;
    for (int i = 0; i < tile_size; ++i) {
        tile_max = std::max(tile_max, scores[i]);
    }
    
    // ========================================
    // STEP 3: Update global softmax state ONCE per tile
    // ========================================
    float correction = 1.0f;
    
    if (!state.initialized) {
        // First tile
        state.running_max = tile_max;
        state.initialized = true;
    } else if (tile_max > state.running_max) {
        // New global max: correct existing accumulation
        correction = fast_exp_poly(state.running_max - tile_max);
        state.running_sum_exp *= correction;
        state.running_max = tile_max;
        
        // Apply correction to context ONCE (not per position!)
        for (int d = 0; d < head_dim; ++d) {
            context[d] *= correction;
        }
    }
    
    // ========================================
    // STEP 4: Compute weights for this tile (SIMD: exp of 16 floats at a time)
    // ========================================
    float tile_sum = 0.0f;
    for (int i = 0; i < tile_size; ++i) {
        weights[i] = fast_exp_poly(scores[i] - state.running_max);
        tile_sum += weights[i];
    }
    state.running_sum_exp += tile_sum;
    
    // ========================================
    // STEP 5: Accumulate weighted V (vectorizable across head_dim)
    // ========================================
    for (int i = 0; i < tile_size; ++i) {
        const Q8_1Block* V_row = V_tile + i * num_blocks;
        
        VWeightedAccumParams accum_params;
        accum_params.v_blocks = V_row;
        accum_params.weight = weights[i];
        accum_params.correction = 1.0f;  // Already applied above
        accum_params.context = context;
        accum_params.num_blocks = num_blocks;
        
        v_weighted_accum_ref(accum_params);
    }
}
```

### Performance Analysis

| Operation | Per-Position (Current) | Per-Tile (FA2) | Reduction |
|-----------|------------------------|----------------|-----------|
| Correction to context | `kv_len × head_dim` | `(kv_len/tile) × head_dim` | `tile×` |
| Softmax state updates | `kv_len` branches | `kv_len/tile` branches | `tile×` |
| Cache efficiency | Random K access | Sequential K access | Better prefetch |

For `kv_tile = 64`:
- Context corrections: 64× fewer
- Branch mispredictions: 64× fewer

---

## Phase 3: JIT Kernel Modifications

**Effort:** High  
**Impact:** Very High (1.5-2× speedup)  
**Risk:** Medium (JIT complexity)

### Current JIT Structure

The existing `JitFusedAttentionWo.h` generates code with this structure:

```asm
; Current JIT-generated code (pseudocode)
for each query q:
    for each head h:
        for each kv position n:
            ; μK1: Q·K dot product
            score = jit_q8_dot_product(Q[q,h], K[n,kv_h])
            
            ; μK2: Online softmax update (per-position)
            if score > max:
                correction = exp(max - score)
                ; Apply correction to context (16 ZMM stores!)
                vmulps zmm0, zmm0, correction
                vmulps zmm1, zmm1, correction
                ; ...
            weight = exp(score - max)
            sum += weight
            
            ; μK3: V weighted accum
            jit_v_weighted_accum(V[n,kv_h], weight, context)
        
        ; μK4: Wo projection
        jit_wo_projection(context, Wo, output[q])
```

### Target JIT Structure (FA2-Style)

```asm
; FA2-style JIT-generated code (pseudocode)
for each query q:
    for each head h:
        for each kv_tile:
            ; ========================================
            ; PHASE A: Compute all scores in tile (vectorized)
            ; ========================================
            ; Unroll by 4 KV positions, compute 4 dot products
            lea    rax, [K + kv_start * kv_stride]
            
            ; Score[0:3] in zmm20-zmm23
            jit_q8_dot_product_4x(Q[q,h], K[kv+0:3], zmm20-zmm23)
            
            ; Score[4:7] in zmm24-zmm27
            jit_q8_dot_product_4x(Q[q,h], K[kv+4:7], zmm24-zmm27)
            
            ; ... continue for tile_size
            
            ; ========================================
            ; PHASE B: Find tile max (horizontal reduction)
            ; ========================================
            ; Reduce zmm20-zmm27 to single max
            vmaxps  zmm28, zmm20, zmm21
            vmaxps  zmm29, zmm22, zmm23
            vmaxps  zmm28, zmm28, zmm29
            ; ... final horizontal max in xmm28[0]
            
            vbroadcastss zmm_tile_max, xmm28
            
            ; ========================================
            ; PHASE C: Compare with running max, apply correction ONCE
            ; ========================================
            vcomiss xmm_tile_max, xmm_running_max
            jbe     .no_correction
            
            ; New max: compute correction factor
            vsubss  xmm_corr, xmm_running_max, xmm_tile_max
            ; Fast exp approximation
            call    jit_fast_exp
            
            ; Apply correction to context (zmm0-zmm3 = 64 floats of context)
            vmulps  zmm0, zmm0, zmm_corr_broadcast
            vmulps  zmm1, zmm1, zmm_corr_broadcast
            vmulps  zmm2, zmm2, zmm_corr_broadcast
            vmulps  zmm3, zmm3, zmm_corr_broadcast
            
            ; Update running max
            vmovss  xmm_running_max, xmm_tile_max
            
            ; Scale running_sum
            vmulss  xmm_running_sum, xmm_running_sum, xmm_corr
            
        .no_correction:
            ; ========================================
            ; PHASE D: Compute weights and tile sum (vectorized exp)
            ; ========================================
            ; weight[i] = exp(score[i] - tile_max)
            vsubps  zmm20, zmm20, zmm_tile_max   ; score - max
            vsubps  zmm21, zmm21, zmm_tile_max
            ; ...
            
            ; Vectorized exp (polynomial approximation)
            jit_fast_exp_16x(zmm20)  ; exp of 16 floats
            jit_fast_exp_16x(zmm21)
            ; ...
            
            ; Horizontal sum for tile_sum
            vaddps  zmm28, zmm20, zmm21
            ; ... reduce to scalar
            vaddss  xmm_running_sum, xmm_running_sum, xmm_tile_sum
            
            ; ========================================
            ; PHASE E: Weighted V accumulation (interleaved loads)
            ; ========================================
            ; Unroll: process 4 V rows at a time
            ; Interleave V loads with FMA to hide latency
            
            vmovups zmm_v0, [V + (kv+0) * v_stride]
            vmovups zmm_v1, [V + (kv+1) * v_stride]
            
            vfmadd231ps zmm0, zmm_v0, zmm_w0  ; context += weight[0] * V[0]
            vmovups     zmm_v2, [V + (kv+2) * v_stride]  ; Overlap load
            
            vfmadd231ps zmm0, zmm_v1, zmm_w1  ; context += weight[1] * V[1]
            vmovups     zmm_v3, [V + (kv+3) * v_stride]  ; Overlap load
            
            ; ... continue interleaved pattern
```

### Key JIT Modifications Required

#### 3.1 Vectorized Multi-Score Computation

**File:** `jit/JitQ8DotProduct.h`

Add a new method to compute multiple dot products simultaneously:

```cpp
/**
 * @brief Generate code to compute 4 Q·K dot products in parallel
 * 
 * Input layout:
 *   Q: [1, num_blocks] - single query row
 *   K: [4, num_blocks] - 4 consecutive key rows
 * 
 * Output:
 *   scores in zmm_out0, zmm_out1, zmm_out2, zmm_out3 (or packed in one zmm)
 * 
 * Strategy:
 *   - Load Q blocks into zmm_q (reused across 4 K rows)
 *   - For each K row: vpdpbusd accumulation
 *   - Convert 4 accumulated int32 sums to FP32 scores
 */
void generate_q8_dot_product_4x(
    Xbyak::CodeGenerator& c,
    const Xbyak::Reg64& Q_ptr,       // Pointer to Q row
    const Xbyak::Reg64& K_ptr,       // Pointer to first K row
    int64_t k_stride,                // Bytes between consecutive K rows
    int num_blocks,                  // head_dim / 32
    const Xbyak::Zmm& score0,        // Output: score for K[0]
    const Xbyak::Zmm& score1,        // Output: score for K[1]
    const Xbyak::Zmm& score2,        // Output: score for K[2]
    const Xbyak::Zmm& score3)        // Output: score for K[3]
{
    using namespace Xbyak;
    
    // Temporary registers
    Zmm zmm_q = zmm28;
    Zmm zmm_k0 = zmm29, zmm_k1 = zmm30, zmm_k2 = zmm31;
    Zmm zmm_acc0 = score0, zmm_acc1 = score1, zmm_acc2 = score2, zmm_acc3 = score3;
    
    // Zero accumulators
    c.vpxord(zmm_acc0, zmm_acc0, zmm_acc0);
    c.vpxord(zmm_acc1, zmm_acc1, zmm_acc1);
    c.vpxord(zmm_acc2, zmm_acc2, zmm_acc2);
    c.vpxord(zmm_acc3, zmm_acc3, zmm_acc3);
    
    // Process each Q8_1 block
    for (int b = 0; b < num_blocks; ++b) {
        int64_t q_offset = b * 36;  // Q8_1Block size
        
        // Load Q block (shared across all 4 K rows)
        c.vmovdqu8(zmm_q, c.ptr[Q_ptr + q_offset]);
        
        // Load and accumulate K[0]
        c.vmovdqu8(zmm_k0, c.ptr[K_ptr + q_offset]);
        c.vpdpbusd(zmm_acc0, zmm_q, zmm_k0);
        
        // Load and accumulate K[1]
        c.vmovdqu8(zmm_k1, c.ptr[K_ptr + k_stride + q_offset]);
        c.vpdpbusd(zmm_acc1, zmm_q, zmm_k1);
        
        // Load and accumulate K[2]
        c.vmovdqu8(zmm_k2, c.ptr[K_ptr + 2*k_stride + q_offset]);
        c.vpdpbusd(zmm_acc2, zmm_q, zmm_k2);
        
        // Load and accumulate K[3] (reuse zmm_k0)
        c.vmovdqu8(zmm_k0, c.ptr[K_ptr + 3*k_stride + q_offset]);
        c.vpdpbusd(zmm_acc3, zmm_q, zmm_k0);
    }
    
    // Horizontal reduction: each zmm has 16 partial sums → reduce to 1
    // ... (existing horizontal sum code from JitQ8DotProduct)
    
    // Scale by global_scale (loaded into zmm_scale beforehand)
    c.vmulps(score0, score0, zmm_scale);
    c.vmulps(score1, score1, zmm_scale);
    c.vmulps(score2, score2, zmm_scale);
    c.vmulps(score3, score3, zmm_scale);
}
```

#### 3.2 Vectorized Tile Max Reduction

**File:** `jit/JitOnlineSoftmax.h`

```cpp
/**
 * @brief Generate code to find max of a tile of scores
 * 
 * Input: scores in zmm0-zmm7 (128 floats = 8 ZMM registers)
 * Output: max value broadcast in zmm_max
 */
void generate_tile_max_reduction(
    Xbyak::CodeGenerator& c,
    const Xbyak::Zmm& zmm_max,
    int num_score_regs)  // How many ZMM registers hold scores (1-8)
{
    using namespace Xbyak;
    
    // Tree reduction for max
    if (num_score_regs >= 2) {
        c.vmaxps(zmm0, zmm0, zmm1);
    }
    if (num_score_regs >= 4) {
        c.vmaxps(zmm2, zmm2, zmm3);
        c.vmaxps(zmm0, zmm0, zmm2);
    }
    if (num_score_regs >= 8) {
        c.vmaxps(zmm4, zmm4, zmm5);
        c.vmaxps(zmm6, zmm6, zmm7);
        c.vmaxps(zmm4, zmm4, zmm6);
        c.vmaxps(zmm0, zmm0, zmm4);
    }
    
    // Now zmm0 has 16 candidates, reduce to 1
    // Extract upper 256 bits, max with lower
    c.vextractf32x8(ymm1, zmm0, 1);
    c.vmaxps(ymm0, ymm0, ymm1);
    
    // Extract upper 128 bits of ymm, max with lower
    c.vextractf128(xmm1, ymm0, 1);
    c.vmaxps(xmm0, xmm0, xmm1);
    
    // Horizontal max within xmm (4 floats)
    c.vpermilps(xmm1, xmm0, 0b01001110);  // Swap pairs
    c.vmaxps(xmm0, xmm0, xmm1);
    c.vpermilps(xmm1, xmm0, 0b10110001);  // Swap within pairs
    c.vmaxps(xmm0, xmm0, xmm1);
    
    // Broadcast result
    c.vbroadcastss(zmm_max, xmm0);
}
```

#### 3.3 Vectorized Exp for Entire Tile

**File:** `jit/JitFastExp.h`

Extend to process 16 floats (one ZMM) at a time:

```cpp
/**
 * @brief Generate vectorized exp approximation for 16 floats
 * 
 * Uses the same polynomial approximation as scalar, but across ZMM.
 * 
 * Input/Output: zmm_x contains values, will be replaced with exp(x)
 * 
 * Algorithm (range-reduced exp):
 *   1. Clamp to avoid overflow: x = clamp(x, -88, 88)
 *   2. Range reduction: x = n*ln2 + r where n = round(x/ln2), |r| < ln2/2
 *   3. Polynomial: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
 *   4. Reconstruct: exp(x) = 2^n * exp(r)
 */
void generate_fast_exp_16x(
    Xbyak::CodeGenerator& c,
    const Xbyak::Zmm& zmm_x,
    const Xbyak::Zmm& zmm_tmp1,
    const Xbyak::Zmm& zmm_tmp2)
{
    using namespace Xbyak;
    
    // Constants (loaded once at kernel init)
    // zmm_log2e: 1/ln(2) = 1.4426950408889634
    // zmm_ln2_hi: ln(2) high bits
    // zmm_ln2_lo: ln(2) low bits (for precision)
    // zmm_c0..c4: polynomial coefficients
    
    // Step 1: Clamp to [-88, 88]
    c.vminps(zmm_x, zmm_x, zmm_clamp_hi);
    c.vmaxps(zmm_x, zmm_x, zmm_clamp_lo);
    
    // Step 2: n = round(x / ln2)
    c.vmulps(zmm_tmp1, zmm_x, zmm_log2e);
    c.vroundps(zmm_tmp1, zmm_tmp1, 0);  // Round to nearest
    
    // Step 3: r = x - n * ln2
    c.vfnmadd231ps(zmm_x, zmm_tmp1, zmm_ln2_hi);  // x -= n * ln2_hi
    c.vfnmadd231ps(zmm_x, zmm_tmp1, zmm_ln2_lo);  // x -= n * ln2_lo
    // Now zmm_x = r
    
    // Step 4: Polynomial exp(r) = 1 + r*(1 + r*(c2 + r*(c3 + r*c4)))
    c.vmovaps(zmm_tmp2, zmm_c4);
    c.vfmadd213ps(zmm_tmp2, zmm_x, zmm_c3);  // c4*r + c3
    c.vfmadd213ps(zmm_tmp2, zmm_x, zmm_c2);  // (c4*r + c3)*r + c2
    c.vfmadd213ps(zmm_tmp2, zmm_x, zmm_one);  // ...
    c.vfmadd213ps(zmm_tmp2, zmm_x, zmm_one);  // = 1 + r*(...)
    
    // Step 5: 2^n via integer add to exponent
    c.vcvtps2dq(zmm_tmp1, zmm_tmp1);         // n as int32
    c.vpslld(zmm_tmp1, zmm_tmp1, 23);        // n << 23 (FP32 exponent position)
    c.vpaddd(zmm_tmp2, zmm_tmp2, zmm_tmp1);  // Add to exponent bits
    
    c.vmovaps(zmm_x, zmm_tmp2);              // Result in zmm_x
}
```

#### 3.4 Interleaved V Accumulation

**File:** `jit/JitVWeightedAccum.h`

```cpp
/**
 * @brief Generate interleaved V accumulation to hide memory latency
 * 
 * Strategy: Start loading V[i+1] while computing context += w[i] * V[i]
 * 
 * Context layout: zmm_ctx0..zmm_ctx3 hold 64 floats (head_dim=64)
 * Weights: w0..w3 broadcast in zmm_w0..zmm_w3
 * V pointer advances through V_tile
 */
void generate_interleaved_v_accum(
    Xbyak::CodeGenerator& c,
    const Xbyak::Reg64& V_ptr,
    int64_t v_stride,
    int tile_size,
    int num_blocks,  // head_dim / 32
    const Xbyak::Zmm& zmm_ctx0,  // Context accumulators
    const Xbyak::Zmm& zmm_ctx1,
    const Xbyak::Zmm& zmm_w_base)  // Weights start here
{
    using namespace Xbyak;
    
    Zmm zmm_v0 = zmm28, zmm_v1 = zmm29;
    Zmm zmm_scale0 = zmm30, zmm_scale1 = zmm31;
    
    // Prefetch first V row
    c.vmovups(zmm_v0, c.ptr[V_ptr]);
    
    for (int i = 0; i < tile_size; i += 2) {
        // Weight for V[i] (broadcast from weight array)
        int w_reg_idx = (i / 16);  // Which weight ZMM
        int w_lane = i % 16;       // Which lane in that ZMM
        c.vbroadcastss(zmm_scale0, c.ptr[rsp + weight_offset + i * 4]);
        
        // Start loading V[i+1] while we compute V[i]
        if (i + 1 < tile_size) {
            c.vmovups(zmm_v1, c.ptr[V_ptr + (i+1) * v_stride]);
        }
        
        // Accumulate V[i]: context += weight[i] * V[i]
        // For Q8_1: dequantize V[i], then FMA
        // (Simplified: assuming FP32 V for illustration)
        c.vfmadd231ps(zmm_ctx0, zmm_v0, zmm_scale0);
        
        // Weight for V[i+1]
        if (i + 1 < tile_size) {
            c.vbroadcastss(zmm_scale1, c.ptr[rsp + weight_offset + (i+1) * 4]);
            
            // Start loading V[i+2]
            if (i + 2 < tile_size) {
                c.vmovups(zmm_v0, c.ptr[V_ptr + (i+2) * v_stride]);
            }
            
            // Accumulate V[i+1]
            c.vfmadd231ps(zmm_ctx0, zmm_v1, zmm_scale1);
        }
    }
}
```

---

## Phase 4: Testing Strategy

The key to safely upgrading to FA2 is maintaining **bit-level correctness** against a trusted reference implementation. We already have a proven FP32 reference kernel that produces correct attention outputs. All FA2 optimizations must produce outputs within tolerance of this reference.

### 4.1 Reference Kernel Hierarchy

We establish a **chain of trust** where each faster implementation is validated against a simpler, more obviously correct one:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Level 0: FP32 Reference (Ground Truth)                                  │
│ ─────────────────────────────────────────────────────────────────────── │
│ File: kernels/cpu/attention/FP32AttentionReference.cpp                  │
│ Algorithm: Naive triple-nested loop, FP32 throughout                    │
│ Purpose: Mathematically correct, human-readable, no optimizations       │
│ Tolerance: Exact (this IS the ground truth)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ validate against
┌─────────────────────────────────────────────────────────────────────────┐
│ Level 1: C++ Online Softmax Reference                                   │
│ ─────────────────────────────────────────────────────────────────────── │
│ File: kernels/cpu/attention/q8_1/ref/FusedAttentionWoRef.cpp            │
│ Algorithm: Online softmax (FA1-style), per-position updates             │
│ Purpose: Validate online softmax correctness                            │
│ Tolerance: < 1e-5 max abs diff vs Level 0                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ validate against
┌─────────────────────────────────────────────────────────────────────────┐
│ Level 2: C++ Tiled Implementation (Current FA1)                         │
│ ─────────────────────────────────────────────────────────────────────── │
│ File: kernels/cpu/attention/q8_1/tiled/FusedAttentionWoTiled.cpp        │
│ Algorithm: Cache-blocked tiling, online softmax per-position            │
│ Purpose: Validate tiling doesn't break correctness                      │
│ Tolerance: < 1e-5 max abs diff vs Level 1                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ validate against
┌─────────────────────────────────────────────────────────────────────────┐
│ Level 3: JIT FA1 Implementation (Current)                               │
│ ─────────────────────────────────────────────────────────────────────── │
│ File: kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h              │
│ Algorithm: AVX-512 VNNI JIT, per-position softmax                       │
│ Purpose: Validate JIT code generation correctness                       │
│ Tolerance: < 1e-4 max abs diff vs Level 2 (quantization noise)          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ validate against (NEW)
┌─────────────────────────────────────────────────────────────────────────┐
│ Level 4: JIT FA2 Implementation (Target)                                │
│ ─────────────────────────────────────────────────────────────────────── │
│ File: kernels/cpu/attention/q8_1/jit/JitFusedAttentionWoFA2.h           │
│ Algorithm: AVX-512 VNNI JIT, tile-wise softmax batching                 │
│ Purpose: FA2 optimizations with proven correctness                      │
│ Tolerance: < 1e-4 max abs diff vs Level 3                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Test Categories

#### Category A: Microkernel Unit Tests

Each microkernel has its own unit test validating against a reference implementation.

| Microkernel | Reference | Test File | Tolerance |
|-------------|-----------|-----------|-----------|
| `q8_dot_product_4x` | `q8_dot_product_ref` (scalar) | `Test__Q8DotProduct4x.cpp` | Exact (integer) |
| `tile_max_reduction` | `std::max_element` | `Test__TileMaxReduction.cpp` | Exact |
| `fast_exp_16x` | `std::exp` | `Test__FastExp16x.cpp` | < 1e-6 relative |
| `tile_softmax_update` | Per-position loop | `Test__TileSoftmaxUpdate.cpp` | < 1e-5 |
| `interleaved_v_accum` | Sequential accum | `Test__InterleavedVAccum.cpp` | < 1e-5 |

```cpp
// Example: Test__Q8DotProduct4x.cpp
TEST(Test__Q8DotProduct4x, MatchesScalarReference) {
    for (int trial = 0; trial < 100; ++trial) {
        auto Q = TestTensorFactory::createQ8_1Random({1, 64});
        auto K = TestTensorFactory::createQ8_1Random({4, 64});  // 4 KV positions
        
        // Reference: 4 separate scalar dot products
        float ref_scores[4];
        for (int i = 0; i < 4; ++i) {
            ref_scores[i] = q8_dot_product_ref(Q, K.row(i), 64);
        }
        
        // Optimized: vectorized 4x dot product
        float opt_scores[4];
        q8_dot_product_4x(Q, K, 64, opt_scores);
        
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(ref_scores[i], opt_scores[i]) 
                << "Mismatch at position " << i << " trial " << trial;
        }
    }
}
```

#### Category B: Algorithm Equivalence Tests

Validate that FA2 algorithmic changes produce identical outputs.

```cpp
// Test__FA2TileSoftmax.cpp

/**
 * @test Tile-wise softmax matches per-position softmax exactly
 * 
 * This is the CRITICAL correctness test. If tile-wise processing
 * produces different results than per-position, the FA2 upgrade is broken.
 */
TEST(Test__FA2TileSoftmax, TileVsPerPositionEquivalence) {
    // Test multiple configurations
    const std::vector<std::tuple<int, int, int>> configs = {
        // {seq_len, kv_len, kv_tile}
        {1, 64, 16},      // Single query, short KV
        {1, 256, 32},     // Single query, medium KV
        {1, 2048, 64},    // Single query, long KV (typical decode)
        {128, 128, 32},   // Square attention (prefill)
        {512, 512, 64},   // Large prefill
        {1, 127, 32},     // Non-tile-aligned KV length
        {1, 65, 64},      // KV length slightly > tile
    };
    
    for (const auto& [seq_len, kv_len, kv_tile] : configs) {
        SCOPED_TRACE("seq=" + std::to_string(seq_len) + 
                     " kv=" + std::to_string(kv_len) + 
                     " tile=" + std::to_string(kv_tile));
        
        auto Q = TestTensorFactory::createQ8_1Random({seq_len, 64});
        auto K = TestTensorFactory::createQ8_1Random({kv_len, 64});
        auto V = TestTensorFactory::createQ8_1Random({kv_len, 64});
        
        // Reference: per-position online softmax (Level 2)
        auto output_ref = compute_attention_per_position(Q, K, V);
        
        // New: tile-wise softmax (Level 4)
        auto output_tiled = compute_attention_tile_batched(Q, K, V, kv_tile);
        
        float max_diff = TestTensorFactory::maxAbsDiff(output_ref, output_tiled);
        float cosine = TestTensorFactory::cosineSimilarity(output_ref, output_tiled);
        
        EXPECT_LT(max_diff, 1e-5f) << "Max diff too high";
        EXPECT_GT(cosine, 0.99999f) << "Cosine similarity too low";
    }
}

/**
 * @test Correction factor handles max transitions correctly
 * 
 * The trickiest part of tile-wise softmax is applying the correction
 * when a new tile has a higher max than previous tiles.
 */
TEST(Test__FA2TileSoftmax, CorrectionFactorStressTest) {
    // Construct adversarial input: each tile has increasing max
    const int num_tiles = 8;
    const int tile_size = 32;
    const int kv_len = num_tiles * tile_size;
    const int head_dim = 64;
    
    // Create scores where tile i has max = i * 2.0
    std::vector<float> adversarial_K(kv_len * head_dim);
    for (int t = 0; t < num_tiles; ++t) {
        float target_max = t * 2.0f;
        for (int i = 0; i < tile_size; ++i) {
            int idx = (t * tile_size + i) * head_dim;
            // Set K values to produce desired dot product range
            for (int d = 0; d < head_dim; ++d) {
                adversarial_K[idx + d] = (target_max / head_dim) * (d < head_dim/2 ? 1.0f : -1.0f);
            }
        }
    }
    
    auto Q = TestTensorFactory::createQ8_1Ones({1, head_dim});
    auto K = TestTensorFactory::createQ8_1FromFP32(adversarial_K.data(), {kv_len, head_dim});
    auto V = TestTensorFactory::createQ8_1Random({kv_len, head_dim});
    
    auto output_ref = compute_attention_per_position(Q, K, V);
    auto output_tiled = compute_attention_tile_batched(Q, K, V, tile_size);
    
    float max_diff = TestTensorFactory::maxAbsDiff(output_ref, output_tiled);
    EXPECT_LT(max_diff, 1e-4f) << "Correction factor not applied correctly";
}
```

#### Category C: Real Model Integration Tests

Test with actual model weights to catch issues that synthetic tests miss.

```cpp
// Test__FA2RealModel.cpp

/**
 * @test FA2 kernel produces same output as FA1 on Qwen 0.5B layer 0
 */
TEST(Test__FA2RealModel, Qwen05B_Layer0_Parity) {
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    auto model_ctx = ModelContext::create(model_path);
    if (!model_ctx) GTEST_SKIP() << "Model not found";
    
    // Load real QKV weights
    Qwen2SchemaFactory schema;
    model_ctx->weightManager()->setWeightShardingConfig(schema.getWeightShardingConfig());
    
    auto wq = model_ctx->getWeight("blk.0.attn_q.weight", -1);
    auto wk = model_ctx->getWeight("blk.0.attn_k.weight", -1);
    auto wv = model_ctx->getWeight("blk.0.attn_v.weight", -1);
    auto wo = model_ctx->getWeight("blk.0.attn_output.weight", -1);
    
    // Test prompts of varying lengths
    const std::vector<std::vector<int>> test_prompts = {
        {9906, 11, 1879},                    // "Hello, world" (3 tokens)
        {791, 4335, 2579, 30354, 11609},     // 5 tokens
        // ... longer sequences for prefill testing
    };
    
    for (const auto& tokens : test_prompts) {
        SCOPED_TRACE("seq_len=" + std::to_string(tokens.size()));
        
        // Compute QKV from embeddings
        auto [Q, K, V] = compute_qkv_projections(model_ctx, tokens, wq, wk, wv);
        
        // FA1 reference (current implementation)
        auto output_fa1 = fused_attention_wo_fa1(Q, K, V, wo);
        
        // FA2 new implementation
        auto output_fa2 = fused_attention_wo_fa2(Q, K, V, wo);
        
        float max_diff = TestTensorFactory::maxAbsDiff(output_fa1, output_fa2);
        float cosine = TestTensorFactory::cosineSimilarity(output_fa1, output_fa2);
        
        EXPECT_LT(max_diff, 1e-4f);
        EXPECT_GT(cosine, 0.9999f);
    }
}

/**
 * @test Full E2E inference produces same tokens with FA1 vs FA2
 */
TEST(Test__FA2RealModel, E2E_TokenParity) {
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
    const std::string prompt = "The capital of France is";
    const int max_tokens = 20;
    
    // Run with FA1 kernel
    auto tokens_fa1 = run_inference_with_kernel(model_path, prompt, max_tokens, KernelType::FA1);
    
    // Run with FA2 kernel  
    auto tokens_fa2 = run_inference_with_kernel(model_path, prompt, max_tokens, KernelType::FA2);
    
    // Tokens should match exactly (greedy sampling, temperature=0)
    ASSERT_EQ(tokens_fa1.size(), tokens_fa2.size());
    for (size_t i = 0; i < tokens_fa1.size(); ++i) {
        EXPECT_EQ(tokens_fa1[i], tokens_fa2[i]) 
            << "Token mismatch at position " << i 
            << ": FA1=" << tokens_fa1[i] << " FA2=" << tokens_fa2[i];
    }
}
```

#### Category D: Edge Case Tests

Specifically test boundary conditions and corner cases.

```cpp
// Test__FA2EdgeCases.cpp

TEST(Test__FA2EdgeCases, SingleKVPosition) {
    // kv_len = 1, tile processing must handle this
    auto Q = TestTensorFactory::createQ8_1Random({1, 64});
    auto K = TestTensorFactory::createQ8_1Random({1, 64});
    auto V = TestTensorFactory::createQ8_1Random({1, 64});
    
    auto ref = compute_attention_per_position(Q, K, V);
    auto tiled = compute_attention_tile_batched(Q, K, V, 32);
    
    EXPECT_LT(TestTensorFactory::maxAbsDiff(ref, tiled), 1e-5f);
}

TEST(Test__FA2EdgeCases, KVLengthEqualsTileSize) {
    auto Q = TestTensorFactory::createQ8_1Random({1, 64});
    auto K = TestTensorFactory::createQ8_1Random({32, 64});  // Exactly one tile
    auto V = TestTensorFactory::createQ8_1Random({32, 64});
    
    auto ref = compute_attention_per_position(Q, K, V);
    auto tiled = compute_attention_tile_batched(Q, K, V, 32);
    
    EXPECT_LT(TestTensorFactory::maxAbsDiff(ref, tiled), 1e-5f);
}

TEST(Test__FA2EdgeCases, KVLengthOneMoreThanTile) {
    auto Q = TestTensorFactory::createQ8_1Random({1, 64});
    auto K = TestTensorFactory::createQ8_1Random({33, 64});  // Tile + 1
    auto V = TestTensorFactory::createQ8_1Random({33, 64});
    
    auto ref = compute_attention_per_position(Q, K, V);
    auto tiled = compute_attention_tile_batched(Q, K, V, 32);
    
    EXPECT_LT(TestTensorFactory::maxAbsDiff(ref, tiled), 1e-5f);
}

TEST(Test__FA2EdgeCases, VeryLargeScores) {
    // Test numerical stability with large dot products
    auto Q = TestTensorFactory::createQ8_1(127);  // Max Q8_1 value
    auto K = TestTensorFactory::createQ8_1(127);  // Will produce large scores
    auto V = TestTensorFactory::createQ8_1Random({64, 64});
    
    // Should not overflow or produce NaN
    auto output = compute_attention_tile_batched(Q, K, V, 32);
    EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(output));
}

TEST(Test__FA2EdgeCases, AllEqualScores) {
    // When all scores are equal, weights should be uniform
    auto Q = TestTensorFactory::createQ8_1Zeros({1, 64});
    auto K = TestTensorFactory::createQ8_1Zeros({64, 64});
    auto V = TestTensorFactory::createQ8_1Random({64, 64});
    
    auto ref = compute_attention_per_position(Q, K, V);
    auto tiled = compute_attention_tile_batched(Q, K, V, 32);
    
    EXPECT_LT(TestTensorFactory::maxAbsDiff(ref, tiled), 1e-5f);
}

TEST(Test__FA2EdgeCases, CausalMaskPartialTile) {
    // Causal mask where tile straddles the diagonal
    const int seq_len = 5;
    const int kv_tile = 4;
    
    auto Q = TestTensorFactory::createQ8_1Random({seq_len, 64});
    auto K = TestTensorFactory::createQ8_1Random({seq_len, 64});
    auto V = TestTensorFactory::createQ8_1Random({seq_len, 64});
    auto mask = create_causal_mask(seq_len, seq_len);
    
    auto ref = compute_attention_per_position_masked(Q, K, V, mask);
    auto tiled = compute_attention_tile_batched_masked(Q, K, V, mask, kv_tile);
    
    EXPECT_LT(TestTensorFactory::maxAbsDiff(ref, tiled), 1e-5f);
}
```

### 4.3 Performance Regression Tests

Ensure FA2 is actually faster, not just correct.

```cpp
// Perf__FA2Comparison.cpp

/**
 * @benchmark FA2 must be at least 1.2x faster than FA1 on prefill
 */
TEST(Perf__FA2Comparison, PrefillSpeedup) {
    const int seq_len = 512;
    const int head_dim = 64;
    const int num_heads = 14;
    const int warmup = 5;
    const int iters = 20;
    
    auto Q = TestTensorFactory::createQ8_1Random({seq_len, num_heads * head_dim});
    auto K = TestTensorFactory::createQ8_1Random({seq_len, num_heads * head_dim});
    auto V = TestTensorFactory::createQ8_1Random({seq_len, num_heads * head_dim});
    
    // Warmup and benchmark FA1
    for (int i = 0; i < warmup; ++i) fused_attention_fa1(Q, K, V);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fused_attention_fa1(Q, K, V);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fa1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    
    // Warmup and benchmark FA2
    for (int i = 0; i < warmup; ++i) fused_attention_fa2(Q, K, V);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fused_attention_fa2(Q, K, V);
    t1 = std::chrono::high_resolution_clock::now();
    double fa2_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    
    double speedup = fa1_ms / fa2_ms;
    LOG_INFO("FA1: " << fa1_ms << " ms, FA2: " << fa2_ms << " ms, Speedup: " << speedup << "x");
    
    EXPECT_GT(speedup, 1.2) << "FA2 should be at least 1.2x faster than FA1";
}

/**
 * @benchmark Decode latency should not regress
 */
TEST(Perf__FA2Comparison, DecodeLatency) {
    const int kv_len = 2048;
    const int head_dim = 64;
    const int warmup = 10;
    const int iters = 100;
    
    auto Q = TestTensorFactory::createQ8_1Random({1, head_dim});
    auto K = TestTensorFactory::createQ8_1Random({kv_len, head_dim});
    auto V = TestTensorFactory::createQ8_1Random({kv_len, head_dim});
    
    // Benchmark FA1 decode
    for (int i = 0; i < warmup; ++i) fused_attention_fa1(Q, K, V);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fused_attention_fa1(Q, K, V);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fa1_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    
    // Benchmark FA2 decode
    for (int i = 0; i < warmup; ++i) fused_attention_fa2(Q, K, V);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fused_attention_fa2(Q, K, V);
    t1 = std::chrono::high_resolution_clock::now();
    double fa2_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    
    LOG_INFO("FA1: " << fa1_us << " μs, FA2: " << fa2_us << " μs");
    
    // FA2 should not be slower than FA1 on decode
    EXPECT_LE(fa2_us, fa1_us * 1.05) << "FA2 decode should not regress vs FA1";
}
```

### 4.4 Test Execution Plan

Tests are run incrementally as each phase is implemented:

| Phase | Tests to Run | Pass Criteria |
|-------|--------------|---------------|
| 1 (OpenMP) | All existing tests | No regressions |
| 2.1 (Tile softmax C++) | `Test__FA2TileSoftmax` | All equivalence tests pass |
| 2.2 (Integrate tiled) | Category B + C | < 1e-5 max diff |
| 3.x (JIT microkernels) | Category A for each | Exact or < 1e-6 |
| 3.5 (JIT integration) | All Category B, C, D | < 1e-4 max diff |
| 4.1 (Performance) | Category D perf tests | ≥ 1.2x speedup |

### 4.5 Running the FA2 Test Suite

```bash
# Full FA2 correctness suite (after implementation)
ctest --test-dir build_v2_release \
  -R "FA2|FusedAttention|TileSoftmax" \
  --output-on-failure --parallel

# Equivalence tests only
ctest --test-dir build_v2_release -R "FA2.*Equivalence|FA2.*Parity"

# Edge cases only  
ctest --test-dir build_v2_release -R "FA2EdgeCases"

# Performance regression tests
ctest --test-dir build_v2_release -R "Perf__FA2" --verbose
```

---

## Phase 5: Register Guard Enhancement (Compile-Time Enforcement)

**Effort:** Low-Medium (2-3 days)  
**Impact:** High (eliminates entire class of bugs)  
**Risk:** Low (infrastructure change, no algorithm changes)  
**Status:** 🔴 Not Started

### 5.1 Problem Statement

The current register guard system (`RegisterGuard.h` + `RegisterAllocation.h`) provides:
- **Compile-time typed registers**: `TypedZmm<Zone, Index>`, `TypedXmm<Zone, Index>`
- **Zone membership verification**: SFINAE helpers like `require_zone<Reg, Zone>`
- **Runtime conflict detection**: `RegisterTracker` + `RegisterGuard<T>` RAII wrappers
- **Overlap awareness**: Static assertions for Score/Scratch aliasing

**What's missing:** The system is **opt-in**. Developers can still use raw `Xbyak::Zmm(20)` or `gen.zmm0` directly, bypassing all safeguards. This creates a gap where:

1. New code may forget to use guards
2. Legacy code may use raw registers
3. Copy-paste from Xbyak examples uses raw registers
4. No compiler error when guards are bypassed

### 5.2 Solution: Compile-Time Enforcement

Create a **sealed register system** where raw Xbyak register usage causes compilation failures in JIT code.

#### 5.2.1 Sealed Register Macros

```cpp
// In RegisterAllocation.h (or new RegisterEnforcement.h)

#ifdef LLAMINAR_ENFORCE_REGISTER_GUARDS

// These macros poison direct Xbyak register access within JIT kernels.
// Any code using raw zmm0, zmm1, etc. will fail to compile.

// Poison individual register names (preprocessor level)
#define zmm0 REGISTER_GUARD_REQUIRED_USE_Accum0_INSTEAD
#define zmm1 REGISTER_GUARD_REQUIRED_USE_Accum1_INSTEAD
// ... etc for all 32 ZMM registers ...
#define zmm20 REGISTER_GUARD_REQUIRED_USE_Scratch0_OR_Score0_INSTEAD
// ... etc ...

// Alternative: Template specialization that produces helpful error
namespace llaminar2::jit::internal {
    template<int N>
    struct RawZmmBlocked {
        static_assert(N < 0, 
            "Direct Xbyak::Zmm(N) usage is prohibited. "
            "Use typed registers: Accum0-7, Input0-7, Scratch0-5, Score0-3, State*, etc. "
            "See RegisterAllocation.h for the full type system.");
    };
}

// Intercept Xbyak::Zmm constructor attempts
#define Zmm(n) ::llaminar2::jit::internal::RawZmmBlocked<n>{}

#endif // LLAMINAR_ENFORCE_REGISTER_GUARDS
```

#### 5.2.2 Scoped Enforcement with Pragma Push/Pop

For incremental migration, allow per-file or per-function enforcement:

```cpp
// Enable enforcement in this file
#define LLAMINAR_ENFORCE_REGISTER_GUARDS
#include "kernels/cpu/jit/RegisterAllocation.h"

// Or use pragmas for specific regions
REGISTER_GUARDS_BEGIN  // Enables poisoning
void emit_critical_section() {
    // zmm0 here would cause compile error
    auto acc = tracker.borrow<Accum0>();  // OK
    gen.vmovaps(acc.zmm(), ...);          // OK - .zmm() accessor still works
}
REGISTER_GUARDS_END    // Disables poisoning
```

#### 5.2.3 Base Class Enforcement Pattern

For JIT kernel classes, enforce via inheritance:

```cpp
// New file: JitKernelBase.h

/**
 * @brief Base class for JIT kernels with enforced register guarding
 * 
 * Inherit from this to get:
 * 1. Built-in RegisterTracker
 * 2. Typed register accessor methods
 * 3. Compile-time enforcement (no raw Zmm allowed)
 */
class JitKernelWithGuards : public Xbyak::CodeGenerator {
protected:
    RegisterTracker reg_tracker_;
    
    // Factory methods return guards, not raw registers
    template<typename RegType>
    [[nodiscard]] RegisterGuard<RegType> borrow() {
        return reg_tracker_.borrow<RegType>();
    }
    
    // Convenience for common operations (return guards)
    [[nodiscard]] auto borrow_accum0() { return borrow<Accum0>(); }
    [[nodiscard]] auto borrow_accum1() { return borrow<Accum1>(); }
    [[nodiscard]] auto borrow_scratch4() { return borrow<Scratch4>(); }
    [[nodiscard]] auto borrow_scratch5() { return borrow<Scratch5>(); }
    // ... etc ...
    
    // DELETED: Raw register accessors
    // (Inheriting classes cannot accidentally use gen.zmm0, etc.)
    
private:
    // Hide Xbyak's direct register access
    using Xbyak::CodeGenerator::zmm0;   // Make private
    using Xbyak::CodeGenerator::zmm1;
    // ... etc for all zmm/ymm/xmm ...
};
```

#### 5.2.4 Static Analysis Custom Check

For CI integration, add a clang-tidy or grep-based check:

```bash
#!/bin/bash
# scripts/check_register_guards.sh

# Find raw Zmm/Xmm usage in JIT files (excluding RegisterAllocation.h itself)
if grep -rn --include="*.h" --include="*.cpp" \
    'Xbyak::Zmm([0-9]' \
    src/v2/kernels/cpu/attention/ \
    src/v2/kernels/cpu/jit/ \
    | grep -v RegisterAllocation.h \
    | grep -v RegisterGuard.h \
    | grep -v "\.zmm()" \
    | grep -v "// RAW_OK"; then
    echo "ERROR: Raw Xbyak::Zmm usage detected. Use typed registers!"
    exit 1
fi

echo "✓ All JIT register usage goes through guards"
```

### 5.3 Migration Plan

#### Step 1: Audit Current Usage

Current JIT files and their register guard status:

| File | Guards Used | Raw Xbyak | Action |
|------|-------------|-----------|--------|
| `JitFusedAttentionWo.h` | ✅ Partial | ⚠️ Some | Migrate remaining |
| `JitOnlineSoftmax.h` | ✅ Yes | ⚠️ `zmm_max`, etc. | Convert accessors |
| `JitFastExp.h` | ✅ Yes | ✅ None | ✓ Already compliant |
| `JitQ8DotProduct.h` | ✅ Yes | ✅ None | ✓ Already compliant |
| `JitVWeightedAccum.h` | ✅ Yes | ✅ None | ✓ Already compliant |
| `JitWoProjection.h` | ⚠️ Partial | ⚠️ Some | Migrate |
| `JitMicrokernelBase.h` | ❌ None | ⚠️ Legacy | Full migration |

#### Step 2: Create Typed Accessors for Common Patterns

Currently `JitMicrokernelBase.h` has:
```cpp
Xbyak::Zmm zmm_max() const { return Xbyak::Zmm(StateRegs::ZMM_MAX); }  // BAD
```

Replace with:
```cpp
StateMax state_max() const { return StateMax{}; }  // GOOD - returns typed register
```

#### Step 3: Add [[nodiscard]] to All Borrow Methods

Ensure guards aren't accidentally discarded:

```cpp
// BAD: Guard immediately destroyed, register not actually protected
tracker.borrow<Scratch0>();  // Compiler warning: discarded [[nodiscard]]

// GOOD: Guard lives until end of scope
auto guard = tracker.borrow<Scratch0>();
```

#### Step 4: Enable Enforcement Incrementally

1. First, enable in new code only (`#define` in new files)
2. Then, migrate legacy files one by one
3. Finally, enable globally in CMakeLists.txt

### 5.4 Enhanced RegisterAllocation.h Additions

```cpp
// ============================================================================
// Compile-Time Register Name Lookup (for better error messages)
// ============================================================================

template<int AbsoluteIndex>
struct RegisterName {
    static constexpr const char* value = 
        (AbsoluteIndex < 8) ? "Accum" :
        (AbsoluteIndex < 16) ? "Input" :
        (AbsoluteIndex < 20) ? "State" :
        (AbsoluteIndex < 26) ? "Scratch" :
        "Reserved";
    
    static constexpr int local_index = 
        (AbsoluteIndex < 8) ? AbsoluteIndex :
        (AbsoluteIndex < 16) ? AbsoluteIndex - 8 :
        (AbsoluteIndex < 20) ? AbsoluteIndex - 16 :
        (AbsoluteIndex < 26) ? AbsoluteIndex - 20 :
        AbsoluteIndex - 26;
};

// ============================================================================
// Register Usage Manifest (for multi-kernel coordination)
// ============================================================================

/**
 * @brief Declare a kernel's complete register footprint
 * 
 * Used to verify that composed kernels don't have conflicting requirements.
 */
template<typename... Regs>
struct KernelRegisterManifest {
    static constexpr size_t count = sizeof...(Regs);
    
    // Check compatibility with another manifest
    template<typename... OtherRegs>
    static constexpr bool compatible_with() {
        // Two manifests are compatible if they can run sequentially
        // (i.e., no register needs to persist across the boundary)
        return true;  // Conservative default
    }
    
    // Check if all registers are from allowed zones
    template<typename... AllowedZones>
    static constexpr bool restricted_to() {
        return (is_any_zone_v<Regs, AllowedZones...> && ...);
    }
};

// Example usage:
// using FastExpManifest = KernelRegisterManifest<Scratch4, Scratch5, Input6, Input7>;
// static_assert(FastExpManifest::restricted_to<ScratchZone, QVectorZone>());

// ============================================================================
// Guarded Register Reference (Compile-Time Safe, Zero Runtime Cost)
// ============================================================================

/**
 * @brief A reference to a typed register that REQUIRES borrowing
 * 
 * Unlike TypedZmm which can be used directly, GuardedReg<T> has no
 * .zmm() accessor. You MUST call .borrow(tracker) to get a guard.
 */
template<typename RegType>
struct GuardedReg {
    using reg_type = RegType;
    
    // NO direct accessor - forces borrow pattern
    // Xbyak::Zmm zmm() = delete;
    
    // Only way to use: borrow first
    [[nodiscard]] RegisterGuard<RegType> borrow(RegisterTracker& tracker) {
        return tracker.borrow<RegType>();
    }
};

// Pre-defined guarded registers (use these in new code)
namespace guarded {
    inline constexpr GuardedReg<Accum0> accum0{};
    inline constexpr GuardedReg<Accum1> accum1{};
    // ... etc ...
    inline constexpr GuardedReg<Scratch4> scratch4{};
    inline constexpr GuardedReg<Scratch5> scratch5{};
}
```

### 5.5 Testing Strategy

#### Unit Tests

```cpp
TEST(RegisterGuardEnforcement, BorrowReturnsWorkingRegister) {
    RegisterTracker tracker;
    auto guard = tracker.borrow<Scratch4>();
    
    // Can access underlying Xbyak register
    EXPECT_EQ(guard.zmm().getIdx(), 24);
    EXPECT_EQ(guard.xmm().getIdx(), 24);
}

TEST(RegisterGuardEnforcement, DoubleConflictDetected) {
    RegisterTracker tracker;
    auto g1 = tracker.borrow<Score0>();  // Physical reg 20
    
    EXPECT_DEATH({
        auto g2 = tracker.borrow<Scratch0>();  // Also physical reg 20 - CONFLICT
    }, "REGISTER CONFLICT DETECTED");
}

TEST(RegisterGuardEnforcement, NodiscardWarning) {
    // This test verifies [[nodiscard]] is present
    // (The actual warning is compile-time, this is a documentation test)
    RegisterTracker tracker;
    
    // This should generate a compiler warning:
    // tracker.borrow<Scratch0>();  // Warning: ignoring return value
    
    // Correct usage:
    [[maybe_unused]] auto guard = tracker.borrow<Scratch0>();
}

TEST(RegisterGuardEnforcement, ManifestConflictDetected) {
    // Verify that conflicting manifests are caught at compile time
    using Manifest1 = KernelRegisterManifest<Scratch0, Scratch1>;
    using Manifest2 = KernelRegisterManifest<Score0, Score1>;  // Aliases Scratch0/1
    
    // This should fail at compile time:
    // static_assert(!Manifest1::conflicts_with<Manifest2::regs...>());
}
```

#### Integration Tests

```cpp
TEST(RegisterGuardMigration, JitFastExpUsesGuards) {
    // Verify JitFastExp uses guards (no raw Zmm)
    // This is a documentation test - actual enforcement is compile-time
    
    JitFastExpEmitter emitter;
    // If JitFastExpEmitter internally uses raw Zmm, this test file
    // won't compile when LLAMINAR_ENFORCE_REGISTER_GUARDS is defined
}
```

### 5.6 Implementation Order

| Step | Task | Effort |
|------|------|--------|
| 5.0.1 | Add `[[nodiscard]]` to all borrow methods | 0.5 day |
| 5.0.2 | Create `check_register_guards.sh` CI script | 0.5 day |
| 5.0.3 | Add `GuardedReg<T>` template and `guarded::` namespace | 0.5 day |
| 5.0.4 | Migrate `JitMicrokernelBase.h` raw accessors | 1 day |
| 5.0.5 | Migrate remaining raw usage in `JitFusedAttentionWo.h` | 0.5 day |
| 5.0.6 | Add `KernelRegisterManifest` for kernel composition | 0.5 day |
| 5.0.7 | Enable `LLAMINAR_ENFORCE_REGISTER_GUARDS` globally | Included |

**Total Effort:** ~3.5 days

### 5.7 Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Raw `Xbyak::Zmm(N)` in JIT code | ~20 instances | 0 |
| `[[nodiscard]]` on borrow methods | Partial | 100% |
| CI enforcement script | None | Pass required |
| Test coverage for guards | Basic | Full (conflict, alias, nodiscard) |

---

## Phase 6: High-Performance Wo Projection

**Effort:** High  
**Impact:** Very High (5-6× speedup on Qwen 7B decode)  
**Risk:** Medium (significant JIT changes)  
**Status:** 🔴 Not Started

### 6.1 Problem Statement

Performance analysis (December 23, 2025) revealed that the fused Wo projection is the **dominant bottleneck**, not attention:

| Model | Attention % of FLOPs | Wo % of FLOPs |
|-------|----------------------|---------------|
| Qwen 0.5B decode (kv=128) | 22.6% | **77.4%** |
| Qwen 7B decode (kv=128) | 6.7% | **93.3%** |
| Qwen 7B prefill (seq=128) | 3.5% | **96.5%** |

The current fused Wo implementation achieves only **~8-9 GFLOP/s** while standalone OpenBLAS SGEMM achieves:

| Operation | Fused Kernel | OpenBLAS | Ratio |
|-----------|-------------|----------|-------|
| `[1 × 3584] × [3584 × 3584]` (decode) | ~8.6 GFLOP/s | **54 GFLOP/s** | **6.3× slower** |
| `[128 × 3584] × [3584 × 3584]` (prefill) | ~8.2 GFLOP/s | **1508 GFLOP/s** | **184× slower** |

### 6.2 Root Cause Analysis

The current JIT Wo projection (`emit_prefill_wo_fp32`) uses a **naive row-by-row dot product**:

```cpp
// Current implementation (SLOW)
void emit_prefill_wo_fp32(reg_Wo, reg_out, reg_ctx_off, d_model) {
    for (row = 0; row < d_model; ++row) {         // Outer: output rows
        acc = 0;
        for (col = 0; col < d_model; col += 16) { // Inner: dot product
            ctx = load(context[col:col+15]);
            wo = load(Wo[row, col:col+15]);
            acc += dot(ctx, wo);
        }
        store(output[row], reduce(acc));
    }
}
```

**Problems:**
1. **No cache blocking**: Wo matrix (49 MB for Qwen 7B) constantly evicted from cache
2. **Single accumulator**: No instruction-level parallelism (ILP)
3. **No prefetching**: Memory latency not hidden
4. **Row-by-row access**: Poor spatial locality for Wo reads
5. **Sequential output**: Each output element computed independently

### 6.3 Solution: BLAS-Style Microkernel

High-performance GEMM implementations use:

1. **Register blocking**: Compute an MR×NR output tile per inner loop iteration
2. **Cache blocking**: Tile at L1/L2/L3 cache sizes
3. **Packing**: Repack matrices for sequential access
4. **Software prefetching**: Hide memory latency
5. **Multiple accumulators**: Maximize FMA throughput

For the fused kernel, we have a choice:

#### Option A: Unfuse Wo (Call OpenBLAS)

**Pros:** Immediate 5-6× speedup, minimal code changes  
**Cons:** Extra memory traffic (context written then read), loses fusion benefit

```cpp
// After attention completes, call OpenBLAS
void fused_attention_wo_with_blas(Q, K, V, Wo, output, seq_len, kv_len) {
    // 1. Compute attention into context buffer
    float* context = alloc_aligned(seq_len * d_model * sizeof(float));
    compute_attention_only(Q, K, V, context, seq_len, kv_len);
    
    // 2. Call OpenBLAS for Wo projection
    // output[seq_len, d_model] = context[seq_len, d_model] × Wo[d_model, d_model]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, d_model, d_model,
                1.0f, context, d_model,
                Wo, d_model,
                0.0f, output, d_model);
    
    free(context);
}
```

#### Option B: JIT BLAS-Quality Microkernel

**Pros:** Maintains fusion, potentially better than BLAS for small M  
**Cons:** Significant implementation effort, needs careful tuning

We implement a **6×16 microkernel** following the BLIS/OpenBLAS approach:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    GEMM MICROKERNEL DESIGN                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  For C[M,N] = A[M,K] × B[K,N]:                                         │
│                                                                        │
│  ┌─────┐     ┌─────────────────────────────────────────────┐           │
│  │  A  │     │                      B                      │           │
│  │ MR  │  ×  │                      NR                     │           │
│  │ × K │     │                   K × NR                    │           │
│  └─────┘     └─────────────────────────────────────────────┘           │
│      │                           │                                     │
│      └──────────────┬────────────┘                                     │
│                     ▼                                                  │
│              ┌───────────┐                                             │
│              │     C     │                                             │
│              │  MR × NR  │  ← MR×NR accumulators in registers          │
│              └───────────┘                                             │
│                                                                        │
│  For AVX-512:                                                          │
│    NR = 16 (one ZMM register width)                                    │
│    MR = 6  (6 output rows, 6 accumulators)                             │
│    → 6 × 16 = 96 output elements per microkernel iteration             │
│    → 6 ZMM accumulators (zmm0-5)                                       │
│    → 1 ZMM for A broadcast (zmm6)                                      │
│    → 1 ZMM for B load (zmm7)                                           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.4 JIT Implementation: 6×16 FP32 GEMV/GEMM Microkernel

**File:** `jit/JitWoProjectionOptimized.h`

```cpp
/**
 * @brief Generate optimized 6×16 microkernel for Wo projection
 *
 * Computes a 6×16 tile of output:
 *   C[6,16] = A[6,K] × B[K,16]
 *
 * Register allocation:
 *   zmm0-5:  6 accumulators (one per output row)
 *   zmm6:    Broadcast of A[row, k]
 *   zmm7:    Load of B[k, 0:15]
 *   zmm8-13: Prefetched B rows (for ILP)
 *   zmm14:   Spare for unrolling
 *
 * K-loop is unrolled by 4 for ILP and prefetch scheduling.
 *
 * @param c         Xbyak code generator
 * @param reg_A     Pointer to A matrix (row-major, context buffer)
 * @param reg_B     Pointer to B matrix (row-major, Wo weights)
 * @param reg_C     Pointer to C matrix (row-major, output buffer)
 * @param K         Reduction dimension (d_model)
 * @param lda       Leading dimension of A (d_model)
 * @param ldb       Leading dimension of B (d_model)
 * @param ldc       Leading dimension of C (d_model)
 */
void generate_wo_microkernel_6x16(
    Xbyak::CodeGenerator& c,
    const Xbyak::Reg64& reg_A,
    const Xbyak::Reg64& reg_B,
    const Xbyak::Reg64& reg_C,
    int K, int lda, int ldb, int ldc)
{
    using namespace Xbyak;
    
    // ═══════════════════════════════════════════════════════════════════
    // ACCUMULATOR INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════
    
    Zmm acc0 = zmm0, acc1 = zmm1, acc2 = zmm2;
    Zmm acc3 = zmm3, acc4 = zmm4, acc5 = zmm5;
    Zmm a_bcast = zmm6;
    Zmm b_col = zmm7;
    
    // Zero all 6 accumulators
    c.vxorps(acc0, acc0, acc0);
    c.vxorps(acc1, acc1, acc1);
    c.vxorps(acc2, acc2, acc2);
    c.vxorps(acc3, acc3, acc3);
    c.vxorps(acc4, acc4, acc4);
    c.vxorps(acc5, acc5, acc5);
    
    // ═══════════════════════════════════════════════════════════════════
    // MAIN K-LOOP (unrolled by 4)
    // ═══════════════════════════════════════════════════════════════════
    
    Reg64 reg_k = rcx;
    Reg64 reg_A_ptr = r8;   // Current A column pointer
    Reg64 reg_B_ptr = r9;   // Current B row pointer
    
    c.mov(reg_A_ptr, reg_A);
    c.mov(reg_B_ptr, reg_B);
    c.xor_(reg_k, reg_k);
    
    Label k_loop, k_remainder, k_done;
    
    // Main loop: process 4 K values per iteration
    c.L(k_loop);
    c.cmp(reg_k, K - 3);
    c.jge(k_remainder, T_NEAR);
    
    // Prefetch next B rows (hide memory latency)
    c.prefetcht0(ptr[reg_B_ptr + 4 * ldb * 4 + 0]);   // B[k+4, 0:15]
    c.prefetcht0(ptr[reg_B_ptr + 4 * ldb * 4 + 64]);  // B[k+4, 16:31] if needed
    
    // ─────────────────────────────────────────────────────────────────
    // K iteration 0
    // ─────────────────────────────────────────────────────────────────
    c.vmovups(b_col, ptr[reg_B_ptr]);                      // B[k, 0:15]
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 0 * lda * 4]); // A[0, k]
    c.vfmadd231ps(acc0, a_bcast, b_col);                   // acc0 += A[0,k] * B[k,:]
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 1 * lda * 4]); // A[1, k]
    c.vfmadd231ps(acc1, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 2 * lda * 4]); // A[2, k]
    c.vfmadd231ps(acc2, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 3 * lda * 4]); // A[3, k]
    c.vfmadd231ps(acc3, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 4 * lda * 4]); // A[4, k]
    c.vfmadd231ps(acc4, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 5 * lda * 4]); // A[5, k]
    c.vfmadd231ps(acc5, a_bcast, b_col);
    
    // ─────────────────────────────────────────────────────────────────
    // K iteration 1
    // ─────────────────────────────────────────────────────────────────
    c.vmovups(b_col, ptr[reg_B_ptr + 1 * ldb * 4]);        // B[k+1, 0:15]
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 0 * lda * 4 + 4]);
    c.vfmadd231ps(acc0, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 1 * lda * 4 + 4]);
    c.vfmadd231ps(acc1, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 2 * lda * 4 + 4]);
    c.vfmadd231ps(acc2, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 3 * lda * 4 + 4]);
    c.vfmadd231ps(acc3, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 4 * lda * 4 + 4]);
    c.vfmadd231ps(acc4, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 5 * lda * 4 + 4]);
    c.vfmadd231ps(acc5, a_bcast, b_col);
    
    // ─────────────────────────────────────────────────────────────────
    // K iterations 2 and 3 (same pattern)
    // ─────────────────────────────────────────────────────────────────
    // ... (repeated for k+2 and k+3)
    
    // Advance pointers
    c.add(reg_A_ptr, 4 * 4);           // A_ptr += 4 (4 floats)
    c.add(reg_B_ptr, 4 * ldb * 4);     // B_ptr += 4 * ldb (4 rows)
    c.add(reg_k, 4);
    c.jmp(k_loop, T_NEAR);
    
    // ═══════════════════════════════════════════════════════════════════
    // K REMAINDER (K % 4 elements)
    // ═══════════════════════════════════════════════════════════════════
    
    c.L(k_remainder);
    c.cmp(reg_k, K);
    c.jge(k_done, T_NEAR);
    
    c.vmovups(b_col, ptr[reg_B_ptr]);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 0 * lda * 4]);
    c.vfmadd231ps(acc0, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 1 * lda * 4]);
    c.vfmadd231ps(acc1, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 2 * lda * 4]);
    c.vfmadd231ps(acc2, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 3 * lda * 4]);
    c.vfmadd231ps(acc3, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 4 * lda * 4]);
    c.vfmadd231ps(acc4, a_bcast, b_col);
    
    c.vbroadcastss(a_bcast, ptr[reg_A_ptr + 5 * lda * 4]);
    c.vfmadd231ps(acc5, a_bcast, b_col);
    
    c.add(reg_A_ptr, 4);
    c.add(reg_B_ptr, ldb * 4);
    c.inc(reg_k);
    c.jmp(k_remainder, T_NEAR);
    
    // ═══════════════════════════════════════════════════════════════════
    // STORE RESULTS
    // ═══════════════════════════════════════════════════════════════════
    
    c.L(k_done);
    c.vmovups(ptr[reg_C + 0 * ldc * 4], acc0);
    c.vmovups(ptr[reg_C + 1 * ldc * 4], acc1);
    c.vmovups(ptr[reg_C + 2 * ldc * 4], acc2);
    c.vmovups(ptr[reg_C + 3 * ldc * 4], acc3);
    c.vmovups(ptr[reg_C + 4 * ldc * 4], acc4);
    c.vmovups(ptr[reg_C + 5 * ldc * 4], acc5);
}
```

### 6.5 GEMV Specialization for Decode (M=1)

For decode (single query), we optimize differently since there's only one output row:

```cpp
/**
 * @brief Generate optimized GEMV for decode (M=1)
 *
 * For decode, we have: output[1, N] = context[1, K] × Wo[K, N]
 *
 * Strategy:
 *   - Vectorize over N (output columns)
 *   - Process 4×16 = 64 output columns per iteration
 *   - Use 4 accumulators for ILP
 *   - Unroll K-loop by 4 for prefetch scheduling
 *
 * Register allocation:
 *   zmm0-3:   4 accumulators for output[0:15], [16:31], [32:47], [48:63]
 *   zmm4:     Context broadcast
 *   zmm5-8:   B loads (4 columns of 16)
 *   zmm9-12:  Prefetched B
 */
void generate_wo_gemv_optimized(
    Xbyak::CodeGenerator& c,
    const Xbyak::Reg64& reg_context,  // [1, K] context vector
    const Xbyak::Reg64& reg_Wo,       // [K, N] Wo matrix  
    const Xbyak::Reg64& reg_output,   // [1, N] output vector
    int K, int N)
{
    using namespace Xbyak;
    
    Zmm acc0 = zmm0, acc1 = zmm1, acc2 = zmm2, acc3 = zmm3;
    Zmm ctx_bcast = zmm4;
    Zmm wo0 = zmm5, wo1 = zmm6, wo2 = zmm7, wo3 = zmm8;
    
    // ═══════════════════════════════════════════════════════════════════
    // OUTER LOOP: N dimension (output columns), step by 64
    // ═══════════════════════════════════════════════════════════════════
    
    Reg64 reg_n = r10;
    Reg64 reg_out_ptr = r11;
    Reg64 reg_Wo_col = r12;
    
    c.xor_(reg_n, reg_n);
    c.mov(reg_out_ptr, reg_output);
    c.mov(reg_Wo_col, reg_Wo);
    
    Label n_loop, n_loop_end;
    
    c.L(n_loop);
    c.cmp(reg_n, N - 63);
    c.jg(n_loop_end, T_NEAR);
    
    // Zero accumulators for this N-tile
    c.vxorps(acc0, acc0, acc0);
    c.vxorps(acc1, acc1, acc1);
    c.vxorps(acc2, acc2, acc2);
    c.vxorps(acc3, acc3, acc3);
    
    // ───────────────────────────────────────────────────────────────────
    // INNER LOOP: K dimension (reduction)
    // ───────────────────────────────────────────────────────────────────
    
    Reg64 reg_k = rcx;
    Reg64 reg_ctx_ptr = r8;
    Reg64 reg_Wo_ptr = r9;
    
    c.mov(reg_ctx_ptr, reg_context);
    c.mov(reg_Wo_ptr, reg_Wo_col);
    c.xor_(reg_k, reg_k);
    
    Label k_loop, k_done;
    
    c.L(k_loop);
    c.cmp(reg_k, K);
    c.jge(k_done, T_NEAR);
    
    // Prefetch next Wo row
    c.prefetcht0(ptr[reg_Wo_ptr + N * 4]);
    
    // Broadcast context[k]
    c.vbroadcastss(ctx_bcast, ptr[reg_ctx_ptr]);
    
    // Load 4 × 16 = 64 Wo elements from row k
    c.vmovups(wo0, ptr[reg_Wo_ptr + 0 * 64]);   // Wo[k, n:n+15]
    c.vmovups(wo1, ptr[reg_Wo_ptr + 1 * 64]);   // Wo[k, n+16:n+31]
    c.vmovups(wo2, ptr[reg_Wo_ptr + 2 * 64]);   // Wo[k, n+32:n+47]
    c.vmovups(wo3, ptr[reg_Wo_ptr + 3 * 64]);   // Wo[k, n+48:n+63]
    
    // FMA: acc += context[k] * Wo[k, :]
    c.vfmadd231ps(acc0, ctx_bcast, wo0);
    c.vfmadd231ps(acc1, ctx_bcast, wo1);
    c.vfmadd231ps(acc2, ctx_bcast, wo2);
    c.vfmadd231ps(acc3, ctx_bcast, wo3);
    
    c.add(reg_ctx_ptr, 4);      // Next context element
    c.add(reg_Wo_ptr, N * 4);   // Next Wo row
    c.inc(reg_k);
    c.jmp(k_loop, T_NEAR);
    
    c.L(k_done);
    
    // Store 64 output elements
    c.vmovups(ptr[reg_out_ptr + 0 * 64], acc0);
    c.vmovups(ptr[reg_out_ptr + 1 * 64], acc1);
    c.vmovups(ptr[reg_out_ptr + 2 * 64], acc2);
    c.vmovups(ptr[reg_out_ptr + 3 * 64], acc3);
    
    // Advance to next N-tile
    c.add(reg_out_ptr, 64 * 4);
    c.add(reg_Wo_col, 64 * 4);
    c.add(reg_n, 64);
    c.jmp(n_loop, T_NEAR);
    
    c.L(n_loop_end);
    
    // Handle N remainder (N % 64 elements) - similar pattern with masking
}
```

### 6.6 Cache Blocking Strategy

For large matrices, add L2/L3 cache blocking:

```cpp
/**
 * @brief Cache-blocked Wo projection
 *
 * Block sizes chosen for typical cache hierarchy:
 *   - L1: 32 KB data → NC_L1 = 64 (64×16×4 = 4 KB per N-block)
 *   - L2: 1 MB → KC = 256 (256 columns of Wo fit in L2)
 *   - L3: 30+ MB → Full Wo matrix (49 MB for Qwen 7B)
 *
 * For Wo projection where context is tiny (1-128 rows × K columns):
 *   - Context fits entirely in L1
 *   - Block over K (reduction dimension) to keep Wo in L2
 *   - Block over N (output columns) for register tiling
 */
void generate_wo_blocked(
    Xbyak::CodeGenerator& c,
    int M,      // Number of queries (1 for decode, seq_len for prefill)
    int N,      // Output dimension (d_model)
    int K,      // Reduction dimension (d_model)
    int MC = 6,      // M-block size (microkernel height)
    int NC = 64,     // N-block size (4 × 16 for decode)
    int KC = 256)    // K-block size (L2 cache)
{
    // For each K-block
    //   For each N-block
    //     For each M-block
    //       Call microkernel(MC, NC, KC_actual)
    
    // The microkernel handles partial tiles at boundaries
}
```

### 6.7 Integration with Fused Attention

Two integration strategies:

#### Strategy A: Separate Attention + Optimized Wo

```cpp
void compute_fused_attention_wo_optimized(
    const Q8_1Block* Q, const Q8_1Block* K, const Q8_1Block* V,
    const float* Wo, float* output,
    int seq_len_q, int seq_len_kv, float scale)
{
    // 1. Allocate context buffer
    alignas(64) float context[seq_len_q * d_model];
    
    // 2. Compute attention into context
    compute_attention_only_jit(Q, K, V, context, seq_len_q, seq_len_kv, scale);
    
    // 3. Apply optimized Wo projection
    if (seq_len_q == 1) {
        // Decode: use GEMV kernel
        wo_gemv_optimized(context, Wo, output, d_model, d_model);
    } else {
        // Prefill: use blocked GEMM kernel
        wo_gemm_blocked(context, Wo, output, seq_len_q, d_model, d_model);
    }
}
```

#### Strategy B: Fused Streaming Wo (Advanced)

Process Wo projection incrementally as attention context becomes available:

```cpp
// As each query's context vector is computed, immediately project through Wo
// This keeps context hot in L1 and enables pipelining
for (int q = 0; q < seq_len_q; ++q) {
    // Compute attention for query q → context[q, :]
    compute_single_query_attention(Q[q], K, V, &context[q * d_model], ...);
    
    // Immediately project context[q, :] through Wo while still in cache
    wo_gemv_optimized(&context[q * d_model], Wo, &output[q * d_model], K, N);
}
```

### 6.8 Expected Performance Improvement

| Configuration | Current | Target | Improvement |
|--------------|---------|--------|-------------|
| Qwen 7B decode (kv=128) | 3.18 ms | ~0.7 ms | **4.5×** |
| Qwen 7B prefill (seq=128) | 417 ms | ~65 ms | **6.4×** |
| Qwen 0.5B decode (kv=128) | 0.17 ms | ~0.06 ms | **2.8×** |

### 6.9 Testing Strategy

```cpp
// Test__WoProjectionOptimized.cpp

TEST(Test__WoProjectionOptimized, GEMV_MatchesReference) {
    const int K = 3584, N = 3584;  // Qwen 7B dimensions
    
    auto context = TestTensorFactory::createFP32Random({1, K});
    auto Wo = TestTensorFactory::createFP32Random({K, N});
    auto ref_output = std::vector<float>(N);
    auto opt_output = std::vector<float>(N);
    
    // Reference: simple GEMV
    for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += context[k] * Wo[k * N + n];
        }
        ref_output[n] = sum;
    }
    
    // Optimized: JIT GEMV
    wo_gemv_optimized(context.data(), Wo.data(), opt_output.data(), K, N);
    
    float max_diff = 0;
    for (int n = 0; n < N; ++n) {
        max_diff = std::max(max_diff, std::abs(ref_output[n] - opt_output[n]));
    }
    
    EXPECT_LT(max_diff, 1e-4f);  // FP32 accumulation tolerance
}

TEST(Perf__WoProjectionOptimized, GEMV_Throughput) {
    const int K = 3584, N = 3584;
    const int warmup = 10, iters = 100;
    
    auto context = TestTensorFactory::createFP32Random({1, K});
    auto Wo = TestTensorFactory::createFP32Random({K, N});
    auto output = std::vector<float>(N);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        wo_gemv_optimized(context.data(), Wo.data(), output.data(), K, N);
    }
    
    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        wo_gemv_optimized(context.data(), Wo.data(), output.data(), K, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double flops = 2.0 * K * N;
    double gflops = (flops / 1e9) / (ms / 1000.0);
    
    LOG_INFO("Optimized GEMV: " << ms << " ms, " << gflops << " GFLOP/s");
    
    // Target: at least 40 GFLOP/s (80% of OpenBLAS)
    EXPECT_GT(gflops, 40.0);
}
```

### 6.10 Implementation Priority

Given the massive performance impact of Wo projection:

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Option A: Unfuse and call OpenBLAS | 1 day | 5-6× immediate |
| P1 | JIT GEMV for decode (M=1) | 2 days | Maintain fusion |
| P2 | JIT blocked GEMM for prefill | 3 days | Full optimization |
| P3 | Streaming fused Wo | 4 days | Maximum throughput |

**Recommendation:** Start with P0 (OpenBLAS unfused) to get immediate gains, then implement P1/P2 to restore fusion benefits.

---

## Implementation Order

| Phase | Task | Dependencies | Tests to Pass | Estimated Effort |
|-------|------|--------------|---------------|------------------|
| 1.1 | Add OpenMP parallel for heads in `FusedAttentionWoTiled.cpp` | None | All existing (no regression) | 1 day |
| 1.2 | Make Wo projection thread-safe (atomic accumulation) | 1.1 | All existing (no regression) | 0.5 day |
| 1.3 | Benchmark Phase 1 performance | 1.2 | N/A (perf only) | 0.5 day |
| 2.1 | Implement `process_kv_tile_batched()` in C++ | None | `Test__FA2TileSoftmax` | 1 day |
| 2.2 | Integrate tile-wise softmax into `FusedAttentionWoTiled` | 2.1 | Category B equivalence tests | 1 day |
| 2.3 | Add edge case tests for tile softmax | 2.2 | `Test__FA2EdgeCases` | 0.5 day |
| 3.1 | JIT: `generate_q8_dot_product_4x()` | None | `Test__Q8DotProduct4x` | 1.5 days |
| 3.2 | JIT: `generate_tile_max_reduction()` | 3.1 | `Test__TileMaxReduction` | 0.5 day |
| 3.3 | JIT: `generate_fast_exp_16x()` | None | `Test__FastExp16x` | 1 day |
| 3.4 | JIT: `generate_interleaved_v_accum()` | 3.3 | `Test__InterleavedVAccum` | 1 day |
| 3.5 | JIT: Integrate all into `JitFusedAttentionWo` main loop | 3.1-3.4 | Category B + C tests | 2 days |
| 3.6 | JIT: Full correctness validation | 3.5 | `Test__FA2RealModel` E2E parity | 1 day |
| 4.1 | Performance benchmarks vs FA1 | 3.6 | `Perf__FA2Comparison` ≥1.2× | 0.5 day |
| 4.2 | Integration into pipeline | 4.1 | E2E token parity test | 0.5 day |
| **5.0.1** | **Add `[[nodiscard]]` to all borrow methods** | None | Compile-time check | **0.5 day** |
| 5.0.2 | Create `check_register_guards.sh` CI script | None | CI integration | 0.5 day |
| 5.0.3 | Add `GuardedReg<T>` template and `guarded::` namespace | 5.0.1 | Unit tests | 0.5 day |
| 5.0.4 | Migrate `JitMicrokernelBase.h` raw accessors | 5.0.3 | Existing tests pass | 1 day |
| 5.0.5 | Migrate remaining raw usage in JIT files | 5.0.4 | CI script passes | 0.5 day |
| 5.0.6 | Add `KernelRegisterManifest` for kernel composition | 5.0.5 | Manifest unit tests | 0.5 day |
| **6.0** | **Unfuse Wo + call OpenBLAS (quick win)** | None | Perf: 5-6× speedup | **1 day** |
| 6.1 | JIT: `generate_wo_gemv_optimized()` for decode | 6.0 | `Test__WoGEMV` parity | 2 days |
| 6.2 | JIT: `generate_wo_microkernel_6x16()` | 6.1 | `Test__WoMicrokernel` | 2 days |
| 6.3 | JIT: Cache-blocked GEMM wrapper | 6.2 | `Test__WoBlockedGEMM` | 1.5 days |
| 6.4 | Re-fuse optimized Wo into attention | 6.3 | Full E2E parity | 1.5 days |
| 6.5 | Streaming fused Wo (advanced) | 6.4 | Perf validation | 2 days |

**Total Estimated Effort:** ~25.5 days (original 12 + Phase 5: 3.5 days + Phase 6: 10 days)

---

## Risk Mitigation

### Numerical Precision

**Risk:** Tile-wise softmax may accumulate differently than per-position, causing output divergence.

**Mitigation:**
- Add E2E parity tests comparing FA2 vs FA1 outputs
- Use Kahan summation for sum_exp if needed
- Tolerance: accept 1e-5 max absolute error (well within inference noise)

### Register Pressure

**Risk:** AVX-512 has 32 ZMM registers; FA2 tile processing may exceed this.

**Mitigation:**
- Carefully plan register allocation:
  - zmm0-3: Context accumulators (4 regs)
  - zmm4-11: Score tile (8 regs)
  - zmm12-19: Weight tile (8 regs)
  - zmm20-27: V loading / temp (8 regs)
  - zmm28-31: Constants (4 regs)
- Spill constants to stack if needed (they're read-only)

### Thread Contention

**Risk:** Wo projection atomic accumulation may create contention.

**Mitigation:**
- Profile with `perf` to measure contention
- If >5% overhead, switch to per-head output buffers + final reduction

---

## Success Criteria

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Prefill throughput (seq=2048) | ~300 tok/s | ≥450 tok/s | `--benchmark` mode |
| Decode latency | ~18 ms/tok | ≤12 ms/tok | `--benchmark` mode |
| Attention kernel time % | 40% of total | ≤25% of total | `LLAMINAR_PROFILE_KERNELS=1` |
| Test parity | N/A | <1e-5 max diff | E2E parity tests |

---

## References

1. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691
2. Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022
3. Intel AVX-512 Programming Reference (2023)
4. Xbyak JIT Assembler Documentation: https://github.com/herumi/xbyak

---

## Appendix: Register Allocation Map

```
+--------+------------------+--------------------------------------------+
| ZMM    | Usage            | Notes                                      |
+--------+------------------+--------------------------------------------+
| zmm0   | context[0:15]    | Head dim floats 0-15                       |
| zmm1   | context[16:31]   | Head dim floats 16-31                      |
| zmm2   | context[32:47]   | Head dim floats 32-47                      |
| zmm3   | context[48:63]   | Head dim floats 48-63                      |
+--------+------------------+--------------------------------------------+
| zmm4   | scores[0:15]     | Tile scores (reused per tile)              |
| zmm5   | scores[16:31]    |                                            |
| zmm6   | scores[32:47]    |                                            |
| zmm7   | scores[48:63]    |                                            |
+--------+------------------+--------------------------------------------+
| zmm8   | weights[0:15]    | exp(score - max) per tile                  |
| zmm9   | weights[16:31]   |                                            |
| zmm10  | weights[32:47]   |                                            |
| zmm11  | weights[48:63]   |                                            |
+--------+------------------+--------------------------------------------+
| zmm12  | V[i] block 0     | Loaded V row, dequantized                  |
| zmm13  | V[i] block 1     |                                            |
| zmm14  | V[i+1] block 0   | Prefetched V row                           |
| zmm15  | V[i+1] block 1   |                                            |
+--------+------------------+--------------------------------------------+
| zmm16  | Q block 0        | Loaded once per head                       |
| zmm17  | Q block 1        |                                            |
+--------+------------------+--------------------------------------------+
| zmm18  | K[i] temp        | For dot product computation                |
| zmm19  | K[i+1] temp      |                                            |
+--------+------------------+--------------------------------------------+
| zmm20  | dot accum 0      | INT32 accumulator for Q·K                  |
| zmm21  | dot accum 1      |                                            |
| zmm22  | dot accum 2      |                                            |
| zmm23  | dot accum 3      |                                            |
+--------+------------------+--------------------------------------------+
| zmm24  | exp polynomial   | Intermediate for fast exp                  |
| zmm25  | exp polynomial   |                                            |
+--------+------------------+--------------------------------------------+
| zmm26  | running_max      | Broadcast softmax max                      |
| zmm27  | running_sum      | Scalar sum_exp (in xmm27)                  |
+--------+------------------+--------------------------------------------+
| zmm28  | const: log2e     | 1/ln(2)                                    |
| zmm29  | const: ln2       | ln(2) for range reduction                  |
| zmm30  | const: exp_c2-c4 | Polynomial coefficients                    |
| zmm31  | const: ones      | 1.0f broadcast                             |
+--------+------------------+--------------------------------------------+
```
