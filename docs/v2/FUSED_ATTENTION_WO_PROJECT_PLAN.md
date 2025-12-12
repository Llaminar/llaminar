# Fused Attention + Wo Projection: Project Plan

**Author:** David Sanftenberg  
**Date:** December 12, 2025  
**Status:** ✅ Phase 2 Complete — Robustness Testing Done  
**Branch:** `feature/fused-attention-wo`

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Microkernel Reference + SIMD + Tests | ✅ **Complete** (48 tests passing) |
| **Phase 2** | Composed Reference Kernel + FP16/BF16 Wo | ✅ **Complete** (48 integration/robustness + 10 FP16/BF16 = 58 tests) |
| **Phase 3** | Cache-Blocked Tiled Reference | ⬜ Not Started |
| **Phase 4** | JIT Kernel (Xbyak) | ⬜ Not Started — Strategy Documented |
| **Phase 5** | Pipeline Integration | ⬜ Not Started |

### Phase 2 Test Coverage

**Integration Tests** (`Test__FusedAttentionWoRef.cpp`):
1. ✅ ValidateParams_NullQ_ReturnsFalse
2. ✅ ValidateParams_ValidParams_ReturnsTrue
3. ✅ SinglePositionSingleHead_MatchesReference
4. ✅ MultiPositionCausal_MatchesReference
5. ✅ GQA_MultipleQueryHeadsPerKV
6. ✅ Q8_1_Wo_Weights
7. ✅ DecodeMode_PositionOffset
8. ✅ ExecuteSingleHead

**Robustness Tests** (`Test__FusedAttentionWoRef_Robustness.cpp`):

*Numerical Stability (16 tests):*
1. ✅ SoftmaxSaturation_LargeScores_NoOverflow
2. ✅ SoftmaxSaturation_VeryLargeScoreDifference (documents expected one-hot behavior)
3. ✅ CatastrophicCancellation_LargeOppositeValues
4. ✅ CatastrophicCancellation_AccumulatorPrecision (1024 uniform contributions)
5. ✅ QuantizationSaturation_ExtremeOutliers
6. ✅ QuantizationClipping_GammaWeightRange (13.0 weights, matches Qwen2.5 gamma[62])
7. ✅ ScaleDisparity_MixedMagnitudeBlocks
8. ✅ SignFlip_SmallValueRoundingError
9. ✅ ResidualCollapse_AccumulatedError (24-layer simulation)
10. ✅ ExtremeOutlier_SingleValueDominatesBlock (documents per-block scaling behavior)
11. ✅ EdgeCase_ZeroInputs
12. ✅ EdgeCase_IdenticalQK
13. ✅ EdgeCase_SinglePosition
14. ✅ EdgeCase_VeryLongSequence (4096 positions)
15. ✅ FastExp_ExtremeInputs
16. ✅ OnlineSoftmax_OrderIndependence (with correction factor)

*Stride Handling (6 tests):*
17. ✅ Stride_QKV_NonContiguous_HeadLayout (padded tensor layouts)
18. ✅ Stride_Output_NonContiguous (strided output buffer)
19. ✅ Stride_KV_Cache_Offset (decode mode position calculations)
20. ✅ Stride_GQA_KVHead_Mapping (GQA/MQA/MHA head ratios including Qwen2.5 14:2)
21. ✅ Stride_Wo_RowMajor_vs_ColMajor (weight layout verification)
22. ✅ Stride_Batch_Dimension (batch stride calculations)

*MPI Attention Slicing (6 tests):*
23. ✅ MPISlice_HeadPartitioning (tensor parallelism head distribution)
24. ✅ MPISlice_HeadExtraction_Interleaved (extract single head from multi-head layout)
25. ✅ MPISlice_KVBroadcast_GQA (K/V head replication for GQA)
26. ✅ MPISlice_OutputAccumulation (allreduce output aggregation)
27. ✅ MPISlice_WoWeight_HeadSlice (column-slice Wo for local heads)
28. ✅ MPISlice_ContextToOutput_Partial (partial context projection)

---

## 1. Problem Statement

### Current Q8_1 Attention Flow (4 quantization steps)
```
Q_fp32 → [Quant1] → Q_q8 ─┐
K_fp32 → [Quant2] → K_q8 ─┼→ Attention → Context_q8 → [Quant4] → Wo GEMM → Out
V_fp32 → [Quant3] → V_q8 ─┘                            ↑
                                                    [Extra quantization]
```

### Observed Divergence
- **Q/K/V Projections:** >0.999 cosine similarity (excellent)
- **ATTENTION_CONTEXT:** ~0.89 cosine similarity (diverging)
- **Root Cause:** Softmax amplifies small quantization differences, AND we immediately quantize the FP32 context to Q8_1 only to dequantize it again for Wo projection

### Key Insight
Looking at `QuantisedAttentionJit_Q8_1_Fused.h` lines 1049-1080:
1. Context is computed as **FP32** accumulator
2. Normalized by `1/sum_exp` (still FP32)
3. **Immediately quantized** to Q8_1
4. Wo projection then **dequantizes** back to FP32

This FP32 → Q8_1 → FP32 round-trip is wasteful and introduces unnecessary noise.

---

## 2. Proposed Solution

### Target Flow (3 quantization steps)
```
Q_fp32 → [Quant1] → Q_q8 ─┐
K_fp32 → [Quant2] → K_q8 ─┼→ Fused(Attention + Wo) → Out_fp32
V_fp32 → [Quant3] → V_q8 ─┘
```

### Benefits
1. **Eliminates one quantization step** (context → Q8_1)
2. **Better numerical accuracy** (context stays FP32 through projection)
3. **Improved cache locality** (context + Wo accessed together)
4. **Minimal changes** to existing tensor types

---

## 3. Architecture: Microkernel Design

### Design Philosophy
Rather than a monolithic fused kernel, we build **composable microkernels** that:
1. Are independently testable
2. Have clear, minimal interfaces
3. Can be composed in both C++ reference and JIT implementations
4. Enable A/B testing of individual components

### Microkernel Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FUSED ATTENTION + Wo KERNEL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐ │
│  │ Q8_1 Dot    │   │ Online      │   │ V Weighted  │   │ Wo Projection   │ │
│  │ Product     │──▶│ Softmax     │──▶│ Sum         │──▶│ (FP32 context)  │ │
│  │ (Q·K)       │   │ (streaming) │   │ (accumulate)│   │                 │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────────┘ │
│        μK1              μK2               μK3                μK4           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Microkernel Specifications

### μK1: Q8_1 Dot Product (`q8_dot_product`)

**Purpose:** Compute dot product between Q8_1 vectors with proper scaling.

**Signature:**
```cpp
namespace llaminar::v2::kernels::microkernels {

struct Q8DotProductParams {
    const Q8_1Block* q_blocks;  // [num_blocks] Q vector blocks
    const Q8_1Block* k_blocks;  // [num_blocks] K vector blocks  
    int num_blocks;             // head_dim / 32
    float global_scale;         // Optional pre-multiplied scale (e.g., 1/sqrt(d))
};

struct Q8DotProductResult {
    float score;                // Final scaled dot product
};

// Reference implementation (testable)
Q8DotProductResult q8_dot_product_ref(const Q8DotProductParams& params);

// SIMD implementation (AVX-512 VNNI)
Q8DotProductResult q8_dot_product_avx512(const Q8DotProductParams& params);

}
```

**Algorithm:**
```
score = 0
for each block b in [0, num_blocks):
    d_q = dequant(q_blocks[b].d)  // FP16 → FP32
    d_k = dequant(k_blocks[b].d)
    block_scale = d_q * d_k
    
    // Integer dot product (vpdpbusd path)
    int32 dot = 0
    for i in [0, 32):
        dot += (q_blocks[b].qs[i] + 128) * k_blocks[b].qs[i]  // unsigned × signed
    
    // Adjust for unsigned conversion bias
    dot -= 128 * k_blocks[b].sum_qs
    
    score += dot * block_scale
    
return score * global_scale
```

**Test Cases:**
- Zero vectors → 0.0
- Identity (same vector) → ||v||²
- Orthogonal vectors → 0.0
- Random vectors → matches FP32 reference (±tolerance)
- Edge cases: max/min values, single block, multiple blocks

---

### μK2: Online Softmax State (`online_softmax`)

**Purpose:** Streaming softmax that can process scores one at a time without storing all scores.

**Signature:**
```cpp
namespace llaminar::v2::kernels::microkernels {

struct OnlineSoftmaxState {
    float max_score;    // Running maximum
    float sum_exp;      // Running sum of exp(score - max)
    bool initialized;   // First score seen?
};

// Initialize state
OnlineSoftmaxState online_softmax_init();

// Update state with new score, returns weight for this score
float online_softmax_update(OnlineSoftmaxState& state, float score);

// Get correction factor for previously computed weights
// Call this when max_score changes to rescale old accumulations
float online_softmax_correction(float old_max, float new_max);

// Finalize: returns 1/sum_exp for final normalization
float online_softmax_finalize(const OnlineSoftmaxState& state);

}
```

**Algorithm:**
```cpp
float online_softmax_update(OnlineSoftmaxState& state, float score) {
    if (!state.initialized) {
        state.max_score = score;
        state.sum_exp = 1.0f;
        state.initialized = true;
        return 1.0f;  // exp(0) = 1
    }
    
    if (score > state.max_score) {
        // New maximum: need to rescale everything
        float correction = exp(state.max_score - score);
        state.sum_exp *= correction;
        state.max_score = score;
        state.sum_exp += 1.0f;
        return 1.0f;  // This score's weight before normalization
    } else {
        float weight = exp(score - state.max_score);
        state.sum_exp += weight;
        return weight;
    }
}
```

**Test Cases:**
- Single score → weight = 1.0, sum = 1.0
- Two equal scores → weights = 0.5, 0.5
- Increasing sequence → correct rescaling
- Decreasing sequence → no rescaling needed
- Large dynamic range → numerical stability
- Matches offline softmax reference

---

### μK3: V Weighted Accumulator (`v_weighted_accum`)

**Purpose:** Accumulate weighted V vectors into FP32 context buffer.

**Signature:**
```cpp
namespace llaminar::v2::kernels::microkernels {

struct VWeightedAccumParams {
    const Q8_1Block* v_blocks;  // [num_blocks] V vector for position n
    float weight;               // Softmax weight for this position
    float correction;           // Rescaling factor (1.0 if no max update)
    float* context;             // [head_dim] FP32 accumulator (in/out)
    int num_blocks;             // head_dim / 32
};

// Apply correction to existing context, then add weighted V
void v_weighted_accum_ref(const VWeightedAccumParams& params);

// SIMD version
void v_weighted_accum_avx512(const VWeightedAccumParams& params);

}
```

**Algorithm:**
```cpp
void v_weighted_accum_ref(const VWeightedAccumParams& params) {
    // Apply correction to existing accumulation (if max changed)
    if (params.correction != 1.0f) {
        for (int d = 0; d < params.num_blocks * 32; ++d) {
            params.context[d] *= params.correction;
        }
    }
    
    // Add weighted V
    for (int b = 0; b < params.num_blocks; ++b) {
        float d_v = fp16_to_fp32(params.v_blocks[b].d);
        for (int i = 0; i < 32; ++i) {
            float v_val = params.v_blocks[b].qs[i] * d_v;
            params.context[b * 32 + i] += params.weight * v_val;
        }
    }
}
```

**Test Cases:**
- Zero weight → context unchanged (except correction)
- Unit weight, zero context → context = V
- Correction factor application
- Multiple accumulations sum correctly
- SIMD matches reference

---

### μK4: Wo Projection (`wo_projection`)

**Purpose:** Project FP32 context through Wo weight matrix.

**Signature:**
```cpp
namespace llaminar::v2::kernels::microkernels {

struct WoProjectionParams {
    const float* context;           // [head_dim] FP32 normalized context
    const void* wo_weights;         // Weight data (Q8_1 or FP32)
    TensorType wo_type;             // GGUF type of Wo weights
    int head_dim;                   // Input dimension (64 or 128)
    int d_model;                    // Output dimension (896 for Qwen2.5-0.5B)
    int head_idx;                   // Which head (for striding into Wo)
    int n_heads;                    // Total heads (for Wo layout)
    float* output;                  // [d_model] FP32 output (accumulated)
    bool accumulate;                // If true, add to output; if false, overwrite
};

// Reference implementation
void wo_projection_ref(const WoProjectionParams& params);

// Optimized (handles Q8_1 Wo with on-the-fly dequant)
void wo_projection_q8_wo(const WoProjectionParams& params);

// Optimized (FP32 Wo)
void wo_projection_fp32_wo(const WoProjectionParams& params);

}
```

**Wo Weight Layout:**
```
Wo shape: [d_model, n_heads * head_dim]
         = [896, 14 * 64] = [896, 896] for Qwen2.5-0.5B

For head h, the relevant slice is:
  Wo[:, h*head_dim : (h+1)*head_dim]
  = Wo[:, h*64 : h*64+64]

For each output dimension o in [0, d_model):
  output[o] += sum_{d=0}^{head_dim-1} context[d] * Wo[o, head_idx*head_dim + d]
```

**Test Cases:**
- Identity-like Wo → output ≈ context (with appropriate tiling)
- Zero context → zero output
- Single head projection → matches naive GEMM
- Accumulation mode (multiple heads)
- Q8_1 Wo vs FP32 Wo accuracy comparison

---

### μK5: Fast Exp Approximation (`fast_exp`)

**Purpose:** Fast exponential approximation for softmax (extracted from existing JIT).

**Signature:**
```cpp
namespace llaminar::v2::kernels::microkernels {

// Scalar reference
float fast_exp_ref(float x);

// AVX-512 vectorized (16 floats)
__m512 fast_exp_avx512(__m512 x);

// Polynomial coefficients (5th order Taylor)
// exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120

}
```

**Test Cases:**
- exp(0) = 1.0
- exp(1) ≈ 2.718...
- Negative values (softmax range: typically [-10, 0])
- Accuracy vs std::exp across softmax-relevant range
- SIMD matches scalar

---

## 5. Composition: Reference Implementation

### FusedAttentionWo Class

```cpp
namespace llaminar::v2::kernels {

class FusedAttentionWoRef {
public:
    struct Params {
        // Input tensors (Q8_1)
        const Q8_1Block* Q;      // [seq_len, num_heads, head_dim/32 blocks]
        const Q8_1Block* K;      // [seq_len, num_kv_heads, head_dim/32 blocks]  
        const Q8_1Block* V;      // [seq_len, num_kv_heads, head_dim/32 blocks]
        
        // Wo weight
        const void* Wo;          // [d_model, n_heads * head_dim]
        TensorType wo_type;
        
        // Output (FP32)
        float* output;           // [seq_len, d_model]
        
        // Dimensions
        int seq_len;
        int num_heads;
        int num_kv_heads;
        int head_dim;
        int d_model;
        
        // Attention config
        float scale;             // 1/sqrt(head_dim)
        const float* mask;       // Optional causal mask [seq_len, seq_len]
        int mask_stride;
    };
    
    // Execute fused attention + Wo using microkernels
    static bool execute(const Params& params);
    
private:
    // Internal: process one query head
    static void process_head(
        const Params& params,
        int query_pos,           // m in [0, seq_len)
        int head_idx,            // h in [0, num_heads)
        float* context_buffer    // [head_dim] scratch space
    );
};

}
```

### Execution Flow

```cpp
bool FusedAttentionWoRef::execute(const Params& params) {
    const int num_blocks = params.head_dim / 32;
    const int kv_head_ratio = params.num_heads / params.num_kv_heads;  // GQA
    
    // Zero output (heads will accumulate into it)
    std::memset(params.output, 0, params.seq_len * params.d_model * sizeof(float));
    
    // Allocate per-thread context buffers
    std::vector<float> context_buffer(params.head_dim);
    
    // For each query position
    for (int m = 0; m < params.seq_len; ++m) {
        // For each query head
        for (int h = 0; h < params.num_heads; ++h) {
            process_head(params, m, h, context_buffer.data());
        }
    }
    
    return true;
}

void FusedAttentionWoRef::process_head(
    const Params& params, int m, int h, float* context
) {
    using namespace microkernels;
    
    const int num_blocks = params.head_dim / 32;
    const int kv_head = h / (params.num_heads / params.num_kv_heads);  // GQA mapping
    
    // Get Q row for this position and head
    const Q8_1Block* Q_row = params.Q + 
        (m * params.num_heads + h) * num_blocks;
    
    // Initialize online softmax
    OnlineSoftmaxState softmax_state = online_softmax_init();
    
    // Zero context accumulator
    std::memset(context, 0, params.head_dim * sizeof(float));
    
    // Iterate over all K/V positions (up to m for causal)
    const int max_n = m + 1;  // Causal: can only attend to past + self
    
    for (int n = 0; n < max_n; ++n) {
        // Get K row for this position and KV head
        const Q8_1Block* K_row = params.K + 
            (n * params.num_kv_heads + kv_head) * num_blocks;
        
        // μK1: Compute Q·K score
        Q8DotProductParams dot_params = {
            .q_blocks = Q_row,
            .k_blocks = K_row,
            .num_blocks = num_blocks,
            .global_scale = params.scale
        };
        float score = q8_dot_product_ref(dot_params).score;
        
        // Apply mask if provided
        if (params.mask) {
            score += params.mask[m * params.mask_stride + n];
        }
        
        // μK2: Online softmax update
        float old_max = softmax_state.max_score;
        float weight = online_softmax_update(softmax_state, score);
        float correction = (softmax_state.max_score != old_max) 
            ? online_softmax_correction(old_max, softmax_state.max_score)
            : 1.0f;
        
        // μK3: Accumulate weighted V
        const Q8_1Block* V_row = params.V + 
            (n * params.num_kv_heads + kv_head) * num_blocks;
        
        VWeightedAccumParams accum_params = {
            .v_blocks = V_row,
            .weight = weight,
            .correction = correction,
            .context = context,
            .num_blocks = num_blocks
        };
        v_weighted_accum_ref(accum_params);
    }
    
    // Normalize context by sum_exp
    float inv_sum = online_softmax_finalize(softmax_state);
    for (int d = 0; d < params.head_dim; ++d) {
        context[d] *= inv_sum;
    }
    
    // μK4: Project through Wo (accumulates into output)
    WoProjectionParams wo_params = {
        .context = context,
        .wo_weights = params.Wo,
        .wo_type = params.wo_type,
        .head_dim = params.head_dim,
        .d_model = params.d_model,
        .head_idx = h,
        .n_heads = params.num_heads,
        .output = params.output + m * params.d_model,
        .accumulate = true  // Multiple heads contribute
    };
    wo_projection_ref(wo_params);
}
```

---

## 6. JIT Implementation Strategy

### Phase 1: JIT Microkernels

Each microkernel gets a JIT version using Xbyak:

```cpp
namespace llaminar::v2::kernels::jit {

class Q8DotProductJit : public Xbyak::CodeGenerator {
    // Emits AVX-512 VNNI code for Q8_1 dot product
    // Uses vpdpbusd for unsigned × signed accumulation
};

class OnlineSoftmaxJit : public Xbyak::CodeGenerator {
    // Emits state machine for streaming softmax
    // Uses fast_exp polynomial approximation
};

class VWeightedAccumJit : public Xbyak::CodeGenerator {
    // Emits vectorized weighted accumulation
    // Handles correction factor multiplication
};

class WoProjectionJit : public Xbyak::CodeGenerator {
    // Emits context × Wo GEMV
    // Handles Q8_1 or FP32 Wo weights
};

}
```

### Phase 2: Composed JIT Kernel

```cpp
class FusedAttentionWoJit : public Xbyak::CodeGenerator {
public:
    FusedAttentionWoJit(int head_dim, int d_model, TensorType wo_type);
    
    // Generated function signature
    using KernelFn = void (*)(const FusedAttentionWoParams* params);
    
    KernelFn getKernel() const { return getCode<KernelFn>(); }
    
private:
    // Emit inlined microkernel calls
    void emit_q8_dot_product(/* registers */);
    void emit_online_softmax_update(/* registers */);
    void emit_v_weighted_accum(/* registers */);
    void emit_wo_projection(/* registers */);
    
    // Register allocation
    // ZMM0-3:  Context accumulators (64 floats = 4 ZMM)
    // ZMM4-5:  Q block data
    // ZMM6-7:  K block data
    // ZMM8-9:  V block data
    // ZMM10:   Softmax state (max, sum_exp)
    // ZMM11-15: Wo projection scratch
    // ZMM16-31: Available for tiling
};
```

---

## 7. Testing Strategy

### Unit Tests (Per Microkernel)

| Test File | Microkernel | Coverage |
|-----------|-------------|----------|
| `Test__Q8DotProduct.cpp` | μK1 | Zero, identity, orthogonal, random, edge cases |
| `Test__OnlineSoftmax.cpp` | μK2 | Single, equal, increasing, decreasing, stability |
| `Test__VWeightedAccum.cpp` | μK3 | Zero weight, unit weight, correction, accumulation |
| `Test__WoProjection.cpp` | μK4 | Identity, zero, single head, multi-head, Q8/FP32 Wo |
| `Test__FastExp.cpp` | μK5 | Accuracy vs std::exp, SIMD consistency |

### Integration Tests

| Test File | Scope | Validation |
|-----------|-------|------------|
| `Test__FusedAttentionWoRef.cpp` | Full reference | Matches separate attention + Wo GEMM |
| `Test__FusedAttentionWoJit.cpp` | JIT vs reference | Bit-exact or ±epsilon |
| `Test__FusedAttentionWoParity.cpp` | vs PyTorch | Cosine similarity > 0.99 |

### Performance Tests

| Test File | Metric |
|-----------|--------|
| `Perf__FusedAttentionWo.cpp` | Throughput (GFLOPS), latency, vs unfused baseline |

---

## 8. Implementation Phases

### Phase 1: Microkernel Reference (Week 1) ✅ COMPLETE
- [x] Implement `q8_dot_product_ref` + `q8_dot_product_avx512_vnni`
- [x] Implement `online_softmax_*` functions
- [x] Implement `v_weighted_accum_ref` + `v_weighted_accum_avx512`
- [x] Implement `wo_projection_ref` (FP32 + Q8_1 paths)
- [x] Implement `fast_exp_ref` + `fast_exp_poly` + `fast_exp_avx512`
- [x] Unit tests for all microkernels (48 tests, all passing)

**Delivered Files:**
- `src/v2/kernels/cpu/microkernels/q8_1/Q8DotProduct.h/.cpp`
- `src/v2/kernels/cpu/microkernels/q8_1/OnlineSoftmax.h/.cpp`
- `src/v2/kernels/cpu/microkernels/q8_1/VWeightedAccum.h/.cpp`
- `src/v2/kernels/cpu/microkernels/q8_1/WoProjection.h/.cpp`
- `src/v2/kernels/cpu/microkernels/q8_1/FastExp.h/.cpp`
- `tests/v2/unit/microkernels/q8_1/Test__Q8DotProduct.cpp` (10 tests)
- `tests/v2/unit/microkernels/q8_1/Test__OnlineSoftmax.cpp` (9 tests)
- `tests/v2/unit/microkernels/q8_1/Test__VWeightedAccum.cpp` (8 tests)
- `tests/v2/unit/microkernels/q8_1/Test__WoProjection.cpp` (8 tests)
- `tests/v2/unit/microkernels/q8_1/Test__FastExp.cpp` (13 tests)

### Phase 2: Composed Reference (Week 1) ✅ COMPLETE
- [x] Implement `FusedAttentionWoRef::execute`
- [x] Integration test: matches separate attention + Wo GEMM (8 tests passing)
- [ ] PyTorch parity test

**Delivered Files:**
- `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoRef.h` - Header with FusedAttentionWoParams
- `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoRef.cpp` - Reference implementation
- `tests/v2/integration/q8_1/Test__FusedAttentionWoRef.cpp` - 8 integration tests

**Test Coverage:**
- Validation (null params, valid params)
- Single position/head
- Multi-position causal attention
- GQA (grouped query attention)
- Q8_1 Wo weights
- Decode mode (KV cache with position offset)
- Single head execution interface

### Phase 3: Cache-Blocked Tiled Attention (Week 2)

**Status:** ⬜ Not Started

**Goal:** Transform the reference implementation from O(N) cache misses per query to O(N/KV_TILE) by loading K/V tiles into L2 cache and reusing them across multiple query positions.

#### Current Implementation Analysis

The reference implementation (`FusedAttentionWoRef.cpp`) has these characteristics:

✅ **What's Correct:**
- Online softmax with correction factor (FlashAttention-style streaming)
- Streaming V accumulation (no explicit score matrix storage)
- Fused Wo projection (eliminates quantization round-trip)

❌ **What's Missing (Performance Bottleneck):**
```cpp
// Current: Each K/V row loaded from RAM per query position
for (int m = 0; m < seq_len; ++m) {           // Query positions
    for (int n = 0; n < max_kv_pos; ++n) {    // K/V positions - CACHE MISS!
        K_row = K + n * ...;                   // Cold load every time
        score = q8_dot_product(...);
    }
}
```

This pattern causes:
1. **No K/V reuse:** Each K/V row loaded `seq_len` times
2. **Cache thrashing:** K/V data evicted before reuse
3. **Poor ILP:** Sequential dependency prevents prefetching

#### Tiled Attention Algorithm

**FlashAttention-style outer loop with dynamic tile sizing:**

```cpp
// Tile sizes computed at runtime from CPUFeatures.h
const uint32_t L2_SIZE = cpu_l2_cache_size();   // e.g., 1MB per core
const uint32_t L3_SIZE = cpu_l3_cache_size();   // e.g., 38MB shared

// KV_TILE: How many K/V positions fit in L2 with room for Q and context
// Each K row: head_dim * sizeof(Q8_1Block)/32 = 64 * 36/32 = 72 bytes
// Each V row: same = 72 bytes
// Working set per KV position: ~144 bytes
// Leave 50% of L2 for Q, context, softmax state
const int KV_TILE = std::min(512, static_cast<int>(L2_SIZE * 0.5f / 144));

// Q_TILE: How many query positions to process together
// Enables K/V tile reuse across multiple queries
const int Q_TILE = std::min(64, static_cast<int>(L2_SIZE * 0.25f / (head_dim * 4)));

for (int q_start = 0; q_start < seq_len; q_start += Q_TILE) {
    int q_end = std::min(q_start + Q_TILE, seq_len);
    
    // Per-query softmax state arrays
    OnlineSoftmaxState softmax_states[Q_TILE];
    float context_buffers[Q_TILE][head_dim];  // Or use L3 for larger tiles
    
    for (int kv_start = 0; kv_start < max_kv_pos; kv_start += KV_TILE) {
        int kv_end = std::min(kv_start + KV_TILE, max_kv_pos);
        
        // PREFETCH: Load K/V tile into L2 (stays resident for Q_TILE queries)
        prefetch_kv_tile(K, V, kv_start, kv_end, kv_head);
        
        // Process all queries against this K/V tile
        for (int m = q_start; m < q_end; ++m) {
            // Causal: only attend to positions <= m
            int effective_kv_end = std::min(kv_end, m + 1);
            if (kv_start > m) continue;  // Skip future K/V tiles
            
            for (int n = kv_start; n < effective_kv_end; ++n) {
                // K/V rows now in L2 cache - FAST!
                const Q8_1Block* K_row = K + n * ...;  // L2 hit
                float score = q8_dot_product(Q_row, K_row, ...);
                
                // Online softmax update
                float weight = online_softmax_update(softmax_states[m - q_start], score);
                
                // Accumulate weighted V
                const Q8_1Block* V_row = V + n * ...;  // L2 hit
                v_weighted_accum(..., context_buffers[m - q_start]);
            }
        }
    }
    
    // Finalize softmax and project through Wo for this Q tile
    for (int m = q_start; m < q_end; ++m) {
        normalize_context(context_buffers[m - q_start], softmax_states[m - q_start]);
        wo_projection(context_buffers[m - q_start], output + m * d_model, ...);
    }
}
```

#### Dynamic Cache Size Detection

**Leverage existing CPUFeatures.h infrastructure:**

```cpp
#include "v2/utils/CPUFeatures.h"

namespace llaminar::v2::kernels {

struct TileConfig {
    int kv_tile;        // K/V positions per tile (fits in L2)
    int q_tile;         // Query positions per tile (for K/V reuse)
    int wo_tile;        // Wo output dimensions per tile (optional L3 blocking)
    uint32_t l2_size;   // Detected L2 cache size
    uint32_t l3_size;   // Detected L3 cache size
};

// Compute optimal tile sizes based on detected cache hierarchy
inline TileConfig compute_tile_config(int head_dim, int d_model) {
    using namespace llaminar2;
    
    TileConfig config;
    config.l2_size = cpu_l2_cache_size();   // Cached static - zero overhead
    config.l3_size = cpu_l3_cache_size();   // Cached static - zero overhead
    
    // Q8_1Block layout: head_dim / 32 blocks per row
    // Each block: 32 int8 + 2 bytes (fp16 scale) + 2 bytes (int16 sum) = 36 bytes
    const int bytes_per_kv_row = (head_dim / 32) * 36 * 2;  // K + V
    
    // Target: Use 50% of L2 for K/V tile (leave room for Q, context, etc.)
    const int l2_for_kv = config.l2_size / 2;
    config.kv_tile = std::max(32, std::min(512, l2_for_kv / bytes_per_kv_row));
    
    // Q_TILE: Each query needs head_dim * 4 bytes for context accumulator
    const int l2_for_context = config.l2_size / 4;
    config.q_tile = std::max(8, std::min(64, l2_for_context / (head_dim * 4)));
    
    // Wo tiling (optional, for very large d_model)
    // Wo slice for one head: d_model * head_dim * weight_bytes
    config.wo_tile = d_model;  // Default: no Wo tiling needed for Qwen2.5
    
    return config;
}

}
```

#### Expected Cache Behavior

| Cache Level | Contents | Size Budget |
|-------------|----------|-------------|
| **L1 (32KB)** | Current Q row, softmax state, registers | ~1KB |
| **L2 (1MB)** | K/V tile (KV_TILE rows), context accumulators | ~500KB |
| **L3 (38MB)** | Wo weight matrix slice, next K/V tiles | ~10MB |

#### Performance Model

**Memory bandwidth analysis:**

Current (no tiling):
- K/V loads: `seq_len × seq_len × 144 bytes` = O(N²) bandwidth
- For seq_len=512: ~38MB of K/V traffic per head

Tiled (KV_TILE=256, Q_TILE=32):
- K/V loads: `(seq_len / KV_TILE) × seq_len × 144 bytes` = O(N²/KV_TILE)
- For seq_len=512: ~150KB of K/V traffic per head (256× reduction!)

#### Implementation Tasks

- [ ] Add `TileConfig` struct and `compute_tile_config()` to attention header
- [ ] Implement `FusedAttentionWoTiled` class with outer tile loops
- [ ] Add K/V prefetch intrinsics for tile loading
- [ ] Update context buffer allocation for Q_TILE batch
- [ ] Handle causal masking at tile boundaries
- [ ] Unit test: Tiled matches reference for various tile sizes
- [ ] Performance test: Measure bandwidth reduction

#### Test Cases

| Test | Description | Validation |
|------|-------------|------------|
| `TileConfig_L2Detection` | Verify cache size detection | `kv_tile > 0`, `q_tile > 0` |
| `TiledVsReference_SmallSeq` | seq_len=64 (< KV_TILE) | Matches reference within ε |
| `TiledVsReference_MediumSeq` | seq_len=256 (= KV_TILE) | Matches reference within ε |
| `TiledVsReference_LargeSeq` | seq_len=512 (> KV_TILE) | Matches reference within ε |
| `TiledVsReference_CausalBoundary` | Verify causal at tile edges | Correct masking |
| `TiledPerformance_BandwidthReduction` | Measure actual bandwidth | ≥2× improvement |

### Phase 3b: SIMD Microkernels (Week 2)
- [x] Implement `q8_dot_product_avx512` (completed in Phase 1)
- [x] Implement `v_weighted_accum_avx512` (completed in Phase 1)
- [ ] Implement `wo_projection_avx512` (FP32 Wo first)
- [x] Implement `fast_exp_avx512` (completed in Phase 1)
- [x] Unit tests: SIMD matches reference (completed in Phase 1)

### Phase 4: JIT Kernel (Week 2-3)
- [ ] Implement `FusedAttentionWoJit` with tiled outer loops
- [ ] Inline microkernels into JIT code generation
- [ ] Integration test: JIT matches tiled reference
- [ ] Performance benchmark vs unfused

**JIT Considerations:**
- Emit tile loop structure with computed tile sizes
- Use register blocking for Q across K positions within tile
- Prefetch next K/V tile while processing current
- ZMM0-3: Context accumulators (64 floats)
- ZMM4-7: Q/K block data
- ZMM8-9: V block data
- ZMM10: Softmax state
- ZMM16-31: K/V tile prefetch buffer

---

## Phase 4 Deep Dive: JIT Strategy for Variable Model Sizes

### Qwen2 Model Dimension Matrix

| Model | d_model | n_heads | n_kv_heads | head_dim | d_ff | GQA Ratio | Wo Shape |
|-------|---------|---------|------------|----------|------|-----------|----------|
| **0.5B** | 896 | 14 | 2 | 64 | 4864 | 7:1 | [896, 896] |
| **1.5B** | 1536 | 12 | 2 | 128 | 8960 | 6:1 | [1536, 1536] |
| **3B** | 2048 | 16 | 2 | 128 | 11008 | 8:1 | [2048, 2048] |
| **7B** | 3584 | 28 | 4 | 128 | 18944 | 7:1 | [3584, 3584] |
| **14B** | 5120 | 40 | 8 | 128 | 13824 | 5:1 | [5120, 5120] |
| **32B** | 5120 | 40 | 8 | 128 | 27648 | 5:1 | [5120, 5120] |
| **72B** | 8192 | 64 | 8 | 128 | 29568 | 8:1 | [8192, 8192] |

### JIT Flexibility Requirements

Unlike static SIMD code, Xbyak JIT allows us to:

1. **Specialize per head_dim**: Different register blocking for 64 vs 128 head_dim
2. **Specialize per GQA ratio**: Unroll KV head replication differently
3. **Specialize per d_model**: Tile Wo projection based on output size
4. **Specialize per cache size**: Dynamic tile sizes from CPUFeatures.h
5. **Specialize per sequence length**: Prefill (large M) vs decode (M=1) kernels

### JIT Kernel Dispatch Architecture

```cpp
namespace llaminar::v2::kernels::jit {

/**
 * @brief Configuration key for JIT kernel specialization
 * 
 * Each unique combination gets a specialized kernel generated once,
 * then cached for reuse. This is Xbyak's key advantage over static SIMD.
 */
struct FusedAttentionWoConfig {
    int head_dim;        // 64 or 128 (affects register blocking)
    int d_model;         // Output dimension (affects Wo tiling)
    int n_heads;         // Query heads per kernel invocation
    int gqa_ratio;       // n_heads / n_kv_heads (affects KV replication)
    WoWeightType wo_type;// FP32, BF16, FP16, Q8_1, Q4_0
    bool is_decode;      // M=1 optimization (different code path)
    int kv_tile_size;    // From CPUFeatures L2 cache detection
    int wo_tile_size;    // From CPUFeatures L3 cache detection
    
    bool operator==(const FusedAttentionWoConfig& o) const {
        return head_dim == o.head_dim && d_model == o.d_model &&
               n_heads == o.n_heads && gqa_ratio == o.gqa_ratio &&
               wo_type == o.wo_type && is_decode == o.is_decode &&
               kv_tile_size == o.kv_tile_size && wo_tile_size == o.wo_tile_size;
    }
};

struct ConfigHash {
    size_t operator()(const FusedAttentionWoConfig& c) const {
        return std::hash<int>()(c.head_dim) ^ (std::hash<int>()(c.d_model) << 4) ^
               (std::hash<int>()(c.gqa_ratio) << 8) ^ (std::hash<int>()(c.is_decode) << 12);
    }
};

/**
 * @brief JIT kernel cache - generate once, use many
 */
class FusedAttentionWoJitFactory {
public:
    using KernelFn = void (*)(const FusedAttentionWoParams*);
    
    static KernelFn getKernel(const FusedAttentionWoConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(config);
        if (it != cache_.end()) {
            return it->second->getKernel();
        }
        
        // Generate specialized kernel for this configuration
        auto jit = std::make_unique<FusedAttentionWoJit>(config);
        KernelFn fn = jit->getKernel();
        cache_[config] = std::move(jit);
        
        LOG_INFO("JIT: Generated FusedAttentionWo kernel for head_dim=" << config.head_dim
                 << " d_model=" << config.d_model << " gqa=" << config.gqa_ratio
                 << " decode=" << config.is_decode);
        
        return fn;
    }
    
private:
    static std::unordered_map<FusedAttentionWoConfig, 
                              std::unique_ptr<FusedAttentionWoJit>, 
                              ConfigHash> cache_;
    static std::mutex mutex_;
};

}
```

### Specialization Strategy per Model Size

#### 1. head_dim Specialization (64 vs 128)

**head_dim=64 (Qwen2 0.5B):**
```
Context: 64 floats = 4 ZMM registers (ZMM0-3)
Q/K blocks: 2 blocks × 36 bytes = 72 bytes
V blocks: 2 blocks × 36 bytes = 72 bytes

Register allocation:
  ZMM0-3:   Context accumulators (64 FP32)
  ZMM4-5:   Q blocks (2 × 32 int8)
  ZMM6-7:   K blocks (2 × 32 int8)  
  ZMM8-9:   V blocks (dequantized to FP32)
  ZMM10-11: Softmax state (max, sum)
  ZMM12-15: Wo column accumulators (4 × 16 FP32)
  ZMM16-31: Available for loop unrolling / prefetch
```

**head_dim=128 (Qwen2 1.5B+):**
```
Context: 128 floats = 8 ZMM registers (ZMM0-7)
Q/K blocks: 4 blocks × 36 bytes = 144 bytes
V blocks: 4 blocks × 36 bytes = 144 bytes

Register allocation:
  ZMM0-7:   Context accumulators (128 FP32)
  ZMM8-11:  Q blocks (4 × 32 int8)
  ZMM12-15: K blocks (4 × 32 int8)
  ZMM16-19: V blocks (dequantized to FP32)
  ZMM20-21: Softmax state
  ZMM22-25: Wo column accumulators
  ZMM26-31: Loop scratch / prefetch
```

#### 2. GQA Ratio Specialization

**Ratio 7:1 (Qwen2 0.5B, 7B):**
```cpp
// JIT emits unrolled KV head replication
// 7 query heads share 1 KV head
void emit_gqa_7_to_1() {
    // Load KV head once
    load_kv_head(zmm_k, zmm_v, kv_head_idx);
    
    // Process 7 query heads against same K/V
    for (int q = 0; q < 7; ++q) {
        load_q_head(zmm_q, q_head_base + q);
        emit_attention_head(zmm_q, zmm_k, zmm_v, zmm_context[q]);
    }
}
```

**Ratio 5:1 (Qwen2 14B, 32B):**
```cpp
void emit_gqa_5_to_1() {
    // Different unroll factor
    load_kv_head(zmm_k, zmm_v, kv_head_idx);
    for (int q = 0; q < 5; ++q) {
        // ... same pattern, different unroll
    }
}
```

#### 3. Decode Mode Optimization (M=1)

**Decode kernel (autoregressive):**
- Single query position attending to all past K/V
- No need for Q tiling (just 1 query)
- Maximize K/V tile size for L2 reuse
- Inline softmax normalization (no separate pass)

```cpp
void generate_decode_kernel(const FusedAttentionWoConfig& cfg) {
    // Simplified: no M loop
    // Load Q once, stream K/V
    
    emit_load_q_row();  // Load single Q row to ZMM4-5 (or ZMM8-11 for 128)
    
    // Initialize online softmax
    emit_init_softmax();  // max = -inf, sum = 0
    
    // K/V streaming loop (all past positions)
    L("kv_loop");
        emit_load_k_row();           // K[n] → ZMM6-7
        emit_q8_dot_product();       // score = Q·K
        emit_softmax_update();       // update max, sum, weight
        emit_load_v_row();           // V[n] → ZMM8-9  
        emit_v_weighted_accum();     // context += weight * V
        inc(reg_n);
        cmp(reg_n, reg_kv_len);
        jl("kv_loop");
    
    // Finalize and project
    emit_softmax_finalize();  // context /= sum
    emit_wo_projection();     // output = context × Wo
}
```

**Prefill kernel (context processing):**
- Multiple query positions (M > 1)
- Q tiling for K/V cache reuse
- Causal mask handling at tile boundaries

```cpp
void generate_prefill_kernel(const FusedAttentionWoConfig& cfg) {
    // Tiled outer loop
    L("q_tile_loop");
        // Load Q tile (Q_TILE query rows)
        emit_load_q_tile();
        
        L("kv_tile_loop");
            // Prefetch next K/V tile
            emit_prefetch_kv_tile();
            
            // Process current K/V tile against all Q in tile
            L("q_inner");
                L("kv_inner");
                    emit_attention_step();  // Q·K, softmax, V accum
                jmp_if_more_kv("kv_inner");
            jmp_if_more_q("q_inner");
        jmp_if_more_kv_tiles("kv_tile_loop");
        
        // Finalize Q tile: normalize and project
        emit_finalize_q_tile();
    jmp_if_more_q_tiles("q_tile_loop");
}
```

#### 4. Wo Projection Tiling (Large d_model)

For large models (d_model > 4096), Wo projection becomes memory-bound. JIT can tile:

```cpp
void emit_wo_projection_tiled(int d_model, int wo_tile) {
    // Process wo_tile output dimensions at a time
    // Keeps Wo slice in L3 cache
    
    for (int o_start = 0; o_start < d_model; o_start += wo_tile) {
        int o_end = std::min(o_start + wo_tile, d_model);
        
        // Emit code for this tile
        mov(reg_wo_ptr, ptr[reg_wo_base + o_start * wo_row_stride]);
        mov(reg_out_ptr, ptr[reg_output + o_start * sizeof(float)]);
        
        L("wo_tile_loop");
            // Load context (already in ZMM0-3 or ZMM0-7)
            // Dot with Wo rows for this output tile
            emit_wo_dot_product();
            add(reg_wo_ptr, wo_row_stride);
            add(reg_out_ptr, sizeof(float));
            dec(reg_wo_count);
            jnz("wo_tile_loop");
    }
}
```

### Wo Weight Type Dispatch

JIT generates different inner loops based on Wo weight type:

| Wo Type | JIT Strategy | Performance Notes |
|---------|--------------|-------------------|
| **FP32** | Direct FMA: `vfmadd231ps` | Fastest, no dequant overhead |
| **BF16** | Shift + FMA: `vpmovzxwd` + FMA | ~5% slower, 50% memory |
| **FP16** | Convert + FMA: inline `vcvtph2ps` | ~10% slower, 50% memory |
| **Q8_1** | Block dequant + FMA | ~20% slower, 25% memory |
| **Q4_0** | Nibble unpack + dequant + FMA | ~40% slower, 12.5% memory |

```cpp
void emit_wo_dot_product_dispatch(WoWeightType wo_type) {
    switch (wo_type) {
        case WoWeightType::FP32:
            emit_wo_dot_fp32();
            break;
        case WoWeightType::BF16:
            emit_wo_dot_bf16();  // vpmovzxwd + vslld + vfmadd
            break;
        case WoWeightType::FP16:
            emit_wo_dot_fp16();  // vcvtph2ps + vfmadd
            break;
        case WoWeightType::Q8_1:
            emit_wo_dot_q8_1();  // Block dequant loop
            break;
        default:
            throw std::runtime_error("Unsupported Wo type for JIT");
    }
}
```

### Expected JIT Kernel Count

For a full Qwen2 model family deployment, expected kernel variants:

| Dimension Combo | Decode | Prefill | Total |
|-----------------|--------|---------|-------|
| head_dim=64, gqa=7:1 (0.5B) | 5 Wo types | 5 Wo types | 10 |
| head_dim=128, gqa=6:1 (1.5B) | 5 Wo types | 5 Wo types | 10 |
| head_dim=128, gqa=8:1 (3B) | 5 Wo types | 5 Wo types | 10 |
| head_dim=128, gqa=7:1 (7B) | 5 Wo types | 5 Wo types | 10 |
| head_dim=128, gqa=5:1 (14B, 32B) | 5 Wo types | 5 Wo types | 10 |
| head_dim=128, gqa=8:1 (72B) | 5 Wo types | 5 Wo types | 10 |
| **Total** | | | **60 kernels** |

Each kernel is ~10-30KB of generated code, total JIT cache: ~1-2MB.
Generation time: ~1-5ms per kernel (one-time cost at model load).

### JIT Test Strategy

| Test | Description | Validation |
|------|-------------|------------|
| `JIT_HeadDim64_MatchesRef` | head_dim=64 kernel | Bit-exact vs reference |
| `JIT_HeadDim128_MatchesRef` | head_dim=128 kernel | Bit-exact vs reference |
| `JIT_GQA_7to1` | 7:1 GQA ratio | Correct KV sharing |
| `JIT_GQA_5to1` | 5:1 GQA ratio | Correct KV sharing |
| `JIT_DecodeVsPrefill` | M=1 vs M>1 | Same output for M=1 |
| `JIT_WoFP32` | FP32 Wo projection | Matches reference |
| `JIT_WoBF16` | BF16 Wo projection | Within BF16 tolerance |
| `JIT_WoQ8_1` | Q8_1 Wo projection | Within Q8_1 tolerance |
| `JIT_CacheReuse` | Tile sizes from CPUFeatures | No cache thrashing |
| `Perf_JIT_vs_Reference` | All model sizes | ≥2x speedup |

---

### Phase 5: Pipeline Integration (Week 3)
- [ ] Add `FusedAttentionWoKernel` to `KernelFactory`
- [ ] Integrate into `Qwen2Pipeline::attention_block`
- [ ] E2E parity test
- [ ] Full benchmark comparison

---

## 9. File Structure

```
src/v2/kernels/cpu/
├── microkernels/q8_1/
│   ├── Q8DotProduct.h          # μK1 interface
│   ├── Q8DotProduct.cpp        # μK1 reference impl
│   ├── Q8DotProductAVX512.cpp  # μK1 SIMD impl
│   ├── OnlineSoftmax.h         # μK2 interface
│   ├── OnlineSoftmax.cpp       # μK2 impl
│   ├── VWeightedAccum.h        # μK3 interface
│   ├── VWeightedAccum.cpp      # μK3 reference impl
│   ├── VWeightedAccumAVX512.cpp# μK3 SIMD impl
│   ├── WoProjection.h          # μK4 interface
│   ├── WoProjection.cpp        # μK4 reference impl
│   ├── WoProjectionAVX512.cpp  # μK4 SIMD impl
│   ├── FastExp.h               # μK5 interface
│   └── FastExp.cpp             # μK5 impl
├── attention/q8_1/
│   ├── FusedAttentionWoRef.h   # Composed reference
│   ├── FusedAttentionWoRef.cpp
│   ├── FusedAttentionWoTiled.h # Cache-blocked tiled
│   ├── FusedAttentionWoTiled.cpp
│   ├── FusedAttentionWoJit.h   # JIT implementation
│   └── FusedAttentionWoJit.cpp
├── jit/q8_1/
│   ├── JitMicrokernelBase.h    # JIT base class
│   ├── JitQ8DotProduct.h       # JIT μK1
│   ├── JitOnlineSoftmax.h      # JIT μK2
│   ├── JitVWeightedAccum.h     # JIT μK3
│   ├── JitWoProjection.h       # JIT μK4
│   ├── JitFastExp.h            # JIT μK5
│   └── JitFusedAttentionWo.h   # JIT fused kernel

tests/v2/unit/microkernels/q8_1/
├── Test__Q8DotProduct.cpp
├── Test__OnlineSoftmax.cpp
├── Test__VWeightedAccum.cpp
├── Test__WoProjection.cpp
└── Test__FastExp.cpp

tests/v2/unit/jit/q8_1/
└── Test__JitMicrokernels.cpp

tests/v2/integration/q8_1/
├── Test__FusedAttentionWoRef.cpp
└── Test__FusedAttentionWoJit.cpp

tests/v2/e2e/
└── Test__FusedAttentionWoParity.cpp

tests/v2/performance/
└── Perf__FusedAttentionWo.cpp
```

---

## 10. Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| ATTENTION_CONTEXT cosine | > 0.99 | ~0.89 |
| ATTENTION_OUTPUT cosine | > 0.99 | ~0.88 |
| Top-1 token accuracy | 100% | ~95% |
| Performance vs unfused | ≥ 1.0x | N/A |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory pressure (FP32 context) | High | Tile Wo projection, stream output |
| Register pressure in JIT | Medium | Careful allocation, spill to stack |
| GQA complexity | Medium | Start with MHA, add GQA after |
| Q8_1 Wo handling | Low | Start with FP32 Wo, add Q8_1 later |

---

## 12. References

- Existing JIT kernel: `src/v2/kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h`
- Q8_1 format: `src/v2/tensors/quantized/Q8_1Tensor.h`
- Online softmax: [Flash Attention paper](https://arxiv.org/abs/2205.14135)
- AVX-512 VNNI: Intel Intrinsics Guide (`vpdpbusd`)
- CPU cache detection: `src/v2/utils/CPUFeatures.h` (provides `cpu_l2_cache_size()`, `cpu_l3_cache_size()`)

---

## 13. Dynamic Cache Detection

### CPUFeatures.h Integration

The tiled attention implementation uses runtime cache detection for portability across different CPU architectures:

```cpp
#include "v2/utils/CPUFeatures.h"

// These functions use CPUID leaf 0x04 with cached static results
uint32_t l2 = llaminar2::cpu_l2_cache_size();  // e.g., 1MB per core (Xeon Gold)
uint32_t l3 = llaminar2::cpu_l3_cache_size();  // e.g., 38MB shared (Xeon Gold)
```

### Fallback Values

For non-x86 platforms or detection failures, conservative defaults are used:
- L2: 256KB (covers most mobile/embedded CPUs)
- L3: 8MB (conservative for server CPUs)

### Tile Size Selection

| L2 Cache Size | KV_TILE | Q_TILE | Notes |
|---------------|---------|--------|-------|
| 256KB | 128 | 16 | Embedded/mobile CPUs |
| 512KB | 256 | 32 | Consumer desktop |
| 1MB | 512 | 64 | Server (Xeon Gold) |
| 2MB | 512 | 64 | High-end server (capped) |

The tile sizes are capped to avoid diminishing returns from increased loop overhead.
