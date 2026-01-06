# Project: HybridQ16 K Precision Fix

**Status**: In Progress  
**Created**: 2026-01-02  
**Branch**: `feature/typed-residuals`  
**Blocking Issue**: K_ROPE cosine = 0.878 vs FP32 (should be >0.99)

---

## Executive Summary

The HybridQ16 pipeline loses precision in the K tensor due to Q8_1 quantization bottleneck in QKV GEMM. K projection outputs have ~130 dynamic range, causing Q8_1 to zero out small values (<0.51). This cascades through RoPE into attention, degrading final logits.

**Solution**: Output K as Q16_1 directly from GEMM, then apply Q16→Q16 dynamic-scale RoPE with per-head scale tracking through to attention.

---

## Problem Analysis

### Root Cause Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CURRENT HYBRIDQ16 PIPELINE (BROKEN)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QKV GEMM → Q8_1 (Q, K, V)                                                 │
│                  ↓                                                          │
│            K has max_abs ≈ 130                                              │
│            Q8_1 step = 130/127 ≈ 1.02                                       │
│            Values |x| < 0.51 → 0  ← PRECISION LOSS                          │
│                  ↓                                                          │
│  RoPE (Q8_1 → Q16_1, fixed scale)                                          │
│                  ↓                                                          │
│            Fixed scale clips at ~4.0                                        │
│            No per-head scale tracking  ← INFORMATION LOSS                   │
│                  ↓                                                          │
│  Attention (head_scales = nullptr → 1.0f)  ← WRONG SCALE MATH              │
│                  ↓                                                          │
│  K_ROPE cosine = 0.878  ← SYMPTOM                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Evidence

```
K_PROJECTION FP32 values: -8.47655, -3.80529, -6.33802, 0.657984, -0.0529036, ...
K_PROJECTION Q8_1 qs:     -9, -3, -6, 1, 0, ...  ← Element 4 rounds to 0!

Q16_1 would preserve: qs ≈ -7 for -0.0529 (step = 130/16383 ≈ 0.008)
```

---

## Solution Architecture

### Target Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIXED HYBRIDQ16 PIPELINE                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QKV GEMM → Q8_1 (Q, V), Q16_1 (K)  ← PHASE 1-3                            │
│                  ↓                                                          │
│            K preserves full precision (256× finer than Q8_1)                │
│                  ↓                                                          │
│  RoPE:                                                                      │
│    Q: Q8_1 → Q16_1 (existing path)                                         │
│    K: Q16_1 → Q16_1 dynamic-scale  ← PHASE 4 (primitives DONE)             │
│                  ↓                                                          │
│            Per-head scales output: Q_head_scales, K_head_scales             │
│                  ↓                                                          │
│  KV Cache stores K + K_head_scales  ← PHASE 7                              │
│                  ↓                                                          │
│  Attention receives head_scales  ← PHASE 5-6                               │
│    score_scale = q_scale × k_scale / sqrt(head_dim)                         │
│                  ↓                                                          │
│  K_ROPE cosine > 0.99  ← TARGET                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: GEMM Q16_1 Output Capability ⬜ NOT STARTED

**Goal**: Add `multiply_with_precomputed_q8_1_to_q16_1()` to QuantisedGemmKernel.

**Files**:
- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h`
- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h`
- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmJit_M2.h`

**Details**:
```cpp
// New output format enum value
enum class GemmOutputFormat { FP32 = 0, Q8_1 = 1, BF16 = 2, FP16 = 3, Q16_1 = 4 };

// New method signature
void multiply_with_precomputed_q8_1_to_q16_1(
    const QuantisedPackedWeights* packed_weights,
    const Q8_1Block* activations,
    Q16_1Block* output,           // or Q16_1Block_64/128 based on block_size
    Q16BlockSize block_size,
    int M, int N, int K,
    float* out_row_scales = nullptr);  // Optional per-row scale output
```

**Q16_1 Quantization Logic** (integer domain):
```cpp
// INT32 accumulator → Q16_1 block
int32_t max_abs = find_max_abs(acc_i32, block_size);
float d = static_cast<float>(max_abs) / 16000.0f;  // Leave headroom
float inv_d = (d > 0) ? (1.0f / d) : 0.0f;

int32_t sum_qs = 0;
for (int i = 0; i < block_size; ++i) {
    int32_t q = static_cast<int32_t>(std::round(acc_i32[i] * inv_d));
    q = std::clamp(q, -16383, 16383);
    block.qs[i] = static_cast<int16_t>(q);
    sum_qs += q;
}
block.d = d;
block.sum_qs = sum_qs;
```

**Tests**: `tests/v2/unit/kernels/gemm_v4/Test__GemmQ16Output.cpp`

---

### Phase 2: GraphOrchestrator K Buffer Allocation ⬜ NOT STARTED

**Goal**: Allocate K projection buffer as Q16_1 in HybridQ16 mode.

**Files**:
- `src/v2/execution/GraphOrchestrator.cpp`

**Changes**:
```cpp
// In allocateQKVBuffers() or equivalent
if (mode == ActivationMode::HybridQ16) {
    // Q and V remain Q8_1 (will be converted to Q16_1 by RoPE/cache)
    Q_buffer = createQ8_1Tensor(seq_len, n_heads * head_dim);
    V_buffer = createQ8_1Tensor(seq_len, n_kv_heads * head_dim);
    
    // K goes directly to Q16_1 from GEMM
    K_buffer = createQ16_1Tensor(seq_len, n_kv_heads * head_dim, 
                                  optimal_q16_block_size(head_dim));
}
```

---

### Phase 3: FusedQKVGEMMStage Mixed Output ⬜ NOT STARTED

**Goal**: Execute QKV GEMM with K→Q16_1, Q/V→Q8_1.

**Files**:
- `src/v2/kernels/cpu/gemm_v4/FusedGEMM.h`
- `src/v2/kernels/cpu/gemm_v4/FusedGEMM.cpp`
- `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp`

**New Method**:
```cpp
// FusedGEMM.h
void execute_qkv_mixed_precision(
    const Q8_1Block* activations,
    const QuantisedPackedWeights* Wq,
    const QuantisedPackedWeights* Wk,
    const QuantisedPackedWeights* Wv,
    Q8_1Block* Q_out,      // Q8_1
    void* K_out,           // Q16_1 (variable block size)
    Q8_1Block* V_out,      // Q8_1
    Q16BlockSize k_block_size,
    int M, int N_q, int N_k, int N_v, int K_dim);
```

---

### Phase 4: RoPEStage Q16→Q16 Dynamic-Scale for K ✅ PRIMITIVES DONE

**Goal**: Wire existing Q16→Q16 dynamic-scale RoPE into RoPEStage.

**Files**:
- `src/v2/execution/compute_stages/stages/RoPEStage.cpp`
- `src/v2/kernels/cpu/ops/CPURoPEKernelT.h`
- `src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp`

**Primitives Ready** (optimized today with AVX2/AVX512):
- `apply_rope_q16_to_q16_head_dynamic_scale<BlockType>()` - 1.55x speedup
- `apply_rope_q16_to_q16_dynamic_scale<BlockType>()` - high-level wrapper
- `apply_rope_q16_to_q16_dynamic_scale_dispatch()` - runtime block size dispatch

**New Kernel Method**:
```cpp
// CPURoPEKernelT<ActivationPrecision::Q16_1>
bool apply_q16_to_q16_dynamic_scale(
    TensorBase* K_in,
    TensorBase* K_out,
    const int* position_ids,
    int seq_len,
    int n_kv_heads,
    int head_dim,
    float rope_theta,
    float* out_head_scales);  // [seq_len * n_kv_heads] output scales
```

**RoPEStage Changes**:
```cpp
// Detect Q16_1 K input (from Phase 2/3)
const bool k_is_q16 = params_.K && 
                       params_.K->native_type() == TensorType::Q16_1;

if (hybrid_q16_mode && k_is_q16) {
    // K: Q16→Q16 dynamic-scale (preserves precision, outputs head_scales)
    kernel->apply_q16_to_q16_dynamic_scale(
        params_.K, params_.K_out,
        position_ids.data(), seq_len, n_kv_heads, params_.head_dim,
        params_.theta_base, params_.k_head_scales_out);
    
    // Q: Q8_1→Q16_1 (existing path)
    kernel->apply_q8_1_to_q16_fixed_scale(...);
}
```

---

### Phase 5: FusedAttentionWoParams head_scales Fields ⬜ NOT STARTED

**Goal**: Add head_scales to attention params struct.

**Files**:
- `src/v2/tensors/TensorKernels.h`

**Changes to `FusedAttentionWoParams`**:
```cpp
struct FusedAttentionWoParams
{
    // ... existing fields ...
    
    // === Per-head scale factors (from RoPE/normalization) ===
    
    /**
     * @brief Q tensor per-head scales [seq_len_q * n_heads]
     * 
     * If nullptr, assumes uniform scale = 1.0f.
     * For HybridQ16: populated by RoPE Q8_1→Q16_1 conversion.
     */
    const float* q_head_scales = nullptr;
    
    /**
     * @brief K/V tensor per-head scales [kv_len * n_kv_heads]
     * 
     * If nullptr, assumes uniform scale = 1.0f.
     * For HybridQ16: populated by RoPE Q16_1→Q16_1 conversion for K,
     * and Q8_1→Q16_1 conversion for V in KVCacheAppendStage.
     */
    const float* kv_head_scales = nullptr;
    
    // === Helper methods ===
    
    float get_qk_scale(int q_head, int kv_head) const {
        float s_q = q_head_scales ? q_head_scales[q_head] : 1.0f;
        float s_k = kv_head_scales ? kv_head_scales[kv_head] : 1.0f;
        return s_q * s_k / std::sqrt(static_cast<float>(head_dim));
    }
};
```

---

### Phase 6: FusedAttentionWoStage Wire head_scales ⬜ NOT STARTED

**Goal**: Pass head_scales from RoPE output to attention kernel.

**Files**:
- `src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp`

**Changes**:
```cpp
// In execute() for Q16_INTEGER backend
q16_params.q_head_scales = params_.q_head_scales;
q16_params.kv_head_scales = params_.kv_head_scales;

// Log for debugging
LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER head_scales: "
          << "q=" << (params_.q_head_scales ? "provided" : "nullptr")
          << " kv=" << (params_.kv_head_scales ? "provided" : "nullptr"));
```

---

### Phase 7: KVCacheAppendStage Store K Scales ⬜ NOT STARTED

**Goal**: Store K head_scales alongside K data for decode lookups.

**Files**:
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp`
- `src/v2/models/KVCache.h`
- `src/v2/models/KVCache.cpp`

**KVCache Changes**:
```cpp
class KVCache {
    // ... existing K/V tensor storage ...
    
    // Per-position, per-head scales for HybridQ16 mode
    std::vector<std::vector<float>> k_scales_;  // [layer][position * n_kv_heads]
    std::vector<std::vector<float>> v_scales_;  // [layer][position * n_kv_heads]
    
public:
    void append_k_scales(int layer, const float* scales, int n_positions, int n_kv_heads);
    const float* get_k_scales(int layer, int start_pos, int len) const;
};
```

---

## Test Plan

### Unit Tests

| Phase | Test File | Coverage |
|-------|-----------|----------|
| 1 | `Test__GemmQ16Output.cpp` | Q16_1 output quantization correctness |
| 4 | `Test__Q16_to_Q16_RoPE_DynamicScale.cpp` | ✅ Already passing |
| 5 | `Test__FusedAttentionWoParams.cpp` | head_scales getter methods |

### Integration Tests

```bash
# Primary parity test
./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline

# Stage-by-stage comparison
LLAMINAR_LOG_LEVEL=INFO ./build_v2_integration/tests/v2/v2_integration_hybridq16_vs_fp32_pipeline 2>&1 | grep cosine
```

### Expected Results After Full Fix

| Stage | Current | Target |
|-------|---------|--------|
| K_PROJECTION | ~0.999 | ~0.999 |
| K_ROPE | **0.878** | **>0.99** |
| ATTENTION_CONTEXT | ~0.95 | >0.98 |
| Final logits | ~0.85 | >0.95 |

---

## Critical Constraints

### Integer Domain Only

**The HybridQ16 path MUST remain 100% integer/quantized domain.**

DO NOT:
- Add FP32 intermediate buffers
- Dequantize to FP32 for processing
- Use FP32 RoPE followed by requantization

### Block Size Consistency

All Q16_1 tensors in a head must use the same block size:
- `Q16_1Block` (32 elements) - legacy, avoid
- `Q16_1Block_64` (64 elements) - head_dim=64
- `Q16_1Block_128` (128 elements) - head_dim=128

Use `optimal_q16_block_size(head_dim)` for automatic selection.

---

## Dependencies

```
Phase 1 (GEMM Q16 output)
    ↓
Phase 2 (K buffer allocation)
    ↓
Phase 3 (FusedQKVGEMMStage mixed output)
    ↓
Phase 4 (RoPEStage Q16→Q16) ← Primitives already done
    ↓
Phase 5 (FusedAttentionWoParams head_scales)
    ↓
Phase 6 (FusedAttentionWoStage wire scales)
    ↓
Phase 7 (KVCache scale storage)
```

Phases 5-7 can be developed in parallel with Phases 1-3.

---

## Progress Tracking

- [x] **Phase 0**: Dynamic-scale Q16→Q16 RoPE primitives (completed 2026-01-02)
  - [x] `apply_rope_q16_to_q16_head_dynamic_scale<BlockType>()` scalar
  - [x] AVX2 optimized (1.16x speedup)
  - [x] AVX512 optimized (1.55x speedup)
  - [x] Multi-block-size support (32, 64, 128)
  - [x] Unit tests passing
- [x] **Phase 1**: GEMM Q16_1 output capability (pre-existing: `GemmOutputFormat::Q16_1 = 4`)
- [x] **Phase 2**: GraphOrchestrator K buffer allocation (completed 2026-01-02)
  - [x] Added `HybridBufferType::K_GEMM_Output`, `Q_GEMM_Output`, `V_GEMM_Output`
  - [x] HybridQ16PrecisionConfig with `k_gemm_output = Q16_1`
  - [x] K allocated with `k_prec` in GraphOrchestrator
- [x] **Phase 3**: FusedQKVGEMMStage mixed output (completed 2026-01-02)
  - [x] Added `GEMMProjectionQ16_1` struct to FusedGEMM.h
  - [x] Implemented `execute_q8_1_mixed_qkv()` for Q=Q8_1, K=Q16_1, V=Q8_1
  - [x] FusedQKVGEMMStage detects mixed-precision QKV and routes appropriately
- [x] **Phase 4**: RoPEStage Q16→Q16 wiring (completed 2026-01-02)
  - [x] Added `apply_mixed_q8_k16_to_q16()` to ITensorRoPE interface
  - [x] Implemented in CPURoPEKernelT (Q→fixed, K→dynamic-scale)
  - [x] RoPEStage detects k_is_q16_1 and routes to new method
- [x] **Phase 5**: FusedAttentionWoParams head_scales fields (completed 2026-01-02)
  - [x] Added `q_head_scales`, `k_head_scales` to FusedAttentionWoParams
  - [x] Added `K_head_scales` to FusedAttentionWoStage::Params
- [x] **Phase 6-7**: Wire K_head_scales pipeline (completed 2026-01-02)
  - [x] K_head_scales buffer in Qwen2ActivationBuffers
  - [x] K_head_scales vector in InferenceState
  - [x] Allocated in GraphOrchestrator for HybridQ16 mode
  - [x] Wired through RoPEStage and FusedAttentionWoStage to kernel
- [ ] **Phase 8**: Attention kernel k_head_scales usage ← **REMAINING WORK**
  - [ ] Extend Q16IntegerAttentionParams for per-position K scales
  - [ ] Wire k_head_scales in Q16FusedAttentionKernel::compute()
  - [ ] Update OnlineSoftmax for per-position K scales (Option C: Pass Scale to Softmax)
  - [ ] Wire k_scales through decode and prefill paths
  - [ ] See [PHASE8_ATTENTION_KERNEL_K_SCALES.md](./PHASE8_ATTENTION_KERNEL_K_SCALES.md)

---

## References

- [HANDOVER_HYBRIDQ16_K_PRECISION.md](./HANDOVER_HYBRIDQ16_K_PRECISION.md) - Original root cause analysis
- [PROJECT_Q16_INTEGER_ATTENTION_V2.md](./PROJECT_Q16_INTEGER_ATTENTION_V2.md) - Full integer attention design
- `src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp` - Q16 RoPE implementations
- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` - Reference for Q8_1 output pattern
