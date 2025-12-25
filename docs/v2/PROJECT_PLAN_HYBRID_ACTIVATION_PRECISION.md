# Project Plan: Hybrid Activation Precision Mode

## Executive Summary

This document outlines the implementation plan for a new `ActivationPrecision::Hybrid` mode in Llaminar V2. The Hybrid mode uses optimal precision for each operation in the transformer pipeline, eliminating unnecessary quantization/requantization steps that currently cause numerical divergence (cosine similarity 0.85-0.89 vs FP32 reference).

**Target**: Achieve ≥0.995 cosine similarity with FP32 reference while maintaining 30-50% of Q8_1 performance gains.

---

## Problem Analysis

### Current Q8_1 Pipeline Issues

The current `ActivationPrecision::Q8_1` mode applies uniform quantization to all activation buffers, causing three unnecessary requantization points:

| Stage | Current Flow | Issue |
|-------|-------------|-------|
| **QKV GEMM** | FP32 input → quantize → Q8_1 × Q8_1 weights → FP32/Q8_1 output | ✅ Acceptable |
| **RoPE** | Q8_1 → dequant → FP32 rotate → requant → Q8_1 | ❌ **Unnecessary requant** |
| **Attention Q×K** | Q8_1 × Q8_1 (VNNI) → FP32 scores | ✅ Acceptable (small error) |
| **Softmax × V** | FP32 weights × dequant(V) → FP32 context | ✅ Correct |
| **Wo Projection** | FP32 context → quantize → Q8_1 × Wo → FP32 | ❌ **Unnecessary context quantization** |
| **Residual Add** | FP32 | ✅ Correct |
| **FFN Gate/Up** | Similar to QKV | Same pattern |
| **FFN Down** | Similar to Wo | Same pattern |

### Root Cause

1. **Softmax amplification**: 0.1% Q/K quantization error can become 10%+ output error through softmax
2. **RoPE requantization**: Dequant→rotate→requant loses precision unnecessarily
3. **Context quantization**: FP32 attention context quantized for VNNI Wo when FP32 Wo exists

---

## Proposed Hybrid Precision Map

| Buffer/Operation | Hybrid Precision | Rationale |
|------------------|------------------|-----------|
| **residual** | FP32 | Always FP32 for numerical stability |
| **normalized** | FP32 | Always FP32 (RMSNorm operates on residual) |
| **hidden** | FP32 | Embedding output |
| **logits** | FP32 | Softmax input |
| **Q, K, V** (after QKV GEMM) | Q8_1 | Quantized GEMM output is acceptable |
| **Q, K** (after RoPE) | FP32 | **Keep in FP32 to avoid requant** |
| **K/V Cache** | BF16 | Better than Q8_1 (2× vs 3.5× compression), 8-bit mantissa |
| **attn_output** (context) | FP32 | Softmax × V output is already FP32 |
| **Wo Projection** | FP32 weights | **Skip context quantization** |
| **attn_proj** | FP32 | Already FP32 |
| **gate, up** | Q8_1 | Acceptable for FFN |
| **ffn_output** | FP32 | Already FP32 |

---

## Implementation Phases

### Phase 1: Add Hybrid Enum and Config (Est: 2 hours)

**Files to modify:**

#### 1.1 `src/v2/execution/RuntimeConfig.h`

Add new enum value:

```cpp
enum class ActivationPrecision
{
    FP32,   ///< 32-bit float activations (default, highest accuracy)
    BF16,   ///< bfloat16 activations (Intel AMX, reduced bandwidth)
    FP16,   ///< 16-bit float activations (ARM/GPU optimization)
    Q8_1,   ///< Block-quantized int8 (36 bytes per 32 elements, 3.5x compression)
    Hybrid  ///< Mixed precision: FP32 residual, BF16 KV cache, Q8_1 activations
};
```

#### 1.2 `src/v2/execution/RuntimeConfig.cpp`

Add parsing support:

```cpp
// In parseActivationPrecision():
if (str == "hybrid" || str == "Hybrid" || str == "HYBRID") {
    return ActivationPrecision::Hybrid;
}
```

#### 1.3 `src/v2/cli/ArgumentParser.cpp`

Update help text for `--activation-precision`:

```cpp
"Activation buffer precision: fp32, bf16, fp16, q8_1, hybrid (default: fp32)"
```

**Testing:**
- Unit test: Parse "hybrid" string correctly
- Integration test: CLI accepts `--activation-precision hybrid`

---

### Phase 2: Buffer Precision Mapping (Est: 4 hours)

**Goal:** Replace single `act_prec` with per-buffer precision selection for Hybrid mode.

#### 2.1 Create `HybridBufferPrecisions` Structure

New file: `src/v2/execution/HybridPrecisionConfig.h`

```cpp
#pragma once
#include "RuntimeConfig.h"

namespace llaminar2 {

/// Per-buffer precision configuration for Hybrid mode
struct HybridBufferPrecisions {
    ActivationPrecision qkv_gemm_output = ActivationPrecision::Q8_1;  // QKV GEMM output
    ActivationPrecision qk_rope_output = ActivationPrecision::FP32;   // After RoPE (skip requant!)
    ActivationPrecision kv_cache = ActivationPrecision::BF16;         // BF16 KV cache
    ActivationPrecision attention_context = ActivationPrecision::FP32; // softmax × V
    ActivationPrecision ffn_gate_up = ActivationPrecision::Q8_1;      // FFN intermediate
    
    /// Get precision for a specific buffer type
    ActivationPrecision getPrecision(const std::string& buffer_name) const;
    
    /// Factory for default hybrid configuration
    static HybridBufferPrecisions defaultHybrid();
};

/// Resolve effective precision given global setting
ActivationPrecision resolveBufferPrecision(
    ActivationPrecision global_precision,
    const std::string& buffer_name,
    const HybridBufferPrecisions* hybrid_config = nullptr);

} // namespace llaminar2
```

#### 2.2 Modify `GraphOrchestrator::initializeInferenceState()`

Location: `src/v2/execution/GraphOrchestrator.cpp` lines 780-910

```cpp
void GraphOrchestrator::initializeInferenceState(...) {
    // ...
    
    // Resolve hybrid config if needed
    std::unique_ptr<HybridBufferPrecisions> hybrid_config;
    if (config.activation_precision == ActivationPrecision::Hybrid) {
        hybrid_config = std::make_unique<HybridBufferPrecisions>(
            HybridBufferPrecisions::defaultHybrid());
        LOG_INFO("[GraphOrchestrator] Using Hybrid activation precision");
    }
    
    // QKV buffers - use hybrid-aware precision
    ActivationPrecision qkv_prec = resolveBufferPrecision(
        act_prec, "qkv_gemm_output", hybrid_config.get());
    state_.Q = factory.createActivation({...}, qkv_prec, device_idx);
    state_.K = factory.createActivation({...}, qkv_prec, device_idx);
    state_.V = factory.createActivation({...}, qkv_prec, device_idx);
    
    // For Hybrid: Q_rope and K_rope buffers in FP32 (post-RoPE)
    if (config.activation_precision == ActivationPrecision::Hybrid) {
        state_.Q_rope = factory.createFP32({...}, device_idx);
        state_.K_rope = factory.createFP32({...}, device_idx);
    }
    
    // KV cache - use BF16 for Hybrid
    ActivationPrecision kv_cache_prec = resolveBufferPrecision(
        act_prec, "kv_cache", hybrid_config.get());
    state_.kv_cache = createUnifiedKVCache(kv_cache_prec, ...);
    
    // attn_output - FP32 for Hybrid (skip context quantization)
    ActivationPrecision attn_out_prec = resolveBufferPrecision(
        act_prec, "attention_context", hybrid_config.get());
    state_.attn_output = factory.createActivation({...}, attn_out_prec, device_idx);
    
    // ...
}
```

#### 2.3 Add New State Buffers

In `GraphOrchestrator.h`:

```cpp
struct InferenceState {
    // ... existing buffers ...
    
    // Hybrid mode: FP32 Q/K after RoPE (to avoid requant)
    std::unique_ptr<TensorBase> Q_rope;
    std::unique_ptr<TensorBase> K_rope;
};
```

**Testing:**
- Unit test: `resolveBufferPrecision()` returns correct values for each buffer
- Unit test: Hybrid mode allocates Q_rope/K_rope buffers
- Integration test: Memory usage comparison (Hybrid vs Q8_1 vs FP32)

---

### Phase 3: RoPE Precision Flow (Est: 6 hours)

**Goal:** Eliminate RoPE requantization by outputting to FP32 buffers.

#### 3.1 Modify RoPE Kernel to Support Mixed I/O Precision

Current: `CPURoPEKernelT<Q8_1>` only accepts Q8_1 → Q8_1

New: Add method for Q8_1 input → FP32 output (dequant + rotate, no requant)

Location: `src/v2/kernels/cpu/ops/CPURoPEKernelT.h`

```cpp
// New method for Hybrid mode: Q8_1 input → FP32 output (no requant)
bool apply_q8_1_to_fp32(
    Q8_1Block* input_q,
    Q8_1Block* input_k,
    float* output_q,    // FP32 output
    float* output_k,    // FP32 output
    const int* position_ids,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta);
```

#### 3.2 Add RoPE Primitive for Q8_1→FP32

Location: `src/v2/kernels/cpu/primitives/RoPEPrimitives.h/cpp`

```cpp
/// RoPE with dequantization - no requantization (for Hybrid mode)
void apply_rope_q8_1_to_fp32(
    const Q8_1Block* Q_in,
    const Q8_1Block* K_in,
    float* Q_out,
    float* K_out,
    const int* position_ids,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    float rope_theta);
```

Implementation (pseudo):
```cpp
void apply_rope_q8_1_to_fp32(...) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int h = 0; h < n_heads; ++h) {
            for (int b = 0; b < blocks_per_head; ++b) {
                // Dequantize Q8_1 block
                float dequant[32];
                dequantize_q8_1_block(Q_in[...], dequant);
                
                // Apply rotation in FP32
                for (int i = 0; i < 32; i += 2) {
                    float c = cos_table[...];
                    float s = sin_table[...];
                    float x = dequant[i];
                    float y = dequant[i + half_dim];
                    Q_out[...] = x * c - y * s;  // Direct to FP32 output
                    Q_out[... + half_dim] = x * s + y * c;
                }
                // NO REQUANTIZATION - stays in FP32
            }
        }
    }
}
```

#### 3.3 Modify RoPEStage for Hybrid Mode

Location: `src/v2/execution/ComputeStage.cpp` RoPEStage::execute()

Add parameter to RoPEStage::Params:

```cpp
struct Params {
    // ... existing ...
    TensorBase* Q_output = nullptr;  // Optional: different output buffer
    TensorBase* K_output = nullptr;  // Optional: different output buffer
    bool output_fp32 = false;        // For Hybrid mode
};
```

Execution logic:
```cpp
if (params_.output_fp32 && params_.Q->native_type() == TensorType::Q8_1) {
    // Hybrid mode: Q8_1 input → FP32 output
    auto* q_q8 = dynamic_cast<Q8_1Tensor*>(params_.Q);
    auto* q_out_fp32 = dynamic_cast<FP32Tensor*>(params_.Q_output);
    primitives::apply_rope_q8_1_to_fp32(
        q_q8->q8_1_blocks(),
        k_q8->q8_1_blocks(),
        q_out_fp32->mutable_data(),
        k_out_fp32->mutable_data(),
        ...);
} else {
    // Existing path: in-place Q8_1→Q8_1 or FP32→FP32
    kernel->apply_tensor(...);
}
```

**Testing:**
- Unit test: `apply_rope_q8_1_to_fp32` matches FP32 reference
- E2E test: Hybrid RoPE output vs Q8_1 RoPE output cosine similarity

---

### Phase 4: Attention Kernel Precision Handling (Est: 8 hours)

**Goal:** Support FP32 Q/K inputs (after Hybrid RoPE) with Q8_1 KV cache.

#### 4.1 FusedAttentionWoKernel Input Validation

Current: Requires Q8_1 for Q/K/V inputs (line 110-120)

For Hybrid: Q/K can be FP32 (from RoPE), but K_cache/V_cache are from BF16 cache.

Location: `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h`

```cpp
bool compute(...) {
    // Validate Q - can be FP32 (Hybrid) or Q8_1 (full Q8_1 mode)
    auto *Q_q8 = dynamic_cast<Q8_1Tensor *>(Q);
    auto *Q_fp32 = dynamic_cast<FP32Tensor *>(Q);
    
    if (!Q_q8 && !Q_fp32) {
        LOG_ERROR("FusedAttentionWoKernel requires Q8_1 or FP32 Q tensor");
        return false;
    }
    
    // For K/V: check cache tensor type
    // In Hybrid mode, K_cache/V_cache come from BF16 cache
    auto *K_q8 = dynamic_cast<Q8_1Tensor *>(K);
    auto *K_bf16 = dynamic_cast<BF16Tensor *>(K);
    // ...
}
```

#### 4.2 JIT Kernel Modification for FP32 Q/K

The JIT attention kernel (`JitFusedAttentionWo`) currently assumes Q8_1 inputs for VNNI dot product.

For Hybrid mode FP32 Q/K:
- Option A: Quantize Q/K on-the-fly in JIT preamble
- Option B: Create separate FP32 Q×K path (slower but simpler)

**Recommended**: Option A - quantize Q on-the-fly per head

Location: `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.cpp`

```cpp
// In emit_prefill_head() / emit_decode_head():
if (config_.q_input_precision == ActivationPrecision::FP32) {
    // Quantize current Q head to scratch Q8_1 buffer
    emit_quantize_fp32_to_q8_1(q_fp32_ptr, q_scratch_q8);
    // Use quantized Q for VNNI dot product
}
```

This is acceptable because:
1. Q is only quantized once per head (not requantized multiple times)
2. The quantization error is bounded by Q8_1 precision
3. VNNI dot product speed gain outweighs quantization overhead

#### 4.3 BF16 KV Cache Integration

Current KV cache types: FP32, BF16, FP16, Q8_1

For attention kernel to consume BF16 K_cache/V_cache:
1. Add BF16→Q8_1 dequantization path in attention kernel
2. Or: dequantize BF16 to FP32, then use FP32 attention path

**Recommended**: Dequantize BF16→FP32 at cache read time (in `KVCacheAppendStage`)

Location: `src/v2/execution/ComputeStage.cpp` KVCacheAppendStage

```cpp
// When reading from BF16 cache for attention:
if (cache->precision() == ActivationPrecision::BF16) {
    // Dequantize to FP32 workspace for attention
    dequantize_bf16_to_fp32(k_cache_bf16, k_workspace_fp32, ...);
    dequantize_bf16_to_fp32(v_cache_bf16, v_workspace_fp32, ...);
}
```

**Testing:**
- Unit test: FusedAttentionWoKernel with FP32 Q/K inputs
- Unit test: BF16 KV cache read/dequantize
- E2E test: Full Hybrid attention layer vs FP32 reference

---

### Phase 5: Wo Projection FP32 Path (Est: 4 hours)

**Goal:** Use FP32 Wo weights to skip context quantization.

#### 5.1 Current Wo Weight Handling

Location: `FusedAttentionWoKernel.h` lines 124-180

Currently supports: Q8_1_VNNI_PACKED, Q8_1, FP32, FP16, BF16

The FP32 path already exists! The issue is that the **context** (attention output) is being quantized before Wo projection.

#### 5.2 Ensure FP32 Context Path

In Hybrid mode with FP32 Wo:
- Context is already FP32 (from softmax × V)
- Just use FP32 GEMM for context × Wo

Location: `src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.cpp`

The current kernel architecture:
1. Computes attention context in FP32 (zmm registers)
2. **For VNNI Wo**: quantizes context to Q8_1, then VNNI GEMM
3. **For FP32 Wo**: should use FP32 GEMM directly

Check if FP32 Wo path quantizes context:

```cpp
// In emit_wo_projection():
switch (wo_type) {
case WoWeightType::Q8_1_VNNI_PACKED:
    emit_quantize_context();  // Quantize FP32 context → Q8_1
    emit_vnni_gemm();
    break;
case WoWeightType::FP32:
    emit_fp32_gemm();  // Should NOT quantize - verify this path
    break;
}
```

**Action Items:**
1. Verify FP32 Wo path doesn't quantize context
2. If it does, fix it to use direct FP32 GEMM
3. Default Hybrid to use FP32 Wo weights (or BF16 for bandwidth)

**Testing:**
- Unit test: FP32 Wo projection with FP32 context
- E2E test: Compare Wo output precision vs Q8_1 mode

---

### Phase 6: FFN Precision Handling (Est: 3 hours)

**Goal:** Apply similar hybrid logic to FFN stages.

#### 6.1 FFN Gate/Up GEMM

Keep Q8_1 output for FFN intermediate activations (acceptable precision loss).

No changes needed - already using `act_prec` for gate/up buffers.

#### 6.2 FFN Down Projection

Similar to Wo: currently may quantize gate output before Down projection.

**Action Items:**
1. Audit FFNDownStage for unnecessary quantization
2. For Hybrid: use FP32 SwiGLU output directly into Down projection
3. Down projection output feeds residual (must be FP32)

**Testing:**
- Unit test: FFN precision flow
- E2E test: FFN layer output vs FP32 reference

---

### Phase 7: Graph Construction Updates (Est: 4 hours)

**Goal:** Wire up Hybrid buffers in `Qwen2Graph` and other model graphs.

#### 7.1 Qwen2Graph Modifications

Location: `src/v2/pipelines/qwen/Qwen2Graph.cpp`

```cpp
void Qwen2Graph::buildAttentionBlock(int layer_idx, ...) {
    // ...
    
    if (config_.activation_precision == ActivationPrecision::Hybrid) {
        // QKV GEMM: output to Q8_1 buffers
        auto qkv_stage = FusedQKVGEMMStage::create(...);
        qkv_stage->setOutputPrecision(ActivationPrecision::Q8_1);
        
        // RoPE: Q8_1 input → FP32 output (Q_rope, K_rope)
        auto rope_stage = RoPEStage::create(...);
        rope_stage->setOutputBuffers(state_.Q_rope.get(), state_.K_rope.get());
        rope_stage->setOutputPrecision(ActivationPrecision::FP32);
        
        // Attention: FP32 Q/K from RoPE, BF16 K_cache/V_cache
        auto attn_stage = FusedAttentionWoStage::create(...);
        attn_stage->setQKPrecision(ActivationPrecision::FP32);
        attn_stage->setKVCachePrecision(ActivationPrecision::BF16);
        // ...
    } else {
        // Existing uniform precision path
        // ...
    }
}
```

#### 7.2 Stage Precision Configuration

Add precision metadata to stage params:

```cpp
struct RoPEStage::Params {
    // ...
    ActivationPrecision input_precision = ActivationPrecision::FP32;
    ActivationPrecision output_precision = ActivationPrecision::FP32;
};
```

**Testing:**
- Integration test: Build Hybrid graph, verify stage connections
- E2E test: Run Hybrid inference end-to-end

---

### Phase 8: Testing and Validation (Est: 6 hours)

#### 8.1 New Unit Tests

| Test File | Coverage |
|-----------|----------|
| `Test__HybridPrecisionConfig.cpp` | Buffer precision resolution |
| `Test__RoPEQ8ToFP32.cpp` | Q8_1→FP32 RoPE kernel |
| `Test__FusedAttentionFP32QK.cpp` | Attention with FP32 Q/K |
| `Test__BF16KVCache.cpp` | BF16 cache read/dequant |

#### 8.2 E2E Parity Tests

```bash
# Run with Hybrid precision
./build_v2_e2e_release/llaminar2 \
    --activation-precision hybrid \
    -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "The capital of France is" \
    -n 20 -t 0

# Compare against FP32 reference
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt "The capital of France is" \
    --max-tokens 20
```

#### 8.3 Cosine Similarity Targets

| Component | Q8_1 Mode | Hybrid Target |
|-----------|-----------|---------------|
| QKV GEMM | ≥0.995 | ≥0.995 |
| RoPE | ~0.98 | ≥0.9995 |
| Attention | 0.85-0.89 | ≥0.995 |
| Wo Projection | ~0.99 | ≥0.999 |
| Full Layer | 0.85-0.89 | ≥0.990 |
| End-to-End | 0.80-0.85 | ≥0.985 |

#### 8.4 Canonical E2E Parity Tests

The following E2E parity tests serve as the primary validation for Hybrid mode correctness:

**Test File:** `tests/v2/e2e/parity/internal/hybrid_vs_fp32_q8_1/Test__HybridPipeline_vs_FP32_Q8_1_LayerByLayer.cpp`

**Test Suite:** `Test__HybridPipeline_vs_FP32_Q8_1`

| Test Name | Purpose |
|-----------|---------|
| `SnapshotComparison_Hybrid_vs_FP32_vs_Q8_1` | Layer-by-layer cosine similarity and relative L2 comparison across all pipeline stages |
| `CriticalStages_AttentionAccuracy` | Focused validation of attention-related stages where Hybrid mode diverges from Q8_1 |

**Metrics Captured:**
- Cosine similarity vs FP32 reference (target: ≥0.985)
- Relative L2 error (informational)
- Max/Mean absolute difference (informational)
- Top-5 logit overlap (end-to-end validation)

**Build & Run:**
```bash
# Build with E2ERelease (includes snapshot capture)
cmake -B build_v2_e2e_release -S src/v2 -DCMAKE_BUILD_TYPE=E2ERelease
cmake --build build_v2_e2e_release --target v2_e2e_parity_hybrid_vs_fp32_q8_1_pipeline --parallel

# Run parity tests
ctest --test-dir build_v2_e2e_release -R "V2_E2E_Parity_HybridPipeline" --output-on-failure
```

**CTest Labels:** `V2;E2E;Parity;Internal;Hybrid;FP32;Q8_1;LayerByLayer`

**Snapshot Keys Compared:**
- `EMBEDDING` - Initial embedding output
- `layer{N}_ATTENTION_NORM` - Pre-attention normalization
- `layer{N}_Q_PROJECTION`, `layer{N}_K_PROJECTION`, `layer{N}_V_PROJECTION` - QKV GEMM outputs
- `layer{N}_Q_ROPE`, `layer{N}_K_ROPE` - Post-RoPE Q/K (critical for Hybrid)
- `layer{N}_ATTENTION_CONTEXT` - Attention softmax × V output
- `layer{N}_ATTENTION_OUTPUT` - Wo projection output
- `layer{N}_ATTENTION_RESIDUAL` - After residual add
- `layer{N}_FFN_NORM`, `layer{N}_FFN_GATE`, `layer{N}_FFN_UP`, `layer{N}_FFN_SWIGLU`, `layer{N}_FFN_DOWN`
- `layer{N}_FFN_RESIDUAL` - After FFN residual add
- `FINAL_NORM`, `LM_HEAD` - Final stages

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FP32 Q/K breaks VNNI dot product | Medium | High | Quantize Q on-the-fly in JIT preamble |
| BF16 KV cache precision loss | Low | Medium | BF16 has 8-bit mantissa (vs 7-bit Q8_1), acceptable |
| Increased memory bandwidth | Medium | Medium | Only Q/K buffers are larger; KV cache is smaller |
| Performance regression | Medium | Medium | Benchmark each phase; fallback to Q8_1 if needed |
| Complex precision dispatch | Low | High | Well-defined precision enum per buffer |

---

## Resource Estimates

| Phase | Estimated Hours | Dependencies |
|-------|-----------------|--------------|
| 1. Enum/Config | 2 | None |
| 2. Buffer Mapping | 4 | Phase 1 |
| 3. RoPE Precision | 6 | Phase 2 |
| 4. Attention Kernel | 8 | Phase 3 |
| 5. Wo Projection | 4 | Phase 4 |
| 6. FFN Handling | 3 | Phase 5 |
| 7. Graph Construction | 4 | Phases 2-6 |
| 8. Testing | 6 | Phases 1-7 |
| **Total** | **37 hours** | |

---

## Success Criteria

1. ✅ `--activation-precision hybrid` CLI option works
2. ✅ Hybrid mode allocates per-buffer precision correctly
3. ✅ RoPE outputs FP32 without requantization
4. ✅ Attention kernel accepts FP32 Q/K inputs
5. ✅ BF16 KV cache integration works
6. ✅ Wo projection uses FP32 path (no context quantization)
7. ✅ E2E cosine similarity ≥0.985 vs FP32 reference
8. ✅ Performance ≥50% of Q8_1 mode throughput
9. ✅ Memory usage ≤ FP32 mode (better than Q8_1 due to BF16 KV cache)

---

## Implementation Status

### Phase Completion Tracker

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Enum/Config | ✅ COMPLETE | `Hybrid` enum, CLI parsing, help text |
| Phase 2: Buffer Mapping | ✅ COMPLETE | `HybridPrecisionConfig`, Q_rope/K_rope buffers allocated |
| Phase 3: RoPE Precision | ✅ COMPLETE | `RoPEStage::Params::hybrid_mode`, FP32 output buffers |
| Phase 4: Attention Kernel | 🔄 IN PROGRESS | FP32 Q/K input support needed |
| Phase 5: Wo Projection | ✅ COMPLETE | FP32 Wo path structure |
| Phase 6: FFN Handling | ✅ COMPLETE | Same precision as Q8_1 mode |
| Phase 7: Graph Construction | 🔄 IN PROGRESS | Hybrid mode flag not wired through |
| Phase 8: Unit Tests | ✅ COMPLETE | 18 unit tests pass |
| E2E Parity Tests | ✅ CREATED | Test infrastructure complete |

### Current Implementation Gap

**Issue Identified by E2E Tests (2025-01-XX):**

The E2E parity test (`Test__HybridPipeline_vs_FP32_Q8_1`) revealed that `hybrid_mode=false` is being passed to RoPEStage even when `ActivationPrecision::Hybrid` is configured. This results in:

| Stage | Hybrid Cosine | Q8_1 Cosine | Issue |
|-------|---------------|-------------|-------|
| `layer0_Q_ROPE` | 0.9849 | 1.0000 | Hybrid mode not using FP32 output |
| `layer0_K_ROPE` | 0.9763 | 1.0000 | Same issue |
| `layer0_ATTENTION_CONTEXT` | 0.0000 | 0.7486 | Attention receiving wrong inputs |
| `layer0_ATTENTION_OUTPUT` | 0.0000 | 0.7985 | Cascading failure |

**Root Cause:** The `hybrid_mode` flag is defined in `RoPEStage::Params` but not being set correctly when the graph is constructed.

**Files Requiring Updates:**
1. `src/v2/pipelines/qwen/Qwen2Graph.cpp` - Wire `hybrid_mode` flag to RoPEStage
2. `src/v2/execution/GraphOrchestrator.cpp` - Pass Hybrid config to graph builder
3. `src/v2/kernels/cpu/attention/*/FusedAttentionWoKernel.*` - Handle FP32 Q/K inputs

### Next Implementation Steps

1. **Wire hybrid_mode through graph construction:**
   - `GraphOrchestrator::buildGraph()` → `Qwen2Graph::build()` → `RoPEStage::Params::hybrid_mode = true`

2. **Verify Q_rope/K_rope buffers used by attention:**
   - Attention stage should read from `Q_rope`/`K_rope` (FP32) not `Q`/`K` (Q8_1)

3. **Re-run E2E parity tests:**
   - Target: `layer0_Q_ROPE` cosine ≥0.9995 vs FP32
   - Target: `layer0_ATTENTION_CONTEXT` cosine ≥0.99 vs FP32

---

## Appendix A: File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `RuntimeConfig.h` | Modify | Add `Hybrid` to `ActivationPrecision` enum |
| `RuntimeConfig.cpp` | Modify | Add "hybrid" string parsing |
| `ArgumentParser.cpp` | Modify | Update help text |
| `HybridPrecisionConfig.h` | **New** | Per-buffer precision configuration |
| `HybridPrecisionConfig.cpp` | **New** | Implementation |
| `GraphOrchestrator.h` | Modify | Add Q_rope/K_rope to InferenceState |
| `GraphOrchestrator.cpp` | Modify | Hybrid buffer allocation logic |
| `TensorFactory.h` | Modify | Add `createHybridBuffers()` helper |
| `CPURoPEKernelT.h` | Modify | Add `apply_q8_1_to_fp32()` |
| `CPURoPEKernelT.cpp` | Modify | Implement Q8_1→FP32 path |
| `RoPEPrimitives.h` | Modify | Add `apply_rope_q8_1_to_fp32()` |
| `RoPEPrimitives.cpp` | Modify | Implement primitive |
| `ComputeStage.h` | Modify | Add precision params to RoPEStage |
| `ComputeStage.cpp` | Modify | RoPE/Attention stage precision handling |
| `FusedAttentionWoKernel.h` | Modify | Support FP32 Q/K inputs |
| `JitFusedAttentionWo.cpp` | Modify | On-the-fly Q quantization |
| `UnifiedKVCache.h/cpp` | Verify | BF16 cache already exists |
| `Qwen2Graph.cpp` | Modify | Hybrid graph construction |
| `tests/v2/unit/Test__HybridPrecisionConfig.cpp` | **New** | Unit tests |
| `tests/v2/unit/Test__RoPEQ8ToFP32.cpp` | **New** | Unit tests |
| `tests/v2/e2e/Test__HybridParity.cpp` | **New** | E2E parity tests |

---

## Appendix B: Precision Flow Diagrams

### Current Q8_1 Flow (Problem)

```
Residual (FP32)
    │
    ▼
┌─────────────┐
│  RMS Norm   │ FP32
└─────────────┘
    │
    ▼
┌─────────────┐
│  QKV GEMM   │ FP32 → Q8_1 (quantize)
└─────────────┘
    │
    ▼
┌─────────────┐
│    RoPE     │ Q8_1 → dequant → FP32 → requant → Q8_1  ❌ UNNECESSARY
└─────────────┘
    │
    ▼
┌─────────────┐
│  Attention  │ Q8_1 × Q8_1 → FP32 context
└─────────────┘
    │
    ▼
┌─────────────┐
│  Wo Proj    │ FP32 → Q8_1 (quantize) → VNNI GEMM  ❌ UNNECESSARY
└─────────────┘
    │
    ▼
Residual Add (FP32)
```

### Hybrid Flow (Solution)

```
Residual (FP32)
    │
    ▼
┌─────────────┐
│  RMS Norm   │ FP32
└─────────────┘
    │
    ▼
┌─────────────┐
│  QKV GEMM   │ FP32 → Q8_1 (quantize) ✅ OK
└─────────────┘
    │
    ▼
┌─────────────┐
│    RoPE     │ Q8_1 → dequant → FP32 (STOP - no requant!) ✅ FIXED
└─────────────┘
    │
    ▼
┌─────────────┐     ┌──────────┐
│  Attention  │◄────│ BF16 KV  │ BF16 dequant → FP32 K/V
│             │     │  Cache   │
└─────────────┘     └──────────┘
    │ FP32 context
    ▼
┌─────────────┐
│  Wo Proj    │ FP32 context × FP32 Wo → FP32 ✅ NO QUANTIZATION
└─────────────┘
    │
    ▼
Residual Add (FP32)
```

---

## Appendix C: Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| ActivationPrecision enum | `RuntimeConfig.h` | 140-147 |
| Buffer allocation | `GraphOrchestrator.cpp` | 780-910 |
| RoPE Q8_1 kernel | `CPURoPEKernelT.h` | 361-430 |
| RoPE Q8_1 primitive | `RoPEPrimitives.cpp` | 1677-1780 |
| RoPEStage execute | `ComputeStage.cpp` | 1611-1730 |
| FusedAttentionWoKernel | `FusedAttentionWoKernel.h` | 100-260 |
| FusedAttentionWoStage | `ComputeStage.cpp` | 3950-4100 |
| FusedQKVGEMMStage | `ComputeStage.cpp` | 918-1050 |
| KV cache creation | `UnifiedKVCache.cpp` | 796-840 |

---

*Document Version: 1.0*  
*Created: 2025-01-XX*  
*Last Updated: 2025-01-XX*
