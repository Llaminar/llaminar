# Fusion Framework Migration Plan

**Author**: David Sanftenberg  
**Date**: 2025-11-22  
**Updated**: 2025-11-24  
**Status**: Phase 2 Complete - Reworking Strategy

## Executive Summary

This document outlines the migration plan for implementing a **lightweight operator fusion framework** in Llaminar V2 to reduce memory bandwidth and improve inference performance. 

### Strategy Pivot (2025-11-24)

**Phase 2 Lessons Learned**:
- ❌ Flat INT8 activations across the pipeline caused unacceptable error accumulation
- ❌ Repeated weight dequantization was performance-hostile
- ✅ The new `Q8_1GemmKernel` JIT kernel solves both problems

**New Strategy: Hybrid FP32 Residual + Q8_1 Compute**

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| **Residual Stream** | FP32 | Preserves dynamic range, prevents error accumulation |
| **GEMM Activations** | Q8_1 (on-the-fly) | Quantized inside kernel, no separate step |
| **GEMM Weights** | Q8_1 (stored) | Direct consumption, no dequantization |
| **Attention Scores** | FP32 (fused softmax) | Kernel computes stats without materializing |
| **RMSNorm/SwiGLU** | FP32 | Low-cost element-wise ops |

**Why Not Q16_1 (Custom 16-bit Block)?**

A 16-bit integer format was considered but rejected:
- No VNNI hardware support (INT16 requires FP32 accumulation anyway)
- Doubles memory bandwidth vs Q8_1 with no throughput gain
- FP32 residuals still needed for stability (dynamic range accumulates with depth)

**New Expected Performance Impact**:
- **GEMM**: Q8_1 JIT kernel with VNNI (4x throughput vs FP32)
- **Fused Softmax**: 1.33x speedup for M=1 (single-token decode)
- **Fused Bias/Mask**: Eliminates post-GEMM memory round-trips
- **Target Speedup**: 20-40% on CPU inference

---

## Problem Statement

### Original Problem: Quantization Overhead (Obsolete)

The original plan focused on reducing quantization/dequantization round-trips. This is now **largely solved** by the Q8_1 JIT kernel which:
1. Quantizes FP32 activations on-the-fly inside the GEMM
2. Consumes Q8_1 weights directly (no dequantization)
3. Outputs FP32 results (with optional fused bias/mask/softmax)

### Current Problem: Memory Bandwidth

The new bottleneck is **memory bandwidth**, not quantization CPU cycles:

| Operation | Read | Write | Notes |
|-----------|------|-------|-------|
| GEMM (Q8_1) | A (FP32) + B (Q8_1) | C (FP32) | A quantized on-the-fly |
| Softmax | C (FP32) | Probs (FP32) | Materializes attention scores |
| Bias | C (FP32) | C (FP32) | Extra read/write pass |
| Mask | C (FP32) | C (FP32) | Extra read/write pass |

**Fused Solution** (already implemented in `Q8_1GemmKernel`):
- Bias/Mask applied inside GEMM kernel → no extra pass
- Softmax statistics computed during GEMM → scores never written to memory
- **Bandwidth Savings**: Up to 2× for attention score computation

---

## Design Overview

### Core Architecture: FP32 Residual + Q8_1 GEMM

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER N                                      │
├─────────────────────────────────────────────────────────────────┤
│  [FP32 Residual] ──┬── RMSNorm ──► [FP32 Normalized]            │
│                    │                       │                     │
│                    │    ┌──────────────────┴─────────────────┐   │
│                    │    │  Q8_1 GEMM (JIT)                   │   │
│                    │    │  • FP32→Q8_1 quantization on-fly   │   │
│                    │    │  • Q8_1 weights (no dequant)       │   │
│                    │    │  • Fused bias/mask/softmax         │   │
│                    │    │  • Output: FP32                    │   │
│                    │    └────────────────────────────────────┘   │
│                    │                       │                     │
│                    │    [FP32 Attention/FFN Output]              │
│                    │                       │                     │
│                    └─────── (+) ◄──────────┘                     │
│                              │                                   │
│                    [FP32 Residual] ──► LAYER N+1                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **FP32 Residual Stream**
   - Residuals accumulate across 24+ layers
   - INT8 would require rescaling at every layer (expensive + error-prone)
   - FP32 provides ~7 digits of precision, sufficient for deep networks

2. **On-the-fly Activation Quantization**
   - Q8_1GemmKernel quantizes FP32 activations inside the kernel
   - Per-row scaling captures local dynamic range
   - Block size = 32 elements (matches VNNI vector width)

3. **Direct Quantised Weight Consumption**
   - Weights stored in Quantised format (from GGUF)
   - VNNI `vpdpbusd` operates on INT8 directly

4. **Fused Post-Ops**
   - Bias, mask, softmax computed during GEMM output phase
   - Results never written to memory in intermediate form
   - Critical for attention score computation

---

## Migration Phases (Revised)

### **Phase 1: Q8_1 JIT Kernel Foundation** ✅ COMPLETE

**Goal**: High-performance INT8 GEMM with VNNI.

**Delivered**:
- `QuantisedGemmKernel` with adaptive blocking (0.5B - 32B models)
- `QuantisedGemmJit_M1` / `QuantisedGemmJit_M2` specialized kernels
- K-Tiling for large models (prevents L2 thrashing)
- **Performance**: ~2100 GFLOPS on Qwen 32B FFN Down

---

### **Phase 2: Fused GEMM Operations** ✅ COMPLETE

**Goal**: Eliminate post-GEMM memory round-trips.

**Delivered**:
- `multiply_fused()` API with bias, mask, softmax parameters
- Fused Online Softmax (computes max/sum during output)
- **Performance**: 1.33x speedup for M=1 (single-token decode)

**Benchmark Results** (N=4096, K=4096):
| Batch Size (M) | Baseline (ms) | Fused (ms) | Speedup |
|----------------|---------------|------------|---------|
| M=1 | 0.109 | 0.082 | **1.33x** |
| M=2 | 0.110 | 0.088 | **1.25x** |
| M=32 | 0.566 | 0.516 | **1.10x** |

---

### **Phase 3: Pipeline Integration** 🔄 IN PROGRESS

**Goal**: Wire fused kernels into `Qwen2Pipeline`.

**Status Update (2025-11-24)**:
- ✅ Fused kernel headers updated to FP32 API (FusedDualGEMM, FusedTripleGEMM, FusedSwiGLU)
- ✅ Kernel implementations updated to use Q8_1GemmKernel
- ✅ Factory methods updated with proper Q8_1Tensor casting
- ⏳ Pipeline integration pending (old INT32 path disabled via `#if 0`)
- ⏳ New unit tests for FP32 fused kernels pending

See `changelog/2025-11-24-fused-kernel-fp32-migration-phase1.md` for details.

#### 3.1 Attention Block Integration
```cpp
// BEFORE: Separate GEMM + Softmax
auto scores = gemm(Q, K.T());          // Writes [M, seq_len]
auto probs = softmax(scores);          // Reads + Writes [M, seq_len]

// AFTER: Fused GEMM with Online Softmax
float* local_max, local_sum;           // Softmax stats only
gemm_fused(Q, K.T(), C, mask,          // C optionally skipped
           /*do_softmax=*/true,
           local_max, local_sum);
// Softmax normalization applied during V multiplication
```

#### 3.2 FFN Block Integration
```cpp
// BEFORE: Separate GEMM + Bias passes
auto gate = gemm(x, W_gate);           // Writes [M, hidden]
auto up = gemm(x, W_up);               // Writes [M, hidden]
gate = gate + b_gate;                  // Read + Write
up = up + b_up;                        // Read + Write

// AFTER: Fused GEMM with Bias
gemm_fused(x, W_gate, gate, b_gate);   // Bias applied during output
gemm_fused(x, W_up, up, b_up);         // No extra memory pass
```

**Tasks**:
1. ⬜ Update `Qwen2Pipeline::attention_block()` to use `multiply_fused()`
2. ⬜ Update `Qwen2Pipeline::ffn_block()` to use `multiply_fused()`
3. ⬜ Add configuration flags for fusion enable/disable
4. ⬜ E2E parity validation with adjusted tolerances

**Success Criteria**:
- E2E parity tests pass (tolerance ≤ 5% degradation)
- End-to-end latency reduction ≥ 15%

---

### **Phase 4: FlashAttention-Style Tiling** ⬜ PLANNED

**Goal**: True online attention without materializing scores.

**Concept**: Instead of computing full attention scores and then applying softmax, compute attention in tiles:

```
For each tile of K, V:
    1. Compute partial Q @ K.T scores
    2. Update running softmax (max, sum)
    3. Rescale previous V accumulator
    4. Add new V contribution
Output: Attention context without ever storing full [M, seq_len] scores
```

**Benefit**: Memory usage O(M * head_dim) instead of O(M * seq_len).

**Tasks**:
1. ⬜ Implement `FlashAttentionKernel` with tiled computation
2. ⬜ Integrate with `CpuAttentionKernelT`
3. ⬜ Support variable sequence lengths (KV cache)

**Success Criteria**:
- Long-context support (32K+ tokens) without OOM
- Performance parity with current implementation for short contexts

---

### **Phase 5: Advanced Optimizations** ⬜ PLANNED

**Goal**: Additional memory and compute optimizations.

#### 5.1 Shared Activation Quantization
```cpp
// Current: Quantize activation 3x for Q, K, V projections
q8_1_quant(x) → gemm(x_q8, W_q)
q8_1_quant(x) → gemm(x_q8, W_k)
q8_1_quant(x) → gemm(x_q8, W_v)

// Optimized: Quantize once, reuse for all projections
auto x_q8 = q8_1_quant(x);  // Single quantization pass
gemm(x_q8, W_q);
gemm(x_q8, W_k);
gemm(x_q8, W_v);
```

**Savings**: 2 quantization passes per layer × 24 layers = 48 ops/forward

#### 5.2 Fused RMSNorm → Quantization
```cpp
// Current: RMSNorm → FP32 → GEMM (internal quantization)
// Optimized: RMSNorm → Q8_1 directly (skip FP32 intermediate)
```

**Savings**: 1 FP32 buffer per normalization

#### 5.4 BF16 Residual Stream

**Goal**: Reduce memory bandwidth for residual connections by using BF16 storage.

**Current State**: FP32 residuals (4 bytes per element)
**Proposed**: BF16 residuals (2 bytes per element) with FP32 accumulation

```
┌─────────────────────────────────────────────────────────────────┐
│  [BF16 Residual] ─── (cvt to FP32) ─── RMSNorm ─► [FP32 Norm]   │
│                                                       │         │
│                          Q8_1GemmKernel (FP32 output) │         │
│                                                       ▼         │
│                                                 [FP32 Output]   │
│                                                       │         │
│  [BF16 Residual] ◄── (cvt to BF16) ── (+) ◄──────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
- **FP32 Addition**: `residual_fp32 = cvt(residual_bf16) + output_fp32`
- **BF16 Storage**: `residual_bf16 = cvt(residual_fp32)` (only storage is reduced-precision)
- **Conversion Ops**: Use AVX512-BF16 `vcvtne2ps2bf16` / `vcvtneps2bf16` (near-zero overhead)

**Memory Impact** (4096-dim model, seq_len=2048):
| Precision | Residual Size | L3 Cache Fit |
|-----------|---------------|--------------|
| FP32 | 32 MB/batch | Often spills |
| BF16 | 16 MB/batch | Usually fits |

**Precision Analysis**:
- BF16: 8-bit exponent (same range as FP32) + 7-bit mantissa (~2.5 decimal digits)
- Residual = Σ GEMM outputs across layers → errors are random, tend to cancel
- Production systems (vLLM, TensorRT-LLM) successfully use BF16/FP16 throughout

**Risks**:
- Accumulation error over 24-72 layers (mitigated by FP32 addition)
- Conversion overhead (mitigated by AVX512-BF16 hardware support)

**Tasks**:
1. ⬜ Add `BF16Tensor` residual buffer option to `Qwen2Pipeline`
2. ⬜ Implement BF16↔FP32 conversion in residual add kernel
3. ⬜ Add E2E parity tests with BF16 residuals
4. ⬜ Benchmark memory bandwidth improvement

**Success Criteria**:
- E2E parity within 1% of FP32 residual baseline
- ≥30% memory bandwidth reduction for residual operations
- No measurable latency regression from conversions

**Prerequisites**: Phase 3 (FP32 fused path) must be complete first

#### 5.3 INT8 KV Cache (Experimental)
Store K/V cache in Q8_1 format to reduce memory footprint.
- **Risk**: Accuracy degradation over long sequences
- **Mitigation**: Per-head or per-layer calibration

---

## Technical Design Details

### Precision Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ PRECISION FLOW: Hybrid FP32 Residual + Q8_1 Compute               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Token Embedding ──────────────────────────────────────────────►   │
│       [FP32]                                                       │
│          │                                                         │
│          ▼                                                         │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │ LAYER 0                                                    │    │
│  │                                                            │    │
│  │  [FP32] ── RMSNorm ──► [FP32] ── Q8_1 GEMM ──► [FP32]     │    │
│  │    │                              ▲                   │     │    │
│  │    │                              │                   │     │    │
│  │    │                        On-the-fly             Fused    │    │
│  │    │                        quantization           bias/    │    │
│  │    │                        (inside kernel)        mask     │    │
│  │    │                                                  │     │    │
│  │    └──────────────────── (+) ◄────────────────────────┘     │    │
│  │                           │                                 │    │
│  │                    [FP32 Residual]                          │    │
│  └───────────────────────────┼─────────────────────────────────┘    │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │ LAYER 1 ... LAYER N-1                                      │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  Final RMSNorm ── LM Head ──► Logits [FP32]                        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Q8_1 Block Structure

```cpp
struct Q8_1Block {
    int8_t qs[32];    // 32 quantized values
    fp16 d;           // Scale factor (max/127)
    int16_t sum_qs;   // Sum of qs (for zero-point compensation)
};
// Size: 36 bytes per 32 elements = 1.125 bytes/element
```

### GEMM Kernel Signature

```cpp
bool multiply_fused(
    const float* A,           // [M, K] FP32 activations
    float* C,                 // [M, N] FP32 output
    int m, int n, int k,
    const float* bias,        // [N] optional, fused addition
    const float* mask,        // [M, N] optional, fused addition (ALiBi/causal)
    bool do_softmax,          // Compute max/sum stats instead of raw output
    float* local_max,         // [M * (N/64)] output: per-block max
    float* local_sum,         // [M * (N/64)] output: per-block sum(exp)
    bool accumulate,          // C = alpha * A @ B + beta * C
    float alpha, float beta,
    const MPIContext* ctx,
    int device_idx
);
```

### Alternative Considered: Q16_1 Format

A 16-bit block format was evaluated but rejected:

```cpp
// Hypothetical Q16_1Block (NOT IMPLEMENTED)
struct Q16_1Block {
    int16_t qs[32];   // 32 quantized values
    float d;          // Scale factor
    int32_t sum_qs;   // Sum of qs
};
// Size: 72 bytes per 32 elements = 2.25 bytes/element
```

**Why Rejected**:
| Factor | Q8_1 | Q16_1 |
|--------|------|-------|
| Memory BW | 1.125 B/elem | 2.25 B/elem (2x worse) |
| VNNI Support | ✅ `vpdpbusd` INT8→INT32 | ❌ No INT16 dot product |
| Accumulation | INT32 (sufficient) | Needs FP32 anyway |
| Dynamic Range | ±127 per block | ±32767 per block (unused) |

The INT8 dynamic range of ±127 is sufficient when:
1. Activations are per-row quantized (captures local range)
2. Residual stream stays in FP32 (accumulates global range)
3. Weights are pre-quantized with proper calibration

---

## Risk Assessment

### Accuracy Risks

| Component | Risk Level | Mitigation |
|-----------|-----------|------------|
| FP32 Residual Stream | **None** | Standard practice, full precision |
| Q8_1 GEMM (per-row quant) | **Low** | Local dynamic range capture, proven in Phase 2 |
| Fused Softmax (fast exp) | **Low** | <5% relative error, validated in unit tests |
| Fused Bias/Mask | **None** | Exact computation, just reordered |
| FlashAttention Tiling | **Medium** | Numerical stability requires careful rescaling |
| INT8 KV Cache | **High** | Accuracy degradation over long sequences |

### Lessons Learned from Phase 2

| Issue | Impact | Resolution |
|-------|--------|------------|
| Flat INT8 activations | Severe error accumulation | Switched to FP32 residual stream |
| Repeated weight dequant | 30% slowdown | Q8_1 kernel consumes weights directly |
| Per-tensor INT8 scaling | Dynamic range overflow | Per-row (per-token) scaling in Q8_1 |

### Performance Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fusion overhead | Low | Benchmarked: fused path is always faster |
| Memory fragmentation | Low | Pre-allocated buffers in pipeline |
| VNNI availability | Medium | Runtime feature detection, FP32 fallback |

---

## Validation Strategy

### Unit Tests
- Per-kernel correctness tests
- Fused kernel vs separate kernel parity
- SIMD variant testing (AVX512/AVX2/scalar)

### Integration Tests
- Pipeline-level fusion integration
- Multi-device compatibility
- Batch size variations

### E2E Parity Tests
- Compare against PyTorch reference
- Tolerance adjustments expected (quantization accumulation)
- Target: ≤20% relative L2 degradation

### Performance Benchmarks
- Per-fusion microbenchmarks
- Layer-level benchmarks
- End-to-end inference benchmarks
- Comparison: baseline vs Phase 1 vs Phase 2 vs Phase 3+4

---

## Success Metrics

### Performance Targets (Revised)

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1 (Q8_1 JIT) | >1500 GFLOPS on Qwen 7B+ | ✅ 2100 GFLOPS |
| Phase 2 (Fused Ops) | 1.2x+ speedup M=1 | ✅ 1.33x achieved |
| Phase 3 (Pipeline) | 15% E2E latency reduction | 🔄 In progress |
| Phase 4 (FlashAttn) | O(M) memory for attention | ⬜ Planned |
| Phase 5 (Optimizations) | 25% cumulative speedup | ⬜ Planned |

### Accuracy Targets

| Metric | Target | Current |
|--------|--------|---------|
| E2E Max Divergence | ≤ 1e-3 | TBD (Phase 3) |
| Per-Layer RMSE | ≤ 1e-4 | ✅ Passing |
| Softmax Error | ≤ 5% relative | ✅ <5% |

### Code Quality Targets
- All fused kernels have unit tests (100% coverage)
- E2E tests pass with adjusted tolerances
- No performance regressions on non-fused paths
- Clean separation: fusion logic encapsulated in kernel

---

## Timeline (Revised)

| Phase | Duration | Status | Completion |
|-------|----------|--------|------------|
| Phase 1 | 2 weeks | ✅ Complete | 2025-11-20 |
| Phase 2 | 2 weeks | ✅ Complete | 2025-11-24 |
| Phase 3 | 2 weeks | 🔄 In Progress | 2025-12-08 |
| Phase 4 | 3 weeks | ⬜ Planned | 2025-12-29 |
| Phase 5 | 2 weeks | ⬜ Planned | 2026-01-12 |

**Total**: ~11 weeks (remaining: ~7 weeks)

---

## Open Questions

1. **FlashAttention Implementation**
   - Use tiled online softmax or full FlashAttention algorithm?
   - How to handle KV cache with tiled approach?

2. **Shared Activation Quantization**
   - Cache quantized activations for Q/K/V projections?
   - Memory vs compute tradeoff (extra buffer vs 2 quant passes)

3. **INT8 KV Cache**
   - Worth the accuracy risk for memory savings?
   - Per-head or per-layer calibration needed?

4. **Backend Portability**
   - How to expose fused ops to CUDA/ROCm backends?
   - OneDNN post-ops integration for Intel GPUs?

---

## References

- **Q8_1 GEMM Architecture**: `src/v2/kernels/cpu/gemm_v4/Q8_1-README.md`
- **Performance Results**: `src/v2/kernels/cpu/gemm_v4/PERF-README.md`
- **FlashAttention Paper**: https://arxiv.org/abs/2205.14135
- **Online Softmax**: https://arxiv.org/abs/1805.02867

---

## Appendix: Code Locations

### Phase 1-2 Deliverables (Complete)
- `src/v2/kernels/cpu/gemm_v4/Q8_1GemmKernel.h` - Main orchestrator
- `src/v2/kernels/cpu/gemm_v4/Q8_1GemmJit_M1.h` - M=1 JIT kernel
- `src/v2/kernels/cpu/gemm_v4/Q8_1GemmJit_M2.h` - M=2 JIT kernel (with fused ops)
- `tests/v2/unit/kernels/gemm/Test__Q8_1GemmFused.cpp` - Fused op unit tests
- `tests/v2/performance/Perf__Q8_1_GEMM_Fused.cpp` - Performance benchmarks

### Phase 3 Integration Points
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - attention_block, ffn_block
- `src/v2/attention/GQAAttention.h` - Attention kernel selection

### Phase 4-5 (Planned)
- `src/v2/kernels/cpu/FlashAttentionKernel.h` - Tiled attention
- `src/v2/kernels/cpu/fused/FusedRMSNormQuantize.h` - Fused normalization

---

**Status Updates**:
- 2025-11-22: Migration plan created, Phase 1 initiated
- 2025-11-24: Phase 2 complete (fused bias/mask/softmax), strategy pivot documented
- Next: Phase 3 pipeline integration
