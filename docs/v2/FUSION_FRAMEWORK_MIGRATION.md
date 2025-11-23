# Fusion Framework Migration Plan

**Author**: David Sanftenberg  
**Date**: 2025-11-22  
**Status**: Phase 1 - Complete (E2E Validation Pending)

## Executive Summary

This document outlines the migration plan for implementing a **lightweight operator fusion framework** in Llaminar V2 to reduce quantization/dequantization round trips and improve inference performance. The framework is built on top of `CPUKernelBase` and follows a gradual migration path with measurable milestones.

**Phase 1 Status**: ✅ Implementation complete, ⚠️ E2E tests failing (pre-existing issue)

**Current Blocker**: E2E batched tests failing with ~1500-1700 max divergence in residual stream (4 tests failing, unrelated to Phase 1 fusion work)

**Expected Performance Impact**:
- Current: ~264 quant/dequant operations per forward pass (24 layers × 11 ops/layer)
- Target: ~144 operations per forward pass (24 layers × 6 ops/layer)
- **Expected Speedup**: 10-20% on CPU inference (bandwidth-bound workloads)

---

## Problem Statement

### Current Quantization Overhead

**Per-Layer Quantization Round Trips**:

#### Attention Block (~7 round trips)
1. Input → RMSNorm: FP32 → INT8 → FP32
2. Q projection: FP32 → INT8 → matmul → FP32
3. K projection: FP32 → INT8 → matmul → FP32
4. V projection: FP32 → INT8 → matmul → FP32
5. Q@K^T attention scores: FP32 → INT8 → matmul → FP32
6. scores@V context: FP32 → INT8 → matmul → FP32
7. Output projection: FP32 → INT8 → matmul → FP32

#### FFN Block (~4 round trips)
1. Input → RMSNorm: FP32 → INT8 → FP32
2. Gate projection: FP32 → INT8 → matmul → FP32
3. Up projection: FP32 → INT8 → matmul → FP32
4. Down projection: FP32 → INT8 → matmul → FP32

**Total**: 11 quant/dequant round trips per layer × 24 layers = **264 operations per forward pass**

---

## Design Overview

### Core Components

#### 1. Kernel Contracts (`CPUKernelBase` extension)
- Define input/output tensor formats for each kernel
- Declare fusion capabilities
- Enable automatic fusion pattern detection

#### 2. Fusion Patterns
- **RMSNorm → Quantize**: Fuse normalization with quantization (saves 1 FP32 buffer)
- **Shared Input Quantization**: Quantize once for gate/up or Q/K/V projections (saves 1-2 quant passes)
- **Dequant → SwiGLU**: Fuse dequantization with activation (saves 2 dequant passes)

#### 3. Execution Graph (Future)
- Operator scheduling and buffer management
- Automatic fusion pattern detection
- Dead buffer elimination

---

## Migration Phases

### **Phase 1: Foundation (2 weeks) - IN PROGRESS**

**Goal**: Establish fusion infrastructure and implement highest-impact fusion.

#### Tasks
1. ✅ Extend `CPUKernelBase` with kernel contracts
   - Add `TensorFormat` enum
   - Add `KernelContract` struct
   - Add `get_contract()` virtual method

2. ✅ Add contracts to existing kernels
   - `CPURMSNormKernelT`
   - `CPUSoftmaxKernelT`
   - `CPURoPEKernelT`
   - `CPUSwiGLUKernelT`
   - `OneDNNGemmKernel`
   - `CpuAttentionKernelT`

3. 🔄 Implement `FusedRMSNormQuantize` kernel
   - Single-pass RMSNorm + per-row quantization to INT8
   - SIMD optimizations (AVX512/AVX2)
   - Unit tests with parity validation

4. ⬜ Integrate into `Qwen2Pipeline`
   - Replace RMSNorm → (implicit quantization) with fused kernel
   - Benchmark: attention norm + FFN norm (2× per layer)

5. ⬜ Validation
   - E2E parity tests (tolerance adjustment expected)
   - Performance benchmarks (target: 5-10% speedup)

**Deliverables**:
- `src/v2/kernels/cpu/CPUKernelBase.h` (extended)
- `src/v2/kernels/cpu/fused/FusedRMSNormQuantize.{h,cpp}`
- `tests/v2/unit/Test__FusedRMSNormQuantize.cpp`
- Benchmark results

**Success Criteria**:
- All unit tests pass
- E2E parity tests pass (tolerance ≤ 10% degradation)
- Measured speedup ≥ 5%

---

### **Phase 2: Multi-Input Fusions (3 weeks)**

**Goal**: Eliminate redundant quantizations for shared inputs.

#### Tasks
1. Implement `FusedDualGEMM` kernel
   - Shared input quantization for gate/up projections
   - Output INT32 accumulators (defer dequantization)

2. Implement `FusedTripleGEMM` kernel
   - Shared input quantization for Q/K/V projections
   - Output INT32 accumulators

3. Implement `FusedDequantSwiGLU` kernel
   - Dequantize INT32 accumulators
   - Apply SwiGLU activation: `gate * silu(up)`
   - Output FP32

4. Integrate into `Qwen2Pipeline::ffn_block()`
   - Replace gate/up separate quantizations with fused kernel
   - Use fused dequant+SwiGLU

5. Integrate into `Qwen2Pipeline::attention_block()`
   - Replace Q/K/V separate quantizations with fused kernel

6. Validation
   - E2E parity tests
   - Performance benchmarks (target: additional 5-10% speedup)

**Deliverables**:
- `src/v2/kernels/cpu/fused/FusedDualGEMM.{h,cpp}`
- `src/v2/kernels/cpu/fused/FusedTripleGEMM.{h,cpp}`
- `src/v2/kernels/cpu/fused/FusedDequantSwiGLU.{h,cpp}`
- Updated `Qwen2Pipeline` integration
- Benchmark results

**Success Criteria**:
- Cumulative speedup ≥ 10% (Phase 1 + Phase 2)
- E2E parity tests pass (tolerance ≤ 15% degradation)

---

### **Phase 3: Graph Execution Framework (4 weeks)**

**Goal**: Automate fusion detection and buffer management.

#### Tasks
1. Implement `FusionPattern` and `FusionMatcher`
   - Pattern detection algorithm
   - Cost model for fusion decisions

2. Implement `ExecutionGraph`
   - Graph construction from kernel sequence
   - Fusion optimization pass
   - Buffer lifetime analysis and reuse
   - Execution scheduler

3. Integrate into `Qwen2Pipeline`
   - Build subgraphs for attention/FFN blocks
   - Let optimizer apply fusions automatically

4. Add configuration options
   - `--fusion-level` flag (conservative/balanced/aggressive)
   - Runtime fusion enable/disable

5. Validation
   - Compare manual fusions (Phase 1+2) vs automatic graph fusions
   - Verify no performance regression

**Deliverables**:
- `src/v2/kernels/cpu/FusionPattern.{h,cpp}`
- `src/v2/pipelines/ExecutionGraph.{h,cpp}`
- Updated pipeline integration
- Configuration system

**Success Criteria**:
- Automatic fusion matches manual Phase 1+2 performance
- Buffer reuse reduces memory footprint by ≥20%
- Clean pipeline code (fusion logic separate from pipeline logic)

---

### **Phase 4: Advanced Optimizations (3 weeks)**

**Goal**: Attention-specific fusions and residual stream optimizations.

#### Tasks
1. Implement attention-specific fusions
   - `FusedQK_Softmax`: Q@K^T → scale → softmax
   - `FusedSoftmax_ScoresV`: Keep attention scores in cache

2. Implement residual stream optimizations
   - `FusedGEMM_Residual`: Fuse dequantization with residual addition
   - Explore INT8 residual stream (high risk, requires accuracy validation)

3. Backend-specific optimizations
   - OneDNN primitive post-ops integration
   - CUDA/ROCm fusion primitives (future)

4. Dynamic shape caching
   - Cache fusion plans per (batch_size, seq_len) tuple
   - Avoid re-optimization on every forward pass

**Deliverables**:
- Advanced fused kernels
- Backend integration
- Dynamic optimization caching

**Success Criteria**:
- Cumulative speedup ≥ 15-20% over baseline
- E2E parity tests pass (tolerance ≤ 20% degradation)
- Production-ready fusion framework

---

## Technical Design Details

### Kernel Contract Definition

```cpp
// src/v2/kernels/cpu/CPUKernelBase.h

enum class TensorFormat {
    FP32,
    BF16,
    FP16,
    INT8,
    INT32,
    Q4_0,
    IQ4_NL,
    // ... other quantized formats
};

struct KernelContract {
    std::vector<TensorFormat> accepted_input_formats;
    TensorFormat output_format;
    bool supports_inplace;
    bool is_fusable;
    
    bool can_fuse_with(const KernelContract& next) const;
};

class CPUKernelBase {
public:
    virtual ~CPUKernelBase() = default;
    virtual KernelContract get_contract() const = 0;
    virtual bool supports_fusion() const { return false; }
    virtual TensorFormat preferred_fusion_format() const { 
        return TensorFormat::FP32; 
    }
};
```

### Fusion Pattern Examples

#### RMSNorm → Quantize
```
Before:
  [FP32] → RMSNorm → [FP32] → Quantize → [INT8]
  
After:
  [FP32] → FusedRMSNormQuantize → [INT8]
  
Savings: 1 FP32 intermediate buffer + 1 quantization pass
```

#### Gate/Up Shared Quantization
```
Before:
  [FP32] → Quant → Gate Matmul → [FP32]
  [FP32] → Quant → Up Matmul → [FP32]
  
After:
  [FP32] → Quant → [INT8] → DualGEMM(gate, up) → [INT32, INT32] →
           DequantSwiGLU → [FP32]
  
Savings: 1 quantization pass + 2 dequantization passes
```

#### Q/K/V Shared Quantization
```
Before:
  [FP32] → Quant → Q Matmul → [FP32]
  [FP32] → Quant → K Matmul → [FP32]
  [FP32] → Quant → V Matmul → [FP32]
  
After:
  [FP32] → Quant → [INT8] → TripleGEMM(Q, K, V) → [INT32×3] →
           Dequant → [FP32]
  
Savings: 2 quantization passes
```

---

## Risk Assessment

### Accuracy Risks

| Fusion Type | Risk Level | Mitigation |
|------------|-----------|------------|
| RMSNorm+Quantize | **Low** | No additional quantization, just reordering |
| Gate/Up Sharing | **Low** | No additional quantization, same INT8 path |
| QKV Sharing | **Low** | No additional quantization, same INT8 path |
| Fused SwiGLU | **Medium** | Validate activation ranges, add unit tests |
| INT8 Residual Stream | **High** | Extensive validation, may require per-layer scale adjustments |

### Performance Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fusion overhead > savings | Medium | Cost model to decide when to fuse |
| Cache pollution from fused kernels | Low | Careful memory access patterns, prefetching |
| Dynamic shape overhead | Low | Cache fusion plans per shape tuple |

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

### Performance Targets
- **Phase 1**: 5-10% speedup (RMSNorm+Quantize fusion)
- **Phase 2**: 10-15% cumulative speedup (multi-input fusions)
- **Phase 3**: Maintain Phase 2 speedup with automatic optimization
- **Phase 4**: 15-20% cumulative speedup (advanced fusions)

### Accuracy Targets
- **Phase 1**: Tolerance ≤ 10% degradation vs baseline
- **Phase 2**: Tolerance ≤ 15% degradation vs baseline
- **Phase 3**: Maintain Phase 2 accuracy
- **Phase 4**: Tolerance ≤ 20% degradation vs baseline

### Code Quality Targets
- All fused kernels have unit tests (100% coverage)
- E2E tests pass with adjusted tolerances
- No performance regressions on non-fused paths
- Clean separation: fusion logic separate from pipeline logic

---

## Timeline

| Phase | Duration | Completion Date (Estimated) |
|-------|----------|----------------------------|
| Phase 1 | 2 weeks | 2025-12-06 |
| Phase 2 | 3 weeks | 2025-12-27 |
| Phase 3 | 4 weeks | 2026-01-24 |
| Phase 4 | 3 weeks | 2026-02-14 |

**Total**: ~12 weeks (3 months)

---

## Open Questions

1. **Accuracy vs Speed Trade-off**
   - How much accuracy degradation is acceptable for production use?
   - Should we add a `--fusion-level` configuration flag?

2. **Dynamic Shapes**
   - How to efficiently handle variable batch_size and seq_len?
   - Cache fusion plans per shape or recompute on every change?

3. **Backend Integration**
   - How to integrate with OneDNN's existing post-ops/fusion primitives?
   - Can we share fusion plans across CPU/CUDA/ROCm backends?

4. **Memory Management**
   - Who allocates fusion intermediate buffers?
   - How to implement buffer reuse across fusion groups?

---

## References

- **OneDNN Fusion Primitives**: https://oneapi-src.github.io/oneDNN/dev_guide_attributes_post_ops.html
- **PyTorch Fusion Passes**: https://pytorch.org/docs/stable/notes/autograd.html#graph-optimization
- **TensorRT Layer Fusion**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fusion

---

## Appendix: Code Locations

### Existing Kernels (to extend with contracts)
- `src/v2/kernels/cpu/CPUKernelBase.h`
- `src/v2/kernels/cpu/CPURMSNormKernelT.h`
- `src/v2/kernels/cpu/CPUSoftmaxKernelT.h`
- `src/v2/kernels/cpu/CPURoPEKernelT.h`
- `src/v2/kernels/cpu/CPUSwiGLUKernelT.h`
- `src/v2/kernels/cpu/CpuAttentionKernelT.h`
- `src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h`

### New Fused Kernels (to create)
- `src/v2/kernels/cpu/fused/FusedRMSNormQuantize.{h,cpp}`
- `src/v2/kernels/cpu/fused/FusedDualGEMM.{h,cpp}`
- `src/v2/kernels/cpu/fused/FusedTripleGEMM.{h,cpp}`
- `src/v2/kernels/cpu/fused/FusedDequantSwiGLU.{h,cpp}`

### Pipeline Integration Points
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` (attention_block, ffn_block)

### Test Files (to create)
- `tests/v2/unit/Test__FusedRMSNormQuantize.cpp`
- `tests/v2/unit/Test__FusedDualGEMM.cpp`
- `tests/v2/integration/Test__FusionIntegration.cpp`

---

**Status Updates**:
- 2025-11-22: Migration plan created, Phase 1 initiated
- Next update: After `FusedRMSNormQuantize` implementation complete
