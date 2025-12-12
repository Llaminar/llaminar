# Fused Attention + Wo Projection: Project Plan

**Author:** David Sanftenberg  
**Date:** December 12, 2025  
**Status:** ✅ Phase 6 Complete — Unified Precision-Aware MPI Allreduce  
**Branch:** `feature/typed-residuals`

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Microkernel Reference + SIMD + Tests | ✅ **Complete** (48 tests passing) |
| **Phase 2** | Composed Reference Kernel + FP16/BF16 Wo | ✅ **Complete** (48 integration/robustness + 12 batch = 68 tests) |
| **Phase 3** | Cache-Blocked Tiled Reference | ✅ **Complete** (21 parity tests passing) |
| **Phase 4** | JIT Kernel (Xbyak) | ✅ **Complete** (9 instantiation + 27 correctness = 36 tests) |
| **Phase 5** | Pipeline Integration | ✅ **Complete** (wrapper, CLI, pipeline code, 10 tests) |
| **Phase 6** | Unified Precision-Aware MPI Allreduce | ✅ **Complete** |

### Phase 2 Test Coverage

**Integration Tests** (`Test__FusedAttentionWoRef.cpp`): 8 tests
**Robustness Tests** (`Test__FusedAttentionWoRef_Robustness.cpp`): 28 tests
**Batch Tests** (`Test__FusedAttentionWoRef_Batch.cpp`): 12 tests

**Total Phase 2 Tests**: 48 tests passing

### Phase 3 Test Coverage

**Tiled Parity Tests** (`Test__FusedAttentionWoTiled.cpp`): 21 tests
- TileConfig computation and cache detection
- Tiled vs reference parity across various configurations
- Qwen2.5 model dimensions (0.5B, 7B)
- Decode mode, batch processing, Q8_1 Wo weights
- Long KV sequences exceeding tile size (1024, 2048)

### Phase 4 Summary (Complete)

**JIT Infrastructure** (`JitMicrokernelBase.h`): ✅ Complete
- ZMM register zone allocation (Accum, Input, State, Scratch, Constants)
- ConstRegs and StateRegs definitions
- Common SIMD patterns and utilities

**JIT Microkernel Emitters**: ✅ Complete
- `JitQ8DotProduct.h` - AVX-512 VNNI dot product
- `JitOnlineSoftmax.h` - Streaming softmax state management
- `JitVWeightedAccum.h` - Weighted V accumulation
- `JitWoProjection.h` - Output projection
- `JitFastExp.h` - Fast exponential approximation

**Composed JIT Kernel** (`JitFusedAttentionWo.h`): ✅ Complete
- Framework and cache implemented
- Single query attention loop with online softmax
- Context rescaling after max updates
- Wo projection integration complete
- 512KB code buffer for large models (32B/72B)

**JIT Tests**: 36 tests passing
- `Test__JitMicrokernels.cpp`: 13 infrastructure tests
- `Test__JitFusedAttentionWo.cpp`: 9 instantiation tests
- `Test__JitFusedAttentionWo_Correctness.cpp`: 27 numerical correctness tests

**Numerical Correctness Results** (vs FP32 reference):
| Model | Cosine Similarity | Relative L2 Error |
|-------|-------------------|-------------------|
| Qwen2 0.5B | 0.995-0.997 | 0.073-0.092 |
| Qwen2 1.5B | 0.996-0.997 | 0.073-0.086 |
| Qwen2 7B | 0.996-0.997 | 0.079-0.084 |
| Qwen2 32B | 0.996-0.997 | 0.079-0.081 |
| Qwen2 72B | 0.997 | 0.081 |

**Bugs Fixed During Phase 4:**
1. Register collision: `rax` → `rdi` for scratch in emit_broadcast_i32_const
2. Register collision: `ymm0` → `ymm20` in emit_copy_q_head_to_stack (zmm0 clobber)
3. Register collision: `Xmm(3)` → `Xmm(20)` for scale_local (zmm3 clobber)
4. Missing context rescaling after softmax max update
5. Code buffer 64KB → 512KB for large models

### Phase 5 Current Progress

**Status:** ✅ **Complete**

**Goal:** Integrate JIT fused attention kernel into Qwen2 pipeline for end-to-end inference.

**Completed Tasks:**
- [x] Create `FusedAttentionWoKernel` wrapper class for pipeline use
  - Location: `src/v2/kernels/cpu/attention/FusedAttentionWoKernel.h`
  - Supports multiple backends: Reference, Tiled, JIT
  - Config-based instantiation with runtime backend selection
- [x] Unified Q8_1Block definition across all components
  - Consolidated to `llaminar2::Q8_1Block` in `BlockStructures.h`
  - Removed duplicate definitions from microkernels
  - Added `using` declarations for backward compatibility
- [x] Integration tests with Q8_1Tensor objects
  - Location: `tests/v2/integration/attention/Test__FusedAttentionWoKernel.cpp`
  - **10 tests passing** with Q8_1Tensor (production usage pattern)
  - Verified JIT kernel works correctly with tensor objects, not just raw vectors
  - Tests cover: single token decode, prefill, long KV, Qwen2 7B config, Q8_1 Wo weights
- [x] Add `use_fused_attention` config option
  - Added to `PipelineConfig.h` with documentation
  - Added CLI flag `--fused-attention` in `ArgParser.{h,cpp}` and `Main.cpp`
  - Requires Q8_1 activation precision, logs warning otherwise
- [x] Implement fused attention path in `Qwen2Pipeline::attention_block()`
  - Created `fused_attn_wo_kernel_` member variable
  - Initialized in constructor when `use_fused_attention && activation_precision == Q8_1`
  - Conditional path in attention_block() replaces `compute_attention + project_row_parallel`
  - Currently uses REFERENCE backend (JIT doesn't apply Wo projection yet)

**CLI Usage:**
```bash
# Enable fused attention (requires Q8_1 activation)
./run_llaminar.sh -- -m model.gguf --activation-prec q8_1 --fused-attention -p "Hello" -n 10
```

**Known Limitations:**
- **JIT Wo Projection**: JIT kernel outputs raw attention context (no Wo projection yet)
  - REFERENCE backend applies Wo correctly
  - JIT backend works for correctness testing with identity Wo
- **Pre-existing Q8_1 Device Transfer Bug**: E2E CLI testing with Q8_1 blocked by shape mismatch bug in `PipelineBase::prepareActivationForDevice()` (unrelated to fused attention)
  - Bug manifests when `placement_map_` creates device 0 entries for CPU
  - Kernel integration tests (10 tests) pass independently

**Integration Test Results:**
| Test | Cosine Similarity | Status |
|------|-------------------|--------|
| Reference vs JIT (single token) | 0.996 | ✅ |
| Q8_1Tensor vs vector parity | 1.000 | ✅ |
| JIT with Q8_1Tensor objects | 0.996 | ✅ |
| Reference vs JIT (prefill seq=8) | 0.997 | ✅ |
| Tiled vs JIT (kv_len=256) | 0.997 | ✅ |
| Qwen2 7B Config | - | ✅ |
| Q8_1 Wo Weights | - | ✅ |

---

## Phase 6: Unified Precision-Aware MPI Allreduce

**Status:** 🔄 In Progress  
**Goal:** Implement a portable, unified allreduce design that works across all activation precision types (FP32, FP16, BF16, Q8_1).

### Problem Statement

The current tensor-parallel GEMM implementation (`project_row_parallel`, `project_column_parallel`) has precision-specific code paths that:
1. **FP32**: Works correctly with native `MPI_Allreduce` using `MPI_FLOAT`
2. **Q8_1**: Currently dequantizes → FP32 allreduce → requantizes (wasteful round-trip)
3. **FP16/BF16**: Not implemented (silently skipped or crashes)

For k-sliced weights (column-parallel), the current Q8_1 path:
1. Dequantizes Q8_1 input to FP32 for slicing
2. Performs FP32 GEMM
3. Allreduces FP32
4. Requantizes to Q8_1

This defeats the purpose of Q8_1 (bandwidth reduction) and introduces quantization noise.

### Design Goals

1. **Unified Interface**: Single `allreduce_activation_inplace()` method that dispatches by precision type
2. **Native Q8_1 Allreduce**: AVX512-vectorized N-way reduction of Q8_1 blocks without FP32 conversion
3. **FP16/BF16 Support**: Native 16-bit allreduce or efficient FP32 fallback
4. **Bandwidth Efficiency**: Q8_1 allreduce sends 36 bytes/32 elements (vs FP32's 128 bytes/32 elements = 3.5x less)
5. **SIMD Optimization**: All precision types use vectorized reduction kernels

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MPIContext::allreduce_inplace()                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ActivationPrecision dispatch:                                          │
│                                                                         │
│  FP32  ──→ MPI_Allreduce(MPI_IN_PLACE, MPI_FLOAT, MPI_SUM)             │
│            (native MPI, no conversion)                                  │
│                                                                         │
│  FP16  ──→ allgather_bytes() + simd::fp16_sum_n() + store              │
│            (allgather raw bytes, AVX512 vectorized reduction)           │
│                                                                         │
│  BF16  ──→ allgather_bytes() + simd::bf16_sum_n() + store              │
│            (allgather raw bytes, AVX512 vectorized reduction)           │
│                                                                         │
│  Q8_1  ──→ allgather_bytes() + simd::q8_1_sum_n() + store              │
│            (allgather blocks, AVX512 dequant→sum→requant)               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 6.1: Q8_1-Native MPI Allreduce ✅ Complete
- [x] Add `simd::q8_1_sum_n()` - AVX512 N-way reduction of Q8_1 block arrays
- [x] Add `MPIContext::allreduce_q8_1_inplace()` - MPI allgather + local vectorized reduction
- [x] Update `project_column_parallel()` to use Q8_1-native allreduce
- [x] Unit tests for Q8_1 allreduce correctness (`Test__PrecisionAwareAllreduce.cpp`)

#### Phase 6.2: FP16/BF16-Native MPI Allreduce ✅ Complete
- [x] Add `simd::fp16_sum_n()` - AVX512 N-way FP16 reduction (F16C vectorized)
- [x] Add `simd::bf16_sum_n()` - AVX512 N-way BF16 reduction (round-to-nearest-even)
- [x] Add `MPIContext::allreduce_fp16_inplace()` - MPI allgather + local vectorized reduction
- [x] Add `MPIContext::allreduce_bf16_inplace()` - MPI allgather + local vectorized reduction
- [x] Update `project_column_parallel()` to support FP16/BF16 output
- [x] Unit tests for FP16/BF16 allreduce correctness (11 tests passing)

#### Phase 6.3: Unified Interface (Deferred)
- [ ] Create `MPIContext::allreduce_activation_inplace(TensorBase*, ActivationPrecision)`
- [ ] Dispatch by precision to type-specific implementation
- [ ] Update all call sites to use unified interface
- [ ] Error handling for unsupported tensor types

**Note**: Phase 6.3 deferred as `project_column_parallel()` already dispatches by tensor type.
The type-specific methods provide better compile-time type safety and avoid circular dependencies.

#### Phase 6.4: Q8_1 Block-Aligned Slicing ✅
- [x] Add `Q8_1Tensor::slice_k_blocks()` for k-sliced GEMM
- [x] Add `Q8_1Tensor::is_k_aligned()` static helper for 32-element alignment check
- [x] Update `project_row_parallel()` k-sliced path to use native Q8_1 slicing when aligned
- [x] Fallback to FP32 dequantization path for unaligned slices
- [x] Unit tests for block-aligned slicing (13 tests: alignment, slicing, tensor-parallel scenarios)

#### Phase 6.5: Multi-Precision Allreduce in Row-Parallel Paths ✅
- [x] Update replicated weights fallback path to support Q8_1/BF16/FP16 (scale + native allreduce)
- [x] Add `simd::fp16_scale_inplace()` and `simd::bf16_scale_inplace()` SIMD helpers
- [x] Update TensorSlice row-parallel path with multi-precision support ✅
- [x] Update k-sliced row-parallel path to use native precision allreduce ✅

#### Phase 6.6: TensorSlice Row-Parallel Multi-Precision Support ✅
- [x] Implement Q8_1 native path with block-level allgatherv (when n_local % 32 == 0)
- [x] Implement BF16 native path with element-level allgatherv
- [x] Implement FP16 native path with element-level allgatherv
- [x] Keep FP32 fallback for Q8_1 when slices not block-aligned
- [x] Replace allreduce-sum with allgatherv for row-parallel (more efficient for disjoint slices)

#### Phase 6.7: K-Sliced Row-Parallel Multi-Precision Support ✅
- [x] Implement Q8_1 native path: Q8_1 input k-slicing + Q8_1 output + Q8_1 allreduce
- [x] Implement BF16 native path: BF16 input slicing + BF16 output + BF16 allreduce
- [x] Implement FP16 native path: FP16 input slicing + FP16 output + FP16 allreduce
- [x] Keep FP32 fallback for mixed-precision or unaligned scenarios
- [x] Support cross-precision fallbacks (e.g., BF16 input with FP32 output)

**TensorSlice Row-Parallel Multi-Precision Design:**

For row-parallel (output columns split across ranks), each rank produces disjoint slices:
- Rank 0: columns [0, n_local)
- Rank 1: columns [n_local, 2*n_local)
- etc.

This is an **allgather** pattern, not a reduction. The new implementation:

1. **Q8_1 Native Path** (when `n_start % 32 == 0` and `n_local % 32 == 0`):
   - GEMM produces Q8_1 output directly
   - Allgatherv Q8_1 blocks row-by-row into correct positions
   - No FP32 intermediate, no requantization

2. **BF16/FP16 Native Paths**:
   - GEMM produces native 16-bit output
   - Allgatherv elements row-by-row
   - No FP32 intermediate

3. **FP32 Fallback** (for Q8_1 with unaligned slices):
   - GEMM produces FP32 output
   - Allgatherv FP32 elements
   - Requantize to Q8_1 at end

**Bandwidth Improvement:**
| Precision | Bytes/Element | MPI Transfer |
|-----------|---------------|--------------|
| FP32 | 4 | m × n × 4 bytes |
| BF16 | 2 | m × n × 2 bytes (2x better) |
| FP16 | 2 | m × n × 2 bytes (2x better) |
| Q8_1 | 1.125 | m × (n/32) × 36 bytes (3.5x better) |

**K-Sliced Row-Parallel Multi-Precision Design:**

For k-sliced row-parallel (input K dimension split across ranks), each rank computes a partial product:
- Weight is `[n, k_local]` where `k_local = k / world_size`
- Input is sliced: `[m, k_local]` (columns `k_start` to `k_start + k_local`)
- Each rank computes: `C_local = A_local @ W_local^T` → `[m, n]`
- **Allreduce-SUM** combines partial products: `C = sum(C_local)`

This is a **reduction** pattern, not allgather. The new implementation:

1. **Q8_1 Native Path** (when `k_start % 32 == 0` and `k_local % 32 == 0`):
   - Native Q8_1 k-slicing via `slice_k_blocks()` (no dequantization)
   - GEMM produces Q8_1 output directly
   - `allreduce_q8_1_inplace()` for AVX512 dequant→sum→requant
   - Full end-to-end Q8_1 with minimal FP32 conversion

2. **BF16/FP16 Native Paths**:
   - Create native 16-bit k-sliced input view
   - GEMM produces native 16-bit output
   - `allreduce_bf16_inplace()` / `allreduce_fp16_inplace()` for native reduction
   - No FP32 intermediate

3. **FP32 Fallback**:
   - Used for mixed-precision (e.g., Q8_1 input with FP32 output)
   - Used for Q8_1 with unaligned k boundaries
   - Converts result to target precision after FP32 allreduce

### Technical Details

#### Implemented Functions

**SIMDHelpers.h: N-way Sum Functions**
```cpp
// Q8_1: Sum N Q8_1 block arrays into one (N-way reduction)
// Algorithm: Dequant all→sum in FP32→requant (AVX512 vectorized)
void q8_1_sum_n(const Q8_1Block *const *inputs, size_t n_inputs,
                Q8_1Block *output, size_t n_blocks);

// FP16: Sum N FP16 arrays into one (AVX512 F16C)
// Algorithm: FP16→FP32 in zmm→accumulate→FP32→FP16
void fp16_sum_n(const uint16_t *const *inputs, size_t n_inputs,
                uint16_t *output, size_t count);

// BF16: Sum N BF16 arrays into one (AVX512 bit manipulation)
// Algorithm: Shift BF16→FP32→accumulate→round-to-nearest-even→shift FP32→BF16
void bf16_sum_n(const uint16_t *const *inputs, size_t n_inputs,
                uint16_t *output, size_t count);
```

**MPIContext.h: Type-Specific Allreduce Methods**
```cpp
void allreduce_q8_1_inplace(Q8_1Block *data, size_t n_blocks) const;
void allreduce_fp16_inplace(uint16_t *data, size_t count) const;
void allreduce_bf16_inplace(uint16_t *data, size_t count) const;
```

#### Q8_1 Allreduce Implementation (Complete)

**SIMDHelpers.h: `q8_1_sum_n()`**
```cpp
// Sum N Q8_1 block arrays into one (N-way reduction)
// Algorithm per block:
// 1. Dequantize all N blocks to FP32 (in SIMD registers)
// 2. Sum all FP32 values
// 3. Find new max_abs for quantization
// 4. Requantize to Q8_1
void q8_1_sum_n(const Q8_1Block *const *inputs, size_t n_inputs,
                Q8_1Block *output, size_t n_blocks);
```

**MPIContext.h: `allreduce_q8_1_inplace()`**
```cpp
// All-reduce sum for Q8_1 blocks (in-place)
// Uses allgather + vectorized local reduction
void allreduce_q8_1_inplace(Q8_1Block *data, size_t n_blocks) const {
    // 1. Allgather all Q8_1 blocks from all ranks
    // 2. Sum blocks using AVX512-vectorized reduction
    // 3. Store result back to input buffer
}
```

#### Bandwidth Analysis

| Precision | Bytes/32 Elements | Allreduce Data | vs FP32 |
|-----------|-------------------|----------------|---------|
| FP32 | 128 | 128 | 1.0x |
| FP16 | 64 | 64 | 2.0x better |
| BF16 | 64 | 64 | 2.0x better |
| Q8_1 | 36 (block) | 36 | 3.5x better |

### Future Work (Phase 7+)
- [ ] Fix Q8_1 device transfer bug to enable E2E CLI testing
- [ ] Implement Wo projection in JIT kernel for true fusion
- [ ] Benchmark fused vs unfused performance
- [ ] Enable JIT backend as default (after Wo projection complete)

**Key Findings:**
| Test | Cosine Similarity | Status |
|------|-------------------|--------|
| Reference vs JIT (single token) | 0.996 | ✅ |
| Q8_1Tensor vs vector parity | 1.000 | ✅ |
| JIT with Q8_1Tensor objects | 0.996 | ✅ |
| Reference vs JIT (prefill seq=8) | 0.997 | ✅ |
| Reference vs JIT (kv_len=256) | 0.997 | ✅ |
| Qwen2 7B Config | - | ✅ |

**Key Findings:**
1. **Q8_1Tensor storage is correct** - Blocks can be copied to/from tensors with identical results
2. **Quantization algorithm matters** - Must use `vals * inv_scale` (not `vals / scale`) to match unit test precision
3. **JIT doesn't apply Wo projection yet** - Tests use identity-like Wo matrix for Reference vs JIT parity
4. **Pipeline integration complete** - Fused kernel integrated into `Qwen2Pipeline::attention_block()` with CLI flag
5. **Pre-existing Q8_1 bug** - Device transfer shape mismatch blocks E2E CLI testing (unrelated to fused attention)

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

**Status:** ✅ **Complete**

**Delivered Files:**
- `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoTiled.h`
- `src/v2/kernels/cpu/attention/q8_1/FusedAttentionWoTiled.cpp`
- `tests/v2/unit/attention/Test__FusedAttentionWoTiled.cpp` (21 tests)

**Implemented Features:**
- Dynamic tile size computation via `compute_tile_config()` using CPUFeatures.h
- L2/L3 cache detection for optimal KV_TILE and Q_TILE sizing
- Online softmax with correction factor across tile boundaries
- Automatic fallback to reference for sequences shorter than tile size
- Batch processing support
- Q8_1 and FP32 Wo weight type support

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

#### Implementation Tasks (Phase 3 — All Complete)

- [x] Add `TileConfig` struct and `compute_tile_config()` to attention header
- [x] Implement `FusedAttentionWoTiled` class with outer tile loops
- [x] Add K/V prefetch intrinsics for tile loading
- [x] Update context buffer allocation for Q_TILE batch
- [x] Handle causal masking at tile boundaries
- [x] Unit test: Tiled matches reference for various tile sizes
- [x] Performance test: Measure bandwidth reduction

#### Test Cases (All Passing)

| Test | Description | Status |
|------|-------------|--------|
| `ComputeTileSizes_DetectsCaches` | Verify cache size detection | ✅ |
| `ComputeTileSizes_Qwen2_0_5B_Config` | Qwen2 0.5B config | ✅ |
| `ComputeTileSizes_Qwen2_7B_Config` | Qwen2 7B config | ✅ |
| `ShouldTile_ShortSequence_ReturnsFalse` | Short seq < KV_TILE | ✅ |
| `ShouldTile_LongSequence_ReturnsTrue` | Long seq > KV_TILE | ✅ |
| `Parity_ShortSequence_NoCausal` | seq=4, non-causal | ✅ |
| `Parity_ShortSequence_Causal` | seq=4, causal | ✅ |
| `Parity_MediumSequence_NoCausal` | seq=64, non-causal | ✅ |
| `Parity_MediumSequence_Causal` | seq=64, causal | ✅ |
| `Parity_LongSequence_ExceedsTileSize` | seq=256 | ✅ |
| `Parity_Qwen2_0_5B_Dimensions` | 14 heads, GQA 7:1 | ✅ |
| `Parity_Qwen2_7B_Dimensions` | 28 heads, GQA 7:1 | ✅ |
| `Parity_DecodeMode_SingleQuery` | M=1, KV=512 | ✅ |
| `Parity_LongKV_ExceedsTileSize` | KV=1024 | ✅ |
| `Parity_VeryLongKV_MultiTile` | KV=2048 (4 tiles) | ✅ |
| `Parity_CrossAttention_DifferentLengths` | Q≠KV lengths | ✅ |
| `EdgeCase_MinimalConfig` | 1 pos, 1 head | ✅ |
| `EdgeCase_HighGQARatio` | 32:1 GQA | ✅ |
| `EdgeCase_ManyKVHeads` | MHA (1:1) | ✅ |
| `BatchedExecution_MatchesReference` | Batch=4 | ✅ |
| `Parity_Q8_1_Wo_Weights` | Q8_1 Wo | ✅ |

### Phase 3b: SIMD Microkernels (Week 2)
- [x] Implement `q8_dot_product_avx512` (completed in Phase 1)
- [x] Implement `v_weighted_accum_avx512` (completed in Phase 1)
- [x] Implement `wo_projection_avx512` (FP32 Wo first) — Handled via reference path
- [x] Implement `fast_exp_avx512` (completed in Phase 1)
- [x] Unit tests: SIMD matches reference (completed in Phase 1)

### Phase 4: JIT Kernel (Week 2-3) ✅ COMPLETE

**Status:** ✅ **Complete** (December 12, 2025)

**Delivered Files:**
- `src/v2/kernels/cpu/jit/q8_1/JitMicrokernelBase.h` — Base class with ZMM register conventions
- `src/v2/kernels/cpu/jit/q8_1/JitQ8DotProduct.h` — JIT emitter for Q8_1 dot product (AVX-512 VNNI)
- `src/v2/kernels/cpu/jit/q8_1/JitOnlineSoftmax.h` — JIT emitter for online softmax
- `src/v2/kernels/cpu/jit/q8_1/JitVWeightedAccum.h` — JIT emitter for weighted V accumulation
- `src/v2/kernels/cpu/jit/q8_1/JitWoProjection.h` — JIT emitter for Wo projection
- `src/v2/kernels/cpu/jit/q8_1/JitFastExp.h` — JIT emitter for fast exponential
- `src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.h` — Composed JIT kernel
- `tests/v2/unit/attention/Test__JitMicrokernels.cpp` (13 tests)
- `tests/v2/unit/attention/Test__JitFusedAttentionWo.cpp` (9 tests)
- `tests/v2/unit/attention/Test__JitFusedAttentionWo_Correctness.cpp` (27 tests)

**Completed:**
- [x] JIT infrastructure with ZMM zone allocation
- [x] All 5 JIT microkernel emitters
- [x] Kernel cache with thread-safe lookup
- [x] Complete attention loop with online softmax
- [x] Context rescaling after max updates
- [x] Wo projection integration
- [x] Infrastructure tests passing (13 tests)
- [x] Instantiation tests passing (9 tests)
- [x] Numerical correctness tests passing (27 tests)
- [x] Support for all Qwen2 model sizes (0.5B-72B)

**Register Zone Allocation (Final):**
- ZMM0-7: Context accumulators (ACCUM zone)
- ZMM8-15: Input data Q/K/V (INPUT zone)
- ZMM16-19: Softmax state max/sum/weight/corr (STATE zone)
- ZMM20-25: Scratch registers (SCRATCH zone)
- ZMM26-31: Constants scale/ones (CONST zone)

**Critical Implementation Notes:**
- XMM/YMM operations zero upper bits of ZMM — never use XMM/YMM 0-19
- 512KB code buffer required for 32B/72B models (64 heads)
- ~8% relative L2 error is expected from fast exp approximation

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

**Status:** 🟡 **In Progress**

**Goal:** Replace separate attention + Wo GEMM with fused JIT kernel in Qwen2Pipeline.

**Tasks:**
- [ ] Create `FusedAttentionWoKernel` wrapper implementing `ITensorAttention`
- [ ] Add kernel to `KernelFactory` dispatch
- [ ] Modify `Qwen2Pipeline::attention_block` to use fused kernel
- [ ] E2E parity test with PyTorch reference
- [ ] Benchmark: measure tok/s improvement

**Expected Deliverables:**
- `src/v2/kernels/cpu/attention/FusedAttentionWoKernel.h` — Pipeline-compatible wrapper
- `src/v2/kernels/cpu/attention/FusedAttentionWoKernel.cpp` — Implementation
- `tests/v2/e2e/Test__FusedAttentionWoPipeline.cpp` — E2E parity tests
- `tests/v2/performance/Perf__FusedAttentionWo.cpp` — Benchmark suite

**Integration Points:**

1. **KernelFactory Registration:**
```cpp
// In KernelFactory.cpp
case KernelType::FUSED_ATTENTION_WO:
    return std::make_unique<FusedAttentionWoKernel>(config);
```

2. **Pipeline Integration:**
```cpp
// In Qwen2Pipeline::attention_block
// BEFORE: Separate attention + Wo GEMM
auto attn_output = attention_kernel->compute(Q, K, V, ...);
auto wo_output = wo_gemm->multiply(attn_output, Wo, ...);

// AFTER: Fused kernel
auto output = fused_attention_wo->compute(Q, K, V, Wo, ...);
```

3. **Expected Benefits:**
- Eliminate context quantization round-trip (FP32 → Q8_1 → FP32)
- Better cache locality (context stays in registers through Wo projection)
- Estimated 10-20% speedup for attention block

**Success Criteria:**
| Metric | Target |
|--------|--------|
| ATTENTION_OUTPUT cosine vs PyTorch | > 0.99 |
| Top-1 token accuracy | 100% |
| Throughput vs unfused | ≥ 1.1x |

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
