# Q16_1 Integer-Domain Fused Attention Kernel

**Status**: Implementation Phase (Reference Kernel Complete, Pipeline Integration In Progress)  
**Created**: 2025-12-27  
**Updated**: 2025-12-28  
**Author**: Llaminar Team  
**Related**: [Exp2FixedSoftmax JIT](../../src/v2/kernels/cpu/attention/q16_1/jit/JitExp2FixedSoftmax.h), [IntAttention Paper](https://arxiv.org/abs/2511.21513)

---

## Recent Progress (2025-12-28)

### Completed This Session

1. **Q16_INTEGER Backend Registration**
   - Added `Q16_INTEGER` to `FusedAttentionBackend` enum in RuntimeConfig.h
   - Updated parser to accept "q16_integer", "q16", "q16_int" CLI options
   - Updated `fusedAttentionBackendToString()` for logging/display

2. **Q16_1 KV Cache Support**
   - Added `UnifiedKVCacheTensor<Q16_1>` template specialization mapping to `Q16_1Tensor`
   - Added `UnifiedKVCacheQ16_1` type alias
   - Implemented `allocate_tensor<Q16_1>()` specialization
   - Implemented `copy_append_data<Q16_1>()` for block-based KV append
   - Implemented `shift_evict_data<Q16_1>()` for sliding window eviction
   - Added Q16_1 branches in `gather_kv_batched()` for both K and V tensors
   - Added explicit template instantiation for Q16_1
   - Updated all 4 factory functions to support Q16_1 precision
   - Added comprehensive unit test coverage (12 new Q16_1 tests)

3. **HybridQ16 Pipeline Precision Configuration**
   - Updated `HybridQ16PrecisionConfig` for Q16 integer attention pipeline:
     - `q_after_rope`: FP32 → **Q16_1** (Q16 kernel expects Q16_1 inputs)
     - `k_after_rope`: FP32 → **Q16_1** (Q16 kernel expects Q16_1 inputs)
     - `kv_cache`: FP32 → **Q16_1** (Q16 kernel uses Q16_1 KV cache)
     - `attention_context`: FP32 → **Q16_1** (for snapshots; fused kernel is INT32 internally)
     - `attention_output`: Q8_1 → **Q16_1** (fused kernel writes Q16_1 directly to residual)
   - Added unit tests for all HybridQ16 precision settings

4. **Unit Test Coverage**
   - `tests/v2/unit/tensors/Test__UnifiedKVCache.cpp`: 12 new Q16_1 tests
   - `tests/v2/unit/Test__HybridPrecisionConfig.cpp`: 21 new HybridQ16 tests

### Previously Completed

- **Q16FusedAttentionRef**: Complete 651-line reference implementation with:
  - Flash Decode path (seq_len=1): Streaming GEMV with online softmax
  - FA2 Prefill path (seq_len>1): Tiled GEMM with blocked computation
  - Exp2FixedSoftmax integration (256-entry LUT)
  - Fused Wo projection and residual add

- **JIT Scaffolding**: All 7 tasks completed:
  - Q16RegisterAllocation.h with zone definitions
  - JitQ16FusedAttention.h base class
  - 4 microkernel stubs (Q8DotProduct, OnlineSoftmax, VNNIMulAccumulate, WoProjection)

- **Microkernel Reference Implementations**:
  - Int8RequantRef: INT32 → INT8 requantization with scaling
  - Exp2FixedSoftmaxRef: Integer-domain softmax with 30-bit precision LUT
  - WoProjectionVNNIRef: Streaming Wo projection with VPDPBUSD patterns

### Files Modified This Session

| File | Changes |
|------|--------|
| `src/v2/execution/RuntimeConfig.h` | Added Q16_INTEGER enum, parser, toString |
| `src/v2/tensors/UnifiedKVCache.h` | Added Q16_1 type mapping and alias |
| `src/v2/tensors/UnifiedKVCache.cpp` | Added Q16_1 specializations and factory support |
| `src/v2/execution/HybridPrecisionConfig.h` | Full Q16 pipeline precision config (Q16_1 for Q/K/KV/context/output) |
| `src/v2/execution/HybridPrecisionConfig.cpp` | Updated comments and getPrecision() for Q16 pipeline |
| `tests/v2/unit/tensors/Test__UnifiedKVCache.cpp` | Added 12 Q16_1 KV cache unit tests |
| `tests/v2/unit/Test__HybridPrecisionConfig.cpp` | Added 21 HybridQ16PrecisionConfig tests |
| `tests/v2/CMakeLists.txt` | Added Q16_1 label to KV cache tests |

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation and Goals](#motivation-and-goals)
3. [Architecture Overview](#architecture-overview)
4. [FA2-Style Tiled Algorithm](#fa2-style-tiled-algorithm)
5. [Prefill vs Decode Execution Strategies](#prefill-vs-decode-execution-strategies)
6. [Fused Kernel Design](#fused-kernel-design)
7. [Development Methodology](#development-methodology)
8. [Component Specifications](#component-specifications)
9. [Exp2FixedSoftmax Integration](#exp2fixedsoftmax-integration)
10. [Q16 vs Q8 Precision Analysis](#q16-vs-q8-precision-analysis)
11. [Implementation Phases](#implementation-phases)
12. [Testing Strategy](#testing-strategy)
13. [Performance Targets](#performance-targets)
14. [Open Questions](#open-questions)
15. [References](#references)
16. [Appendices](#appendix-a-vnni-instruction-reference)

---

## Executive Summary

This document describes the design for a **fully integer-domain FA2-style fused attention kernel** operating on Q16_1 tensors. The kernel fuses:

1. **Q×K^T dot products** (tiled, online softmax)
2. **Exp2FixedSoftmax** (256-entry LUT-based integer softmax using exp2 approximation)
3. **P×V accumulation** (weighted value sum)
4. **Wo projection** (streaming weight repack)
5. **Residual add** (Q16_1 output)

### Critical: AVX512-VNNI Hardware Constraints

**All integer arithmetic must work within AVX512-VNNI limitations:**

| Instruction | Operation | Constraint |
|-------------|-----------|------------|
| `VPDPWSSD` | INT16 × INT16 → INT32 | **Signed** multiply-accumulate |
| `VPDPBUSD` | UINT8 × INT8 → INT32 | Unsigned × Signed multiply-accumulate |
| Output | INT32 accumulator | Cannot use INT64 in hardware |
| Weights | Must be INT16 | Values >32767 interpreted as negative! |

**Design Implications:**
- **Exp2FixedSoftmax outputs INT16** weights in range `[0, 32767]` (not UINT16)
- **P×V uses INT32 accumulators** (not INT64) to match VPDPWSSD output
- **LUT max value = 32767** (15 bits precision, always non-negative)
- **FP32 only for scale factors** — data stays quantized (INT16/INT32) throughout

### VNNI Instruction Usage in Q16 Pipeline

| Stage | Left Operand | Right Operand | Instruction |
|-------|--------------|---------------|-------------|
| Q×K^T (requant) | INT8 Q_requant | INT8 K_requant | scalar (or VPDPBUSD) |
| P×V accumulation | INT16 softmax weights | INT16 V values | **VPDPWSSD** |
| Wo projection | UINT8 (Q8_1 context) | INT8 (packed Wo) | **VPDPBUSD** |
| FFN projections | UINT8 (Q8_1 activations) | INT8 (packed weights) | **VPDPBUSD** |

**Key Clarification**: Model weights (Wo, FFN, etc.) are **always INT8 VNNI-packed**.
VPDPWSSD is used **only** for P×V where both operands are runtime activations (INT16).

### Implementation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Scalar C++ Reference Kernel                           │
│  ─────────────────────────────────────                          │
│  • Readable, debuggable implementation                          │
│  • Bit-exact reference for JIT validation                       │
│  • Full test coverage with FP32 comparison                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: JIT Kernel Implementation                             │
│  ──────────────────────────────────                             │
│  • Xbyak-generated AVX-512/VNNI code                            │
│  • Validated against scalar reference (bit-exact or tolerance)  │
│  • Register allocation via existing guard system                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Single fused kernel** — FA2-style tiling, not separate microkernels
2. **All-integer dataflow** — No FP32 dequantization until output
3. **Q16_1 precision preservation** — 16-bit integers throughout
4. **Exp2FixedSoftmax** — 256-entry LUT for exp2 approximation, INT16 output [0, 32767]
5. **INT32 accumulators** — Match VPDPWSSD hardware output width
6. **Wo + Residual fusion** — Complete attention block in one kernel call
7. **Scalar-first development** — Reference implementation before JIT

### Expected Benefits

| Metric | Current Hybrid | Q16 Fused | Improvement |
|--------|----------------|-----------|-------------|
| Cosine Similarity | 0.897 | ~0.98+ (target) | +9% |
| Memory Bandwidth | High (FP32 intermediates) | Low (INT16 tiled) | ~60% reduction |
| Kernel Calls | 6+ per attention | 1 | 6× fewer |
| Cache Efficiency | Poor | Excellent (FA2 tiling) | Significant |

---

## Motivation and Goals

### Problem Statement

The current `HybridQ16` attention mode achieves only **0.799 cosine similarity** vs FP32 reference due to:

1. **KV Cache stored as FP32** — Fused kernel receives FP32 K/V despite Q16_1 projections
2. **Precision loss at softmax** — Q8_1 dot products lose precision before softmax
3. **Type conversion overhead** — Repeated quant/dequant cycles

### Goals

1. **Achieve ≥0.98 cosine similarity** with FP32 reference attention output
2. **Eliminate FP32 intermediates** — Pure integer until final output
3. **Maintain Q16_1 KV cache** — Store K/V in Q16_1 format natively
4. **Maximize VNNI utilization** — INT16×INT16 instructions throughout
5. **Support streaming decode** — Efficient single-token inference

### Non-Goals (for v1)

- GPU/CUDA implementation (CPU-first)
- Flash Attention memory optimization (standard attention pattern)
- Grouped Query Attention optimization (future work)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            Q16_1 FA2-STYLE FUSED ATTENTION + Wo + RESIDUAL                  │
│                    (Single Kernel, Tiled Computation)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUTS                                                                      │
│  ───────                                                                     │
│  Q: Q16_1[seq_len_q × n_heads × head_dim] + per-row scale                   │
│  K: Q16_1[kv_len × n_kv_heads × head_dim] + per-row scale (KV cache)        │
│  V: Q16_1[kv_len × n_kv_heads × head_dim] + per-row scale (KV cache)        │
│  Wo: Q8_0[d_model × n_heads × head_dim] (quantized weights)                 │
│  residual: Q16_1[seq_len_q × d_model] (input residual stream)               │
│                                                                              │
│  ╔══════════════════════════════════════════════════════════════════════╗   │
│  ║                     FUSED KERNEL (FA2-Style Tiling)                   ║   │
│  ╠══════════════════════════════════════════════════════════════════════╣   │
│  ║                                                                       ║   │
│  ║  for each query_tile in Q (Br queries at a time):                    ║   │
│  ║    Initialize: O_acc[Br, head_dim] = 0                               ║   │
│  ║                m[Br] = -INF (online softmax max)                     ║   │
│  ║                l[Br] = 0    (online softmax sum)                     ║   │
│  ║                                                                       ║   │
│  ║    for each kv_tile in K,V (Bc keys at a time):                      ║   │
│  ║      ┌────────────────────────────────────────────────────────────┐  ║   │
│  ║      │ STEP 1: Q16×K16 Dot Products                                │  ║   │
│  ║      │   S[Br, Bc] = Q_tile × K_tile^T  (INT32 scores)            │  ║   │
│  ║      │   Instruction: vpdpwssd (AVX-512 VNNI)                     │  ║   │
│  ║      └────────────────────────────────────────────────────────────┘  ║   │
│  ║                              │                                        ║   │
│  ║                              ▼                                        ║   │
│  ║      ┌────────────────────────────────────────────────────────────┐  ║   │
│  ║      │ STEP 2: Online Exp2FixedSoftmax                             │  ║   │
│  ║      │   m_new = max(m, rowmax(S))                                │  ║   │
│  ║      │   P[Br, Bc] = Exp2FixedSoftmax(S, m_new) → INT16           │  ║   │
│  ║      │   (Uses 256-entry exp2 LUT, outputs INT16 in [0, 32767])   │  ║   │
│  ║      │   l_new = l × exp2_ratio + rowsum(P)                       │  ║   │
│  ║      │   O_acc = O_acc × exp2_ratio (rescale previous accum)      │  ║   │
│  ║      └────────────────────────────────────────────────────────────┘  ║   │
│  ║                              │                                        ║   │
│  ║                              ▼                                        ║   │
│  ║      ┌────────────────────────────────────────────────────────────┐  ║   │
│  ║      │ STEP 3: INT16×INT16 V Accumulation (VNNI VPDPWSSD)          │  ║   │
│  ║      │   O_acc += P × V_tile  (INT32 accumulator)                 │  ║   │
│  ║      └────────────────────────────────────────────────────────────┘  ║   │
│  ║                                                                       ║   │
│  ║    end kv_tile loop                                                  ║   │
│  ║                                                                       ║   │
│  ║    ┌──────────────────────────────────────────────────────────────┐  ║   │
│  ║    │ STEP 4: Final Normalization                                   │  ║   │
│  ║    │   O[Br, head_dim] = O_acc / l  (normalize by softmax sum)    │  ║   │
│  ║    └──────────────────────────────────────────────────────────────┘  ║   │
│  ║                              │                                        ║   │
│  ║                              ▼                                        ║   │
│  ║    ┌──────────────────────────────────────────────────────────────┐  ║   │
│  ║    │ STEP 5: Wo Projection (INT32 context → INT16 → VPDPWSSD)     │  ║   │
│  ║    │   context_int16 = requant(O_int32)  (INT32 → INT16)         │  ║   │
│  ║    │   proj[d_model] = context_int16 × Wo_int16  (vpdpwssd)      │  ║   │
│  ║    │   (Wo is INT8 sign-extended to INT16 at runtime)            │  ║   │
│  ║    └──────────────────────────────────────────────────────────────┘  ║   │
│  ║                              │                                        ║   │
│  ║                              ▼                                        ║   │
│  ║    ┌──────────────────────────────────────────────────────────────┐  ║   │
│  ║    │ STEP 6: Fused Residual Add + Requantize                      │  ║   │
│  ║    │   output_q16 = residual_q16 + proj  (with scale adjustment) │  ║   │
│  ║    │   Store: Q16_1 updated residual                              │  ║   │
│  ║    └──────────────────────────────────────────────────────────────┘  ║   │
│  ║                                                                       ║   │
│  ║  end query_tile loop                                                 ║   │
│  ║                                                                       ║   │
│  ╚══════════════════════════════════════════════════════════════════════╝   │
│                                                                              │
│  OUTPUT: Q16_1[seq_len_q × d_model] updated residual (in-place)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FA2 Tiling Benefits

| Aspect | Standard Attention | FA2-Style Tiled |
|--------|-------------------|-----------------|
| Memory for S matrix | O(seq² × heads) | O(Br × Bc) tile only |
| Memory for P matrix | O(seq² × heads) | O(Br × Bc) tile only |
| HBM reads | 3× (Q, K, V separately) | 1× (fused, cached) |
| Kernel launches | 6+ (QK, softmax, PV, Wo, add, ...) | 1 (fully fused) |

### Tile Size Selection

For CPU with L2 cache ~1MB per core:

| Parameter | Value | Memory |
|-----------|-------|--------|
| Br (query tile) | 16 | 16 × 128 × 2 = 4KB |
| Bc (kv tile) | 64 | 64 × 128 × 2 = 16KB |
| S tile | 16×64 | 16 × 64 × 4 = 4KB |
| P tile (INT16) | 16×64 | 16 × 64 × 2 = 2KB |
| V tile | 64×128 | 64 × 128 × 2 = 16KB |
| O accumulator (INT32) | 16×128 | 16 × 128 × 4 = 8KB |
| **Total** | | **~50KB** ✓ fits L2 |

---

## Cache Hierarchy Strategy

### Overview

The Q16 fused attention kernel operates on a three-level cache hierarchy. We use **dynamic detection** via `CPUFeatures.h` to adapt tile sizes and prefetch strategies at JIT compile time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CACHE HIERARCHY ASSIGNMENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L1 DATA CACHE (32KB typical, per-core private)                       │    │
│  │ ═══════════════════════════════════════════════════════════════════ │    │
│  │ • Register spill area for ZMM registers                              │    │
│  │ • Current Q/K/V block being processed (Q16_1 blocks)                │    │
│  │ • Score accumulator tile (S[Br, Bc_micro] or s[Bc_micro] for decode)│    │
│  │ • Exp2FixedSoftmax LUT (1KB: 256 × 4-byte entries)                   │    │
│  │ • Active softmax state (m, l per row)                               │    │
│  │                                                                      │    │
│  │ Prefetch: PREFETCHT0 (non-temporal to L1)                           │    │
│  │ Distance: 2-8 KV positions ahead                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L2 CACHE (256KB-1MB typical, per-core private)                       │    │
│  │ ═══════════════════════════════════════════════════════════════════ │    │
│  │ DECODE (Flash Decode):                                               │    │
│  │   • q vector (head_dim × 2B = 256B for 128-dim)                     │    │
│  │   • O accumulator (head_dim × 8B = 1KB for INT64)                   │    │
│  │   • Streaming K/V through L2 → L1                                    │    │
│  │                                                                      │    │
│  │ PREFILL (FA2):                                                       │    │
│  │   • Q tile [Br × head_dim] (Br=16, 128-dim → 4KB)                   │    │
│  │   • K tile [Bc × head_dim] (Bc=64, 128-dim → 16KB)                  │    │
│  │   • V tile [Bc × head_dim] (Bc=64, 128-dim → 16KB)                  │    │
│  │   • O accumulators [Br × head_dim × 8B] (16KB)                      │    │
│  │   • S/P tiles [Br × Bc] (~6KB)                                      │    │
│  │   Total: ~58KB per tile iteration                                    │    │
│  │                                                                      │    │
│  │ Prefetch: PREFETCHT1 (to L2)                                        │    │
│  │ Distance: 8-32 KV positions ahead                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ L3 CACHE (8-40MB typical, shared across cores)                       │    │
│  │ ═══════════════════════════════════════════════════════════════════ │    │
│  │ DECODE:                                                              │    │
│  │   • KV cache for current head (streams through)                      │    │
│  │   • Next head's KV data (prefetched)                                 │    │
│  │                                                                      │    │
│  │ PREFILL:                                                             │    │
│  │   • KV cache tiles beyond current working set                        │    │
│  │   • Wo weight blocks (for projection phase)                          │    │
│  │                                                                      │    │
│  │ Prefetch: PREFETCHT2 (to L3, non-temporal)                          │    │
│  │ Distance: 32-128 KV positions ahead (for XL sequences)               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ MAIN MEMORY (DRAM)                                                   │    │
│  │ ═══════════════════════════════════════════════════════════════════ │    │
│  │ • Full KV cache (may be TBs for long context)                        │    │
│  │ • Wo weight matrix (d_model² elements)                               │    │
│  │ • Q, residual tensors (seq_len × d_model)                           │    │
│  │                                                                      │    │
│  │ Strategy: Sequential access for KV, blocked access for Wo            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Flash Decode: Cache Strategy

Flash Decode processes all KV positions in a single pass without tiling. The key is to **keep q in registers** and **stream K/V sequentially**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLASH DECODE CACHE FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │ REGISTERS (ZMM)  │  q[head_dim]: 4 ZMM for 128-dim                       │
│  │ ════════════════ │  O_acc[head_dim]: 8 ZMM for INT64 accumulator         │
│  │ Total: 12 ZMM    │  m, l: 2 scalars (softmax state)                      │
│  └────────┬─────────┘  Scratch: 8 ZMM for K/V streaming                     │
│           │                                                                  │
│           │ K/V stream: 4 positions per micro-iteration                      │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ L1 (32KB)        │  Current K/V block: 4×72B = 288B                       │
│  │ ════════════════ │  LUT: 256B                                            │
│  │ Working: ~1KB    │  Scores: 4×4B = 16B                                    │
│  └────────┬─────────┘  Weights: 4×2B = 8B                                    │
│           │                                                                  │
│           │ Prefetch T0: 8 positions ahead                                   │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ L2 (256KB-1MB)   │  Streaming buffer: ~16KB (next 64 positions)          │
│  │ ════════════════ │  Output buffer: head_dim × 8B = 1KB                    │
│  │ Working: ~20KB   │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           │ Prefetch T1/T2: 32-128 positions ahead (sequence-dependent)      │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ L3 / DRAM        │  Full KV cache: kv_len × head_dim × 72 bytes          │
│  │ ════════════════ │  For 4K context, 128-dim: ~37MB per KV head            │
│  └──────────────────┘                                                        │
│                                                                              │
│  PREFETCH STRATEGY:                                                          │
│  ──────────────────                                                          │
│  • KV fits in L2 (short seq): PREFETCHT0, distance=4                        │
│  • KV fits in L3 (medium seq): PREFETCHT1, distance=16                      │
│  • KV exceeds L3 (long seq): PREFETCHT2, distance=64                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Flash Decode Data Layout per Stage**:

| Stage | L1 | L2 | L3/DRAM | Notes |
|-------|----|----|---------|-------|
| **Q×K^T** | K[4, head_dim], LUT | q (from reg), scores | KV cache | Stream K, q pinned in registers |
| **Softmax** | scores[4], weights[4], (m,l) | - | - | All in registers/L1 |
| **P×V** | V[4, head_dim], weights[4] | O_acc[head_dim] | KV cache | Stream V, accumulate to L2 |
| **Wo** | Wo block[d, 32] | O_norm[n_heads×head_dim] | Wo matrix | Stream Wo rows |
| **Residual** | proj[d_model] | residual I/O | - | In-place update |

### FA2 Prefill: Cache Strategy

FA2 Prefill processes in tiles [Br × Bc]. The key is to **fit all working data in L2** and **amortize K/V loads across Br queries**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FA2 PREFILL CACHE FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │ REGISTERS (ZMM)  │  Q micro-tile[4, head_dim]: 4 ZMM                      │
│  │ ════════════════ │  K micro-tile[4, head_dim]: 4 ZMM                      │
│  │ Total: 24 ZMM    │  S micro-tile[4, 4]: 4 ZMM (INT32 scores)              │
│  │                  │  O accumulators[4, head_dim]: 8 ZMM (INT64)            │
│  └────────┬─────────┘  Constants: 4 ZMM                                      │
│           │                                                                  │
│           │ Micro-tile: 4×4 GEMM blocking                                    │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ L1 (32KB)        │  LUT: 256B                                             │
│  │ ════════════════ │  P micro-tile[4, Bc]: 512B                             │
│  │ Working: ~2KB    │  Score scratch: 256B                                   │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           │ Tile buffers                                                     │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ L2 (256KB-1MB)   │  Q tile[Br, head_dim]: 4KB (Br=16, 128-dim)            │
│  │ ════════════════ │  K tile[Bc, head_dim]: 16KB (Bc=64)                    │
│  │ Working: ~58KB   │  V tile[Bc, head_dim]: 16KB                            │
│  │                  │  S tile[Br, Bc]: 4KB (INT32)                           │
│  │                  │  P tile[Br, Bc]: 2KB (INT16)                           │
│  │                  │  O_acc[Br, head_dim]: 16KB (INT64)                     │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           │ KV cache and Wo streaming                                        │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ L3 / DRAM        │  Full KV cache                                         │
│  │ ════════════════ │  Wo matrix (streamed for projection)                   │
│  └──────────────────┘  Next Q tile (prefetched)                              │
│                                                                              │
│  PREFETCH STRATEGY:                                                          │
│  ──────────────────                                                          │
│  • Next KV tile: PREFETCHT1 (to L2), 1 tile ahead                           │
│  • Next Q tile: PREFETCHT2 (to L3), when nearing end of KV loop             │
│  • Wo weights: PREFETCHT2 (stream from L3/DRAM during Wo phase)             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**FA2 Prefill Tile Sizing Formula**:

```cpp
// Dynamic tile sizing based on detected cache
struct Q16TileConfig {
    int Br;  // Query tile size
    int Bc;  // KV tile size
    
    static Q16TileConfig compute(int head_dim) {
        const auto& cache = llaminar2::cache_info();
        
        // Target: 50% of L2 for working set
        size_t l2_budget = cache.l2_size / 2;
        
        // Per-query overhead: head_dim × 2 (Q) + head_dim × 8 (O_acc) = 10 × head_dim
        size_t per_query = head_dim * 10;
        
        // Per-KV overhead: head_dim × 4 (K + V as Q16_1 = 72B blocks)
        //                  + Br × 4 (S) + Br × 2 (P)
        // For Br=16: head_dim × 4 + 96 per KV position
        size_t per_kv = head_dim * 4 + 96;  // assuming Br=16
        
        // Start with Br=16 (good register blocking)
        int Br = 16;
        size_t q_budget = Br * per_query;  // ~20KB for 128-dim
        
        // Remaining budget for K/V tiles
        size_t kv_budget = l2_budget - q_budget;
        int Bc = kv_budget / per_kv;
        
        // Round down to multiple of 4 (VNNI blocking)
        Bc = (Bc / 4) * 4;
        
        // Clamp to reasonable range
        Bc = std::max(16, std::min(256, Bc));
        
        return {Br, Bc};
    }
};
```

### Wo Projection: Cache Strategy

Wo projection is **memory-bound** because the weight matrix is large. We use **row-streaming** with **batch accumulation**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Wo PROJECTION CACHE FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DECODE (Single Row GEMV):                                                   │
│  ─────────────────────────                                                   │
│  • Input: O_norm[n_heads × head_dim] in L2                                  │
│  • Wo: d_model rows × (n_heads × head_dim) cols                             │
│  • Strategy: Stream Wo rows, keep O_norm in L2                              │
│                                                                              │
│  for each output_row in d_model:                                            │
│      proj[row] = O_norm · Wo[row, :]                                        │
│      // Wo row loaded once, O_norm reused from L2                           │
│                                                                              │
│  Memory traffic: d_model × (n_heads × head_dim) × weight_size               │
│  For 3584×3584 Q8_0: ~13MB streamed                                         │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                              │
│  PREFILL (Batched GEMM):                                                     │
│  ───────────────────────                                                     │
│  • Input: O_norm[seq_len_q, n_heads × head_dim] in L3                       │
│  • Wo: d_model × (n_heads × head_dim)                                       │
│  • Strategy: Batch queries to amortize Wo loads                             │
│                                                                              │
│  // Wo batch size computed dynamically (see CacheInfo::optimal_wo_batch_size)│
│  for each wo_batch in seq_len_q / batch_size:                               │
│      // Load batch of O_norm into L2 (~batch_size × d_model × 4B)           │
│      // Stream Wo once, computing batch_size outputs                        │
│      proj[batch, :] = O_norm[batch, :] × Wo^T                               │
│                                                                              │
│  Memory traffic: seq_len_q × d_model² / batch_size (Wo reuse)               │
│  With batch=8: 8× reduction in Wo memory traffic                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Residual Add: Cache Strategy

Residual add is **simple** — the projection output and residual are both in L2/L3 from previous stages.

| Mode | Input Location | Strategy |
|------|---------------|----------|
| Decode | proj[d_model] in L2, residual row in L2 | In-place, single pass |
| Prefill | proj[seq_len, d_model] in L3, residual in L3 | Streaming, row-by-row |

### Q16_1 Block Memory Layout

Understanding the Q16_1 block layout is critical for prefetch efficiency:

```
Q16_1 Block (72 bytes, 32 elements):
┌─────────────────────────────────────────────────────────────┐
│ d (float, 4B) │ sum_qs (int32_t, 4B) │ qs[32] (int16_t, 64B)│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
For head_dim=128: 4 blocks per head → 288 bytes per KV position per head
```

**Prefetch Distance Calculation**:

```cpp
// From CPUFeatures.h - AttentionCacheConfig::prefetch_config()
// Compute optimal prefetch distance based on cache level and latency hiding

size_t bytes_per_kv_pos = (head_dim / 32) * 72;  // K only, or ×2 for K+V

// L1 prefetch: hide ~4 cycle latency, ~4 cache lines
int l1_distance = std::max(2, (4 * 64) / bytes_per_kv_pos);

// L2 prefetch: hide ~12 cycle latency, ~16 cache lines  
int l2_distance = std::max(8, (16 * 64) / bytes_per_kv_pos);

// L3/DRAM prefetch: hide ~100+ cycle latency, ~64 cache lines
int l3_distance = std::max(32, (64 * 64) / bytes_per_kv_pos);
```

### Dynamic Configuration via CPUFeatures.h

The JIT kernel queries cache sizes at compile time:

```cpp
// In JIT kernel constructor
void Q16AttentionJit::configure() {
    const auto& cache = llaminar2::cache_info();
    
    // Determine work size class
    AttentionCacheConfig cfg(head_dim_, num_kv_heads_, kv_seq_len_);
    work_size_ = cfg.work_size();
    prefetch_ = cfg.prefetch_config();
    
    // Set tile sizes based on cache
    if (seq_len_q_ == 1) {
        // Flash Decode: no Q tiling, process all KV
        Br_ = 1;
        Bc_ = kv_seq_len_;  // Process all at once
        kv_micro_tile_ = cfg.prefer_kv8_tile() ? 8 : 4;
    } else {
        // FA2 Prefill: tile both Q and KV
        auto tiles = Q16TileConfig::compute(head_dim_);
        Br_ = tiles.Br;
        Bc_ = tiles.Bc;
    }
    
    // Configure Wo batching
    wo_batch_size_ = cache.optimal_wo_batch_size(d_model_);
}
```

### Summary: Cache Assignment Table

| Data | Flash Decode | FA2 Prefill | Notes |
|------|-------------|-------------|-------|
| **q vector** | Registers (ZMM0-3) | L2 (Q tile) | Decode: pinned; Prefill: tiled |
| **K tile** | L1 (streaming) | L2 | Prefetched from L3/DRAM |
| **V tile** | L1 (streaming) | L2 | Loaded with K |
| **S scores** | L1 (micro-tile) | L2 (S tile) | INT32 |
| **P weights** | L1 (micro-tile) | L2 (P tile) | INT16 |
| **O accumulator** | L2 | L2 | INT64, head_dim sized |
| **Softmax state** | Registers | L1 | (m, l) per row |
| **Index16 LUT** | L1 | L1 | 256 bytes |
| **Wo weights** | L3→L2 (streaming) | L3→L2 (streaming) | Batched in prefill |
| **Residual** | L2 | L3 | In-place output |

---

## Register Allocation Contract

### Overview

The Q16 attention JIT kernel uses the **Register Guard** system from `src/v2/kernels/cpu/jit/` to enforce a contract on register usage across all microkernels. This prevents register clobbering bugs at compile time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ZMM REGISTER ZONE LAYOUT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  zmm0-7   │ ACCUMULATOR ZONE   │ O accumulators (INT64 context output)     │
│           │ [VEX-safe LOW]     │ 8 regs × 64B = 512B (4 head_dim blocks)   │
│  ─────────┼────────────────────┼─────────────────────────────────────────── │
│  zmm8-15  │ INPUT ZONE         │ Q vectors (decode) / Q tile row (prefill) │
│           │ [VEX-safe LOW]     │ Also K/V streaming during dot product     │
│  ─────────┼────────────────────┼─────────────────────────────────────────── │
│  zmm16-19 │ STATE ZONE         │ Online softmax state (m, l, weight, corr) │
│           │ [EVEX-only HIGH]   │ Persistent across KV iterations           │
│  ─────────┼────────────────────┼─────────────────────────────────────────── │
│  zmm20-25 │ SCRATCH ZONE       │ Temporaries, intermediate results         │
│           │ [EVEX-only HIGH]   │ NOTE: zmm20-23 alias xmm20-23 (ScoreZone) │
│  ─────────┼────────────────────┼─────────────────────────────────────────── │
│  zmm26-31 │ RESERVED ZONE      │ Preloaded constants (never modified)      │
│           │ [EVEX-only HIGH]   │ scale, LUT base, thresholds, etc.         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    XMM OVERLAY (ScoreZone)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  xmm20-23 │ SCORE ZONE         │ FA2 scalar scores (aliases Scratch0-3)    │
│           │ [EVEX-only HIGH]   │ 4 scores for 4×4 micro-tile               │
│                                                                              │
│  CRITICAL: Using ScoreZone invalidates Scratch0-3 until scores consumed!    │
│            Safe scratch during scoring: Scratch4 (zmm24), Scratch5 (zmm25)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Q16 Reserved Constants (zmm26-31)

| Register | Alias | Contents | Usage |
|----------|-------|----------|-------|
| zmm26 | `Const128` | 128.0f broadcast | Q16 dequant: val × d / 128 |
| zmm27 | `ConstScale` | 1/√head_dim | Attention scale factor |
| zmm28 | `ConstLog2E` | 1.4427f broadcast | Exp2FixedSoftmax: log₂(e) conversion |
| zmm29 | `Const256` | 256.0f broadcast | Exp2FixedSoftmax: fractional LUT index |
| zmm30 | `ConstOne` | 1.0f broadcast | Various normalization |
| zmm31 | `ConstZero` | 0.0f broadcast | Initialization, masking |

### Microkernel Register Contracts

Each microkernel declares its register usage. The JIT kernel orchestrates handoffs between microkernels, ensuring no conflicts.

#### Q16DotProduct (Q×K^T)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Q16 DOT PRODUCT MICROKERNEL                                                │
│  Purpose: Compute s[kv] = q · k[kv] for INT32 scores                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUTS (read-only):                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input0-3 (zmm8-11)  │ Q vector: 4 blocks × 32 INT16 = head_dim=128    │ │
│  │                     │ Loaded once per query, reused across all K       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STREAMING (cyclic reuse):                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input4-7 (zmm12-15) │ K vectors: Stream 4 KV positions at a time      │ │
│  │                     │ Block-by-block: k[kv, block] for dot product    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUTS (write):                                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch0-3 (zmm20-23) │ INT32 scores: 4 KV positions per iteration    │ │
│  │                       │ s[kv] = Σ_block (q[block] · k[kv, block])     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SCRATCH (internal temporaries):                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch4-5 (zmm24-25) │ Intermediate dot products, scale factors      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  CONSTANTS (read-only):                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Const128 (zmm26)      │ For dequant scale computation                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Exp2FixedSoftmax

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EXP2 FIXED-POINT SOFTMAX MICROKERNEL                                       │
│  Purpose: Convert INT32 scores → INT16 weights via exp2 LUT approximation  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ALGORITHM: For each score, compute exp2(-delta * beta) where:              │
│    delta = max_score - score                                                │
│    beta = alpha * log2(e)                                                   │
│    t = delta * beta → decompose into integer + fractional parts            │
│    result = LUT[frac] >> int_part                                           │
│                                                                              │
│  LUT: 256-entry table for 2^(-frac/256) with 30-bit precision              │
│                                                                              │
│  INPUTS (read, then overwritten):                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch0-3 (zmm20-23) │ INT32 scores from Q16DotProduct               │ │
│  │                       │ Consumed and overwritten with INT16 weights   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUTS (overwrite input registers):                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch0-3 (zmm20-23) │ INT16 weights in range [0, 32767]             │ │
│  │                       │ (VNNI-compatible signed INT16)                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STATE (persistent across KV iterations - FA2 only):                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ StateMax (zmm16)    │ Running max score (INT32) for online softmax    │ │
│  │ StateSum (zmm17)    │ Running sum of weights (INT64) for normalization│ │
│  │ StateWeight (zmm18) │ Current max weight (for rescaling check)        │ │
│  │ StateCorr (zmm19)   │ Correction factor (rescale accumulator)         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SCRATCH (internal temporaries):                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch4-5 (zmm24-25) │ Delta computation, exp2 decomposition         │ │
│  │ Input4-7 (zmm12-15)   │ LUT entries (256 × 4B = 1KB, L1 resident)     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  CONSTANTS (read-only):                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ ConstLog2E (zmm28)    │ log2(e) ≈ 1.4427 for ln→log2 conversion      │ │
│  │ Const256 (zmm29)      │ 256.0 for fractional index calculation       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  FLASH DECODE vs FA2 PREFILL:                                               │
│  • Flash Decode: StateZone unused (single-pass, no online tracking)         │
│  • FA2 Prefill: StateZone tracks running (m, l) across KV tiles             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### PVAccumulate (P×V)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PV ACCUMULATE MICROKERNEL                                                  │
│  Purpose: O_acc[d] += Σ_kv weight[kv] × V[kv, d]                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUTS (read-only):                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch0-3 (zmm20-23) │ INT16 weights from Exp2FixedSoftmax           │ │
│  │                       │ 4 KV positions × 32 weights (though only 1    │ │
│  │                       │ weight per KV position actually used)         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STREAMING (cyclic reuse):                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input4-7 (zmm12-15) │ V vectors: 4 KV positions, block-by-block       │ │
│  │                     │ V[kv, block] loaded and immediately consumed    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUTS (accumulate):                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Accum0-7 (zmm0-7)   │ INT64 accumulators: O_acc[head_dim]             │ │
│  │                     │ 8 regs × 8 INT64 = 64 elements (head_dim=128    │ │
│  │                     │ requires 2 passes through 64-element chunks)    │ │
│  │                     │ Accumulated: += weight × V_int16                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SCRATCH (internal temporaries):                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch4-5 (zmm24-25) │ Broadcast weight, intermediate products       │ │
│  │ Input0-3 (zmm8-11)    │ Widened V values (INT16 → INT32 for multiply) │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  CONSTANTS (read-only):                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ (none required)       │ Pure integer multiply-accumulate             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  NOTE: After all KV processed, Accum0-7 contain INT64 weighted sums.       │
│        Normalization (÷ weight_sum) and dequant happen in finalization.    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### WoProjection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WO PROJECTION MICROKERNEL                                                  │
│  Purpose: proj[d_model] = O_norm[n_heads×head_dim] × Wo^T                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUTS (read-only):                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Input0-7 (zmm8-15)  │ O_int32: INT32 context from P×V accumulation    │ │
│  │                     │ Requantized to INT16 before VPDPWSSD multiply   │ │
│  │                     │ Concatenated from all heads: [n_heads×head_dim] │ │
│  │                     │ For models with input_dim > 128, reload in tiles│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  STREAMING (from L3/DRAM):                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch0-3 (zmm20-23) │ Wo weight rows: Q8_0 blocks, dequantized      │ │
│  │                       │ One output element per row of Wo              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUTS (Q16_1 projection written to memory):                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Output buffer        │ proj[d_model]: Q16_1 projection blocks         │ │
│  │                      │ INT32 accumulator → requant → Q16_1            │ │
│  │                      │ Processed in chunks of 32 elements (1 block)   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  SCRATCH (internal temporaries):                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Scratch4-5 (zmm24-25) │ Dequant temporaries, FMA intermediates        │ │
│  │ Accum4-7 (zmm4-7)     │ Additional accumulators for 8-way unroll      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  CONSTANTS (read-only):                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Const128 (zmm26)      │ Q8_0 dequant: val / 127.0 (or similar)        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ALL-INTEGER OUTPUT: This microkernel outputs Q16_1 blocks to memory.      │
│  The subsequent Q16_1 + Q16_1 residual add uses simd::q16_1_add_q16_1()    │
│  which is a proven all-integer operation (dequant, add, requant).          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### ResidualAddQ16

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RESIDUAL ADD Q16 (via simd::q16_1_add_q16_1)                               │
│  Purpose: output_q16 = residual_q16 + projection_q16 (ALL INTEGER)         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  This is NOT a standalone microkernel - uses SIMDHelpers.h::q16_1_add_q16_1 │
│                                                                              │
│  ALGORITHM (per Q16_1 block of 32 elements):                                │
│  1. Load Q16_1 projection block (scale_p, qs_p[32]) from Wo output         │
│  2. Load Q16_1 residual block (scale_r, qs_r[32]) from residual buffer     │
│  3. Dequant both to FP32: val_p = qs_p * scale_p, val_r = qs_r * scale_r  │
│  4. Add in FP32: sum = val_p + val_r                                       │
│  5. Find max_abs for new output scale                                       │
│  6. Requant to Q16_1: scale_out = max_abs/32767, qs_out = round(sum/scale) │
│  7. Store Q16_1 output block (in-place to residual buffer)                 │
│                                                                              │
│  KEY POINT: The "FP32" here is just for scale reconciliation during        │
│  dequant/requant. The DATA stays quantized (INT16) on both sides.          │
│  This is a proven all-integer pattern from SIMDHelpers.h.                  │
│                                                                              │
│  INPUTS:                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ projection_q16      │ Q16_1 output from WoProjection [d_model/32 blks] │ │
│  │ residual_q16        │ Q16_1 residual from previous layer [d_model/32]  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  OUTPUT (in-place to residual buffer):                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ residual_q16        │ Q16_1 output = residual + projection             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  IMPLEMENTATION: simd::q16_1_add_q16_1(residual, projection, output, n)    │
│  - AVX512 optimized with vectorized dequant/add/requant                    │
│  - Processes 32 elements (1 Q16_1 block) at a time                         │
│  - In-place operation: output can alias residual                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Register Flow: Flash Decode Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FLASH DECODE: REGISTER FLOW (seq_len_q = 1)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INITIALIZATION:                                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Load Q[head] → Input0-3 (zmm8-11)                                      │ │
│  │ Clear Accum0-7 (zmm0-7) ← 0                                            │ │
│  │ Initialize m = INT32_MIN, l = 0 (scalars or StateZone if needed)       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  KV LOOP (kv = 0; kv < kv_len; kv += 4):                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: Q×K^T (Q16DotProduct)                                          │ │
│  │   Input0-3 (Q, read-only) × K[kv:kv+4] → Scratch0-3 (scores)           │ │
│  │   K loaded via Input4-7 (streaming)                                    │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ STEP 2: Softmax (Exp2FixedSoftmax)                                     │ │
│  │   Scratch0-3 (scores) → Scratch0-3 (INT16 weights [0, 32767])          │ │
│  │   Update m, l for normalization                                        │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ STEP 3: P×V (PVAccumulate)                                             │ │
│  │   Scratch0-3 (weights) × V[kv:kv+4] → Accum0-7 (accumulate)            │ │
│  │   V loaded via Input4-7 (streaming)                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼ (repeat for all KV)                           │
│  FINALIZATION:                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ O_int32 = Accum0-7 / weight_sum → Input0-7 (INT32 context)             │ │
│  │   (Normalize INT64 accumulators to INT32 context for Wo projection)    │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ STEP 4: Wo Projection (WoProjection)                                   │ │
│  │   Input0-7 (O_norm) × Wo → proj (to memory via Accum0-3)               │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │ STEP 5: Residual Add (ResidualAddQ16)                                  │ │
│  │   proj + residual → output_q16 (to memory)                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Register Flow: FA2 Prefill Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FA2 PREFILL: REGISTER FLOW (seq_len_q > 1)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OUTER LOOP: Q tiles (q_tile = 0; q_tile < seq_len_q; q_tile += Br)         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Load Q[q_tile:q_tile+Br] → L2 buffer (not registers - too large)       │ │
│  │ Clear O_acc[Br × head_dim] in L2                                       │ │
│  │ Initialize StateZone: m[Br] = INT32_MIN, l[Br] = 0                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│    INNER LOOP: KV tiles (kv_tile = 0; kv_tile < kv_len; kv_tile += Bc)      │
│    ┌──────────────────────────────────────────────────────────────────────┐ │
│    │ For each Q row in tile (q_local = 0; q_local < Br; q_local++):       │ │
│    │   Load Q[q_tile + q_local] → Input0-3 (single row at a time)         │ │
│    │                                                                      │ │
│    │   For kv in KV tile (process 4 at a time):                           │ │
│    │     STEP 1: Q×K^T → Scratch0-3 (4 scores)                            │ │
│    │     STEP 2: Exp2FixedSoftmax → Scratch0-3 (4 weights [0, 32767])     │ │
│    │             Update StateZone for online softmax                      │ │
│    │     STEP 3: P×V → Accum0-7 (accumulate for this Q row)               │ │
│    │                                                                      │ │
│    │   Store Accum0-7 back to O_acc[q_local] in L2                        │ │
│    └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼ (after all KV tiles)                          │
│  Q TILE FINALIZATION:                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ For each Q row in tile:                                                │ │
│  │   Normalize O_acc[q_local] by l[q_local]                               │ │
│  │   STEP 4: Wo Projection (batched or row-by-row)                        │ │
│  │   STEP 5: Residual Add                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Register Guard Integration

The JIT kernel uses `borrow<RegType>()` to obtain tracked access to registers:

```cpp
// Example: Q16DotProduct JIT emission
void emit_q16_dot_product(Xbyak::CodeGenerator& gen, RegisterTracker& tracker) {
    // Borrow Q input registers (read-only, but tracked)
    auto q0 = tracker.borrow<Input0>();
    auto q1 = tracker.borrow<Input1>();
    auto q2 = tracker.borrow<Input2>();
    auto q3 = tracker.borrow<Input3>();
    
    // Borrow K streaming registers
    auto k0 = tracker.borrow<Input4>();
    auto k1 = tracker.borrow<Input5>();
    auto k2 = tracker.borrow<Input6>();
    auto k3 = tracker.borrow<Input7>();
    
    // Borrow score output registers
    auto s0 = tracker.borrow<Scratch0>();
    auto s1 = tracker.borrow<Scratch1>();
    auto s2 = tracker.borrow<Scratch2>();
    auto s3 = tracker.borrow<Scratch3>();
    
    // Clear accumulators
    gen.vpxord(s0.zmm(), s0.zmm(), s0.zmm());
    gen.vpxord(s1.zmm(), s1.zmm(), s1.zmm());
    // ... emit VPMADDWD / VPADDD for dot product ...
    
    // Guards automatically release at scope end
}
```

---

## FA2-Style Tiled Algorithm

### Online Softmax with Exp2FixedSoftmax

The key challenge is combining FA2's online softmax rescaling with our integer-domain Exp2FixedSoftmax. Here's the approach:

```
Standard FA2 Online Softmax (FP32):
  m_new = max(m_old, rowmax(S))
  P = exp(S - m_new)
  l_new = l_old × exp(m_old - m_new) + rowsum(P)
  O = O × exp(m_old - m_new) + P × V

Our Integer Adaptation (using exp2 approximation):
  m_new = max(m_old, rowmax(S))             // INT32 max tracking
  P = Exp2FixedSoftmax(S, m_new)            // INT16 weights via 256-entry LUT
  exp2_ratio = LUT[m_old - m_new]           // Integer rescale factor (exp2)
  l_new = (l_old × exp2_ratio) >> shift + rowsum(P)
  O = (O × exp2_ratio) >> shift + P × V

Algorithm: exp2(-delta * beta) where beta = alpha * log2(e)
  1. delta = max_score - score
  2. t = delta * beta
  3. Decompose t into integer (ip) and fractional (frac) parts
  4. result = LUT[frac] >> ip (256-entry LUT with 30-bit precision)
```

### Rescaling in Integer Domain

The tricky part is the `exp(m_old - m_new)` rescaling. Options:

**Option A: Lazy Rescaling (Recommended)**
- Track `m` per-row, defer rescaling to final normalization
- Simpler, avoids repeated rescaling
- Slightly less numerically stable for very long sequences

**Option B: Integer Rescale via Exp2 LUT**
- Use same Exp2FixedSoftmax LUT for rescaling
- `exp2_ratio = LUT[delta_m_frac] >> delta_m_int` where delta_m = m_old - m_new
- More FA2-faithful but adds complexity

**Option C: FP32 Rescaling**
- Use FP32 for just the rescale factor computation
- Hybrid approach, simpler correctness

**Recommendation**: Option A for v1 (lazy rescaling), Option B as optimization.

---

## Prefill vs Decode Execution Strategies

The Q16 fused attention kernel must handle two fundamentally different execution modes with distinct computational characteristics:

### Comparison Overview

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Query count | `seq_len_q` queries (e.g., 512) | 1 query |
| Core operation | **GEMM** (matrix × matrix) | **GEMV** (matrix × vector) |
| Q×K^T shape | [Br, head_dim] × [kv_len, head_dim]^T → [Br, kv_len] | [1, head_dim] × [kv_len, head_dim]^T → [1, kv_len] |
| Softmax shape | [Br, kv_len] tile | [1, kv_len] single row |
| P×V shape | [Br, kv_len] × [kv_len, head_dim] → [Br, head_dim] | [1, kv_len] × [kv_len, head_dim] → [1, head_dim] |
| Memory bound? | Compute-bound (GEMM) | Memory-bound (streaming KV) |
| K/V reuse | High (amortized over Br queries) | Low (single use per token) |
| Optimal strategy | FA2-style tiling, cache blocking | Streaming, minimize latency |

### Prefill Mode: GEMM-Centric

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREFILL: GEMM-CENTRIC PATH                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: Q[seq_len_q, num_heads, head_dim], K/V[kv_len, ...]                 │
│                                                                              │
│  for each query_tile (Br queries):                                          │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │ Q_tile: [Br, head_dim] - multiple queries loaded once            │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│    for each kv_tile (Bc keys):                                              │
│      ┌────────────────────────────────────────────────────────────────┐    │
│      │ STEP 1: Batched QK Dot Product (GEMM)                          │    │
│      │   S[Br, Bc] = Q_tile × K_tile^T                                │    │
│      │   • Uses vpdpwssd with register blocking                       │    │
│      │   • K_tile loaded once, reused across Br queries               │    │
│      │   • Output: Br × Bc INT32 scores                               │    │
│      └────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│      ┌────────────────────────────────────────────────────────────────┐    │
│      │ STEP 2: Batched Exp2FixedSoftmax                               │    │
│      │   P[Br, Bc] = Exp2FixedSoftmax(S, m_tile)                     │    │
│      │   • Per-row max tracking (Br rows)                            │    │
│      │   • 256-entry exp2 LUT lookup (vpgatherdd)                    │    │
│      │   • Output: Br × Bc INT16 weights [0, 32767]                  │    │
│      └────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│      ┌────────────────────────────────────────────────────────────────┐    │
│      │ STEP 3: Batched PV Accumulation (GEMM-style)                   │    │
│      │   O_tile[Br, head_dim] += P[Br, Bc] × V_tile[Bc, head_dim]    │    │
│      │   • V_tile loaded once, reused across Br queries               │    │
│      │   • INT16 × INT16 → INT64 accumulation                        │    │
│      └────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│    end kv_tile loop                                                         │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐    │
│    │ STEP 4-6: Batched Wo Projection + Residual (GEMM)                │    │
│    │   proj[Br, d_model] = O_norm[Br, n_heads*head_dim] × Wo^T       │    │
│    │   residual[Br, d_model] += proj                                  │    │
│    └──────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  end query_tile loop                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Prefill Tiling Strategy**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Br (query tile) | 16-32 | Balance K/V reuse vs register pressure |
| Bc (kv tile) | 64-128 | Fit S/P tiles + V tile in L2 |
| Register blocking | 4×4 or 8×4 | GEMM micro-tile for vpdpwssd |

**Prefill Characteristics**:
- **Compute-bound**: High arithmetic intensity from GEMM operations
- **K/V amortization**: Each K/V element used Br times → worthwhile to optimize loads
- **Cache blocking critical**: S/P/V tiles must fit in L2 together
- **Parallelism**: Parallelize over query tiles (OpenMP for loop)

### Decode Mode: GEMV-Centric (Streaming)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DECODE: GEMV-CENTRIC PATH                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: q[1, num_heads, head_dim], K/V[kv_len, ...] (from KV cache)         │
│                                                                              │
│  // NO query tiling - single query                                          │
│  // Focus: stream through KV cache efficiently                              │
│                                                                              │
│  for each kv_chunk (Bc keys, streaming):                                    │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │ STEP 1: QK Dot Product (GEMV - single row output)                  │   │
│    │   s[Bc] = q × K_chunk^T                                            │   │
│    │   • q stays in registers (head_dim values)                         │   │
│    │   • Stream K_chunk from memory                                     │   │
│    │   • Output: Bc INT32 scores                                        │   │
│    │   • Bc chosen to maximize memory bandwidth utilization             │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │ STEP 2: Online Exp2FixedSoftmax (single row)                       │   │
│    │   p[Bc] = Exp2FixedSoftmax(s, m)                                   │   │
│    │   • Single max tracker (scalar m)                                  │   │
│    │   • 256-entry exp2 LUT lookup                                      │   │
│    │   • Update running softmax state (m, l)                            │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                          │                                                   │
│                          ▼                                                   │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │ STEP 3: PV Weighted Sum (streaming reduction)                      │   │
│    │   o[head_dim] += Σ(p[i] × V[i, :])                                │   │
│    │   • Stream V alongside K (same memory access pattern)              │   │
│    │   • Accumulate into head_dim-sized register buffer                │   │
│    │   • INT64 accumulators for precision                               │   │
│    └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  end kv_chunk loop                                                          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ STEP 4-6: Wo Projection + Residual (GEMV - single row)             │     │
│  │   proj[d_model] = o_norm[n_heads*head_dim] × Wo^T                  │     │
│  │   residual[d_model] += proj                                         │     │
│  │   • Single row output - GEMV not GEMM                               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Decode Characteristics**:
- **Memory-bound**: Single query → low arithmetic intensity, bottlenecked by KV cache reads
- **Latency-critical**: Single token generation, minimize time-to-first-byte
- **No query tiling**: Single query means Br=1 always
- **Streaming focus**: Optimize for sequential KV cache traversal
- **Register allocation**: q vector stays in registers, stream K/V from memory

**Decode Chunk Size**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Bc (kv chunk) | 256-512 | Large enough to amortize loop overhead |
| q in registers | zmm0-7 | head_dim=128 → 128 INT16 = 4 ZMM registers |
| Accumulators | zmm8-15 | head_dim=128 INT64 → need 8 ZMM for O accumulator |

### Microkernel Variants

Given the fundamental differences, we define **separate microkernel variants** for prefill and decode:

```
src/v2/kernels/cpu/attention/q16_1/
├── ref/                                    # Scalar reference implementation
│   ├── Q16FusedAttentionRef.h             # Scalar reference header
│   ├── Q16FusedAttentionRef.cpp           # Scalar reference orchestrator
│   └── microkernels/                       # Reference microkernel implementations
│       ├── Q16DotProductRef.h              # Q×K^T dot product
│       ├── Q16DotProductRef.cpp
│       ├── Exp2FixedSoftmaxRef.h           # exp2 LUT-based softmax
│       ├── Exp2FixedSoftmaxRef.cpp
│       ├── PVAccumulateRef.h               # P×V weighted accumulation
│       ├── PVAccumulateRef.cpp
│       ├── WoProjectionVNNIRef.h           # Wo projection → Q16_1 output
│       ├── WoProjectionVNNIRef.cpp         # INT32 → requant → Q16_1
│       ├── Int8RequantRef.h                # INT16→INT8 requantization
│       └── Int8RequantRef.cpp
│
└── jit/                                    # JIT-compiled implementation
    ├── JitQ16FusedAttention.h              # Fused macro kernel (composes microkernels)
    └── microkernels/                       # JIT microkernel implementations
        ├── JitQ16DotProduct.h              # Q×K^T JIT microkernel
        ├── JitExp2FixedSoftmax.h           # exp2 softmax JIT microkernel
        ├── JitPVAccumulate.h               # P×V JIT microkernel
        └── JitWoProjectionVNNI.h           # Wo projection → Q16_1 output
```

**Note on ResidualAddQ16**: This is NOT a standalone microkernel file. The residual add
uses `SIMDHelpers.h::q16_1_add_q16_1()` which is a proven all-integer operation:
  - Wo projection outputs Q16_1 (INT32 → requant → Q16_1)
  - Residual add: Q16_1 + Q16_1 → Q16_1 (dequant both, add, requant)
  - **NO FP32 in the data path** - scales are FP32 but arithmetic is integer

### Kernel Selection Logic

```cpp
bool Q16FusedAttentionKernel::compute(
    TensorBase* Q,
    TensorBase* K,
    TensorBase* V,
    TensorBase* Wo,
    TensorBase* residual,
    int seq_len_q,
    int kv_len,
    bool causal,
    int position_offset)
{
    // Dispatch based on query count
    if (seq_len_q == 1) {
        // DECODE PATH: Single query, GEMV-centric
        // Optimized for latency, streaming KV access
        return config_.use_jit 
            ? compute_jit_decode(Q, K, V, Wo, residual, kv_len, causal, position_offset)
            : compute_ref_decode(Q, K, V, Wo, residual, kv_len, causal, position_offset);
    } else {
        // PREFILL PATH: Multiple queries, GEMM-centric
        // Optimized for throughput, cache blocking
        return config_.use_jit
            ? compute_jit_prefill(Q, K, V, Wo, residual, seq_len_q, kv_len, causal)
            : compute_ref_prefill(Q, K, V, Wo, residual, seq_len_q, kv_len, causal);
    }
}
```

### Performance Targets by Mode

| Metric | Prefill Target | Decode Target |
|--------|----------------|---------------|
| Primary metric | Throughput (tok/s) | Latency (ms/tok) |
| Bottleneck | Compute (FLOPS) | Memory bandwidth |
| Target improvement | ≥15% over hybrid | ≤1.2× current latency |
| K/V cache access | Amortized (Br reuse) | Streaming (1× use) |
| Parallelism | Query-parallel | Head-parallel |

### Register Allocation Comparison

**Prefill Register Layout** (GEMM micro-tile 4×4):
```
ZMM Registers (32 total):
├── zmm0-3:   Q tile buffer (4 rows × head_dim packed)
├── zmm4-7:   K tile buffer (4 cols × head_dim packed)
├── zmm8-11:  S tile accumulators (4×4 INT32 scores)
├── zmm12-15: P tile (4×4 INT16 softmax weights)
├── zmm16-23: O accumulators (4 rows × head_dim INT64)
├── zmm24-27: V tile buffer (for PV accumulation)
└── zmm28-31: Constants (scale, LUT base, masks)
```

**Decode Register Layout** (GEMV streaming):
```
ZMM Registers (32 total):
├── zmm0-3:   q vector (head_dim=128 → 4 ZMM of INT16)
├── zmm4-7:   K chunk buffer (streaming, 4 rows at a time)
├── zmm8-11:  s scores (Bc INT32 values, chunked)
├── zmm12-15: p weights (Bc INT16 values, chunked)
├── zmm16-23: O accumulator (head_dim INT64, 8 ZMM)
├── zmm24-27: V chunk buffer (streaming, same pattern as K)
├── zmm28:    m (current max, broadcast)
├── zmm29:    l (current sum, broadcast)
└── zmm30-31: Constants (scale, LUT base)
```

---

## Fused Kernel Design

### Scalar Reference Kernel Interface

The interface follows existing patterns in the codebase (see `FusedAttentionWoKernel.h`):

```cpp
#pragma once

#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include <memory>
#include <cmath>

namespace llaminar2
{

/**
 * @file Q16FusedAttentionKernel.h
 * @brief Q16 FA2-style fused attention + Wo + residual kernel
 *
 * Single kernel call for complete attention block:
 *   output = residual + Wo × softmax(Q × K^T / sqrt(d)) × V
 *
 * All computation in integer domain until final output.
 * Follows the same pattern as FusedAttentionWoKernel for consistency.
 */
class Q16FusedAttentionKernel
{
public:
    /**
     * @brief Configuration for the fused Q16 attention kernel
     */
    struct Config
    {
        int num_heads = 0;       ///< Number of query heads
        int num_kv_heads = 0;    ///< Number of KV heads (GQA support)
        int head_dim = 64;       ///< Dimension per head
        int d_model = 0;         ///< Model dimension (num_heads * head_dim)
        int Br = 16;             ///< Query tile size (FA2 tiling)
        int Bc = 64;             ///< KV tile size (FA2 tiling)
        bool use_jit = true;     ///< Use JIT kernel (false = scalar reference)
    };

    explicit Q16FusedAttentionKernel(const Config& config)
        : config_(config)
        , scale_(1.0f / std::sqrt(static_cast<float>(config.head_dim)))
    {
        LOG_DEBUG("Q16FusedAttentionKernel created: heads=" << config.num_heads
                  << "/" << config.num_kv_heads << ", head_dim=" << config.head_dim
                  << ", d_model=" << config.d_model
                  << ", tiling=(" << config.Br << "×" << config.Bc << ")"
                  << ", jit=" << config.use_jit);
    }

    /**
     * @brief Compute fused attention + Wo projection + residual add
     *
     * @param Q Query tensor Q16_1 [seq_len_q, num_heads * head_dim]
     * @param K Key tensor Q16_1 [kv_len, num_kv_heads * head_dim] (from KV cache)
     * @param V Value tensor Q16_1 [kv_len, num_kv_heads * head_dim] (from KV cache)
     * @param Wo Output projection weight (Q8_0 or other quantized format)
     * @param residual Input/output Q16_1 [seq_len_q, d_model] (modified in-place)
     * @param seq_len_q Query sequence length
     * @param kv_len Key/Value sequence length
     * @param causal Whether to apply causal masking
     * @param position_offset Position offset for causal mask (decode mode)
     * @return true on success
     */
    bool compute(
        TensorBase* Q,
        TensorBase* K,
        TensorBase* V,
        TensorBase* Wo,
        TensorBase* residual,
        int seq_len_q,
        int kv_len,
        bool causal = true,
        int position_offset = 0);

private:
    Config config_;
    float scale_;

    // Dispatch to reference or JIT implementation
    bool compute_reference(
        Q16_1Tensor* Q,
        Q16_1Tensor* K,
        Q16_1Tensor* V,
        TensorBase* Wo,
        Q16_1Tensor* residual,
        int seq_len_q,
        int kv_len,
        bool causal,
        int position_offset);

    bool compute_jit(
        Q16_1Tensor* Q,
        Q16_1Tensor* K,
        Q16_1Tensor* V,
        TensorBase* Wo,
        Q16_1Tensor* residual,
        int seq_len_q,
        int kv_len,
        bool causal,
        int position_offset);
};

} // namespace llaminar2
```

### Reference Implementation Parameters (following FusedAttentionWoRef.h pattern)

```cpp
namespace llaminar::v2::kernels::q16_1
{

// Use Q16_1Block from tensors namespace
using llaminar2::Q16_1Block;

/**
 * @brief Parameters for the fused Q16 attention + Wo projection kernel.
 *
 * Follows the same pattern as FusedAttentionWoParams for consistency.
 */
struct Q16FusedAttentionParams
{
    // ============== Input tensors (Q16_1 blocks) ==============
    
    const Q16_1Block* Q;       ///< Query blocks [seq_len_q, num_heads, head_dim/16 blocks]
    const Q16_1Block* K;       ///< Key blocks [kv_len, num_kv_heads, head_dim/16 blocks]
    const Q16_1Block* V;       ///< Value blocks [kv_len, num_kv_heads, head_dim/16 blocks]
    
    // ============== Weight tensor ==============
    
    const void* Wo;            ///< Wo weight matrix (Q8_0 or other format)
    TensorType wo_type;        ///< Type of Wo weights
    
    // ============== Output tensor (in-place) ==============
    
    Q16_1Block* residual;      ///< Input/output residual [seq_len_q, d_model/16 blocks]
    
    // ============== Dimensions ==============
    
    int seq_len_q = 0;         ///< Number of query positions
    int kv_len = 0;            ///< Number of KV positions
    int num_heads = 0;         ///< Number of query heads
    int num_kv_heads = 0;      ///< Number of KV heads (for GQA)
    int head_dim = 0;          ///< Dimension per head (e.g., 128)
    int d_model = 0;           ///< Model dimension
    
    // ============== FA2 Tiling ==============
    
    int Br = 16;               ///< Query tile size
    int Bc = 64;               ///< KV tile size
    
    // ============== Options ==============
    
    bool causal = true;        ///< Apply causal masking
    int position_offset = 0;   ///< Position offset for decode mode
    float scale = 0.0f;        ///< 1/sqrt(head_dim), computed if 0
};

/**
 * @brief Scalar reference implementation of Q16 fused attention
 *
 * @param params Kernel parameters
 * @return true on success
 */
bool q16_fused_attention_reference(const Q16FusedAttentionParams& params);

} // namespace llaminar::v2::kernels::q16_1
```

### Scalar Reference Implementation Structure

```cpp
// File: src/v2/kernels/cpu/attention/q16_1/ref/Q16FusedAttentionRef.cpp

#include "Q16FusedAttentionRef.h"
#include <algorithm>
#include <cstdint>
#include <cmath>

namespace llaminar::v2::kernels::q16_1
{

// Exp2FixedSoftmax LUT - 256-entry exp2 LUT with 30-bit precision
// lut[i] = round(2^30 * 2^(-i/256)) for i in [0, 255]
alignas(64) static uint32_t EXP2_LUT[256];
static bool exp2_lut_initialized = false;

void init_exp2_lut() {
    if (exp2_lut_initialized) return;
    for (int i = 0; i < 256; ++i) {
        double val = std::pow(2.0, -static_cast<double>(i) / 256.0);
        EXP2_LUT[i] = static_cast<uint32_t>(val * (1ULL << 30));
    }
    exp2_lut_initialized = true;
}

// Helper: decode Q16_1Block to get INT16 value at position
inline int16_t q16_block_value(const Q16_1Block* blocks, int row, int col, 
                                int head, int head_dim, int blocks_per_row)
{
    // Q16_1Block has 16 INT16 values per block
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;  // 16
    
    // Compute linear column index within head
    int linear_col = head * head_dim + col;
    int block_idx = linear_col / BLOCK_SIZE;
    int elem_idx = linear_col % BLOCK_SIZE;
    
    const Q16_1Block& block = blocks[row * blocks_per_row + block_idx];
    return block.qs[elem_idx];
}

bool q16_fused_attention_reference(const Q16FusedAttentionParams& params)
{
    const int Br = params.Br;
    const int Bc = params.Bc;
    const int head_dim = params.head_dim;
    const int num_heads = params.num_heads;
    const int num_kv_heads = params.num_kv_heads;
    const int seq_len_q = params.seq_len_q;
    const int kv_len = params.kv_len;
    const int d_model = params.d_model;
    const bool causal = params.causal;
    const int position_offset = params.position_offset;
    
    // Compute scale if not provided
    const float scale = (params.scale > 0) ? params.scale 
                                           : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    
    // GQA: heads per KV group
    const int heads_per_kv = num_heads / num_kv_heads;
    
    // Block layout (Q16_1Block has BLOCK_SIZE=16)
    constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
    const int q_blocks_per_row = (num_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int kv_blocks_per_row = (num_kv_heads * head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int out_blocks_per_row = (d_model + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Scratch space for attention output (all heads concatenated)
    // In full impl, this would be per-thread
    std::vector<int64_t> O_all_heads(num_heads * head_dim, 0);
    
    // Per-head processing
    for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / heads_per_kv;  // GQA mapping
        
        // Query tile loop
        for (int q_start = 0; q_start < seq_len_q; q_start += Br) {
            int q_end = std::min(q_start + Br, seq_len_q);
            int Br_actual = q_end - q_start;
            
            // Initialize accumulators for this query tile
            constexpr int HEAD_DIM_MAX = 256;
            alignas(64) int64_t O_acc[HEAD_DIM_MAX] = {0};  // Per-query row
            int32_t m = INT32_MIN;  // Running max
            int64_t l = 0;          // Running sum
            
            // KV tile loop
            int kv_end_limit = causal ? std::min(q_start + position_offset + Br_actual, kv_len) 
                                      : kv_len;
            for (int kv_start = 0; kv_start < kv_end_limit; kv_start += Bc) {
                int kv_end = std::min(kv_start + Bc, kv_end_limit);
                int Bc_actual = kv_end - kv_start;
                
                // ════════════════════════════════════════════════════
                // STEP 1: Compute Q×K^T scores (INT32)
                // ════════════════════════════════════════════════════
                alignas(64) int32_t S[Bc_actual];
                for (int ki = 0; ki < Bc_actual; ++ki) {
                    // Causal mask check
                    if (causal && (kv_start + ki) > (q_start + position_offset)) {
                        S[ki] = INT32_MIN;  // Masked
                        continue;
                    }
                    
                    // Q16×Q16 dot product
                    int32_t dot = 0;
                    for (int d = 0; d < head_dim; ++d) {
                        int16_t q_val = q16_block_value(params.Q, q_start, d, h, 
                                                        head_dim, q_blocks_per_row);
                        int16_t k_val = q16_block_value(params.K, kv_start + ki, d, 
                                                        kv_head, head_dim, kv_blocks_per_row);
                        dot += static_cast<int32_t>(q_val) * k_val;
                    }
                    S[ki] = dot;
                }
                
                // ════════════════════════════════════════════════════
                // STEP 2: Online Exp2FixedSoftmax
                // ════════════════════════════════════════════════════
                init_exp2_lut();  // Ensure LUT is initialized
                alignas(64) int16_t P[Bc_actual];
                
                // Find new max for this tile
                int32_t m_new = m;
                for (int ki = 0; ki < Bc_actual; ++ki) {
                    if (S[ki] != INT32_MIN) {
                        m_new = std::max(m_new, S[ki]);
                    }
                }
                
                // Compute Exp2FixedSoftmax
                constexpr float LOG2_E = 1.4426950408889634f;
                constexpr int16_t WEIGHT_MAX = 32767;  // VNNI-compatible INT16
                float alpha = 1.0f / 32768.0f;  // Typical Q16 scale
                float beta = alpha * LOG2_E * scale;
                
                // First pass: compute exp2 values
                std::vector<uint64_t> exp_vals(Bc_actual);
                uint64_t sum_exp = 0;
                for (int ki = 0; ki < Bc_actual; ++ki) {
                    if (S[ki] == INT32_MIN) {
                        exp_vals[ki] = 0;  // Masked
                        continue;
                    }
                    
                    int32_t delta = m_new - S[ki];
                    float t = delta * beta;
                    int32_t t_fixed = static_cast<int32_t>(t * 256.0f);
                    int32_t ip = t_fixed >> 8;
                    int32_t frac = t_fixed & 255;
                    
                    if (ip >= 30) {
                        exp_vals[ki] = 0;
                    } else {
                        exp_vals[ki] = static_cast<uint64_t>(EXP2_LUT[frac]) >> ip;
                    }
                    sum_exp += exp_vals[ki];
                }
                
                // Second pass: normalize to INT16 weights
                int32_t tile_sum = 0;
                for (int ki = 0; ki < Bc_actual; ++ki) {
                    if (sum_exp > 0) {
                        P[ki] = static_cast<int16_t>((exp_vals[ki] * WEIGHT_MAX) / sum_exp);
                    } else {
                        P[ki] = 0;
                    }
                    tile_sum += P[ki];
                }
                
                // Update running state (lazy rescaling - simplified)
                m = m_new;
                l += tile_sum;
                
                // ════════════════════════════════════════════════════
                // STEP 3: Accumulate P × V (INT64)
                // ════════════════════════════════════════════════════
                for (int ki = 0; ki < Bc_actual; ++ki) {
                    int16_t w = P[ki];  // INT16 weight from Exp2FixedSoftmax
                    if (w == 0) continue;
                    
                    for (int d = 0; d < head_dim; ++d) {
                        int16_t v_val = q16_block_value(params.V, kv_start + ki, d,
                                                        kv_head, head_dim, kv_blocks_per_row);
                        O_acc[d] += static_cast<int64_t>(w) * v_val;
                    }
                }
                
            }  // end kv_tile loop
            
            // ════════════════════════════════════════════════════
            // STEP 4: Normalize O by softmax sum
            // ════════════════════════════════════════════════════
            for (int d = 0; d < head_dim; ++d) {
                int32_t normalized = (l > 0) 
                    ? static_cast<int32_t>(O_acc[d] / l)
                    : 0;
                O_all_heads[h * head_dim + d] = normalized;
            }
            
        }  // end query_tile loop
    }  // end head loop
    
    // ════════════════════════════════════════════════════
    // STEP 5: Fused Wo Projection (simplified - single query)
    // ════════════════════════════════════════════════════
    // NOTE: Full implementation would handle batched queries
    // and use efficient VNNI-style GEMM for Wo projection
    
    std::vector<int32_t> proj(d_model, 0);
    // ... Wo projection implementation ...
    
    // ════════════════════════════════════════════════════
    // STEP 6: Fused Residual Add + Requantize to Q16_1
    // ════════════════════════════════════════════════════
    // ... residual fusion implementation ...
    
    return true;
}

} // namespace llaminar::v2::kernels::q16_1
```

---

## Development Methodology

### Scalar → JIT Validation Approach

This kernel will be developed in two phases:

**Phase 1: Scalar C++ Reference**
- Fully functional, correct implementation
- Clear, readable code (no SIMD intrinsics)
- Used as ground truth for JIT validation
- Test coverage for all edge cases

**Phase 2: JIT Kernel Implementation**
- Xbyak-based code generation
- AVX-512 VNNI intrinsics
- Must produce bit-identical output to scalar reference
- Performance target: 10-50× speedup over scalar

### Validation Test Matrix

| Test Case | Scalar | JIT | Metric |
|-----------|--------|-----|--------|
| Single query, short KV | ✓ | ✓ | Bit-exact match |
| Single query, long KV (4096) | ✓ | ✓ | Bit-exact match |
| Batch query (prefill) | ✓ | ✓ | Bit-exact match |
| Causal mask | ✓ | ✓ | Bit-exact match |
| GQA (grouped query) | ✓ | ✓ | Bit-exact match |
| Edge: seq_len=1 | ✓ | ✓ | Bit-exact match |
| Edge: kv_len=1 | ✓ | ✓ | Bit-exact match |
| Edge: all masked | ✓ | ✓ | Bit-exact match |

### File Structure

```
src/v2/kernels/cpu/attention/q16_1/
├── ref/                                    # Scalar reference implementation
│   ├── Q16FusedAttentionRef.h             # Scalar reference header
│   ├── Q16FusedAttentionRef.cpp           # Scalar reference orchestrator
│   └── microkernels/                       # Reference microkernel implementations
│       ├── Q16DotProductRef.h              # Q×K^T dot product reference
│       ├── Q16DotProductRef.cpp
│       ├── Exp2FixedSoftmaxRef.h           # exp2 LUT-based softmax reference
│       ├── Exp2FixedSoftmaxRef.cpp
│       ├── PVAccumulateRef.h               # P×V weighted accumulation reference
│       ├── PVAccumulateRef.cpp
│       ├── WoProjectionVNNIRef.h           # Wo projection → Q16_1 output
│       ├── WoProjectionVNNIRef.cpp         # INT32 → requant → Q16_1
│       ├── Int8RequantRef.h                # INT16→INT8 requantization
│       └── Int8RequantRef.cpp
│
└── jit/                                    # JIT-compiled implementation
    ├── JitQ16FusedAttention.h              # Fused macro kernel (composes microkernels)
    └── microkernels/                       # JIT microkernel implementations
        ├── JitQ16DotProduct.h              # Q×K^T JIT microkernel (scaffolding)
        ├── JitExp2FixedSoftmax.h           # exp2 softmax JIT microkernel (scaffolding)
        ├── JitPVAccumulate.h               # P×V JIT microkernel (scaffolding)
        └── JitWoProjectionVNNI.h           # Wo projection → Q16_1 (scaffolding)

tests/v2/unit/jit/
└── Test__JitQ16MicrokernelParity.cpp      # JIT vs scalar parity tests (7/7 passing)
```

**Note on ResidualAddQ16**: The residual add uses `SIMDHelpers.h::q16_1_add_q16_1()`
which is a proven all-integer operation:
  - Wo projection outputs Q16_1 (INT32 → requant → Q16_1)
  - Residual add: Q16_1 + Q16_1 → Q16_1 (dequant both, add in FP32, requant)
  - **NO FP32 intermediate data** - FP32 only used for scale reconciliation

**Directory Layout Rationale**:
- `ref/` and `jit/` share the same structure for consistency
- Root level contains the macro kernel (orchestrator/composition)
- `microkernels/` contains individual kernel implementations
- Both macro kernels compose their respective microkernels

**Current Status (December 28, 2025)**:
- Reference microkernels: Implemented and tested
- JIT scaffolding: Complete with reference delegation
- JIT code generation: TODO (Phase 2)

---

## Component Specifications

### μK1: Q16 Dot Product (`Q16DotProductMicrokernel`)

**Purpose**: Compute Q×K^T attention scores in INT32

**Input**:
- `Q_int16`: INT16 query vector [head_dim] with scale `d_Q`
- `K_int16`: INT16 key matrix [kv_len × head_dim] with per-row scales `d_K[kv]`

**Output**:
- `scores_int32`: INT32 attention scores [kv_len]
- `combined_scale`: FP32 scale factor for later stages

**Algorithm**:
```cpp
// For each KV position
for (int kv = 0; kv < kv_len; ++kv) {
    int32_t dot = 0;
    
    // VNNI-accelerated dot product (16 elements per vpdpwssd)
    for (int d = 0; d < head_dim; d += 16) {
        __m512i q_vec = _mm512_loadu_si512(&Q_int16[d]);
        __m512i k_vec = _mm512_loadu_si512(&K_int16[kv * head_dim + d]);
        dot = _mm512_dpwssd_epi32(_mm512_set1_epi32(dot), q_vec, k_vec);
    }
    
    scores_int32[kv] = dot;
}

// Combined scale: d_Q * d_K * (1/sqrt(head_dim))
combined_scale = d_Q * d_K_avg * inv_sqrt_d;
```

**Key Considerations**:
- `vpdpwssd` computes INT16×INT16→INT32 with saturation
- Head dim 128: max |dot| ≈ 128 × 32767² ≈ 1.37e14 — **needs INT64 accumulator**!
- Alternative: Scale Q/K to [-127, 127] range pre-dot

**Accumulator Overflow Analysis**:
```
INT16 × INT16 = INT32 per element (max 2^30)
Sum of 128 elements: max 2^30 × 128 = 2^37 — OVERFLOWS INT32!

Solution: Use wider accumulator or prescale inputs
Option A: INT64 accumulator (slower but safe)
Option B: Prescale Q,K by 1/16 → max sum = 2^37 / 256 = 2^29 ✓
Option C: Use vpmaddwd + horizontal add pattern
```

---

### μK2: Exp2FixedSoftmax (`Exp2FixedSoftmaxMicrokernel`)

**Purpose**: Integer-domain softmax using exp2 (base-2 exponential) LUT approximation

**Innovation**: Unlike the IntAttention paper's 32-entry LUT with natural exp, we use a **256-entry exp2 LUT** with 30-bit precision. This provides:
- Higher precision (256 entries vs 32)
- Cleaner integer decomposition (powers of 2 via bit shifts)
- VNNI-compatible INT16 output [0, 32767]

**Input**:
- `scores_int32`: INT32 attention scores [kv_len]
- `alpha`: Scale factor from μK1 (combined Q/K scale)

**Output**:
- `weights_int16`: INT16 attention weights [kv_len] in range [0, 32767]
- `sum_weights`: INT32 sum of weights (for normalization)

**Algorithm**:
```cpp
// 256-entry LUT for 2^(-frac/256) where frac in [0, 255]
// Each entry has 30-bit precision
constexpr int LUT_SIZE = 256;
constexpr int LUT_VALUE_BITS = 30;
static uint32_t EXP2_LUT[256];  // Initialized: lut[i] = 2^(-i/256) * 2^30

void exp2_fixed_softmax(
    const int32_t* scores,
    int16_t* weights,
    int n,
    float alpha,
    int32_t* sum_out)
{
    constexpr float LOG2_E = 1.4426950408889634f;
    constexpr int16_t WEIGHT_MAX = 32767;  // VNNI-compatible (signed INT16)
    
    // Step 1: Find max score (excluding masked positions)
    int32_t max_score = INT32_MIN;
    for (int i = 0; i < n; ++i) {
        if (scores[i] != INT32_MIN)  // INT32_MIN = masked
            max_score = std::max(max_score, scores[i]);
    }
    
    // Step 2: Compute exp2 values using LUT
    float beta = alpha * LOG2_E;  // Convert ln domain to log2 domain
    std::vector<uint64_t> exp_vals(n);
    uint64_t sum_exp = 0;
    
    for (int i = 0; i < n; ++i) {
        if (scores[i] == INT32_MIN) {
            exp_vals[i] = 0;  // Masked position
            continue;
        }
        
        int32_t delta = max_score - scores[i];  // Always >= 0
        float t = delta * beta;                  // log2 domain
        
        // Decompose t into integer and fractional parts (8 frac bits)
        int32_t t_fixed = static_cast<int32_t>(t * 256.0f);
        int32_t ip = t_fixed >> 8;     // Integer part (shift amount)
        int32_t frac = t_fixed & 255;  // Fractional part [0, 255]
        
        // exp2(-t) = LUT[frac] >> ip
        if (ip >= 30) {
            exp_vals[i] = 0;  // Underflow to zero
        } else {
            exp_vals[i] = static_cast<uint64_t>(EXP2_LUT[frac]) >> ip;
        }
        sum_exp += exp_vals[i];
    }
    
    // Step 3: Normalize to INT16 weights [0, 32767]
    int32_t weight_sum = 0;
    for (int i = 0; i < n; ++i) {
        if (sum_exp > 0) {
            weights[i] = static_cast<int16_t>(
                (exp_vals[i] * WEIGHT_MAX) / sum_exp);
        } else {
            weights[i] = 0;
        }
        weight_sum += weights[i];
    }
    
    if (sum_out) *sum_out = weight_sum;
}
```

**Exp2 vs Natural Exp Advantages**:

| Aspect | IntAttention (exp) | Exp2FixedSoftmax (exp2) |
|--------|-------------------|-------------------------|
| LUT entries | 32 | 256 |
| Precision | ~5 bits | ~8 bits fractional |
| Integer part | Requires division | Simple bit shift |
| LUT value precision | 16-bit | 30-bit |
| Output range | [0, 65535] UINT16 | [0, 32767] INT16 |
| VNNI compatible | No (unsigned) | Yes (signed) |

**Memory Impact**: LUT grows from 32 bytes to 64 bytes — negligible.

---

### μK3: Integer V Accumulation (`IntVAccumMicrokernel`)

**Purpose**: Weighted sum of V vectors in integer domain

**Input**:
- `weights_int16`: INT16 attention weights [kv_len] in range [0, 32767]
- `V_int16`: INT16 value matrix [kv_len × head_dim] with scales `d_V[kv]`
- `sum_weights`: INT64 sum from μK2

**Output**:
- `context_int32`: INT32 attention context [head_dim]
- `context_scale`: FP32 scale factor

**Algorithm**:
```cpp
void int_v_accum(
    const int16_t* weights,
    const int16_t* V,
    int64_t sum_weights,
    int kv_len,
    int head_dim,
    int32_t* context_out,
    float* scale_out)
{
    // INT64 accumulators to avoid overflow
    // max: 32767 × 32767 × kv_len ≈ 2^30 × kv_len
    // For kv_len=4096: 2^42 — needs INT64!
    alignas(64) int64_t accum[HEAD_DIM_MAX] = {0};
    
    for (int kv = 0; kv < kv_len; ++kv) {
        int16_t w = weights[kv];
        if (w == 0) continue;  // Skip zero weights (masked positions)
        
        const int16_t* v_row = &V[kv * head_dim];
        
        // Vectorized: accum[d] += w * v_row[d]
        for (int d = 0; d < head_dim; d += 16) {
            __m512i v_vec = _mm512_loadu_si512(&v_row[d]);
            __m512i w_vec = _mm512_set1_epi32(w);
            
            // Sign-extend INT16 to INT32, multiply, accumulate to INT64
            __m512i v_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_vec, 0));
            __m512i v_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v_vec, 1));
            
            // ... accumulate to INT64 ...
        }
    }
    
    // Normalize: context = accum / sum_weights
    // This is integer division — some precision loss here
    for (int d = 0; d < head_dim; ++d) {
        context_out[d] = static_cast<int32_t>(accum[d] / sum_weights);
    }
    
    // Scale factor carries through
    *scale_out = d_V_avg;
}
```

**Optimization**: For decode (kv_len large, single query), consider tiled accumulation with partial sums.

---

### μK4: Integer RMSNorm (Optional) (`IntRMSNormMicrokernel`)

**Purpose**: Normalize attention context if required before Wo projection

**Note**: In standard Transformer, RMSNorm is applied to the residual AFTER attention output projection. This microkernel is included for flexibility but may be skipped.

**Algorithm**:
```cpp
void int_rmsnorm(
    const int32_t* input,
    int16_t* output,
    const float* gamma,  // FP32 gamma weights
    int dim,
    float* scale_out)
{
    // Compute sum of squares (INT64)
    int64_t sum_sq = 0;
    for (int d = 0; d < dim; ++d) {
        sum_sq += static_cast<int64_t>(input[d]) * input[d];
    }
    
    // Integer sqrt approximation
    // RMS = sqrt(sum_sq / dim)
    int64_t mean_sq = sum_sq / dim;
    int32_t rms = int_sqrt(mean_sq);  // Newton-Raphson or bit manipulation
    
    // Normalize and apply gamma
    // output[d] = (input[d] * gamma[d] / rms) quantized to INT16
    float inv_rms = 1.0f / (rms + 1e-6f);
    for (int d = 0; d < dim; ++d) {
        float normalized = input[d] * inv_rms * gamma[d];
        output[d] = static_cast<int16_t>(std::round(normalized * (*scale_out)));
    }
}
```

---

### μK5: Wo Projection with VPDPWSSD (INT16×INT16) (`WoProjectionVNNI`)

**Purpose**: Project attention context through Wo weights using VPDPWSSD for maximum precision,
with sign-extended INT8 weights and INT16 context.

**Key Insight**: INT8 context quantization loses too much precision!
```
WRONG:  INT32 context → INT8 (Q8_1) → VPDPBUSD  (loses 8 bits of precision!)
RIGHT:  INT32 context → INT16       → VPDPWSSD  (keeps 16 bits of precision!)
```

Model weights arrive as INT8 VNNI-packed, but we **sign-extend to INT16** at runtime.
This is essentially free (`vpmovsxbw` instruction).

**ALL-INTEGER Data Flow (MAXIMUM PRECISION)**:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VPDPWSSD (INT16×INT16) Wo PROJECTION → Q16_1 OUTPUT                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  P×V Output: INT32 accumulators                                             │
│       │                                                                     │
│       │  int32_context[num_heads × head_dim]                                │
│       │  + weight_sum (INT32, sum of softmax weights)                       │
│       │  + v_scale_product (FP32 scale, tracked separately)                 │
│       │                                                                     │
│       ▼                                                                     │
│  requantize_int32_to_int16_context() ─── KEEPS 16 BITS!                     │
│       │                                                                     │
│       │  Computes: maxabs of INT32 values                                   │
│       │  Scale: combined_scale = maxabs/32767 * v_scale / weight_sum        │
│       │  Output: INT16 packed context (16-bit precision preserved!)         │
│       │                                                                     │
│       ▼                                                                     │
│  INT16 context_packed[inner_dim]                                            │
│       │                                                                     │
│       └───────────────┬────────────────┐                                    │
│                       │                │                                    │
│                       ▼                ▼                                    │
│               Wo_int8[K][N] ──► vpmovsxbw ──► Wo_int16[K][N]                │
│                       │         (sign-extend, FREE!)                        │
│                       ▼                                                     │
│               VPDPWSSD (INT16 × INT16 → INT32) ← Maximum precision!         │
│                       │                                                     │
│                       ▼                                                     │
│               INT32 GEMM output[d_model]                                    │
│                       │                                                     │
│                       ▼                                                     │
│               requantize_int32_to_q16_1() ─── Integer-to-Integer            │
│                       │                                                     │
│                       ▼                                                     │
│               Q16_1 output[d_model / 32 blocks]                             │
│                       │                                                     │
│                       ▼                                                     │
│               q16_1_add_q16_1() ─── Native Q16_1 + Q16_1 residual!          │
│                       │             (from SIMDHelpers.h)                    │
│                       ▼                                                     │
│               Q16_1 updated_residual[d_model / 32 blocks]                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**VNNI Instruction Comparison**:
| Instruction | Operands | Use Case |
|-------------|----------|----------|
| VPDPBUSD | UINT8 × INT8 → INT32 | Original Q8_0 GEMM (loses context precision) |
| VPDPWSSD | INT16 × INT16 → INT32 | **Our choice** (preserves context precision) |

**Input**:
- `IntegerContext`: INT32 P×V accumulators + weight_sum + v_scale_product
- `Wo_packed`: QuantisedPackedWeights (INT8, sign-extended to INT16 at runtime)

**Output**:
- `Q16_1Projection`: Q16_1 blocks [(d_model + 31) / 32] - ready for native residual add!

**Algorithm**:
```cpp
void wo_projection_vpdpwssd_to_q16_1_gemv(
    const WoProjectionVNNIParams& params,
    const IntegerContext& context,      // INT32 from P×V
    Q16_1Projection& output)            // Q16_1 output
{
    // Step 1: Requantize INT32 context to INT16 (KEEP 16 bits!)
    std::vector<int16_t> context_int16(input_dim);
    requantize_int32_to_int16_context(context.int32_data, ...);

    // Step 2: VPDPWSSD GEMV (INT16 × INT16 → INT32)
    // Sign-extend INT8 weights to INT16 on the fly (vpmovsxbw)
    std::vector<int32_t> int32_output(d_model);
    for (int n = 0; n < d_model; ++n) {
        for (int k = 0; k < input_dim; ++k) {
            int16_t ctx_val = context_int16[k];
            int16_t w_int16 = static_cast<int16_t>(Wo_int8[k, n]);  // Sign-extend!
            int32_output[n] += ctx_val * w_int16;  // VPDPWSSD
        }
    }

    // Step 3: Requantize INT32 → Q16_1
    requantize_int32_to_q16_1(int32_output.data(), d_model, scale, output.blocks);
    
    // Now caller can do native Q16_1 residual add:
    // q16_1_add_q16_1(residual, output.blocks, residual, d_model);
}
```

**Key Benefits**:
- **2× context precision**: INT32→INT16 vs INT32→INT8
- **Lossless weight upscaling**: INT8→INT16 sign-extension is exact
- **Same throughput**: VPDPWSSD has same throughput as VPDPBUSD
- **Zero FP32 conversions** in entire attention → residual path
- **Native Q16_1 residual add**: Uses `q16_1_add_q16_1()` from SIMDHelpers.h
```

---

### μK6: Q16_1 Residual Add (`Q16ResidualAddMicrokernel`)

**Purpose**: Add Wo projection to Q16_1 residual, requantize output

**Input**:
- `residual_q16`: Q16_1 residual tensor [d_model]
- `projection_int32`: INT32 Wo output [d_model]
---

### μK6: Q16_1 + Q16_1 Residual Add (NATIVE - uses SIMDHelpers)

**Purpose**: Add Wo projection (Q16_1) to Q16_1 residual - uses existing native operation!

**Key Insight**: Since μK5 now outputs Q16_1 directly, we use the existing
`q16_1_add_q16_1()` from `SIMDHelpers.h` - NO custom microkernel needed!

**Input**:
- `residual_q16`: Q16_1 residual tensor [d_model / 32 blocks]
- `projection_q16`: Q16_1 Wo output [d_model / 32 blocks] (from μK5)

**Output**:
- `residual_q16`: Updated Q16_1 residual (in-place or to separate output)

**Usage** (no new code needed!):
```cpp
#include "tensors/SIMDHelpers.h"

void apply_attention_residual(
    Q16_1Block* residual,           // Q16_1 residual
    const Q16_1Block* projection,   // Q16_1 from wo_projection_vnni_to_q16_1_gemv
    int d_model)
{
    // Native Q16_1 + Q16_1 addition - SIMD accelerated!
    simd::q16_1_add_q16_1(residual, projection, residual, d_model);
    
    // That's it! No FP32 conversions, no custom logic.
}
```

**Why This Works**:
- `q16_1_add_q16_1` already implements the full algorithm:
  1. Dequant both blocks to FP32 in registers (minimal - just scale multiply)
  2. Add FP32 values
  3. Find new max_abs for requantization
  4. Requantize to Q16_1

**Performance**:
- AVX-512 vectorized (32 elements per iteration)
- Works entirely in registers
- No intermediate memory allocation
- Already battle-tested in typed residual pipeline

---

## Exp2FixedSoftmax Integration

### From IntAttention's IndexSoftmax to Exp2FixedSoftmax

The IntAttention paper uses a 32-entry natural exponential LUT. Our Exp2FixedSoftmax improves on this with:

| Aspect | IntAttention IndexSoftmax | Our Exp2FixedSoftmax |
|--------|--------------------------|---------------------|
| LUT entries | 32 | 256 |
| Base | e (natural exp) | 2 (exp2) |
| LUT value precision | 16-bit | 30-bit |
| Output format | UINT16 [0, 65535] | INT16 [0, 32767] |
| Integer decomposition | Division-based | Bit shift |
| VNNI compatibility | No (unsigned) | Yes (signed) |
| Precision | ~5 bits | ~8 bits fractional |

### Key Innovation: exp2 Decomposition

The natural exponential `exp(-x)` requires division to decompose into LUT index.
Our exp2 approach uses `2^(-x)` which decomposes cleanly:

```
exp(-x) = exp(-x)                    // Requires: x * c / (entries-1) for index
exp2(-x) = 2^(-x) = 2^(-ip) * 2^(-frac)  // Just: ip = x >> 8, frac = x & 255
         = LUT[frac] >> ip           // Bit shift for integer part!
```

### LUT Construction for Exp2FixedSoftmax

```cpp
// 256-entry LUT for 2^(-frac/256) with 30-bit precision
// Precomputed at initialization time
void init_exp2_lut(uint32_t* lut) {
    for (int i = 0; i < 256; ++i) {
        double val = std::pow(2.0, -static_cast<double>(i) / 256.0);
        lut[i] = static_cast<uint32_t>(val * (1ULL << 30));
    }
}

// Example values (first 16 entries):
// lut[0]   = 1073741824  // 2^30 * 2^(-0/256) = 2^30 * 1.0
// lut[16]  = 1027745383  // 2^30 * 2^(-16/256) ≈ 2^30 * 0.957
// lut[32]  =  983525516  // 2^30 * 2^(-32/256) ≈ 2^30 * 0.916
// lut[128] =  759250125  // 2^30 * 2^(-128/256) ≈ 2^30 * 0.707
// lut[255] =  537231780  // 2^30 * 2^(-255/256) ≈ 2^30 * 0.500
```

### VNNI Compatibility: INT16 vs UINT16

**Critical constraint**: VPDPWSSD interprets operands as **signed INT16**.

| Value | As UINT16 | As INT16 (VPDPWSSD) |
|-------|-----------|---------------------|
| 32767 | 32767 | 32767 ✓ |
| 32768 | 32768 | -32768 ✗ |
| 65535 | 65535 | -1 ✗ |

Our Exp2FixedSoftmax outputs INT16 in range [0, 32767], ensuring:
- Weights are always non-negative in P×V computation
- Full VNNI VPDPWSSD compatibility
- 15-bit precision (sufficient for softmax probabilities)

### Alternative: Hybrid Precision

For maximum flexibility, support configurable precision:

```cpp
enum class SoftmaxPrecision {
    INT16_VNNI,  // Default: INT16 [0, 32767], VNNI-compatible
    INT16_FP32,  // Same INT16 output, but use FP32 for P×V (future)
    INT8_FAST    // Lowest precision, fastest (for less critical layers)
};
```

---

## Q16 vs Q8 Precision Analysis

### Theoretical Precision Comparison

| Format | Range | Precision | Decimal Digits |
|--------|-------|-----------|----------------|
| Q8_0 | ±127 | 1/127 ≈ 0.79% | ~2.1 |
| Q8_1 | ±127 | 1/127 ≈ 0.79% | ~2.1 |
| Q16_1 | ±32767 | 1/32767 ≈ 0.003% | ~4.5 |
| FP16 | ±65504 | 2^-10 ≈ 0.1% | ~3.3 |
| FP32 | ±3.4e38 | 2^-23 ≈ 0.00001% | ~7.2 |

### Where Precision Matters Most

1. **Softmax probabilities** — Small differences in exp(-x) can flip argmax
2. **V accumulation** — Errors compound with kv_len
3. **Residual stream** — Carries information across layers

### Recommendation

**Use Q16_1 throughout attention** (Q, K, V, context, residual) with:
- INT16 softmax probabilities [0, 32767]
- INT64 accumulators where needed
- Single FP32 scale per tensor (not per-element)

This provides ~100× precision improvement over Q8 approaches with ~2× memory usage.

---

## Implementation Phases

### Phase 1: Scalar Reference - Decode Path ✅ COMPLETE

**Goal**: Implement scalar C++ reference for the **decode path** (GEMV-centric, single query)

**Rationale**: Start with decode because it's simpler (no query tiling) and immediately useful for inference testing.

**Principle**: Correctness over performance. Clear, readable, debuggable code.

- [x] **Task 1.1**: Create `Q16FusedAttentionRefDecode` class
  - Single query (seq_len_q=1) optimization
  - Streaming KV chunk processing (Bc=256)
  - Online softmax state (m, l scalars)
  - INT64 accumulator for O (head_dim values)
  - **Implemented in**: `Q16FusedAttentionRef.cpp` `flash_decode_single_head()`
  
- [x] **Task 1.2**: Implement Exp2FixedSoftmax for single row
  - 256-entry exp2 LUT with 30-bit precision
  - Single max tracker (not Br max values)
  - INT16 output weights [0, 32767] (VNNI-compatible)
  - Online rescaling via lazy normalization
  - **Implemented in**: `Exp2FixedSoftmaxRef.cpp`
  
- [x] **Task 1.3**: Implement streaming PV accumulation
  - GEMV: [1, Bc] × [Bc, head_dim] → [1, head_dim]
  - Chunk-by-chunk accumulation
  - **Implemented in**: `flash_decode_single_head()` inner loop
  
- [x] **Task 1.4**: Implement Wo projection (GEMV variant)
  - Single row output: [1, n_heads*head_dim] × Wo^T → [1, d_model]
  - **Implemented in**: `WoProjectionVNNIRef.cpp`
  
- [x] **Task 1.5**: Decode-specific test suite
  - Basic correctness vs FP32 attention
  - Long KV cache (kv_len=4096)
  - Causal masking with position_offset
  - **Tests in**: `Test__Q16FusedAttentionRef.cpp`

**Deliverable**: ✅ Passing `Test__Q16FusedAttentionRefDecode` for single-token generation

### Phase 2: Scalar Reference - Prefill Path ✅ COMPLETE

**Goal**: Extend scalar reference to **prefill path** (GEMM-centric, batched queries)

- [x] **Task 2.1**: Create `Q16FusedAttentionRefPrefill` class
  - Query tiling: Br=16-32 queries per tile
  - FA2-style nested loop: query_tile × kv_tile
  - Per-row softmax state arrays (m[Br], l[Br])
  - **Implemented in**: `Q16FusedAttentionRef.cpp` `fa2_prefill_single_head()`
  
- [x] **Task 2.2**: Implement batched QK dot product (GEMM)
  - [Br, head_dim] × [Bc, head_dim]^T → [Br, Bc]
  - Naive triple loop for scalar reference
  - **Implemented in**: `fa2_prefill_single_head()` inner loops
  
- [x] **Task 2.3**: Implement batched Exp2FixedSoftmax
  - Multi-row max tracking
  - Per-row exp2 LUT application
  - INT16 output weights [0, 32767]
  - **Implemented in**: `Exp2FixedSoftmaxRef.cpp` with per-row state
  
- [x] **Task 2.4**: Implement batched PV accumulation (GEMM)
  - [Br, Bc] × [Bc, head_dim] → [Br, head_dim]
  - Multiple output rows per tile
  - **Implemented in**: `fa2_prefill_single_head()` accumulation
  
- [x] **Task 2.5**: Implement Wo projection (GEMM variant)
  - Multiple rows: [Br, n_heads*head_dim] × Wo^T → [Br, d_model]
  - **Implemented in**: `WoProjectionVNNIRef.cpp` batched path
  
- [x] **Task 2.6**: Prefill-specific test suite
  - Batched queries (seq_len_q=128, 512, 2048)
  - Verify tiling produces same results as non-tiled
  - Compare throughput vs decode path
  - **Tests in**: `Test__Q16FusedAttentionRef.cpp`

**Deliverable**: ✅ Passing `Test__Q16FusedAttentionRefPrefill` for prompt processing

### Phase 2.5: Pipeline Integration 🔄 IN PROGRESS

**Goal**: Integrate Q16 reference kernel into HybridQ16 inference pipeline

**Rationale**: Prove out the reference kernel with real inference before JIT optimization.

- [x] **Task 2.5.1**: Add Q16_INTEGER backend enum
  - Added to `FusedAttentionBackend` in RuntimeConfig.h
  - CLI parsing: "q16_integer", "q16", "q16_int"
  - **Completed**: 2025-12-28

- [x] **Task 2.5.2**: Enable Q16_1 KV cache storage
  - Added `UnifiedKVCache<Q16_1>` template instantiation
  - Implemented block-based copy/shift operations
  - Updated all 4 factory functions
  - **Completed**: 2025-12-28

- [x] **Task 2.5.3**: Update HybridQ16 precision config
  - Changed `kv_cache` from FP32 to Q16_1
  - Preserves precision through attention computation
  - **Completed**: 2025-12-28

- [ ] **Task 2.5.4**: Add Q16_INTEGER dispatch in FusedAttentionWoKernel
  - Wire `FusedAttentionBackend::Q16_INTEGER` case
  - Call `Q16FusedAttentionRef` from the kernel
  - Handle parameter conversion from stage interface

- [ ] **Task 2.5.5**: End-to-end inference test
  - Run HybridQ16 mode with Q16_INTEGER backend
  - Verify token predictions match FP32 reference
  - Measure cosine similarity improvement

**Deliverable**: Working Q16 attention in HybridQ16 inference mode

### Phase 3: JIT Decode Microkernels 🔄 IN PROGRESS

**Goal**: Implement JIT microkernels for **decode path** (latency-optimized)

**Key Focus**: Minimize latency, optimize for streaming KV access

**Scaffolding Complete** ✅:
- `Q16RegisterAllocation.h`: Zone definitions for Q16 JIT kernels
- `JitQ16FusedAttention.h`: Base class with register accessors
- Microkernel stubs created with interfaces defined

- [x] **Task 3.1**: `Q16DotProductGemv` JIT microkernel (scaffold)
  - q vector in zmm0-3 (stays in registers)
  - Stream K chunks through zmm4-7
  - Output: s[Bc] INT32 scores
  - Target: 4 K rows per iteration, `vpdpwssd` for dot
  - **File**: `JitQ8DotProduct.h` (stub created)
  
- [x] **Task 3.2**: `Exp2FixedSoftmaxSingle` JIT microkernel (scaffold)
  - Single row: vectorized max-find, exp2 LUT lookup
  - Scalar m/l state tracking
  - 256-entry LUT for 2^(-frac/256)
  - `vpgatherdd` for LUT lookup
  - **File**: `JitOnlineSoftmax.h` (stub created)
  
- [x] **Task 3.3**: `PVAccumulateGemv` JIT microkernel (scaffold)
  - Stream V alongside K (fused memory access)
  - INT16 × INT16 → INT64 accumulation
  - Output: O[head_dim] in zmm16-23
  - **File**: `JitVNNIMulAccumulate.h` (stub created)
  
- [ ] **Task 3.4**: JIT decode kernel orchestrator
  - Wire microkernels together
  - Manage register handoff between stages
  
- [ ] **Task 3.5**: JIT vs scalar decode parity tests
  - **Bit-exact match** required
  - Test all kv_len sizes (1, 64, 512, 4096)

**Deliverable**: JIT decode path producing bit-exact output vs scalar reference

### Phase 4: JIT Prefill Microkernels (Week 4-5)

**Goal**: Implement JIT microkernels for **prefill path** (throughput-optimized)

**Key Focus**: Maximize FLOPS utilization, optimize cache blocking

- [ ] **Task 4.1**: `Q16DotProductGemm` JIT microkernel
  - Batched: [Br, head_dim] × [Bc, head_dim]^T → [Br, Bc]
  - GEMM micro-tile: 4×4 or 8×4 register blocking
  - K tile reused across Br queries
  
- [ ] **Task 4.2**: `Exp2FixedSoftmaxBatched` JIT microkernel
  - Multi-row: Br independent softmax rows
  - Per-row max in zmm registers
  - 256-entry exp2 LUT lookup per row
  - INT16 output [0, 32767]
  
- [ ] **Task 4.3**: `PVAccumulateGemm` JIT microkernel
  - Batched: [Br, Bc] × [Bc, head_dim] → [Br, head_dim]
  - GEMM-style accumulation
  - V tile reused across Br queries
  
- [ ] **Task 4.4**: JIT prefill kernel orchestrator
  - Outer query tile loop (Br at a time)
  - Inner KV tile loop (Bc at a time)
  - Cache blocking for L2 residency
  
- [ ] **Task 4.5**: JIT vs scalar prefill parity tests
  - **Bit-exact match** required
  - Test various seq_len_q × kv_len combinations

**Deliverable**: JIT prefill path producing bit-exact output vs scalar reference

### Phase 5: Shared Microkernels + Fusion (Week 5-6)

**Goal**: Implement shared Wo projection + residual microkernels

- [ ] **Task 5.1**: `WoProjectionMicrokernel` (shared)
  - Single-row variant for decode
  - Multi-row variant for prefill (configurable)
  - Streaming Q8→INT16 repack
  
- [ ] **Task 5.2**: `ResidualAddMicrokernel` (shared)
  - Q16_1 residual load/add/store
  - Scale factor alignment
  - Handles both single and batched rows
  
- [ ] **Task 5.3**: Fused kernel assembly
  - Wire Wo + residual into main kernel
  - Single continuous execution (no intermediate stores)
  
- [ ] **Task 5.4**: End-to-end fused kernel tests
  - Complete attention block correctness
  - Memory bandwidth reduction verification

**Deliverable**: Complete Q16 fused attention kernel with Wo + residual fusion

### Phase 6: Performance Optimization (Week 6-7)

**Goal**: Optimize both paths for production performance

- [ ] **Task 6.1**: Decode latency optimization
  - Minimize memory stalls (prefetch K/V)
  - Reduce loop overhead
  - Target: ≤1.2× current decode latency
  
- [ ] **Task 6.2**: Prefill throughput optimization
  - Tune Br/Bc tile sizes for L2 residency
  - Register allocation refinement
  - Target: ≥15% throughput improvement
  
- [ ] **Task 6.3**: NUMA-aware allocation
  - First-touch initialization for KV cache buffers
  - Thread affinity optimization
  
- [ ] **Task 6.4**: Performance benchmarking
  - Decode: ms/token across kv_len
  - Prefill: tok/s across prompt lengths
  - Comparison vs hybrid attention

**Deliverable**: Production-ready performance meeting targets

### Phase 7: Integration & Validation (Week 7-8)

**Goal**: Full integration into inference pipeline

- [ ] **Task 7.1**: Create `Q16FusedAttentionStage` for ComputeGraph
- [ ] **Task 7.2**: Wire into Qwen2Graph as attention option
- [ ] **Task 7.3**: Add runtime mode selection (JIT vs reference)
- [ ] **Task 7.4**: Full model accuracy validation
  - Perplexity benchmarks
  - Output quality comparison
- [ ] **Task 7.5**: Documentation and changelog

**Deliverable**: Production-ready Q16 FA2-style fused attention

---

## Testing Strategy

### Decode Path Tests (GEMV)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `Test__Q16DecodeBasic` | Single query, kv_len=64 | cos_sim > 0.99 vs FP32 |
| `Test__Q16DecodeLongKV` | kv_len=4096 | cos_sim > 0.98 vs FP32 |
| `Test__Q16DecodeCausal` | Causal with position offset | Correct future masking |
| `Test__Q16DecodeGQA` | Grouped query attention | Correct head mapping |
| `Test__Q16DecodeEdgeCases` | kv_len=1, all masked | No crashes, correct output |

### Prefill Path Tests (GEMM)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `Test__Q16PrefillBasic` | seq_len_q=128, kv_len=128 | cos_sim > 0.99 vs FP32 |
| `Test__Q16PrefillLongSeq` | seq_len_q=2048, kv_len=2048 | cos_sim > 0.98 vs FP32 |
| `Test__Q16PrefillCausal` | Full causal mask | Correct triangular mask |
| `Test__Q16PrefillGQA` | Batched GQA | Correct head sharing |
| `Test__Q16PrefillTiling` | Various tile boundaries | Tiled == non-tiled output |

### JIT vs Scalar Parity Tests (Decode)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `Test__Q16JitDecodeBasic` | Basic decode parity | **Bit-exact** match |
| `Test__Q16JitDecodeLongKV` | kv_len=4096 parity | **Bit-exact** match |
| `Test__Q16JitDecodeCausal` | Causal mask parity | **Bit-exact** match |
| `Test__Q16JitDecodeGQA` | GQA parity | **Bit-exact** match |

### JIT vs Scalar Parity Tests (Prefill)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `Test__Q16JitPrefillBasic` | Basic prefill parity | **Bit-exact** match |
| `Test__Q16JitPrefillLongSeq` | Long sequence parity | **Bit-exact** match |
| `Test__Q16JitPrefillCausal` | Causal mask parity | **Bit-exact** match |
| `Test__Q16JitPrefillGQA` | Batched GQA parity | **Bit-exact** match |

### Integration Tests

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| `Test__Q16AttentionBlock` | Full attention block | cos_sim > 0.98 vs FP32 |
| `Test__Q16SingleLayer` | One transformer layer | cos_sim > 0.95 vs FP32 |
| `Test__Q16FullModelPrefill` | Prompt processing | perplexity within 0.5% |
| `Test__Q16FullModelDecode` | Token generation | perplexity within 0.5% |

### Performance Benchmarks

| Benchmark | Mode | Metric | Target |
|-----------|------|--------|--------|
| Prefill throughput | Prefill | tok/s | ≥ 350 (vs 304 current) |
| Decode latency | Decode | ms/tok | ≤ 1.2× current |
| Long KV cache | Decode | ms/tok @ kv_len=8K | < 20ms |
| Memory bandwidth | Both | GB/s | < 50% of FP32 path |
| L2 cache hit rate | Prefill | % | > 90% |
| JIT vs Scalar speedup | Both | × | > 20× |

---

## Performance Targets

### Compute Analysis - Decode Mode (GEMV)

For a single query with head_dim=128, kv_len=1024:

| Operation | INT Ops | Description | Bottleneck |
|-----------|---------|-------------|------------|
| q×K^T | 128×1024 = 131K | INT16×INT16 dot products | Memory (K streaming) |
| Softmax | ~3×1024 = 3K | LUT lookup + scaling | Memory (LUT access) |
| w×V | 1024×128 = 131K | INT16×INT16 products | Memory (V streaming) |
| **Total** | ~265K | ~1 vector ops | **Memory bound** |

**Memory Access** (Decode):
- K cache: 256 KB (streamed once)
- V cache: 256 KB (streamed once)
- q vector: 256 bytes (in registers)
- Total BW: ~512 KB per attention head

### Compute Analysis - Prefill Mode (GEMM)

For batch_size=128 queries with head_dim=128, kv_len=1024:

| Operation | INT Ops | Description | Bottleneck |
|-----------|---------|-------------|------------|
| Q×K^T | 128×128×1024 = 16.8M | Batched dot products | **Compute** |
| Softmax | ~3×128×1024 = 393K | Batched LUT lookup | Memory |
| W×V | 128×1024×128 = 16.8M | Batched weight×V | **Compute** |
| **Total** | ~33.9M | | **Compute bound** |

**Memory Access** (Prefill):
- Q tile: 32 KB (Br=128 queries)
- K cache: 256 KB (reused across Br queries)
- V cache: 256 KB (reused across Br queries)
- Compute-to-memory ratio: 66× (GEMM-like)

### Working Set Analysis

| Component | Size Per Head | Fits In |
|-----------|---------------|---------|
| Q tile (Br=32) | 8 KB | L1 (32KB) |
| K tile (Bc=64) | 16 KB | L1 |
| V tile (Bc=64) | 16 KB | L1 |
| Score tile (Br×Bc) | 8 KB | L1 |
| Weight tile (Br×Bc) | 4 KB | L1 |
| O accumulator (Br) | 8 KB | L1 |
| **Total tile** | ~60 KB | **L2 (1MB)** ✓ |
| Full KV cache | 512 KB | L2 ✓ |

### Target Performance by Mode

| Metric | Mode | Current (Hybrid) | Target (Q16) | Δ |
|--------|------|------------------|--------------|---|
| Throughput | Prefill | 304 tok/s | 350+ tok/s | **+15%** |
| Latency | Decode | 18.5 ms/tok | ≤22 ms/tok | ≤1.2× |
| Long KV (8K) | Decode | 35 ms/tok | 40 ms/tok | ~1.1× |
| Memory BW | Both | High | Medium | **-40%** |
| Accuracy | Both | cos=0.897 | cos≥0.98 | **+9%** |

### Performance Design Choices

| Choice | Decode | Prefill | Rationale |
|--------|--------|---------|-----------|
| Tile Br | 1 (single query) | 16-32 | Match query count |
| Tile Bc | 256-512 | 64-128 | Decode: stream KV; Prefill: fit L1 |
| q location | ZMM registers | L1 tile | Decode: never spill; Prefill: tile reuse |
| K/V access | Streaming | Cached | Decode: one-pass; Prefill: reuse |
| Priority | Latency | Throughput | Decode: real-time; Prefill: batch |

---

## Open Questions

### Q1: INT64 Accumulator Overhead

Using INT64 accumulators for V accumulation adds overhead. Alternatives:
- **A**: Accept INT64 cost (correctness > speed)
- **B**: Prescale weights to fit in INT32 accumulator
- **C**: Use FP32 accumulator (breaks integer-only goal)

**Recommendation**: Option A for v1, optimize later if needed.

### Q2: Per-Row vs Per-Tensor Scales

Q16_1 uses per-row scales for K/V cache. Should softmax use:
- **A**: Average scale across all KV rows (simpler)
- **B**: Per-row alpha computation (more accurate)
- **C**: Per-block scales like SageAttention

**Recommendation**: Option A for v1, Option B if accuracy insufficient.

### Q3: Causal Mask Handling

How to handle masked positions in integer domain:
- **A**: Set weight=0 in Exp2FixedSoftmax (current approach)
- **B**: Use sentinel value in scores (e.g., INT32_MIN)
- **C**: Skip masked positions entirely (sparse attention)

**Recommendation**: Option A, verified in Exp2FixedSoftmax implementation.

### Q4: RMSNorm Placement

Standard Transformer applies RMSNorm to residual after attention:
- **A**: Fuse RMSNorm into attention block (μK4)
- **B**: Keep RMSNorm separate as current design
- **C**: Skip attention-internal norm entirely

**Recommendation**: Option B for compatibility, Option A as optimization.

### Q5: Backward Compatibility

Support both Q16 and legacy FP32/Hybrid modes:
- **A**: Compile-time selection via template
- **B**: Runtime selection via `HybridPrecisionConfig`
- **C**: Separate kernel implementations

**Recommendation**: Option B for flexibility.

---

## References

1. **IntAttention Paper**: [arXiv:2511.21513](https://arxiv.org/abs/2511.21513) — IndexSoftmax technique
2. **Intel VNNI Guide**: AVX-512 VNNI instruction reference
3. **SageAttention**: Per-block quantization for attention
4. **Flash Attention**: Memory-efficient attention (future integration)

---

## Appendix A: VNNI Instruction Reference

### Critical Hardware Constraint

> **VPDPWSSD interprets both operands as SIGNED INT16.** Values in range [32768, 65535]
> are treated as negative numbers [-32768, -1]. This means **softmax weights
> must be INT16 in range [0, 32767]** — the upper half of UINT16 range is unusable.

### `vpdpwssd` — Packed Dot Product of Signed Words with Dword Accumulation

```
vpdpwssd zmm1, zmm2, zmm3

For each dword position i:
    zmm1[i] += zmm2[i].word[0] * zmm3[i].word[0] 
             + zmm2[i].word[1] * zmm3[i].word[1]

CRITICAL: Both word[0] and word[1] are interpreted as SIGNED INT16!
         Range: [-32768, 32767]
         Output: INT32 accumulator
```

- Input: Two ZMM registers with **signed** INT16 pairs
- Output: INT32 accumulation in destination
- Throughput: 0.5 CPI on recent Intel CPUs
- **Constraint**: Cannot use UINT16 values > 32767 (would be negative)

### `vpmovsxbw` — Sign-Extend INT8 to INT16

```
vpmovsxbw zmm1, ymm2

For each position i:
    zmm1.word[i] = sign_extend(ymm2.byte[i])
```

- Input: 32 × INT8 in YMM
- Output: 32 × INT16 in ZMM
- Use for Q8→INT16 weight conversion

---

## Appendix B: Exp2FixedSoftmax LUT Generation

The exp2 LUT approximates `2^(-frac/256)` for fractional values in [0, 255].

```python
import numpy as np

def generate_exp2_lut():
    """Generate 256-entry LUT for Exp2FixedSoftmax.
    
    Each entry approximates 2^(-i/256) with 30-bit precision.
    This allows integer-only softmax computation using:
      exp2(-x) = LUT[frac] >> int_part
    
    where x is decomposed into int_part and frac (8 fractional bits).
    """
    LUT_SIZE = 256
    LUT_PRECISION_BITS = 30
    
    lut = []
    for i in range(LUT_SIZE):
        # 2^(-i/256) scaled to 30-bit integer
        val = 2.0 ** (-i / 256.0)
        int_val = int(round(val * (1 << LUT_PRECISION_BITS)))
        lut.append(int_val)
    
    return lut

def exp2_fixed_softmax_example():
    """Example: compute softmax weight for a given delta.
    
    delta = max_score - score (always >= 0)
    alpha = combined scale factor
    beta = alpha * log2(e) = alpha * 1.4427
    t = delta * beta (in log2 domain)
    """
    lut = generate_exp2_lut()
    
    # Example values
    delta = 100  # score difference (integer)
    beta = 0.01  # typical scale factor
    t = delta * beta  # = 1.0 in this example
    
    # Decompose t into fixed-point (8 frac bits)
    t_fixed = int(t * 256)  # = 256
    int_part = t_fixed >> 8  # = 1
    frac_part = t_fixed & 255  # = 0
    
    # Lookup and shift
    if int_part >= 30:
        result = 0  # Underflow
    else:
        result = lut[frac_part] >> int_part
    
    # result ≈ 2^(-1.0) * 2^30 = 536870912
    print(f"exp2(-{t:.2f}) ≈ {result} (out of {1 << 30})")

# Generate and print C++ LUT
lut = generate_exp2_lut()
print("// Exp2FixedSoftmax: 256-entry LUT for 2^(-frac/256)")
print("// 30-bit precision, indexed by fractional part [0, 255]")
print("static uint32_t EXP2_LUT[256] = {")
for i in range(0, 256, 8):
    row = ", ".join(f"{v:10d}" for v in lut[i:i+8])
    print(f"    {row},")
print("};")
```

### Example LUT Values

```cpp
// First 16 entries (frac = 0 to 15):
// lut[0]  = 1073741824  // 2^30 * 2^(-0/256)   = 2^30 * 1.0000
// lut[1]  = 1070923228  // 2^30 * 2^(-1/256)   ≈ 2^30 * 0.9973
// lut[8]  = 1050866141  // 2^30 * 2^(-8/256)   ≈ 2^30 * 0.9786
// lut[16] = 1028443332  // 2^30 * 2^(-16/256)  ≈ 2^30 * 0.9576

// Key values:
// lut[128] = 759250125  // 2^30 * 2^(-128/256) = 2^30 * 0.7071 (√½)
// lut[255] = 537919669  // 2^30 * 2^(-255/256) ≈ 2^30 * 0.5010
```

---

*End of Document*
