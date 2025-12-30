# Q16_1 Integer-Domain Attention Kernel v2

**Status**: In Progress (Phase 5.3 - MLA Microkernel Support)  
**Created**: 2025-12-30  
**Updated**: 2025-12-31  
**Author**: Llaminar Team  
**Supersedes**: [PROJECT_Q16_INTEGER_ATTENTION.md](./PROJECT_Q16_INTEGER_ATTENTION.md)

## Progress

- [x] Phase 0: Rename failed v1 kernel and microkernels to `.deprecated`
  - [x] `Q16FusedAttentionRef.cpp/h` → `.deprecated.cpp/h`
  - [x] `microkernels/*.cpp/h` → `.deprecated.cpp/h` (5 microkernels)
  - [x] Fixed all internal includes to use `.deprecated.h`
  - [x] Added deprecated sources back to CMakeLists for existing infrastructure
- [x] Phase 1: Scaffold new `Q16IntegerAttentionRef.h/.cpp`
  - [x] Variable block size structs (`Q16_1Block_64`, `Q16_1Block_128`, `Q16_1Block_192`)
  - [x] `Q16IntegerAttentionParams` with head scales
  - [x] Integer Q×K^T dot product (templated)
  - [x] Integer P×V accumulation (templated)
  - [x] Integration with new `Exp2FixedSoftmax` microkernel
  - [x] Fresh `Exp2FixedSoftmax.h/.cpp` with cleaner API
  - [x] **Expanded LUT from 256→2048 entries** (11-bit index, 4KB, ~0.05% error) ✅
  - [x] 28 unit tests for Exp2FixedSoftmax (16 basic + 12 spiky activation tests, all passing)
- [x] Phase 1.5: Codebase audit for block size dependencies
  - [x] Identified 25 files with 40 functions requiring updates
  - [x] Categorized by priority: P0 (blocking) → P3 (integration)
  - [x] See "Q16_1 Codebase Audit" section below
- [x] Phase 2: Core tensor + RoPE block size support (P0) ✅
  - [x] `BlockStructures.h`: Add `Q16_1Block_64/128/192` with `sum_qs` field ✅
  - [x] `BlockStructures.h`: Add `Q16BlockSize` enum and type traits ✅
  - [x] `BlockStructures.h`: Add `optimal_q16_block_size()` function ✅
  - [x] `Test__Q16BlockStructures.cpp`: 17 unit tests all passing ✅
  - [x] `Q16IntegerAttentionRef.h`: Refactored to use canonical types from BlockStructures.h ✅
  - [x] `Q16_1Tensor`: Add `block_size_` member (default: BLOCK_32 for backward compat) ✅
  - [x] `Q16_1Tensor`: Add constructor with `Q16BlockSize` parameter ✅
  - [x] `Q16_1Tensor`: Update `blocks_per_row()` to use `block_size_` ✅
  - [x] `Q16_1Tensor`: Add `q16_block_size()` accessor ✅
  - [x] `Test__Q16_1Tensor.cpp`: 9 new unit tests for variable block sizes (all passing) ✅
  - [x] `RoPEPrimitives`: Templatize functions on BlockType (scalar, AVX2, AVX512) ✅
  - [x] `Test__Q16_1RoPE.cpp`: 25 tests including variable block sizes (all passing) ✅
- [x] Phase 3: Residual + MPI + KV Cache (P1) ✅
  - [x] `SIMDHelpers.h`: Templatize 4 q16_1 operations (q16_add_q16, q16_add_fp32, q16_add_q8, q16_sum_n) ✅
  - [x] `Test__Q16VariableBlockSIMD.cpp`: 19 tests for variable block SIMD operations (all passing) ✅
  - [x] `MPIContext`: Add `allreduce_q16_inplace<BlockType>` template for variable block sizes ✅
  - [x] `BlockStructures.h`: Add `q16_block_size_bytes()` and `q16_block_size_elements()` helpers ✅
  - [x] `TensorFactory`: Add `createQ16_1(shape, block_size, device_idx)` overload ✅
  - [x] `UnifiedKVCache`: Auto-select optimal block size from head_dim, variable-size copy/shift ✅
  - [x] **Unit Test Coverage Audit** (2025-12-30):
    - [x] `Test__Q16BlockStructures.cpp`: Added 4 tests for helper functions (`q16_block_size_bytes`, `q16_block_size_elements`) ✅
    - [x] `Test__TensorFactory_Q16BlockSize.cpp`: NEW - 11 tests for `createQ16_1(shape, block_size)` overload ✅
    - [x] `Test__UnifiedKVCache.cpp`: Added 7 tests for variable block size auto-selection and operations ✅
    - [x] `Test__Q16MPI_Allreduce.cpp`: NEW - 10 MPI integration tests (2-rank) for all block sizes ✅
    - [x] Fixed 2 pre-existing KV cache tests (`AppendQ16_1_MultipleAppends`, `EvictQ16_1`) ✅
    - [x] Fixed `MPIContext::allreduce_q16_inplace` to correctly loop over blocks for `q16_sum_n` ✅
  - [x] **Kernel-Level Updates** (2025-12-30):
    - [x] `CPURoPEKernelT<Q16_1>`: Updated `apply_tensor()` to dispatch by `q16_block_size()` ✅
    - [x] `CPURoPEKernelT<Q16_1>`: Added `apply_typed_block<BlockType>()` template method ✅
    - [x] `Q16_1Tensor::copyFrom_fp32()`: Fixed to support all block sizes (32, 64, 128, 192) ✅
    - [x] `Q16_1Tensor::decode_block_at()`: Fixed to dispatch by block size ✅
    - [x] `Q16_1Tensor::get_raw_block_at()`: Fixed to dispatch by block size ✅
    - [x] `Test__CPURoPEKernelT_Q16_1_BlockSizes.cpp`: NEW - 23 tests for kernel-level RoPE with all block sizes ✅
- [x] Phase 4: Per-head scale normalization ✅
  - [x] `Q16HeadNormalization.h/.cpp`: Core normalization primitives ✅
    - [x] `normalize_q16_head<BlockType>()`: Requantize blocks within a head to unified scale ✅
    - [x] `find_head_max_scale<BlockType>()`: Find max |d| across blocks in head ✅
    - [x] `requantize_blocks_to_scale<BlockType>()`: Adjust qs values for new scale ✅
    - [x] `needs_normalization<BlockType>()`: Check if head needs normalization ✅
  - [x] `Q16_1HeadMetadata` struct for storing per-head scales ✅
  - [x] `blocks_per_head()` and `is_optimal_block_size()` constexpr helpers ✅
  - [x] Explicit template instantiations for all 4 block types (32, 64, 128, 192) ✅
  - [x] `Test__Q16HeadNormalization.cpp`: NEW - 27 unit tests (all passing) ✅
    - [x] `find_head_max_scale` tests: SingleBlock, MultipleBlocks, NegativeScales, AllZero, Empty, AllBlockTypes
    - [x] `requantize_blocks_to_scale` tests: SingleBlock, MultipleBlocks, ZeroBlock
    - [x] `normalize_q16_head` tests: SingleBlock optimal path, MultipleBlocks, PreservesSum, Block192
    - [x] `needs_normalization` tests: SingleBlock, DifferentScales, SameScales, WithTolerance, AfterNormalize
    - [x] Precision tests: LargeScaleDifference (10×), TypicalDistribution
    - [x] Edge cases: NullPointer, ZeroNumBlocks, ZeroHeadScale, MaxINT16Values
  - [ ] Integration with FusedQKVGEMMStage for Q normalization (Phase 5+)
  - [ ] Integration with KVCacheAppendStage for K/V normalization (Phase 5+)
- [x] Phase 5: Implement Wo projection (VPDPWSSD) ✅
  - [x] `WoProjection.h/.cpp`: INT32→INT16 normalization, GEMV, quantization microkernels ✅
  - [x] `q16_context_normalize_to_int16()`: Requantize INT32 context for Wo projection ✅
  - [x] `q16_wo_row_gemv<BlockType>()`: Single row × weight GEMV with VPDPWSSD ✅
  - [x] `q16_wo_projection<BlockType>()`: Full context → projected output ✅
  - [x] `q16_wo_projection_batched<BlockType>()`: Multi-query batched projection ✅
  - [x] `q16_quantize_to_q16_1<BlockType>()`: INT32→Q16_1 final quantization ✅
  - [x] `Test__Q16WoProjection.cpp`: 12 unit tests (all passing) ✅
- [x] Phase 5.1: QK Dot Product Microkernels ✅
  - [x] `Q16DotProduct.h/.cpp`: Templated dot products for all block sizes ✅
  - [x] `q16_dot_single<BlockType>()`: Inner loop for Q·K^T computation ✅
  - [x] `q16_qk_gemv<BlockType>()`: Flash Decode path (seq_len_q=1) ✅
  - [x] `q16_qk_gemm_tile<BlockType>()`: FA2 Prefill tiled path ✅
  - [x] Runtime dispatch functions (`dispatch_q16_dot_single`, etc.) ✅
  - [x] `Test__Q16DotProduct.cpp`: 11 unit tests (all passing) ✅
- [x] Phase 5.2: PV Accumulation Microkernels ✅
  - [x] `PVAccumulate.h/.cpp`: Weighted V accumulation (P×V) ✅
  - [x] `q16_pv_accumulate<BlockType>()`: Zero context + accumulate ✅
  - [x] `q16_pv_accumulate_add<BlockType>()`: Add to existing context ✅
  - [x] `q16_pv_gemm_tile<BlockType>()`: FA2 tiled P×V path ✅
  - [x] `q16_context_rescale()`: Online softmax rescaling ✅
  - [x] `Test__PVAccumulate.cpp`: 12 unit tests (all passing) ✅
- [ ] Phase 5.3: MLA (Multi-head Latent Attention) Support
  - [ ] `MLAAttentionParams` struct with split NOPE/ROPE tensors
  - [ ] `q16_qk_gemv_mla()`: Dual dot product (NOPE + ROPE)
  - [ ] `q16_qk_gemm_tile_mla()`: Tiled MLA variant
  - [ ] MLA-aware KV cache with separate NOPE/ROPE storage
  - [ ] Integration tests with DeepSeek V3 / Kimi K2 configs
- [ ] Phase 6: Implement integer residual add
- [ ] Phase 7: FA2 Prefill tiled implementation
- [ ] Phase 8: Unit tests for Q16IntegerAttention
- [ ] Phase 9: E2E parity tests
- [ ] Phase 10: KV Cache Scale Profiling Tool
  - [ ] Create `python/tools/profile_kv_activations.py` 
  - [ ] Extend existing PyTorch snapshot infrastructure
  - [ ] Collect per-layer, per-head K/V activation statistics (min, max, mean, std, percentiles)
  - [ ] Run on representative prompts (diverse dataset)
  - [ ] Output recommended `kv_cache_scale` values per model family
  - [ ] Profile: Qwen 2.5 (0.5B, 7B), Llama 3 (8B), DeepSeek V3

---

## Design Goal: Full Integer Residual Stream

### Core Principle: NO FP32 Intermediate Residuals

**The residual stream is the backbone of transformer inference.** In v2, the residual is maintained as **Q16_1 throughout the entire forward pass**, with block sizes selected based on the model's head dimension.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Q16_1 RESIDUAL STREAM (FULL INTEGER)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EMBEDDING                                                                   │
│  ─────────                                                                   │
│  Token IDs → Embedding Table → Q16_1 residual (block size = head_dim)       │
│                                                                              │
│  FOR EACH TRANSFORMER LAYER:                                                 │
│  ────────────────────────────                                                │
│                                                                              │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  ATTENTION BLOCK                                                     │  │
│    │                                                                      │  │
│    │  1. RMSNorm(Q16_1 residual) → Q16_1 normalized                      │  │
│    │                                                                      │  │
│    │  2. QKV PROJECTION (see "QKV Projection Exception" below)           │  │
│    │     Q16_1 → Q8_1 → GEMM → Q8_1 Q,K,V → RoPE → Q16_1 Q,K,V          │  │
│    │                                                                      │  │
│    │  3. KV Cache: Store Q16_1 K,V (model-appropriate block size)        │  │
│    │                                                                      │  │
│    │  4. Integer Attention: Q16_1 Q,K,V → INT32 → Q16_1 context          │  │
│    │                                                                      │  │
│    │  5. Wo Projection: Q16_1 context × Q8_0 Wo → Q16_1 projected        │  │
│    │                                                                      │  │
│    │  6. Residual Add: Q16_1 residual + Q16_1 projected → Q16_1 residual │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │  FFN BLOCK                                                           │  │
│    │                                                                      │  │
│    │  1. RMSNorm(Q16_1 residual) → Q16_1 normalized                      │  │
│    │                                                                      │  │
│    │  2. Gate+Up GEMM: Q16_1 → Q8_1 → GEMM → Q16_1 gate, up              │  │
│    │                                                                      │  │
│    │  3. SiLU(gate) × up → Q16_1 activated                               │  │
│    │                                                                      │  │
│    │  4. Down GEMM: Q16_1 → Q8_1 → GEMM → Q16_1 down                     │  │
│    │                                                                      │  │
│    │  5. Residual Add: Q16_1 residual + Q16_1 down → Q16_1 residual      │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  FINAL OUTPUT                                                                │
│  ────────────                                                                │
│  RMSNorm(Q16_1 residual) → LM Head GEMM → FP32 logits (only FP32 output!)   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What This Means

| Aspect | v1 (Broken) | v2 (Goal) |
|--------|-------------|-----------|
| **Residual storage** | FP32 | **Q16_1** |
| **Intermediate activations** | FP32 dequantize/requantize | **Q16_1 or Q8_1 only** |
| **Attention KV cache** | FP32 | **Q16_1** |
| **Scale factors** | FP32 (unavoidable) | FP32 (unavoidable) |
| **Data computation** | FP32 GEMM/attention | **INT16/INT32 VNNI** |

**The ONLY FP32 in the forward pass should be:**
1. Scale factors (the `float d` in Q16_1/Q8_1 blocks) — unavoidable
2. Final logits output — required for sampling

### QKV Projection Exception: The Q16_1 → Q8_1 → Q16_1 Path

The QKV projections are the **one place where we temporarily convert to Q8_1** for GEMM efficiency. This is a deliberate design choice:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                QKV PROJECTION PATH (FusedQKVGEMMStage)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Q16_1 residual (after RMSNorm)                                       │
│         Block size: model-appropriate (64 or 128)                            │
│                                                                              │
│  STEP 1: Convert Q16_1 → Q8_1                                               │
│  ─────────────────────────────                                               │
│  Why: VPDPBUSD (UINT8×INT8→INT32) is 2× faster than VPDPWSSD (INT16×INT16)  │
│  How: Requantize with scale adjustment (preserves information, loses range) │
│                                                                              │
│  STEP 2: Q8_1 × Q8_0 Wq/Wk/Wv GEMM → Q8_1 Q, K, V                          │
│  ───────────────────────────────────────────────────                         │
│  Why: VNNI-optimized GEMM path (existing infrastructure)                     │
│  Note: Output is Q8_1 with per-row scales                                    │
│                                                                              │
│  STEP 3: Apply RoPE in Q8_1 → Q16_1                                         │
│  ──────────────────────────────────                                          │
│  Input:  Q8_1 Q and K tensors                                                │
│  Output: Q16_1 Q and K tensors with per-head scales                          │
│                                                                              │
│  Why Q16_1 output?                                                           │
│  1. RoPE involves sin/cos multiplication - expands dynamic range             │
│  2. Q16_1 preserves precision for subsequent Q×K^T dot products             │
│  3. Per-head scale normalization is applied at this stage                    │
│                                                                              │
│  STEP 4: Store K, V in Q16_1 KV Cache                                        │
│  ─────────────────────────────────                                           │
│  Block size: model-appropriate (matches head_dim for 1 block/head)           │
│  Scale: Per-head normalized scale                                            │
│                                                                              │
│  OUTPUT: Q16_1 Q, K, V with per-head scales                                  │
│          Ready for integer attention kernel                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why not stay in Q16_1 for GEMM?**

| Approach | Memory BW | Compute | Trade-off |
|----------|-----------|---------|-----------|
| Q16_1 activations × Q16_0 weights | 2× | VPDPWSSD | Higher precision, slower |
| Q8_1 activations × Q8_0 weights | 1× | VPDPBUSD | Lower precision, **2× faster** |

For projection layers (Wq, Wk, Wv, Wo, FFN), the GEMM is compute-bound. The Q8_1 path is preferred because:
1. **Weights are stored as Q8_0** — no choice, must use INT8
2. **VPDPBUSD is optimized** — 2× throughput vs VPDPWSSD  
3. **Post-GEMM expansion** — RoPE and normalization expand back to Q16_1

### Model-Appropriate Block Sizes

The Q16_1 residual block size is selected **at model load time** based on `head_dim`:

```cpp
Q16BlockSize select_residual_block_size(const ModelConfig& config) {
    const int head_dim = config.hidden_size / config.n_heads;
    
    // Goal: 1 block per head for optimal integer attention
    switch (head_dim) {
        case 64:  return Q16BlockSize::BLOCK_64;   // Qwen2.5-0.5B, GPT-2
        case 128: return Q16BlockSize::BLOCK_128;  // Llama, Mistral, Qwen3
        default:  return Q16BlockSize::BLOCK_64;   // Universal fallback
    }
    // NOTE: MLA models (DeepSeek V3, Kimi K2) use separate NOPE/ROPE tensors
    // with their own block sizes (128 for NOPE, 64 for ROPE) - see MLA section
}
```

**This block size is used for:**
- Embedding output tensor
- All Q16_1 residual tensors throughout the forward pass
- KV cache storage (K and V tensors)
- Intermediate Q16_1 activations

### Summary: Zero FP32 Intermediate Data

| Stage | Input | Output | FP32 Data? |
|-------|-------|--------|------------|
| Embedding | Token IDs | Q16_1 | ❌ No |
| RMSNorm | Q16_1 | Q16_1 | ❌ No (scale only) |
| QKV input convert | Q16_1 | Q8_1 | ❌ No |
| QKV GEMM | Q8_1 × Q8_0 | Q8_1 | ❌ No |
| RoPE | Q8_1 | Q16_1 | ❌ No (sin/cos LUT) |
| KV Cache | Q16_1 | Q16_1 | ❌ No |
| Q×K^T | INT16 × INT16 | INT32 | ❌ No |
| Softmax | INT32 | INT16 | ❌ No (LUT) |
| P×V | INT16 × INT16 | INT32 | ❌ No |
| Wo convert | Q16_1 | Q8_1 | ❌ No |
| Wo GEMM | Q8_1 × Q8_0 | Q16_1 | ❌ No |
| Residual Add | Q16_1 + Q16_1 | Q16_1 | ❌ No |
| FFN (same pattern) | Q16_1 | Q16_1 | ❌ No |
| Final Norm + LM Head | Q16_1 | **FP32** | ✅ Only here! |

---

## Lessons Learned from v1 Implementation

The original Q16 integer attention implementation **failed to achieve integer-only computation** despite extensive documentation claiming otherwise. Post-mortem analysis revealed:

### What Went Wrong

| Stage | Design Intent | Actual Implementation | Root Cause |
|-------|---------------|----------------------|------------|
| **Q×K^T** | INT16×INT16→INT32 | **FP32** accumulator | Per-block scales forced `float(block_dot) * q.d * k.d` |
| **Softmax** | Exp2FixedSoftmax LUT | **`std::exp()`** | Score range unpredictable for LUT input mapping |
| **P×V** | INT32 accumulator | INT32 + **FP64 scale tracking** | Per-element V scales required double precision |
| **Wo projection** | VPDPWSSD INT32 | **FP32** accumulator | Reference impl took easy path |
| **Residual add** | Q16_1 + Q16_1 | ✓ Q16_1 | Only stage that worked correctly |

### The Fundamental Problem: Per-Block Scales

Q16_1 format stores one `float d` scale per 32 elements. For `head_dim=64`:
- Q has 2 blocks with scales `q.d[0]`, `q.d[1]`  
- K has 2 blocks with scales `k.d[0]`, `k.d[1]`
- V has 2 blocks with scales `v.d[0]`, `v.d[1]`

Computing Q×K^T requires combining **4 different scale pairs** per score:
```
score = (q_block0 · k_block0) × q.d[0] × k.d[0]
      + (q_block1 · k_block1) × q.d[1] × k.d[1]
```

Without normalization, agents fell back to FP32 to handle arbitrary scale combinations.

### Design Oversight: Fixed 32-Element Block Size

The Q16_1 format inherited Q8_1's 32-element block size without considering:

1. **INT16 has 256× more precision** than INT8 - can handle larger block variance
2. **Common head dimensions** (64, 128) would benefit from aligned block sizes
3. **Attention algorithm** specifically needs per-head (not per-block) scales

This wasn't obvious at design time because Q16_1 was designed for **GEMM**, where per-block scales work fine. The problem only manifests in **attention** where multiple scale combinations compound.

---

## v2 Solution Part A: Model-Aware Block Sizes

### The Key Insight: 64 is the GCD

Common head dimensions across model families:
- **64**: Qwen2.5-0.5B, GPT-2
- **128**: Qwen3, MiniMax-M1, Llama-2/3, Mistral
- **MLA (DeepSeek V3, Kimi K2)**: 128 NOPE + 64 ROPE (separate tensors, separate scales)

**64 divides all standard head_dims evenly**, making it the universal fallback block size.

### Model Survey

| Model | Architecture | Q/K head_dim | V head_dim | 64-block | 128-block | Notes |
|-------|-------------|--------------|------------|----------|-----------|-------|
| Qwen2.5-0.5B | Standard | 64 | 64 | ✅ 1/head | ❌ Broken | Optimal |
| Qwen3-8B | Standard | 128 | 128 | 2/head | ✅ 1/head | Optimal |
| MiniMax-M1 | Standard | 128 | 128 | 2/head | ✅ 1/head | Optimal |
| Llama-3-8B | Standard | 128 | 128 | 2/head | ✅ 1/head | Optimal |
| DeepSeek V3 | **MLA** | 128+64 | 128 | See below | See below | Separate NOPE/ROPE |
| Kimi K2 | **MLA** | 128+64 | 128 | See below | See below | Separate NOPE/ROPE |

### Block Size Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                   BLOCK SIZE SELECTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AT QUANTIZATION TIME (model-aware):                            │
│                                                                  │
│    if model.architecture == MLA:                                │
│      // Separate NOPE and ROPE with their own scales            │
│      Q_nope, K_nope → Q16_128 blocks (1 per head)              │
│      Q_rope, K_rope → Q16_64 blocks (1 per head)               │
│      V weights      → Q16_128 blocks (1 per head)              │
│                                                                  │
│    else if model.head_dim == 64:                                │
│      → Q16_64 blocks (1 per head) ← OPTIMAL, no normalization! │
│                                                                  │
│    else if model.head_dim == 128:                               │
│      → Q16_128 blocks (1 per head) ← OPTIMAL                   │
│                                                                  │
│    else:                                                         │
│      → Q16_64 blocks (universal fallback)                      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  AT RUNTIME (attention kernel):                                 │
│                                                                  │
│    if blocks_per_head == 1:                                     │
│      → Pure integer path, no normalization overhead             │
│                                                                  │
│    else:  // 2-3 blocks per head                                │
│      → Per-head normalization (Part B of this plan)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### MLA Architecture: Separate NOPE/ROPE Scales

DeepSeek V3 and Kimi K2 use Multi-head Latent Attention (MLA), which splits Q/K into:
- **NOPE (No Position Embedding)**: 128 dimensions - context-dependent, no rotation
- **ROPE (Rotary Position Embedding)**: 64 dimensions - position-encoded via rotation

**Why NOT merge into 192-block:**
1. NOPE and ROPE have fundamentally different value distributions
2. A single scale for both loses precision where it matters most
3. The 192-block approach was optimizing for "blocks per head" metric, not accuracy

**Correct approach: Separate tensors with optimal block sizes**

```
┌─────────────────────────────────────────────────────────────────┐
│              MLA Q/K STORAGE (DeepSeek V3, Kimi K2)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Q_nope: Q16_1Block_128 × n_heads                               │
│          → 1 block per head, 1 scale per head                   │
│          → Stores context-dependent query components            │
│                                                                  │
│  Q_rope: Q16_1Block_64 × n_heads                                │
│          → 1 block per head, 1 scale per head                   │
│          → Stores position-encoded query components             │
│                                                                  │
│  K_nope: Q16_1Block_128 × n_kv_heads  (cached)                  │
│  K_rope: Q16_1Block_64 × n_kv_heads   (cached)                  │
│                                                                  │
│  V:      Q16_1Block_128 × n_kv_heads  (standard, no split)      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ATTENTION COMPUTATION:                                          │
│                                                                  │
│  score = Q_nope × K_nope^T  +  Q_rope × K_rope^T                │
│          ↑                     ↑                                 │
│          INT32 (scale_q_nope   INT32 (scale_q_rope              │
│                 × scale_k_nope)       × scale_k_rope)           │
│                                                                  │
│  Combined score uses 2 scale pairs (still tractable!)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits of separate NOPE/ROPE:**
| Aspect | 192-block (rejected) | Separate tensors (correct) |
|--------|---------------------|---------------------------|
| Scales per head | 1 | 2 (one per component) |
| Precision | Compromised | Optimal for each component |
| Memory overhead | ~4.2% | ~6.25% (NOPE) + ~12.5% (ROPE) |
| Implementation | Simpler | Slightly more complex |
| Accuracy | Potentially lossy | Preserves semantics |

### Memory Efficiency Comparison

| Block Size | Overhead | Bytes/Value | Use Case |
|------------|----------|-------------|----------|
| 32 (legacy) | 8B / 32 = 25% | 2.25 | GEMM compatibility |
| 64 | 8B / 64 = 12.5% | 2.125 | head_dim=64, MLA ROPE, universal fallback |
| 128 | 8B / 128 = 6.25% | 2.0625 | head_dim=128, MLA NOPE/V (most modern models) |
| 192 | 8B / 192 = 4.2% | 2.04 | *Not recommended* - see MLA section |

### New Q16 Format Variants

```cpp
// Existing (keep for GEMM compatibility)
struct Q16_1Block_32 {
    float d;
    int32_t sum_qs;
    int16_t qs[32];
};

// New: 64-element blocks (universal for attention)
struct Q16_1Block_64 {
    float d;
    int32_t sum_qs;
    int16_t qs[64];
};

// New: 128-element blocks (optimal for head_dim=128)
struct Q16_1Block_128 {
    float d;
    int32_t sum_qs;
    int16_t qs[128];
};

// 192-element blocks exist but NOT recommended for MLA
// See "MLA Architecture: Separate NOPE/ROPE Scales" section
struct Q16_1Block_192 {
    float d;
    int32_t sum_qs;
    int16_t qs[192];
};

// Tensor metadata includes block size
struct Q16_1TensorMetadata {
    uint32_t block_size;  // 32, 64, 128 (192 exists but not recommended)
    uint32_t num_blocks;
    // ... existing fields ...
};
```

### MLA Block Size: Separate NOPE/ROPE (Recommended)

For DeepSeek V3 / Kimi K2 with MLA architecture, the correct approach is **separate tensors**:

```cpp
// MLA Q/K layout with separate NOPE and ROPE tensors
// Each gets optimal 1-block-per-head with its own scale
struct MLAAttentionTensors {
    // NOPE (No Position Embedding) - context-dependent
    Q16_1Block_128* Q_nope;     // [n_heads, 1 block] - 128-dim
    Q16_1Block_128* K_nope;     // [n_kv_heads, seq_len, 1 block]
    
    // ROPE (Rotary Position Embedding) - position-encoded
    Q16_1Block_64* Q_rope;      // [n_heads, 1 block] - 64-dim
    Q16_1Block_64* K_rope;      // [n_kv_heads, seq_len, 1 block]
    
    // V is standard (no NOPE/ROPE split)
    Q16_1Block_128* V;          // [n_kv_heads, seq_len, 1 block] - 128-dim
};

// Attention score computation:
// score[q][k] = dot(Q_nope[q], K_nope[k]) * scale_nope
//             + dot(Q_rope[q], K_rope[k]) * scale_rope
//
// Only 2 scale pairs per score - tractable integer arithmetic!
```

**Why this is better than 192-block:**
- NOPE and ROPE have fundamentally different value distributions
- Each component gets its own optimal scale
- The 192-block loses semantic information for marginal memory savings

---

## MLA Microkernel Implementation Design

### Overview

The Q16 microkernel framework is designed from the ground up to support MLA (Multi-head Latent Attention) models like DeepSeek V3 and Kimi K2. The key insight is that our **templated microkernels work for any block size**, so MLA support is achieved through orchestration rather than new inner loops.

### MLA Attention Parameters

```cpp
// Standard attention (Qwen, Llama, etc.)
struct Q16IntegerAttentionParams {
    const void* Q;              // Q16_1Block_64 or Q16_1Block_128
    const void* K;              // Same block type as Q
    const void* V;              // Same block type as Q
    float* scale_q;             // [n_heads] per-head scales
    float* scale_k;             // [n_kv_heads]
    float* scale_v;             // [n_kv_heads]
    Q16BlockSize block_size;    // 64 or 128
    // ... other params
};

// MLA attention (DeepSeek V3, Kimi K2)
struct MLAAttentionParams {
    // NOPE component (128-dim, no position embedding)
    const Q16_1Block_128* Q_nope;  // [n_heads × 1 block]
    const Q16_1Block_128* K_nope;  // [n_kv_heads × seq_len × 1 block]
    float* scale_q_nope;           // [n_heads] per-head scales
    float* scale_k_nope;           // [n_kv_heads]
    
    // ROPE component (64-dim, rotary position embedding)
    const Q16_1Block_64* Q_rope;   // [n_heads × 1 block]
    const Q16_1Block_64* K_rope;   // [n_kv_heads × seq_len × 1 block]
    float* scale_q_rope;           // [n_heads]
    float* scale_k_rope;           // [n_kv_heads]
    
    // V is standard (no NOPE/ROPE split)
    const Q16_1Block_128* V;       // [n_kv_heads × seq_len × 1 block]
    float* scale_v;                // [n_kv_heads]
    
    // Dimensions
    int n_heads;
    int n_kv_heads;
    int seq_len_q;
    int seq_len_kv;
};
```

### MLA QK Dot Product Kernel

The MLA variant calls **two separate dot products** and sums the results in INT32:

```cpp
// MLA-aware QK dot product (Flash Decode path, seq_len_q=1)
void q16_qk_gemv_mla(
    const Q16_1Block_128* Q_nope,    // [n_heads × 1 block]
    const Q16_1Block_128* K_nope,    // [n_kv_heads × seq_len × 1 block]
    const Q16_1Block_64* Q_rope,     // [n_heads × 1 block]
    const Q16_1Block_64* K_rope,     // [n_kv_heads × seq_len × 1 block]
    int32_t* scores,                  // Output: [seq_len_kv]
    int seq_len_kv,
    int head_idx,                     // Which query head
    int kv_head_idx                   // Which KV head (for GQA)
) {
    const Q16_1Block_128* q_nope = &Q_nope[head_idx];
    const Q16_1Block_64* q_rope = &Q_rope[head_idx];
    
    for (int k = 0; k < seq_len_kv; ++k) {
        // NOPE dot product (128-dim, one Q16_1Block_128 each)
        int32_t nope_score = q16_dot_single<Q16_1Block_128>(
            q_nope,
            &K_nope[kv_head_idx * seq_len_kv + k],
            1  // num_blocks = 1 (optimal path)
        );
        
        // ROPE dot product (64-dim, one Q16_1Block_64 each)
        int32_t rope_score = q16_dot_single<Q16_1Block_64>(
            q_rope,
            &K_rope[kv_head_idx * seq_len_kv + k],
            1  // num_blocks = 1 (optimal path)
        );
        
        // Combine in INT32 domain (scales applied later)
        scores[k] = nope_score + rope_score;
    }
}
```

### MLA Scale Combination

After softmax, the combined scale for the attention output is:

```cpp
// Scale combination for MLA (applied once at output)
float compute_mla_qk_scale(
    float scale_q_nope, float scale_k_nope,
    float scale_q_rope, float scale_k_rope,
    int head_dim_nope,   // 128
    int head_dim_rope    // 64
) {
    // Both components contribute to the score
    // The INT32 score = nope_dot + rope_dot
    // where nope_dot has implicit scale (scale_q_nope * scale_k_nope)
    // and rope_dot has implicit scale (scale_q_rope * scale_k_rope)
    //
    // For softmax input, we need:
    //   float_score = INT32_score * combined_scale / sqrt(total_head_dim)
    //
    // With per-head normalization, all blocks share the same scale,
    // so we can compute the combined scale factor once.
    
    float nope_scale = scale_q_nope * scale_k_nope;
    float rope_scale = scale_q_rope * scale_k_rope;
    float total_head_dim = static_cast<float>(head_dim_nope + head_dim_rope);
    
    // Approximation: weight by dimension ratio
    // (More sophisticated: track separate accumulators)
    float combined = (nope_scale * head_dim_nope + rope_scale * head_dim_rope) 
                   / total_head_dim;
    
    return combined / std::sqrt(total_head_dim);
}
```

### MLA KV Cache Storage

The KV cache for MLA models stores NOPE and ROPE components separately:

```cpp
// MLA KV Cache layout
struct MLAKVCacheConfig {
    Q16BlockSize nope_block_size = Q16BlockSize::BLOCK_128;  // 128-dim
    Q16BlockSize rope_block_size = Q16BlockSize::BLOCK_64;   // 64-dim
    Q16BlockSize v_block_size = Q16BlockSize::BLOCK_128;     // 128-dim (no split)
    
    int n_kv_heads;
    int max_seq_len;
};

// Cache stores 4 separate tensors:
// K_nope: [n_kv_heads, max_seq_len, 128]  as Q16_1Block_128
// K_rope: [n_kv_heads, max_seq_len, 64]   as Q16_1Block_64
// V:      [n_kv_heads, max_seq_len, 128]  as Q16_1Block_128
// (No V_nope/V_rope split - V is standard)
```

### FA2 Prefill with MLA

The tiled prefill path also uses dual dot products:

```cpp
// MLA FA2 Prefill (tiled GEMM)
void q16_qk_gemm_tile_mla(
    const Q16_1Block_128* Q_nope,  // [tile_q × 1 block]
    const Q16_1Block_128* K_nope,  // [tile_k × 1 block]
    const Q16_1Block_64* Q_rope,   // [tile_q × 1 block]
    const Q16_1Block_64* K_rope,   // [tile_k × 1 block]
    int32_t* scores,                // [tile_q × tile_k]
    int tile_q, int tile_k
) {
    // Outer loops over query and key positions
    for (int q = 0; q < tile_q; ++q) {
        for (int k = 0; k < tile_k; ++k) {
            int32_t nope = q16_dot_single<Q16_1Block_128>(&Q_nope[q], &K_nope[k], 1);
            int32_t rope = q16_dot_single<Q16_1Block_64>(&Q_rope[q], &K_rope[k], 1);
            scores[q * tile_k + k] = nope + rope;
        }
    }
}
```

### Why This Design Works

| Aspect | Standard Attention | MLA Attention |
|--------|-------------------|---------------|
| **Block types used** | Single (64 or 128) | Two (128 + 64) |
| **Dot products per score** | 1 | 2 (summed in INT32) |
| **Scale pairs per head** | 1 | 2 (NOPE + ROPE) |
| **KV cache tensors** | 2 (K, V) | 4 (K_nope, K_rope, V) |
| **Inner loop changes** | None | None (same microkernels) |
| **Orchestration changes** | None | New dispatch path |

The key advantage of our templated microkernel design:
- `q16_dot_single<Q16_1Block_128>()` and `q16_dot_single<Q16_1Block_64>()` are **already implemented**
- MLA support is purely **orchestration** - calling the same inner loops twice
- No new SIMD code required for MLA models

### Implementation Checklist for MLA Support

```
[ ] Phase 5.3: MLA Microkernel Support
    [ ] Add MLAAttentionParams struct to Q16IntegerAttentionRef.h
    [ ] Implement q16_qk_gemv_mla() in Q16DotProduct.cpp
    [ ] Implement q16_qk_gemm_tile_mla() in Q16DotProduct.cpp  
    [ ] Add compute_mla_qk_scale() helper
    [ ] Extend UnifiedKVCache for separate NOPE/ROPE storage
    [ ] Update GraphOrchestrator to detect MLA architecture from model config
    [ ] Create Test__MLAAttention.cpp with DeepSeek V3 config
    [ ] E2E parity test with PyTorch MLA reference
```

---

## v2 Solution Part B: Per-Head Scale Normalization

### Core Insight

**Normalize Q, K, V to a single scale per head at pipeline boundaries**, not during attention:

| Tensor | Where Normalized | When |
|--------|------------------|------|
| **Q** | FusedQKVGEMMStage (after projection + bias + RoPE) | Every forward pass |
| **K** | KV Cache write | Once per token |
| **V** | KV Cache write | Once per token |

After normalization:
- All blocks within a head share **one scale factor**
- Integer dot products accumulate without per-block scale tracking
- Scales are combined **once** at attention output, not per-element

### Normalized Q16_1 Format

```cpp
// Standard Q16_1Block (per-32-element scale)
struct Q16_1Block {
    float d;           // Per-block scale
    int32_t sum_qs;    // Sum of quantized values
    int16_t qs[32];    // Quantized values
};

// For attention, we store per-head metadata alongside the tensor
struct Q16_1HeadMetadata {
    float head_scale;      // Max |d| across all blocks in this head
    int num_blocks;        // Blocks per head (head_dim / 32)
};

// Normalized representation (logical, not new struct):
// qs_normalized[i] ≈ qs[i] × (block.d / head_scale)
// Actual value = qs_normalized[i] × head_scale
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Q16_1 INTEGER ATTENTION v2                                │
│                (Per-Head Scale Normalization)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  PREPROCESSING: FusedQKVGEMMStage                                      ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  Input × Wq/Wk/Wv → FP32 Q, K, V                                      ║  │
│  ║  + Bias (if present)                                                   ║  │
│  ║  + RoPE                                                                ║  │
│  ║  ───────────────────────────────────────────────────────────────────  ║  │
│  ║  NORMALIZE Q: For each head h:                                         ║  │
│  ║    1. Quantize to Q16_1 blocks                                        ║  │
│  ║    2. Q_head_scale[h] = max(|block.d|) across head's blocks           ║  │
│  ║    3. Requantize: qs_new = qs × (block.d / Q_head_scale) (INT16)     ║  │
│  ║    4. Set all block.d = Q_head_scale[h]                               ║  │
│  ║  Output: Q_normalized[seq_len × n_heads × head_dim] + Q_head_scales   ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  KV CACHE WRITE (K and V normalization)                                ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║  For new K, V tokens being cached:                                     ║  │
│  ║    1. Quantize to Q16_1 blocks                                        ║  │
│  ║    2. K_head_scale = max(|block.d|), V_head_scale = max(|block.d|)    ║  │
│  ║    3. Requantize to unified scale (same process as Q)                 ║  │
│  ║    4. Store normalized K, V in cache with head scales                 ║  │
│  ║                                                                        ║  │
│  ║  Note: Head scales may evolve as new tokens added. Options:            ║  │
│  ║    A. Track running max, renormalize when scale increases >2×         ║  │
│  ║    B. Use fixed conservative scale based on model statistics          ║  │
│  ║    C. Normalize per-token-group (e.g., every 64 tokens)               ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                              │                                               │
│                              ▼                                               │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  Q16 INTEGER ATTENTION KERNEL                                          ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  INPUTS (all normalized to per-head scales):                           ║  │
│  ║    Q: INT16[seq_len_q × n_heads × head_dim] + Q_head_scale[n_heads]   ║  │
│  ║    K: INT16[kv_len × n_kv_heads × head_dim] + K_head_scale[n_kv_heads]║  │
│  ║    V: INT16[kv_len × n_kv_heads × head_dim] + V_head_scale[n_kv_heads]║  │
│  ║                                                                        ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 1: Q×K^T → INT32 Scores                                    │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ for each query q in [0, seq_len_q):                              │ ║  │
│  ║  │   for each key k in [0, kv_end):                                 │ ║  │
│  ║  │     int32_t score = 0;                                           │ ║  │
│  ║  │     for (int i = 0; i < head_dim; ++i)                           │ ║  │
│  ║  │       score += Q_int16[q][i] × K_int16[k][i];  // VPDPWSSD      │ ║  │
│  ║  │     scores[k] = score;                                           │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ Combined scale = Q_head_scale × K_head_scale × (1/√head_dim)     │ ║  │
│  ║  │ (Applied ONCE at output, not per-element)                        │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 2: Score Normalization + Exp2FixedSoftmax                  │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Find max for numerical stability (INT32)                      │ ║  │
│  ║  │ int32_t max_score = max(scores[0..kv_end]);                      │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Map to LUT input range: INT32 → INT8 [-128, 0]                │ ║  │
│  ║  │ // score_shift chosen so (max_score - min_score) >> shift ≈ 128  │ ║  │
│  ║  │ int score_shift = compute_score_shift(max_score, min_score);     │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ for each k:                                                       │ ║  │
│  ║  │   int32_t relative = scores[k] - max_score;  // Always ≤ 0       │ ║  │
│  ║  │   int8_t lut_idx = clamp(relative >> score_shift, -128, 0);      │ ║  │
│  ║  │   weights_int16[k] = exp2_lut[lut_idx + 128];  // [0, 32767]    │ ║  │
│  ║  │   weight_sum += weights_int16[k];                                │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 3: P×V → INT32 Context (VPDPWSSD)                          │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ int32_t context[head_dim] = {0};                                 │ ║  │
│  ║  │ for each k in [0, kv_end):                                       │ ║  │
│  ║  │   int16_t w = weights_int16[k];                                  │ ║  │
│  ║  │   for (int d = 0; d < head_dim; ++d)                             │ ║  │
│  ║  │     context[d] += w × V_int16[k][d];  // INT16×INT16→INT32      │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // All V values share V_head_scale - no per-element tracking!    │ ║  │
│  ║  │ Context scale = V_head_scale / weight_sum × (1/32767)            │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 4: Wo Projection (VPDPWSSD)                                │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Requantize INT32 context → INT16 for VPDPWSSD                 │ ║  │
│  ║  │ int16_t ctx_int16[n_heads × head_dim];                           │ ║  │
│  ║  │ float ctx_scale = requantize_int32_to_int16(context, ctx_int16); │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Wo projection: INT16 context × INT8 weights (sign-extended)   │ ║  │
│  ║  │ int32_t proj[d_model] = {0};                                     │ ║  │
│  ║  │ for (int n = 0; n < d_model; ++n)                                │ ║  │
│  ║  │   for (int k = 0; k < input_dim; ++k)                            │ ║  │
│  ║  │     proj[n] += ctx_int16[k] × wo_int16[k][n];  // VPDPWSSD      │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ // Requantize INT32 → Q16_1                                      │ ║  │
│  ║  │ combined_scale = ctx_scale × wo_block_scales[...]               │ ║  │
│  ║  │ requantize_int32_to_q16_1(proj, combined_scale, output);        │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║  ┌──────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │ STAGE 5: Residual Add (Q16_1 + Q16_1 → Q16_1)                    │ ║  │
│  ║  │                                                                   │ ║  │
│  ║  │ simd::q16_1_add_q16_1(residual_in, projection, residual_out);   │ ║  │
│  ║  │ // Already implemented correctly in v1                           │ ║  │
│  ║  └──────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  OUTPUT: Q16_1[seq_len_q × d_model] (residual stream)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Precision Flow Summary (Attention Kernel)

This table covers the **integer attention kernel** specifically. For the full forward pass precision flow, see [Design Goal: Full Integer Residual Stream](#design-goal-full-integer-residual-stream) above.

| Stage | Input Types | Accumulator | Output | FP32 Usage |
|-------|-------------|-------------|--------|------------|
| Block size selection | Model config | - | Block size enum | None |
| Q normalization | Q16_1 (block scales) | - | Q16_1 (head scale) | Scale computation only |
| K/V normalization | Q16_1 (block scales) | - | Q16_1 (head scale) | Scale computation only |
| Q×K^T | INT16 × INT16 | **INT32** | INT32 scores | None |
| Softmax | INT32 scores | INT32 | INT16 weights | None (LUT) |
| P×V | INT16 × INT16 | **INT32** | INT32 context | None |
| Wo projection | INT16 × INT16 | **INT32** | Q16_1 | Scale factor only |
| Residual add | Q16_1 + Q16_1 | - | Q16_1 | None |

**FP32 is used ONLY for scale factors, never for data computation.**

### Critical: QKV Projection Uses Q8_1 GEMM

The attention kernel receives Q16_1 Q, K, V as inputs. However, these are produced by the **FusedQKVGEMMStage** which uses Q8_1 GEMM for efficiency:

```
Q16_1 residual → [RMSNorm] → Q16_1 norm → [Q8_1 convert] → Q8_1 activations
                                                              ↓
                                        [Q8_1 × Q8_0 GEMM] → Q8_1 Q, K, V
                                                              ↓
                                              [RoPE Q8→Q16] → Q16_1 Q, K, V (normalized)
                                                              ↓
                                         [Integer Attention Kernel starts here]
```

This Q16_1 → Q8_1 → Q16_1 roundtrip is **intentional** — see "QKV Projection Exception" section above.

---

## Q16_1 Codebase Audit: Files Requiring Block Size Updates

This section documents ALL locations in the codebase where Q16_1 operations assume the fixed 32-element block size. Each must be updated to support the new variable block sizes (64, 128, 192).

### Legend
- ⬜ Not started
- 🟨 In progress
- ✅ Complete

---

### A. Core Block Structure Definition

| File | Function/Struct | Current State | Action Required |
|------|-----------------|---------------|-----------------|
| [BlockStructures.h](../../src/v2/tensors/BlockStructures.h) | `Q16_1Block` | Fixed 32-element | ⬜ Keep for GEMM, add `Q16_1Block_64/128/192` |

---

### B. Q16_1Tensor Class

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | Constructor (5 overloads) | Uses `Q16_1Block::BLOCK_SIZE` | ⬜ Add block_size parameter |
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | `fp32_data()` | Hardcoded 32-element decode | ⬜ Dispatch on block_size |
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | `copyFrom()` | 32-element block loop | ⬜ Dispatch on block_size |
| [Q16_1Tensor.cpp](../../src/v2/tensors/Q16_1Tensor.cpp) | `applyRoPE()` | Uses 32-block kernel | ⬜ Add block_size dispatch |
| [TensorFactory.cpp](../../src/v2/tensors/TensorFactory.cpp) | `createQ16_1()` | Default 32-block | ⬜ Add block_size parameter |

---

### C. RoPE Operations (Q8_1 → Q16_1 and Q16_1 → Q16_1)

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [RoPEPrimitives.h](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h#L569) | `apply_rope_q8_1_to_q16<T>()` | ✅ Templatized on OutBlockType | ✅ **DONE** - Supports 32/64/128/192 |
| [RoPEPrimitives.h](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h#L521) | `apply_rope_q16_1_integer()` | In-place 32-block | ⬜ Add block_size param |
| [RoPEPrimitives.h](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.h#L465) | `apply_rope_q16_1_integer_head()` | Per-head RoPE (32-block) | ⬜ Templatize on BlockType |
| [RoPEPrimitives.cpp](../../src/v2/kernels/cpu/primitives/RoPEPrimitives.cpp#L3593) | `apply_rope_q8_1_to_q16_head<T>()` | ✅ Templatized, AVX2/AVX512 | ✅ **DONE** - 5.6x/10x speedup |
| [CPURoPEKernelT.cpp](../../src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp#L644) | `apply_q8_1_to_q16_1()` | Kernel dispatch (32-block) | ⬜ Add block_size dispatch |
| [CPURoPEKernelT.cpp](../../src/v2/kernels/cpu/ops/CPURoPEKernelT.cpp#L717) | `apply_typed()` (Q16_1) | In-place Q16_1 RoPE | ⬜ Add block_size dispatch |
| [RoPEStage.cpp](../../src/v2/execution/compute_stages/stages/RoPEStage.cpp) | Stage execution | Creates 32-block output | ⬜ Pass block_size to kernel |

---

### D. Residual Add Operations

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2211) | `q16_1_add_fp32()` | `n_blocks = count / 32` | ⬜ Templatize on BlockType |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2614) | `q16_1_add_q8_1()` | `n_blocks = count / 32` | ⬜ Templatize on BlockType |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2024) | `q16_1_add_q16_1()` | 32-element loop | ⬜ Templatize on BlockType |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L1781) | `q16_1_sum_multi()` | MPI reduction helper | ⬜ Templatize on BlockType |

---

### E. MPI Allreduce for Q16_1

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [MPIContext.h](../../src/v2/utils/MPIContext.h#L132) | `allreduce_q16_1_inplace()` | `Q16_1Block*` (32-element) | ⬜ Templatize or add block_size |
| [AllreduceStage.cpp](../../src/v2/execution/compute_stages/stages/AllreduceStage.cpp#L86) | Q16_1 path | Uses `mutable_q16_1_blocks()` | ⬜ Get block_size from tensor |

---

### F. KV Cache (Q16_1 Storage)

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [UnifiedKVCache.h](../../src/v2/tensors/UnifiedKVCache.h#L183) | `UnifiedKVCacheTensor<Q16_1>` | Specialization | ⬜ Add block_size to config |
| [UnifiedKVCache.cpp](../../src/v2/tensors/UnifiedKVCache.cpp#L52) | `allocate_tensor()` | 32-block allocation | ⬜ Pass block_size |
| [UnifiedKVCache.cpp](../../src/v2/tensors/UnifiedKVCache.cpp#L111) | `copy_append_data()` | 32-block copy | ⬜ Dispatch on block_size |
| [UnifiedKVCache.cpp](../../src/v2/tensors/UnifiedKVCache.cpp#L172) | `shift_evict_data()` | 32-block shift | ⬜ Dispatch on block_size |
| [KVCacheAppendStage.cpp](../../src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp#L280) | Hardcoded comment | `block_size = 32` | ⬜ Get from tensor |

---

### G. Quantization/Dequantization

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [QuantizeToQ16_1Stage.h](../../src/v2/execution/compute_stages/stages/QuantizeToQ16_1Stage.h) | FP32→Q16_1 | 32-block output | ⬜ Add block_size param |
| [SIMDHelpers.h](../../src/v2/tensors/SIMDHelpers.h#L2447) | `q16_1_to_q8_1()` | 32-block conversion | ⬜ Templatize on BlockType |
| [WoProjectionVNNIRef.deprecated.cpp](../../src/v2/kernels/cpu/attention/q16_1/ref/microkernels/WoProjectionVNNIRef.deprecated.cpp#L158) | `requantize_fp32_to_q16_1()` | 32-block output | ⬜ Fresh impl in v2 kernel |

---

### H. Attention Kernel Infrastructure

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [JitQ16FusedAttention.h](../../src/v2/kernels/cpu/attention/q16_1/jit/JitQ16FusedAttention.h#L131) | `blocks_per_head()` | `Q16_1Block::BLOCK_SIZE` | ⬜ Use configurable block_size |
| [Q16FusedAttentionKernel.cpp](../../src/v2/kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp) | `convert_params()` | Assumes 32-block | ⬜ Add block_size dispatch |
| [FusedAttentionWoStage.cpp](../../src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp#L194) | Debug dump loop | `Q16_1Block::BLOCK_SIZE` | ⬜ Get from tensor |
| [Q8_1 JitFusedAttentionWo.h](../../src/v2/kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h#L6443) | Q16_1 residual write | `Q16_1_BLOCK_SIZE = 32` | ⬜ Get from params |

---

### I. Buffer Size Calculations

| File | Location | Current State | Action Required |
|------|----------|---------------|-----------------|
| [BufferRole.h](../../src/v2/execution/BufferRole.h#L328) | `BLOCK_SIZE = 32` | Hardcoded constant | ⬜ Make configurable |
| [EmbeddingStage.cpp](../../src/v2/execution/compute_stages/stages/EmbeddingStage.cpp#L236) | `block_size = 32` | Hardcoded | ⬜ Get from config |

---

### J. RMSNorm for Q16_1 Input

| File | Function | Current State | Action Required |
|------|----------|---------------|-----------------|
| [KernelFactory.cpp](../../src/v2/kernels/KernelFactory.cpp#L1469) | `createRMSNorm(Q16_1Tensor*)` | Dequants 32-block | ⬜ Dispatch on block_size |
| [CPURMSNormKernelT](../../src/v2/kernels/cpu/ops/) | `<Q16_1>` specialization | 32-block dequant | ⬜ Add block_size dispatch |

---

### K. HybridQ16 Mode Integration

| File | Component | Current State | Action Required |
|------|-----------|---------------|-----------------|
| [HybridPrecisionConfig.h](../../src/v2/execution/HybridPrecisionConfig.h) | Q16_1 config | No block_size field | ⬜ Add block_size config |
| [Qwen2Graph.cpp](../../src/v2/models/qwen/Qwen2Graph.cpp) | HybridQ16 wiring | Assumes 32-block | ⬜ Pass block_size to stages |
| [GraphOrchestrator.cpp](../../src/v2/execution/GraphOrchestrator.cpp) | Buffer allocation | 32-block buffers | ⬜ Use model-aware sizes |

---

### Summary Statistics

| Category | Files | Functions | Priority |
|----------|-------|-----------|----------|
| **Core Tensor** | 3 | 8 | **P0** (blocking) |
| **RoPE** | 3 | 7 | **P0** (blocking) |
| **Residual Add** | 1 | 4 | **P1** (required) |
| **MPI Allreduce** | 2 | 2 | **P1** (required) |
| **KV Cache** | 2 | 5 | **P1** (required) |
| **Quantize/Dequant** | 3 | 3 | **P1** (required) |
| **Attention Kernel** | 4 | 4 | **P2** (v2 kernel) |
| **Buffer Calcs** | 2 | 2 | **P2** (integration) |
| **RMSNorm** | 2 | 2 | **P2** (integration) |
| **HybridQ16** | 3 | 3 | **P3** (after core) |
| **Total** | **25** | **40** | |

### Recommended Implementation Order

1. **Phase A** (Blocking): Core tensor + RoPE
   - `BlockStructures.h`: Add new block types
   - `Q16_1Tensor`: Add block_size member, update constructors
   - `RoPEPrimitives`: Templatize on BlockType

2. **Phase B** (Required): Residual + MPI + KV Cache
   - `SIMDHelpers.h`: Templatize q16_1 operations
   - `MPIContext`: Add block_size to allreduce
   - `UnifiedKVCache`: Add block_size config

3. **Phase C** (v2 Kernel): Fresh attention implementation
   - `Q16IntegerAttentionRef`: Already uses new block types ✅
   - Kernel uses `Q16_1Block_64/128/192` directly

4. **Phase D** (Integration): Wire it all together
   - `HybridPrecisionConfig`: Add block_size config
   - `Qwen2Graph`: Pass block_size through pipeline
   - `GraphOrchestrator`: Model-aware buffer allocation

---

## Implementation Phases

### Phase 0: Variable Block Size Infrastructure

#### Task 0.1: Define Block Size Enum and Formats
```cpp
// src/v2/tensors/Q16BlockFormats.h

enum class Q16BlockSize : uint32_t {
    BLOCK_32 = 32,    // Legacy, GEMM compatibility
    BLOCK_64 = 64,    // Universal for attention (GCD), MLA ROPE
    BLOCK_128 = 128,  // Optimal for head_dim=128, MLA NOPE/V
    BLOCK_192 = 192,  // Available but NOT recommended for MLA (see MLA section)
};

struct Q16_1Block_64 {
    float d;
    int32_t sum_qs;
    int16_t qs[64];
};

struct Q16_1Block_128 {
    float d;
    int32_t sum_qs;
    int16_t qs[128];
};

struct Q16_1Block_192 {
    float d;
    int32_t sum_qs;
    int16_t qs[192];
};
```

#### Task 0.2: Model-Aware Block Size Selection
```cpp
// src/v2/loaders/BlockSizeSelector.h

struct AttentionBlockSizes {
    Q16BlockSize q_block_size;     // For standard Q
    Q16BlockSize k_block_size;     // For standard K
    Q16BlockSize v_block_size;     // For V
    // MLA-specific (optional, only set for MLA architectures)
    Q16BlockSize nope_block_size;  // For Q_nope, K_nope (128-dim)
    Q16BlockSize rope_block_size;  // For Q_rope, K_rope (64-dim)
    bool is_mla;                   // True for DeepSeek V3, Kimi K2
};

AttentionBlockSizes select_attention_block_sizes(const ModelConfig& cfg) {
    if (cfg.architecture == Architecture::MLA) {
        // DeepSeek V3, Kimi K2: separate NOPE (128) + ROPE (64)
        return {
            .q_block_size = Q16BlockSize::BLOCK_128,   // Not used directly
            .k_block_size = Q16BlockSize::BLOCK_128,   // Not used directly
            .v_block_size = Q16BlockSize::BLOCK_128,   // 1 block/head - optimal!
            .nope_block_size = Q16BlockSize::BLOCK_128, // 1 block/head
            .rope_block_size = Q16BlockSize::BLOCK_64,  // 1 block/head
            .is_mla = true
        };
    }
    
    switch (cfg.head_dim) {
        case 64:
            return { Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_64, 
                     Q16BlockSize::BLOCK_64, {}, {}, false };
        case 128:
            return { Q16BlockSize::BLOCK_128, Q16BlockSize::BLOCK_128,
                     Q16BlockSize::BLOCK_128, {}, {}, false };
        default:
            return { Q16BlockSize::BLOCK_64, Q16BlockSize::BLOCK_64,
                     Q16BlockSize::BLOCK_64, {}, {}, false };
    }
}
```

#### Task 0.3: Update Q16 Tensor to Support Variable Block Size
- [ ] Add `block_size` field to Q16_1Tensor metadata
- [ ] Implement `typed_data<BlockType>()` accessor
- [ ] Update quantization/dequantization for each block size

#### Task 0.4: Update GGUF Loader for New Block Formats
- [ ] Register `Q16_1_64` and `Q16_1_128` type codes
- [ ] Add block size detection from tensor metadata
- [ ] Implement conversion path from 32-block to 64/128-block at load time

### Phase 1: Per-Head Scale Metadata Infrastructure

#### Task 1.1: Add Head Scale Storage
- [ ] Add `head_scales` array to attention kernel params
- [ ] Define storage location (alongside tensor? separate buffer?)
- [ ] Update `Q16FusedAttentionWoResidualParams` struct

#### Task 1.2: Exp2FixedSoftmax Integration Check
- [ ] Verify `Exp2FixedSoftmaxRef` exists and is functional
- [ ] Ensure LUT covers INT8 input range [-128, 0] → INT16 [0, 32767]
- [ ] Add score-shift computation helper

### Phase 2: Q Normalization in FusedQKVGEMMStage

#### Task 2.1: Add Normalization Option to FusedQKVGEMMStage
```cpp
struct FusedQKVGEMMStageParams {
    // ... existing fields ...
    Q16BlockSize q_block_size = Q16BlockSize::BLOCK_64;  // NEW
    bool normalize_q_to_head_scale = false;  // NEW
    float* q_head_scales_out = nullptr;      // NEW: [num_heads]
};
```

#### Task 2.2: Implement Q Normalization Logic
```cpp
// After Q projection + bias + RoPE, before storing to output:
if (params.normalize_q_to_head_scale && params.q_head_scales_out) {
    const int blocks_per_head = head_dim / static_cast<int>(params.q_block_size);
    
    for (int h = 0; h < num_heads; ++h) {
        if (blocks_per_head == 1) {
            // OPTIMAL: Single block per head, no normalization needed!
            params.q_head_scales_out[h] = Q_q16_block[h].d;
        } else {
            // Multiple blocks: normalize to head scale
            float head_scale = compute_head_max_scale(Q_q16, h, head_dim);
            params.q_head_scales_out[h] = head_scale;
            normalize_head_to_scale(Q_q16, h, head_dim, head_scale);
        }
    }
}
```

#### Task 2.3: Unit Tests for Q Normalization
- [ ] Test: normalized blocks all have same scale
- [ ] Test: dequantized values match original within tolerance
- [ ] Test: head_scale equals max of original block scales
- [ ] Test: single-block heads skip normalization (optimal path)

### Phase 3: K/V Normalization in KV Cache

#### Task 3.1: Add Normalization to KVCacheUpdateStage
```cpp
struct KVCacheUpdateParams {
    // ... existing fields ...
    Q16BlockSize k_block_size = Q16BlockSize::BLOCK_64;  // NEW
    Q16BlockSize v_block_size = Q16BlockSize::BLOCK_128; // NEW (often different!)
    bool normalize_to_head_scale = false;     // NEW
    float* k_head_scales = nullptr;           // NEW: [num_kv_heads], running max
    float* v_head_scales = nullptr;           // NEW: [num_kv_heads], running max
};
```

#### Task 3.2: Implement K/V Normalization Strategy

**Option A: Conservative Fixed Scale** (Recommended for v2.0)
```cpp
// Use model-wide statistics to set conservative fixed scales
// Avoids renormalization complexity - no cache re-quantization needed
// Default: ±8.0 range configured in GraphSchema::kv_cache_scale
// Architecture overrides in Qwen2Schema, LlamaSchema, etc.

// The scale is set in GraphSchema and flows through to KVCacheUpdateParams:
struct GraphSchema {
    // ...
    float kv_cache_scale = 8.0f;  // Default ±8.0 range, covers typical activations
};

// Usage in KV cache stage:
float fixed_scale = schema.kv_cache_scale;  // e.g., 8.0f
// Q16 range: [-32767, 32767] × (scale / 32767) = [-scale, +scale]
```

**Why ±8.0 default?**
- Post-RMSNorm activations typically fall in [-3, 3] range
- K/V linear projections can amplify slightly (typical max ~4.0)
- 8.0 provides ~2× headroom over typical maximum values
- Use profiling tool (Phase 10) to determine tighter scales per-model

**Option B: Running Max with Renormalization** (Future)
```cpp
// Track running max, renormalize cache when scale increases significantly
if (new_max > current_head_scale * RENORM_THRESHOLD) {
    renormalize_cache_for_head(cache, layer, head, new_max);
    head_scale = new_max;
}
```

#### Task 3.3: Update KV Cache Tensor Type
- [ ] Ensure KV cache stores normalized INT16 + head scale metadata
- [ ] Update `UnifiedKVCache<Q16_1>` to track head scales
- [ ] Support different block sizes for K vs V (important for MLA!)

#### Task 3.4: Unit Tests for K/V Normalization
- [ ] Test: cache entries have consistent head scale
- [ ] Test: dequantized K/V values match non-normalized baseline
- [ ] Test: fixed scale handles typical value ranges without clipping

### Phase 4: Integer-Only Attention Kernel

#### Task 4.1: Create Q16IntegerAttentionRef (New File)
```
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h
```

**Do NOT modify Q16FusedAttentionRef.cpp** - create fresh implementation.

#### Task 4.2: Implement Stage 1 - Integer Q×K^T
```cpp
// Pure INT16×INT16→INT32, no scale tracking during loop
// Works with any block size as long as inputs are pre-normalized
void compute_qk_scores_int32(
    const int16_t* Q,      // Normalized Q [head_dim]
    const int16_t* K,      // Normalized K [kv_len × head_dim]  
    int kv_len,
    int head_dim,
    int32_t* scores_out)   // [kv_len]
{
    for (int kv = 0; kv < kv_len; ++kv) {
        int32_t score = 0;
        for (int d = 0; d < head_dim; ++d) {
            score += static_cast<int32_t>(Q[d]) * static_cast<int32_t>(K[kv * head_dim + d]);
        }
        scores_out[kv] = score;
    }
}
```

#### Task 4.3: Implement Stage 2 - Exp2FixedSoftmax
```cpp
// INT32 scores → INT16 weights via LUT
void exp2_fixed_softmax_int32_to_int16(
    const int32_t* scores,
    int kv_len,
    int16_t* weights_out,
    int32_t* weight_sum_out)
{
    // Find max score
    int32_t max_score = scores[0];
    for (int i = 1; i < kv_len; ++i)
        max_score = std::max(max_score, scores[i]);
    
    // Compute score range for shift calculation
    int32_t min_score = scores[0];
    for (int i = 1; i < kv_len; ++i)
        min_score = std::min(min_score, scores[i]);
    
    // Adaptive shift: map score range to ~128 LUT entries
    int score_shift = compute_adaptive_shift(max_score - min_score);
    
    // Apply LUT
    int32_t sum = 0;
    for (int i = 0; i < kv_len; ++i) {
        int32_t relative = scores[i] - max_score;  // ≤ 0
        int8_t lut_idx = static_cast<int8_t>(
            std::max(-128, std::min(0, static_cast<int>(relative >> score_shift))));
        weights_out[i] = exp2_lut_256[lut_idx + 128];  // [0, 32767]
        sum += weights_out[i];
    }
    *weight_sum_out = sum;
}
```

#### Task 4.4: Implement Stage 3 - Integer P×V
```cpp
// INT16 weights × INT16 V → INT32 context
void compute_pv_int32(
    const int16_t* weights,  // [kv_len]
    const int16_t* V,        // [kv_len × head_dim]
    int kv_len,
    int head_dim,
    int32_t* context_out)    // [head_dim]
{
    std::memset(context_out, 0, head_dim * sizeof(int32_t));
    for (int kv = 0; kv < kv_len; ++kv) {
        int16_t w = weights[kv];
        if (w == 0) continue;
        for (int d = 0; d < head_dim; ++d) {
            context_out[d] += static_cast<int32_t>(w) * static_cast<int32_t>(V[kv * head_dim + d]);
        }
    }
}
```

#### Task 4.5: Implement Stage 4 - Integer Wo Projection
```cpp
// INT32 context → INT16 → VPDPWSSD → INT32 → Q16_1
void wo_projection_int32(
    const int32_t* context,      // [input_dim]
    const WoPackedWeights* Wo,   // INT8 packed weights
    int input_dim,
    int d_model,
    int32_t weight_sum,
    float v_head_scale,
    float q_head_scale,
    float k_head_scale,
    Q16_1Block* output)
{
    // Step 1: Requantize context INT32 → INT16
    int16_t* ctx_int16 = ...;
    float ctx_scale;
    requantize_int32_to_int16_uniform(context, input_dim, ctx_int16, &ctx_scale);
    
    // Step 2: VPDPWSSD GEMV (INT16 × INT16 → INT32)
    int32_t* proj_int32 = ...;
    for (int n = 0; n < d_model; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < input_dim; ++k) {
            int16_t w = sign_extend_int8(Wo->get(k, n));
            acc += static_cast<int32_t>(ctx_int16[k]) * static_cast<int32_t>(w);
        }
        proj_int32[n] = acc;
    }
    
    // Step 3: Compute combined output scale
    // output = (context_int32 / weight_sum × V_scale) × ctx_scale × Wo_scale
    float combined_scale = ctx_scale * v_head_scale / static_cast<float>(weight_sum)
                         / 32767.0f  // weight normalization
                         * ...;      // Wo block scales
    
    // Step 4: Requantize INT32 → Q16_1
    requantize_int32_to_q16_1(proj_int32, d_model, combined_scale, output);
}
```

#### Task 4.6: Full Kernel Integration
```cpp
bool q16_integer_attention_decode(const Q16IntegerAttentionParams& params) {
    // Validate all inputs are normalized
    LLAMINAR_ASSERT(params.q_head_scale > 0, "Q must be normalized");
    LLAMINAR_ASSERT(params.k_head_scale > 0, "K must be normalized");
    LLAMINAR_ASSERT(params.v_head_scale > 0, "V must be normalized");
    
    for (int h = 0; h < params.num_heads; ++h) {
        // Stage 1: Q×K^T
        compute_qk_scores_int32(Q_head, K_head, kv_len, head_dim, scores);
        
        // Stage 2: Softmax
        exp2_fixed_softmax_int32_to_int16(scores, kv_len, weights, &weight_sum);
        
        // Stage 3: P×V
        compute_pv_int32(weights, V_head, kv_len, head_dim, context);
        
        // Concatenate context for Wo
        concat_context(context, full_context, h, head_dim);
    }
    
    // Stage 4: Wo projection
    wo_projection_int32(full_context, Wo, input_dim, d_model,
                        weight_sum, v_head_scale, q_head_scale, k_head_scale,
                        projection);
    
    // Stage 5: Residual add
    simd::q16_1_add_q16_1(residual_in, projection, residual_out, d_model);
    
    return true;
}
```

### Phase 5: Integration and Testing

#### Task 5.1: Create Q16_INTEGER_V2 Backend
- [ ] Add `Q16_INTEGER_V2` to `FusedAttentionBackend` enum
- [ ] Wire to new `Q16IntegerAttentionRef` kernel
- [ ] Update `FusedAttentionWoStage` dispatch
- [ ] Add block size configuration to backend params

#### Task 5.2: Update GraphOrchestrator for Block Sizes + Normalization
- [ ] Detect model architecture (MLA vs Standard)
- [ ] Select appropriate block sizes based on head dimensions
- [ ] Enable Q normalization when using Q16_INTEGER_V2
- [ ] Pass head scales to attention kernel params
- [ ] Configure KV cache normalization with appropriate block sizes

#### Task 5.3: Parity Tests
- [ ] Create `Test__Q16IntegerAttention_vs_FP32.cpp`
- [ ] Target: ≥0.95 cosine similarity for ATTENTION_CONTEXT
- [ ] Compare against both FP32 baseline and current Q16FusedAttentionRef
- [ ] Test with 64-block and 128-block configurations

#### Task 5.4: Layer-by-Layer Validation
- [ ] Run layer-by-layer comparison similar to existing HybridQ16 test
- [ ] Verify no layer shows catastrophic degradation (>10% cosine drop)
- [ ] Specifically validate layer 22/23 (problem layers in v1)

### Phase 6: MLA Architecture Support (DeepSeek V3, Kimi K2)

#### Task 6.1: MLA-Specific Attention Kernel
```cpp
// MLA has split Q/K: NOPE (128-dim) + ROPE (64-dim)
// Stored as SEPARATE tensors with their own scales (NOT merged into 192-block)
struct MLAAttentionParams {
    // Q/K split structure
    const int16_t* Q_nope;      // [n_heads × 128]
    const int16_t* Q_rope;      // [n_heads × 64]
    const int16_t* K_nope;      // [kv_len × n_kv_heads × 128]
    const int16_t* K_rope;      // [kv_len × n_kv_heads × 64]
    float q_nope_scale, q_rope_scale;
    float k_nope_scale, k_rope_scale;
    
    // V is standard 128-dim
    const int16_t* V;           // [kv_len × n_kv_heads × 128]
    float v_head_scale;         // Single scale (128-block optimal!)
    
    // ... rest of params
};

// MLA dot product naturally splits
void compute_mla_qk_scores_int32(const MLAAttentionParams& params, ...) {
    for (int kv = 0; kv < kv_len; ++kv) {
        // Nope dot (2 × 64-blocks → can normalize to 1 scale)
        int32_t nope_score = dot_int32(Q_nope, K_nope[kv], 128);
        
        // Rope dot (1 × 64-block → already optimal!)
        int32_t rope_score = dot_int32(Q_rope, K_rope[kv], 64);
        
        // Combine with only 2 scale pairs (not 3!)
        scores[kv] = nope_score + rope_score;
    }
}
```

#### Task 6.2: MLA KV Cache Layout
- [ ] Store K_nope and K_rope separately in cache
- [ ] K_nope: 128-dim as 1×Q16_1Block_128 (1 block/head - optimal!)
- [ ] K_rope: 64-dim as 1×Q16_1Block_64 (1 block/head - optimal!)
- [ ] V: 128-dim as 1×Q16_1Block_128 (1 block/head - optimal!)

#### Task 6.3: MLA Unit Tests
- [ ] Test: nope/rope split produces same scores as combined 192-dim approach
- [ ] Test: MLA-specific normalization works correctly
- [ ] Test: All tensors achieve 1-block-per-head optimal path

### Phase 7: Performance Optimization (Future)

#### Task 7.1: SIMD Vectorization
- [ ] Vectorize `compute_qk_scores_int32` with VPDPWSSD
- [ ] Vectorize `compute_pv_int32` with VPDPWSSD
- [ ] Vectorize Wo projection
- [ ] Specialized versions for 64-block vs 128-block

#### Task 7.2: JIT Implementation
- [ ] Port scalar reference to Xbyak JIT
- [ ] Reuse existing register guard system
- [ ] Target: 2-3× speedup over scalar
- [ ] Block-size-specialized codegen

---

## Key Differences from v1

| Aspect | v1 (Failed) | v2 (Proposed) |
|--------|-------------|---------------|
| **Block size** | Fixed 32 | Model-aware (64, 128) |
| **Scale handling** | Per-block, tracked at runtime | Per-head, normalized upfront |
| **Optimal path** | Never | 1-block-per-head skips normalization |
| **MLA support** | None | Native nope/rope split |
| **Softmax** | `std::exp()` FP32 | Exp2FixedSoftmax LUT |
| **V scale tracking** | Per-element FP64 | Single head scale |
| **Implementation** | Modified existing file | Fresh implementation |
| **FP32 in compute** | Everywhere | Never (scales only) |
| **Validation** | Post-hoc discovery | Built-in assertions |

---

## Risk Mitigation

### Risk 1: Normalization Precision Loss
**Concern**: Requantizing to head scale may lose precision  
**Mitigation**: 
- Head scale is max of block scales, so ratios are ≤1.0
- INT16 has enough headroom for typical value distributions
- Monitor clipping rate in normalization

### Risk 2: KV Cache Scale Evolution
**Concern**: As tokens accumulate, scale may need updating  
**Mitigation**:
- Start with fixed conservative scale
- Add renormalization support in v2.1 if needed
- Profile actual scale distributions on real prompts

### Risk 3: Softmax LUT Precision
**Concern**: 256-entry LUT may be too coarse  
**Mitigation**:
- Adaptive shift maps actual score range to LUT
- exp2 is well-behaved for softmax (no exp overflow issues)
- Can increase to 512 or 1024 entries if needed

---

## Success Criteria

1. **Precision**: ≥0.95 cosine similarity vs FP32 on ATTENTION_CONTEXT
2. **Integer Purity**: Zero FP32 operations in data path (verified by code review)
3. **No Layer Collapse**: All layers within 5% cosine of average
4. **Documentation Accuracy**: Comments match actual implementation

---

## File Manifest

### New Files (Implemented)
```
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp      # Macrokernel
src/v2/kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.h
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProduct.h    # QK dot product
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProduct.cpp
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulate.h     # P×V accumulation
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulate.cpp
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/WoProjection.h     # Wo projection
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/WoProjection.cpp
tests/v2/unit/microkernels/Test__Q16DotProduct.cpp                     # 11 tests
tests/v2/unit/microkernels/Test__PVAccumulate.cpp                      # 12 tests
tests/v2/unit/microkernels/Test__Q16WoProjection.cpp                   # 12 tests
```

### New Files (Pending - MLA Support)
```
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/MLADotProduct.h    # MLA QK dual dot
src/v2/kernels/cpu/attention/q16_1/ref/microkernels/MLADotProduct.cpp
src/v2/tensors/MLAKVCache.h                                             # NOPE/ROPE split cache
src/v2/tensors/MLAKVCache.cpp
tests/v2/unit/microkernels/Test__MLADotProduct.cpp
tests/v2/integration/Test__Q16IntegerAttention_vs_FP32.cpp
tests/v2/unit/Test__Q16Normalization.cpp
tests/v2/unit/Test__Q16BlockFormats.cpp
python/tools/profile_kv_activations.py                                  # KV scale profiler
```

### Modified Files
```
src/v2/execution/RuntimeConfig.h                  # Add Q16_INTEGER_V2 enum
src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.h/cpp  # Q normalization + block size
src/v2/execution/compute_stages/stages/KVCacheUpdateStage.h/cpp # K/V normalization + block size
src/v2/execution/compute_stages/stages/FusedAttentionWoStage.cpp # Dispatch
src/v2/execution/GraphSchema.h                    # Add kv_cache_scale field
src/v2/models/qwen/Qwen2Schema.h                  # Document kv_cache_scale usage
src/v2/tensors/UnifiedKVCache.h/cpp              # Head scale storage + variable block size
src/v2/loaders/GGUFLoader.cpp                     # Block size selection based on model
```

### Unchanged (Reference Only)
```
src/v2/kernels/cpu/attention/q16_1/ref/Q16FusedAttentionRef.cpp  # Keep as cautionary tale
```

---

## Phase 10: KV Cache Scale Profiling Tool

### Overview

The profiling tool addresses the "Conservative Fixed Scale" approach by empirically determining optimal `kv_cache_scale` values for different model architectures. This eliminates the "growing scale" problem while maximizing quantization precision.

### Task 10.1: Create Profiling Script

**File**: `python/tools/profile_kv_activations.py`

```python
#!/usr/bin/env python3
"""
KV Cache Activation Profiler for Q16_1 Integer Attention

Profiles K/V activations across transformer layers to determine optimal
fixed scales for the Conservative Fixed Scale quantization strategy.

Usage:
    python profile_kv_activations.py \
        --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
        --prompts data/diverse_prompts.txt \
        --output kv_scale_report.json
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ActivationStats:
    min_val: float
    max_val: float
    mean: float
    std: float
    p99: float      # 99th percentile
    p999: float     # 99.9th percentile  
    p9999: float    # 99.99th percentile
    
def recommend_scale(stats: ActivationStats, headroom: float = 2.0) -> float:
    """Recommend kv_cache_scale with headroom above p99.9"""
    return stats.p999 * headroom

# Hook-based capture extending existing snapshot infrastructure
class KVActivationProfiler:
    def __init__(self, model):
        self.k_activations: Dict[int, List[torch.Tensor]] = {}
        self.v_activations: Dict[int, List[torch.Tensor]] = {}
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        for layer_idx, layer in enumerate(model.layers):
            # Hook K projection output
            layer.self_attn.k_proj.register_forward_hook(
                lambda m, inp, out, idx=layer_idx: 
                    self.k_activations.setdefault(idx, []).append(out.detach()))
            # Hook V projection output  
            layer.self_attn.v_proj.register_forward_hook(
                lambda m, inp, out, idx=layer_idx:
                    self.v_activations.setdefault(idx, []).append(out.detach()))
    
    def compute_stats(self) -> Dict[str, ActivationStats]:
        results = {}
        for layer_idx in self.k_activations:
            k_all = torch.cat(self.k_activations[layer_idx]).flatten()
            v_all = torch.cat(self.v_activations[layer_idx]).flatten()
            
            results[f"layer{layer_idx}_K"] = ActivationStats(
                min_val=k_all.min().item(),
                max_val=k_all.max().item(),
                mean=k_all.mean().item(),
                std=k_all.std().item(),
                p99=np.percentile(k_all.numpy(), 99),
                p999=np.percentile(k_all.numpy(), 99.9),
                p9999=np.percentile(k_all.numpy(), 99.99))
            # ... similar for V
        return results
```

### Task 10.2: Representative Prompt Dataset

Create diverse prompt set covering:
- Short prompts (1-10 tokens)
- Medium prompts (50-200 tokens) 
- Long context (1000+ tokens)
- Code, prose, dialog, technical content
- Multiple languages (if multilingual model)

### Task 10.3: Run Profiling on Target Models

| Model | Expected K Range | Expected V Range | Recommended Scale |
|-------|------------------|------------------|-------------------|
| Qwen2.5-0.5B | TBD | TBD | TBD |
| Qwen2.5-7B | TBD | TBD | TBD |
| Llama-3-8B | TBD | TBD | TBD |
| DeepSeek V3 | TBD | TBD | TBD |

### Task 10.4: Update Architecture Schemas

After profiling, update per-architecture schemas:

```cpp
// If Qwen2 profiling shows max ~3.5 with p99.9 ~3.2:
GraphSchema createSchema() const override {
    GraphSchema schema;
    // ...
    schema.kv_cache_scale = 4.0f;  // Tighter than default 8.0, better precision
    return schema;
}
```

### Expected Outcomes

1. **Model-specific optimal scales** - Better precision than conservative default
2. **Validation of ±8.0 default** - Confirm it covers all models with headroom
3. **Documentation** - Per-model activation characteristics for future reference
4. **Regression test data** - Known activation ranges for E2E validation

---

## Timeline Estimate

| Phase | Tasks | Estimated Effort | Status |
|-------|-------|------------------|--------|
| Phase 0 | Deprecate v1 kernel, scaffold v2 | 2-3 hours | ✅ Complete |
| Phase 1 | Variable block size infrastructure | 3-4 hours | ✅ Complete |
| Phase 2 | Core tensor + RoPE block size support | 4-5 hours | ✅ Complete |
| Phase 3 | Residual + MPI + KV Cache | 4-5 hours | ✅ Complete |
| Phase 4 | Per-head scale normalization | 3-4 hours | ✅ Complete |
| Phase 5 | Wo projection microkernels | 2-3 hours | ✅ Complete |
| Phase 5.1 | QK dot product microkernels | 2-3 hours | ✅ Complete |
| Phase 5.2 | PV accumulation microkernels | 2-3 hours | ✅ Complete |
| Phase 5.3 | **MLA microkernel support** | 3-4 hours | 🔜 Next |
| Phase 6 | Integer residual add | 2-3 hours | Pending |
| Phase 7 | FA2 Prefill tiled implementation | 4-6 hours | Pending |
| Phase 8 | Unit tests for Q16IntegerAttention | 3-4 hours | Pending |
| Phase 9 | E2E parity tests | 4-6 hours | Pending |
| Phase 10 | KV Cache Scale Profiling Tool | 4-6 hours | Pending |
| **Completed** | Phases 0-5.2 | **~22-28 hours** | ✅ |
| **Remaining (core)** | Phases 5.3-9 | **~16-23 hours** | |
| **Remaining (with profiling)** | Phases 5.3-10 | **~20-29 hours** | |

---

## Appendix A: Model Block Size Reference

| Model | Architecture | Q/K head_dim | V head_dim | Q/K Block | V Block | Blocks/Head |
|-------|-------------|--------------|------------|-----------|---------|-------------|
| Qwen2.5-0.5B | Standard | 64 | 64 | 64 | 64 | **1** (optimal) |
| GPT-2 | Standard | 64 | 64 | 64 | 64 | **1** (optimal) |
| Qwen3-8B | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| Llama-3-8B | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| MiniMax-M1 | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| Mistral-7B | Standard | 128 | 128 | 128 | 128 | **1** (optimal) |
| DeepSeek V3 | MLA | 128+64 | 128 | 128/64 | 128 | **1** (separate NOPE/ROPE) |
| Kimi K2 | MLA | 128+64 | 128 | 128/64 | 128 | **1** (separate NOPE/ROPE) |

**Key insight**: With model-aware block sizes, **all model families get optimal 1-block-per-head path**:
- Standard 64-dim models → 64-block
- Standard 128-dim models → 128-block  
- MLA models → 128-block NOPE + 64-block ROPE (separate tensors, separate scales)

No normalization needed for any model when using the correct block size!
// For i=0   (input=-128): LUT[0] = ~0 (negligible weight)
static const int16_t exp2_lut_256[256] = {
    0, 0, 0, 0, 0, 0, 0, 0,     // -128 to -121: essentially zero
    // ... gradual increase ...
    32767                        // 0: full weight
};
```

The adaptive shift computation:
```cpp
int compute_adaptive_shift(int32_t score_range) {
    // Target: map score_range to ~128 LUT entries (7 bits)
    // shift = log2(score_range) - 7, minimum 0
    if (score_range <= 128) return 0;
    int shift = 0;
    while ((score_range >> shift) > 128) shift++;
    return shift;
}
```
