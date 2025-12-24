# JIT Fused Attention-Wo Kernel: Architecture Analysis

This document provides a comprehensive analysis of the JIT Fused Attention-Wo kernel implementation, covering operation order, cache blocking, tiling strategy, register pressure, and performance characteristics.

## Executive Summary

The kernel is **well-designed** for streaming attention workloads with good cache locality and register pressure management. There are **no major cache thrashing issues**, but there are a few areas where we're paying unavoidable costs due to algorithmic constraints.

---

## 1. High-Level Operation Order

### DECODE Mode (seq_len_q = 1, typical inference)

```
For each query position q:
    For each attention head h (0..num_heads):
        ┌─────────────────────────────────────────────────────────────┐
        │ PHASE 1: Q LOAD (once per head)                             │
        │   • Load Q[q,h] blocks into zmm10-13 (persistent)           │
        │   • XOR with 0x80 for unsigned VNNI format                  │
        └─────────────────────────────────────────────────────────────┘
        ┌─────────────────────────────────────────────────────────────┐
        │ PHASE 2: KV LOOP (FA2 tiled, 4 or 8 positions per tile)     │
        │   Tile loop: kv = 0..seq_len_kv step 4/8                    │
        │     ├─ PREFETCH K[kv+N] (N varies by work-size)             │
        │     ├─ μK1: Q·K dot products ×4 → 4 scores                  │
        │     ├─ μK2: Tile max reduction → single max                 │
        │     ├─ μK2: Softmax state update (max, sum) ONCE            │
        │     ├─ μK3: Context rescale by correction factor ONCE       │
        │     ├─ μK5: exp(score - max) → 4 weights                    │
        │     ├─ μK2: sum += weight (×4)                              │
        │     ├─ PREFETCH V[kv+N]                                     │
        │     └─ μK3: Interleaved V accum (4 rows, overlapped)        │
        └─────────────────────────────────────────────────────────────┘
        ┌─────────────────────────────────────────────────────────────┐
        │ PHASE 3: NORMALIZE (once per head)                          │
        │   • context[h] *= 1/sum                                     │
        │   • Store to context buffer                                 │
        └─────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────┐
    │ PHASE 4: Wo PROJECTION (once per query)                         │
    │   μK4: output = context[all_heads] × Wo[d_model, d_model]       │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Cache Blocking Strategy

### Memory Footprint per Component

| Component | Size (Qwen 7B, head_dim=128, num_heads=28, seq_len_kv=2048) |
|-----------|-------------------------------------------------------------|
| **Q** (1 head) | 4 blocks × 36B = **144B** → fits in L1 |
| **K** (1 position) | 4 blocks × 36B = **144B** → streamed |
| **V** (1 position) | 4 blocks × 36B = **144B** → streamed |
| **Context** (1 head) | 128 floats × 4B = **512B** → fits in registers + L1 |
| **Context buffer** (all heads) | 28 × 128 × 4B = **14KB** → fits in L1 |
| **Wo** (full) | 3584 × 3584 × 4B = **~49MB** → DRAM |

### Blocking Decisions

1. **Q**: Loaded once per head, kept in **zmm10-13** (4 ZMM = 256B). No re-fetching during KV loop. ✅

2. **K/V**: **Streamed** with software prefetch. Each position loaded once, used immediately, discarded:
   - SMALL (kv ≤ 1024): `prefetcht0` (L1), 4 positions ahead
   - LARGE: `prefetcht1` (L2), 16 positions ahead  
   - XL (kv > 4096): `prefetcht2` (L3), 64 positions ahead

3. **Context accumulators**: 
   - **Blocks 0-1**: Kept in **zmm0-3** (register-resident, zero cache traffic) ✅
   - **Blocks 2+**: **Spilled to stack** (L1 hit, 128B per extra block per KV iteration)

4. **Wo**: No blocking. GEMV does **64 columns × K rows** per outer loop, streaming through Wo row-by-row. Memory-bound.

---

## 3. Tiling Strategy Analysis

### FA2 KV Tiling (4x or 8x)

**Why tile?** Amortize softmax state updates and context rescaling.

| Operation | FA1 (1x) | FA2 4x | FA2 8x |
|-----------|----------|--------|--------|
| Q·K dots | 1 | 4 | 8 |
| Softmax state updates | 1 | 1 | 1 |
| Context rescales | 1 | 1 | 1 |
| V accum calls | 1 | 1 (interleaved 4x) | 2 (interleaved 4x each) |

**FA2 4x Register Budget**:

| Zone | Registers | Usage During Tile |
|------|-----------|-------------------|
| Accum (zmm0-7) | 8 | Context blocks 0-1 (zmm0-3), temporary (zmm4-7) |
| Input (zmm8-15) | 8 | Q data (zmm10-13), K/V loading (zmm8-9) |
| State (zmm16-19) | 4 | max, sum, weight, correction |
| Scratch (zmm20-25) | 6 | Scores (xmm20-23), temp (zmm24-25) |
| Const (zmm26-31) | 6 | 0x80, scale, log2e, exp_min, 1.0f |

**Key Insight**: 4 scores fit in **xmm20-23** (alias zmm20-23 lower 128-bits). This is why we borrow Score0-3, then release before borrowing Scratch0-3 for weights.

### V Interleaving Pattern

Instead of: `Load V[0], FMA, Load V[1], FMA, ...`

We do:
```
Load V[0] scale → Load V[0] data → dequant V[0]
Load V[1] scale | FMA ctx += V[0]*w0 | Load V[1] data
dequant V[1]    | Load V[2] scale    | FMA ctx += V[1]*w1 | Load V[2] data
... (pipeline continues)
```

**Result**: Memory latency (~100 cycles for L3, ~300 for DRAM) is hidden by FMA execution.

---

## 4. Cache Reuse Analysis

### What We're Reusing Well ✅

| Data | Reuse Factor | How |
|------|--------------|-----|
| **Q** | 1× load, seq_len_kv× use | Stays in zmm10-13 entire KV loop |
| **Context blocks 0-1** | Entire KV loop | Stays in zmm0-3 |
| **Constants** | Entire kernel | Pinned in zmm26-31 |
| **Softmax state** | Entire head | Stays in zmm16-19 |

### What We're Streaming (Intentionally)

| Data | Access Pattern | Why Streaming is Correct |
|------|----------------|--------------------------|
| **K** | Sequential, each position 1× | seq_len_kv × head_dim > L2 |
| **V** | Sequential, each position 1× | Same reasoning |

### Context Spill Traffic

For **head_dim > 64** (2+ blocks), blocks 2+ are spilled:

```
Per KV iteration, per extra block:
  Load spilled context: 128B (2 ZMM loads)
  Store spilled context: 128B (2 ZMM stores)
  = 256B per block per KV iteration
```

For head_dim=128 (4 blocks), blocks 2-3 are spilled:
```
Per KV iteration: 2 blocks × 256B = 512B
For seq_len_kv=2048: 512B × 2048 = 1MB total spill traffic per head
```

**Is this cache thrashing?** No! This hits **L1** (spill area is at known stack offset, hot in cache). The 1MB amortizes across 2048 iterations = **512B/iteration** which is:
- 8 cache line loads + 8 stores = 16 cache operations
- At ~4 cycles/L1 hit = ~64 cycles
- vs. ~300 cycles for a single DRAM round-trip

**Verdict**: Spilling is acceptable, not thrashing.

---

## 5. Cache Thrashing Analysis

### Places We Checked

| Area | Status | Analysis |
|------|--------|----------|
| **K/V streaming** | ✅ OK | Prefetch hides latency, sequential access |
| **Context spill** | ✅ OK | L1-resident stack, not evicting working data |
| **Wo projection** | ⚠️ Concern | 49MB Wo matrix, row-major GEMV |
| **Q reloads** | ✅ OK | Registers, no reloads |

### Wo Projection: The Real Bottleneck

The Wo projection is a **GEMV** (d_model outputs × d_model reduction):
```cpp
for n = 0 to d_model step 64:   // Outer loop: 64 outputs
    acc[0:63] = 0
    for k = 0 to d_model:        // Inner loop: dot product
        wo_row = Wo[k, n:n+63]   // Load 256 bytes
        ctx_k = context[k]       // Broadcast 4 bytes
        acc += ctx_k * wo_row
    store output[n:n+63]
```

**Problem**: Wo is accessed **row-major**, but we're iterating N (columns) in outer loop, K (rows) in inner. Each inner iteration touches a **different cache line** of Wo.

For d_model=3584:
- Inner loop: 3584 iterations × 256B = **~900KB** of Wo touched per outer iteration
- That's larger than L2, so we're streaming from L3/DRAM

**Mitigation**: The code uses **prefetcht0** for next row, 4 rows ahead.

**Verdict**: Wo is memory-bound but correctly prefetched. No cache **thrashing** (repeated eviction+reload of same data). Each Wo row loaded exactly once.

---

## 6. Register Pressure Assessment

### Zone Occupancy During Critical Phases

#### Phase: FA2 4x Tile (Q·K scoring)

| Zone | Registers | Active Usage |
|------|-----------|--------------|
| **Accum** (zmm0-7) | 8 | Context (zmm0-3), vpdpbusd temp (ymm4-7) |
| **Input** (zmm8-15) | 8 | Q unsigned (zmm10-13), K load (ymm8), K scales (xmm9) |
| **State** (zmm16-19) | 4 | max, sum (only 2 active here) |
| **Scratch** (zmm20-25) | 6 | Scores (xmm20-23), correction (xmm25), scale (xmm24) |
| **Const** (zmm26-31) | 6 | All 6 used |

**Occupancy**: ~24/32 ZMM (**75%**). Not spilling due to register pressure.

#### Phase: V Interleaved Accumulation

| Zone | Registers | Active Usage |
|------|-----------|--------------|
| **Accum** (zmm0-7) | 8 | Context (zmm0-3), V data (zmm4-5 or zmm6-7 for spilled) |
| **Input** (zmm8-15) | 8 | V lo/hi pair 0 (zmm8-9), V lo/hi pair 1 (zmm10-11), V scale (zmm12) |
| **State** (zmm16-19) | 4 | Not used here |
| **Scratch** (zmm20-25) | 6 | Weights broadcast (zmm20-23) |

**Occupancy**: ~20/32 ZMM (**63%**). Healthy headroom.

#### Phase: Wo Projection

| Zone | Registers | Active Usage |
|------|-----------|--------------|
| **Accum** (zmm0-7) | 8 | 4 output accumulators (zmm0-3), context broadcast (zmm4), Wo loads (zmm5-7) |
| **Scratch** (zmm20) | 1 | Extra Wo load |

**Occupancy**: ~9/32 ZMM (**28%**). Register-light phase.

### Stack Spills

| Spill Type | When | Frequency | Cost |
|------------|------|-----------|------|
| **Context blocks 2+** | During KV loop | Every KV iteration | L1 hit (~4 cycles) |
| **Scores (FA2 8x)** | Between 4x tile pairs | Once per 8x tile | L1 hit |
| **seq_len_kv, position_offset** | Between heads | Once per head | L1 hit |

**Verdict**: No excessive spilling. All spills are to **L1-hot stack locations**.

---

## 7. Identified Inefficiencies

### Minor Issues (Not Worth Fixing)

1. **d_Q loaded from memory** in dot product loop instead of registers:
   - Why: Saving zmm registers for more important data (Q unsigned, accumulators)
   - Cost: 4 bytes per block × num_blocks loads per KV position
   - Verdict: Acceptable, L1 hit

2. **Scalar horizontal reductions** in Q·K dot product:
   - Why: vpdpbusd produces 8×int32, need scalar score
   - Cost: ~10 instructions per score
   - Verdict: Unavoidable with Q8_1 format

3. **Context spill for head_dim > 64**:
   - Why: Only 4 ZMM pairs for context, need 2 ZMM per block
   - Cost: L1 load/store traffic
   - Verdict: Correct trade-off (L1 vs. register pressure)

### Non-Issues (Design Decisions)

1. **Wo not blocked**: Correct for GEMV where M=1. Blocking helps GEMM (M>1).

2. **No K/V caching across heads**: Would need massive buffer. Streaming is better.

---

## 8. Summary and Recommendations

### What We're Doing Right ✅

1. **Q register residency**: Zero reloads during KV loop
2. **FA2 tiling**: Amortizes softmax overhead 4-8×
3. **V interleaving**: Hides memory latency effectively
4. **Work-size prefetch tuning**: Adapts to KV length
5. **Context blocks 0-1 in registers**: Avoids spill for common head_dim=64
6. **Register zone discipline**: Clear ownership, no conflicts

### Performance Characteristics

| Workload | Bottleneck | Utilization |
|----------|------------|-------------|
| **Small KV (≤256)** | Compute (FMA throughput) | High |
| **Medium KV (256-2048)** | Balanced | Good |
| **Large KV (>4096)** | Memory (K/V streaming) | Limited by DRAM BW |
| **Wo projection** | Memory (Wo streaming) | Always memory-bound |

### Potential Future Optimizations

1. **FP8 K/V**: Would halve memory traffic, double effective cache capacity
2. **Wo packing**: Column-major layout could improve GEMV locality
3. **Multi-query batching**: For batch decode, could amortize Wo loads (see Section 10)

---

## 10. Cache-Aware Wo Batching (Optimization Opportunity)

The Wo projection is the primary bottleneck for decode. Currently, we do a GEMV (m=1) for each query, loading the entire 49MB Wo matrix from DRAM each time.

### The Problem

```
Current decode flow:
for each query q:
    Phase 1: Compute attention context[q] for all heads
    Phase 2: GEMV: output[q] = Wo × context[q]  ← Loads 49MB from DRAM
```

For batch decode with N queries, we load Wo **N times** from DRAM.

### The Solution: Batched Wo Projection

Accumulate multiple context vectors before doing the Wo multiplication:

```
Batched flow:
for batch_start = 0 to num_queries step BATCH_SIZE:
    Phase 1: For each q in batch:
        Compute attention → store to batch_context[q_local]
    
    Phase 2: GEMM: output[batch] = Wo × batch_context  ← Loads 49MB ONCE
```

### Cache Budget Analysis

Use `CacheInfo` from `v2/utils/CPUFeatures.h` to determine optimal batch size:

```cpp
#include "v2/utils/CPUFeatures.h"

const CacheInfo& cache = cache_info();
int batch = cache.optimal_wo_batch_size(d_model);
// Example: d_model=3584, L2=1MB → batch=16

// Context buffer sizing:
// batch=16 × d_model=3584 × 4B = 224KB (22% of L2)
// Remaining 78% of L2 available for Wo tiles during GEMM
```

### Expected Speedup

| Batch Size | Wo Loads | Speedup (Wo-bound) |
|------------|----------|-------------------|
| 1 (current) | N | 1× |
| 4 | N/4 | ~4× |
| 8 | N/8 | ~8× |
| 16 | N/16 | ~16× |

**Note**: Actual speedup depends on attention compute time. If attention dominates, batching helps less. Use profiling to measure real impact.

### Implementation Status

- ✅ `CacheInfo` struct for cache detection (`CPUFeatures.h`)
- ✅ `optimal_wo_batch_size()` calculates L2-aware batch size
- ✅ `print_analysis()` for debugging cache decisions
- ✅ Batched decode loop in `JitFusedAttentionWo.h`
- ✅ `emit_wo_projection_batched()` for multi-query GEMM
- ✅ VNNI-packed Q8_1 Wo weights for batched projection

---

## 11. VNNI-Optimized Wo Projection (December 2024)

### Overview

The Wo projection kernel now supports **AVX-512 VNNI-packed Q8_1 weights**, eliminating the need for runtime dequantization and significantly improving throughput for the memory-bound Wo GEMV/GEMM.

### New `WoFormat::Q8_1_VNNI_PACKED`

A new weight format optimized for VNNI dot products:

| Format | Storage | Compute Path |
|--------|---------|--------------|
| `WoFormat::FP32` | 4B/element | FMA |
| `WoFormat::FP16` | 2B/element | Convert → FMA |
| `WoFormat::BF16` | 2B/element | Convert → FMA |
| `WoFormat::Q8_1` | 36B/32 elements | Dequant → FMA |
| `WoFormat::Q8_1_VNNI_PACKED` | 36B/32 elements | **Direct VNNI** |

### Memory Layout

```
Q8_1_VNNI_PACKED block (36 bytes):
┌────────────────────────────────────────────────────────────┐
│ d (FP16, 2B) │ sum_qs (INT16, 2B) │ qs[32] (INT8, 32B)     │
└────────────────────────────────────────────────────────────┘

Weights are packed so that 4 consecutive rows share the same 
K-dimension block alignment, enabling 4-row parallel VNNI ops.
```

### VNNI Dot Product Flow

```cpp
// For 4 output rows simultaneously:
for k_block = 0 to num_blocks:
    // Load weight scales (4 FP16 → 4 FP32)
    d0, d1, d2, d3 = vcvtph2ps(Wo_packed[row0..row3].d)
    
    // Load INT8 weights (4 × 32 bytes)
    w0_lo = vpmovsxbd(Wo[row0].qs[0:15])   // 16 INT8 → 16 INT32
    w0_hi = vpmovsxbd(Wo[row0].qs[16:31])
    // ... same for w1, w2, w3
    
    // Load context (FP32 → quantize to INT8 on the fly)
    ctx_q8 = quantize_to_int8(context[k_block * 32 : +32])
    
    // VNNI 4-way dot product
    acc0 += vpdpbusd(ctx_q8, w0)  // Unsigned × Signed
    acc1 += vpdpbusd(ctx_q8, w1)
    acc2 += vpdpbusd(ctx_q8, w2)
    acc3 += vpdpbusd(ctx_q8, w3)

// Finalize: acc * d * ctx_scale → output
```

### Key Implementation Functions

| Function | Purpose |
|----------|---------|
| `emit_wo_projection_vnni_inline()` | Single-query VNNI Wo projection (DECODE) |
| `emit_wo_projection_vnni_inline_with_reg_offset()` | Batched VNNI projection with dynamic offset |
| `llaminar2_wo_q8_1_vnni_packed_gemm()` | External C function for batched GEMM |

### Fallback Path Fixes (December 2024)

The fallback path (`emit_wo_projection_with_reg_offset`) was updated to correctly handle all weight formats when VNNI is unavailable:

| Format | Fix Applied |
|--------|-------------|
| `BF16` | Use `vpmovzxwd` + `vpslld` for BF16 → FP32 conversion |
| `FP16` | Use `vcvtph2ps` for FP16 → FP32 conversion |
| `Q8_1` | Added `emit_wo_projection_q8_1_with_reg_offset()` for raw block handling |

### Performance Characteristics

| Scenario | Before (FP32 Wo) | After (Q8_1 VNNI) | Speedup |
|----------|------------------|-------------------|---------|
| Wo memory traffic | 49MB (7B model) | 12.25MB | 4× reduction |
| Batched (m=8) | Memory-bound | Compute-bound | ~3-4× |
| Single query (m=1) | Memory-bound | Memory-bound | ~1.5× |

### Usage

```cpp
// Configure JIT kernel for VNNI-packed weights
JitAttentionConfig config;
config.wo_format = WoFormat::Q8_1_VNNI_PACKED;
config.batch_size = 8;  // Batched decode

JitFusedAttentionWo kernel(config);
kernel.compute(Q, K, V, Wo_vnni_packed, output, ...);
```

### Requirements

- **Hardware**: AVX-512 VNNI (Ice Lake, Sapphire Rapids, or newer)
- **Fallback**: Automatically falls back to dequantize+FMA path on non-VNNI CPUs

---

## 12. Cache-Aware Kernel Configuration (December 2024)

### Overview

The attention kernel now derives its heuristics from **detected CPU cache sizes** rather than hardcoded thresholds, improving portability across different microarchitectures (Zen 4, Golden Cove, Sapphire Rapids, etc.).

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `AttentionCacheConfig` | `CPUFeatures.h` | Computes config from cache sizes |
| `AttentionWorkSize` | `CPUFeatures.h` | SMALL/LARGE/XL enum |
| `PrefetchConfig` | `CPUFeatures.h` | Distance and cache level |
| `effectivePrefetchDistance()` | `JitFusedAttentionWo.h` | JIT config method |

### Cache-Derived Heuristics

| Parameter | Derivation |
|-----------|------------|
| **WorkSizeClass** | KV footprint vs L2/L3 capacity |
| **Prefetch Distance** | Cache line count × latency hiding target |
| **FA2 Tile Width** | KV8 if per-head KV fits L2, else KV4 |

### Memory Model

```
KV footprint per head (Q8_1):
  bytes_per_pos = (head_dim / 32) × 36 × 2  (K + V)
  footprint = bytes_per_pos × kv_seq_len

Work size classification:
  SMALL: footprint ≤ 50% L2 → L1 prefetch (prefetcht0)
  LARGE: footprint ≤ L3/8   → L2 prefetch (prefetcht1)
  XL:    footprint > L3/8   → L3 prefetch (prefetcht2)
```

### Example Configuration

For Qwen2 0.5B (head_dim=64) at kv_seq_len=2048:
```
Per-head footprint: ((64+31)/32) × 36 × 2 × 2048 = 147,456 bytes ≈ 144 KB
L2 threshold (256KB × 50%): 128 KB
→ WorkSize: LARGE (spills L2)
→ Prefetch: 16 positions to L2 (prefetcht1)
→ FA2 Tile: KV8 (fits per-iteration working set)
```

### API Usage

```cpp
// Automatic cache-aware configuration in FusedAttentionWoKernel
AttentionCacheConfig cache_cfg(head_dim, num_kv_heads, kv_seq_len);

// Map to JIT config
jit_config.work_size = /* from cache_cfg.work_size() */;
jit_config.prefetch_distance = cache_cfg.prefetch_config().distance;
jit_config.prefetch_level = cache_cfg.prefetch_config().cache_level;
jit_config.fa2_tile_width = cache_cfg.prefer_kv8_tile() ? KV8 : KV4;
```

### Debug Output

```cpp
// Print detailed cache analysis
AttentionCacheConfig cfg(128, 4, 4096);
cfg.print_config();
```

Output:
```
╔════════════════════════════════════════════════════════════╗
║      Cache-Aware Attention Configuration                   ║
╠════════════════════════════════════════════════════════════╣
║ Input Parameters:                                          ║
║   head_dim:      128                                       ║
║   num_kv_heads:    4                                       ║
║   kv_seq_len:   4096                                       ║
╠════════════════════════════════════════════════════════════╣
║ Memory Footprint:                                          ║
║   Per head:     1152 KB                                    ║
║   Total:        4608 KB                                    ║
╠════════════════════════════════════════════════════════════╣
║ Derived Configuration:                                     ║
║   WorkSize:   LARGE (per-head vs L2/L3)                    ║
║   Prefetch:   16 positions → L2 (prefetcht1)               ║
║   FA2 Tile:   KV4 (KV8 spills L2)                          ║
╚════════════════════════════════════════════════════════════╝
```

### Implementation Status

- ✅ `AttentionCacheConfig` struct in `CPUFeatures.h`
- ✅ `work_size()` derives SMALL/LARGE/XL from KV footprint
- ✅ `prefetch_config()` computes distance and cache level
- ✅ `prefer_kv8_tile()` based on L2 capacity
- ✅ `FusedAttentionWoKernel.h` uses cache config for JIT selection
- ✅ `JitFusedAttentionWo.h` uses `effectivePrefetchDistance/Level()`
- ✅ All 4 prefetch code blocks updated to cache-aware API

---

## 13. Microkernel Reference

| μK | Name | File | Purpose |
|----|------|------|---------|
| μK1 | `JitQ8DotProduct` | `jit/JitQ8DotProduct.h` | Q·K scoring with VNNI |
| μK2 | `JitOnlineSoftmax` | `jit/JitOnlineSoftmax.h` | Online softmax state |
| μK3 | `JitVWeightedAccum` | `jit/JitVWeightedAccum.h` | V accumulation |
| μK4 | `JitWoProjectionOptimized` | `jit/JitWoProjectionOptimized.h` | Wo GEMV |
| μK5 | `JitFastExp` | `jit/JitFastExp.h` | Fast exp approximation |
| μK6 | `emit_wo_projection_vnni_inline` | `jit/JitFusedAttentionWo.h` | VNNI Wo projection |

---

## 14. Conclusion

**No major issues found.** The kernel is well-architected for single-query decode with:
- Register allocation matching data reuse patterns
- Cache blocking where it matters (Q, context)
- Streaming where blocking doesn't help (K, V, Wo)
- Latency hiding through prefetch and interleaving
- **Cache-aware configuration** for portability across CPUs

The main limitation is inherent: attention is memory-bound for large sequences, and Wo projection is always memory-bound for decode. The kernel is doing the right things given these constraints.

**Recent Improvements**:
- VNNI-packed Wo weights for direct integer dot products
- Adaptive prefetch distance based on detected cache sizes
- WorkSizeClass derived from KV footprint vs L2/L3 capacity

The cache-aware approach means the kernel will automatically adapt to:
- Desktop CPUs (smaller L2/L3): More conservative prefetch, earlier SMALL→LARGE transitions
- Server CPUs (larger L2/L3): Longer prefetch distances, KV8 tiles for larger contexts
- Different microarchitectures (Intel vs AMD): Cache detection via CPUID
