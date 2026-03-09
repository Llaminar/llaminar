# Native-VNNI GEMM Kernel Design Proposal

## 1. Executive Summary

**Goal**: Design a new family of GEMM kernels (`ROCmQuantisedGemmKernel_native_VNNI.hip`) that execute M>1 prefill using native sub-8-bit quantization formats (1-bit to 6-bit), bypassing the current INT8 intermediary path.

**Why**: Today, M>1 prefill for all quantization formats goes through the INT8 VNNI GEMM pipeline: weights are dequantized to INT8, VNNI-packed into `[K/4 × N × 4]`, and processed by V3/V7 kernels. This works but discards per-block FP16 scales (replaced by a single global scale pair), and reads 8 bpw of weight data regardless of the original quantization. A native-VNNI GEMM kernel would:

1. **Read fewer bytes**: Q4_0 reads 4.5 bpw instead of 8 bpw — **1.78× less HBM traffic**
2. **Preserve accuracy**: Per-block FP16 scales are lossless (same as GEMV path)
3. **Enable lower-BPW models for prefill**: Q2_K at 2.6 bpw = **3.1× bandwidth savings**

**Target hardware**: AMD MI50/MI60 (gfx906), 60 CUs, 4 SIMDs/CU, 256 VGPRs/SIMD, 64 KB LDS/CU, 16 KB L1I$/CU.

**Scope**: All 17 `NativeVNNIFormat` values currently supported by the GEMV kernel, plus the 18th (IQ4_XS). The kernel set is independent of the existing INT8 GEMM V3/V7 kernels but draws architectural inspiration from their LDS-pipelined designs.

---

## 2. The Core Challenge: Decode-in-LDS vs Decode-in-Register

The fundamental architectural decision is **where** to decode quantized weights to INT8 for `v_dot4_i32_i8` consumption. This choice dominates the entire kernel design.

### Option A: Decode-in-Load (Global → Decode → INT8 LDS)

```
Global Memory (native quant)  →  Cooperative Decode (ALU)  →  INT8 data in LDS
                                                                    ↓
                                                             v_dot4 from LDS
```

**How it works**: During cooperative B-tile loading, threads decode quantized payloads into INT8 and write the decoded INT8 bytes into LDS — exactly the same layout as INT8 VNNI's `b_lds[buf][kk * N_TILE + ni]`. The compute phase is then identical to INT8 V3/V7: pure `v_dot4` from LDS.

**Pros**:
- Compute phase is unchanged from INT8 GEMM — proven, optimal `v_dot4` inner loop
- Clear separation of concerns: load+decode vs compute
- Decode ALU naturally overlaps with A-tile loads and compute of previous tile
- LDS layout is format-agnostic after decode

**Cons**:
- LDS stores are INT8 (full 8 bpw), so LDS bandwidth saving is zero — only HBM reads are reduced
- Decode VGPRs (for staging raw payload) coexist with compute accumulators — VGPR pressure
- Per-block FP16 scales cannot be folded into `v_dot4` (INT32 accumulation) — scales must be stored separately and applied per-block in the epilogue or during accumulation

**Critical issue — Per-block scale accumulation**:

INT8 GEMM uses a single `int32_t acc[mm][nn]` that accumulates across ALL K-tiles, then applies a single global scale at the end. Native-VNNI has **per-block scales** (every 32 K-elements). If we decode to INT8 in LDS, we lose the per-block scale boundary information. We'd need either:

- (a) **Track block boundaries in the compute phase**: Complex K-loop that knows when a block boundary is crossed → eliminates the "format-agnostic compute" advantage
- (b) **Fuse scale into decoded values**: Convert INT8→FP16 in LDS and use FP16 dot products → 2× LDS footprint, kills bandwidth win
- (c) **Accumulate per-block and apply scale between blocks**: Forces K-tile size = quant block size (32), eliminating the KT flexibility that makes V3/V7 efficient

### Option B: Decode-in-Compute (Native quant in LDS → Decode+v_dot4 per block)

```
Global Memory (native quant)  →  Cooperative Load (raw)  →  Quantized data in LDS
                                                                    ↓
                                                             Decode + v_dot4 + scale
                                                             (per-block, in registers)
```

**How it works**: Cooperative loading writes raw quantized payloads + scales into LDS (maintaining the native format). The compute phase reads raw data from LDS, decodes to INT8 in registers, runs `v_dot4`, and applies per-block FP16 scales — all within the K-loop body. This is essentially the GEMV decode pipeline, but reading from LDS instead of global memory.

**Pros**:
- **LDS stores the compact native format**: Q4_0 uses 4.5 bpw in LDS instead of 8 bpw → 1.78× more data fits per LDS buffer → either smaller LDS footprint or larger tiles
- Per-block scales are naturally handled — each decode+dot4 iteration corresponds to exactly one quant block
- Weight data read from LDS into registers is a fraction of INT8 (saving LDS→register bandwidth)
- Format-specific decode overhead is amortized across M rows (same block decoded once, dotted against M activations)

**Cons**:
- Compute phase is format-specific (templated per NativeVNNIFormat) — separate ISA per format
- Decode ALU runs inside the hot inner loop instead of in the load phase
- L1I$ pressure from format-specific compute code (mitigated by template instantiation — one kernel active at a time)

**Critical advantage — Block reuse across M rows**:

This is the key insight that makes GEMM fundamentally different from GEMV for native-VNNI. In GEMV (M=1), each thread decodes a block and dots it against one activation row. In GEMM (M>1), the **same decoded block** (one column of B at one K-block) is dotted against **M_PER_THREAD activation rows** (4 in our design). The decode cost is paid once per block per thread, but produces M_PER_THREAD × 8 useful `v_dot4` operations. For M_PER_THREAD=4:

| | GEMV (M=1) | GEMM (M=4/thread) | Ratio |
|---|---|---|---|
| Decode cost | ~120 ALU | ~120 ALU | 1× |
| v_dot4 ops | 8 | 32 | 4× |
| Decode:Compute | 15:1 (Q4_0) | 3.75:1 | 4× better |

This M-reuse dramatically shifts the decode:compute balance. Formats that were heavily decode-bound at M=1 become much more balanced at M>1.

### Decision: **Option B — Decode-in-Compute**

Option B is the clear winner because:
1. Per-block FP16 scales are preserved naturally (Option A requires awkward workarounds)
2. LDS stores compact data (better effective LDS utilization)
3. M-row reuse amortizes decode overhead by M_PER_THREAD (the whole point of GEMM)
4. Template specialization per format is already the pattern in the GEMV kernel

---

## 3. Kernel Architecture

### 3.1. High-Level Pipeline

The kernel follows the V3 double-buffered pipeline structure, but with the compute phase replaced by a format-specific decode+dot4+scale loop:

```
PRELOAD: Cooperative load of raw quantized B-tile (block 0) + A-tile → LDS buf[0]

MAIN LOOP (tiles 1..T-1):
    ┌───────────────────────────────────────────────────────────────────┐
    │ STEP 1: Issue cooperative loads for tile T+1                      │
    │         → raw quantized B payload + scales → staging registers    │
    │         → INT8 A-tile                      → staging registers    │
    │                                                                   │
    │ STEP 2: Compute tile T from LDS buf[T%2]                         │
    │         for each quant block within the K-tile:                   │
    │           a. Read raw payload + scale from B LDS                  │
    │           b. Decode payload → INT8 packed_groups[8]               │
    │           c. Read A[m_base..m_base+3] from A LDS                  │
    │           d. v_dot4 × M_PER_THREAD × 8 groups                    │
    │           e. Apply FP16 scale to FP32 accumulators                │
    │                                                                   │
    │ __syncthreads()                                                   │
    │                                                                   │
    │ STEP 3: Write staging → LDS buf[(T+1)%2]                         │
    │                                                                   │
    │ __syncthreads()                                                   │
    └───────────────────────────────────────────────────────────────────┘

FINAL TILE: Compute (no more loads)
STORE: FP32 accumulators → global memory
```

### 3.2. Thread Layout and Tile Geometry

```
BLOCK_SIZE = 256 threads
M_PER_THREAD = 4         (each thread handles 4 M-rows, vectorized A load)
THREADS_M = M_TILE / 4
THREADS_N = 256 / THREADS_M
N_PER_THREAD = N_TILE / THREADS_N

Grid: (ceil(N / N_TILE), ceil(M / M_TILE), 1)
```

**Three N_TILE configurations** (inspired by V3/V7 but adapted):

| Variant | N_TILE | M_TILE=128 Layout | Acc/Thread | Target Shapes |
|---------|--------|-------------------|------------|---------------|
| **S64** | 64 | 32×8, N/thr=8 | 32 FP32 | K-heavy (attention, FFN_Down) |
| **S128** | 128 | 32×8, N/thr=16 | 64 FP32 | N-heavy (FFN_Up, FFN_Gate) |
| **S32** | 32 | 32×8, N/thr=4 | 16 FP32 | VGPR-constrained formats (IQ4_NL) |

The S32 variant is new — it halves the accumulator count to free VGPRs for decode-heavy formats.

### 3.3. LDS Layout

Unlike INT8 GEMM where B is stored as decoded INT8 in LDS, native-VNNI GEMM stores **raw quantized data** in LDS:

```
A LDS (same as INT8 GEMM):
  a_lds[2][BK * M_TILE]  — INT32 (4 packed INT8 activations per element)
  Layout: a_lds[buf][(a_kk * M_TILE) + mi]
  BK = number of A k-groups per tile = BLOCKS_PER_TILE * 8
       (each quant block has 32 elements = 8 k-groups of 4)

B LDS (NEW — raw quantized payload + scales):
  b_payload_lds[2][BLOCKS_PER_TILE * N_TILE * MAX_PAYLOAD_BYTES]  — raw bytes
  b_scale_lds[2][BLOCKS_PER_TILE * N_TILE * SCALE_ENTRY_SIZE]    — FP16 scales
  Layout: b_payload_lds[buf][(block_in_tile * N_TILE + ni) * payload_bytes + byte]
          b_scale_lds[buf][(block_in_tile * N_TILE + ni) * scale_entry_bytes + byte]
```

**Key design: K-tile = integer multiple of quant blocks**

Because per-block scales must be applied at block boundaries, the K-tile depth is defined as `BLOCKS_PER_TILE` quant blocks rather than arbitrary k-groups:

| BLOCKS_PER_TILE | K-elements | BK (k-groups) | Equivalent INT8 KT |
|-----------------|------------|---------------|---------------------|
| 1 | 32 | 8 | KT=8 |
| 2 | 64 | 16 | KT=16 |
| 4 | 128 | 32 | KT=32 |

Default: **BLOCKS_PER_TILE = 2** (64 K-elements per tile), matching INT8 GEMM's KT=16 operating point.

### 3.4. LDS Budget Analysis

For a given format with `payload_bytes` and `scale_bytes` per block, with N_TILE and M_TILE:

```
A LDS per buffer = BK × M_TILE × 4 bytes
                 = (BLOCKS_PER_TILE × 8) × M_TILE × 4

B LDS per buffer = BLOCKS_PER_TILE × N_TILE × (payload_bytes + scale_bytes)

Total LDS = 2 × (A_per_buf + B_per_buf)    [double-buffered]
```

| Format | payload_bytes | scale_bytes | N=64, BPT=2, M=128 | N=128, BPT=2, M=128 | N=32, BPT=2, M=128 |
|--------|--------------|-------------|---------------------|----------------------|---------------------|
| Q4_0 | 16 | 2 | 2×(8KB + 2.25KB) = 20.5 KB | 2×(8KB + 4.5KB) = 25 KB | 2×(8KB + 1.125KB) = 18.25 KB |
| Q4_1 | 16 | 4 | 2×(8KB + 2.5KB) = 21 KB | 2×(8KB + 5KB) = 26 KB | — |
| Q5_0 | 20 | 2 | 2×(8KB + 2.75KB) = 21.5 KB | 2×(8KB + 5.5KB) = 27 KB | — |
| Q6_K | 24 | 4 | 2×(8KB + 3.5KB) = 23 KB | 2×(8KB + 7KB) = 30 KB | — |
| Q2_K | 12 | 8* | 2×(8KB + 2.5KB) = 21 KB | 2×(8KB + 5KB) = 26 KB | — |
| IQ4_NL | 16 | 2 | 2×(8KB + 2.25KB) = 20.5 KB | 2×(8KB + 4.5KB) = 25 KB | — |
| IQ3_S | 13 | 2 | 2×(8KB + 1.875KB) = 19.75 KB | — | — |
| IQ2_XXS | 8 | 2 | 2×(8KB + 1.25KB) = 18.5 KB | — | — |
| IQ1_S | 6 | 4 | 2×(8KB + 1.25KB) = 18.5 KB | — | — |

*Q2_K scale_bytes includes 4 bytes dual-scale + 4 bytes embedded min

**Comparison with INT8 GEMM**:
- INT8 V3 (N=64, KT=16): 2×(16×128 + 16×64)×4 = 2×(8192 + 4096)×4 = 24,576 = **24 KB**
- INT8 V7 (N=128, KT=16): 2×(16×128 + 16×128)×4 = 2×(8192 + 8192)×4 = **32 KB**

Native-VNNI uses **less LDS** than INT8 GEMM for the same tile size because raw quantized data is smaller than decoded INT8. For Q2_K at N=64, the B tile is only 2.5 KB per buffer vs INT8's 4 KB — a 37% reduction.

### 3.5. IQ Grid LUT in LDS

IQ grid formats (IQ3_S, IQ2_S, etc.) need lookup tables for decode. In the GEMV kernel, these are accessed from `__constant__` memory (~100 cycle latency per lookup). For GEMM, where the same LUT is accessed repeatedly, preloading into LDS is strongly preferred:

| LUT | Size | Formats |
|-----|------|---------|
| IQ4_NL codebook | 16 B | IQ4_NL, IQ4_XS |
| IQ3 grid | 2 KB | IQ3_S, IQ3_XXS |
| IQ2 grid | 4 KB | IQ2_S, IQ2_XS, IQ2_XXS |
| IQ1 grid | 16 KB | IQ1_S, IQ1_M |

**Strategy**: Preload the required LUT into LDS at kernel start (before the main loop). This LDS is persistent (not double-buffered) and reduces LUT access from ~100 cycles (constant memory) to ~4 cycles (LDS). The IQ4_NL codebook at 16 bytes can alternatively be loaded into 4 scalar registers for zero-latency access.

**LDS budget with LUT** (worst case: IQ1 with 16 KB LUT):
- IQ1_S with N=64, BPT=2, M=128: 18.5 KB (tile buffers) + 16 KB (LUT) = 34.5 KB
- Still within the 64 KB LDS budget, but may limit occupancy. For IQ1, N_TILE=32 may be needed.

---

## 4. VGPR Budget Analysis

This is the critical constraint. The kernel must stay at ≤128 VGPRs for 2 waves/SIMD (the V7 operating point). Exceeding 128 → 1 wave → catastrophic performance loss (V4 lesson).

### 4.1. VGPR Components

| Component | VGPRs (N_TILE=64) | VGPRs (N_TILE=128) | VGPRs (N_TILE=32) | Notes |
|-----------|-------------------|---------------------|-------------------|-------|
| **FP32 Accumulators** | 32 (4×8) | 64 (4×16) | 16 (4×4) | M_PER_THREAD × N_PER_THREAD |
| **A staging** | 8-16 | 8-16 | 8-16 | A_NUM_PASSES × 4 (uint4) |
| **B staging** | 4-8 | 4-8 | 4-8 | B payload staging (format-dependent) |
| **A LDS operands** | 4 | 4 | 4 | a_reg[M_PER_THREAD] per K-step |
| **B decode registers** | 8 | 8 | 8 | packed_groups[8] (decoded INT8) |
| **Decode scratch** | 4-12 | 4-12 | 4-12 | Format-specific decode temporaries |
| **Scale registers** | 2-4 | 2-4 | 2-4 | FP16→FP32 scale conversion |
| **Pointers / indices** | 6-8 | 6-8 | 6-8 | Global/LDS addressing |
| **Loop control** | 2-4 | 2-4 | 2-4 | Tile counter, block counter |
| **TOTAL** | **~70-96** | **~104-132** | **~54-80** | |

### 4.2. Per-Format VGPR Estimates

Using the GEMV ISA analysis as the decode VGPR baseline, and adding GEMM accumulator/staging overhead:

| Format | GEMV VGPRs | Decode Scratch | Est. S64 VGPRs | Est. S128 VGPRs | Est. S32 VGPRs |
|--------|-----------|----------------|----------------|-----------------|----------------|
| Q4_0 | 29 | 4 | **~72** | ~104 | ~56 |
| Q4_1 | 33 | 6 | **~76** | ~108 | ~60 |
| Q5_0 | 35 | 6 | **~78** | ~110 | ~62 |
| Q5_1 | 45 | 8 | **~82** | ~114 | ~66 |
| Q6_K | 45 | 8 | **~82** | ~114 | ~66 |
| Q3_K | 39 | 6 | **~78** | ~110 | ~62 |
| Q2_K | 35 | 6 | **~78** | ~110 | ~62 |
| IQ4_NL | 55 | 12 | **~90** | ~122 | ~74 |
| IQ3_S | 35 | 8 | **~82** | ~114 | ~66 |
| IQ2_S | 38 | 8 | **~84** | ~116 | ~68 |
| IQ1_S | 35 | 8 | **~82** | ~114 | ~66 |
| IQ1_M | 37 | 8 | **~84** | ~116 | ~68 |

**Key observations**:

1. **S64 (N_TILE=64)**: All formats fit within 2-wave occupancy (≤128 VGPRs). Most formats are at 3 waves (≤84 VGPRs). This is the safest option.

2. **S128 (N_TILE=128)**: Simple formats (Q4_0, Q4_1, Q2_K) fit at 2 waves. Complex formats (IQ4_NL at ~122) are at the ragged edge — compiler register allocation may push them over 128 into 1-wave territory. **Risky for IQ formats**.

3. **S32 (N_TILE=32)**: All formats comfortably at 3+ waves. Use as a VGPR-pressure escape valve for IQ4_NL and other decode-heavy formats.

### 4.3. Format-to-Variant Mapping (Recommended)

| Format Group | Primary Variant | Fallback Variant | Rationale |
|-------------|----------------|------------------|-----------|
| Simple (Q4_0, Q4_1, Q5_0, Q5_1) | S64 or S128 | — | Low decode overhead, fits 2 waves |
| K-quant (Q6_K, Q3_K, Q2_K) | S64 | S128 (simple K-quants) | Moderate decode, 2-3 waves at S64 |
| IQ4 (IQ4_NL, IQ4_XS) | S64 | S32 | High LUT overhead, S32 as safety net |
| IQ grid (IQ3_S/XXS, IQ2_S/XS/XXS) | S64 | S32 | Grid LUT decode + LDS pressure |
| IQ1 (IQ1_S, IQ1_M) | S64 | S32 | 16 KB LUT in LDS already constrains tile size |

**Auto-dispatch**: Select S64 as the default. Use S128 for simple formats when N > K (N-heavy shapes). Use S32 only if ISA analysis shows a format exceeds 128 VGPRs at S64.

---

## 5. Accumulator Type: FP32, Not INT32

Unlike INT8 GEMM (which accumulates in INT32 and applies a single global scale post-loop), native-VNNI GEMM must accumulate in **FP32** because per-block FP16 scales vary across K-blocks.

```
INT8 GEMM:    acc_int32 += sdot4(a, b)    ... for all K ...    result = acc_int32 * global_scale
Native-VNNI:  for each block:  acc_fp32 += float(sdot4(a, b)) * block_scale_fp16
```

**Impact on compute inner loop**:

```cpp
// Per-block compute (inside K-block loop):
int32_t block_acc = 0;
#pragma unroll
for (int g = 0; g < 8; ++g)
    block_acc = __builtin_amdgcn_sdot4(a_reg[g][mm], packed_groups[g], block_acc, false);

// Apply per-block scale to FP32 accumulator:
f_acc[mm][nn] += static_cast<float>(block_acc) * block_scale;
```

This adds 1 `v_cvt_f32_i32` + 1 `v_fma_f32` per block per accumulator element. For M_PER_THREAD=4 × N_PER_THREAD=8 = 32 accumulators and BLOCKS_PER_TILE=2:

- **64 extra FMA + 64 CVT = 128 extra instructions per K-tile** (vs zero for INT8 GEMM)
- At 1 cycle each, this is ~128 cycles per K-tile
- Compared to decode ALU (~120-250 cycles for Q4_0/Q6_K) and dot4 compute (256+ cycles for 32 acc × 8 groups × 2 blocks), this is a modest overhead (~15-30%)

**Asymmetric format additional overhead**:

Asymmetric formats (Q4_1, Q5_1, Q2_K) need `sum_a * block_min` correction. This requires computing `sum_a` per block — the sum of activation INT8 values in the 32-element block. Computed once per block (shared across N columns):

```cpp
constexpr int32_t ones = 0x01010101;
int32_t sum_a = 0;
#pragma unroll
for (int g = 0; g < 8; ++g)
    sum_a = __builtin_amdgcn_sdot4(a_reg[g][0], ones, sum_a, false);  // Any M-row works
```

Cost: 8 extra `v_dot4` + 1 FMA per N-column per block. Since `sum_a` is the same for all N-columns of the same block, it's computed once and reused N_PER_THREAD times.

---

## 6. Compute Phase: Block-Oriented Inner Loop

### 6.1. Single-Scale Symmetric (Q4_0, Q5_0, IQ4_NL, IQ3_S, IQ2_XXS, etc.)

This is the simplest and most common pattern:

```cpp
// For each quant block within the K-tile:
for (int blk = 0; blk < BLOCKS_PER_TILE; ++blk)
{
    // (a) Read A from LDS: 8 k-groups × M_PER_THREAD values
    int32_t a_reg[8][M_PER_THREAD];
    #pragma unroll
    for (int g = 0; g < 8; ++g)
        *reinterpret_cast<uint4*>(&a_reg[g][0]) =
            *reinterpret_cast<const uint4*>(&a_lds[buf][(blk * 8 + g) * M_TILE + m_base]);

    // For each N-column this thread owns:
    #pragma unroll
    for (int nn = 0; nn < N_PER_THREAD; ++nn)
    {
        // (b) Read raw payload from B LDS
        const uint8_t* payload = &b_payload_lds[buf][(blk * N_TILE + t_n * N_PER_THREAD + nn)
                                                      * Traits::payload_bytes];
        const uint16_t scale_bits = b_scale_lds[buf][blk * N_TILE + t_n * N_PER_THREAD + nn];

        // (c) Decode payload → INT8 packed_groups[8]
        int32_t packed_groups[8];
        decode_block<FMT>(payload, packed_groups);  // Format-specific decode

        // (d) v_dot4 accumulation
        int32_t block_acc = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                block_acc = __builtin_amdgcn_sdot4(a_reg[g][mm], packed_groups[g], block_acc, false);
        // Wait — this is wrong. We need per-mm accumulation!

        // (d) CORRECT: v_dot4 accumulation (per M-row)
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            int32_t block_acc = 0;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                block_acc = __builtin_amdgcn_sdot4(a_reg[g][mm], packed_groups[g], block_acc, false);

            // (e) Apply per-block FP16 scale
            const float block_d = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            f_acc[mm][nn] += static_cast<float>(block_acc) * block_d;
        }
    }
}
```

**Instruction count per tile (S64, M_PER_THREAD=4, N_PER_THREAD=8, BLOCKS_PER_TILE=2)**:

| Operation | Count per tile | Notes |
|-----------|---------------|-------|
| A LDS reads | 2 × 8 × 1 = 16 uint4 | 8 k-groups × 2 blocks, vectorized |
| B payload LDS reads | 2 × 8 × ~4 = ~64 | Format-dependent, ~16B each |
| B scale LDS reads | 2 × 8 = 16 | FP16 scale per block per col |
| Decode ALU | 2 × 8 × ~120 = ~1,920 | Format-dependent (Q4_0 estimate) |
| v_dot4 | 2 × 8 × 4 × 8 = 512 | 8 groups × 4 M-rows × 8 N-cols × 2 blocks |
| CVT + FMA (scale) | 2 × 8 × 4 = 64 | Per block × N-cols × M-rows |

**Problem**: The decode is per-N-column, repeated N_PER_THREAD=8 times. But for a given K-block index, all N-columns have **different payloads** (different weight values). So decode cannot be shared across N. However, decode can be shared across M-rows (same weights, different activations).

**Optimization**: Restructure the loop to decode once per (block, N-col), then dot against all M-rows:

```cpp
for (int blk = 0; blk < BLOCKS_PER_TILE; ++blk)
{
    // Load ALL A vectors for this block (8 k-groups × M_PER_THREAD)
    int32_t a_reg[8][M_PER_THREAD];
    load_a_tile(blk, a_reg);

    for (int nn = 0; nn < N_PER_THREAD; ++nn)
    {
        // Decode once for this (block, N-col)
        int32_t packed_groups[8];
        decode_block<FMT>(payload_ptr(blk, nn), packed_groups);

        float scale = read_scale(blk, nn);

        // Dot against all M-rows
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            int32_t block_acc = 0;
            for (int g = 0; g < 8; ++g)
                block_acc = __builtin_amdgcn_sdot4(a_reg[g][mm], packed_groups[g], block_acc, false);
            f_acc[mm][nn] += float(block_acc) * scale;
        }
    }
}
```

**Revised instruction count** (decode NOT repeated per M-row):

| Operation | Count per tile | Cycles (est.) |
|-----------|---------------|---------------|
| Decode ALU | 2 × 8 × ~120 = ~1,920 | ~1,920 |
| v_dot4 | 2 × 8 × 4 × 8 = 512 | ~512 |
| Scale CVT+FMA | 2 × 8 × 4 = 64 | ~64 |
| LDS reads (A) | 16 uint4 | ~32 |
| LDS reads (B payload) | 16 = 2×8 | ~32 |
| **Total** | | **~2,560** |

Compared to INT8 GEMM V3 (KT=16, N=64, M=128):
- v_dot4: 16 × 8 × 4 = 512 (same!)
- No decode overhead → ~64 cycles for LDS reads + ~512 cycles for dot4 = **~576 cycles**

**Decode overhead ratio**: ~1,920 / 512 = **3.75:1** for Q4_0. Much better than GEMV's 15:1 (thanks to M-reuse), but still significant. The kernel will be compute-bound rather than memory-bound for most formats during prefill.

### 6.2. Dual-Scale (Q6_K, Q3_K, IQ2_S, IQ2_XS)

Dual-scale formats split each 32-element block into two 16-element halves with different scales:

```cpp
// Elements 0-15 → acc_lo with scale_lo
// Elements 16-31 → acc_hi with scale_hi
int32_t acc_lo = 0, acc_hi = 0;
for (int g = 0; g < 4; ++g)
    acc_lo = __builtin_amdgcn_sdot4(a_reg[g][mm], packed_groups[g], acc_lo, false);
for (int g = 4; g < 8; ++g)
    acc_hi = __builtin_amdgcn_sdot4(a_reg[g][mm], packed_groups[g], acc_hi, false);

f_acc[mm][nn] += float(acc_lo) * scale_lo + float(acc_hi) * scale_hi;
```

Cost: 2 extra CVT + 2 FMA (vs 1 CVT + 1 FMA for single-scale). Minimal overhead.

### 6.3. Asymmetric (Q4_1, Q5_1, Q2_K)

Asymmetric adds `sum_a * min` correction:

```cpp
// Compute sum_a once per block (shared across all N-columns)
int32_t sum_a = 0;
constexpr int32_t ones = 0x01010101;
for (int g = 0; g < 8; ++g)
    sum_a = __builtin_amdgcn_sdot4(a_reg[g][mm], ones, sum_a, false);

// For each N-column:
f_acc[mm][nn] += float(block_acc) * scale + float(sum_a) * min_val;
```

Cost: 8 extra `v_dot4` per block per M-row for `sum_a` computation + 1 FMA for min correction per (block, N-col, M-row).

**Optimization**: `sum_a` is the same for all N-columns at the same M-row and block. Compute it once and reuse across N_PER_THREAD columns. Restructure the loop order:

```cpp
for (int blk = 0; blk < BLOCKS_PER_TILE; ++blk)
{
    load_a_tile(blk, a_reg);

    // Precompute sum_a for each M-row (shared across all N-cols)
    int32_t sum_a[M_PER_THREAD];
    if constexpr (Traits::is_asymmetric) {
        constexpr int32_t ones = 0x01010101;
        for (int mm = 0; mm < M_PER_THREAD; ++mm) {
            sum_a[mm] = 0;
            for (int g = 0; g < 8; ++g)
                sum_a[mm] = __builtin_amdgcn_sdot4(a_reg[g][mm], ones, sum_a[mm], false);
        }
    }

    for (int nn = 0; nn < N_PER_THREAD; ++nn)
    {
        decode_block<FMT>(payload_ptr(blk, nn), packed_groups);
        float scale = read_scale(blk, nn);
        float min_val = read_min(blk, nn);

        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            int32_t block_acc = 0;
            for (int g = 0; g < 8; ++g)
                block_acc = sdot4(a_reg[g][mm], packed_groups[g], block_acc);
            f_acc[mm][nn] += float(block_acc) * scale;
            if constexpr (Traits::is_asymmetric)
                f_acc[mm][nn] += float(sum_a[mm]) * min_val;
        }
    }
}
```

Cost of `sum_a`: 8 × M_PER_THREAD = 32 extra `v_dot4` per block. Since BLOCKS_PER_TILE=2 and these are per-tile, that's 64 extra `v_dot4` per tile — a ~12% overhead on the dot4 count.

---

## 7. Cooperative B-Tile Loading

This is the most format-dependent part. Unlike INT8 GEMM where B is stored as uniform `int32_t` elements, native-VNNI B tiles have variable payload sizes.

### 7.1. Loading Strategy

Each thread cooperatively loads a portion of the B tile's raw payload and scales into LDS. The challenge is that payload sizes vary by format (6-24 bytes per block per N-column).

**Approach: Byte-granularity cooperative load**

```
Total B bytes per buffer = BLOCKS_PER_TILE × N_TILE × (payload_bytes + scale_bytes)
Total loads needed       = ceil(total_bytes / (256 threads × load_width))
```

For Q4_0 (payload=16, scale=2, total=18 bytes/block) with N_TILE=64, BPT=2:
- Total B bytes = 2 × 64 × 18 = 2,304 bytes
- With 256 threads doing 16-byte loads: ceil(2304/4096) = 1 pass
- With 4-byte loads: ceil(2304/1024) = 3 passes

**Preferred: Vectorized uint4 loads (16 bytes) where possible, scalar for tail**

The host-side packing can ensure payload+scale data is 16-byte aligned per (block, N) pair by padding. This allows `global_load_dwordx4` for the majority of the B-tile load.

### 7.2. B Payload LDS Layout

For coalesced global reads, the payload must be interleaved by N (already done by `packNativeVNNI`):

```
Global layout: [block_idx × N × payload_bytes]
  For block b, column n: offset = (b * N + n) * payload_bytes

LDS layout: [block_in_tile × N_TILE × payload_bytes]
  For block b_local, column n_local: b_payload_lds[buf][(b_local * N_TILE + n_local) * payload_bytes]
```

Cooperative load maps each thread to a (byte_offset) in the tile, loads from global, writes to LDS.

### 7.3. Scale Loading

Scales are loaded separately from payload (they're in separate arrays on the GPU):

```
d_native_vnni_scales[block_idx * N + n]  — FP16 scale
d_native_vnni_mins[block_idx * N + n]    — FP16 min (for asymmetric/dual-scale)
```

Each thread loads 1-2 scale values (FP16 = 2 bytes each). With N_TILE=64 and BPT=2: 128 scale entries × 2 bytes = 256 bytes. Trivially loaded by 256 threads in one pass (1 byte each) or 64 threads (4-byte load each).

---

## 8. Template Structure

### 8.1. Kernel Function Signature

```cpp
template <int N_TILE, int BLOCKS_PER_TILE, int M_TILE, NativeVNNIFormat FMT>
__global__ __launch_bounds__(256, 2)
void native_vnni_gemm_kernel(
    const int8_t*   __restrict__ d_A_int8,            // [M × K] INT8 activations
    const uint8_t*  __restrict__ d_B_payload,          // [blocks_per_row × N × payload_bytes]
    const uint16_t* __restrict__ d_B_scales,           // [blocks_per_row × N] FP16 scales
    const uint16_t* __restrict__ d_B_mins,             // [blocks_per_row × N] FP16 mins (or nullptr)
    float*          __restrict__ d_C_fp32,             // [M × N] FP32 output
    int M, int N, int K,
    int blocks_per_row);                               // K / 32
```

### 8.2. Format Traits (Reuse from GEMV)

```cpp
// Already defined in ROCmGemvKernel_native_VNNI.hip — extract to shared header
template <NativeVNNIFormat FMT>
struct NVNNITraits {
    static constexpr int payload_bytes;
    static constexpr int block_size = 32;
    static constexpr bool is_asymmetric;
    static constexpr bool is_dual_scale;
    static constexpr bool is_dual_scale_asym;
    static constexpr bool is_iq_grid;
    static constexpr bool is_iq1_grid;
    static constexpr bool has_embedded_scales;
    static constexpr int embedded_scale_offset;
    static constexpr int embedded_min_offset;
    // ... etc
};
```

### 8.3. Decode Functions (Reuse from GEMV)

The decode logic from the GEMV kernel's `iq_decode_accumulate_block` and inline decode paths can be extracted into standalone `__device__ __forceinline__` functions:

```cpp
template <NativeVNNIFormat FMT>
__device__ __forceinline__ void decode_native_vnni_block(
    const uint8_t* __restrict__ payload,
    int32_t packed_groups[8],
    const uint32_t* lds_grid32 = nullptr,    // For IQ3 formats
    const uint64_t* lds_grid64 = nullptr);   // For IQ2/IQ1 formats
```

### 8.4. Template Instantiation

Each `(N_TILE, BLOCKS_PER_TILE, M_TILE, FMT)` combination is a separate kernel instantiation. To limit combinatorial explosion:

**Phase 1 (launch)**: Instantiate only the most common configurations:
- `N_TILE ∈ {32, 64}` × `BPT=2` × `M_TILE ∈ {32, 128}` × `FMT ∈ {all 18 formats}`
- Total: 2 × 1 × 2 × 18 = **72 kernel instantiations**

**Phase 2 (tuned)**: Add `N_TILE=128` for select formats that can handle the VGPR pressure:
- `N_TILE=128` × `BPT=2` × `M_TILE ∈ {32, 128}` × `FMT ∈ {Q4_0, Q4_1, Q2_K, Q3_K}`
- Additional: 1 × 1 × 2 × 4 = **8 more instantiations**

---

## 9. Dispatch Logic

### 9.1. When to Use Native-VNNI GEMM vs INT8 GEMM

The dispatch decision depends on whether native-VNNI GEMM is faster than INT8 GEMM for a given shape and format. The key trade-off:

| Factor | Native-VNNI GEMM | INT8 GEMM |
|--------|-------------------|-----------|
| HBM reads | BPW/8 × weights | 1.0× weights (8 bpw) |
| Decode ALU | Per-block | Zero |
| Accumulator | FP32 | INT32 |
| Scale application | Per-block in inner loop | Single global scale post-loop |
| Accuracy | Lossless per-block FP16 | Truncated to single scale pair |

**Expected crossover**: For M=1, native-VNNI is always better (bandwidth-bound, decode hidden by memory latency). As M increases, the kernel becomes compute-bound, and the decode overhead matters more. At very large M (>128+), INT8 GEMM may win because its compute phase is pure `v_dot4` with no decode overhead.

**Proposed heuristic**:

```
if (M == 1):
    → Native-VNNI GEMV (existing kernel)
elif (M <= NATIVE_GEMM_M_THRESHOLD):
    → Native-VNNI GEMM (this new kernel)
else:
    → INT8 GEMM V3/V7 (existing kernel)
```

Where `NATIVE_GEMM_M_THRESHOLD` is format-dependent and determined empirically:
- Simple formats (Q4_0): threshold ~128-256 (decode is cheap)
- Complex formats (IQ4_NL): threshold ~32-64 (decode is expensive)
- Very low BPW (Q2_K, IQ2): threshold ~64-128 (bandwidth savings are large)

**Override**: `LLAMINAR_ROCM_NATIVE_VNNI_GEMM=1` to force native-VNNI GEMM for all M>1.

### 9.2. N_TILE / M_TILE Selection

```
For a given (M, N, K, FMT):
    if format_has_high_vgpr_pressure(FMT):
        N_TILE = 32
    elif (N > K):
        N_TILE = 64 (or 128 for simple formats)
    else:
        N_TILE = 64

    if (M <= 32):
        M_TILE = 32
    elif (M <= 64):
        M_TILE = 64
    else:
        M_TILE = 128
```

---

## 10. Performance Projections

### 10.1. Roofline Analysis

For M>1 prefill, the kernel is typically **compute-bound** (unlike GEMV which is memory-bound). The relevant compute throughput is:

- **gfx906 v_dot4 peak**: 60 CUs × 4 SIMDs × 64 lanes × 1 dot4/cycle × 1.725 GHz = **26.6 Tdot4/s**
- Each `v_dot4` = 4 MACs → **106.4 TMAC/s** INT8 peak

For a GEMM of shape [M×N] = A[M×K] × B[K×N]:
- Total MACs = M × N × K
- Theoretical time = M × N × K / peak_throughput

**Q4_0 overhead model**:
- Decode ALU: ~120 cycles per block per N-column = (K/32) × N × 120 cycles
- v_dot4: (K/32) × 8 groups × M × N = K × M × N / 4 cycles
- Scale: (K/32) × M × N × 2 cycles (CVT + FMA)
- Effective throughput = v_dot4 cycles / (v_dot4 + decode + scale) = **~21% at M=4, ~51% at M=16, ~72% at M=64**

This means for M=64:
- If INT8 GEMM achieves 70% of peak → ~74.5 TMAC/s effective
- Native-VNNI Q4_0 achieves 72% of that due to decode → ~53.6 TMAC/s
- But reads **1.78× less HBM data** → still ~53.6 TMAC/s (compute-bound, not BW-limited)

For large M where INT8 GEMM is also compute-bound, INT8 GEMM will be faster in raw throughput. But native-VNNI has 2 advantages:
1. **Less VRAM**: Weights stay at native BPW, no INT8 conversion copy needed
2. **Better accuracy**: Per-block FP16 scales preserved

### 10.2. Expected Speedup vs INT8 GEMM

| Format | BPW | Bandwidth Ratio | Est. Speedup (M=8-32) | Est. Speedup (M=128+) |
|--------|-----|-----------------|-----------------------|------------------------|
| Q4_0 | 4.5 | 1.78× | 1.0-1.3× | 0.7-0.9× |
| Q4_1 | 5.0 | 1.60× | 1.0-1.2× | 0.7-0.9× |
| Q3_K | 3.4 | 2.35× | 1.2-1.6× | 0.7-0.9× |
| Q2_K | 2.6 | 3.08× | 1.5-2.2× | 0.8-1.0× |
| IQ4_NL | 4.5 | 1.78× | 0.8-1.1× | 0.5-0.7× |
| IQ2_XXS | 2.1 | 3.81× | 1.5-2.5× | 0.9-1.1× |
| IQ1_S | 1.6 | 5.00× | 1.5-2.5× | 0.9-1.2× |

**Key insight**: Native-VNNI GEMM is most beneficial for:
1. **Low BPW formats** (Q2_K, IQ2, IQ1) where bandwidth savings dominate
2. **Medium M** (8-64) where the kernel is in the bandwidth/compute transition zone
3. **Accuracy-critical use cases** where per-block FP16 scales matter

For high BPW formats (Q5_1, Q6_K) at large M, INT8 GEMM may be faster — the dispatch heuristic should select INT8 GEMM in these cases.

---

## 11. Implementation Plan

### Phase 1: Core Infrastructure + Q4_0 (week 1-2)

**Goal**: Working native-VNNI GEMM for the simplest format, proving the architecture.

1. **Extract shared code from GEMV** into a common header:
   - `NativeVNNIFormat` enum → `NativeVNNICommon.h`
   - `NVNNITraits<FMT>` → `NativeVNNICommon.h`
   - Decode functions → `NativeVNNIDecode.hip` (device-only header)

2. **Implement `native_vnni_gemm_kernel` for Q4_0**:
   - S64 variant (N_TILE=64, BPT=2, M_TILE=32/128)
   - LDS double-buffered pipeline
   - Decode-in-compute with per-block FP16 scale
   - FP32 output (no INT32→FP32 conversion needed)

3. **Cooperative B-tile load**:
   - Payload + scale loading into LDS
   - Handle alignment/padding

4. **Dispatch integration**:
   - Add to `tryPrefillNativeGemm()` path in `ROCmQuantisedGemmKernel.cpp`
   - M-threshold gating (initially M ≤ 64)

5. **Tests**:
   - Unit: `Test__NativeVNNI_GEMM_Q4_0` — random weights, verify vs FP32 reference
   - Integration: Compare against INT8 GEMM output for small M shapes
   - ISA analysis: Verify VGPR count < 128, measure occupancy

### Phase 2: Simple Formats (Q4_1, Q5_0, Q5_1) (week 2-3)

Extend the template to symmetric+asymmetric simple formats. Add `sum_a` computation for asymmetric formats.

### Phase 3: K-quant Formats (Q6_K, Q3_K, Q2_K) (week 3-4)

Add dual-scale support (Q6_K, Q3_K) and dual-scale-asymmetric (Q2_K). These formats have the largest bandwidth savings potential.

### Phase 4: IQ Formats (IQ4_NL, IQ3_S, IQ2_S, etc.) (week 4-5)

Add LDS grid LUT preloading. Implement IQ decode paths. May need S32 variant for VGPR-heavy formats.

### Phase 5: IQ1 Formats + S128 Variant (week 5-6)

IQ1_S/IQ1_M with 16 KB LUT. Add S128 variant for simple formats with N-heavy shapes.

### Phase 6: Tuning + Dispatch Optimization (week 6-7)

Per-format ISA analysis. Determine optimal M-threshold for native vs INT8 dispatch. Benchmark against INT8 GEMM across all shapes and formats.

---

## 12. File Structure

| File | Purpose |
|------|---------|
| `src/v2/kernels/rocm/NativeVNNICommon.h` | Shared: `NativeVNNIFormat` enum, `NVNNITraits`, constants |
| `src/v2/kernels/rocm/NativeVNNIDecode.hip` | Shared: `decode_native_vnni_block<FMT>()` device functions |
| `src/v2/kernels/rocm/gemm/ROCmQuantisedGemmKernel_native_VNNI.hip` | **NEW**: Native-VNNI GEMM kernels |
| `src/v2/kernels/rocm/ROCmGemvKernel_native_VNNI.hip` | Existing GEMV — refactored to use shared headers |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp` | Dispatch: add `tryPrefillNativeGemm()` path |
| `tests/v2/unit/kernels/rocm/Test__NativeVNNI_GEMM.cpp` | Unit tests: per-format GEMM accuracy |
| `tests/v2/integration/kernels/rocm/Test__NativeVNNI_GEMM_Parity.cpp` | Parity vs INT8 GEMM + FP32 reference |

---

## 13. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| VGPR overflow (>128) for IQ formats at S64 | High | S32 fallback variant; ISA analysis before committing to a tile size |
| Decode ALU dominates at large M (native GEMM slower than INT8) | Medium | M-threshold dispatch; INT8 GEMM remains the fallback for large M |
| L1I$ pressure from large per-format kernel ISA | Medium | Keep compute loop bodies small; avoid aggressive unrolling; `#pragma clang loop unroll(disable)` on N-loop |
| Compiler inserts eager `s_waitcnt` breaking pipeline overlap | Medium | ISA analysis post-compilation; use `__builtin_amdgcn_s_waitcnt` hints if needed |
| LDS bank conflicts from non-power-of-2 payload sizes | Medium | Pad payload to power-of-2 boundaries in host packing; use `__align__` on LDS arrays |
| Template instantiation explosion (72+ kernels) | Low | Compile-time gating via `#if`; only instantiate formats in use |
| LDS budget exceeded for IQ1 + large N_TILE | Low | Cap N_TILE to 32 for IQ1 formats; LDS budget analysis per-format |

---

## 14. Success Criteria

1. **Accuracy**: Every native-VNNI GEMM kernel produces output with cosine similarity ≥ 0.9999 vs FP32 dequantized reference across all tested shapes (M=2..128, N=896..18944, K=896..18944)

2. **Performance**: For M ≤ 64:
   - Q4_0/Q4_1: ≥ 0.9× INT8 GEMM throughput (bandwidth savings offset decode overhead)
   - Q2_K/Q3_K: ≥ 1.3× INT8 GEMM throughput (3× bandwidth savings dominate)
   - IQ formats: ≥ 0.8× INT8 GEMM throughput

3. **Occupancy**: All kernels compile to ≤ 128 VGPRs (2 waves/SIMD) with 0 register spills

4. **No regressions**: Existing INT8 GEMM and native-VNNI GEMV paths remain unchanged

5. **Dispatch correctness**: Auto-dispatch selects native-VNNI GEMM or INT8 GEMM based on empirically-validated M-thresholds, never producing worse results than the pure INT8 path

---

## 15. Open Questions

1. **Should the first implementation target S64 only, or S64 + S32?** S32 is a safe bet for all formats but wastes N-dimension parallelism. S64 gives better tile reuse but may not fit all IQ formats.

   → **Recommendation**: Start with S64 only. Add S32 after ISA analysis reveals any format exceeding 128 VGPRs.

2. **Should B payload in LDS be padded to power-of-2 bytes per entry?** Padding wastes LDS space but avoids bank conflicts. Without profiling, it's unclear if bank conflicts matter.

   → **Recommendation**: No padding initially. Profile with `rocprof --hsa-trace` and add padding if LDS bank conflict stalls are measured.

3. **Should the native-VNNI GEMM path be enabled by default or opt-in?** The INT8 GEMM path is proven and production-quality. Switching to native GEMM for prefill is a significant change.

   → **Recommendation**: Opt-in via `LLAMINAR_ROCM_NATIVE_VNNI_GEMM=1` in Phase 1-3. Enable by default in Phase 6 after comprehensive benchmarking.

4. **Should `NVNNITraits` and decode functions live in a shared `.hip` header or be duplicated?** Shared headers reduce duplication but complicate the build (HIP device code in headers).

   → **Recommendation**: Shared `NativeVNNICommon.h` for host-visible traits, shared `NativeVNNIDecode.hip` for `__device__` decode functions (included by both GEMV and GEMM `.hip` files).

5. **What is the right `BLOCKS_PER_TILE` (BPT)?** BPT=1 minimizes LDS but gives less K-tile depth for pipeline overlap. BPT=2 matches INT8 GEMM's KT=16 operating point. BPT=4 doubles LDS but provides more overlap.

   → **Recommendation**: BPT=2 as default. Add BPT=1 as a fallback for LDS-constrained configurations (IQ1 with large LUT).

---

## 16. Profiling Analysis and Tuning Roadmap

### 16.1 Methodology

Counter data collected with `rocprofv3` across 10 passes (5 focused + 5 full-test) on MI50 (gfx906, 60 CUs). All kernels profiled at M=32, M=128, test shape N=4864, K=896 (Qwen2.5 0.5B FFN). INT8 V7 GEMM counters extracted as baseline.

### 16.2 Key Findings

**Finding 1: All NVNNI variants are compute-bound, not memory-bound.**

| Metric             | NVNNI N128/M32 | INT8 N128/M32 |
|--------------------|-----------------|---------------|
| VALUBusy           | 60%             | 39%           |
| MemUnitBusy        | 10%             | 18%           |
| LDSBankConflict    | 1.2%            | 1.4%          |

The decode-in-load design successfully eliminated memory bottlenecks. NVNNI loads ~50% less HBM data than INT8 (4-bit vs 8-bit weights), but spends the savings on VALU decode work.

**Finding 2: Occupancy gap is the #1 performance bottleneck.**

| Kernel              | VGPRs | Waves/SIMD | Occupancy |
|---------------------|-------|------------|-----------|
| INT8 N128/M32       | 72    | 3          | 75%       |
| **NVNNI N128/M32 Q4_0**  | **128** | **2** | **50%** |
| **NVNNI N128/M32 IQ4_NL**| **116** | **2** | **50%** |
| INT8 N64/M32        | 176   | 1          | 25%       |
| NVNNI N64/M32 Q4_0  | 84    | 3          | 75%       |
| NVNNI N64/M32 IQ4_NL| 88    | 2          | 50%       |

INT8's N128/M32 achieves 3-wave occupancy (72 VGPRs) vs NVNNI's 2-wave (128 VGPRs). This 50% occupancy gap directly explains NVNNI's 0.84× performance at M=32 — the GPU cannot hide latency with only 2 waves per SIMD.

**Finding 3: NVNNI has 34% more VALU instructions per wave than INT8.**

Decode overhead (shift, mask, subtract-8 for Q4_0; v_perm LUT for IQ4_NL) adds ~34% more VALU instructions compared to INT8's straight loads. IQ4_NL costs ~8% more VALU than Q4_0. At 3-wave occupancy, this decode overhead can be hidden behind memory latency; at 2-wave, it cannot.

**Finding 4: LDS stalls are negligible for NVNNI.**

NVNNI's A-matrix LDS loads generate only 1.2-1.4% bank conflict stalls. This is excellent — the INT8 N64 path suffers 21-25% LDS stalls by comparison. NVNNI's larger N_TILE naturally spreads LDS accesses across banks.

### 16.3 VGPR Breakdown: Why N128/M32 Hits 128

Per-thread accumulator count = `M_TILE × N_TILE / BLOCK_SIZE`:

| Geometry    | Accumulators/Thread | FP32 acc | INT32 block_acc | Total VGPR Pressure |
|-------------|---------------------|----------|-----------------|---------------------|
| N128/M32    | 32×128/256 = 16     | 16       | 16              | 32 regs → 128 VGPRs |
| N128/M16    | 16×128/256 = 8      | 8        | 8               | 16 regs → 84 VGPRs  |
| N64/M32     | 32×64/256 = 8       | 8        | 8               | 16 regs → 84 VGPRs  |
| N64/M64     | 64×64/256 = 16      | 16       | 16              | 32 regs → 116 VGPRs |

The accumulator registers (FP32 partial sums + INT32 block-level accumulators) dominate VGPR usage. Halving the accumulator count from M32→M16 drops from 128→84 VGPRs — exactly at the 3-wave boundary.

### 16.4 Verified VGPR Counts (Hybrid Dispatch Build)

Extracted via `llvm-objdump --disassemble-all` on the compiled GPU code object.
The hybrid dispatch instantiates 12 kernel variants: N128×{M16,M32}×{Q4_0,IQ4_NL} + N64×{M16,M32,M64}×{Q4_0,IQ4_NL}. M16 variants use `MIN_BLOCKS=3` (`__launch_bounds__(256, 3)`), all others use `MIN_BLOCKS=2`.

| Kernel                  | VGPRs | Scratch | Occupancy  | MIN_BLOCKS |
|-------------------------|-------|---------|------------|------------|
| N128/M16/Q4_0/MB3       | **84**| 0       | **3-wave** | 3          |
| N128/M16/IQ4_NL/MB3     | **84**| **8B**  | **3-wave** | 3          |
| N128/M32/Q4_0/MB2       | 128   | 0       | 2-wave     | 2          |
| N128/M32/IQ4_NL/MB2     | 116   | 0       | 2-wave     | 2          |
| N64/M16/Q4_0/MB2        | 80    | 0       | 3-wave     | 2          |
| N64/M16/IQ4_NL/MB2      | 80    | 0       | 3-wave     | 2          |
| N64/M32/Q4_0/MB2        | 84    | 0       | 3-wave     | 2          |
| N64/M32/IQ4_NL/MB2      | 88    | 0       | 2-wave     | 2          |
| N64/M64/Q4_0/MB2        | 116   | 0       | 2-wave     | 2          |
| N64/M64/IQ4_NL/MB2      | 116   | 0       | 2-wave     | 2          |

**IQ4_NL N128/M16 spill**: `__launch_bounds__(256, 3)` forces IQ4_NL from 88→84 VGPRs, causing 8 bytes (2 registers) of scratch spill. This is a net positive: 3-wave occupancy (50% more waves) far outweighs the cost of 2 register spills to scratch memory.

### 16.5 Implemented: Hybrid N128 Dispatch (M16/3-wave + M32/2-wave)

#### Design Evolution

Three dispatch strategies were tested:

1. **Always M32/2-wave** (original baseline): simple, consistent, but bottlenecked by 2-wave occupancy for small grids.
2. **Always M16/3-wave**: regressed badly for large N shapes (LM_Head 0.68-0.74× vs baseline 0.85×) because M32/2-wave has 2× better A-tile reuse per WG and wins when the GPU is already saturated.
3. **Hybrid dispatch** (implemented): M16/3-wave when GPU is undersaturated, M32/2-wave when saturated. Zero regression, targeted wins.

#### Hybrid Dispatch Algorithm

```cpp
if (use_n128) {
    const int n128_blocks = (N + 127) / 128;
    const int m16_blocks  = (M + 15) / 16;
    const int total_wgs_m16 = n128_blocks * m16_blocks;

    if (total_wgs_m16 <= 256) {
        return launch3(N128{}, M16{});   // 3-wave, undersaturated GPU
    } else {
        // GPU saturated: M32/2-wave → better tile efficiency
        if (m_tile <= 16)
            return launch3(N128{}, M16{});
        else
            return launch(N128{}, M32{});
    }
}
```

**Threshold rationale**: 256 WGs ≈ 1 WG per SIMD on MI50 (60 CUs × 4 SIMDs = 240 SIMDs). Below this, the GPU cannot fill all SIMDs even at 3-wave occupancy, so extra occupancy directly improves throughput. Above this, all SIMDs are occupied and M32's 2× better A-tile reuse per WG dominates.

#### Why Always-M16 Regresses for Large N

When the GPU is already saturated with workgroups:
- M16 tile: ceil(M/16) × ceil(N/128) WGs, each processing 16×128 output tiles
- M32 tile: ceil(M/32) × ceil(N/128) WGs, each processing 32×128 output tiles

M16 launches 2× more WGs doing half the work each. The 50% occupancy increase (3-wave vs 2-wave = 720 vs 480 concurrent WGs) cannot compensate for:
1. **Halved A-tile reuse**: M16 loads 16 A-rows from LDS vs M32's 32 — each A-row load amortizes over half as many output elements.
2. **Diminishing occupancy returns**: When total WGs >> SIMD slots, more concurrent slots just mean more scheduling overhead without throughput gain.

Effective throughput ratio at saturation: `(720/480) × (WG_work_M16/WG_work_M32) = 1.5 × 0.5 = 0.75×`. Confirmed empirically — LM_Head (N=151936) at M32 showed 0.68-0.74× with always-M16 vs 0.85× baseline.

#### Benchmark Results (MI50, gfx906)

**Aggregate averages (vs INT8 VNNI GEMM baseline):**

| Format | M=32 | M=64 | M=128 | M=256 | Δ vs baseline |
|--------|------|------|-------|-------|---------------|
| Q4_0   | 0.84×| 0.94×| 1.09× | 1.11× | ═ / ═ / ═ / ▲+0.02 |
| IQ4_NL | 0.80×| 0.89×| 1.03× | 1.05× | ═ / ═ / ═ / ═ |

**Zero regression on aggregate** — the hybrid dispatch exactly matches baseline averages since only small shapes trigger M16/3-wave.

**M16/3-wave wins (GPU undersaturated, WGs ≤ 256):**

| Shape | Format | M | WGs | Hybrid | Baseline* | Improvement |
|-------|--------|---|-----|--------|-----------|-------------|
| 0.5B_FFN_Up (N=4864, K=896) | Q4_0 | 32 | 76 | **1.06×** | ~0.85× | +25% |
| 0.5B_FFN_Up (N=4864, K=896) | Q4_0 | 64 | 152 | **1.25×** | ~0.95× | +32% |
| 3B_FFN_Up (N=11008, K=2048) | Q4_0 | 32 | 172 | **1.02×** | ~0.90× | +13% |
| 0.5B_FFN_Up (N=4864, K=896) | IQ4_NL | 64 | 152 | **1.17×** | ~0.85× | +38% |
| 0.5B_FFN_Up (N=4864, K=896) | IQ4_NL | 32 | 76 | **1.01×** | ~0.80× | +26% |

*Baseline estimates for per-shape numbers (only aggregate averages were recorded pre-hybrid).

**Threshold validation against real workloads:**

| Shape | M | M16 WGs | Dispatch | Speedup | Correct? |
|-------|---|---------|----------|---------|----------|
| 0.5B_FFN_Up | 32 | 76 | M16/3w | 1.06× | ✓ undersaturated, 3-wave helps |
| 3B_FFN_Up | 32 | 172 | M16/3w | 1.02× | ✓ undersaturated, marginal win |
| 7B_FFN_Up | 32 | 296 | M32/2w | 0.82× | ✓ near-saturated, M32 better |
| LM_Head | 32 | 2374 | M32/2w | 0.84-0.93× | ✓ heavily saturated, M32 correct |
| 0.5B_FFN_Up | 64 | 152 | M16/3w | 1.25× | ✓ big win at undersaturation |
| 3B_FFN_Up | 64 | 344 | M32/2w | 1.03× | ✓ just above threshold, M32 safe |

### 16.6 Further Tuning Directions

1. **IQ4_NL spill elimination**: The 8B scratch spill in N128/M16/IQ4_NL/MB3 comes from forcing 88→84 VGPRs. Options:
   - `#pragma clang loop unroll(disable)` on the `v_perm` LUT decode loop
   - Restructure `decode_iq4_nl_block()` to reduce live register overlap
   - Accept the 8B spill (3-wave >> 2-register spill cost)

2. **VALU instruction reduction**: NVNNI has 34% more VALU instructions than INT8 (decode overhead). Strategies:
   - Fuse shift+mask+subtract operations where possible
   - Exploit `v_perm_b32` more aggressively for multi-element decode
   - Move scale multiplication outside the K-loop (accumulate INT32, scale once)

3. **Threshold tuning**: The 256-WG threshold is tuned for MI50 (240 SIMDs). For MI100 (120 CUs = 480 SIMDs), the threshold should scale to ~480.

4. **N64 path improvements**: N64/M32/IQ4_NL sits at 88 VGPRs (2-wave). Forcing 3-wave via `MIN_BLOCKS=3` may help small shapes similarly.

5. **LM_Head-specific optimization**: LM_Head shapes (N=151936) show 0.73-0.93× across all M values — the largest gap. Since N>>K, these shapes have very high N-block counts and M32/2-wave is correct, but per-block scale accumulation overhead may be disproportionate. Investigating scale pre-reduction or wider K-blocks could help.

### 16.7 ISA Audit (Release Build Disassembly)

Full ISA audit of all 10 kernel variants extracted from the release binary via `llvm-objdump`.
Disassembly: 14,024 lines from GPU code object #5 (100 symbols, 12 kernel variants, 10 unique).

#### 16.7.1 Instruction Count Summary

| Variant | Total | Compute | Decode | V_MOV | Addr Calc | Control | Scalar ALU |
|---------|-------|---------|--------|-------|-----------|---------|------------|
| N128/M16/IQ4_NL/MB3 | 1354 | 268 (19.8%) | 255 (18.8%) | 130 | 153 (11.3%) | 116 (8.6%) | 146 (10.8%) |
| N128/M16/Q4_0/MB3 | 1237 | 268 (21.7%) | 135 (10.9%) | 124 | 177 (14.3%) | 115 (9.3%) | 144 (11.6%) |
| N128/M32/IQ4_NL/MB2 | 1766 | 548 (31.0%) | 245 (13.9%) | 194 | 158 (8.9%) | 128 (7.2%) | 164 (9.3%) |
| N128/M32/Q4_0/MB2 | 1658 | 548 (33.1%) | 125 (7.5%) | 188 | 182 (11.0%) | 128 (7.7%) | 164 (9.9%) |
| N64/M16/IQ4_NL/MB2 | 1071 | 134 (12.5%) | 237 (22.1%) | 94 | 126 (11.8%) | 112 (10.5%) | 139 (13.0%) |
| N64/M16/Q4_0/MB2 | 963 | 134 (13.9%) | 117 (12.1%) | 88 | 150 (15.6%) | 112 (11.6%) | 139 (14.4%) |
| N64/M32/IQ4_NL/MB2 | 1345 | 268 (19.9%) | 254 (18.9%) | 130 | 153 (11.4%) | 115 (8.6%) | 144 (10.7%) |
| N64/M32/Q4_0/MB2 | 1237 | 268 (21.7%) | 134 (10.8%) | 124 | 177 (14.3%) | 115 (9.3%) | 144 (11.6%) |
| N64/M64/IQ4_NL/MB2 | 1735 | 548 (31.6%) | 254 (14.6%) | 193 | 160 (9.2%) | 113 (6.5%) | 145 (8.4%) |
| N64/M64/Q4_0/MB2 | 1627 | 548 (33.7%) | 134 (8.2%) | 187 | 184 (11.3%) | 113 (6.9%) | 145 (8.9%) |

**Compute** = `v_dot4_i32_i8` + `v_cvt_f32_i32` + `v_fmac_f32` + `v_mul_f32` + `v_cvt_f32_f16`
**Decode** = `v_and` + `v_or` + `v_xor` + `v_lshr` + `v_lshl` + `v_perm` (IQ4_NL only)

**Key scaling pattern**: Compute instructions are identical across Q4_0/IQ4_NL for the same tile geometry (e.g., both N128/M32 variants have exactly 384 `v_dot4`, 96 `v_cvt_f32_i32`, etc.). The only differences are decode and address calculation.

#### 16.7.2 Positive Findings (Compiler Doing Well)

1. **✅ NO flat_load/flat_store in any kernel**: All global memory access uses coalesced `global_load_dwordx4` / `global_store_dword[x4]`. Zero uncoalesced operations across all 10 variants.

2. **✅ NO scratch spills for Q4_0 variants**: Only IQ4_NL N128/M16/MB3 has 1 `buffer_load` + 1 `buffer_store` (8 bytes scratch from forcing `__launch_bounds__(256,3)` compressing 88→84 VGPRs). All other variants are spill-free.

3. **✅ v_fma_mix_f32 for scale handling**: The compiler fuses the FP16→FP32 scale conversion + FMA accumulation into a single `v_fma_mix_f32` instruction with `op_sel_hi:[0,1,0]`. This replaces a separate `v_cvt_f32_f16` + `v_fmac_f32` pair, saving 1 instruction per scale application. For BPT=2 with 4 M-rows per thread, this saves 8 instructions per tile iteration.

4. **✅ 128-bit LDS reads**: All compute reads use `ds_read_b128` (16 bytes per read), maximizing LDS bandwidth utilization.

5. **✅ Bank-conflict-free LDS writes**: Decoded INT8 data uses `ds_write2st64_b32` (strided 2-element writes with 64-dword stride) for bank-conflict avoidance.

6. **✅ Optimal s_waitcnt pipelining in hot loop**: The inner k-group loop uses progressive countdown `lgkmcnt(3)` → `lgkmcnt(2)` → `lgkmcnt(1)` → `lgkmcnt(0)`, each followed by 16 `v_dot4_i32_i8` instructions. This properly pipelines LDS reads with compute — later reads complete while early results are consumed.

7. **✅ Double-buffered load/decode overlaps with compute**: The safe loop issues `s_waitcnt vmcnt(1)` to start B-matrix decode while the A-matrix global_load is still in flight. Decode bitops execute concurrently with the A-data transfer latency.

8. **✅ 4× unroll confirmed**: The `#pragma clang loop unroll_count(4)` correctly produces 4 groups of 16 `v_dot4` per BPT block (64 v_dot4 per block, 128 per tile iteration for BPT=2), matching 8 k-groups with 4 unrolled.

9. **✅ Vectorized output stores on fast path**: `global_store_dwordx4` (128-bit) is used for the non-boundary output path. The boundary path falls back to per-element `global_store_dword` with exec masking.

10. **✅ Minimal NOPs**: Only 3 `s_nop` instructions per kernel — negligible.

#### 16.7.3 Findings That Look Concerning But Are Not Actionable

1. **v_mov_b32 overhead (88-194 per kernel, 10-12% of instructions)**:
   - **Source**: Primarily integer accumulator zeroing (16 `v_mov vX, 0` per BPT block = 32 per tile), plus compiler register-file pressure relief moves.
   - **Why not actionable**: Zeroing runs once per tile iteration (before the k-group inner loop), not in the hot path. Compiler-inserted v_mov for VGPR shuffling is a consequence of register pressure — reducing VGPRs would require reducing the M or N tile size, which hurts compute density.

2. **Address calculation overhead (8.9-15.6%)**:
   - Dominated by 64-bit pointer arithmetic: `v_mad_i64_i32`, `v_lshlrev_b64`, `v_add_co_u32`/`v_addc_co_u32` pairs.
   - **Note**: The Q4_0 "decode" bias (`v_add_u32 v, 0x78787878, v`) is categorized as address calc by instruction name but is actually part of the SWAR decode. True Q4_0 decode overhead is ~28 VALU per cooperative load block (not the reported 10.9%).
   - **Why not actionable**: 64-bit pointer math is unavoidable for addressing matrices with >4GB. The compiler already fuses shift+OR (`v_lshl_or_b32`) and uses `v_mad_i64_i32` for combined multiply-add.

3. **Boundary-checked output stores**:
   - The output section emits conditional scalar `global_store_dword` per M-row element, guarded by `v_cmp_gt_i32` + `s_and_saveexec` + `s_cbranch_execz` per element, with a `global_store_dwordx4` fast path when all 4 N-columns are in bounds.
   - **Why not actionable**: These branch instructions are predicted/eliminated in the common case where the tile fits entirely within the matrix. The branch code costs only icache space, not execution cycles for non-boundary tiles.

#### 16.7.4 IQ4_NL Decode Overhead Analysis

**IQ4_NL v_perm LUT decode per cooperative load block (32 values from 16 packed bytes):**

| Instruction | Count | Purpose |
|------------|-------|---------|
| `v_lshrrev_b32` | 12 | Extract nibble bits (shift by 1, 4, or 5) |
| `v_and_b32` | 12 | Mask 3-bit LUT index and 1-bit mux flag |
| `v_perm_b32` | 24 | 3 LUT lookups per nibble-pair × 2 pairs × 4 dwords |
| `v_or_b32` | 8 | Combine mux flag with identity permutation |
| `s_mov_b32` + `v_mov_b32` | 4 | Load LUT constants (amortized) |
| **Total** | **60** | |

**Q4_0 SWAR decode per cooperative load block (32 values from 16 packed bytes):**

| Instruction | Count | Purpose |
|------------|-------|---------|
| `v_and_b32` | 8 | Mask nibbles (low + high) |
| `v_lshrrev_b32` | 4 | Shift high nibbles to position |
| `v_add_u32` | 8 | SWAR bias (+0x78 per byte) |
| `v_xor_b32` | 8 | Sign conversion (XOR 0x80) |
| **Total** | **28** | |

**IQ4_NL decode is 2.14× more instructions than Q4_0** (+32 VALU per cooperative load).

With ~4 decode sites per kernel (preload + safe loop + boundary loop + final), the total per-kernel overhead is approximately **+120 instructions**, which matches the observed difference: N128/M32/IQ4_NL has 245 decode bitops vs Q4_0's 125 (Δ=120).

This decode overhead is **fundamental to the format** — the v_perm LUT path is the optimal decode for IQ4_NL on GCN (confirmed by Phase 4-5 analysis). The overhead directly explains the ~5% IQ4_NL-vs-Q4_0 performance gap at each M value.

#### 16.7.5 Hot Loop Structure (N128/M32/Q4_0 Representative)

The inner loop for one BPT block (verified from ISA at lines 1612-1682):

```
    ┌─ ds_read_b128 × 4     (B-matrix: 4 × 16B = 64B from LDS)
    │  ds_read_b128 × 4     (A-matrix: 4 × 16B = 64B from LDS)
    │  s_addk_i32            (loop counter)
    │
    │  s_waitcnt lgkmcnt(3)  ─── first A-row ready ───
    │  v_dot4 × 16           (4 M-rows × 4 N-cols = 16 MACs, k-group 0)
    │
    │  s_waitcnt lgkmcnt(2)  ─── second A-row ready ───
    │  v_dot4 × 16           (k-group 1)
    │
    │  s_waitcnt lgkmcnt(1)  ─── third A-row ready ───
    │  v_dot4 × 16           (k-group 2)
    │
    │  s_waitcnt lgkmcnt(0)  ─── all reads ready ───
    │  v_dot4 × 16           (k-group 3)
    │
    └─ s_cbranch_scc0 (back-edge)
```

This loop body runs twice per BPT block (8 k-groups / 4 unrolled = 2 iterations), producing:
- 2 iters × 64 v_dot4 = **128 v_dot4 per BPT block**
- 2 BPT blocks = **256 v_dot4 per tile** (128 from each of the 2 register-file–separated BPT loops)

The remaining 128 v_dot4 (of the kernel's 384 total) come from the boundary loop and final tile compute, which have identical inner structure but with bounds-checked global loads.

#### 16.7.6 Instruction Efficiency Ratios

| Variant | Compute % | Non-Compute Overhead |
|---------|-----------|---------------------|
| N64/M64/Q4_0 | 33.7% | 66.3% |
| N128/M32/Q4_0 | 33.1% | 66.9% |
| N64/M64/IQ4_NL | 31.6% | 68.4% |
| N128/M32/IQ4_NL | 31.0% | 69.0% |
| N128/M16/Q4_0 | 21.7% | 78.3% |
| N128/M16/IQ4_NL | 19.8% | 80.2% |
| N64/M16/Q4_0 | 13.9% | 86.1% |
| N64/M16/IQ4_NL | 12.5% | 87.5% |

**Observations**:
- M64/M32 variants achieve 31-34% compute density — comparable to hand-tuned GEMM kernels on GCN (typical range 30-40% for quantized formats with per-block scales).
- M16 variants have much lower compute density (13-22%) because the fixed per-tile overhead (cooperative load, decode, LDS write, barrier, accumulator zero) is amortized over fewer compute instructions (96-192 v_dot4 vs 384).
- This confirms the hybrid dispatch decision: M16 is only beneficial when workgroup count is low enough that 3-wave occupancy compensates for the lower compute density.

#### 16.7.7 Conclusions

The compiler is generating **high-quality ISA** for these kernels:
- No memory access pathologies (no flat loads, no unnecessary spills, coalesced global access)
- Optimal LDS usage (128-bit reads, strided writes, proper bank-conflict avoidance)
- Excellent instruction scheduling (progressive waitcnt pipelining, overlapping memory and compute)
- Smart instruction selection (`v_fma_mix_f32` fusion, `ds_write2st64_b32` strided writes)

The remaining performance gaps vs INT8 baseline are attributable to:
1. **Decode overhead**: +28-60 VALU per cooperative load block (Q4_0 vs IQ4_NL)
2. **Per-block scale accumulation**: `v_cvt_f32_i32` + `v_fma_mix_f32` per block boundary (2× per tile)
3. **Accumulator zeroing**: 16× `v_mov_b32 v, 0` per BPT block (INT8 has no block boundaries)

These are all **fundamental to the quantized format** — the kernel is operating near the floor set by the decode-in-load architecture. Further gains require algorithm-level changes (e.g., multi-block fusion to amortize scales, or format-specific compute paths that avoid INT8 intermediation entirely).

### 16.8 N64 Hybrid Dispatch Extension

#### 16.8.1 Problem

The N128 path (N-heavy shapes: FFN_Up, AttnQKV, LM_Head) had hybrid M16/3-wave dispatch since Phase 10, but the **N64 path (K-heavy shapes: FFN_Dn, AttnOut) completely lacked it**. This meant K-heavy shapes at small M launched far too few workgroups:

| Shape | Path | M | Tile | WGs | CU Coverage | Speedup |
|-------|------|---|------|-----|-------------|---------|
| 7B_FFN_Dn | N64 | 32 | M32 | 56 | 23% | 0.64-0.67× |
| 0.5B_FFN_Dn | N64 | 32 | M32 | 14 | 6% | 0.67-0.70× |
| 7B_AttnOut | N64 | 32 | M32 | 56 | 23% | 0.70-0.73× |
| 0.5B_AttnOut | N64 | 64 | M64 | 14 | 6% | 0.95-0.97× |

These were the **worst-performing shapes** across the entire benchmark suite.

#### 16.8.2 Solution

Added hybrid M16/3-wave dispatch to the N64 path, mirroring the N128 strategy with a **tighter WG threshold of 128** (vs 256 for N128). The lower threshold accounts for N64 tiles having half the column-output of N128 tiles, meaning lower per-WG compute density — above 128 WGs, the GPU has enough parallelism for M32/M64 tiles to win via better A-data reuse.

**Dispatch logic (N64 path)**:
```
if M ≤ 64 AND total_m16_wgs ≤ 128:
    dispatch M16/3-wave (N64/M16/MB3)
else:
    dispatch original M32/M64/2-wave
```

**Threshold derivation**: First attempt used 256 WGs (matching N128). This caused catastrophic regressions for 7B shapes at M=64 (IQ4_NL 7B_FFN_Dn: 0.81→0.56×, -31%). Root cause: 7B M=64 shapes create 224 M16 WGs — the GPU is already 93% saturated, but each WG has poor compute density with M16 tiles over long K-loops (K=18944). The 128 threshold excludes 7B M=64 shapes (224 WGs > 128) while keeping 0.5B/3B M=64 wins (56-128 WGs ≤ 128).

#### 16.8.3 ISA Quality

Two new kernel variants were added:

| Kernel | VGPRs | Occupancy | Spills | Notes |
|--------|-------|-----------|--------|-------|
| N64/M16/Q4_0/MB3 | 64 | **4-wave** | 0 | Compiler beat 3-wave target! |
| N64/M16/IQ4_NL/MB3 | 78 | 3-wave | 0 | Clean, zero spills |

The Q4_0 variant achieved 4-wave occupancy (64 VGPRs ≤ 64 threshold on gfx906) — the compiler aggressively compressed registers below the 3-wave target of 84 VGPRs. Total kernel count: 12 (was 10).

#### 16.8.4 Results

**Grand Summary (averaged across all 11 shapes per M value):**

| Format | M | Before | After | Delta |
|--------|---|--------|-------|-------|
| Q4_0 | 32 | 0.84× | **0.89×** | **+0.05** |
| Q4_0 | 64 | 0.94× | **0.99×** | **+0.05** |
| Q4_0 | 128 | 1.09× | 1.09× | 0.00 |
| Q4_0 | 256 | 1.11× | 1.11× | 0.00 |
| IQ4_NL | 32 | 0.80× | **0.86×** | **+0.07** |
| IQ4_NL | 64 | 0.89× | **0.96×** | **+0.07** |
| IQ4_NL | 128 | 1.03× | 1.03× | 0.00 |
| IQ4_NL | 256 | 1.05× | 1.05× | 0.00 |

**Per-shape breakdown**: 11 wins (>+0.05), 76 neutral, 1 borderline regression (-0.05, within noise). Zero regressions at M=128/256.

**Biggest wins (all M=32/64, undersaturated grids):**

| Shape | M | Before | After | Delta |
|-------|---|--------|-------|-------|
| IQ4_NL 0.5B_FFN_Dn | 64 | 0.83× | 1.25× | **+0.42** |
| IQ4_NL 0.5B_AttnOut | 32 | 0.83× | 1.18× | **+0.35** |
| IQ4_NL 0.5B_AttnOut | 64 | 0.95× | 1.28× | **+0.33** |
| Q4_0 0.5B_FFN_Dn | 64 | 0.85× | 1.14× | **+0.29** |
| Q4_0 0.5B_AttnOut | 32 | 0.86× | 1.11× | **+0.25** |
| IQ4_NL 0.5B_FFN_Dn | 32 | 0.67× | 0.91× | **+0.24** |

#### 16.8.5 Threshold Selection: 128 vs 256 WGs

First attempt (threshold=256) showed a split pattern at M=64:

| Shape | M=64 WGs | Threshold 256 | Threshold 128 |
|-------|----------|---------------|---------------|
| 0.5B_FFN_Dn | 56 | +0.29/+0.42 | +0.29/+0.42 |
| 3B_FFN_Dn | 128 | +0.07/+0.05 | -0.05/neutral |
| 7B_FFN_Dn | 224 | **-0.09/-0.25** | neutral |

At 224 WGs (7B shapes), the GPU already has 93% SIMD coverage at 2-wave. Switching to M16/3-wave fragments work into 4× more WGs with 4× less output per WG, and the long K-loops (K=18944) amplify per-WG overhead. The N128 path doesn't have this problem because N128 tiles have 2× the column width, giving each WG better compute density.

The 128 threshold matches a ≈53% SIMD saturation point (128/240 SIMDs). Below this, the occupancy advantage of 3-wave matters more than M-tile efficiency. Above this, the GPU has enough parallelism for larger tiles to dominate.

### 16.9 M_TILE > 32 Investigation (LM_Head)

#### 16.9.1 Motivation

LM_Head (N=151936, K=896/2048) is the worst-performing shape, running at 0.67-0.93× vs INT8 V7 across all M values.  The current N128 dispatch uses M32 (THREADS_M=8, THREADS_N=32, N_PER_THREAD=4), while INT8 V7 uses M128 (THREADS_M=32, THREADS_N=8, N_PER_THREAD=16) — a **4× A-data reuse advantage** per thread.

The hypothesis was that increasing M_TILE to 64 or 128 would double N_PER_THREAD from 4 to 8, improving A-data reuse and closing the gap with INT8 V7.

#### 16.9.2 Resource Budget Analysis

M128/N128 is infeasible due to the **dual accumulator architecture** (acc[] + block_acc[] coexist during the fmaf epilogue):

| Config | acc VGPRs | block_acc VGPRs | Total accum | Other | Est. total | Occupancy |
|--------|-----------|-----------------|-------------|-------|-----------|-----------|
| M32/N128 | 4×4=16 | 4×4=16 | 32 | ~94 | ~126 | 2-wave ✓ |
| M64/N128 | 4×8=32 | 4×8=32 | 64 | ~55 | ~119 | 2-wave ✓ |
| M128/N64 | 4×8=32 | 4×8=32 | 64 | ~56 | ~120 | 2-wave ✓ |
| M128/N128 | 4×16=64 | 4×16=64 | **128** | ~75 | **~203** | **1-wave ✗** |

M64/N128 and M128/N64 both achieve N_PER_THREAD=8 (2× improvement) within the 2-wave VGPR budget.  Actual compiler output confirmed exactly 128 VGPRs for all 4 new variants (2 formats × 2 configs).

#### 16.9.3 The A-LDS Bank Conflict Wall

**Empirical result**: Both M64/N128 and M128/N64 produced **catastrophic regressions** on all N-heavy shapes:

| Config | Q4_0 0.5B LM_Head M=128 | Q4_0 3B LM_Head M=128 | IQ4_NL 0.5B LM_Head M=128 |
|--------|--------------------------|------------------------|---------------------------|
| M32/N128 (baseline) | 0.88× | 0.74× | 0.80× |
| M128/N64 | **0.37×** | **0.31×** | **0.39×** |
| M64/N128 (at M=64) | **0.54×** (was 0.91×) | **0.50×** (was 0.82×) | **0.49×** (was 0.83×) |

**Root cause**: A-LDS bank conflicts on gfx906.

gfx906 has 32 LDS banks at 4-byte (dword) granularity.  Each thread reads `M_PER_THREAD=4` consecutive int32 from A-LDS at addresses `[t_m*4, t_m*4+1, t_m*4+2, t_m*4+3]`, spanning 4 consecutive banks starting at bank `(t_m*4) % 32`.  Since `gcd(4, 32) = 4`, only **8 distinct bank groups** exist (banks {0-3}, {4-7}, ..., {28-31}).  When `THREADS_M > 8`, multiple threads map to the same bank group within a half-wavefront (32 threads):

| THREADS_M | M_TILE | Threads per half-wave | Unique bank groups | Bank conflict factor |
|-----------|--------|----------------------|-------------------|---------------------|
| 8 | 32 | 8 | 8 | **1× (zero conflicts)** |
| 16 | 64 | 16 | 8 | **2-way** |
| 32 | 128 | 32 | 8 | **4-way** |

The bank conflict penalty scales linearly: each A-LDS read takes N cycles instead of 1, where N is the conflict factor.  With 4 A-register loads per thread per k-group, 8 k-groups per block, and 2 blocks per tile:
- M32: 4 reads × 1 cyc × 16 = 64 total A-LDS cycles per tile → baseline
- M64: 4 reads × 2 cyc × 16 = 128 cycles → **2× penalty**
- M128: 4 reads × 4 cyc × 16 = 256 cycles → **4× penalty**

This penalty **completely negates** the 2× A-data reuse improvement from N_PER_THREAD doubling, resulting in net 1.7-2.5× regressions.

#### 16.9.4 Why INT8 V7 Tolerates M128

The INT8 V7 kernel uses the *same* A-LDS layout with the *same* bank conflicts at M128 (THREADS_M=32 → 4-way), yet still performs well.  The key differences:

1. **No decode overhead**: INT8 reads pre-packed data — zero VALU for B-tile loading.  Native VNNI spends 28-60 extra VALU per B-tile on decode, so any compute-phase penalty is proportionally larger.

2. **No block_acc**: INT8 accumulates directly into INT32 without per-block FP16 scale conversion.  This reduces peak live register count and gives the compiler more room to reorder instructions around bank conflicts (117 VGPRs vs 128).

3. **Lower conflict sensitivity**: INT8's lighter inner loop (no `v_cvt_f32_f16` + `v_fmaf_rn` per block boundary) means the bank conflict stalls are a smaller fraction of total execution time.

#### 16.9.5 Fundamental Constraint

**M_TILE ≤ 32 is the hardware-optimal M-tile for native-VNNI GEMM on gfx906** with the current LDS layout (`a_lds[kk * M_TILE + mi]`) and `M_PER_THREAD=4`.

The constraint derives from: `max_conflict_free_THREADS_M = 32_banks / gcd(M_PER_THREAD, 32_banks) = 32 / 4 = 8`, giving `M_TILE_max = 8 × 4 = 32`.

Alternative approaches considered but rejected:
- **LDS padding** (`a_lds[kk * (M_TILE + pad) + mi]`): Padding shifts bank alignment across K-groups but does NOT break within-kk conflicts (the problem is same-kk, different-t_m collisions).
- **A-LDS transpose** (`a_lds[mi * KT + kk]`): Puts ALL threads on the same bank for a given (kk, mm). Worse than current layout.
- **M_PER_THREAD=1** with THREADS_M=32, N_PER_THREAD=16: Zero bank conflicts but 17 LDS reads/kgroup for 16 compute ops (1.06 cycles/output) vs M_PER_THREAD=4's 8 reads/16 ops (0.5 cycles/output). The 4×4 cross product is fundamentally more LDS-efficient.

#### 16.9.6 Implications for LM_Head Optimization

The LM_Head performance gap (0.67-0.93× vs INT8) cannot be closed by M_TILE changes alone.  The remaining gap comes from:

1. **Decode overhead** (~28-60 VALU per B-tile, fixed cost)
2. **Per-block scale accumulation** (`v_cvt_f32_f16` + `v_fmaf_rn` per block, K/32 times)
3. **Dual accumulator pressure** (block_acc + acc → 128 VGPRs, no compiler slack)

These are architectural costs of the decode-in-load approach.  Potential future paths:
- **Wider BPT** (e.g., BPT=4): Amortize pipeline overhead across more K-elements per tile, reducing decode-per-output ratio.  Risk: LDS budget doubles per K-dimension.
- **Asymmetric M_PER_THREAD**: Use M_PER_THREAD=2 with THREADS_M=16 → conflict-free up to M_TILE=32 still, but reduces accumulator pressure.  Not obviously better.
- **Format-specific optimizations**: Q4_0's simpler decode (SWAR subtract, no LUT) may tolerate different tradeoffs than IQ4_NL.
