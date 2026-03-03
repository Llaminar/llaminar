# Native-VNNI GEMM Kernel Design Proposal

## 1. Executive Summary

**Goal**: Design a new family of GEMM kernels (`ROCmGemmKernel_native_VNNI.hip`) that execute M>1 prefill using native sub-8-bit quantization formats (1-bit to 6-bit), bypassing the current INT8 intermediary path.

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
| `src/v2/kernels/rocm/ROCmGemmKernel_native_VNNI.hip` | **NEW**: Native-VNNI GEMM kernels |
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
