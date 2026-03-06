# Native-VNNI GEMV ISA Analysis — AMD MI50/MI60 (gfx906)

This document presents a detailed ISA-level analysis of the native-VNNI GEMV decode kernels,
following the same methodology used in `README.vnni-gemm-tuning.md` for the INT8 GEMM kernels.
The goal is to identify performance bottlenecks and actionable optimization paths.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Analysis Setup](#analysis-setup)
- [Resource Metadata: All Kernels](#resource-metadata-all-kernels)
- [Decode:Compute Ratio Table](#decodecompute-ratio-table)
- [The Gold Standard: INT8 Inner Loop](#the-gold-standard-int8-inner-loop)
- [Q4_0 Hot Loop: Anatomy of the Bottleneck](#q40-hot-loop-anatomy-of-the-bottleneck)
- [Q4_1 vs Q4_0 Anomaly](#q41-vs-q40-anomaly)
- [IQ4_NL: Death by LUT](#iq4_nl-death-by-lut)
- [IQ Grid Formats: Progressive Waitcnt](#iq-grid-formats-progressive-waitcnt)
- [GEMM V3 Pipeline: The Reference Pattern](#gemm-v3-pipeline-the-reference-pattern)
- [Optimization #1: Software-Pipelined K-Loop](#optimization-1-software-pipelined-k-loop)
- [Optimization #2: LDS LUT Migration](#optimization-2-lds-lut-migration)
- [Optimization #3: Q4_0 Compiler Codegen Anomaly](#optimization-3-q40-compiler-codegen-anomaly)
- [Optimization Priority Matrix](#optimization-priority-matrix)
- [Theoretical Speedup Estimates](#theoretical-speedup-estimates)

---

## Executive Summary

The native-VNNI GEMV kernels decode quantized weights to INT8 on-the-fly and use `v_dot4_i32_i8`
for accumulation. ISA analysis reveals **three systemic bottlenecks** that explain the ~1.0x
performance parity with the pre-quantized INT8 GEMV baseline:

1. **Serialized load/decode pipeline** — `s_waitcnt vmcnt(0)` at the top of each K-loop iteration
   creates a hard stall. The ~120 decode ALU cycles happen *after* the stall, providing zero
   overlap with memory latency.

2. **Global memory LUT lookups** — IQ4_NL issues 68 `global_load_ubyte` instructions per block
   to look up 16-byte codebook entries via global memory (~100 cycles/lookup). The codebook
   is only 16 bytes and should be in LDS or scalar registers.

3. **Compiler codegen divergence** — Q4_0 compiles to 686 ISA lines with 38 global_loads and
   27 branches, while the structurally-identical Q4_1 compiles to 254 lines with 6 loads and
   9 branches.

The primary optimization — **software-pipelined K-loop** inspired by our GEMM V3's LDS
double-buffer — would restructure the loop to issue loads for block B+1 *before* decoding block B,
using the 120+ decode ALU cycles to hide memory latency instead of wasting them.

---

## Analysis Setup

**Binary analyzed**: `build_v2_release/tests/v2/v2_perf_native_vnni_throughput`
(Release build with `-O3 -march=native`)

**Code object extraction**:
```bash
# List embedded GPU code objects
roc-obj-ls build_v2_release/tests/v2/v2_perf_native_vnni_throughput

# Extract native-VNNI kernels (co5) and INT8 baseline (co4)
dd if=<binary> of=/tmp/nvnni_co5.elf bs=1 skip=<offset_5> count=118672
dd if=<binary> of=/tmp/nvnni_co4.elf bs=1 skip=<offset_4> count=217712

# Full disassembly
llvm-objdump -d --mcpu=gfx906 /tmp/nvnni_co5.elf > /tmp/nvnni_full_disasm.s   # 6328 lines
llvm-objdump -d --mcpu=gfx906 /tmp/nvnni_co4.elf > /tmp/int8_vnni_disasm.s    # 10238 lines
```

**Toolchain**: `/opt/rocm/llvm/bin/{llvm-readelf, llvm-objcopy, llvm-objdump}`, `/opt/rocm/bin/roc-obj-ls`

---

## Resource Metadata: All Kernels

Extracted via `llvm-readelf --notes` from the release binary code objects.

### Native-VNNI Kernels (15 format variants + reduce)

| Format   | BPW  | VGPRs | SGPRs | LDS | Spills | Waves/SIMD | ISA Lines |
|----------|------|-------|-------|-----|--------|------------|-----------|
| Q4_0     | 4.5  | 29    | 56    | 0   | 0      | 4          | 686       |
| IQ4_NL   | 4.5  | 55    | 56    | 0   | 0      | 4          | 466       |
| Q4_1     | 5.0  | 33    | 56    | 0   | 0      | 4          | 254       |
| Q5_0     | 5.5  | 35    | 56    | 0   | 0      | 4          | 458       |
| Q5_1     | 6.0  | 45    | 56    | 0   | 0      | 4          | 402       |
| Q6_K     | 6.6  | 45    | 54    | 0   | 0      | 4          | 468       |
| Q3_K     | 3.4  | 39    | 58    | 0   | 0      | 4          | 400       |
| Q2_K     | 2.6  | 35    | 58    | 0   | 0      | 4          | 300       |
| IQ3_S    | 3.4  | 35    | 34    | 0   | 0      | 4          | 302       |
| IQ3_XXS  | 3.4  | 32    | 34    | 0   | 0      | 4          | 291       |
| IQ2_S    | 2.5  | 38    | 38    | 0   | 0      | 4          | 257       |
| IQ2_XS   | 2.3  | 38    | 38    | 0   | 0      | 4          | 255       |
| IQ2_XXS  | 2.1  | 32    | 32    | 0   | 0      | 4          | 225       |
| IQ1_S    | 1.6  | 35    | 38    | 0   | 0      | 4          | 176       |
| IQ1_M    | 1.6  | 37    | 38    | 0   | 0      | 4          | 216       |
| reduce   | —    | 5     | 18    | 0   | 0      | 10         | 50        |

### INT8 Baseline Kernels (reference)

| Kernel                    | VGPRs | SGPRs | LDS | Spills | Waves/SIMD |
|---------------------------|-------|-------|-----|--------|------------|
| scatter_selfreduce (TN128, CPT2) | 17 | 48 | 0 | 0 | 4 |
| direct (TN128, CPT2)     | 13    | 34    | 0   | 0      | 4          |
| reduce                    | 5     | 18    | 0   | 0      | 10         |

**Key finding**: All kernels achieve 4 waves/SIMD with zero register spills and zero LDS.
Occupancy is NOT the bottleneck. VGPRs range from 29 (Q4_0) to 55 (IQ4_NL), all well within
the 64 VGPR budget for 4-wave occupancy on gfx906. The INT8 baseline uses only 17 VGPRs.

---

## Decode:Compute Ratio Table

Instruction census from the full disassembly, counted per format kernel's ISA range:

| Format   | BPW  | v_dot4 | global_loads | decode ALU | Decode:Compute | Kernel Eff |
|----------|------|--------|--------------|------------|----------------|------------|
| Q4_0     | 4.5  | 16     | 38           | 253        | 15.8×          | 58%        |
| IQ4_NL   | 4.5  | 16     | 68           | 135        | 8.4×           | 58%        |
| Q4_1     | 5.0  | 24     | 6            | 79         | 3.3×           | 62%        |
| Q5_0     | 5.5  | 16     | 6            | 280        | 17.5×          | 70%        |
| Q5_1     | 6.0  | 24     | 8            | 236        | 9.8×           | 74%        |
| Q6_K     | 6.6  | 16     | 8            | 426        | 26.6×          | 85%        |
| Q3_K     | 3.4  | 16     | 6            | 282        | 17.6×          | 44%        |
| Q2_K     | 2.6  | 24     | 8            | 146        | 6.1×           | 33%        |
| IQ3_S    | 3.4  | 16     | 23           | 204        | 12.8×          | 44%        |
| IQ3_XXS  | 3.1  | 16     | 20           | 166        | 10.4×          | 41%        |
| IQ2_S    | 2.5  | 16     | 16           | 180        | 11.2×          | 32%        |
| IQ2_XS   | 2.3  | 16     | 16           | 180        | 11.2×          | 32%        |
| IQ2_XXS  | 2.1  | 16     | 12           | 159        | 9.9×           | 30%        |
| IQ1_S    | 1.6  | 24     | 16           | 30         | 1.2×           | 26%        |
| IQ1_M    | 1.6  | 24     | 18           | 30         | 1.2×           | 25%        |
| **INT8** | 8.0  | 10     | 12           | **0**      | **0.0×**       | 100%       |

**Decode:Compute** = decode ALU instructions / v_dot4 instructions. This measures how many cycles
of "overhead" exist per cycle of useful compute.

**Kernel Efficiency** = BPW / 8.0 — the theoretical bandwidth savings vs INT8. A kernel with
4.5 BPW reads 56% of the bytes that INT8 reads. If it were purely memory-bandwidth-bound,
it would be 1.78× faster. Instead, it's ~1.0× — confirming that decode ALU dominates.

**Interpretation**: For Q6_K, there are 26.6 decode ALU instructions for every v_dot4. This means
the actual dot-product computation is less than 4% of the kernel's work. Even if v_dot4 were
free, eliminating it would only save 4% — the kernel is overwhelmingly decode-bound.

---

## The Gold Standard: INT8 Inner Loop

The INT8 `scatter_selfreduce` kernel demonstrates near-optimal GEMV structure. Here is the
main loop (ISA lines 6560–6604, from `/tmp/int8_vnni_disasm.s`):

```asm
;; === LOAD PHASE: 4 loads issued in parallel ===
global_load_dwordx2 v[9:10],  v[3:4],   off      ; Load 8B INT8 weights (pass 0)
global_load_dwordx2 v[13:14], v[11:12], off      ; Load 8B (pass 1)
global_load_dwordx2 v[15:16], v[11:12], off      ; Load 8B (pass 2)
global_load_dwordx2 v[11:12], v[11:12], off      ; Load 8B (pass 3)
;; ... ~24 instructions of pointer arithmetic ...

;; === COMPUTE PHASE: progressive waitcnt + dot4 ===
s_waitcnt vmcnt(3) lgkmcnt(0)                     ; Wait for pass 0 only
v_dot4_i32_i8 v6, s27, v9,  v6                    ; dot4 col 0 (pass 0)
v_dot4_i32_i8 v5, s27, v10, v5                    ; dot4 col 1 (pass 0)
s_waitcnt vmcnt(2)                                 ; Wait for pass 1
v_dot4_i32_i8 v6, s21, v13, v6                    ; dot4 col 0 (pass 1)
v_dot4_i32_i8 v5, s21, v14, v5                    ; dot4 col 1 (pass 1)
s_waitcnt vmcnt(1)                                 ; Wait for pass 2
v_dot4_i32_i8 v6, s35, v15, v6                    ; dot4 col 0 (pass 2)
v_dot4_i32_i8 v5, s35, v16, v5                    ; dot4 col 1 (pass 2)
s_waitcnt vmcnt(0)                                 ; Wait for pass 3
v_dot4_i32_i8 v6, s0, v11, v6                     ; dot4 col 0 (pass 3)
v_dot4_i32_i8 v5, s0, v12, v5                     ; dot4 col 1 (pass 3)
s_cbranch_scc0 back_edge                           ; Loop back
```

**What makes this optimal**:
1. **All 4 loads issued up front** — the memory system can service them in parallel
2. **Progressive waitcnt** — `vmcnt(3)→(2)→(1)→(0)` consumes each load as it arrives,
   allowing later loads to overlap with earlier dot4 computation
3. **A-vector in scalar registers** — `s27`, `s21`, `s35`, `s0` loaded via `s_load_dwordx4`
   in the prologue, no per-iteration scalar loads
4. **Zero decode overhead** — the path from load to dot4 has no ALU between them
5. **~17 instructions total** per inner loop iteration (4 loads + 8 dot4 + 4 waitcnt + 1 branch)

This processes 32 INT8 elements × 2 columns = 64 dot products per iteration.

---

## Q4_0 Hot Loop: Anatomy of the Bottleneck

The Q4_0 kernel (ISA lines 92–253, from `/tmp/nvnni_full_disasm.s`) has a dramatically
different structure:

```
┌──────────────────── ITERATION START ────────────────────┐
│                                                          │
│  s_or_b64 exec, exec, s[0:1]     ; restore exec mask   │
│  s_waitcnt vmcnt(0)              ; ■■■ HARD STALL ■■■  │ ← Loads just issued 2-4 cycles ago!
│                                   ; Stalls ~100-300 cy  │
│                                                          │
│  ┌── DECODE PHASE (~120 ALU instructions) ──────────┐   │
│  │  v_and_b32, v_lshrrev_b32, v_add_u16 ...        │   │
│  │  v_perm_b32, v_or3_b32, ...                      │   │ 120 cycles of useful-but-wasted
│  │  [nibble unpack, bias subtract, byte packing]     │   │ compute that COULD be hiding
│  │  ... ~120 instructions for 8 packed_groups ...    │   │ memory latency
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  s_waitcnt lgkmcnt(0)            ; wait for A-vector    │
│                                                          │
│  ┌── COMPUTE PHASE (8× v_dot4) ────────────────────┐   │
│  │  v_dot4_i32_i8 v7, s8,  v20, 0    ; groups 0-7  │   │
│  │  v_dot4_i32_i8 v4, s9,  v4,  v7                 │   │
│  │  v_dot4_i32_i8 v4, s10, v5,  v4                 │   │
│  │  v_dot4_i32_i8 v4, s11, v6,  v4                 │   │
│  │  v_dot4_i32_i8 v0, s12, v0,  v4                 │   │
│  │  v_dot4_i32_i8 v0, s13, v1,  v0                 │   │
│  │  v_dot4_i32_i8 v0, s14, v2,  v0                 │   │
│  │  v_dot4_i32_i8 v0, s15, v3,  v0                 │   │ 8 cycles
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  v_cvt_f32_i32 + v_fma_mix_f32   ; scale application   │
│  pointer advance, loop counter++                        │
│  s_cmp_ge_i32 + s_cbranch_scc1   ; exit if done        │
│                                                          │
│  ┌── NEXT ITERATION LOADS (too late!) ──────────────┐   │
│  │  global_load_ushort v19       ; scale (col 0)    │   │
│  │  global_load_dwordx4 v[0:3]   ; payload (col 0)  │   │ Loads issued HERE,
│  │  s_load_dwordx8 s[8:15]       ; A-vector         │   │ but waitcnt is only
│  │  global_load_ushort v20       ; scale (col 1)    │   │ 2-4 instructions away
│  │  global_load_dwordx4 v[4:7]   ; payload (col 1)  │   │ at the TOP of the loop!
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  s_cbranch → ITERATION START (2 instructions later!)    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**The critical problem**: Loads for the NEXT iteration are issued at the BOTTOM of the loop
body, just 2-4 instructions before the `s_waitcnt vmcnt(0)` at the TOP. This gives the memory
system virtually zero time to service the loads before the hard stall.

Meanwhile, the 120 decode ALU instructions — which take ~120 cycles on VALU — execute AFTER
the data has already arrived. These 120 cycles could be overlapping with memory loads for the
NEXT block, but instead they run with the memory pipeline completely idle.

**Cycle budget per iteration (current)**:
| Phase | Cycles | Memory Active? |
|-------|--------|----------------|
| `s_waitcnt vmcnt(0)` stall | ~100-300 | Idle (waiting for data) |
| Decode ALU | ~120 | **Idle** (data arrived, no loads pending) |
| 8× v_dot4 | ~8 | Idle |
| Pointer math + loads | ~15 | Loads issued (too late) |
| **Total** | ~243-443 | Memory active: ~15/443 = 3% |

---

## Q4_1 vs Q4_0 Anomaly

Despite near-identical source code (Q4_1 is Q4_0 without the `-8` bias subtract), the compiler
produces dramatically different ISA:

| Metric | Q4_0 | Q4_1 | Ratio |
|--------|------|------|-------|
| ISA lines | 686 | 254 | 2.7× |
| global_loads | 38 | 6 | 6.3× |
| s_waitcnt vmcnt | 36 | 4 | 9.0× |
| Back-edge branches | 27 | 9 | 3.0× |
| Decode ALU | 253 | 79 | 3.2× |
| v_dot4 | 16 | 24 | 0.67× |

**Q4_0** appears to have been aggressively unrolled by the compiler into multiple specialized
code paths — possibly a main loop processing 2 blocks, a tail loop for 1 block, plus separate
paths for each CPT column. The 38 global_loads and 27 branches suggest 3-4 distinct code paths
within a single kernel invocation.

**Q4_1** has a clean, tight single-path loop with interleaved decode+dot4. It gets more v_dot4
instructions (24 vs 16) despite fewer total instructions, suggesting it processes 3 blocks per
iteration (3 × 8 = 24 dot4) in a compact loop body.

**Investigation needed**: The Q4_0 codegen may respond to:
- `#pragma clang loop unroll(disable)` / `unroll_count(1)` on the K-loop
- `__attribute__((noinline))` on the decode function
- Restructuring the source to match Q4_1's pattern more closely
- Manually specifying loop trip counts via `__builtin_assume`

---

## IQ4_NL: Death by LUT

IQ4_NL uses a 16-entry codebook (`k_iq4_codebook_i8[16]`) for dequantization. Each 4-bit nibble
is used as an index into this LUT to get the corresponding INT8 value.

**Source pattern** (per block of 32 elements):
```cpp
const int8_t lo0 = k_iq4_codebook_i8[b0 & 0x0F];  // LUT lookup
const int8_t lo1 = k_iq4_codebook_i8[b1 & 0x0F];  // LUT lookup
// ... 32 lookups total per block × CPT columns
```

**Compiled ISA**: 68 `global_load_ubyte` instructions per K-loop iteration. Each one is a
**scattered random access** to global memory — the address depends on the data (nibble value),
so coalescing is impossible.

```asm
;; IQ4_NL inner loop (ISA lines 770-860)
global_load_ubyte v24, v[24:25], off    ; LUT[nibble] → 1 byte from global memory
global_load_ubyte v25, v[25:26], off    ; another scattered LUT access
;; ... 66 more global_load_ubyte ...
s_waitcnt vmcnt(31)                      ; progressive: consume as loads arrive
v_lshl_or_b32 v28, v24, 8, v28          ; pack into INT32
s_waitcnt vmcnt(30)
v_lshlrev_b32 v24, 16, v25
;; ... interleaved pack + waitcnt ...
```

**The compiler IS doing software pipelining** — the progressive `vmcnt(31)→vmcnt(30)→...` pattern
consumes each LUT byte as it arrives while later lookups are in flight. But the fundamental
problem remains: 68 scattered global memory accesses at ~100 cycles each.

**The codebook is only 16 bytes**:

| Entry | Size | Current Location | Access Latency |
|-------|------|------------------|----------------|
| `k_iq4_codebook_i8[16]` | 16 B | Global memory (L2 hit ~100 cy) | ~100 cycles/access |
| Same in LDS | 16 B | LDS (`__shared__`) | ~4 cycles/access |
| Same in scalar regs | 16 B | 4× SGPRs | 0 cycles (register file) |

Moving the 16-byte codebook to LDS would reduce lookup overhead by **25×**:
`68 loads × 100 cycles = 6,800 cycles → 68 loads × 4 cycles = 272 cycles`

Even better: load the 16-byte LUT into 4 scalar registers (`s_load_dwordx4`) and use
`v_perm_b32` for register-level lookup. This eliminates memory accesses entirely.

---

## IQ Grid Formats: Progressive Waitcnt

The IQ grid formats (IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS) use larger LUTs for dequantization:

| Format  | Grid Table | Size | Entries | Bytes/Entry |
|---------|------------|------|---------|-------------|
| IQ3_S   | `iq3s_grid` | 2 KB | 512 | 4 |
| IQ3_XXS | `iq3xxs_grid` (same as IQ3_S) | 2 KB | 512 | 4 |
| IQ2_S   | `iq2s_grid` | 4 KB | 512 | 8 |
| IQ2_XS  | `iq2xs_grid` (same as IQ2_S) | 4 KB | 512 | 8 |
| IQ2_XXS | `iq2xxs_grid` (same as IQ2_S) | 4 KB | 512 | 8 |
| IQ1_S/M | `iq1_grid` | 16 KB | 2048 | 8 |

These kernels already use progressive waitcnt patterns:
```asm
;; IQ3_S inner loop
global_load_dword v..., v[...], off      ; grid4 = iq3s_grid[idx]
;; ... 7 more grid lookups ...
s_waitcnt vmcnt(7)                        ; consume first lookup
;; sign application + packing
s_waitcnt vmcnt(6)                        ; consume second
;; ... progressive consumption ...
```

**Optimization**: The 2-4 KB grid tables fit comfortably in LDS (64 KB budget on gfx906).
Pre-loading them to LDS at kernel start would replace `global_load_dword` (~100 cy) with
`ds_read_b32` (~4 cy). The IQ1 grid at 16 KB is larger but still fits in LDS.

---

## GEMM V3 Pipeline: The Reference Pattern

Our INT8 GEMM V3 kernel (`ROCmQuantisedGemmKernel_INT8_VNNI.hip`) achieves excellent performance
through a software-pipelined LDS double-buffer. The core trick is directly applicable to GEMV.

### GEMM V3 Pipeline Structure

```
PRELOAD: Issue loads for Tile 0
         ┌─────────────────────────────────────────────────┐
         │  global_load → a_staging[], b_staging           │
         └─────────────────────────────────────────────────┘

MAIN LOOP (tiles 1..T-1):
    ┌───────────────────────────────────────────────────────┐
    │  STEP 1: Issue loads for Tile T+1 → staging regs     │ ← Async: memory in flight
    │                                                       │
    │  STEP 2: Compute Tile T from LDS (current buffer)    │ ← Overlaps with Step 1!
    │          v_dot4 chains on previously-loaded data      │
    │                                                       │
    │  __syncthreads()                                      │
    │                                                       │
    │  STEP 3: Write staging regs → LDS (next buffer)      │ ← s_waitcnt vmcnt(0) here
    │          (loads from Step 1 must be done by now)      │    (naturally, after compute)
    │                                                       │
    │  __syncthreads()                                      │
    │  Swap ping-pong: cur_buf = nxt_buf                    │
    └───────────────────────────────────────────────────────┘

FINAL TILE: Compute (no more loads needed)
```

### Key Tricks from GEMM V3

1. **Staging registers**: Global loads target dedicated `a_staging[]`/`b_staging` registers,
   not the compute registers. This decouples the load destination from the compute source.

2. **Load-before-compute**: Main loop body issues ALL loads for the next tile FIRST (Step 1),
   then computes on the CURRENT tile (Step 2). The loads overlap with the entire compute phase.

3. **Natural waitcnt placement**: `s_waitcnt vmcnt(0)` occurs in Step 3 (writing staging regs
   to LDS), well after the compute phase. By this point, loads have had the entire compute
   phase (~hundreds of cycles) to complete.

4. **Cooperative loads**: All 256 threads cooperate on loading A/B tiles, maximizing bandwidth.

5. **LDS double buffer**: Ping-pong between two LDS buffers allows overlapping load→LDS writes
   for tile T+1 with compute from LDS reads for tile T.

---

## Optimization #1: Software-Pipelined K-Loop

**The primary optimization.** Apply the GEMM V3's load-before-compute pattern to the GEMV kernel.

### Current Structure (Serialized)

```
for each block b:
    WAIT for block b loads              ← STALL ~100-300 cycles
    DECODE block b → packed_groups[8]   ← 120 cycles, memory idle
    8× sdot4 accumulate                 ← 8 cycles
    ISSUE loads for block b+1           ← loads in flight for ~4 cycles before WAIT
```

### Proposed Structure (Pipelined)

```
PRELOAD: issue loads for block 0

for b = 0 to end-2:
    WAIT for block b loads              ← loads have been flying for ~130+ cycles!
    ISSUE loads for block b+1           ← fire-and-forget, async
    DECODE block b → packed_groups[8]   ← 120 cycles (hides b+1 load latency!)
    8× sdot4 accumulate                 ← 8 cycles

FINAL BLOCK (b = end-1):
    WAIT for block end-1 loads
    DECODE → packed_groups[8]
    8× sdot4 accumulate
```

### Why This Works

The decode phase is ~120 ALU instructions taking ~120 cycles. Memory latency for global loads on
gfx906 is ~100-400 cycles (L2 hit: ~100 cy, HBM: ~300-400 cy). With pipelining:

| Phase | Cycles | Overlap |
|-------|--------|---------|
| WAIT for block b | ~0 (loads already done) | — |
| ISSUE block b+1 loads | ~2 | Loads start flying |
| DECODE block b | ~120 | **Block b+1 loads in flight!** |
| 8× sdot4 | ~8 | Still in flight |
| **Total per iteration** | ~130 | ~130 cycles of load overlap |

With 4 waves/SIMD, the GPU can further hide latency by switching between wavefronts:
- 4 waves × 130 cycles = 520 cycles of useful work per iteration
- Even HBM latency of 400 cycles is fully hidden

### Source Code Change

The key change in `ROCmGemvKernel_native_VNNI.hip`:

```cpp
// Current: load + decode + compute are interleaved within each column/block
for (int b = b_start; b < b_end; ++b) {
    for (int c = 0; c < CPT; ++c) {
        const uint8_t* payload = d_payload + linear * Traits::payload_bytes;  // implicit load
        // ... decode payload to packed_groups ...
        // ... sdot4 accumulate ...
    }
}

// Proposed: separate load staging from decode
// Use staging buffers to preload next block while decoding current
uint8_t payload_staging[2][CPT][Traits::payload_bytes];  // double-buffer
uint16_t scale_staging[2][CPT];
int cur = 0, nxt = 1;

// Preload block 0 into staging[0]
for (int c = 0; c < CPT; ++c) {
    load_payload(b_start, c, payload_staging[0][c], scale_staging[0][c]);
}

for (int b = b_start; b < b_end; ++b) {
    // Issue loads for NEXT block (if not last)
    if (b + 1 < b_end) {
        for (int c = 0; c < CPT; ++c)
            load_payload(b + 1, c, payload_staging[nxt][c], scale_staging[nxt][c]);
    }
    
    // Decode CURRENT block from staging (overlaps with loads above)
    for (int c = 0; c < CPT; ++c) {
        decode_to_packed_groups(payload_staging[cur][c], packed_groups);
        sdot4_accumulate(a4, packed_groups, f_acc[c], scale_staging[cur][c]);
    }
    
    // Swap buffers
    cur ^= 1; nxt ^= 1;
}
```

**VGPR impact**: The staging buffers need ~20-30 additional VGPRs. Current usage:
- Q4_0: 29 VGPRs → ~55 VGPRs (still 4 waves, limit is 64)
- Q4_1: 33 VGPRs → ~55 VGPRs (still 4 waves)
- IQ4_NL: 55 VGPRs → may push to ~70 VGPRs (drops to 3 waves)

For IQ4_NL, the VGPR pressure from LUT staging may drop occupancy. This should be combined
with Optimization #2 (LDS LUT migration) which reduces VGPRs by eliminating LUT register usage.

### Compiler Hint: `__builtin_amdgcn_s_waitcnt`

To prevent the compiler from inserting eager `s_waitcnt vmcnt(0)` that would defeat the
pipelining, use explicit waitcnt hints:

```cpp
// Force the compiler to NOT insert waitcnt before decode
// This is the HIP equivalent of __asm__ volatile("s_nop 0") but for memory ordering
__builtin_amdgcn_s_waitcnt(/*vmcnt*/0x3f | /*lgkmcnt*/0x0);  // wait only for scalar loads
```

Alternatively, restructure using `volatile` pointer accesses or intermediate `__asm__` fences
to control when the compiler inserts synchronization.

---

## Optimization #2: LDS LUT Migration

**Target formats**: IQ4_NL (16-byte codebook), IQ3_S/IQ3_XXS (2 KB grid), IQ2_S/IQ2_XS/IQ2_XXS
(4 KB grid), IQ1_S/IQ1_M (16 KB grid).

### IQ4_NL: Scalar Register LUT (Optimal)

The IQ4_NL codebook is only 16 bytes (16 entries × 1 byte). On gfx906, this fits in 4 scalar
registers. Using `v_perm_b32` (which can extract any byte from a pair of 32-bit registers),
we can implement the LUT lookup entirely in registers:

```cpp
// Load 16-byte codebook into 4 scalar registers at kernel start
// s_cb[0] = codebook[0:3],  s_cb[1] = codebook[4:7], etc.
uint32_t cb[4];  // codebook in registers
cb[0] = *(uint32_t*)(k_iq4_codebook_i8 + 0);
cb[1] = *(uint32_t*)(k_iq4_codebook_i8 + 4);
cb[2] = *(uint32_t*)(k_iq4_codebook_i8 + 8);
cb[3] = *(uint32_t*)(k_iq4_codebook_i8 + 12);

// Lookup: select byte from cb[nibble/4] at position nibble%4
// This replaces global_load_ubyte with register-only operations
int8_t value = extract_byte(cb[nibble >> 2], nibble & 3);
```

**Expected impact**: Eliminates all 68 `global_load_ubyte` instructions. Replaces with ~68
`v_bfe_u32` + `v_perm_b32` ALU instructions (~68 cycles vs 6,800 cycles of memory latency).
That's a **100× reduction** in the LUT access cost.

### IQ Grid Formats: LDS Preload

For IQ3/IQ2 grids (2-4 KB), use cooperative LDS preload at kernel start:

```cpp
__shared__ uint32_t s_iq3s_grid[512];   // 2 KB in LDS
__shared__ uint64_t s_iq2s_grid[512];   // 4 KB in LDS

// Cooperative load at kernel start
for (int i = threadIdx.x; i < 512; i += blockDim.x)
    s_iq3s_grid[i] = iq3s_grid[i];
__syncthreads();

// Replace global_load_dword with ds_read_b32 in the hot loop
const uint32_t grid4 = s_iq3s_grid[idx];  // ~4 cycles vs ~100 cycles
```

**LDS budget**: 64 KB per CU on gfx906. With 4 KB of grid table + no other LDS usage,
this leaves 60 KB available — well within budget.

**Expected impact for IQ3_S**: 23 `global_load_dword` × (100-4) = **2,208 cycles saved per
K-loop iteration**. With ~28 iterations (896/32), total savings: ~61,824 cycles per GEMV.

---

## Optimization #3: Q4_0 Compiler Codegen Anomaly

Q4_0 compiles to 2.7× more ISA than Q4_1 despite near-identical source. Potential fixes:

### Approach A: Disable Unrolling

```cpp
#pragma clang loop unroll(disable)
for (int b = b_start; b < b_end; ++b) {
    // ... existing Q4_0 decode loop ...
}
```

This may produce a compact single-path loop similar to Q4_1.

### Approach B: Provide Trip Count Hints

```cpp
__builtin_assume(b_end - b_start >= 1);
__builtin_assume(b_end - b_start <= 128);  // K=4096, blocks_per_row=128
```

### Approach C: Restructure Source to Match Q4_1 Pattern

Q4_1's source uses `center_4bit = 0` (no subtraction), which the compiler may optimize better.
For Q4_0, the `-8` bias can be folded into the accumulation:

```cpp
// Current: w = nibble - 8 → packed into INT8 → sdot4
// Proposed: w = nibble → packed into INT8 → sdot4 → subtract bias correction
//   bias_correction = 8 * Σ|a_i| computed once per block
```

This eliminates 32 `v_add_u16` instructions from the decode path, making Q4_0 structurally
identical to Q4_1.

---

## Optimization Priority Matrix

| # | Optimization | Target Formats | Effort | Impact | Risk |
|---|-------------|----------------|--------|--------|------|
| 1 | Software-pipelined K-loop | ALL 15 formats | High | High (~2-3×) | Medium (VGPR pressure) |
| 2a | Scalar reg LUT for IQ4_NL | IQ4_NL | Low | Very High (100×) | Low |
| 2b | LDS grid preload | IQ3_S/XXS, IQ2_S/XS/XXS, IQ1 | Medium | High (25×) | Low |
| 3 | Q4_0 codegen fix | Q4_0 | Low | Medium (up to 2.7×) | Low |

**Recommended order**: 2a → 3 → 1 → 2b

Rationale:
- **2a** (IQ4_NL scalar LUT) is a small, isolated change with massive impact
- **3** (Q4_0 codegen) is quick to try — just add pragma/builtin hints
- **1** (K-loop pipelining) is the big systemic win but requires careful restructuring
- **2b** (LDS grid) is straightforward but only helps grid-based formats

---

## Theoretical Speedup Estimates

### With Software-Pipelined K-Loop Only

| Format | Current cy/iter | Pipelined cy/iter | Stall Eliminated | Speedup |
|--------|-----------------|-------------------|------------------|---------|
| Q4_0   | ~343 (200 stall + 120 decode + 15 other + 8 dot4) | ~145 (0 stall + 120 decode + 2 load + 15 other + 8 dot4) | 200 cy | ~2.4× |
| Q4_1   | ~207 (100 stall + 79 decode + 12 other + 16 dot4) | ~109 (0 stall + 79 decode + 2 load + 12 other + 16 dot4) | 100 cy | ~1.9× |
| Q5_0   | ~408 (200 stall + 180 decode + 12 other + 16 dot4) | ~210 (0 stall + 180 decode + 2 load + 12 other + 16 dot4) | 200 cy | ~1.9× |
| Q6_K   | ~542 (300 stall + 226 decode + 8 other + 8 dot4) | ~244 (0 stall + 226 decode + 2 load + 8 other + 8 dot4) | 300 cy | ~2.2× |
| IQ4_NL | ~285 (0 stall* + 135 decode + 68 LUT_load + 68 pack + 8 dot4) | ~148 (LDS+pipe) | varies | ~1.9× |

*IQ4_NL's "stall" is distributed across 68 progressive waitcnts, not a single hard stall.

### With All Optimizations Combined

The theoretical best case for native-VNNI vs INT8 is determined by the bandwidth ratio:

| Format | BPW | Bytes/block | Bandwidth Savings vs INT8 | Theoretical Max Speedup |
|--------|-----|-------------|--------------------------|------------------------|
| Q4_0   | 4.5 | 18 B        | 1.78×                    | 1.78× (if decode-free) |
| Q4_1   | 5.0 | 20 B        | 1.60×                    | 1.60×                  |
| Q3_K   | 3.4 | 14 B        | 2.29×                    | 2.29×                  |
| Q2_K   | 2.6 | 12 B        | 3.08×                    | 3.08×                  |
| IQ4_NL | 4.5 | 18 B        | 1.78×                    | 1.78×                  |
| IQ2_XXS| 2.1 | 10 B        | 3.81×                    | 3.81×                  |
| IQ1_S  | 1.6 | 8 B         | 5.00×                    | 5.00×                  |

**Reality check**: These theoretical maxima assume the kernel becomes fully memory-bandwidth-bound
after optimizations. In practice, even with perfect pipelining, some decode ALU overhead remains.
A realistic target for the common Q4_0 case is **1.3-1.5× vs INT8** after all optimizations.

For very low BPW formats (IQ2, IQ1), the bandwidth savings are more dramatic, and a
**2.0-3.0× improvement over INT8** may be achievable.

---

## Appendix: ISA Toolchain Quick Reference

```bash
# List GPU code objects in a binary
roc-obj-ls <binary>

# Extract code object (use offset/size from roc-obj-ls)
dd if=<binary> of=kernel.elf bs=1 skip=<offset> count=<size>

# Full disassembly
llvm-objdump -d --mcpu=gfx906 kernel.elf > disasm.s

# Resource metadata (VGPRs, SGPRs, LDS, spills)
llvm-readelf --notes kernel.elf | grep -A5 'amdhsa.kernels'

# Instruction census (e.g., count v_dot4 occurrences)
grep -c 'v_dot4_i32_i8' disasm.s

# Quick loop analysis
grep -n 'v_dot4\|s_cbranch\|s_waitcnt vmcnt\|global_load' disasm.s | head -100
```
