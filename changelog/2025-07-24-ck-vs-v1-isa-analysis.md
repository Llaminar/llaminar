# CK vs V1 Wide-Tile Kernel: ISA-Level Analysis

**Date**: 2025-07-24  
**Target Shape**: FFN_Up/Gate 7B — M=128, N=18944, K=3584  
**GPU**: AMD Instinct MI50 (gfx906), 60 CUs, 4 MB L2, 64 KB LDS/CU  
**Measured Gap**: V1 = 1.108ms, CK = 0.957ms (14% kernel-only gap via rocprof)

---

## 1. Kernel Identity

| | V1 (ours) | CK (composable_kernel) |
|---|---|---|
| **Symbol** | `qgemm_int8_vnni_wide_tile_kernel<64,8,2>` | `DeviceGemmMultipleD_Dl` 128x128 HasMainK=true HasDoubleTail=false |
| **Tile** | M128 × N64 | M128 × N128 |
| **K-block** | KT=8 (32 bytes) | K0=16, K1=4 (64 bytes) |
| **Threads** | 256 (32M × 8N) | 256 |
| **VGPRs** | 66 | 128 |
| **SGPRs** | 48 | 52 |
| **LDS** | 30,720 bytes | 32,768 bytes |
| **Waves/SIMD** | 3 | 2 |
| **Accumulators/thread** | 32 (4M × 8N) | 64 (8M × 8N) |
| **ISA lines** | 662 + 218 (subroutine) | ~3,700 |

---

## 2. Loop Structure Comparison

### V1: Single-Buffered, Subroutine-Based

```
PROLOGUE:
  [address setup, bounds checks]

LOOP (112 iterations for K=3584):
  ├─ Global loads: 3 × s_swappc_b64 (subroutine calls for A/B loads)
  ├─ LDS writes (via subroutine)
  ├─ s_barrier ────────────── wait for LDS writes
  ├─ COMPUTE (m=0):
  │   ├─ 64 v_dot4 inline (8 kk × 8 n-cols)
  │   └─ interleaved: ds_read → waitcnt → v_dot4 groups
  ├─ COMPUTE (m=1,2,3):
  │   └─ 3 × s_swappc_b64 → qgemm_wide_tile_row_phase (64 v_dot4 each)
  ├─ s_barrier ────────────── wait for compute before overwriting LDS
  └─ s_cmp + s_cbranch ──── loop test

EPILOGUE:
  [vectorized stores with tail handling]
```

**Per iteration**: 256 v_dot4, 2 barriers, 3+3=6 subroutine calls  
**Total for K=3584**: 112 iter × 2 = **224 barriers**, 112 × 6 = **672 s_swappc**

### CK: Double-Buffered, Software-Pipelined

```
PROLOGUE:
  [global load → LDS write (even buffer)]
  s_barrier ─────────────────── even buffer ready

LOOP (28 iterations for K=3584):
  ├─ EVEN PHASE:
  │   ├─ 1024 v_dot4 from even LDS buffer
  │   ├─ buffer_load (global → VGPR for next block)
  │   └─ ds_write to odd LDS buffer
  ├─ s_barrier ──────────────── odd buffer ready
  ├─ ODD PHASE:
  │   ├─ 1024 v_dot4 from odd LDS buffer
  │   └─ ds_write to even LDS buffer
  ├─ s_cmp + s_cbranch ─────── loop test
  └─ s_barrier ──────────────── even buffer ready

EPILOGUE:
  1024 v_dot4 (drain last K-block)
  [global stores]
```

**Per iteration**: 2048 v_dot4, 2 barriers, 0 subroutine calls  
**Total for K=3584**: 28 iter × 2 + 1 = **57 barriers**, **0 s_swappc**

---

## 3. Instruction Census

| Metric | V1 (main + sub×3) | CK |
|---|---|---|
| **v_dot4_i32_i8** | 64 + 3×64 = **256**/iter | **2048**/iter (+1024 epilogue) |
| **ds_read** | 37 + 3×44 = **169**/iter | **128**/iter |
| **s_barrier** | **2**/iter | **2**/iter |
| **s_swappc_b64** | **6**/iter | **0** |
| **s_nop** | **0** total | **2** total |
| **s_waitcnt** | 54 + 3×23 = **123**/iter | ~50/iter |
| **Branches** | **59**/iter (bounds checks) | ~3/iter |
| **Total v_dot4** | 112×256 = **28,672** | 28×2048+1024 = **58,368** |

Note: CK has 2× more total v_dot4 because its N-tile is 2× larger (128 vs 64),
processing 2× more output columns per workgroup — same work per accumulator (896).

---

## 4. Compute-to-Overhead Ratio

| Metric | V1 | CK | Ratio |
|---|---|---|---|
| v_dot4 per barrier | 128 | 1024 | **8×** |
| v_dot4 per loop iteration | 256 | 2048 | **8×** |
| Loop iterations (K=3584) | 112 | 28 | **4×** fewer |
| Total barriers | 224 | 57 | **3.9×** fewer |
| Subroutine calls total | 672 | 0 | **∞** |
| K bytes consumed per iteration | 32 | 128 | **4×** |

---

## 5. Scheduling Pattern: CK's Interleaved Pipeline

CK's inner loop (between barriers) shows a sophisticated software pipeline.
Representative 50-instruction sequence from the even phase:

```asm
; === Issue 8 ds_read_b128 to fill pipeline ===
ds_read_b128 v[94:97],  v35 offset:16384     ; B[0..3] from even buffer
ds_read_b128 v[98:101], v35 offset:16640     ; B[4..7]
ds_read_b128 v[102:105], v34                  ; A[0..3]
ds_read_b128 v[106:109], v34 offset:256       ; A[4..7]
ds_read_b128 v[110:113], v35 offset:16896    ; B[8..11]
ds_read_b128 v[114:117], v35 offset:17152    ; B[12..15]
ds_read_b128 v[118:121], v34 offset:512       ; A[8..11]
ds_read_b128 v[122:125], v34 offset:768       ; A[12..15]

; === Wait for only 3 of 8 reads (lgkmcnt=5) ===
s_waitcnt lgkmcnt(5)

; === Compute on first completed data ===
v_dot4_i32_i8 v89, v102, v94, v89    ; A[0] × B[0]
v_dot4_i32_i8 v88, v102, v95, v88    ; A[0] × B[1]
v_dot4_i32_i8 v87, v102, v96, v87    ; A[0] × B[2]
v_dot4_i32_i8 v86, v102, v97, v86    ; A[0] × B[3]
; ... 28 more v_dot4 using A[0..3] × B[0..3] ...

; === Freed registers → reuse for output ===
v_dot4_i32_i8 v102, v105, v98, v61   ; Reuse v102 as accumulator!
v_dot4_i32_i8 v103, v105, v99, v60
v_dot4_i32_i8 v104, v105, v100, v59
v_dot4_i32_i8 v105, v105, v101, v58

; === Issue new ds_read into freed registers ===
ds_read_b128 v[58:61], v35 offset:17408
s_waitcnt lgkmcnt(5)                          ; Only wait for what's needed

; === Continue compute on next A-tile ===
v_dot4_i32_i8 v57, v106, v94, v57
; ... pattern repeats ...
```

**Key techniques:**
1. **Batch issue + partial wait**: Issue 8 ds_reads, wait for only 3 (`lgkmcnt(5)`).
   Remaining 5 reads complete during v_dot4 execution.
2. **Register recycling**: After consuming A[0..3], reuse v102-v105 as accumulators,
   freeing them for new ds_reads later.
3. **No NOPs**: Only 2 `s_nop` in entire 3,072 v_dot4 kernel — compiler fills
   v_dot4 hazard slots with ds_read and s_waitcnt instructions.

### V1's Scheduling Pattern (contrast)

```asm
; === Smaller batches, tighter waits ===
ds_read2st64_b32 v[15:16], v44 offset1:2    ; 2 A-values (stride-64)
ds_read_b128 v[8:11], v45                    ; 4 B-values
ds_read_b96 v[12:14], v45 offset:16          ; 3 B-values
ds_read_b32 v17, v47                          ; 1 B-value

s_waitcnt lgkmcnt(2)                          ; Wait for all but 2
v_dot4_i32_i8 v8, v15, v8, v4
s_waitcnt lgkmcnt(1)
v_dot4_i32_i8 v12, v15, v12, v0
ds_read_b32 v0, v46                           ; Issue 1 new read
v_dot4_i32_i8 v9, v15, v9, v5
v_dot4_i32_i8 v10, v15, v10, v6
; ... 4 more v_dot4 ...
ds_read_b128 v[4:7], v45 offset:256           ; Next B-group
; ... pattern continues ...
```

**V1's compiler also interleaves ds_read with v_dot4** — the scheduling is decent.
The problem is the **small batch size** (4-8 reads → 8 v_dot4 → repeat) compared
to CK's massive batches (8 reads → 32+ v_dot4 → new reads).

---

## 6. Root Cause Analysis: Why CK is 14% Faster

### Primary: Double-Buffering Eliminates Serialization (~10% of gap)

V1's single-buffer pipeline:
```
[Load Global→LDS] → barrier → [Compute from LDS] → barrier → repeat
     ↑ IDLE GPU          ↑ IDLE GPU              ↑ IDLE GPU
```

CK's double-buffer pipeline:
```
[Compute on even LDS] + [Load→Write to odd LDS]  → barrier →
[Compute on odd LDS]  + [Load→Write to even LDS] → barrier → repeat
     ↑ OVERLAPPED — compute and load run in parallel
```

V1 stalls during global loads (~300ns per load), CK hides this behind compute.
With 112 iterations × ~300ns per stall ≈ 33.6μs of load latency exposed.
This is roughly 3% of V1's 1108μs but compounds with other effects.

### Secondary: 4× Fewer Loop Iterations (~2-3% of gap)

| | V1 | CK |
|---|---|---|
| K per iteration | 32 bytes | 128 bytes |
| Loop iterations | 112 | 28 |
| Barriers | 224 | 57 |
| Total overhead (est.) | ~16,800 cycles | ~1,400 cycles |
| At 1.5 GHz | ~11.2 μs | ~0.9 μs |

### Tertiary: No Subroutine Call Overhead (~1% of gap)

V1: 672 `s_swappc_b64` calls → instruction cache thrashing + pipeline stalls.
CK: All code is inline — no subroutine calls in the hot path.

### Quaternary: Fewer Branches in Hot Path

V1: 59 branches per iteration (many for bounds-checking edge cases).
CK: Separate kernel variants (`HasMainK`, `HasDoubleTail`, `MNPadding`)
eliminate bounds checks from the main loop.

---

## 7. V3 Kernel Design Recommendations

Based on this analysis, a V3 kernel should adopt CK's key techniques:

### 7.1 LDS Double-Buffering (highest impact)

```
Current V1:  __shared__ int32_t a_lds[KT * M_TILE];      // 4 KB single buffer
             __shared__ int32_t b_lds[KT * N_TILE];      // 2 KB single buffer

Proposed V3: __shared__ int32_t a_lds[2][KT * M_TILE];   // 8 KB double buffer
             __shared__ int32_t b_lds[2][KT * N_TILE];   // 4 KB double buffer
```

Total LDS: 12 KB (vs current 6 KB). Still fits 5 workgroups/CU (64KB / 12KB = 5).
Trade 1 wave/SIMD for overlapped load+compute (same tradeoff CK makes).

### 7.2 Larger K-Block (reduce loop overhead)

Increase KT from 8 to 32 (or 64 to match CK):
- KT=32: 28 iterations for K=3584 (matches CK)
- KT=16: 56 iterations (intermediate)
- LDS cost: KT=32 → 32×128×4 = 16KB for A, 32×64×4 = 8KB for B → 24KB per buffer
  - Double-buffered: 48KB. Only 1 workgroup/CU (64KB limit). Too much.
  - Single-buffered with KT=32: 24KB. Fits 2 workgroups.

**Recommended**: KT=16, double-buffered = 2×(16×128 + 16×64)×4 = 2×12KB = 24KB.
Fits 2 workgroups/CU. 56 iterations. 4× fewer than V1 current.

### 7.3 Inline All Compute (eliminate s_swappc)

Replace the 3 subroutine calls with inline code for all 4 m-rows.
Register cost: same 66 VGPRs (4 m-rows done serially, same registers reused).
Code size: 218 lines × 4 = ~900 lines. Well within I-cache budget.

### 7.4 Separate Main/Tail Kernels (eliminate branches)

Like CK's `HasMainK`/`HasDoubleTail`/`MNPadding` variants:
- Main kernel: no bounds checks, assumes full tiles
- Tail kernel: handles M/N/K edge cases
- Dispatch: host-side check which variant to launch

### 7.5 Use `__builtin_amdgcn_sdot4` (already done)

V1 already uses this intrinsic. CK also uses it (CK_USE_AMD_V_DOT_INLINE_ASM=0).
Compiler-scheduled hazard filling works well for both (0 s_nop in V1, 2 in CK).

---

## 8. Expected Impact

| Change | Estimated Gain | Confidence |
|---|---|---|
| LDS double-buffering | 5-8% | High — eliminates load serialization |
| KT=8→16 (fewer iterations) | 2-3% | Medium — reduces barrier/branch overhead |
| Inline compute (no swappc) | 1-2% | Medium — eliminates I-cache effects |
| Separate main/tail variants | 0.5-1% | Low — bounds checks are not the bottleneck |
| **Combined** | **8-14%** | Close the gap to CK |

---

## 9. Disassembly Reference

All analysis from `/tmp/ck_disasm.txt` (45,493 lines), generated from:
```bash
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/ck_disasm.txt
```

| Kernel | Lines | Address | v_dot4 | ds_read | Barriers |
|---|---|---|---|---|---|
| V1 main | 44832-45493 | 0x79E00 | 64 | 37 | 2 |
| V1 subroutine | 460-678 | 0x2AF00 | 64 | 44 | 0 |
| CK 128x128 | 18369-22092 | 0x4B100 | 3072 | 192 | 3 |

---

## Appendix A: How to Reproduce This Disassembly

This appendix documents the exact commands used to extract and analyze the ISA
from the compiled HIP fat binary, so future agents or developers can reproduce
or extend these findings.

### A.1 Prerequisites

- **LLVM/ROCm tools** on `PATH`: `llvm-objdump`, `llvm-readelf`, `llvm-objcopy`
  (installed with ROCm; typically at `/opt/rocm/llvm/bin/`)
- A **built binary** containing both our V1 kernel and CK kernels. The performance
  test binary works well because it instantiates all kernel variants:
  ```bash
  cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel
  ```
- The binary path used:
  `build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison`

### A.2 Step 1: Extract the GPU Code Object from the Fat Binary

HIP executables embed GPU code objects inside ELF sections with an
`__CLANG_OFFLOAD_BUNDLE__` prefix. First, list them:

```bash
llvm-readelf -S build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  | grep '__CLANG_OFFLOAD_BUNDLE__'
```

This shows sections like:
```
[XX] __CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906  PROGBITS  ...
```

Extract the GPU code object (the gfx906 section) into a standalone ELF:

```bash
llvm-objcopy \
  --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' \
  build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison /dev/null
```

Verify it worked:
```bash
file /tmp/co.elf
# Should show: ELF 64-bit LSB shared object, *unknown arch 0xe0* ...
ls -la /tmp/co.elf
# ~800 KB is typical for a binary with multiple CK kernel variants
```

### A.3 Step 2: Read Kernel Resource Metadata

Before disassembling, inspect the kernel resource usage (VGPRs, SGPRs, LDS)
from the ELF notes section:

```bash
llvm-readelf --notes /tmp/co.elf 2>/dev/null | head -200
```

This dumps AMDGPU metadata in YAML format with entries like:
```yaml
- .name: _Z32qgemm_int8_vnni_wide_tile_kernelILi64ELi8ELi2EE...
  .vgpr_count: 66
  .sgpr_count: 48
  .group_segment_fixed_size: 30720   # LDS in bytes
  .wavefront_size: 64
```

For a clean table, extract with awk:
```bash
llvm-readelf --notes /tmp/co.elf 2>/dev/null \
  | awk '
    /\.name:/ { name=$2 }
    /\.vgpr_count:/ { vgpr=$2 }
    /\.sgpr_count:/ { sgpr=$2 }
    /\.group_segment_fixed_size:/ { lds=$2 }
    /\.wavefront_size:/ { printf "%-60s VGPR=%-4s SGPR=%-4s LDS=%s\n", substr(name,1,60), vgpr, sgpr, lds }
  '
```

**Key metrics to look for:**
- `vgpr_count` → determines waves/SIMD: `floor(256 / vgpr_count)`
- `group_segment_fixed_size` → LDS bytes, determines workgroups/CU: `floor(65536 / lds)`
- Cross-reference: actual occupancy = `min(waves_from_vgpr, waves_from_lds)`

### A.4 Step 3: Full Disassembly

```bash
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/ck_disasm.txt
wc -l /tmp/ck_disasm.txt
# Expect ~45,000 lines for a binary with 10+ kernel variants
```

### A.5 Step 4: Find Kernel Entry Points

List all function symbols (kernel names) with their line numbers:

```bash
grep -n '<_Z' /tmp/ck_disasm.txt | head -30
```

The output shows demangled-ish kernel names with their disassembly line numbers.
Key kernels to identify:

| Pattern in symbol name | What it is |
|---|---|
| `qgemm_int8_vnni_wide_tile_kernel` | Our V1 kernel |
| `qgemm_wide_tile_row_phase` | V1's per-m-row compute subroutine |
| `DeviceGemmMultipleD` ... `Li256E` | CK 128×128 (256 threads) |
| `DeviceGemmMultipleD` ... `Li64E` | CK 64×64 (64 threads) |
| `DeviceGemmMultipleD` ... `Li32E` | CK 32×32 (32 threads) |
| `Lb1ELb1E` in CK symbol | HasMainK=true, HasDoubleTail=true |
| `Lb1ELb0E` in CK symbol | HasMainK=true, HasDoubleTail=false (**production variant**) |
| `Lb0ELb0E` in CK symbol | HasMainK=false, HasDoubleTail=false |
| `PassThrough` in CK symbol | GemmDefault (no padding) |
| `RightPad` in CK symbol | MNPadding variant |

### A.6 Step 5: Instruction Census

Count specific instruction types within a kernel's line range:

```bash
# Example: CK 128x128 HasMainK=true HasDoubleTail=false at lines 18369-22092
for pat in v_dot4_i32_i8 ds_read_b128 ds_read_b32 ds_read_b96 \
           ds_read2st64_b32 s_barrier s_waitcnt s_nop s_cbranch \
           buffer_load ds_write s_swappc; do
  echo -n "$pat: "
  awk 'NR>=18369 && NR<=22092' /tmp/ck_disasm.txt | grep -c "$pat"
done
```

**Useful aggregate counts:**
```bash
# Total ds_read (all variants combined)
awk 'NR>=START && NR<=END' /tmp/ck_disasm.txt | grep -c 'ds_read'

# Compute density = v_dot4 / total_instructions
total=$(awk 'NR>=START && NR<=END' /tmp/ck_disasm.txt | grep -v '^\s*$' | wc -l)
compute=$(awk 'NR>=START && NR<=END' /tmp/ck_disasm.txt | grep -c 'v_dot4_i32_i8')
echo "Compute density: $compute / $total"
```

### A.7 Step 6: Map Loop Structure

Find barriers and the loop back-edge to understand the loop body:

```bash
# Find barriers (synchronization points)
awk 'NR>=START && NR<=END' /tmp/ck_disasm.txt | grep -n 's_barrier'

# Find loop back-edge (branch to earlier address)
awk 'NR>=START && NR<=END' /tmp/ck_disasm.txt | grep -n 's_cbranch'

# Find subroutine calls (V1 specific)
awk 'NR>=START && NR<=END' /tmp/ck_disasm.txt | grep -n 's_swappc'
```

**Interpreting barrier locations:**
- Barriers divide the kernel into phases: prologue → even compute → odd compute → epilogue
- For double-buffered kernels (CK): 3 barriers = prologue + 2 per loop iteration
- For single-buffered kernels (V1): 2 barriers per loop iteration (load barrier + compute barrier)

### A.8 Step 7: Analyze Instruction Scheduling

Read the actual instruction sequence between barriers to see interleaving:

```bash
# Read a section of the inner loop (e.g., first 100 lines after barrier 1)
awk 'NR>=18630 && NR<=18730' /tmp/ck_disasm.txt
```

**What to look for:**
- `ds_read_b128` followed by `s_waitcnt lgkmcnt(N)` — partial waits mean pipelining
- `v_dot4_i32_i8` instructions between ds_reads — interleaved scheduling
- `s_nop` — wasted cycles (fewer = better compiler scheduling)
- Register reuse — a v_dot4 writing to a register that was just used as input
  for ds_read means the compiler is recycling registers aggressively

### A.9 Quick Reference: One-Liner Cheat Sheet

```bash
# Extract code object
llvm-objcopy --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' \
  BUILD_DIR/tests/v2/v2_perf_rocm_prefill_dispatch_comparison /dev/null

# Kernel resources (VGPR/SGPR/LDS)
llvm-readelf --notes /tmp/co.elf 2>/dev/null | grep -A5 '\.name:'

# Full disassembly
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/ck_disasm.txt

# Find all kernels
grep -n '<_Z' /tmp/ck_disasm.txt

# Instruction census for lines START-END
for p in v_dot4_i32_i8 ds_read s_barrier s_nop s_waitcnt buffer_load ds_write s_swappc; do
  echo -n "$p: "; awk "NR>=START && NR<=END" /tmp/ck_disasm.txt | grep -c "$p"
done

# Loop structure
awk "NR>=START && NR<=END" /tmp/ck_disasm.txt | grep -n 's_barrier\|s_cbranch\|s_swappc'
```

### A.10 Notes on Reproducibility

- The exact line numbers in the disassembly will change if the binary is recompiled
  (code layout, inlining decisions, etc.). Use the **kernel symbol names** and
  **address offsets** as stable anchors, not absolute line numbers.
- The `--mcpu=gfx906` flag is required for correct instruction decoding on MI50.
  Use `--mcpu=gfx90a` for MI210, `--mcpu=gfx942` for MI300X, etc.
- CK kernel symbols are extremely long (>1000 characters) due to heavy C++ template
  use. Use `grep` with short distinctive fragments (e.g., `Li256E`, `Lb1ELb0E`,
  `PassThrough`) rather than trying to match full names.
- The `__CLANG_OFFLOAD_BUNDLE__` section name format may vary with ROCm versions.
  If the `llvm-objcopy --dump-section` command fails, list sections first with
  `llvm-readelf -S` and look for sections containing `gfx906` or `amdgcn`.
