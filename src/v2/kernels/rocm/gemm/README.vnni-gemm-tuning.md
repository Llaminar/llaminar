# VNNI INT8 GEMM Kernel Tuning Guide — AMD MI50/MI60 (gfx906)

This document summarizes the multi-session tuning process that produced V7, a native HIP INT8 GEMM kernel that matches or exceeds AMD's Composable Kernel (CK) library on N-heavy LLM inference shapes. It covers the tools, process, architectural constraints, ISA analysis techniques, and latency-hiding strategies developed along the way.

---

## Table of Contents

- [Target Workload](#target-workload)
- [MI50 Architecture Reference](#mi50-architecture-reference)
- [CK Library Data Mining](#ck-library-data-mining)
- [Kernel Version History (V1–V7)](#kernel-version-history-v1v7)
- [ISA Analysis Toolchain](#isa-analysis-toolchain)
- [Profiling: rocprof Per-Dispatch Analysis](#profiling-rocprof-per-dispatch-analysis)
- [Latency Hiding and Interleaving Techniques](#latency-hiding-and-interleaving-techniques)
- [V7: The Winning Design](#v7-the-winning-design)
- [Final Benchmark Results](#final-benchmark-results)
- [Key Lessons Learned](#key-lessons-learned)
- [Command Reference](#command-reference)

---

## Target Workload

**INT8×INT8→INT32 GEMM** for quantized LLM inference on AMD Instinct MI50/MI60 GPUs.

Primary target shapes (Qwen2.5 family):

| Shape | M | N | K | Description |
|-------|-----|-------|------|-------------|
| 0.5B FFN_Up | 128 | 4864 | 896 | Feedforward up-projection |
| 3B FFN_Up | 128 | 11008 | 2048 | Feedforward up-projection |
| 7B FFN_Up | 128 | 18944 | 3584 | Feedforward up-projection (primary benchmark) |
| 7B LM_Head | 128 | 152064 | 3584 | Language model head (extreme-wide) |
| 7B AttnOut | 128 | 3584 | 3584 | Attention output projection (K-heavy) |

The core intrinsic is `v_dot4_i32_i8` — a single-cycle 4-way INT8 dot product that accumulates into INT32. This is the gfx906 equivalent of Intel's VNNI `vpdpbusd` (hence the naming).

---

## MI50 Architecture Reference

### gfx906 Key Specifications

| Parameter | Value | Impact on GEMM |
|-----------|-------|----------------|
| Compute Units (CUs) | 60 | Grid sizing — blocks distributed across CUs |
| SIMDs per CU | 4 | Each SIMD executes one wavefront (64 threads) |
| VGPRs per SIMD | 256 | **Critical**: determines occupancy (waves/SIMD) |
| SGPRs per SIMD | 800 (102 per wave) | Rarely the bottleneck |
| LDS per CU | 64 KB | Shared across all workgroups on a CU |
| L1 Instruction Cache | 16 KB per CU | **Important**: constrains loop body ISA size |
| L2 Cache | 4 MB (shared) | Caches A matrix (128×K fit for K≤8192) |
| Wavefront size | 64 threads | One instruction operates on 64 lanes |
| Peak HBM bandwidth | 1024 GB/s (theoretical) | Measured: 378–416 GB/s D2D |
| `v_dot4_i32_i8` throughput | 1 per cycle per lane | Compute-bound for large enough tiles |

### Occupancy Model

The number of concurrent wavefronts per SIMD determines latency-hiding capability:

| VGPRs/wave | Waves/SIMD | Explanation |
|-----------|-----------|-------------|
| 1–64 | 4 | Maximum occupancy |
| 65–84 | 3 | `floor(256/VGPRs)` |
| 85–128 | 2 | **CK's operating point** |
| 129–256 | 1 | Insufficient latency hiding |

**Key tradeoff**: More accumulators → more VGPRs → fewer waves. For M128×N128 tiles with 64 accumulators, 2 waves/SIMD is the natural operating point. Forcing 1 wave (e.g., via register spills) is catastrophic — V4 demonstrated this with 256 VGPRs + 118 spills.

### LDS Allocation

LDS is partitioned across workgroups on the same CU:

| LDS per workgroup | Max workgroups/CU | Concurrent waves (×4 SIMDs) |
|-------------------|--------------------|-----------------------------|
| 16 KB | 4 | Up to 16 |
| 24 KB | 2 | Up to 8 |
| 32 KB | 2 | Up to 8 |
| 48 KB | 1 | Up to 4 |

For our N128 double-buffered kernels, LDS usage is 16–32 KB depending on KT, comfortably allowing 2+ workgroups per CU.

### Memory Hierarchy Latencies

| Access Type | Latency (cycles) | Bandwidth |
|-------------|-------------------|-----------|
| VGPR (register) | 0 | — |
| LDS | ~20–30 | ~12 TB/s aggregate |
| L1 data cache | ~80 | ~6 TB/s |
| L2 cache | ~200–300 | ~2 TB/s |
| HBM | ~400–600 | 378–416 GB/s measured |

These latencies explain why double-buffering and progressive waitcnt management are essential — a single LDS read stall of 25 cycles "wastes" 25 potential `v_dot4` instructions per lane.

---

## CK Library Data Mining

### What is CK?

AMD's [Composable Kernel](https://github.com/ROCm/composable_kernel) is a template-heavy C++ library that generates near-optimal GPU kernels by composing tile-level primitives. For gfx906 INT8 GEMM, it uses the `DeviceGemmMultipleD_Dl` template family ("Dl" = Deep Learning, using `v_dot4_i32_i8`).

### Template Variant Discovery

CK generates **multiple kernel variants** at compile time, selected at runtime based on problem shape:

| Symbol Pattern | Variant | Purpose |
|----------------|---------|---------|
| `Lb1ELb0E` | HasMainK=true, HasDoubleTail=false | **Production hot path** — fully aligned K tiles |
| `Lb1ELb1E` | HasMainK=true, HasDoubleTail=true | Handles K remainder with tail phase |
| `Lb0ELb0E` | HasMainK=false, HasDoubleTail=false | Small-K fallback |
| `PassThrough` | GemmDefault (no padding) | N,M aligned to tile size |
| `RightPad` | MNPadding | Handles partial M/N tiles at matrix edges |
| `Li256E` | 256 threads → 128×128 tile | N-heavy shapes (FFN_Up, LM_Head) |
| `Li64E` | 64 threads → 64×64 tile | Medium shapes |
| `Li32E` | 32 threads → 32×32 tile | Decode (M=1–32) |

**Key insight**: CK's `HasMainK` variant eliminates all K-boundary checks from the main loop. This is the same strategy V7 adopts via safe-tile splitting.

### CK ISA Analysis Summary

Disassembly of the CK 128×128 `HasMainK=true, HasDoubleTail=false` kernel revealed:

| Feature | CK Approach | Why It's Fast |
|---------|-------------|---------------|
| K-loop structure | Fully unrolled (28568 `v_dot4` total) | Zero loop overhead, maximal icache usage |
| LDS buffering | Double-buffered (even/odd) | Overlaps load with compute |
| Global loads | `buffer_load_dwordx4` (SRD-based) | Hardware bounds checking, no software branches |
| LDS reads | `ds_read_b128` | 16 bytes per read, maximizes throughput |
| LDS writes | `ds_write2_b64` (paired 8-byte) | Efficient for non-contiguous patterns |
| Waitcnt strategy | Progressive `lgkmcnt(5)` → compute → `lgkmcnt(3)` → ... | Always 5–6 reads in flight |
| Branching | ~3 branches per loop iteration | Minimal icache pollution |
| Subroutine calls | 0 (`s_swappc_b64`) | All code inline in loop body |
| Total `s_nop` | 2 (in entire kernel) | Near-perfect scheduling |

### How We Extracted CK ISA

See the [ISA Analysis Toolchain](#isa-analysis-toolchain) section for exact commands. The process was:

1. Build the benchmark binary (both CK and native kernels in one executable)
2. Extract the gfx906 code object from the HIP fat binary
3. Read kernel metadata (VGPRs, SGPRs, LDS) from ELF notes
4. Full disassembly with `llvm-objdump`
5. Locate CK kernels by symbol name patterns
6. Instruction census (count `v_dot4`, `ds_read`, `s_barrier`, etc.)
7. Map loop structure via barrier and branch locations
8. Analyze scheduling patterns between barriers

---

## Kernel Version History (V1–V7)

### V1: Baseline Wide-Tile (N64, Single-Buffered)

**Design**: M128×N64 tile, 256 threads, KT=8, single LDS buffer, subroutine-based compute.

**Resources**: 66 VGPRs, 48 SGPRs, 6 KB LDS → 3 waves/SIMD.

**Characteristics**:
- Each m-row computed via `s_swappc_b64` subroutine call (3 calls per iteration × 112 iterations = 672 total)
- 224 barriers for K=3584
- 59 branches per loop iteration (bounds checking)
- Load and compute fully serialized (no overlap)

**Performance**: 1.108ms on 7B FFN_Up (CK: 0.957ms, 14% gap in rocprof kernel-only timing).

**Key Problem**: No latency hiding. Load → barrier → compute → barrier is fully serial.

### V2: A from L2, B in LDS

**Design**: Removed A from LDS entirely. A matrix (128×K) fits in 4 MB L2 cache for K≤8192. B remains in LDS.

**Rationale**: Eliminates A LDS bank conflicts (4-way conflict in V1 due to stride-4 access pattern). Free 4 KB LDS for other waves.

**Result**: Similar performance to V1 — the bottleneck was load/compute serialization, not LDS bank conflicts.

### V3: LDS Double-Buffered Pipeline (N64)

**Design**: Adopted CK's primary technique — double-buffered LDS with software-pipelined main loop.

**Key changes**:
- `a_lds[2][KT×128]` + `b_lds[2][KT×64]` — ping-pong between buffers
- KT=16 support via multi-pass A cooperative loading
- Issue global loads → compute on current buffer → barrier → write to next buffer → barrier
- Load and compute now partially overlapped

**Resources**: ~66 VGPRs, 24 KB LDS (KT=16) → 3 waves/SIMD, 2 workgroups/CU.

**Result**: Measurably faster than V1/V2, but still 8–10% behind CK. The N64 tile covers fewer columns per block → more grid blocks → higher overhead.

### V4: N128 Double-Buffered (CK Tile Parity)

**Design**: Matched CK's M128×N128 tile geometry. 64 accumulators per thread (4M × 16N).

**Key insight**: N128 halves the grid block count vs N64 (148 vs 296 for 7B FFN_Up), doubling K-work per block.

**Disaster**: Compiler used 256 VGPRs + 118 register spills → 1 wave/SIMD. The `#pragma unroll` on the KT loop with 64 accumulators forced all KT × 20 operand VGPRs to be live simultaneously.

**Result**: **Slower than V1**. Catastrophic occupancy (1 wave) overwhelmed any tiling benefit.

### V5: N128 + Single-Buffered Inner Loop

**Design**: Fixed V4's register explosion with three changes:

1. `__launch_bounds__(256, 2)` — forces 128 VGPR budget (2 waves/SIMD)
2. `#pragma clang loop unroll(disable)` on the inner kk loop — prevents VGPR explosion
3. Single register set for operands (20 VGPRs) instead of V4's dual-buffered approach

**Resources**: 117 VGPRs, 0 spills, 2 waves/SIMD — **matches CK's occupancy profile** (128 VGPRs, 2 waves, 0 spills).

**Result**: 0.771ms GPU kernel time on 7B FFN_Up (CK: 0.723ms, 6.6% gap). But wallclock was 8.72ms vs CK's 7.32ms (19% gap) — the difference was PCIe overhead, not kernel performance.

**Critical finding**: All native kernel variants (V1–V5) converged to ~8.72ms wallclock regardless of occupancy or tiling strategy. The ISA-level gap was only 6.6%.

### V6: Cross-Iteration Read-Compute Overlap

**Design**: Replaced V5's `unroll(disable)` with `unroll_count(2)` to give the compiler visibility across two consecutive kk iterations, enabling CK-style read/compute interleaving.

**Hypothesis**: V5's `unroll(disable)` prevents the compiler from scheduling kk+1's LDS reads during kk's compute phase. With `unroll_count(2)`, the compiler can interleave reads and dot products.

**ISA confirmation**: Using `s_sched_barrier` variants confirmed the compiler *did* achieve CK-like ISA interleaving patterns.

**Result**: **Zero performance improvement** over V5. This definitively proved the inner compute loop (which all V1–V6 optimizations targeted) is NOT the bottleneck. The remaining 6.6% gap is in the outer pipeline — global load issuance and boundary checking.

### V7: Safe-Tile Split (Winner)

See [V7: The Winning Design](#v7-the-winning-design) below.

### Resource Summary Across Versions

| Version | Tile | VGPRs | Spills | Waves/SIMD | LDS (KB) | Inner Loop |
|---------|------|-------|--------|------------|----------|------------|
| V1 | M128×N64 | 66 | 0 | 3 | 6 | Single-buffered, subroutine |
| V2 | M128×N64 | ~66 | 0 | 3 | 2 | A from L2, B in LDS |
| V3 | M128×N64 | ~66 | 0 | 3 | 24 | Double-buffered LDS |
| V4 | M128×N128 | 256 | 118 | 1 | 32 | Double-buffered, unrolled |
| V5 | M128×N128 | 117 | 0 | 2 | 16–32 | Double-buffered, loop not unrolled |
| V6 | M128×N128 | ~117 | 0 | 2 | 16–32 | Double-buffered, `unroll_count(2)` |
| V7 | M128×N128 | ~117 | 0 | 2 | 16–32 | Double-buffered, safe-tile split |
| CK | M128×N128 | 128 | 0 | 2 | 32 | Fully unrolled K dimension |

---

## ISA Analysis Toolchain

All ISA analysis was performed using LLVM tools shipped with ROCm.

### Step 1: Build the Binary

```bash
cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel
```

The benchmark binary instantiates all kernel variants (V1–V7 + CK), making it ideal for comparative disassembly.

### Step 2: Extract the GPU Code Object

HIP executables embed GPU code objects as ELF sections. List them:

```bash
llvm-readelf -S build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  | grep '__CLANG_OFFLOAD_BUNDLE__'
```

Extract the gfx906 section:

```bash
llvm-objcopy \
  --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' \
  build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison /dev/null
```

### Step 3: Read Kernel Resource Metadata

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

This produces a table of all kernels with their VGPR count, SGPR count, and LDS usage — the three parameters that determine occupancy.

### Step 4: Full Disassembly

```bash
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/ck_disasm.txt
wc -l /tmp/ck_disasm.txt   # Expect ~45,000 lines
```

### Step 5: Locate Kernel Entry Points

```bash
grep -n '<_Z' /tmp/ck_disasm.txt | head -30
```

CK kernel symbols are extremely long (>1000 characters) due to C++ templates. Use distinctive fragments to identify them:

| Grep Pattern | Kernel |
|-------------|--------|
| `qgemm_int8_vnni_wide_tile_v5` | Our V5 kernel |
| `Li256E.*Lb1ELb0E.*PassThrough` | CK 128×128, HasMainK=true, no padding |
| `Li64E` | CK 64×64 |
| `Li32E` | CK 32×32 |

### Step 6: Instruction Census

Count instruction types within a kernel's disassembly line range:

```bash
START=18369; END=22092  # CK 128x128 example
for pat in v_dot4_i32_i8 ds_read_b128 ds_read_b32 ds_write s_barrier \
           s_waitcnt s_nop s_cbranch buffer_load s_swappc; do
  echo -n "$pat: "
  awk "NR>=$START && NR<=$END" /tmp/ck_disasm.txt | grep -c "$pat"
done
```

### Step 7: Map Loop Structure

```bash
# Find barriers (phase boundaries)
awk "NR>=$START && NR<=$END" /tmp/ck_disasm.txt | grep -n 's_barrier'

# Find loop back-edges
awk "NR>=$START && NR<=$END" /tmp/ck_disasm.txt | grep -n 's_cbranch'

# Find subroutine calls (V1 only)
awk "NR>=$START && NR<=$END" /tmp/ck_disasm.txt | grep -n 's_swappc'
```

**Interpreting barriers**:
- Double-buffered kernels (CK, V3–V7): 3 barriers = prologue + 2 per loop body
- Single-buffered kernels (V1): 2 barriers per iteration (load barrier + compute barrier)

### Step 8: Analyze Instruction Scheduling

Read 50–100 ISA lines between barriers to observe interleaving patterns:

```bash
awk 'NR>=18630 && NR<=18730' /tmp/ck_disasm.txt
```

**What to look for**:
- `ds_read_b128` followed by `s_waitcnt lgkmcnt(N)` where N > 0 — progressive waits mean pipelining
- `v_dot4_i32_i8` interleaved between `ds_read` instructions — good scheduling
- `s_nop` — wasted cycles (fewer = better)
- Register reuse — a register appearing as `v_dot4` destination and then `ds_read` source means aggressive recycling

### One-Liner Cheat Sheet

```bash
# Extract → metadata → disassembly → find kernels → census
llvm-objcopy --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' \
  BUILD/tests/v2/v2_perf_rocm_prefill_dispatch_comparison /dev/null && \
llvm-readelf --notes /tmp/co.elf 2>/dev/null | grep -A5 '\.name:' && \
llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/ck_disasm.txt && \
grep -n '<_Z' /tmp/ck_disasm.txt
```

---

## Profiling: rocprof Per-Dispatch Analysis

### The Wallclock Illusion

Early benchmarking showed a consistent ~19% wallclock gap between native kernels and CK:

| Variant | Wallclock (ms) | Apparent Gap |
|---------|---------------|-------------|
| V5/KT8 | 8.72 | 19% slower |
| V5/KT16 | 8.74 | 19% slower |
| CK | 7.32 | baseline |

This was misleading — **every native variant converged to ~8.72ms** regardless of inner loop changes (V1 through V6). This suggested the gap was not in the GPU kernel itself.

### rocprof Per-Dispatch Timing

`rocprof` measures actual GPU kernel execution time, excluding host-side overhead and PCIe transfers:

```bash
# Trace individual kernel dispatches
rocprof --stats --timestamp on \
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*"

# Results appear in results.csv and results.stats.csv
```

The actual GPU kernel times told a completely different story:

| Variant | GPU Kernel (ms) | Actual Gap |
|---------|---------------|------------|
| V5/KT8 | 0.771 | **6.6%** slower |
| V5/KT16 | 0.780 | 7.3% slower |
| CK | 0.723 | baseline |

### The PCIe Bottleneck

The ~1.4ms wallclock difference was almost entirely PCIe overhead:

- **Measured PCIe bandwidth**: 1.79 GB/s (on our topology)
- **Per-GEMM transfer**: ~1.3 MB input + ~9.7 MB output = ~11 MB
- **PCIe time**: 11 MB / 1.79 GB/s ≈ 6.1ms
- **Total wallclock** ≈ 0.77ms (kernel) + 6.1ms (PCIe) + ~1.8ms (overhead) ≈ 8.7ms

CK's slightly lower wallclock suggested it had marginally better PCIe pipelining or allocation patterns, but the kernel-level gap was only 6.6% — a much more tractable target.

### Implications for Optimization

This profiling discovery redirected the entire optimization strategy:

1. **Inner loop optimizations (V1–V6) had diminishing returns** — the 6.6% gap was in the outer pipeline, not the compute loop
2. **Reducing branch count and icache pressure in the outer loop** became the priority
3. **Wallclock benchmarks are unreliable** for kernel-level comparisons on PCIe-bottlenecked systems

---

## Latency Hiding and Interleaving Techniques

### 1. LDS Double-Buffering

**Concept**: Use two LDS buffers (even/odd). While computing from buffer 0, load data into buffer 1. Swap on the next iteration.

```
Iteration N:   [Compute buf[0]] + [Load → buf[1]]  → s_barrier
Iteration N+1: [Compute buf[1]] + [Load → buf[0]]  → s_barrier
```

**Impact**: Eliminates the serial load→barrier→compute→barrier cycle. The primary technique behind CK's advantage over V1 (~5–8% gain).

**Implementation** (V3 onwards):

```cpp
__shared__ int32_t a_lds[2][KT * M_TILE];  // Double-buffered
__shared__ int32_t b_lds[2][KT * N_TILE];

for (int tile = 1; tile < num_tiles; ++tile) {
    int cur = (tile - 1) & 1;  // Compute from this buffer
    int nxt = tile & 1;         // Load into this buffer

    issue_global_loads(tile * KT);  // Async: loads into staging VGPRs
    compute_tile(cur);               // Compute: overlaps with loads!
    __syncthreads();
    write_staging_to_lds(nxt);       // Staging VGPRs → LDS
    __syncthreads();
}
```

### 2. Progressive `s_waitcnt` Management

**Concept**: Instead of waiting for ALL outstanding memory operations (`lgkmcnt(0)`), wait for only as many as needed for the next computation.

**CK's pattern** (observed in disassembly):

```asm
; Issue 8 ds_read_b128 (fills LDS read pipeline)
ds_read_b128 v[94:97],  v35 offset:16384
ds_read_b128 v[98:101], v35 offset:16640
ds_read_b128 v[102:105], v34
ds_read_b128 v[106:109], v34 offset:256
ds_read_b128 v[110:113], v35 offset:16896
ds_read_b128 v[114:117], v35 offset:17152
ds_read_b128 v[118:121], v34 offset:512
ds_read_b128 v[122:125], v34 offset:768

; Wait for only 3 of 8 reads (5 still in-flight)
s_waitcnt lgkmcnt(5)

; Compute on the 3 completed reads while 5 are still streaming
v_dot4_i32_i8 v89, v102, v94, v89
v_dot4_i32_i8 v88, v102, v95, v88
; ... 30 more v_dot4 ...

; Issue new reads into freed registers
ds_read_b128 v[58:61], v35 offset:17408
s_waitcnt lgkmcnt(5)     ; Keep 5 reads in-flight at all times
```

**Key insight**: `lgkmcnt(5)` means "wait until at most 5 LDS reads are still pending." With 8 outstanding and lgkmcnt(5), the GPU waits for exactly 3 to complete, then proceeds. This keeps the LDS read queue constantly fed.

**V5's pattern** (good but not optimal):

```asm
; 5 ds_read_b128 per kk step
ds_read_b128 v[...], ...
ds_read_b128 v[...], ...
ds_read_b128 v[...], ...
ds_read_b128 v[...], ...
ds_read_b128 v[...], ...

s_waitcnt lgkmcnt(3)     ; Wait for 2 of 5
; 16 v_dot4 overlap with 3 outstanding
; ... but remaining 48 v_dot4 have ZERO reads in-flight
```

The `#pragma clang loop unroll(disable)` prevents the compiler from seeing across kk iterations, so it cannot issue kk+1's reads during kk's compute.

### 3. Register Recycling

**Concept**: After consuming a register as a `v_dot4` source operand, immediately reuse it as an accumulator or load target.

```asm
; v102-v105 loaded from LDS (A tile data)
v_dot4_i32_i8 v89, v102, v94, v89   ; Use v102 as source
; ... more v_dot4 using v102 ...

; v102 is now dead as source — reuse as accumulator  
v_dot4_i32_i8 v102, v105, v98, v61  ; v102 is now an accumulator!

; v102-v105 freed as A data — issue new reads into them
ds_read_b128 v[102:105], v34 offset:1024
```

This technique keeps the VGPR file fully utilized. CK achieves 128-VGPR occupancy (2 waves) despite having 64 accumulators + staging + operands because it recycles aggressively.

### 4. `__launch_bounds__` for Occupancy Control

```cpp
__global__ __launch_bounds__(256, 2)  // 256 threads, minimum 2 waves/SIMD
void kernel(...) { ... }
```

The second parameter (`2`) tells the compiler to target 128 VGPRs maximum (`256 / 2 = 128`). Without this, the compiler may use all 256 VGPRs for a single wave, destroying latency hiding. V4's omission of this constraint was the root cause of its failure (256 VGPRs + 118 spills).

### 5. Controlling Inner Loop Unrolling

The inner loop unroll strategy turned out to be critical:

| Strategy | Effect | Outcome |
|----------|--------|---------|
| `#pragma unroll` | Full unroll — all KT iterations visible | V4: VGPR explosion (256+118 spills) |
| `#pragma clang loop unroll(disable)` | No unroll — compiler sees 1 iteration | V5: Clean 117 VGPRs, no cross-step overlap |
| `#pragma clang loop unroll_count(2)` | 2× unroll — compiler sees 2 iterations | V6: CK-like interleaving, same performance |

**Lesson**: On gfx906, `unroll(disable)` and `unroll_count(2)` produce identical performance for this workload. The inner loop scheduling is NOT the bottleneck once you have 2 waves/SIMD and double-buffered LDS.

### 6. Vectorized Load/Store Alignment

All cooperative loads use `uint4` (16-byte) vectorized access:

```cpp
// 16-byte vectorized global load (compiles to global_load_dwordx4)
a_staging[p] = *reinterpret_cast<const uint4*>(
    a_global + mi * a_row_stride + gkg);

// 16-byte vectorized LDS write (compiles to ds_write_b128)
*reinterpret_cast<uint4*>(&b_lds[buf][offset]) = b_staging[p];
```

This is 4× more efficient than scalar loads and maps directly to the widest load instruction available on gfx906.

### 7. Cooperative Loading Geometry

256 threads must cooperatively load an M_TILE × KT buffer. The load is partitioned:

**A matrix** (layout: row-major, each row = K/4 int32 elements):
```
A_K_CHUNKS = KT / 4          // e.g., KT=8 → 2 chunks
A_ROWS_PER_PASS = 256 / 2    // 128 rows per pass
A_NUM_PASSES = M_TILE / 128  // 1 pass for M_TILE=128
```

Each thread loads one `uint4` (4 int32 = 16 bytes of packed int8) from its assigned row/chunk coordinate.

**B matrix** (layout: VNNI — K-groups as leading dimension):
```
B_N_CHUNKS = N_TILE / 4       // 128/4 = 32 chunks
B_TOTAL_GROUPS = KT × 32     // KT=8 → 256 groups
B_NUM_PASSES = 256 / 256     // 1 pass for KT=8
```

For KT=16 with N=128, `B_TOTAL_GROUPS = 512 > 256 threads` → 2 passes.

---

## V7: The Winning Design

### Motivation

V6 proved that inner-loop scheduling is fully optimized. The remaining 6.6% GPU kernel gap had to be in the **outer pipeline** — specifically, the boundary-checking code in the global load path.

V5 ISA analysis revealed:
- **1308 ISA lines** total in the kernel function
- **15 branches** in the "fast path" of the main loop body
- **84 branches** in the "slow/boundary path" of the same loop body
- **99 total branches per loop iteration**

Even with perfect branch prediction, the duplicated boundary code:
1. Wastes 16 KB L1 instruction cache (gfx906 has only 16 KB L1I per CU)
2. Prevents optimal instruction scheduling (compiler must handle both paths)
3. Adds control flow overhead (branch resolution, program counter updates)

CK avoids this by generating separate kernel variants (`HasMainK`, `HasDoubleTail`, `MNPadding`) at compile time, ensuring the hot loop contains zero boundary checks.

### V7 Strategy: Runtime Safe-Tile Split

Instead of compile-time kernel variants (which would require N template instantiations), V7 uses a **runtime split** of the tile loop into two phases:

1. **Safe loop** (tiles 0 through `num_safe-1`): All loads guaranteed in-bounds. Uses unconditional vectorized loads with zero boundary checks.
2. **Boundary loop** (tiles `num_safe` through `num_tiles-1`): Full boundary checking with scalar fallback for edge cases.

### Safe Tile Computation

A tile is "safe" when ALL loads for ALL threads are guaranteed in-bounds:

```
Safe conditions (all must hold simultaneously):
  1. M >= M_TILE (128)          — all A row indices valid
  2. n_block + N_TILE <= N      — all B column indices valid  
  3. (tile+1) * KT <= total_k_groups  — all K-group indices valid
```

```cpp
const bool a_m_safe = (M >= M_TILE);
const bool b_n_safe = (n_block + N_TILE <= N);
const int  safe_k   = total_k_groups / KT;  // floor division
const int  num_safe = (a_m_safe && b_n_safe)
                    ? min(safe_k, num_tiles) : 0;
```

**For 7B FFN_Up** (M=128, N=18944, K=3584):
- `M >= 128` ✓
- `18944 % 128 == 0` → all N-blocks are interior ✓
- `total_k_groups = 896`, `KT=8` → `safe_k = 112 = num_tiles` → **ALL tiles safe**
- Boundary loop never executes

### Two Separate Load Lambdas

**Safe loads** — unconditional vectorized (`global_load_dwordx4`):

```cpp
auto safe_issue_a_loads = [&](int kt_base) __attribute__((always_inline)) {
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p) {
        a_staging[p] = *reinterpret_cast<const uint4*>(
            a_global + a_mi[p] * a_row_stride + kt_base + a_kg);
    }
};
```

**Boundary loads** — with full checks and scalar fallback:

```cpp
auto boundary_issue_a_loads = [&](int kt_base) __attribute__((always_inline)) {
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p) {
        if (__builtin_expect(mi < M && gkg + 3 < total_k_groups, 1)) {
            a_staging[p] = *reinterpret_cast<const uint4*>(...);  // vectorized
        } else {
            a_staging[p].x = (mi < M && gkg + 0 < total_k_groups) ? ... : 0;  // scalar
            a_staging[p].y = (mi < M && gkg + 1 < total_k_groups) ? ... : 0;
            a_staging[p].z = (mi < M && gkg + 2 < total_k_groups) ? ... : 0;
            a_staging[p].w = (mi < M && gkg + 3 < total_k_groups) ? ... : 0;
        }
    }
};
```

### Main Loop Structure

```cpp
// SAFE LOOP: zero boundary checks
for (int tile = 1; tile < num_safe; ++tile) {
    safe_issue_a_loads(tile * KT);
    safe_issue_b_loads(tile * KT);
    compute_tile((tile - 1) & 1);
    __syncthreads();
    write_a_to_lds(tile & 1);
    write_b_to_lds(tile & 1);
    __syncthreads();
}

// BOUNDARY LOOP: full checks (runs 0–2 tiles, or never for aligned shapes)
for (int tile = max(1, num_safe); tile < num_tiles; ++tile) {
    boundary_issue_a_loads(tile * KT);
    boundary_issue_b_loads(tile * KT);
    compute_tile((tile - 1) & 1);
    __syncthreads();
    write_a_to_lds(tile & 1);
    write_b_to_lds(tile & 1);
    __syncthreads();
}
```

### Why This Works

For aligned shapes (the common case in LLM inference):

1. **All boundary code is dead** — the boundary loop's iteration count is 0, so the compiler can place all boundary code in a cold section
2. **Safe loop ISA is maximally tight** — unconditional loads compile to straight-line `global_load_dwordx4` without any comparing, branching, or scalar fallback code
3. **Better icache utilization** — the safe loop body is smaller, fits better in the 16 KB L1I$
4. **Same register budget** — 117 VGPRs, 0 spills, 2 waves/SIMD (identical to V5)

---

## Final Benchmark Results

Benchmark: `WideTileVariantComparison` test, best-of-3 wallclock from `hipDeviceSynchronize`-bracketed measurement. All variants tested on the same GPU in the same run.

### N-Heavy Shapes (V7's Target — FFN_Up/Gate)

| Shape | CK (ms) | V5/KT8 (ms) | V7/KT16 (ms) | V7 vs CK |
|-------|---------|-------------|--------------|----------|
| 0.5B FFN_Up | 1.891 | 1.917 | **1.858** | **1.018×** faster |
| 3B FFN_Up | 4.308 | 4.318 | **4.284** | **1.006×** faster |
| 7B FFN_Up | 7.323 | 7.397 | **7.300** | **1.003×** faster |

### Extreme-Wide Shapes (LM_Head)

| Shape | CK (ms) | V5/KT8 (ms) | V7/KT16 (ms) | V7 vs CK |
|-------|---------|-------------|--------------|----------|
| 0.5B LM_Head | 45.342 | 45.614 | **45.493** | 0.997× |
| 3B LM_Head | 47.303 | 47.631 | **47.279** | **1.001×** faster |
| 7B LM_Head | 49.310 | 49.732 | **49.251** | **1.001×** faster |

### K-Heavy Shapes (V3/V1 Still Best)

| Shape | CK (ms) | Best Native | Winner |
|-------|---------|------------|--------|
| 0.5B FFN_Down | 2.245 | 2.146 (V3/KT8) | V3 |
| 7B FFN_Down | 8.323 | 8.101 (V1/N128) | V1 |
| 7B AttnOut | 2.503 | 2.490 (V1/N64) | V1 |

**Key takeaway**: V7/KT16 is the best native kernel for N-heavy shapes, matching or exceeding CK on all FFN_Up and LM_Head workloads. K-heavy shapes still favor V1 or V3 (their tall-K geometry maps better to V1's simpler N64 tiling or V3's smaller LDS footprint).

---

## Key Lessons Learned

### 1. Profile Before Optimizing

The single most impactful technique was `rocprof` per-dispatch analysis, which revealed the 6.6% real GPU kernel gap vs the misleading 19% wallclock gap. Without this, we would have continued optimizing an inner loop that was already within 6.6% of optimal.

### 2. Occupancy is Table Stakes, Not a Differentiator

V4's catastrophic failure (1 wave/SIMD) showed that occupancy is a hard floor — below 2 waves, nothing else matters. But V5 vs V6 showed that going from "acceptable occupancy" to "perfect scheduling" yielded zero measurable improvement on gfx906. The occupancy sweet spot is exactly 2 waves/SIMD for this workload.

### 3. The Compiler Is Impressively Good

V5 with `#pragma clang loop unroll(disable)` achieved within 6.6% of hand-tuned CK despite preventing any cross-iteration optimization. V6 proving that enabling cross-iteration visibility yields zero improvement suggests the compiler's single-iteration scheduling is already near-optimal.

### 4. Where You Branch Matters More Than How You Compute

All inner loop optimizations (V1–V6) converged to the same performance. The winning optimization (V7) reduced branch count in the outer pipeline by splitting safe from boundary tiles — a structural change, not a scheduling one.

### 5. CK's Real Advantage: Compile-Time Specialization

CK generates separate kernel variants for different boundary conditions at compile time (`HasMainK`, `HasDoubleTail`, `MNPadding`). This is fundamentally optimal — the hot path has zero boundary code at the ISA level. V7 approximates this at runtime with safe-tile splitting, which is nearly as effective for shapes where most tiles are safe.

### 6. `__launch_bounds__` is Non-Negotiable

Always specify `__launch_bounds__(threads, min_waves)` for performance-critical HIP kernels. Without the second parameter, the compiler may use all 256 VGPRs for a single wave. The difference between 1 wave and 2 waves is not 2× — it's often 3–5× due to the lost latency-hiding capability.

### 7. Wallclock Benchmarks on PCIe-Bottlenecked Systems Are Misleading

On our topology with 1.79 GB/s PCIe bandwidth, over 90% of GEMM wallclock time is memory transfer. Kernel-level differences vanish in the noise unless you use `rocprof` or similar hardware-level profiling.

---

## Command Reference

### Build & Benchmark

```bash
# Build the benchmark binary (Release mode)
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_ROCM=ON
cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel

# Run all wide-tile variant comparison benchmarks
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*"

# Run a specific V7-only benchmark
LLAMINAR_ROCM_WIDE_TILE_V7=1 \
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*"
```

### ISA Extraction & Analysis

```bash
# Full pipeline: extract → metadata → disassemble → find kernels
BINARY=build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison

llvm-objcopy \
  --dump-section='__CLANG_OFFLOAD_BUNDLE__hipv4-amdgcn-amd-amdhsa--gfx906=/tmp/co.elf' \
  "$BINARY" /dev/null

llvm-readelf --notes /tmp/co.elf 2>/dev/null | awk '
  /\.name:/{name=$2} /\.vgpr_count:/{vgpr=$2} /\.sgpr_count:/{sgpr=$2}
  /\.group_segment_fixed_size:/{lds=$2}
  /\.wavefront_size:/{printf "%-60s VGPR=%-4s SGPR=%-4s LDS=%s\n",substr(name,1,60),vgpr,sgpr,lds}'

llvm-objdump -d --mcpu=gfx906 /tmp/co.elf > /tmp/ck_disasm.txt

grep -n '<_Z' /tmp/ck_disasm.txt | head -40
```

### Kernel Selection Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLAMINAR_ROCM_WIDE_TILE_V7=1` | Force V7 kernel |
| `LLAMINAR_ROCM_WIDE_TILE_V6=1` | Force V6 kernel |
| `LLAMINAR_ROCM_WIDE_TILE_V5=1` | Force V5 kernel |
| `LLAMINAR_ROCM_WIDE_TILE_V4=1` | Force V4 kernel |
| `LLAMINAR_ROCM_WIDE_TILE_V3=1` | Force V3 kernel |
| `LLAMINAR_ROCM_WIDE_TILE_V2=1` | Force V2 kernel |


### rocprof Profiling

```bash
# Per-dispatch kernel timing
rocprof --stats --timestamp on \
  ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
  --gtest_filter="*WideTileVariantComparison*"

# Results in results.csv (per-dispatch) and results.stats.csv (aggregate)
cat results.stats.csv | column -t -s,
```

---

## Source Files

| File | Purpose |
|------|---------|
| [ROCmQuantisedGemmKernel.hip](ROCmQuantisedGemmKernel.hip) | All V1–V7 kernel templates, dispatch functions, and CK stubs |
| [ROCmQuantisedGemmKernel.cpp](ROCmQuantisedGemmKernel.cpp) | Dispatch wiring (env var → kernel selection) |
| [../../utils/DebugEnv.h](../../utils/DebugEnv.h) | Environment variable definitions |
| [../../../../tests/v2/performance/kernels/rocm/Perf__ROCmPrefillDispatchComparison.cpp](../../../../tests/v2/performance/kernels/rocm/Perf__ROCmPrefillDispatchComparison.cpp) | Benchmark test |
| [../../../../changelog/2025-07-24-ck-vs-v1-isa-analysis.md](../../../../changelog/2025-07-24-ck-vs-v1-isa-analysis.md) | CK vs V1 ISA deep dive |
| [../../../../changelog/2025-07-25-v5-kernel-occupancy-experiment.md](../../../../changelog/2025-07-25-v5-kernel-occupancy-experiment.md) | V5 design and occupancy results |
