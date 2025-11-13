# OneDNN INT8 GEMM Architecture Deep Dive

**Author**: David Sanftenberg  
**Date**: November 12, 2025  
**Source**: `external/onednn/src/cpu/x64/gemm/s8x8s32/`

---

## Overview

This document provides a detailed technical analysis of OneDNN's INT8 GEMM implementation, extracted from the source code. Use this as a reference when implementing our optimizations.

---

## 1. Microkernel Architecture

### Register Allocation Strategy

**File**: `jit_avx512_core_gemm_s8u8s32_kern.hpp` (lines 77-84)

```cpp
static const int IGEMM_UNROLL_M_ = 48;  // M-dimension blocking
static const int IGEMM_UNROLL_N_ = 8;   // N-dimension blocking

// Vector register assignments
Xbyak::Zmm dp_scratch_, ones_, a_regs_[max_unroll_m_ >> 4], b_regs_[2];
Xbyak::Zmm c_regs_[max_unroll_m_ >> 4][max_unroll_n_];
```

**Register Breakdown**:
- `c_regs_[3][8]`: **24 ZMM registers** for 48×8 output tile
  - 48 rows / 16 INT32 per ZMM = 3 ZMM per column
  - 8 columns = 3×8 = 24 ZMM total
- `a_regs_[3]`: **3 ZMM registers** for A matrix rows
  - Reused across all 8 N-columns
- `b_regs_[2]`: **2 ZMM registers** for B matrix (ping-pong)
- `dp_scratch_`, `ones_`: **2 ZMM registers** for VNNI emulation
- **Total: 31 ZMM registers used** (out of 32 available)

**Key Insight**: Minimal register pressure allows aggressive reuse.

---

## 2. Inner Kernel Loop

### Main Computation Loop

**File**: `jit_avx512_core_gemm_s8u8s32_kern.cpp` (lines 93-155)

```cpp
void jit_avx512_core_gemm_s8u8s32_kern_t::kernel_loop(
        int unroll_m, int unroll_n, bool cfetch) {
    int um_vecs = (unroll_m + 15) >> 4;  // Number of ZMM for M-dimension
    Label label_kernel_loop;

    L_aligned(label_kernel_loop);
    {
        for (int h = 0; h < 4; h++) {  // Process 8 K-elements per iteration (4×2)
            for (int j = 0; j < unroll_n; j++) {  // N-dimension loop
                const Zmm b = b_regs_[j & 1];  // Ping-pong B registers
                
                // Broadcast one 32-bit element from B (4 INT8 values)
                vpbroadcastd(b, ptr[BO_ + isize_ * (2 * j + 2 * h * unroll_n - offset_b_)]);
                
                // Accumulate into first row of C
                dot_product(c_regs_[0][j], b, a_regs_[0]);
                
                // Prefetch control (interleaved with computation)
                if (j == 1 && !(h & 1))
                    prefetch_b(ptr[BO_ + isize_ * (prefetch_size_b_ + 2 * h * unroll_n - offset_b_)]);
                else if (j % 3 == 0)
                    prefetch_a(ptr[AO_ + isize_ * (prefetch_size_a_ + 32 * (j / 3) + 2 * h * unroll_m - offset_a_)]);
                
                // Accumulate into remaining rows of C
                for (int i = 1; i < um_vecs; i++)
                    dot_product(c_regs_[i][j], b, a_regs_[i]);
                
                // Prefetch C for writeback (streaming stores)
                if (cfetch && (j == std::min(1, unroll_n - 1))) {
                    if (h == 3)
                        lea(CO2_, ptr[CO2_ + LDC_]);
                    else if (h < um_vecs)
                        prefetch_c(ptr[CO2_ + (16 * h * size_)]);
                }
            }
            
            // Load next A tile for next iteration
            for (int i = 0; i < um_vecs; i++)
                vmovups(a_regs_[i], ptr[AO_ + isize_ * (32 * i + 2 * (h + 1) * unroll_m - offset_a_)]);
            
            if (h == 2) prefetch_x(ptr[AA_ - (offset_a_ * isize_)]);  // Extra prefetch
        }
        
        add(AO_, 8 * isize_ * unroll_m);  // Advance A pointer
        add(BO_, 8 * isize_ * unroll_n);  // Advance B pointer
        sub(LoopCount_, 1);
        jg(label_kernel_loop, T_NEAR);
    }
}
```

**Loop Structure Analysis**:
- **Outer loop** (implicit via `LoopCount_`): K-dimension, 8 elements per iteration
- **h loop** (0-3): Process 2 K-elements per iteration (4×2 = 8 total)
- **j loop** (0-7): N-dimension columns
- **i loop** (0-2): M-dimension row blocks (implicit in dot_product calls)

**Key Techniques**:
1. **Interleaved prefetching**: Every 3rd operation prefetches next data
2. **Ping-pong B registers**: `b_regs_[j & 1]` alternates to avoid WAR hazards
3. **C prefetching**: Prefetch write targets (`prefetchw`) to avoid RFO stalls
4. **Loop unrolling**: h-loop fully unrolled (4 iterations)

---

## 3. VNNI Dot Product

### VNNI vs Emulation

**File**: `jit_avx512_core_gemm_s8u8s32_kern.cpp` (lines 75-85)

```cpp
void jit_avx512_core_gemm_s8u8s32_kern_t::dot_product(
        const Xmm &dst, const Xmm &src1, const Xmm &src2) {
    if (vnni_)
        vpdpbusd(dst, src1, src2);  // 1 instruction: 4×(u8×s8)→s32
    else {
        vpmaddubsw(dp_scratch_, src1, src2);  // 2×(u8×s8)→s16
        vpmaddwd(dp_scratch_, ones_, dp_scratch_);  // 2×s16→s32
        vpaddd(dst, dst, dp_scratch_);  // Accumulate s32
    }
}
```

**VNNI Instruction Details**:
- `vpdpbusd zmm1, zmm2, zmm3`:
  - Multiplies 64 unsigned×signed INT8 pairs from zmm2/zmm3
  - Groups into 16 sets of 4 products
  - Sums each set of 4 → 16 INT32 results
  - Adds to zmm1 (accumulator)
- **Throughput**: 1 cycle on Ice Lake/Cascade Lake (dual-issue)
- **Latency**: 4 cycles

**Emulation Cost** (no VNNI):
- `vpmaddubsw`: 1 cycle throughput, 4 cycles latency
- `vpmaddwd`: 1 cycle throughput, 4 cycles latency
- `vpaddd`: 0.33 cycles throughput, 1 cycle latency
- **Total**: 3 instructions vs 1, 2-3× slower

---

## 4. Data Packing Strategy

### B Matrix Packing (Column-Major → Panels)

**File**: `jit_avx512_core_u8_copy_bn_kern_autogen.cpp`

**Packing Concept**:
```
Original B (column-major): [N][K]
  - Strided access: Load column N requires K×N stride
  - Cache unfriendly: Load 64-byte line, use 1 byte

Packed B (panel format): [N/8][K][8]
  - Sequential access: Load column N requires K×8 stride
  - Cache friendly: Load 64-byte line, use 8 bytes
```

**Implementation Pattern** (pseudo-assembly):
```cpp
// For each N/8 panel:
for (int nc = 0; nc < N; nc += 8) {
    // For each K row:
    for (int k = 0; k < K; ++k) {
        // Copy 8 consecutive columns into packed format
        vmovdqu(xmm0, ptr[B + nc*K + k]);      // Load 16 bytes (overkill, but aligned)
        vmovsd(ptr[PackedB + panel_offset], xmm0);  // Store 8 bytes continuously
        panel_offset += 8;
    }
}
```

**Benefits**:
1. **Sequential access**: Next 8 columns are adjacent in memory
2. **Cache efficiency**: 64-byte cache line holds 8×8 = 64 INT8 values (fully utilized)
3. **Prefetching**: Hardware prefetcher can predict next cache line
4. **SIMD alignment**: 8-column panels match broadcast width

---

## 5. Prefetching Configuration

### Prefetch Distances

**File**: `jit_avx512_core_gemm_s8u8s32_kern.hpp` (lines 69-70)

```cpp
static const int prefetch_size_a_ = 32 * 5;  // 160 elements ahead
static const int prefetch_size_b_ = 32 * 4;  // 128 elements ahead
```

**Rationale**:
- **A matrix**: 160 elements = 160 bytes = 2.5 cache lines
  - Memory latency: ~50-100 cycles
  - Kernel processes ~20 elements per iteration
  - 160 elements = 8 iterations ahead → perfect timing
  
- **B matrix**: 128 elements = 128 bytes = 2 cache lines
  - Broadcast from B is faster (single load)
  - Less prefetch distance needed

**Prefetch Types**:
```cpp
void prefetch_a(const Xbyak::Address &src) { prefetcht0(src); }  // L2 cache
void prefetch_b(const Xbyak::Address &src) { prefetcht0(src); }  // L2 cache
void prefetch_c(const Xbyak::Address &src) { prefetchw(src); }   // L2 cache (exclusive)
```

- `prefetcht0`: Prefetch to all cache levels (L1/L2/L3)
- `prefetchw`: Prefetch for write (request exclusive ownership)

---

## 6. Cache Blocking Parameters

### Multi-Level Blocking

**OneDNN uses dynamic blocking** based on cache sizes, but typical values:

```cpp
// L3 cache (shared across cores, ~35MB)
MC_L3 = 192-384   // M-dimension: 192-384 rows
NC_L3 = 4096-8192 // N-dimension: 4096-8192 columns
KC_L3 = 512-1024  // K-dimension: 512-1024 elements

// Working set size:
// A: MC_L3 × KC_L3 = 384 × 1024 INT8 = 384KB
// B: NC_L3 × KC_L3 = 8192 × 1024 INT8 = 8MB
// C: MC_L3 × NC_L3 = 384 × 8192 INT32 = 12MB
// Total: ~20MB (fits in 35MB L3)

// L2 cache (per-core, ~1MB)
MC_L2 = 48-96     // M-dimension: 48-96 rows
NC_L2 = 256-512   // N-dimension: 256-512 columns
KC_L2 = 256-512   // K-dimension: 256-512 elements

// Working set size:
// A: 96 × 512 INT8 = 48KB
// B: 512 × 512 INT8 = 256KB
// C: 96 × 512 INT32 = 192KB
// Total: ~500KB (fits in 1MB L2)

// L1 cache (register blocking)
MR = 48  // Matches IGEMM_UNROLL_M
NR = 8   // Matches IGEMM_UNROLL_N
```

**Key Principle**: Each level's working set fits entirely in that cache level.

---

## 7. Loop Ordering and Parallelization

### OneDNN's Loop Nest

**Conceptual structure** (from `gemm_driver.cpp`):

```cpp
#pragma omp parallel for  // Parallelize on M-blocks (coarse-grained)
for (int mc_l3 = 0; mc_l3 < M; mc_l3 += MC_L3) {
    // Pack A matrix for this M-block (amortized cost)
    pack_a_matrix(A, mc_l3, MC_L3);
    
    for (int nc_l3 = 0; nc_l3 < N; nc_l3 += NC_L3) {
        // Pack B matrix for this N-block (done once per thread)
        pack_b_matrix(B, nc_l3, NC_L3);
        
        for (int kc_l3 = 0; kc_l3 < K; kc_l3 += KC_L3) {
            // L3 block: Fits in shared cache
            
            for (int mc_l2 = mc_l3; mc_l2 < mc_l3 + MC_L3; mc_l2 += MC_L2) {
                for (int nc_l2 = nc_l3; nc_l2 < nc_l3 + NC_L3; nc_l2 += NC_L2) {
                    // L2 block: Fits in private cache
                    
                    for (int kc_l2 = kc_l3; kc_l2 < kc_l3 + KC_L3; kc_l2 += KC_L2) {
                        // L1 block: Call register-blocked microkernel
                        
                        for (int i = mc_l2; i < mc_l2 + MC_L2; i += MR) {
                            for (int j = nc_l2; j < nc_l2 + NC_L2; j += NR) {
                                // Microkernel: Stays in registers
                                microkernel_48x8(A_packed, B_packed, C, i, j, kc_l2, KC_L2);
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**Parallelization Strategy**:
- **Outer M-loop parallelized**: Each thread gets independent M-blocks
- **No inner parallelization**: Avoids OpenMP overhead in hot loops
- **Work distribution**: Static scheduling (predictable load balance)

---

## 8. JIT Assembly Benefits

### Why JIT vs C++ Intrinsics?

**OneDNN uses Xbyak** (JIT assembler library):

**Advantages**:
1. **Precise instruction scheduling**:
   ```cpp
   // Can guarantee this sequence:
   vpbroadcastd(b0, ptr[BO_]);        // Load B (4 cycles latency)
   vmovups(a0, ptr[AO_]);             // Load A (4 cycles latency)
   prefetch_a(ptr[AO_ + 160]);        // Prefetch (overlaps with latency)
   vpdpbusd(c0, a0, b0);              // Use A/B (executes when ready)
   ```
   
2. **No register spills**:
   - Compiler might spill registers to stack under pressure
   - JIT explicitly controls all 32 ZMM registers
   
3. **Explicit loop unrolling**:
   ```cpp
   for (int j = 0; j < 8; ++j) {
       vpbroadcastd(b, ptr[BO_ + j*4]);
       vpdpbusd(c0, a0, b);
       vpdpbusd(c1, a1, b);
       vpdpbusd(c2, a2, b);
   }
   // Fully unrolled at compile time, no loop overhead
   ```

4. **Alignment control**:
   ```cpp
   L_aligned(label_kernel_loop);  // Force 32-byte alignment
   ```

**Downsides**:
- More complex to write and debug
- Portability (x64-specific)
- Learning curve for Xbyak API

---

## 9. Performance Bottleneck Analysis

### Why OneDNN is 14× Faster

**Memory Traffic Reduction**:
```
Our approach (naive):
- Load C: M×N×K times (every iteration)
- Load A: M×N×K times (every iteration)
- Load B: M×N×K times (every iteration)
- Store C: M×N×K times (every iteration)
Total memory ops: 4×M×N×K

OneDNN approach (48×8 blocking):
- Load C: (M/48)×(N/8) times (once per tile)
- Load A: (M/48)×N×K times (broadcast reused)
- Load B: M×(N/8)×K times (broadcast reused)
- Store C: (M/48)×(N/8) times (once per tile)
Total memory ops: (M/48)×(N/8)×(2 + 48K + 8K)

For M=4096, N=896, K=4864:
- Naive: 4 × 4096 × 896 × 4864 = 71.3 billion ops
- OneDNN: (4096/48) × (896/8) × (2 + 48×4864 + 8×4864) = 2.3 billion ops
Reduction: 31× less memory traffic!
```

**Instruction-Level Parallelism**:
```
Our approach:
- 1 VNNI instruction in flight at a time
- Pipeline bubbles due to dependencies

OneDNN:
- 24 independent C accumulators (c_regs[3][8])
- 8 VNNI instructions in flight simultaneously
- CPU can execute 2 VNNI/cycle (dual-issue)
- Perfect pipeline utilization
```

---

## 10. Implementation Checklist

### Phase 1: Register Blocking

- [ ] Allocate 6 ZMM registers for C (6×16 = 96 INT32)
- [ ] Implement K-loop with VNNI accumulation
- [ ] Load C once, store once (minimize memory traffic)
- [ ] Verify register usage (no spills to stack)

### Phase 2: Data Packing

- [ ] Pack B into [N/16][K][16] panels
- [ ] Store scales separately (per-panel)
- [ ] Update microkernel to use packed format
- [ ] Verify sequential access pattern

### Phase 3: Prefetching

- [ ] Add `_mm_prefetch` for A (160 elements ahead)
- [ ] Add `_mm_prefetch` for B (128 elements ahead)
- [ ] Profile L2 cache miss rate (should drop by 50-70%)

### Phase 4: Cache Blocking

- [ ] Implement 3-level blocking (L3/L2/L1)
- [ ] Calculate block sizes for cache hierarchy
- [ ] Parallelize on L3 blocks (not inner loops)
- [ ] Verify working sets fit in respective caches

---

## Summary: OneDNN's Secret Sauce

1. **Massive register reuse**: 48×8 tiles stay in registers across entire K-loop
2. **Sequential data layout**: Packed panels eliminate strided access
3. **Explicit prefetching**: Software-controlled cache management
4. **Multi-level blocking**: Working sets perfectly sized for cache hierarchy
5. **JIT assembly**: Precise control over instruction scheduling and register allocation

**Our path forward**: Implement these techniques incrementally (Phases 1-4) to close the 14× gap.

---

**Next**: Follow `INT8_GEMM_QUICK_ACTION_PLAN.md` to start Phase 1 implementation.
