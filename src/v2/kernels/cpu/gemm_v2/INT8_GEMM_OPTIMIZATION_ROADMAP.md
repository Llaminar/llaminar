# INT8 GEMM Optimization Roadmap
## Achieving OneDNN Performance (6,610 GOPS Target)

**Author**: David Sanftenberg  
**Date**: November 12, 2025  
**Current Performance**: 472 GOPS (7.14% of target)  
**Target Performance**: 6,610 GOPS (OneDNN baseline, 84% of theoretical peak)  
**Gap**: 14× speedup needed

---

## Executive Summary

After investigating OneDNN's INT8 GEMM implementation (`external/onednn/src/cpu/x64/gemm/s8x8s32/`), we've identified the key architectural differences between our current approach and OneDNN's highly optimized implementation. This document provides a step-by-step roadmap to close the 14× performance gap.

**Key Finding**: OneDNN uses a fundamentally different architecture:
1. **JIT-compiled assembly kernels** (not C++ intrinsics)
2. **48×8 register blocking** (massive register reuse)
3. **Pre-packed data layouts** (column-major panels for cache efficiency)
4. **Explicit prefetching** (software-controlled cache management)
5. **VNNI-optimized dot products** (`vpdpbusd` instruction)

---

## OneDNN Architecture Analysis

### 1. Register Blocking Strategy

**File**: `jit_avx512_core_gemm_s8u8s32_kern.hpp`

```cpp
static const int IGEMM_UNROLL_M_ = 48;  // M-dimension blocking
static const int IGEMM_UNROLL_N_ = 8;   // N-dimension blocking
```

**Key Insights**:
- **48×8 output tile** processed in registers simultaneously
- 48 M-dimension = 3 ZMM registers (16 INT32 values each)
- 8 N-dimension = 8 separate accumulator sets
- **Total: 24 ZMM registers for C accumulators** (3×8)
- Remaining registers for A/B tiles and temporaries

**Why This Matters**:
- Each 48×8 tile stays in registers for entire K-dimension loop
- Minimizes C matrix memory traffic (load once, accumulate many times, store once)
- Our current approach: 1×1 blocking (no register reuse across loop iterations)

### 2. Data Packing (Copy Kernels)

**Files**: 
- `jit_avx512_core_u8_copy_bn_kern_autogen.cpp` (pack B matrix)
- `jit_avx512_core_u8_copy_an_kern_autogen.cpp` (pack A matrix)

**OneDNN Packing Strategy**:
- **B matrix**: Packed into column-major panels (N-dimension grouped)
- **Panel width**: 8 columns (matches UNROLL_N)
- **Block size**: 32 rows (K-dimension, matches cache line)
- **Layout**: Continuous in memory for sequential access

**Example B Matrix Packing**:
```
Original B (column-major): [N, K]
Packed B: [N/8, K/32, 8, 32] with elements continuous
```

**Benefits**:
- Eliminates column-major striding (sequential access pattern)
- Cache lines fully utilized (32 bytes = 32 INT8 values)
- Prefetching becomes effective (predictable access pattern)

**Our Current Approach**:
- Q8_0 block format: [N, K_blocks] with 32-element blocks
- Column-major access requires K_blocks stride per element
- Cache lines partially wasted (load 64 bytes, use 1-2 bytes)

### 3. JIT Assembly Generation

**File**: `jit_avx512_core_gemm_s8u8s32_kern.cpp`

**OneDNN Uses Xbyak JIT**:
```cpp
void jit_avx512_core_gemm_s8u8s32_kern_t::generate() {
    // Runtime assembly generation
    vpbroadcastd(b, ptr[BO_ + offset]);
    vpdpbusd(c_regs_[i][j], b, a_regs_[i]);  // VNNI dot product
    prefetcht0(ptr[AO_ + prefetch_offset]);   // Explicit prefetch
}
```

**Advantages over C++ Intrinsics**:
- **Precise instruction scheduling** (avoid compiler reordering)
- **Explicit register allocation** (no spills to stack)
- **Custom prefetching** (tuned for specific access patterns)
- **Minimal overhead** (no function call boundaries)

**Our Current Approach**:
- C++ with AVX512 intrinsics (`_mm512_dpbusd_epi32`)
- Compiler decides instruction scheduling
- Register allocation subject to compiler heuristics
- No explicit prefetching

### 4. Cache Blocking Hierarchy

**OneDNN's 3-Level Blocking**:

```cpp
// L3 cache blocks (shared across cores)
MC = 192-384   // M-dimension L3 block
NC = 4096-8192 // N-dimension L3 block
KC = 512-1024  // K-dimension L3 block

// L2 cache blocks (per-core)
MC2 = 48       // Matches register blocking (IGEMM_UNROLL_M)
NC2 = 256-512  // Fits in L2 with KC
KC2 = 256-512  // Fits in L2 with MC2/NC2

// L1 cache blocks (register blocking)
MR = 48        // Register tile M-dimension
NR = 8         // Register tile N-dimension
```

**Our Current Attempt**:
- Single-level blocking: MC=256, KC=512, NC=128
- No L3/L2/L1 hierarchy
- OpenMP parallelization overhead dominates small blocks

### 5. VNNI Optimization

**OneDNN's VNNI Usage**:
```cpp
void dot_product(const Xmm &dst, const Xmm &src1, const Xmm &src2) {
    if (vnni_)
        vpdpbusd(dst, src1, src2);  // 4×INT8→INT32 in one instruction
    else {
        // Fallback: 3 instructions
        vpmaddubsw(dp_scratch_, src1, src2);  // 2×INT8→INT16
        vpmaddwd(dp_scratch_, ones_, dp_scratch_); // 2×INT16→INT32
        vpaddd(dst, dst, dp_scratch_);        // Accumulate
    }
}
```

**Our Current Approach**:
- Uses VNNI but with bias correction overhead
- Converts signed→unsigned (+128 offset)
- Post-processing bias subtraction
- OneDNN uses native unsigned×signed VNNI (no conversion)

### 6. Prefetching Strategy

**OneDNN's Explicit Prefetching**:
```cpp
static const int prefetch_size_a_ = 32 * 5;  // 160 elements ahead
static const int prefetch_size_b_ = 32 * 4;  // 128 elements ahead

// In kernel loop:
prefetch_a(ptr[AO_ + isize_ * (prefetch_size_a_ + offset)]);
prefetch_b(ptr[BO_ + isize_ * (prefetch_size_b_ + offset)]);
prefetch_c(ptr[CO2_ + (16 * h * size_)]);  // Prefetch write
```

**Benefits**:
- Hides memory latency (data ready when needed)
- Tuned for specific access patterns
- Separate prefetch distances for A/B/C

**Our Current Approach**:
- No explicit prefetching
- Relies on hardware prefetcher (less effective for irregular patterns)

---

## Current vs OneDNN Comparison

| Feature | Our Implementation | OneDNN | Impact |
|---------|-------------------|---------|---------|
| **Register blocking** | 1×1 (no blocking) | 48×8 (24 ZMM regs) | **10-15×** |
| **Data layout** | Q8_0 blocks (strided) | Packed panels (sequential) | **2-3×** |
| **Code generation** | C++ intrinsics | JIT assembly | **1.5-2×** |
| **Prefetching** | Hardware only | Explicit software | **1.3-1.5×** |
| **Cache blocking** | Single level (failing) | 3-level hierarchy | **1.2-1.5×** |
| **VNNI usage** | With bias correction | Native unsigned×signed | **1.1-1.2×** |
| **Combined** | **472 GOPS** | **6,610 GOPS** | **14×** |

**Note**: Impacts are multiplicative, not additive. Product ≈ 10 × 2 × 1.5 × 1.3 × 1.2 × 1.1 ≈ **50×** theoretical, but diminishing returns → **14× actual**

---

## Optimization Roadmap

### Phase 1: Register Blocking (Target: 2,000-3,000 GOPS, 4-6× speedup)

**Goal**: Implement 6×16 register microkernel

**Tasks**:
1. **Design 6×16 microkernel**:
   - 6 M-dimension = 1 ZMM register (16 INT32 values, but process 6 rows)
   - 16 N-dimension = 16 separate accumulator columns
   - Keep 96 INT32 accumulators in registers (6 ZMM regs)
   
2. **Implement K-dimension loop**:
   ```cpp
   // Pseudocode
   for (int k = 0; k < K; k += 32) {
       load_a_panel(6 rows, 32 elements);  // 1.5 ZMM
       for (int n = 0; n < 16; ++n) {
           broadcast_b(column n, 32 elements);  // 1 ZMM
           vnni_dot_product(c_regs[n], a_panel, b_broadcast);
       }
   }
   ```

3. **Outer M×N tiling**:
   - Process output in 6×16 tiles
   - Load C tile once, accumulate over K, store once

**Expected Performance**:
- Reduces C memory traffic by 32× (load/store once per 512 K-iterations, not per element)
- Improves instruction-level parallelism (16 independent accumulators)
- Better register utilization (96/512 = 19% of registers vs current <5%)

**Implementation Effort**: 2-3 days

**Files to Create**:
- `src/v2/kernels/cpu/gemm/int8/RegisterBlockedInt8Gemm.h`
- `tests/v2/performance/cpu/kernels/gemm/Perf__RegisterBlockedInt8Gemm.cpp`

---

### Phase 2: Data Packing (Target: 4,000-5,000 GOPS, 2× speedup on top of Phase 1)

**Goal**: Pack B matrix into cache-friendly panel format

**Tasks**:
1. **Design panel layout**:
   ```cpp
   // Original: B[N, K_blocks] with Q8_0 blocks
   // Packed:   B_panels[N/16, K/32, 16, 32] continuous INT8
   
   struct PackedB {
       std::vector<int8_t> data;      // [N/16 * K/32 * 16 * 32]
       std::vector<float> scales;     // [N/16 * K/32] per-panel scales
       int N, K;
   };
   ```

2. **Implement packing function**:
   ```cpp
   PackedB pack_b_matrix(const Q8_0Block* B, int N, int K) {
       // For each N/16 panel:
       //   For each K/32 block:
       //     Extract 16×32 tile from Q8_0 blocks
       //     Store continuously in packed layout
       //     Store combined scale for tile
   }
   ```

3. **Update microkernel for packed format**:
   - Sequential loads instead of strided
   - Prefetching becomes effective

**Expected Performance**:
- Eliminates column-major stride overhead
- Full cache line utilization (32 bytes = 32 INT8)
- Hardware prefetcher works effectively

**Implementation Effort**: 2-3 days

**Memory Trade-off**:
- Packed format same size as original Q8_0
- One-time packing cost amortized over many inferences

---

### Phase 3: Explicit Prefetching (Target: 5,000-6,000 GOPS, 1.3× speedup)

**Goal**: Add software prefetch instructions

**Tasks**:
1. **Tune prefetch distances**:
   ```cpp
   constexpr int PREFETCH_A_AHEAD = 160;  // 5 iterations ahead
   constexpr int PREFETCH_B_AHEAD = 128;  // 4 iterations ahead
   ```

2. **Insert prefetch instructions**:
   ```cpp
   for (int k = 0; k < K; k += 32) {
       _mm_prefetch((char*)(A + k + PREFETCH_A_AHEAD), _MM_HINT_T0);
       _mm_prefetch((char*)(B + k + PREFETCH_B_AHEAD), _MM_HINT_T0);
       // ... computation ...
   }
   ```

3. **Profile and tune**:
   - Measure L2 cache miss rate
   - Adjust prefetch distances based on latency

**Expected Performance**:
- Hides 10-20ns memory latency
- Reduces pipeline stalls

**Implementation Effort**: 1 day

---

### Phase 4: 3-Level Cache Blocking (Target: 6,000-6,500 GOPS, 1.2× speedup)

**Goal**: Implement L3/L2/L1 cache hierarchy

**Tasks**:
1. **Calculate block sizes**:
   ```cpp
   // L3 (shared): Fit working set in 35MB L3
   const int MC_L3 = 384;   // 384 × 512 INT8 ≈ 200KB (A)
   const int KC_L3 = 512;
   const int NC_L3 = 8192;  // 8192 × 512 INT8 ≈ 4MB (B)
   
   // L2 (per-core): Fit in 1MB L2
   const int MC_L2 = 96;    // 96 × 256 INT8 ≈ 25KB
   const int KC_L2 = 256;
   const int NC_L2 = 512;   // 512 × 256 INT8 ≈ 130KB
   
   // L1 (register): Microkernel blocking
   const int MR = 6;
   const int NR = 16;
   ```

2. **Implement nested loops**:
   ```cpp
   for (mc_l3 = 0; mc_l3 < M; mc_l3 += MC_L3) {
     for (nc_l3 = 0; nc_l3 < N; nc_l3 += NC_L3) {
       for (kc_l3 = 0; kc_l3 < K; kc_l3 += KC_L3) {
         // L3 block fits in shared cache
         for (mc_l2 = mc_l3; mc_l2 < mc_l3 + MC_L3; mc_l2 += MC_L2) {
           for (nc_l2 = nc_l3; nc_l2 < nc_l3 + NC_L3; nc_l2 += NC_L2) {
             // L2 block fits in private cache
             for (kc_l2 = kc_l3; kc_l2 < kc_l3 + KC_L3; kc_l2 += KC_L2) {
               // Call register-blocked microkernel
               microkernel_6x16(mc_l2, nc_l2, kc_l2, ...);
             }
           }
         }
       }
     }
   }
   ```

3. **Parallelize outer loops**:
   - OpenMP on L3 blocks (coarse-grained, low overhead)
   - Each thread gets independent L2/L1 working sets

**Expected Performance**:
- Maximizes cache hit rates at all levels
- Reduces memory bandwidth pressure

**Implementation Effort**: 2-3 days

---

### Phase 5 (Optional): JIT Assembly (Target: 6,500-7,000 GOPS, 1.1-1.3× speedup)

**Goal**: Replace C++ intrinsics with JIT-compiled assembly

**Tasks**:
1. **Integrate Xbyak**:
   - Add Xbyak library (header-only, already in OneDNN)
   - Create JIT class inheriting from `Xbyak::CodeGenerator`

2. **Generate microkernel at runtime**:
   ```cpp
   class JitInt8Microkernel : public Xbyak::CodeGenerator {
   public:
       JitInt8Microkernel(int MR, int NR) {
           // Generate assembly for MR×NR microkernel
           // Precise register allocation
           // Explicit instruction scheduling
       }
   };
   ```

3. **Benchmark vs intrinsics**:
   - May not provide significant speedup if compiler does well
   - Main benefit: explicit control (debugging, tuning)

**Expected Performance**:
- 10-30% speedup if compiler isn't optimal
- More predictable performance (no compiler surprises)

**Implementation Effort**: 3-5 days (learning curve for Xbyak)

**Risk**: May not be worth effort if Phases 1-4 achieve target

---

## Alternative: BRGEMM (Block Repetitive GEMM)

**OneDNN's Newer Approach** (Post-2020):

Instead of classic GEMM with 3-level blocking, OneDNN now uses **BRGEMM**:

**Key Concept**:
- Split GEMM into many small "blocks"
- Each block is 32×32 or 48×16 (fits in registers)
- Blocks processed repetitively with same kernel
- Better instruction cache utilization

**File**: `external/onednn/src/cpu/x64/brgemm/brgemm.hpp`

**Advantages**:
- Simpler kernel (one block size, reused)
- Better for variable-size matrices
- Easier to optimize (tune one kernel)

**Trade-offs**:
- Requires more sophisticated driver code
- May not beat hand-tuned full GEMM for fixed sizes

**Recommendation**: Start with classic GEMM (Phases 1-4), consider BRGEMM if hitting limits.

---

## Implementation Priority

### Immediate (This Week):
1. ✅ **Phase 1a**: Implement 6×16 register microkernel
   - Expected: 2,000-3,000 GOPS (4-6× current)
   - Critical path: Register blocking is the biggest win

2. ✅ **Phase 1b**: Benchmark and validate
   - Confirm speedup matches expectations
   - Profile to identify next bottleneck

### Short-term (Next Week):
3. **Phase 2**: Data packing
   - Expected: 4,000-5,000 GOPS total (2× on top of Phase 1)
   - Prerequisite for effective prefetching

4. **Phase 3**: Explicit prefetching
   - Expected: 5,000-6,000 GOPS total
   - Quick win once packing is done

### Medium-term (2-3 Weeks):
5. **Phase 4**: 3-level cache blocking
   - Expected: 6,000-6,500 GOPS total
   - Should reach OneDNN parity

### Optional (Future):
6. **Phase 5**: JIT assembly
   - Only if Phases 1-4 fall short of target
   - Diminishing returns vs implementation cost

---

## Performance Milestones

| Phase | Expected GOPS | Efficiency | vs Target | Cumulative Effort |
|-------|---------------|------------|-----------|-------------------|
| **Current** | 472 | 5.99% | 14.0× behind | - |
| **Phase 1** | 2,500 | 31.7% | 2.6× behind | 2-3 days |
| **Phase 2** | 4,500 | 57.1% | 1.5× behind | 5-6 days |
| **Phase 3** | 5,500 | 69.8% | 1.2× behind | 6-7 days |
| **Phase 4** | 6,500 | 82.5% | 1.0× (GOAL!) | 9-10 days |
| **Phase 5** | 7,000 | 88.8% | Exceeds target | 13-15 days |

---

## Risk Assessment

### High Risk:
- **Q8_0 format incompatibility**: Our quantization format may not map cleanly to OneDNN's packing
  - **Mitigation**: Design packing that preserves per-block scales
  - **Fallback**: Convert to uniform quantization (sacrifice quality for speed)

### Medium Risk:
- **Compiler optimization interference**: Intrinsics may not generate optimal code
  - **Mitigation**: Use `__attribute__((always_inline))`, `-march=native`, inspect assembly
  - **Fallback**: JIT assembly (Phase 5)

### Low Risk:
- **Cache model assumptions**: Block sizes tuned for specific CPU
  - **Mitigation**: Auto-tuning framework (detect cache sizes at runtime)
  - **Fallback**: Multiple pre-tuned configurations

---

## Validation Strategy

### Correctness:
1. **Unit tests**: Each phase has corresponding correctness test
2. **Parity tests**: Compare against naive implementation
3. **Numerical stability**: Monitor max absolute error (<1e-3)

### Performance:
1. **Microbenchmarks**: Isolated microkernel performance
2. **Integration benchmarks**: Full GEMM with packing overhead
3. **End-to-end**: Qwen inference latency/throughput

### Profiling Tools:
- **perf**: Cache miss rates, instruction counts
- **Intel VTune**: Bottleneck analysis (if available)
- **Manual instrumentation**: Cycle-accurate timing

---

## Resources

### OneDNN Source Files (Reference):
- **Core kernel**: `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp`
- **Packing**: `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_u8_copy_bn_kern_autogen.cpp`
- **BRGEMM**: `external/onednn/src/cpu/x64/brgemm/brgemm.hpp`

### Documentation:
- OneDNN Developer Guide: https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html
- Intel Intrinsics Guide: https://software.intel.com/sites/landingpage/IntrinsicsGuide/
- AVX512 VNNI: https://en.wikichip.org/wiki/x86/avx512_vnni

### Papers:
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn)
- "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (Van Zee & van de Geijn)

---

## Conclusion

Achieving OneDNN's 6,610 GOPS performance requires **fundamental architectural changes**:

1. **Register blocking** (Phase 1) is the most critical - provides 4-6× speedup alone
2. **Data packing** (Phase 2) enables sequential access patterns - doubles performance again
3. **Prefetching + cache blocking** (Phases 3-4) close the remaining gap

**Total estimated effort**: 9-10 days for Phases 1-4 to reach target.

**Key insight**: The 14× gap is not from micro-optimizations, but from **algorithmic differences** in how data flows through the cache hierarchy. OneDNN's architecture minimizes memory traffic through aggressive register reuse and cache-aware data layouts.

Our current approach (simple triple-loop with cache blocking) cannot close this gap without adopting OneDNN's core techniques.

---

**Next Steps**:
1. Implement Phase 1a (6×16 microkernel) - Start immediately
2. Benchmark Phase 1a - Validate 4-6× speedup
3. If Phase 1a successful, proceed to Phase 2 (packing)
4. If Phase 1a disappointing, profile and debug before continuing

**Success Criteria**:
- Phase 1: ≥2,000 GOPS (4× improvement)
- Phase 2: ≥4,000 GOPS (8× improvement)  
- Phase 4: ≥6,000 GOPS (13× improvement, within 10% of target)
