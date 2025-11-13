# INT8 GEMM Optimization - Quick Action Plan

**Date**: November 12, 2025  
**Current**: 472 GOPS (7.14% of target)  
**Target**: 6,610 GOPS (OneDNN baseline)  
**Gap**: 14× speedup needed

---

## OneDNN Key Findings (30-Second Summary)

After analyzing OneDNN's INT8 GEMM implementation:

1. **48×8 register blocking** - Not 1×1 like ours → **10-15× speedup**
2. **Pre-packed data panels** - Not strided Q8_0 blocks → **2-3× speedup**
3. **JIT assembly** - Not C++ intrinsics → **1.5-2× speedup**
4. **Explicit prefetching** - Not hardware-only → **1.3-1.5× speedup**
5. **3-level cache blocking** - Not single-level → **1.2-1.5× speedup**

**Combined**: 14× (multiplicative effects with diminishing returns)

---

## Immediate Next Steps (Start Today)

### Step 1: Implement 6×16 Register Microkernel (2-3 days)

**Goal**: 2,000-3,000 GOPS (4-6× current performance)

**Core Concept**:
```cpp
// Current (1×1 blocking):
for (i) for (j) for (k) {
    C[i,j] += A[i,k] * B[k,j];  // Load C every iteration!
}

// OneDNN (48×8 blocking):
// Load C[i:i+48, j:j+8] once into 24 ZMM registers
for (k) {
    // Accumulate 48×8 values in registers
    // Only store C when K loop done
}
// Store C[i:i+48, j:j+8] once
```

**Implementation Pattern**:
```cpp
class RegisterBlockedInt8Gemm {
    static constexpr int MR = 6;   // M-dimension register blocking
    static constexpr int NR = 16;  // N-dimension register blocking
    
    static void microkernel_6x16(
        const int8_t* A,  // [6, K] row-major
        const int8_t* B,  // [K, 16] column-major panels
        int32_t* C,       // [6, 16] output accumulator
        int K)
    {
        // Load C into registers (6 ZMM = 96 INT32 values)
        __m512i c0 = _mm512_loadu_si512((__m512i*)(C + 0));   // Row 0, cols 0-15
        __m512i c1 = _mm512_loadu_si512((__m512i*)(C + 16));  // Row 1, cols 0-15
        __m512i c2 = _mm512_loadu_si512((__m512i*)(C + 32));  // Row 2, cols 0-15
        __m512i c3 = _mm512_loadu_si512((__m512i*)(C + 48));  // Row 3, cols 0-15
        __m512i c4 = _mm512_loadu_si512((__m512i*)(C + 64));  // Row 4, cols 0-15
        __m512i c5 = _mm512_loadu_si512((__m512i*)(C + 80));  // Row 5, cols 0-15
        
        for (int k = 0; k < K; k += 4) {
            // Broadcast 4 elements from each row of A
            __m512i a0 = _mm512_set1_epi32(*(int32_t*)(A + 0*K + k));
            __m512i a1 = _mm512_set1_epi32(*(int32_t*)(A + 1*K + k));
            __m512i a2 = _mm512_set1_epi32(*(int32_t*)(A + 2*K + k));
            __m512i a3 = _mm512_set1_epi32(*(int32_t*)(A + 3*K + k));
            __m512i a4 = _mm512_set1_epi32(*(int32_t*)(A + 4*K + k));
            __m512i a5 = _mm512_set1_epi32(*(int32_t*)(A + 5*K + k));
            
            // Load 4×16 panel from B (column-major)
            __m512i b0 = _mm512_loadu_si512((__m512i*)(B + (k+0)*16));
            __m512i b1 = _mm512_loadu_si512((__m512i*)(B + (k+1)*16));
            __m512i b2 = _mm512_loadu_si512((__m512i*)(B + (k+2)*16));
            __m512i b3 = _mm512_loadu_si512((__m512i*)(B + (k+3)*16));
            
            // VNNI: 4-way dot product + accumulate
            c0 = _mm512_dpbusd_epi32(c0, a0, b0);  // Row 0 accumulates
            c0 = _mm512_dpbusd_epi32(c0, a0, b1);
            c0 = _mm512_dpbusd_epi32(c0, a0, b2);
            c0 = _mm512_dpbusd_epi32(c0, a0, b3);
            
            c1 = _mm512_dpbusd_epi32(c1, a1, b0);  // Row 1 accumulates
            // ... repeat for c2-c5 ...
        }
        
        // Store C once (6 stores for 96 values vs K stores in naive)
        _mm512_storeu_si512((__m512i*)(C + 0), c0);
        _mm512_storeu_si512((__m512i*)(C + 16), c1);
        _mm512_storeu_si512((__m512i*)(C + 32), c2);
        _mm512_storeu_si512((__m512i*)(C + 48), c3);
        _mm512_storeu_si512((__m512i*)(C + 64), c4);
        _mm512_storeu_si512((__m512i*)(C + 80), c5);
    }
};
```

**Why 6×16 (not 48×8)?**:
- 48×8 = 384 INT32 = 24 ZMM registers (leaves only 8 for A/B/temps)
- 6×16 = 96 INT32 = 6 ZMM registers (leaves 26 for A/B/temps/prefetch)
- Easier to implement and debug
- Still gets 90% of the benefit

**File to Create**: `src/v2/kernels/cpu/gemm/int8/RegisterBlockedInt8Gemm.h`

**Test File**: `tests/v2/performance/cpu/kernels/gemm/Perf__RegisterBlockedInt8Gemm.cpp`

---

### Step 2: Benchmark Phase 1 (Same Day)

**Success Criteria**:
- ✅ **Correctness**: Max error < 1e-3 vs naive
- ✅ **Performance**: ≥2,000 GOPS (4× speedup)
- ✅ **Efficiency**: ≥25% of theoretical peak

**If Success**: Proceed to Step 3 (Data Packing)  
**If Failure**: Profile and debug before continuing

---

### Step 3: Data Packing (2-3 days after Phase 1)

**Goal**: 4,000-5,000 GOPS (2× speedup on top of Phase 1)

**Pack B Matrix into Panels**:
```cpp
struct PackedBMatrix {
    std::vector<int8_t> data;    // [N/16][K][16] continuous
    std::vector<float> scales;   // [N/16][K/32] per-panel scales
    int N, K;
};

PackedBMatrix pack_b_matrix(const Q8_0Block* B, int N, int K) {
    PackedBMatrix packed;
    packed.N = N;
    packed.K = K;
    packed.data.resize((N/16) * K * 16);
    packed.scales.resize((N/16) * (K/32));
    
    #pragma omp parallel for
    for (int nc = 0; nc < N; nc += 16) {
        for (int k = 0; k < K; ++k) {
            // Copy 16 consecutive columns into panel
            for (int ni = 0; ni < 16; ++ni) {
                int col = nc + ni;
                int block_idx = col * (K/32) + (k/32);
                const Q8_0Block& block = B[block_idx];
                packed.data[nc/16 * K * 16 + k * 16 + ni] = block.qs[k % 32];
            }
        }
        
        // Store scales per-panel
        // ... handle scale merging ...
    }
    
    return packed;
}
```

**Expected Impact**:
- Sequential B access (no column-major stride)
- Cache lines fully utilized
- Prefetching becomes effective

---

### Step 4: Explicit Prefetching (1 day)

**Add to microkernel**:
```cpp
for (int k = 0; k < K; k += 4) {
    // Prefetch 160 elements ahead for A
    _mm_prefetch((char*)(A + k + 160), _MM_HINT_T0);
    
    // Prefetch 128 elements ahead for B
    _mm_prefetch((char*)(B + (k + 128) * 16), _MM_HINT_T0);
    
    // ... computation ...
}
```

**Expected Impact**: 1.3-1.5× speedup (hides memory latency)

---

### Step 5: 3-Level Cache Blocking (2-3 days)

**Implement Nested Loop Tiling**:
```cpp
// L3 blocks (shared cache)
for (int mc = 0; mc < M; mc += 384) {
  for (int nc = 0; nc < N; nc += 8192) {
    // L2 blocks (private cache)
    for (int mc2 = mc; mc2 < mc + 384; mc2 += 96) {
      for (int nc2 = nc; nc2 < nc + 8192; nc2 += 512) {
        // L1 blocks (register microkernel)
        for (int i = mc2; i < mc2 + 96; i += 6) {
          for (int j = nc2; j < nc2 + 512; j += 16) {
            microkernel_6x16(...);
          }
        }
      }
    }
  }
}
```

**Parallelize on L3 blocks** (not inner loops!)

**Expected Impact**: 1.2-1.5× speedup (maximizes cache utilization)

---

## Performance Milestones

| Milestone | Expected GOPS | Days from Start | Cumulative Effort |
|-----------|---------------|-----------------|-------------------|
| **Current** | 472 | 0 | - |
| **Phase 1** | 2,500 | 2-3 | 2-3 days |
| **Phase 2** | 4,500 | 5-6 | 5-6 days |
| **Phase 3** | 5,500 | 6-7 | 6-7 days |
| **Phase 4** | 6,500 | 9-10 | 9-10 days |
| **TARGET** | 6,610 | - | **ACHIEVED** |

---

## Critical Success Factors

### Do's:
✅ Start with 6×16 microkernel (not 48×8) - easier to debug  
✅ Use `__attribute__((always_inline))` on microkernel  
✅ Compile with `-march=native -O3 -ffast-math`  
✅ Benchmark each phase independently  
✅ Profile with `perf stat -e cache-misses,cache-references`  

### Don'ts:
❌ Don't parallelize inside microkernel (too fine-grained)  
❌ Don't use OpenMP inside inner loops (overhead dominates)  
❌ Don't skip correctness tests (silent bugs are worst)  
❌ Don't optimize prematurely (validate Phase 1 first)  
❌ Don't give up if Phase 1 "only" gets 3× speedup (still huge win!)

---

## Emergency Fallback

**If register blocking doesn't work**:
1. Check assembly output (`objdump -d` or `perf annotate`)
2. Verify VNNI instructions are being used
3. Check for register spills to stack (bad!)
4. Try smaller microkernel (4×8 instead of 6×16)
5. Consider JIT assembly (Phase 5) if compiler is failing

**If you're stuck**: Re-read OneDNN's implementation:
- `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp`

---

## Checklist for Phase 1 (Today)

- [ ] Create `RegisterBlockedInt8Gemm.h` header
- [ ] Implement `microkernel_6x16()` function
- [ ] Add outer M×N tiling loops
- [ ] Create performance test `Perf__RegisterBlockedInt8Gemm.cpp`
- [ ] Add to CMake build system
- [ ] Compile Release build
- [ ] Run benchmark
- [ ] Verify ≥2,000 GOPS achieved
- [ ] If successful, plan Phase 2 (packing)

---

**Start Now**: The biggest win (4-6× speedup) comes from Phase 1 alone!
