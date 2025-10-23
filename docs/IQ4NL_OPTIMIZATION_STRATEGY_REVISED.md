# IQ4_NL GEMM Optimization Strategy - Revised After Failures

**Date**: October 22, 2025  
**Baseline**: Q-Projection 4096 = 410 GFLOPS (FP32), 228 GFLOPS (BF16)

## Failed Attempts - Lessons Learned

### ❌ OpenMP Optimization
- **Attempted**: `#pragma omp parallel for` on outer loop
- **Result**: 394 → 219 GFLOPS (-44% regression)
- **Why**: Thread-local storage semantics, synchronization overhead
- **Lesson**: OpenMP on small-M matrix multiply is counterproductive

### ❌ Fused Dequant+GEMM Kernel
- **Attempted**: Single-pass dequantize + dot product
- **Result**: 410 → 44 GFLOPS (-89% regression!)
- **Why**: Still requires serial nibble extraction, breaks compiler opts, targets non-bottleneck
- **Lesson**: 128-byte L1 buffers are free; focus on real bottlenecks

## Profiling Reality Check

From our testing:

| Component | % of Time | Optimization Potential |
|-----------|-----------|------------------------|
| OpenMP overhead | ~30% | ❌ Already tried, failed |
| Dot product computation | 18-20% | ✅ Viable target |
| Dequantization | 4-5% | ⚠️ Limited headroom |
| Buffer operations | <1% | ❌ Not worth it (tried, failed) |

## Viable Optimization Paths

### 1. **Dot Product Micro-Optimizations** ⭐ (HIGH PRIORITY)

**Current Implementation**:
```cpp
static inline float dot_product_simd(const float* a, const float* b, size_t n) {
    __m512 acc = _mm512_setzero_ps();
    for (size_t i = 0; i < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    return _mm512_reduce_add_ps(acc);
}
```

**Bottleneck**: Long dependency chain (each FMA depends on previous accumulator)

**Optimization A: Multiple Accumulators** (Expected: +10-15%)
```cpp
static inline float dot_product_simd_4acc(const float* a, const float* b, size_t n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    
    for (size_t i = 0; i < n; i += 64) {  // Process 4× vectors per iteration
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    
    // Combine accumulators
    __m512 sum01 = _mm512_add_ps(acc0, acc1);
    __m512 sum23 = _mm512_add_ps(acc2, acc3);
    __m512 total = _mm512_add_ps(sum01, sum23);
    return _mm512_reduce_add_ps(total);
}
```

**Why This Works**:
- Breaks long dependency chain into 4 independent chains
- CPU can execute all 4 FMAs in parallel (ILP)
- FMA latency: ~4 cycles, throughput: 2/cycle on modern CPUs
- 4 accumulators saturate FMA units

**Risk**: Low - simple, well-understood optimization  
**Effort**: 30 mins implementation + testing  
**Expected Gain**: +10-15% (410 → 450-470 GFLOPS)

---

### 2. **Software Prefetching** ⭐ (MEDIUM PRIORITY)

**Problem**: CPU stalls waiting for next block's data

**Solution**: Prefetch next iteration's data while computing current
```cpp
for (size_t i = 0; i < n; i += 64) {
    // Prefetch next iteration (2 cache lines ahead)
    _mm_prefetch((const char*)(a + i + 128), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + i + 128), _MM_HINT_T0);
    
    // Compute current iteration
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    // ... rest of loop
}
```

**Risk**: Low - CPU ignores bad prefetch hints  
**Effort**: 15 mins  
**Expected Gain**: +3-8% (overlaps with multiple accumulators)

---

### 3. **Intel MKL Integration** ⭐ (HIGH PRIORITY, EASY WIN)

**Current**: Using OpenBLAS `cblas_sgemm`

**Alternative**: Intel MKL - likely better optimized for our AVX-512 hardware

**Test Without Rebuild**:
```bash
# Install MKL (if not present)
sudo apt install intel-mkl

# Override OpenBLAS with MKL
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libmkl_rt.so.2
./run_benchmark.sh benchmark_iq4nl_gemm
```

**Expected**: +10-25% if MKL has better tuned kernels for our shapes

**Risk**: Very low - just LD_PRELOAD test, no code changes  
**Effort**: 5 mins test, 30 mins integration if successful  
**Expected Gain**: +10-25% (410 → 450-510 GFLOPS)

---

### 4. **Profile-Guided Optimization (PGO)** ⭐ (MEDIUM PRIORITY)

**Idea**: Let compiler optimize based on actual runtime behavior

**Steps**:
```bash
# Build with instrumentation
cmake -B build_pgo -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fprofile-generate"
cmake --build build_pgo --target benchmark_iq4nl_gemm --parallel

# Run representative workload
./build_pgo/benchmark_iq4nl_gemm

# Rebuild with profile data
cmake -B build_pgo -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fprofile-use -fprofile-correction"
cmake --build build_pgo --target benchmark_iq4nl_gemm --parallel

# Test
./run_benchmark.sh build_pgo/benchmark_iq4nl_gemm
```

**Why This Works**:
- Better branch prediction
- Better inlining decisions
- Better code layout (hot paths together)

**Risk**: Low - fallback to non-PGO build if worse  
**Effort**: 1 hour  
**Expected Gain**: +5-12% (410 → 430-460 GFLOPS)

---

### 5. **Assembly Inspection + Manual Tuning** (LOW PRIORITY)

**Check Current Codegen**:
```bash
# Generate assembly
g++ -S -O3 -mavx512f -fverbose-asm -masm=intel \
  -I./src -o /tmp/iq4nl.s src/tensors/IQ4_NLTensor.h

# Inspect dot_product_simd
grep -A 50 "dot_product_simd" /tmp/iq4nl.s
```

**Look For**:
- Unnecessary `vzeroupper` (AVX-SSE transition penalties)
- Suboptimal register allocation
- Missed auto-vectorization opportunities

**Risk**: Medium - assembly is fragile  
**Effort**: 2-4 hours  
**Expected Gain**: +5-10% if we find inefficiencies

---

## Recommended Sequence

**Phase 1: Quick Wins** (2 hours total)
1. ✅ Test Intel MKL with LD_PRELOAD (5 mins)
2. ✅ If MKL wins, integrate it (30 mins)
3. ✅ Implement 4-accumulator dot product (30 mins)
4. ✅ Add software prefetching (15 mins)
5. ✅ Test combined (15 mins)

**Expected Result**: 410 → 500-550 GFLOPS (+22-34%)

**Phase 2: Compiler Optimizations** (1-2 hours)
6. ✅ Profile-Guided Optimization (1 hour)
7. ✅ Check assembly, tune if needed (1 hour)

**Expected Result**: 500-550 → 550-600 GFLOPS (cumulative +34-46%)

**Phase 3: Diminishing Returns** (3-5 hours)
8. ⚠️ Hand-tuned assembly for critical kernels
9. ⚠️ Custom allocator to reduce overhead
10. ⚠️ Block-level caching for m > 1 cases

**Expected Result**: 550-600 → 600-650 GFLOPS (cumulative +46-58%)

---

## Target Goals

| Metric | Current | Optimistic Target | Stretch Goal |
|--------|---------|-------------------|--------------|
| Q-Proj 4096 | 410 GFLOPS | 550 GFLOPS (+34%) | 650 GFLOPS (+58%) |
| FFN 512 | 505 GFLOPS | 650 GFLOPS (+29%) | 750 GFLOPS (+48%) |
| FFN 8192 | 387 GFLOPS | 500 GFLOPS (+29%) | 580 GFLOPS (+50%) |

## What NOT to Do (Learned the Hard Way)

❌ **Don't microoptimize L1 cache traffic** - 128-byte buffers are essentially free  
❌ **Don't add OpenMP to small matrix ops** - Synchronization overhead dominates  
❌ **Don't fuse unrelated operations** - Breaks compiler optimizations  
❌ **Don't optimize based on theory** - Always measure, many "obvious" wins fail  

## Success Criteria

- ✅ Each optimization tested independently
- ✅ Measure impact before combining
- ✅ Keep code readable (no "clever" tricks)
- ✅ Document what works AND what doesn't
- ✅ Revert immediately if regression detected

---

**Next Action**: Start with Intel MKL test (5 minutes, zero risk)
