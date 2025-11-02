# CUDA GEMM Kernel Optimization Roadmap

**Date**: November 1, 2025  
**Current Performance**: 3010 GFLOPS (8.46% of RTX 3090 peak)  
**Target Performance**: 12,000-15,000 GFLOPS (33-42% of peak)  
**Status**: 🔴 **CRITICAL PERFORMANCE ISSUE** - 4-5× improvement needed

---

## Executive Summary

The current CUDA GEMM kernel achieves only **8.46% of theoretical peak** due to:
1. ❌ **No Tensor Core utilization** (3-4x potential speedup)
2. ❌ **Poor memory coalescing** (2-3x potential speedup)  
3. ❌ **No microkernel/register tiling** (1.5-2x potential speedup)
4. ❌ **Catastrophic single-token performance** (0.06% of peak, 4% GPU utilization)

**Optimization plan**: 3-phase approach targeting 4-5× total speedup.

---

## Performance Baseline

### Current Performance (October 31, 2025)

| Workload | Shape | Best GFLOPS | % of Peak | Arithmetic Intensity |
|----------|-------|-------------|-----------|----------------------|
| **Large Batch** (best case) |
| Qwen_14B_Batch256_QKV | 256×5120×5120 | 3010 | 8.46% | 116 FLOPs/byte |
| Qwen_7B_Batch128_QKV | 128×4096×4096 | 2264 | 6.36% | 60 FLOPs/byte |
| Qwen_4B_Batch128_QKV | 128×2560×2560 | 2532 | 7.12% | 58 FLOPs/byte |
| **Medium Batch** |
| Qwen_0_5B_Batch32_QKV | 32×896×896 | 585 | 1.64% | 15 FLOPs/byte |
| **Single-Token** (catastrophic) |
| Qwen_7B_SingleToken_QKV | 1×4096×4096 | 45.5 | 0.13% | 0.50 FLOPs/byte |
| Qwen_0_5B_SingleToken_QKV | 1×896×896 | 22.7 | 0.06% | 0.50 FLOPs/byte |

**Key Findings**:
- ✅ Fused dequant+GEMM working (no separate dequant kernel)
- ❌ Memory bandwidth limited (0.5 FLOPs/byte for m=1)
- ❌ Compute limited (large batches not saturating GPU)
- ❌ Grid size limited (single-token only uses ~4% of SMs)

### Competitor Benchmarks

**llama.cpp CUDA backend** (Q4_0 quantization, similar to IQ4_NL):
- RTX 3090: **~15,000 GFLOPS** for batch GEMM (**42% of peak**)
- Uses Tensor Cores (FP16/BF16 mixed precision)
- Optimized memory coalescing and microkernel design

**llaminar gap**: **5× slower** than llama.cpp

---

## Root Cause Analysis

### Issue 1: No Tensor Core Utilization ❌

**Current**: FP32 scalar FFMA instructions
```cuda
#pragma unroll
for (int wm = 0; wm < WORK_M; ++wm) {
    #pragma unroll
    for (int wn = 0; wn < WORK_N; ++wn) {
        acc[wm][wn] += a_frag[wm] * b_frag[wn];  // Scalar FP32 FMA
    }
}
```

**Theoretical throughput**:
- FP32 FMA: 35.58 TFLOPS (achieved: 3.01 TFLOPS, 8.46%)

**Tensor Core potential** (RTX 3090 has 328 2nd-gen Tensor Cores):
- FP16 input, FP32 accumulate: **~140 TFLOPS** (4× FP32)
- With mixed precision losses: **~100 TFLOPS** realistic
- **Target**: 30-40% of Tensor Core peak = **30,000-40,000 GFLOPS**

**Impact**: **3-4× speedup** (8% → 24-32% of FP32 peak, or 6-8% of Tensor peak)

### Issue 2: Poor Memory Coalescing ❌

**Current load pattern**:
```cuda
for (int load_idx = 0; load_idx < A_LOADS_PER_THREAD; ++load_idx) {
    const int flat_idx = tid * A_LOADS_PER_THREAD + load_idx;
    const int a_row = flat_idx / TILE_K;  // Division in loop!
    const int a_col = flat_idx % TILE_K;  // Modulo in loop!
    val = A[global_row * k + global_col];  // Non-coalesced
}
```

**Problems**:
1. Division/modulo per iteration (expensive on GPU)
2. No guarantee of coalesced 128-byte transactions
3. No vectorized loads (float4 would load 16 bytes vs 4 bytes per instruction)

**Optimal pattern** (coalesced vectorized):
```cuda
// Reorganize so adjacent threads load adjacent addresses
const int a_col_base = threadIdx.x * 4;  // Each thread loads float4
const int a_row = threadIdx.y + blockIdx.y * WORK_M;
float4 vals = reinterpret_cast<const float4*>(&A[a_row * k + a_col_base])[0];
// Now 32 threads load 128 bytes in single transaction (coalesced)
```

**Impact**: **2-3× speedup** on memory-bound kernels (single-token)

### Issue 3: No Microkernel (Poor ILP) ⚠️

**Current**: 1D unrolling over K with immediate accumulate
```cuda
for (int k_idx = 0; k_idx < TILE_K; ++k_idx) {
    // Load A/B fragments
    // Immediate FMA (no pipelining)
    acc[wm][wn] += a_frag[wm] * b_frag[wn];
}
```

**Problem**: Load → FMA → Load → FMA (serial, no instruction overlap)

**Microkernel solution**: 2D tiling + software pipelining
```cuda
// 8×8 microkernel with 4-stage pipeline
for (int k_tile = 0; k_tile < TILE_K; k_tile += 8) {
    // Load next 8×8 tile into registers (async)
    load_microkernel_tile(k_tile);
    
    // Compute previous 8×8 tile (8×8 = 64 FMAs)
    #pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
        // 8×8 outer product (fully unrolled, high ILP)
    }
}
```

**Impact**: **1.5-2× speedup** via instruction-level parallelism

### Issue 4: Single-Token Grid Size ❌

**Root cause**: Grid = (1, ceil(n/TILE_N), 1)
- Example: n=896, TILE_N=64 → 14 thread blocks
- RTX 3090: 82 SMs → only 14 active → **17% SM utilization**
- With 64 threads/block → only 896 threads active (out of 125,952 max)
- **GPU utilization**: **0.7%**

**Solutions**:
1. **N-slicing**: Reduce TILE_N for m=1 (64 → 16), increase grid size 4×
2. **Warp-specialized kernel**: 1 warp per output row (32 threads compute 896 outputs)
3. **Persistent threads**: Reuse threads across multiple rows (wavefront pattern)

**Impact**: **5-10× speedup** for single-token (still memory-bound)

### Issue 5: Shared Memory Bank Conflicts ⚠️

**Current**: `TRANSPOSE_SMEM` flag does nothing
```cuda
if constexpr (TRANSPOSE_SMEM) {
    s_A[buffer_idx][a_row][a_col] = val;  // Same as below!
} else {
    s_A[buffer_idx][a_row][a_col] = val;
}
```

**Problem**: Reading `s_A[tid_m * WORK_M + wm][k_idx]` with stride-1 in k causes 2-way bank conflicts when multiple threads in same warp access same k_idx.

**Solution**: Pad shared memory or true transpose
```cuda
__shared__ float s_A[NUM_BUFFERS][TILE_M][TILE_K + 1];  // +1 padding avoids conflicts
// OR
s_A[buffer_idx][a_col][a_row] = val;  // Transpose layout
```

**Impact**: **1.2-1.5× speedup** on large tiles

### Issue 6: Redundant Dequantization ❌

**Current**: Dequant happens every kernel launch
```cuda
// B weights are dequantized EVERY time for batch processing
decoder.decode_block(block_ptr, decoded);
```

**Problem**: For batch=256, same weights dequantized 256 times!

**Solutions**:
1. **Persistent dequant cache**: Dequant once, store in L2 (6 MB on RTX 3090)
2. **JIT compiled kernel**: Embed dequantized values as constants
3. **Separate dequant pass**: Pre-dequant to temp buffer, use standard FP16 GEMM

**Impact**: **1.5-2× speedup** for large batches (amortizes dequant cost)

---

## Optimization Phases

### Phase 1: Memory Optimization (Target: 2-3× speedup)

**Goal**: Fix memory access patterns to achieve better coalescing

**Tasks**:
1. ✅ **Coalesced loads**: Reorganize A/B loading for adjacent threads → adjacent addresses
   - Change flat_idx pattern to guarantee coalescing
   - Remove division/modulo from loop
   - **Expected**: +30-50% on memory-bound kernels

2. ✅ **Vectorized loads**: Use float4 for aligned accesses
   - Check alignment conditions at compile time
   - Fall back to scalar for unaligned
   - **Expected**: +20-30% on coalesced loads

3. ✅ **Shared memory padding**: Add +1 to TILE_K to avoid bank conflicts
   ```cuda
   __shared__ float s_A[NUM_BUFFERS][TILE_M][TILE_K + 1];
   __shared__ float s_B[NUM_BUFFERS][TILE_N][TILE_K + 1];
   ```
   - **Expected**: +10-20% on large tiles

4. ✅ **Implement TRANSPOSE_SMEM**: True transpose layout for bank conflict avoidance
   ```cuda
   if constexpr (TRANSPOSE_SMEM) {
       s_A[buffer_idx][a_col][a_row] = val;  // Swap indices
   }
   ```
   - **Expected**: +15-25% when enabled

**Phase 1 Target**: **6,000-9,000 GFLOPS** (17-25% of FP32 peak)

### Phase 2: Compute Optimization - Tensor Cores (Target: 3-4× speedup)

**Goal**: Utilize Tensor Cores for 3-4× compute throughput

**Background**: RTX 3090 (Ampere, SM_86) supports:
- `wmma` (Warp Matrix Multiply Accumulate) - high-level API
- `mma.sync` - low-level PTX instructions for finer control

**Tasks**:
1. ✅ **Add FP16 accumulator path**:
   ```cuda
   template<typename AccumType = float>  // or __half for FP16
   __global__ void quantized_gemm_kernel_wmma(...) {
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, AccumType> acc_frag;
       
       // Load 16×16 tiles into fragments
       wmma::load_matrix_sync(a_frag, s_A_fp16, TILE_K);
       wmma::load_matrix_sync(b_frag, s_B_fp16, TILE_K);
       
       // Tensor Core matrix multiply
       wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);  // 1 instruction, 256 FMAs!
   }
   ```

2. ✅ **Mixed precision pipeline**:
   - Dequant to FP16 (not FP32)
   - Compute in FP16 (Tensor Cores)
   - Accumulate in FP32 (numerical stability)
   - **Expected**: 3-4× throughput increase

3. ✅ **Warp-level synchronization**: Replace thread-level with warp-level
   - 16×16×16 wmma tiles (1 tile per warp)
   - Reduce shared memory pressure
   - **Expected**: Better occupancy

**Phase 2 Target**: **12,000-15,000 GFLOPS** (33-42% of FP32 peak, ~10% of Tensor peak)

### Phase 3: Advanced Optimization (Target: 1.5-2× speedup)

**Goal**: Further optimizations for state-of-the-art performance

**Tasks**:
1. ⚠️ **Async copy (`cp.async`)**:
   ```cuda
   // Ampere (SM_80+) async memory copy
   __pipeline_memcpy_async(&s_A[buffer][row][col], &A_global[idx], sizeof(float4));
   __pipeline_commit();
   __pipeline_wait_prior(1);  // Wait for previous stage
   ```
   - Overlaps memory copy with compute
   - **Expected**: +15-25% via latency hiding

2. ⚠️ **Software pipelining**: Double/triple buffering with async
   - Load tile N+1 while computing tile N
   - **Expected**: +10-20% on large tiles

3. ⚠️ **Persistent dequant cache**:
   - Store dequantized weights in L2 (6 MB)
   - Reuse across batch dimension
   - **Expected**: +50-100% for large batches

4. ⚠️ **Separate single-token kernel**:
   - m=1 specialized with N-slicing or warp-specialized
   - **Expected**: 5-10× speedup for single-token

**Phase 3 Target**: **18,000-25,000 GFLOPS** (50-70% of FP32 peak, ~15-18% of Tensor peak)

---

## Implementation Plan

### Week 1: Phase 1 - Memory Optimization

**Day 1-2**: Coalesced loads
- [ ] Reorganize A/B loading pattern for coalescing
- [ ] Add coalescing test (verify 128-byte transactions)
- [ ] Benchmark: Expect +30-50% for memory-bound

**Day 3-4**: Vectorized loads + shared memory
- [ ] Implement float4 loads with alignment checks
- [ ] Add shared memory padding (+1 to avoid bank conflicts)
- [ ] Implement true TRANSPOSE_SMEM
- [ ] Benchmark: Expect +40-60% cumulative

**Day 5**: Validation and tuning
- [ ] Run full benchmark suite
- [ ] Verify no regressions
- [ ] Update autotuner with new configs
- [ ] **Target**: 6,000+ GFLOPS on large batches

### Week 2: Phase 2 - Tensor Cores

**Day 1-2**: wmma infrastructure
- [ ] Add wmma-based kernel variant
- [ ] Implement FP16 dequant path (currently only FP32)
- [ ] Handle mixed precision accumulation

**Day 3-4**: Integration and tuning
- [ ] Add wmma configs to autotuner
- [ ] Tune tile sizes for Tensor Cores (16×16×16 multiples)
- [ ] Benchmark vs Phase 1

**Day 5**: Validation
- [ ] Numerical validation (FP16 vs FP32 comparison)
- [ ] Performance validation (expect 3-4× over Phase 1)
- [ ] **Target**: 12,000+ GFLOPS on large batches

### Week 3: Phase 3 - Advanced (Optional)

**Day 1-2**: Async copy
- [ ] Implement `cp.async` for SM_80+
- [ ] Add pipeline stages (double/triple buffering)

**Day 3-4**: Single-token optimization
- [ ] Implement N-slicing for m=1
- [ ] Add warp-specialized kernel variant

**Day 5**: Final validation
- [ ] Full benchmark sweep
- [ ] Compare vs llama.cpp
- [ ] **Target**: Match or exceed llama.cpp (15,000 GFLOPS)

---

## Success Metrics

### Performance Targets

| Milestone | Large Batch (256×5120×5120) | Medium Batch (32×896×896) | Single-Token (1×896×896) |
|-----------|----------------------------|---------------------------|--------------------------|
| **Baseline** | 3010 GFLOPS (8.46%) | 585 GFLOPS (1.64%) | 22.7 GFLOPS (0.06%) |
| **Phase 1** | 6,000-9,000 (17-25%) | 1,200-1,800 (3-5%) | 50-100 (0.14-0.28%) |
| **Phase 2** | 12,000-15,000 (33-42%) | 2,400-3,000 (7-8%) | 100-200 (0.28-0.56%) |
| **Phase 3** | 18,000-25,000 (50-70%) | 3,600-5,000 (10-14%) | 500-1,000 (1.4-2.8%) |

### Validation

- ✅ **Correctness**: All outputs match FP32 baseline within 1e-3 relative error
- ✅ **Performance**: Phase 2 matches or exceeds llama.cpp
- ✅ **Efficiency**: Large batches achieve 30%+ of FP32 peak (or 10%+ of Tensor peak)
- ✅ **Autotuner**: ML heuristic correctly ranks new variants

---

## Risk Mitigation

### Risk 1: FP16 Numerical Instability

**Mitigation**: Mixed precision (FP16 compute, FP32 accumulate)
- Tensor Cores support FP32 accumulators natively
- Keep critical paths (RMSNorm, Softmax) in FP32

### Risk 2: Tensor Core Alignment Requirements

**Constraint**: Tensor Cores require 16×16×16 aligned tiles

**Mitigation**: 
- Add padding for non-aligned shapes
- Fall back to FP32 path for small/odd dimensions
- Autotuner learns when to use wmma vs scalar

### Risk 3: Increased Code Complexity

**Mitigation**:
- Keep scalar FP32 path as reference
- Add extensive validation tests
- Gradual rollout (Phase 1 → Phase 2 → Phase 3)

---

## Next Steps

**IMMEDIATE** (this session):
1. Create Phase 1 implementation branch
2. Implement coalesced load pattern
3. Add shared memory padding
4. Run quick benchmark to verify 2× speedup

**WEEK 1**:
1. Complete Phase 1 (memory optimization)
2. Achieve 6,000+ GFLOPS on large batches
3. Update autotuner with new variants

**WEEK 2**:
1. Implement wmma Tensor Core path
2. Achieve 12,000+ GFLOPS
3. Match llama.cpp performance

---

**Status**: 🔴 **READY TO START** - Optimization roadmap complete, implementation can begin
