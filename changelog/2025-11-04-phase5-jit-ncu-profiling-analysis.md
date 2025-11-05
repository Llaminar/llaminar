# Phase 5 JIT Kernel NCU Profiling Analysis
**Date**: November 4, 2025  
**Kernel**: `iq4nl_gemm_phase5_kernel`  
**Configuration**: p5_64_64_64_sub16_mma2x2_buf2_thr128_swz333  
**Matrix Size**: 1024×896×896  
**Current Performance**: 8.86 TFLOPS (baseline achieved ✓)

## Executive Summary

Comprehensive Nsight Compute profiling reveals **significant optimization opportunities** beyond current 8.86 TFLOPS:

### Top 5 Optimization Opportunities (Ranked by Impact)

1. **🔴 CRITICAL: Uncoalesced Global Memory (78% waste)** → **Est. +15-25% speedup**
2. **🔴 CRITICAL: Shared Memory Bank Conflicts (83% conflicts)** → **Est. +10-20% speedup**  
3. **🟡 IMPORTANT: Low Occupancy (20% vs 33% theoretical)** → **Est. +5-10% speedup**
4. **🟡 IMPORTANT: Issue Slot Utilization (23% vs 100%)** → **Est. +5-15% speedup**
5. **🟢 MODERATE: Grid Too Small (0.7 waves)** → **Est. +3-7% speedup** (batch workloads)

**Cumulative Potential**: 38-77% performance improvement → **12-16 TFLOPS achievable**

---

## Detailed NCU Metrics

### Performance Characteristics

| Metric | Value | Analysis |
|--------|-------|----------|
| **Duration** | 248.89 μs | Baseline reference |
| **Throughput (Compute)** | 56.87% | **Memory-bound, not compute-bound** |
| **Throughput (Memory)** | 56.87% | Primary bottleneck |
| **SM Busy** | 24.88% | **Low utilization - opportunity!** |
| **Tensor Core Utilization** | 24.9% | Dominated by Tensor (FP) pipeline |

### Occupancy Analysis

```
Theoretical Occupancy:   33.33% (4 warps/scheduler max)
Achieved Occupancy:      20.41% (9.79 warps/SM actual)
Gap:                     -38.76% (underutilization)

Limiting Factors:
  ✗ Registers:      113 regs/thread (limits to 4 blocks/SM)
  ✗ Shared Memory:  24.58 KB/block (limits to 4 blocks/SM)
  ✓ Warps:          Could support 12 blocks/SM
```

**Analysis**: Both registers AND shared memory limit occupancy. Need to reduce one or both.

### Memory System Bottlenecks

#### 1. Global Memory Access Pattern

```
L1/TEX Hit Rate:             75.99% (good)
L2 Hit Rate:                 94.26% (excellent)
BUT:
  Global Load Efficiency:    21.56% (6.9 / 32 bytes used) ⚠️
  Global Store Efficiency:   50.00% (16.0 / 32 bytes used) ⚠️
  Excessive Sectors (loads): 7,038,976 / 9,060,352 (78% waste!)
```

**Root Cause**: Uncoalesced memory accesses - threads within warp access non-contiguous addresses.

**Impact**: 
- 78% of global memory bandwidth wasted
- ~3.6× more transactions than necessary
- **Major performance limiter**

#### 2. Shared Memory Bank Conflicts

```
Average Bank Conflicts:      7.3-way conflicts
Conflicted Requests:         3,024,872 / 3,638,576 (83.13%)
Excessive Wavefronts:        3,010,560 / 10,035,200 (30%)
```

**Root Cause**: Swizzle pattern (swz333) not fully eliminating conflicts for this tile size.

**Impact**:
- 7.3× serialization on shared memory stores
- 30% extra shared memory wavefronts
- **Significant performance limiter**

### Warp Scheduling Efficiency

```
Issue Slot Utilization:      22.82% (each scheduler issues 1 inst / 4.4 cycles)
Active Warps/Scheduler:      2.45 warps (out of 12 max)
Eligible Warps/Scheduler:    0.29 warps (ready to issue)
No Eligible Warp:            77.18% of cycles

Stall Reasons (inferred):
  - Memory latency (global/shared access)
  - Bank conflicts (shared memory serialization)
  - Low occupancy (insufficient warps to hide latency)
```

**Analysis**: 77% of cycles have no warp ready to issue → severe underutilization.

### Workload Distribution

```
Grid Size:              224 blocks (16×14×1)
SMs Available:          82 SMs
Waves per SM:           0.68 waves (224 / 82 = 2.73 blocks, but uneven)
Workload Imbalance:     -19% (min) to +7.5% (max) SM active cycles
```

**Analysis**: Small grid causes load imbalance. Not critical for single-batch, but batching would help.

---

## Optimization Roadmap

### Phase 1: Fix Memory Access Patterns (High Impact, Medium Effort)

#### 1.1 Global Memory Coalescing

**Problem**: Only 21.56% load efficiency, 50% store efficiency.

**Diagnosis Steps**:
```bash
# Generate source-level memory access report
sudo /usr/local/cuda/bin/ncu --set full \
  --section SourceCounters \
  --section MemoryWorkloadAnalysis \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
```

**Likely Issues**:
1. **IQ4_NL block dequantization**: Non-contiguous access to quantized blocks
2. **Tensor Core fragment loading**: Strided access to shared memory
3. **Output epilogue**: Bank conflicts on reduction

**Fixes**:
```cpp
// BEFORE (strided access - uncoalesced)
for (int k_offset = 0; k_offset < K; k_offset += BLOCK_K) {
    const IQ4_NLBlock& block = B_blocks[row * blocks_per_row + k_offset/32];
    // Each thread accesses different row → stride = blocks_per_row
}

// AFTER (contiguous access - coalesced)
// Transpose layout or use vectorized loads
float4* block_ptr = reinterpret_cast<float4*>(&B_blocks[...]);
float4 data = block_ptr[threadIdx.x];  // Coalesced 16-byte load
```

**Expected Gain**: +15-25% (reduce excessive sectors from 78% to <20%)

#### 1.2 Shared Memory Bank Conflict Resolution

**Problem**: 7.3-way conflicts, 83% of stores serialized.

**Diagnosis**:
```bash
# Check swizzle pattern effectiveness
CUDA_JIT_DEBUG_DUMP=1 ./v2_test_phase5_parity ...
# Examine generated smem layout in debug_source_*.cu
```

**Current Swizzle**: `swz333` (3-bit XOR on B, C, D dims)

**Investigation**:
1. Profile without swizzle: `swz000` (baseline)
2. Try alternative swizzles: `swz222`, `swz444`, `swz555`
3. Analyze conflict pattern per tile size

**Potential Fix**:
```cpp
// Current: 3-bit swizzle may be insufficient for 64×64 tile
using SmemLayoutB = decltype(
    composition(Swizzle<3,3,3>{}, ...)  // 3-bit swizzle
);

// Try: 4-bit or 5-bit swizzle
using SmemLayoutB = decltype(
    composition(Swizzle<4,4,4>{}, ...)  // 4-bit swizzle (more aggressive)
);
```

**Expected Gain**: +10-20% (reduce conflicts from 83% to <30%)

### Phase 2: Improve Occupancy (Medium Impact, Medium Effort)

#### 2.1 Register Pressure Reduction

**Current**: 113 regs/thread → limits to 4 warps/scheduler (33% occupancy)

**Target**: <96 regs/thread → allows 6 warps/scheduler (50% occupancy)

**Techniques**:
1. **Reduce accumulator footprint**:
   ```cpp
   // BEFORE: Separate accumulators per MMA
   float acc_mma0[...], acc_mma1[...];
   
   // AFTER: Reuse accumulators
   float acc[...];  // Unified accumulator
   ```

2. **Spill to shared memory**:
   ```cpp
   // Store intermediate results in smem instead of registers
   __shared__ float temp_storage[...];
   ```

3. **Compiler hints**:
   ```cpp
   #pragma unroll 2  // Reduce unroll factor
   ```

**Expected Gain**: +5-10% (increase occupancy 33% → 50%)

#### 2.2 Shared Memory Reduction

**Current**: 24.58 KB/block → limits to 4 blocks/SM

**Target**: <18 KB/block → allows 5 blocks/SM (better occupancy)

**Techniques**:
1. **Reduce buffering**: Currently using 2 buffers (buf2), try 1 buffer
2. **Tile size adjustment**: 64×64 → 32×64 or 64×32
3. **Reuse epilogue smem for mainloop**

**Trade-off**: May reduce register reuse efficiency. Profile both.

**Expected Gain**: +3-5% (marginal occupancy improvement)

### Phase 3: Workload Distribution (Low-Medium Impact, Low Effort)

#### 3.1 Batch Processing

**Current**: Grid = 224 blocks → 0.68 waves/SM (underutilized)

**Target**: Grid = 2000+ blocks → 6+ waves/SM

**Implementation**:
```cpp
// Batch multiple GEMM operations
for (int batch = 0; batch < BATCH_SIZE; ++batch) {
    gemm_jit(A[batch], B, C[batch], ...);
}
```

**Use Cases**:
- Multi-head attention (16-64 heads)
- Batch inference (decode multiple sequences)
- Pipeline parallelism (multiple layers)

**Expected Gain**: +5-10% (better SM utilization, amortize kernel launch)

#### 3.2 Persistent Kernel

**Idea**: Grid = SMs, each block processes multiple output tiles.

```cpp
__global__ void persistent_gemm(...) {
    int block_id = blockIdx.x;
    for (int tile = block_id; tile < num_tiles; tile += gridDim.x) {
        // Process tile
    }
}
```

**Benefits**:
- Perfect load balancing
- Reduced kernel launch overhead
- Better instruction cache reuse

**Expected Gain**: +3-7% (eliminate workload imbalance)

### Phase 4: Instruction-Level Optimizations (Low Impact, High Effort)

#### 4.1 Fused Multiply-Add (FMA)

**Current**: 0 fused FP32, 401,408 non-fused FP32

**Issue**: Compiler not generating FMAs (multiply + add → fused).

**Investigation**:
```bash
# Check PTX disassembly
cuobjdump --dump-ptx phase5_jit_profile.cubin
# Look for: fma.rn.f32 vs mul.f32 + add.f32
```

**Potential Fix**:
```cpp
// Ensure FMA-friendly code
float result = a * b + c;  // Should compile to FMA
// vs
float temp = a * b;
result = temp + c;  // May not fuse
```

**Expected Gain**: +1-2% (theoretical 50% FP32 improvement unlikely - Tensor Cores dominate)

#### 4.2 Predication Efficiency

**Current**: Avg 30.61/32 threads not predicated off (95.7% efficiency)

**Analysis**: Excellent predication efficiency. No action needed.

---

## Prioritized Implementation Plan

### Week 1: Quick Wins (Est. +20-30%)

1. **Day 1-2**: Global memory coalescing analysis
   - Run source-level profiling
   - Identify strided access patterns
   - Implement vectorized loads (float4)

2. **Day 3-4**: Shared memory bank conflict tuning
   - Test swizzle variations (swz000, swz222, swz444, swz555)
   - Profile each configuration
   - Select optimal swizzle

3. **Day 5**: Validate and benchmark
   - Run parity tests
   - Measure performance gains
   - Document findings

**Target**: 8.86 → 10.5-11.5 TFLOPS

### Week 2: Occupancy Improvements (Est. +10-15%)

1. **Day 1-3**: Register pressure reduction
   - Analyze register allocation (cuobjdump)
   - Reduce accumulator footprint
   - Tune unroll factors

2. **Day 4-5**: Shared memory optimization
   - Test single buffering (buf1)
   - Experiment with tile sizes
   - Profile occupancy changes

**Target**: 10.5-11.5 → 11.5-13.0 TFLOPS

### Week 3: Advanced Optimizations (Est. +5-10%)

1. **Day 1-2**: Batch processing infrastructure
   - Implement batched kernel interface
   - Test with multi-head attention workload

2. **Day 3-5**: Persistent kernel prototype
   - Implement grid-stride loop
   - Tune work distribution
   - Compare vs standard approach

**Target**: 11.5-13.0 → 12.0-14.0 TFLOPS

---

## Measurement Protocol

### Before Each Optimization

```bash
# 1. Baseline performance (3 runs)
cd /workspaces/llaminar/build_v2_release/tests/v2
for i in {1..3}; do
    ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"
done

# 2. Full NCU profile
sudo /usr/local/cuda/bin/ncu --set full \
  --export baseline_profile_$(date +%Y%m%d_%H%M%S) \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"

# 3. Source-level metrics
sudo /usr/local/cuda/bin/ncu --set full \
  --section SourceCounters \
  --section MemoryWorkloadAnalysis \
  --page raw \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config" \
  > baseline_source_metrics.txt
```

### After Each Optimization

```bash
# 1. Performance regression test
./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.*"

# 2. Compare metrics
sudo /usr/local/cuda/bin/ncu --set full \
  --export optimized_profile_$(date +%Y%m%d_%H%M%S) \
  ./v2_test_phase5_parity --gtest_filter="Phase5ParityTest.Phase5A_Baseline_Config"

# 3. Diff analysis
ncu-ui  # Load both profiles, compare side-by-side
```

### Success Criteria

| Metric | Baseline | Target (Week 1) | Target (Week 2) | Target (Week 3) |
|--------|----------|-----------------|-----------------|-----------------|
| **Throughput** | 8.86 TFLOPS | 10.5 TFLOPS | 11.5 TFLOPS | 12.5 TFLOPS |
| **Global Load Eff** | 21.56% | >60% | >70% | >80% |
| **Global Store Eff** | 50.00% | >80% | >90% | >95% |
| **Bank Conflicts** | 83% | <50% | <30% | <20% |
| **Occupancy** | 20.4% | 20-25% | 30-35% | 35-40% |
| **Issue Slot Util** | 22.8% | 25-30% | 30-35% | 35-40% |

---

## Risk Analysis

### High Risk

1. **Memory coalescing changes may break correctness**
   - Mitigation: Comprehensive parity testing after each change
   - Rollback plan: Git branches for each optimization

2. **Swizzle tuning may degrade other tile sizes**
   - Mitigation: Profile multiple configurations (16×16, 32×32, 64×64, 128×128)
   - Fallback: Per-config swizzle selection in autotuner

### Medium Risk

3. **Register reduction may increase spilling**
   - Mitigation: Monitor L1 traffic, check for reg spills in SASS
   - Threshold: Accept 5% spill if occupancy gain >10%

4. **Single buffering may expose pipeline bubbles**
   - Mitigation: Profile critical path, ensure gmem latency hidden
   - Fallback: Keep dual buffering for large tiles

### Low Risk

5. **Batching infrastructure overhead**
   - Mitigation: Measure per-call overhead, ensure <1% for batch>8

---

## Appendix: Key NCU Metrics

### Compute Workload

```
Executed IPC (Active):       0.91 inst/cycle
Executed IPC (Elapsed):      0.84 inst/cycle
Issued IPC:                  0.91 inst/cycle
Branch Efficiency:           100.00% (perfect!)
```

### Memory Workload

```
Memory Throughput:           28.85 GB/s
DRAM Throughput:             3.17% (excellent L2 hit rate)
L1/TEX Throughput:           62.03%
L2 Throughput:               13.88%
```

### Launch Configuration

```
Block Size:                  128 threads
Grid Size:                   224 blocks
Shared Memory (Static):      24.58 KB/block
Shared Memory (Dynamic):     0 bytes
Registers/Thread:            113 regs
Waves/SM:                    0.68 (low!)
```

### Roofline Analysis

```
FP32 Peak Utilization:       ~0% (misleading - Tensor Cores dominate)
FP64 Peak Utilization:       0%
Note: Tensor Core throughput not shown in roofline - actual compute is TF32
```

---

## Conclusion

Current Phase 5 JIT kernel achieves **8.86 TFLOPS** with **clean parity** to baseline. NCU profiling reveals:

1. ✅ **Tensor Core utilization is good** (24.9% of active cycles)
2. ✅ **Cache hierarchy is efficient** (94% L2 hit, 76% L1 hit)
3. ✅ **No divergence or predication issues**

**BUT**:

4. ❌ **78% global memory bandwidth wasted** (uncoalesced access)
5. ❌ **83% shared memory bank conflicts** (poor swizzle for this tile)
6. ❌ **77% cycles with no eligible warp** (low occupancy + memory latency)

**Fixing these issues should yield 12-16 TFLOPS** (35-80% improvement), approaching theoretical peak for this workload.

**Next Steps**:
1. Run source-level profiling to identify exact uncoalesced access locations
2. Implement global memory coalescing (vectorized loads, layout changes)
3. Tune swizzle pattern for 64×64 tile size
4. Iterate with NCU profiling after each optimization

---

**Generated**: November 4, 2025  
**Profiling Tool**: NVIDIA Nsight Compute 2024.3  
**Report Location**: `phase5_jit_profile.ncu-rep`
