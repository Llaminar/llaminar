# Phase 5 JIT Auto-Tuning Implementation Plan

**Date**: November 4, 2025  
**Author**: David Sanftenberg  
**Status**: Ready for implementation

---

## Executive Summary

After profiling Phase 5A, we discovered the real bottleneck is **shared memory limiting occupancy** (22.62% warp active, only 3 blocks/SM), not decode overhead or synchronization. 

**Solution**: Convert Phase 5 to JIT-compiled kernel with parameterized buffer stages, allowing us to sweep the configuration space and find the optimal occupancy/latency-hiding tradeoff.

---

## Profiling Findings (NCU Analysis)

### Phase 5A Current Performance
- **Throughput**: 8.86 TFLOPS (+1% vs Phase 4)
- **SM Utilization**: 26.34% (low!)
- **Tensor Core Utilization**: 6.29% (critically low!)
- **Warp Occupancy**: 22.62% (target: 50%+)
- **DRAM Bandwidth**: 3.49% (not the bottleneck)

### Root Cause: Shared Memory Occupancy Limit
```
launch__occupancy_limit_blocks:        16  (theoretical max)
launch__occupancy_limit_registers:      8  (register-limited)
launch__occupancy_limit_shared_mem:     3  (ACTUAL LIMIT!) ❌
```

**Analysis**:
- Each block uses 32 KB shared memory (2× 64×64×FP16 double-buffered tiles)
- A100 has 164 KB shared memory per SM
- 164KB / 32KB = **only 5 blocks/SM**, but registers limit to 3
- With only 3 concurrent blocks, we can't hide latency → Tensor Cores idle

**Trade-off Space**:
| Buffer Stages | Shared Memory | Occupancy (blocks/SM) | Latency Hiding | Tensor Core Util |
|---------------|---------------|-----------------------|----------------|------------------|
| 1 (single) | 16 KB | 10 blocks | Poor | Medium |
| 2 (double) | 32 KB | 5 blocks | Good | Low (current) |
| 3 (triple) | 48 KB | 3 blocks | Excellent | Very Low |

**Hypothesis**: Single buffering with larger tiles may outperform double buffering!

---

## JIT Auto-Tuning Architecture

### Configuration Parameters

```cpp
struct Phase5GemmConfig {
    // Tile dimensions
    int tile_m, tile_n, tile_k;       // 32, 64, 128, 256
    
    // Streaming granularity
    int sub_k;                         // 16, 32, 64, tile_k
    
    // CuTe atom layout
    int mma_m, mma_n;                  // 1×1, 2×2, 4×4
    
    // CRITICAL: Buffer stages
    int buffer_stages;                 // 1, 2, or 3
    
    // Derived
    int threads_per_block;             // 32, 128, 512
    int swizzle_b, swizzle_m, swizzle_s; // 3, 3, 3
};
```

### Configuration Space Size

**Full space**: ~2,000 configurations
- Tile sizes: 4 × 4 × 3 = 48 combinations
- Sub-K: 4 options × streaming factor
- MMA layouts: 3 options (1×1, 2×2, 4×4)
- Buffer stages: 3 options (1, 2, 3)
- After validation: ~800-1200 valid configs

**Focused space** (for quick iteration): 11 hand-picked configurations
- Single vs double buffering comparison
- Small vs large tiles
- Streaming vs no-streaming

### JIT Template Structure

**File**: `CudaGemmKernelTemplatePhase5.h`

**Placeholders**:
```cpp
const char* PHASE5_GEMM_KERNEL_TEMPLATE = R"(
    template <
        int TILE_M = ${TILE_M},           // e.g., 64
        int TILE_N = ${TILE_N},           // e.g., 64
        int TILE_K = ${TILE_K},           // e.g., 64
        int SUB_K = ${SUB_K},             // e.g., 16
        int MMA_M = ${MMA_M},             // e.g., 2
        int MMA_N = ${MMA_N},             // e.g., 2
        int BUFFER_STAGES = ${BUFFER_STAGES} // e.g., 1
    >
    __global__ void iq4nl_gemm_phase5_kernel(...) {
        __shared__ __half s_A[BUFFER_STAGES][TILE_M][TILE_K];
        __shared__ __half s_B_decoded[BUFFER_STAGES][TILE_N][TILE_K];
        // ... (rest of Phase 5A kernel logic)
    }
)";
```

### Compilation Pipeline

```
1. Generate config space (800-1200 configs)
2. For each valid config:
   a. Substitute template parameters
   b. JIT compile with NVRTC (~500ms first time)
   c. Cache compiled kernel (disk + memory)
3. Benchmark on real model shapes (Qwen 0.5B-72B)
4. Profile top 10 + worst 10 with NCU
5. Train ONNX model on profiling features
6. Runtime: ML heuristic selects best config (<1ms overhead)
```

---

## Expected Performance Improvements

### Hypothesis 1: Single Buffering + Larger Tiles
**Config**: 128×128×64, SUB_K=32, buffer_stages=1
- **Shared memory**: 16 KB (vs 32 KB current)
- **Occupancy**: 10 blocks/SM (vs 3 current)
- **Tensor Core util**: 15-20% (vs 6.29% current)
- **Expected gain**: +30-50% over current

**Trade-off**: Slightly higher decode overhead (not overlapped), but much better occupancy

### Hypothesis 2: Larger Tiles + Double Buffering
**Config**: 128×128×64, SUB_K=16, buffer_stages=2, MMA 1×1 (fewer threads)
- **Shared memory**: 32 KB
- **Occupancy**: 5 blocks/SM (registers may limit)
- **Tensor Core util**: 20-25%
- **Expected gain**: +40-60% if occupancy holds

**Trade-off**: Maximal work per block, decode overlap, but occupancy-limited

### Hypothesis 3: Extreme Tiles + Single Buffering
**Config**: 256×256×64, SUB_K=64 (no streaming), buffer_stages=1
- **Shared memory**: 32 KB (but huge tile)
- **Occupancy**: 5 blocks/SM
- **Tensor Core util**: 30-40% (massive tiles)
- **Expected gain**: +60-80% (if we don't run out of registers)

**Trade-off**: Maximum Tensor Core work, but may thrash cache or exceed register limits

---

## Implementation Plan

### Phase 1: JIT Infrastructure (1-2 hours)
- [x] Create `CudaGemmKernelTemplatePhase5.h` with CuTe kernel template
- [x] Create `Phase5ConfigSpace.h` with config generation
- [ ] Extend `CudaGemmJIT` to support Phase 5 template
- [ ] Add Phase 5 config substitution logic

### Phase 2: Focused Sweep (2-3 hours)
- [ ] Benchmark 11 focused configs on Qwen 0.5B shapes
- [ ] Identify top 3 performers
- [ ] Profile with NCU to validate hypotheses
- [ ] Document shared memory vs occupancy correlation

### Phase 3: Full Sweep (8-12 hours overnight)
- [ ] Generate full 800-1200 config space
- [ ] Benchmark on Qwen 0.5B, 1.5B, 7B shapes
- [ ] Profile top 30 + worst 30 configs
- [ ] Extract profiling features (SM util, Tensor Core util, etc.)

### Phase 4: ML Model Training (2-4 hours)
- [ ] Add profiling features to training data
- [ ] Retrain ONNX model with Phase 5 configs
- [ ] Validate top-30 hit rate (target: 75%+)
- [ ] Deploy updated model

### Phase 5: Production Integration (1-2 hours)
- [ ] Update `CudaGemmBackend` to use JIT Phase 5
- [ ] Add runtime heuristic selection
- [ ] Run end-to-end Qwen inference benchmarks
- [ ] Document performance improvements

---

## Success Criteria

### Minimum Viable Performance
- **+10-15% over Phase 4** (8.75 → 9.7 TFLOPS)
- Demonstrates JIT auto-tuning is viable
- Validates configuration space exploration

### Target Performance
- **+30-40% over Phase 4** (8.75 → 11.4-12.3 TFLOPS)
- Tensor Core utilization: 15-20% (vs 6.29% current)
- Warp occupancy: 40-50% (vs 22.62% current)

### Stretch Goal
- **+50-80% over Phase 4** (8.75 → 13.1-15.8 TFLOPS)
- Tensor Core utilization: 25-35%
- Warp occupancy: 50-60%
- Approaches cutlass baseline (18-20 TFLOPS)

---

## Key Insights from Profiling

1. **Shared memory is the bottleneck**, not decode overhead
2. **Streaming helped eliminate sync tax** (+17% from fixing redundant decode)
3. **Overlapping decode with MMA had minimal impact** (+0.3%)
4. **Tensor Cores are starved** (only 6% utilized)
5. **Need to explore occupancy vs work-per-block tradeoff**

**Next action**: Implement JIT infrastructure and run focused sweep to test hypotheses.

---

## Files Created

1. **`CudaGemmKernelTemplatePhase5.h`** - JIT kernel template with CuTe
2. **`Phase5ConfigSpace.h`** - Configuration space generator
3. **`phase5-jit-transition-plan.md`** - This document

**Next steps**:
- Extend `CudaGemmJIT` to compile Phase 5 templates
- Create benchmark harness for config sweeps
- Profile and iterate!
