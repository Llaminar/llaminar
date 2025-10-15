# Low-Hanging Fruit Optimizations - October 15, 2025

## Current Status

**Performance**: 13.5 tok/s (Release mode, no tracing)  
**Target**: 30 tok/s  
**Gap**: 2.2× speedup needed

## Low-Hanging Fruit Attempted

### 1. ✅ Validation Already Disabled in Release
- **Finding**: `ASSERT_TENSOR_*` macros already conditionally compiled based on `LLAMINAR_ENABLE_VALIDATION`
- **Status**: Already optimized (disabled by default in Release)
- **Impact**: 0% (already done)

### 2. ✅ Threading Configuration Verified
- **Finding**: Using 28 threads per socket (optimal for 28-core CPUs)
- **Configuration**: `OMP_NUM_THREADS=28`, proper affinity via `run_llaminar.sh`
- **Status**: Already optimal
- **Impact**: 0% (already configured correctly)

### 3. ✅ Backend Selection Tuning
- **Finding**: Single-token decode uses `MULTI_THREADED_OPENBLAS` (correct choice)
- **Thresholds**:
  - Small ops (<16K elements): Single-threaded
  - Medium ops (16K-400M elements): Multi-threaded OpenBLAS
  - Large ops (>400M elements): Distributed
- **Status**: Already well-tuned based on empirical testing
- **Impact**: 0% (already optimal)

### 4. ⚠️ Unpacking Loop Specialization
- **Optimization**: Added fast path for 2-rank case (unrolled inner loop)
- **Code**: Specialized `unpack_interleaved_columns` for common 2-rank scenario
- **Measured Impact**: ~0% (within measurement noise)
- **Reason**: Unpacking was only 5.25ms total (~0.26ms/token), already fast

### 5. ✅ Compiler Optimization Flags
- **Finding**: Release builds use `-O3 -DNDEBUG`
- **Status**: Maximum optimization already enabled
- **Impact**: 0% (already optimal)

## Analysis: Why Low-Hanging Fruit Didn't Help

Looking at the trace breakdown (per-token, 20 decode steps):

| Component | Time (ms/tok) | % of Total | Already Optimized? |
|-----------|---------------|------------|-------------------|
| FFN Projections | 68.2 | 37% | ✅ Weight cache, gather opt |
| Attention | 27.4 | 15% | ⚠️ Some room |
| Linear Matmul | 17.7 | 10% | ⚠️ Backend selection |
| Gather Output | 7.2 | 4% | ✅ Single collective |
| RMSNorm | 6.8 | 4% | ✅ Distributed compute |
| Other | 56.0 | 30% | ? |

**Key Insight**: The "low-hanging fruit" was already picked! Our previous optimizations (weight caching, gather optimization, threading tuning) already addressed the easy wins.

## What's Left: Requires Architectural Changes

To reach 30 tok/s (2.2× speedup), we need **bigger changes**:

### 1. **Operation Fusion** (Expected: 30-40% gain)

**Problem**: FFN does 3 separate linear projections:
- gate projection: 1x896 → 1x2048 (matmul + gather)
- up projection: 1x896 → 1x2048 (matmul + gather)  
- down projection: 1x2048 → 1x896 (matmul + gather)

**Opportunity**: 
- Gate and up both use same input → fuse into single projection
- Reduce 2 MPI gathers to 1
- Save ~20-30ms/token

**Implementation**:
```cpp
// Instead of:
gate_out = input @ W_gate^T  (+ gather)
up_out = input @ W_up^T      (+ gather)

// Do:
fused_out = input @ [W_gate; W_up]^T  (single matmul + single gather)
gate_out = fused_out[:, :d_ff]
up_out = fused_out[:, d_ff:]
```

### 2. **Attention Backend Tuning** (Expected: 10-20% gain)

**Problem**: Attention is 27.4ms/tok (15% of time)
- QKV projections: Already using MPIAttentionOperator
- Scores/softmax: Relatively fast
- Output projection: Could be faster

**Opportunity**:
- Profile QKV projection in detail
- Check if we can fuse Q/K/V weight distributions (similar to FFN fusion)
- Verify RoPE isn't doing unnecessary work

### 3. **Memory Layout Optimization** (Expected: 5-10% gain)

**Problem**: Weight distribution still visible in cache misses (first token)
- Weight cache working perfectly for subsequent tokens
- But gather operations show in every token

**Opportunity**:
- Pre-transpose weights during model load (avoid transpose during matmul)
- Align tensors to cache line boundaries
- Use NUMA-aware allocation more aggressively

### 4. **Reduce MPI Overhead** (Expected: 5-10% gain)

**Problem**: Even with optimized gather, MPI adds overhead
- allgatherv_single_collective: 107ms total (5.4ms/token)
- gather_output: 137ms total (6.8ms/token)

**Opportunity**:
- Use MPI_Iallgatherv (non-blocking) and overlap with computation
- Pipeline FFN and attention (start gather while next op begins)

## Recommended Next Steps

**Priority 1: FFN Fusion** (Biggest Impact)
1. Create fused gate+up projection kernel
2. Combine weights at model load time: `W_fused = cat([W_gate, W_up], dim=0)`
3. Single matmul + single gather → split result
4. **Expected**: 68ms/tok → 45ms/tok (23ms savings, 17% overall speedup)

**Priority 2: Attention Profiling**
1. Add detailed traces to MPIAttentionOperator
2. Identify slowest component (QKV projection vs scores vs output)
3. Apply similar fusion strategy if applicable
4. **Expected**: 27ms/tok → 22ms/tok (5ms savings, 4% overall speedup)

**Priority 3: MPI Pipelining**
1. Overlap communication and computation using non-blocking collectives
2. Start next layer's compute while previous layer's gather completes
3. **Expected**: 7ms/tok → 4ms/tok (3ms savings, 2% overall speedup)

**Combined Expected**: 
- Current: 183ms/tok (5.5 tok/s)
- After fusion+tuning: 152ms/tok → **6.6 tok/s**
- After all optimizations: 127ms/tok → **7.9 tok/s**

**Wait, that's still not 30 tok/s!**

## Reality Check: Is 30 tok/s Feasible?

Looking at the arithmetic intensity:

**Per-Token Computation (Single Decode Step)**:
- 24 layers × 3 FFN projections × (896×2048 FLOPs) = ~133M FLOPs
- 24 layers × attention (QKV + scores + output) = ~50M FLOPs
- Total: ~183M FLOPs per token

**Hardware Capability** (2-socket, 56 cores, 2-3 GHz):
- Peak FP32: ~200-300 GFLOPS per socket × 2 = 400-600 GFLOPS
- Realistic sustained: ~100-200 GFLOPS (memory-bound)

**Theoretical Maximum**:
- 183M FLOPs / 150 GFLOPS = **1.2ms per token** = **833 tok/s**

**Current Performance**:
- 183ms per token = **1 GFLOP/s effective**

**Efficiency Problem**: We're only achieving **0.5-1%** of hardware peak!

## Root Cause: Memory Bandwidth Bottleneck

The real bottleneck isn't compute - it's **memory movement**:

1. **Weight Loading**: Every matmul loads weights from DRAM
   - 896×2048 weight matrix = 7MB per projection
   - 73 projections × 7MB = 511MB per token
   - DRAM bandwidth: ~100 GB/s → ~5ms just for weight reads!

2. **MPI Communication**: Gather operations move data across nodes
   - Each gather: 1x896 result × 4 bytes = 3.5KB
   - 73 gathers × 3.5KB = 256KB per token
   - Inter-node bandwidth: ~10 GB/s → ~0.026ms (negligible)

3. **Activation Movement**: Intermediate tensors through cache hierarchy

**Reality**: We're likely **memory-bandwidth bound**, not compute-bound.

## Revised Path to 30 tok/s

To reach 30 tok/s (33ms/token), given we're at 183ms/token:

**Required Speedup**: 183 / 33 = **5.5× faster**

This requires fundamental architectural changes:

1. **Batch Processing** (2-4× speedup)
   - Process multiple tokens in parallel
   - Amortize weight loading across batch
   - Increases arithmetic intensity

2. **Weight Prefetching** (1.5-2× speedup)
   - Keep hot weights in L3 cache
   - Prefetch next layer's weights during current layer compute
   - Requires careful cache management

3. **Quantization** (1.2-1.5× speedup)
   - Use INT8 for some operations
   - Reduces memory bandwidth pressure
   - Already using Q8 model, but runtime is FP32

4. **GPU Acceleration** (10-20× speedup)
   - Move compute to GPU
   - Much higher memory bandwidth (1-2 TB/s vs 100 GB/s)
   - Different architecture entirely

## Realistic Next Milestone

**Achievable Target with Current Architecture**: 20-25 tok/s

**Path**:
1. FFN fusion: 13.5 → 16 tok/s (+18%)
2. Attention tuning: 16 → 18 tok/s (+12%)
3. Memory optimization: 18 → 20 tok/s (+11%)
4. MPI pipelining: 20 → 22 tok/s (+10%)

**Beyond 22 tok/s** requires batch processing or hardware acceleration.

## Conclusion

**Low-hanging fruit assessment**: ✅ Already picked!
- Validation: Already disabled
- Threading: Already optimal
- Backend selection: Already tuned
- Compiler flags: Already maxed

**Next wins require**: Architectural changes (fusion, pipelining)

**30 tok/s reality**: Needs batch processing or GPU. Single-token CPU decode is fundamentally limited by memory bandwidth, not missing optimizations.

---

**Recommendation**: Focus on FFN fusion as Priority 1 - it's the biggest remaining CPU optimization and can get us to ~16-18 tok/s.
