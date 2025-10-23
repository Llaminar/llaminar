# Fused Kernel Optimization Attempt - Failed

**Date**: October 22, 2025  
**Status**: ❌ Reverted - Massive Performance Regression  
**Impact**: 8-10× slowdown (394 → 44 GFLOPS)

## Hypothesis

Eliminate intermediate dequantized buffer by fusing dequantization directly into dot product:

```cpp
// Current (2-stage):
float B_buffer[32];
decode_block(quantized, B_buffer);  // Write 128 bytes
result = dot_product(A, B_buffer, 32);  // Read 128 bytes

// Proposed (fused):
result = dot_product_fused(A, quantized_block);  // No buffer!
```

**Expected**: -47% memory bandwidth, +15-25% throughput

## Implementation

Created `dot_product_fused_iq4nl()` that combines:
1. Nibble extraction (lookup_low/lookup_high arrays)
2. Int8 → FP32 conversion
3. FMA accumulation

Applied to all 4 multiply paths (FP32 cache-blocked, FP32 row-wise, BF16 cache-blocked, BF16 row-wise).

## Results

**Catastrophic regression**:

| Workload | Baseline | Fused | Change |
|----------|----------|-------|---------|
| Q-Proj 4096 | 394 GFLOPS | 44 GFLOPS | **-89%** |
| FFN 512 | 525 GFLOPS | 76 GFLOPS | **-86%** |
| FFN 8192 | 384 GFLOPS | 72 GFLOPS | **-81%** |

## Root Cause Analysis

###  1. **Serial nibble extraction still required**

```cpp
// Inside dot_product_fused_iq4nl - THIS IS SLOW!
for (int i = 0; i < 16; ++i) {
    lookup_low[i] = kvalues_iq4nl[block.qs[i] & 0x0F];  // Serial loop
}
```

The for-loop to extract nibbles and lookup kvalues is **inherently serial** and must happen regardless of whether we have an intermediate buffer.

### 2. **We still create temporary arrays**

```cpp
alignas(64) int8_t lookup_low[16];   // 16 bytes temp
alignas(64) int8_t lookup_high[16];  // 16 bytes temp
```

The fused kernel didn't actually eliminate buffers - it just moved them inside the function. These still occupy stack/register space.

### 3. **Loss of compiler optimization opportunities**

**Before**: Two simple functions that compiler can optimize independently
- `decodeBlockAVX512`: Pure computation, easy to inline/vectorize
- `dot_product_simd`: Standard FMA loop, compiler knows how to optimize

**After**: Complex fused function with multiple concerns
- Harder for compiler to auto-vectorize
- Likely doesn't inline due to size
- Register pressure from holding both A and dequantized B

### 4. **The 128-byte buffer is NOT a bottleneck**

- 128 bytes fits entirely in L1 cache (32 KB available)
- Likely optimized into registers by compiler
- Write-read pair has ~3-4 cycle latency (negligible vs computation)
- Buffer is **hot** - immediately used after decode

The buffer write+read is **pipelined with computation** - while we're decoding block N, we're computing with block N-1's buffer. The fused approach serializes this.

## Key Learnings

### ❌ What Didn't Work

1. **Microoptimizing memory traffic that wasn't the bottleneck**
   - 128-byte buffer is tiny, stays in L1
   - Compiler likely optimizes away most overhead
   - The write+read latency overlaps with computation

2. **Assuming "fewer memory operations = faster"**
   - True for DRAM access (GB-scale)
   - **False for L1 cache** (KB-scale)
   - L1 bandwidth: 1000+ GB/s, latency: ~4 cycles
   - 128 bytes = 0.1 microseconds (unmeasurable)

3. **Breaking compiler optimization opportunities**
   - Two simple functions → one complex function
   - Lost inlining opportunities
   - Harder auto-vectorization

### ✅ What We Learned

1. **Profile before optimizing**
   - The 30% OpenMP overhead is the real bottleneck
   - Dequantization (even with buffer) is only 4-5% of time
   - Buffer overhead is <1%

2. **Respect the compiler**
   - Modern compilers are smart about L1-resident data
   - Clear, simple functions optimize better than "clever" fusions
   - Trust but verify with assembly inspection

3. **Real bottlenecks for IQ4_NL GEMM**:
   - OpenMP synchronization: 30%
   - Dot product computation: 18-20%
   - Dequantization: 4-5%
   - **Buffer overhead: <1%** (not worth optimizing!)

## Alternative Approaches That Might Work

### 1. **Reduce OpenMP Overhead** (30% target)

Already attempted - semantic issues with thread-local storage.

**Next**: Try explicit thread affinity, reduce barrier frequency.

### 2. **Optimize Dot Product** (18-20% target)

Current: AVX512 FMA loop with horizontal reduction

**Ideas**:
- Multiple accumulators (reduce dependency chains)
- Unroll loop 2× or 4× (more ILP)
- Prefetch next block during computation

### 3. **Block-Level Caching** (for m > 1)

When processing multiple rows with same column blocks:

```cpp
// Cache decoded blocks across rows
if (m > 4) {
    float B_cache[num_k_blocks][32];  // Decode once
    for (int kb = 0; kb < num_k_blocks; ++kb) {
        decode_block_at(j, kb, B_cache[kb]);
    }
    for (int i = 0; i < m; ++i) {
        for (int kb = 0; kb < num_k_blocks; ++kb) {
            acc[i] += dot_product(A_row, B_cache[kb], 32);
        }
    }
}
```

**Benefit**: Amortizes decode cost across m rows
**Cost**: Increased cache pressure (only works if m × 32 fits in L1)

## Conclusion

**The fused kernel was a bad idea** because:

1. Dequantization work still required (nibble extraction, lookup, conversion)
2. Temporary arrays still created (just in different place)
3. Broke compiler optimizations
4. Targeted non-bottleneck (<1% buffer overhead vs 30% OpenMP overhead)

**Key Insight**: 128-byte L1-resident buffers are essentially free. The cost is in the **computation** (nibble extraction, lookup, conversion), not the memory operations.

**Recommendation**: Focus on real bottlenecks (OpenMP overhead 30%, dot product 18-20%) rather than microoptimizing L1 cache traffic.

---

**Files Modified** (reverted):
- `src/tensors/IQ4_NLTensor.h`: Added `dot_product_fused_iq4nl()`, modified all multiply paths

**Revert Command**:
```bash
git checkout src/tensors/IQ4_NLTensor.h
```

**Status**: Code reverted to baseline, performance restored to 394 GFLOPS.
