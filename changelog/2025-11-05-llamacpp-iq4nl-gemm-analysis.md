# llama.cpp IQ4_NL GEMM Analysis

**Date**: November 5, 2025  
**Purpose**: Understand how llama.cpp avoids conversion overhead in IQ4_NL GEMM

## Key Insight: Fused Dequantization During GEMM

llama.cpp does **NOT** dequantize IQ4_NL to INT8 before GEMM. Instead, it **dequantizes on-the-fly during the dot product** using efficient table lookups.

## Implementation Strategy

### 1. IQ4_NL Lookup Table (Device Memory)

**File**: `ggml/src/ggml-common.h`

```cpp
// Expanded from macro:
static const __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 
       1,   13,  25,  38,  53,  69,  89, 113
};
```

- **16-byte table** stored in device constant/read-only memory
- Accessed via 4-bit indices (0-15)
- Maps 4-bit quantized values to INT8 dequantized values

### 2. Efficient Table Lookup (`get_int_from_table_16`)

**File**: `ggml/src/ggml-cuda/vecdotq.cuh`

**NVIDIA CUDA Implementation**:
```cpp
static __device__ __forceinline__ int2 get_int_from_table_16(
    const int & q4,        // 8 × 4-bit indices packed into 32 bits  
    const int8_t * table   // Lookup table (kvalues_iq4nl)
) {
    const uint32_t * table32 = (const uint32_t *) table;

    // Process 32 bits in 2 iterations (16 bits each)
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
    
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        
        // Use __byte_perm to select bytes based on low 3 bits
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        
        // Use 4th bit to select between low/high
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // Rearrange bytes: even indices in .x, odd indices in .y
    return make_int2(
        __byte_perm(tmp[0], tmp[1], 0x6420),  // Even indices
        __byte_perm(tmp[0], tmp[1], 0x7531)   // Odd indices
    );
}
```

**Key Features**:
- Uses CUDA intrinsic `__byte_perm` for parallel byte selection
- Processes **8 × 4-bit values → 8 × INT8 values** in ~3-4 instructions
- Returns `int2` with even/odd bytes separated for DP4A
- **Zero overhead** - compiles to a few PTX instructions

### 3. Fused Dot Product with Dequantization

**File**: `ggml/src/ggml-cuda/vecdotq.cuh`

```cpp
static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq,              // IQ4_NL weights
    const block_q8_1 * __restrict__ bq8_1,      // Quantized activations (INT8)
    const int & kbx,                            // Block index
    const int & iqs                             // Sub-block index
) {
    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;
    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        // Extract 32 bits (8 × 4-bit indices) from IQ4_NL block
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        
        // Dequantize 8 × 4-bit → 8 × INT8 using table lookup
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        
        // Perform 2 × DP4A operations (each processes 4 × INT8)
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);  // Even indices
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);  // Odd indices
    }

    // Apply scaling factors
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}
```

**Workflow**:
1. **Load** 32 bits (8 × 4-bit indices) from IQ4_NL block
2. **Dequantize** via table lookup → `int2` (8 × INT8 values)
3. **Compute** 2 × DP4A operations (INT8 dot products)
4. **Accumulate** into INT32 sum
5. **Scale** with FP16 scale factors

### 4. Matrix Multiply Quantized (MMQ) Kernel

**File**: `ggml/src/ggml-cuda/mmq.cuh`

The MMQ kernel orchestrates this for large matrix multiplies:

```
Input: A (FP32/Q8_1), B (IQ4_NL quantized weights)
Output: C (FP32)

For each tile:
    1. Load tile of A into shared memory
    2. Convert A → Q8_1 (if needed)
    3. Load tile of B (IQ4_NL) into shared memory
    4. For each warp:
        - Call vec_dot_iq4_nl_q8_1() repeatedly
        - Dequantize B on-the-fly using kvalues_iq4nl lookup
        - Accumulate dot products with DP4A
    5. Write results to C
```

**Key Design Choices**:
- **B stays in IQ4_NL** throughout computation
- **A converted to Q8_1** once per tile (amortized cost)
- **Dequantization happens in registers** during DP4A loop
- **No global memory writes** for dequantized data

## Performance Advantages

### vs Our Phase 7 Implementation

**Our Approach** (Current):
1. Dequantize IQ4_NL → INT8 (separate kernel)
2. Dequantized B written to global memory (~16 MB for 4096×4096)
3. CUTLASS reads dequantized B from global memory
4. Total: **~2 memory passes** (write + read)

**llama.cpp Approach**:
1. Load IQ4_NL directly from global memory (~8 MB for 4096×4096)
2. Dequantize in registers during computation
3. No intermediate writes
4. Total: **1 memory pass** (read only)

**Memory Savings**:
- **2× less memory bandwidth** (no dequantized write)
- **2× less storage** (keeps quantized format)
- **Cache-friendly**: Dequantized values stay in registers/L1

### Why It's Fast

1. **Table lookup is cheap**: 
   - `kvalues_iq4nl` fits in constant cache (16 bytes)
   - `__byte_perm` instruction is 1-2 cycles
   - Total: **~5 instructions** for 8 × INT8 values

2. **Registers instead of memory**:
   - Dequantized values never touch global memory
   - DP4A operates on register values
   - Memory hierarchy: Global → Shared → Register → DP4A

3. **DP4A perfect match**:
   - `int2` output maps directly to DP4A input
   - Even/odd split allows 2 × DP4A per lookup
   - Maximizes ALU utilization

## Implications for Our Implementation

### Option 1: Adopt llama.cpp's Approach (Recommended)

**Pros**:
- ✅ Eliminates conversion overhead
- ✅ 2× less memory bandwidth
- ✅ Proven to work well in production

**Cons**:
- ❌ Cannot use CUTLASS directly (expects INT8, not IQ4_NL)
- ❌ Need custom kernel or CUTLASS extension
- ❌ More complex than template instantiation

**Implementation Path**:
1. Port `get_int_from_table_16` + `vec_dot_iq4_nl_q8_1`
2. Build MMQ-style kernel or CUTLASS epilogue extension
3. Keep quantized format throughout

### Option 2: Optimize Current Approach

**Keep CUTLASS, optimize conversions**:
1. **Fuse quantize_A with GEMM**: Quantize A tiles in shared memory
2. **Pre-dequantize B weights**: One-time conversion at model load
3. **Persistent dequantized cache**: Keep INT8 weights in VRAM

**Pros**:
- ✅ Simpler - keeps CUTLASS tensor core path
- ✅ Incremental improvement

**Cons**:
- ❌ Still 2× memory usage vs quantized
- ❌ Misses half the bandwidth savings
- ❌ Not aligned with llama.cpp's proven approach

## Recommendation

**Adopt llama.cpp's fused dequantization approach** for Phase 8:

1. Port `get_int_from_table_16` to our codebase
2. Implement custom fused dequant-GEMM kernel based on llama.cpp's MMQ
3. Compare performance:
   - llama.cpp MMQ (DP4A)
   - Custom fused dequant + CUTLASS (tensor cores)
   - Potential: **Best of both** (fused dequant WITH tensor cores)

The key insight is that **dequantization overhead is a fundamental bottleneck**, not just a benchmark artifact. Production inference will hit the same issue every forward pass.

## Files to Study Further

1. `external/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` - Full MMQ kernel
2. `external/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` - All quantization formats
3. `external/llama.cpp/ggml/src/ggml-common.h` - Lookup tables
4. `external/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu` - Instantiation

## Next Steps

1. ✅ Understand llama.cpp's approach (COMPLETE)
2. ⏳ Benchmark llama.cpp IQ4_NL GEMM performance
3. ⏳ Prototype fused dequant kernel
4. ⏳ Compare: llama.cpp DP4A vs custom tensor cores
5. ⏳ Integrate best approach into Phase 7/8
