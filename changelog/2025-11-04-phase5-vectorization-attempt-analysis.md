# Phase 5 JIT Kernel Vectorization Attempt - Analysis

**Date**: November 4, 2025  
**Author**: Development Session  
**Status**: ⚠️ **Vectorization Added But No Performance Improvement**

## Summary

Added float4 vectorized A-matrix loading to Phase 5 JIT kernel template, but NCU profiling shows **no improvement in memory coalescing**. The issue is architectural: CuTe's swizzled shared memory layout breaks coalescing patterns.

## Work Completed

### 1. Vectorization Implementation ✅

**Files Modified:**
- `src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h` - Added vectorized loading with preprocessor branches
- `src/v2/kernels/cuda/CudaGemmConfigPhase5.h` - Added `vectorize_a` parameter (default: 4)
- `src/v2/kernels/cuda/CudaGemmJITPhase5.cu` - Added template substitution for `${VECTORIZE_A}`

**Implementation Pattern:**
```cpp
#if VECTORIZE_A == 4
    // float4 vectorized loads (128-bit, coalesced)
    for (int vec_idx = tid; vec_idx < VEC_ELEMENTS; vec_idx += THREADS_PER_BLOCK) {
        int linear_idx = vec_idx * 4;
        int m = linear_idx / TILE_K;
        int k_base = (linear_idx % TILE_K) & ~3;  // Align to 4
        
        float4 vec4 = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
        
        // Write through CuTe swizzled tensor
        sA_write(m, k_base + 0) = half_t(__float2half(vec4.x));
        sA_write(m, k_base + 1) = half_t(__float2half(vec4.y));
        sA_write(m, k_base + 2) = half_t(__float2half(vec4.z));
        sA_write(m, k_base + 3) = half_t(__float2half(vec4.w));
    }
#elif VECTORIZE_A == 2
    // float2 vectorized loads (64-bit)
    ...
#else
    // Scalar fallback (VECTORIZE_A == 1)
    for (int k = tx; k < TILE_K; k += 32) {
        for (int m = ty; m < TILE_M; m += (THREADS_PER_BLOCK / 32)) {
            sA_write(m, k) = half_t(__float2half(A[gm * K + gk]));
        }
    }
#endif
```

### 2. NCU Profiling Results ❌

**Baseline (scalar loads):**
- Excessive sectors: 7,038,976 / 9,060,352 (78%)
- Load efficiency: 21.56% (6.9/32 bytes used per sector)
- Performance: 8.86 TFLOPS

**With vectorize_a=4:**
- Excessive sectors: 7,038,976 / 9,060,352 (78%) ← **NO CHANGE**
- Load efficiency: Still ~6.9/32 bytes per sector ← **NO IMPROVEMENT**
- Performance: 8.83 TFLOPS ← **IDENTICAL**

### 3. Root Cause Analysis 🔍

**Problem**: Vectorization doesn't match CuTe swizzle MBase configuration.

**Critical Discovery** (from [CuTe Swizzle blog](https://leimao.github.io/blog/CuTe-Swizzle/)):

> "when we know the number of elements in the vector, we could set `MBase` to the log2 of the number of elements in the vector, and as we will prove later, the swizzle operation will guarantee the contiguous memory access."

**Our Configuration Mismatch:**

1. **Shared Memory Type**: `half_t` (FP16, 2 bytes)
2. **Swizzle Config**: `Swizzle<3, 3, 3>` → `MBase = 3`
3. **Expected Vector Size**: `2^MBase = 2^3 = 8` FP16 elements (128 bits)
4. **Actual Vector Size**: `4` FP32 elements → `4` FP16 elements (64 bits) ❌

**The Formula** (from blog):
```
For element size S bytes, vector size N elements:
- MBase = log2(N)
- BBits = log2(32 * 4 / S) - MBase
- SShift = log2(X) - MBase  (X = row size)
```

**Our Case** (FP16, 64-wide tile):
```
S = 2 bytes (half_t)
X = 64 (TILE_K)
```

**If using 8-element vectors** (correct for MBase=3):
```
N = 8 FP16 elements (128 bits total)
MBase = log2(8) = 3 ✅ (matches our swizzle!)
BBits = log2(128 / 2) - 3 = 6 - 3 = 3 ✅
SShift = log2(64) - 3 = 6 - 3 = 3 ✅
Config: Swizzle<3, 3, 3> ✅ CORRECT for 8-element vectors
```

**If using 4-element vectors** (what we implemented):
```
N = 4 FP16 elements (64 bits total)
MBase = log2(4) = 2 ❌ (but we have MBase=3!)
BBits = log2(128 / 2) - 2 = 6 - 2 = 4
SShift = log2(64) - 2 = 6 - 2 = 4
Config: Swizzle<4, 2, 4> ← What we SHOULD use for 4-element vectors
```

**Why Vectorization Failed:**

1. **Global Memory Load**: Thread reads 4 consecutive FP32 (coalesced) ✅
   ```cpp
   float4 vec4 = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
   ```

2. **Shared Memory Write**: Writes 4 FP16 through `Swizzle<3,3,3>` ❌
   ```cpp
   // MBase=3 expects 8-element vectors, but we only have 4!
   sA_write(m, k_base + 0) = half_t(__float2half(vec4.x));
   sA_write(m, k_base + 1) = half_t(__float2half(vec4.y));
   sA_write(m, k_base + 2) = half_t(__float2half(vec4.z));
   sA_write(m, k_base + 3) = half_t(__float2half(vec4.w));
   // Missing: k_base+4, k_base+5, k_base+6, k_base+7 to complete the 8-element group!
   ```

3. **Swizzle Breaks Contiguity**:
   - With `MBase=3`, swizzle preserves **8 consecutive elements**
   - But we only write **4 elements**, then jump to next thread
   - The 4-element write doesn't align with swizzle's 8-element boundary
   - Result: Scattered writes → uncoalesced access

4. **NCU Sees Uncoalesced Access**:
   - Global load is 128-bit aligned but only serves 4 elements, not 8
   - Swizzle expects 8-element groups for contiguous placement
   - Mismatch causes many small transactions → 78% waste

**Key Insight**: The uncoalesced access is **NOT from the global load** itself, but from **thread mapping mismatch**. Our loop:
```cpp
for (int vec_idx = tid; vec_idx += THREADS_PER_BLOCK)
```
Maps threads to (m, k) coordinates, but CuTe's swizzle expects **different thread mapping** for coalescing.

## What Works vs What Doesn't

### ✅ Works (Implemented Successfully)
- Parameterized vectorization (1/2/4)
- JIT template substitution
- Correct boundary handling (scalar fallback)
- Compilation and execution (no crashes)

### ❌ Doesn't Work (Performance Issues)
- Memory coalescing improvement: **0% gain**
- Load efficiency: Still 21.56% (vs target >60%)
- Excessive sectors: Still 78% (vs target <30%)

### 🤔 Why It Doesn't Work
- **Architectural Mismatch**: CuTe swizzle vs vectorized loads
- **Thread Mapping**: Need to respect CuTe's expected thread→memory mapping
- **Swizzle Layout**: `Swizzle<3,3,3>` permutes addresses for bank conflict avoidance
- **Global Load Coalescing**: Requires **all threads in warp** loading consecutive addresses

## Alternative Approaches

### Option 1: Use 8-Element Vectors (Match Current Swizzle) ⭐ RECOMMENDED
```cpp
// Load 8×float (256-bit) or 2×float4 from global memory
// Convert to 8×half_t and write as complete 8-element group
#if VECTORIZE_A == 8
    for (int vec_idx = tid; vec_idx < TOTAL_ELEMENTS/8; vec_idx += THREADS_PER_BLOCK) {
        int linear_idx = vec_idx * 8;
        int m = linear_idx / TILE_K;
        int k_base = (linear_idx % TILE_K) & ~7;  // Align to 8
        
        // Load 8 floats (two float4 or use custom float8 if available)
        float4 vec4_lo = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
        float4 vec4_hi = *reinterpret_cast<const float4*>(&A[gm * K + gk + 4]);
        
        // Write all 8 elements (matches MBase=3)
        sA_write(m, k_base + 0) = half_t(__float2half(vec4_lo.x));
        sA_write(m, k_base + 1) = half_t(__float2half(vec4_lo.y));
        sA_write(m, k_base + 2) = half_t(__float2half(vec4_lo.z));
        sA_write(m, k_base + 3) = half_t(__float2half(vec4_lo.w));
        sA_write(m, k_base + 4) = half_t(__float2half(vec4_hi.x));
        sA_write(m, k_base + 5) = half_t(__float2half(vec4_hi.y));
        sA_write(m, k_base + 6) = half_t(__float2half(vec4_hi.z));
        sA_write(m, k_base + 7) = half_t(__float2half(vec4_hi.w));
    }
#endif
```
**Pros**: 
- ✅ Matches `Swizzle<3,3,3>` expectation (MBase=3 → 8 elements)
- ✅ No swizzle parameter changes needed
- ✅ Should enable contiguous writes per CuTe blog proof

**Cons**: 
- ⚠️ 256-bit loads may not be optimal (depends on architecture)
- ⚠️ More registers per thread

### Option 2: Change Swizzle to Match 4-Element Vectors
```cpp
// In CudaGemmConfigPhase5.h:
int swizzle_b = 4;  // BBits = log2(128/2) - 2 = 4
int swizzle_m = 2;  // MBase = log2(4) = 2 (for 4-element vectors)
int swizzle_s = 4;  // SShift = log2(64) - 2 = 4
```

**Pros**:
- ✅ Matches current float4 implementation
- ✅ 128-bit loads (optimal for most GPUs)

**Cons**:
- ⚠️ Changes shared memory layout (may affect bank conflicts)
- ⚠️ Need to validate parity tests with new swizzle
- ⚠️ May need different swizzle for B-matrix

### Option 3: Trust CuTe's Copy Operations (Investigate)
- CuTe may have vectorized copy primitives we're missing
- `cute::copy()` with proper layouts might auto-vectorize
- Need to study CuTe documentation for proper usage

**Pros**: 
- ✅ CuTe-native approach (framework intended design)
- ✅ May handle coalescing automatically

**Cons**: 
- ⚠️ Requires CuTe deep dive
- ⚠️ May not provide explicit control

### Option 4: Bypass CuTe for A-Matrix (Not Recommended)
```cpp
// Load directly to raw shared memory
__shared__ half_t s_A_raw[TILE_M][TILE_K];
// ... vectorized loads to s_A_raw ...
__syncthreads();
// Then swizzle copy
```
**Drawback**: Extra synchronization, defeats CuTe design

## Conclusions

1. **Vectorization alone doesn't fix coalescing** when CuTe swizzle is involved
2. **Thread mapping must match layout** for true coalescing
3. **CuTe's design trades global coalescing for shared bank conflict avoidance**
4. **Current performance (8.86 TFLOPS) is acceptable baseline**

## Recommendations

**Immediate Next Steps:**

1. **✅ Test Option 1: 8-Element Vectorization** (matches current Swizzle<3,3,3>)
   - Add `vectorize_a = 8` support to template
   - Load 2×float4 (256-bit total) → write 8×half_t
   - Run parity test to verify correctness
   - Profile with NCU to measure improvement
   - **Expected**: Load efficiency >60%, excessive sectors <30%

2. **⏸️ Test Option 2: Change Swizzle to <4,2,4>** (matches current float4)
   - Create new config with swizzle_b=4, swizzle_m=2, swizzle_s=4
   - Keep vectorize_a=4
   - Run parity test (shared memory layout changed)
   - Profile with NCU
   - **Expected**: Similar gains to Option 1

3. **📊 Compare Performance**:
   ```
   | Config                  | TFLOPS | Load Eff | Excessive Sectors |
   |-------------------------|--------|----------|-------------------|
   | Baseline (scalar)       | 8.86   | 21.56%   | 78%              |
   | 8-elem + Swizzle<3,3,3> | ?      | ?        | ?                |
   | 4-elem + Swizzle<4,2,4> | ?      | ?        | ?                |
   ```

4. **🎯 Select Production Config**:
   - Choose best performing option
   - Update defaults in config
   - Add to autotuner search space
   - Document in code comments

**Long Term:**

- Investigate CuTe-native `cute::copy()` patterns
- Add swizzle parameters to autotuner exploration
- Consider different swizzles for different tile sizes
- Apply same principles to B-matrix vectorization
1. Keep vectorization code (for documentation, may help other configs)
2. Set `vectorize_a = 1` (scalar) as default until we understand CuTe better
3. Focus on B-matrix vectorization (IQ4_NL dequantization)
4. Profile B-matrix loads specifically

**Medium Term:**
1. Study CuTe copy semantics (`cute::copy()` with proper src/dst layouts)
2. Investigate if CuTe has vectorization-aware primitives
3. Compare with official CuTe GEMM examples
4. Consider non-swizzled layouts for comparison

**Long Term:**
1. Custom thread mapping that respects swizzle
2. Or redesign A-loading to not use swizzle (if justified by profiling)
3. Autotuner sweep across swizzle params (swz222, swz444) - may find better trade-off

## Files Changed

```
src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h  (+120 lines vectorization)
src/v2/kernels/cuda/CudaGemmConfigPhase5.h          (+4 lines vectorize_a param)
src/v2/kernels/cuda/CudaGemmJITPhase5.cu            (+1 line template substitution)
```

## Profiling Data

**Original Profile**: `build_v2_release/tests/v2/phase5_jit_profile.ncu-rep`  
**Vectorized Profile**: `build_v2_release/tests/v2/phase5_jit_vectorized.ncu-rep`

**Comparison**:
- No measurable difference in any metric
- Both show 78% excessive sectors
- Both show ~6.9 bytes/sector utilization
- Confirms vectorization is not addressing root cause

## Next Steps

1. **Revert default to scalar** (`vectorize_a = 1`) to avoid misleading optimization
2. **Profile B-matrix dequantization** - may have more impactful vectorization opportunities
3. **Study CuTe GEMM examples** from CUTLASS repo for proper usage patterns
4. **Focus on other NCU opportunities**: Occupancy (+5-10%), issue slots (+5-15%)
