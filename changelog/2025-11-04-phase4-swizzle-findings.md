# Phase 4 Swizzle Implementation Findings

**Date**: November 4, 2025  
**Author**: David Sanftenberg (with GitHub Copilot)  
**Session Duration**: ~3 hours

## Summary

After extensive debugging, we successfully:
1. ✅ Fixed **critical test bug** in Phase 4 (dequantization missing from CPU reference)
2. ✅ Verified Phase 4 kernel **passes correctness** with Phase 3 Pipelined code (0.0039 error)
3. ❌ Discovered swizzle implementation issue (21.69 error when swizzle enabled)
4. 📝 Documented root cause and path forward

---

## Bug #1: Test Correctness Failure (FIXED)

### Symptom
- Phase 4 correctness test failed with **1.32 error** vs CPU reference
- Even with EXACT same kernel body as Phase 3 Pipelined (which passes with 0.0039 error)

### Root Cause
**Test was comparing apples to oranges!**

```cpp
// WRONG (Phase 4 original test):
auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);
cpu_gemm_reference(A, B_fp32, C_ref, M, N, K);  // ❌ Uses original FP32!
launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);  // ✅ Uses quantized blocks

// CORRECT (Phase 3 test):
auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);
// Dequantize for CPU reference
std::vector<float> B_dequant(N * K);
// ... dequantize B_blocks ...
cpu_gemm_reference(A, B_dequant, C_ref, M, N, K);  // ✅ Uses dequantized!
```

**Why This Failed**:
- Quantization is **lossy** (4-bit approximation of FP32)
- GPU computed: `C_gpu = A × dequant(B_blocks)`
- CPU computed: `C_ref = A × B_fp32` (original, unquantized)
- These are fundamentally different matrix products!

### Fix Applied
Added dequantization step to Phase 4 test:

```cpp
// Dequantize B_blocks for CPU reference
std::vector<float> B_dequant(N * K);
for (int n = 0; n < N; n++) {
    for (int kb = 0; kb < K/32; kb++) {
        const auto& block = B_blocks[n * (K/32) + kb];
        for (int i = 0; i < 16; i++) {
            uint8_t q = block.quants[i];
            B_dequant[n * K + kb * 32 + i * 2] = block.scale * iq4nl_values[q & 0xF];
            B_dequant[n * K + kb * 32 + i * 2 + 1] = block.scale * iq4nl_values[q >> 4];
        }
    }
}
cpu_gemm_reference(A, B_dequant, C_ref, M, N, K);  // ✅ Now correct!
```

**Result**: Test now passes with **0.00394034** error (identical to Phase 3 Pipelined)

**Changed File**: `tests/v2/unit/Test__CudaGemmPhase4QuickWins.cpp` (lines 110-145)

---

## Bug #2: Swizzle Implementation Issue (OPEN)

### Symptom
- Without swizzle: **0.0039** error ✅ (passes)
- With swizzle: **21.69** error ❌ (fails catastrophically)

### Swizzle Configuration
```cpp
using SmemLayoutA_Swizzled = decltype(composition(
    Swizzle<3, 3, 3>{},  // XOR pattern: bits [3:5] ^= bits [0:2]
    Layout<Shape<Int<TILE_M>, Int<TILE_K>>,
           Stride<Int<TILE_K>, Int<1>>>{}
));
```

**Formula validation** (from Lei Mao's blog):
- FP16 (S=2), 64-wide (X=64), 128-bit vector (N=8)
- MBase = log2(8) = 3 ✅
- BBits = log2(128/2) - 3 = 3 ✅
- SShift = log2(64) - 3 = 3 ✅
- **`Swizzle<3, 3, 3>` is correct for our configuration!**

### Problem Identified

**We write with linear indexing, read with swizzled indexing!**

```cpp
// PROLOGUE: Write to shared memory (linear indexing)
for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads) {
    int m = idx / TILE_K;
    int k = idx % TILE_K;
    s_A[0][m][k] = __float2half(A[gm * K + gk]);  // Linear array indexing
}

// COMPUTE: Read from shared memory (swizzled layout)
auto sA_tensor = make_tensor(
    make_smem_ptr(s_A[read_stage][0]),
    SmemLayoutA_Swizzled{}  // Swizzled layout!
);
```

**Why This Breaks**:
1. Linear write stores element (0,0) at memory address `0`
2. Swizzled layout maps (0,0) to address `swizzle(0,0)` (e.g., address `5`)
3. When we read (0,0) from swizzled tensor, we fetch from address `5`
4. We get wrong data → catastrophic error!

**Example**:
```
Write: s_A[0][0] = value_0_0  // Writes to physical address 0
Read:  sA_tensor(0, 0) → swizzle(0,0) = 5 → reads from address 5 (wrong value!)
```

### Solutions Attempted

**Attempt 1: Use `tensor(m,k)` indexing for writes** ❌
```cpp
auto sA_write = make_tensor(make_smem_ptr(smem_A_ptr), SmemLayoutA{});
for (...) {
    sA_write(m, k) = value;  // Apply swizzle to writes too
}
```
- **Problem**: Still used `SmemLayoutA{}` (linear), not swizzled
- **Result**: Same 1.32 error

**Attempt 2: Apply swizzle to both write and read tensors** ❌
```cpp
auto sA_write = make_tensor(make_smem_ptr(...), SmemLayoutA_Swizzled{});
sA_write(m, k) = value;  // Swizzled write

auto sA_read = make_tensor(make_smem_ptr(...), SmemLayoutA_Swizzled{});
auto tCrA = thr_mma.partition_fragment_A(sA_read);  // Swizzled read
```
- **Problem**: This should work in theory, but...
- **Issue**: We're using dynamic shared memory allocation (`extern __shared__`)
- **Conflict**: CuTe expects statically allocated arrays for swizzle
- **Result**: Didn't test this variant properly yet

---

## Next Steps

### Option 1: Fix Swizzle Implementation (Proper CuTe Pattern)

**From Lei Mao's blog**: Use `cute::copy()` for **both** global→shared and shared→register transfers.

```cpp
// Step 1: Create swizzled layout for global A (FP32)
auto gA = make_tensor(
    make_gmem_ptr(A),
    make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
    make_stride(Int<K>{}, Int<1>{})
);

// Step 2: Create swizzled shared memory tensor
auto sA = make_tensor(
    make_smem_ptr(s_A[stage][0]),
    SmemLayoutA_Swizzled{}
);

// Step 3: Use cute::copy() - automatically handles swizzle!
cute::copy(gA, sA);  // Respects swizzled layout
```

**Challenge**: `cute::copy()` expects **same data types**:
- Global A: FP32
- Shared A: FP16

**Need**: Custom `Copy_Atom` for FP32→FP16 conversion during copy.

**Estimated effort**: 2-4 hours to implement custom Copy_Atom + test

### Option 2: Swizzle Only During Read (If Possible)

**Idea**: Keep linear writes, but ensure physical memory layout matches logical swizzle.

**Hypothesis**: Maybe we can pre-compute swizzled indices and write with those?

```cpp
for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads) {
    int m = idx / TILE_K;
    int k = idx % TILE_K;
    
    // Compute swizzled address
    int swizzled_addr = compute_swizzle_address(m, k);
    
    // Write to swizzled location
    flat_smem[swizzled_addr] = __float2half(A[gm * K + gk]);
}
```

**Problem**: This defeats the purpose - we're manually computing swizzle instead of letting CuTe do it.

### Option 3: Skip Swizzle, Ship Phase 3 Part 2

**Rationale**:
- Phase 3 Part 2 already achieves **6.56 TFLOPS at M=1024** (18.45% of peak)
- Swizzle optimizes bank conflicts (~5-10% gain expected)
- Primary bottleneck is **memory bandwidth** (30-40%), not bank conflicts
- Swizzle won't fix bandwidth saturation

**Expected gain**: 7.2 TFLOPS (+10%) vs current 6.56 TFLOPS
**Realistic gain**: 6.9-7.0 TFLOPS (+5-7%) due to bandwidth limits

**Time vs reward**:
- Proper swizzle: 2-4 hours implementation + testing
- Potential gain: 0.34-0.64 TFLOPS absolute
- **Is this worth it?** Debatable given other optimization opportunities

---

## Recommendation

**Ship Phase 3 Part 2 as production-ready Phase 4**:
1. ✅ **6.56 TFLOPS** at sweet spot (M=1024)
2. ✅ **Correctness validated** (0.0039 error vs CPU)
3. ✅ **No swizzle complexity** (simple, maintainable)
4. ✅ **Double-buffered pipelining** (compute/memory overlap)

**Defer swizzle to Phase 5**:
- Learn proper CuTe Copy_Atom patterns
- Implement custom FP32→FP16 conversion atom
- Validate swizzle gains with profiler (ncu bank conflict metrics)
- Only if profiler shows bank conflicts are >10% of runtime

**Alternative next steps**:
- **Phase 5**: Larger tiles (128×128×64) for M≥2048 to reduce block count
- **Phase 6**: cp.async for true async global→shared transfer (+3-5%)
- **Phase 7**: Multi-stage pipelining (prefetch 2 tiles ahead)

---

## Files Modified

### Kernel
- `src/v2/kernels/cuda/CudaGemmKernelPhase4QuickWins.cu`
  - Lines 1-20: Updated header comments (swizzle-focused)
  - Lines 95-115: Swizzled layout definitions (currently causing 21.69 error)
  - Lines 245-270: Swizzled tensor creation in compute stage

### Test
- `tests/v2/unit/Test__CudaGemmPhase4QuickWins.cpp`
  - Lines 110-145: **CRITICAL FIX** - Added dequantization step
  - Now matches Phase 3 test correctness methodology

### Documentation
- `changelog/2025-11-04-phase4-swizzle-findings.md` (this file)

---

## Lessons Learned

1. **Always dequantize for CPU reference** when testing quantized kernels
   - Quantization is lossy → comparing quantized vs unquantized is meaningless
   
2. **Swizzle must be applied to both writes and reads** in CuTe
   - Can't mix linear writes with swizzled reads
   - CuTe's `cute::copy()` is the canonical way to handle swizzled layouts

3. **Test incrementally** when adding optimizations
   - Phase 3 → Phase 4 with no swizzle: ✅ Works (0.0039 error)
   - Phase 4 with swizzle: ❌ Breaks (21.69 error)
   - Isolates the problem to swizzle implementation

4. **CuTe Copy_Atom is necessary** for type conversions with swizzle
   - Can't just use `cute::copy(gA_fp32, sA_fp16)` without custom atom
   - This is why CUTLASS has dozens of Copy_Atom specializations

5. **Profiler-guided optimization** is critical
   - Don't assume swizzle helps without measuring bank conflicts
   - Use `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`

---

## Performance Summary

| Phase | Description | M=128 (GFLOPS) | M=1024 (GFLOPS) | % of Peak | Status |
|-------|-------------|----------------|-----------------|-----------|--------|
| 3.1   | Large tiles | 695 | ~4,500 | 12.65% | ✅ Shipped |
| 3.2   | Pipelined   | 1,347 | **6,564** | **18.45%** | ✅ Shipped |
| 4     | Swizzle (target) | 1,480 | 7,200 | 20.24% | ❌ Broken |
| 4     | No swizzle (actual) | 1,347 | 6,564 | 18.45% | ✅ **Working** |

**Current best**: **6.56 TFLOPS** (Phase 3 Part 2 / Phase 4 without swizzle)

---

## Conclusion

We've successfully debugged the Phase 4 implementation and identified:
1. **Test bug** (fixed): Dequantization missing → 1.32 error reduced to 0.0039 ✅
2. **Swizzle bug** (open): Linear write + swizzled read mismatch → 21.69 error ❌

**Next decision point**: Ship Phase 3 Part 2 as Phase 4, or invest 2-4 hours in proper swizzle implementation?

**User's choice** (based on earlier "no time pressure" statement): Continue with proper swizzle implementation using CuTe patterns.

**Action items**:
- [ ] Implement custom `Copy_Atom<FP32_to_FP16>` for global→shared transfer
- [ ] Test swizzled writes using `cute::copy()`
- [ ] Validate bank conflict reduction with `ncu` profiler
- [ ] Benchmark performance gain (expect 6.9-7.2 TFLOPS)

**Estimated completion**: 2-4 hours remaining work
