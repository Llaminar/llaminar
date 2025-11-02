# Phase 2.5: FP16 Async Copy Success (3.06× Speedup)

**Date**: January 27, 2025  
**Status**: ✅ **SUCCESS** - Exceeded 3× performance target  
**Performance**: 1,666 GFLOPS (3.06× speedup over Phase 2.0)

---

## Executive Summary

Successfully implemented **Phase 2.5: FP16 Async Copy** with explicit `cp.async` instruction usage via CuTe's `TiledCopy` API. Achieved **1,666 GFLOPS**, representing a **3.06× speedup** over Phase 2.0's manual copy approach (545 GFLOPS) and **3.92× speedup** over Phase 1 baseline (425 GFLOPS).

**Key Innovation**: FP16 input tensors + explicit SM80_CP_ASYNC copy atom + proper tensor partitioning.

---

## Performance Results

### Benchmark Configuration
- **Matrix Size**: m=32, n=896, k=896
- **Architecture**: NVIDIA Ampere (SM80) with Tensor Cores
- **Tile Size**: 64×64×16 (M×N×K)
- **Input Type**: `cutlass::half_t` (FP16)
- **Weight Format**: IQ4_NL quantized
- **Iterations**: 100 (timed)

### Performance Progression

| Phase | Technique | GFLOPS | Speedup vs Phase 1 | Speedup vs Previous |
|-------|-----------|--------|--------------------|--------------------|
| **Phase 1** | Baseline IQ4_NL GEMM | 425 | 1.0× | - |
| **Phase 2.0** | Tensor Core + Manual Copy | 545 | 1.28× | 1.28× |
| **Phase 2.5** | **FP16 + Async Copy** | **1,666** | **3.92×** | **3.06×** |

**Target**: 3× speedup over Phase 2.0 ✅ **EXCEEDED** (3.06×)

---

## Implementation Details

### File Changes

**`src/v2/kernels/cuda/CudaGemmKernelTensorCoreCuTe.cuh`** (Modified):

#### Change 1: Template on InputType (Lines 60-88)
```cpp
// BEFORE (FP32 only):
template<typename Decoder, int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
__global__ void quantized_gemm_kernel_cute(const float* A, ...)

// AFTER (FP32 or FP16):
template<typename InputType,  // FP32 (float) or FP16 (cutlass::half_t)
         typename Decoder, 
         int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
__global__ void quantized_gemm_kernel_cute(const InputType* A, ...)
```

**Rationale**: Enable runtime selection between FP32 (manual copy) and FP16 (async copy) paths.

#### Change 2: Explicit Async Copy with Tensor Partitioning (Lines 178-210)
```cpp
if constexpr (can_use_async) {
    // Create explicit SM80_CP_ASYNC copy atom
    using CopyAtomA = cute::Copy_Atom<
        cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, 
        cutlass::half_t
    >;
    
    auto copyA = cute::make_tiled_copy(
        CopyAtomA{},
        cute::Layout<cute::Shape<cute::_32, cute::_8>>{},  // Thread layout
        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}    // Value layout
    );
    
    // Partition tensors per-thread
    auto thr_copy_A = copyA.get_thread_slice(tid);
    auto tAgA = thr_copy_A.partition_S(gA_k);  // Source (gmem)
    auto tAsA = thr_copy_A.partition_D(sA);    // Destination (smem)
    
    // Async copy with cp.async
    copy(copyA, tAgA, tAsA);
    cp_async_fence();
    
    // ... (later in loop) ...
    
    cp_async_wait<0>();  // Wait for all async copies
} else {
    // FP32 manual copy path (Phase 2.0)
    for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
        smem_A_flat[row * TILE_K + col] = cutlass::half_t(gA_k(row, col));
    }
}
```

**Critical Insights**:
1. **Generic `copy()` doesn't auto-select cp.async**: Must create explicit `TiledCopy` with `SM80_CP_ASYNC` atom
2. **Tensor partitioning required**: Use `get_thread_slice()` + `partition_S/D()` to match instruction requirements
3. **CACHEALWAYS policy**: Ensures data stays in L2 cache for Tensor Core consumption

#### Change 3: Updated Launcher (Lines 280-300)
```cpp
template<typename InputType, typename Decoder, int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
inline cudaError_t launchQuantizedGemmCuTe(
    const InputType* A,  // Changed from const float*
    float* C, int m, int n, int k, Decoder decoder, cudaStream_t stream = 0)
{
    quantized_gemm_kernel_cute<InputType, Decoder, TILE_M, TILE_N, TILE_K>
        <<<blocks, threads, 0, stream>>>(A, C, m, n, k, decoder);
    return cudaGetLastError();
}
```

**`src/v2/kernels/cuda/CudaGemmVariantsTensorCore.cu`** (Modified):
- Line 76: Explicit `<float, ...>` template argument for Phase 2.0 FP32 path

**`tests/v2/performance/Perf__Phase2_5_TensorCore_FP16.cu`** (New):
- Created comprehensive FP16 performance test
- **Note**: Initial version corrupted due to heredoc Unicode issues
- **Fix**: Recreated with ASCII-only output formatting

**`tests/v2/CMakeLists.txt`** (Modified):
- Added Phase 2.5 FP16 test target
- Labels: `V2;Performance;CUDA;Phase2_5;TensorCore;FP16Input;AsyncCopy;GEMM`

---

## Technical Learnings

### 1. CuTe's `copy()` Behavior

**Myth**: `copy(gmem_tensor, smem_tensor)` automatically uses `cp.async` for gmem→smem transfers.

**Reality**: Generic `copy()` uses **default copy policy** (synchronous load/store). Must create **explicit TiledCopy** with `SM80_CP_ASYNC` atom.

**Evidence**:
- Initial implementation: `copy(gA_k, sA)` → 96 GFLOPS (5.6× **slower** than manual!)
- Explicit TiledCopy: `copy(copyA, tAgA, tAsA)` → 1,666 GFLOPS (17× faster)

### 2. Tensor Partitioning Requirements

**Error**: `"Src/Dst partitioning does not match the instruction requirement"`

**Cause**: Passing unpartitioned tensors to TiledCopy.

**Solution**: Per-thread partitioning via:
```cpp
auto thr_copy_A = copyA.get_thread_slice(tid);
auto tAgA = thr_copy_A.partition_S(source);  // Partition source
auto tAsA = thr_copy_A.partition_D(dest);    // Partition destination
```

**Rationale**: Each thread operates on a different slice of the tile. CuTe needs to know which slice belongs to which thread.

### 3. FP16 vs FP32 Input Trade-offs

| Aspect | FP32 Input (Phase 2.0) | FP16 Input (Phase 2.5) |
|--------|------------------------|------------------------|
| **Memory Bandwidth** | 2× higher (4 bytes/element) | 2× lower (2 bytes/element) |
| **Conversion Overhead** | FP32→FP16 in smem copy loop | None (already FP16) |
| **Async Copy** | Not applicable (needs FP16) | ✅ Enabled via cp.async |
| **Performance** | 545 GFLOPS | **1,666 GFLOPS** |
| **Speedup** | 1.28× | **3.06×** |

**Recommendation**: Always use FP16 input for Tensor Core kernels when possible.

### 4. File Creation Best Practices

**Issue**: Heredoc with Unicode characters corrupted test file (254 lines instead of 130).

**Root Cause**: Shell heredoc doesn't handle Unicode properly → mangled characters in source.

**Solution**: ASCII-only output formatting in test files.

**Example**:
```cpp
// ❌ BAD (causes compilation errors):
std::cout << "╔════════════════╗\n";  // Unicode box-drawing

// ✅ GOOD (ASCII-only):
std::cout << "===== PHASE 2.5 =====\n";
```

---

## Debugging Timeline

### Initial Failure (96 GFLOPS)

**Symptoms**:
- Performance: 96 GFLOPS (5.6× **slower** than Phase 2.0)
- Speedup: 0.177× (regression, not improvement)

**Diagnosis**:
- Generic `copy(gA_k, sA)` did not use cp.async
- Fell back to synchronous load/store
- Extra overhead from function call

### First Fix Attempt (Compilation Error)

**Approach**: Create TiledCopy but pass unpartitioned tensors.

**Error**:
```
error: static assertion failed with "CopyAtom: Src/Dst partitioning does not 
match the instruction requirement."
```

**Root Cause**: TiledCopy expects per-thread partitioned tensors.

### Second Fix (Success!)

**Changes**:
1. Created explicit `SM80_CP_ASYNC_CACHEALWAYS` copy atom
2. Partitioned source and destination tensors per-thread
3. Passed partitioned tensors to `copy()`

**Result**: 1,666 GFLOPS (3.06× speedup) ✅

---

## Next Steps

### Phase 2.7: Multi-Stage Pipeline (Planned)

**Objective**: Overlap async copy of tile K+1 with MMA of tile K.

**Expected Speedup**: 1.5-2× over Phase 2.5 (2,500-3,300 GFLOPS).

**Implementation**:
```cpp
// Double-buffered shared memory
__shared__ cutlass::half_t smem_A[2][TILE_M * TILE_K];
int read_stage = 0, write_stage = 1;

for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    // Load tile K+1 to write_stage (async)
    if (k_tile + 1 < num_k_tiles) {
        auto gA_next = gA(_, _, k_tile + 1);
        copy(copyA, gA_next, smem_A[write_stage]);
        cp_async_fence();
    }
    
    // Compute tile K from read_stage (overlapped!)
    cp_async_wait<0>();
    __syncthreads();
    gemm(tiled_mma, smem_A[read_stage], ...);
    
    // Swap buffers
    read_stage ^= 1;
    write_stage ^= 1;
}
```

**Key Benefits**:
- Hide memory latency behind computation
- Maximize Tensor Core utilization
- Approach peak hardware GFLOPS

### Phase 3: Tile Size Autotuning (Planned)

**Objective**: Find optimal TILE_M, TILE_N, TILE_K for different matrix sizes.

**Approach**: Grid search or heuristic-based tuning.

**Expected Benefit**: 10-30% improvement via better register/smem utilization.

---

## References

**Documentation**:
- `.github/instructions/cutlass.instructions.md`: Comprehensive CUTLASS/CuTe guide
- CUTLASS CuTe Tutorial: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md

**Related Changes**:
- `changelog/2025-01-26-cutlass-instructions-guide.md`: Created comprehensive guide
- `changelog/2025-01-25-phase2-tensorcore-implementation.md`: Phase 2.0 results

**Performance Baseline**:
- Phase 1: 425 GFLOPS (IQ4_NL GEMM baseline)
- Phase 2.0: 545 GFLOPS (Tensor Core + manual copy)
- Phase 2.5: **1,666 GFLOPS** (Tensor Core + async copy) ← **This work**

---

## Conclusion

Phase 2.5 successfully demonstrates the power of **explicit async memory transfer** via CuTe's `TiledCopy` API with `SM80_CP_ASYNC` copy atoms. By eliminating FP32→FP16 conversion overhead and properly partitioning tensors for asynchronous copy, we achieved a **3.06× speedup** over Phase 2.0 and **3.92× speedup** over the Phase 1 baseline.

**Key Takeaways**:
1. ✅ Generic `copy()` does **not** automatically use cp.async
2. ✅ Must create explicit `TiledCopy` with `SM80_CP_ASYNC` atom
3. ✅ Tensor partitioning (`get_thread_slice`, `partition_S/D`) is mandatory
4. ✅ FP16 input eliminates conversion overhead and enables true async copy
5. ✅ Performance: **1,666 GFLOPS** (exceeds 3× target)

**Next Goal**: Phase 2.7 multi-stage pipeline targeting **2,500-3,300 GFLOPS**.
