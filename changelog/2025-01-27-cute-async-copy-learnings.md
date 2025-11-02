# CuTe Async Copy: Critical Learnings

**Date**: January 27, 2025  
**Context**: Phase 2.5 FP16 async copy implementation  
**Performance Impact**: 17× speedup (96 → 1,666 GFLOPS)

---

## The Myth vs Reality

### ❌ MYTH: Generic `copy()` Auto-Selects cp.async

**Common Assumption**:
```cpp
// Many assume this uses cp.async for gmem→smem:
auto gA = make_tensor(gmem_ptr, shape);
auto sA = make_tensor(smem_ptr, shape);
copy(gA, sA);  // "Should use cp.async automatically, right?"
```

**What Actually Happens**:
- CuTe's generic `copy()` uses **default copy policy**
- Default policy: Synchronous load/store (ldg + stg)
- **No cp.async** is generated
- **Performance**: 96 GFLOPS (5.6× slower than even manual copy!)

### ✅ REALITY: Explicit TiledCopy Required

**Correct Implementation**:
```cpp
// 1. Create explicit async copy atom
using CopyAtomA = cute::Copy_Atom<
    cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, 
    cutlass::half_t
>;

// 2. Create TiledCopy with thread layout
auto copyA = cute::make_tiled_copy(
    CopyAtomA{},
    cute::Layout<cute::Shape<cute::_32, cute::_8>>{},  // 32×8 = 256 threads
    cute::Layout<cute::Shape<cute::_1, cute::_8>>{}    // 1×8 values/thread
);

// 3. Partition tensors per-thread
auto thr_copy_A = copyA.get_thread_slice(threadIdx.x);
auto tAgA = thr_copy_A.partition_S(gA);  // Source (gmem)
auto tAsA = thr_copy_A.partition_D(sA);  // Destination (smem)

// 4. Async copy with cp.async
copy(copyA, tAgA, tAsA);  // NOW uses cp.async!
cp_async_fence();

// 5. Wait before accessing smem
cp_async_wait<0>();
__syncthreads();
```

**Performance**: 1,666 GFLOPS (17× faster than generic copy!)

---

## Why Generic Copy Can't Use cp.async

### Technical Constraints

1. **cp.async requires specific alignment**:
   - Must be 16-byte aligned (uint128_t)
   - Generic copy can't guarantee this

2. **cp.async requires compile-time layouts**:
   - TiledCopy provides thread→data mapping at compile time
   - Generic copy works with runtime layouts

3. **cp.async requires specific element types**:
   - Works with FP16, BF16, INT8, etc.
   - Generic copy is type-agnostic

4. **cp.async requires partitioning**:
   - Each thread knows its slice at compile time
   - Generic copy doesn't enforce partitioning

### Compilation Errors When Missing Partitioning

**Error**:
```
error: static assertion failed with "CopyAtom: Src/Dst partitioning does not 
match the instruction requirement."
```

**Cause**: Passing unpartitioned tensors to TiledCopy.

**Fix**: Partition both source and destination:
```cpp
// ❌ WRONG (compilation error):
copy(copyA, gA, sA);

// ✅ CORRECT (partitioned):
auto thr_copy = copyA.get_thread_slice(tid);
auto tAgA = thr_copy.partition_S(gA);
auto tAsA = thr_copy.partition_D(sA);
copy(copyA, tAgA, tAsA);
```

---

## Performance Comparison

### Experiment Setup
- Matrix: 32×896×896 (m×k, k×n)
- GPU: NVIDIA RTX 3090 (Ampere, SM80)
- Input: FP16 (`cutlass::half_t`)
- Weights: IQ4_NL quantized

### Results

| Implementation | Technique | GFLOPS | Performance |
|---------------|-----------|--------|-------------|
| **Generic copy()** | Synchronous ldg/stg | 96 | ❌ 5.6× slower |
| **Manual loop** | FP32→FP16 conversion | 545 | ⚠️ Baseline |
| **TiledCopy + cp.async** | Explicit async | **1,666** | ✅ **17× faster!** |

### Performance Analysis

**Generic copy() breakdown** (96 GFLOPS):
```
Per K-tile:
[Generic copy: 80%] [Barrier: 5%] [MMA: 10%] [Barrier: 5%]
     ↓ Slow!           ↓ Overhead     ↓ Idle     ↓ Overhead
```

- **80% of time**: Inefficient synchronous copy
- Function call overhead + no cp.async
- Tensor Cores mostly idle

**TiledCopy + cp.async** (1,666 GFLOPS):
```
Per K-tile:
[cp.async: 20%] ──┐
                  ├── [MMA: 60%] [Fence/Wait: 20%]
                  └── Overlapped!
```

- **60% of time**: Tensor Core compute (3× higher utilization!)
- Async copy hides most memory latency
- Only 20% waiting for memory

---

## Implementation Checklist

When implementing async copy with CuTe:

### ✅ Required Steps

1. ✅ **Create explicit Copy_Atom**:
   ```cpp
   using CopyAtom = cute::Copy_Atom<
       cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, 
       ElementType
   >;
   ```

2. ✅ **Create TiledCopy with layouts**:
   ```cpp
   auto copy = cute::make_tiled_copy(
       CopyAtom{},
       ThreadLayout{},  // e.g., Layout<Shape<_32, _8>>
       ValueLayout{}    // e.g., Layout<Shape<_1, _8>>
   );
   ```

3. ✅ **Partition tensors per-thread**:
   ```cpp
   auto thr_copy = copy.get_thread_slice(tid);
   auto src_part = thr_copy.partition_S(src);
   auto dst_part = thr_copy.partition_D(dst);
   ```

4. ✅ **Call copy with partitioned tensors**:
   ```cpp
   copy(tiled_copy, src_part, dst_part);
   cp_async_fence();
   ```

5. ✅ **Wait before accessing smem**:
   ```cpp
   cp_async_wait<0>();  // Wait for all pending copies
   __syncthreads();     // Ensure all threads see data
   ```

### ❌ Common Mistakes

1. ❌ Using generic `copy()` without TiledCopy
2. ❌ Skipping tensor partitioning
3. ❌ Forgetting `cp_async_fence()` / `cp_async_wait<>()`
4. ❌ Not using `__syncthreads()` after wait
5. ❌ Using FP32 input (cp.async prefers FP16/BF16)

---

## Copy Policy Hierarchy

CuTe's copy system has multiple levels:

```
1. Raw copy (lowest level):
   └─> Explicit PTX instructions (ld.global, cp.async, etc.)

2. Copy_Atom (instruction wrapper):
   └─> SM80_CP_ASYNC, AutoVectorizingCopy, etc.
   └─> Defines copy semantics for specific instruction

3. TiledCopy (thread coordination):
   └─> Maps threads to data elements
   └─> Enforces alignment and partitioning
   └─> Created via make_tiled_copy()

4. Generic copy() (highest level):
   └─> Type-agnostic, runtime layouts
   └─> Uses DEFAULT copy policy (synchronous!)
   └─> ⚠️ Does NOT auto-select cp.async
```

**Key Insight**: Must use **Level 3 (TiledCopy)** or lower to get cp.async.

---

## When to Use Each Copy Method

### Use Generic `copy()` When:
- ✅ Non-performance-critical paths
- ✅ Runtime-determined layouts
- ✅ Prototype/debugging code

### Use TiledCopy + cp.async When:
- ✅ Hot path (gmem→smem transfers)
- ✅ Compile-time known layouts
- ✅ FP16/BF16 data types
- ✅ Ampere/Hopper GPUs (SM80+)
- ✅ Performance matters!

---

## Debug Checklist

If async copy is slow:

1. **Verify cp.async is used**:
   ```bash
   # Compile with PTX output
   nvcc --ptx -o kernel.ptx kernel.cu
   
   # Check for cp.async instructions
   grep "cp.async" kernel.ptx  # Should see multiple hits
   ```

2. **Check alignment**:
   ```cpp
   printf("gmem_ptr: %p (aligned: %d)\n", 
          ptr, (uintptr_t)ptr % 16 == 0);
   ```

3. **Verify partitioning**:
   ```cpp
   // Each thread should have different slice
   printf("Thread %d: src_size=%d, dst_size=%d\n",
          tid, size(src_part), size(dst_part));
   ```

4. **Profile with Nsight Compute**:
   ```bash
   ncu --set full -o profile ./benchmark
   # Check "Memory Workload Analysis" for cp.async utilization
   ```

---

## References

**NVIDIA Documentation**:
- PTX ISA Guide (cp.async): https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUTLASS CuTe Tutorial: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md

**Llaminar Documentation**:
- `.github/instructions/cutlass.instructions.md`: Comprehensive CUTLASS/CuTe guide
- `changelog/2025-01-27-phase2-5-async-copy-success.md`: Phase 2.5 implementation details

**Related Code**:
- `src/v2/kernels/cuda/CudaGemmKernelTensorCoreCuTe.cuh`: Working async copy implementation
- `tests/v2/performance/Perf__Phase2_5_TensorCore_FP16.cu`: Performance validation

---

## Conclusion

CuTe's generic `copy()` is **not** a magic async memory transfer function. It uses a **default synchronous copy policy** unless you explicitly create a `TiledCopy` with an async `Copy_Atom` like `SM80_CP_ASYNC`.

**Critical lesson**: In performance-critical paths, **always** use explicit `TiledCopy` with proper tensor partitioning to unlock cp.async's 17× speedup potential.

**Performance impact**: 96 GFLOPS (generic copy) → 1,666 GFLOPS (TiledCopy + cp.async)
