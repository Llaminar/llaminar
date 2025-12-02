# GEMM Kernel Caching Performance Fix

**Date**: 2025-01-XX (Session)
**Author**: Claude via GitHub Copilot

## Summary

Implemented GEMM kernel caching to eliminate repeated weight repacking overhead during inference, resulting in **~30% performance improvement**.

## Root Cause Analysis

Profiling with `perf record` and `perf stat` revealed a critical performance bottleneck:

### Before Fix Profile
```
~45% in libgomp (OpenMP synchronization)
~15% in pack_weights_generic() - Weight repacking!
~5.7% in unpack_superblock_to_int8 - Dequantization

Hardware Counters:
- IPC: 0.28 (very low, target is 2-4)
- Cache miss rate: 35%
- LLC miss rate: 70%
```

### Problem Identified
The `createGemm()` method in `GemmOp.h` was being called for **every GEMM operation** during the forward pass. This caused:

1. **Repeated kernel construction** - `QuantisedGemmKernel` constructor runs `pack_weights_generic()`
2. **Weight repacking on every token** - For a 24-layer model with ~73 GEMMs per forward pass, this meant repacking ALL weights 73×N times (where N = number of tokens)
3. **Cache pollution** - The massive repeated memory operations destroyed cache locality

## Solution

### 1. Added GEMM Kernel Caching to TensorBase

Added `getOrCreateGemm()` method that caches the GEMM kernel on first creation:

```cpp
// src/v2/tensors/Tensors.h
class TensorBase {
public:
    ITensorGemm *getOrCreateGemm() {
        std::lock_guard<std::mutex> lock(gemm_cache_mutex_);
        if (!cached_gemm_) {
            cached_gemm_ = createGemm();  // Only called once!
        }
        return cached_gemm_.get();
    }

protected:
    mutable std::mutex gemm_cache_mutex_;
    mutable std::unique_ptr<ITensorGemm> cached_gemm_;
};
```

### 2. Made TensorBase Non-Copyable

Since `std::mutex` and `std::unique_ptr` are non-copyable, explicitly deleted copy operations:

```cpp
class TensorBase {
public:
    TensorBase(const TensorBase &) = delete;
    TensorBase &operator=(const TensorBase &) = delete;
    TensorBase(TensorBase &&) = delete;
    TensorBase &operator=(TensorBase &&) = delete;
protected:
    TensorBase() = default;  // Allow derived classes to construct
};
```

### 3. Updated GemmOp to Use Cached Kernels

Changed from `createGemm()` (new kernel every call) to `getOrCreateGemm()` (cached):

```cpp
// src/v2/pipelines/ops/GemmOp.h
bool operator()(const TensorBase *A, TensorBase *W, TensorBase *C, ...) {
    // BEFORE: auto gemm_kernel = W->createGemm();  // New kernel every call!
    auto *gemm_kernel = W->getOrCreateGemm();       // Cached kernel
    // ...
}
```

### 4. Fixed Test Using Tensor Copy

Updated `Test__AttentionParity.cpp` to use `copyFrom()` instead of copy construction:

```cpp
// BEFORE (broken with non-copyable TensorBase):
auto Q_llama = std::make_shared<FP32Tensor>(*Q);

// AFTER (correct):
auto Q_llama = std::make_shared<FP32Tensor>(Q->shape());
Q_llama->copyFrom(Q.get());
```

## Results

### After Fix Profile
```
~36% in libgomp (OpenMP synchronization)
~7.7% in decodeBlock (actual dequantization - necessary work)
~0.9% in pack_weights_generic() - Down from 15%!
```

### Performance Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `pack_weights_generic` CPU % | ~15% | ~0.9% | **16× reduction** |
| Total inference time (30 tokens) | ~11.5s | ~8s | **~30% faster** |
| Decode throughput | ~2.6 tok/s | ~3.8 tok/s | **~46% improvement** |

## Files Changed

1. **`src/v2/tensors/Tensors.h`**
   - Added `getOrCreateGemm()` method with mutex-protected lazy initialization
   - Added `gemm_cache_mutex_` and `cached_gemm_` protected members
   - Made TensorBase non-copyable (deleted copy/move constructors and assignment)
   - Added protected default constructor

2. **`src/v2/pipelines/ops/GemmOp.h`**
   - Changed `W->createGemm()` to `W->getOrCreateGemm()` in both `operator()` overloads

3. **`tests/v2/unit/Test__AttentionParity.cpp`**
   - Fixed tensor copying to use `copyFrom()` instead of copy construction

## Remaining Bottlenecks

The profile after the fix shows remaining optimization opportunities:

1. **~36% OpenMP overhead** - Synchronization/barriers dominate
   - Potential: Use OpenMP tasks instead of parallel for, reduce critical sections

2. **~7.7% decodeBlock** - Actual computation (necessary)
   - Potential: SIMD optimizations, better memory access patterns

3. **Cache miss rates still high** - 63% cache miss rate
   - Potential: Better prefetching, NUMA-aware allocation improvements

## Testing

All tests pass:
- 108 V2 unit tests: ✅ PASSED
- V2 Integration tests: ✅ PASSED
- Manual inference testing: ✅ Working correctly

## Future Work

1. Profile and optimize OpenMP synchronization overhead
2. Investigate per-layer or per-device kernel caching strategies
3. Consider kernel fusion to reduce synchronization points
4. Improve cache locality with better data layout
