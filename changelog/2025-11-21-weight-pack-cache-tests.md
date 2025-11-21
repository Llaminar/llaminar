# WeightPack Cache Unit Tests Implementation

**Date**: 2025-11-21  
**Author**: David Sanftenberg  
**Type**: Test Coverage + Memory Analysis

## Summary

Implemented comprehensive unit tests for OneDNN INT8 weight packing cache mechanism stored in `TensorBase::cache_`. Tests document current behavior and validate cache correctness, revealing a **36% memory optimization opportunity** by freeing original quantized buffers after repacking to INT8.

## Background

### Architecture

OneDNN GEMM kernels convert quantized weights (Q4_0, IQ4_NL, etc.) to INT8 column-major format for optimal performance:

1. **Original Format**: Q4_0 (~4.5 bits/weight) stored in row-major blocks
2. **WeightPack Format**: INT8 (8 bits/weight) + per-column FP32 scales
3. **Cache Location**: `TensorBase::cache_` (mutable `std::any`)
4. **Lazy Population**: `OneDNNGemmKernel` checks `cache_.has_value()`, calls `pack_weights_to_int8()` on miss

### Memory Overhead Issue

**Current Behavior**: Both original quantized buffer AND WeightPack coexist in memory after first GEMM operation.

**Example** (2048×4096 weight matrix):
- Q4_0 blocks: 4.7 MB
- WeightPack: 8.4 MB (8.4MB INT8 + 8KB scales)
- **Total: 13.1 MB** (current)
- **Desired: 8.4 MB** (after deallocation)
- **Savings: 36%**

For multi-billion parameter models, this represents **hundreds of megabytes** of wasted memory per weight tensor.

## Test Coverage

### File Structure

```
tests/v2/unit/Test__WeightPackCache.cpp (315 lines)
├── Test fixture: Test__WeightPackCache
│   ├── createQ4_0Tensor(): Generate test weight tensors
│   └── verifyCacheExists(): Validate cache_ has WeightPack
└── 6 Test Cases (all passing)
```

### Test Cases

#### 1. CacheLazyInitialization
**Purpose**: Verify cache NOT created until needed  
**Validates**: `cache_.has_value()` returns false initially  
**Runtime**: ~7ms

#### 2. CachePopulationByKernel
**Purpose**: Verify `pack_weights_to_int8()` populates cache correctly  
**Validates**:
- Cache populated after packing
- WeightPack has correct dimensions (K×N)
- Per-column scales allocated (N scales)

**Runtime**: ~5ms

#### 3. CacheCorrectness
**Purpose**: Verify repacking produces identical data  
**Validates**:
- Fresh pack matches cached pack
- INT8 data byte-for-byte identical
- FP32 scales match (FLOAT_EQ tolerance)

**Runtime**: ~4ms

#### 4. MultipleKernelsShareCache
**Purpose**: Verify cache persists across kernel instances  
**Validates**:
- Second kernel accesses same cached WeightPack
- No redundant repacking occurs
- Cache shared via `TensorBase*` pointer

**Runtime**: <1ms

#### 5. ThreadSafeCacheAccess
**Purpose**: Verify concurrent packing is thread-safe  
**Validates**:
- 4 threads concurrently call `pack_weights_to_int8()`
- No data corruption or exceptions
- All threads produce valid WeightPacks

**Runtime**: ~6ms

#### 6. MemoryFootprintAfterCaching
**Purpose**: Document memory overhead and optimization potential  
**Validates**:
- Measures Q4_0 buffer size (4.7 MB)
- Measures WeightPack size (8.4 MB)
- Calculates current overhead (0.39× FP32 baseline)
- Calculates desired overhead (0.25× FP32 baseline)
- **Documents 36% potential savings**

**Runtime**: ~2590ms (large 2048×4096 matrix)

**Log Output** (with INFO level):
```
Q4_0 weight memory: 4718592 bytes
  N=2048, K=4096, blocks_per_row=128
WeightPack memory: 8396800 bytes
  INT8 data: 8388608 bytes
  FP32 scales: 8192 bytes
Current memory overhead: 0.390869× (Q4_0 + WeightPack)
Desired memory overhead: 0.250244× (WeightPack only)
Potential memory savings: 35.9775%
```

## Build Integration

### CMake Configuration

```cmake
# tests/v2/CMakeLists.txt
add_executable(v2_test_weight_pack_cache
    unit/Test__WeightPackCache.cpp
)

target_link_libraries(v2_test_weight_pack_cache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)

add_v2_test(V2_Unit_WeightPackCache
    COMMAND v2_test_weight_pack_cache
    LABELS "V2;Unit;TensorOperations;WeightCaching;INT8;MemoryOptimization;OneDNN;CPU"
    MPI_PROCS 1
)
```

### Test Execution

```bash
# Direct execution
./build_v2/tests/v2/v2_test_weight_pack_cache

# Via CTest
ctest --test-dir build_v2 -R "V2_Unit_WeightPackCache" --verbose

# Memory footprint details (with logging)
LLAMINAR_LOG_LEVEL=INFO ./build_v2/tests/v2/v2_test_weight_pack_cache \
  --gtest_filter="*MemoryFootprint*"
```

## Test Results

```
[==========] Running 6 tests from 1 test suite.
[----------] 6 tests from Test__WeightPackCache
[ RUN      ] Test__WeightPackCache.CacheLazyInitialization
[       OK ] Test__WeightPackCache.CacheLazyInitialization (7 ms)
[ RUN      ] Test__WeightPackCache.CachePopulationByKernel
[       OK ] Test__WeightPackCache.CachePopulationByKernel (5 ms)
[ RUN      ] Test__WeightPackCache.CacheCorrectness
[       OK ] Test__WeightPackCache.CacheCorrectness (4 ms)
[ RUN      ] Test__WeightPackCache.MultipleKernelsShareCache
[       OK ] Test__WeightPackCache.MultipleKernelsShareCache (0 ms)
[ RUN      ] Test__WeightPackCache.ThreadSafeCacheAccess
[       OK ] Test__WeightPackCache.ThreadSafeCacheAccess (6 ms)
[ RUN      ] Test__WeightPackCache.MemoryFootprintAfterCaching
[       OK ] Test__WeightPackCache.MemoryFootprintAfterCaching (2590 ms)
[----------] 6 tests from Test__WeightPackCache (2613 ms total)

[  PASSED  ] 6 tests.
```

**Total Runtime**: ~2.6 seconds (dominated by large matrix packing in memory footprint test)  
**Pass Rate**: 100% (6/6)

## Memory Analysis

### Current Memory Layout

For a 2048×4096 weight matrix after INT8 repacking:

| Component | Format | Size | Notes |
|-----------|--------|------|-------|
| Q4_0 Blocks | Row-major, 4-bit nibbles + FP16 scales | 4.7 MB | Original quantized format |
| WeightPack INT8 | Column-major, 8-bit signed | 8.4 MB | Transposed for OneDNN |
| WeightPack Scales | Per-column FP32 | 8 KB | N scales (2048 floats) |
| **Total** | - | **13.1 MB** | Both coexist in memory |

### Optimized Memory Layout (Post-Deallocation)

| Component | Format | Size | Notes |
|-----------|--------|------|-------|
| ~~Q4_0 Blocks~~ | ~~Deallocated~~ | ~~4.7 MB~~ | Freed after repacking |
| WeightPack INT8 | Column-major, 8-bit signed | 8.4 MB | Only format needed |
| WeightPack Scales | Per-column FP32 | 8 KB | Preserved |
| **Total** | - | **8.4 MB** | **36% reduction** |

### Scaling Analysis

For various model sizes:

| Model Size | Weight Params | Q4_0 Size | Current Total | Optimized | Savings |
|------------|---------------|-----------|---------------|-----------|---------|
| 0.5B | ~250M | ~140 MB | ~390 MB | ~250 MB | ~140 MB (36%) |
| 7B | ~3.5B | ~2 GB | ~5.5 GB | ~3.5 GB | ~2 GB (36%) |
| 13B | ~6.5B | ~3.7 GB | ~10.2 GB | ~6.5 GB | ~3.7 GB (36%) |
| 70B | ~35B | ~20 GB | ~55 GB | ~35 GB | ~20 GB (36%) |

**Note**: Savings apply only to weight tensors that undergo INT8 repacking (typically 80-90% of model weights for OneDNN GEMM operations).

## Implementation Roadmap

### Phase 1: Test Coverage (✅ Complete)
- [x] Create `Test__WeightPackCache.cpp` with 6 test cases
- [x] Integrate with CMake build system
- [x] Validate cache behavior and correctness
- [x] Document memory overhead

### Phase 2: Memory Deallocation (Pending)
**Goal**: Implement `TensorBase::release_native_buffer()` or similar

**Requirements**:
1. Only apply to **weight tensors** (read-only), NOT activations
2. Call after `cache_.has_value() == true` (cache populated)
3. Mark buffer as "deallocated" (prevent double-free)
4. Update tests to verify actual memory reduction

**Proposed API**:
```cpp
// In TensorBase (or derived quantized tensor classes)
void release_native_buffer() {
    if (is_weight_tensor_ && cache_.has_value()) {
        raw_data_.clear();          // Free original buffer
        raw_data_.shrink_to_fit();  // Release memory
        buffer_deallocated_ = true; // Flag for safety
    }
}
```

**Call Site** (in `OneDNNGemmKernel::multiply_activations()`):
```cpp
if (!weight_tensor_->cache_.has_value()) {
    weight_tensor_->cache_ = pack_weights_to_int8(*weight_tensor_, k, n);
    weight_tensor_->release_native_buffer(); // FREE ORIGINAL BUFFER
}
```

### Phase 3: Validation (Pending)
- [ ] Update `MemoryFootprintAfterCaching` test to verify actual deallocation
- [ ] Benchmark memory usage before/after on full models
- [ ] Ensure no performance regression (cache hit path unchanged)
- [ ] Verify thread safety of deallocation (multiple kernels)

## Technical Details

### pack_weights_to_int8() Workflow

```cpp
// src/v2/kernels/cpu/gemm_v4/OneDNNGemmAdapter.h
WeightPack pack_weights_to_int8(const TensorBase& tensor, int K, int N) {
    // Step 1: Convert quantized format → INT8 row-major + scales
    auto [int8_row_major, scales] = tensor.to_int8_perchannel();
    
    // Step 2: Transpose to column-major [K, N] for OneDNN
    std::vector<int8_t> int8_col_major(K * N);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            int8_col_major[j * K + i] = int8_row_major[i * N + j];
        }
    }
    
    // Step 3: Return packed representation
    return WeightPack{
        .data = std::move(int8_col_major),
        .col_scales = std::move(scales),
        .rows = K,
        .cols = N
    };
}
```

### Cache Access Pattern

```cpp
// src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h (lines 1070-1090)
bool OneDNNGemmKernel::multiply_activations(...) {
    // Check cache
    if (!weight_tensor_->cache_.has_value()) {
        // Cache miss: pack weights to INT8
        weight_tensor_->cache_ = pack_weights_to_int8(*weight_tensor_, k, n);
        
        // TODO: Free original buffer here
        // weight_tensor_->release_native_buffer();
    }
    
    // Cache hit: extract packed weights
    const auto& weight_pack = std::any_cast<const WeightPack&>(
        weight_tensor_->cache_
    );
    
    // Use cached INT8 weights for OneDNN GEMM
    // ... (no repacking needed)
}
```

## Constraints and Safety

### Read-Only Constraint
- **Deallocation ONLY for weight tensors** (marked as read-only at load time)
- **NEVER deallocate activation tensors** (reused across decode steps)
- Rationale: Weights are loaded once and never modified; activations are reused

### Thread Safety
- `TensorBase::cache_` is mutable (allows lazy init in const methods)
- Multiple `OneDNNGemmKernel` instances may share same `TensorBase*`
- First kernel to populate cache "wins" (others hit cache)
- Deallocation must be thread-safe (consider `std::call_once` or mutex)

### Memory Safety
- Mark buffer as "deallocated" to prevent double-free
- Accessing `raw_data_` after deallocation should be safe (empty vector)
- Cache must remain valid (not affected by buffer deallocation)

## Related Files

### Source Code
- `src/v2/tensors/Tensors.h` (line 358): `TensorBase::cache_` definition
- `src/v2/kernels/cpu/gemm_v4/OneDNNGemmAdapter.h`: `pack_weights_to_int8()`
- `src/v2/kernels/cpu/gemm_v4/OneDNNGemmKernel.h`: Cache population logic

### Tests
- `tests/v2/unit/Test__WeightPackCache.cpp`: Complete test suite
- `tests/v2/CMakeLists.txt`: Build configuration

### Documentation
- `.github/copilot-instructions.md`: Development guidelines
- `.github/instructions/llaminar-v2-architecture.instructions.md`: V2 architecture

## Next Steps

1. **Implement `release_native_buffer()`** in quantized tensor classes (Q4_0Tensor, IQ4_NLTensor, etc.)
2. **Call deallocation** after cache population in `OneDNNGemmKernel::multiply_activations()`
3. **Update test** to verify actual memory reduction (not just calculate potential savings)
4. **Benchmark** full model memory footprint before/after on multi-billion parameter models
5. **Document** in changelog with real-world memory savings data

## Conclusion

This test suite establishes a foundation for safely implementing the 36% memory optimization. All 6 tests pass, validating cache correctness and documenting the memory overhead issue. The next phase will implement actual buffer deallocation with appropriate safety constraints (weight tensors only, thread-safe, no double-free).

**Key Insight**: For large models (7B-70B parameters), this optimization saves **2-20 GB of memory** per node, enabling larger batch sizes or fitting larger models in limited memory environments.
