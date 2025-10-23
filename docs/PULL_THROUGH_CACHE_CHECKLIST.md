# Pull-Through Cache: Implementation Checklist
**Date**: October 20, 2025  
**Goal**: Fix BF16 memory leak with graceful migration path  
**Timeline**: 2 weeks

## ✅ Design Complete
- [x] Architecture document created
- [x] Migration path defined
- [x] Success metrics identified

## Phase 1: Cache Infrastructure (Days 1-2)

### Files to Create/Modify

**src/operators/TensorSlabCache.h**:
- [x] Design complete (documented)
- [ ] Add `TensorCacheKey` struct
- [ ] Add `CachedTensorData` struct  
- [ ] Add `CachedDataType` enum
- [ ] Add `getOrDecodeTensor<T>()` template method
- [ ] Add `evict_if_needed()` and `evict_lru_entry()`
- [ ] Add cache statistics (hits, misses, evictions)
- [ ] Add `invalidate(tensor_ptr)` for cache invalidation

**src/operators/TensorSlabCache.cpp**:
- [ ] Implement `getOrDecodeTensor<T>()` with cache lookup
- [ ] Implement LRU eviction logic
- [ ] Implement statistics tracking
- [ ] Add logging for cache events

### Unit Tests to Create

**tests/test_tensor_cache.cpp**:
- [ ] Cache hit/miss behavior
- [ ] LRU eviction correctness
- [ ] Memory capacity enforcement
- [ ] Statistics tracking
- [ ] Thread safety (single mutex)

### Validation
```bash
# Build and run cache tests
cmake --build build --target test_tensor_cache
./build/test_tensor_cache --gtest_filter="TensorCache.*"
```

## Phase 2: TensorBase Interface (Day 3)

### Files to Modify

**src/tensors/TensorBase.h**:
- [ ] Add `virtual const float* data_fp32() const`
- [ ] Add `virtual const bfloat16* data_bf16() const`
- [ ] Add `virtual DataType native_type() const = 0`
- [ ] Add `virtual size_t element_count() const = 0`
- [ ] Add `protected virtual void decode_to_fp32(float*) const = 0`
- [ ] Add `protected virtual void decode_to_bf16(bfloat16*) const = 0`
- [ ] Add `protected virtual const float* data_native_fp32() const { return nullptr; }`
- [ ] Add `protected virtual const bfloat16* data_native_bf16() const { return nullptr; }`
- [ ] Update `data()` to call `data_fp32()` and const_cast

**src/tensors/TensorBase.cpp**:
- [ ] Implement `data_fp32()` with fast path + cache lookup
- [ ] Implement `data_bf16()` with fast path + cache lookup
- [ ] Update `data()` wrapper

### Validation
```bash
# Should compile (but subclasses don't implement yet)
cmake --build build --target llaminar_core
```

## Phase 3: Fix BF16Tensor (Days 4-5) **← CRITICAL**

### Files to Modify

**src/tensors/BF16Tensor.h**:
- [ ] **REMOVE** `mutable std::vector<float> fp32_cache_`
- [ ] **REMOVE** `mutable bool cache_valid_`
- [ ] **REMOVE** `void update_cache() const`
- [ ] **REMOVE** `void invalidate_cache()`
- [ ] Add `DataType native_type() const override { return DataType::BF16; }`
- [ ] Add `size_t element_count() const override`
- [ ] Add `const bfloat16* data_native_bf16() const override`
- [ ] Add `void decode_to_fp32(float* dst) const override`
- [ ] Add `void decode_to_bf16(bfloat16* dst) const override`

**src/tensors/BF16Tensor.cpp**:
- [ ] Implement `decode_to_fp32()` with OMP parallel BF16→FP32
- [ ] Implement `decode_to_bf16()` with memcpy (already BF16)

### Memory Test

**tests/test_bf16_memory_fixed.cpp**:
- [ ] Create large BF16Tensor
- [ ] Call `data()` multiple times
- [ ] Measure memory: should be ~50% of FP32 + cache overhead
- [ ] Verify no per-tensor cache allocation

### Validation
```bash
# Build and run memory test
cmake --build build --target test_bf16_memory_fixed
./build/test_bf16_memory_fixed

# Expected output:
# BF16Tensor: 2000 MB (50% of FP32)
# Cache: 64 MB (shared)
# Total: 2064 MB (vs 5000 MB before)
```

### Integration Test

Run existing parity tests to ensure numerics unchanged:
```bash
ctest --test-dir build -R "ParityFrameworkTest" --output-on-failure
# All tests should pass (numerics identical to before)
```

## Phase 4: SimpleTensor (Day 6)

### Files to Modify

**src/tensors/SimpleTensor.h**:
- [ ] Add `DataType native_type() const override { return DataType::FP32; }`
- [ ] Add `size_t element_count() const override`
- [ ] Add `const float* data_native_fp32() const override`
- [ ] Add `void decode_to_fp32(float* dst) const override`
- [ ] Add `void decode_to_bf16(bfloat16* dst) const override`

**src/tensors/SimpleTensor.cpp**:
- [ ] Implement `decode_to_fp32()` with memcpy (already FP32)
- [ ] Implement `decode_to_bf16()` with OMP parallel FP32→BF16

### Validation
```bash
# Build and run tests
cmake --build build --target test_simple_tensor
./build/test_simple_tensor

# Verify:
# - FP32 access is direct (no cache lookup)
# - BF16 conversion uses cache
```

## Phase 5: QuantizedTensor (Days 7-8)

### Files to Modify

**src/tensors/QuantizedTensor.h**:
- [ ] Add `DataType native_type() const override { return DataType::QUANTIZED; }`
- [ ] Add `size_t element_count() const override`
- [ ] Add `void decode_to_fp32(float* dst) const override`
- [ ] Add `void decode_to_bf16(bfloat16* dst) const override`

**src/tensors/QuantizedTensor.cpp**:
- [ ] Implement `decode_to_fp32()` by calling `decodeBlock()` for all blocks
- [ ] Implement `decode_to_bf16()` via FP32 intermediate (or direct if supported)

### Notes
- Column slab cache path remains **unchanged** (separate cache)
- Full tensor decode only used when operator calls `data()`
- Most weight operations use slab path (not full tensor)

### Validation
```bash
# Verify full tensor decode works
./build/test_quantized_tensor

# Verify column slab path still works (unchanged)
./build/test_cosma_prefill
```

## Phase 6: Integration Testing (Days 9-10)

### Memory Profiling

**Test 1: Measure total cache usage**
```bash
# FP32 baseline
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "test" -n 100
# Note peak RSS: ~5171 MB

# BF16 with pull-through cache
LLAMINAR_QUANT_OUTPUT_BF16=1 \
LLAMINAR_TENSOR_CACHE_STATS=1 \
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "test" -n 100
# Expected: ~2585 MB + 64 MB cache = ~2650 MB (48% reduction)
```

**Test 2: Verify no per-tensor allocations**
```bash
# Run with memory profiling
valgrind --tool=massif ./build/llaminar ...
ms_print massif.out.<pid>
# Check: No large allocations proportional to number of tensors
```

### Parity Tests

**Full test suite**:
```bash
ctest --test-dir build -R "ParityFrameworkTest" --output-on-failure --verbose
# Expected: All 387/387 stages pass, rel L2 <1e-5
```

**Batch correctness**:
```bash
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.*"
# Expected: All stages pass
```

### Performance Benchmarking

**Throughput test**:
```bash
# FP32 baseline
./run_llaminar.sh --benchmark -m model.gguf -p "test prompt" -n 100
# Note: X.X tok/s

# BF16 mode
LLAMINAR_QUANT_OUTPUT_BF16=1 \
./run_llaminar.sh --benchmark -m model.gguf -p "test prompt" -n 100
# Expected: >95% of FP32 throughput
```

**Cache efficiency test**:
```bash
# Long sequence to stress cache
LLAMINAR_TENSOR_CACHE_STATS=1 \
LLAMINAR_QUANT_OUTPUT_BF16=1 \
./run_llaminar.sh -m model.gguf -p "$(cat long_prompt.txt)" -n 200

# Check output:
# [TensorCache] Hit rate: >70%
# [TensorCache] Evictions: <20% of misses
```

### Stress Tests

**Large batch size** (cache thrashing):
```bash
# Should still work, may have lower hit rate
LLAMINAR_TENSOR_CACHE_SIZE_MB=128 \
./run_batch_performance.sh --batch-size 32
```

**Small cache size** (eviction pressure):
```bash
# Verify graceful degradation
LLAMINAR_TENSOR_CACHE_SIZE_MB=16 \
./run_llaminar.sh -m model.gguf -p "test" -n 100
# Should work, but more cache misses
```

## Phase 7: Operator Migration (Ongoing, Optional)

### High-Priority Operators

**MPILinearOperator** (highest frequency):
- [ ] Check for `BF16Tensor`, call `bf16_data()` directly
- [ ] Fallback to `data()` for other types
- [ ] Test: Attention parity tests

**MPIAttentionOperator**:
- [ ] Q/K/V projections use direct BF16
- [ ] Scores/softmax stay FP32 (numerical stability)
- [ ] Context aggregation can use BF16
- [ ] Test: Attention stage parity

**MPISwiGLUOperator**:
- [ ] Gate/up/down projections use BF16
- [ ] SwiGLU activation in FP32 or BF16
- [ ] Test: FFN parity tests

### Low-Priority Operators

**MPIRMSNormOperator**:
- [ ] Can use BF16 input if allowed
- [ ] Output to BF16 tensor

**MPIRoPEOperator**:
- [ ] Can work directly on BF16
- [ ] In-place modification

### Migration Template

```cpp
// Before (uses cache):
float* input_data = input_tensor->data();

// After (direct BF16 access):
const bfloat16* input_bf16 = nullptr;
const float* input_fp32 = nullptr;

if (auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(input_tensor)) {
    input_bf16 = bf16->bf16_data();  // Direct, no cache
} else {
    input_fp32 = input_tensor->data();  // FP32 path
}

if (input_bf16) {
    // Use BF16-aware kernel
} else {
    // Use FP32 kernel
}
```

## Success Checklist

### Memory (Critical)
- [ ] BF16 mode uses ~50% memory vs FP32 (not 150%)
- [ ] Total cache size bounded to configured limit (64 MB default)
- [ ] No memory leaks (valgrind clean)
- [ ] Memory usage stable over time (no growth)

### Correctness (Critical)
- [ ] All parity tests pass (387/387 stages)
- [ ] Batch correctness tests pass (17/17 stages)
- [ ] Relative L2 error <1e-5 across all tests
- [ ] No numerical regressions vs previous version

### Performance (Important)
- [ ] Throughput within 95% of FP32 baseline
- [ ] Cache hit rate >70% for typical workloads
- [ ] Decode overhead <5% on average
- [ ] Eviction rate <20% of cache misses

### Robustness (Important)
- [ ] Works with various cache sizes (16 MB - 256 MB)
- [ ] Graceful degradation with small cache
- [ ] No crashes under memory pressure
- [ ] Thread-safe (single mutex verified)

## Environment Variables

### Configuration
```bash
LLAMINAR_QUANT_OUTPUT_BF16=1          # Enable BF16 activations
LLAMINAR_TENSOR_CACHE_SIZE_MB=64      # Cache size (default 64)
LLAMINAR_TENSOR_CACHE_STATS=1         # Log hit/miss statistics
LLAMINAR_TENSOR_CACHE_TRACE=1         # Verbose cache events
LLAMINAR_TENSOR_CACHE_DISABLE=1       # Disable cache (fallback to decode always)
```

### Testing
```bash
# Memory test with stats
LLAMINAR_QUANT_OUTPUT_BF16=1 \
LLAMINAR_TENSOR_CACHE_STATS=1 \
./run_llaminar.sh ...

# Performance test with profiling
LLAMINAR_QUANT_OUTPUT_BF16=1 \
LLAMINAR_TENSOR_CACHE_TRACE=1 \
./run_llaminar.sh --benchmark ...

# Stress test with small cache
LLAMINAR_TENSOR_CACHE_SIZE_MB=16 \
./run_llaminar.sh ...
```

## Rollback Plan

If issues arise, graceful rollback:

1. **Disable cache globally**:
   ```bash
   LLAMINAR_TENSOR_CACHE_DISABLE=1
   ```
   Falls back to per-tensor decode (slower but works)

2. **Revert to FP32**:
   ```bash
   unset LLAMINAR_QUANT_OUTPUT_BF16
   ```
   Uses SimpleTensor (baseline behavior)

3. **Git revert**:
   ```bash
   git revert <commit-hash>  # Revert cache implementation
   ```

## Next Steps

**Week 1 Priority**:
1. Implement cache infrastructure (Days 1-2)
2. Update TensorBase interface (Day 3)
3. **Fix BF16Tensor** (Days 4-5) ← CRITICAL for memory fix

**Week 2 Priority**:
1. SimpleTensor + QuantizedTensor (Days 6-8)
2. Testing and validation (Days 9-10)

**Week 3+ (Optional)**:
1. Operator migration for performance
2. Cache tuning and optimization
3. Production deployment

---

**This checklist tracks the pull-through cache implementation. Update status as tasks complete.**
