# Pull-Through Tensor Cache Architecture
**Date**: October 20, 2025  
**Status**: PROPOSED DESIGN  
**Author**: David Sanftenberg

## Motivation

**Current Problem**: BF16Tensor has persistent `fp32_cache_` that causes memory explosion (150% overhead instead of 50% savings).

**Original Fix**: Remove cache, break all operators, fix them all at once (risky).

**This Solution**: Centralized LRU cache with pull-through pattern (graceful migration).

## Core Concept

**Pull-Through Cache Pattern**:
```
Operator calls tensor->data_fp32()
    ↓
TensorBase checks shared LRU cache
    ↓
Cache miss? → Decode tensor → Insert into LRU → Return pointer
Cache hit? → Return cached pointer
    ↓
Operator uses pointer (doesn't care about cache)
```

**Key Benefits**:
1. **Bounded memory**: 64MB shared LRU (configurable) instead of unbounded per-tensor caches
2. **Non-breaking**: Operators keep calling `data()`, get cached pointers
3. **Incremental migration**: Migrate to `data_bf16()` one operator at a time for performance
4. **Automatic eviction**: LRU handles memory pressure
5. **Unified design**: Works for all tensor types (Quantized, BF16, Simple)

## Architecture Design

### Extended TensorBase Interface

```cpp
// src/tensors/TensorBase.h

class TensorBase {
public:
    // NEW: Type-aware accessors with pull-through cache
    virtual const float* data_fp32() const;
    virtual const bfloat16* data_bf16() const;
    
    // LEGACY: Compatibility wrapper (calls data_fp32)
    virtual float* data() {
        return const_cast<float*>(data_fp32());
    }
    
    // Required by subclasses for decode logic
    virtual DataType native_type() const = 0;
    virtual size_t element_count() const = 0;
    
protected:
    // Called by data_fp32/data_bf16 on cache miss
    virtual void decode_to_fp32(float* dst) const = 0;
    virtual void decode_to_bf16(bfloat16* dst) const = 0;
};
```

### Generalized Tensor Cache

```cpp
// src/operators/TensorSlabCache.h (extends QuantSlabCache)

enum class CachedDataType : uint8_t {
    FP32 = 0,
    BF16 = 1
};

struct TensorCacheKey {
    const void* tensor_ptr;      // Tensor identity
    CachedDataType data_type;    // FP32 or BF16
    size_t element_count;        // For validation
    
    bool operator<(const TensorCacheKey& other) const {
        if (tensor_ptr != other.tensor_ptr) return tensor_ptr < other.tensor_ptr;
        return data_type < other.data_type;
    }
    
    size_t hash() const {
        return std::hash<const void*>{}(tensor_ptr) ^ 
               static_cast<size_t>(data_type);
    }
};

struct CachedTensorData {
    CachedDataType type;
    size_t element_count;
    size_t bytes;
    std::vector<uint8_t> data;     // Type-erased storage
    std::chrono::steady_clock::time_point last_access;
    
    template<typename T>
    const T* typed_data() const {
        return reinterpret_cast<const T*>(data.data());
    }
    
    size_t memory_bytes() const { return data.size(); }
};

class TensorSlabCache {
public:
    static TensorSlabCache& instance() {
        static TensorSlabCache cache;
        return cache;
    }
    
    // NEW: Full tensor decode (for activations)
    template<typename T>
    const T* getOrDecodeTensor(
        const TensorBase* tensor,
        CachedDataType type,
        std::function<void(T*)> decode_fn
    );
    
    // EXISTING: Column slab decode (for weights) - unchanged
    bool getOrDecode(
        const QuantizedTensor& tensor,
        size_t col_start, size_t col_count,
        QuantSlab& out_slab,
        bool reuse_allowed = true
    );
    
    void clear() { 
        tensor_cache_.clear(); 
        // Keep existing slab cache
    }
    
    size_t total_cached_bytes() const;
    void set_capacity(size_t bytes) { capacity_bytes_ = bytes; }
    
private:
    TensorSlabCache() : capacity_bytes_(64 * 1024 * 1024) {}  // 64MB default
    
    std::map<TensorCacheKey, CachedTensorData> tensor_cache_;
    size_t capacity_bytes_;
    
    // Existing slab cache members (unchanged)
    // ...
    
    void evict_if_needed(size_t bytes_needed);
    void evict_lru_entry();
};
```

### TensorBase Default Implementation

```cpp
// src/tensors/TensorBase.cpp

const float* TensorBase::data_fp32() const {
    auto& cache = TensorSlabCache::instance();
    
    // Fast path: Native FP32 tensor (SimpleTensor)
    if (native_type() == DataType::FP32) {
        return data_native_fp32();  // Virtual hook for direct access
    }
    
    // Pull-through cache path
    return cache.getOrDecodeTensor<float>(
        this,
        CachedDataType::FP32,
        [this](float* dst) { this->decode_to_fp32(dst); }
    );
}

const bfloat16* TensorBase::data_bf16() const {
    auto& cache = TensorSlabCache::instance();
    
    // Fast path: Native BF16 tensor
    if (native_type() == DataType::BF16) {
        return data_native_bf16();  // Virtual hook for direct access
    }
    
    // Pull-through cache path
    return cache.getOrDecodeTensor<bfloat16>(
        this,
        CachedDataType::BF16,
        [this](bfloat16* dst) { this->decode_to_bf16(dst); }
    );
}
```

### BF16Tensor Implementation (Fixed)

```cpp
// src/tensors/BF16Tensor.h

class BF16Tensor : public TensorBase {
public:
    BF16Tensor(const std::vector<int>& shape)
        : shape_(shape) {
        size_t total = 1;
        for (auto dim : shape) total *= dim;
        data_.resize(total);
    }
    
    // REMOVED: fp32_cache_, cache_valid_, update_cache()
    
    // Direct BF16 access (fast path)
    bfloat16* bf16_data() { return data_.data(); }
    const bfloat16* bf16_data() const { return data_.data(); }
    
    // TensorBase interface
    DataType native_type() const override { return DataType::BF16; }
    size_t element_count() const override { return data_.size(); }
    
protected:
    // Fast path for native type
    const bfloat16* data_native_bf16() const override { 
        return data_.data(); 
    }
    
    // Decode to FP32 (cache miss path)
    void decode_to_fp32(float* dst) const override {
        #pragma omp parallel for if(data_.size() > 10000)
        for (size_t i = 0; i < data_.size(); ++i) {
            dst[i] = static_cast<float>(data_[i]);
        }
    }
    
    // BF16 is native, so this is a copy
    void decode_to_bf16(bfloat16* dst) const override {
        std::memcpy(dst, data_.data(), data_.size() * sizeof(bfloat16));
    }

private:
    std::vector<bfloat16> data_;  // Only storage!
    std::vector<int> shape_;
};
```

### SimpleTensor Implementation

```cpp
// src/tensors/SimpleTensor.h

class SimpleTensor : public TensorBase {
public:
    // Existing constructor, resize(), etc.
    
    DataType native_type() const override { return DataType::FP32; }
    size_t element_count() const override { return data_.size(); }
    
protected:
    // Fast path for FP32 (direct access, no cache)
    const float* data_native_fp32() const override {
        return data_.data();
    }
    
    // FP32 is native, so this is a copy
    void decode_to_fp32(float* dst) const override {
        std::memcpy(dst, data_.data(), data_.size() * sizeof(float));
    }
    
    // Decode to BF16 (cache miss path)
    void decode_to_bf16(bfloat16* dst) const override {
        #pragma omp parallel for if(data_.size() > 10000)
        for (size_t i = 0; i < data_.size(); ++i) {
            dst[i] = static_cast<bfloat16>(data_[i]);
        }
    }

private:
    std::vector<float> data_;
};
```

### QuantizedTensor Implementation

```cpp
// src/tensors/QuantizedTensor.h (already exists, extend)

class QuantizedTensor : public TensorBase {
public:
    // Existing: raw_data_, layout_, etc.
    
    DataType native_type() const override { 
        return DataType::QUANTIZED;  // Not FP32 or BF16
    }
    
    size_t element_count() const override {
        return layout_.total_elements;
    }
    
protected:
    // No fast path (always needs decode)
    
    void decode_to_fp32(float* dst) const override {
        // Decode all blocks to FP32
        for (size_t block_idx = 0; block_idx < num_blocks(); ++block_idx) {
            decodeBlock(block_idx, dst + block_idx * block_size);
        }
    }
    
    void decode_to_bf16(bfloat16* dst) const override {
        // Decode via FP32 intermediate (or direct if supported)
        std::vector<float> tmp(layout_.total_elements);
        decode_to_fp32(tmp.data());
        
        #pragma omp parallel for if(tmp.size() > 10000)
        for (size_t i = 0; i < tmp.size(); ++i) {
            dst[i] = static_cast<bfloat16>(tmp[i]);
        }
    }

private:
    // Existing quantized storage
};
```

## Cache Implementation Details

### Cache Lookup Logic

```cpp
template<typename T>
const T* TensorSlabCache::getOrDecodeTensor(
    const TensorBase* tensor,
    CachedDataType type,
    std::function<void(T*)> decode_fn)
{
    TensorCacheKey key{tensor, type, tensor->element_count()};
    
    // Cache hit?
    auto it = tensor_cache_.find(key);
    if (it != tensor_cache_.end()) {
        it->second.last_access = std::chrono::steady_clock::now();
        cache_hits_++;
        return it->second.template typed_data<T>();
    }
    
    // Cache miss: Allocate and decode
    cache_misses_++;
    
    size_t bytes_needed = tensor->element_count() * sizeof(T);
    evict_if_needed(bytes_needed);
    
    CachedTensorData entry;
    entry.type = type;
    entry.element_count = tensor->element_count();
    entry.bytes = bytes_needed;
    entry.data.resize(bytes_needed);
    entry.last_access = std::chrono::steady_clock::now();
    
    // Decode into cache
    decode_fn(reinterpret_cast<T*>(entry.data.data()));
    
    // Insert and return
    auto [inserted_it, success] = tensor_cache_.emplace(key, std::move(entry));
    return inserted_it->second.template typed_data<T>();
}
```

### LRU Eviction Logic

```cpp
void TensorSlabCache::evict_if_needed(size_t bytes_needed) {
    size_t current_bytes = total_cached_bytes();
    
    while (current_bytes + bytes_needed > capacity_bytes_ && !tensor_cache_.empty()) {
        evict_lru_entry();
        current_bytes = total_cached_bytes();
        evictions_++;
    }
}

void TensorSlabCache::evict_lru_entry() {
    if (tensor_cache_.empty()) return;
    
    // Find oldest accessed entry
    auto oldest = tensor_cache_.begin();
    auto oldest_time = oldest->second.last_access;
    
    for (auto it = tensor_cache_.begin(); it != tensor_cache_.end(); ++it) {
        if (it->second.last_access < oldest_time) {
            oldest = it;
            oldest_time = it->second.last_access;
        }
    }
    
    LOG_DEBUG("Evicting tensor cache entry: " 
              << oldest->second.memory_bytes() << " bytes");
    tensor_cache_.erase(oldest);
}
```

## Migration Path

### Phase 1: Implement Cache Infrastructure (1-2 days)

**Tasks**:
1. Add `TensorCacheKey` and `CachedTensorData` to `TensorSlabCache.h`
2. Implement `getOrDecodeTensor()` template method
3. Add eviction logic with LRU policy
4. Add cache statistics (hits, misses, evictions)

**Testing**:
- Unit tests: Cache hit/miss behavior
- Eviction tests: Verify LRU works correctly
- Memory tests: Cache stays under capacity

### Phase 2: Update TensorBase Interface (1 day)

**Tasks**:
1. Add `data_fp32()` and `data_bf16()` virtual methods
2. Add protected `decode_to_fp32()` and `decode_to_bf16()` hooks
3. Add `native_type()` and `element_count()` methods
4. Implement default `data_fp32()`/`data_bf16()` with cache lookup
5. Update `data()` to call `data_fp32()` for compatibility

**Testing**:
- Interface tests: Verify methods callable
- No functional changes yet (subclasses not updated)

### Phase 3: Fix BF16Tensor (1 day) **← CRITICAL**

**Tasks**:
1. **Remove** `fp32_cache_` and `cache_valid_` members
2. **Remove** `update_cache()` method
3. Implement `decode_to_fp32()` (BF16→FP32 conversion)
4. Implement `decode_to_bf16()` (memcpy, already BF16)
5. Implement `data_native_bf16()` for fast path
6. Keep `bf16_data()` for direct access

**Testing**:
- Memory tests: **CRITICAL** - should be ~50% of FP32, not 150%
- Parity tests: Numerics unchanged
- Cache tests: Verify cache used for FP32 conversion

**Expected Result**: BF16 activations now use ~50% memory!

### Phase 4: Update SimpleTensor (1 day)

**Tasks**:
1. Implement `decode_to_fp32()` (memcpy, already FP32)
2. Implement `decode_to_bf16()` (FP32→BF16 conversion)
3. Implement `data_native_fp32()` for fast path

**Testing**:
- Fast path tests: FP32 access should be direct (no cache)
- Conversion tests: FP32→BF16 uses cache

### Phase 5: Update QuantizedTensor (1-2 days)

**Tasks**:
1. Implement `decode_to_fp32()` (full tensor dequant)
2. Implement `decode_to_bf16()` (dequant to BF16)
3. Note: Column slab path remains unchanged (separate cache)

**Testing**:
- Full tensor decode tests
- Verify column slab path still works
- Memory tests: Full decode only when needed

### Phase 6: Validation (2-3 days)

**Testing**:
1. **Memory profiling**: 
   - Total cache usage should be ~64MB
   - No per-tensor allocations
   - BF16 mode: ~50% memory vs FP32 (not 150%!)

2. **Parity tests**:
   - All 1557 tests should pass
   - Relative L2 error <1e-5

3. **Performance tests**:
   - Cache hit rate >80% for typical workloads
   - Throughput within 95% of FP32 (cache overhead)

4. **Stress tests**:
   - Cache eviction under memory pressure
   - Large batch sizes (cache thrashing)

### Phase 7: Operator Migration (Ongoing, optional)

**Gradual performance optimization**:

Operators can stay on `data()` (works, uses cache) or migrate to `data_bf16()` for better performance:

```cpp
// BEFORE (uses cache):
float* input_data = input_tensor->data();  // Calls data_fp32()

// AFTER (direct access, faster):
if (auto bf16 = std::dynamic_pointer_cast<BF16Tensor>(input_tensor)) {
    bfloat16* bf16_data = bf16->bf16_data();  // Direct, no cache
    // Use BF16-aware kernel
} else {
    float* fp32_data = input_tensor->data();  // Still works
}
```

**Priority order** (most benefit first):
1. MPILinearOperator (Q/K/V/O projections) - high frequency
2. MPIAttentionOperator (scores, context) - high frequency
3. MPISwiGLUOperator (FFN) - medium frequency
4. Other operators - low frequency

## Performance Characteristics

### Memory Usage

**Before (broken)**:
- BF16Tensor: 2 bytes (BF16) + 4 bytes (FP32 cache) = 6 bytes/element
- Total: 150% of FP32 baseline
- Example: 5171 MB (FP32) → 6498 MB (BF16+cache) = +26%

**After (pull-through cache)**:
- BF16Tensor: 2 bytes (BF16 only)
- Shared cache: 64 MB total (all tensors)
- Total: ~50% of FP32 + 64 MB
- Example: 5171 MB (FP32) → 2585 MB (BF16) + 64 MB (cache) = -48%

### Cache Hit Rates (Expected)

**Typical pipeline execution**:
1. Embedding output → used once in layer 0 → evicted
2. Layer N attention output → used in FFN norm, gate, up → 3 hits
3. Layer N FFN output → used in next layer norm → 1 hit
4. Most tensors accessed 2-4 times before eviction

**Estimated hit rate**: 70-85% (depends on cache size and model)

### Decode Overhead

**FP32→BF16 conversion cost**:
- Sequential: ~5-10 ns/element (simple cast)
- Parallel (OMP): ~1-2 ns/element (10K+ elements)
- Negligible vs GEMM cost (100-1000x more expensive)

**BF16→FP32 conversion cost**:
- Same as above
- Only on cache miss

**Cache lookup cost**:
- std::map lookup: ~50-100 ns
- Amortized over tensor size: <0.1% overhead

### Tuning Parameters

```cpp
// Environment variables for tuning
LLAMINAR_TENSOR_CACHE_SIZE_MB=64      // Default cache size
LLAMINAR_TENSOR_CACHE_STATS=1         // Log hit/miss rates
LLAMINAR_TENSOR_CACHE_TRACE=1         // Verbose logging
```

## Edge Cases and Concerns

### 1. Tensor Lifetime vs Cache Lifetime

**Problem**: What if tensor destroyed while cached pointer in use?

**Solution**: Cache stores **copy** of decoded data, not reference to tensor buffer
- Decoded data lives in cache independently
- Tensor destruction doesn't invalidate cached pointers
- Cache key uses tensor pointer as identity, but data is copied

**Validation**: When tensor destroyed, future cache misses won't match its key

### 2. Cache Thrashing

**Problem**: Working set > cache size → frequent evictions

**Scenarios**:
- Very large batch sizes
- Deep model with many layers
- Small cache size

**Mitigation**:
1. Tune cache size via `LLAMINAR_TENSOR_CACHE_SIZE_MB`
2. Monitor eviction rate (if >20%, increase cache)
3. Consider per-layer cache pinning (advanced)
4. Migrate hot operators to `data_bf16()` (bypass cache)

### 3. Thread Safety

**Problem**: Multiple threads accessing cache simultaneously

**Solution**: 
- Add mutex for cache operations (simple)
- Or thread-local caches (complex, higher memory)
- Or lock-free hash map (complex, best performance)

**Initial approach**: Single mutex (simple, sufficient for single-threaded per rank)

```cpp
class TensorSlabCache {
    mutable std::mutex cache_mutex_;
    
    template<typename T>
    const T* getOrDecodeTensor(...) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        // ... existing logic
    }
};
```

### 4. MPI and Multi-Rank

**Problem**: Each rank has separate cache instance

**Solution**: This is **correct behavior**!
- Each rank has different activations (partitioned)
- Weights use column slab cache (already rank-aware)
- No cross-rank cache synchronization needed

### 5. Invalidation on Tensor Mutation

**Problem**: Tensor data changes, cache becomes stale

**Solution**: Invalidation API

```cpp
class TensorBase {
    void invalidate_cache() {
        TensorSlabCache::instance().invalidate(this);
    }
};

// In TensorSlabCache:
void invalidate(const TensorBase* tensor) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Remove all entries for this tensor
    auto it = tensor_cache_.begin();
    while (it != tensor_cache_.end()) {
        if (it->first.tensor_ptr == tensor) {
            it = tensor_cache_.erase(it);
        } else {
            ++it;
        }
    }
}
```

**Usage**: Call `invalidate_cache()` after in-place mutations (rare in inference)

## Success Metrics

### Memory (Primary Goal)

✅ **Target**: BF16 mode uses ~50% memory vs FP32 (not 150%)

**Measurement**:
```bash
# FP32 baseline
./run_llaminar.sh -m model.gguf -p "test" -n 100
# Note peak RSS: ~5171 MB

# BF16 with pull-through cache
LLAMINAR_QUANT_OUTPUT_BF16=1 ./run_llaminar.sh -m model.gguf -p "test" -n 100
# Expected peak RSS: ~2585 MB (50%) + 64 MB (cache) = ~2650 MB
```

### Correctness (Critical)

✅ **Target**: All parity tests pass (387/387 stages)

**Validation**:
```bash
ctest --test-dir build -R "ParityFrameworkTest" --output-on-failure
# All tests pass, relative L2 <1e-5
```

### Performance (Acceptable Overhead)

✅ **Target**: Throughput within 95% of FP32 baseline

**Measurement**:
- Cache lookup overhead: <1% (std::map lookup amortized)
- Decode overhead on miss: ~5-10% (first access only)
- Typical hit rate >80% → effective overhead <2%

### Cache Efficiency

✅ **Target**: Hit rate >70% for typical workloads

**Monitoring**:
```bash
LLAMINAR_TENSOR_CACHE_STATS=1 ./run_llaminar.sh ...
# Output:
# [TensorCache] Hits: 1240, Misses: 358, Evictions: 12
# [TensorCache] Hit rate: 77.6%
```

## Comparison to Alternatives

### Alternative 1: Per-Tensor Caches (Current - Broken)

**Memory**: Unbounded, 150% overhead  
**Migration**: None needed (already exists)  
**Complexity**: Low  
**Verdict**: ❌ Unacceptable memory usage

### Alternative 2: Break Everything (Original Plan)

**Memory**: Optimal (50% for BF16)  
**Migration**: High risk (all operators at once)  
**Complexity**: Medium  
**Verdict**: ✅ Correct but risky

### Alternative 3: Pull-Through Cache (This Proposal)

**Memory**: Near-optimal (50% + 64 MB)  
**Migration**: Incremental, low risk  
**Complexity**: Medium  
**Verdict**: ✅✅ **Best of both worlds**

## Implementation Priority

**Week 1** (Critical Path):
- Day 1-2: Cache infrastructure (TensorCacheKey, getOrDecodeTensor, eviction)
- Day 3: TensorBase interface (data_fp32, data_bf16)
- Day 4-5: **Fix BF16Tensor** (remove fp32_cache_)

**Week 2** (Completion):
- Day 6: SimpleTensor implementation
- Day 7-8: QuantizedTensor implementation
- Day 9-10: Testing and validation

**Week 3+** (Optimization):
- Migrate high-frequency operators to `data_bf16()`
- Performance profiling and tuning
- Cache size optimization

## Conclusion

The pull-through cache design elegantly solves the BF16 memory leak while enabling **graceful migration**:

1. ✅ **Non-breaking**: Operators keep calling `data()`, work unchanged
2. ✅ **Bounded memory**: 64 MB shared cache vs unbounded per-tensor caches
3. ✅ **Incremental**: Migrate operators one at a time for better performance
4. ✅ **Unified**: All tensor types (Quantized, BF16, Simple) use same pattern
5. ✅ **Production-ready**: Low risk, easy to validate, simple to tune

**This is the right approach.** Let's implement it.
