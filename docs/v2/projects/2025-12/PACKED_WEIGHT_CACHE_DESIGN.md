# Packed Weight Cache Architecture

## Problem Statement

The current `KernelFactory` caches GEMM kernels by `TensorBase*` pointer, and kernels themselves store packed weight data internally. This creates several issues:

1. **Stateful Kernels**: `QuantisedGemmKernel` holds packed weights in `packed_weights_` member
2. **Instance-Based Caching**: Cache key is tensor instance pointer, not (shape, format, device)
3. **Duplicate Packing**: Same weights in different contexts may be packed multiple times
4. **Tight Coupling**: Kernels are bound to specific tensor instances

## Proposed Architecture

### Key Insight

There are **two separate concerns** that should be independently cached:

1. **JIT Kernel Code** (already stateless)
   - `QuantisedGemmJit_M1`, `QuantisedGemmJit_M2` are singleton JIT-compiled functions
   - Shape-independent (M is a runtime parameter)
   - Keyed by: (TensorFormat, DeviceType) - essentially just one per format/device combo

2. **Packed Weights** (currently in kernel, should be separate)
   - VNNI-optimal layout of weight data
   - Keyed by: (TensorBase*, row_start, row_end) for sharding support
   - Expensive to create, should be cached and reused

### New Components

#### 1. PackedWeightCache

A singleton cache that stores packed weights keyed by tensor + range:

```cpp
class PackedWeightCache {
public:
    struct CacheKey {
        const TensorBase* tensor;
        int row_start;
        int row_end;  // -1 means full tensor
        
        bool operator==(const CacheKey& other) const;
        size_t hash() const;
    };
    
    static PackedWeightCache& instance();
    
    // Get or create packed weights for a tensor range
    const QuantisedPackedWeights* getOrCreate(
        const TensorBase* tensor,
        int row_start = 0,
        int row_end = -1);
    
    // Clear cache for a specific tensor (call when tensor is destroyed)
    void clearFor(const TensorBase* tensor);
    
    // Clear entire cache
    void clear();
    
    // Statistics
    size_t cacheSize() const;
    size_t totalPackedBytes() const;

private:
    std::unordered_map<CacheKey, std::unique_ptr<QuantisedPackedWeights>> cache_;
    std::mutex mutex_;
};
```

#### 2. Stateless QuantisedGemmKernel

The kernel becomes a thin wrapper around JIT code:

```cpp
class QuantisedGemmKernel : public ITensorGemm {
public:
    // Constructor takes device type and format - NOT tensor
    QuantisedGemmKernel(DeviceType device, TensorType format);
    
    // Execute method takes packed weights as parameter
    bool multiply_with_packed_weights(
        const QuantisedPackedWeights& packed,
        const void* q8_1_activations,
        float* C,
        int m, int n, int k,
        const GemmFusedOps& fused_ops);
    
    // Convenience: Get packed weights from cache and execute
    bool multiply(
        const TensorBase* weight_tensor,
        const void* q8_1_activations, 
        float* C,
        int m,
        const GemmFusedOps& fused_ops);

private:
    // JIT kernels are static singletons
    static QuantisedGemmJit_M1& jit_m1();
    static QuantisedGemmJit_M2& jit_m2();
    
    // No weight storage - kernels are truly stateless
};
```

#### 3. Updated KernelFactory

The factory becomes simpler - just returns appropriate kernel type:

```cpp
class KernelFactory {
public:
    // Get stateless kernel for format/device combo
    // These can be shared across all tensors of same format/device
    static ITensorGemm* getGemmKernel(TensorType format, DeviceType device);
    
    // Packed weight cache access
    static const QuantisedPackedWeights* getPackedWeights(
        const TensorBase* tensor,
        int row_start = 0, 
        int row_end = -1);
    
    static void clearPackedWeightsFor(const TensorBase* tensor);
    static void clearAllPackedWeights();
};
```

### Usage Pattern

**Before** (current problematic pattern):
```cpp
// Each tensor gets its own kernel with embedded packed weights
auto* kernel = KernelFactory::getOrCreateGemm(weight_tensor);
kernel->multiply(activations, output, m, n, k, ...);
```

**After** (proposed pattern):
```cpp
// Get stateless kernel (shared by all Q4_0 tensors on CPU)
auto* kernel = KernelFactory::getGemmKernel(TensorType::Q4_0, DeviceType::CPU);

// Get packed weights from cache (created once per tensor)
const auto* packed = KernelFactory::getPackedWeights(weight_tensor);

// Execute with external packed weights
kernel->multiply_with_packed_weights(*packed, activations, output, m, n, k, ...);

// Or use convenience method that does both:
kernel->multiply(weight_tensor, activations, output, m, ...);
```

### Benefits

1. **True Kernel Reuse**: One kernel instance per (format, device) combination
2. **Packed Weight Sharing**: Same packed data reused even if accessed through different code paths
3. **Clear Ownership**: Cache owns packed weights, lifetime well-defined
4. **Sharding Support**: Cache key includes row range for tensor-parallel packed weights
5. **Simpler Debugging**: No hidden state in kernels - all data flows through parameters

### Migration Path

1. **Phase 1**: Add `PackedWeightCache` alongside existing infrastructure
2. **Phase 2**: Update `QuantisedGemmKernel` to optionally use external packed weights
3. **Phase 3**: Update callers to use new pattern
4. **Phase 4**: Remove `packed_weights_` member from kernel, simplify `KernelFactory`

### Tensor Lifecycle

When a tensor is destroyed:
```cpp
TensorBase::~TensorBase() {
    // Clear any cached packed weights for this tensor
    PackedWeightCache::instance().clearFor(this);
}
```

This is already done for `KernelFactory` cache, just needs to be updated to use the new cache.
