# GPU Tensor Allocation and Buffer Management Analysis

## Overview

This document analyzes the current tensor allocation and buffer management system in Llaminar V2, focusing on what infrastructure exists for GPU tensor allocation and what changes are needed to support GPU buffer allocation in the graph execution system.

---

## 1. Current CPU Tensor Allocation Flow

### TensorFactory (Primary Entry Point)

**File**: [src/v2/tensors/TensorFactory.h](../../../src/v2/tensors/TensorFactory.h) and [.cpp](../../../src/v2/tensors/TensorFactory.cpp)

The `TensorFactory` is the central factory for creating tensors with NUMA-aware allocation:

```cpp
// Key APIs:
class TensorFactory {
public:
    explicit TensorFactory(const MPIContext &mpi_ctx);
    
    // Unified creation (DeviceId-based) - THE PRIMARY API
    std::unique_ptr<ITensor> create(TensorType type,
                                    const std::vector<size_t> &shape,
                                    DeviceId device = DeviceId::cpu());
    
    // Activation buffer creation (convenience)
    std::unique_ptr<ITensor> createActivation(const std::vector<size_t> &shape,
                                              ActivationPrecision precision,
                                              DeviceId device = DeviceId::cpu());
    
    // Type-specific CPU creation methods
    std::unique_ptr<FP32Tensor> createFP32(const std::vector<size_t> &shape, DeviceId device);
    std::unique_ptr<Q8_1Tensor> createQ8_1(const std::vector<size_t> &shape, DeviceId device);
    std::unique_ptr<Q16_1Tensor> createQ16_1(const std::vector<size_t> &shape, DeviceId device);
    // ... etc
};
```

### CPU Allocation Flow

1. **MPI Context** → `TensorFactory` constructor extracts MPI rank
2. **NUMA Node Mapping** → `getNumaNodeForRank(rank)` determines NUMA node
3. **NUMA Binding** → `bindToNumaNode()` sets memory policy before allocation
4. **Tensor Creation** → Type-specific `create*()` methods allocate tensors

```cpp
// Example: createFP32 with NUMA awareness
std::unique_ptr<FP32Tensor> TensorFactory::createFP32(const std::vector<size_t> &shape, DeviceId device) {
    if (numa_node_ >= 0) {
        bindToNumaNode();  // Bind to local NUMA node before allocation
    }
    return std::make_unique<FP32Tensor>(shape, device.toLegacyIndex());
}
```

---

## 2. CUDA Tensor Types

### CUDATensorBase (GPU Tensor Base Class)

**File**: [src/v2/tensors/cuda/CUDATensorBase.h](../../../src/v2/tensors/cuda/CUDATensorBase.h)

Key characteristics of GPU tensors:
- **GPU-Primary**: `data_ptr()` returns DEVICE pointer (not host)
- **`is_on_gpu()` returns `true`**, `is_on_cpu()` returns `false`
- **No lazy transfer**: Explicit `copyToHost()`/`copyFromHost()` methods
- **Device index tracking**: Each tensor knows its CUDA ordinal

```cpp
class CUDATensorBase : public ITensor {
public:
    CUDATensorBase(std::vector<size_t> shape, TensorType dtype, 
                   int device_idx, cudaStream_t stream = nullptr);
    
    // Device awareness - GPU tensors are GPU-primary
    DeviceId home_device() const override { return DeviceId::cuda(device_idx_); }
    bool is_on_cpu() const override { return false; }
    bool is_on_gpu() const override { return true; }
    
    // GPU-Specific API
    void* device_ptr();
    int device_index() const;
    cudaStream_t stream() const;
    
    // Host Copy Utilities
    void copyToHost(void* host_dst, size_t bytes) const;
    void copyFromHost(const void* host_src, size_t bytes);
    
protected:
    bool allocateDevice(size_t bytes);  // cudaMalloc wrapper
    void freeDevice();                   // cudaFree wrapper
    
    void* device_ptr_ = nullptr;
    int device_idx_ = 0;
    cudaStream_t stream_ = nullptr;
    bool owns_memory_ = true;
};
```

### CUDATypedTensor (Typed GPU Tensors)

**File**: [src/v2/tensors/cuda/CUDATypedTensor.h](../../../src/v2/tensors/cuda/CUDATypedTensor.h)

Template class providing type-safe GPU tensor access:

```cpp
template<typename T, TensorType DType>
class CUDATypedTensor : public CUDATensorBase {
public:
    // Allocating constructor
    CUDATypedTensor(std::vector<size_t> shape, int device_idx, cudaStream_t stream = nullptr);
    
    // Non-owning view over existing device memory
    CUDATypedTensor(std::vector<size_t> shape, T* device_ptr, int device_idx, cudaStream_t stream = nullptr);
    
    // Type-safe accessors
    T* typed_device_ptr();
    const T* typed_device_ptr() const;
    T* typed_data();  // Alias for typed_device_ptr
};
```

### Available CUDA Tensor Type Aliases

```cpp
// Floating point
using CUDAFp32Tensor    = CUDATypedTensor<float, TensorType::FP32>;
using CUDAFP16Tensor    = CUDATypedTensor<__half, TensorType::FP16>;
using CUDABF16Tensor    = CUDATypedTensor<__nv_bfloat16, TensorType::BF16>;

// Integer
using CUDAINT8Tensor    = CUDATypedTensor<int8_t, TensorType::INT8>;
using CUDAINT32Tensor   = CUDATypedTensor<int32_t, TensorType::INT32>;

// Quantized activations (block-structured)
using CUDAQ8_1Tensor    = CUDATypedTensor<Q8_1Block, TensorType::Q8_1>;
using CUDAQ16_1Tensor   = CUDATypedTensor<Q16_1Block, TensorType::Q16_1>;      // 32-element blocks
using CUDAQ16_1_64Tensor = CUDATypedTensor<Q16_1Block_64, TensorType::Q16_1>;  // 64-element blocks
using CUDAQ16_1_128Tensor = CUDATypedTensor<Q16_1Block_128, TensorType::Q16_1>; // 128-element blocks
```

---

## 3. GPU Tensor Creation in TensorFactory

The `TensorFactory::create()` method already supports CUDA tensor creation:

```cpp
std::unique_ptr<ITensor> TensorFactory::create(TensorType type,
                                               const std::vector<size_t> &shape,
                                               DeviceId device) {
#ifdef HAVE_CUDA
    if (device.is_cuda()) {
        switch (type) {
        case TensorType::FP32:
            return std::make_unique<CUDAFp32Tensor>(shape, device.ordinal);
        case TensorType::FP16:
            return std::make_unique<CUDAFP16Tensor>(shape, device.ordinal);
        case TensorType::BF16:
            return std::make_unique<CUDABF16Tensor>(shape, device.ordinal);
        case TensorType::INT8:
            return std::make_unique<CUDAINT8Tensor>(shape, device.ordinal);
        case TensorType::INT32:
            return std::make_unique<CUDAINT32Tensor>(shape, device.ordinal);
        case TensorType::Q8_1:
            return std::make_unique<CUDAQ8_1Tensor>(shape, device.ordinal);
        case TensorType::Q16_1:
            return std::make_unique<CUDAQ16_1Tensor>(shape, device.ordinal);
        default:
            throw std::runtime_error("CUDA tensor type not yet supported");
        }
    }
#endif
    // Fall through to CPU tensor creation...
}
```

### Code Examples for Creating GPU Tensors

```cpp
// Method 1: Using TensorFactory::create() with DeviceId
TensorFactory factory(mpi_ctx);

// Create FP32 tensor on GPU 0
auto gpu_fp32 = factory.create(TensorType::FP32, {1024, 896}, DeviceId::cuda(0));

// Create Q8_1 tensor on GPU 1
auto gpu_q8 = factory.create(TensorType::Q8_1, {batch_size, seq_len, hidden_dim}, DeviceId::cuda(1));

// Create BF16 tensor on GPU 0
auto gpu_bf16 = factory.create(TensorType::BF16, {32, 64, 128}, DeviceId::cuda(0));


// Method 2: Using createActivation() with precision enum
auto gpu_activation = factory.createActivation(
    {seq_len, d_model}, 
    ActivationPrecision::FP32, 
    DeviceId::cuda(0)
);

// With head_dim for Q16 block size selection
auto gpu_q16 = factory.createActivation(
    {seq_len, qkv_dim}, 
    ActivationPrecision::Q16_1, 
    64,  // head_dim -> selects Q16_1_64 blocks
    DeviceId::cuda(0)
);


// Method 3: Direct CUDA tensor construction (low-level)
#include "tensors/cuda/CUDATypedTensor.h"

// Allocating tensor
CUDAFp32Tensor gpu_tensor({batch, seq_len, hidden}, 0);  // GPU ordinal 0

// Non-owning view over existing device memory
float* existing_d_ptr = ...;  // From cuBLAS, cuDNN, etc.
CUDAFp32Tensor view({batch, seq_len, hidden}, existing_d_ptr, 0);
// view doesn't own memory, won't free on destruction
```

---

## 4. Device Management

### DeviceId (Type-Safe Device Identification)

**File**: [src/v2/backends/DeviceId.h](../../../src/v2/backends/DeviceId.h)

Replaces legacy integer device indices with explicit type+ordinal:

```cpp
struct DeviceId {
    DeviceType type;
    int ordinal;  // GPU ordinal (0-based), 0 for CPU
    
    // Factory methods
    static DeviceId cpu();
    static DeviceId cuda(int gpu_ordinal);
    static DeviceId rocm(int gpu_ordinal);
    
    // Predicates
    bool is_cpu() const;
    bool is_cuda() const;
    bool is_gpu() const;
    
    // Conversion from legacy index (for gradual migration)
    static DeviceId fromLegacyIndex(int idx);  // -1 or 0 = CPU, 1+ = GPU
    int toLegacyIndex() const;
};
```

### DeviceManager (Device Enumeration)

**File**: [src/v2/backends/ComputeBackend.h](../../../src/v2/backends/ComputeBackend.h)

Singleton that enumerates all available compute devices:

```cpp
class DeviceManager {
public:
    static DeviceManager& instance();
    
    // Initialize with NUMA filtering
    void initialize(int local_numa_node = -1);
    
    // Device enumeration
    const std::vector<ComputeDevice>& devices() const;
    int find_device(ComputeBackendType type, int device_id = 0) const;
    int cpuDeviceIndex() const { return 0; }  // CPU is always index 0
    
    // GPU availability
    bool has_gpu() const;
    std::vector<size_t> get_devices_by_type(ComputeBackendType type) const;
    
    // Context creation
    std::shared_ptr<ComputeContext> create_context(size_t device_index);
};

// Usage:
auto& dm = DeviceManager::instance();
dm.initialize(-1);  // -1 = enumerate all devices

if (dm.has_gpu()) {
    int cuda_idx = dm.find_device(ComputeBackendType::GPU_CUDA, 0);
    // cuda_idx can be used with DeviceId::cuda(cuda_idx)
}
```

---

## 5. Graph Buffer Management System

### DeviceGraphBufferManager

**File**: [src/v2/execution/DeviceGraphBufferManager.h](../../../src/v2/execution/DeviceGraphBufferManager.h)

Centralized buffer management for DeviceGraphExecutor:

```cpp
class DeviceGraphBufferManager {
public:
    explicit DeviceGraphBufferManager(TensorFactory* factory, const MPIContext* mpi_ctx = nullptr);
    
    // Allocation
    bool allocateForGraph(ComputeGraph& graph);
    bool allocateWithAliasing(ComputeGraph& graph);  // With SCRATCH buffer aliasing
    bool allocateBuffer(const std::string& node_name, const BufferDescriptor& desc);
    
    // Retrieval
    TensorBase* getBuffer(const std::string& node_name, const std::string& buffer_name);
    bool hasBuffer(const std::string& node_name, const std::string& buffer_name) const;
    
    // Binding
    bool bindBuffer(const std::string& node_name, const std::string& buffer_name, 
                    TensorBase** target_ptr);
};
```

### BufferDescriptor

**File**: [src/v2/execution/BufferRole.h](../../../src/v2/execution/BufferRole.h)

Describes a single buffer's requirements:

```cpp
struct BufferDescriptor {
    std::string name;
    BufferRole role;              // INPUT, OUTPUT, INOUT, SCRATCH, WEIGHT
    std::vector<size_t> shape;
    BufferTensorType tensor_type; // FP32, BF16, Q8_1, etc.
    bool required = true;
    size_t alignment = 64;
    int device_idx = -1;          // -1 = CPU, >=0 = GPU ordinal
    
    // Convenience factory methods
    static BufferDescriptor output(const std::string& name, 
                                   std::vector<size_t> shape,
                                   BufferTensorType type = BufferTensorType::FP32);
    static BufferDescriptor scratch(const std::string& name,
                                    std::vector<size_t> shape,
                                    BufferTensorType type = BufferTensorType::FP32);
    // ... etc
};
```

### Current Buffer Allocation Flow (CPU)

```cpp
// DeviceGraphBufferManager::createTensorFromDescriptor() (current implementation)
std::unique_ptr<TensorBase> DeviceGraphBufferManager::createTensorFromDescriptor(const BufferDescriptor& desc) {
    int device_idx = desc.device_idx;  // Currently always -1 (CPU)
    
    switch (desc.tensor_type) {
    case BufferTensorType::FP32:
        return factory_->createFP32(shape, DeviceId::fromLegacyIndex(device_idx));
    case BufferTensorType::BF16:
        return factory_->createBF16(shape, DeviceId::fromLegacyIndex(device_idx));
    case BufferTensorType::Q8_1:
        return factory_->createQ8_1(shape, DeviceId::fromLegacyIndex(device_idx));
    // ... etc
    }
}
```

**Problem**: The `TensorBase` alias is `CPUTensorBase`, not `ITensor`. GPU tensors inherit from `CUDATensorBase`, not `CPUTensorBase`. This is the main blocker for GPU buffer allocation.

---

## 6. What's Needed for GPU Buffer Allocation

### Issue 1: Return Type Mismatch

**Current State**:
- `DeviceGraphBufferManager` stores `std::unique_ptr<TensorBase>` (alias for `CPUTensorBase`)
- `CUDATensorBase` does NOT inherit from `CPUTensorBase`
- Both inherit from `ITensor` (common interface)

**Solution**: Change buffer storage to `std::unique_ptr<ITensor>`:

```cpp
// DeviceGraphBufferManager.h
class DeviceGraphBufferManager {
private:
    // OLD:
    // std::unordered_map<BufferKey, std::unique_ptr<TensorBase>, BufferKeyHash> buffers_;
    
    // NEW:
    std::unordered_map<BufferKey, std::unique_ptr<ITensor>, BufferKeyHash> buffers_;

public:
    // OLD:
    // TensorBase* getBuffer(...);
    
    // NEW:
    ITensor* getBuffer(...);
};
```

### Issue 2: BufferDescriptor Device Specification

The `device_idx` field uses legacy integer convention:
- `-1` = CPU (interpreted as DeviceId::cpu())
- `>=0` = GPU ordinal (but currently always -1)

**Solution**: Either keep using integers with `DeviceId::fromLegacyIndex()` or add a `DeviceId device` field:

```cpp
struct BufferDescriptor {
    // Option A: Keep legacy int (simpler, already works)
    int device_idx = -1;  // -1 = CPU, >=0 = CUDA ordinal
    
    // Option B: Add DeviceId (cleaner, explicit)
    DeviceId device = DeviceId::cpu();
    
    // Factory methods updated:
    static BufferDescriptor gpuOutput(const std::string& name,
                                      std::vector<size_t> shape,
                                      int gpu_ordinal,
                                      BufferTensorType type = BufferTensorType::FP32) {
        BufferDescriptor desc;
        desc.device_idx = gpu_ordinal;  // or desc.device = DeviceId::cuda(gpu_ordinal)
        // ...
        return desc;
    }
};
```

### Issue 3: DeviceGraphBufferManager::createTensorFromDescriptor

Needs to use `TensorFactory::create()` (which already supports CUDA) instead of type-specific methods:

```cpp
std::unique_ptr<ITensor> DeviceGraphBufferManager::createTensorFromDescriptor(const BufferDescriptor& desc) {
    DeviceId device = (desc.device_idx >= 0) 
        ? DeviceId::cuda(desc.device_idx) 
        : DeviceId::cpu();
    
    TensorType type = bufferTensorTypeToTensorType(desc.tensor_type);
    
    // Use unified create() - already handles CPU and CUDA
    return factory_->create(type, desc.shape, device);
}
```

### Issue 4: GraphSchema BufferSpec Device Placement

**File**: [src/v2/execution/GraphSchema.h](../../../src/v2/execution/GraphSchema.h)

The declarative schema `BufferSpec` needs device placement:

```cpp
struct BufferSpec {
    std::string name;
    std::vector<std::string> shape;  // Shape formulas like "seq_len", "d_model"
    std::string dtype = "fp32";
    BufferSemantic semantic;
    std::string alias_group;
    int alias_priority = 0;
    std::string description;
    
    // MISSING: Device placement
    // Add:
    std::string device = "cpu";  // "cpu", "cuda:0", "cuda:1", etc.
    // Or:
    int device_idx = -1;
};
```

The `ResolvedBufferSpec` in [GraphResolver.h](../../../src/v2/execution/GraphResolver.h) already has `device_idx`:

```cpp
struct ResolvedBufferSpec {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;
    BufferSemantic semantic;
    int device_idx;  // ✓ Already present
    // ...
};
```

---

## 7. Concrete Implementation Plan

### Phase 1: Update DeviceGraphBufferManager to Support ITensor

```cpp
// DeviceGraphBufferManager.h changes:

// 1. Change storage type
std::unordered_map<BufferKey, std::unique_ptr<ITensor>, BufferKeyHash> buffers_;

// 2. Change return types
ITensor* getBuffer(const std::string& node_name, const std::string& buffer_name);
ITensor* getBuffer(const BufferKey& key);

// 3. Update createTensorFromDescriptor signature
std::unique_ptr<ITensor> createTensorFromDescriptor(const BufferDescriptor& desc);
```

### Phase 2: Implement GPU Buffer Creation

```cpp
// DeviceGraphBufferManager.cpp changes:

std::unique_ptr<ITensor> DeviceGraphBufferManager::createTensorFromDescriptor(const BufferDescriptor& desc) {
    DeviceId device = DeviceId::fromLegacyIndex(desc.device_idx);
    
    // Map BufferTensorType to TensorType
    TensorType type;
    switch (desc.tensor_type) {
    case BufferTensorType::FP32:  type = TensorType::FP32;  break;
    case BufferTensorType::BF16:  type = TensorType::BF16;  break;
    case BufferTensorType::FP16:  type = TensorType::FP16;  break;
    case BufferTensorType::Q8_1:  type = TensorType::Q8_1;  break;
    case BufferTensorType::INT32: type = TensorType::INT32; break;
    default:                      type = TensorType::FP32;  break;
    }
    
    // Use unified create() - works for both CPU and CUDA
    return factory_->create(type, desc.shape, device);
}
```

### Phase 3: Add Device Placement to Schema

```cpp
// GraphSchema.h - update BufferSpec
struct BufferSpec {
    // ... existing fields ...
    
    /// Target device: "cpu", "cuda:0", etc.
    std::string device = "cpu";
};

// GraphResolver.cpp - parse device string
int parseDeviceSpec(const std::string& device_str) {
    if (device_str == "cpu" || device_str.empty()) return -1;
    if (device_str.substr(0, 5) == "cuda:") {
        return std::stoi(device_str.substr(5));
    }
    return -1;  // Default to CPU
}
```

### Phase 4: Stage Integration

Update stages to specify device placement in `getBufferRequirements()`:

```cpp
StageBufferRequirements CUDAAttentionStage::getBufferRequirements() const {
    StageBufferRequirements reqs;
    
    // GPU output buffer
    BufferDescriptor output_desc = BufferDescriptor::output(
        "attention_output", 
        {seq_len_, d_model_}, 
        BufferTensorType::FP32
    );
    output_desc.device_idx = gpu_ordinal_;  // Set GPU device
    
    reqs.buffers.push_back(output_desc);
    return reqs;
}
```

---

## 8. Summary

### What Already Exists ✅

| Component | Status | File |
|-----------|--------|------|
| CUDA tensor types | ✅ Working | `tensors/cuda/CUDATypedTensor.h` |
| CUDATensorBase | ✅ Working | `tensors/cuda/CUDATensorBase.h` |
| TensorFactory CUDA support | ✅ Working | `tensors/TensorFactory.cpp` |
| DeviceId type-safe IDs | ✅ Working | `backends/DeviceId.h` |
| DeviceManager enumeration | ✅ Working | `backends/ComputeBackend.h` |
| CUDABackend allocation | ✅ Working | `backends/cuda/CUDABackend.cu` |
| BufferDescriptor.device_idx | ✅ Present | `execution/BufferRole.h` |

### What Needs Changes 🔧

| Component | Change Needed | Priority |
|-----------|---------------|----------|
| DeviceGraphBufferManager storage | `TensorBase` → `ITensor` | High |
| DeviceGraphBufferManager::getBuffer | Return `ITensor*` | High |
| createTensorFromDescriptor | Use `factory_->create()` | High |
| BufferSpec (schema) | Add `device` field | Medium |
| Stage buffer requirements | Set `device_idx` for GPU | Medium |
| Aliasing groups | GPU buffer aliasing | Low |

### Code Example: Full GPU Buffer Allocation

```cpp
// After implementing changes:

// 1. Create factory and manager
TensorFactory factory(mpi_ctx);
DeviceGraphBufferManager manager(&factory, &mpi_ctx);

// 2. Allocate GPU buffer manually
BufferDescriptor gpu_output;
gpu_output.name = "attention_context";
gpu_output.role = BufferRole::OUTPUT;
gpu_output.shape = {seq_len, d_model};
gpu_output.tensor_type = BufferTensorType::FP32;
gpu_output.device_idx = 0;  // CUDA GPU 0

manager.allocateBuffer("layer0_attention", gpu_output);

// 3. Retrieve (returns ITensor*, not TensorBase*)
ITensor* buffer = manager.getBuffer("layer0_attention", "attention_context");
ASSERT_TRUE(buffer->is_on_gpu());
ASSERT_EQ(buffer->home_device(), DeviceId::cuda(0));

// 4. Cast to CUDA tensor for typed access
auto* cuda_tensor = dynamic_cast<CUDAFp32Tensor*>(buffer);
float* d_ptr = cuda_tensor->typed_data();  // Device pointer
```

---

## Appendix: File References

- **TensorFactory**: [src/v2/tensors/TensorFactory.h](../../../src/v2/tensors/TensorFactory.h)
- **CUDATensorBase**: [src/v2/tensors/cuda/CUDATensorBase.h](../../../src/v2/tensors/cuda/CUDATensorBase.h)
- **CUDATypedTensor**: [src/v2/tensors/cuda/CUDATypedTensor.h](../../../src/v2/tensors/cuda/CUDATypedTensor.h)
- **DeviceGraphBufferManager**: [src/v2/execution/DeviceGraphBufferManager.h](../../../src/v2/execution/DeviceGraphBufferManager.h)
- **BufferRole/BufferDescriptor**: [src/v2/execution/BufferRole.h](../../../src/v2/execution/BufferRole.h)
- **GraphSchema (BufferSpec)**: [src/v2/execution/GraphSchema.h](../../../src/v2/execution/GraphSchema.h)
- **GraphResolver (ResolvedBufferSpec)**: [src/v2/execution/GraphResolver.h](../../../src/v2/execution/GraphResolver.h)
- **DeviceId**: [src/v2/backends/DeviceId.h](../../../src/v2/backends/DeviceId.h)
- **DeviceManager**: [src/v2/backends/ComputeBackend.h](../../../src/v2/backends/ComputeBackend.h)
- **CUDABackend**: [src/v2/backends/cuda/CUDABackend.h](../../../src/v2/backends/cuda/CUDABackend.h)
