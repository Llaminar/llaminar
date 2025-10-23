# Multi-GPU Heterogeneous Architecture Design

**Author:** David Sanftenberg  
**Date:** 2025-10-XX  
**Status:** Design Complete, Implementation Pending

## Executive Summary

This document describes the architectural refactoring to support **heterogeneous multi-GPU execution on a single MPI rank**. The design enables mixing different GPU backends (CUDA, ROCm, Vulkan) simultaneously, such as using an RTX 3090 (CUDA) and RX 7900 XTX (ROCm) together for load-balanced inference.

## Motivation

### Problem Statement

The previous architecture had a 1:1 relationship between MPI rank and compute backend:
- Each rank had one `ComputeContext` (CPU or single GPU)
- Cannot use multiple GPUs of different types on same rank
- Cannot load-balance layers across heterogeneous GPUs

### User Requirements

> "I do want to support multiple types of GPU at the same time. So we should be able to mix hipBLAS, cuBLAS, vulkan on the same mpi rank. Like if my machine has both an RTX 7900 and an RTX 3090"

## Architecture Overview

### Component Hierarchy

```
DeviceManager (singleton)
  ├─ enumerate all devices (CUDA, ROCm, Vulkan, CPU)
  ├─ create_context(device_index) → ComputeContext
  └─ auto-select device based on memory/round-robin

TensorBase (per-tensor device affinity)
  ├─ device_index() → -1 (host) or ≥0 (device index)
  ├─ set_device(device_idx) → upload to specific device
  └─ createGemm() → returns kernel for tensor's device

ITensorGemm (kernel interface)
  └─ multiply(..., mpi_ctx, device_idx) → executes on specified device

QwenPipeline (orchestrator)
  ├─ uses device_idx for tensor placement
  └─ kernels automatically execute on tensor's device
```

## Key Design Decisions

### 1. Device Manager Singleton

**Why:** Single source of truth for all available devices across system

```cpp
class DeviceManager {
public:
    static DeviceManager& instance();  // Singleton
    void initialize();  // Enumerate all CUDA, ROCm, Vulkan, CPU devices
    const std::vector<ComputeDevice>& devices() const;
    
    // Context creation
    std::shared_ptr<ComputeContext> create_context(size_t device_index);
    
    // Device selection
    int find_device(ComputeBackendType type, int device_id = 0) const;
    size_t select_device(size_t estimated_memory_bytes = 0);  // Auto-select
    
    // Queries
    bool has_gpu() const;
    std::vector<size_t> get_devices_by_type(ComputeBackendType type) const;

private:
    std::vector<ComputeDevice> devices_;  // All enumerated devices
    std::vector<std::shared_ptr<ComputeContext>> contexts_;  // Cached per-device
    size_t last_selected_device_ = 0;  // Round-robin state
};
```

**Benefits:**
- ✅ Enumerate devices once at startup
- ✅ Cache ComputeContext instances (expensive to create)
- ✅ Round-robin load balancing across GPUs
- ✅ Memory-aware device selection

### 2. Per-Tensor Device Affinity

**Why:** Each tensor can reside on different device, enabling layer-level load balancing

```cpp
class TensorBase {
    virtual int device_index() const = 0;  // -1 = host, ≥0 = device index
    virtual bool set_device(int device_idx) = 0;  // Upload to device
    virtual bool is_on_device(int device_idx) const = 0;
};

class SimpleTensor : public TensorBase {
    int device_idx_ = -1;  // Current device
    std::vector<float> host_data_;  // Always allocated
    void* device_data_ = nullptr;  // Allocated if device_idx ≥ 0
    
    bool host_dirty_ = false;  // Host modified, needs upload
    bool device_dirty_ = false;  // Device modified, needs download
    
    // Per-device kernel cache
    std::map<int, std::unique_ptr<ITensorGemm>> gemm_kernels_;
    
    void sync_to_device();  // Upload if host_dirty
    void sync_from_device();  // Download if device_dirty
};
```

**Benefits:**
- ✅ Lazy transfer (only upload/download when dirty)
- ✅ Host fallback (always have CPU copy)
- ✅ Per-device kernel caching (avoid recreation)
- ✅ Cross-device transfers handled automatically

### 3. Device Index vs ComputeContext Pointer

**Before:** Kernels accepted `ComputeContext* compute_ctx`
**After:** Kernels accept `int device_idx`

**Rationale:**
- `ComputeContext*` tied execution to single context → cannot switch devices
- `device_idx` allows kernel to query DeviceManager for correct context
- Enables pipeline to orchestrate multi-device execution without managing contexts

```cpp
// Old API (single context per rank)
class ITensorGemm {
    virtual bool multiply(..., const ComputeContext* compute_ctx) = 0;
};

// New API (multi-device per rank)
class ITensorGemm {
    virtual bool multiply(..., int device_idx) = 0;
    // Kernel internally does: auto ctx = DeviceManager::instance().create_context(device_idx);
};
```

### 4. MPIContext Separation

**Why:** MPI coordination orthogonal to device execution

```cpp
// MPI handles distributed coordination
class MPIContext {
    void allreduce_sum(const float*, float*, size_t);
    void broadcast(float*, size_t, int root = 0);
    std::pair<size_t, size_t> get_local_slice(size_t total);
};

// Device handles local execution
class ComputeContext {
    virtual void* allocate(size_t bytes) = 0;
    virtual void copy_to_device(void* dst, const void* src, size_t bytes) = 0;
};
```

**Benefits:**
- ✅ Can do distributed inference without GPU (OpenBLAS + MPI)
- ✅ Can do single-GPU inference without MPI
- ✅ Clean separation of concerns

## Usage Examples

### Example 1: Heterogeneous Multi-GPU

```cpp
// Initialize device manager
DeviceManager::instance().initialize();
// Devices: [CPU_OPENBLAS(0), GPU_CUDA(1), GPU_ROCM(2)]

// Create pipeline with MPI context
auto mpi_ctx = MPIContextFactory::global();
auto pipeline = std::make_unique<QwenPipeline>("model.gguf", mpi_ctx, -1 /*CPU by default*/);

// Place different layers on different devices
for (int i = 0; i < 12; ++i) {
    pipeline->get_layer_weight(i, "wq")->set_device(1);  // First 12 layers on CUDA
}
for (int i = 12; i < 24; ++i) {
    pipeline->get_layer_weight(i, "wq")->set_device(2);  // Last 12 layers on ROCm
}

// Forward pass automatically routes to correct device per layer
pipeline->forward(tokens, seq_len);
```

### Example 2: Auto-Select Device

```cpp
// Let DeviceManager choose best device (prefers GPU, uses round-robin)
size_t device_idx = DeviceManager::instance().select_device(1024*1024*1024);  // 1GB estimate

auto pipeline = std::make_unique<QwenPipeline>("model.gguf", mpi_ctx, device_idx);
// All tensors default to device_idx, but can override per-tensor
```

### Example 3: Cross-Device Transfer

```cpp
// Tensor on CUDA device
auto wq = std::make_shared<IQ4_NLTensor>(...);
wq->set_device(1);  // Upload to CUDA device 1

// Activation on ROCm device
auto activation = std::make_shared<SimpleTensor>(...);
activation->set_device(2);  // Place on ROCm device 2

// GEMM kernel handles cross-device transfer automatically
auto gemm = wq->createGemm();
gemm->multiply(activation->data(), output->data(), m, n, k, true, 1.0f, 0.0f, nullptr, 1);
// Kernel detects: activation on device 2, weight on device 1, output wants device 1
// → transfers activation from device 2 to device 1, executes GEMM, result on device 1
```

## Implementation Status

### Completed (Design Phase)

- ✅ MPIContext abstraction (src/MPIContext.h)
- ✅ ComputeContext interface (src/ComputeBackend.h)
- ✅ DeviceManager singleton (src/ComputeBackend.h)
- ✅ TensorBase device affinity API (src/tensors/TensorBaseInterface.h)
- ✅ SimpleTensor multi-device storage (src/tensors/TensorBaseInterface.h)
- ✅ IQ4_NLTensor device upload (src/tensors/TensorBaseInterface.h)
- ✅ Kernel interfaces updated (src/kernels/TensorKernels.h)
- ✅ QwenPipeline example updated (src/QwenPipelineExample.h)

### Pending (Implementation Phase)

#### P0: Core Infrastructure
- ❌ DeviceManager::initialize() implementation
  - Enumerate CUDA devices (cudaGetDeviceCount, cudaGetDeviceProperties)
  - Enumerate ROCm devices (hipGetDeviceCount, hipGetDeviceProperties)
  - Enumerate Vulkan devices (vkEnumeratePhysicalDevices)
  - Add CPU as device index 0

- ❌ SimpleTensor::sync_to_device() / sync_from_device()
  - cudaMemcpy for CUDA
  - hipMemcpy for ROCm
  - vkCmdCopyBuffer for Vulkan

- ❌ IQ4_NLTensor::set_device() implementation
  - Upload quantized blocks to GPU memory
  - Handle 64B aligned block structure

#### P1: CUDA Backend (Most Common)
- ❌ CUDAComputeContext implementation
  - allocate() → cudaMalloc
  - free() → cudaFree
  - copy_to_device() → cudaMemcpy H2D
  - copy_from_device() → cudaMemcpy D2H
  - synchronize() → cudaDeviceSynchronize

- ❌ CUDAGemmKernel implementation
  - Wrap cuBLAS cublasGemmEx
  - Support BF16 via CUDA_R_16BF
  - Fused dequant for IQ4_NL weights (CUDA kernel)

#### P2: ROCm Backend (AMD GPUs)
- ❌ ROCmComputeContext implementation
  - Similar to CUDA but hipMalloc, hipFree, hipMemcpy

- ❌ ROCmGemmKernel implementation
  - Wrap hipBLAS hipblasGemmEx
  - Fused dequant kernel for IQ4_NL

#### P3: Vulkan Backend (Cross-Platform)
- ❌ VulkanComputeContext implementation
  - vkAllocateMemory, vkFreeMemory
  - vkCmdCopyBuffer for transfers

- ❌ VulkanGemmKernel implementation
  - Compute shader for GEMM
  - Shader specialization for different sizes

#### P4: Multi-GPU Load Balancing
- ❌ Heuristics for automatic layer placement
  - Memory-based (fit largest model on GPU with most memory)
  - Performance-based (benchmark and assign to fastest)
  - Round-robin (simple default)

- ❌ Pipeline orchestration with cross-device transfers
  - Detect when tensor on device A, next op needs device B
  - Insert explicit transfer operation
  - Optimize to minimize transfers (graph analysis)

## Performance Expectations

### Single GPU Baseline
- Qwen 2.5 0.5B Q8: ~1210 tok/s @ batch=512 (llama.cpp baseline)

### Multi-GPU Speedup
- **Linear scaling ideal:** 2 GPUs = 2× throughput
- **Realistic:** 1.7-1.8× (transfer overhead, load imbalance)
- **Heterogeneous:** RTX 3090 (24GB, fast) + RX 7900 XTX (24GB, fast) → ~1.8× expected

### Bottlenecks
- PCIe transfer overhead for cross-device ops
- Load imbalance if one GPU much slower
- MPI synchronization points (barriers between layers)

## Testing Strategy

### Unit Tests
1. **DeviceManager enumeration**
   - Mock CUDA/ROCm device lists
   - Verify correct ordering (GPU before CPU)
   - Test find_device() with various queries

2. **Tensor sync logic**
   - Host → device → host roundtrip
   - Verify dirty flags set/cleared correctly
   - Test lazy transfer (no sync when not dirty)

3. **Per-device kernel caching**
   - Create GEMM kernel for device 1
   - Create GEMM kernel for device 2
   - Verify different instances (not shared)

### Integration Tests
1. **Single-GPU inference**
   - CUDA only: set all tensors to device 1
   - ROCm only: set all tensors to device 2
   - Vulkan only: set all tensors to device 3
   - Verify correctness vs CPU reference

2. **Multi-GPU heterogeneous**
   - Layers 0-11 on CUDA device 1
   - Layers 12-23 on ROCm device 2
   - Verify correctness vs CPU reference
   - Verify speedup >1.5×

3. **Cross-device transfer**
   - Weight on device 1, activation on device 2
   - Verify kernel handles transfer automatically
   - Verify result correct vs CPU

### Benchmarks
1. **Single-GPU vs Multi-GPU**
   - Measure throughput (tok/s) with 1 GPU
   - Measure throughput with 2 GPUs (same type)
   - Measure throughput with 2 GPUs (heterogeneous)
   - Expected: 1.7-1.8× speedup with 2 GPUs

2. **Transfer overhead**
   - Measure time for explicit host ↔ device transfer
   - PCIe 3.0 x16: ~12 GB/s (theoretical)
   - Realistic: ~10 GB/s (measure actual)

## Migration Path

### Phase 1: Core Infrastructure (Week 1)
- Implement DeviceManager::initialize() with CUDA/ROCm enumeration
- Implement SimpleTensor sync logic (cudaMemcpy, hipMemcpy)
- Basic unit tests for enumeration and sync

### Phase 2: CUDA Backend (Week 2)
- CUDAComputeContext implementation
- CUDAGemmKernel with cuBLAS
- Fused dequant CUDA kernel for IQ4_NL
- Integration test: single-GPU CUDA inference

### Phase 3: ROCm Backend (Week 3)
- ROCmComputeContext implementation
- ROCmGemmKernel with hipBLAS
- Fused dequant kernel for IQ4_NL
- Integration test: single-GPU ROCm inference

### Phase 4: Multi-GPU Orchestration (Week 4)
- Cross-device transfer logic in kernels
- Load balancing heuristics (round-robin first, then memory-based)
- Integration test: heterogeneous multi-GPU
- Benchmark: measure speedup vs single GPU

### Phase 5: Optimization (Week 5+)
- NCCL/RCCL integration for GPU-to-GPU collectives (faster than MPI)
- Overlap computation and transfer (async streams)
- Graph-based transfer optimization (minimize cross-device ops)
- Performance tuning based on profiling

## Success Criteria

### Functional Requirements
- ✅ Can enumerate all devices (CPU + all GPUs)
- ✅ Can place tensors on specific devices
- ✅ Can execute GEMM on CUDA device
- ✅ Can execute GEMM on ROCm device
- ✅ Can use RTX 3090 (CUDA) + RX 7900 XTX (ROCm) simultaneously
- ✅ Produces correct results (parity with CPU reference)

### Performance Requirements
- ✅ Single-GPU: Match or exceed llama.cpp performance (1210 tok/s @ batch=512)
- ✅ Multi-GPU: ≥1.7× speedup with 2 GPUs of same type
- ✅ Heterogeneous: ≥1.5× speedup with 2 different GPUs
- ✅ Transfer overhead: <5% of total runtime for typical workloads

## References

### Code Files
- `src/MPIContext.h` - MPI coordination abstraction
- `src/ComputeBackend.h` - Device manager and context interfaces
- `src/tensors/TensorBaseInterface.h` - Tensor device affinity
- `src/kernels/TensorKernels.h` - Kernel interfaces with device_idx
- `src/QwenPipelineExample.h` - Example pipeline orchestration

### External Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Vulkan Specification](https://www.khronos.org/vulkan/)
- [cuBLAS API Reference](https://docs.nvidia.com/cuda/cublas/)
- [hipBLAS Documentation](https://rocm.docs.amd.com/projects/hipBLAS/)

### Prior Work
- IQ4_NL tile sweep optimization: +41% FP32, +26% BF16 (64×32 optimal tiles)
- BF16 activation strategy: Selective use for bandwidth-bound ops
- Operator elimination design: Direct kernel orchestration from pipelines

---

**Document Status:** ✅ Design Complete  
**Next Steps:** Begin Phase 1 implementation (DeviceManager::initialize())
