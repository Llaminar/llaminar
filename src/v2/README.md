# Llaminar v2 - Clean Greenfield Implementation

**Status:** Architecture Complete, Implementation Pending  
**Namespace:** `llaminar2`  
**Target:** Multi-GPU heterogeneous inference with direct kernel orchestration

## Architecture Overview

Llaminar v2 is a radical simplification of the original codebase, eliminating:
- ❌ Operator layer (MPILinearOperator, MPIAttentionOperator, etc.)
- ❌ Slab cache (WeightSlab, LRU caching)
- ❌ Complex abstraction layers

And focusing on:
- ✅ Direct kernel orchestration from pipelines
- ✅ Per-tensor device placement
- ✅ Heterogeneous multi-GPU support (CUDA + ROCm + Vulkan on same rank)
- ✅ Clean, minimal interfaces

## Directory Structure

```
src/v2/
├── utils/
│   └── MPIContext.h          # MPI coordination abstraction
├── backends/
│   ├── ComputeBackend.h      # Device manager + compute contexts
│   ├── ComputeBackend.cpp    # CPU implementation
│   ├── CUDABackend.cu        # CUDA implementation (pending)
│   ├── ROCmBackend.cpp       # ROCm implementation (pending)
│   └── VulkanBackend.cpp     # Vulkan implementation (pending)
├── tensors/
│   ├── Tensors.h             # Tensor interface and implementations (FP32, IQ4_NL)
│   ├── TensorBase.cpp        # FP32Tensor + IQ4_NLTensor implementations
│   └── TensorKernels.h       # Kernel interfaces (GEMM, RoPE, SwiGLU, etc.)
├── kernels/
│   ├── CUDAGemm.cu           # CUDA GEMM kernel (pending)
│   ├── ROCmGemm.cpp          # ROCm GEMM kernel (pending)
│   ├── iq4nl_dequant.cu      # CUDA IQ4_NL dequant kernel (pending)
│   └── iq4nl_dequant.hip     # HIP IQ4_NL dequant kernel (pending)
├── pipelines/
│   ├── QwenPipeline.h        # Qwen transformer pipeline
│   └── QwenPipeline.cpp      # Pipeline implementation
├── tools/
│   └── (benchmark tools, profilers, etc.)
├── Main.cpp                  # Entry point
├── CMakeLists.txt            # Build configuration
└── README.md                 # This file
```

## Key Design Principles

### 1. Separation of Concerns

```cpp
MPIContext       // Distributed coordination (Allreduce, broadcast, barrier)
ComputeContext   // Device execution (allocate, copy, sync)
TensorBase       // Data storage with device affinity
ITensorGemm      // Kernel execution
QwenPipeline     // Orchestration
```

### 2. Per-Tensor Device Affinity

```cpp
// Each tensor knows which device it's on
auto wq = std::make_shared<IQ4_NLTensor>(...);
wq->set_device(1);  // Upload to device 1 (e.g., CUDA GPU)

auto wk = std::make_shared<IQ4_NLTensor>(...);
wk->set_device(2);  // Upload to device 2 (e.g., ROCm GPU)

// Kernels execute on tensor's device
auto gemm = wq->createGemm();
gemm->multiply(...);  // Runs on device 1
```

### 3. Direct Kernel Orchestration

```cpp
// No operators, pipelines call kernels directly
auto wq_gemm = wq->createGemm();
wq_gemm->multiply(x->data(), Q->data(), seq_len, d_model, d_model,
                  true, 1.0f, 0.0f, mpi_ctx.get(), device_idx);

auto rope = Q->createRoPE();
rope->apply(Q->data(), K->data(), position_ids, seq_len, n_heads, head_dim,
            true, mpi_ctx.get(), device_idx);
```

### 4. Selective BF16

```cpp
// Use BF16 for bandwidth-bound operations
rope->apply(..., use_bf16=true);        // RoPE: 2× bandwidth savings
swiglu->apply(..., use_bf16=true);      // SwiGLU: element-wise, bandwidth-bound
gemm->multiply(...);                     // GEMM: uses BF16 for IQ4_NL weights (+26% speedup)

// Keep FP32 for precision-critical operations
softmax->apply(..., use_bf16=false);    // Softmax: numerical stability critical
rmsnorm->apply(..., use_bf16=false);    // RMSNorm: precision matters
```

## Device Manager

Central singleton for device enumeration and management:

```cpp
// Initialize once at startup
DeviceManager::instance().initialize();

// Enumerate devices
const auto& devices = DeviceManager::instance().devices();
// Output:
// Device 0: CPU (OpenBLAS)
// Device 1: GPU (CUDA) - NVIDIA GeForce RTX 3090 (24 GB)
// Device 2: GPU (ROCm) - AMD Radeon RX 7900 XTX (24 GB)

// Find specific device
int cuda_idx = DeviceManager::instance().find_device(ComputeBackendType::GPU_CUDA, 0);

// Auto-select best device (prefers GPU, uses round-robin)
int device_idx = DeviceManager::instance().select_device(1024*1024*1024);  // 1GB estimate
```

## Example Usage

### Heterogeneous Multi-GPU Inference

```cpp
#include "QwenPipeline.h"
#include "ComputeBackend.h"
#include "MPIContext.h"

int main() {
    MPI_Init(nullptr, nullptr);
    
    // Initialize device manager
    DeviceManager::instance().initialize();
    
    // Create pipeline
    auto mpi_ctx = MPIContextFactory::global();
    auto pipeline = std::make_unique<QwenPipeline>("model.gguf", mpi_ctx, -1);
    
    // Place layers on different devices
    int cuda_idx = DeviceManager::instance().find_device(ComputeBackendType::GPU_CUDA, 0);
    int rocm_idx = DeviceManager::instance().find_device(ComputeBackendType::GPU_ROCM, 0);
    
    // First 12 layers on CUDA (RTX 3090)
    for (int i = 0; i < 12; ++i) {
        pipeline->get_layer_weight(i, "wq")->set_device(cuda_idx);
        pipeline->get_layer_weight(i, "wk")->set_device(cuda_idx);
        pipeline->get_layer_weight(i, "wv")->set_device(cuda_idx);
        pipeline->get_layer_weight(i, "wo")->set_device(cuda_idx);
    }
    
    // Last 12 layers on ROCm (RX 7900 XTX)
    for (int i = 12; i < 24; ++i) {
        pipeline->get_layer_weight(i, "wq")->set_device(rocm_idx);
        // ... other weights ...
    }
    
    // Run inference (automatically handles cross-device transfers)
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    pipeline->forward(tokens.data(), tokens.size());
    
    MPI_Finalize();
    return 0;
}
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [ ] `DeviceManager::initialize()` - Enumerate CUDA/ROCm/Vulkan/CPU devices
- [ ] `FP32Tensor::sync_to_device()` / `sync_from_device()` - Host↔device transfers
- [ ] `CPUComputeContext` implementation (OpenBLAS)

### Phase 2: CUDA Backend (Week 2)
- [ ] `CUDAComputeContext` implementation
- [ ] `CUDAGemmKernel` - cuBLAS wrapper
- [ ] Fused IQ4_NL dequant CUDA kernel
- [ ] Integration test: Single-GPU CUDA inference

### Phase 3: ROCm Backend (Week 3)
- [ ] `ROCmComputeContext` implementation
- [ ] `ROCmGemmKernel` - hipBLAS wrapper
- [ ] Fused IQ4_NL dequant HIP kernel
- [ ] Integration test: Single-GPU ROCm inference

### Phase 4: Multi-GPU Orchestration (Week 4)
- [ ] Cross-device transfer logic in kernels
- [ ] Auto layer placement (`QwenPipeline::auto_place_layers()`)
- [ ] Integration test: Heterogeneous multi-GPU (CUDA + ROCm)
- [ ] Benchmark: Measure speedup vs single GPU

### Phase 5: Optimization (Week 5+)
- [ ] Async streams (overlap transfer + compute)
- [ ] NCCL/RCCL integration (GPU-to-GPU collectives)
- [ ] Graph-based transfer optimization
- [ ] Performance tuning

## Building

**CMake configuration:**

```cmake
# Enable backends
option(HAVE_CUDA "Enable CUDA backend" ON)
option(HAVE_ROCM "Enable ROCm backend" OFF)
option(HAVE_VULKAN "Enable Vulkan backend" OFF)

# Build v2 library
add_library(llaminar2_core
    src/v2/ComputeBackend.cpp
    src/v2/TensorBase.cpp
    src/v2/QwenPipeline.cpp
    # ... other sources
)

target_include_directories(llaminar2_core PUBLIC src/v2)
target_link_libraries(llaminar2_core OpenBLAS::OpenBLAS MPI::MPI_CXX)

if(HAVE_CUDA)
    enable_language(CUDA)
    target_sources(llaminar2_core PRIVATE
        src/v2/backends/CUDABackend.cu
        src/v2/kernels/CUDAGemm.cu
    )
    target_link_libraries(llaminar2_core CUDA::cudart CUDA::cublas)
endif()

# Executable
add_executable(llaminar2 src/v2/Main.cpp)
target_link_libraries(llaminar2 llaminar2_core)
```

**Build commands:**

```bash
# Release build
cmake -B build_v2 -S . -DCMAKE_BUILD_TYPE=Release \
  -DHAVE_CUDA=ON -DHAVE_ROCM=OFF
cmake --build build_v2 --target llaminar2 --parallel

# Run
./build_v2/llaminar2 --list-devices
./build_v2/llaminar2 -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Hello" -n 128 --device cuda:0
```

## Testing

```bash
# Unit tests
./build_v2/test_device_manager
./build_v2/test_tensor_sync

# Integration tests
./build_v2/test_cuda_inference
./build_v2/test_rocm_inference
./build_v2/test_multi_gpu

# Benchmarks
./build_v2/benchmark_single_gpu
./build_v2/benchmark_multi_gpu
```

## Performance Targets

- **Single GPU (CUDA):** ≥1210 tok/s @ batch=512 (match llama.cpp baseline)
- **Multi-GPU (2× same type):** ≥1.7× speedup (2057 tok/s)
- **Heterogeneous (CUDA + ROCm):** ≥1.5× speedup (1815 tok/s)

## Documentation

- **Architecture Overview:** `/workspaces/llaminar/docs/MULTI_GPU_ARCHITECTURE.md`
- **Implementation Guide:** `/workspaces/llaminar/docs/MULTI_GPU_IMPLEMENTATION_GUIDE.md`
- **Refactoring Summary:** `/workspaces/llaminar/docs/ARCHITECTURE_REFACTORING_SUMMARY.md`

## Migration from v1

Llaminar v2 is **not backward compatible** with v1. Key differences:

| Aspect | v1 | v2 |
|--------|----|----|
| Namespace | `llaminar` | `llaminar2` |
| Operators | MPILinearOperator, etc. | Eliminated |
| Slab cache | WeightSlab, LRU | Eliminated |
| Kernel API | `compute_ctx` pointer | `device_idx` integer |
| Device placement | Per-rank | Per-tensor |
| Multi-GPU | One GPU per rank | Multiple GPUs per rank |

v1 will remain in `src/` for reference, v2 is in `src/v2/`.

---

**Status:** ✅ Architecture Complete  
**Next Step:** Phase 1 implementation (DeviceManager::initialize())
