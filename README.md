# Llaminar v2 - Clean Greenfield Implementation

**Status**: Core Architecture Complete, CPU Backend Operational, GPU Backends Pending

**Current State** (October 2025):
- ✅ **IBlockDecoder Strategy Pattern**: Generic quantized GEMM kernel complete
- ✅ **IQ4_NL Quantized Tensors**: Fused dequantization with 335-451 GFLOPS performance
- ✅ **CPU Backend**: QuantizedGemmKernel fully functional
- ✅ **Basic Pipeline**: QwenPipeline structure established
- 🔄 **Full Pipeline**: Attention/FFN operators in progress
- ❌ **GPU Backends**: CUDA/ROCm/Vulkan not yet implemented
- ❌ **MPI Distribution**: Not yet ported from V1

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
- ✅ Heterogeneous multi-GPU support (CUDA + ROCm + Vulkan on same rank) - planned
- ✅ Clean, minimal interfaces

## Project Structure

```
src/v2/
├── utils/              # Low-level utilities (no dependencies)
│   ├── MPIContext      # MPI initialization and rank info
│   ├── CPUFeatures     # SIMD capability detection
│   └── Logging         # Simple logging system
├── backends/           # Device abstraction
│   ├── ComputeBackend  # ✅ Base device interface (implemented)
│   ├── CPUBackend      # ✅ CPU implementation (implicit, via kernels)
│   ├── CUDABackend     # ❌ CUDA implementation (pending)
│   ├── ROCmBackend     # ❌ AMD GPU implementation (pending)
│   └── VulkanBackend   # ❌ Vulkan compute implementation (pending)
├── tensors/            # Tensor data structures
│   ├── TensorBase      # ✅ Base tensor interface (implemented)
│   ├── FP32Tensor      # ✅ Dense float32 tensor (implemented)
│   ├── BF16Tensor      # ❌ Dense bfloat16 tensor (pending)
│   ├── IQ4_NLTensor    # ✅ Quantized IQ4_NL tensor (implemented)
│   └── Q6_KTensor      # ❌ Quantized Q6_K tensor (pending)
├── kernels/            # Compute kernels (per-device)
│   ├── cpu/            # ✅ QuantizedGemm.{h,cpp} (implemented)
│   ├── cuda/           # ❌ GPU kernels (pending)
│   ├── rocm/           # ❌ GPU kernels (pending)
│   └── vulkan/         # ❌ GPU kernels (pending)
└── pipelines/          # Model execution
    └── QwenPipeline    # 🔄 Basic structure (attention/FFN in progress)
```

## Key Design Principles

### 1. Separation of Concerns (Operator-Free Architecture)

**No Operator Layer** - V2 eliminates the operator abstraction entirely:

```cpp
// V1 (Operator-Based) - REMOVED in V2:
// MPILinearOperator, MPIAttentionOperator, MPIRMSNormOperator, etc.

// V2 (Operator-Free) - Direct orchestration:
MPIContext       // Distributed coordination (Allreduce, broadcast, barrier)
ComputeContext   // Device execution (allocate, copy, sync)
TensorBase       // Data storage with device affinity
ITensorGemm      // Kernel execution interface
QwenPipeline     // Direct kernel orchestration (no operators)
```

**Benefit**: Reduces indirection, simplifies debugging, enables fine-grained device control.

### 2. Per-Tensor Device Affinity (Future Feature)

**Current**: Single CPU device, multi-device planned

```cpp
// FUTURE: Each tensor will know which device it's on
auto wq = std::make_shared<IQ4_NLTensor>(...);
wq->set_device(1);  // Upload to device 1 (e.g., CUDA GPU) - NOT YET IMPLEMENTED

auto wk = std::make_shared<IQ4_NLTensor>(...);
wk->set_device(2);  // Upload to device 2 (e.g., ROCm/GPU) - NOT YET IMPLEMENTED

// Kernels execute on tensor's device
auto gemm = wq->createGemm();
gemm->multiply(...);  // Runs on device 1 - NOT YET IMPLEMENTED
```

### 3. IBlockDecoder Strategy Pattern (✅ IMPLEMENTED)

**Generic quantized GEMM kernel** that works with all quantized formats:

```cpp
// IBlockDecoder interface (zero-overhead inline virtual methods)
class IBlockDecoder {
public:
    __attribute__((always_inline))
    virtual void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const = 0;
    
    __attribute__((always_inline))
    virtual size_t block_size() const = 0;
};

// IQ4_NL tensor implements IBlockDecoder
class IQ4_NLTensor : public TensorBase, public IBlockDecoder {
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const IQ4_NLBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeBlock(block, output);  // IQ4_NL-specific decode (inlined)
    }
    
    size_t block_size() const override { return 32; }  // 32 elements/block
    
    std::unique_ptr<ITensorGemm> createGemm() const override {
        return std::make_unique<QuantizedGemmKernel>(this);  // Generic kernel!
    }
};

// Generic QuantizedGemmKernel works for ALL quantized formats
class QuantizedGemmKernel : public ITensorGemm {
    const IBlockDecoder* decoder_;  // Strategy interface
    
    bool multiply(...) override {
        // Generic implementation - no format-specific code here!
        decoder_->decode_block_at(j, kb, B_block);  // Inlined (zero overhead)
        // ... accumulate ...
    }
};
```

**Performance**: 335-451 GFLOPS on CPU (measured with IQ4_NL weights)

**Code Reuse**: ~350 lines generic kernel vs ~1000 lines per format (3× reduction)

### 4. Selective BF16 (Future Feature)

```cpp
// FUTURE: Use BF16 for bandwidth-bound operations - NOT YET IMPLEMENTED
rope->apply(..., use_bf16=true);        // RoPE: 2× bandwidth savings
swiglu->apply(..., use_bf16=true);      // SwiGLU: element-wise, bandwidth-bound

// Keep FP32 for precision-critical operations
softmax->apply(..., use_bf16=false);    // Softmax: numerical stability critical
rmsnorm->apply(..., use_bf16=false);    // RMSNorm: precision matters
```

## Device Manager (Partially Implemented)

Central singleton for device enumeration and management:

```cpp
// Initialize once at startup
DeviceManager::instance().initialize();

// Enumerate devices (CURRENT: CPU only)
const auto& devices = DeviceManager::instance().devices();
// Current Output:
// Device 0: CPU (OpenBLAS)

// FUTURE Output (when GPU backends implemented):
// Device 0: CPU (OpenBLAS)
// Device 1: GPU (CUDA) - NVIDIA GeForce RTX 3090 (24 GB)
// Device 2: GPU (ROCm) - AMD Radeon RX 7900 XTX (24 GB)

// Find specific device - NOT YET IMPLEMENTED
// int cuda_idx = DeviceManager::instance().find_device(ComputeBackendType::GPU_CUDA, 0);

// Auto-select best device - NOT YET IMPLEMENTED (always CPU currently)
// int device_idx = DeviceManager::instance().select_device(1024*1024*1024);  // 1GB estimate
```

## Example Usage (Future Features)

**Note**: The following examples describe planned V2 functionality. Current implementation supports CPU-only inference with IQ4_NL quantization.

### Heterogeneous Multi-GPU Inference (Planned)

**Current Status**: CUDA/ROCm backends not yet implemented. This example shows the intended API design.

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
    
    // FUTURE: Place layers on different devices (NOT YET IMPLEMENTED)
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

**Current Status**: Phase 1 partially complete. Core infrastructure exists, GPU backends pending.

### Phase 1: Core Infrastructure ✅ MOSTLY COMPLETE
- [x] **TensorBase** - Base tensor interface with device affinity
- [x] **IQ4_NLTensor** - Quantized IQ4_NL tensor with IBlockDecoder pattern
- [x] **FP32Tensor** - Dense float32 tensor
- [x] **QuantizedGemmKernel** - Generic CPU GEMM (335-451 GFLOPS)
- [x] **ComputeBackend** - Base device interface
- [x] **QwenPipeline** - Basic pipeline structure
- [ ] **DeviceManager::initialize()** - Enumerate CUDA/ROCm/Vulkan/CPU devices (CPU only currently)
- [ ] **FP32Tensor::sync_to_device()** / `sync_from_device()` - Host↔device transfers (NOT YET IMPLEMENTED)
- [ ] **CPUComputeContext** implementation (OpenBLAS) - Partially implemented

### Phase 2: CUDA Backend ❌ NOT STARTED
- [ ] `CUDAComputeContext` implementation
- [ ] `CUDAGemmKernel` - cuBLAS wrapper
- [ ] Fused IQ4_NL dequant CUDA kernel
- [ ] Integration test: Single-GPU CUDA inference

### Phase 3: ROCm Backend ❌ NOT STARTED
- [ ] `ROCmComputeContext` implementation
- [ ] `ROCmGemmKernel` - hipBLAS wrapper
- [ ] Fused IQ4_NL dequant HIP kernel
- [ ] Integration test: Single-GPU ROCm inference

### Phase 4: Multi-GPU Orchestration ❌ NOT STARTED
- [ ] Cross-device transfer logic in kernels
- [ ] Auto layer placement (`QwenPipeline::auto_place_layers()`)
- [ ] Integration test: Heterogeneous multi-GPU (CUDA + ROCm)
- [ ] Benchmark: Measure speedup vs single GPU

### Phase 5: Optimization ❌ NOT STARTED
- [ ] Async streams (overlap transfer + compute)
- [ ] NCCL/RCCL integration (GPU-to-GPU collectives)
- [ ] Graph-based transfer optimization
- [ ] Performance tuning

## Building

**Current**: CPU-only builds work. GPU backend options exist but backends are not yet implemented.

**CMake configuration** (see `src/v2/CMakeLists.txt`):

```cmake
# Backend options (default OFF - GPU backends not yet implemented)
option(HAVE_CUDA "Enable CUDA backend" OFF)    # NOT YET IMPLEMENTED
option(HAVE_ROCM "Enable ROCm backend" OFF)    # NOT YET IMPLEMENTED
option(HAVE_VULKAN "Enable Vulkan backend" OFF)  # NOT YET IMPLEMENTED

# Build v2 library (CURRENT IMPLEMENTATION)
add_library(llaminar2_core STATIC
    backends/ComputeBackend.cpp
    pipelines/QwenPipeline.cpp
    tensors/FP32Tensor.cpp
    tensors/IQ4_NLTensor.cpp
    kernels/cpu/QuantizedGemm.cpp
)

target_link_libraries(llaminar2_core PUBLIC
    MPI::MPI_CXX
    ${OPENBLAS_LIBRARIES}
)

# FUTURE: GPU backends (not yet created)
if(HAVE_CUDA)
    # Will add: src/v2/backends/CUDABackend.cu
    # Will add: src/v2/kernels/cuda/*.cu
    target_link_libraries(llaminar2_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Executable
add_executable(llaminar2 Main.cpp)
target_link_libraries(llaminar2 llaminar2_core)
```

**Build commands:**

```bash
# Debug build (from workspace root)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --parallel

# Release build
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run (device enumeration only - inference not yet functional)
./build_v2/llaminar2 --list-devices
# Output: Device 0: CPU (OpenBLAS)

# FUTURE: When inference is implemented
# ./build_v2/llaminar2 -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "Hello" -n 128
```

## Testing

**Current**: Basic unit tests exist. Full integration tests pending pipeline completion.

```bash
# FUTURE: Unit tests (planned, not yet created)
# ./build_v2/test_device_manager
# ./build_v2/test_tensor_sync

# FUTURE: Integration tests (planned)
# ./build_v2/test_cuda_inference
# ./build_v2/test_rocm_inference
# ./build_v2/test_multi_gpu

# FUTURE: Benchmarks (planned)
# ./build_v2/benchmark_single_gpu
# ./build_v2/benchmark_multi_gpu
```

## Performance Targets (Aspirational)

**Note**: These are targets for when GPU backends are implemented. Current CPU implementation achieves 335-451 GFLOPS with IQ4_NL quantization.

- **Single GPU (CUDA):** ≥1210 tok/s @ batch=512 (match llama.cpp baseline)
- **Multi-GPU (2× same type):** ≥1.7× speedup (2057 tok/s)
- **Heterogeneous (CUDA + ROCm):** ≥1.5× speedup (1815 tok/s)

## Documentation

- **V2 Architecture Guide:** `.github/instructions/llaminar-v2-architecture.instructions.md` (comprehensive)
- **Development Guidelines:** `.github/copilot-instructions.md` (sections on V1 vs V2)
- **IBlockDecoder Pattern:** See V2 Architecture Guide, section "Quantized Tensor Strategy Pattern"

## Migration from v1

Llaminar v2 is **not backward compatible** with v1. Key differences:

| Aspect | v1 | v2 |
|--------|----|----|
| Namespace | `llaminar` | `llaminar2` |
| Operators | MPILinearOperator, etc. | Eliminated |
| Slab cache | WeightSlab, LRU | Eliminated |
| Kernel API | `compute_ctx` pointer | `device_idx` integer |
| Device placement | Per-rank | Per-tensor (planned) |
| Multi-GPU | One GPU per rank | Multiple GPUs per rank (planned) |

v1 remains in `src/` (production), v2 is in `src/v2/` (experimental).

---

**Status:** ✅ Core Architecture Complete, CPU Backend Operational  
**Next Step:** GPU backend implementation (Phase 2/3)
