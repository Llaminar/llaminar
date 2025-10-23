# Multi-GPU Implementation Guide

**Status:** Ready to implement  
**Priority:** P0 - Core infrastructure blocking all GPU work  
**Estimated Effort:** 4-5 weeks

## Quick Start Checklist

### Week 1: Core Infrastructure ✅ Design Complete, ❌ Implementation Pending

- [ ] **DeviceManager::initialize()** (src/ComputeBackend.cpp)
  ```cpp
  void DeviceManager::initialize() {
      // 1. Add CPU as device 0
      devices_.push_back({ComputeBackendType::CPU_OPENBLAS, 0, "CPU OpenBLAS", 0, 0});
      
      #ifdef HAVE_CUDA
      // 2. Enumerate CUDA devices
      int cuda_count = 0;
      cudaGetDeviceCount(&cuda_count);
      for (int i = 0; i < cuda_count; ++i) {
          cudaDeviceProp props;
          cudaGetDeviceProperties(&props, i);
          devices_.push_back({
              ComputeBackendType::GPU_CUDA, 
              i, 
              props.name, 
              props.totalGlobalMem,
              props.major * 10 + props.minor  // e.g., 8.6 → 86
          });
      }
      #endif
      
      #ifdef HAVE_ROCM
      // 3. Enumerate ROCm devices
      int rocm_count = 0;
      hipGetDeviceCount(&rocm_count);
      for (int i = 0; i < rocm_count; ++i) {
          hipDeviceProp_t props;
          hipGetDeviceProperties(&props, i);
          devices_.push_back({
              ComputeBackendType::GPU_ROCM,
              i,
              props.name,
              props.totalGlobalMem,
              props.gcnArch  // GCN architecture version
          });
      }
      #endif
      
      #ifdef HAVE_VULKAN
      // 4. Enumerate Vulkan devices
      // TODO: vkEnumeratePhysicalDevices
      #endif
  }
  ```

- [ ] **SimpleTensor::sync_to_device()** (src/tensors/SimpleTensor.cpp)
  ```cpp
  bool SimpleTensor::sync_to_device() {
      if (!host_dirty_ || device_idx_ < 0) return true;
      
      auto& dm = DeviceManager::instance();
      const auto& device = dm.devices()[device_idx_];
      
      if (!device_data_) {
          // Allocate device memory first time
          auto ctx = dm.create_context(device_idx_);
          device_data_ = ctx->allocate(host_data_.size() * sizeof(float));
      }
      
      auto ctx = dm.create_context(device_idx_);
      ctx->copy_to_device(device_data_, host_data_.data(), host_data_.size() * sizeof(float));
      
      host_dirty_ = false;
      return true;
  }
  ```

- [ ] **SimpleTensor::sync_from_device()** (src/tensors/SimpleTensor.cpp)
  ```cpp
  bool SimpleTensor::sync_from_device() {
      if (!device_dirty_ || device_idx_ < 0) return true;
      
      auto ctx = DeviceManager::instance().create_context(device_idx_);
      ctx->copy_from_device(host_data_.data(), device_data_, host_data_.size() * sizeof(float));
      
      device_dirty_ = false;
      return true;
  }
  ```

- [ ] **Unit tests** (tests/test_device_manager.cpp)
  - Test device enumeration (mock CUDA/ROCm counts)
  - Test find_device() queries
  - Test select_device() round-robin
  - Test tensor sync roundtrip (host → device → host)

### Week 2: CUDA Backend ❌ Not Started

- [ ] **CUDAComputeContext::allocate()** (src/backends/CUDABackend.cpp)
  ```cpp
  void* CUDAComputeContext::allocate(size_t bytes) {
      void* ptr = nullptr;
      cudaSetDevice(device_id_);
      CUDA_CHECK(cudaMalloc(&ptr, bytes));
      return ptr;
  }
  ```

- [ ] **CUDAComputeContext::copy_to_device()** (src/backends/CUDABackend.cpp)
  ```cpp
  void CUDAComputeContext::copy_to_device(void* dst, const void* src, size_t bytes) {
      cudaSetDevice(device_id_);
      CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
  }
  ```

- [ ] **CUDAGemmKernel** (src/kernels/CUDAGemmKernel.cpp)
  - Wrap cuBLAS cublasGemmEx for FP32×FP32 → FP32
  - Wrap cuBLAS for FP32×BF16 → FP32 (CUDA_R_16BF)
  - Implement fused dequant+GEMM CUDA kernel for IQ4_NL

- [ ] **IQ4_NL dequant kernel** (src/kernels/iq4nl_dequant.cu)
  ```cuda
  __global__ void iq4nl_dequant_kernel(
      const uint8_t* __restrict__ blocks,  // Quantized blocks [n_blocks]
      float* __restrict__ output,          // Dequantized output [n, k]
      int n, int k) {
      
      int block_idx = blockIdx.x;
      int tid = threadIdx.x;
      
      // Load block (64 bytes)
      const IQ4NLBlock* block = (const IQ4NLBlock*)(blocks + block_idx * 64);
      
      // Dequantize 4-bit values to FP32
      // TODO: Implement using shared memory for block scale
  }
  ```

- [ ] **Integration test** (tests/test_cuda_inference.cpp)
  - Load Qwen 0.5B Q8 model
  - Place all tensors on CUDA device 0
  - Run prefill + decode
  - Compare vs CPU reference (parity test)

### Week 3: ROCm Backend ❌ Not Started

- [ ] **ROCmComputeContext** (src/backends/ROCmBackend.cpp)
  - Similar to CUDA but hipMalloc, hipMemcpy, etc.

- [ ] **ROCmGemmKernel** (src/kernels/ROCmGemmKernel.cpp)
  - Wrap hipBLAS hipblasGemmEx
  - Fused dequant+GEMM HIP kernel for IQ4_NL

- [ ] **Integration test** (tests/test_rocm_inference.cpp)
  - Same as CUDA test but on ROCm device

### Week 4: Multi-GPU Orchestration ❌ Not Started

- [ ] **Cross-device transfer in kernels**
  ```cpp
  bool CUDAGemmKernel::multiply(..., int device_idx) {
      // Check if activation tensor is on different device
      if (activation->device_index() != device_idx) {
          // Transfer activation to this device
          activation->set_device(device_idx);
      }
      
      // Execute GEMM on device_idx
      // ...
  }
  ```

- [ ] **QwenPipeline::auto_place_layers()**
  ```cpp
  void QwenPipeline::auto_place_layers() {
      auto& dm = DeviceManager::instance();
      auto gpu_devices = dm.get_devices_by_type(ComputeBackendType::GPU_CUDA);
      gpu_devices.insert(gpu_devices.end(),
                         dm.get_devices_by_type(ComputeBackendType::GPU_ROCM).begin(),
                         dm.get_devices_by_type(ComputeBackendType::GPU_ROCM).end());
      
      if (gpu_devices.empty()) return;  // No GPUs, stay on CPU
      
      // Round-robin layer placement
      for (int i = 0; i < n_layers_; ++i) {
          int device_idx = gpu_devices[i % gpu_devices.size()];
          layer_weights_[i].wq->set_device(device_idx);
          layer_weights_[i].wk->set_device(device_idx);
          layer_weights_[i].wv->set_device(device_idx);
          layer_weights_[i].wo->set_device(device_idx);
          // ... other weights ...
      }
  }
  ```

- [ ] **Integration test: heterogeneous multi-GPU**
  ```cpp
  TEST(MultiGPU, HeterogeneousCUDAROCm) {
      auto& dm = DeviceManager::instance();
      dm.initialize();
      
      auto cuda_idx = dm.find_device(ComputeBackendType::GPU_CUDA, 0);
      auto rocm_idx = dm.find_device(ComputeBackendType::GPU_ROCM, 0);
      
      ASSERT_GE(cuda_idx, 0) << "No CUDA device found";
      ASSERT_GE(rocm_idx, 0) << "No ROCm device found";
      
      auto pipeline = std::make_unique<QwenPipeline>("model.gguf", mpi_ctx, -1);
      
      // Place first half on CUDA, second half on ROCm
      for (int i = 0; i < 12; ++i) {
          pipeline->get_layer_weight(i, "wq")->set_device(cuda_idx);
      }
      for (int i = 12; i < 24; ++i) {
          pipeline->get_layer_weight(i, "wq")->set_device(rocm_idx);
      }
      
      // Run inference
      std::vector<int> tokens = {1, 2, 3, 4, 5};
      pipeline->forward(tokens.data(), tokens.size());
      
      // Compare vs CPU reference
      auto cpu_pipeline = std::make_unique<QwenPipeline>("model.gguf", mpi_ctx, -1);
      cpu_pipeline->forward(tokens.data(), tokens.size());
      
      compare_logits(pipeline->logits(), cpu_pipeline->logits(), 1e-3);
  }
  ```

- [ ] **Benchmark: multi-GPU speedup**
  ```bash
  # Baseline: Single CUDA GPU
  ./run_llaminar.sh --benchmark -m model.gguf -n 128 --device cuda:0
  # Expected: ~1200 tok/s
  
  # Two CUDA GPUs (same type)
  ./run_llaminar.sh --benchmark -m model.gguf -n 128 --device cuda:0,cuda:1
  # Expected: ~2000 tok/s (1.7× speedup)
  
  # Heterogeneous: CUDA + ROCm
  ./run_llaminar.sh --benchmark -m model.gguf -n 128 --device cuda:0,rocm:0
  # Expected: ~1800 tok/s (1.5× speedup)
  ```

## File Structure

### New Files to Create

```
src/
├── backends/
│   ├── CUDABackend.cpp       # CUDAComputeContext implementation
│   ├── ROCmBackend.cpp       # ROCmComputeContext implementation
│   └── VulkanBackend.cpp     # VulkanComputeContext implementation
│
├── kernels/
│   ├── CUDAGemmKernel.cpp    # cuBLAS wrapper
│   ├── ROCmGemmKernel.cpp    # hipBLAS wrapper
│   ├── iq4nl_dequant.cu      # CUDA dequant kernel
│   └── iq4nl_dequant.hip     # HIP dequant kernel
│
└── tensors/
    └── SimpleTensor.cpp       # Sync logic implementation

tests/
├── test_device_manager.cpp   # Unit tests for DeviceManager
├── test_tensor_sync.cpp      # Unit tests for host↔device sync
├── test_cuda_inference.cpp   # Integration test: CUDA inference
├── test_rocm_inference.cpp   # Integration test: ROCm inference
└── test_multi_gpu.cpp        # Integration test: heterogeneous multi-GPU
```

### CMake Changes

```cmake
# Option to enable CUDA backend
option(HAVE_CUDA "Enable CUDA backend" ON)
if(HAVE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DHAVE_CUDA)
    target_link_libraries(llaminar_core CUDA::cudart CUDA::cublas)
endif()

# Option to enable ROCm backend
option(HAVE_ROCM "Enable ROCm backend" OFF)
if(HAVE_ROCM)
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    add_definitions(-DHAVE_ROCM)
    target_link_libraries(llaminar_core hip::host roc::hipblas)
endif()

# Option to enable Vulkan backend
option(HAVE_VULKAN "Enable Vulkan backend" OFF)
if(HAVE_VULKAN)
    find_package(Vulkan REQUIRED)
    add_definitions(-DHAVE_VULKAN)
    target_link_libraries(llaminar_core Vulkan::Vulkan)
endif()
```

## Common Pitfalls

### Pitfall 1: Forgetting to Set Device Before CUDA Calls

**Wrong:**
```cpp
cudaMalloc(&ptr, size);  // Uses current device (undefined!)
```

**Right:**
```cpp
cudaSetDevice(device_id_);
cudaMalloc(&ptr, size);
```

### Pitfall 2: Not Checking Tensor Device Before Access

**Wrong:**
```cpp
// Assume tensor is on device
kernel->multiply(activation->data(), output, ...);  // Crash if on host!
```

**Right:**
```cpp
if (activation->device_index() != device_idx_) {
    activation->set_device(device_idx_);  // Transfer first
}
kernel->multiply(activation->data(), output, ...);
```

### Pitfall 3: Forgetting to Sync After Host Modification

**Wrong:**
```cpp
tensor->data()[0] = 1.0f;  // Modify host
// device_data_ is now stale!
kernel->multiply(tensor->data(), ...);  // Wrong result!
```

**Right:**
```cpp
tensor->data()[0] = 1.0f;  // Modify host
tensor->mark_host_dirty();  // Flag for sync
// OR: tensor->sync_to_device() explicitly
kernel->multiply(tensor->data(), ...);  // Correct, auto-syncs
```

### Pitfall 4: Race Conditions in Async Streams

**Problem:** Using async streams without proper synchronization

**Solution:** Start with synchronous transfers (cudaMemcpy), optimize to async later

```cpp
// Week 2-4: Use synchronous (safe, simple)
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

// Week 5+: Optimize with async streams
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
cudaStreamSynchronize(stream);
```

## Debugging Tips

### CUDA Debugging

```bash
# Enable CUDA error checking (adds cudaGetLastError after each call)
export LLAMINAR_CUDA_DEBUG=1

# Use cuda-memcheck to detect memory errors
cuda-memcheck ./build/test_cuda_inference

# Use Nsight Systems for profiling
nsys profile ./build/llaminar --benchmark -m model.gguf
```

### ROCm Debugging

```bash
# Enable HIP error checking
export HIP_VISIBLE_DEVICES=0  # Use only GPU 0
export ROCR_VISIBLE_DEVICES=0

# Use rocprof for profiling
rocprof --stats ./build/llaminar --benchmark -m model.gguf
```

### Multi-GPU Debugging

```bash
# Enable device placement logging
export LLAMINAR_LOG_DEVICE_PLACEMENT=1

# Output:
# [INFO] Tensor wq_layer_0: placed on device 1 (GPU_CUDA)
# [INFO] Tensor wq_layer_12: placed on device 2 (GPU_ROCM)
# [INFO] Transfer: wq_layer_0 device 1 → activation device 2 (128 MB)
```

## Performance Targets

### Single GPU (CUDA)
- **Target:** Match llama.cpp (1210 tok/s @ batch=512, pp512)
- **Minimum:** ≥1000 tok/s (acceptable)
- **Excellent:** ≥1500 tok/s (exceeds baseline)

### Multi-GPU (2× same type)
- **Target:** 1.7× speedup (2040 tok/s)
- **Minimum:** 1.5× speedup (1815 tok/s)
- **Excellent:** 1.9× speedup (2279 tok/s)

### Multi-GPU (heterogeneous CUDA + ROCm)
- **Target:** 1.5× speedup (1815 tok/s)
- **Minimum:** 1.3× speedup (1573 tok/s)
- **Excellent:** 1.7× speedup (2057 tok/s)

## Next Steps

1. **Week 1 (Current):** Implement DeviceManager::initialize() and tensor sync logic
2. **Week 2:** CUDA backend (highest priority, most common GPU)
3. **Week 3:** ROCm backend (AMD GPUs, growing market share)
4. **Week 4:** Multi-GPU orchestration and load balancing
5. **Week 5+:** Performance optimization (async streams, NCCL, graph optimization)

## Questions to Resolve

1. **Memory Management:** Should we use unified memory (cudaMallocManaged) or explicit transfers?
   - **Decision:** Start with explicit (more control), consider unified in Phase 5

2. **Stream Management:** One stream per device or multiple streams per device?
   - **Decision:** One stream initially, add multiple in Phase 5 for overlap

3. **Context Caching:** Cache contexts per device or create on-demand?
   - **Decision:** Cache (already in DeviceManager design)

4. **Error Handling:** Fallback to CPU on GPU OOM or fail hard?
   - **Decision:** Fail hard initially, add CPU fallback in Phase 5

---

**Ready to start:** Week 1 implementation  
**First task:** DeviceManager::initialize() in src/ComputeBackend.cpp
