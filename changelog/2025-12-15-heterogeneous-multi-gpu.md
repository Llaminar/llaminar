# Heterogeneous Multi-GPU Support: CUDA + ROCm in Same Binary

**Date**: 2025-12-15  
**Phase**: 6 (Multi-GPU) - Foundation

## Summary

Enabled simultaneous CUDA and ROCm support in a single binary, allowing Llaminar to enumerate and use NVIDIA and AMD GPUs together. This is the foundation for Phase 6 multi-GPU tensor parallelism.

## Hardware Validated

- **NVIDIA GeForce RTX 3090** (24 GB, SM 8.6, Ampere)
- **2× AMD Instinct MI50** (32 GB each, gfx906)
- Total: 3 GPUs + CPU, all visible in single process

## Changes

### CMakeLists.txt

1. **Removed Mutual Exclusion**: Deleted `FATAL_ERROR` that prevented both `HAVE_CUDA` and `HAVE_ROCM` being enabled
2. **Updated CUDA Architectures**: Changed from `sm_70` (Volta) to `sm_75`+ (Turing and newer) for CUDA 13.0 compatibility
3. **Added HIP Language Support**: 
   ```cmake
   enable_language(HIP)
   set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx1030;gfx1100")
   ```
4. **Fixed hipblas Linking**: Changed from `hipblas` to `roc::hipblas` target for proper library path resolution

### New Files

- **`backends/CUDAEnumeration.cu`**: CUDA device enumeration in isolated compilation unit
  - `enumerate_cuda_devices()` - Returns vector of ComputeDevice
  - `get_cuda_device_numa_node()` - NUMA affinity detection
  
- **`backends/ROCmEnumeration.cpp`**: ROCm device enumeration (compiled with HIP)
  - `enumerate_rocm_devices()` - Returns vector of ComputeDevice
  - `get_rocm_device_numa_node()` - NUMA affinity detection
  
- **`backends/GPUEnumeration.h`**: Extern declarations for both enumeration functions

### Modified Files

- **`backends/ComputeBackend.cpp`**: Removed inline CUDA/HIP code, now calls extern functions from GPUEnumeration.h
- **`utils/NUMATopology.cpp`**: Added mutual exclusion guards for GPU headers to prevent conflicts when both backends enabled

## Technical Challenge: Header Conflicts

CUDA and HIP headers define conflicting types (e.g., `dim3`, `longlong4`) that cause "redefinition" errors when included in the same translation unit. 

**Solution**: Separate compilation units:
- `.cu` files only include CUDA headers (compiled with nvcc)
- ROCm `.cpp` files only include HIP headers (compiled with hipcc via `LANGUAGE HIP`)
- Common code uses extern declarations

## Device Enumeration Output

```
[DeviceManager] Enumerated 4 device(s):
  [0] CPU (376 GB, NUMA node 0)
  [1] NVIDIA GeForce RTX 3090 (23 GB, SM 8.6, NUMA node 0)
  [2] AMD Instinct MI60 / MI50 (gfx906) (31 GB, NUMA node 0)
  [3] AMD Instinct MI60 / MI50 (gfx906) (31 GB, NUMA node 0)
```

## Build Command

```bash
cmake -B build_v2_multigpu -S src/v2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DHAVE_CUDA=ON \
    -DHAVE_ROCM=ON

cmake --build build_v2_multigpu --parallel
```

## Test Results

- **Unit Tests**: 149/149 passed ✅
- **Integration Tests**: 44/44 passed ✅

## Next Steps (Phase 6 Remaining)

1. **Multi-GPU DeviceManager**: Layer-wise GPU assignment
2. **Cross-device Memory Transfers**: GPU↔GPU, GPU↔CPU data movement
3. **GPU-aware Weight Placement**: Distribute model layers across GPUs
4. **Pipeline Modifications**: Use multiple GPUs during inference

## Known Limitations

- NUMA detection uses fallback when both CUDA and ROCm enabled (sysfs-only)
- SM version for AMD GPUs shown as "90.6" (placeholder, could be improved)
