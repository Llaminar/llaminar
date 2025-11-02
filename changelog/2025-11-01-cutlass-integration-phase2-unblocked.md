# CUTLASS Integration - Phase 2 Unblocked

**Date**: November 1, 2025  
**Session Type**: CUTLASS installation + CuTe API adoption  
**Status**: ✅ **UNBLOCKED** - Modern Tensor Core path available  

## Executive Summary

After discovering WMMA API is unavailable in CUDA 12.x, we've successfully integrated NVIDIA CUTLASS 4.2.1 and will use the **CuTe template API** for Phase 2 Tensor Core implementation.

## Actions Taken

### 1. CUTLASS Installation

**Installed CUTLASS 4.2.1** to `/opt/cutlass`:
```bash
sudo git clone --branch v4.2.1 --depth 1 \
  https://github.com/NVIDIA/cutlass.git /opt/cutlass
```

**Updated devcontainer Dockerfile**:
```dockerfile
# Install NVIDIA CUTLASS (Tensor Core library) - header-only, no build needed
# CUTLASS 4.2.1 supports CUDA 12.x and provides modern Tensor Core APIs
RUN git clone --branch v4.2.1 --depth 1 \
    https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    chown -R vscode:vscode /opt/cutlass

# Set CUTLASS environment variable for CMake
ENV CUTLASS_DIR=/opt/cutlass
```

### 2. CMake Integration

**Updated `src/v2/CMakeLists.txt`**:
```cmake
# Add CUTLASS support for Tensor Cores (Phase 2)
if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
else()
    set(CUTLASS_DIR "/opt/cutlass")
endif()

if(EXISTS "${CUTLASS_DIR}/include")
    message(STATUS "V2: Found CUTLASS at ${CUTLASS_DIR}")
    target_include_directories(cuda_backend PUBLIC "${CUTLASS_DIR}/include")
    target_compile_definitions(cuda_backend PUBLIC HAVE_CUTLASS)
    
    # CUTLASS requires C++17 minimum
    target_compile_features(cuda_backend PUBLIC cxx_std_17)
    
    # Add CUTLASS NVCC flags for Tensor Cores
    # SM 86 = RTX 3090 (Ampere), SM 80 = A100, SM 75 = Turing
    target_compile_options(cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>  # Turing
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=sm_80>  # Ampere A100
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>  # Ampere RTX 3090
    )
    
    message(STATUS "V2: CUTLASS Tensor Core support enabled (Phase 2)")
else()
    message(WARNING "V2: CUTLASS not found at ${CUTLASS_DIR} - Tensor Core kernels disabled")
endif()
```

**Re-enabled Tensor Core source file**:
```cmake
set(CUDA_KERNEL_SOURCES
    kernels/cuda/IQ4_NL_BlockDecoder.cu
    kernels/cuda/CudaGemmVariants.cu
    kernels/cuda/CudaGemmVariantsOptimized.cu
    kernels/cuda/CudaGemmVariantsTensorCore.cu  # Phase 2: Tensor Cores via CUTLASS
    kernels/cuda/CudaGemmAutoTuner.cu
    kernels/cuda/CudaGemmFactory.cu
)
```

## Why CuTe > WMMA

| Feature | WMMA (deprecated) | CuTe (modern) |
|---------|-------------------|---------------|
| **API Level** | Low-level (fragment management) | High-level (template abstractions) |
| **CUDA 12.x Support** | ❌ **Removed** | ✅ **Fully supported** |
| **Tile Configuration** | Manual 16×1616 only | Flexible (any power-of-2) |
| **Code Complexity** | ~300 lines of manual fragment handling | ~150 lines with templates |
| **Performance** | Good | **Optimal** (compiler-optimized) |
| **Learning Curve** | Moderate | Steep (but better documented) |
| **Future Support** | Deprecated | **Active development** |

## CuTe API Key Concepts

### 1. Tensor Core MMA Atoms

**SM80 (Ampere) Tensor Core operations:**
```cpp
using namespace cute;

// Define 16x8x16 Tensor Core operation (FP16 → FP16 with FP32 accumulation)
TiledMMA mmaC = make_tiled_mma(
    SM80_16x8x16_F16F16F16F16_TN{},     // MMA atom: 16×8×16 FP16 TensorOp
    Layout<Shape<_2,_2>>{},             // 2×2 atom tiling
    Tile<_32,_32,_16>{}                 // 32×32×16 final tile size
);
```

### 2. Async Copy (cp.async)

**Efficient global → shared memory transfer:**
```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_32,_8>>{},            // Thread layout
    Layout<Shape<_1,_8>>{}              // Value layout
);
```

### 3. Shared Memory Layout

**Software pipelining with 3-stage buffering:**
```cpp
auto bP = Int<3>{};  // 3-stage pipeline
auto sA = make_layout(make_shape(bM, bK, bP));  // (128, 16, 3)
auto sB = make_layout(make_shape(bN, bK, bP));  // (128, 16, 3)
```

### 4. Tensor Partitioning

**Automatic thread partitioning:**
```cpp
ThrMMA thr_mma = mmaC.get_slice(threadIdx.x);
Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));  // Register fragments
Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
Tensor tCrC = thr_mma.make_fragment_C(tCgC);           // Accumulator
```

## Phase 2 Implementation Plan (CuTe-based)

### Step 1: Adapt Dequantization for CuTe Tensors

Current approach loads quantized blocks into FP16 shared memory:
```cpp
// Load quantized B matrix → dequantize to FP16 shared memory
Tensor sB = make_tensor(make_smem_ptr(smem.B), sB_layout);

// Decoder fills sB with dequantized FP16 values
decoder.decode_block_fp16(block, &sB[...]);
```

### Step 2: Define CuTe Kernel Template

```cpp
template <class Decoder, int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
__global__ void cute_quantized_gemm_kernel(
    const float* A,              // [m × k] FP32 input
    const IQ4_NLBlock* B_blocks, // [n × k/32] quantized weights
    float* C,                    // [m × n] FP32 output
    int m, int n, int k
) {
    using namespace cute;
    
    // Define problem shape
    auto prob_shape = make_shape(m, n, k);
    
    // Define Tensor Core MMA
    TiledMMA mmaC = make_tiled_mma(
        SM80_16x8x16_F16F16F32F32_TN{},  // FP16→FP32 accumulation
        Layout<Shape<_2,_2>>{},
        Tile<TILE_M, TILE_N, TILE_K>{}
    );
    
    // Async copy for A (FP32 → FP16 shared memory)
    TiledCopy copyA = make_tiled_copy(...);
    
    // Custom dequant+copy for B (quantized → FP16 shared memory)
    // ... dequantization logic here ...
    
    // Tensor Core compute loop
    gemm_compute_loop(mmaC, tCrA, tCrB, tCrC);
}
```

### Step 3: Performance Targets

Using CuTe optimizations (3-stage pipelining, async copy, optimized layouts):
- **Phase 1 baseline**: 425 GFLOPS (optimized FP32)
- **Phase 2 target**: **1,700-2,550 GFLOPS** (4-6× speedup)
  - Tensor Cores: 3-4× (FP16 compute)
  - Async copy + pipelining: 1.5×

**Why higher than original 3-4× target?**
- CuTe's async copy eliminates memory bottlenecks
- 3-stage pipelining hides latency
- Optimized layouts improve shared memory bandwidth

## Next Steps

1. ✅ CUTLASS installed and integrated
2. ✅ CMake configured with CUTLASS paths
3. ⏳ **Implement CuTe-based quantized GEMM kernel** (next session)
4. ⏳ Create Phase 2 performance test
5. ⏳ Validate correctness and performance

## Files Modified

1. **`.devcontainer/Dockerfile`**:
   - Added CUTLASS 4.2.1 installation
   - Set `CUTLASS_DIR` environment variable

2. **`src/v2/CMakeLists.txt`**:
   - Added CUTLASS include paths
   - Defined `HAVE_CUTLASS` compile flag
   - Added SM 75/80/86 NVCC architecture flags
   - Re-enabled `CudaGemmVariantsTensorCore.cu`

3. **`PHASE2_BLOCKED.md`**:
   - Updated to reflect CUTLASS solution

## References

- **CUTLASS GitHub**: https://github.com/NVIDIA/cutlass
- **CUTLASS Documentation**: https://docs.nvidia.com/cutlass/
- **CuTe Tutorial**: `/opt/cutlass/examples/cute/tutorial/`
- **SM80 GEMM Example**: `/opt/cutlass/examples/cute/tutorial/sgemm_sm80.cu`

## Session Metrics

- **Time Spent**: ~1 hour (installation + CMake integration)
- **Files Modified**: 3 files
- **CUTLASS Size**: 31.17 MiB (header-only library)
- **Build Status**: ✅ Compiles successfully with CUTLASS support

---

**Status**: Phase 2 **UNBLOCKED** - CuTe API provides modern Tensor Core path  
**Next Target**: Implement CuTe-based quantized GEMM kernel (4-6× speedup)
