# Llaminar v2 - Clean greenfield implementation
cmake_minimum_required(VERSION 3.18)

# Define custom build types BEFORE project() so they're recognized
# Define custom build type: E2ERelease (Release + Snapshots for E2E testing)
# Usage: cmake -DCMAKE_BUILD_TYPE=E2ERelease
set(CMAKE_CXX_FLAGS_E2ERELEASE "-O3 -DNDEBUG" CACHE STRING "Flags for E2E Release builds")
set(CMAKE_C_FLAGS_E2ERELEASE "-O3 -DNDEBUG" CACHE STRING "Flags for E2E Release builds")
set(CMAKE_EXE_LINKER_FLAGS_E2ERELEASE "" CACHE STRING "Linker flags for E2E Release builds")
mark_as_advanced(CMAKE_CXX_FLAGS_E2ERELEASE CMAKE_C_FLAGS_E2ERELEASE CMAKE_EXE_LINKER_FLAGS_E2ERELEASE)

# Define custom build type: Integration (Release + Snapshots + Debug symbols + Assertions)
# Usage: cmake -DCMAKE_BUILD_TYPE=Integration
# This enables optimized code with snapshot capture, backtraces for debugging, AND assertions.
# Note: NDEBUG is still set to get Release-like behavior from stdlib, but LLAMINAR_ENABLE_ASSERTIONS
# keeps our custom assertions active for integration testing.
set(CMAKE_CXX_FLAGS_INTEGRATION "-O3 -g -DNDEBUG -DLLAMINAR_ENABLE_ASSERTIONS" CACHE STRING "Flags for Integration builds (Release + debug symbols + assertions)")
set(CMAKE_C_FLAGS_INTEGRATION "-O3 -g -DNDEBUG -DLLAMINAR_ENABLE_ASSERTIONS" CACHE STRING "Flags for Integration builds (Release + debug symbols + assertions)")
set(CMAKE_EXE_LINKER_FLAGS_INTEGRATION "" CACHE STRING "Linker flags for Integration builds")
mark_as_advanced(CMAKE_CXX_FLAGS_INTEGRATION CMAKE_C_FLAGS_INTEGRATION CMAKE_EXE_LINKER_FLAGS_INTEGRATION)

# Add custom build types to valid build types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;E2ERelease;Integration" CACHE STRING "" FORCE)

# Project declaration - MUST come after custom build type definitions
# The build type flags are picked up by the compiler setup during project()
project(llaminar2 LANGUAGES C CXX)

# =============================================================================
# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "V2: Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    # Also cache nvcc and hipcc TUs — these are the slowest in the build
    # (multi-arch fatbin + ROCm headers). Ccache 4.0+ supports both.
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_HIP_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "V2: ccache not found - using direct compilation")
endif()

# =============================================================================
# Code Coverage Support (Debug builds only)
# =============================================================================
# When coverage flags are detected in CMAKE_CXX_FLAGS_DEBUG, ensure gcov
# library is linked to all targets. This is required because --coverage
# generates profiling calls that need libgcov at link time.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_FLAGS_DEBUG MATCHES "--coverage")
        message(STATUS "V2: Code coverage enabled - linking gcov library")
        link_libraries(gcov)
    endif()
endif()

# Enable native CPU optimizations (AVX512/AVX2/FMA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")

# Backend options (Phase 6: Heterogeneous Multi-GPU)
# CUDA and ROCm CAN coexist - compiled in separate translation units
# Enable both for mixed NVIDIA + AMD GPU systems
# DEFAULT: ON - most development machines have at least one GPU type
option(HAVE_CUDA "Enable CUDA backend" ON)
option(HAVE_ROCM "Enable ROCm backend" ON)
option(HAVE_VULKAN "Enable Vulkan backend" OFF)

# ML Heuristic auto-retraining option
# When enabled, automatically checks CSV hash and retrains ML model on every build
# Only retrains if benchmark data actually changed (hash-based, fast check)
# DEFAULT: OFF (run manually to avoid long builds)
option(AUTO_RETRAIN_ML_HEURISTIC "Automatically retrain ML heuristic if CSV changed" OFF)

# ONNX Runtime option for neural network heuristic
option(USE_ONNX_HEURISTIC "Enable ONNX Runtime for neural network heuristic" ON)

# OneDNN option for optimized INT8 GEMM
option(USE_ONEDNN "Enable OneDNN for INT8 GEMM acceleration" ON)

# Attention input dumping (for debugging JIT kernel issues)
# When enabled, dumps Q/K/V/mask inputs to disk for each JIT attention invocation
option(DUMP_ATTENTION_INPUTS "Dump attention kernel inputs to disk for debugging" OFF)

# =============================================================================
# GPU Backend Configuration (Phase 6: Multi-GPU)
# =============================================================================
# CUDA and ROCm CAN coexist in the same binary when:
#   1. CUDA code is in separate .cu files (compiled by nvcc)
#   2. ROCm code is in separate .cpp files (compiled by hipcc)
#   3. Common code uses abstract IBackend interface (no GPU headers)
#
# This enables heterogeneous multi-GPU: NVIDIA + AMD in the same system
if(HAVE_CUDA AND HAVE_ROCM)
    message(STATUS "V2: Both CUDA and ROCm enabled (heterogeneous multi-GPU)")
    message(STATUS "V2: NVIDIA GPUs via CUDA, AMD GPUs via ROCm")
endif()

# =============================================================================
# CUDA Backend Configuration
# =============================================================================
if(HAVE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "V2: CUDA enabled")
    message(STATUS "V2: CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "V2: CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "V2: nvcc: ${CMAKE_CUDA_COMPILER}")
    
    # ONNX Runtime for neural network heuristic
    if(USE_ONNX_HEURISTIC)
        # Look for ONNX Runtime in /opt/onnxruntime (manual install)
        set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
        
        if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
            message(STATUS "V2: ONNX Runtime found at ${ONNXRUNTIME_ROOT}")
            message(STATUS "V2: Neural network heuristic enabled (Test R²=0.96)")
            
            set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
            set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
            
            add_compile_definitions(HAVE_ONNX_RUNTIME)
        else()
            message(WARNING "V2: ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
            message(WARNING "V2: Neural network heuristic disabled")
            message(WARNING "V2: Download from: https://github.com/microsoft/onnxruntime/releases")
            message(WARNING "V2: Extract to: /opt/onnxruntime")
            set(USE_ONNX_HEURISTIC OFF)
        endif()
    endif()
    
    # CUDA compile flags (CUDA 13.0+ dropped sm_70 Volta support)
    # RTX 3090 = sm_86 (Ampere)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")  # Base: Turing
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")  # Turing (RTX 20xx)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")  # Ampere (A100)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")  # Ampere (RTX 30xx)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")  # Ada (RTX 40xx)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90")  # Hopper (H100)
    
    # Enable nvcc multi-threaded compilation (auto-detect optimal thread count)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 0")
    
    # CUDA optimization flags for Release builds
    # --use_fast_math: Enable fast math intrinsics (~10-30% speedup, slight precision loss)
    # --ftz=true: Flush denormals to zero (faster, no denormals in LLM inference anyway)
    # --prec-div=false: Fast division (slightly less accurate, negligible for inference)
    # --prec-sqrt=false: Fast sqrt (slightly less accurate)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math --ftz=true --prec-div=false --prec-sqrt=false")
    
    # NOTE: CUDA_SEPARABLE_COMPILATION is set per-target (on cuda_backend and
    # device_benchmarks) rather than globally. Combined with CUDA_RESOLVE_DEVICE_SYMBOLS,
    # this resolves device symbols at library build time. Without this, every downstream
    # executable (580+ test binaries) gets its own nvcc device-link step, adding ~19MB
    # per executable and massive link time. With per-library resolution, only the 3 GPU
    # libraries need device linking — executables link normally.
    #
    # The nvlink_pool job limit is no longer needed (3 library device links vs 580+
    # executable device links), but we keep a small pool as a safety net.
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set_property(GLOBAL APPEND PROPERTY JOB_POOLS nvlink_pool=8)
    endif()
endif()

# =============================================================================
# ROCm Backend Configuration
# =============================================================================
if(HAVE_ROCM)
    # Set ROCm path (standard install location)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation path")
    endif()
    
    # Enable HIP language for ROCm compilation
    enable_language(HIP)
    
    # Find HIP
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    
    message(STATUS "V2: ROCm enabled")
    message(STATUS "V2: ROCm Path: ${ROCM_PATH}")
    message(STATUS "V2: HIP Version: ${hip_VERSION}")
    message(STATUS "V2: hipcc: ${HIP_HIPCC_EXECUTABLE}")
    
    # ROCm compile flags - only architectures supporting DL (Deep Learning) instructions
    # CK requires specific arch macros: gfx906, gfx908, gfx90a (MI50/MI100/MI200)
    # NOT supported: gfx900 (no DL), gfx940 (CK expects gfx942), gfx1030/gfx1100 (RDNA)
    # Currently targeting only gfx906 (MI50) for faster compilation
    set(CMAKE_HIP_ARCHITECTURES "gfx906")
    # Note: hipcc's -parallel-jobs conflicts with CMake's --parallel, so we rely
    # on CMake's build parallelism instead (multiple hipcc processes in parallel)
endif()

# =============================================================================
# Backend Library Compilation (Phase 3)
# =============================================================================

# Find Python3 for code generation scripts (needed before backend compilation)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "V2: Found Python3: ${Python3_EXECUTABLE}")

# CUDA backend (separate .cu compilation unit)
if(HAVE_CUDA)
    # JIT compilation architecture - no precompiled variants needed
    # Kernels compiled on-demand at runtime via NVRTC
    # Benefits:
    # - Build time: 25 minutes → 30 seconds (50× faster)
    # - Binary size: 1GB → 10MB (100× smaller)
    # - Unlimited dynamic configs (vs 37,380 fixed)
    # - GPU-adaptive (auto-compiles for target architecture)
    
    # CUDA kernels
    set(CUDA_KERNEL_SOURCES
        kernels/cuda/gemm/CuBLASGemmKernel.cu              # cuBLAS wrapper for FP32/FP16/BF16 dense GEMM
        kernels/cuda/gemm/CUDAQuantisedGemmKernel_CUTLASS.cu  # CUTLASS INT8 GEMM + TC weight repacking
        kernels/cuda/gemm/CUDANativeVNNIIQTables.cu    # shared IQ lookup table symbol definitions for native-vnni decode
        kernels/cuda/gemm/CUDANativeVNNIGemvTuned.cu  # tuned shape-specific native-vnni GEMV (Wide/KPar/Direct)
        kernels/cuda/gemm/CUDANativeVNNIPrefillKernels.cu  # custom native-vnni prefill GEMM family (Q4_0 first)
        kernels/cuda/gemm/CUDAcuBLASQuantGemm.cu          # cuBLAS FP16 GEMM path for quantized weights (env: LLAMINAR_CUBLAS_GEMM=1)
        # Ops kernels (split from monolithic CUDAOpsKernels.cu)
        kernels/cuda/ops/CUDARMSNormKernels.cu        # RMSNorm kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDAResidualAddKernels.cu    # Residual Add kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDASwiGLUKernels.cu         # SwiGLU kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDAFusedOpsKernels.cu       # Fused ops: SwiGLU+Quant, ResidualAdd+RMSNorm
        kernels/cuda/ops/CUDARoPEKernels.cu           # RoPE kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDAEmbeddingKernels.cu      # Embedding lookup kernel
        kernels/cuda/ops/CUDAVectorAddKernels.cu      # Vector add kernels (for collective reductions)
        kernels/cuda/ops/CUDASamplingKernels.cu        # Argmax + Top-K kernels for GPU-side sampling
        kernels/cuda/ops/CUDACastKernels.cu            # FP32↔FP16 cast kernels for mixed-precision allreduce
        kernels/cuda/attention/CUDAFlashAttentionKernels.cu  # Flash Attention 2 and Flash Decoding kernels
        kernels/cuda/kvcache/CUDARingKVCache.cu       # Ring buffer KV cache with O(1) eviction
        kernels/cuda/kvcache/CUDATurboQuantKernels.cu  # TurboQuant quantize/dequant/RoPE GPU kernels
        kernels/cuda/kvcache/CUDARingKVCacheTQ.cu     # TurboQuant KV cache (TQ8-K/TQ4-V split)
        # GDN (Gated Delta Net) kernels for Qwen 3.5
        kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu  # GDN recurrence, conv1d, GatedRMSNorm, OutputGate
        # Profiling utilities
        utils/CUDAKernelProfiler.cu                   # CUDA event-based kernel profiling
    )
    
    # NOTE: CUDA kernel ITensorGemm adapters (CUDAFloatingPointGemmKernel.cpp, CUDAQuantisedGemmKernel.cpp)
    # are added to llaminar2_core with target_sources() below, compiled with regular C++ compiler (NOT nvcc)
    # because they include MPI headers which nvcc cannot process
    
    # CUDA backend library includes backend + kernels + enumeration + NCCLBackend
    # 
    # CRITICAL: NCCLBackend MUST be in cuda_backend (not llaminar2_core) to isolate
    # NCCL symbols from RCCL. Both NCCL and RCCL export identical symbols (ncclCommInitRank,
    # ncclAllReduce, etc.). If both are linked to llaminar2_core, whichever loads first
    # will shadow the other's symbols, causing AMD code to call NVIDIA's NCCL or vice versa.
    #
    # Note: NCCLBackendCUDA.cu isolates CUDA runtime calls from NCCLBackend.cpp
    # Note: HostBackendCUDA.cu isolates CUDA runtime calls from HostBackend.cpp
    # Note: CUDARingKVCacheTensorAdapter.cpp provides ITensor interface implementations for
    # CUDARingKVCache and must be in cuda_backend to satisfy template instantiation dependencies
    # Note: NvidiaContextFactory.cu registers the CUDA factory with GPUDeviceContextPool
    # Note: CUDAWorkerImpl.cu provides cuda_worker::initializeContext/cleanupContext for GPUDeviceWorker
    add_library(cuda_backend STATIC 
        backends/cuda/CUDABackend.cu
        backends/cuda/CUDATensorValidation.cu
        backends/cuda/NvidiaDeviceContext.cu
        backends/cuda/NvidiaContextFactory.cu
        backends/cuda/CUDAWorkerImpl.cu
        backends/cuda/CUDAGraphCapture.cu
        backends/CUDAEnumeration.cu
        collective/backends/NCCLBackendCUDA.cu
        collective/backends/HostBackendCUDA.cu
        collective/coordinators/NCCLCoordinator.cu
        kernels/cuda/kvcache/CUDARingKVCacheTensorAdapter.cpp
        kernels/cuda/kvcache/CUDARingKVCacheBase.cpp
        ${CUDA_KERNEL_SOURCES}
    )
    # Resolve CUDA device symbols at library build time so downstream executables
    # (500+ test binaries) don't each need their own nvcc device-link step.
    set_target_properties(cuda_backend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set_property(TARGET cuda_backend PROPERTY JOB_POOL_LINK nvlink_pool)
    endif()
    target_include_directories(cuda_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    
    # NCCLBackend.cpp must be compiled with g++ (not nvcc) because it includes MPI headers.
    # NCCLDynamicLoader.cpp must also be compiled with g++ as it uses dlopen/dlsym.
    # Add NCCLBackend sources as separate with regular C++ compilation.
    target_sources(cuda_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/collective/backends/NCCLBackend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/collective/backends/NCCLDynamicLoader.cpp
    )
    # Set NCCLBackend source files to be compiled with C++ compiler, not nvcc
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/collective/backends/NCCLBackend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/collective/backends/NCCLDynamicLoader.cpp
        PROPERTIES LANGUAGE CXX
    )
    
    # Link CUDA libraries:
    # - cudart: CUDA Runtime API (for memory management, streams, etc.)
    # - nvrtc: NVRTC runtime compiler (for JIT kernel compilation)
    # - cuda: CUDA Driver API (for cuLaunchKernel, cuModuleLoad, etc.)
    # - cublas: cuBLAS library for dense GEMM (FP32/FP16/BF16)
    # - cublasLt: cuBLAS Light for fused GEMM+bias operations
    target_link_libraries(cuda_backend PUBLIC 
        CUDA::cudart
        CUDA::nvrtc
        CUDA::cuda_driver
        CUDA::cublas
        CUDA::cublasLt
    )
    
    # Link MPI for NCCLBackend.cpp (which uses MPI_Comm_rank for distributed communication)
    target_link_libraries(cuda_backend PUBLIC MPI::MPI_CXX)
    
    # Link ONNX Runtime if enabled
    if(USE_ONNX_HEURISTIC AND ONNXRUNTIME_INCLUDE_DIR)
        target_include_directories(cuda_backend PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(cuda_backend PRIVATE ${ONNXRUNTIME_LIB})
        message(STATUS "V2: CUDA backend linked with ONNX Runtime")
    endif()
    
    # Hash-based automatic ML heuristic retraining
    # Checks if CSV changed and retrains model before building CUDA backend
    # Only retrains if benchmark data actually changed (hash-based, <1 sec overhead)
    # Disable with: cmake -DAUTO_RETRAIN_ML_HEURISTIC=OFF
    if(AUTO_RETRAIN_ML_HEURISTIC)
        message(STATUS "V2: Auto-retrain ML heuristic enabled (hash-based check on every build)")
        # Script path: go up 2 levels from src/v2 to workspace root
        set(WORKSPACE_ROOT "${PROJECT_SOURCE_DIR}/../..")
        add_custom_command(
            TARGET cuda_backend PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "[Auto-Retrain] Checking CUDA GEMM heuristic..."
            COMMAND bash ${WORKSPACE_ROOT}/scripts/check_cuda_heuristic_needs_retrain.sh && 
                    ${CMAKE_COMMAND} -E echo "[Auto-Retrain] CSV hash changed, retraining ML model..." &&
                    bash ${WORKSPACE_ROOT}/scripts/train_cuda_heuristic.sh || 
                    ${CMAKE_COMMAND} -E echo "[Auto-Retrain] Heuristic weights up-to-date (hash unchanged)"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Auto-checking CUDA GEMM heuristic (hash-based)"
            VERBATIM
        )
    else()
        message(STATUS "V2: Auto-retrain ML heuristic disabled (manual training only)")
    endif()
    
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
        
        # CUTLASS requires C++17 minimum, we use C++23 for stacktrace
        target_compile_features(cuda_backend PUBLIC cxx_std_23)
        
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
    
    # Define HAVE_CUDA for MPIStager and other consumers
    add_compile_definitions(HAVE_CUDA)
    
    message(STATUS "V2: CUDA backend library configured with ${CMAKE_CURRENT_LIST_LENGTH} kernel(s)")
    
    # =========================================================================
    # NCCL Detection (NVIDIA Collective Communications Library)
    # =========================================================================
    # NCCL provides optimized multi-GPU and multi-node collective operations.
    # Used by NCCLBackend for AllReduce, AllGather, etc. in tensor parallelism.
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CUDA_TOOLKIT_ROOT_DIR}/include
        PATHS
            /usr/include
            /usr/local/include
            /opt/nccl/include
            /usr/local/cuda/include
    )
    find_library(NCCL_LIBRARY nccl
        HINTS
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib
        PATHS
            /usr/lib/x86_64-linux-gnu
            /usr/lib64
            /usr/local/lib
            /opt/nccl/lib
            /usr/local/cuda/lib64
    )
    
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        set(HAVE_NCCL ON CACHE BOOL "NCCL library found")
        add_compile_definitions(HAVE_NCCL)
        message(STATUS "V2: NCCL found")
        message(STATUS "V2:   Include: ${NCCL_INCLUDE_DIR}")
        message(STATUS "V2:   Library: ${NCCL_LIBRARY}")
        
        # CRITICAL: NCCL is now loaded dynamically via dlopen() in NCCLDynamicLoader.cpp
        # to avoid symbol conflicts with RCCL, which exports identical symbol names.
        # We only need the headers for NCCLBackendCUDA.cu (it doesn't actually link NCCL anymore).
        # NCCLDynamicLoader loads NCCL at runtime with RTLD_LOCAL to isolate symbols.
        target_include_directories(cuda_backend PUBLIC ${NCCL_INCLUDE_DIR})
        target_compile_definitions(cuda_backend PUBLIC HAVE_NCCL)
        # Link dl for dlopen/dlsym
        target_link_libraries(cuda_backend PUBLIC ${CMAKE_DL_LIBS})
        message(STATUS "V2: NCCL will be loaded dynamically (dlopen) from cuda_backend for symbol isolation")
    else()
        set(HAVE_NCCL OFF CACHE BOOL "NCCL library not found")
        message(WARNING "V2: NCCL not found - NCCLBackend will be disabled")
        if(NOT NCCL_INCLUDE_DIR)
            message(WARNING "V2:   nccl.h not found in standard paths")
        endif()
        if(NOT NCCL_LIBRARY)
            message(WARNING "V2:   libnccl.so not found in standard paths")
        endif()
    endif()
endif()

# amdgpu DKMS Auto-Clone for P2P Fix — REMOVED
# =============================================================================
# Previously cloned dbsanfte/amdgpu@fix/pex-pcie-switch-p2p-enable into
# external/amdgpu so the host driver could be rebuilt with a P2P-through-
# PCIe-switch patch. The build never compiled or linked anything from that
# tree — kernel-driver rebuild is a host-side concern. The clone added
# ~1 minute and ~80 MB to every fresh build for no in-tree benefit.

# ROCm backend (separate .cpp compilation unit with HIP)
if(HAVE_ROCM)
    # ROCm kernels compiled with hipcc
    set(ROCM_KERNEL_SOURCES
        kernels/rocm/gemm/ROCmQuantisedGemmKernel.hip           # Common INT8 GEMM utilities (quantization, scaling, memory)
        kernels/rocm/gemm/ROCmQuantisedGemmKernel_INT8_VNNI.hip # INT8 VNNI wide-tile V3+V7 kernels (gfx906)
        kernels/rocm/gemm/ROCmGemvKernel.hip               # GEMV kernel suite documentation index (split into files below)
        kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip    # INT8 VNNI GEMV/GEMM kernels for decode + prefill (gfx906)
        kernels/rocm/gemm/ROCmGemvKernel_native_VNNI.hip  # Native-VNNI 4-bit (Q4_0/IQ4_NL) GEMV kernels (gfx906)
        kernels/rocm/gemm/ROCmQuantisedGemmKernel_native_VNNI.hip  # Native-VNNI 4-bit (Q4_0/IQ4_NL) GEMM kernels for M>1 prefill (gfx906)
        kernels/rocm/attention/ROCmFlashAttentionKernels.hip  # Flash Attention 2 + Flash Decoding for MI50
        kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip       # Ring buffer KV cache kernels
        kernels/rocm/kvcache/ROCmTurboQuantKernels.hip        # TurboQuant quantize/dequant/RoPE GPU kernels
        kernels/rocm/kvcache/ROCmRingKVCacheTQ.hip            # TurboQuant KV cache (TQ8-K/TQ4-V split)
        kernels/rocm/ops/ROCmSwiGLUKernels.hip               # SwiGLU kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmRoPEKernels.hip                 # RoPE kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmRMSNormKernels.hip              # RMSNorm kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmEmbeddingKernels.hip            # Embedding kernels (FP32, BF16, FP16, Q8_1)
        kernels/rocm/ops/ROCmResidualAddKernels.hip          # Residual Add kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmArgmaxKernels.hip               # Argmax kernel for GPU-side greedy sampling
        kernels/rocm/ops/ROCmSamplingKernels.hip              # Top-K kernel for GPU-side top-k/top-p sampling
        kernels/rocm/ops/ROCmFusedOpsKernels.hip              # Fused Residual+RMSNorm, Fused SwiGLU+Quant kernels
        kernels/rocm/ops/ROCmCastKernels.hip                  # FP32↔FP16 cast kernels for mixed-precision allreduce
        # GDN (Gated Delta Net) kernels for Qwen 3.5
        kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip        # GDN recurrence, conv1d, GatedRMSNorm, OutputGate
        # Profiling utilities
        utils/ROCmKernelProfiler.hip                         # ROCm/HIP event-based kernel profiling
    )
    
    # ROCm enumeration and backend must be compiled with hipcc to use HIP runtime
    # CRITICAL: RCCLBackend MUST be in rocm_backend (not llaminar2_core) to isolate
    # RCCL symbols from NCCL. Both libraries export the same symbol names
    # (ncclCommInitRank, etc.). If both are linked to the same target, only one
    # set resolves, causing ROCm code to accidentally call NVIDIA NCCL.
    #
    # CRITICAL: RCCL is now loaded dynamically via dlopen() in RCCLDynamicLoader.cpp
    # to avoid symbol conflicts with NCCL, which exports identical symbol names.
    # RCCLBackendHIP.cpp isolates HIP runtime calls and uses the dynamic loader.
    # Note: AMDContextFactory.cpp registers the ROCm factory with GPUDeviceContextPool
    # Note: ROCmWorkerImpl.cpp provides rocm_worker::initializeContext/cleanupContext for GPUDeviceWorker
    add_library(rocm_backend STATIC 
        backends/rocm/ROCmBackend.cpp
        backends/rocm/ROCmTensorValidation.cpp
        backends/rocm/AMDDeviceContext.cpp
        backends/rocm/AMDContextFactory.cpp
        backends/rocm/ROCmWorkerImpl.cpp
        backends/rocm/HipDeviceGuard.cpp
        backends/rocm/HIPGraphCapture.cpp
        backends/ROCmEnumeration.cpp
        collective/backends/HostBackendROCm.cpp              # HIP-specific host memory registration
        collective/backends/RCCLBackend.cpp                  # RCCL backend (MUST be here, not in llaminar2_core)
        collective/backends/RCCLDynamicLoader.cpp            # Dynamic loader for RCCL (dlopen/dlsym)
        collective/backends/RCCLBackendHIP.cpp               # HIP runtime wrappers for RCCL backend
        collective/coordinators/RCCLCoordinator.cpp          # RCCL collective coordinator
        ${ROCM_KERNEL_SOURCES}
    )
    # Resolve HIP device symbols at library build time (same rationale as cuda_backend).
    set_target_properties(rocm_backend PROPERTIES
        HIP_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_include_directories(rocm_backend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ROCM_PATH}/include
        ${ROCM_PATH}/hip/include
        ${ROCM_PATH}/hipblas/include
        ${ROCM_PATH}/hipblaslt/include
    )
    target_link_libraries(rocm_backend PUBLIC
        hip::host                                # HIP runtime
        roc::hipblas                             # hipBLAS
        ${ROCM_PATH}/lib/libhipblaslt.so         # hipBLASLt for fused GEMM+bias
    )
    
    # RCCLBackend.cpp needs MPI for MPIContext
    target_link_libraries(rocm_backend PUBLIC MPI::MPI_CXX)
    
    # Link dl for dlopen/dlsym (RCCLDynamicLoader uses dlopen to load RCCL dynamically)
    target_link_libraries(rocm_backend PUBLIC ${CMAKE_DL_LIBS})
    
    # CRITICAL: RCCL is now loaded dynamically via dlopen() instead of direct linking
    # This avoids symbol conflicts with NCCL - both libraries export identical symbol
    # names (ncclAllReduce, ncclCommInitRank, etc.). Dynamic loading with RTLD_LOCAL
    # keeps symbols in isolated namespaces.
    if(HAVE_RCCL)
        # Only need RCCL headers (if available) - library is loaded at runtime
        if(RCCL_INCLUDE_DIR)
            target_include_directories(rocm_backend PUBLIC ${RCCL_INCLUDE_DIR})
        endif()
        target_compile_definitions(rocm_backend PUBLIC HAVE_RCCL)
        message(STATUS "V2: RCCL will be loaded dynamically (dlopen) from rocm_backend for symbol isolation")
    endif()
    
    # Set HIP compiler for ROCm source files
    # NOTE: RCCLBackend.cpp, RCCLDynamicLoader.cpp, and RCCLBackendHIP.cpp are NOT
    # compiled with HIP because:
    # - RCCLBackend.cpp needs MPI headers which hipcc cannot process
    # - RCCLDynamicLoader.cpp uses dlopen/dlsym
    # - RCCLBackendHIP.cpp uses HIP runtime API via standard C++ includes, but
    #   is compiled with HIP to access hip/hip_runtime.h
    set_source_files_properties(
        backends/rocm/ROCmBackend.cpp
        backends/rocm/ROCmTensorValidation.cpp
        backends/ROCmEnumeration.cpp
        collective/backends/HostBackendROCm.cpp
        collective/backends/RCCLBackendHIP.cpp
        ${ROCM_KERNEL_SOURCES}
        PROPERTIES LANGUAGE HIP
    )
    
    # RCCLBackend.cpp and RCCLDynamicLoader.cpp must be compiled with regular C++ compiler
    set_source_files_properties(
        collective/backends/RCCLBackend.cpp
        collective/backends/RCCLDynamicLoader.cpp
        PROPERTIES LANGUAGE CXX
    )
    
    # =============================================================================
    # CRITICAL: Force full optimization for ALL HIP GPU kernel code
    # =============================================================================
    # GPU kernels without -O3 are pathologically slow (100-30000× slower than optimized).
    # The optimizer is essential for:
    #   1. Inlining template call chains (especially CK's deep template metaprogramming)
    #   2. Eliminating dead code paths
    #   3. Constant folding tile sizes, loop bounds, and kernel parameters
    #   4. Register allocation and instruction scheduling for GPU wavefronts
    #
    # Debug builds: Host code remains debuggable, but GPU kernel code is optimized.
    # This is the standard pattern for CUDA/HIP - GPU code MUST be optimized to function.
    #
    # Measured impact (CK INT8 GEMM M=8, N=896, K=896):
    #   - Without -O3: ~1800-2600 ms  (unusable)
    #   - With -O3:    ~0.06 ms       (30,000× faster)
    set_source_files_properties(
        ${ROCM_KERNEL_SOURCES}
        PROPERTIES COMPILE_FLAGS "-O3 -DNDEBUG"
    )
    
    # NOTE: ROCm kernel ITensorGemm adapters (ROCmFloatingPointGemmKernel.cpp)
    # are added to llaminar2_core with target_sources() below, compiled with regular C++ compiler (NOT hipcc)
    # because they include MPI headers which hipcc cannot process
    
    # Define HAVE_ROCM for MPIStager and other consumers
    add_compile_definitions(HAVE_ROCM)
    
    message(STATUS "V2: ROCm backend library configured with hipBLAS GEMM kernels")
    
    # =========================================================================
    # RCCL Auto-Clone and Build for ROCm Multi-GPU Support
    # =========================================================================
    # The Ubuntu-packaged RCCL (rccl 2.27.7.70101-38~24.04) has broken gfx906
    # (MI50/MI60) kernel binaries. When using multi-GPU on these architectures,
    # the library crashes with SIGSEGV in ihipLaunchKernel() because the GPU
    # kernels fail to load (hipErrorNoBinaryForGpu).
    #
    # Workaround: Build RCCL from source with explicit GPU_TARGETS for gfx906.
    # See: https://github.com/ROCm/rocm-systems/issues/2805
    #
    # This is OPT-IN because building RCCL from source takes 30-60+ minutes
    # and requires hipcc (so it can't share the C++23 compiler we use for
    # llaminar). The Docker release image and CI default to OFF and rely on
    # the system librccl.so loaded dynamically at runtime via dlopen(). Set
    # LLAMINAR_BUILD_RCCL_FROM_SOURCE=ON locally if you need the gfx906 fix.
    option(LLAMINAR_BUILD_RCCL_FROM_SOURCE
           "Clone and build RCCL from source (workaround for broken gfx906 binaries in packaged RCCL)"
           OFF)

    if(LLAMINAR_BUILD_RCCL_FROM_SOURCE)
        find_package(Git REQUIRED)
    
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(RCCL_EXTERNAL_DIR "${REPO_ROOT}/external/rccl")
    set(RCCL_GIT_URL "https://github.com/ROCm/rccl.git")
    set(RCCL_GIT_REF "rocm-7.1.1")  # Match our ROCm version
    
    # Auto-detect GPU architecture for RCCL build
    # Default to gfx906 (MI50/MI60) which is known to be broken in packaged RCCL
    if(NOT DEFINED RCCL_GPU_TARGETS)
        # Try to detect from rocminfo
        execute_process(
            COMMAND bash -c "rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-z]+' | head -1"
            OUTPUT_VARIABLE DETECTED_GPU_TARGET
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(DETECTED_GPU_TARGET)
            set(RCCL_GPU_TARGETS "${DETECTED_GPU_TARGET}")
            message(STATUS "V2: Auto-detected GPU target for RCCL: ${RCCL_GPU_TARGETS}")
        else()
            set(RCCL_GPU_TARGETS "gfx906")
            message(STATUS "V2: Using default GPU target for RCCL: ${RCCL_GPU_TARGETS}")
        endif()
    endif()
    
    if(NOT EXISTS "${REPO_ROOT}/external")
        file(MAKE_DIRECTORY "${REPO_ROOT}/external")
    endif()
    
    if(NOT EXISTS "${RCCL_EXTERNAL_DIR}")
        message(STATUS "V2: RCCL not found in external/, cloning ${RCCL_GIT_URL} (${RCCL_GIT_REF})...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch ${RCCL_GIT_REF}
                ${RCCL_GIT_URL}
                rccl
            WORKING_DIRECTORY "${REPO_ROOT}/external"
            RESULT_VARIABLE GIT_CLONE_RESULT
            OUTPUT_VARIABLE GIT_CLONE_OUTPUT
            ERROR_VARIABLE GIT_CLONE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        
        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to clone RCCL: ${GIT_CLONE_ERROR}")
        endif()
        
        message(STATUS "V2: RCCL ${RCCL_GIT_REF} cloned successfully")
    else()
        message(STATUS "V2: RCCL already present at ${RCCL_EXTERNAL_DIR}; leaving existing checkout untouched")
    endif()
    
    # Get current commit hash for cache validation
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY "${RCCL_EXTERNAL_DIR}"
        RESULT_VARIABLE RCCL_REV_PARSE_RESULT
        OUTPUT_VARIABLE RCCL_COMMIT
        ERROR_VARIABLE RCCL_REV_PARSE_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT RCCL_REV_PARSE_RESULT EQUAL 0)
        message(FATAL_ERROR "V2: Failed to determine RCCL commit: ${RCCL_REV_PARSE_ERROR}")
    endif()
    
    # Check if RCCL build is up-to-date
    set(RCCL_BUILD_DIR "${RCCL_EXTERNAL_DIR}/build")
    set(RCCL_VERSION_FILE "${RCCL_BUILD_DIR}/.llaminar-rccl-commit")
    
    set(RCCL_BUILD_COMMIT "missing")
    if(EXISTS "${RCCL_VERSION_FILE}")
        file(READ "${RCCL_VERSION_FILE}" RCCL_BUILD_COMMIT_RAW)
        string(STRIP "${RCCL_BUILD_COMMIT_RAW}" RCCL_BUILD_COMMIT)
    endif()
    
    if(EXISTS "${RCCL_BUILD_DIR}" AND NOT RCCL_BUILD_COMMIT STREQUAL RCCL_COMMIT)
        message(STATUS "V2: RCCL build cache (${RCCL_BUILD_COMMIT}) is stale vs ${RCCL_COMMIT}, rebuilding...")
        file(REMOVE_RECURSE "${RCCL_BUILD_DIR}")
    endif()
    
    # Build RCCL if needed
    if(NOT EXISTS "${RCCL_BUILD_DIR}/librccl.so")
        message(STATUS "V2: Building RCCL (${RCCL_GIT_URL}@${RCCL_GIT_REF} @ ${RCCL_COMMIT}) from source...")
        message(STATUS "V2: RCCL GPU targets: ${RCCL_GPU_TARGETS}")
        message(STATUS "V2: NOTE: This may take 30-60+ minutes on first build")
        
        # Configure RCCL
        execute_process(
            COMMAND ${CMAKE_COMMAND} -B "${RCCL_BUILD_DIR}" -S "${RCCL_EXTERNAL_DIR}"
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_PREFIX_PATH=/opt/rocm
                    -DGPU_TARGETS=${RCCL_GPU_TARGETS}
                    -DBUILD_TESTS=OFF
            RESULT_VARIABLE RCCL_CONFIG_RESULT
            OUTPUT_VARIABLE RCCL_CONFIG_OUTPUT
            ERROR_VARIABLE RCCL_CONFIG_ERROR
        )
        
        if(NOT RCCL_CONFIG_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to configure RCCL: ${RCCL_CONFIG_ERROR}")
        endif()
        
        # Build RCCL
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build "${RCCL_BUILD_DIR}" --parallel
            RESULT_VARIABLE RCCL_BUILD_RESULT
            OUTPUT_VARIABLE RCCL_BUILD_OUTPUT
            ERROR_VARIABLE RCCL_BUILD_ERROR
        )
        
        if(NOT RCCL_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to build RCCL: ${RCCL_BUILD_ERROR}")
        endif()
        
        # Write version file for cache validation
        file(WRITE "${RCCL_VERSION_FILE}" "${RCCL_COMMIT}\n")
        
        message(STATUS "V2: RCCL built successfully in ${RCCL_BUILD_DIR}")
    else()
        message(STATUS "V2: RCCL already built in ${RCCL_BUILD_DIR}")
    endif()
    endif() # LLAMINAR_BUILD_RCCL_FROM_SOURCE
    
    # =========================================================================
    # RCCL Detection (ROCm Collective Communications Library)
    # =========================================================================
    # Search for RCCL - prioritize our local build over system packages
    # (system packages may have broken GPU binaries for certain architectures)
    find_path(RCCL_INCLUDE_DIR rccl/rccl.h rccl.h
        HINTS
            ${RCCL_BUILD_DIR}/include        # Our local build (highest priority)
            ${RCCL_EXTERNAL_DIR}/src/include # RCCL source headers
            ${ROCM_PATH}/include
            ${ROCM_PATH}/rccl/include
        PATHS
            /opt/rocm/include
            /opt/rocm/rccl/include
            /usr/include
            /usr/local/include
        NO_DEFAULT_PATH
    )
    find_library(RCCL_LIBRARY rccl
        HINTS
            ${RCCL_BUILD_DIR}                # Our local build (highest priority)
            ${ROCM_PATH}/lib
            ${ROCM_PATH}/rccl/lib
        PATHS
            /opt/rocm/lib
            /opt/rocm/rccl/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib64
            /usr/local/lib
        NO_DEFAULT_PATH
    )
    
    if(RCCL_INCLUDE_DIR AND RCCL_LIBRARY)
        set(HAVE_RCCL ON CACHE BOOL "RCCL library found")
        add_compile_definitions(HAVE_RCCL)
        
        # Check if we're using our local build or system RCCL
        string(FIND "${RCCL_LIBRARY}" "${RCCL_BUILD_DIR}" RCCL_IS_LOCAL)
        if(NOT RCCL_IS_LOCAL EQUAL -1)
            message(STATUS "V2: RCCL found (locally built for ${RCCL_GPU_TARGETS})")
        else()
            message(STATUS "V2: RCCL found (system)")
            message(WARNING "V2: Using system RCCL - may have broken GPU binaries for gfx906")
        endif()
        message(STATUS "V2:   Include: ${RCCL_INCLUDE_DIR}")
        message(STATUS "V2:   Library: ${RCCL_LIBRARY}")
    else()
        set(HAVE_RCCL OFF CACHE BOOL "RCCL library not found")
        message(WARNING "V2: RCCL not found - RCCLBackend will be disabled")
        if(NOT RCCL_INCLUDE_DIR)
            message(WARNING "V2:   rccl.h not found in standard paths")
        endif()
        if(NOT RCCL_LIBRARY)
            message(WARNING "V2:   librccl.so not found in standard paths")
        endif()
    endif()
endif()

# =============================================================================
# Device Benchmarks (CPU + CUDA + ROCm)
# =============================================================================
# Lightweight benchmarks for measuring actual device capabilities:
# - Memory bandwidth (GB/s)
# - Compute throughput (GFLOPS/GOPS)
# - Host-device transfer rates
# - Peer-to-peer transfer rates (multi-GPU)
#
# Results are used by PlacementStrategy to make informed placement decisions
# based on measured values rather than heuristic estimates.

set(BENCHMARK_SOURCES
    backends/benchmarks/DeviceBenchmark.cpp
    backends/benchmarks/CPUBenchmark.cpp
)

# Add CUDA benchmark if CUDA backend enabled
if(HAVE_CUDA)
    list(APPEND BENCHMARK_SOURCES backends/benchmarks/CUDABenchmark.cu)
endif()

# Add ROCm benchmark if ROCm backend enabled
if(HAVE_ROCM)
    list(APPEND BENCHMARK_SOURCES backends/benchmarks/ROCmBenchmark.cpp)
endif()



add_library(device_benchmarks STATIC ${BENCHMARK_SOURCES})
# Resolve device symbols at library build time (same rationale as cuda_backend/rocm_backend).
if(HAVE_CUDA)
    set_target_properties(device_benchmarks PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    if(CMAKE_GENERATOR MATCHES "Ninja")
        set_property(TARGET device_benchmarks PROPERTY JOB_POOL_LINK nvlink_pool)
    endif()
endif()
if(HAVE_ROCM)
    set_target_properties(device_benchmarks PROPERTIES
        HIP_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
target_include_directories(device_benchmarks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(device_benchmarks PRIVATE OpenMP::OpenMP_CXX)

if(HAVE_CUDA)
    target_link_libraries(device_benchmarks PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(device_benchmarks PRIVATE HAVE_CUDA=1)
endif()

if(HAVE_ROCM)
    target_link_libraries(device_benchmarks PRIVATE hip::host)
    target_compile_definitions(device_benchmarks PRIVATE HAVE_ROCM=1)
    # ROCm files need HIP compilation
    set_source_files_properties(
        backends/benchmarks/ROCmBenchmark.cpp
        PROPERTIES LANGUAGE HIP
    )
    if(HAVE_CUDA)
        set_source_files_properties(
            collective/backends/HostBackendROCm.cpp
            PROPERTIES LANGUAGE HIP
        )
    endif()
endif()

message(STATUS "V2: Device benchmarks library configured")

# =============================================================================
# OneDNN Configuration for INT8/BF16 GEMM Acceleration
# =============================================================================
# OneDNN provides optimized matmul primitives for:
#   - INT8: s8s8s32 with post-ops (used by INT8PackedGemm)
#   - BF16: bf16bf16f32 (used by BF16GemmKernel)
#   - FP32: f32 matmul baseline
if(USE_ONEDNN)
    find_package(Git REQUIRED)

    # First, ensure OneDNN exists in external/ and is on the expected revision
    # Note: Repository root is two levels up from src/v2
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(ONEDNN_EXTERNAL_DIR "${REPO_ROOT}/external/onednn")
    set(ONEDNN_GIT_URL "https://github.com/uxlfoundation/oneDNN.git")
    # Pinned to an upstream SemVer release tag for reproducible builds.
    # We previously tracked dbsanfte/oneDNN@main for "fused softmax matmul"
    # APIs, but that codepath is no longer used in llaminar. Bump this tag
    # when picking up new upstream features. Latest releases:
    # https://github.com/uxlfoundation/oneDNN/releases
    set(ONEDNN_GIT_REF "v3.11.3")

    if(NOT EXISTS "${REPO_ROOT}/external")
        file(MAKE_DIRECTORY ${REPO_ROOT}/external)
    endif()

    # If an existing checkout points at the wrong remote (e.g. the old
    # dbsanfte/oneDNN fork), wipe it so we re-clone the pinned upstream tag.
    if(EXISTS "${ONEDNN_EXTERNAL_DIR}/.git")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${ONEDNN_EXTERNAL_DIR} remote get-url origin
            OUTPUT_VARIABLE _ONEDNN_EXISTING_URL
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT _ONEDNN_EXISTING_URL STREQUAL ONEDNN_GIT_URL)
            message(STATUS "V2: OneDNN at ${ONEDNN_EXTERNAL_DIR} points at '${_ONEDNN_EXISTING_URL}', expected '${ONEDNN_GIT_URL}'; re-cloning")
            file(REMOVE_RECURSE "${ONEDNN_EXTERNAL_DIR}")
        endif()
    endif()

    if(NOT EXISTS "${ONEDNN_EXTERNAL_DIR}")
        message(STATUS "V2: OneDNN not found in external/, cloning ${ONEDNN_GIT_URL} (${ONEDNN_GIT_REF})...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch ${ONEDNN_GIT_REF}
                ${ONEDNN_GIT_URL}
                onednn
            WORKING_DIRECTORY ${REPO_ROOT}/external
            RESULT_VARIABLE GIT_CLONE_RESULT
            OUTPUT_VARIABLE GIT_CLONE_OUTPUT
            ERROR_VARIABLE GIT_CLONE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to clone OneDNN: ${GIT_CLONE_ERROR}")
        endif()

        message(STATUS "V2: OneDNN ${ONEDNN_GIT_REF} cloned successfully")
    else()
        message(STATUS "V2: OneDNN already present at ${ONEDNN_EXTERNAL_DIR}; leaving existing checkout untouched")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${ONEDNN_EXTERNAL_DIR}
        RESULT_VARIABLE ONEDNN_REV_PARSE_RESULT
        OUTPUT_VARIABLE ONEDNN_COMMIT
        ERROR_VARIABLE ONEDNN_REV_PARSE_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT ONEDNN_REV_PARSE_RESULT EQUAL 0)
        message(FATAL_ERROR "V2: Failed to determine OneDNN commit: ${ONEDNN_REV_PARSE_ERROR}")
    endif()

    # Now check if OneDNN is built for the requested commit
    set(ONEDNN_BUILD_DIR "${ONEDNN_EXTERNAL_DIR}/build")
    set(ONEDNN_VERSION_FILE "${ONEDNN_BUILD_DIR}/.llaminar-onednn-commit")

    set(ONEDNN_BUILD_COMMIT "missing")
    if(EXISTS "${ONEDNN_VERSION_FILE}")
        file(READ "${ONEDNN_VERSION_FILE}" ONEDNN_BUILD_COMMIT_RAW)
        string(STRIP "${ONEDNN_BUILD_COMMIT_RAW}" ONEDNN_BUILD_COMMIT)
    endif()

    if(EXISTS "${ONEDNN_BUILD_DIR}" AND NOT ONEDNN_BUILD_COMMIT STREQUAL ONEDNN_COMMIT)
        message(STATUS
            "V2: OneDNN build cache (${ONEDNN_BUILD_COMMIT}) is stale vs ${ONEDNN_COMMIT}, rebuilding...")
        file(REMOVE_RECURSE "${ONEDNN_BUILD_DIR}")
    endif()

    if(NOT EXISTS "${ONEDNN_BUILD_DIR}/include/oneapi/dnnl/dnnl.hpp")
        message(STATUS "V2: Building OneDNN (${ONEDNN_GIT_URL}@${ONEDNN_GIT_REF} @ ${ONEDNN_COMMIT}) from source...")

        # Configure OneDNN
        execute_process(
            COMMAND ${CMAKE_COMMAND} -B ${ONEDNN_BUILD_DIR} -S ${ONEDNN_EXTERNAL_DIR}
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_INSTALL_PREFIX=${ONEDNN_BUILD_DIR}
                    -DDNNL_CPU_RUNTIME=OMP
                    -DDNNL_BUILD_TESTS=OFF
                    -DDNNL_BUILD_EXAMPLES=OFF
                    -DDNNL_EXPERIMENTAL_UKERNEL=ON
                    -DCMAKE_CXX_FLAGS=-march=native
                    -DCMAKE_C_FLAGS=-march=native
            RESULT_VARIABLE ONEDNN_CONFIG_RESULT
            OUTPUT_VARIABLE ONEDNN_CONFIG_OUTPUT
            ERROR_VARIABLE ONEDNN_CONFIG_ERROR
        )

        if(NOT ONEDNN_CONFIG_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to configure OneDNN: ${ONEDNN_CONFIG_ERROR}")
        endif()

        # Build OneDNN
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${ONEDNN_BUILD_DIR} --parallel --target install
            RESULT_VARIABLE ONEDNN_BUILD_RESULT
            OUTPUT_VARIABLE ONEDNN_BUILD_OUTPUT
            ERROR_VARIABLE ONEDNN_BUILD_ERROR
        )

        if(NOT ONEDNN_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to build OneDNN: ${ONEDNN_BUILD_ERROR}")
        endif()

        file(WRITE "${ONEDNN_VERSION_FILE}" "${ONEDNN_COMMIT}\n")

        message(STATUS "V2: OneDNN built and installed to ${ONEDNN_BUILD_DIR}")
    else()
        message(STATUS "V2: OneDNN already built in ${ONEDNN_BUILD_DIR}")
    endif()
    
    # Look for OneDNN in standard locations + our build dir
    set(ONEDNN_SEARCH_PATHS
        "${ONEDNN_BUILD_DIR}"           # Our freshly built version (highest priority)
        "/opt/onednn"
        "/usr/local"
        "/usr"
        "$ENV{HOME}/onednn"
    )
    
    set(ONEDNN_FOUND FALSE)
    foreach(SEARCH_PATH ${ONEDNN_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}/include/oneapi/dnnl/dnnl.hpp")
            set(ONEDNN_ROOT "${SEARCH_PATH}")
            set(ONEDNN_INCLUDE_DIR "${SEARCH_PATH}/include")
            set(ONEDNN_FOUND TRUE)
            
            # Look for library
            find_library(ONEDNN_LIB
                NAMES dnnl
                PATHS "${SEARCH_PATH}/lib" "${SEARCH_PATH}/lib64"
                NO_DEFAULT_PATH
            )
            
            if(ONEDNN_LIB)
                message(STATUS "V2: OneDNN found at ${ONEDNN_ROOT}")
                message(STATUS "V2: OneDNN include: ${ONEDNN_INCLUDE_DIR}")
                message(STATUS "V2: OneDNN library: ${ONEDNN_LIB}")
                message(STATUS "V2: OneDNN provides: INT8 s8s8s32, BF16 bf16bf16f32 matmul")
                add_compile_definitions(HAVE_ONEDNN)
                break()
            endif()
        endif()
    endforeach()
    
    if(NOT ONEDNN_FOUND)
        message(WARNING "V2: OneDNN build failed or library not found - INT8/BF16 GEMM will use fallback paths")
        message(WARNING "V2: INT8: Falls back to AVX512-VNNI or scalar")
        message(WARNING "V2: BF16: Falls back to FP32 expansion")
        set(USE_ONEDNN OFF)
    endif()
endif()

# Find Python3 for code generation scripts
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "V2: Found Python3: ${Python3_EXECUTABLE}")

# CPU BLAS backend: OneDNN for all GEMM (FP32, BF16, INT8)
message(STATUS "V2: BLAS backend: OneDNN")

# Define GEMM script path for microkernel generation
# =============================================================================
# GEMM v1, v2, and v3 have been removed - only gemm (OneDNN-based) remains
# All legacy GEMM microkernel generation and integer GEMM implementations are obsolete
# =============================================================================

# =============================================================================
# Vendored header-only libraries (cpp-httplib, nlohmann/json)
# =============================================================================
# These are header-only libraries vendored in external/vendor/.
# =============================================================================

# =============================================================================
# FTXUI Integration (for interactive chat UI)
# =============================================================================
include(FetchContent)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui
    GIT_TAG v5.0.0
    GIT_SHALLOW TRUE
)

# Disable FTXUI examples and tests to speed up build
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "Disable FTXUI docs" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "Disable FTXUI examples" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "Disable FTXUI tests" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "Disable FTXUI install" FORCE)

FetchContent_MakeAvailable(ftxui)
message(STATUS "V2: FTXUI v5.0.0 configured for interactive chat UI")

# =============================================================================
# libfort Integration (for formatted ASCII/Unicode tables in test output)
# =============================================================================
# Set policy compatibility for libfort's older CMakeLists.txt
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(libfort
    GIT_REPOSITORY https://github.com/seleznevae/libfort
    GIT_TAG v0.4.2
    GIT_SHALLOW TRUE
)

# Disable libfort tests and examples
set(FORT_ENABLE_TESTING OFF CACHE BOOL "Disable libfort tests" FORCE)

FetchContent_MakeAvailable(libfort)
# Reset policy version
unset(CMAKE_POLICY_VERSION_MINIMUM)
message(STATUS "V2: libfort v0.4.2 configured for formatted table output")

# Link fort to GPU backends (for CUDAKernelProfiler.h and ROCmKernelProfiler.h table output)
if(HAVE_CUDA)
    target_link_libraries(cuda_backend PUBLIC fort)
endif()
if(HAVE_ROCM)
    target_link_libraries(rocm_backend PUBLIC fort)
endif()

# =============================================================================
# Typed Register Enforcement (JIT Code Quality)
# =============================================================================
# V2 Core Library
add_library(llaminar2_core STATIC
    # Interfaces
    interfaces/IMPIContext.cpp
    backends/ComputeBackend.cpp
    backends/HardwareInventory.cpp
    backends/InventoryPrinter.cpp
    backends/PCIeEnumeration.cpp
    backends/BackendManager.cpp
    backends/CPUBackend.cpp
    # GPUDeviceContextPool is backend-agnostic (uses factory functions registered by cuda_backend/rocm_backend)
    backends/GPUDeviceContextPool.cpp
    # GPUDeviceWorker is backend-agnostic (uses forward-declared cuda_worker/rocm_worker namespaces)
    backends/GPUDeviceWorker.cpp
    # Execution framework (hierarchical work distribution)
    execution/mpi_orchestration/WorkDistributor.cpp
    execution/local_execution/device/DeviceContext.cpp
    # ComputeStage base classes and factory in compute_stages/
    execution/compute_stages/ComputeStageBase.cpp
    execution/compute_stages/ComputeStageFactory.cpp
    # Individual stages in compute_stages/stages/
    execution/compute_stages/stages/GEMMStage.cpp
    execution/compute_stages/stages/FusedQKVGEMMStage.cpp
    execution/compute_stages/stages/FusedGateUpGEMMStage.cpp
    execution/compute_stages/stages/RMSNormStage.cpp
    execution/compute_stages/stages/QKNormStage.cpp
    execution/compute_stages/stages/RoPEStage.cpp
    execution/compute_stages/stages/ResidualAddStage.cpp
    execution/compute_stages/stages/FusedResidualNormStage.cpp
    execution/compute_stages/stages/AllreduceStage.cpp
    execution/compute_stages/stages/AllGatherStage.cpp
    execution/compute_stages/stages/AllGatherVStage.cpp
    execution/compute_stages/stages/SendActivationsStage.cpp
    execution/compute_stages/stages/ReceiveActivationsStage.cpp
    execution/compute_stages/stages/MoEStages.cpp
    execution/compute_stages/stages/AttentionOutputGateStage.cpp
    execution/compute_stages/stages/GatedRMSNormStage.cpp
    execution/compute_stages/stages/GDNProjectionStage.cpp
    execution/compute_stages/stages/ShortConv1dStage.cpp
    execution/compute_stages/stages/GDNRecurrenceStage.cpp
    execution/compute_stages/stages/QGateSplitStage.cpp
    execution/moe/IMoERouter.cpp
    execution/moe/IMoEDispatcher.cpp
    execution/moe/IMoEExpertExecutor.cpp
    execution/moe/IMoECombiner.cpp
    execution/moe/ExpertPlacementMap.cpp
    execution/moe/ExpertWeightCache.cpp
    execution/moe/ExpertRebalancer.cpp
    execution/moe/MoEPipeline.cpp
    execution/cache/HybridCacheManager.cpp
    execution/compute_stages/stages/AttentionWithKVCacheStage.cpp
    execution/compute_stages/stages/KVCacheAppendStage.cpp
    execution/compute_stages/stages/KVCacheGatherStage.cpp
    execution/compute_stages/stages/AttentionComputeStage.cpp
    execution/compute_stages/stages/EmbeddingStage.cpp
    execution/compute_stages/stages/LMHeadStage.cpp
    execution/local_execution/graph/DeviceGraphCaptureController.cpp
    execution/local_execution/graph/ComputeGraph.cpp
    execution/local_execution/graph/StageVerifier.cpp
    execution/local_execution/graph/DeviceGraphExecutor.cpp
    execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp
    execution/local_execution/graph/StageTimeline.cpp
    execution/local_execution/device/DomainAwareBufferManager.cpp
    execution/local_execution/graph/GraphResolver.cpp
    execution/local_execution/graph/GraphBuilderRegistry.cpp
    execution/local_execution/graph/SchemaFactoryRegistry.cpp
    execution/local_execution/model/LayerExecutor.cpp
    execution/local_execution/model/ModelExecutor.cpp
    execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp
    execution/local_execution/orchestrators/MultiDomainOrchestrator.cpp
    execution/local_execution/orchestrators/LogitsGatherer.cpp
    execution/local_execution/orchestrators/DeviceSampler.cpp
    execution/local_execution/orchestrators/MultiDeviceOrchestrator.cpp
    execution/local_execution/model/HeterogeneousLayerExecutor.cpp
    execution/local_execution/coherence/CrossDomainTransfer.cpp
    execution/config/HybridPrecisionConfig.cpp
    execution/config/InferenceMode.cpp
    execution/local_execution/coherence/StageCoherence.cpp
    execution/local_execution/device/DeviceWorkspaceManager.cpp
    execution/local_execution/device/WorkspaceAllocator.cpp
    # Unified transfer engine (explicit coherence state machine + centralized data movement)
    transfer/TransferEngine.cpp
    # Tensor parallelism configuration (proportional sharding for heterogeneous TP)
    config/TensorParallelConfig.cpp
    # Pipeline parallelism configuration (layer range assignment for PP)
    config/PipelineParallelConfig.cpp
    # Layer placement configuration (layer-to-device assignment for CPU pipeline stages)
    config/LayerPlacementConfig.cpp
    # TP Domain configuration (multi-domain tensor parallelism for hybrid systems)
    config/TPDomain.cpp
    # TP Domain Config (Phase 1.1 - Unified PP Graph Architecture)
    config/TPDomainConfig.cpp
    # PP Stage Config (Phase 1.2 - Unified PP Graph Architecture)
    config/PPStageConfig.cpp
    # Pipeline Config (Phase 1.3 - Unified PP Graph Architecture)
    config/PipelineConfig.cpp
    # Collective backend type (extracted to avoid circular dependency)
    config/CollectiveBackendType.cpp
    # Orchestration configuration (Phase 0 - CLI/Config architecture for heterogeneous orchestration)
    config/OrchestrationConfig.cpp
    config/OrchestrationConfigParser.cpp
    # Declarative config validation rule framework (device selection conflict detection)
    config/ConfigValidator.cpp
    # ParallelismTree parser (Phase 7 - CLI/YAML parser for ParallelismTree)
    config/ParallelismTreeParser.cpp
    # Backend selector (Part 2 - automatic backend selection for PP/TP)
    config/BackendSelector.cpp
    # Global device address (fully-qualified device identification across nodes)
    backends/GlobalDeviceAddress.cpp
    # Device registry and adapter (Phase 6 - GlobalDeviceAddress migration)
    backends/DeviceRegistry.cpp
    backends/DeviceAddressAdapter.cpp
    # Slab GEMM configuration (for chunked FP16 GEMM workspace management)
    kernels/SlabGemmConfig.cpp
    # Placement strategy (weight/compute distribution)
    execution/mpi_orchestration/PlacementPlan.cpp
    execution/mpi_orchestration/PlacementStrategy.cpp
    execution/mpi_orchestration/placement/HeterogeneousMultiDomainStrategy.cpp
    # Memory planning (pre-flight VRAM/RAM budget estimation)
    planning/ClusterInventoryGatherer.cpp
    planning/ModelMemoryProfile.cpp
    planning/WeightMemoryEstimator.cpp
    planning/KVCacheMemoryEstimator.cpp
    planning/ActivationMemoryEstimator.cpp
    planning/WorkspaceMemoryEstimator.cpp
    planning/MemoryPlanner.cpp
    # Execution plan builder (Phase 1 - Plan-to-Graph translation)
    execution/mpi_orchestration/ExecutionPlanBuilder.cpp
    # Pipeline parallel graph builder (Phase 3 - PP stage insertion)
    execution/local_execution/graph/PipelineParallelGraphBuilder.cpp
    # Inference dispatch (InferenceRunner decouples pipeline vs graph paths)
    execution/factory/InferenceRunnerFactory.cpp
    # App-level components (extracted from Main.cpp)
    app/AppLifecycle.cpp
    app/SubcommandRouter.cpp
    app/Splash.cpp
    app/InferenceRunnerAdapter.cpp
    app/ChatTemplateResolver.cpp
    app/MPIBootstrapPhase.cpp
    app/RuntimeInitPhase.cpp
    # Subcommands
    app/commands/CommandMPI.cpp
    app/commands/BenchmarkCommand.cpp
    app/commands/LegacyCommand.cpp
    app/commands/DescribeCommand.cpp
    app/commands/OneshotCommand.cpp
    app/commands/ServeCommand.cpp
    app/commands/PlanCommand.cpp
    # Execution modes (used by LegacyCommand)
    app/modes/InteractiveChatMode.cpp
    app/modes/SingleShotChatMode.cpp
    app/modes/BenchmarkMode.cpp
    app/modes/ServerMode.cpp
    app/modes/ChatCompletionHandler.cpp
    app/modes/CompletionMode.cpp
    loaders/ModelLoader.cpp
    loaders/ModelContext.cpp
    loaders/ModelContextConfig.cpp
    loaders/WeightManager.cpp
    loaders/WeightSlicer.cpp
    loaders/WeightPlacementMap.cpp
    loaders/DeviceOrchestrator.cpp
    loaders/LayerWeightStreamer.cpp
    loaders/WeightStreamerFactory.cpp
    # Partial weight loader (Phase 3 - PP partial weight loading)
    loaders/PartialWeightLoader.cpp
    # Layer device placement (Phase 2 - multi-device graph building)
    execution/local_execution/device/LayerDevicePlacement.cpp
    # Qwen2 model and graph configuration
    models/ModelRegistrations.cpp
    models/qwen/QwenGraphBase.cpp
    models/qwen/Qwen2Graph.cpp
    models/qwen/Qwen2GraphConfigBuilder.cpp
    models/qwen/Qwen2BufferSpec.cpp
    # Qwen3.5 Dense (hybrid GDN + Full Attention)
    models/qwen35/Qwen35Graph.cpp
    models/qwen35/Qwen35GraphConfigBuilder.cpp    tensors/TensorBase.cpp
    tensors/FP32Tensor.cpp
    tensors/FP16Tensor.cpp
    tensors/BF16Tensor.cpp
    tensors/INT32Tensor.cpp
    tensors/INT8Tensor.cpp
    tensors/TensorFactory.cpp
    tensors/GPUTensorVerification.cpp
    tensors/IQ4_NLTensor.cpp
    # Simple quantization formats (Q8_0, Q8_1, Q16_1, Q4_0, Q4_1, Q5_0, Q5_1)
    tensors/Q8_0Tensor.cpp
    tensors/Q8_1Tensor.cpp
    tensors/Q16_1Tensor.cpp
    tensors/Q4_0Tensor.cpp
    tensors/Q4_1Tensor.cpp
    tensors/Q5_0Tensor.cpp
    tensors/Q5_1Tensor.cpp
    # K-quant formats (Q6_K, Q2_K, Q5_K, Q3_K, Q4_K, Q8_K)
    tensors/Q6_KTensor.cpp
    tensors/Q2_KTensor.cpp
    tensors/Q5_KTensor.cpp
    tensors/Q3_KTensor.cpp
    tensors/Q4_KTensor.cpp
    tensors/Q8_KTensor.cpp
    # IQ formats (IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, IQ3_S, IQ1_S, IQ1_M)
    tensors/IQ4_XSTensor.cpp
    tensors/IQ2_XXSTensor.cpp
    tensors/IQ2_XSTensor.cpp
    tensors/IQ3_XXSTensor.cpp
    tensors/IQ2_STensor.cpp
    tensors/IQ3_STensor.cpp
    tensors/IQ1_STensor.cpp
    tensors/IQ1_MTensor.cpp
    # TurboQuant KV cache tensors
    tensors/TQ4Tensor.cpp
    tensors/TQ8Tensor.cpp
    # KV Cache implementation (moved from tensors/)
    kernels/cpu/CPURingKVCache.cpp
    # Vectorized primitives (ported from V1)
    kernels/cpu/primitives/RoPEPrimitives.cpp
    kernels/cpu/primitives/RMSNormPrimitives.cpp
    kernels/cpu/primitives/SoftmaxPrimitives.cpp
    kernels/cpu/primitives/SwiGLUPrimitives.cpp
    kernels/cpu/primitives/Q16HeadNormalization.cpp
    # CPU GEMM kernels and auto-tuning infrastructure
    # NOTE: Legacy GEMM implementations (v1/v2/v3) removed - only gemm (OneDNN) remains
    # All old GEMM kernels removed: GemmAutoTuner, SmartGemmSearch, QuantizedGemm,
    # BF16PackedGemm, FP16PackedGemm, Q8_1GemmKernelInit, VNNIGemmKernelInit
    kernels/cpu/ops/CPURoPEKernelT.cpp     # Typed RoPE with fused precision conversion (FP32/BF16/FP16/Q8_1)
    kernels/cpu/ops/CPURMSNormKernelT.cpp  # Typed RMSNorm with fused precision conversion (replaced CPURMSNormKernelT)
    kernels/cpu/ops/CPUSoftmaxKernelT.cpp  # Typed Softmax with native precision paths (FP32/BF16/FP16/Q8_1)
    kernels/cpu/ops/CPUSwiGLUKernelT.cpp   # Typed SwiGLU with native precision paths (FP32/BF16/FP16/Q8_1)
    kernels/cpu/ops/CPUEmbeddingKernelT.cpp    # Typed Embedding lookup with native precision paths (FP32/BF16/FP16/Q8_1)
    kernels/cpu/attention/CPUFlashAttentionKernelT.cpp # FlashAttention2-style tiled CPU attention (FP32 with fallbacks)
    # GDN kernel implementations (Gated Delta Net)
    kernels/cpu/gdn/CPUShortConvolution.cpp
    kernels/cpu/gdn/CPUGatedDeltaNet.cpp
    # Kernel dispatch factory (centralized device-aware kernel creation)
    kernels/KernelFactory.cpp
    kernels/HybridKVCacheConfig.cpp
    kernels/common/PreparedEmbeddingWeights.cpp
    # Universal device kernel cache (hipBLAS, cuBLAS handles, etc.)
    kernels/DeviceKernelCache.cpp
    # Testing infrastructure
    testing/PerfHarness.cpp
    utils/BatchPaddingUtils.cpp
    utils/MPIBootstrap.cpp
    utils/NUMATopology.cpp
    utils/NodeTopology.cpp
    utils/MPITopology.cpp
    utils/NodeDetection.cpp
    utils/MPIStager.cpp
    utils/Tokenizer.cpp
    utils/Sampler.cpp
    utils/ChatTemplate.cpp
    utils/ChatUI.cpp
    utils/BenchmarkRunner.cpp
    # Collective communication infrastructure
    collective/DeviceGroup.cpp
    collective/BackendRouter.cpp
    collective/backends/HostBackend.cpp
    collective/backends/MPIBackend.cpp
    collective/backends/HeterogeneousBackend.cpp
    # NOTE: NCCLBackend.cpp is NOT here - it's in cuda_backend to isolate NCCL symbols
    # from RCCL. Both export the same symbols (ncclCommInitRank, etc.) and would conflict.
    # NOTE: RCCLBackend.cpp is NOT here - it's in rocm_backend to isolate RCCL symbols
    # from NCCL. Both export the same symbols (ncclCommInitRank, etc.) and would conflict.
    collective/backends/UPIBackend.cpp
    execution/local_execution/collective/CollectiveContext.cpp
    # NUMA-aware memory allocation
    memory/NUMAAllocator.cpp
    # Buffer harmonization (Phase 1: BufferArena + CoherenceTracker)
    memory/BufferArena.cpp
    memory/CoherenceTracker.cpp
    # LOCAL tensor parallelism (Phase 4: multi-device within single MPI rank)
    collective/LocalTPContext.cpp
    collective/LocalPPContext.cpp
    # GLOBAL tensor parallelism (cross-MPI-rank TP)
    collective/GlobalTPContext.cpp
    # TP context factory (Phase 3 Global TP Unified Interface)
    collective/TPContextFactory.cpp
    collective/PPStage.cpp
    loaders/LocalTPWeightSharder.cpp
    # TP allreduce stage (Phase 4: supports both LOCAL and GLOBAL TP)
    execution/compute_stages/stages/TPAllreduceStage.cpp
    # Legacy LOCAL TP allreduce stage (deprecated, use TPAllreduceStage)
    execution/compute_stages/stages/LocalTPAllreduceStage.cpp
    execution/compute_stages/stages/LocalPPTransferStage.cpp
    execution/compute_stages/stages/GlobalPPTransferStage.cpp
    # Global Pipeline Parallelism topology and planning
    execution/global_pp/GlobalPPTopology.cpp
    execution/global_pp/GlobalPPRankPlanBuilder.cpp
    # Parallelism tree (recursive PP/TP composition)
    execution/parallelism_tree/ParallelismTree.cpp
    execution/parallelism_tree/TransferSpec.cpp
    execution/parallelism_tree/CompoundShardInfo.cpp
    execution/parallelism_tree/HierarchicalCommBuilder.cpp
    execution/parallelism_tree/PipelineRunner.cpp
    execution/parallelism_tree/TreeToRunnerCompiler.cpp
    # Orchestration runner (Phase 5: end-to-end integration)
    execution/runner/OrchestrationRunner.cpp
    execution/runner/OrchestrationRunnerFactory.cpp
    # Snapshot capture (Phase 2: extracted from DeviceGraphOrchestrator)
    snapshots/SnapshotCapture.cpp
    # Forward execution engine (Phase 3: extracted from DeviceGraphOrchestrator)
    execution/local_execution/engine/ForwardExecutionEngine.cpp
    # NOTE: Legacy GEMM microkernel and instantiation systems removed (gemm v1/v2/v3)
    # GemmMicroKernelInit.cpp, Q8_1/VNNI instantiation sources - REMOVED
    # Vendored jinja template engine (from llama.cpp, MIT license)
    vendor/jinja/caps.cpp
    vendor/jinja/lexer.cpp
    vendor/jinja/parser.cpp
    vendor/jinja/runtime.cpp
    vendor/jinja/string.cpp
    vendor/jinja/value.cpp
)

# ---------------------------------------------------------------------------
# Embed model-specific Jinja chat templates
#
# Each (jinja file -> generated header) pair lives under jinja/<family>/<ver>/.
# The generator script `cmake/EmbedJinjaTemplate.cmake` reads the .jinja file
# and wraps its contents in a C++ raw-string literal. Generated headers land
# in ${CMAKE_CURRENT_BINARY_DIR}/generated/jinja_templates/ so we add that to
# the include path.
# ---------------------------------------------------------------------------
set(LLAMINAR2_JINJA_EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/EmbedJinjaTemplate.cmake")
set(LLAMINAR2_JINJA_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/jinja_templates")
file(MAKE_DIRECTORY "${LLAMINAR2_JINJA_GEN_DIR}")

function(llaminar2_embed_jinja_template)
    cmake_parse_arguments(ARG "" "INPUT;OUTPUT;NAMESPACE;SYMBOL;SOURCE_URL;LICENSE" "" ${ARGN})
    if(NOT ARG_INPUT OR NOT ARG_OUTPUT OR NOT ARG_NAMESPACE OR NOT ARG_SYMBOL)
        message(FATAL_ERROR "llaminar2_embed_jinja_template: INPUT, OUTPUT, NAMESPACE, SYMBOL required")
    endif()
    if(NOT ARG_SOURCE_URL)
        set(ARG_SOURCE_URL "(not provided)")
    endif()
    if(NOT ARG_LICENSE)
        set(ARG_LICENSE "(not provided)")
    endif()
    add_custom_command(
        OUTPUT "${ARG_OUTPUT}"
        COMMAND ${CMAKE_COMMAND}
            -DJINJA_INPUT=${ARG_INPUT}
            -DHEADER_OUTPUT=${ARG_OUTPUT}
            -DNAMESPACE=${ARG_NAMESPACE}
            -DSYMBOL=${ARG_SYMBOL}
            -DSOURCE_URL=${ARG_SOURCE_URL}
            -DLICENSE=${ARG_LICENSE}
            -P "${LLAMINAR2_JINJA_EMBED_SCRIPT}"
        DEPENDS "${ARG_INPUT}" "${LLAMINAR2_JINJA_EMBED_SCRIPT}"
        COMMENT "Embedding Jinja template: ${ARG_INPUT}"
        VERBATIM
    )
endfunction()

# --- Qwen 3.5 community chat template (MIT) --------------------------------
set(QWEN35_JINJA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../jinja/qwen/qwen35/template.jinja")
set(QWEN35_JINJA_HDR "${LLAMINAR2_JINJA_GEN_DIR}/qwen/qwen35/Qwen35ChatTemplate.generated.h")
llaminar2_embed_jinja_template(
    INPUT       "${QWEN35_JINJA_SRC}"
    OUTPUT      "${QWEN35_JINJA_HDR}"
    NAMESPACE   "llaminar2::qwen35"
    SYMBOL      "kCommunityChatTemplate"
    SOURCE_URL  "https://huggingface.co/froggeric/Qwen3.5-35B-A3B-Uncensored-FernflowerAI-MLX-8bit/resolve/main/chat_template.jinja"
    LICENSE     "MIT"
)
target_sources(llaminar2_core PRIVATE "${QWEN35_JINJA_HDR}")
target_include_directories(llaminar2_core PRIVATE "${LLAMINAR2_JINJA_GEN_DIR}")

target_include_directories(llaminar2_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# cpp-httplib and nlohmann/json for ServerMode
target_include_directories(llaminar2_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor
)

# Enable pipeline snapshot capture (for parity testing and debugging)
# - Debug builds: Always enabled (development/debugging)
# - E2ERelease builds: Always enabled (optimized E2E parity testing)
# - Integration builds: Always enabled (Release + debug symbols + snapshots)
# - Release builds: Disabled by default (performance), enable with -DENABLE_SNAPSHOTS=ON
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (Debug build)")
elseif(CMAKE_BUILD_TYPE MATCHES "E2ERelease")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (E2ERelease build - optimized with snapshots for E2E testing)")
elseif(CMAKE_BUILD_TYPE MATCHES "Integration")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (Integration build - optimized with debug symbols and snapshots)")
else()
    option(ENABLE_SNAPSHOTS "Enable pipeline snapshot capture" OFF)
    if(ENABLE_SNAPSHOTS)
        target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
        message(STATUS "V2: Pipeline snapshot capture enabled (explicit option)")
    else()
        message(STATUS "V2: Pipeline snapshot capture disabled (Release build, use -DENABLE_SNAPSHOTS=ON to enable)")
    endif()
endif()

# Attention input dumping for debugging
if(DUMP_ATTENTION_INPUTS)
    target_compile_definitions(llaminar2_core PUBLIC LLAMINAR_DUMP_ATTENTION_INPUTS)
    message(STATUS "V2: Attention input dumping ENABLED (will dump Q/K/V/mask to disk)")
endif()

# Compile features - C++23 for std::stacktrace (transfer diagnostics) and C++20 features
target_compile_features(llaminar2_core PUBLIC cxx_std_23)

# C++23 stacktrace library support
# GCC 13+ supports stacktrace with -lstdc++_libbacktrace, BUT when ROCm is enabled
# the linker is clang++ which can't link against this GCC-specific library.
# Therefore we disable stacktrace when ROCm is enabled.
if(HAVE_ROCM)
    message(STATUS "V2: C++23 stacktrace DISABLED (ROCm uses clang++ linker, incompatible with stdc++_libbacktrace)")
    target_compile_definitions(llaminar2_core PUBLIC LLAMINAR_DISABLE_STACKTRACE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    # Check if the library exists
    find_library(STDCXX_LIBBACKTRACE stdc++_libbacktrace)
    if(STDCXX_LIBBACKTRACE)
        message(STATUS "V2: Found stdc++_libbacktrace for C++23 stacktrace support")
        target_link_libraries(llaminar2_core PUBLIC ${STDCXX_LIBBACKTRACE})
    else()
        message(STATUS "V2: stdc++_libbacktrace not found - stacktrace disabled")
        target_compile_definitions(llaminar2_core PUBLIC LLAMINAR_DISABLE_STACKTRACE)
    endif()
else()
    message(STATUS "V2: C++23 stacktrace disabled (requires GCC 13+)")
    target_compile_definitions(llaminar2_core PUBLIC LLAMINAR_DISABLE_STACKTRACE)
endif()

# Dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
message(STATUS "V2: MPI enabled (required dependency)")

# Generate microkernel instantiations (auto-run Python script at configure time)
find_library(NUMA_LIBRARY NAMES numa)
if(NUMA_LIBRARY)
    message(STATUS "Found libnuma: ${NUMA_LIBRARY}")
    target_compile_definitions(llaminar2_core PUBLIC HAVE_NUMA)
    target_link_libraries(llaminar2_core PUBLIC ${NUMA_LIBRARY})
else()
    message(STATUS "libnuma not found - NUMA-aware allocation disabled")
endif()

target_link_libraries(llaminar2_core PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ftxui::screen
    ftxui::dom
    ftxui::component
    fort
)

# OneDNN for INT8 GEMM acceleration
if(USE_ONEDNN AND ONEDNN_FOUND)
    target_include_directories(llaminar2_core PUBLIC ${ONEDNN_INCLUDE_DIR})
    target_link_libraries(llaminar2_core PUBLIC ${ONEDNN_LIB})
    message(STATUS "V2: Linked OneDNN library for INT8 GEMM")
endif()

# BLAS libraries are linked above in the backend selection section

# =============================================================================
# Link GPU Backend Libraries (Phase 3)
# =============================================================================
if(HAVE_CUDA)
    target_link_libraries(llaminar2_core PUBLIC cuda_backend)
    
    # Add CUDA kernel C++ adapter files (compiled with C++ compiler, NOT nvcc)
    # These implement ITensorGemm interface and can include MPI headers
    # NOTE: CUDARingKVCacheTensorAdapter.cpp moved to cuda_backend to satisfy
    # template instantiation dependencies from CUDARingKVCache.cu
    target_sources(llaminar2_core PRIVATE
        kernels/cuda/gemm/CUDAFloatingPointGemmKernel.cpp
        kernels/cuda/gemm/CUDAWeightPacker.cpp
        kernels/cuda/gemm/CUDAQuantisedGemmKernel.cpp
        kernels/cuda/ops/CUDAOpsKernels.cpp           # RMSNorm, SwiGLU, RoPE, Embedding ITensor implementations
        kernels/cuda/attention/CUDAFlashAttentionKernelT.cpp  # Flash Attention ITensor implementation
    )
    
    message(STATUS "V2: Linked CUDA backend library")
endif()

if(HAVE_ROCM)
    target_link_libraries(llaminar2_core PUBLIC rocm_backend)
    
    # Add ROCm kernel C++ adapter files (compiled with C++ compiler, NOT hipcc)
    # These implement ITensorGemm interface and can include MPI headers
    target_sources(llaminar2_core PRIVATE
        kernels/rocm/ROCmWeightPacker.cpp               # Weight packing utilities (native-VNNI + INT8 conversion)
        kernels/rocm/gemm/ROCmFloatingPointGemmKernel.cpp
        kernels/rocm/gemm/ROCmQuantisedGemmKernel.cpp  # INT8 GEMM adapter (uses CK via extern "C")
        kernels/rocm/gemm/HipBLASGemmKernel.cpp        # hipBLAS GEMM for FP32/FP16 (compiled with g++, linked with hipBLAS)
        kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp  # Flash Attention ITensor implementation
        kernels/rocm/kvcache/ROCmRingKVCache.cpp              # Ring buffer KV cache implementation
        kernels/rocm/kvcache/ROCmRingKVCacheBase.cpp          # Shared base class for ring buffer KV caches
        kernels/rocm/kvcache/ROCmRingKVCacheFactory.cpp       # Factory for creating ROCm KV cache instances
        kernels/rocm/kvcache/ROCmHybridRingKVCacheFactory.cpp # Factory for creating ROCm Hybrid KV cache instances
        kernels/rocm/ops/ROCmSwiGLUKernelT.cpp                # SwiGLU ITensor implementation
        kernels/rocm/ops/ROCmRoPEKernelT.cpp                  # RoPE ITensor implementation
        kernels/rocm/ops/ROCmRMSNormKernelT.cpp               # RMSNorm ITensor implementation
        kernels/rocm/ops/ROCmEmbeddingKernelT.cpp             # Embedding ITensor implementation
    )
    
    message(STATUS "V2: Linked ROCm backend library with hipBLAS GEMM")
endif()

# Link device benchmarks library
target_link_libraries(llaminar2_core PUBLIC device_benchmarks)
message(STATUS "V2: Linked device benchmarks library")

if(HAVE_CUDA)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_CUDA)
    target_include_directories(llaminar2_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(llaminar2_core PUBLIC 
        CUDA::cudart 
        CUDA::cublas
        CUDA::cublasLt  # For fused GEMM+bias operations
        CUDA::nvml  # For GPU NUMA node detection
    )
    message(STATUS "V2: Linked CUDA runtime, cuBLAS, cuBLASLt, and NVML")
    
    # NOTE: NCCL is linked to cuda_backend (not llaminar2_core) to isolate symbols
    # from RCCL. Both export identical symbols and would conflict if in same library.
    # If HAVE_NCCL is set, cuda_backend already has NCCL linked via HAVE_NCCL check above
    if(HAVE_NCCL)
        # Only include headers for compilation - the library is linked via cuda_backend
        target_include_directories(llaminar2_core PUBLIC ${NCCL_INCLUDE_DIR})
        # DO NOT link NCCL library here - it comes via cuda_backend to prevent symbol conflicts
        message(STATUS "V2: NCCL headers available (library linked via cuda_backend)")
    endif()
endif()

# =============================================================================
# Link ROCm backend if enabled
# =============================================================================
if(HAVE_ROCM)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_ROCM)
    # NOTE: HIP runtime, hipBLAS, hipBLASLt, and ROCm include directories are
    # linked to rocm_backend (not llaminar2_core) to consolidate all HIP
    # dependencies in one place. They flow transitively via the PUBLIC link
    # from rocm_backend → llaminar2_core → executables.
    message(STATUS "V2: ROCm libraries available via rocm_backend (hip::host, hipBLAS, hipBLASLt)")
    
    # NOTE: RCCL is linked to rocm_backend (not llaminar2_core) to isolate symbols
    # from NCCL. Both libraries export the same symbols (ncclCommInitRank, etc.)
    # If HAVE_RCCL is set, rocm_backend already has RCCL linked via HAVE_RCCL check above
    if(HAVE_RCCL)
        target_compile_definitions(llaminar2_core PUBLIC HAVE_RCCL)
        target_include_directories(llaminar2_core PUBLIC ${RCCL_INCLUDE_DIR})
        # DO NOT link RCCL library here - it comes via rocm_backend to prevent symbol conflicts
        message(STATUS "V2: RCCL headers available (library linked via rocm_backend)")
    endif()
endif()

if(HAVE_VULKAN)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(llaminar2_core PUBLIC Vulkan::Vulkan)
endif()

# =============================================================================
# libmvec (GCC Vector Math Library for vectorized exp/log/sin/cos)
# =============================================================================
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # libmvec is part of glibc, link with -lmvec
    target_link_libraries(llaminar2_core PUBLIC mvec)
    message(STATUS "V2: Linked libmvec for vectorized math (exp, log, sin, cos)")
endif()

# =============================================================================
# OneDNN Configuration (INT8/BF16 optimization, replaces MKL)
# =============================================================================
# Link OneDNN if found
if(USE_ONEDNN AND ONEDNN_FOUND)
    target_include_directories(llaminar2_core PUBLIC ${ONEDNN_INCLUDE_DIR})
    target_link_libraries(llaminar2_core PUBLIC ${ONEDNN_LIB})
    message(STATUS "V2: Linked OneDNN for INT8/BF16 GEMM acceleration")
endif()

# Main executable
add_executable(llaminar2 Main.cpp)
target_link_libraries(llaminar2 llaminar2_core)

# GEMM tile-config sweep benchmark (CUDA only, optional)
if(HAVE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../benchmarks/v2/gemm_sweep_main.cu")
    add_executable(gemm_sweep ../../benchmarks/v2/gemm_sweep_main.cu)
    target_link_libraries(gemm_sweep PRIVATE llaminar2_core)
    target_include_directories(gemm_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(gemm_sweep PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Enable testing for V2
enable_testing()

# V2 Tests
# Include test subdirectory (unit, integration, e2e, pytorch_parity)
# Note: The source tree is at ../../tests/v2 relative to src/v2
add_subdirectory(../../tests/v2 ${CMAKE_BINARY_DIR}/tests/v2)

# TODO: GPU backends (CUDA, ROCm, Vulkan) - not yet implemented
# TODO: Intel MKL backend - not yet implemented

# Install targets
install(TARGETS llaminar2
    RUNTIME DESTINATION bin
)

install(FILES
    utils/MPIContext.h
    DESTINATION include/llaminar2/utils
)

install(FILES
    backends/ComputeBackend.h
    DESTINATION include/llaminar2/backends
)

install(FILES
    tensors/Tensors.h
    tensors/TensorKernels.h
    DESTINATION include/llaminar2/tensors
)

install(FILES
    models/QwenPipeline.h
    DESTINATION include/llaminar2/models
)

# =============================================================================
# CUDA GEMM Retraining Pipeline
# =============================================================================
if(HAVE_CUDA AND USE_ONNX_HEURISTIC)
    # Path relative to workspace root (CMAKE_SOURCE_DIR points to src/v2)
    set(PIPELINE_FILE "${CMAKE_SOURCE_DIR}/../../cmake/CudaGemmPipeline.cmake")
    if(EXISTS "${PIPELINE_FILE}")
        include("${PIPELINE_FILE}")
    else()
        message(WARNING "CUDA GEMM Pipeline file not found: ${PIPELINE_FILE}")
    endif()
endif()
