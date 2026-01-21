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
# Automatic Linux Capability Assignment (PCIe BAR P2P support)
# =============================================================================
# This MUST be included early, before any add_executable() calls, because it
# overrides add_executable to automatically set CAP_SYS_ADMIN on all executables.
# This enables PCIe BAR direct P2P transfers between CUDA and ROCm GPUs.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AutoSetCapabilities.cmake)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "V2: Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
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
    
    # Relocatable device code for separate compilation
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
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
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a")
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
        kernels/cuda/CuBLASGemmKernel.cu              # cuBLAS wrapper for FP32/FP16/BF16 dense GEMM
        kernels/cuda/CUDAQuantisedGemmKernel_CUTLASS.cu  # CUTLASS INT8 GEMM implementation
        # Ops kernels (split from monolithic CUDAOpsKernels.cu)
        kernels/cuda/ops/CUDARMSNormKernels.cu        # RMSNorm kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDAResidualAddKernels.cu    # Residual Add kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDASwiGLUKernels.cu         # SwiGLU kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDARoPEKernels.cu           # RoPE kernels (FP32, BF16, FP16)
        kernels/cuda/ops/CUDAEmbeddingKernels.cu      # Embedding lookup kernel
        kernels/cuda/ops/CUDAVectorAddKernels.cu      # Vector add kernels (for collective reductions)
        kernels/cuda/attention/CUDAFlashAttentionKernels.cu  # Flash Attention 2 and Flash Decoding kernels
        kernels/cuda/kvcache/CUDARingKVCache.cu       # Ring buffer KV cache with O(1) eviction
    )
    
    # NOTE: CUDA kernel ITensorGemm adapters (CUDAFloatingPointGemmKernel.cpp, CUDAQuantisedGemmKernel.cpp)
    # are added to llaminar2_core with target_sources() below, compiled with regular C++ compiler (NOT nvcc)
    # because they include MPI headers which nvcc cannot process
    
    # CUDA backend library includes backend + kernels + enumeration
    add_library(cuda_backend STATIC 
        backends/cuda/CUDABackend.cu
        backends/cuda/CUDATensorValidation.cu
        backends/CUDAEnumeration.cu
        ${CUDA_KERNEL_SOURCES}
    )
    target_include_directories(cuda_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    
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
        
        # CUTLASS requires C++17 minimum, we use C++20
        target_compile_features(cuda_backend PUBLIC cxx_std_20)
        
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
endif()

# =============================================================================
# ComposableKernel (CK) Auto-Clone for ROCm INT8 GEMM
# =============================================================================
# CK is AMD's template-based GPU kernel library. We use a fork that includes
# fixes for gfx906 (MI50/MI60) GPUs which don't have MFMA instructions.
# See: https://github.com/ROCm/composable_kernel/pull/3593
if(HAVE_ROCM)
    find_package(Git REQUIRED)

    # Ensure CK exists in external/ and is on our fix branch
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(CK_EXTERNAL_DIR "${REPO_ROOT}/external/composable_kernel")
    set(CK_GIT_URL "https://github.com/dbsanfte/composable_kernel.git")
    # Track our fork's fix branch for gfx906 issues (OOB buffer loads, etc.)
    set(CK_GIT_REF "gfx-906-issues")

    if(NOT EXISTS "${REPO_ROOT}/external")
        file(MAKE_DIRECTORY ${REPO_ROOT}/external)
    endif()

    if(NOT EXISTS "${CK_EXTERNAL_DIR}")
        message(STATUS "V2: ComposableKernel not found in external/, cloning ${CK_GIT_URL} (${CK_GIT_REF})...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch ${CK_GIT_REF}
                ${CK_GIT_URL}
                composable_kernel
            WORKING_DIRECTORY ${REPO_ROOT}/external
            RESULT_VARIABLE GIT_CLONE_RESULT
            OUTPUT_VARIABLE GIT_CLONE_OUTPUT
            ERROR_VARIABLE GIT_CLONE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )

        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "V2: Failed to clone ComposableKernel: ${GIT_CLONE_ERROR}")
        endif()

        message(STATUS "V2: ComposableKernel ${CK_GIT_REF} cloned successfully")
    else()
        # Verify CK is on the expected branch (don't force-checkout, just warn if different)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${CK_EXTERNAL_DIR}
            RESULT_VARIABLE CK_BRANCH_RESULT
            OUTPUT_VARIABLE CK_CURRENT_BRANCH
            ERROR_VARIABLE CK_BRANCH_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(CK_BRANCH_RESULT EQUAL 0)
            if(NOT CK_CURRENT_BRANCH STREQUAL CK_GIT_REF)
                message(WARNING "V2: ComposableKernel is on branch '${CK_CURRENT_BRANCH}', expected '${CK_GIT_REF}'. "
                                "INT8 GEMM may not work correctly without the fix branch.")
            else()
                message(STATUS "V2: ComposableKernel already present at ${CK_EXTERNAL_DIR} on branch ${CK_GIT_REF}")
            endif()
        else()
            message(STATUS "V2: ComposableKernel already present at ${CK_EXTERNAL_DIR}; leaving existing checkout untouched")
        endif()
    endif()
endif()

# ROCm backend (separate .cpp compilation unit with HIP)
if(HAVE_ROCM)
    # Find ComposableKernel library (AMD's template-based GPU kernel library)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    find_package(ComposableKernel QUIET)
    
    # ROCm kernels compiled with hipcc
    set(ROCM_KERNEL_SOURCES
        kernels/rocm/ROCmQuantisedGemmKernel_CK.hip  # ComposableKernel INT8 GEMM implementation
        kernels/rocm/attention/ROCmFlashAttentionKernels.hip  # Flash Attention 2 + Flash Decoding for MI50
        kernels/rocm/kvcache/ROCmRingKVCacheKernels.hip       # Ring buffer KV cache kernels
        kernels/rocm/ops/ROCmSwiGLUKernels.hip               # SwiGLU kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmRoPEKernels.hip                 # RoPE kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmRMSNormKernels.hip              # RMSNorm kernels (FP32, BF16, FP16)
        kernels/rocm/ops/ROCmEmbeddingKernels.hip            # Embedding kernels (FP32, BF16, FP16, Q8_1)
        kernels/rocm/ops/ROCmResidualAddKernels.hip          # Residual Add kernels (FP32, BF16, FP16)
    )
    
    # ROCm enumeration and backend must be compiled with hipcc to use HIP runtime
    add_library(rocm_backend STATIC 
        backends/rocm/ROCmBackend.cpp
        backends/rocm/ROCmTensorValidation.cpp
        backends/ROCmEnumeration.cpp
        ${ROCM_KERNEL_SOURCES}
    )
    target_include_directories(rocm_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(rocm_backend PUBLIC hip::host roc::hipblas)
    
    # Link ComposableKernel if found
    if(ComposableKernel_FOUND)
        target_link_libraries(rocm_backend PUBLIC ComposableKernel::ComposableKernel)
        target_compile_definitions(rocm_backend PUBLIC HAVE_COMPOSABLE_KERNEL)
        message(STATUS "V2: ROCm backend using ComposableKernel for INT8 GEMM")
    else()
        message(WARNING "V2: ComposableKernel not found - ROCm INT8 GEMM will be disabled")
    endif()
    
    # Set HIP compiler for ROCm source files
    set_source_files_properties(
        backends/rocm/ROCmBackend.cpp
        backends/rocm/ROCmTensorValidation.cpp
        backends/ROCmEnumeration.cpp
        ${ROCM_KERNEL_SOURCES}
        PROPERTIES LANGUAGE HIP
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

# Add cross-vendor P2P if both CUDA and ROCm enabled
if(HAVE_CUDA AND HAVE_ROCM)
    list(APPEND BENCHMARK_SOURCES
        # Direct P2P via PCIe BAR mapping (no DMA-BUF, no fallback)
        backends/p2p/DirectP2P.cpp
        backends/p2p/DirectP2P_CUDA.cu
        backends/p2p/DirectP2P_ROCm.cpp
        # HostBackend runtime-specific helpers (must be in separate compilation units)
        collective/backends/HostBackendCUDA.cu
        collective/backends/HostBackendROCm.cpp
        # PCIe BAR collective backend for CUDA↔ROCm direct P2P
        collective/backends/PCIeBARBackend.cpp
    )
endif()

add_library(device_benchmarks STATIC ${BENCHMARK_SOURCES})
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
            backends/p2p/DirectP2P_ROCm.cpp
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
#   - BF16: bf16bf16f32 (used by BF16GemmKernel, replaces MKL)
#   - FP32: Baseline handled by OpenBLAS
if(USE_ONEDNN)
    find_package(Git REQUIRED)

    # First, ensure OneDNN exists in external/ and is on the expected revision
    # Note: Repository root is two levels up from src/v2
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    set(ONEDNN_EXTERNAL_DIR "${REPO_ROOT}/external/onednn")
    set(ONEDNN_GIT_URL "https://github.com/dbsanfte/oneDNN.git")
    # Track dbsanfte fork main branch so fused softmax patches are always available
    set(ONEDNN_GIT_REF "main")

    if(NOT EXISTS "${REPO_ROOT}/external")
        file(MAKE_DIRECTORY ${REPO_ROOT}/external)
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
        message(WARNING "V2: BF16: Falls back to FP32 expansion + OpenBLAS")
        set(USE_ONEDNN OFF)
    endif()
endif()

# Find Python3 for code generation scripts
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "V2: Found Python3: ${Python3_EXECUTABLE}")

# CPU BLAS backend: OpenBLAS for FP32 GEMM
# Note: OneDNN handles INT8/BF16 optimizations, OpenBLAS for FP32 baseline
message(STATUS "V2: BLAS backend: OpenBLAS (FP32 only)")

# Define GEMM script path for microkernel generation
# =============================================================================
# GEMM v1, v2, and v3 have been removed - only gemm_v4 (OneDNN-based) remains
# All legacy GEMM microkernel generation and integer GEMM implementations are obsolete
# =============================================================================

# =============================================================================
# llama.cpp Integration (REMOVED)
# =============================================================================
# llama.cpp submodule is kept in the repo for reference but not linked in V2.
# V2 uses native implementations for all operations (tokenization, dequantization, etc.)
# The submodule remains available at external/llama.cpp for:
#   - Reference implementations
#   - GGUF format documentation
#   - Quantization algorithm reference
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
# Typed Register Enforcement (JIT Code Quality)
# =============================================================================
# Enforce that JIT code uses the typed register system (RegisterGuard.h)
# instead of raw Xbyak register instantiations like Zmm(5) or Xmm(20).
# This prevents developers from bypassing the tracking system.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EnforceTypedRegisters.cmake)
enforce_typed_register_usage(${CMAKE_CURRENT_SOURCE_DIR})

# V2 Core Library
add_library(llaminar2_core STATIC
    # Interfaces
    interfaces/IMPIContext.cpp
    backends/ComputeBackend.cpp
    backends/BackendManager.cpp
    backends/CPUBackend.cpp
    # Execution framework (hierarchical work distribution)
    execution/WorkDistributor.cpp
    execution/DeviceContext.cpp
    # ComputeStage base classes and factory in compute_stages/
    execution/compute_stages/ComputeStageBase.cpp
    execution/compute_stages/ComputeStageFactory.cpp
    # Individual stages in compute_stages/stages/
    execution/compute_stages/stages/GEMMStage.cpp
    execution/compute_stages/stages/FusedQKVGEMMStage.cpp
    execution/compute_stages/stages/FusedGateUpGEMMStage.cpp
    execution/compute_stages/stages/RMSNormStage.cpp
    execution/compute_stages/stages/RoPEStage.cpp
    execution/compute_stages/stages/SwiGLUStage.cpp
    execution/compute_stages/stages/ResidualAddStage.cpp
    execution/compute_stages/stages/AllreduceStage.cpp
    execution/compute_stages/stages/AllGatherStage.cpp
    execution/compute_stages/stages/AllGatherVStage.cpp
    execution/compute_stages/stages/SendActivationsStage.cpp
    execution/compute_stages/stages/ReceiveActivationsStage.cpp
    execution/compute_stages/stages/MoEStages.cpp
    execution/compute_stages/stages/AttentionWithKVCacheStage.cpp
    execution/compute_stages/stages/KVCacheAppendStage.cpp
    execution/compute_stages/stages/KVCacheGatherStage.cpp
    execution/compute_stages/stages/AttentionComputeStage.cpp
    execution/compute_stages/stages/FusedAttentionWoStage.cpp
    execution/compute_stages/stages/EmbeddingStage.cpp
    execution/compute_stages/stages/LMHeadStage.cpp
    execution/GraphExecutor.cpp
    execution/GraphBufferManager.cpp
    execution/DomainAwareBufferManager.cpp
    execution/GraphResolver.cpp
    execution/LivenessAnalyzer.cpp
    execution/LayerExecutor.cpp
    execution/ModelExecutor.cpp
    execution/GraphOrchestrator.cpp
    execution/MultiDomainOrchestrator.cpp
    execution/HeterogeneousLayerExecutor.cpp
    execution/CrossDomainTransfer.cpp
    execution/HybridPrecisionConfig.cpp
    execution/InferenceMode.cpp
    execution/StageCoherence.cpp
    execution/DeviceWorkspaceManager.cpp
    # Tensor parallelism configuration (proportional sharding for heterogeneous TP)
    config/TensorParallelConfig.cpp
    # Pipeline parallelism configuration (layer range assignment for PP)
    config/PipelineParallelConfig.cpp
    # Layer placement configuration (layer-to-device assignment for CPU pipeline stages)
    config/LayerPlacementConfig.cpp
    # TP Domain configuration (multi-domain tensor parallelism for hybrid systems)
    config/TPDomain.cpp
    # Slab GEMM configuration (for chunked FP16 GEMM workspace management)
    kernels/SlabGemmConfig.cpp
    # Placement strategy (weight/compute distribution)
    execution/PlacementPlan.cpp
    execution/PlacementStrategy.cpp
    execution/placement/HeterogeneousMultiDomainStrategy.cpp
    # Inference dispatch (InferenceRunner decouples pipeline vs graph paths)
    execution/InferenceRunnerFactory.cpp
    loaders/ModelLoader.cpp
    loaders/ModelContext.cpp
    loaders/WeightManager.cpp
    loaders/WeightPlacementMap.cpp
    loaders/WeightPreloader.cpp
    loaders/DeviceOrchestrator.cpp
    loaders/LayerWeightStreamer.cpp
    loaders/WeightStreamerFactory.cpp
    models/qwen/Qwen2Graph.cpp
    tensors/cpu/CPUTensorBase.cpp
    tensors/cpu/FP32Tensor.cpp
    tensors/cpu/FP16Tensor.cpp
    tensors/cpu/BF16Tensor.cpp
    tensors/cpu/INT32Tensor.cpp
    tensors/cpu/INT8Tensor.cpp
    tensors/TensorFactory.cpp
    tensors/TensorValidation.cpp
    tensors/cpu/IQ4_NLTensor.cpp
    # Simple quantization formats (Q8_0, Q8_1, Q16_1, Q4_0, Q4_1, Q5_0, Q5_1)
    tensors/cpu/Q8_0Tensor.cpp
    tensors/cpu/Q8_1Tensor.cpp
    tensors/cpu/Q16_1Tensor.cpp
    tensors/cpu/Q4_0Tensor.cpp
    tensors/cpu/Q4_1Tensor.cpp
    tensors/cpu/Q5_0Tensor.cpp
    tensors/cpu/Q5_1Tensor.cpp
    # K-quant formats (Q6_K, Q2_K, Q5_K, Q3_K, Q4_K, Q8_K)
    tensors/cpu/Q6_KTensor.cpp
    tensors/cpu/Q2_KTensor.cpp
    tensors/cpu/Q5_KTensor.cpp
    tensors/cpu/Q3_KTensor.cpp
    tensors/cpu/Q4_KTensor.cpp
    tensors/cpu/Q8_KTensor.cpp
    # IQ formats (IQ4_XS, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, IQ3_S, IQ1_S, IQ1_M)
    tensors/cpu/IQ4_XSTensor.cpp
    tensors/cpu/IQ2_XXSTensor.cpp
    tensors/cpu/IQ2_XSTensor.cpp
    tensors/cpu/IQ3_XXSTensor.cpp
    tensors/cpu/IQ2_STensor.cpp
    tensors/cpu/IQ3_STensor.cpp
    tensors/cpu/IQ1_STensor.cpp
    tensors/cpu/IQ1_MTensor.cpp
    # KV Cache implementation (moved from tensors/cpu/)
    kernels/cpu/CPUKVCache.cpp
    # Vectorized primitives (ported from V1)
    kernels/cpu/primitives/RoPEPrimitives.cpp
    kernels/cpu/primitives/RMSNormPrimitives.cpp
    kernels/cpu/primitives/SoftmaxPrimitives.cpp
    kernels/cpu/primitives/SwiGLUPrimitives.cpp
    kernels/cpu/primitives/Q16HeadNormalization.cpp
    # CPU GEMM kernels and auto-tuning infrastructure
    # NOTE: Legacy GEMM implementations (v1/v2/v3) removed - only gemm_v4 (OneDNN) remains
    # All old GEMM kernels removed: GemmAutoTuner, SmartGemmSearch, QuantizedGemm,
    # BF16PackedGemm, FP16PackedGemm, Q8_1GemmKernelInit, VNNIGemmKernelInit
    kernels/cpu/ops/CPURoPEKernelT.cpp     # Typed RoPE with fused precision conversion (FP32/BF16/FP16/Q8_1)
    kernels/cpu/ops/CPURMSNormKernelT.cpp  # Typed RMSNorm with fused precision conversion (replaced CPURMSNormKernelT)
    kernels/cpu/ops/CPUSoftmaxKernelT.cpp  # Typed Softmax with native precision paths (FP32/BF16/FP16/Q8_1)
    kernels/cpu/ops/CPUSwiGLUKernelT.cpp   # Typed SwiGLU with native precision paths (FP32/BF16/FP16/Q8_1)
    kernels/cpu/ops/CPUEmbeddingKernelT.cpp    # Typed Embedding lookup with native precision paths (FP32/BF16/FP16/Q8_1)
    kernels/cpu/attention/CPUAttentionKernelT.cpp # Typed attention kernel (FP32/BF16/FP16/Q8_1)
    # Q8_1 fused attention + Wo projection (microkernels and composed kernels)
    kernels/cpu/attention/q8_1/microkernels/Q8DotProduct.cpp       # μK1: Q8_1 dot product with VNNI
    kernels/cpu/attention/q8_1/microkernels/OnlineSoftmax.cpp      # μK2: Streaming online softmax
    kernels/cpu/attention/q8_1/microkernels/VWeightedAccum.cpp     # μK3: Weighted V accumulation
    kernels/cpu/attention/q8_1/microkernels/WoProjection.cpp       # μK4: Wo output projection
    kernels/cpu/attention/q8_1/microkernels/FastExp.cpp            # μK5: Fast exponential approximation
    kernels/cpu/attention/q8_1/jit/WoProjectionVNNIPacked.cpp       # JIT helper thunk for VNNI-packed Wo GEMM
    kernels/cpu/attention/q8_1/FusedAttentionWoRef.cpp             # Reference implementation
    kernels/cpu/attention/q8_1/FusedAttentionWoTiled.cpp           # Cache-blocked tiled implementation
    # Q16_1 fused attention (TRUE INTEGER PIPELINE v2)
    kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.cpp        # ITensorFusedAttentionWo implementation
    kernels/cpu/attention/q16_1/ref/Q16IntegerAttentionRef.cpp     # v2: TRUE integer attention (replaces Q16FusedAttentionRef)
    kernels/cpu/attention/q16_1/ref/microkernels/Exp2Core.cpp            # v2: shared exp2 LUT primitives for all softmax impls
    kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmax.cpp    # v2: exp2 fixed-point softmax (INT32→INT16, 2048-entry LUT)
    kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProduct.cpp       # v2: Q×K^T integer dot product microkernel
    kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulate.cpp        # v2: P×V integer accumulation microkernel
    kernels/cpu/attention/q16_1/ref/microkernels/WoProjection.cpp        # v2: Wo projection INT32→Q16_1 microkernel
    kernels/cpu/attention/q16_1/ref/microkernels/OnlineSoftmax.cpp       # v2: FlashDecode online softmax microkernel
    kernels/cpu/attention/q16_1/ref/microkernels/MLAAttention.cpp        # v2: MLA (DeepSeek V3/Kimi K2) dual NOPE+ROPE attention
    # Fused kernels (operator fusion framework)
    kernels/cpu/gemm_v4/FusedGEMM.cpp           # Unified fused GEMM kernel (supports Gate/Up and Q/K/V patterns)
    # Kernel dispatch factory (centralized device-aware kernel creation)
    kernels/KernelFactory.cpp
    # Universal device kernel cache (hipBLAS, cuBLAS handles, etc.)
    kernels/DeviceKernelCache.cpp
    # Testing infrastructure
    testing/PerfHarness.cpp
    utils/ArgParser.cpp
    utils/BatchPaddingUtils.cpp
    utils/MPIBootstrap.cpp
    utils/NUMATopology.cpp
    utils/NodeTopology.cpp
    utils/MPITopology.cpp
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
    collective/backends/NCCLBackend.cpp
    collective/backends/RCCLBackend.cpp
    collective/backends/UPIBackend.cpp
    execution/CollectiveContext.cpp
    # NUMA-aware memory allocation
    memory/NUMAAllocator.cpp
    # NOTE: Legacy GEMM microkernel and instantiation systems removed (gemm v1/v2/v3)
    # GemmMicroKernelInit.cpp, Q8_1/VNNI instantiation sources - REMOVED
)

target_include_directories(llaminar2_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
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

# Compile features - C++20 for concepts, ranges, and improved constexpr
target_compile_features(llaminar2_core PUBLIC cxx_std_20)

# Dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
message(STATUS "V2: MPI enabled (required dependency)")
message(STATUS "V2: OpenBLAS configured for FP32 GEMM baseline")

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
    target_sources(llaminar2_core PRIVATE
        kernels/cuda/CUDAFloatingPointGemmKernel.cpp
        kernels/cuda/CUDAQuantisedGemmKernel.cpp
        kernels/cuda/kvcache/CUDARingKVCacheTensorAdapter.cpp # ICUDARingKVCache::append(TensorBase*) impl
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
        kernels/rocm/ROCmFloatingPointGemmKernel.cpp
        kernels/rocm/ROCmQuantisedGemmKernel.cpp  # INT8 GEMM adapter (uses CK via extern "C")
        kernels/rocm/HipBLASGemmKernel.cpp        # hipBLAS GEMM for FP32/FP16 (compiled with g++, linked with hipBLAS)
        kernels/rocm/attention/ROCmFlashAttentionKernelT.cpp  # Flash Attention ITensor implementation
        kernels/rocm/kvcache/ROCmRingKVCache.cpp              # Ring buffer KV cache implementation
        kernels/rocm/kvcache/ROCmRingKVCacheFactory.cpp       # Factory for creating ROCm KV cache instances
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
endif()

# =============================================================================
# Link ROCm backend if enabled
# =============================================================================
if(HAVE_ROCM)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_ROCM)
    target_include_directories(llaminar2_core PUBLIC 
        ${ROCM_PATH}/include
        ${ROCM_PATH}/hip/include
        ${ROCM_PATH}/hipblas/include
        ${ROCM_PATH}/hipblaslt/include
    )
    target_link_libraries(llaminar2_core PUBLIC 
        hip::host      # HIP runtime
        roc::hipblas   # hipBLAS
        ${ROCM_PATH}/lib/libhipblaslt.so  # hipBLASLt for fused GEMM+bias
    )
    message(STATUS "V2: Linked HIP runtime, hipBLAS, and hipBLASLt")
endif()

if(HAVE_VULKAN)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(llaminar2_core PUBLIC Vulkan::Vulkan)
endif()

# =============================================================================
# OpenBLAS Configuration (FP32 GEMM baseline)
# =============================================================================
if(NOT DEFINED OPENBLAS_LIBRARIES)
    find_package(BLAS REQUIRED)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include /usr/include/openblas /usr/local/include /usr/include/x86_64-linux-gnu)
    if(CBLAS_INCLUDE_DIR)
        message(STATUS "V2: Found CBLAS include: ${CBLAS_INCLUDE_DIR}")
        target_include_directories(llaminar2_core PUBLIC ${CBLAS_INCLUDE_DIR})
    else()
        message(WARNING "V2: CBLAS header not found - using fallback")
    endif()
    set(OPENBLAS_LIBRARIES ${BLAS_LIBRARIES})
    message(STATUS "V2: Using OpenBLAS: ${BLAS_LIBRARIES}")
endif()
target_compile_definitions(llaminar2_core PUBLIC HAVE_OPENBLAS)
target_link_libraries(llaminar2_core PUBLIC ${OPENBLAS_LIBRARIES})
message(STATUS "V2: OpenBLAS configured for FP32 GEMM baseline")

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

# =============================================================================
# llama.cpp Library (REMOVED - using native BPE tokenizer)
# =============================================================================
# NOTE: Previously linked llama.cpp for tokenization, but now we have
# native BPETokenizer that reads vocabulary directly from GGUF metadata.
# This eliminates duplicate file loading and external dependency.

# If llama.cpp is needed in future for other purposes, re-enable:
# target_link_libraries(llaminar2_core PUBLIC llama)
# target_include_directories(llaminar2_core PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../external/llama.cpp/include
# )
# message(STATUS "V2: Linked llama.cpp library for tokenization")

# Main executable
add_executable(llaminar2 Main.cpp)
target_link_libraries(llaminar2 llaminar2_core)

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
