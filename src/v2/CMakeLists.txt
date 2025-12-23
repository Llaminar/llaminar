# Llaminar v2 - Clean greenfield implementation
cmake_minimum_required(VERSION 3.18)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "V2: Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "V2: ccache not found - using direct compilation")
endif()

# Define custom build type: E2ERelease (Release + Snapshots for E2E testing)
# Usage: cmake -DCMAKE_BUILD_TYPE=E2ERelease
set(CMAKE_CXX_FLAGS_E2ERELEASE "-O3 -DNDEBUG" CACHE STRING "Flags for E2E Release builds")
set(CMAKE_C_FLAGS_E2ERELEASE "-O3 -DNDEBUG" CACHE STRING "Flags for E2E Release builds")
set(CMAKE_EXE_LINKER_FLAGS_E2ERELEASE "" CACHE STRING "Linker flags for E2E Release builds")
mark_as_advanced(CMAKE_CXX_FLAGS_E2ERELEASE CMAKE_C_FLAGS_E2ERELEASE CMAKE_EXE_LINKER_FLAGS_E2ERELEASE)

# Add E2ERelease to valid build types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;E2ERelease" CACHE STRING "" FORCE)

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
    
    # ROCm compile flags (support common AMD architectures)
    # MI50 = gfx906
    set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx1030;gfx1100")
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
    
    # CUDA kernels (JIT-based compilation)
    set(CUDA_KERNEL_SOURCES
        kernels/cuda/IQ4_NL_BlockDecoder.cu
        kernels/cuda/CudaGemmVariantsBaseline.cu      # Template definitions + JIT launcher
        kernels/cuda/CudaGemmVariantsMemoryOpt.cu     # Memory-optimized kernels (Phase 1)
        # kernels/cuda/CudaGemmVariantsTensorCore.cu  # OBSOLETE: Unused CuTe-based Tensor Core kernels (replaced by JIT)
        kernels/cuda/CudaGemmJIT.cu                   # JIT compiler implementation (NVRTC)
        kernels/cuda/CudaGemmAutoTuner.cu
        kernels/cuda/CudaGemmFactory.cu
        kernels/cuda/CudaGemmCuTeWrapper.cu           # CuTe Tensor Core wrapper (Phase 1)
        kernels/cuda/CudaGemmKernelPhase3.cu          # Phase 3 Part 1: Large tiles (128×128×64)
        kernels/cuda/CudaGemmKernelPhase3Pipelined.cu # Phase 3 Part 2: Pipelined (900-1,000 GFLOPS target)
        kernels/cuda/CudaGemmKernelPhase4QuickWins.cu # Phase 4: Swizzled + cp.async (7.6 TFLOPS target)
        kernels/cuda/CudaGemmKernelPhase5ASimple.cu   # Phase 5A: Streaming dequant (10.0 TFLOPS target)
        kernels/cuda/CudaGemmJITPhase5.cu             # Phase 5 JIT: Auto-tuning infrastructure
        kernels/cuda/CudaGemmKernelPhase7_CUTLASS.cu  # Phase 7: CUTLASS int8 GEMM (50-100+ TFLOPS target)
    )
    
    # Add neural network if ONNX Runtime enabled
    if(USE_ONNX_HEURISTIC AND ONNXRUNTIME_INCLUDE_DIR)
        list(APPEND CUDA_KERNEL_SOURCES kernels/cuda/CudaGemmNeuralNetwork.cu)
        message(STATUS "V2: Neural network source added to CUDA backend")
    endif()
    
    # CUDA backend library includes backend + kernels + enumeration
    add_library(cuda_backend STATIC 
        backends/cuda/CUDABackend.cu
        backends/CUDAEnumeration.cu
        ${CUDA_KERNEL_SOURCES}
    )
    target_include_directories(cuda_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    
    # Link CUDA libraries:
    # - cudart: CUDA Runtime API (for memory management, streams, etc.)
    # - nvrtc: NVRTC runtime compiler (for JIT kernel compilation)
    # - cuda: CUDA Driver API (for cuLaunchKernel, cuModuleLoad, etc.)
    target_link_libraries(cuda_backend PUBLIC 
        CUDA::cudart
        CUDA::nvrtc
        CUDA::cuda_driver
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
    
    # Define HAVE_CUDA for MPIStager and other consumers
    add_compile_definitions(HAVE_CUDA)
    
    message(STATUS "V2: CUDA backend library configured with ${CMAKE_CURRENT_LIST_LENGTH} kernel(s)")
endif()

# ROCm backend (separate .cpp compilation unit with HIP)
if(HAVE_ROCM)
    # ROCm enumeration must be compiled with hipcc to use HIP runtime
    add_library(rocm_backend STATIC 
        backends/rocm/ROCmBackend.cpp
        backends/ROCmEnumeration.cpp
    )
    target_include_directories(rocm_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(rocm_backend PUBLIC hip::host roc::hipblas)
    
    # Set HIP compiler for ROCm source files
    set_source_files_properties(
        backends/rocm/ROCmBackend.cpp
        backends/ROCmEnumeration.cpp
        PROPERTIES LANGUAGE HIP
    )
    
    # Define HAVE_ROCM for MPIStager and other consumers
    add_compile_definitions(HAVE_ROCM)
    
    message(STATUS "V2: ROCm backend library configured")
endif()

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
# llama.cpp Integration (for dequant parity testing)
# =============================================================================
message(STATUS "V2: Configuring llama.cpp submodule for dequant reference...")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Disable llama.cpp CLI tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama.cpp server" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Disable llama.cpp common utils" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "Disable llama.cpp libcurl dependency" FORCE)
set(GGML_CUDA OFF CACHE BOOL "Disable GGML CUDA backend" FORCE)
set(GGML_METAL OFF CACHE BOOL "Disable GGML Metal backend" FORCE)
set(GGML_SYCL OFF CACHE BOOL "Disable GGML SYCL backend" FORCE)
set(GGML_RPC OFF CACHE BOOL "Disable GGML RPC backend" FORCE)
set(GGML_NATIVE ON CACHE BOOL "Enable native CPU optimizations for GGML" FORCE)

# Only add llama.cpp if not already added by parent project
if(NOT TARGET ggml)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../external/llama.cpp 
                     ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)
endif()

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

# V2 Core Library
add_library(llaminar2_core STATIC
    backends/ComputeBackend.cpp
    backends/BackendManager.cpp
    # Execution framework (hierarchical work distribution)
    execution/WorkDistributor.cpp
    execution/DeviceContext.cpp
    execution/ComputeStage.cpp
    execution/GraphExecutor.cpp
    execution/GraphBufferManager.cpp
    execution/GraphResolver.cpp
    execution/LivenessAnalyzer.cpp
    execution/LayerExecutor.cpp
    execution/ModelExecutor.cpp
    execution/GraphOrchestrator.cpp
    # Placement strategy (weight/compute distribution)
    execution/PlacementPlan.cpp
    execution/PlacementStrategy.cpp
    # Inference dispatch (InferenceRunner decouples pipeline vs graph paths)
    execution/InferenceRunnerFactory.cpp
    loaders/ModelLoader.cpp
    loaders/ModelContext.cpp
    loaders/WeightManager.cpp
    loaders/WeightPlacementMap.cpp
    loaders/DeviceOrchestrator.cpp
    models/qwen/Qwen2Graph.cpp
    tensors/TensorBase.cpp
    tensors/FP32Tensor.cpp
    tensors/FP16Tensor.cpp
    tensors/BF16Tensor.cpp
    tensors/INT32Tensor.cpp
    tensors/INT8Tensor.cpp
    tensors/TensorFactory.cpp
    tensors/IQ4_NLTensor.cpp
    # Simple quantization formats (Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1)
    tensors/Q8_0Tensor.cpp
    tensors/Q8_1Tensor.cpp
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
    tensors/UnifiedKVCache.cpp
    # Vectorized primitives (ported from V1)
    kernels/cpu/primitives/RoPEPrimitives.cpp
    kernels/cpu/primitives/RMSNormPrimitives.cpp
    kernels/cpu/primitives/SoftmaxPrimitives.cpp
    kernels/cpu/primitives/SwiGLUPrimitives.cpp
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
    kernels/cpu/attention/q8_1/FusedAttentionWoRef.cpp             # Reference implementation
    kernels/cpu/attention/q8_1/FusedAttentionWoTiled.cpp           # Cache-blocked tiled implementation
    # Fused kernels (operator fusion framework)
    kernels/cpu/gemm_v4/FusedGEMM.cpp           # Unified fused GEMM kernel (supports Gate/Up and Q/K/V patterns)
    # Kernel dispatch factory (centralized device-aware kernel creation)
    kernels/KernelFactory.cpp
    # Testing infrastructure
    testing/PerfHarness.cpp
    utils/ArgParser.cpp
    utils/BatchPaddingUtils.cpp
    utils/MPIBootstrap.cpp
    utils/NUMATopology.cpp
    utils/MPITopology.cpp
    utils/MPIStager.cpp
    utils/Tokenizer.cpp
    utils/Sampler.cpp
    utils/ChatTemplate.cpp
    utils/ChatUI.cpp
    utils/BenchmarkRunner.cpp
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
# - Release builds: Disabled by default (performance), enable with -DENABLE_SNAPSHOTS=ON
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (Debug build)")
elseif(CMAKE_BUILD_TYPE MATCHES "E2ERelease")
    target_compile_definitions(llaminar2_core PUBLIC ENABLE_PIPELINE_SNAPSHOTS)
    message(STATUS "V2: Pipeline snapshot capture enabled (E2ERelease build - optimized with snapshots for E2E testing)")
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

# Compile features
target_compile_features(llaminar2_core PUBLIC cxx_std_17)

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
    message(STATUS "V2: Linked CUDA backend library")
endif()

if(HAVE_ROCM)
    target_link_libraries(llaminar2_core PUBLIC rocm_backend)
    message(STATUS "V2: Linked ROCm backend library")
endif()
if(HAVE_CUDA)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_CUDA)
    target_include_directories(llaminar2_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(llaminar2_core PUBLIC 
        CUDA::cudart 
        CUDA::cublas
        CUDA::nvml  # For GPU NUMA node detection
    )
    message(STATUS "V2: Linked CUDA runtime, cuBLAS, and NVML")
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
    )
    target_link_libraries(llaminar2_core PUBLIC 
        hip::host      # HIP runtime
        roc::hipblas   # hipBLAS
    )
    message(STATUS "V2: Linked HIP runtime and hipBLAS")
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

# Profiling executable (for ncu profiling data collection)
if(HAVE_CUDA)
    add_executable(profile_cuda_config kernels/cuda/ProfileSingleConfig.cu)
    target_link_libraries(profile_cuda_config cuda_backend CUDA::cudart)
    target_include_directories(profile_cuda_config PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "V2: Added profile_cuda_config executable for profiling data collection")
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
