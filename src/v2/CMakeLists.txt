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

# Enable native CPU optimizations (AVX512/AVX2/FMA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")

# Backend options (temporarily disabled due to header conflicts - will enable separately)
# CUDA and ROCm cannot be linked together due to type conflicts in headers
# We'll add separate GPU kernel compilation units later
option(HAVE_CUDA "Enable CUDA backend" OFF)
option(HAVE_ROCM "Enable ROCm backend" OFF)
option(HAVE_VULKAN "Enable Vulkan backend" OFF)

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
    
    # CUDA compile flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -gencode=arch=compute_70,code=sm_70")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_86,code=sm_86")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_89,code=sm_89")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90")
    
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
    
    # Find HIP
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    
    message(STATUS "V2: ROCm enabled")
    message(STATUS "V2: ROCm Path: ${ROCM_PATH}")
    message(STATUS "V2: HIP Version: ${hip_VERSION}")
    message(STATUS "V2: hipcc: ${HIP_HIPCC_EXECUTABLE}")
    
    # ROCm compile flags (support common AMD architectures)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx900")  # Vega 10
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx906")  # Vega 20
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx908")  # MI100
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx90a")  # MI200
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx940")  # MI300
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx1030") # Navi 21
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx1100") # Navi 31
endif()

# Find Python3 for code generation scripts
find_package(Python3 COMPONENTS Interpreter REQUIRED)
message(STATUS "V2: Found Python3: ${Python3_EXECUTABLE}")

# CPU BLAS backend: Auto-select based on CPU vendor, or manual override
# Note: MKL and OpenBLAS cannot be linked together due to symbol conflicts
# We select one at CMake configure time based on CPU detection
set(BLAS_BACKEND "AUTO" CACHE STRING "BLAS backend: AUTO, MKL, or OPENBLAS")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS AUTO MKL OPENBLAS)

# Generate microkernel instantiations (auto-run Python script at configure time)
set(GENERATE_GEMM_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu/generate_gemm_microkernel_instantiations.py")
set(GENERATED_SOURCES_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu/generated/sources.cmake")

# Check if generation is needed (script newer than output, or output doesn't exist)
if(NOT EXISTS ${GENERATED_SOURCES_CMAKE} OR 
   ${GENERATE_GEMM_SCRIPT} IS_NEWER_THAN ${GENERATED_SOURCES_CMAKE})
    message(STATUS "V2: Generating GEMM microkernel instantiations...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${GENERATE_GEMM_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu
        RESULT_VARIABLE GENERATE_RESULT
        OUTPUT_VARIABLE GENERATE_OUTPUT
        ERROR_VARIABLE GENERATE_ERROR
    )
    
    if(NOT GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "V2: Failed to generate GEMM microkernels:\n${GENERATE_ERROR}")
    endif()
    
    message(STATUS "V2: Generated 64 microkernel files with 1225 variants")
else()
    message(STATUS "V2: GEMM microkernel instantiations are up-to-date")
endif()

# Include generated microkernel instantiations
include(${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu/generated/sources.cmake)

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

# V2 Core Library
add_library(llaminar2_core STATIC
    backends/ComputeBackend.cpp
    loaders/ModelLoader.cpp
    loaders/ModelContext.cpp
    loaders/WeightManager.cpp
    loaders/WeightPlacementMap.cpp
    loaders/DeviceOrchestrator.cpp
    pipelines/PipelineBase.cpp
    pipelines/PipelineFactory.cpp
    pipelines/qwen/Qwen2Pipeline.cpp
    pipelines/attention/GQAAttention.cpp
    tensors/TensorBase.cpp
    tensors/FP32Tensor.cpp
    tensors/FP16Tensor.cpp
    tensors/BF16Tensor.cpp
    tensors/TensorFactory.cpp
    tensors/KVCache.cpp
    tensors/IQ4_NLTensor.cpp
    # Simple quantization formats (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1)
    tensors/Q8_0Tensor.cpp
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
    tensors/BatchedKVCache.cpp
    # Vectorized primitives (ported from V1)
    kernels/cpu/primitives/RoPEPrimitives.cpp
    kernels/cpu/primitives/RMSNormPrimitives.cpp
    kernels/cpu/primitives/SoftmaxPrimitives.cpp
    # CPU kernels and auto-tuning infrastructure
    kernels/cpu/GemmAutoTuner.cpp         # Auto-tuner singleton + factory function
    kernels/cpu/SmartGemmSearch.cpp       # Intelligent hierarchical search strategies
    # kernels/cpu/GemmVariantGenerator.cpp  # OBSOLETE: Replaced by MicroKernelRegistry (backed up as .bak)
    # kernels/cpu/GemmVariantGenerator_Generated.cpp # OBSOLETE: Replaced by MicroKernelRegistry (backed up as .bak)
    kernels/cpu/FP32GemmKernel.cpp
    kernels/cpu/FP32StandaloneGemm.cpp
    kernels/cpu/BF16GemmKernel.cpp
    kernels/cpu/CPURoPEKernel.cpp
    kernels/cpu/CPURMSNormKernel.cpp
    kernels/cpu/CPUSoftmaxKernel.cpp
    kernels/cpu/CPUSwiGLUKernel.cpp
    kernels/cpu/CPUAttention.cpp
    # Testing infrastructure (Phase 3a & 3b)
    testing/MockPipeline.cpp
    testing/AttentionTestHarness.cpp
    testing/PerfHarness.cpp
    utils/ArgParser.cpp
    utils/BatchPaddingUtils.cpp
    utils/NUMATopology.cpp
    utils/MPIStager.cpp
    # Generated GEMM microkernel template instantiations (1225 total across 64 files)
    kernels/cpu/GemmMicroKernelInit.cpp
    ${MICROKERNEL_INSTANTIATION_SOURCES}
)

target_include_directories(llaminar2_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compile features
target_compile_features(llaminar2_core PUBLIC cxx_std_17)

# Dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# ============================================================================
# BLAS Backend Selection (MKL vs OpenBLAS)
# ============================================================================
# Detect CPU vendor to choose optimal BLAS backend
if(BLAS_BACKEND STREQUAL "AUTO")
    # Detect CPU vendor at configure time
    execute_process(
        COMMAND bash -c "lscpu | grep 'Vendor ID:' | awk '{print $3}'"
        OUTPUT_VARIABLE CPU_VENDOR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(CPU_VENDOR STREQUAL "GenuineIntel")
        message(STATUS "V2: Detected Intel CPU - preferring Intel MKL")
        set(BLAS_BACKEND_SELECTED "MKL")
    else()
        message(STATUS "V2: Detected non-Intel CPU (${CPU_VENDOR}) - preferring OpenBLAS")
        set(BLAS_BACKEND_SELECTED "OPENBLAS")
    endif()
else()
    set(BLAS_BACKEND_SELECTED ${BLAS_BACKEND})
    message(STATUS "V2: Manual BLAS backend selection: ${BLAS_BACKEND_SELECTED}")
endif()

# Find and link the selected BLAS backend
if(BLAS_BACKEND_SELECTED STREQUAL "MKL")
    # Try to find Intel MKL
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "V2: Intel MKL found - using MKL backend")
        target_compile_definitions(llaminar2_core PUBLIC HAVE_MKL)
        target_include_directories(llaminar2_core PUBLIC ${MKL_INCLUDE})
        target_link_libraries(llaminar2_core PUBLIC MKL::MKL)
        set(BLAS_LIBRARIES_SELECTED ${MKL_LIBRARIES})
    else()
        message(WARNING "V2: Intel MKL requested but not found - falling back to OpenBLAS")
        message(WARNING "V2: Install MKL with: sudo apt install intel-oneapi-mkl-devel")
        set(BLAS_BACKEND_SELECTED "OPENBLAS")
    endif()
endif()

if(BLAS_BACKEND_SELECTED STREQUAL "OPENBLAS" OR NOT MKL_FOUND)
    # Find OpenBLAS
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
    set(BLAS_LIBRARIES_SELECTED ${OPENBLAS_LIBRARIES})
endif()

message(STATUS "V2: Selected BLAS backend: ${BLAS_BACKEND_SELECTED}")

# Try to find libnuma (optional, for NUMA-aware allocation)
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
)

# BLAS libraries are linked above in the backend selection section

# =============================================================================
# Link CUDA backend if enabled
# =============================================================================
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

if(HAVE_MKL)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_MKL)
endif()

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
    pipelines/QwenPipeline.h
    DESTINATION include/llaminar2/pipelines
)
