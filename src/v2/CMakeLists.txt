# Llaminar v2 - Clean greenfield implementation
cmake_minimum_required(VERSION 3.18)

# Enable native CPU optimizations (AVX512/AVX2/FMA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")

# Backend options (default to OFF - enable explicitly when ready)
option(HAVE_CUDA "Enable CUDA backend" OFF)
option(HAVE_ROCM "Enable ROCm backend" OFF)
option(HAVE_VULKAN "Enable Vulkan backend" OFF)

# CPU BLAS backend: Auto-select based on CPU vendor, or manual override
# Note: MKL and OpenBLAS cannot be linked together due to symbol conflicts
# We select one at CMake configure time based on CPU detection
set(BLAS_BACKEND "AUTO" CACHE STRING "BLAS backend: AUTO, MKL, or OPENBLAS")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS AUTO MKL OPENBLAS)

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
    tensors/FP32Tensor.cpp
    tensors/FP16Tensor.cpp
    tensors/BF16Tensor.cpp
    tensors/TensorFactory.cpp
    tensors/IQ4_NLTensor.cpp
    # Simple quantization formats (Q8_0, Q4_0, Q4_1)
    tensors/Q8_0Tensor.cpp
    tensors/Q4_0Tensor.cpp
    tensors/Q4_1Tensor.cpp
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
    kernels/cpu/QuantizedGemm.cpp
    kernels/cpu/FP32GemmKernel.cpp
    kernels/cpu/BF16GemmKernel.cpp
    kernels/cpu/CPURoPEKernel.cpp
    kernels/cpu/CPURMSNormKernel.cpp
    kernels/cpu/CPUSoftmaxKernel.cpp
    kernels/cpu/CPUSwiGLUKernel.cpp
    utils/ArgParser.cpp
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

# Add conditional backend support
if(HAVE_CUDA)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(llaminar2_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(HAVE_ROCM)
    target_compile_definitions(llaminar2_core PUBLIC HAVE_ROCM)
    # ROCm libraries will be found by parent CMakeLists.txt
    if(HIP_LIBRARY AND HIPBLAS_LIBRARY)
        target_link_libraries(llaminar2_core PUBLIC ${HIP_LIBRARY} ${HIPBLAS_LIBRARY})
    endif()
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
