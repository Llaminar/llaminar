# Llaminar v2 - Clean greenfield implementation
cmake_minimum_required(VERSION 3.18)

# Backend options (default to OFF - enable explicitly when ready)
option(HAVE_CUDA "Enable CUDA backend" OFF)
option(HAVE_ROCM "Enable ROCm backend" OFF)
option(HAVE_VULKAN "Enable Vulkan backend" OFF)
option(HAVE_MKL "Enable Intel MKL" OFF)

# V2 Core Library
add_library(llaminar2_core STATIC
    backends/ComputeBackend.cpp
    loaders/ModelLoader.cpp
    loaders/ModelContext.cpp
    loaders/WeightManager.cpp
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
)

target_include_directories(llaminar2_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Compile features
target_compile_features(llaminar2_core PUBLIC cxx_std_17)

# Dependencies
find_package(MPI REQUIRED)
# OpenBLAS already found in parent CMakeLists.txt

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
    ${OPENBLAS_LIBRARIES}
)

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
