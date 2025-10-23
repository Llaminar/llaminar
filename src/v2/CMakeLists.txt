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
    pipelines/QwenPipeline.cpp
    tensors/FP32Tensor.cpp
    tensors/IQ4_NLTensor.cpp
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
