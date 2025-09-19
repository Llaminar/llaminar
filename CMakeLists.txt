cmake_minimum_required(VERSION 3.16)
project(llaminar LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Find Google Test for unit testing
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Fallback to manual finding
    find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS /usr/include)
    find_library(GTEST_LIBRARY gtest PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
    find_library(GTEST_MAIN_LIBRARY gtest_main PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
    
    if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
        set(GTEST_FOUND TRUE)
        message(STATUS "Google Test found manually")
    else()
        message(WARNING "Google Test not found - kernel tests will be disabled")
        set(GTEST_FOUND FALSE)
    endif()
endif()

# Find BLAS/CBLAS for linear algebra operations
find_package(BLAS REQUIRED)
find_path(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include /usr/include/openblas)
if(NOT CBLAS_INCLUDE_DIR)
    message(WARNING "CBLAS header not found - using fallback")
    set(CBLAS_INCLUDE_DIR "")
endif()

# Check for NUMA support
find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARY numa)

if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
    set(NUMA_FOUND TRUE)
    message(STATUS "NUMA support found")
    add_definitions(-DHAVE_NUMA)
else()
    set(NUMA_FOUND FALSE)
    message(STATUS "NUMA support not found - continuing without it")
endif()

# Check for CUDA support (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(ENABLE_CUDA TRUE)
    message(STATUS "CUDA support found - enabling GPU detection")
    add_definitions(-DENABLE_CUDA)
else()
    set(ENABLE_CUDA FALSE)
    message(STATUS "CUDA support not found - GPU detection will be limited")
endif()

# Check for ROCm support (optional)
find_path(ROCM_INCLUDE_DIR hip/hip_runtime.h PATHS /opt/rocm/include)
find_library(HIP_LIBRARY hip PATHS /opt/rocm/lib64)
if(ROCM_INCLUDE_DIR AND HIP_LIBRARY)
    set(ENABLE_ROCM TRUE)
    message(STATUS "ROCm support found - enabling AMD GPU detection")
    add_definitions(-DENABLE_ROCM)
else()
    set(ENABLE_ROCM FALSE)
    message(STATUS "ROCm support not found - AMD GPU detection disabled")
endif()

# Configure COSMA submodule
message(STATUS "Configuring COSMA submodule...")
set(COSMA_BLAS "OPENBLAS" CACHE STRING "BLAS backend for COSMA")
set(COSMA_SCALAPACK "OFF" CACHE BOOL "Disable ScaLAPACK for COSMA")
set(COSMA_WITH_TESTS OFF CACHE BOOL "Build COSMA tests")
set(COSMA_WITH_APPS OFF CACHE BOOL "Build COSMA applications")  
set(COSMA_WITH_BENCHMARKS OFF CACHE BOOL "Build COSMA benchmarks")
set(COSMA_WITH_INSTALL OFF CACHE BOOL "Install COSMA")
set(COSMA_WITH_PROFILING OFF CACHE BOOL "Enable COSMA profiling")
set(COSMA_WITH_NCCL OFF CACHE BOOL "Disable NCCL for COSMA")
set(COSMA_WITH_GPU OFF CACHE BOOL "Disable GPU for COSMA")
add_subdirectory(COSMA)



# Source files for the core library
set(LLAMINAR_CORE_SOURCES
    src/common.cpp
    src/argument_parser.cpp
    src/topology_manager.cpp
    src/graph_compute.cpp
    src/kernels/MatMulKernel.cpp
    src/kernels/RMSNormKernel.cpp
    src/kernels/LinearKernel.cpp
    src/kernels/EmbeddingKernel.cpp
    src/kernels/MLPKernel.cpp
    src/kernels/AttentionKernel.cpp
    src/tensors/simple_tensor.cpp
    src/tensors/cosma_tensor.cpp
    src/tensors/tensor_factory.cpp
    src/model_loader.cpp
    src/repacker.cpp
)

# Create the core library
add_library(llaminar_core STATIC ${LLAMINAR_CORE_SOURCES})

# Link libraries to core
target_link_libraries(llaminar_core
    PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cosma
    ${BLAS_LIBRARIES}
)

# Add NUMA if available
if(NUMA_FOUND)
    target_include_directories(llaminar_core PRIVATE ${NUMA_INCLUDE_DIR})
    target_link_libraries(llaminar_core PRIVATE ${NUMA_LIBRARY})
endif()

# Add CUDA if available
if(ENABLE_CUDA)
    target_include_directories(llaminar_core PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(llaminar_core PRIVATE ${CUDA_LIBRARIES} ${CUDA_NVML_LIBRARY})
endif()

# Add ROCm if available
if(ENABLE_ROCM)
    target_include_directories(llaminar_core PRIVATE ${ROCM_INCLUDE_DIR})
    target_link_libraries(llaminar_core PRIVATE ${HIP_LIBRARY})
endif()

# Set compile features
target_compile_features(llaminar_core PUBLIC cxx_std_17)

# Include directories for core library
target_include_directories(llaminar_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/COSMA/src
    ${CBLAS_INCLUDE_DIR}
)

# Main executable
add_executable(llaminar src/main.cpp)
target_link_libraries(llaminar llaminar_core)

# Enable testing
enable_testing()

# Test sources and executables
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic llaminar_core)
add_test(NAME BasicTest COMMAND test_basic)

add_executable(test_numa tests/test_numa.cpp)
target_link_libraries(test_numa llaminar_core)
add_test(NAME NumaTest COMMAND test_numa)

add_executable(test_cosma tests/test_cosma.cpp)
target_link_libraries(test_cosma llaminar_core)
add_test(NAME CosmaTest COMMAND test_cosma)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph llaminar_core)
add_test(NAME GraphTest COMMAND test_graph)

# Test our custom transformer compute nodes
add_executable(test_transformer src/test_transformer.cpp)
target_link_libraries(test_transformer llaminar_core)
add_test(NAME TransformerTest COMMAND test_transformer)

# Kernel unit tests (if Google Test is available)
if(GTEST_FOUND)
    message(STATUS "Building kernel unit tests with Google Test")
    
    # RMSNorm kernel test
    add_executable(test_rmsnorm_kernel tests/test_rmsnorm_kernel.cpp)
    target_link_libraries(test_rmsnorm_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_rmsnorm_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_rmsnorm_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_rmsnorm_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME RMSNormKernelTest COMMAND test_rmsnorm_kernel)
    
    # Linear kernel test
    add_executable(test_linear_kernel tests/test_linear_kernel.cpp)
    target_link_libraries(test_linear_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_linear_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_linear_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_linear_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME LinearKernelTest COMMAND test_linear_kernel)
    
    # Embedding kernel test
    add_executable(test_embedding_kernel tests/test_embedding_kernel.cpp)
    target_link_libraries(test_embedding_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_embedding_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_embedding_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_embedding_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME EmbeddingKernelTest COMMAND test_embedding_kernel)
    
    # MLP kernel test
    add_executable(test_mlp_kernel tests/test_mlp_kernel.cpp)
    target_link_libraries(test_mlp_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mlp_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mlp_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mlp_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME MLPKernelTest COMMAND test_mlp_kernel)
    
    # Attention kernel test
    add_executable(test_attention_kernel tests/test_attention_kernel.cpp)
    target_link_libraries(test_attention_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_attention_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_attention_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME AttentionKernelTest COMMAND test_attention_kernel)
    
    set(KERNEL_TEST_TARGETS test_rmsnorm_kernel test_linear_kernel test_embedding_kernel test_mlp_kernel test_attention_kernel)
else()
    message(STATUS "Google Test not found - kernel unit tests disabled")
    set(KERNEL_TEST_TARGETS "")
endif()

# Set output directories
set_target_properties(llaminar
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(test_basic test_numa test_cosma test_graph test_transformer ${KERNEL_TEST_TARGETS}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Custom test target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_basic test_numa test_cosma test_graph test_transformer ${KERNEL_TEST_TARGETS}
    COMMENT "Running all tests"
)

# Print configuration summary
message(STATUS "=== Llaminar Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C standard: ${CMAKE_C_STANDARD}")
message(STATUS "MPI found: ${MPI_FOUND}")
message(STATUS "MPI include path: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_LIBRARIES}")
message(STATUS "OpenMP found: ${OpenMP_FOUND}")
message(STATUS "NUMA found: ${NUMA_FOUND}")
message(STATUS "Google Test found: ${GTEST_FOUND}")
message(STATUS "BLAS found: ${BLAS_FOUND}")
message(STATUS "CBLAS include: ${CBLAS_INCLUDE_DIR}")
message(STATUS "COSMA BLAS backend: ${COSMA_BLAS}")
message(STATUS "COSMA ScaLAPACK backend: ${COSMA_SCALAPACK}")
message(STATUS "===============================")
