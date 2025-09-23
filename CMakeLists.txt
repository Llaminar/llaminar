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
# Note: Removed -DDEBUG from global flags to prevent COSMA verbose output
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
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
    src/mpi_kernel_base.cpp
    src/cosma_prefill_manager.cpp
    src/kernels/MPILinearKernel.cpp
    src/kernels/MPIRMSNormKernel.cpp
    src/kernels/MPIAttentionKernel.cpp
    src/kernels/MPIEmbeddingKernel.cpp
    src/kernels/MPIMLPKernel.cpp
    src/kernels/MPIResidualKernel.cpp
    src/kernels/MPIRoPEKernel.cpp
    src/kernels/MPISwiGLUKernel.cpp
    src/mpi_transformer_pipeline.cpp
    src/pipeline_base.cpp
    src/tensors/simple_tensor.cpp
    src/tensors/cosma_tensor.cpp
    src/tensors/tensor_factory.cpp
    src/model_loader.cpp
    src/repacker.cpp
    src/socket_aware_inference.cpp
    # Adaptive architecture components
    src/adaptive_matmul.h
    src/adaptive_transformer_pipeline.h
    # Chat interface components
    src/chat/gguf_tokenizer.cpp
    src/chat/bpe_processor.cpp
    src/chat/chat_session.cpp
    src/chat/response_generator.cpp
    src/chat/chat_interface.cpp
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

# Add DEBUG flag only to our targets (not COSMA) when in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(llaminar_core PRIVATE DEBUG)
endif()

# Include directories for core library
target_include_directories(llaminar_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/COSMA/src
    ${CBLAS_INCLUDE_DIR}
)

# Main executable
add_executable(llaminar src/main.cpp)
target_link_libraries(llaminar llaminar_core)

# Adaptive architecture demonstration
add_executable(adaptive_demo src/adaptive_demo.cpp)
target_link_libraries(adaptive_demo llaminar_core)

# Production adaptive architecture demonstration
add_executable(production_demo src/production_demo.cpp)
target_link_libraries(production_demo llaminar_core)

# Add DEBUG flag to main executable when in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(llaminar PRIVATE DEBUG)
    target_compile_definitions(adaptive_demo PRIVATE DEBUG)
    target_compile_definitions(production_demo PRIVATE DEBUG)
endif()

# Enable testing
enable_testing()

# Minimal ggml reference library for golden quant/dequant (Option A)
add_library(ggml_min STATIC
    llama.cpp/ggml/src/ggml.c
    llama.cpp/ggml/src/ggml-quants.c
    tests/ggml_backend_stubs.c
)
target_include_directories(ggml_min PUBLIC
    llama.cpp/ggml/src
    llama.cpp/ggml/include
    llama.cpp/ggml
)
target_compile_definitions(ggml_min PRIVATE GGML_USE_FAKE_DEPS=1)
target_compile_definitions(ggml_min PRIVATE
    GGML_VERSION="0.0.0"
    GGML_COMMIT="llaminar-min"
    GGML_USE_K_QUANTS=1
    GGML_USE_QKK_64=1
)


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

# Kernel unit tests (if Google Test is available)
if(GTEST_FOUND)
    message(STATUS "Building kernel unit tests with Google Test")
    

    

    

    

    

    
    # MPI linear kernel test
    add_executable(test_mpi_linear_kernel tests/test_mpi_linear_kernel.cpp)
    target_link_libraries(test_mpi_linear_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_linear_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_linear_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_linear_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME MPILinearKernelTest COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/tests/test_mpi_linear_kernel)

    add_executable(test_mpi_rmsnorm_kernel tests/test_mpi_rmsnorm_kernel.cpp)
    target_link_libraries(test_mpi_rmsnorm_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_rmsnorm_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_rmsnorm_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_rmsnorm_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME MPIRMSNormKernelTest COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/tests/test_mpi_rmsnorm_kernel)

    add_executable(test_mpi_attention_kernel tests/test_mpi_attention_kernel.cpp)
    target_link_libraries(test_mpi_attention_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_attention_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_attention_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_attention_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME MPIAttentionKernelTest COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/tests/test_mpi_attention_kernel)
    
    # MPI Transformer Pipeline test
    add_executable(test_mpi_transformer_pipeline tests/test_mpi_transformer_pipeline.cpp)
    target_link_libraries(test_mpi_transformer_pipeline llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_transformer_pipeline GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_transformer_pipeline PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_transformer_pipeline ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME MPITransformerPipelineTest COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/tests/test_mpi_transformer_pipeline)
    # Ensure this long-running MPI test cannot hang indefinitely; internal watchdog < 60s, external timeout at 60s
    set_tests_properties(MPITransformerPipelineTest PROPERTIES TIMEOUT 60)
    
    # Adaptive Matrix Multiplication test
    # Shared logging bootstrap (object library) to unify test logging levels
    add_library(test_logging_bootstrap OBJECT tests/test_logging_bootstrap.cpp)
    
    add_executable(test_adaptive_matmul tests/test_adaptive_matmul.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_adaptive_matmul llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_adaptive_matmul GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_adaptive_matmul PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_adaptive_matmul ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME AdaptiveMatMulTest COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/tests/test_adaptive_matmul)
    
    # Minimal COSMA Custom Layout test
    add_executable(test_cosma_minimal tests/test_cosma_minimal.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_minimal llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_minimal GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_minimal PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_minimal ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaMinimalTest COMMAND mpirun -np 2 ${CMAKE_BINARY_DIR}/tests/test_cosma_minimal)

    # Cosma Prefill Manager correctness (single rank baseline)
    add_executable(test_cosma_prefill_manager tests/test_cosma_prefill_manager.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager)
    set_tests_properties(CosmaPrefillManagerTest PROPERTIES TIMEOUT 60)

    add_executable(test_cosma_prefill_manager_multi tests/test_cosma_prefill_manager_multi.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_multi PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_test(NAME CosmaPrefillManagerMultiTest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_cosma_prefill_manager_multi>)
    set_tests_properties(CosmaPrefillManagerMultiTest PROPERTIES TIMEOUT 60)

    # Cosma Prefill Manager distributed full-path (large GEMM) correctness test
    add_executable(test_cosma_prefill_manager_distributed tests/test_cosma_prefill_manager_distributed.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_distributed PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_test(NAME CosmaPrefillManagerDistributedFullTest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_cosma_prefill_manager_distributed>)
    set_tests_properties(CosmaPrefillManagerDistributedFullTest PROPERTIES TIMEOUT 60)

    # Population regression (destination-local round-trip invariants)
    add_executable(test_cosma_prefill_population_regression tests/test_cosma_prefill_population_regression.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_population_regression PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_test(NAME CosmaPrefillManagerPopulationRegressionTest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_cosma_prefill_population_regression>)
    set_tests_properties(CosmaPrefillManagerPopulationRegressionTest PROPERTIES TIMEOUT 60)

    # Population performance benchmark (env-gated)
    add_executable(test_cosma_prefill_population_perf tests/test_cosma_prefill_population_perf.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_population_perf PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_test(NAME CosmaPrefillManagerPopulationPerfTest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_cosma_prefill_population_perf>)
    set_tests_properties(CosmaPrefillManagerPopulationPerfTest PROPERTIES TIMEOUT 120)

    # Dequantization tests (Q4_0 correctness + Q4_K placeholder sanity)
    add_executable(test_dequant tests/test_dequant.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_dequant llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_dequant GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant)

    # Extra quantization tests (all types)
    add_executable(test_dequant_extra tests/test_dequant_extra.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_dequant_extra llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_dequant_extra GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant_extra PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant_extra ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantExtraTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant_extra)

    # Golden reference comparison test (ggml quant vs llaminar dequant)
    add_executable(test_dequant_golden tests/test_dequant_golden.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_include_directories(test_dequant_golden PRIVATE
        llama.cpp/ggml/src
        llama.cpp/ggml/include
        llama.cpp/ggml
    )
    target_link_libraries(test_dequant_golden llaminar_core ggml_min)
    if(GTest_FOUND)
        target_link_libraries(test_dequant_golden GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant_golden PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant_golden ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantGoldenTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant_golden)

    # Integration test for real Q4_K tensors (statistical sanity)
    add_executable(test_integration_q4k tests/test_integration_q4k.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_integration_q4k llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_integration_q4k GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_integration_q4k PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_integration_q4k ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME IntegrationQ4K COMMAND ${CMAKE_BINARY_DIR}/tests/test_integration_q4k)
    
    set(KERNEL_TEST_TARGETS test_mpi_linear_kernel test_mpi_rmsnorm_kernel test_mpi_attention_kernel test_mpi_transformer_pipeline test_adaptive_matmul test_cosma_minimal test_cosma_prefill_manager test_dequant test_dequant_extra test_dequant_golden test_integration_q4k)
else()
    message(STATUS "Google Test not found - kernel unit tests disabled")
    set(KERNEL_TEST_TARGETS "")
endif()

# Set output directories
set_target_properties(llaminar adaptive_demo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(test_basic test_numa test_cosma test_graph ${KERNEL_TEST_TARGETS}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Custom test target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_basic test_numa test_cosma test_graph ${KERNEL_TEST_TARGETS}
    COMMENT "Running all tests"
)

# Add DEBUG flag to all test targets when in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # List of all test targets
    set(TEST_TARGETS test_basic test_numa test_cosma test_graph)
    
    # Add any kernel test targets if they exist
    if(GTEST_FOUND)
        list(APPEND TEST_TARGETS ${KERNEL_TEST_TARGETS})
    endif()
    
    # Apply DEBUG flag to all test targets
    foreach(target ${TEST_TARGETS})
        if(TARGET ${target})
            target_compile_definitions(${target} PRIVATE DEBUG)
        endif()
    endforeach()
endif()

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
