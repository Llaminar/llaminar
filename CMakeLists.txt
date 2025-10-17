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

# Option: enable heavy RMSNorm reference validation code
option(LLAMINAR_ENABLE_RMSNORM_REFERENCE "Enable RMSNorm reference validation path" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Default OFF in Release unless explicitly turned on
    if(NOT DEFINED CACHE{LLAMINAR_ENABLE_RMSNORM_REFERENCE_SET})
        set(LLAMINAR_ENABLE_RMSNORM_REFERENCE OFF)
    endif()
endif()
if(LLAMINAR_ENABLE_RMSNORM_REFERENCE)
    add_compile_definitions(LLAMINAR_ENABLE_RMSNORM_REFERENCE)
endif()

# Option: enable tensor validation, contracts, and health checks
# Defaults to ON in Debug, OFF in Release for maximum performance
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT DEFINED LLAMINAR_ENABLE_VALIDATION)
    option(LLAMINAR_ENABLE_VALIDATION "Enable tensor validation, contracts, and health checks" OFF)
else()
    option(LLAMINAR_ENABLE_VALIDATION "Enable tensor validation, contracts, and health checks" ON)
endif()
if(LLAMINAR_ENABLE_VALIDATION)
    add_compile_definitions(LLAMINAR_ENABLE_VALIDATION)
    message(STATUS "Tensor validation and contracts enabled")
else()
    message(STATUS "Tensor validation and contracts disabled for maximum performance")
endif()

# Option: enable performance tracing framework (zero-overhead when disabled)
option(LLAMINAR_ENABLE_PERF_TRACE "Enable hierarchical performance tracing" ON)
if(LLAMINAR_ENABLE_PERF_TRACE)
    add_compile_definitions(LLAMINAR_ENABLE_PERF_TRACE)
    message(STATUS "Performance tracing enabled (use LLAMINAR_PERF_TRACE=1 to activate)")
endif()

# Compiler flags
# Note: Removed -DDEBUG from global flags to prevent COSMA verbose output
# Debug build with gcov coverage support:
#   -g: debug symbols
#   -O0: no optimizations (required for accurate coverage)
#   --coverage: enables gcov instrumentation (equivalent to -fprofile-arcs -ftest-coverage)
#   -fprofile-abs-path: use absolute paths in .gcno files (helps gcov viewer match files)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 --coverage -fprofile-abs-path")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 --coverage -fprofile-abs-path")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Linker flags for coverage (Debug only)
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "--coverage")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "--coverage")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Ensure MPI dependent code paths are enabled
add_compile_definitions(LLAMINAR_HAVE_MPI)

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

message(STATUS "Configuring llama.cpp submodule...")
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
# Use -march=native to avoid illegal instruction errors on varying CPU architectures
set(GGML_NATIVE ON CACHE BOOL "Enable native CPU optimizations for GGML" FORCE)
add_subdirectory(llama.cpp)



# Source files for the core library
set(LLAMINAR_CORE_SOURCES
    src/SystemTopology.cpp
    src/ArgumentParser.cpp
    src/BenchmarkRunner.cpp
    src/SequentialBatchBenchmark.cpp
    src/BatchPaddingUtils.cpp
    src/BatchQwenPipeline.cpp
    src/BatchQwenPipelineAdapter.cpp
    src/JsonExport.cpp
    src/TopologyManager.cpp
    src/MpiKernelBase.cpp
    src/CosmaPrefillManager.cpp
    src/utils/DebugEnv.cpp
    src/PerformanceTracer.cpp
    src/operators/MPILinearOperator.cpp
    src/operators/MPILinearBatchOperator.cpp
    src/operators/MPIRMSNormOperator.cpp
    src/operators/MPIAttentionOperator.cpp
    src/operators/MPIAttentionBatchOperator.cpp
    src/operators/MPIEmbeddingOperator.cpp
    src/operators/MPIResidualOperator.cpp
    src/operators/MPIRoPEOperator.cpp
    src/operators/MPISwiGLUOperator.cpp
    src/operators/MPISwiGLUBatchOperator.cpp
    src/operators/attention/AttentionValidator.cpp
    src/operators/common/AttentionPrimitives.cpp
    src/operators/common/RmsnormCore.cpp
    src/operators/common/RmsnormT5.cpp
    src/operators/common/SoftmaxCore.cpp
    src/QwenPipeline.cpp
    src/PipelineBase.cpp
    src/ParityHooks.cpp
    src/BackendSelector.cpp
    src/MatmulBackendSelection.cpp
    src/LargeMatmulPlan.cpp
    src/PrefillDiagnostics.cpp
    src/AbstractPipeline.cpp
    src/PipelineSnapshotManager.cpp
    src/PrefillProvider.cpp
    src/PrefillProviderBaseImpl.cpp
    src/OpenblasPrefillProvider.cpp
    src/CosmaPrefillProvider.cpp
    src/QwenPipelineAdapter.cpp
    src/LlamaPipelineAdapter.cpp
    src/tensors/SimpleTensor.cpp
    src/tensors/CosmaTensor.cpp
    src/tensors/TensorFactory.cpp
    src/tensors/BatchedKVCache.cpp
    src/ModelLoader.cpp
    src/ModelWeightsProvider.cpp
    src/MpiSlicingHelper.cpp
    src/Repacker.cpp
    src/SocketAwareInference.cpp
    # Backend abstraction (prefill / inference future GPU integration)
    src/backends/DeviceKind.h
    src/backends/PrefillBackend.h
    src/backends/PrefillBackend.cpp
    src/backends/InferenceBackend.h
    src/backends/InferenceBackend.cpp
    # Adaptive architecture components
    src/AdaptiveMatmul.h
    src/MatmulBackendSelection.h
    src/AdaptiveTransformerPipeline.h
    # Chat interface components
    src/chat/GgufTokenizer.cpp
    src/chat/BpeProcessor.cpp
    src/chat/ChatSession.cpp
    src/chat/ResponseGenerator.cpp
    src/chat/ChatInterface.cpp
    # Weight slicing Phase 0 helpers
    src/weights/WeightRoles.cpp
)

# Create the core library
add_library(llaminar_core STATIC ${LLAMINAR_CORE_SOURCES})

# Redundant safety: explicitly add WeightRoles.cpp in case list edits are missed
target_sources(llaminar_core PRIVATE src/weights/WeightRoles.cpp)

# Link libraries to core
target_link_libraries(llaminar_core
    PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cosma
    ggml
    ${BLAS_LIBRARIES}
)

# Optional GNU libmvec (vector math) for accelerated transcendentals (e.g., RoPE sin/cos)
find_library(MVEC_LIB mvec)
if(MVEC_LIB)
    message(STATUS "Found libmvec: ${MVEC_LIB}")
    target_link_libraries(llaminar_core PUBLIC ${MVEC_LIB})
    target_compile_definitions(llaminar_core PUBLIC LLAMINAR_HAVE_LIBMVEC=1)
endif()

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
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml
)

# Main executable
add_executable(llaminar src/Main.cpp)
target_link_libraries(llaminar llaminar_core)

# Adaptive architecture demonstration
add_executable(adaptive_demo src/AdaptiveDemo.cpp)
target_link_libraries(adaptive_demo llaminar_core)

# Production adaptive architecture demonstration
add_executable(production_demo src/ProductionDemo.cpp)
target_link_libraries(production_demo llaminar_core)

# Tensor-parallel output projection micro-benchmark
# add_executable(bench_tp_output_projection bench_tp_output_projection.cpp)
# target_link_libraries(bench_tp_output_projection llaminar_core)

# RMSNorm micro-benchmark harness
# add_executable(bench_rmsnorm bench_rmsnorm.cpp)
# target_link_libraries(bench_rmsnorm llaminar_core)

# Attention primitives benchmark (primitives vs legacy path)
# add_executable(bench_attention bench_attention.cpp)
# target_link_libraries(bench_attention llaminar_core)

# Softmax micro-benchmark harness
# add_executable(bench_softmax bench_softmax.cpp)
# target_link_libraries(bench_softmax llaminar_core)

# Distributed softmax benchmark (MPI required)
# add_executable(bench_softmax_distributed bench_softmax_distributed.cpp)
# target_link_libraries(bench_softmax_distributed llaminar_core)

# RoPE micro-benchmark harness
# add_executable(bench_rope bench_rope.cpp)
# target_link_libraries(bench_rope llaminar_core)

# Add DEBUG flag to main executable when in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(llaminar PRIVATE DEBUG)
    target_compile_definitions(adaptive_demo PRIVATE DEBUG)
    target_compile_definitions(production_demo PRIVATE DEBUG)
endif()

# Enable testing
enable_testing()

# Helper function to add tests with DEBUG log level by default
# This ensures all CTest runs include DEBUG level logs for better diagnostics
# Usage: add_test_with_debug_logging(TestName COMMAND test_executable [args...])
function(add_test_with_debug_logging TEST_NAME)
    # Forward all arguments to add_test, inserting NAME before TEST_NAME
    add_test(NAME ${TEST_NAME} ${ARGN})
    set_tests_properties(${TEST_NAME} PROPERTIES
        ENVIRONMENT "LLAMINAR_LOG_LEVEL=DEBUG"
    )
endfunction()

# Weight role classification unit test (non-MPI)
if(GTEST_FOUND)
    add_executable(test_weight_role_classification
        tests/TestWeightRoleClassification.cpp
        src/weights/WeightRoles.cpp
        src/utils/DebugEnv.cpp
    )
    target_link_libraries(test_weight_role_classification
        PRIVATE
        llaminar_core
        GTest::gtest
        GTest::gtest_main
    )
    add_test_with_debug_logging(WeightRoleClassification COMMAND test_weight_role_classification)
endif()

# RMSNorm core parity regression test
if(GTEST_FOUND)
    add_executable(test_rmsnorm_core_correctness
        tests/TestRmsnormCoreCorrectness.cpp
        src/utils/DebugEnv.cpp
    )
    target_link_libraries(test_rmsnorm_core_correctness
        PRIVATE
            llaminar_core
            GTest::gtest
            GTest::gtest_main
    )
    add_test_with_debug_logging(RMSNormCoreCorrectness COMMAND test_rmsnorm_core_correctness)

    add_executable(test_softmax_core_correctness
        tests/TestSoftmaxCoreCorrectness.cpp
    )
    target_link_libraries(test_softmax_core_correctness
        PRIVATE
            llaminar_core
            GTest::gtest
            GTest::gtest_main
    )
    add_test_with_debug_logging(SoftmaxCoreCorrectness COMMAND test_softmax_core_correctness)
endif()


function(add_llaminar_mpi_test TEST_NAME NUM_PROCS TARGET_NAME)
    if(NOT MPIEXEC_EXECUTABLE)
        message(FATAL_ERROR "MPI executable not available for ${TEST_NAME}")
    endif()

    set(_mpi_command ${MPIEXEC_EXECUTABLE})
    if(MPIEXEC_NUMPROC_FLAG)
        list(APPEND _mpi_command ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS})
    else()
        list(APPEND _mpi_command -n ${NUM_PROCS})
    endif()

    foreach(flag ${MPIEXEC_PREFLAGS})
        if(NOT flag)
            continue()
        endif()
        if(flag MATCHES "^-?-oversubscribe$")
            continue()
        endif()
        list(APPEND _mpi_command ${flag})
    endforeach()

    list(APPEND _mpi_command --oversubscribe)
    list(APPEND _mpi_command $<TARGET_FILE:${TARGET_NAME}>)

    foreach(flag ${MPIEXEC_POSTFLAGS})
        if(NOT flag)
            continue()
        endif()
        if(flag MATCHES "^-?-oversubscribe$")
            continue()
        endif()
        list(APPEND _mpi_command ${flag})
    endforeach()

    add_test(NAME ${TEST_NAME}
        COMMAND ${_mpi_command}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    set_tests_properties(${TEST_NAME} PROPERTIES 
        ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1;LLAMINAR_LOG_LEVEL=DEBUG"
    )
endfunction()

# Weight slice parity validation test (requires MPI, launched with 2 ranks)
if(GTEST_FOUND)
    add_executable(test_weight_slice_correctness
        tests/TestWeightSliceCorrectness.cpp
        src/weights/WeightRoles.cpp
        src/utils/DebugEnv.cpp
    )
    target_link_libraries(test_weight_slice_correctness
        PRIVATE
        llaminar_core
        GTest::gtest
        GTest::gtest_main
    )
    add_llaminar_mpi_test(WeightSliceCorrectness 2 test_weight_slice_correctness)
endif()

# Test sources and executables
add_executable(test_basic tests/TestBasic.cpp)
target_link_libraries(test_basic llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(BasicTest COMMAND test_basic)

add_executable(test_pipeline_factory tests/TestPipelineFactory.cpp)
target_link_libraries(test_pipeline_factory llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(PipelineFactoryTest COMMAND test_pipeline_factory)

add_executable(test_weight_contracts tests/TestWeightContracts.cpp)
target_link_libraries(test_weight_contracts llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(WeightContractsTest COMMAND test_weight_contracts)

add_executable(test_weight_validation_integration tests/TestWeightValidationIntegration.cpp)
target_link_libraries(test_weight_validation_integration llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(WeightValidationIntegration COMMAND test_weight_validation_integration)

add_executable(test_large_matmul_plan tests/TestLargeMatmulPlan.cpp)
target_link_libraries(test_large_matmul_plan llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(LargeMatmulPlanTest COMMAND test_large_matmul_plan)

add_executable(test_numa tests/TestNuma.cpp)
target_link_libraries(test_numa llaminar_core)
add_test_with_debug_logging(NumaTest COMMAND test_numa)

# Batch padding utilities test
add_executable(test_batch_padding tests/test_batch_padding.cpp)
target_link_libraries(test_batch_padding llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(BatchPaddingTest COMMAND test_batch_padding)

# Tensor batch operations test
add_executable(test_tensor_batch_ops tests/test_tensor_batch_ops.cpp)
target_link_libraries(test_tensor_batch_ops llaminar_core GTest::gtest GTest::gtest_main)
add_test_with_debug_logging(TensorBatchOpsTest COMMAND test_tensor_batch_ops)

add_executable(test_contract_loading tests/TestContractLoading.cpp)
target_link_libraries(test_contract_loading llaminar_core)
add_test_with_debug_logging(ContractLoadingTest COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_contract_loading ${MPIEXEC_POSTFLAGS})

add_executable(test_bias_contracts tests/TestBiasContracts.cpp)
target_link_libraries(test_bias_contracts llaminar_core GTest::gtest)
add_test_with_debug_logging(BiasContractTest COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ./test_bias_contracts ${MPIEXEC_POSTFLAGS})

add_executable(test_cosma tests/TestCosma.cpp)
target_link_libraries(test_cosma llaminar_core)
add_test(NAME CosmaTest COMMAND test_cosma)

# COSMA precision comparison test (float vs double in multi-rank)
add_executable(test_cosma_precision tests/TestCosmaPrecision.cpp)
target_link_libraries(test_cosma_precision 
    PRIVATE 
    cosma 
    GTest::gtest 
    MPI::MPI_CXX
    ${BLAS_LIBRARIES}
)
add_llaminar_mpi_test(CosmaPrecision 2 test_cosma_precision)
set_tests_properties(CosmaPrecision PROPERTIES TIMEOUT 120)

# Embedding standalone test (minimal isolation test for debugging)
add_executable(test_embedding_standalone 
    tests/TestEmbeddingStandalone.cpp
    tests/ParityTestFramework.cpp
)
target_link_libraries(test_embedding_standalone llaminar_core GTest::gtest MPI::MPI_CXX)
add_test(NAME EmbeddingStandaloneTest COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 
    ${CMAKE_CURRENT_BINARY_DIR}/test_embedding_standalone)
set_tests_properties(EmbeddingStandaloneTest PROPERTIES TIMEOUT 60)

# Kernel unit tests (if Google Test is available)
if(GTEST_FOUND)
    message(STATUS "Building kernel unit tests with Google Test")
    

    

    

    

    

    
    # MPI linear kernel test
    add_executable(test_mpi_linear_kernel tests/TestMpiLinearKernel.cpp)
    target_link_libraries(test_mpi_linear_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_linear_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_linear_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_linear_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPILinearKernelTest 2 test_mpi_linear_kernel)

    # MPI Linear Batch Operator test - comprehensive parity and equivalence validation
    add_executable(test_linear_batch_operator tests/test_linear_batch_operator.cpp)
    target_link_libraries(test_linear_batch_operator llaminar_core GTest::gtest)
    add_llaminar_mpi_test(LinearBatchOperatorTest 2 test_linear_batch_operator)
    set_tests_properties(LinearBatchOperatorTest PROPERTIES TIMEOUT 90)
    set_tests_properties(LinearBatchOperatorTest PROPERTIES LABELS "linear;batch;parity;kernel")

    # MPI SwiGLU Batch Operator test - comprehensive parity and equivalence validation
    add_executable(test_swiglu_batch_operator tests/test_swiglu_batch_operator.cpp)
    target_link_libraries(test_swiglu_batch_operator llaminar_core GTest::gtest)
    add_llaminar_mpi_test(SwiGLUBatchOperatorTest 2 test_swiglu_batch_operator)
    set_tests_properties(SwiGLUBatchOperatorTest PROPERTIES TIMEOUT 60)
    set_tests_properties(SwiGLUBatchOperatorTest PROPERTIES LABELS "swiglu;batch;parity;kernel;activation")

    # MPI RMSNorm Batch support test
    add_executable(test_mpi_rmsnorm_batch tests/test_mpi_rmsnorm_batch.cpp)
    target_link_libraries(test_mpi_rmsnorm_batch llaminar_core GTest::gtest)
    add_llaminar_mpi_test(MPIRMSNormBatchTest 2 test_mpi_rmsnorm_batch)
    set_tests_properties(MPIRMSNormBatchTest PROPERTIES TIMEOUT 60)
    set_tests_properties(MPIRMSNormBatchTest PROPERTIES LABELS "rmsnorm;batch;normalization")

    # MPI Attention Batch support test
    add_executable(test_mpi_attention_batch tests/test_mpi_attention_batch.cpp)
    target_link_libraries(test_mpi_attention_batch llaminar_core GTest::gtest)
    add_llaminar_mpi_test(MPIAttentionBatchTest 2 test_mpi_attention_batch)
    set_tests_properties(MPIAttentionBatchTest PROPERTIES TIMEOUT 120)
    set_tests_properties(MPIAttentionBatchTest PROPERTIES LABELS "attention;batch;kernel")

    # Minimal MPI finalize probe test (diagnostic for finalize hang)
    add_executable(test_mpi_finalize_probe tests/TestMpiFinalizeProbe.cpp)
    target_link_libraries(test_mpi_finalize_probe llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_finalize_probe GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_finalize_probe PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_finalize_probe ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPIFinalizeProbeTest 2 test_mpi_finalize_probe)

    add_executable(test_mpi_rmsnorm_kernel tests/TestMpiRmsnormKernel.cpp)
    target_link_libraries(test_mpi_rmsnorm_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_rmsnorm_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_rmsnorm_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_rmsnorm_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPIRMSNormKernelTest 2 test_mpi_rmsnorm_kernel)
    # Distributed softmax parity test (2 ranks)
    add_executable(test_mpi_softmax_correctness tests/TestMpiSoftmaxCorrectness.cpp)
    target_link_libraries(test_mpi_softmax_correctness llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_softmax_correctness GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_softmax_correctness PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_softmax_correctness ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPISoftmaxCorrectnessTest 2 test_mpi_softmax_correctness)

    # Reconfigure test_mpi_attention_kernel as a standalone minimal MPI binary (no GTest) to avoid
    # pre-main static initialization hang (see tests/AGENTS.md Case Study 11).
    add_executable(test_mpi_attention_kernel tests/TestMpiAttentionKernel.cpp)
    target_compile_definitions(test_mpi_attention_kernel PRIVATE LLAMINAR_STANDALONE_MPI_ATTENTION_TEST=1)
    target_link_libraries(test_mpi_attention_kernel PRIVATE llaminar_core MPI::MPI_CXX)
    # Register as CTest with explicit mpiexec invocation.
    add_test(NAME MPIAttentionKernelTest COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_attention_kernel>)
    set_tests_properties(MPIAttentionKernelTest PROPERTIES TIMEOUT 45 LABELS "attention;mpi;kernel" RUN_SERIAL TRUE)

    # Clean implementation unit tests (comprehensive test suite for refactored kernel)
    add_executable(test_mpi_attention_kernel_clean tests/TestMpiAttentionKernelClean.cpp)
    target_link_libraries(test_mpi_attention_kernel_clean llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_mpi_attention_kernel_clean GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_mpi_attention_kernel_clean PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_attention_kernel_clean ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    
    # Batch prefill skeleton test (validates BatchQwenPipeline foundation)
    add_executable(test_batch_prefill_skeleton tests/TestBatchPrefillSkeleton.cpp)
    target_link_libraries(test_batch_prefill_skeleton llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_batch_prefill_skeleton GTest::gtest pthread)
    else()
        target_include_directories(test_batch_prefill_skeleton PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_batch_prefill_skeleton ${GTEST_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(BatchPrefillSkeletonTest 2 test_batch_prefill_skeleton)
    set_tests_properties(BatchPrefillSkeletonTest PROPERTIES TIMEOUT 120 LABELS "batch;pipeline;skeleton")
    # Single-rank tests
    add_test(NAME MPIAttentionKernelClean_SingleRank 
             COMMAND $<TARGET_FILE:test_mpi_attention_kernel_clean> --gtest_filter=*SingleRank*:*Determinism*:*InvalidInput*:*MismatchedDimensions*:*WithBias*:*GQASupport*:*GQAVsMHA*:*SmallScale*:*LargeSequence*:*SnapshotCallback*)
    set_tests_properties(MPIAttentionKernelClean_SingleRank PROPERTIES TIMEOUT 60 LABELS "attention;mpi;kernel;clean" RUN_SERIAL TRUE)
    # Multi-rank tests
    add_test(NAME MPIAttentionKernelClean_MultiRank 
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_attention_kernel_clean> --gtest_filter=*MultiRank*:*BiasDistribution*)
    set_tests_properties(MPIAttentionKernelClean_MultiRank PROPERTIES TIMEOUT 60 LABELS "attention;mpi;kernel;clean" RUN_SERIAL TRUE)

    # Regression tests for specific bugs (bias overflow, RoPE GQA, dimension validation)
    add_executable(test_attention_regression tests/TestAttentionRegression.cpp)
    target_link_libraries(test_attention_regression llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_attention_regression GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_attention_regression PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_regression ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME AttentionRegressionTests 
             COMMAND $<TARGET_FILE:test_attention_regression>)
    set_tests_properties(AttentionRegressionTests PROPERTIES 
        TIMEOUT 60 
        LABELS "attention;regression;safety" 
        RUN_SERIAL TRUE)

    # Real weights test (uses actual GGUF model weights instead of synthetic data)
    add_executable(test_mpi_attention_real_weights tests/TestMpiAttentionRealWeights.cpp)
    target_link_libraries(test_mpi_attention_real_weights llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_mpi_attention_real_weights GTest::gtest pthread)
    else()
        target_include_directories(test_mpi_attention_real_weights PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_attention_real_weights ${GTEST_LIBRARY} pthread)
    endif()
    # Single-rank test with real weights
    add_test(NAME MPIAttentionRealWeights_SingleRank 
             COMMAND $<TARGET_FILE:test_mpi_attention_real_weights>)
    set_tests_properties(MPIAttentionRealWeights_SingleRank PROPERTIES TIMEOUT 120 LABELS "attention;mpi;real-weights" RUN_SERIAL TRUE)
    # Multi-rank test with real weights
    add_test(NAME MPIAttentionRealWeights_MultiRank 
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_attention_real_weights>)
    set_tests_properties(MPIAttentionRealWeights_MultiRank PROPERTIES TIMEOUT 120 LABELS "attention;mpi;real-weights" RUN_SERIAL TRUE)

    # Bias validation test (ensures kernel rejects incorrect bias sizes)
    add_executable(test_attention_bias_validation tests/TestAttentionBiasValidation.cpp)
    target_link_libraries(test_attention_bias_validation llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_attention_bias_validation GTest::gtest pthread)
    else()
        target_include_directories(test_attention_bias_validation PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_bias_validation ${GTEST_LIBRARY} pthread)
    endif()
    add_test(NAME AttentionBiasValidation COMMAND $<TARGET_FILE:test_attention_bias_validation>)
    set_tests_properties(AttentionBiasValidation PROPERTIES TIMEOUT 30 LABELS "attention;validation" RUN_SERIAL TRUE)

    # Minimal standalone MPI probe (no GTest, used to debug hangs prior to test harness)
    add_executable(minimal_mpi_probe tests/MinimalMpiProbe.cpp)
    target_link_libraries(minimal_mpi_probe llaminar_core MPI::MPI_CXX)

    # Prefill Performance Benchmark (MPI/OpenMP parallelization analysis)
    add_executable(test_prefill_performance_bench tests/test_prefill_performance_bench.cpp)
    target_link_libraries(test_prefill_performance_bench llaminar_core MPI::MPI_CXX OpenMP::OpenMP_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_prefill_performance_bench GTest::gtest pthread)
    else()
        target_include_directories(test_prefill_performance_bench PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_prefill_performance_bench ${GTEST_LIBRARY} pthread)
    endif()
    # Multi-rank performance benchmark tests
    add_test(NAME PrefillPerformanceBench_OpenBLAS_StrongScaling
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_prefill_performance_bench> 
             --gtest_filter=PrefillPerformanceBench.OpenBLAS_StrongScaling_Prefill)
    set_tests_properties(PrefillPerformanceBench_OpenBLAS_StrongScaling 
        PROPERTIES TIMEOUT 180 LABELS "performance;benchmark;openblas" RUN_SERIAL TRUE)
    
    add_test(NAME PrefillPerformanceBench_OpenBLAS_ModelShapes
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_prefill_performance_bench> 
             --gtest_filter=PrefillPerformanceBench.OpenBLAS_ModelShapes)
    set_tests_properties(PrefillPerformanceBench_OpenBLAS_ModelShapes 
        PROPERTIES TIMEOUT 240 LABELS "performance;benchmark;openblas" RUN_SERIAL TRUE)
    
    add_test(NAME PrefillPerformanceBench_ThreadUtilization
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_prefill_performance_bench> 
             --gtest_filter=PrefillPerformanceBench.ThreadUtilizationAnalysis)
    set_tests_properties(PrefillPerformanceBench_ThreadUtilization 
        PROPERTIES TIMEOUT 180 LABELS "performance;benchmark;threading" RUN_SERIAL TRUE)
    
    add_test(NAME PrefillPerformanceBench_COSMA_StrongScaling
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_prefill_performance_bench> 
             --gtest_filter=PrefillPerformanceBench.COSMA_StrongScaling_Prefill)
    set_tests_properties(PrefillPerformanceBench_COSMA_StrongScaling 
        PROPERTIES TIMEOUT 240 LABELS "performance;benchmark;cosma" RUN_SERIAL TRUE)
    
    add_test(NAME PrefillPerformanceBench_COSMA_ModelShapes
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_prefill_performance_bench> 
             --gtest_filter=PrefillPerformanceBench.COSMA_ModelShapes)
    set_tests_properties(PrefillPerformanceBench_COSMA_ModelShapes 
        PROPERTIES TIMEOUT 300 LABELS "performance;benchmark;cosma" RUN_SERIAL TRUE)
    
    add_test(NAME PrefillPerformanceBench_Comparative
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_prefill_performance_bench> 
             --gtest_filter=PrefillPerformanceBench.ComparativePerformance)
    set_tests_properties(PrefillPerformanceBench_Comparative 
        PROPERTIES TIMEOUT 360 LABELS "performance;benchmark;comparative" RUN_SERIAL TRUE)

    # Attention Output Mode Validation Test (regression test for LocalHeads bug)
    # This test ensures that LocalHeads mode correctly throws an error when used with
    # non-sharded weights on multiple ranks (the bug that caused 98.6% parity failures)
    add_executable(test_attention_output_mode_validation tests/TestAttentionOutputModeValidation.cpp)
    target_link_libraries(test_attention_output_mode_validation llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_attention_output_mode_validation GTest::gtest pthread)
    else()
        target_include_directories(test_attention_output_mode_validation PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_output_mode_validation ${GTEST_LIBRARY} pthread)
    endif()
    add_test(NAME AttentionOutputModeValidation COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_output_mode_validation>)
    set_tests_properties(AttentionOutputModeValidation PROPERTIES TIMEOUT 30 LABELS "attention;mpi;regression;validation" RUN_SERIAL TRUE)

    # Parity test wrapper integration: use the standalone binary with parameter variants.
    # We keep the binary (test_attention_shard_correctness) but add several CTest entries.
    # Variant matrix chosen to exercise small vs medium shapes and uneven head distribution.
    add_test(NAME AttentionShardCorrectness_Small COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_correctness> --heads=4 --head-dim=16 --seq-len=5)
    set_tests_properties(AttentionShardCorrectness_Small PROPERTIES TIMEOUT 30 LABELS "attention;parity;sharding")
    add_test(NAME AttentionShardCorrectness_Medium COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_correctness> --heads=8 --head-dim=8 --seq-len=32)
    set_tests_properties(AttentionShardCorrectness_Medium PROPERTIES TIMEOUT 30 LABELS "attention;parity;sharding")
    add_test(NAME AttentionShardCorrectness_Uneven COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_correctness> --heads=6 --head-dim=12 --seq-len=9)
    set_tests_properties(AttentionShardCorrectness_Uneven PROPERTIES TIMEOUT 30 LABELS "attention;parity;sharding")
    add_test(NAME AttentionShardCorrectness_Stress COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_correctness> --heads=12 --head-dim=16 --seq-len=24)

    # Unified TP GEMM parity matrix driver (row + col executor cases)
    add_test(NAME TPGemmCorrectness_Matrix
             COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/run_tp_gemm_parity_matrix.sh)
    set_tests_properties(TPGemmCorrectness_Matrix PROPERTIES TIMEOUT 240 LABELS "tp;gemm;parity;matrix")

    # ---------------------------------------------------------------------------
    # Generic TP GEMM executor parity tests (row & column partition)
    # ---------------------------------------------------------------------------
    add_executable(test_tp_gemm_correctness tests/TestTpGemmCorrectness.cpp)
    target_link_libraries(test_tp_gemm_correctness PRIVATE llaminar_core GTest::gtest_main)

    # Column mode tests
    add_test(NAME TPGemmCorrectness_ColEven_2way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.ColumnEvenSplit_2way)
    set_tests_properties(TPGemmCorrectness_ColEven_2way PROPERTIES TIMEOUT 40 LABELS "tp;gemm;parity;col;tp2")
    add_test(NAME TPGemmCorrectness_ColRagged_3way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 3 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.ColumnRaggedPrime_3way)
    set_tests_properties(TPGemmCorrectness_ColRagged_3way PROPERTIES TIMEOUT 50 LABELS "tp;gemm;parity;col;ragged;tp3")
    add_test(NAME TPGemmCorrectness_ColEven_4way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.ColumnEven_4way)
    set_tests_properties(TPGemmCorrectness_ColEven_4way PROPERTIES TIMEOUT 60 LABELS "tp;gemm;parity;col;tp4")

    # Row mode tests
    add_test(NAME TPGemmCorrectness_RowEven_2way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.RowEvenSplit_2way)
    set_tests_properties(TPGemmCorrectness_RowEven_2way PROPERTIES TIMEOUT 40 LABELS "tp;gemm;parity;row;tp2")
    add_test(NAME TPGemmCorrectness_RowRagged_3way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 3 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.RowRaggedPrime_3way)
    set_tests_properties(TPGemmCorrectness_RowRagged_3way PROPERTIES TIMEOUT 50 LABELS "tp;gemm;parity;row;ragged;tp3")
    add_test(NAME TPGemmCorrectness_RowEven_4way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.RowEven_4way)
    set_tests_properties(TPGemmCorrectness_RowEven_4way PROPERTIES TIMEOUT 60 LABELS "tp;gemm;parity;row;tp4")

    # Degenerate single-rank path
    add_test(NAME TPGemmCorrectness_DegenerateSingle
             COMMAND $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.DegenerateSingle)
    set_tests_properties(TPGemmCorrectness_DegenerateSingle PROPERTIES TIMEOUT 20 LABELS "tp;gemm;parity;degenerate")

    # Additional shape coverage (wider, taller, moderate, larger ragged)
    add_test(NAME TPGemmCorrectness_ColWide_4way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.ColumnWide_4way)
    set_tests_properties(TPGemmCorrectness_ColWide_4way PROPERTIES TIMEOUT 70 LABELS "tp;gemm;parity;col;tp4;extra")
    add_test(NAME TPGemmCorrectness_RowTall_4way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 4 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.RowTall_4way)
    set_tests_properties(TPGemmCorrectness_RowTall_4way PROPERTIES TIMEOUT 70 LABELS "tp;gemm;parity;row;tp4;extra")
    add_test(NAME TPGemmCorrectness_ColModerate_2way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.ColumnModerate_2way)
    set_tests_properties(TPGemmCorrectness_ColModerate_2way PROPERTIES TIMEOUT 50 LABELS "tp;gemm;parity;col;tp2;extra")
    add_test(NAME TPGemmCorrectness_RowModerate_2way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.RowModerate_2way)
    set_tests_properties(TPGemmCorrectness_RowModerate_2way PROPERTIES TIMEOUT 50 LABELS "tp;gemm;parity;row;tp2;extra")
    add_test(NAME TPGemmCorrectness_ColRaggedLarge_3way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 3 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.ColumnRaggedLarge_3way)
    set_tests_properties(TPGemmCorrectness_ColRaggedLarge_3way PROPERTIES TIMEOUT 60 LABELS "tp;gemm;parity;col;tp3;ragged;extra")
    add_test(NAME TPGemmCorrectness_RowRaggedLarge_3way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 3 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.RowRaggedLarge_3way)
    set_tests_properties(TPGemmCorrectness_RowRaggedLarge_3way PROPERTIES TIMEOUT 60 LABELS "tp;gemm;parity;row;tp3;ragged;extra")

    # Batched GEMM parity (multiple independent GEMMs)
    add_test(NAME TPGemmCorrectness_BatchRowEven_2way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.BatchRowEvenSplit_2way)
    set_tests_properties(TPGemmCorrectness_BatchRowEven_2way PROPERTIES TIMEOUT 70 LABELS "tp;gemm;parity;row;tp2;batch")
    add_test(NAME TPGemmCorrectness_BatchColEven_2way
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_tp_gemm_correctness> --gtest_filter=TPGemmCorrectness.BatchColEvenSplit_2way)
    set_tests_properties(TPGemmCorrectness_BatchColEven_2way PROPERTIES TIMEOUT 70 LABELS "tp;gemm;parity;col;tp2;batch")
    set_tests_properties(AttentionShardCorrectness_Stress PROPERTIES TIMEOUT 60 LABELS "attention;parity;sharding;stress")

    # Diagnostic path test: force loader sharded-mode + shard diagnostics for a tiny config to exercise verification logic.
    add_test(NAME AttentionShardLoad_Diag
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_correctness> --heads=4 --head-dim=8 --seq-len=4)
    # Environment variables enabling shard-mode and diagnostic logging
    set_tests_properties(AttentionShardLoad_Diag PROPERTIES
        TIMEOUT 40
        LABELS "attention;parity;sharding;diagnostic"
        ENVIRONMENT "LLAMINAR_SHARD_LOAD_DIAG=1")

    # Uneven head distribution diagnostic (5 heads -> ranks get 3 and 2) to stress offset logic
    add_test(NAME AttentionShardLoad_Diag_Uneven
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_correctness> --heads=5 --head-dim=8 --seq-len=6)
    set_tests_properties(AttentionShardLoad_Diag_Uneven PROPERTIES
        TIMEOUT 50
        LABELS "attention;parity;sharding;diagnostic;uneven"
        ENVIRONMENT "LLAMINAR_SHARD_LOAD_DIAG=1")

    # --- Role tagging synthetic log capture test (no MPI) ---
    add_executable(test_role_tagging_logs tests/TestRoleTaggingLogs.cpp)
    target_link_libraries(test_role_tagging_logs llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_role_tagging_logs GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME RoleTaggingSyntheticTest COMMAND test_role_tagging_logs)
    set_tests_properties(RoleTaggingSyntheticTest PROPERTIES TIMEOUT 15 LABELS "logging;roles;synthetic")

    # --- Role tagging integration tests (per quant group via GTest discovery) ---
    add_executable(test_model_load_role_tags tests/TestModelLoadRoleTags.cpp)
    target_link_libraries(test_model_load_role_tags llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_model_load_role_tags GTest::gtest GTest::gtest_main)
        include(GoogleTest)
        # Discover individual GTest cases (ModelLoadRoleTags.*_Group) so failures isolate the quant type.
        gtest_discover_tests(test_model_load_role_tags
            DISCOVERY_TIMEOUT 30
            PROPERTIES TIMEOUT 120 LABELS "logging;roles;integration" ENVIRONMENT "LLAMINAR_SHARD_LOAD_DIAG=1")
    else()
        # Fallback single test registration if GTest not found
        add_test(NAME RoleTaggingIntegrationTest COMMAND test_model_load_role_tags)
        set_tests_properties(RoleTaggingIntegrationTest PROPERTIES TIMEOUT 120 LABELS "logging;roles;integration" ENVIRONMENT "LLAMINAR_SHARD_LOAD_DIAG=1")
    endif()

    # Attention sharding parity harness (transitional). Uses MPI; currently reconstructs legacy sum output.
    add_executable(test_attention_shard_correctness tests/TestAttentionShardCorrectness.cpp)
    add_executable(test_attention_missing_reduction_negative tests/TestAttentionMissingReductionNegative.cpp)
    target_link_libraries(test_attention_shard_correctness llaminar_core)
    target_link_libraries(test_attention_missing_reduction_negative llaminar_core)
    # Standalone binary (no GTest) used for debugging; do not register with add_llaminar_mpi_test macro.
    add_test(NAME AttentionShardCorrectnessStandalone 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_attention_shard_correctness>)
    set_tests_properties(AttentionShardCorrectnessStandalone PROPERTIES TIMEOUT 30)
    set_tests_properties(AttentionShardCorrectnessStandalone PROPERTIES LABELS "attention;parity;sharding;standalone")

    # Attention stage contracts validation tests
    add_executable(test_attention_stage_contracts tests/TestAttentionStageContracts.cpp)
    target_link_libraries(test_attention_stage_contracts llaminar_core GTest::gtest)
    
    # Basic execution tests (single and multi-rank)
    add_test(NAME AttentionStageContracts_BasicExecution
             COMMAND ${MPIEXEC_EXECUTABLE} -np 1 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.BasicExecution)
    set_tests_properties(AttentionStageContracts_BasicExecution PROPERTIES TIMEOUT 30 LABELS "attention;contracts;validation")
    
    add_test(NAME AttentionStageContracts_MultiRankExecution
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.MultiRankExecution)
    set_tests_properties(AttentionStageContracts_MultiRankExecution PROPERTIES TIMEOUT 30 LABELS "attention;contracts;validation;mpi")
    
    # Prefill and decode path tests
    add_test(NAME AttentionStageContracts_PrefillExecution
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.PrefillExecution)
    set_tests_properties(AttentionStageContracts_PrefillExecution PROPERTIES TIMEOUT 40 LABELS "attention;contracts;validation;prefill")
    
    add_test(NAME AttentionStageContracts_DecodeExecution
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.DecodeExecution)
    set_tests_properties(AttentionStageContracts_DecodeExecution PROPERTIES TIMEOUT 40 LABELS "attention;contracts;validation;decode")
    
    # Contract violation tests (should fail gracefully)
    add_test(NAME AttentionStageContracts_InvalidInputShape
             COMMAND ${MPIEXEC_EXECUTABLE} -np 1 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.InvalidInputShape)
    set_tests_properties(AttentionStageContracts_InvalidInputShape PROPERTIES TIMEOUT 30 LABELS "attention;contracts;validation;negative")
    
    add_test(NAME AttentionStageContracts_InvalidWeightShape
             COMMAND ${MPIEXEC_EXECUTABLE} -np 1 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.InvalidWeightShape)
    set_tests_properties(AttentionStageContracts_InvalidWeightShape PROPERTIES TIMEOUT 30 LABELS "attention;contracts;validation;negative")
    
    # Visual verification test (shows contract messages)
    add_test(NAME AttentionStageContracts_ContractMessagesVisible
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_stage_contracts> --gtest_filter=AttentionStageContractsTest.ContractMessagesVisible)
    set_tests_properties(AttentionStageContracts_ContractMessagesVisible PROPERTIES TIMEOUT 30 LABELS "attention;contracts;validation;verbose")

    # RMSNorm shard stats test (validates distributed inverse scale parity vs reference)
    add_executable(test_rmsnorm_shard_stats tests/TestRmsnormShardStats.cpp)
    target_link_libraries(test_rmsnorm_shard_stats llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_rmsnorm_shard_stats GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME RMSNormShardStats_Parity
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_rmsnorm_shard_stats>)

    # RMSNorm sharded gamma parity test (validates gamma shard scaling parity)
    add_executable(test_rmsnorm_sharded_gamma tests/TestRmsnormShardedGamma.cpp)
    target_link_libraries(test_rmsnorm_sharded_gamma llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_rmsnorm_sharded_gamma GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME RMSNormShardedGamma_Parity
                 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_rmsnorm_sharded_gamma>)
    set_tests_properties(RMSNormShardedGamma_Parity PROPERTIES TIMEOUT 40 LABELS "rmsnorm;sharding;gamma;parity")

    # TP splitters unit tests (pure local, no MPI required)
    add_executable(test_tp_splitters tests/TestTpSplitters.cpp)
    target_link_libraries(test_tp_splitters llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_tp_splitters GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME TPSplittersTest COMMAND test_tp_splitters)

    # Attention TP parity test (column split reconstruction)
    add_executable(test_attention_tp_correctness tests/TestAttentionTpCorrectness.cpp)
    target_link_libraries(test_attention_tp_correctness llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_attention_tp_correctness GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME AttentionTPCorrectnessTest COMMAND test_attention_tp_correctness)

    # Generic TP GEMM executor parity test (row & column partition, uneven splits)
    add_executable(test_tp_generic_gemm_executor tests/TestTpGenericGemmExecutor.cpp)
    target_link_libraries(test_tp_generic_gemm_executor llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_tp_generic_gemm_executor GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME TPGemmGenericParityTest COMMAND test_tp_generic_gemm_executor)
    set_tests_properties(TPGemmGenericParityTest PROPERTIES TIMEOUT 25 LABELS "tp;gemm;parity")

    # Attention TP simulation parity test (row/col & auto heuristic)
    add_executable(test_attention_tp_sim_correctness tests/TestAttentionTpSimCorrectness.cpp)
    target_link_libraries(test_attention_tp_sim_correctness llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_attention_tp_sim_correctness GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME AttentionTPSimParityTest COMMAND test_attention_tp_sim_correctness)
    set_tests_properties(AttentionTPSimParityTest PROPERTIES TIMEOUT 30 LABELS "attention;parity;tp;simulation")


    # Negative misuse test: ensures missing reduction leads to detectable mismatch
    add_test(NAME AttentionShardMissingReduction_Negative
        COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_missing_reduction_negative> --heads=4 --head-dim=16 --seq-len=5)
    set_tests_properties(AttentionShardMissingReduction_Negative PROPERTIES TIMEOUT 30 LABELS "attention;negative;sharding")

    # Attention micro parity test (single small config, scalar reference comparison)
    add_executable(test_attention_micro tests/TestAttentionMicro.cpp)
    target_link_libraries(test_attention_micro llaminar_core)
    add_executable(test_attention_primitives tests/TestAttentionPrimitives.cpp)
    target_link_libraries(test_attention_primitives llaminar_core)
    if (GTest_FOUND)
        target_link_libraries(test_attention_primitives GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_attention_primitives PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_primitives ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    # Single-rank pure math primitives test (no MPI). Register with CTest explicitly.
    # Executable may live either in top-level or tests/ depending on generator/output rules.
    # Use generator expression for reliable path regardless of multi-config or output dir layout
    add_test(NAME AttentionPrimitivesTest COMMAND $<TARGET_FILE:test_attention_primitives>)
    set_tests_properties(AttentionPrimitivesTest PROPERTIES TIMEOUT 5)

    # K tensor gathering and RoPE GQA correctness tests
    add_executable(test_k_gathering_and_rope_gqa tests/TestKGatheringAndRopeGqa.cpp)
    target_link_libraries(test_k_gathering_and_rope_gqa llaminar_core MPI::MPI_CXX)
    if (GTest_FOUND)
        target_link_libraries(test_k_gathering_and_rope_gqa GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_k_gathering_and_rope_gqa PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_k_gathering_and_rope_gqa ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    # Single-rank tests (RoPE GQA logic)
    add_test(NAME KGatheringAndRopeGQA_SingleRank
             COMMAND $<TARGET_FILE:test_k_gathering_and_rope_gqa> --gtest_filter=*RopeUsesCorrectKVHeadIndices*:*RopeKIndependentOfQHeadCount*)
    set_tests_properties(KGatheringAndRopeGQA_SingleRank PROPERTIES TIMEOUT 10 LABELS "attention;rope;gqa")
    # Multi-rank tests (K gathering order)
    add_test(NAME KGatheringAndRopeGQA_MultiRank
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_k_gathering_and_rope_gqa> --gtest_filter=*AllgatherProducesSequentialHeadOrder*:*MultiRankKGatheringAfterRope*)
    set_tests_properties(KGatheringAndRopeGQA_MultiRank PROPERTIES TIMEOUT 15 LABELS "attention;rope;gqa;mpi" RUN_SERIAL TRUE)

    # Golden oracle tests for attention primitives (PyTorch reference)
    add_executable(test_attention_golden tests/TestAttentionGolden.cpp)
    target_link_libraries(test_attention_golden llaminar_core)
    if (GTest_FOUND)
        target_link_libraries(test_attention_golden GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_attention_golden PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_golden ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME AttentionGoldenTest COMMAND $<TARGET_FILE:test_attention_golden>)
    set_tests_properties(AttentionGoldenTest PROPERTIES TIMEOUT 10 LABELS "golden;quick")
    if(GTest_FOUND)
        target_link_libraries(test_attention_micro GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_attention_micro PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_micro ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(AttentionMicroTest 2 test_attention_micro)
    set_tests_properties(AttentionMicroTest PROPERTIES TIMEOUT 30)
    set_tests_properties(AttentionMicroTest PROPERTIES LABELS "attention;parity;kernel")

    # RoPE recurrence parity test (single-rank; validates recurrence vs legacy bitwise/tolerance parity)
    add_executable(test_rope_recurrence_correctness tests/TestRopeRecurrenceCorrectness.cpp)
    target_link_libraries(test_rope_recurrence_correctness llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_rope_recurrence_correctness GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_rope_recurrence_correctness PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_rope_recurrence_correctness ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME RoPERecurrenceParityTest COMMAND $<TARGET_FILE:test_rope_recurrence_correctness>)
    set_tests_properties(RoPERecurrenceParityTest PROPERTIES TIMEOUT 10 LABELS "rope;parity;attention")

    # RoPE corruption test (single-rank; tests for memory corruption with exact MPIAttentionKernel dimensions)
    add_executable(test_rope_corruption tests/TestRopeCorruption.cpp)
    target_link_libraries(test_rope_corruption llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_rope_corruption GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_rope_corruption PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_rope_corruption ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME RoPECorruptionTest COMMAND $<TARGET_FILE:test_rope_corruption>)
    set_tests_properties(RoPECorruptionTest PROPERTIES TIMEOUT 30 LABELS "rope;corruption;attention")

    # TP column-partition WO projection parity (single-rank; ensure partitions vs disabled match)
    add_test(NAME TPColumnPartitionWOParity
        COMMAND $<TARGET_FILE:test_attention_micro> --gtest_filter=AttentionMicroTestFixture.TPColumnPartitionWOParity)
    set_tests_properties(TPColumnPartitionWOParity PROPERTIES TIMEOUT 20 LABELS "attention;parity;tp")

    # GatherHeadsPreProjection multi-rank parity (filtered gtest run)
    add_test(NAME AttentionGatherPreProjectionParity
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_attention_micro>
                --gtest_filter=AttentionMicroTestFixture.GatherPreProjectionMultiRankParity)
    # Ensure the test runs with the intended output mode unless explicitly overridden.
    set_tests_properties(AttentionGatherPreProjectionParity PROPERTIES
        ENVIRONMENT "LLAMINAR_ATTN_OUTPUT_MODE=gather_pre;OMPI_MCA_rmaps_base_oversubscribe=1"
        TIMEOUT 30
        LABELS "attention;parity;gather_pre")

    # MPI Embedding kernel comprehensive correctness tests (full vs sharded, transposed vs non-transposed)
    add_executable(test_mpi_embedding_kernel tests/TestMpiEmbeddingKernel.cpp)
    target_link_libraries(test_mpi_embedding_kernel llaminar_core)
    # Link only gtest, not gtest_main, since file provides its own main via LLAMINAR_DEFINE_GTEST_MPI_MAIN
    if(GTest_FOUND)
        target_link_libraries(test_mpi_embedding_kernel GTest::gtest)
    else()
        target_include_directories(test_mpi_embedding_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_embedding_kernel ${GTEST_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPIEmbeddingKernelTest 2 test_mpi_embedding_kernel)
    # Temporary workaround: embedding kernel test hangs in MPI_Finalize; skip finalize to avoid CTRL-C.
    # Provide only oversubscribe flag; per-test selective finalize skip is handled inside the test now.
    set_tests_properties(MPIEmbeddingKernelTest PROPERTIES ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
    # Enforce a hard timeout so the test cannot hang indefinitely (auto-kill after 60s)
    set_tests_properties(MPIEmbeddingKernelTest PROPERTIES TIMEOUT 60)
    set_tests_properties(MPIEmbeddingKernelTest PROPERTIES LABELS "embedding;normal")

    # Extended embedding tests (fuzz + precision boundary) in a separate target so they can run under a distinct label.
    add_executable(test_mpi_embedding_kernel_extended tests/TestMpiEmbeddingKernel.cpp)
    target_link_libraries(test_mpi_embedding_kernel_extended llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_embedding_kernel_extended GTest::gtest)
    else()
        target_include_directories(test_mpi_embedding_kernel_extended PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_embedding_kernel_extended ${GTEST_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPIEmbeddingKernelExtendedTest 2 test_mpi_embedding_kernel_extended)
    # Force extended env var so fuzz + precision tests run; still allow user to add others.
    set_tests_properties(MPIEmbeddingKernelExtendedTest PROPERTIES ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1;LLAMINAR_EMBED_EXTENDED=1")
    set_tests_properties(MPIEmbeddingKernelExtendedTest PROPERTIES TIMEOUT 120)
    set_tests_properties(MPIEmbeddingKernelExtendedTest PROPERTIES LABELS "embedding;extended")
    
    # MPI Embedding Batch support test
    add_executable(test_mpi_embedding_batch tests/test_mpi_embedding_batch.cpp)
    target_link_libraries(test_mpi_embedding_batch llaminar_core GTest::gtest)
    add_llaminar_mpi_test(MPIEmbeddingBatchTest 2 test_mpi_embedding_batch)
    set_tests_properties(MPIEmbeddingBatchTest PROPERTIES TIMEOUT 60)
    set_tests_properties(MPIEmbeddingBatchTest PROPERTIES LABELS "embedding;batch")
    
    # MPI Transformer Pipeline test
    add_executable(test_mpi_transformer_pipeline tests/TestMpiTransformerPipeline.cpp)
    target_link_libraries(test_mpi_transformer_pipeline llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_transformer_pipeline GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_transformer_pipeline PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_transformer_pipeline ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    # Renamed to reflect QwenPipeline implementation (was MPITransformerPipelineTest)
    add_llaminar_mpi_test(QwenPipelineTest 2 test_mpi_transformer_pipeline)
    # Ensure this long-running MPI test cannot hang indefinitely; internal watchdog < 60s, external timeout at 60s
    set_tests_properties(QwenPipelineTest PROPERTIES TIMEOUT 60)
    
    # Adaptive Matrix Multiplication test
    # Shared logging bootstrap (object library) to unify test logging levels
    add_library(test_logging_bootstrap OBJECT tests/TestLoggingBootstrap.cpp)
    
    add_executable(test_adaptive_matmul tests/TestAdaptiveMatmul.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_adaptive_matmul llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_adaptive_matmul GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_adaptive_matmul PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_adaptive_matmul ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(AdaptiveMatMulTest 2 test_adaptive_matmul)
    
    # Minimal COSMA Custom Layout test
    add_executable(test_cosma_minimal tests/TestCosmaMinimal.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_minimal llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_minimal GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_minimal PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_minimal ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(CosmaMinimalTest 2 test_cosma_minimal)

    # Cosma Prefill Manager correctness (single rank baseline)
    add_executable(test_cosma_prefill_manager tests/TestCosmaPrefillManager.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager)
    set_tests_properties(CosmaPrefillManagerTest PROPERTIES TIMEOUT 60)



    add_executable(test_cosma_prefill_attention_integration tests/TestCosmaPrefillAttentionIntegration.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_attention_integration PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaPrefillAttentionIntegrationTest 2 test_cosma_prefill_attention_integration)
    set_tests_properties(CosmaPrefillAttentionIntegrationTest PROPERTIES TIMEOUT 120)

    add_executable(test_cosma_prefill_manager_orientation_autofix tests/TestCosmaPrefillManagerOrientationAutofix.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_orientation_autofix llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager_orientation_autofix GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager_orientation_autofix PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager_orientation_autofix ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerOrientationAutoFixTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager_orientation_autofix)
    set_tests_properties(CosmaPrefillManagerOrientationAutoFixTest PROPERTIES TIMEOUT 60)

    # Minimal scatter/gather coordinate mapping test
    # add_executable(test_cosma_scatter_gather test_cosma_scatter_gather.cpp)
    # target_link_libraries(test_cosma_scatter_gather PRIVATE llaminar_core MPI::MPI_CXX)
    # add_llaminar_mpi_test(CosmaScatterGatherTest 2 test_cosma_scatter_gather)
    # set_tests_properties(CosmaScatterGatherTest PROPERTIES TIMEOUT 60)



    add_executable(test_cosma_prefill_manager_stats tests/TestCosmaPrefillManagerStats.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_stats llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager_stats GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager_stats PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager_stats ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerStatsTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager_stats)
    set_tests_properties(CosmaPrefillManagerStatsTest PROPERTIES TIMEOUT 60)

    # TP output projection executor (simulation) test
    add_executable(test_tp_output_projection_executor tests/TestTpOutputProjectionExecutor.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_tp_output_projection_executor llaminar_core)
    add_test(NAME TPOutputProjectionExecutorTest COMMAND $<TARGET_FILE:test_tp_output_projection_executor>)



    # (Removed) test_abstract_pipeline_parity: disabled due to untrusted baseline; source retained for manual runs.
    # KV cache growth dynamic capacity test
    add_executable(test_kv_cache_growth tests/TestKvCacheGrowth.cpp)
    add_executable(test_incremental_generation tests/TestIncrementalGeneration.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_kv_cache_growth PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    target_link_libraries(test_incremental_generation PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(KVCacheGrowthTest 2 test_kv_cache_growth)
    add_llaminar_mpi_test(IncrementalGenerationTest 1 test_incremental_generation)
    add_executable(test_incremental_generation_multi tests/TestIncrementalGenerationMulti.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_incremental_generation_multi PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(IncrementalGenerationMultiRankTest 2 test_incremental_generation_multi)

    # Incremental decode replay vs incremental parity tests
    add_executable(test_incremental_decode_correctness tests/TestIncrementalDecodeCorrectness.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_incremental_decode_correctness PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    # Single-rank invocation (multi-rank test section will internally GTEST_SKIP multi-rank portion)
    add_llaminar_mpi_test(IncrementalDecodeCorrectnessSingle 1 test_incremental_decode_correctness)
    
    # ===== OBSOLETE 2025-10-15 =====
    # IncrementalDecodeCorrectnessMulti test disabled - superseded by ParityFramework.TrueIncrementalDecodeVsPyTorch
    # Reason: 
    #   - TrueIncrementalDecodeVsPyTorch provides superior coverage (validates against PyTorch reference)
    #   - Works with any rank count (1 or 2), uses real model files
    #   - IncrementalDecodeCorrectnessMulti only tests internal replay consistency with synthetic tiny model
    #   - Multi-rank version fails with GQA edge case (n_head_kv=1 with 2 ranks) not found in real models
    # Replacement: ParityFramework.TrueIncrementalDecodeVsPyTorch in TestParityFramework.cpp
    # add_llaminar_mpi_test(IncrementalDecodeCorrectnessMulti 2 test_incremental_decode_correctness)

    # (Removed) test_kv_cache_growth_correctness: superseded by functional KVCacheGrowthTest + decode instrumentation.
    set_tests_properties(KVCacheGrowthTest PROPERTIES TIMEOUT 60)

    # ===== DEPRECATED 2025-10-06 =====
    # test_prefill_attention_golden has been DEPRECATED and moved to tests/deprecated/
    # Reason: 80% overlap with ParityFramework, timeout issues
    # Replacement: ParityFramework.CosmaModeValidation in TestParityFramework.cpp
    # Migration complete - see COSMA_MIGRATION_COMPLETE.md for details
    # 
    # Original lines (now removed):
    # add_executable(test_prefill_attention_golden tests/TestPrefillAttentionGolden.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    # target_link_libraries(test_prefill_attention_golden PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
    # add_llaminar_mpi_test(PrefillAttentionGoldenTest 2 test_prefill_attention_golden)
    # =================================

# Parity test framework for comparing Llaminar with llama.cpp at intermediate stages
add_executable(test_parity_framework 
    tests/TestParityFramework.cpp 
    tests/ParityTestFramework.cpp 
    tests/WeightVerifier.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_parity_framework PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(ParityFrameworkTest 2 test_parity_framework)

# Inference parity test - compares generated text with PyTorch for real prompts
add_executable(test_inference_parity
    tests/TestInferenceParity.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_inference_parity PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(InferenceParityTest 2 test_inference_parity)
set_tests_properties(InferenceParityTest PROPERTIES TIMEOUT 300 LABELS "integration;parity;inference")

# Incremental snapshot helper test - verifies .npy writer and per-token snapshot saving
add_executable(test_incremental_snapshot_helper
    tests/TestIncrementalSnapshotHelper.cpp
    tests/ParityTestFramework.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_incremental_snapshot_helper PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)

# QwenPipeline batch processing tests - validates prefillBatch and decodeBatch
add_executable(test_qwen_pipeline_batch
    tests/test_qwen_pipeline_batch.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_qwen_pipeline_batch PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(QwenPipelineBatchTest 2 test_qwen_pipeline_batch)
set_tests_properties(QwenPipelineBatchTest PROPERTIES TIMEOUT 120 LABELS "integration;batch;qwen")

# Batch benchmark tests - validates batch benchmarking infrastructure and correctness
add_executable(test_batch_benchmark
    tests/test_batch_benchmark.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_batch_benchmark PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(BatchBenchmarkTest 2 test_batch_benchmark)
set_tests_properties(BatchBenchmarkTest PROPERTIES TIMEOUT 60 LABELS "unit;batch;benchmark")

# Batch correctness tests - validates batch vs sequential equivalence with real models
add_executable(test_batch_correctness
    tests/test_batch_correctness.cpp
    tests/ParityTestFramework.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_batch_correctness PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(BatchCorrectnessTest 2 test_batch_correctness)
set_tests_properties(BatchCorrectnessTest PROPERTIES TIMEOUT 600 LABELS "integration;batch;correctness")  # 10 minutes - these tests load models multiple times

# Batch vs Sequential parity test with pipeline snapshots - DISABLED (incomplete implementation)
# TODO: This test was started but never finished - API doesn't match PipelineSnapshotManager
# Use test_batch_correctness.cpp instead which has working batch parity tests
# add_executable(test_batch_sequential_parity
#     tests/test_batch_sequential_parity.cpp
#     tests/ParityTestFramework.cpp
#     $<TARGET_OBJECTS:test_logging_bootstrap>)
# target_link_libraries(test_batch_sequential_parity PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
# add_llaminar_mpi_test(BatchSequentialParityTest 2 test_batch_sequential_parity)
# set_tests_properties(BatchSequentialParityTest PROPERTIES TIMEOUT 180 LABELS "integration;batch;parity;snapshots")

# Batch embedding debug test - isolates embedding comparison
add_executable(test_batch_embedding_debug
    tests/test_batch_embedding_debug.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_batch_embedding_debug PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(BatchEmbeddingDebug 2 test_batch_embedding_debug)
set_tests_properties(BatchEmbeddingDebug PROPERTIES TIMEOUT 60 LABELS "integration;batch;debug")

# Batch performance tests - measures throughput scaling and bandwidth utilization
add_executable(test_batch_performance
    tests/test_batch_performance.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_batch_performance PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(BatchPerformanceTest 2 test_batch_performance)
set_tests_properties(BatchPerformanceTest PROPERTIES TIMEOUT 300 LABELS "integration;batch;performance")

# Batch tensor operations tests - Day 2: SimpleTensor batch dimension support
add_executable(test_batch_tensor_operations
    tests/test_batch_tensor_operations.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_batch_tensor_operations PRIVATE llaminar_core GTest::gtest_main)
add_test(NAME BatchTensorOperationsTest COMMAND test_batch_tensor_operations)
set_tests_properties(BatchTensorOperationsTest PROPERTIES TIMEOUT 30 LABELS "unit;batch;tensor;day2")

# Batched KV cache tests - Day 3: Per-sequence cache management
add_executable(test_batched_kv_cache
    tests/test_batched_kv_cache.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_batched_kv_cache PRIVATE llaminar_core GTest::gtest_main)
add_test(NAME BatchedKVCacheTest COMMAND test_batched_kv_cache)
set_tests_properties(BatchedKVCacheTest PROPERTIES TIMEOUT 30 LABELS "unit;batch;kvcache;day3")

# Attention batch integration tests - Day 3: MPIAttentionOperator with BatchedKVCache
add_executable(test_attention_batch_integration
    tests/test_attention_batch_integration.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_attention_batch_integration PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(AttentionBatchIntegrationTest 2 test_attention_batch_integration)
set_tests_properties(AttentionBatchIntegrationTest PROPERTIES TIMEOUT 120 LABELS "integration;batch;attention;kvcache;day3")

# Operator batch interface tests - Day 4-5: Batch dimension detection and validation
add_executable(test_operator_batch_interfaces
    tests/test_operator_batch_interfaces.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_operator_batch_interfaces PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(OperatorBatchInterfaceTest 2 test_operator_batch_interfaces)
set_tests_properties(OperatorBatchInterfaceTest PROPERTIES TIMEOUT 60 LABELS "unit;batch;operators;day4")

add_test(NAME IncrementalSnapshotHelper COMMAND test_incremental_snapshot_helper)
set_tests_properties(IncrementalSnapshotHelper PROPERTIES TIMEOUT 30 LABELS "unit;parity;snapshot")

# Weight passthrough test - verifies dequantized weights reach attention kernel intact
add_executable(test_weight_passthrough
    tests/TestWeightPassthrough.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_weight_passthrough PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(WeightPassthroughTest 2 test_weight_passthrough)
set_tests_properties(WeightPassthroughTest PROPERTIES TIMEOUT 60 LABELS "unit;weights;dequant")

# Isolated PrefillProvider unit tests (OpenBLAS & COSMA providers)
add_executable(test_prefill_providers 
    tests/TestPrefillProviders.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_prefill_providers PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_llaminar_mpi_test(PrefillProvidersTest 2 test_prefill_providers)
set_tests_properties(PrefillProvidersTest PROPERTIES TIMEOUT 120 LABELS "prefill;providers;unit")

# ============================================================================
# Integration Test Framework
# ============================================================================
# Comprehensive model integration tests with PyTorch layer-by-layer parity
# Tests both prefill and decode stages across multiple precisions

# Base model integration test framework
add_executable(test_qwen_integration
    tests/integration/QwenIntegrationTest.cpp
    tests/integration/ModelIntegrationTestBase.cpp
    tests/ParityTestFramework.cpp
    $<TARGET_OBJECTS:test_logging_bootstrap>)
target_include_directories(test_qwen_integration PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/tests/integration)
target_link_libraries(test_qwen_integration PRIVATE llaminar_core GTest::gtest MPI::MPI_CXX)

# Register integration tests with CTest
# These tests auto-skip if golden references are missing
add_llaminar_mpi_test(QwenIntegrationTests 2 test_qwen_integration)
set_tests_properties(QwenIntegrationTests PROPERTIES 
    TIMEOUT 300
    LABELS "integration;qwen;parity"
    ENVIRONMENT "LLAMINAR_PARITY_CAPTURE=1"
)

# Note: Tests require golden references in tests/golden_references/
# Generate with: python3 tests/integration/generate_golden_references.py --model qwen2.5-0.5b-instruct
# ============================================================================

add_executable(test_tp_partition tests/TestTpPartition.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
target_link_libraries(test_tp_partition PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
add_test(NAME TPPartitionSpecTest COMMAND test_tp_partition)    # Embedding orientation diagnostic test (non-MPI critical, rank 0 does work)
    add_executable(test_embedding_orientation tests/TestEmbeddingOrientation.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_embedding_orientation PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
    target_include_directories(test_embedding_orientation PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/src)
    add_test(NAME EmbeddingOrientationTest COMMAND test_embedding_orientation)

    # Dedicated fused RMSNorm+QKV correctness test (isolates CosmaPrefillManager::fused_rmsnorm_qkv)
    add_executable(test_cosma_fused_rmsnorm_qkv tests/TestCosmaFusedRmsnormQkv.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_fused_rmsnorm_qkv PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaFusedRmsnormQkvTest 2 test_cosma_fused_rmsnorm_qkv)
    # REMOVED: set_tests_properties(PrefillAttentionGoldenTest PROPERTIES TIMEOUT 180) - test deprecated 2025-10-06

    # Dequantization tests (Q4_0 correctness + Q4_K placeholder sanity)
    add_executable(test_dequant tests/TestDequant.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_dequant llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_dequant GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant)

    # Extra quantization tests (all types)
    add_executable(test_dequant_extra tests/TestDequantExtra.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_dequant_extra llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_dequant_extra GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant_extra PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant_extra ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantExtraTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant_extra)

    # Golden reference comparison test (ggml quant vs llaminar dequant)
    add_executable(test_dequant_golden tests/TestDequantGolden.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_include_directories(test_dequant_golden PRIVATE
        llama.cpp/ggml/src
        llama.cpp/ggml/include
        llama.cpp/ggml
    )
    target_link_libraries(test_dequant_golden llaminar_core llama)
    if(GTest_FOUND)
        target_link_libraries(test_dequant_golden GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant_golden PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant_golden ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantGoldenTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant_golden)

    # Model loader alignment regression test (ensures offset chain + alignment invariants)
    add_executable(test_model_loader_alignment tests/TestModelLoaderAlignment.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_include_directories(test_model_loader_alignment PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
    )
    target_link_libraries(test_model_loader_alignment llaminar_core llama)
    if (GTEST_FOUND)
        target_link_libraries(test_model_loader_alignment GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_model_loader_alignment PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_model_loader_alignment ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME ModelLoaderAlignmentTest COMMAND ${CMAKE_BINARY_DIR}/test_model_loader_alignment)

    # Integration test for real Q4_K tensors (statistical sanity)
    add_executable(test_integration_q4k tests/TestIntegrationQ4k.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_integration_q4k llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_integration_q4k GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_integration_q4k PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_integration_q4k ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME IntegrationQ4K COMMAND ${CMAKE_BINARY_DIR}/tests/test_integration_q4k)

    # Golden model loader comparison (float tensors vs llama.cpp reference)
    add_executable(test_model_loader_golden tests/TestModelLoaderGolden.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_model_loader_golden PRIVATE llaminar_core llama GTest::gtest_main)
    # Executable ended up at top-level build dir (not in tests/); reference directly
    if (EXISTS ${CMAKE_BINARY_DIR}/tests/test_model_loader_golden)
        add_test(NAME ModelLoaderGoldenTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_model_loader_golden)
    else()
        add_test(NAME ModelLoaderGoldenTest COMMAND ${CMAKE_BINARY_DIR}/test_model_loader_golden)
    endif()
    set_tests_properties(ModelLoaderGoldenTest PROPERTIES TIMEOUT 60)


    # Quant / Dequant parity tests (single-rank deterministic)
    add_executable(test_quant_dequant_correctness tests/TestQuantDequantCorrectness.cpp)
    target_link_libraries(test_quant_dequant_correctness llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME QuantDequantCorrectnessTest COMMAND test_quant_dequant_correctness)

    # Quantized shard regression test (validates column/row shard parity for quant tensors)
    add_executable(test_model_loader_quant_shard_regression tests/TestModelLoaderQuantShardRegression.cpp)
    target_link_libraries(test_model_loader_quant_shard_regression llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME ModelLoaderQuantShardRegression COMMAND test_model_loader_quant_shard_regression)
    add_executable(test_quant_shard_cache_stats tests/TestQuantShardCacheStats.cpp)
    target_link_libraries(test_quant_shard_cache_stats llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME QuantShardCacheStatsTest COMMAND test_quant_shard_cache_stats)
    add_executable(test_partial_decode_q4_0 tests/TestPartialDecodeQ40.cpp)
    target_link_libraries(test_partial_decode_q4_0 llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME PartialDecodeQ4_0Test COMMAND test_partial_decode_q4_0)
    # Multi-format partial decode parity (legacy formats Q4_1/Q5_1/Q8_1 removed)
    if (EXISTS ${CMAKE_SOURCE_DIR}/tests/TestPartialDecodeMultiFormats.cpp)
        add_executable(test_partial_decode_multi_formats tests/TestPartialDecodeMultiFormats.cpp)
        target_link_libraries(test_partial_decode_multi_formats llaminar_core GTest::gtest GTest::gtest_main)
        add_test(NAME PartialDecodeMultiFormatsTest COMMAND test_partial_decode_multi_formats)
    endif()
    # Aggregated parity test iterating present supported formats (Q4_0, Q5_0, Q8_0, K variants)
    if (EXISTS ${CMAKE_SOURCE_DIR}/tests/TestPartialDecodeAggregated.cpp)
        add_executable(test_partial_decode_aggregated tests/TestPartialDecodeAggregated.cpp)
        target_link_libraries(test_partial_decode_aggregated llaminar_core GTest::gtest GTest::gtest_main)
        add_test(NAME PartialDecodeAggregatedTest COMMAND test_partial_decode_aggregated)
    endif()
    if (EXISTS ${CMAKE_SOURCE_DIR}/tests/TestPartialDecodeKFormats.cpp)
        add_executable(test_partial_decode_k_formats tests/TestPartialDecodeKFormats.cpp)
        target_link_libraries(test_partial_decode_k_formats llaminar_core GTest::gtest GTest::gtest_main)
        add_test(NAME PartialDecodeKFormatsTest COMMAND test_partial_decode_k_formats)
    endif()

    # (Removed) Q5_1 layout micro test – deprecated quantization formats are no longer supported.
    # K-format (Q4_K / Q4_K_M) partial decode parity (added once implementation matures)
    # Placeholder future extension: we could add a dedicated test when models contain those tensors.
    set_tests_properties(QuantDequantCorrectnessTest PROPERTIES TIMEOUT 10)

    # ------------------------------------------------------------------
    # Test model download fixture: ensures required Qwen 0.5B GGUF variants
    # are present before any model-dependent tests run. Downloads are skipped
    # if files already exist. Users can export LLAMINAR_SKIP_MODEL_DOWNLOAD=1
    # to bypass network fetch (tests will then skip if models absent).
    # ------------------------------------------------------------------
    add_test(NAME FetchTestModels COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/fetch_test_models.sh)
    set_tests_properties(FetchTestModels PROPERTIES FIXTURES_SETUP TestModels TIMEOUT 420)

    # Helper macro to associate tests with the model fixture.
    function(mark_requires_models test_name)
        if (TEST ${test_name})
            set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED TestModels)
        endif()
    endfunction()

    # Explicit tests that rely on presence of GGUF model files.
    mark_requires_models(RoleTaggingIntegrationTest)
    mark_requires_models(ModelLoaderGoldenTest)
    mark_requires_models(ModelLoaderQuantShardRegression)
    mark_requires_models(IntegrationQ4K)
    mark_requires_models(PartialDecodeQ4_0Test)
    mark_requires_models(PartialDecodeAggregatedTest)
    mark_requires_models(PartialDecodeKFormatsTest)

    # End-to-end / parity tests that may scan models directory.
    mark_requires_models(End2EndShardParity_Small)
    mark_requires_models(End2EndShardParity_UnevenHeads)

    # Some GTest-discovered group tests live inside test_model_load_role_tags; we mark the container binary.
    mark_requires_models(RoleTaggingIntegrationTest)

    set(KERNEL_TEST_TARGETS
        test_mpi_linear_kernel
        test_mpi_rmsnorm_kernel
        test_mpi_attention_kernel
        test_attention_micro
        test_attention_primitives
        test_mpi_transformer_pipeline
        test_adaptive_matmul
        test_cosma_minimal
        test_cosma_prefill_manager
        test_cosma_prefill_manager_orientation_autofix
        test_cosma_prefill_manager_stats
        test_dequant
        test_dequant_extra
        test_dequant_golden
        test_integration_q4k)
else()
    message(STATUS "Google Test not found - kernel unit tests disabled")
    set(KERNEL_TEST_TARGETS "")
endif()

# Set output directories
set_target_properties(llaminar adaptive_demo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(test_basic test_numa test_cosma ${KERNEL_TEST_TARGETS}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# Custom test target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_basic test_numa test_cosma ${KERNEL_TEST_TARGETS}
    COMMENT "Running all tests"
)

# Add DEBUG flag to all test targets when in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # List of all test targets
    set(TEST_TARGETS test_basic test_numa test_cosma)
    
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

# ---------------------------------------------------------------------------
# Global Test Environment Injection (Threading & Topology Defaults)
#
# We want ctest-driven executions to inherit the same canonical threading
# defaults that the run_llaminar.sh launcher provides, without requiring
# contributors to manually export environment variables. This block appends
# (does not overwrite) two environment variables to every registered test:
#   LLAMINAR_OMP_USE_PHYSICAL=1   -> prefer physical core count for OpenMP
#   LLAMINAR_BIND_PER_SOCKET=1    -> engage per-socket binding heuristics
#
# If a test already defines ENVIRONMENT, we extend it unless the variables
# are already present (idempotent across re-configures). A user can still
# override behavior at runtime with LLAMINAR_OMP_FORCE which the code checks
# with higher precedence.
# ---------------------------------------------------------------------------
get_property(_ALL_LLAMINAR_TESTS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
if(_ALL_LLAMINAR_TESTS)
    foreach(_t IN LISTS _ALL_LLAMINAR_TESTS)
        get_test_property(${_t} ENVIRONMENT _CURRENT_ENV)
        if(_CURRENT_ENV)
            string(FIND "${_CURRENT_ENV}" "LLAMINAR_OMP_USE_PHYSICAL" _have_phys)
            if(_have_phys EQUAL -1)
                set(_CURRENT_ENV "${_CURRENT_ENV};LLAMINAR_OMP_USE_PHYSICAL=1")
            endif()
            string(FIND "${_CURRENT_ENV}" "LLAMINAR_BIND_PER_SOCKET" _have_bind)
            if(_have_bind EQUAL -1)
                set(_CURRENT_ENV "${_CURRENT_ENV};LLAMINAR_BIND_PER_SOCKET=1")
            endif()
            set_tests_properties(${_t} PROPERTIES ENVIRONMENT "${_CURRENT_ENV}")
        else()
            set_tests_properties(${_t} PROPERTIES ENVIRONMENT "LLAMINAR_OMP_USE_PHYSICAL=1;LLAMINAR_BIND_PER_SOCKET=1")
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

# Test dequantization verification tool (TODO: create test_dequant_verify.cpp)
# add_executable(test_dequant_verify test_dequant_verify.cpp)
# target_link_libraries(test_dequant_verify PRIVATE ggml)
# target_include_directories(test_dequant_verify PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include)

