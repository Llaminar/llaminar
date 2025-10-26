# Llaminar V2 Test Suite
# Test categories: unit, integration, e2e, pytorch_parity

# Note: enable_testing() is called in parent CMakeLists.txt (src/v2/CMakeLists.txt)

# =============================================================================
# CPU Topology Detection for Test Configuration
# =============================================================================
# Detect system topology to configure optimal MPI/OpenMP settings for tests
execute_process(
    COMMAND bash -c "lscpu | grep 'Socket(s):' | awk '{print $2}'"
    OUTPUT_VARIABLE V2_TEST_SOCKETS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND bash -c "lscpu | grep 'Core(s) per socket:' | awk '{print $4}'"
    OUTPUT_VARIABLE V2_TEST_CORES_PER_SOCKET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT V2_TEST_SOCKETS OR V2_TEST_SOCKETS EQUAL "")
    set(V2_TEST_SOCKETS 1)
endif()
if(NOT V2_TEST_CORES_PER_SOCKET OR V2_TEST_CORES_PER_SOCKET EQUAL "")
    set(V2_TEST_CORES_PER_SOCKET 1)
endif()

# Always use at least 2 MPI ranks for testing (even on single-socket systems)
if(V2_TEST_SOCKETS LESS 2)
    set(V2_TEST_MIN_RANKS 2)
    set(V2_TEST_NEEDS_OVERSUBSCRIBE TRUE)
    message(STATUS "V2 Test Configuration: ${V2_TEST_SOCKETS} socket(s), ${V2_TEST_CORES_PER_SOCKET} cores/socket")
    message(STATUS "V2 Tests will use: mpirun -np 2 --oversubscribe with OMP_NUM_THREADS=${V2_TEST_CORES_PER_SOCKET}")
else()
    set(V2_TEST_MIN_RANKS ${V2_TEST_SOCKETS})
    set(V2_TEST_NEEDS_OVERSUBSCRIBE FALSE)
    message(STATUS "V2 Test Configuration: ${V2_TEST_SOCKETS} sockets, ${V2_TEST_CORES_PER_SOCKET} cores/socket")
    message(STATUS "V2 Tests will use: mpirun -np ${V2_TEST_SOCKETS} with OMP_NUM_THREADS=${V2_TEST_CORES_PER_SOCKET}")
endif()

# =============================================================================
# Test Model Fetching Fixture
# =============================================================================
# All V2 tests require GGUF models to be present. This fixture ensures
# the fetch_test_models.sh script runs before any test suite executes.

# Path to the model fetching script (relative to this CMakeLists.txt)
set(FETCH_MODELS_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/../../scripts/fetch_test_models.sh")
set(PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

# Create a test that runs the fetch script
# This test will always pass (exit 0) but ensures models are downloaded
add_test(
    NAME V2_FetchModelsFixture
    COMMAND bash -c "bash ${FETCH_MODELS_SCRIPT} || exit 0"
    WORKING_DIRECTORY ${PROJECT_ROOT}
)

# Set properties for the fixture test
set_tests_properties(V2_FetchModelsFixture PROPERTIES
    FIXTURES_SETUP V2_Models           # This test provides the V2_Models fixture
    TIMEOUT 600                         # 10 minute timeout for large downloads
    LABELS "V2;Fixture;Models"
)

# =============================================================================
# Helper Function: Add V2 Test with Model Dependency and MPI
# =============================================================================
# Usage: add_v2_test(TestName COMMAND executable [args...] LABELS label1 label2 [MPI_PROCS n])
# This automatically:
#   - Wraps the test with mpirun and optimal MPI/OpenMP settings
#   - Makes the test depend on the V2_Models fixture
#   - Sets environment variables for optimal performance

function(add_v2_test TEST_NAME)
    cmake_parse_arguments(
        ARG                             # Prefix for parsed args
        ""                              # Options (boolean flags)
        "MPI_PROCS"                     # One-value keywords
        "COMMAND;LABELS"                # Multi-value keywords
        ${ARGN}                         # Arguments to parse
    )
    
    # Detect CPU topology for optimal thread/process counts
    # Use lscpu for reliable topology detection
    execute_process(
        COMMAND bash -c "lscpu | grep 'Socket(s):' | awk '{print $2}'"
        OUTPUT_VARIABLE NUM_SOCKETS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND bash -c "lscpu | grep 'Core(s) per socket:' | awk '{print $4}'"
        OUTPUT_VARIABLE CORES_PER_SOCKET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Fallback to safe defaults if detection fails
    if(NOT NUM_SOCKETS OR NUM_SOCKETS EQUAL "")
        set(NUM_SOCKETS 1)
    endif()
    if(NOT CORES_PER_SOCKET OR CORES_PER_SOCKET EQUAL "")
        set(CORES_PER_SOCKET 1)
    endif()
    
    # Default MPI processes: at least 2 (or user override)
    if(NOT DEFINED ARG_MPI_PROCS)
        if(NUM_SOCKETS LESS 2)
            set(ARG_MPI_PROCS 2)
        else()
            set(ARG_MPI_PROCS ${NUM_SOCKETS})
        endif()
    endif()
    
    # Build mpirun command with optimal settings
    set(MPI_CMD
        mpirun
        -np ${ARG_MPI_PROCS}
        --bind-to socket
        --map-by socket
        --mca mpi_leave_pinned 1
        --mca btl_vader_single_copy_mechanism none
    )
    
    # Add --oversubscribe if we have fewer sockets than MPI ranks
    if(NUM_SOCKETS LESS ARG_MPI_PROCS)
        list(APPEND MPI_CMD --oversubscribe)
    endif()
    
    # E2E and Integration tests should use Release build for performance
    # Unit tests can use Debug build for better diagnostics
    set(USE_RELEASE_BUILD FALSE)
    if("${ARG_LABELS}" MATCHES "E2E" OR "${ARG_LABELS}" MATCHES "Integration")
        set(USE_RELEASE_BUILD TRUE)
    endif()
    
    # Determine executable path based on build type
    if(USE_RELEASE_BUILD)
        # Use Release build (build_v2_release)
        set(RELEASE_BUILD_DIR "${CMAKE_SOURCE_DIR}/../../build_v2_release")
        list(GET ARG_COMMAND 0 EXECUTABLE_TARGET)
        # Extract just the target name (remove any path components)
        get_filename_component(EXECUTABLE_NAME ${EXECUTABLE_TARGET} NAME)
        
        # Check if this is a target file reference or direct path
        if(EXECUTABLE_TARGET MATCHES "\\$<TARGET_FILE:")
            # Extract target name from generator expression
            string(REGEX REPLACE ".*TARGET_FILE:([^>]+)>.*" "\\1" TARGET_NAME "${EXECUTABLE_TARGET}")
            set(EXECUTABLE_PATH "${RELEASE_BUILD_DIR}/tests/v2/${TARGET_NAME}")
        else()
            set(EXECUTABLE_PATH "${RELEASE_BUILD_DIR}/tests/v2/${EXECUTABLE_NAME}")
        endif()
        
        # Replace first argument with Release build path
        list(REMOVE_AT ARG_COMMAND 0)
        set(FINAL_COMMAND ${EXECUTABLE_PATH} ${ARG_COMMAND})
    else()
        # Use Debug build (default, CMAKE_CURRENT_BINARY_DIR)
        set(FINAL_COMMAND ${ARG_COMMAND})
    endif()
    
    # Create the test with mpirun wrapper
    add_test(NAME ${TEST_NAME} COMMAND ${MPI_CMD} ${FINAL_COMMAND})
    
    # Determine default log level based on test type (Unit/Integration = DEBUG, Performance = INFO)
    set(DEFAULT_LOG_LEVEL "DEBUG")
    if("${ARG_LABELS}" MATCHES "Performance")
        set(DEFAULT_LOG_LEVEL "INFO")
    endif()
    
    # Set environment variables for optimal performance + logging
    # Working directory: project root (so model paths like "models/..." work)
    # CMAKE_SOURCE_DIR is /workspaces/llaminar/src/v2, so go up two levels to reach project root
    set_tests_properties(${TEST_NAME} PROPERTIES
        FIXTURES_REQUIRED V2_Models                    # Requires V2_Models fixture
        LABELS "${ARG_LABELS}"
        ENVIRONMENT "LLAMINAR_LOG_LEVEL=${DEFAULT_LOG_LEVEL};OMP_NUM_THREADS=${CORES_PER_SOCKET};OMP_PLACES=sockets;OMP_PROC_BIND=close;OMP_NESTED=false;OMP_DYNAMIC=false;KMP_AFFINITY=granularity=fine,compact,1,0;KMP_BLOCKTIME=0;OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET};GOTO_NUM_THREADS=${CORES_PER_SOCKET};MKL_NUM_THREADS=${CORES_PER_SOCKET};MKL_DYNAMIC=false;OMPI_MCA_mpi_leave_pinned=1;OMPI_MCA_btl_vader_single_copy_mechanism=none;OMPI_MCA_btl_openib_allow_ib=1"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/../.."
    )
endfunction()

# =============================================================================
# Unit Tests
# =============================================================================
# Fast, isolated tests for individual components (tensors, kernels, etc.)
# No model loading, no MPI (or single-rank MPI only)
# Naming convention: Test files use CamelCase format Test__ClassName.cpp

# Find GTest (should be available from parent CMake)
find_package(GTest REQUIRED)

# Test: FP32Tensor class
add_executable(v2_test_tensor_basics unit/tensors/Test__FP32Tensor.cpp)
target_link_libraries(v2_test_tensor_basics 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorBasics 
    COMMAND $<TARGET_FILE:v2_test_tensor_basics>
    LABELS "V2;Unit;TensorOperations;FP32;Basics;GEMM"
)

# Test: BF16Tensor class
add_executable(v2_test_bf16_tensor unit/tensors/Test__BF16Tensor.cpp)
target_link_libraries(v2_test_bf16_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BF16Tensor 
    COMMAND $<TARGET_FILE:v2_test_bf16_tensor>
    LABELS "V2;Unit;TensorOperations;BF16;Quantization;GEMM"
)

# Test: KVCache
add_executable(v2_test_kv_cache unit/tensors/Test__KVCache.cpp)
target_link_libraries(v2_test_kv_cache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KVCache 
    COMMAND $<TARGET_FILE:v2_test_kv_cache>
    LABELS "V2;Unit;TensorOperations;KVCache;AutoregressiveDecode"
)

# Test: WeightPlacementMap
# WeightPlacementMap Tests (Tensor-Level Device Placement - Phase 2)
add_executable(v2_test_weight_placement_map unit/loaders/Test__WeightPlacementMap.cpp)
target_link_libraries(v2_test_weight_placement_map llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightPlacementMap
    COMMAND $<TARGET_FILE:v2_test_weight_placement_map>
    LABELS "V2;Unit;DeviceManagement;WeightPlacement;Heterogeneous;MoE"
)

# Test: PipelineBase Device Detection (Phase 3)
# Tests pipeline integration helpers: detectAttentionDevices, detectFFNDevices
add_executable(v2_test_pipeline_device_detection unit/pipelines/Test__PipelineBase_DeviceDetection.cpp)
target_link_libraries(v2_test_pipeline_device_detection llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PipelineDeviceDetection
    COMMAND $<TARGET_FILE:v2_test_pipeline_device_detection>
    LABELS "V2;Unit;PipelineExecution;DeviceManagement;MoE;Phase3"
)

# Test: ModelLoader
add_executable(v2_test_model_loader unit/loaders/Test__ModelLoader.cpp)
target_link_libraries(v2_test_model_loader 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ModelLoader 
    COMMAND $<TARGET_FILE:v2_test_model_loader>
    MPI_PROCS 1
    LABELS "V2;Unit;ModelLoading;GGUF;WeightLoading"
)

# Test: IQ4_NL Tensor GEMM
add_executable(v2_test_iq4nl_tensor unit/tensors/Test__IQ4_NLTensor.cpp)
target_link_libraries(v2_test_iq4nl_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_NLTensor 
    COMMAND $<TARGET_FILE:v2_test_iq4nl_tensor>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ4_NL;GEMM"
)

# Test: BatchedKVCache
add_executable(v2_test_batched_kv_cache unit/tensors/Test__BatchedKVCache.cpp)
target_link_libraries(v2_test_batched_kv_cache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BatchedKVCache
    COMMAND $<TARGET_FILE:v2_test_batched_kv_cache>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Batching;KVCache;MultiSequence"
)

# Test: TensorViews (regression test for view creation)
add_executable(v2_test_tensor_views unit/Test__TensorViews.cpp)
target_link_libraries(v2_test_tensor_views
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorViews
    COMMAND $<TARGET_FILE:v2_test_tensor_views>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Views;MemoryManagement;Regression"
)

# Test: Qwen2 Null MPI Context (regression test for null pointer crash)
add_executable(v2_test_qwen2_null_mpi_context unit/Test__Qwen2NullMPIContext.cpp)
target_link_libraries(v2_test_qwen2_null_mpi_context
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Qwen2NullMPIContext
    COMMAND $<TARGET_FILE:v2_test_qwen2_null_mpi_context>
    MPI_PROCS 1
    LABELS "V2;Unit;PipelineExecution;Qwen;SingleRank;Regression;Models"
    FIXTURES_REQUIRED "V2_Models"
)

# Test: BatchPaddingUtils
add_executable(v2_test_batch_padding_utils unit/utils/Test__BatchPaddingUtils.cpp)
target_link_libraries(v2_test_batch_padding_utils
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BatchPaddingUtils
    COMMAND $<TARGET_FILE:v2_test_batch_padding_utils>
    MPI_PROCS 1
    LABELS "V2;Unit;Batching;Padding;Masking;Bucketing"
)

# Test: PipelineFactory
add_executable(v2_test_pipeline_factory unit/pipelines/Test__PipelineFactory.cpp)
target_link_libraries(v2_test_pipeline_factory 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PipelineFactory 
    COMMAND $<TARGET_FILE:v2_test_pipeline_factory>
    LABELS "V2;Unit;PipelineExecution;Factory;Registration"
)

# Test: DeviceOrchestrator
add_executable(v2_test_device_orchestrator Test__DeviceOrchestrator.cpp)
target_link_libraries(v2_test_device_orchestrator 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceOrchestrator 
    COMMAND $<TARGET_FILE:v2_test_device_orchestrator>
    LABELS "V2;Unit;DeviceManagement;Orchestration;BasicFeatures"
    MPI_PROCS 1
)

# Test: ArgParser
add_executable(v2_test_arg_parser Test__ArgParser.cpp)
target_link_libraries(v2_test_arg_parser 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ArgParser 
    COMMAND $<TARGET_FILE:v2_test_arg_parser>
    LABELS "V2;Unit;CLI;ArgumentParsing"
)

# Test: Vectorized Primitives
add_executable(v2_test_vectorized_primitives Test__VectorizedPrimitives.cpp)
target_link_libraries(v2_test_vectorized_primitives 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_VectorizedPrimitives 
    COMMAND $<TARGET_FILE:v2_test_vectorized_primitives>
    LABELS "V2;Unit;Kernels;VectorizedPrimitives;RoPE;RMSNorm;Softmax;AVX512;AVX2"
    MPI_PROCS 1
)

# Test: MPI Vectorized Attention (Phase 2 Integration)
add_executable(v2_test_mpi_vectorized_attention Test__MPIVectorizedAttention.cpp)
target_link_libraries(v2_test_mpi_vectorized_attention 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_MPIVectorizedAttention 
    COMMAND $<TARGET_FILE:v2_test_mpi_vectorized_attention>
    LABELS "V2;Integration;MPI;Attention;PipelineExecution;VectorizedPrimitives;TensorParallel"
    MPI_PROCS 2
)# Test: DeviceOrchestrator Phase 2
add_executable(v2_test_device_orchestrator_phase2 Test__DeviceOrchestrator_Phase2.cpp)
target_link_libraries(v2_test_device_orchestrator_phase2 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceOrchestrator_Phase2 
    COMMAND $<TARGET_FILE:v2_test_device_orchestrator_phase2>
    LABELS "V2;Unit;DeviceManagement;Orchestration;AdvancedFeatures"
    MPI_PROCS 1
)

# Test: CPU Kernels (Phase 3)
add_executable(v2_test_cpu_kernels Test__CPUKernels.cpp)
target_link_libraries(v2_test_cpu_kernels 
    llaminar2_core 
    ${OPENBLAS_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUKernels 
    COMMAND $<TARGET_FILE:v2_test_cpu_kernels>
    LABELS "V2;Unit;Kernels;CPU;ComputeBackends"
    MPI_PROCS 1
)

# Test: MPI Tensor-Parallel Correctness (Phase 3a)
add_executable(v2_test_mpi_tensor_parallel_correctness Test__MPITensorParallelCorrectness.cpp)
target_link_libraries(v2_test_mpi_tensor_parallel_correctness 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_MPITensorParallelCorrectness 
    COMMAND $<TARGET_FILE:v2_test_mpi_tensor_parallel_correctness>
    LABELS "V2;Integration;MPI;TensorParallel;Attention;Correctness"
    MPI_PROCS 2
)

# Test: TensorDimensions (Phase 3)
add_executable(v2_test_tensor_dimensions Test__TensorDimensions.cpp)
target_link_libraries(v2_test_tensor_dimensions 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorDimensions 
    COMMAND $<TARGET_FILE:v2_test_tensor_dimensions>
    LABELS "V2;Unit;TensorOperations;Dimensions;ShapeHandling"
    MPI_PROCS 1
)

# Test: Multi-Device Support (Phase 4.1)
add_executable(v2_test_multi_device Test__MultiDevice.cpp)
target_link_libraries(v2_test_multi_device 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MultiDevice 
    COMMAND $<TARGET_FILE:v2_test_multi_device>
    LABELS "V2;Unit;DeviceManagement;MultiDevice;HeterogeneousExecution"
    MPI_PROCS 1
)

# Test: Device Transfer (Phase 4.2)
add_executable(v2_test_device_transfer Test__DeviceTransfer.cpp)
target_link_libraries(v2_test_device_transfer 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceTransfer 
    COMMAND $<TARGET_FILE:v2_test_device_transfer>
    LABELS "V2;Unit;DataTransfer;CrossDevice;BufferManagement"
    MPI_PROCS 1
)

# Test: Device-Aware Execution (Phase 4.3)
add_executable(v2_test_device_aware_execution Test__DeviceAwareExecution.cpp)
target_link_libraries(v2_test_device_aware_execution 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceAwareExecution 
    COMMAND $<TARGET_FILE:v2_test_device_aware_execution>
    LABELS "V2;Unit;DeviceManagement;Orchestration;DeviceAwareness"
    MPI_PROCS 1
)

# =============================================================================
# MPI Tensor-Parallel Tests (Phase 2)
# =============================================================================

add_executable(v2_test_mpi_tensor_parallel Test__MPITensorParallel.cpp)
target_link_libraries(v2_test_mpi_tensor_parallel 
    llaminar2_core 
    GTest::gtest 
    MPI::MPI_CXX
)
add_v2_test(V2_Unit_MPITensorParallel 
    COMMAND $<TARGET_FILE:v2_test_mpi_tensor_parallel>
    LABELS "V2;Unit;MPI;TensorParallel;StrategySelection;DistributedExecution"
    MPI_PROCS 2
)

# MPI Batched Attention E2E - End-to-end validation of tensor-parallel batched attention
add_executable(v2_test_mpi_batched_attention 
    integration/Test__MPIBatchedAttention.cpp
)
target_link_libraries(v2_test_mpi_batched_attention 
    llaminar2_core 
    GTest::gtest  # Provides its own MPI-aware main()
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_MPIBatchedAttention 
    COMMAND $<TARGET_FILE:v2_test_mpi_batched_attention>
    LABELS "V2;Integration;MPI;Batching;TensorParallel;E2E;Attention;Models"
    MPI_PROCS 2
)

# Placeholder for additional unit test executables (to be added)
# Example:
# add_executable(v2_test_tensor_factory unit/test_tensor_factory.cpp)
# target_link_libraries(v2_test_tensor_factory llaminar2_core GTest::gtest_main)
# add_v2_test(V2_Unit_TensorFactory 
#     COMMAND $<TARGET_FILE:v2_test_tensor_factory>
#     LABELS "V2;Unit;TensorFactory"
# )

# =============================================================================
# Integration Tests
# =============================================================================
# Medium-complexity tests involving multiple components
# May load models, test pipelines, verify end-to-end data flow

# Test: Batched Attention (Phase 2)
add_executable(v2_test_batched_attention integration/Test__BatchedAttention.cpp)
target_link_libraries(v2_test_batched_attention 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_BatchedAttention
    COMMAND $<TARGET_FILE:v2_test_batched_attention>
    MPI_PROCS 1
    LABELS "V2;Integration;Batching;Attention;CausalMasking;PaddingMasking;GQA"
)

# Placeholder for integration test executables (to be added)
# Example:
# add_executable(v2_test_model_loader integration/test_model_loader.cpp)
# target_link_libraries(v2_test_model_loader llaminar2_core GTest::gtest_main)
# add_v2_test(V2_Integration_ModelLoader
#     COMMAND $<TARGET_FILE:v2_test_model_loader>
#     LABELS "V2;Integration;ModelLoader"
# )

# =============================================================================
# End-to-End Tests (Phase 3c)
# =============================================================================
# Full pipeline tests with real models and inference
# Tests complete workflows: load -> prefill -> decode -> validate output

# Qwen2 E2E Correctness (single-rank vs multi-rank comparison)
add_executable(v2_test_qwen2_e2e_correctness 
    e2e/Test__Qwen2E2ECorrectness.cpp
    e2e/gtest_mpi_main.cpp
)
target_link_libraries(v2_test_qwen2_e2e_correctness llaminar2_core GTest::gtest MPI::MPI_CXX)

add_v2_test(V2_E2E_Qwen2Correctness
    COMMAND $<TARGET_FILE:v2_test_qwen2_e2e_correctness>
    LABELS "V2;E2E;Qwen;Correctness;Models"
    MPI_PROCS 2
)

# =============================================================================
# PyTorch Parity Tests
# =============================================================================
# Ground truth validation against PyTorch reference implementation
# Compares layer-by-layer outputs, logits, attention scores, etc.

# Placeholder for parity test executables (to be added)
# Example:
# add_executable(v2_test_pytorch_parity pytorch_parity/test_pytorch_parity.cpp)
# target_link_libraries(v2_test_pytorch_parity llaminar2_core GTest::gtest_main)
# add_v2_test(V2_Parity_PyTorch
#     COMMAND $<TARGET_FILE:v2_test_pytorch_parity>
#     LABELS "V2;Parity;PyTorch"
# )

# =============================================================================
# Performance Tests
# =============================================================================
# Benchmarks for throughput, latency, memory usage
# Performance tests require Release builds and optimal MPI/OpenMP settings
# They are integrated into CTest but labeled separately for manual execution
#
# Usage:
#   ctest -L Performance --verbose                    # Run all performance tests
#   ctest -L "Performance;GEMM" --verbose             # Run GEMM benchmarks
#   ctest -L "Performance;IQ4_NL" --verbose           # Run IQ4_NL benchmarks
#
# Performance tests automatically:
#   - Build/use Release configuration (build_v2_release)
#   - Apply optimal MPI/OpenMP settings (from run_benchmark.sh)
#   - Pin to physical cores for consistent measurements
#   - Report GFLOPS, bandwidth, and latency metrics

# Helper function: Add V2 Performance Test
# Similar to add_v2_test but optimized for performance measurement
function(add_v2_perf_test TEST_NAME)
    cmake_parse_arguments(
        ARG                             # Prefix for parsed args
        ""                              # Options (boolean flags)
        "MPI_PROCS"                     # One-value keywords
        "COMMAND;LABELS"                # Multi-value keywords
        ${ARGN}                         # Arguments to parse
    )
    
    # Detect CPU topology for optimal thread/process counts
    execute_process(
        COMMAND bash -c "lscpu | grep 'Socket(s):' | awk '{print $2}'"
        OUTPUT_VARIABLE NUM_SOCKETS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND bash -c "lscpu | grep 'Core(s) per socket:' | awk '{print $4}'"
        OUTPUT_VARIABLE CORES_PER_SOCKET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Fallback to safe defaults if detection fails
    if(NOT NUM_SOCKETS OR NUM_SOCKETS EQUAL "")
        set(NUM_SOCKETS 1)
    endif()
    if(NOT CORES_PER_SOCKET OR CORES_PER_SOCKET EQUAL "")
        set(CORES_PER_SOCKET 1)
    endif()
    
    # Default MPI processes: 1 for pure performance tests (no distribution overhead)
    if(NOT DEFINED ARG_MPI_PROCS)
        set(ARG_MPI_PROCS 1)
    endif()
    
    # Build mpirun command with optimal settings (from run_benchmark.sh)
    set(MPI_CMD
        mpirun
        -np ${ARG_MPI_PROCS}
        --bind-to socket
        --map-by socket
        --mca mpi_leave_pinned 1
        --mca btl_vader_single_copy_mechanism none
        --report-bindings               # Verify core pinning
        # Export OpenMP environment variables to MPI ranks
        -x OMP_NUM_THREADS=${CORES_PER_SOCKET}
        -x OMP_PLACES=sockets
        -x OMP_PROC_BIND=close
        -x OMP_NESTED=false
        -x OMP_DYNAMIC=false
        -x KMP_AFFINITY=granularity=fine,compact,1,0
        -x KMP_BLOCKTIME=0
        -x OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET}
        -x GOTO_NUM_THREADS=${CORES_PER_SOCKET}
        -x MKL_NUM_THREADS=${CORES_PER_SOCKET}
        -x MKL_DYNAMIC=false
    )
    
    # Add --oversubscribe if we have fewer sockets than MPI ranks
    if(NUM_SOCKETS LESS ARG_MPI_PROCS)
        list(APPEND MPI_CMD --oversubscribe)
    endif()
    
    # Check if Release build exists, warn if not
    set(RELEASE_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../../build_v2_release")
    if(NOT EXISTS ${RELEASE_BUILD_DIR})
        message(STATUS "Performance tests require Release build (build_v2_release)")
        message(STATUS "Will attempt to build Release configuration automatically")
        
        # Add custom target to build Release if needed
        add_custom_target(ensure_v2_release_build
            COMMAND ${CMAKE_COMMAND} -E echo "Building V2 Release configuration..."
            COMMAND ${CMAKE_COMMAND} -B ${RELEASE_BUILD_DIR} -S ${CMAKE_CURRENT_LIST_DIR}/../.. -DCMAKE_BUILD_TYPE=Release
            COMMAND ${CMAKE_COMMAND} --build ${RELEASE_BUILD_DIR} --parallel
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Ensuring V2 Release build exists for performance tests"
        )
    endif()
    
    # Create the test with mpirun wrapper
    # Use full path to executable so it works regardless of working directory
    list(GET ARG_COMMAND 0 EXECUTABLE_NAME)
    set(EXECUTABLE_PATH "${CMAKE_BINARY_DIR}/performance/${EXECUTABLE_NAME}")
    list(REMOVE_AT ARG_COMMAND 0)
    add_test(NAME ${TEST_NAME} COMMAND ${MPI_CMD} ${EXECUTABLE_PATH} ${ARG_COMMAND})
    
    # Set environment variables for optimal performance (from run_benchmark.sh)
    # Working directory: project root so model paths work correctly
    # Performance tests default to INFO log level (less verbose than Unit/Integration)
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "${ARG_LABELS}"
        ENVIRONMENT "LLAMINAR_LOG_LEVEL=INFO;OMP_NUM_THREADS=${CORES_PER_SOCKET};OMP_PLACES=sockets;OMP_PROC_BIND=close;OMP_NESTED=false;OMP_DYNAMIC=false;KMP_AFFINITY=granularity=fine,compact,1,0;KMP_BLOCKTIME=0;OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET};GOTO_NUM_THREADS=${CORES_PER_SOCKET};MKL_NUM_THREADS=${CORES_PER_SOCKET};MKL_DYNAMIC=false;OMPI_MCA_mpi_leave_pinned=1;OMPI_MCA_btl_vader_single_copy_mechanism=none;OMPI_MCA_btl_openib_allow_ib=1"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../.."
    )
    
    # Note: Performance tests do NOT require V2_Models fixture
    # They control model loading themselves for benchmarking
endfunction()

# Performance Test: IQ4_NL Quantized GEMM
# Benchmarks IQ4_NL quantized matrix multiplication with FP32 and BF16 activations
# Test cases: SingleToken, SmallBatch, MediumBatch, LargeBatch (prefill)
add_executable(v2_perf_iq4nl_gemm performance/Perf__IQ4_NL_GEMM.cpp)
target_link_libraries(v2_perf_iq4nl_gemm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_iq4nl_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_IQ4NL_GEMM
    COMMAND $<TARGET_FILE:v2_perf_iq4nl_gemm>
    LABELS "V2;Performance;GEMM;IQ4_NL;Quantization;Throughput"
    MPI_PROCS 2  # Match V1 baseline (2 ranks) for fair comparison
)

# Performance Test: Qwen2 E2E Pipeline Throughput
# Benchmarks full pipeline: model loading, prefill, decode
# Measures tokens/sec, GFLOPS, memory usage
add_executable(v2_perf_qwen2_e2e 
    performance/Perf__Qwen2E2EPerformance.cpp
    e2e/gtest_mpi_main.cpp
)
target_link_libraries(v2_perf_qwen2_e2e 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
set_target_properties(v2_perf_qwen2_e2e PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Qwen2E2E
    COMMAND $<TARGET_FILE:v2_perf_qwen2_e2e>
    LABELS "V2;Performance;E2E;Qwen;Pipeline;Throughput;Models"
    MPI_PROCS 2
)
# Performance Test: MPI Tensor-Parallel (Phase 3b)
add_executable(v2_perf_mpi_tensor_parallel performance/Perf__MPITensorParallel.cpp)
target_link_libraries(v2_perf_mpi_tensor_parallel 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
set_target_properties(v2_perf_mpi_tensor_parallel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)
add_v2_perf_test(V2_Perf_MPITensorParallel 
    COMMAND $<TARGET_FILE:v2_perf_mpi_tensor_parallel>
    LABELS "V2;Performance;MPI;TensorParallel;Attention;Speedup"
    MPI_PROCS 2
)

# Profiling tool: GEMM hotpath analysis
# TEMPORARILY DISABLED - needs update for IBlockDecoder interface
# add_executable(v2_profile_gemm_hotpath performance/profile_gemm_hotpath.cpp)
# target_link_libraries(v2_profile_gemm_hotpath llaminar2_core)
# set_target_properties(v2_profile_gemm_hotpath PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )

# Perf: IQ4_NL GEMM tile sweep optimization
add_executable(v2_perf_iq4nl_tile_sweep performance/Perf__IQ4_NL_TileSweep.cpp)
target_link_libraries(v2_perf_iq4nl_tile_sweep 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    MPI::MPI_CXX
)
set_target_properties(v2_perf_iq4nl_tile_sweep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)
add_v2_perf_test(V2_Perf_IQ4NL_TileSweep
    COMMAND $<TARGET_FILE:v2_perf_iq4nl_tile_sweep>
    LABELS "V2;Performance;TensorOperations;Quantization;IQ4_NL;GEMM;CacheOptimization"
    MPI_PROCS 2
)

# Perf: L1 cache optimization comparison (Original vs L1-Opt)
add_executable(v2_perf_l1_opt_comparison performance/Perf__L1OptimizationComparison.cpp)
target_link_libraries(v2_perf_l1_opt_comparison 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
set_target_properties(v2_perf_l1_opt_comparison PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)
add_v2_perf_test(V2_Perf_L1OptComparison
    COMMAND $<TARGET_FILE:v2_perf_l1_opt_comparison>
    LABELS "V2;Performance;TensorOperations;Quantization;IQ4_NL;GEMM;CacheOptimization;L1Cache"
    MPI_PROCS 2
)

# Future performance tests can be added here following the same pattern:
# Example:
# add_executable(v2_perf_attention performance/Perf__Attention.cpp)
# target_link_libraries(v2_perf_attention llaminar2_core GTest::gtest GTest::gtest_main)
# add_v2_perf_test(V2_Perf_Attention
#     COMMAND $<TARGET_FILE:v2_perf_attention>
#     LABELS "V2;Performance;Attention;Throughput"
# )

# =============================================================================
# Test Suite Labels
# =============================================================================
# V2 tests use hierarchical, feature-based labels for flexible filtering:
#
# TIER 1 - Test Type (Required):
#   Unit                    - Isolated component tests (no model loading)
#   Integration             - Multi-component tests (may load models)
#   E2E                     - End-to-end pipeline tests (full inference)
#   Parity                  - Ground truth validation (PyTorch/llama.cpp)
#   Performance             - Benchmarks (run manually, not in CTest)
#
# TIER 2 - Architecture (Optional):
#   V2                      - V2 architecture tests
#
# TIER 3 - Component (Specific to what's being tested):
#   DeviceManagement        - Device orchestration, discovery, selection
#   TensorOperations        - Tensor creation, manipulation, conversion
#   Kernels                 - Computational kernels (GEMM, RoPE, Attention)
#   ModelLoading            - GGUF parsing, weight loading
#   PipelineExecution       - Pipeline lifecycle, forward pass
#   WeightPlacement         - Weight distribution strategies
#   DataTransfer            - Cross-device buffer management
#   ArgumentParsing         - CLI argument handling
#
# TIER 4 - Feature (Granular specifics):
#   Orchestration           - Orchestrator-specific features
#   Quantization            - Quantized tensor formats
#   MultiDevice             - Heterogeneous execution
#   CrossDevice             - Cross-device operations
#   HeterogeneousExecution  - Mixed device types
#   BasicFeatures           - Core functionality
#   AdvancedFeatures        - Extended capabilities
#   etc. (feature-specific)
#
# FILTERING EXAMPLES:
#   ctest -L "V2;Unit"                          # All V2 unit tests
#   ctest -L DeviceManagement                   # All device management tests
#   ctest -L "TensorOperations;Quantization"    # All quantized tensor tests
#   ctest -L Orchestration                      # All orchestrator tests
#   ctest -L GEMM                               # All GEMM-related tests
#   ctest -R "V2_Unit_.*"                       # All V2 unit tests (by name)
#
# NAMING CONVENTIONS:
#   - Use CamelCase for label names (matches file naming)
#   - Be specific: "PipelineFactory" not "Factory"
#   - Feature-based, not timeline-based: "WeightPlacement" not "Phase1"
#   - Multiple labels per test enable flexible filtering dimensions
#
# =============================================================================
# Test Suite Labels
# =============================================================================
# Tests can be run by category using CTest filters:
#
#   ctest -L V2                    # All V2 tests
#   ctest -L "V2;Unit"             # V2 unit tests only
#   ctest -L "V2;Integration"      # V2 integration tests only
#   ctest -L "V2;E2E"              # V2 end-to-end tests only
#   ctest -L "V2;Parity"           # V2 PyTorch parity tests only
#   ctest -R "V2_Unit_.*"          # All V2 unit tests (by name regex)
#
# The model fetching fixture (V2_FetchModelsFixture) will run automatically
# before any test that requires the V2_Models fixture.

# =============================================================================
# Notes on Test Dependencies
# =============================================================================
# The CMake FIXTURES system ensures:
#   1. V2_FetchModelsFixture runs before any test requiring V2_Models
#   2. Models are fetched only once per CTest invocation
#   3. If model fetch fails, dependent tests still run (script returns 0)
#   4. Individual tests can check for model availability and SKIP if needed
#
# Environment variables respected by fetch_test_models.sh:
#   LLAMINAR_SKIP_MODEL_DOWNLOAD=1    # Skip all downloads (use existing)
#   LLAMINAR_FETCH_EXPERIMENTAL=1     # Include experimental/404-prone models
#   LLAMINAR_FETCH_FP32=1             # Include large FP32 models
#   LLAMINAR_FETCH_7B_IQ_MODELS=1     # Include large 7B IQ models
#   LLAMINAR_ENFORCE_MODELS=1         # Fail if no models present after fetch

