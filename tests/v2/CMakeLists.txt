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
    
    # Create the test with mpirun wrapper
    add_test(NAME ${TEST_NAME} COMMAND ${MPI_CMD} ${ARG_COMMAND})
    
    # Set environment variables for optimal performance
    set_tests_properties(${TEST_NAME} PROPERTIES
        FIXTURES_REQUIRED V2_Models                    # Requires V2_Models fixture
        LABELS "${ARG_LABELS}"
        ENVIRONMENT "OMP_NUM_THREADS=${CORES_PER_SOCKET};OMP_PLACES=sockets;OMP_PROC_BIND=close;OMP_NESTED=false;OMP_DYNAMIC=false;KMP_AFFINITY=granularity=fine,compact,1,0;KMP_BLOCKTIME=0;OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET};GOTO_NUM_THREADS=${CORES_PER_SOCKET};MKL_NUM_THREADS=${CORES_PER_SOCKET};MKL_DYNAMIC=false;OMPI_MCA_mpi_leave_pinned=1;OMPI_MCA_btl_vader_single_copy_mechanism=none;OMPI_MCA_btl_openib_allow_ib=1"
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
    COMMAND v2_test_tensor_basics
    LABELS "V2;Unit;Tensor"
)

# Test: ModelLoader
add_executable(v2_test_model_loader unit/loaders/Test__ModelLoader.cpp)
target_link_libraries(v2_test_model_loader 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ModelLoader 
    COMMAND v2_test_model_loader
    LABELS "V2;Unit;ModelLoader"
)

# Test: IQ4_NL Tensor GEMM
add_executable(v2_test_iq4nl_tensor unit/tensors/Test__IQ4_NLTensor.cpp)
target_link_libraries(v2_test_iq4nl_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_NLTensor 
    COMMAND v2_test_iq4nl_tensor
    LABELS "V2;Unit;IQ4_NL;GEMM"
)

# Test: PipelineFactory
add_executable(v2_test_pipeline_factory unit/pipelines/Test__PipelineFactory.cpp)
target_link_libraries(v2_test_pipeline_factory 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PipelineFactory 
    COMMAND v2_test_pipeline_factory
    LABELS "V2;Unit;Pipeline;Factory"
)

# Placeholder for additional unit test executables (to be added)
# Example:
# add_executable(v2_test_tensor_factory unit/test_tensor_factory.cpp)
# target_link_libraries(v2_test_tensor_factory llaminar2_core GTest::gtest_main)
# add_v2_test(V2_Unit_TensorFactory 
#     COMMAND v2_test_tensor_factory
#     LABELS "V2;Unit;TensorFactory"
# )

# =============================================================================
# Integration Tests
# =============================================================================
# Medium-complexity tests involving multiple components
# May load models, test pipelines, verify end-to-end data flow

# Placeholder for integration test executables (to be added)
# Example:
# add_executable(v2_test_model_loader integration/test_model_loader.cpp)
# target_link_libraries(v2_test_model_loader llaminar2_core GTest::gtest_main)
# add_v2_test(V2_Integration_ModelLoader
#     COMMAND v2_test_model_loader
#     LABELS "V2;Integration;ModelLoader"
# )

# =============================================================================
# End-to-End Tests
# =============================================================================
# Full pipeline tests with real models and inference
# Tests complete workflows: load -> prefill -> decode -> validate output

# Placeholder for e2e test executables (to be added)
# Example:
# add_executable(v2_test_qwen_inference e2e/test_qwen_inference.cpp)
# target_link_libraries(v2_test_qwen_inference llaminar2_core GTest::gtest_main)
# add_v2_test(V2_E2E_QwenInference
#     COMMAND v2_test_qwen_inference
#     LABELS "V2;E2E;Qwen"
# )

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
#     COMMAND v2_test_pytorch_parity
#     LABELS "V2;Parity;PyTorch"
# )

# =============================================================================
# Performance Tests (Optional)
# =============================================================================
# Benchmarks for throughput, latency, memory usage
# Not part of standard test suites (run manually or in CI performance stage)

# Placeholder for performance test executables (to be added)
# Example:
# add_executable(v2_benchmark_gemm performance/benchmark_gemm.cpp)
# target_link_libraries(v2_benchmark_gemm llaminar2_core GTest::gtest_main)
# # Note: Performance tests don't use add_v2_test since they're not part of CTest by default
# # They can be run manually: ./build_v2/tests/v2/v2_benchmark_gemm

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

