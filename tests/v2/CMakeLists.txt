# Llaminar V2 Test Suite
# Test categories: unit, integration, e2e, pytorch_parity

# Note: enable_testing() is called in parent CMakeLists.txt (src/v2/CMakeLists.txt)

# =============================================================================
# Linker Workaround for Circular Static Library Dependencies
# =============================================================================
# llaminar2_core links cuda_backend, but cuda_backend uses symbols from llaminar2_core.
# On GNU ld, we need --start-group / --end-group to resolve circular dependencies.
# This function wraps target_link_libraries to add the needed linker flags.
function(target_link_libraries_with_groups target)
    if(UNIX AND NOT APPLE)
        # Add start-group before libraries and end-group after
        target_link_libraries(${target} PRIVATE 
            -Wl,--start-group
            ${ARGN}
            -Wl,--end-group
        )
    else()
        # On macOS/others, just link normally
        target_link_libraries(${target} PRIVATE ${ARGN})
    endif()
endfunction()

set(V2_ENABLE_CUDA_GPU_CONTEXT_TESTS_DEFAULT ON)
if(HAVE_CUDA AND HAVE_ROCM)
    # CUDA device-code registration for these context/graph-capture tests can
    # fault before main() in dual-backend builds. ROCm variants keep coverage in
    # the default suite; enable this option when debugging CUDA startup itself.
    set(V2_ENABLE_CUDA_GPU_CONTEXT_TESTS_DEFAULT OFF)
endif()
set(V2_CUDA_GPU_CONTEXT_TESTS_OPT_IN_HINT
    "Reconfigure with -DV2_ENABLE_CUDA_GPU_CONTEXT_TESTS=ON to run the _CUDA split tests")
option(V2_ENABLE_CUDA_GPU_CONTEXT_TESTS
    "Enable CUDA GPU context/graph-capture split tests in dual-backend builds"
    ${V2_ENABLE_CUDA_GPU_CONTEXT_TESTS_DEFAULT})
if(HAVE_CUDA AND NOT V2_ENABLE_CUDA_GPU_CONTEXT_TESTS)
    message(STATUS "V2 Tests: Macro-generated CUDA GPU context/graph-capture split tests will be built but registered disabled. ${V2_CUDA_GPU_CONTEXT_TESTS_OPT_IN_HINT}.")
endif()

# Build GPU tests as one binary per backend. CUDA and HIP device runtimes can
# register linked device code before main(), so combined CUDA+ROCm test binaries
# can fail before gtest skip guards run. Split binaries preserve backend coverage
# while keeping startup deterministic.
function(add_v2_gpu_backend_test TARGET_BASE SOURCE TEST_BASE)
    set(TEST_LABELS ${ARGN})

    if(HAVE_ROCM)
        set(ROCM_TARGET ${TARGET_BASE}_rocm)
        add_executable(${ROCM_TARGET} ${SOURCE})
        target_compile_definitions(${ROCM_TARGET} PRIVATE GPU_CONTEXT_TEST_BACKEND_ROCM=1)
        if(LLAMINAR_SHARED_CORE)
            target_link_libraries(${ROCM_TARGET} PRIVATE
                llaminar2_core
                GTest::gtest
                GTest::gtest_main
            )
        else()
            target_link_libraries(${ROCM_TARGET} PRIVATE
                -Wl,--start-group
                llaminar2_core
                -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive
                -Wl,--end-group
                GTest::gtest
                GTest::gtest_main
            )
        endif()
        add_v2_test(${TEST_BASE}_ROCm
            COMMAND $<TARGET_FILE:${ROCM_TARGET}>
            LABELS ${TEST_LABELS} ROCm
            MPI_PROCS 1
        )
    endif()

    if(HAVE_CUDA)
        set(CUDA_TARGET ${TARGET_BASE}_cuda)
        add_executable(${CUDA_TARGET} ${SOURCE})
        target_compile_definitions(${CUDA_TARGET} PRIVATE GPU_CONTEXT_TEST_BACKEND_CUDA=1)
        if(LLAMINAR_SHARED_CORE)
            target_link_libraries(${CUDA_TARGET} PRIVATE
                llaminar2_core
                GTest::gtest
                GTest::gtest_main
            )
        else()
            target_link_libraries(${CUDA_TARGET} PRIVATE
                -Wl,--start-group
                llaminar2_core
                -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive
                -Wl,--end-group
                GTest::gtest
                GTest::gtest_main
            )
        endif()
        add_v2_test(${TEST_BASE}_CUDA
            COMMAND $<TARGET_FILE:${CUDA_TARGET}>
            LABELS ${TEST_LABELS} CUDA
            MPI_PROCS 1
        )
        if(NOT V2_ENABLE_CUDA_GPU_CONTEXT_TESTS AND TEST ${TEST_BASE}_CUDA)
            set_tests_properties(${TEST_BASE}_CUDA PROPERTIES
                DISABLED TRUE
                DISABLED_REASON "Macro-generated CUDA context/graph-capture split tests are opt-in for dual-backend builds. ${V2_CUDA_GPU_CONTEXT_TESTS_OPT_IN_HINT}."
            )
        endif()
    endif()

    if(NOT HAVE_ROCM AND NOT HAVE_CUDA)
        set(CPU_ONLY_TARGET ${TARGET_BASE})
        add_executable(${CPU_ONLY_TARGET} ${SOURCE})
        target_link_libraries(${CPU_ONLY_TARGET} PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
        add_v2_test(${TEST_BASE}
            COMMAND $<TARGET_FILE:${CPU_ONLY_TARGET}>
            LABELS ${TEST_LABELS}
            MPI_PROCS 1
        )
    endif()
endfunction()

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
# Release-mode guard: only build performance test targets
# =============================================================================
# When V2_PERF_TESTS_ONLY is ON (set by parent for Release builds), we override
# add_executable so that only targets with the v2_perf_ prefix are created.
# All other test executables (v2_test_*, v2_integration_*, v2_unit_*,
# v2_orchestration_*, v2_e2e_*) become no-ops, keeping the Release build lean.
# The add_v2_test() and discover_v2_parity_tests() functions also early-return
# for non-perf tests when this flag is set.
if(V2_PERF_TESTS_ONLY)
    message(STATUS "V2 Tests: Release mode — building performance tests only")

    # Override add_executable: only pass through v2_perf_* targets
    macro(add_executable target)
        if("${target}" MATCHES "^v2_perf_")
            _add_executable(${target} ${ARGN})
        endif()
    endmacro()

    # Skip target_link_libraries for non-existent (skipped) targets
    macro(target_link_libraries target)
        if(TARGET ${target})
            _target_link_libraries(${target} ${ARGN})
        endif()
    endmacro()

    # Skip target_include_directories for non-existent (skipped) targets
    macro(target_include_directories target)
        if(TARGET ${target})
            _target_include_directories(${target} ${ARGN})
        endif()
    endmacro()

    # Skip target_compile_definitions for non-existent (skipped) targets
    macro(target_compile_definitions target)
        if(TARGET ${target})
            _target_compile_definitions(${target} ${ARGN})
        endif()
    endmacro()

    # Skip target_compile_options for non-existent (skipped) targets
    macro(target_compile_options target)
        if(TARGET ${target})
            _target_compile_options(${target} ${ARGN})
        endif()
    endmacro()
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

if(NOT V2_PERF_TESTS_ONLY)
    add_test(
        NAME V2_Unit_Static_NoKernelFactoryPreparedEmbeddingRegistry
        COMMAND bash ${PROJECT_ROOT}/scripts/check_no_kernel_factory_prepared_embedding_registry.sh ${PROJECT_ROOT}
        WORKING_DIRECTORY ${PROJECT_ROOT}
    )
    set_tests_properties(V2_Unit_Static_NoKernelFactoryPreparedEmbeddingRegistry PROPERTIES
        LABELS "V2;Unit;StaticAnalysis;Weights;PreparedWeights;KernelFactory;Embedding"
        TIMEOUT 30
    )

    add_test(
        NAME V2_Unit_Static_NoDirectGetenvOutsideDebugEnv
        COMMAND bash ${PROJECT_ROOT}/scripts/check_no_direct_getenv_outside_debugenv.sh ${PROJECT_ROOT}
        WORKING_DIRECTORY ${PROJECT_ROOT}
    )
    set_tests_properties(V2_Unit_Static_NoDirectGetenvOutsideDebugEnv PROPERTIES
        LABELS "V2;Unit;StaticAnalysis;DebugEnv;Environment"
        TIMEOUT 30
    )

    add_test(
        NAME V2_Unit_Static_NoDefaultStreamInGPUCode
        COMMAND bash ${PROJECT_ROOT}/scripts/check_no_default_stream_in_gpu_code.sh ${PROJECT_ROOT}
        WORKING_DIRECTORY ${PROJECT_ROOT}
    )
    set_tests_properties(V2_Unit_Static_NoDefaultStreamInGPUCode PROPERTIES
        LABELS "V2;Unit;StaticAnalysis;GPU;CUDA;ROCm;Streams"
        TIMEOUT 30
    )

    add_test(
        NAME V2_Unit_Static_NoROCmHotpathHiddenAllocations
        COMMAND bash ${PROJECT_ROOT}/scripts/check_no_rocm_hotpath_hidden_allocations.sh ${PROJECT_ROOT}
        WORKING_DIRECTORY ${PROJECT_ROOT}
    )
    set_tests_properties(V2_Unit_Static_NoROCmHotpathHiddenAllocations PROPERTIES
        LABELS "V2;Unit;StaticAnalysis;GPU;ROCm;Workspace;GraphCapture"
        TIMEOUT 30
    )
endif()

# =============================================================================
# Helper Function: Add V2 Test with Model Dependency and MPI
# =============================================================================
# Usage: add_v2_test(TestName COMMAND executable [args...] LABELS label1 label2 [MPI_PROCS n])
# This automatically:
#   - Wraps the test with mpirun and optimal MPI/OpenMP settings
#   - Makes the test depend on the V2_Models fixture
#   - Sets environment variables for optimal performance

function(add_v2_test TEST_NAME)
    # In Release mode (V2_PERF_TESTS_ONLY), skip non-performance tests
    if(V2_PERF_TESTS_ONLY AND NOT "${TEST_NAME}" MATCHES "^V2_Perf_")
        return()
    endif()

    cmake_parse_arguments(
        ARG                             # Prefix for parsed args
        ""                              # Options (boolean flags)
        "MPI_PROCS;TIMEOUT"             # One-value keywords
        "COMMAND;LABELS;RESOURCE_LOCK"  # Multi-value keywords
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
        --mca orte_allowed_exit_without_sync 1
    )
    
    # Add --oversubscribe if we have fewer sockets than MPI ranks
    if(NUM_SOCKETS LESS ARG_MPI_PROCS)
        list(APPEND MPI_CMD --oversubscribe)
    endif()
    
    # Integration tests use build_v2_integration (Release + snapshots + debug symbols)
    # E2E tests use build_v2_e2e_release
    # Unit tests use current build (Debug or whatever is configured)
    set(USE_INTEGRATION_BUILD FALSE)
    set(USE_E2E_BUILD FALSE)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        if("${ARG_LABELS}" MATCHES "Integration")
            set(USE_INTEGRATION_BUILD TRUE)
        elseif("${ARG_LABELS}" MATCHES "E2E")
            set(USE_E2E_BUILD TRUE)
        endif()
    endif()
    
    # Determine executable path based on test type
    if(USE_INTEGRATION_BUILD)
        # Use Integration build (build_v2_integration)
        set(INTEGRATION_BUILD_DIR "${CMAKE_SOURCE_DIR}/../../build_v2_integration")
        list(GET ARG_COMMAND 0 EXECUTABLE_TARGET)
        get_filename_component(EXECUTABLE_NAME ${EXECUTABLE_TARGET} NAME)
        
        if(EXECUTABLE_TARGET MATCHES "\\$<TARGET_FILE:")
            string(REGEX REPLACE ".*TARGET_FILE:([^>]+)>.*" "\\1" TARGET_NAME "${EXECUTABLE_TARGET}")
            set(EXECUTABLE_PATH "${INTEGRATION_BUILD_DIR}/tests/v2/${TARGET_NAME}")
        else()
            set(EXECUTABLE_PATH "${INTEGRATION_BUILD_DIR}/tests/v2/${EXECUTABLE_NAME}")
        endif()
        
        list(REMOVE_AT ARG_COMMAND 0)
        set(FINAL_COMMAND ${EXECUTABLE_PATH} ${ARG_COMMAND})
    elseif(USE_E2E_BUILD)
        # Use E2E Release build (build_v2_e2e_release)
        set(E2E_BUILD_DIR "${CMAKE_SOURCE_DIR}/../../build_v2_e2e_release")
        list(GET ARG_COMMAND 0 EXECUTABLE_TARGET)
        get_filename_component(EXECUTABLE_NAME ${EXECUTABLE_TARGET} NAME)
        
        if(EXECUTABLE_TARGET MATCHES "\\$<TARGET_FILE:")
            string(REGEX REPLACE ".*TARGET_FILE:([^>]+)>.*" "\\1" TARGET_NAME "${EXECUTABLE_TARGET}")
            set(EXECUTABLE_PATH "${E2E_BUILD_DIR}/tests/v2/${TARGET_NAME}")
        else()
            set(EXECUTABLE_PATH "${E2E_BUILD_DIR}/tests/v2/${EXECUTABLE_NAME}")
        endif()
        
        list(REMOVE_AT ARG_COMMAND 0)
        set(FINAL_COMMAND ${EXECUTABLE_PATH} ${ARG_COMMAND})
    else()
        # Use Debug build (default, CMAKE_CURRENT_BINARY_DIR)
        # Check if COMMAND uses $<TARGET_FILE:...> generator expression
        list(GET ARG_COMMAND 0 FIRST_ARG)
        if(FIRST_ARG MATCHES "\\$<TARGET_FILE:")
            # Already has full path via generator expression, use as-is
            set(FINAL_COMMAND ${ARG_COMMAND})
        else()
            # Bare executable name, need to prepend CMAKE_CURRENT_BINARY_DIR
            set(EXECUTABLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${FIRST_ARG}")
            list(REMOVE_AT ARG_COMMAND 0)
            set(FINAL_COMMAND ${EXECUTABLE_PATH} ${ARG_COMMAND})
        endif()
    endif()
    
    # Create the test with mpirun wrapper
    add_test(NAME ${TEST_NAME} COMMAND ${MPI_CMD} ${FINAL_COMMAND})
    
    # Determine default log level based on test type (Unit/Integration = DEBUG, Performance = INFO)
    set(DEFAULT_LOG_LEVEL "DEBUG")
    if("${ARG_LABELS}" MATCHES "Performance")
        set(DEFAULT_LOG_LEVEL "INFO")
    endif()
    
    # Build base environment variables for optimal performance + logging
    set(TEST_ENV 
        "LLAMINAR_LOG_LEVEL=${DEFAULT_LOG_LEVEL}"
        "LLAMINAR_TP_COLLECT_TIMEOUT_MS=30000"
        "HWLOC_COMPONENTS=-gl,-opencl"
        "OMP_NUM_THREADS=${CORES_PER_SOCKET}"
        "OMP_PLACES=sockets"
        "OMP_PROC_BIND=close"
        "OMP_NESTED=false"
        "OMP_DYNAMIC=false"
        "KMP_AFFINITY=granularity=fine,compact,1,0"
        "KMP_BLOCKTIME=0"
        "MKL_NUM_THREADS=${CORES_PER_SOCKET}"
        "MKL_DYNAMIC=false"
        "OMPI_MCA_mpi_leave_pinned=1"
        "OMPI_MCA_btl_vader_single_copy_mechanism=none"
        "OMPI_MCA_btl_openib_allow_ib=1"
    )
    
    # Add ROCm environment variables for ROCm tests
    # HSA_OVERRIDE_GFX_VERSION=9.0.6 enables gfx906 (MI50/MI60) codepaths on compatible GPUs
    if("${ARG_LABELS}" MATCHES "ROCm")
        list(APPEND TEST_ENV "HSA_OVERRIDE_GFX_VERSION=9.0.6")
    endif()
    
    # Set environment variables
    # Working directory: project root (so model paths like "models/..." work)
    # CMAKE_SOURCE_DIR is /workspaces/llaminar/src/v2, so go up two levels to reach project root
    set_tests_properties(${TEST_NAME} PROPERTIES
        FIXTURES_REQUIRED V2_Models                    # Requires V2_Models fixture
        LABELS "${ARG_LABELS}"
        ENVIRONMENT "${TEST_ENV}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/../.."
    )
    
    # =========================================================================
    # PROCESSORS: Tell CTest how many CPU slots each test consumes.
    #
    # CTest uses PROCESSORS to cap concurrent tests so the total does not
    # exceed the --parallel limit (defaults to nproc = 112 on 2×28c HT).
    # Setting each test to 4 processors means at most ~28 tests run
    # concurrently (112 / 4), which avoids MPI/OpenMP oversubscription
    # while still allowing good parallelism across the test suite.
    # =========================================================================
    set_tests_properties(${TEST_NAME} PROPERTIES
        PROCESSORS 4
    )

    # Add RESOURCE_LOCK if specified (prevents parallel execution of tests needing same resource)
    # All integration tests share a single lock to force serial execution even
    # under `ctest --parallel`.  Parity tests have their own lock
    # ("Parity_Model_Execution") set by discover_v2_parity_tests(), so they are
    # unaffected here.
    if("${ARG_LABELS}" MATCHES "Integration")
        list(APPEND ARG_RESOURCE_LOCK "Integration_Serial")
    endif()
    if(ARG_RESOURCE_LOCK)
        set_tests_properties(${TEST_NAME} PROPERTIES
            RESOURCE_LOCK "${ARG_RESOURCE_LOCK}"
        )
    endif()

    # Add TIMEOUT if specified (useful for slow integration/parity tests)
    if(ARG_TIMEOUT)
        set_tests_properties(${TEST_NAME} PROPERTIES
            TIMEOUT ${ARG_TIMEOUT}
        )
    endif()
endfunction()

# =============================================================================
# discover_v2_parity_tests() — Dynamic GTest discovery with per-test process isolation
#
# Runs the test binary with --gtest_list_tests at POST_BUILD time (no mpirun),
# parses discovered test cases, and generates individual CTest entries for each.
# Each test case runs in its own process, avoiding state accumulation bugs
# (e.g. ROCm CLR corruption between RCCL init/destroy cycles).
#
# Usage:
#   discover_v2_parity_tests(my_target
#       TEST_PREFIX V2_Integration_Parity_Qwen2_SingleDevice
#       LABELS "V2;Integration;Parity;..."
#       MPI_PROCS 1
#   )
# =============================================================================
function(discover_v2_parity_tests TARGET)
    # In Release mode, parity tests are never perf tests — skip entirely
    if(V2_PERF_TESTS_ONLY)
        return()
    endif()

    cmake_parse_arguments(ARG "" "MPI_PROCS;TEST_PREFIX;TIMEOUT" "LABELS" ${ARGN})

    if(NOT ARG_TEST_PREFIX)
        set(ARG_TEST_PREFIX "${TARGET}")
    endif()
    if(NOT ARG_MPI_PROCS)
        set(ARG_MPI_PROCS 1)
    endif()

    # Paths for generated artifacts
    set(_ctest_file "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_discovered_tests.cmake")

    # Working directory: project root (CMAKE_SOURCE_DIR is src/v2, go up two levels)
    get_filename_component(_working_dir "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)

    # POST_BUILD: run discovery script (no mpirun needed, just the binary directly)
    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -D "TEST_EXECUTABLE=$<TARGET_FILE:${TARGET}>"
            -D "CTEST_FILE=${_ctest_file}"
            -D "TEST_PREFIX=${ARG_TEST_PREFIX}"
            -D "LABELS=${ARG_LABELS}"
            -D "MPI_PROCS=${ARG_MPI_PROCS}"
            -D "NUM_SOCKETS=${V2_TEST_SOCKETS}"
            -D "CORES_PER_SOCKET=${V2_TEST_CORES_PER_SOCKET}"
            -D "WORKING_DIR=${_working_dir}"
            -D "TIMEOUT=${ARG_TIMEOUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/V2ParityTestDiscovery.cmake"
        BYPRODUCTS "${_ctest_file}"
        COMMENT "Discovering parity tests in ${TARGET}..."
        VERBATIM
    )

    # Tell CTest to include the generated file (tests appear after first build)
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_ctest_file}")

    # Create an empty file so CTest doesn't error before the first build
    if(NOT EXISTS "${_ctest_file}")
        file(WRITE "${_ctest_file}" "# Placeholder — tests will be discovered after building ${TARGET}\n")
    endif()
endfunction()

# =============================================================================
# Unit Tests
# =============================================================================
# Fast, isolated tests for individual components (tensors, kernels, etc.)
# No model loading, no MPI (or single-rank MPI only)
# Naming convention: Test files use CamelCase format Test__ClassName.cpp

# Find GTest (should be available from parent CMake)
find_package(GTest REQUIRED)

# Enable pipeline snapshot capture for test builds (matches core library)
# This compile definition enables the snapshot infrastructure in PipelineBase.
# In release builds (without this flag), all snapshot code compiles away to NOOPs.
# CRITICAL: Must match core library definition to avoid ODR violations!
# We also set a CMake variable for conditional test registration.
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
    set(ENABLE_PIPELINE_SNAPSHOTS TRUE)
elseif(CMAKE_BUILD_TYPE MATCHES "E2ERelease")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
    set(ENABLE_PIPELINE_SNAPSHOTS TRUE)
elseif(CMAKE_BUILD_TYPE MATCHES "Integration")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
    set(ENABLE_PIPELINE_SNAPSHOTS TRUE)
elseif(ENABLE_SNAPSHOTS)
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
    set(ENABLE_PIPELINE_SNAPSHOTS TRUE)
endif()

# =============================================================================
# Test Utilities Library
# =============================================================================
# Shared utilities for tests (TestOrchestrationHelper, etc.)
# Tests that need these utilities link against v2_test_utils

add_library(v2_test_utils STATIC
    utils/TestOrchestrationHelper.cpp
)
target_link_libraries(v2_test_utils PUBLIC
    llaminar2_core
    GTest::gtest
)
target_include_directories(v2_test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}  # For utils/*.h includes
    ${V2_SOURCE_DIR}             # For src/v2/ includes
)

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

# Test: ExpertGemmRegistry - MoE expert GEMM engine registry
add_executable(v2_unit_expert_gemm_registry unit/loaders/Test__ExpertGemmRegistry.cpp)
target_link_libraries(v2_unit_expert_gemm_registry
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpertGemmRegistry
    COMMAND $<TARGET_FILE:v2_unit_expert_gemm_registry>
    LABELS "V2;Unit;Loaders;MoE"
    MPI_PROCS 1
)

# Test: Unified Pipeline MoE Collection - MoE expert weight collection in GPU pipeline
add_executable(v2_unit_unified_pipeline_moe unit/loaders/Test__UnifiedPipelineMoECollection.cpp)
target_include_directories(v2_unit_unified_pipeline_moe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes
)
target_link_libraries(v2_unit_unified_pipeline_moe
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_UnifiedPipelineMoECollection
    COMMAND $<TARGET_FILE:v2_unit_unified_pipeline_moe>
    LABELS "V2;Unit;Loaders;MoE"
    MPI_PROCS 1
)

add_executable(v2_unit_gpu_vram_preflight
    unit/loaders/Test__GPUVramPreflight.cpp
)
target_link_libraries(v2_unit_gpu_vram_preflight
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GPUVramPreflight
    COMMAND $<TARGET_FILE:v2_unit_gpu_vram_preflight>
    LABELS "V2;Unit;Loaders;GpuPipeline;Diagnostics"
    MPI_PROCS 1
)

# Test: MoE Expert Overlay Preparation - tier-aware preparation requests and CPU fallback packing
add_executable(v2_unit_weight_manager_moe_expert_overlay_preparation
    unit/loaders/Test__WeightManagerMoEExpertOverlayPreparation.cpp
)
target_include_directories(v2_unit_weight_manager_moe_expert_overlay_preparation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_unit_weight_manager_moe_expert_overlay_preparation
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_WeightManagerMoEExpertOverlayPreparation
    COMMAND $<TARGET_FILE:v2_unit_weight_manager_moe_expert_overlay_preparation>
    LABELS "V2;Unit;Loaders;MoE;ExpertParallel"
    MPI_PROCS 1
)

# Test: MoE Expert Parallel Plan - same-layer expert overlay value validation
add_executable(v2_unit_moe_expert_parallel_plan unit/execution/moe/Test__MoEExpertParallelPlan.cpp)
target_link_libraries(v2_unit_moe_expert_parallel_plan
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEExpertParallelPlan
    COMMAND $<TARGET_FILE:v2_unit_moe_expert_parallel_plan>
    LABELS "V2;Unit;Execution;MoE;ExpertParallel;Configuration"
    MPI_PROCS 1
)

# Test: MoE Expert Overlay Runtime Plan - resolved domain roles and rank plans
add_executable(v2_unit_moe_expert_overlay_runtime_plan
    unit/execution/moe/Test__MoEExpertOverlayRuntimePlan.cpp
)
target_link_libraries(v2_unit_moe_expert_overlay_runtime_plan
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEExpertOverlayRuntimePlan
    COMMAND $<TARGET_FILE:v2_unit_moe_expert_overlay_runtime_plan>
    LABELS "V2;Unit;Execution;MoE;ExpertParallel;Configuration"
    MPI_PROCS 1
)

# Test: MoE Expert Parallel Planner - static same-layer expert residency planning
add_executable(v2_unit_moe_expert_parallel_planner unit/execution/moe/Test__MoEExpertParallelPlanner.cpp)
target_link_libraries(v2_unit_moe_expert_parallel_planner
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEExpertParallelPlanner
    COMMAND $<TARGET_FILE:v2_unit_moe_expert_parallel_planner>
    LABELS "V2;Unit;Execution;MoE;ExpertParallel;Planner"
    MPI_PROCS 1
)

# Test: MoE graph-native/stage source hygiene guardrails
add_executable(v2_unit_moe_forbidden_dependency_scan unit/execution/moe/Test__MoEGraphNative_ForbiddenDependencyScan.cpp)
target_link_libraries(v2_unit_moe_forbidden_dependency_scan
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEForbiddenDependencyScan
    COMMAND $<TARGET_FILE:v2_unit_moe_forbidden_dependency_scan>
    LABELS "V2;Unit;Execution;MoE;SourceHygiene;GraphNative"
    MPI_PROCS 1
)

# Test: MoE overlay collective workspace and key namespace behavior
add_executable(v2_unit_moe_overlay_collective_workspace
    unit/execution/moe/Test__MoEOverlayCollectiveWorkspace.cpp
)
target_include_directories(v2_unit_moe_overlay_collective_workspace PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_unit_moe_overlay_collective_workspace
    llaminar2_core
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEOverlayCollectiveWorkspace
    COMMAND $<TARGET_FILE:v2_unit_moe_overlay_collective_workspace>
    LABELS "V2;Unit;Execution;MoE;Overlay;SparseCollective"
    MPI_PROCS 1
)

# Test: MoE graph-native profiler rows and machine-readable metrics
add_executable(v2_unit_moe_graph_native_profiling_metrics
    unit/execution/moe/Test__MoEGraphNativeProfilingMetrics.cpp
)
target_link_libraries(v2_unit_moe_graph_native_profiling_metrics
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEGraphNativeProfilingMetrics
    COMMAND $<TARGET_FILE:v2_unit_moe_graph_native_profiling_metrics>
    LABELS "V2;Unit;Execution;MoE;GraphNative;Profiler;SparseCollective"
    MPI_PROCS 1
)

# Test: Tensor Host Release Lifecycle (CPU-only tensor release guard regression)
# Regression tests for the bug where releaseAllHostWeightData() released CPU-only
# tensors that were never uploaded to GPU, causing null data access in graph stages.
# Tests: FP32 release basics, isDeviceValid() guard pattern, Q8_0 quantized tensor
# release, double-release idempotency, metadata preservation after release.
add_executable(v2_test_tensor_host_release unit/tensors/Test__TensorHostRelease.cpp)
target_link_libraries(v2_test_tensor_host_release
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorHostRelease
    COMMAND $<TARGET_FILE:v2_test_tensor_host_release>
    LABELS "V2;Unit;TensorOperations;HostRelease;Regression;Lifecycle;FP32;Q8_0"
)

# Test: TensorAllocateOnDevice - CPUTensorBase::allocateOnDevice() method
add_executable(v2_test_tensor_allocate_on_device unit/tensors/Test__TensorAllocateOnDevice.cpp)
target_link_libraries(v2_test_tensor_allocate_on_device 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_TensorAllocateOnDevice 
    COMMAND $<TARGET_FILE:v2_test_tensor_allocate_on_device>
    LABELS "V2;Integration;TensorOperations;GPU;Coherence;DeviceMemory"
    MPI_PROCS 1
)

# Test: TensorByteSize - Verify byte_size() implementation for all tensor types
# This is critical because CPUTensorBase::byte_size() throws if not overridden,
# which causes GPU transfer failures.
add_executable(v2_test_tensor_byte_size unit/tensors/Test__TensorByteSize.cpp)
target_link_libraries(v2_test_tensor_byte_size 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorByteSize 
    COMMAND $<TARGET_FILE:v2_test_tensor_byte_size>
    LABELS "V2;Unit;TensorOperations;ByteSize;Quantization;GPU"
    MPI_PROCS 1
)

# Test: TensorBase TransferTo - Direct GPU-to-GPU transfer (Phase 2 GPU-Native Coherence)
# Test: TensorBase Authoritative Device tracking - Multi-device coherence API
# NOTE: These tests have been moved to the integration test suite (integration/tensors/)
#       because they require GPU allocation and fail with OOM when run in parallel
#       with other GPU-heavy unit tests.

# Test: Qwen2BufferSpec - Buffer allocation strategy
add_executable(v2_test_qwen2_buffer_spec unit/models/qwen/Test__Qwen2BufferSpec.cpp)
target_link_libraries(v2_test_qwen2_buffer_spec 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Qwen2BufferSpec 
    COMMAND $<TARGET_FILE:v2_test_qwen2_buffer_spec>
    LABELS "V2;Unit;BufferAllocation;Heterogeneous;TensorParallelism"
    MPI_PROCS 1
)

# Test: CacheInfo - CPU cache detection and batch size optimization
add_executable(v2_test_cache_info unit/utils/Test__CacheInfo.cpp)
target_link_libraries(v2_test_cache_info 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CacheInfo 
    COMMAND $<TARGET_FILE:v2_test_cache_info>
    LABELS "V2;Unit;CPUFeatures;Cache;Optimization"
    MPI_PROCS 1
)

# Test: StageDumpConfig - Environment variable parsing and filtering for stage dumps
add_executable(v2_test_stage_dump_config unit/execution/debug/Test__StageDumpConfig.cpp)
target_link_libraries(v2_test_stage_dump_config 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_StageDumpConfig 
    COMMAND $<TARGET_FILE:v2_test_stage_dump_config>
    LABELS "V2;Unit;Debug;StageDump;Config"
    MPI_PROCS 1
)

# Test: CrossDomainTransfer - Cross-domain activation transfer utility
add_executable(v2_test_cross_domain_transfer unit/execution/local_execution/coherence/Test__CrossDomainTransfer.cpp)
target_link_libraries(v2_test_cross_domain_transfer 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_test_cross_domain_transfer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
add_v2_test(V2_Unit_CrossDomainTransfer 
    COMMAND $<TARGET_FILE:v2_test_cross_domain_transfer>
    LABELS "V2;Unit;Execution;Transfer;Heterogeneous"
    MPI_PROCS 1
)

# Test: DeviceGroup - Device group construction and predicates for collective operations
add_executable(v2_unit_device_group unit/collective/Test__DeviceGroup.cpp)
target_link_libraries(v2_unit_device_group 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceGroup 
    COMMAND $<TARGET_FILE:v2_unit_device_group>
    LABELS "V2;Unit;Collective;DeviceGroup;Topology"
    MPI_PROCS 1
)

# Test: PPStage and HierarchicalPPConfig - Hierarchical PP stage model and configuration
add_executable(v2_unit_ppstage_hierarchical_config unit/collective/Test__PPStageAndHierarchicalPPConfig.cpp)
target_include_directories(v2_unit_ppstage_hierarchical_config PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_ppstage_hierarchical_config 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PPStage_HierarchicalConfig 
    COMMAND $<TARGET_FILE:v2_unit_ppstage_hierarchical_config>
    LABELS "V2;Unit;Collective;PPStage;HierarchicalPPConfig;PipelineParallelism;TensorParallel"
    MPI_PROCS 1
)

# Test: PPStage Global TP - GLOBAL_TP_DOMAIN support in PPStage
# Note: Uses mpi_gtest_main.cpp for MPI initialization (needed for GlobalTPContext tests with MPI_COMM_SELF)
add_executable(v2_unit_ppstage_globaltp 
    unit/collective/Test__PPStage_GlobalTP.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_unit_ppstage_globaltp 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
add_v2_test(V2_Unit_PPStage_GlobalTP 
    COMMAND $<TARGET_FILE:v2_unit_ppstage_globaltp>
    LABELS "V2;Unit;Collective;PPStage;GlobalTP;Phase5"
    MPI_PROCS 1
)

# Test: LocalPPContext - LOCAL pipeline parallelism context configuration and transfer tracking
add_executable(v2_unit_localppcontext unit/collective/Test__LocalPPContext.cpp)
target_include_directories(v2_unit_localppcontext PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_localppcontext 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_LocalPPContext 
    COMMAND $<TARGET_FILE:v2_unit_localppcontext>
    LABELS "V2;Unit;Collective;LocalPP;PipelineParallelism"
    MPI_PROCS 1
)

# Integration Test: LocalPPContext - LOCAL pipeline parallelism integration with OrchestrationRunner
# Tests: context creation, backend selection, transfer flow, Scenario 7 (PP of TP domains)
add_executable(v2_integration_local_pp_context integration/pipeline_parallel/Test__LocalPPIntegration.cpp)
target_include_directories(v2_integration_local_pp_context PRIVATE 
    ${CMAKE_SOURCE_DIR}          # For v2/ includes
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_local_pp_context 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Integration_LocalPP 
    COMMAND $<TARGET_FILE:v2_integration_local_pp_context>
    LABELS "V2;Integration;Collective;LocalPP;PipelineParallelism;OrchestrationRunner"
    MPI_PROCS 1
)

# Integration Test: HierarchicalPPTransfers - Smart source selection and broadcast for PP transfers
# Tests: findBestSourceDevice() for heterogeneous TP domains, broadcast to TP domain
# Validates optimizations in HierarchicalPPContext::transferFromTPDomain() and transferToTPDomain()
add_executable(v2_integration_hierarchical_pp_transfers integration/pipeline_parallel/Test__HierarchicalPPTransfers.cpp)
target_include_directories(v2_integration_hierarchical_pp_transfers PRIVATE 
    ${CMAKE_SOURCE_DIR}          # For v2/ includes
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_hierarchical_pp_transfers 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Integration_HierarchicalPP_Transfers 
    COMMAND $<TARGET_FILE:v2_integration_hierarchical_pp_transfers>
    LABELS "V2;Integration;Collective;LocalPP;PipelineParallelism;HierarchicalPPConfig;SmartSourceSelection;Broadcast"
    MPI_PROCS 1
)

# Integration Test: HierarchicalPP_GlobalTP - PP transfers with GLOBAL_TP_DOMAIN stages (Phase 5)
# Tests: PP transfers from local TP → global TP → single device
# Validates: PPStage::fromGlobalTPContext(), GlobalTPContext integration, cross-rank coordination
# Requires: MPI with 2 ranks for cross-rank global TP testing
# Note: Uses mpi_gtest_main.cpp for MPI initialization (needed for cross-rank tests)
add_executable(v2_integration_hierarchical_pp_globaltp 
    integration/pipeline_parallel/Test__HierarchicalPP_GlobalTP.cpp
    integration/mpi_gtest_main.cpp
)
target_include_directories(v2_integration_hierarchical_pp_globaltp PRIVATE 
    ${CMAKE_SOURCE_DIR}          # For v2/ includes
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_hierarchical_pp_globaltp 
    llaminar2_core 
    GTest::gtest
    GTest::gmock
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_HierarchicalPP_GlobalTP 
    COMMAND $<TARGET_FILE:v2_integration_hierarchical_pp_globaltp>
    LABELS "V2;Integration;Collective;LocalPP;PipelineParallelism;GlobalTP;Phase5"
    MPI_PROCS 2
)

# Integration Test: UnifiedPP_Execution - Unified PP graph execution via executeMultiDevice()
# Tests: Multi-device contexts, LocalPPTransferStage execution, data flow through PP pipeline
# Requires real model weights (Qwen2.5-0.5B)
add_executable(v2_integration_unified_pp_execution integration/pp/Test__UnifiedPP_Execution.cpp)
target_include_directories(v2_integration_unified_pp_execution PRIVATE 
    ${CMAKE_SOURCE_DIR}          # For v2/ includes
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_unified_pp_execution 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Integration_UnifiedPP_Execution 
    COMMAND $<TARGET_FILE:v2_integration_unified_pp_execution>
    LABELS "V2;Integration;PipelineParallelism;UnifiedPPGraph;Execution;QwenStandardGraph"
    MPI_PROCS 1
)



# Cross-vendor integration tests (require both CUDA and ROCm)
if(HAVE_CUDA AND HAVE_ROCM)

    # Test: CUDA Vector Add Kernels - Accuracy tests for FP32/FP16/BF16/INT8/INT32 reduction kernels
    add_executable(v2_integration_cuda_vector_add_kernels integration/kernels/cuda/ops/Test__CUDAVectorAddKernels.cpp)
    target_link_libraries(v2_integration_cuda_vector_add_kernels 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_CUDAVectorAddKernels 
        COMMAND $<TARGET_FILE:v2_integration_cuda_vector_add_kernels>
        LABELS "V2;Integration;CUDA;Kernels;VectorAdd;FP16;BF16;INT8"
        MPI_PROCS 1
    )

    # Test: CollectiveBackendIntegration - Comprehensive collective backend tests
    # Tests CPU+CPU (MPI), NVIDIA+NVIDIA (NCCL), AMD+AMD (RCCL), NVIDIA+AMD (HOST),
    # and complex heterogeneous scenarios with multi-domain tensor parallelism
    add_executable(v2_integration_collective_backend integration/collective/Test__CollectiveBackendIntegration.cpp)
    target_link_libraries(v2_integration_collective_backend 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    if(NOT LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_collective_backend device_benchmarks)
    endif()
    target_include_directories(v2_integration_collective_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    # Multi-rank collective tests (requires 2 MPI ranks)
    add_v2_test(V2_Integration_CollectiveBackendIntegration 
        COMMAND $<TARGET_FILE:v2_integration_collective_backend>
        LABELS "V2;Integration;Collective;BackendRouter;MPI;NCCL;RCCL;TensorParallel;Heterogeneous;AllReduce"
        MPI_PROCS 2
    )
    # Single-rank collective tests (requires exactly 1 MPI rank)
    # These tests verify single-rank optimizations like no-op AllReduce
    add_v2_test(V2_Integration_CollectiveSingleRank
        COMMAND $<TARGET_FILE:v2_integration_collective_backend> --gtest_filter="Test__CollectiveSingleRank.*"
        LABELS "V2;Integration;Collective;SingleRank;AllReduce"
        MPI_PROCS 1
    )

endif()

# Test: CrossDomainTransferGPU - GPU integration tests for cross-domain activation transfer
# Requires either CUDA or ROCm - tests actual PCIe transfers
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_cross_domain_transfer_gpu integration/execution/coherence/Test__CrossDomainTransferGPU.cpp)
    target_link_libraries(v2_integration_cross_domain_transfer_gpu 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    target_include_directories(v2_integration_cross_domain_transfer_gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    add_v2_test(V2_Integration_CrossDomainTransferGPU 
        COMMAND $<TARGET_FILE:v2_integration_cross_domain_transfer_gpu>
        LABELS "V2;Integration;Execution;Transfer;Heterogeneous;GPU"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CrossDomainTransfer GPU integration tests")
endif()

# Test: BackendRouter - Backend selection and routing for collective operations
add_executable(v2_unit_backend_router unit/collective/Test__BackendRouter.cpp)
target_link_libraries(v2_unit_backend_router 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BackendRouter 
    COMMAND $<TARGET_FILE:v2_unit_backend_router>
    LABELS "V2;Unit;Collective;BackendRouter;Routing"
    MPI_PROCS 1
)

# Test: IBufferRegistration - Buffer registration interface for collective backends
add_executable(v2_unit_buffer_registration unit/collective/Test__IBufferRegistration.cpp)
target_link_libraries(v2_unit_buffer_registration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BufferRegistration 
    COMMAND $<TARGET_FILE:v2_unit_buffer_registration>
    LABELS "V2;Unit;Collective;BufferRegistration;Interface"
    MPI_PROCS 1
)

# Test: HostBackend - CPU-based fallback collective backend
add_executable(v2_unit_host_backend unit/collective/backends/Test__HostBackend.cpp)
target_link_libraries(v2_unit_host_backend 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HostBackend 
    COMMAND $<TARGET_FILE:v2_unit_host_backend>
    LABELS "V2;Unit;Collective;HostBackend;CPU"
    MPI_PROCS 1
)

# Test: CPUBackend - CPU/NUMA backend implementing IBackend interface
add_executable(v2_unit_cpu_backend unit/backends/Test__CPUBackend.cpp)
target_link_libraries(v2_unit_cpu_backend 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Unit_CPUBackend 
    COMMAND $<TARGET_FILE:v2_unit_cpu_backend>
    LABELS "V2;Unit;Backend;CPUBackend;NUMA;Memory"
    MPI_PROCS 1
)

# Test: DeviceWorkspaceManager - Per-device GPU workspace buffer management
add_executable(v2_unit_device_workspace_manager unit/execution/local_execution/device/Test__DeviceWorkspaceManager.cpp)
target_link_libraries(v2_unit_device_workspace_manager 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceWorkspaceManager 
    COMMAND $<TARGET_FILE:v2_unit_device_workspace_manager>
    LABELS "V2;Unit;Execution;DeviceWorkspaceManager;Memory;Allocation"
    MPI_PROCS 1
)

# Test: WorkspaceAllocator - per-device workspace reuse and growth
add_executable(v2_unit_workspace_allocator unit/execution/local_execution/device/Test__WorkspaceAllocator.cpp)
target_link_libraries(v2_unit_workspace_allocator
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_WorkspaceAllocator
    COMMAND $<TARGET_FILE:v2_unit_workspace_allocator>
    LABELS "V2;Unit;Execution;WorkspaceAllocator;Memory;Allocation;Regression"
    MPI_PROCS 1
)

# Test: IWorkspaceConsumerStage - Mixin for stages that delegate workspace to kernels
add_executable(v2_unit_workspace_consumer_stage unit/execution/compute_stages/Test__IWorkspaceConsumerStage.cpp)
target_link_libraries(v2_unit_workspace_consumer_stage 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IWorkspaceConsumerStage 
    COMMAND $<TARGET_FILE:v2_unit_workspace_consumer_stage>
    LABELS "V2;Unit;Execution;ComputeStages;Workspace;IWorkspaceConsumer;Mixin"
    MPI_PROCS 1
)

# Test: ROCmQuantisedGemmKernel_Workspace - IWorkspaceConsumer interface integration
add_executable(v2_unit_rocm_gemm_workspace unit/kernels/rocm/Test__ROCmQuantisedGemmKernel_Workspace.cpp)
target_link_libraries(v2_unit_rocm_gemm_workspace 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GemmWorkspaceConsumer
    COMMAND $<TARGET_FILE:v2_unit_rocm_gemm_workspace>
    LABELS "V2;Unit;Kernels;Workspace;IWorkspaceConsumer"
    MPI_PROCS 1
)

# Test: PackedWeightsSerialization - Round-trip serialization/deserialization of packed GEMM weights
# Tests: Q4_0 round-trip, asymmetric Q4_1, WithNativeBlocks, empty sections, invalid magic/version,
#        truncated buffer, zero-size, large weights, size calculation, int8_flat, superblock format
add_executable(v2_unit_packed_weights_serialization unit/kernels/Test__PackedWeightsSerialization.cpp)
target_link_libraries(v2_unit_packed_weights_serialization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PackedWeightsSerialization
    COMMAND $<TARGET_FILE:v2_unit_packed_weights_serialization>
    LABELS "V2;Unit;Kernels;Serialization;PackedWeights;MoE"
    MPI_PROCS 1
)

# Test: GPUPackedWeightsStubs - Phase 4 GPU packed weight stub IPackedWeights implementations
# Tests: CUDA/ROCm construction, format, clone, convertTo, serialize, compatibility, empty data
add_executable(v2_unit_gpu_packed_weights_stubs unit/kernels/Test__GPUPackedWeightsStubs.cpp)
target_link_libraries(v2_unit_gpu_packed_weights_stubs
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PackedWeightsInterface
    COMMAND $<TARGET_FILE:v2_unit_gpu_packed_weights_stubs>
    LABELS "V2;Unit;Kernels;PackedWeights;IPackedWeights"
    MPI_PROCS 1
)

# Test: ExpertWeightTransfer — MoE cross-rank weight transfer logic
add_executable(v2_unit_expert_weight_transfer unit/moe/Test__ExpertWeightTransfer.cpp)
target_link_libraries(v2_unit_expert_weight_transfer
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpertWeightTransfer
    COMMAND $<TARGET_FILE:v2_unit_expert_weight_transfer>
    LABELS "V2;Unit;MoE;WeightTransfer;PackedWeights"
    MPI_PROCS 1
)

# Test: MPIBackend - MPI-based collective backend for inter-node communication
add_executable(v2_unit_mpi_backend unit/collective/backends/Test__MPIBackend.cpp)
target_link_libraries(v2_unit_mpi_backend 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MPIBackend 
    COMMAND $<TARGET_FILE:v2_unit_mpi_backend>
    LABELS "V2;Unit;Collective;MPIBackend;MPI"
    MPI_PROCS 1
)

# Test: UPIBackend - UPI-based collective backend for cross-socket CPU tensor parallelism
add_executable(v2_unit_upi_backend unit/collective/backends/Test__UPIBackend.cpp)
target_link_libraries(v2_unit_upi_backend 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_UPIBackend 
    COMMAND $<TARGET_FILE:v2_unit_upi_backend>
    LABELS "V2;Unit;Collective;UPIBackend;UPI;CPU"
    MPI_PROCS 1
)

# Test: NCCLBackend - NCCL collective backend for NVIDIA GPUs (P2P operations)
if(HAVE_CUDA)
    add_executable(v2_unit_nccl_backend unit/collective/backends/Test__NCCLBackend.cpp)
    target_link_libraries(v2_unit_nccl_backend 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_NCCLBackend_P2P 
        COMMAND $<TARGET_FILE:v2_unit_nccl_backend>
        LABELS "V2;Integration;Collective;NCCLBackend;P2P;CUDA;NCCL"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added NCCLBackend P2P integration tests")
endif()

# Test: RCCLBackend - RCCL collective backend for AMD GPUs (P2P operations)
if(HAVE_ROCM)
    add_executable(v2_unit_rccl_backend unit/collective/backends/Test__RCCLBackend.cpp)
    target_link_libraries(v2_unit_rccl_backend 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_RCCLBackend_P2P 
        COMMAND $<TARGET_FILE:v2_unit_rccl_backend>
        LABELS "V2;Integration;Collective;RCCLBackend;P2P;ROCm;RCCL"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added RCCLBackend P2P integration tests")
endif()

# Test: HeterogeneousBackend - Heterogeneous multi-GPU collective backend (CUDA + ROCm)
# NOTE: Unit tests work even without hardware by testing validation logic
add_executable(v2_unit_heterogeneous_backend unit/collective/backends/Test__HeterogeneousBackend.cpp)
target_link_libraries(v2_unit_heterogeneous_backend 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
if(NOT LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_unit_heterogeneous_backend device_benchmarks)
endif()
add_v2_test(V2_Unit_HeterogeneousBackend 
    COMMAND $<TARGET_FILE:v2_unit_heterogeneous_backend>
    LABELS "V2;Unit;Collective;HeterogeneousBackend;CUDA;ROCm;CrossVendor"
    MPI_PROCS 1
)

# Test: CollectiveContext - Runtime context for collective operations
add_executable(v2_unit_collective_context unit/execution/local_execution/collective/Test__CollectiveContext.cpp)
target_link_libraries(v2_unit_collective_context 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CollectiveContext 
    COMMAND $<TARGET_FILE:v2_unit_collective_context>
    LABELS "V2;Unit;Collective;CollectiveContext;Execution"
    MPI_PROCS 1
)

# Test: GraphExecutorCollective - CollectiveContext integration with DeviceGraphExecutor
add_executable(v2_unit_graph_executor_collective unit/execution/local_execution/graph/Test__GraphExecutorCollective.cpp)
target_include_directories(v2_unit_graph_executor_collective PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_graph_executor_collective 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GraphExecutorCollective 
    COMMAND $<TARGET_FILE:v2_unit_graph_executor_collective>
    LABELS "V2;Unit;Collective;DeviceGraphExecutor;Execution"
    MPI_PROCS 1
)

# Test: StageRunPolicy - Unified execution policy tests
add_executable(v2_unit_stage_run_policy unit/execution/local_execution/graph/Test__StageRunPolicy.cpp)
target_include_directories(v2_unit_stage_run_policy PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_stage_run_policy 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_StageRunPolicy
    COMMAND $<TARGET_FILE:v2_unit_stage_run_policy>
    LABELS "V2;Unit;DeviceGraphExecutor;Execution;StageRunPolicy"
    MPI_PROCS 1
)

# Test: StageTimeline - GPU event-based timeline profiler regression tests
add_executable(v2_unit_stage_timeline unit/execution/local_execution/graph/Test__StageTimeline.cpp)
target_include_directories(v2_unit_stage_timeline PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_unit_stage_timeline 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_StageTimeline
    COMMAND $<TARGET_FILE:v2_unit_stage_timeline>
    LABELS "V2;Unit;DeviceGraphExecutor;Execution;StageTimeline;GraphCapture"
    MPI_PROCS 1
)

# Test: MockMPIContext - Mock MPI context for unit testing distributed logic
add_executable(v2_unit_mock_mpi_context unit/mocks/Test__MockMPIContext.cpp)
target_include_directories(v2_unit_mock_mpi_context PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_mpi_context 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockMPIContext 
    COMMAND $<TARGET_FILE:v2_unit_mock_mpi_context>
    LABELS "V2;Unit;Mock;MPI;Testing"
    MPI_PROCS 1
)

# Test: MockMPITopology - Mock MPI topology for unit testing distributed logic
add_executable(v2_unit_mock_mpi_topology unit/mocks/Test__MockMPITopology.cpp)
target_include_directories(v2_unit_mock_mpi_topology PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_mpi_topology 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockMPITopology 
    COMMAND $<TARGET_FILE:v2_unit_mock_mpi_topology>
    LABELS "V2;Unit;Mock;MPI;Topology;Testing"
    MPI_PROCS 1
)

# Test: MockModelLoader - Mock model loader for unit testing without GGUF files
add_executable(v2_unit_mock_model_loader unit/mocks/Test__MockModelLoader.cpp)
target_include_directories(v2_unit_mock_model_loader PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_model_loader 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockModelLoader 
    COMMAND $<TARGET_FILE:v2_unit_mock_model_loader>
    LABELS "V2;Unit;Mock;ModelLoader;Testing"
    MPI_PROCS 1
)

# Test: MockWeightManager - Mock weight manager for unit testing without GGUF files
add_executable(v2_unit_mock_weight_manager unit/mocks/Test__MockWeightManager.cpp)
target_include_directories(v2_unit_mock_weight_manager PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_weight_manager 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockWeightManager 
    COMMAND $<TARGET_FILE:v2_unit_mock_weight_manager>
    LABELS "V2;Unit;Mock;WeightManager;Testing"
    MPI_PROCS 1
)

# Test: MockModelContext - Mock model context composing loader and weight manager
add_executable(v2_unit_mock_model_context unit/mocks/Test__MockModelContext.cpp)
target_include_directories(v2_unit_mock_model_context PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_model_context 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockModelContext 
    COMMAND $<TARGET_FILE:v2_unit_mock_model_context>
    LABELS "V2;Unit;Mock;ModelContext;Testing"
    MPI_PROCS 1
)

# Test: MockCollectiveContext - Mock collective context for unit testing without MPI/NCCL
add_executable(v2_unit_mock_collective_context unit/mocks/Test__MockCollectiveContext.cpp)
target_include_directories(v2_unit_mock_collective_context PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_collective_context 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockCollectiveContext 
    COMMAND $<TARGET_FILE:v2_unit_mock_collective_context>
    LABELS "V2;Unit;Mock;CollectiveContext;Testing"
    MPI_PROCS 1
)

# Test: MockWorkDistributor - Mock work distributor for unit testing work distribution
add_executable(v2_unit_mock_work_distributor unit/mocks/Test__MockWorkDistributor.cpp)
target_include_directories(v2_unit_mock_work_distributor PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_work_distributor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockWorkDistributor 
    COMMAND $<TARGET_FILE:v2_unit_mock_work_distributor>
    LABELS "V2;Unit;Mock;WorkDistributor;Testing"
    MPI_PROCS 1
)

# Test: MockWeightPlacementMap - Mock weight placement map for unit testing without real device placement
add_executable(v2_unit_mock_weight_placement_map unit/mocks/Test__MockWeightPlacementMap.cpp)
target_include_directories(v2_unit_mock_weight_placement_map PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_weight_placement_map 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockWeightPlacementMap 
    COMMAND $<TARGET_FILE:v2_unit_mock_weight_placement_map>
    LABELS "V2;Unit;Mock;WeightPlacementMap;Testing"
    MPI_PROCS 1
)

# Test: SwiGLU formula correctness (regression test for gate/up swap bug)
# This test verifies that SwiGLU computes silu(gate) * up, NOT gate * silu(up)
add_executable(v2_test_swiglu_formula unit/kernels/cpu/ops/Test__SwiGLUFormula.cpp)
target_link_libraries(v2_test_swiglu_formula 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SwiGLUFormula 
    COMMAND $<TARGET_FILE:v2_test_swiglu_formula>
    LABELS "V2;Unit;Kernels;SwiGLU;FFN;Activation;RegressionTest"
)

# Test: FP16Tensor class
add_executable(v2_test_fp16_tensor unit/tensors/Test__FP16Tensor.cpp)
target_link_libraries(v2_test_fp16_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FP16Tensor 
    COMMAND $<TARGET_FILE:v2_test_fp16_tensor>
    LABELS "V2;Unit;TensorOperations;FP16;Quantization;ViewSupport"
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
    LABELS "V2;Unit;TensorOperations;BF16;Quantization;GEMM;ViewSupport"
)

# Test: Q16_1 Block Structures (variable block sizes for integer attention)
add_executable(v2_test_q16_block_structures unit/tensors/Test__Q16BlockStructures.cpp)
target_link_libraries(v2_test_q16_block_structures 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16BlockStructures 
    COMMAND $<TARGET_FILE:v2_test_q16_block_structures>
    LABELS "V2;Unit;TensorOperations;Q16_1;BlockStructures;IntegerAttention"
    MPI_PROCS 1
)

# Test: KV Cache Block Size Alignment (regression tests for block size mismatch bug)
add_executable(v2_test_kv_cache_block_size_alignment unit/tensors/Test__KVCacheBlockSizeAlignment.cpp)
target_link_libraries(v2_test_kv_cache_block_size_alignment 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KVCacheBlockSizeAlignment 
    COMMAND $<TARGET_FILE:v2_test_kv_cache_block_size_alignment>
    LABELS "V2;Unit;TensorOperations;Q16_1;KVCache;BlockSize;Regression;IntegerAttention"
    MPI_PROCS 1
)

# Test: Q8_1Tensor class
add_executable(v2_test_q8_1_tensor unit/tensors/Test__Q8_1Tensor.cpp)
target_link_libraries(v2_test_q8_1_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1Tensor 
    COMMAND $<TARGET_FILE:v2_test_q8_1_tensor>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;Activations"
    MPI_PROCS 1  # Quantized activation tests are CPU-only
)

# Test: Q16_1Tensor class (high-precision quantized activations)
add_executable(v2_test_q16_1_tensor unit/tensors/Test__Q16_1Tensor.cpp)
target_link_libraries(v2_test_q16_1_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16_1Tensor 
    COMMAND $<TARGET_FILE:v2_test_q16_1_tensor>
    LABELS "V2;Unit;TensorOperations;Q16_1;Quantization;Residuals;HighPrecision"
    MPI_PROCS 1  # Quantized activation tests are CPU-only
)

# Test: TensorLayout enum and layout tracking (Phase 3 tensor contracts)
add_executable(v2_test_tensor_layout unit/tensors/Test__TensorLayout.cpp)
target_link_libraries(v2_test_tensor_layout
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorLayout
    COMMAND $<TARGET_FILE:v2_test_tensor_layout>
    LABELS "V2;Unit;TensorOperations;Layout;Contracts;Attention"
    MPI_PROCS 1
)

# Test: NativeVNNI intrinsic format metadata
# Backend-independent regression for codebook IDs consumed by CUDA/ROCm/CPU sweep generators.
add_executable(v2_test_native_vnni_format_info unit/tensors/Test__NativeVnniFormatInfo.cpp)
target_link_libraries(v2_test_native_vnni_format_info
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_NativeVnniFormatInfo
    COMMAND $<TARGET_FILE:v2_test_native_vnni_format_info>
    LABELS "V2;Unit;TensorOperations;NativeVNNI;Quantization;Codebook;Regression"
    MPI_PROCS 1
)

add_test(NAME V2_Unit_NativeVNNICodebookCsvValidator
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/validate_native_vnni_codebook_csv.py
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/fixtures/native_vnni_codebook_aliases.csv
)
set_tests_properties(V2_Unit_NativeVNNICodebookCsvValidator PROPERTIES
    LABELS "V2;Unit;NativeVNNI;Quantization;Codebook;CSV;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_MTPPerfStatsSummary
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_mtp_perfstats_summary.py
)
set_tests_properties(V2_Unit_MTPPerfStatsSummary PROPERTIES
    LABELS "V2;Unit;MTP;PerfStats;Benchmark;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_MTPIterationBenchmarkMatrix
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_mtp_iteration_benchmark_matrix.py
)
set_tests_properties(V2_Unit_MTPIterationBenchmarkMatrix PROPERTIES
    LABELS "V2;Unit;MTP;Benchmark;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_MTPDepthPolicyTrainer
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_mtp_depth_policy_trainer.py
)
set_tests_properties(V2_Unit_MTPDepthPolicyTrainer PROPERTIES
    LABELS "V2;Unit;MTP;Benchmark;Trainer;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_NativeVNNIDispatchRefreshScript
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit/scripts/test_native_vnni_dispatch_refresh.py
)
set_tests_properties(V2_Unit_NativeVNNIDispatchRefreshScript PROPERTIES
    LABELS "V2;Unit;NativeVNNI;CUDA;ROCm;Dispatch;Trainer;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_CUDAGemvDispatchGeneratorAliases
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/validate_cuda_gemv_dispatch_generator.py
            --generator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/infer_gemv_dispatch_heuristic.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/fixtures/cuda_gemv_dispatch_alias_smoke.csv
)
set_tests_properties(V2_Unit_CUDAGemvDispatchGeneratorAliases PROPERTIES
    LABELS "V2;Unit;CUDA;GEMV;NativeVNNI;Codebook;Dispatch;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_CUDAGemvDispatchBaseMerge
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/validate_cuda_gemv_dispatch_base_merge.py
            --generator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/analyze_cuda_tc_gemv_dispatch.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/fixtures/cuda_gemv_dispatch_alias_smoke.csv
            --base-include ${CMAKE_SOURCE_DIR}/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc
            --dispatch-validator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/validate_native_vnni_generated_dispatch_ids.py
)
set_tests_properties(V2_Unit_CUDAGemvDispatchBaseMerge PROPERTIES
    LABELS "V2;Unit;CUDA;GEMV;NativeVNNI;Codebook;Dispatch;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_CUDAPrefillDispatchGeneratorAliases
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/validate_cuda_prefill_dispatch_generator.py
            --generator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/analyze_cuda_tc_gemm_dispatch.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/cuda/gemm/fixtures/cuda_prefill_dispatch_alias_smoke.csv
)
set_tests_properties(V2_Unit_CUDAPrefillDispatchGeneratorAliases PROPERTIES
    LABELS "V2;Unit;CUDA;GEMM;Prefill;NativeVNNI;Codebook;Dispatch;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_NativeVNNIGeneratedDispatchCodebooks
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/validate_native_vnni_generated_dispatch_ids.py
            ${CMAKE_SOURCE_DIR}/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc
            ${CMAKE_SOURCE_DIR}/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchGenerated.inc
            ${CMAKE_SOURCE_DIR}/kernels/rocm/gemm/ROCmNativeVNNIDecodeDispatchGenerated.inc
            ${CMAKE_SOURCE_DIR}/kernels/rocm/gemm/ROCmNativeVNNIPrefillDispatchGenerated.inc
)
set_tests_properties(V2_Unit_NativeVNNIGeneratedDispatchCodebooks PROPERTIES
    LABELS "V2;Unit;NativeVNNI;Quantization;Codebook;Dispatch;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_ROCmNativeVNNITrainerCsvValidator
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/validate_rocm_native_vnni_trainer_csv.py
            --require-alias-group
            --require-policy-prefill-m
            --require-prefill-m 600
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/fixtures/rocm_native_vnni_decode_alias_smoke.csv
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/fixtures/rocm_native_vnni_prefill_alias_smoke.csv
)
set_tests_properties(V2_Unit_ROCmNativeVNNITrainerCsvValidator PROPERTIES
    LABELS "V2;Unit;ROCm;GEMM;GEMV;NativeVNNI;Codebook;Trainer;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_ROCmNativeVNNITrainerGenerator
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/validate_rocm_native_vnni_trainer_generator.py
            --generator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/analyze_rocm_native_vnni_trainer.py
            --dispatch-validator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/validate_native_vnni_generated_dispatch_ids.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/fixtures/rocm_native_vnni_prefill_alias_smoke.csv
)
set_tests_properties(V2_Unit_ROCmNativeVNNITrainerGenerator PROPERTIES
    LABELS "V2;Unit;ROCm;GEMM;NativeVNNI;Codebook;Trainer;Dispatch;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/validate_rocm_native_vnni_decode_trainer_generator.py
            --generator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py
            --dispatch-validator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/validate_native_vnni_generated_dispatch_ids.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/fixtures/rocm_native_vnni_decode_alias_smoke.csv
)
set_tests_properties(V2_Unit_ROCmNativeVNNIDecodeTrainerGenerator PROPERTIES
    LABELS "V2;Unit;ROCm;GEMV;NativeVNNI;Codebook;Trainer;Dispatch;Regression"
    TIMEOUT 30
)

add_test(NAME V2_Unit_ROCmNativeVNNIBatchedDecodeTrainerGenerator
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/validate_rocm_native_vnni_batched_decode_trainer_generator.py
            --generator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/analyze_rocm_native_vnni_batched_decode_trainer.py
            --dispatch-validator ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/validate_native_vnni_generated_dispatch_ids.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/performance/kernels/rocm/fixtures/rocm_native_vnni_batched_decode_smoke.csv
)
set_tests_properties(V2_Unit_ROCmNativeVNNIBatchedDecodeTrainerGenerator PROPERTIES
    LABELS "V2;Unit;ROCm;GEMV;NativeVNNI;Codebook;Trainer;Dispatch;Regression"
    TIMEOUT 30
)

# Performance: Q16 Fixed-Scale SIMD Quantization
add_executable(v2_perf_q16_fixed_scale_simd performance/tensors/Perf__Q16_FixedScaleSIMD.cpp)
target_link_libraries(v2_perf_q16_fixed_scale_simd
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_Q16_FixedScaleSIMD
    COMMAND $<TARGET_FILE:v2_perf_q16_fixed_scale_simd>
    LABELS "V2;Performance;Quantization;Q16_1;SIMD;AVX2;AVX512;VNNI"
    MPI_PROCS 1
)

# Test: Q16 Decode (Dequant) SIMD - Block to FP32 conversion
# Tests SIMD implementations (Scalar, AVX2, AVX512) of Q16 block dequantization
add_executable(v2_test_q16_decode_simd unit/tensors/Test__Q16DecodeSIMD.cpp)
target_link_libraries(v2_test_q16_decode_simd
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16DecodeSIMD
    COMMAND $<TARGET_FILE:v2_test_q16_decode_simd>
    LABELS "V2;Unit;Quantization;Q16_1;SIMD;AVX2;AVX512;Dequantization;Decode"
    MPI_PROCS 1
)

# Performance: Q16 Decode (Dequant) SIMD - Block to FP32 conversion
add_executable(v2_perf_q16_decode_simd performance/tensors/Perf__Q16_DecodeSIMD.cpp)
target_link_libraries(v2_perf_q16_decode_simd
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_Q16_DecodeSIMD
    COMMAND $<TARGET_FILE:v2_perf_q16_decode_simd>
    LABELS "V2;Performance;Quantization;Q16_1;SIMD;AVX2;AVX512;Dequantization"
    MPI_PROCS 1
)

# Test: Q8_1 Block-Aligned Slicing for tensor-parallel GEMM
# Phase 6.4: Validates slice_k_blocks() for k-dimension splitting at 32-element boundaries
add_executable(v2_test_q8_1_block_aligned_slicing unit/tensors/Test__Q8_1_BlockAlignedSlicing.cpp)
target_link_libraries(v2_test_q8_1_block_aligned_slicing 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_BlockAlignedSlicing 
    COMMAND $<TARGET_FILE:v2_test_q8_1_block_aligned_slicing>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;TensorParallel;KSlicing"
    MPI_PROCS 1  # Block slicing is local, no MPI needed
)

# Test: TensorFactory device_index assignment (regression test for CLI Q8_1 bug)
# Verifies that Q8_1, BF16, FP16 activation tensors have correct device_index
# after creation via createActivation(). Bug: tensors were created with device=-1
# even when device_idx=0 was passed, causing spurious device transfer attempts.
add_executable(v2_test_tensor_factory_device_idx unit/tensors/Test__TensorFactory_DeviceIndex.cpp)
target_link_libraries(v2_test_tensor_factory_device_idx 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorFactory_DeviceIndex 
    COMMAND $<TARGET_FILE:v2_test_tensor_factory_device_idx>
    LABELS "V2;Unit;TensorOperations;TensorFactory;DeviceManagement;RegressionTest"
    MPI_PROCS 1  # Single-rank test, no MPI needed
)

# Test: TensorFactory Q16_1 variable block size creation (Phase 3)
# Verifies that createQ16_1(shape, block_size, device_idx) creates tensors
# with correct block sizes (BLOCK_64, BLOCK_128) for different head_dims.
add_executable(v2_test_tensor_factory_q16_block_size unit/tensors/Test__TensorFactory_Q16BlockSize.cpp)
target_link_libraries(v2_test_tensor_factory_q16_block_size 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorFactory_Q16BlockSize 
    COMMAND $<TARGET_FILE:v2_test_tensor_factory_q16_block_size>
    LABELS "V2;Unit;TensorOperations;TensorFactory;Q16;VariableBlockSize"
    MPI_PROCS 1  # Single-rank test, no MPI needed
)

# Test: ITensor Runtime Interface
# Tests the ITensor interface including native_type_id(), is<T>(), typed_as<T>(),
# try_as<T>(), and static_type_id() for type-safe runtime polymorphism.
add_executable(v2_test_itensor unit/tensors/Test__ITensor.cpp)
target_link_libraries(v2_test_itensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ITensor 
    COMMAND $<TARGET_FILE:v2_test_itensor>
    LABELS "V2;Unit;TensorOperations;ITensor;TypeSafety"
    MPI_PROCS 1  # Single-rank test, no MPI needed
)

# Test: AttentionMode enum and detection function
# Tests the AttentionMode dispatch (PREFILL, DECODE, BATCHED_DECODE, CHUNKED_PREFILL)
add_executable(v2_test_attention_mode unit/kernels/cpu/attention/Test__AttentionMode.cpp)
target_link_libraries(v2_test_attention_mode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionMode 
    COMMAND $<TARGET_FILE:v2_test_attention_mode>
    LABELS "V2;Unit;Kernels;Attention;ModeDetection"
    MPI_PROCS 1
)

# Test: Q8_1 Add Reproduction (Divergence Investigation)
add_executable(v2_test_q8_1_add_repro unit/tensors/Test__Q8_1_Add_Repro.cpp)
target_link_libraries(v2_test_q8_1_add_repro 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_Add_Repro 
    COMMAND $<TARGET_FILE:v2_test_q8_1_add_repro>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;Residuals;Reproduction"
    MPI_PROCS 1
)

# Test: Q8_1 Add Extreme Tests (Comprehensive Edge Case Coverage)
add_executable(v2_test_q8_1_add_extreme unit/tensors/Test__Q8_1_Add_Extreme.cpp)
target_link_libraries(v2_test_q8_1_add_extreme 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_Add_Extreme 
    COMMAND $<TARGET_FILE:v2_test_q8_1_add_extreme>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;Residuals;EdgeCases;CatastrophicCancellation"
    MPI_PROCS 1
)

# Test: Q8_1 Add Debug (Layer 21 pattern investigation)
add_executable(v2_test_q8_1_add_debug unit/tensors/Test__Q8_1_Add_Debug.cpp)
target_link_libraries(v2_test_q8_1_add_debug 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_Add_Debug 
    COMMAND $<TARGET_FILE:v2_test_q8_1_add_debug>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;Debug"
    MPI_PROCS 1
)

# Test: Q8_1 Add Comprehensive (exhaustive test suite for q8_1_add_q8_1)
add_executable(v2_test_q8_1_add_comprehensive unit/tensors/Test__Q8_1_Add_Comprehensive.cpp)
target_link_libraries(v2_test_q8_1_add_comprehensive 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_Add_Comprehensive 
    COMMAND $<TARGET_FILE:v2_test_q8_1_add_comprehensive>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;Comprehensive;CatastrophicCancellation"
    MPI_PROCS 1
)

# Test: Q8_1 Residual Add Layer 21 (using actual layer 21 dump data patterns)
add_executable(v2_test_q8_1_residual_layer21 unit/tensors/Test__Q8_1_ResidualAdd_Layer21.cpp)
target_link_libraries(v2_test_q8_1_residual_layer21 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_ResidualAdd_Layer21 
    COMMAND $<TARGET_FILE:v2_test_q8_1_residual_layer21>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;Residuals;Layer21;RealData"
    MPI_PROCS 1
)

# Test: INT8Tensor class (runtime activation quantization with per-row scales)
add_executable(v2_test_int8_tensor unit/tensors/Test__INT8Tensor.cpp)
target_link_libraries(v2_test_int8_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_INT8Tensor 
    COMMAND $<TARGET_FILE:v2_test_int8_tensor>
    LABELS "V2;Unit;TensorOperations;INT8;Quantization;PerRowScales;Dequantization;BatchedExecution"
    MPI_PROCS 1  # INT8 activation tests are CPU-only, single rank
)

add_executable(v2_test_int32_tensor unit/tensors/Test__INT32Tensor.cpp)
target_link_libraries(v2_test_int32_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_INT32Tensor 
    COMMAND $<TARGET_FILE:v2_test_int32_tensor>
    LABELS "V2;Unit;TensorOperations;INT32;Quantization;Requantization;INT8Pipeline"
    MPI_PROCS 1  # INT32Tensor operations are CPU-only, single rank
)

# NOTE: INT32 RMSNorm test (Test__INT32RMSNorm.cpp) removed - was for orphaned INT8 pipeline
# INT32→INT8 RMSNorm is no longer used; pipeline uses CPURMSNormKernelT<Q8_1> instead

# Test: Q8_1 Pure-Integer RMSNorm (NO floating-point operations)
add_executable(v2_test_q8_1_pure_integer_rmsnorm unit/kernels/cpu/ops/Test__Q8_1PureIntegerRMSNorm.cpp)
target_link_libraries(v2_test_q8_1_pure_integer_rmsnorm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1PureIntegerRMSNorm 
    COMMAND $<TARGET_FILE:v2_test_q8_1_pure_integer_rmsnorm>
    LABELS "V2;Unit;Kernels;RMSNorm;Q8_1;PureInteger;TypedResiduals;CPU"
    MPI_PROCS 1
)

# Test: Q16_1 RMSNorm (Q16_1 input → FP32 output)
add_executable(v2_test_q16_1_rmsnorm unit/kernels/cpu/ops/Test__Q16_1RMSNorm.cpp)
target_link_libraries(v2_test_q16_1_rmsnorm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16_1RMSNorm 
    COMMAND $<TARGET_FILE:v2_test_q16_1_rmsnorm>
    LABELS "V2;Unit;Kernels;RMSNorm;Q16_1;TypedResiduals;CPU"
    MPI_PROCS 1
)

# Test: Q4_1 SIMD Unpack
add_executable(v2_test_q4_1_simd_unpack unit/tensors/Test__Q4_1_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q4_1_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_1_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q4_1_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Quantization;Q4_1;SIMD"
    MPI_PROCS 1
)

# Test: Q4_1 Transcode (Fused Dequant/Requant)
add_executable(v2_test_q4_1_transcode unit/tensors/Test__Q4_1_Transcode.cpp)
target_link_libraries(v2_test_q4_1_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_1_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q4_1_transcode>
    LABELS "V2;Unit;TensorOperations;Quantization;Q4_1;SIMD;Transcode"
    MPI_PROCS 1
)

# Test: IQ4_NL Transcode (Unpack to INT8)
add_executable(v2_test_iq4_nl_transcode unit/tensors/Test__IQ4_NL_Transcode.cpp)
target_link_libraries(v2_test_iq4_nl_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_NL_Transcode 
    COMMAND $<TARGET_FILE:v2_test_iq4_nl_transcode>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ4_NL;SIMD;Transcode"
    MPI_PROCS 1
)

# Test: Q2_K SIMD Unpack
add_executable(v2_test_q2_k_simd_unpack unit/tensors/Test__Q2_K_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q2_k_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q2_K_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q2_k_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Quantization;Q2_K;SIMD"
    MPI_PROCS 1
)

# Test: Q3_K SIMD Unpack
add_executable(v2_test_q3_k_simd_unpack unit/tensors/Test__Q3_K_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q3_k_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q3_K_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q3_k_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Quantization;Q3_K;SIMD"
    MPI_PROCS 1
)

# Test: Q3_K Bug Repro
add_executable(v2_test_q3k_bug unit/tensors/Test__Q3K_Bug.cpp)
target_link_libraries(v2_test_q3k_bug 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q3KBug 
    COMMAND $<TARGET_FILE:v2_test_q3k_bug>
    LABELS "V2;Unit;Q3K;Bug"
    MPI_PROCS 1
)

# Test: Q2_K Transcode (Fused Dequant/Requant)
add_executable(v2_test_q2_k_transcode unit/tensors/Test__Q2_K_Transcode.cpp)
target_link_libraries(v2_test_q2_k_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q2_K_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q2_k_transcode>
    LABELS "V2;Unit;TensorOperations;Quantization;Q2_K;SIMD;Transcode"
    MPI_PROCS 1
)

# Test: Q3_K Transcode (Fused Dequant/Requant)
add_executable(v2_test_q3_k_transcode unit/tensors/Test__Q3_K_Transcode.cpp)
target_link_libraries(v2_test_q3_k_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q3_K_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q3_k_transcode>
    LABELS "V2;Unit;TensorOperations;Quantization;Q3_K;SIMD;Transcode"
    MPI_PROCS 1
)

# Test: Q4_K Transcode (Fused Dequant/Requant)
add_executable(v2_test_q4_k_transcode unit/tensors/Test__Q4_K_Transcode.cpp)
target_link_libraries(v2_test_q4_k_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_K_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q4_k_transcode>
    LABELS "V2;Unit;TensorOperations;Quantization;Q4_K;SIMD;Transcode"
    MPI_PROCS 1
)

# Test: Q6_K Transcode (Fused Dequant/Requant)
add_executable(v2_test_q6_k_transcode unit/tensors/Test__Q6_K_Transcode.cpp)
target_link_libraries(v2_test_q6_k_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q6_K_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q6_k_transcode>
    LABELS "V2;Unit;TensorOperations;Quantization;Q6_K;SIMD;Transcode"
    MPI_PROCS 1
)

# Test: Q4_1 Integration (GEMM)
add_executable(v2_test_q4_1_integration integration/tensors/Test__Q4_1Tensor.cpp)
target_link_libraries(v2_test_q4_1_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q4_1Tensor 
    COMMAND $<TARGET_FILE:v2_test_q4_1_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;Q4_1;GEMM"
)

# Test: Q2_K Integration (GEMM)
add_executable(v2_test_q2_k_integration integration/tensors/Test__Q2_KTensor.cpp)
target_link_libraries(v2_test_q2_k_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q2_KTensor 
    COMMAND $<TARGET_FILE:v2_test_q2_k_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;Q2_K;GEMM"
)

# Test: Q3_K Integration (GEMM)
add_executable(v2_test_q3_k_integration integration/tensors/Test__Q3_KTensor.cpp)
target_link_libraries(v2_test_q3_k_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q3_KTensor 
    COMMAND $<TARGET_FILE:v2_test_q3_k_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;Q3_K;GEMM"
)

# Test: Q4_K Integration (GEMM)
add_executable(v2_test_q4_k_integration integration/tensors/Test__Q4_KTensor.cpp)
target_link_libraries(v2_test_q4_k_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q4_KTensor 
    COMMAND $<TARGET_FILE:v2_test_q4_k_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;Q4_K;GEMM"
)

# Test: Q6_K Integration (GEMM)
add_executable(v2_test_q6_k_integration integration/tensors/Test__Q6_KTensor.cpp)
target_link_libraries(v2_test_q6_k_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q6_KTensor 
    COMMAND $<TARGET_FILE:v2_test_q6_k_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;Q6_K;GEMM"
)

# Test: IQ4_NL Integration (GEMM)
add_executable(v2_test_iq4_nl_integration integration/tensors/Test__IQ4_NLTensor.cpp)
target_link_libraries(v2_test_iq4_nl_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_IQ4_NLTensor 
    COMMAND $<TARGET_FILE:v2_test_iq4_nl_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;IQ4_NL;GEMM"
)

# Test: TensorBase Authoritative Device tracking - Multi-device coherence API (requires GPU)
add_executable(v2_integration_tensor_authoritative_device integration/tensors/Test__TensorBase_AuthoritativeDevice.cpp)
target_link_libraries(v2_integration_tensor_authoritative_device 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_TensorBase_AuthoritativeDevice 
    COMMAND $<TARGET_FILE:v2_integration_tensor_authoritative_device>
    LABELS "V2;Integration;Tensors;Coherence;AuthoritativeDevice;GPU"
    MPI_PROCS 1
)

# Test: TensorBase TransferTo - Direct GPU-to-GPU transfer (requires GPU)
add_executable(v2_integration_tensor_transfer_to integration/tensors/Test__TensorBase_TransferTo.cpp)
target_link_libraries(v2_integration_tensor_transfer_to 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_TensorBase_TransferTo 
    COMMAND $<TARGET_FILE:v2_integration_tensor_transfer_to>
    LABELS "V2;Integration;Tensors;Coherence;TransferTo;GPU"
    MPI_PROCS 1
)

# Test: TransferEngine copyActivation - tensor→tensor copy transport selection (requires GPU)
add_executable(v2_integration_transfer_engine_copy_activation integration/transfer/Test__TransferEngine_CopyActivation.cpp)
target_link_libraries(v2_integration_transfer_engine_copy_activation 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_TransferEngine_CopyActivation 
    COMMAND $<TARGET_FILE:v2_integration_transfer_engine_copy_activation>
    LABELS "V2;Integration;Transfer;TransferEngine;Coherence;CopyActivation;GPU"
    MPI_PROCS 1
)

# Test: FusedQKVGEMMStage compute stage (regression test for QKV bias bug)
add_executable(v2_test_fused_qkv_gemm_stage unit/execution/compute_stages/stages/Test__FusedQKVGEMMStage.cpp)
target_link_libraries(v2_test_fused_qkv_gemm_stage
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedQKVGEMMStage
    COMMAND $<TARGET_FILE:v2_test_fused_qkv_gemm_stage>
    LABELS "V2;Unit;ComputeStage;FusedKernels;GEMM;Bias;Attention;QKV;RegressionTest;CPU"
    MPI_PROCS 1  # CPU compute stage tests, single rank
)

# Test: FusedGateUpGEMMStage compute stage (FFN gate/up bias handling)
add_executable(v2_test_fused_gate_up_gemm_stage unit/execution/compute_stages/stages/Test__FusedGateUpGEMMStage.cpp)
target_link_libraries(v2_test_fused_gate_up_gemm_stage
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedGateUpGEMMStage
    COMMAND $<TARGET_FILE:v2_test_fused_gate_up_gemm_stage>
    LABELS "V2;Unit;ComputeStage;FusedKernels;GEMM;Bias;FFN;RegressionTest;CPU"
    MPI_PROCS 1  # CPU compute stage tests, single rank
)

# Test: Device index handling in CPU kernels (regression test for device_idx=0 bug)
add_executable(v2_test_device_index_cpu_kernels unit/kernels/cpu/Test__DeviceIndexCPUKernels.cpp)
target_link_libraries(v2_test_device_index_cpu_kernels
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceIndexCPUKernels
    COMMAND $<TARGET_FILE:v2_test_device_index_cpu_kernels>
    LABELS "V2;Unit;DeviceManagement;Kernels;RegressionTest;CPU"
    MPI_PROCS 1  # CPU kernel tests, single rank
)

# Test: KernelFactory centralized kernel dispatch
add_executable(v2_test_kernel_factory unit/kernels/Test__KernelFactory.cpp)
target_link_libraries(v2_test_kernel_factory
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelFactory
    COMMAND $<TARGET_FILE:v2_test_kernel_factory>
    LABELS "V2;Unit;Kernels;DeviceManagement;KernelFactory;GEMM;Quantization"
    MPI_PROCS 1  # Kernel factory tests, single rank
)

# Test: CPU Quantized GEMM/GEMV Parity — all formats × thread counts 1-56
# Covers GEMV (M=1), GEMM (M=9 odd, M=64), standard N=512, and small-N
# (N=16, N=32) for ldc<64 overflow regression. Checks relative L2 error,
# cosine similarity, NaN/Inf, and all-zero-row detection.
add_executable(v2_test_cpu_quantized_gemm_parity unit/kernels/Test__CPUQuantizedGemmParity.cpp)
target_link_libraries(v2_test_cpu_quantized_gemm_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUQuantizedGemmParity
    COMMAND $<TARGET_FILE:v2_test_cpu_quantized_gemm_parity>
    LABELS "V2;Unit;Kernels;GEMM;GEMV;Quantization;Parity;DeferredPacking;VNNI;LowThreadCount;Regression"
    MPI_PROCS 1
)

# Test: AVX2 vs AVX512-VNNI parity (emulated dpbusd, chunk GEMV, full GEMV/GEMM)
add_executable(v2_test_avx2_vnni_parity unit/kernels/Test__AVX2VNNIParity.cpp)
target_link_libraries(v2_test_avx2_vnni_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AVX2VNNIParity
    COMMAND $<TARGET_FILE:v2_test_avx2_vnni_parity>
    LABELS "V2;Unit;Kernels;GEMM;GEMV;AVX2;Parity;VNNI"
    MPI_PROCS 1
)

# Test: AVX2 Fallback Parity (all non-GEMM AVX2 fallback functions)
add_executable(v2_test_avx2_fallback_parity unit/kernels/Test__AVX2FallbackParity.cpp)
target_link_libraries(v2_test_avx2_fallback_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AVX2FallbackParity
    COMMAND $<TARGET_FILE:v2_test_avx2_fallback_parity>
    LABELS "V2;Unit;Kernels;AVX2;Parity;GDN;TurboQuant;FWHT;Attention"
    MPI_PROCS 1
)

# Test: SIMDHelpers ISA Parity (scalar vs AVX2 vs AVX512 for residual add, quantize, etc.)
add_executable(v2_test_simd_helpers_isa_parity unit/Test__SIMDHelpersISAParity.cpp)
target_link_libraries(v2_test_simd_helpers_isa_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SIMDHelpersISAParity
    COMMAND $<TARGET_FILE:v2_test_simd_helpers_isa_parity>
    LABELS "V2;Unit;Kernels;AVX2;AVX512;Parity;SIMD;Quantization"
    MPI_PROCS 1
)

# Test: Fused Projection Parity (multiply_fused_tensor vs multiply_tensor)
# Regression test for Q4_0 M=1 decode bug where fused path produced wrong output
add_executable(v2_test_fused_projection_parity unit/kernels/Test__FusedProjectionParity.cpp)
target_link_libraries(v2_test_fused_projection_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedProjectionParity
    COMMAND $<TARGET_FILE:v2_test_fused_projection_parity>
    LABELS "V2;Unit;Kernels;GEMM;GEMV;FusedKernels;Quantization;Parity;RegressionTest"
    MPI_PROCS 1
)

# Performance: VNNI repack micro-benchmark (deferred packing hot path)
add_executable(v2_perf_vnni_repack performance/kernels/Perf__VNNIRepack.cpp)
target_link_libraries(v2_perf_vnni_repack
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_VNNIRepack
    COMMAND $<TARGET_FILE:v2_perf_vnni_repack>
    LABELS "V2;Performance;GEMM;VNNI;DeferredPacking;Repack;Benchmark"
    MPI_PROCS 1
)

# Test: PreparedWeightStore sliced GEMM for tensor parallelism (Phase 2.1)
add_executable(v2_test_prepared_weight_store_sliced unit/loaders/Test__PreparedWeightStoreSliced.cpp)
target_link_libraries(v2_test_prepared_weight_store_sliced
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PreparedWeightStoreSliced
    COMMAND $<TARGET_FILE:v2_test_prepared_weight_store_sliced>
    LABELS "V2;Unit;Loaders;PreparedWeightStore;TensorParallelism;SlicedGEMM;Phase2"
    MPI_PROCS 1  # Sliced kernel tests, single rank
)

# Test: Kernel dynamic state lifecycle (session boundary resets)
add_executable(v2_test_kernel_dynamic_state_lifecycle unit/kernels/Test__KernelDynamicStateLifecycle.cpp)
target_link_libraries(v2_test_kernel_dynamic_state_lifecycle
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelDynamicStateLifecycle
    COMMAND $<TARGET_FILE:v2_test_kernel_dynamic_state_lifecycle>
    LABELS "V2;Unit;Kernels;KernelFactory;DynamicState;SessionLifecycle"
    MPI_PROCS 1
)

# Test: KernelFactory TensorSlice unwrapping in CUDA dispatch path
# Regression test for: "unsupported tensor type 10 for device CUDA:0"
# Root cause: dynamic_cast<Q4_0Tensor*>(TensorSlice*) returns nullptr
# Solution: Unwrap TensorSlice via slice->inner() before type dispatch
add_executable(v2_test_kernel_factory_tensorslice_cuda unit/kernels/Test__KernelFactoryTensorSliceCUDA.cpp)
target_link_libraries(v2_test_kernel_factory_tensorslice_cuda
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_KernelFactoryTensorSliceCUDA
    COMMAND $<TARGET_FILE:v2_test_kernel_factory_tensorslice_cuda>
    LABELS "V2;Integration;Kernels;KernelFactory;TensorSlice;CUDA;TensorParallelism;Regression"
    MPI_PROCS 1  # TensorSlice tests, single rank
)

# Test: TensorSlice device coherence delegation
# Verifies TensorSlice properly delegates coherence methods to inner tensor
# Critical for MPI tensor parallelism where weights are wrapped in TensorSlice
# Root cause: TensorSlice inherits from CPUTensorBase but doesn't override coherence methods
# Solution: StageCoherence.cpp unwraps TensorSlice to access inner tensor
add_executable(v2_test_tensorslice_coherence unit/tensors/Test__TensorSliceCoherence.cpp)
target_link_libraries(v2_test_tensorslice_coherence
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorSliceCoherence
    COMMAND $<TARGET_FILE:v2_test_tensorslice_coherence>
    LABELS "V2;Unit;Tensors;TensorSlice;Coherence;DeviceManagement;TensorParallelism;Regression"
    MPI_PROCS 1  # Coherence tests, single rank
)

# Test: NUMATopology NUMA detection (local node, GPU affinity) - INTEGRATION (requires real GPUs)
add_executable(v2_integration_numa_topology integration/utils/Test__NUMATopology.cpp)
target_link_libraries(v2_integration_numa_topology
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_NUMATopology
    COMMAND $<TARGET_FILE:v2_integration_numa_topology>
    LABELS "V2;Integration;NUMA;Topology;DeviceManagement;GPU"
    MPI_PROCS 1  # NUMA topology tests, single rank
)

# Test: KernelFactory ops kernel creation (RMSNorm, SwiGLU, RoPE, Attention, Embedding, ResidualAdd)
add_executable(v2_test_kernel_factory_ops unit/kernels/Test__KernelFactoryOps.cpp)
target_link_libraries(v2_test_kernel_factory_ops
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelFactoryOps
    COMMAND $<TARGET_FILE:v2_test_kernel_factory_ops>
    LABELS "V2;Unit;Kernels;KernelFactory;Ops;RMSNorm;SwiGLU;RoPE;Attention;Embedding;ResidualAdd"
    MPI_PROCS 1  # Ops kernel factory tests, single rank
)

# Test: CUDAKernelBase and ROCmKernelBase device context support (Phase 4)
# Tests setDeviceContext/hasDeviceContext/getStream/getBlasHandle with mock contexts
# No actual GPU hardware required - uses MockGPUContext for unit testing
add_executable(v2_test_kernel_base_device_context unit/kernels/Test__KernelBaseDeviceContext.cpp)
target_link_libraries(v2_test_kernel_base_device_context
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelBaseDeviceContext
    COMMAND $<TARGET_FILE:v2_test_kernel_base_device_context>
    LABELS "V2;Unit;Kernels;DeviceContext;CUDAKernelBase;ROCmKernelBase;Phase4;GPUDeviceContextRefactor"
    MPI_PROCS 1  # Mock context tests, single rank, no GPU needed
)

# Test: RMSNorm precision variants (FP32/BF16/FP16/INT32 scalar/AVX2/AVX512 parity)
add_executable(v2_test_rmsnorm_precision unit/kernels/cpu/ops/Test__RMSNormPrecision.cpp)
target_link_libraries(v2_test_rmsnorm_precision 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RMSNormPrecision 
    COMMAND $<TARGET_FILE:v2_test_rmsnorm_precision>
    LABELS "V2;Unit;Kernels;RMSNorm;FP32;BF16;FP16;INT32;SIMD;AVX2;AVX512;CPU;PrecisionTesting"
    MPI_PROCS 1  # RMSNorm primitives are CPU-only, single rank
)

# Test: INT32→INT8 RMSNorm with dynamic requantization (full INT8 pipeline)
add_executable(v2_test_rmsnorm_int32 unit/kernels/cpu/ops/Test__RMSNormINT32.cpp)
target_link_libraries(v2_test_rmsnorm_int32
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RMSNormINT32
    COMMAND $<TARGET_FILE:v2_test_rmsnorm_int32>
    LABELS "V2;Unit;Kernels;RMSNorm;INT32;INT8;Requantization;DynamicScaling;AVX512;AVX2;CPU;QuantizationAccuracy"
    MPI_PROCS 1  # RMSNorm primitives are CPU-only, single rank
)

# NOTE: RMSNorm interface test (Test__RMSNormInterface.cpp) removed - redundant with Test__CPURMSNormKernelT.cpp
# The typed kernel test suite provides comprehensive coverage of all precision variants

# Test: Softmax primitives (scalar/AVX2/AVX512 parity, cross-precision accuracy)
add_executable(v2_test_softmax_primitives unit/kernels/cpu/primitives/Test__SoftmaxPrimitives.cpp)
target_link_libraries(v2_test_softmax_primitives
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SoftmaxPrimitives
    COMMAND $<TARGET_FILE:v2_test_softmax_primitives>
    LABELS "V2;Unit;Primitives;Softmax;FP32;BF16;FP16;SIMD;AVX2;AVX512;CPU;PrecisionTesting;NumericalStability"
    MPI_PROCS 1  # Softmax primitives are CPU-only, single rank
)

# Test: Softmax -Inf handling (AVX512 specific regression test)
add_executable(v2_test_softmax_inf unit/kernels/cpu/primitives/Test__SoftmaxInf.cpp)
target_link_libraries(v2_test_softmax_inf
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SoftmaxInf
    COMMAND $<TARGET_FILE:v2_test_softmax_inf>
    LABELS "V2;Unit;Primitives;Softmax;AVX512;Regression;InfHandling"
    MPI_PROCS 1
)

# Test: Q8_1 Integer-Aware Softmax primitives (native Q8_1 softmax vs FP32 reference)
add_executable(v2_test_q8_1_softmax_primitives unit/kernels/cpu/primitives/Test__Q8_1SoftmaxPrimitives.cpp)
target_link_libraries(v2_test_q8_1_softmax_primitives
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1SoftmaxPrimitives
    COMMAND $<TARGET_FILE:v2_test_q8_1_softmax_primitives>
    LABELS "V2;Unit;Primitives;Softmax;Q8_1;Quantization;SIMD;AVX2;AVX512;CPU;PrecisionTesting;IntegerDomain"
    MPI_PROCS 1
)

# Test: IndexSoftmax integer-domain softmax (IntAttention paper arXiv:2511.21513)
add_executable(v2_test_index_softmax unit/kernels/cpu/primitives/Test__IndexSoftmax.cpp)
target_link_libraries(v2_test_index_softmax
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IndexSoftmax
    COMMAND $<TARGET_FILE:v2_test_index_softmax>
    LABELS "V2;Unit;Primitives;Softmax;IntegerDomain;IndexSoftmax;LUT;Quantization"
    MPI_PROCS 1
)

# Test: ActivationTraits (trait dispatch to softmax primitives, GEMM, workspace allocation)
add_executable(v2_test_activation_traits unit/kernels/cpu/primitives/Test__ActivationTraits.cpp)
target_link_libraries(v2_test_activation_traits
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ActivationTraits
    COMMAND $<TARGET_FILE:v2_test_activation_traits>
    LABELS "V2;Unit;Primitives;ActivationTraits;FP32;BF16;FP16;INT32;TraitDispatch;TemplateSystem;CPU"
    MPI_PROCS 1  # ActivationTraits tests are CPU-only, single rank
)

# Test: CPURMSNormKernelT (typed RMSNorm with fused precision conversion)
add_executable(v2_test_rmsnorm_typed_kernel unit/kernels/cpu/ops/Test__CPURMSNormKernelT.cpp)
target_link_libraries(v2_test_rmsnorm_typed_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURMSNormKernelT
    COMMAND $<TARGET_FILE:v2_test_rmsnorm_typed_kernel>
    LABELS "V2;Unit;Kernels;RMSNorm;FP32;BF16;FP16;Q8_1;FusedPrecision;TypedResidual;CPU"
    MPI_PROCS 1  # CPURMSNormKernelT tests are CPU-only, single rank
)

# Test: CPURoPEKernelT (typed RoPE with fused precision conversion)
add_executable(v2_test_rope_typed_kernel unit/kernels/cpu/ops/Test__CPURoPEKernelT.cpp)
target_link_libraries(v2_test_rope_typed_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURoPEKernelT
    COMMAND $<TARGET_FILE:v2_test_rope_typed_kernel>
    LABELS "V2;Unit;Kernels;RoPE;FP32;BF16;FP16;Q8_1;FusedPrecision;TypedResidual;CPU"
    MPI_PROCS 1  # CPURoPEKernelT tests are CPU-only, single rank
)

# Test: CPURoPEKernelT<Q16_1> with variable block sizes (32, 64, 128, 192)
add_executable(v2_test_rope_q16_block_sizes unit/kernels/cpu/ops/Test__CPURoPEKernelT_Q16_1_BlockSizes.cpp)
target_link_libraries(v2_test_rope_q16_block_sizes
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURoPEKernelT_Q16_1_BlockSizes
    COMMAND $<TARGET_FILE:v2_test_rope_q16_block_sizes>
    LABELS "V2;Unit;Kernels;RoPE;Q16_1;BlockSizes;TypedResidual;CPU"
    MPI_PROCS 1
)

# Test: Q16HeadNormalization primitives (per-head scale normalization)
add_executable(v2_test_q16_head_normalization unit/kernels/cpu/primitives/Test__Q16HeadNormalization.cpp)
target_link_libraries(v2_test_q16_head_normalization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16HeadNormalization
    COMMAND $<TARGET_FILE:v2_test_q16_head_normalization>
    LABELS "V2;Unit;Primitives;Q16_1;Normalization;IntegerAttention"
    MPI_PROCS 1
)

# Test: Q16 VNNI-Safe Normalization (rescaling for safe VNNI dot products)
add_executable(v2_test_q16_vnni_safe_normalization unit/kernels/cpu/ops/Test__Q16_VNNI_Safe_Normalization.cpp)
target_link_libraries(v2_test_q16_vnni_safe_normalization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16VNNISafeNormalization
    COMMAND $<TARGET_FILE:v2_test_q16_vnni_safe_normalization>
    LABELS "V2;Unit;Primitives;Q16_1;VNNI;Normalization;IntegerAttention"
    MPI_PROCS 1
)

# Test: CPUFlashAttentionKernelT (Flash attention accuracy: FP32, VNNI i16/i12, causal, GQA, window)
add_executable(v2_test_cpu_flash_attention_kernel_t unit/kernels/cpu/attention/Test__CPUFlashAttentionKernelT.cpp)
target_link_libraries(v2_test_cpu_flash_attention_kernel_t
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUFlashAttentionKernelT
    COMMAND $<TARGET_FILE:v2_test_cpu_flash_attention_kernel_t>
    LABELS "V2;Unit;Kernels;Attention;FlashAttention;FP32;VNNI;CPU;Accuracy;GQA;TensorParallel"
    MPI_PROCS 1
)

# Test: CPU attention padding parity for Phase 6 prefill bucket safety
add_executable(v2_test_attention_padding_parity unit/kernels/cpu/attention/Test__AttentionPaddingParity.cpp)
target_link_libraries(v2_test_attention_padding_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionPaddingParity
    COMMAND $<TARGET_FILE:v2_test_attention_padding_parity>
    LABELS "V2;Unit;Kernels;Attention;Padding;CausalMask;CPU;Phase6"
    MPI_PROCS 1
)

# Test: Q16 decode HEAD_MAJOR layout regression (KV cache addressing)
add_executable(v2_test_q16_decode_head_major unit/kernels/cpu/attention/Test__Q16DecodeHeadMajor.cpp)
target_link_libraries(v2_test_q16_decode_head_major
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16DecodeHeadMajor
    COMMAND $<TARGET_FILE:v2_test_q16_decode_head_major>
    LABELS "V2;Unit;Kernels;Attention;Q16_1;HeadMajor;KVCache;Regression;CPU"
    MPI_PROCS 1
)

# Test: AttentionComputeStage (Phase 9 - ComputeStage for attention via KernelFactory)
add_executable(v2_test_attention_compute_stage unit/execution/compute_stages/stages/Test__AttentionComputeStage.cpp)
target_link_libraries(v2_test_attention_compute_stage
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionComputeStage
    COMMAND $<TARGET_FILE:v2_test_attention_compute_stage>
    LABELS "V2;Unit;Kernels;Attention;ComputeStage;KernelFactory;CPU"
    MPI_PROCS 1
)

# Test: CPUSoftmaxKernelT (Enum-based softmax kernel with native precision paths)
add_executable(v2_test_cpu_softmax_kernel_typed unit/kernels/cpu/ops/Test__CPUSoftmaxKernelT.cpp)
target_link_libraries(v2_test_cpu_softmax_kernel_typed
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUSoftmaxKernelT
    COMMAND $<TARGET_FILE:v2_test_cpu_softmax_kernel_typed>
    LABELS "V2;Unit;Kernels;Softmax;FP32;BF16;FP16;Q8_1;TypedKernel;CPU"
    MPI_PROCS 1
)

# Test: BF16 SIMD Accuracy (Stage 2 - SIMD-optimized BF16 conversion)
add_executable(v2_test_bf16_simd_accuracy unit/tensors/Test__BF16_SIMD_Accuracy.cpp)
target_link_libraries(v2_test_bf16_simd_accuracy
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BF16_SIMD_Accuracy
    COMMAND $<TARGET_FILE:v2_test_bf16_simd_accuracy>
    LABELS "V2;Unit;TensorOperations;SIMD;BF16;AVX2;AVX512;Quantization;AccuracyTest"
    MPI_PROCS 1  # CPU-only SIMD tests, single rank
)

# Test: Precision-aware allreduce N-way sum functions (FP16, BF16, Q8_1, Q16_1)
add_executable(v2_test_precision_aware_allreduce unit/collective/Test__PrecisionAwareAllreduce.cpp)
target_link_libraries(v2_test_precision_aware_allreduce 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PrecisionAwareAllreduce
    COMMAND $<TARGET_FILE:v2_test_precision_aware_allreduce>
    LABELS "V2;Unit;MPI;Allreduce;SIMD;FP16;BF16;Q8_1;Q16_1;AVX512;Precision"
    MPI_PROCS 1  # Tests SIMD sum functions locally, no MPI communication needed
)

# NOTE: Q8_0 decode vectorization test removed - depends on obsolete gemm implementations

# NOTE: VNNIGemmKernel and VNNIGemmAdapter tests removed - gemm_v3 is obsolete

# NOTE: FP16 GEMM tests removed - depend on obsolete gemm implementations (FP16GemmImpl, FP16PackedGemm)

# Test: RoPE primitives (scalar, AVX2, AVX512 parity)
add_executable(v2_test_rope_primitives unit/kernels/cpu/primitives/Test__RoPEPrimitives.cpp)
target_link_libraries(v2_test_rope_primitives 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RoPEPrimitives 
    COMMAND $<TARGET_FILE:v2_test_rope_primitives>
    LABELS "V2;Unit;Primitives;RoPE;SIMD;AVX2;AVX512;RotaryPositionEncoding;ImplementationParity;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: RoPE precision correctness (scalar/AVX2/AVX512 parity + cross-precision accuracy)
add_executable(v2_test_rope_precision unit/kernels/cpu/ops/Test__RoPEPrecisionCorrectness.cpp)
target_link_libraries(v2_test_rope_precision 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RoPEPrecisionCorrectness 
    COMMAND $<TARGET_FILE:v2_test_rope_precision>
    LABELS "V2;Unit;Primitives;RoPE;SIMD;AVX2;AVX512;Correctness;FP32;BF16;FP16;INT32;PrecisionTesting;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: Q16_1 RoPE SIMD correctness and performance
add_executable(v2_test_q16_1_rope unit/kernels/cpu/ops/Test__Q16_1RoPE.cpp)
target_link_libraries(v2_test_q16_1_rope 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16_1RoPE_Correctness 
    COMMAND $<TARGET_FILE:v2_test_q16_1_rope>
    LABELS "V2;Unit;Primitives;RoPE;Q16_1;SIMD;AVX2;AVX512;Correctness;Performance;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: Q16 Variable Block Size SIMD Operations
add_executable(v2_test_q16_variable_block_simd unit/tensors/Test__Q16VariableBlockSIMD.cpp)
target_link_libraries(v2_test_q16_variable_block_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16VariableBlockSIMD 
    COMMAND $<TARGET_FILE:v2_test_q16_variable_block_simd>
    LABELS "V2;Unit;SIMD;Q16_1;AVX2;AVX512;Correctness;Performance;CPU;ResidualAdd"
    MPI_PROCS 1
)

# Test: Q8_1 to Q16 RoPE Block Size Conversion (32/64/128/192 output blocks)
add_executable(v2_test_q8_1_to_q16_rope_blocksizes unit/kernels/cpu/ops/Test__Q8_1_to_Q16_RoPE_BlockSizes.cpp)
target_link_libraries(v2_test_q8_1_to_q16_rope_blocksizes 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_to_Q16_RoPE_BlockSizes 
    COMMAND $<TARGET_FILE:v2_test_q8_1_to_q16_rope_blocksizes>
    LABELS "V2;Unit;Primitives;RoPE;Q8_1;Q16_1;SIMD;AVX2;AVX512;Correctness;CPU;BlockSizes"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: Q8_1 RoPE SIMD correctness and performance
add_executable(v2_test_q8_1_rope unit/kernels/cpu/ops/Test__Q8_1RoPE.cpp)
target_link_libraries(v2_test_q8_1_rope 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1RoPE_Correctness 
    COMMAND $<TARGET_FILE:v2_test_q8_1_rope>
    LABELS "V2;Unit;Primitives;RoPE;Q8_1;SIMD;AVX2;AVX512;Correctness;Performance;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: Fixed-Scale Q8_1 to Q16 RoPE (for integer attention)
add_executable(v2_test_q8_1_to_q16_rope_fixed_scale unit/kernels/cpu/ops/Test__Q8_1_to_Q16_RoPE_FixedScale.cpp)
target_link_libraries(v2_test_q8_1_to_q16_rope_fixed_scale 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_to_Q16_RoPE_FixedScale 
    COMMAND $<TARGET_FILE:v2_test_q8_1_to_q16_rope_fixed_scale>
    LABELS "V2;Unit;Primitives;RoPE;Q8_1;Q16_1;FixedScale;IntegerAttention;Correctness;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: Fixed-Scale Q16 to Q16 RoPE
add_executable(v2_test_q16_to_q16_rope_fixed_scale unit/kernels/cpu/ops/Test__Q16_to_Q16_RoPE_FixedScale.cpp)
target_link_libraries(v2_test_q16_to_q16_rope_fixed_scale 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16_to_Q16_RoPE_FixedScale 
    COMMAND $<TARGET_FILE:v2_test_q16_to_q16_rope_fixed_scale>
    LABELS "V2;Unit;Primitives;RoPE;Q16_1;FixedScale;IntegerAttention;Correctness;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: CPURoPEKernelT class interface
add_executable(v2_test_cpu_rope_kernel unit/kernels/cpu/ops/Test__CPURoPEKernelT.cpp)
target_link_libraries(v2_test_cpu_rope_kernel 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURoPEKernelT_Interface 
    COMMAND $<TARGET_FILE:v2_test_cpu_rope_kernel>
    LABELS "V2;Unit;Kernels;RoPE;CPU"
    MPI_PROCS 1
)

# Test: CPUSwiGLUKernelT class interface
add_executable(v2_test_cpu_swiglu_kernel unit/kernels/cpu/ops/Test__CPUSwiGLUKernelT.cpp)
target_link_libraries(v2_test_cpu_swiglu_kernel 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUSwiGLUKernelT 
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu_kernel>
    LABELS "V2;Unit;Kernels;SwiGLU;CPU"
    MPI_PROCS 1
)

# Test: CPUSwiGLUKernelT (Enum-based SwiGLU kernel with native precision paths)
add_executable(v2_test_cpu_swiglu_kernel_typed unit/kernels/cpu/ops/Test__CPUSwiGLUKernelT.cpp)
target_link_libraries(v2_test_cpu_swiglu_kernel_typed
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUSwiGLUKernelT_Precision
    COMMAND $<TARGET_FILE:v2_test_cpu_swiglu_kernel_typed>
    LABELS "V2;Unit;Kernels;SwiGLU;TypedPrecision;FP32;BF16;FP16;Q8_1;CPU"
    MPI_PROCS 1
)

# Test: BPETokenizer (native tokenizer implementation)
add_executable(v2_test_bpe_tokenizer integration/loaders/Test__BPETokenizer.cpp)
target_link_libraries(v2_test_bpe_tokenizer 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_BPETokenizer 
    COMMAND $<TARGET_FILE:v2_test_bpe_tokenizer>
    LABELS "V2;Integration;Tokenization;BPE;NLP;TextProcessing"
    MPI_PROCS 1  # Tokenizer is single-rank only
)



# Test: Sampler (token sampling strategies for generation)
add_executable(v2_test_sampler unit/utils/Test__Sampler.cpp)
target_link_libraries(v2_test_sampler 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Sampler 
    COMMAND $<TARGET_FILE:v2_test_sampler>
    LABELS "V2;Unit;Sampling;Generation;BasicFeatures;Greedy;Temperature;TopK;TopP"
    MPI_PROCS 1  # Sampler is single-rank only
)

# Test: Deterministic mode policy gates
add_executable(v2_test_deterministic_mode unit/utils/Test__DeterministicMode.cpp)
target_link_libraries(v2_test_deterministic_mode
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeterministicMode
    COMMAND $<TARGET_FILE:v2_test_deterministic_mode>
    LABELS "V2;Unit;Determinism;DebugEnv;CUDA;ROCm;Policy"
    MPI_PROCS 1
)

# Test: BenchmarkRunnerCPU (CPU device handling in benchmark runner)
add_executable(v2_test_benchmark_runner_cpu unit/utils/Test__BenchmarkRunnerCPU.cpp)
target_include_directories(v2_test_benchmark_runner_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor  # For nlohmann/json.hpp
)
target_link_libraries(v2_test_benchmark_runner_cpu
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Unit_BenchmarkRunnerCPU
    COMMAND $<TARGET_FILE:v2_test_benchmark_runner_cpu>
    LABELS "V2;Unit;Benchmark;CPU;Regression"
    MPI_PROCS 1
)

# Test: PerfStatsCollector (unified structured perf counter export)
add_executable(v2_test_perf_stats_collector unit/utils/Test__PerfStatsCollector.cpp)
target_include_directories(v2_test_perf_stats_collector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor  # For nlohmann/json.hpp
)
target_link_libraries(v2_test_perf_stats_collector
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PerfStatsCollector
    COMMAND $<TARGET_FILE:v2_test_perf_stats_collector>
    LABELS "V2;Unit;Profiling;Benchmark;JSON;CSV"
    MPI_PROCS 1
)

# Test: ChatTemplate (chat template detection and formatting)
add_executable(v2_test_chat_template unit/utils/Test__ChatTemplate.cpp)
target_link_libraries(v2_test_chat_template 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ChatTemplate 
    COMMAND $<TARGET_FILE:v2_test_chat_template>
    LABELS "V2;Unit;Chat;Template;Tokenization;TextProcessing"
    MPI_PROCS 1  # ChatTemplate is single-rank only
)

# Test: ToolCallParser (tool call parsing for Hermes 2 Pro, Generic, etc.)
add_executable(v2_test_tool_call_parser unit/utils/Test__ToolCallParser.cpp)
target_include_directories(v2_test_tool_call_parser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor  # For nlohmann/json.hpp
)
target_link_libraries(v2_test_tool_call_parser
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ToolCallParser
    COMMAND $<TARGET_FILE:v2_test_tool_call_parser>
    LABELS "V2;Unit;ToolCalling;TextProcessing"
    MPI_PROCS 1
)

# Note: ArgParser tests removed - replaced by OrchestrationConfigParser tests

# Test: NodeTopology (NUMA/socket topology detection for CPU tensor parallelism)
add_executable(v2_test_node_topology unit/utils/Test__NodeTopology.cpp)
target_link_libraries(v2_test_node_topology 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_NodeTopology 
    COMMAND $<TARGET_FILE:v2_test_node_topology>
    LABELS "V2;Unit;Topology;NUMA;Socket;CPU;MPI;TensorParallelism"
    MPI_PROCS 1  # NodeTopology is single-rank only (reads sysfs)
)

# Test: NUMAAllocator (NUMA-aware memory allocation for cross-socket CPU tensor parallelism)
add_executable(v2_test_numa_allocator unit/memory/Test__NUMAAllocator.cpp)
target_link_libraries(v2_test_numa_allocator 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_NUMAAllocator 
    COMMAND $<TARGET_FILE:v2_test_numa_allocator>
    LABELS "V2;Unit;Memory;NUMA;Allocation;TensorParallelism"
    MPI_PROCS 1  # NUMAAllocator is single-rank only (memory allocation)
)

# Test: BufferArena (centralized buffer management, coherence tracking, borrow validation)
add_executable(v2_test_buffer_arena unit/memory/Test__BufferArena.cpp)
target_link_libraries(v2_test_buffer_arena 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BufferArena 
    COMMAND $<TARGET_FILE:v2_test_buffer_arena>
    LABELS "V2;Unit;Memory;BufferArena;Coherence;BufferHarmonization"
    MPI_PROCS 1
)

# Test: NUMATopology sysfs detection (hardware-independent unit tests with temp files)
add_executable(v2_test_numa_topology_sysfs unit/utils/Test__NUMATopologySysfs.cpp)
target_link_libraries(v2_test_numa_topology_sysfs 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_NUMATopologySysfs 
    COMMAND $<TARGET_FILE:v2_test_numa_topology_sysfs>
    LABELS "V2;Unit;NUMA;Sysfs;Topology;HardwareIndependent"
    MPI_PROCS 1  # Pure unit test using temp files, no hardware required
)

# Test: BPETokenizer byte encoding (GPT-2 style byte-level encoding/decoding)
add_executable(v2_test_bpe_byte_encoding unit/utils/Test__BPETokenizerByteEncoding.cpp)
target_link_libraries(v2_test_bpe_byte_encoding 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BPEByteEncoding 
    COMMAND $<TARGET_FILE:v2_test_bpe_byte_encoding>
    LABELS "V2;Unit;Tokenization;ByteEncoding;UTF8;Unicode;TextProcessing"
    MPI_PROCS 1  # Pure unit test, no MPI needed
)

# Test: Attention parity regression tests (locks in RoPE theta, causal masking, GQA behavior)

# Test: AttentionComputeStage dynamic kv_len from KV cache
add_executable(v2_test_attention_dynamic_kvlen unit/execution/compute_stages/stages/Test__AttentionComputeStage_DynamicKVLen.cpp)
target_link_libraries(v2_test_attention_dynamic_kvlen 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionComputeStage_DynamicKVLen 
    COMMAND $<TARGET_FILE:v2_test_attention_dynamic_kvlen>
    LABELS "V2;Unit;Attention;ComputeStage;DynamicKVLen;GraphBufferManagement;Phase9;Phase6;Padding;DecodeContinuation;CPU"
    MPI_PROCS 1  # Single rank for unit testing
)

# Test: Decode-mode KV cache override (regression test for stale activation buffer bug)
add_executable(v2_test_attention_decode_kv_override unit/execution/compute_stages/stages/Test__AttentionStage_DecodeKVOverride.cpp)
target_link_libraries(v2_test_attention_decode_kv_override
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionStage_DecodeKVOverride
    COMMAND $<TARGET_FILE:v2_test_attention_decode_kv_override>
    LABELS "V2;Unit;Attention;ComputeStage;DecodeKVOverride;Regression;CPU"
    MPI_PROCS 1
)

# Test: EmbeddingStage graph capture support
add_executable(v2_test_embedding_stage_graph_capture unit/execution/compute_stages/stages/Test__EmbeddingStage_GraphCapture.cpp)
target_link_libraries(v2_test_embedding_stage_graph_capture
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_EmbeddingStage_GraphCapture
    COMMAND $<TARGET_FILE:v2_test_embedding_stage_graph_capture>
    LABELS "V2;Unit;Execution;ComputeStage;Embedding;GraphCapture;DynamicParams;CPU"
    MPI_PROCS 1
)

# Test: ComputeStage tracing must not materialize tensor payloads during graph capture
add_executable(v2_test_compute_stage_trace_graph_capture unit/execution/compute_stages/Test__ComputeStageTraceGraphCapture.cpp)
target_link_libraries(v2_test_compute_stage_trace_graph_capture
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ComputeStageTraceGraphCapture
    COMMAND $<TARGET_FILE:v2_test_compute_stage_trace_graph_capture>
    LABELS "V2;Unit;Execution;ComputeStage;GraphCapture;Tracing;Regression;CPU"
    MPI_PROCS 1
)

# Test: SwiGLU parity regression tests (locks in correct formula: gate * silu(up))
add_executable(v2_test_swiglu_parity unit/kernels/cpu/ops/Test__SwiGLUParity.cpp)
target_link_libraries(v2_test_swiglu_parity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SwiGLUParity 
    COMMAND $<TARGET_FILE:v2_test_swiglu_parity>
    LABELS "V2;Unit;FFN;RegressionTests;SwiGLU;ParityTesting;CPU;Activation"
    MPI_PROCS 1  # Single rank for unit testing
)

# Test: Batch attention correctness - mask generation and batched processing

# Test: Sequential vs Batched Attention Comparison
add_executable(v2_test_attention_sequential_vs_batched unit/kernels/cpu/attention/Test__AttentionSequentialVsBatched.cpp)
target_link_libraries(v2_test_attention_sequential_vs_batched 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionSequentialVsBatched 
    COMMAND $<TARGET_FILE:v2_test_attention_sequential_vs_batched>
    LABELS "V2;Unit;Attention;Batched;Padding;SequentialComparison"
    MPI_PROCS 1  # Unit test comparing sequential vs batched execution
)

# Test: GQAAttention batched path (comprehensive batched attention tests)

# Test: GQAAttention padding mask correctness

# Test: GQAAttention mask passing verification (critical bug investigation)

# Test: MpiAttentionOrchestrator mask passing verification

# Test: Attention mask diagnostic (basic mask construction verification)
add_executable(v2_test_attention_mask_diagnostic unit/kernels/cpu/attention/Test__AttentionMaskDiagnostic.cpp)
target_link_libraries(v2_test_attention_mask_diagnostic 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionMaskDiagnostic
    COMMAND $<TARGET_FILE:v2_test_attention_mask_diagnostic>
    LABELS "V2;Unit;Attention;Masking;Diagnostic;MaskConstruction"
    MPI_PROCS 1  # Unit test for mask utility functions
)

# Test: INT8GemmKernel (OneDNN per-channel quantization)
## NOTE: INT8GemmKernel test removed - depends on obsolete GEMM v4 weight-owned APIs

# Test: Full INT8 Pipeline (end-to-end INT8 inference) (DISABLED - missing INT8GemmKernel.h)
# add_executable(v2_test_full_int8_pipeline Test__FullINT8Pipeline.cpp)
# target_link_libraries(v2_test_full_int8_pipeline 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_FullINT8Pipeline 
#     COMMAND $<TARGET_FILE:v2_test_full_int8_pipeline>
#     LABELS "V2;Unit;Kernels;INT8;Quantization;GEMM;Pipeline;INT32;Requantization;EndToEnd"
#     MPI_PROCS 1  # CPU-only, single rank
# )

# Test: TensorBase Conversion Interface
add_executable(v2_test_tensor_conversion unit/tensors/Test__TensorConversion.cpp)
target_link_libraries(v2_test_tensor_conversion 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorConversion 
    COMMAND $<TARGET_FILE:v2_test_tensor_conversion>
    LABELS "V2;Unit;TensorOperations;Conversion;FP32;BF16;FP16;INT8;RoundTrip;Correctness"
)

# Test: IQ4_NL Tensor Views
add_executable(v2_test_iq4nl_views unit/tensors/Test__IQ4_NLTensor_Views.cpp)
target_link_libraries(v2_test_iq4nl_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
# NOTE: IQ4_NL GEMM/Tilesweep perf tests removed - depends on obsolete
# GEMM v4 paths. IQ4_NL is now validated via tensor views and activation-
# centric kernels.

# Test: CPURingKVCache (Phase 1 ring metadata + append/evict state transitions)
add_executable(v2_test_cpu_ring_kv_cache unit/kernels/cpu/Test__CPURingKVCache.cpp)
target_link_libraries(v2_test_cpu_ring_kv_cache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURingKVCache
    COMMAND $<TARGET_FILE:v2_test_cpu_ring_kv_cache>
    LABELS "V2;Unit;KVCache;Ring;CPU;FP32;Phase1"
    MPI_PROCS 1
)

# Test: CPURingKVCache get_kv_converted (ring wrap correctness for FP32 shadow buffers)
add_executable(v2_test_cpu_ring_kv_cache_get_kv_converted unit/kernels/cpu/Test__CPURingKVCache_GetKvConverted.cpp)
target_link_libraries(v2_test_cpu_ring_kv_cache_get_kv_converted
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURingKVCache_GetKvConverted
    COMMAND $<TARGET_FILE:v2_test_cpu_ring_kv_cache_get_kv_converted>
    LABELS "V2;Unit;KVCache;Ring;CPU;FP32;FP16;GetKvConverted"
    MPI_PROCS 1
)

# Test: CPURingKVCache comprehensive (shadow invalidation, edge cases, multi-seq, eviction, factory, stress)
add_executable(v2_test_cpu_ring_kv_cache_comprehensive unit/kernels/cpu/Test__CPURingKVCache_Comprehensive.cpp)
target_link_libraries(v2_test_cpu_ring_kv_cache_comprehensive
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURingKVCache_Comprehensive
    COMMAND $<TARGET_FILE:v2_test_cpu_ring_kv_cache_comprehensive>
    LABELS "V2;Unit;KVCache;Ring;CPU;FP32;FP16;BF16;Shadow;Eviction;Factory;Comprehensive"
    MPI_PROCS 1
)

# Test: CPURingKVCache Q16_1 (variable block sizes, HEAD_MAJOR layout, get_kv_converted)
add_executable(v2_test_cpu_ring_kv_cache_q16_1 unit/kernels/cpu/Test__CPURingKVCache_Q16_1.cpp)
target_link_libraries(v2_test_cpu_ring_kv_cache_q16_1
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPURingKVCache_Q16_1
    COMMAND $<TARGET_FILE:v2_test_cpu_ring_kv_cache_q16_1>
    LABELS "V2;Unit;KVCache;Ring;CPU;Q16_1;HeadMajor;BlockSize;Shadow;Quantization"
    MPI_PROCS 1
)

# Test: IKVCache logical block IO for prefix cache restore/rollback
add_executable(v2_test_ikv_cache_logical_block_io unit/kernels/Test__IKVCacheLogicalBlockIO.cpp)
target_link_libraries(v2_test_ikv_cache_logical_block_io
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IKVCacheLogicalBlockIO
    COMMAND $<TARGET_FILE:v2_test_ikv_cache_logical_block_io>
    LABELS "V2;Unit;KVCache;PrefixCache;LogicalBlockIO;CPU;Q16_1;TQ"
    MPI_PROCS 1
)

# Test: KernelFactory KVCache methods (unified factory interface for KV cache creation)
add_executable(v2_test_kernelfactory_kvcache unit/kernels/Test__KernelFactory_KVCache.cpp)
target_link_libraries(v2_test_kernelfactory_kvcache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelFactory_KVCache 
    COMMAND $<TARGET_FILE:v2_test_kernelfactory_kvcache>
    LABELS "V2;Unit;KernelFactory;KVCache;FP32;BF16;FP16;Q8_1;Q16_1;Sharding;TensorParallelism"
    MPI_PROCS 1
)

# Test: HybridKVCache (hybrid FA+GDN cache: layer mapping, GDN state, factory creation)
add_executable(v2_test_hybrid_kvcache unit/kernels/Test__HybridKVCache.cpp)
target_link_libraries(v2_test_hybrid_kvcache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HybridKVCache 
    COMMAND $<TARGET_FILE:v2_test_hybrid_kvcache>
    LABELS "V2;Unit;KVCache;HybridKVCache;GDN;Qwen35;KernelFactory"
    MPI_PROCS 1
)

# Test: Q16SafeBlockAccess (Phase 2 - Safe Q16 block accessors and dispatch helpers)
add_executable(v2_test_q16_safe_block_access unit/tensors/Test__Q16SafeBlockAccess.cpp)
target_link_libraries(v2_test_q16_safe_block_access 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16SafeBlockAccess 
    COMMAND $<TARGET_FILE:v2_test_q16_safe_block_access>
    LABELS "V2;Unit;TensorOperations;Q16_1;Quantization;SafeBlockAccess;Phase2"
    MPI_PROCS 1
)

# Test: ShardedKVCache (tensor parallelism support - local KV heads)
add_executable(v2_test_sharded_kv_cache unit/execution/kvcache/Test__ShardedKVCache.cpp)
target_link_libraries(v2_test_sharded_kv_cache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ShardedKVCache 
    COMMAND $<TARGET_FILE:v2_test_sharded_kv_cache>
    LABELS "V2;Unit;KVCache;Sharding;TensorParallelism;FP32;BF16;FP16;Q8_1;Q16_1;DistributedArchitecture"
    MPI_PROCS 1
)

# Test: Q4_0 SIMD Unpacking (AVX-512, AVX2, SSE, Scalar)
add_executable(v2_test_q4_0_simd_unpack unit/tensors/Test__Q4_0_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q4_0_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_0_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q4_0_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Q4_0;Quantization;SIMD;AVX512;AVX2;SSE;Unpack;IINT8Unpackable;Parity"
    MPI_PROCS 1
)

# Test: Q4_K SIMD Unpacking (AVX-512, AVX2, Scalar)
add_executable(v2_test_q4_k_simd_unpack unit/tensors/Test__Q4_K_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q4_k_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_K_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q4_k_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Q4_K;Quantization;SIMD;AVX512;AVX2;Unpack"
    MPI_PROCS 1
)

# Test: Q5_K SIMD Unpacking (AVX-512, AVX2, Scalar)
add_executable(v2_test_q5_k_simd_unpack unit/tensors/Test__Q5_K_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q5_k_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_K_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q5_k_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Q5_K;Quantization;SIMD;AVX512;AVX2;Unpack"
    MPI_PROCS 1
)

# Test: Q6_K SIMD Unpacking (AVX-512, AVX2, Scalar)
add_executable(v2_test_q6_k_simd_unpack unit/tensors/Test__Q6_K_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q6_k_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q6_K_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q6_k_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Q6_K;Quantization;SIMD;AVX512;AVX2;Unpack"
    MPI_PROCS 1
)

# Test: Q5_0 SIMD Unpacking (AVX-512, AVX2, Scalar)
add_executable(v2_test_q5_0_simd_unpack unit/tensors/Test__Q5_0_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q5_0_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_0_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q5_0_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Q5_0;Quantization;SIMD;AVX512;AVX2;Unpack"
    MPI_PROCS 1
)

# Test: Q5_0 Transcoding and SIMD Unpacking
add_executable(v2_test_q5_0_transcode unit/tensors/Test__Q5_0_Transcode.cpp)
target_link_libraries(v2_test_q5_0_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_0_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q5_0_transcode>
    LABELS "V2;Unit;TensorOperations;Q5_0;Quantization;SIMD;Unpack;IINT8Unpackable"
    MPI_PROCS 1
)

# Test: Q5_1 Transcoding and SIMD Unpacking
add_executable(v2_test_q5_1_transcode unit/tensors/Test__Q5_1_Transcode.cpp)
target_link_libraries(v2_test_q5_1_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_1_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q5_1_transcode>
    LABELS "V2;Unit;TensorOperations;Q5_1;Quantization;SIMD;Unpack;IINT8Unpackable"
    MPI_PROCS 1
)

# Test: Q5_K Transcoding and SIMD Unpacking
add_executable(v2_test_q5_k_transcode unit/tensors/Test__Q5_K_Transcode.cpp)
target_link_libraries(v2_test_q5_k_transcode 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_K_Transcode 
    COMMAND $<TARGET_FILE:v2_test_q5_k_transcode>
    LABELS "V2;Unit;TensorOperations;Q5_K;Quantization;SIMD;Unpack;IINT8Unpackable"
    MPI_PROCS 1
)

# Test: IQ2_XS Tensor Views
add_executable(v2_test_iq2xs_views unit/tensors/Test__IQ2_XSTensor_Views.cpp)
target_link_libraries(v2_test_iq2xs_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XS_Views 
    COMMAND $<TARGET_FILE:v2_test_iq2xs_views>
    LABELS "V2;Unit;TensorOperations;IQ2_XS;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: IQ2_XS Decode Vectorization
add_executable(v2_test_iq2xs_decode_vectorization unit/tensors/Test__IQ2_XS_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq2xs_decode_vectorization 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XS_DecodeVectorization 
    COMMAND $<TARGET_FILE:v2_test_iq2xs_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;IQ2_XS;Quantization;SIMD;Unpack;IINT8Unpackable"
    MPI_PROCS 1
)

# Test: IQ2_XS Unpack Vectorization (AVX512)
add_executable(v2_test_iq2_xs_unpack_vectorization unit/tensors/Test__IQ2_XS_UnpackVectorization.cpp)
target_link_libraries(v2_test_iq2_xs_unpack_vectorization 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XS_UnpackVectorization 
    COMMAND $<TARGET_FILE:v2_test_iq2_xs_unpack_vectorization>
    LABELS "V2;Unit;TensorOperations;IQ2_XS;Quantization;SIMD;AVX512"
    MPI_PROCS 1
)

# Test: IQ1_M Decode Vectorization
add_executable(v2_test_iq1m_decode_vectorization unit/tensors/Test__IQ1_M_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq1m_decode_vectorization 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_M_DecodeVectorization 
    COMMAND $<TARGET_FILE:v2_test_iq1m_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;IQ1_M;Quantization;SIMD;Unpack;IINT8Unpackable"
    MPI_PROCS 1
)

# Test: IQ3_S Decode Vectorization
add_executable(v2_test_iq3s_decode_vectorization unit/tensors/Test__IQ3_S_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq3s_decode_vectorization 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_S_DecodeVectorization 
    COMMAND $<TARGET_FILE:v2_test_iq3s_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;IQ3_S;Quantization;SIMD;Unpack;IINT8Unpackable"
    MPI_PROCS 1
)

# Test: IQ3_XXS Tensor Views
add_executable(v2_test_iq3xxs_views unit/tensors/Test__IQ3_XXSTensor_Views.cpp)
target_link_libraries(v2_test_iq3xxs_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_XXS_Views 
    COMMAND $<TARGET_FILE:v2_test_iq3xxs_views>
    LABELS "V2;Unit;TensorOperations;IQ3_XXS;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: IQ3_XXS Tensor Interface (IINT8Unpackable)
add_executable(v2_test_iq3xxs_tensor unit/tensors/Test__IQ3_XXSTensor.cpp)
target_link_libraries(v2_test_iq3xxs_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_XXS_Tensor 
    COMMAND $<TARGET_FILE:v2_test_iq3xxs_tensor>
    LABELS "V2;Unit;TensorOperations;IQ3_XXS;Quantization;IINT8Unpackable"
)

# Test: IQ3_S Tensor Interface (IINT8Unpackable)
add_executable(v2_test_iq3s_tensor unit/tensors/Test__IQ3_STensor.cpp)
target_link_libraries(v2_test_iq3s_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_S_Tensor 
    COMMAND $<TARGET_FILE:v2_test_iq3s_tensor>
    LABELS "V2;Unit;TensorOperations;IQ3_S;Quantization;IINT8Unpackable"
)



# Test: IQ2_XXS Tensor (IINT8Unpackable)
add_executable(v2_test_iq2xxs_tensor unit/tensors/Test__IQ2_XXSTensor.cpp)
target_link_libraries(v2_test_iq2xxs_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XXSTensor 
    COMMAND $<TARGET_FILE:v2_test_iq2xxs_tensor>
    LABELS "V2;Unit;TensorOperations;IQ2_XXS;Quantization;IINT8Unpackable"
)

# Test: IQ2_XXS Unpack Vectorization (AVX512)
add_executable(v2_test_iq2_xxs_unpack_vectorization unit/tensors/Test__IQ2_XXS_UnpackVectorization.cpp)
target_link_libraries(v2_test_iq2_xxs_unpack_vectorization 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XXS_UnpackVectorization 
    COMMAND $<TARGET_FILE:v2_test_iq2_xxs_unpack_vectorization>
    LABELS "V2;Unit;TensorOperations;IQ2_XXS;Quantization;SIMD;AVX512"
    MPI_PROCS 1
)



# Test: IQ2_S Tensor Unit
add_executable(v2_test_iq2s_tensor unit/tensors/Test__IQ2_STensor.cpp)
target_link_libraries(v2_test_iq2s_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_STensor
    COMMAND $<TARGET_FILE:v2_test_iq2s_tensor>
    LABELS "V2;Unit;TensorOperations;IQ2_S;Quantization"
)



# Test: IQ2_S Tensor Views
add_executable(v2_test_iq2s_views unit/tensors/Test__IQ2_STensor_Views.cpp)
target_link_libraries(v2_test_iq2s_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_S_Views 
    COMMAND $<TARGET_FILE:v2_test_iq2s_views>
    LABELS "V2;Unit;TensorOperations;IQ2_S;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: IQ3_S Tensor Views
add_executable(v2_test_iq3s_views unit/tensors/Test__IQ3_STensor_Views.cpp)
target_link_libraries(v2_test_iq3s_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_S_Views 
    COMMAND $<TARGET_FILE:v2_test_iq3s_views>
    LABELS "V2;Unit;TensorOperations;IQ3_S;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: IQ1_S Tensor Views
add_executable(v2_test_iq1s_views unit/tensors/Test__IQ1_STensor_Views.cpp)
target_link_libraries(v2_test_iq1s_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_S_Views 
    COMMAND $<TARGET_FILE:v2_test_iq1s_views>
    LABELS "V2;Unit;TensorOperations;IQ1_S;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: IQ1_M Tensor Views
add_executable(v2_test_iq1m_views unit/tensors/Test__IQ1_MTensor_Views.cpp)
target_link_libraries(v2_test_iq1m_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_M_Views 
    COMMAND $<TARGET_FILE:v2_test_iq1m_views>
    LABELS "V2;Unit;TensorOperations;IQ1_M;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: Q8_0 Tensor Views
add_executable(v2_test_q8_0_views unit/tensors/Test__Q8_0Tensor_Views.cpp)
target_link_libraries(v2_test_q8_0_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_0_Views 
    COMMAND $<TARGET_FILE:v2_test_q8_0_views>
    LABELS "V2;Unit;TensorOperations;Q8_0;Quantization;ViewSupport;RowSlice"
)

# Test: Q8_0 Tensor Partial Blocks (Regression test for Jan 2026 buffer overflow fix)
add_executable(v2_test_q8_0_partial_blocks unit/tensors/Test__Q8_0Tensor_PartialBlocks.cpp)
target_link_libraries(v2_test_q8_0_partial_blocks 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_0_PartialBlocks 
    COMMAND $<TARGET_FILE:v2_test_q8_0_partial_blocks>
    LABELS "V2;Unit;TensorOperations;Q8_0;Quantization;Regression;PartialBlocks;BufferOverflow"
)

# Test: Quantized tensor tail decode regressions (K not divisible by 256)
add_executable(v2_test_quantized_tensor_tail_decode_regression unit/tensors/Test__QuantizedTensorTailDecodeRegression.cpp)
target_link_libraries(v2_test_quantized_tensor_tail_decode_regression
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_test_quantized_tensor_tail_decode_regression PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_QuantizedTensorTailDecodeRegression
    COMMAND $<TARGET_FILE:v2_test_quantized_tensor_tail_decode_regression>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Quantization;Regression;PartialBlocks;TailDecode;BufferOverflow"
)

# Test: Q4_0 Tensor Views
add_executable(v2_test_q4_0_views unit/tensors/Test__Q4_0Tensor_Views.cpp)
target_link_libraries(v2_test_q4_0_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_0_Views 
    COMMAND $<TARGET_FILE:v2_test_q4_0_views>
    LABELS "V2;Unit;TensorOperations;Q4_0;Quantization;ViewSupport;RowSlice"
)

# Test: Q4_1 Tensor Views
add_executable(v2_test_q4_1_views unit/tensors/Test__Q4_1Tensor_Views.cpp)
target_link_libraries(v2_test_q4_1_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_1_Views 
    COMMAND $<TARGET_FILE:v2_test_q4_1_views>
    LABELS "V2;Unit;TensorOperations;Q4_1;Quantization;ViewSupport;RowSlice"
)

# Test: Q6_K Tensor Views
add_executable(v2_test_q6_k_views unit/tensors/Test__Q6_KTensor_Views.cpp)
target_link_libraries(v2_test_q6_k_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q6_K_Views 
    COMMAND $<TARGET_FILE:v2_test_q6_k_views>
    LABELS "V2;Unit;TensorOperations;Q6_K;Quantization;ViewSupport;RowSlice;KQuant;SuperBlock"
)

# Test: Q2_K Tensor Views
add_executable(v2_test_q2_k_views unit/tensors/Test__Q2_KTensor_Views.cpp)
target_link_libraries(v2_test_q2_k_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q2_K_Views 
    COMMAND $<TARGET_FILE:v2_test_q2_k_views>
    LABELS "V2;Unit;TensorOperations;Q2_K;Quantization;ViewSupport;RowSlice;KQuant;SuperBlock"
)

# Test: Q3_K Tensor Views
add_executable(v2_test_q3_k_views unit/tensors/Test__Q3_KTensor_Views.cpp)
target_link_libraries(v2_test_q3_k_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q3_K_Views 
    COMMAND $<TARGET_FILE:v2_test_q3_k_views>
    LABELS "V2;Unit;TensorOperations;Q3_K;Quantization;ViewSupport;RowSlice;KQuant;SuperBlock"
)

# Test: Q4_K Tensor Views
add_executable(v2_test_q4_k_views unit/tensors/Test__Q4_KTensor_Views.cpp)
target_link_libraries(v2_test_q4_k_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_K_Views 
    COMMAND $<TARGET_FILE:v2_test_q4_k_views>
    LABELS "V2;Unit;TensorOperations;Q4_K;Quantization;ViewSupport;RowSlice;KQuant;SuperBlock"
)

# Test: Q5_K Tensor Views
add_executable(v2_test_q5_k_views unit/tensors/Test__Q5_KTensor_Views.cpp)
target_link_libraries(v2_test_q5_k_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_K_Views 
    COMMAND $<TARGET_FILE:v2_test_q5_k_views>
    LABELS "V2;Unit;TensorOperations;Q5_K;Quantization;ViewSupport;RowSlice;KQuant;SuperBlock"
)

# Test: Q8_K Tensor Views
add_executable(v2_test_q8_k_views unit/tensors/Test__Q8_KTensor_Views.cpp)
target_link_libraries(v2_test_q8_k_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_K_Views 
    COMMAND $<TARGET_FILE:v2_test_q8_k_views>
    LABELS "V2;Unit;TensorOperations;Q8_K;Quantization;ViewSupport;RowSlice;KQuant;SuperBlock"
)

# Test: Q5_0 Tensor Views
add_executable(v2_test_q5_0_views unit/tensors/Test__Q5_0Tensor_Views.cpp)
target_link_libraries(v2_test_q5_0_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_0_Views 
    COMMAND $<TARGET_FILE:v2_test_q5_0_views>
    LABELS "V2;Unit;TensorOperations;Q5_0;Quantization;ViewSupport;RowSlice"
)

# Test: Q5_1 Tensor Views
add_executable(v2_test_q5_1_views unit/tensors/Test__Q5_1Tensor_Views.cpp)
target_link_libraries(v2_test_q5_1_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_1_Views 
    COMMAND $<TARGET_FILE:v2_test_q5_1_views>
    LABELS "V2;Unit;TensorOperations;Q5_1;Quantization;ViewSupport;RowSlice"
)

# Test: Q8_1 Tensor Views
add_executable(v2_test_q8_1_views unit/tensors/Test__Q8_1Tensor_Views.cpp)
target_link_libraries(v2_test_q8_1_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_Views 
    COMMAND $<TARGET_FILE:v2_test_q8_1_views>
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;ViewSupport;RowSlice"
)

# Test: Q16_1 Tensor Views
add_executable(v2_test_q16_1_views unit/tensors/Test__Q16_1Tensor_Views.cpp)
target_link_libraries(v2_test_q16_1_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16_1_Views 
    COMMAND $<TARGET_FILE:v2_test_q16_1_views>
    LABELS "V2;Unit;TensorOperations;Q16_1;Quantization;ViewSupport;RowSlice"
)

# =============================================================================
# SIMD Path Equivalency Tests (Phase 1: Q4_0, Q4_1, Q5_0, Q5_1, Q6_K)
# =============================================================================
# These tests validate that scalar, AVX2, and AVX512 dequantization paths
# produce identical results for quantized tensors with SIMD optimizations.
# Tests use direct method calls (decodeBlockScalar, decodeBlockAVX2, decodeBlockAVX512)
# with synthetic test data to ensure numerical equivalency across all code paths.

# Test: Q4_0 SIMD Equivalency
add_executable(v2_test_q4_0_simd integration/tensors/Test__Q4_0Tensor.cpp)
target_link_libraries(v2_test_q4_0_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q4_0_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q4_0_simd>
    MPI_PROCS 1
    LABELS "V2;Integration;TensorOperations;Q4_0;Quantization;SIMD;AVX2;AVX512;Equivalency"
)

# Test: Q4_1 SIMD Equivalency
add_executable(v2_test_q4_1_simd unit/tensors/Test__Q4_1Tensor.cpp)
target_link_libraries(v2_test_q4_1_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_1_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q4_1_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q4_1;Quantization;SIMD;AVX2;AVX512;Equivalency;FMA"
)

# Test: Q5_0 SIMD Equivalency
add_executable(v2_test_q5_0_simd integration/tensors/Test__Q5_0Tensor.cpp)
target_link_libraries(v2_test_q5_0_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q5_0_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q5_0_simd>
    MPI_PROCS 1
    LABELS "V2;Integration;TensorOperations;Q5_0;Quantization;SIMD;AVX2;AVX512;Equivalency;HighBit"
)



# Test: Q5_1 SIMD Equivalency
add_executable(v2_test_q5_1_simd unit/tensors/Test__Q5_1Tensor.cpp)
target_link_libraries(v2_test_q5_1_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_1_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q5_1_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q5_1;Quantization;SIMD;AVX2;AVX512;Equivalency;HighBit;FMA"
)

# Test: Q5_1 SIMD Unpack
add_executable(v2_test_q5_1_simd_unpack unit/tensors/Test__Q5_1_SIMD_Unpack.cpp)
target_link_libraries(v2_test_q5_1_simd_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_1_SIMD_Unpack 
    COMMAND $<TARGET_FILE:v2_test_q5_1_simd_unpack>
    LABELS "V2;Unit;TensorOperations;Quantization;Q5_1;SIMD"
    MPI_PROCS 1
)

# Test: Q6_K SIMD Equivalency
add_executable(v2_test_q6_k_simd unit/tensors/Test__Q6_KTensor.cpp)
target_link_libraries(v2_test_q6_k_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q6_K_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q6_k_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q6_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock"
)

# =============================================================================
# SIMD Path Equivalency Tests (Phase 2: Q_K Variants)
# =============================================================================
# Q2_K, Q3_K, Q4_K, Q5_K, Q8_K SIMD equivalency tests

# Test: Q2_K SIMD Equivalency
add_executable(v2_test_q2_k_simd unit/tensors/Test__Q2_KTensor.cpp)
target_link_libraries(v2_test_q2_k_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q2_K_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q2_k_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q2_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock"
)

# Test: Q8_1 Quantize SIMD Equivalency (FP32 → Q8_1)
add_executable(v2_test_q8_1_quantize_simd unit/tensors/Test__Q8_1_QuantizeSIMD.cpp)
target_link_libraries(v2_test_q8_1_quantize_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_QuantizeSIMD 
    COMMAND $<TARGET_FILE:v2_test_q8_1_quantize_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q8_1;Quantization;SIMD;AVX2;AVX512;Equivalency"
)

# Test: Q8_0 SIMD Equivalency
add_executable(v2_test_q8_0_simd integration/tensors/Test__Q8_0Tensor.cpp)
target_link_libraries(v2_test_q8_0_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Q8_0_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q8_0_simd>
    MPI_PROCS 1
    LABELS "V2;Integration;TensorOperations;Q8_0;Quantization;SIMD;AVX2;AVX512;Equivalency"
)

# Test: Q3_K SIMD Equivalency
add_executable(v2_test_q3_k_simd unit/tensors/Test__Q3_KTensor.cpp)
target_link_libraries(v2_test_q3_k_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q3_K_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q3_k_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q3_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock"
)

# Test: Q4_K SIMD Equivalency
add_executable(v2_test_q4_k_simd unit/tensors/Test__Q4_KTensor.cpp)
target_link_libraries(v2_test_q4_k_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_K_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q4_k_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q4_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock"
)

# Test: Q5_K SIMD Equivalency
add_executable(v2_test_q5_k_simd unit/tensors/Test__Q5_KTensor.cpp)
target_link_libraries(v2_test_q5_k_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_K_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q5_k_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q5_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock;HighBits;MaskShifting"
)

# Test: Q8_K SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate Q8_K int8→float conversion across SIMD implementations
# Expected: 9 tests total (3 equivalency + 6 edge cases)
# Q8_K: Simplest quantization - just direct int8→float type conversion, no scale/lookup
add_executable(v2_test_q8_k_simd unit/tensors/Test__Q8_KTensor.cpp)
target_link_libraries(v2_test_q8_k_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_K_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q8_k_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q8_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock;Int8Conversion"
)

# Test: requantizeRowToInt8() accuracy for Q8_0, Q8_1, Q8_K
# Purpose: Validate SIMD (AVX-512/AVX2) and scalar requantization paths produce
#          accurate INT8 output with correct row scales for ROCm weight packing.
# Coverage: reconstruction accuracy, clamping, zero rows, multi-row independence,
#           SIMD-vs-scalar consistency, Q8_K direct copy and -128 rescaling.
add_executable(v2_test_requantize_row_to_int8 unit/tensors/Test__RequantizeRowToInt8.cpp)
target_link_libraries(v2_test_requantize_row_to_int8
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RequantizeRowToInt8
    COMMAND $<TARGET_FILE:v2_test_requantize_row_to_int8>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Quantization;SIMD;AVX2;AVX512;Q8_0;Q8_1;Q8_K;Int8Conversion;ROCm"
)

# =============================================================================
# TurboQuant KV Cache Quantization Tests (Phase 1: Foundation)
# =============================================================================

# Test: TurboQuant Codebook
# Validates Lloyd-Max codebook centroids for 4-bit and 3-bit quantization.
# Tests centroid accuracy vs re-derived values, symmetry, sorted order,
# threshold midpoints, nearest-centroid lookup, TQ3 bit packing, and
# theoretical MSE bounds.
add_executable(v2_test_turboquant_codebook unit/Test__TurboQuantCodebook.cpp)
target_link_libraries(v2_test_turboquant_codebook
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TurboQuantCodebook
    COMMAND $<TARGET_FILE:v2_test_turboquant_codebook>
    MPI_PROCS 1
    LABELS "V2;Unit;Quantization;TurboQuant;Codebook;LloydMax"
)

# Test: TurboQuant Roundtrip
# Validates full quantize→dequantize roundtrip for TQ4 and TQ3 at
# head_dim=64 and 128. Tests rotation orthogonality, norm preservation,
# determinism, zero vectors, MSE within paper bounds, batch consistency,
# and TQ4 < TQ3 quality ordering.
add_executable(v2_test_turboquant_roundtrip unit/Test__TurboQuantRoundtrip.cpp)
target_link_libraries(v2_test_turboquant_roundtrip
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TurboQuantRoundtrip
    COMMAND $<TARGET_FILE:v2_test_turboquant_roundtrip>
    MPI_PROCS 1
    LABELS "V2;Unit;Quantization;TurboQuant;Roundtrip;MSE;Rotation"
)

# Split TQ KV Cache Integration (TQ8-K / TQ4-V)
# Tests split precision data flowing through the CPURingKVCache:
# append, gather, ring wrap, incremental decode, multi-layer, clear.
add_executable(v2_test_splittq_kvcache unit/kernels/cpu/Test__CPURingKVCache_SplitTQ.cpp)
target_link_libraries(v2_test_splittq_kvcache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SplitTQ_KVCache
    COMMAND $<TARGET_FILE:v2_test_splittq_kvcache>
    MPI_PROCS 1
    LABELS "V2;Unit;KVCache;TurboQuant;TQ8;TQ4;SplitTQ;Ring;CPU;Quantization"
)

# Split TQ Attention with head_dim=128 (Qwen3-like)
# Tests split TQ8-K/TQ4-V round-trip, cache round-trip, dequant correctness, and full
# attention pipeline with head_dim=128 — the code path that differs from Qwen2's head_dim=64.
add_executable(v2_test_splittq_attention_headdim128 unit/kernels/cpu/Test__SplitTQ_Attention_HeadDim128.cpp)
target_link_libraries(v2_test_splittq_attention_headdim128
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SplitTQ_Attention_HeadDim128
    COMMAND $<TARGET_FILE:v2_test_splittq_attention_headdim128>
    MPI_PROCS 1
    LABELS "V2;Unit;KVCache;TurboQuant;TQ8;TQ4;SplitTQ;Attention;CPU;Quantization"
)

# Fused TQ attention decode path: validates the rotation-orthogonality trick where
# Q is pre-rotated once and QK/V ops use centroids directly (zero shadow buffers).
add_executable(v2_test_tq_fused_attention unit/kernels/cpu/attention/Test__TQFusedAttention.cpp)
target_link_libraries(v2_test_tq_fused_attention
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TQFusedAttention
    COMMAND $<TARGET_FILE:v2_test_tq_fused_attention>
    MPI_PROCS 1
    LABELS "V2;Unit;TurboQuant;TQ8;TQ4;Attention;CPU;Kernels;FlashAttention"
)

# Q8_1 / Q16_1 Fused Attention Tests
# Validates that fused attention paths (bypassing FP32 shadow buffers) for Q8_1
# and Q16_1 KV cache formats match the dequant-first reference path.
add_executable(v2_test_q8q16_fused_attention unit/kernels/cpu/attention/Test__Q8Q16FusedAttention.cpp)
target_link_libraries(v2_test_q8q16_fused_attention
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8Q16FusedAttention
    COMMAND $<TARGET_FILE:v2_test_q8q16_fused_attention>
    MPI_PROCS 1
    LABELS "V2;Unit;Q8_1;Q16_1;Attention;CPU;Kernels;FlashAttention"
)

# Split TQ Regression Tests
# Targeted regression tests for 4 specific bugs found during split TQ8/TQ4 integration:
# 1. Per-head context derivation in turboquant_dequantize_row_tq8
# 2. Row byte-offset calculation with typed_data() pointer arithmetic
# 3. Buffer overflow from dequantize_to_fp32 on cache-sized tensors
# 4. Context mismatch between quantize and dequantize layer contexts
add_executable(v2_test_splittq_regression unit/kernels/cpu/Test__SplitTQ_Regression.cpp)
target_link_libraries(v2_test_splittq_regression
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SplitTQ_Regression
    COMMAND $<TARGET_FILE:v2_test_splittq_regression>
    MPI_PROCS 1
    LABELS "V2;Unit;TurboQuant;TQ8;TQ4;SplitTQ;Regression;CPU;Quantization"
)

# TQ8 Round-trip Quality Tests
# Tests TQ8 block-level quantize/dequantize quality, TQ8 vs TQ4 comparison,
# TQ8Tensor high-level API, determinism, extreme norms, inner product preservation.
add_executable(v2_test_tq8_roundtrip unit/Test__TQ8Roundtrip.cpp)
target_link_libraries(v2_test_tq8_roundtrip
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TQ8Roundtrip
    COMMAND $<TARGET_FILE:v2_test_tq8_roundtrip>
    MPI_PROCS 1
    LABELS "V2;Unit;Quantization;TurboQuant;TQ8;Roundtrip;MSE;Tensor"
)

# TQ8 and TQ8/TQ4 Split-Precision KV Cache Tests
# Tests CPURingKVCacheTQ8 (symmetric) and CPURingKVCacheTQ (asymmetric TQ8-K / TQ4-V):
# precision reporting, append/gather round-trip, ring wrap, incremental decode,
# multi-layer independence, clear, K quality >> V quality validation.
add_executable(v2_test_tq8_kvcache unit/kernels/cpu/Test__CPURingKVCache_TQ8.cpp)
target_link_libraries(v2_test_tq8_kvcache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TQ8_KVCache
    COMMAND $<TARGET_FILE:v2_test_tq8_kvcache>
    MPI_PROCS 1
    LABELS "V2;Unit;KVCache;TurboQuant;TQ8;TQ4;Ring;CPU;Quantization;SplitPrecision"
)

# Test: ModelContextConfig
# Tests configuration struct for ModelContext creation (PP/TP/combined scenarios)
add_executable(v2_unit_modelcontextconfig unit/loaders/Test__ModelContextConfig.cpp)
target_link_libraries(v2_unit_modelcontextconfig llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_ModelContextConfig
    COMMAND $<TARGET_FILE:v2_unit_modelcontextconfig>
    MPI_PROCS 1
    LABELS "V2;Unit;Loaders;ModelContext;PipelineParallelism;TensorParallelism"
)

# Test: WeightPlacementMap
# WeightPlacementMap Tests (Tensor-Level Device Placement - Phase 2)
add_executable(v2_test_weight_placement_map unit/loaders/Test__WeightPlacementMap.cpp)
target_link_libraries(v2_test_weight_placement_map llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightPlacementMap
    COMMAND $<TARGET_FILE:v2_test_weight_placement_map>
    LABELS "V2;Unit;DeviceManagement;WeightPlacement;Heterogeneous;MoE"
)

# Test: Weight identity and lifecycle trace scaffolding
add_executable(v2_unit_weight_identity unit/loaders/Test__WeightIdentity.cpp)
target_link_libraries(v2_unit_weight_identity llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightIdentity
    COMMAND $<TARGET_FILE:v2_unit_weight_identity>
    LABELS "V2;Unit;Loaders;Weights;Lifecycle;Identity"
    MPI_PROCS 1
)

add_executable(v2_unit_weight_plan unit/loaders/Test__WeightPlan.cpp)
target_link_libraries(v2_unit_weight_plan llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightPlan
    COMMAND $<TARGET_FILE:v2_unit_weight_plan>
    LABELS "V2;Unit;Loaders;Weights;Lifecycle;Planning"
    MPI_PROCS 1
)

add_executable(v2_unit_prepared_weight_store unit/loaders/Test__PreparedWeightStore.cpp)
target_link_libraries(v2_unit_prepared_weight_store llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PreparedWeightStore
    COMMAND $<TARGET_FILE:v2_unit_prepared_weight_store>
    LABELS "V2;Unit;Loaders;Weights;Lifecycle;PreparedWeights"
    MPI_PROCS 1
)

add_executable(v2_unit_prepared_weight_store_expert_slab unit/loaders/Test__PreparedWeightStore_ExpertSlab.cpp)
target_link_libraries(v2_unit_prepared_weight_store_expert_slab llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PreparedWeightStoreExpertSlab
    COMMAND $<TARGET_FILE:v2_unit_prepared_weight_store_expert_slab>
    LABELS "V2;Unit;Loaders;Weights;Lifecycle;PreparedWeights;MoE;ExpertSlab"
    MPI_PROCS 1
)

# Test: WeightManager Sharding
# Tests weight sharding for tensor parallelism (Phase 1: row-parallel only)
add_executable(v2_test_weight_manager_sharding unit/loaders/Test__WeightManagerSharding.cpp)
target_link_libraries(v2_test_weight_manager_sharding llaminar2_core GTest::gtest_main)
target_include_directories(v2_test_weight_manager_sharding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ includes
)
add_v2_test(V2_Unit_WeightManagerSharding
    COMMAND $<TARGET_FILE:v2_test_weight_manager_sharding>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorParallelism;WeightSharding;RowParallel"
)

# Test: WeightManager Decode Shards
# Tests decode weight shard slicing for CPU decode participation (Option A: Selective Duplication)
add_executable(v2_test_weight_manager_decode_shards unit/loaders/Test__WeightManagerDecodeShards.cpp)
target_link_libraries(v2_test_weight_manager_decode_shards llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightManagerDecodeShards
    COMMAND $<TARGET_FILE:v2_test_weight_manager_decode_shards>
    LABELS "V2;Unit;TensorParallelism;WeightSharding;DecodeShards;CPUDecode"
)

# Test: WeightManager INPUT_PARALLEL Slicing
# Exhaustive tests for INPUT_PARALLEL weight slicing (attn_output vs ffn_down)
# Tests the critical distinction between ATTENTION_WO (head-based slicing) and
# FFN_DOWN (d_ff-based slicing) - regression tests for the bug where d_ff dims
# were incorrectly used for all INPUT_PARALLEL weights.
add_executable(v2_test_weight_manager_input_parallel_slicing unit/loaders/Test__WeightManagerInputParallelSlicing.cpp)
target_include_directories(v2_test_weight_manager_input_parallel_slicing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes (TestTensorFactory)
)
target_link_libraries(v2_test_weight_manager_input_parallel_slicing llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightManagerInputParallelSlicing
    COMMAND $<TARGET_FILE:v2_test_weight_manager_input_parallel_slicing>
    LABELS "V2;Unit;TensorParallelism;WeightSharding;InputParallel;AttentionWo;FFNDown;LocalTP"
    MPI_PROCS 1
)

# Test: WeightManager FusedQKV Sharding + Tied Embeddings
# Regression tests for two critical TP bugs:
# 1. FusedQKV: getShardedWeight() must slice 3 sub-blocks independently, not contiguously
# 2. Tied Embeddings: output.weight falls back to column-parallel token_embd.weight
add_executable(v2_test_weight_manager_fusedqkv_tied_embeddings unit/loaders/Test__WeightManager_FusedQKV_TiedEmbeddings.cpp)
target_include_directories(v2_test_weight_manager_fusedqkv_tied_embeddings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ includes
)
target_link_libraries(v2_test_weight_manager_fusedqkv_tied_embeddings llaminar2_core GTest::gtest_main)

# Test: WeightSlicer — stateless slicing math
# Pure computation tests for weight slicing without I/O or MPI dependencies.
# Tests column-parallel, row-parallel, FusedQKV sub-block, GDN asymmetric,
# proportional (TensorParallelConfig), and device-assignment slicing.
add_executable(v2_unit_weight_slicer unit/loaders/Test__WeightSlicer.cpp)
target_link_libraries(v2_unit_weight_slicer llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightSlicer
    COMMAND $<TARGET_FILE:v2_unit_weight_slicer>
    LABELS "V2;Unit;TensorParallelism;WeightSharding;Slicing"
    MPI_PROCS 1
)

# Test: WeightLoadProgress - Progress bar rendering, IProgressPublisher, aggregator integration
add_executable(v2_unit_weight_load_progress unit/loaders/Test__WeightLoadProgress.cpp)
target_link_libraries(v2_unit_weight_load_progress llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightLoadProgress
    COMMAND $<TARGET_FILE:v2_unit_weight_load_progress>
    LABELS "V2;Unit;Loaders;Progress;MPI"
    MPI_PROCS 1
)

# Test: PPActivationContract (PP inter-stage transfer contract)
add_executable(v2_unit_pp_activation_contract unit/collective/Test__PPActivationContract.cpp)
target_link_libraries(v2_unit_pp_activation_contract llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PPActivationContract
    COMMAND $<TARGET_FILE:v2_unit_pp_activation_contract>
    LABELS "V2;Unit;PipelineParallelism;Collective"
    MPI_PROCS 1
)

add_v2_test(V2_Unit_WeightManagerFusedQKVTiedEmbeddings
    COMMAND $<TARGET_FILE:v2_test_weight_manager_fusedqkv_tied_embeddings>
    LABELS "V2;Unit;TensorParallelism;WeightSharding;FusedQKV;TiedEmbeddings;GatedDeltaNet;Qwen35"
    MPI_PROCS 1
)

# Test: GDN FusedQKV TP Sharding
# Regression tests for GDN (Gated DeltaNet) asymmetric QKV weight sharding.
# GDN layers have Q=K≠V layout that doesn't match FA-based detection.
# setGDNDimensions() enables correct sub-block slicing under TP.
add_executable(v2_test_weight_manager_gdn_fusedqkv unit/loaders/Test__WeightManager_GDNFusedQKV.cpp)
target_include_directories(v2_test_weight_manager_gdn_fusedqkv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ includes
)
target_link_libraries(v2_test_weight_manager_gdn_fusedqkv llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightManagerGDNFusedQKV
    COMMAND $<TARGET_FILE:v2_test_weight_manager_gdn_fusedqkv>
    LABELS "V2;Unit;TensorParallelism;WeightSharding;FusedQKV;GatedDeltaNet;Qwen35"
    MPI_PROCS 1
)

# Test: WeightManager Device-Aware Features
# Tests features folded from WeightPreloader: getWeightForDevice, preloadForDevices,
# cloneTensorForDevice, packGemmWeights, uploadNonGemmWeights, preloadStats,
# per-device caching with first_device tracking
add_executable(v2_test_weight_manager_device_aware unit/loaders/Test__WeightManager_DeviceAware.cpp)
target_include_directories(v2_test_weight_manager_device_aware PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ and tests/v2/utils/ includes
)
target_link_libraries(v2_test_weight_manager_device_aware
    llaminar2_core
    GTest::gtest_main
)
add_v2_test(V2_Unit_WeightManager_DeviceAware
    COMMAND $<TARGET_FILE:v2_test_weight_manager_device_aware>
    LABELS "V2;Unit;WeightManager;DeviceManagement;TensorParallelism;LocalTP;MultiDevice"
    MPI_PROCS 1
)

# Test: WeightManager Virtual Weight Preloading (Regression)
# Regression tests for TP GEMM cache miss when tied embeddings (output.weight)
# aren't in the GGUF tensor list. Verifies that virtual weights from
# WeightShardingConfig.exact_matches are included in preloadForDevices().
add_executable(v2_test_weight_preload_virtual_weights unit/Test__WeightPreloadVirtualWeights.cpp)
target_include_directories(v2_test_weight_preload_virtual_weights PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ includes
)
target_link_libraries(v2_test_weight_preload_virtual_weights
    llaminar2_core
    GTest::gtest_main
)
add_v2_test(V2_Unit_WeightPreloadVirtualWeights
    COMMAND $<TARGET_FILE:v2_test_weight_preload_virtual_weights>
    LABELS "V2;Unit;WeightManager;TensorParallelism;TiedEmbeddings;Regression"
    MPI_PROCS 1
)

# Test: NullWeightStreamer
# No-op weight streamer for RESIDENT mode (Option B - weights fit in GPU VRAM)
add_executable(v2_test_null_weight_streamer unit/loaders/Test__NullWeightStreamer.cpp)
target_link_libraries(v2_test_null_weight_streamer llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_NullWeightStreamer
    COMMAND $<TARGET_FILE:v2_test_null_weight_streamer>
    LABELS "V2;Unit;WeightStreaming;OptionB;RESIDENT"
)

# Test: LayerWeightStreamer
# Weight streaming implementation for STREAMING mode (Option B - weights exceed GPU VRAM)
add_executable(v2_test_layer_weight_streamer unit/loaders/Test__LayerWeightStreamer.cpp)
target_link_libraries(v2_test_layer_weight_streamer llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_LayerWeightStreamer
    COMMAND $<TARGET_FILE:v2_test_layer_weight_streamer>
    LABELS "V2;Unit;WeightStreaming;OptionB;STREAMING;LRU;Cache"
)

# Test: WeightStreamerFactory
# Factory for creating IWeightStreamer implementations based on config/env/auto-detection
add_executable(v2_test_weight_streamer_factory unit/loaders/Test__WeightStreamerFactory.cpp)
target_link_libraries(v2_test_weight_streamer_factory llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightStreamerFactory
    COMMAND $<TARGET_FILE:v2_test_weight_streamer_factory>
    LABELS "V2;Unit;WeightStreaming;OptionB;Factory;AutoDetection"
)

# =============================================================================
# Execution Framework Tests (WorkDistributor, DeviceContext)
# =============================================================================

# Test: WorkDistributor (hierarchical work distribution across ranks/devices)
# Tests rank-level slicing, device-level slicing with weights, and hierarchical distribution
add_executable(v2_test_work_distributor unit/execution/mpi_orchestration/Test__WorkDistributor.cpp)
target_link_libraries(v2_test_work_distributor llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_WorkDistributor
    COMMAND $<TARGET_FILE:v2_test_work_distributor>
    LABELS "V2;Unit;Execution;WorkDistribution;MPI;DeviceManagement;HierarchicalSlicing"
    MPI_PROCS 1  # Tests slice computation, not actual MPI communication
)

# Test: DeviceContext (abstract execution context for CPU/GPU)
# Tests CPUDeviceContext memory allocation, parallel execution, and workspace management
add_executable(v2_test_device_context unit/execution/local_execution/device/Test__DeviceContext.cpp)
target_link_libraries(v2_test_device_context llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceContext
    COMMAND $<TARGET_FILE:v2_test_device_context>
    LABELS "V2;Unit;Execution;DeviceContext;OpenMP;MemoryManagement;ParallelExecution"
    MPI_PROCS 1  # Tests CPU context locally, no MPI needed
)

# Test: MultiGPU (Phase 6 - heterogeneous GPU support with CUDA + ROCm)
# Tests DeviceManager enumeration, backend-specific device ID mapping, 
# tensor GPU transfers, cross-device transfers, and kernel fallbacks
add_executable(v2_test_multi_gpu unit/backends/Test__MultiGPU.cpp)
target_link_libraries(v2_test_multi_gpu llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_MultiGPU
    COMMAND $<TARGET_FILE:v2_test_multi_gpu>
    LABELS "V2;Integration;DeviceManagement;MultiGPU;CUDA;ROCm;HeterogeneousBackends;TensorTransfer;KernelFallback"
    MPI_PROCS 1  # Tests device management locally, no MPI needed
)

# =============================================================================
# CUDA Kernel Infrastructure Tests (Phase 1)
# =============================================================================
# Test: CUDA Tensor Transfer (Phase 1 - GPU upload/download via TensorBase::ensureOnDevice)
# Tests FP32, BF16, FP16 tensor upload/download, round-trip verification, edge cases
add_executable(v2_test_cuda_tensor_transfer unit/kernels/cuda/Test__CUDATensorTransfer.cpp)
target_link_libraries(v2_test_cuda_tensor_transfer llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDATensorTransfer
    COMMAND $<TARGET_FILE:v2_test_cuda_tensor_transfer>
    LABELS "V2;Integration;CUDA;TensorTransfer;FP32;BF16;FP16;GPUMemory;DeviceManagement"
    MPI_PROCS 1  # Tests GPU transfers locally, no MPI needed
)

# Test: CUDA Flash Attention Decode KV Length Bug (Bug fix validation)
# Bug discovered 2026-01-07: compute() hardcoded kv_len = seq_len. For decode mode
# with seq_len=1 but kv_len=7, GPU only attended to 1 token instead of 7.
# Fix: Use compute_decode() which takes separate seq_len and kv_len parameters.
add_executable(v2_unit_cuda_flash_attention_decode_kvlen unit/kernels/cuda/Test__CUDAFlashAttention_DecodeKVLen.cpp)
target_link_libraries(v2_unit_cuda_flash_attention_decode_kvlen llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAFlashAttention_DecodeKVLen
    COMMAND $<TARGET_FILE:v2_unit_cuda_flash_attention_decode_kvlen>
    LABELS "V2;Integration;CUDA;Attention;FlashAttention;FlashDecode;DecodeMode;KVCache;BugFix"
    MPI_PROCS 1  # Tests GPU attention locally, no MPI needed
)

# Test: CUDA Quantized GEMM workspace declarations
# Host-only structural regressions for workspace-owned CUDA GEMM scratch.
add_executable(v2_unit_cuda_quantised_gemm_workspace unit/kernels/cuda/Test__CUDAQuantisedGemmKernel_Workspace.cpp)
target_link_libraries(v2_unit_cuda_quantised_gemm_workspace llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CUDAQuantisedGemmWorkspace
    COMMAND $<TARGET_FILE:v2_unit_cuda_quantised_gemm_workspace>
    LABELS "V2;Unit;CUDA;GEMM;Workspace;GraphCapture;Regression"
    MPI_PROCS 1
)

# Integration Test: cuBLAS GEMM (Phase 1 - dense GEMM via cuBLAS for FP32/FP16/BF16)
# Tests correctness against CPU reference, various matrix sizes, transpose variants
add_executable(v2_test_cublas_gemm integration/kernels/cuda/Test__CuBLASGemm.cpp)
target_link_libraries(v2_test_cublas_gemm llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CuBLASGemm
    COMMAND $<TARGET_FILE:v2_test_cublas_gemm>
    LABELS "V2;Integration;CUDA;cuBLAS;GEMM;FP32;FP16;BF16;DenseGEMM;KernelCorrectness"
    MPI_PROCS 1  # Tests GPU GEMM locally, no MPI needed
)

# Integration Test: GEMM kernel device context integration (Phase 4)
# Tests CuBLASGemmKernel and HipBLASGemmKernel with device context vs own handle
# Verifies backward compatibility and result parity between construction modes
add_executable(v2_integration_gemm_kernel_context integration/kernels/Test__GemmKernelContextIntegration.cpp)
target_link_libraries(v2_integration_gemm_kernel_context llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_GemmKernelContextIntegration
    COMMAND $<TARGET_FILE:v2_integration_gemm_kernel_context>
    LABELS "V2;Integration;CUDA;ROCm;cuBLAS;hipBLAS;GEMM;DeviceContext;Phase4;GPUDeviceContextRefactor;BackwardCompatibility"
    MPI_PROCS 1  # Tests GPU GEMM locally, no MPI needed
)

# Test: hipBLAS GEMM (ROCm - dense GEMM via hipBLAS for FP32/FP16)
# Tests ROCmFloatingPointGemmKernel correctness against CPU reference, various matrix sizes
# MI50: 13.4 TFLOPS FP32, 26.5 TFLOPS FP16 (no native BF16)
add_executable(v2_test_hipblas_gemm unit/kernels/rocm/Test__ROCmFloatingPointGemmKernel.cpp)
target_link_libraries(v2_test_hipblas_gemm llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_HipBLASGemm
    COMMAND $<TARGET_FILE:v2_test_hipblas_gemm>
    LABELS "V2;Integration;ROCm;hipBLAS;GEMM;FP32;FP16;DenseGEMM;KernelCorrectness;MI50"
    MPI_PROCS 1  # Tests GPU GEMM locally, no MPI needed
)

# Test: CUDA Basic Pipeline (Phase 1 - end-to-end KernelFactory + tensor transfer)
# Tests KernelFactory dispatch to cuBLAS, tensor upload/download in GEMM context
add_executable(v2_integration_cuda_basic_pipeline integration/kernels/cuda/Test__CUDABasicPipeline.cpp)
target_link_libraries(v2_integration_cuda_basic_pipeline llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDABasicPipeline
    COMMAND $<TARGET_FILE:v2_integration_cuda_basic_pipeline>
    LABELS "V2;Integration;CUDA;Pipeline;KernelFactory;TensorTransfer;GEMM;FP32"
    MPI_PROCS 1  # Tests GPU pipeline locally, no MPI needed
)

# Test: Tensor Host/Device Memory Coherence (comprehensive state machine validation)
# Tests the two-flag coherence model (host_valid_, device_valid_) with full state
# transition coverage, edge cases, and efficiency verification.
# See: docs/v2/TENSOR_MEMORY_COHERENCE_DESIGN.md
add_executable(v2_integration_tensor_coherence integration/execution/coherence/Test__TensorCoherence.cpp)
target_link_libraries(v2_integration_tensor_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_TensorCoherence
    COMMAND $<TARGET_FILE:v2_integration_tensor_coherence>
    LABELS "V2;Integration;CUDA;TensorTransfer;Coherence;StateMachine;FP32;BF16;FP16;GPUMemory;DeviceManagement"
    MPI_PROCS 1  # Tests GPU coherence locally, no MPI needed
)

# Test: Host Release After GPU Upload (CUDA Pinned Memory Regression)
# Regression tests for the bug where release_host_weight_data() after ensureOnDevice()
# corrupted CUDA state because cudaHostUnregister() was not called before freeing
# pinned host memory. Tests: single/multi-tensor release, double release idempotency,
# interleaved upload-release cycles, large tensors, and post-release CUDA operations.
add_executable(v2_integration_host_release_after_gpu_upload integration/execution/coherence/Test__HostReleaseAfterGpuUpload.cpp)
target_link_libraries(v2_integration_host_release_after_gpu_upload llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_HostReleaseAfterGpuUpload
    COMMAND $<TARGET_FILE:v2_integration_host_release_after_gpu_upload>
    LABELS "V2;Integration;CUDA;PinnedMemory;HostRelease;Regression;Coherence;GPUMemory;DeviceManagement"
    MPI_PROCS 1  # Tests GPU pinned memory locally, no MPI needed
)

# Test: GPU-Resident Data Flow (Phase 1: verifies no unnecessary D2H transfers)
# Tests StageCoherence functions with MockBackend transfer tracking
# Verifies: cohereInputs, cohereOutputs, markOutputsDirty behavior
# See: docs/v2/GPU_RESIDENT_EXECUTION_PROJECT_PLAN.md
add_executable(v2_integration_gpu_resident_data_flow integration/execution/Test__GpuResidentDataFlow.cpp)
target_link_libraries(v2_integration_gpu_resident_data_flow llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_GpuResidentDataFlow
    COMMAND $<TARGET_FILE:v2_integration_gpu_resident_data_flow>
    LABELS "V2;Integration;StageCoherence;GpuResident;TransferTracking;MockBackend;FP32"
    MPI_PROCS 1  # Tests coherence logic locally, no MPI needed
)

# Test: Mapped Memory Coherence (Phase 3: verifies zero-copy snapshot mode)
# Tests FP32Tensor::createMapped() behavior for GPU↔CPU shared memory
# Verifies: gpu_data_ptr valid after creation, ensureOnHost is no-op, 
#           kernel writes visible on host, both valid flags stay true
# See: docs/v2/GPU_RESIDENT_EXECUTION_PROJECT_PLAN.md
add_executable(v2_integration_mapped_memory_coherence integration/execution/coherence/Test__MappedMemoryCoherence.cpp)
target_include_directories(v2_integration_mapped_memory_coherence PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_mapped_memory_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_MappedMemoryCoherence
    COMMAND $<TARGET_FILE:v2_integration_mapped_memory_coherence>
    LABELS "V2;Integration;MappedMemory;ZeroCopy;Coherence;Snapshot;FP32;GPUMemory;DeviceManagement"
    MPI_PROCS 1  # Tests mapped memory locally, no MPI needed
)

# Test: Stream/Coherence Integration (regression tests for TP stale event bugs)
# Tests real GPU event recording/waiting, stale event persistence through flags-only marking,
# and MockBackend event tracking verification. Requires at least one GPU (CUDA or ROCm).
# Skips automatically on CPU-only machines.
add_executable(v2_integration_stream_coherence integration/execution/coherence/Test__StreamCoherenceIntegration.cpp)
target_include_directories(v2_integration_stream_coherence PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_stream_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_StreamCoherence
    COMMAND $<TARGET_FILE:v2_integration_stream_coherence>
    LABELS "V2;Integration;Coherence;StreamCoherence;StaleEvent;EventTracking;MockBackend;GPUMemory"
    MPI_PROCS 1  # Tests coherence locally, no MPI needed
)

# Performance: Coherence Transfer Overhead Benchmark (Phase 4: Performance Validation)
# Measures transfer overhead for different tensor allocation strategies:
# - ensureOnDevice (baseline with H2D upload)
# - allocateOnDevice (optimized, no H2D upload)
# - mapped memory (zero-copy)
# Validates GPU-resident execution optimization
# See: docs/v2/GPU_RESIDENT_EXECUTION_PROJECT_PLAN.md
add_executable(v2_perf_coherence_transfer_overhead performance/Test__CoherenceTransferOverhead.cpp)
target_include_directories(v2_perf_coherence_transfer_overhead PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_perf_coherence_transfer_overhead llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_CoherenceTransferOverhead
    COMMAND $<TARGET_FILE:v2_perf_coherence_transfer_overhead>
    LABELS "V2;Performance;Coherence;TransferOverhead;H2D;DeviceManagement;GPUMemory;ZeroCopy"
    MPI_PROCS 1  # Benchmarks run locally, no MPI needed
)

# Prefix-cache / MTP benchmark JSON smoke.
# Keeps Phase 14 machine-readable benchmark acceptance counters covered without
# requiring large model files in the performance suite.
add_executable(v2_perf_prefix_cache_mtp_benchmark performance/Perf__PrefixCacheMTPBenchmark.cpp)
target_include_directories(v2_perf_prefix_cache_mtp_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor
)
target_link_libraries(v2_perf_prefix_cache_mtp_benchmark
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_PrefixCacheMTP
    COMMAND $<TARGET_FILE:v2_perf_prefix_cache_mtp_benchmark>
    LABELS "V2;Performance;Benchmark;PrefixCache;MTP;JSON;Smoke"
    MPI_PROCS 1
)

# MoE graph-native overlay benchmark harness smoke.
# Validates curated overlay configs and the analyzer script without requiring
# large model files in the performance suite.
add_executable(v2_perf_moe_graph_native_overlay
    performance/moe/Perf__MoEGraphNativeOverlayHarness.cpp
    performance/moe/Perf__MoEGraphNativeOverlayAnalysis.cpp
)
target_link_libraries(v2_perf_moe_graph_native_overlay
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_MoEGraphNativeOverlay
    COMMAND $<TARGET_FILE:v2_perf_moe_graph_native_overlay>
    LABELS "V2;Performance;Benchmark;MoE;Overlay;ExpertParallel;GraphNative;Smoke"
    MPI_PROCS 1
)

# Focused MoE MTP verifier-prefill speedometer.
# Times CUDA/ROCm grouped verifier prefill for M=2/3/4 routed and shared expert
# cases, with parity against decode-equivalent rows before reporting CSV rows.
add_executable(v2_perf_moe_verifier_prefill
    performance/kernels/moe/Perf__MoEVerifierPrefill.cpp
    performance/kernels/moe/Perf__ROCmMoEVerifierPrefill.cpp
)
target_link_libraries(v2_perf_moe_verifier_prefill
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
if(HAVE_CUDA)
    target_link_libraries(v2_perf_moe_verifier_prefill CUDA::cudart)
endif()
add_v2_test(V2_Perf_MoEVerifierPrefill
    COMMAND $<TARGET_FILE:v2_perf_moe_verifier_prefill>
    LABELS "V2;Performance;Benchmark;MoE;MTP;Verifier;CUDA;ROCm;GraphCapture"
    MPI_PROCS 1
)

# Focused GPU speculative-summary speedometer.
# Isolates the Qwen3.6-sized verifier argmax + compact greedy summary path from
# full-model graph stream handoff timing, so MoE MTP tuning can tell whether a
# backend primitive is slow or merely waiting on earlier queued verifier work.
add_executable(v2_perf_gpu_speculative_summary
    performance/kernels/Perf__GPUSpeculativeSummary.cpp
)
target_link_libraries(v2_perf_gpu_speculative_summary
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
if(HAVE_CUDA)
    target_link_libraries(v2_perf_gpu_speculative_summary CUDA::cudart)
endif()
add_v2_test(V2_Perf_GPUSpeculativeSummary
    COMMAND $<TARGET_FILE:v2_perf_gpu_speculative_summary>
    LABELS "V2;Performance;Benchmark;MTP;Sampling;CUDA;ROCm;GraphCapture"
    MPI_PROCS 1
)

# HostBackend Cross-Vendor Copy Benchmark
# =============================================================================
add_executable(v2_perf_host_backend_cross_vendor performance/Perf__HostBackendCrossVendorCopy.cpp)
target_include_directories(v2_perf_host_backend_cross_vendor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_perf_host_backend_cross_vendor llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_HostBackendCrossVendorCopy
    COMMAND $<TARGET_FILE:v2_perf_host_backend_cross_vendor>
    LABELS "V2;Performance;HostBackend;CrossVendor;PCIe;Bandwidth"
    MPI_PROCS 1
)

# CUDA Event Synchronization Tests
# =============================================================================
# Tests for CUDA event-based synchronization in the backend.
# Validates that waitForEvent() uses cudaEventSynchronize() instead of
# cudaDeviceSynchronize(), which is essential for snapshot capture performance.

add_executable(v2_integration_cuda_event_synchronization
    integration/backends/cuda/Test__CUDAEventSynchronization.cpp
)
target_link_libraries(v2_integration_cuda_event_synchronization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_CUDAEventSynchronization
    COMMAND $<TARGET_FILE:v2_integration_cuda_event_synchronization>
    LABELS "V2;Integration;CUDA;Backend;EventSync;Coherence;Performance"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA Event Synchronization tests")

# Test: CUDA GEMM Parity (FP32 and IQ4_NL: CUDA vs CPU kernel comparison)
# Tests CUDAFloatingPointGemmKernel and CUDAQuantisedGemmKernel through KernelFactory
# with cosine similarity metrics against CPU reference kernels
add_executable(v2_integration_cuda_gemm_parity integration/kernels/cuda/Test__CUDAGemmParity.cpp)
target_link_libraries(v2_integration_cuda_gemm_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAGemmParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_gemm_parity>
    LABELS "V2;Integration;CUDA;GEMM;Parity;FP32;IQ4_NL;Quantized;KernelFactory;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# Test: CUDA GEMM Non-Determinism (self-consistency across repeated calls)
# Exercises multiply_fused_tensor() N times with same input and checks output stability.
# Reproduces split-K atomicAdd non-determinism seen in PP parity tests.
add_executable(v2_integration_cuda_gemm_nondet integration/kernels/cuda/Test__CUDAGemmNonDeterminism.cpp)
target_link_libraries(v2_integration_cuda_gemm_nondet llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAGemmNonDeterminism
    COMMAND $<TARGET_FILE:v2_integration_cuda_gemm_nondet>
    LABELS "V2;Integration;CUDA;GEMM;NonDeterminism;Quantized;SplitK"
    MPI_PROCS 1
)

if(HAVE_CUDA)
add_executable(v2_integration_cuda_moe_kernel integration/kernels/cuda/Test__CUDAMoEKernel.cpp)
target_link_libraries(v2_integration_cuda_moe_kernel llaminar2_core GTest::gtest GTest::gtest_main CUDA::cudart)
add_v2_test(V2_Integration_CUDAMoEKernel
    COMMAND $<TARGET_FILE:v2_integration_cuda_moe_kernel>
    LABELS "V2;Integration;CUDA;MoE;Routing;Gather;Scatter;SwiGLU;KernelFactory"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA MoE kernel integration tests")
endif()

# Test: CUDA Ops Parity (RMSNorm, SwiGLU, RoPE)
# Tests CUDA kernel implementations of normalization and activation functions
# against CPU reference kernels with cosine similarity metrics
add_executable(v2_integration_cuda_ops_parity integration/kernels/cuda/ops/Test__CUDAOpsParity.cpp)
target_link_libraries(v2_integration_cuda_ops_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAOpsParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_ops_parity>
    LABELS "V2;Integration;CUDA;Parity;RMSNorm;SwiGLU;RoPE;FP32;BF16;FP16;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# Test: CUDA ResidualAdd Parity
# Tests CUDA ResidualAdd kernels (FP32, BF16, FP16) against CPU reference
# with cosine similarity and L2 error metrics
add_executable(v2_integration_cuda_residual_add_parity integration/kernels/cuda/ops/Test__CUDAResidualAddParity.cpp)
target_link_libraries(v2_integration_cuda_residual_add_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAResidualAddParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_residual_add_parity>
    LABELS "V2;Integration;CUDA;Parity;ResidualAdd;FP32;BF16;FP16;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# Test: CUDA HiddenStateRowSelectStage graph capture/replay
# Verifies bucketed-prefill LM-head row selection can update real length across graph replays.
add_executable(v2_integration_cuda_hidden_state_row_select_stage integration/kernels/cuda/ops/Test__CUDAHiddenStateRowSelectStage.cpp)
target_link_libraries(v2_integration_cuda_hidden_state_row_select_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAHiddenStateRowSelectStage
    COMMAND $<TARGET_FILE:v2_integration_cuda_hidden_state_row_select_stage>
    LABELS "V2;Integration;CUDA;GraphCapture;RowSelect;LMHead;Phase6;FP32"
    MPI_PROCS 1
)

if(HAVE_CUDA)
# Test: CUDA RoPE graph-capture dynamic metadata
# Verifies contiguous RoPE replay uses pre-uploaded device params and explicit
# non-contiguous position IDs fail before H2D during graph capture.
add_executable(v2_integration_cuda_rope_graph_capture_no_h2d integration/kernels/Test__RoPEGraphCaptureNoH2D.cpp)
target_compile_definitions(v2_integration_cuda_rope_graph_capture_no_h2d PRIVATE GPU_CONTEXT_TEST_BACKEND_CUDA=1)
target_include_directories(v2_integration_cuda_rope_graph_capture_no_h2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_integration_cuda_rope_graph_capture_no_h2d llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARoPEGraphCaptureNoH2D
    COMMAND $<TARGET_FILE:v2_integration_cuda_rope_graph_capture_no_h2d>
    LABELS "V2;Integration;CUDA;GraphCapture;RoPE;NoH2D;FP32"
    MPI_PROCS 1
)
endif()

# Test: CUDA Flash Attention Parity
# Tests Flash Attention 2 (prefill) and Flash Decoding (decode) CUDA kernels
# against CPU attention reference with cosine similarity metrics
add_executable(v2_integration_cuda_flash_attention_parity integration/kernels/cuda/attention/Test__CUDAFlashAttentionParity.cpp)
target_link_libraries(v2_integration_cuda_flash_attention_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAFlashAttentionParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_flash_attention_parity>
    LABELS "V2;Integration;CUDA;Parity;Attention;FlashAttention;FlashDecode;GQA;FP32;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# Test: CUDA attention padding parity for Phase 6 prefill bucket safety
# Verifies causal real-row outputs are unchanged by hostile bucket-tail padding
add_executable(v2_integration_cuda_attention_padding_parity integration/kernels/cuda/attention/Test__CUDAAttentionPaddingParity.cpp)
target_link_libraries(v2_integration_cuda_attention_padding_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAAttentionPaddingParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_attention_padding_parity>
    LABELS "V2;Integration;CUDA;Attention;Padding;CausalMask;Phase6;Parity;FP32"
    MPI_PROCS 1
)

if(HAVE_CUDA)
# Test: CUDA padded GDN real-length semantics
# Verifies effective-length recurrence and short-conv prefill preserve decode state.
add_executable(v2_integration_cuda_gdn_padded_real_length integration/kernels/cuda/Test__CUDAGDNPaddedRealLength.cpp)
target_link_libraries(v2_integration_cuda_gdn_padded_real_length llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAGDNPaddedRealLength
    COMMAND $<TARGET_FILE:v2_integration_cuda_gdn_padded_real_length>
    LABELS "V2;Integration;CUDA;GatedDeltaNet;ShortConv;Padding;Phase6;RealLength;FP32"
    MPI_PROCS 1
)
endif()

# Performance: CUDA Flash Attention Throughput Benchmarks
# Tests FA3 pipelined, WMMA, and scalar prefill + decode kernels
# Measures TFLOPS, GB/s, latency across Qwen model configurations
add_executable(v2_perf_cuda_flash_attention performance/kernels/cuda/attention/Perf__CUDAFlashAttention.cpp)
target_link_libraries(v2_perf_cuda_flash_attention llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_CUDAFlashAttention
    COMMAND $<TARGET_FILE:v2_perf_cuda_flash_attention>
    LABELS "V2;Performance;CUDA;Attention;FlashAttention;FA3;WMMA;TensorCore;Throughput;Latency"
    MPI_PROCS 1  # Tests GPU attention locally, no MPI needed
)

add_executable(v2_perf_cuda_blockwise_tensorcore_gemm performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemm.cpp)
target_link_libraries(v2_perf_cuda_blockwise_tensorcore_gemm llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_CUDABlockwiseTensorCoreGemm
    COMMAND $<TARGET_FILE:v2_perf_cuda_blockwise_tensorcore_gemm>
    LABELS "V2;Performance;CUDA;GEMM;GEMV;TensorCore;BlockwiseQuantization;Qwen;Tuning"
    MPI_PROCS 1
)

add_executable(v2_perf_cuda_blockwise_tensorcore_gemm_sweep performance/kernels/cuda/gemm/Perf__CUDABlockwiseTensorCoreGemmSweep.cpp)
target_link_libraries(v2_perf_cuda_blockwise_tensorcore_gemm_sweep llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_CUDABlockwiseTensorCoreGemmSweep
    COMMAND $<TARGET_FILE:v2_perf_cuda_blockwise_tensorcore_gemm_sweep>
    LABELS "V2;Performance;CUDA;GEMV;Decode;NativeVNNI;Codebook;Qwen;Tuning;Sweep;Trainer"
    MPI_PROCS 1
)

add_executable(v2_perf_cuda_native_vnni_gemm performance/kernels/cuda/gemm/Perf__CUDANativeVNNIGemm.cpp)
target_link_libraries(v2_perf_cuda_native_vnni_gemm llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_CUDANativeVNNIGemm
    COMMAND $<TARGET_FILE:v2_perf_cuda_native_vnni_gemm>
    LABELS "V2;Performance;CUDA;GEMM;Prefill;NativeVNNI;Codebook;Qwen;Tuning;StreamK;Sweep"
    MPI_PROCS 1
)

# Test: CUDA Ring Buffer KV Cache Parity
# Tests ring buffer KV cache with O(1) eviction, wrap-around, batched gather, FP32/FP16/BF16
if(HAVE_CUDA)
add_executable(v2_integration_cuda_ring_kv_cache_parity integration/kernels/cuda/Test__CUDARingKVCacheParity.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCacheParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_parity>
    LABELS "V2;Integration;CUDA;Parity;KVCache;RingBuffer;Eviction;SlidingWindow;FP32;FP16;BF16"
    MPI_PROCS 1  # Tests GPU KV cache locally, no MPI needed
)

# Integration Test: CUDA Ring Buffer KV Cache Sharding (Tensor Parallelism)
# Tests sharded KV cache for tensor-parallel inference: per-rank metadata,
# local_n_kv_heads, kv_head_start offsets, memory allocation, IKVCache interface.
add_executable(v2_integration_cuda_ring_kv_cache_sharding integration/kernels/cuda/Test__CUDARingKVCache_Sharding.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_sharding llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCache_Sharding
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_sharding>
    LABELS "V2;Integration;CUDA;KVCache;RingBuffer;Sharding;TensorParallelism;FP32;FP16;BF16;GQA"
    MPI_PROCS 1  # Tests GPU KV cache sharding locally, no MPI needed
)
message(STATUS "V2 Tests: Added CUDA Ring KV Cache sharding integration tests")

# Test: CUDA Ring Buffer KV Cache with TurboQuant (TQ8-K / TQ4-V)
# Tests TurboQuant quantize/dequant roundtrip, ring wrap, RoPE-on-read, shadow invalidation
add_executable(v2_integration_cuda_ring_kv_cache_tq integration/kernels/cuda/Test__CUDARingKVCacheTQ.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_tq llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCacheTQ
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_tq>
    LABELS "V2;Integration;CUDA;KVCache;TurboQuant;TQ8;TQ4;RoPE;Quantization"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA TurboQuant KV Cache integration tests")

# Test: CUDA Ring Buffer KV Cache RoPE-on-read (FP16, Q8_1, FP32 caches)
# Tests get_kv_converted() with fused RoPE for existing cache precisions
add_executable(v2_integration_cuda_ring_kv_cache_rope integration/kernels/cuda/Test__CUDARingKVCache_RoPEOnRead.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_rope llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCache_RoPEOnRead
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_rope>
    LABELS "V2;Integration;CUDA;KVCache;RoPE;FP16;Q8_1;FP32"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA KV Cache RoPE-on-read integration tests")

# Comprehensive edge-case tests for CUDA Ring KV Cache
# Covers metadata accessors, eviction edge cases, single-token ops, multi-sequence, wrapping stress
add_executable(v2_integration_cuda_ring_kv_cache_comprehensive integration/kernels/cuda/Test__CUDARingKVCache_Comprehensive.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_comprehensive llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCache_Comprehensive
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_comprehensive>
    LABELS "V2;Integration;CUDA;KVCache;RingBuffer;Eviction;Metadata;EdgeCases;FP32;FP16;BF16"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA Ring KV Cache comprehensive edge-case tests")

# Integration Test: CUDA Ring Buffer KV Cache Logical Block IO
# Proves host-backed prefix payload export/import/truncate over real CUDA storage.
add_executable(v2_integration_cuda_ring_kv_cache_logical_block_io integration/kernels/cuda/Test__CUDARingKVCache_LogicalBlockIO.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_logical_block_io llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCache_LogicalBlockIO
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_logical_block_io>
    LABELS "V2;Integration;CUDA;KVCache;RingBuffer;PrefixCache;LogicalBlockIO;FP32;Q8_1"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA Ring KV Cache logical block IO integration tests")

# Integration Test: CUDA Hybrid KV Cache Reset
# Verifies GDN GPU kernel state and compressed FA metadata resets through
# standard clear()/clear_layer() semantics without recreating cache kernels.
add_executable(v2_integration_cuda_hybrid_kv_cache_reset integration/kernels/cuda/Test__CUDAHybridKVCacheReset.cpp)
target_link_libraries(v2_integration_cuda_hybrid_kv_cache_reset llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAHybridKVCacheReset
    COMMAND $<TARGET_FILE:v2_integration_cuda_hybrid_kv_cache_reset>
    LABELS "V2;Integration;CUDA;KVCache;Hybrid;GatedDeltaNet;Reset;Regression"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA Hybrid KV Cache reset regression tests")

# Test: NCCL Backend Integration
# Tests NCCL-based collective operations for multi-GPU communication
# Validates initialization, allreduce, allgather, broadcast, and clean shutdown
if(HAVE_NCCL)
add_executable(v2_integration_nccl_backend integration/collective/backends/Test__NCCLBackend.cpp)
target_link_libraries(v2_integration_nccl_backend 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_NCCLBackend
    COMMAND $<TARGET_FILE:v2_integration_nccl_backend>
    LABELS "V2;Integration;CUDA;NCCL;Collective;AllReduce;AllGather;Broadcast;MultiGPU"
    MPI_PROCS 1  # Tests NCCL locally - single rank
)
message(STATUS "V2 Tests: Added NCCL backend integration tests")
endif()

endif()

# Test: RCCL Backend Integration
# Tests RCCL-based collective operations for multi-GPU communication on AMD ROCm
# Validates initialization, allreduce, allgather, broadcast, and clean shutdown
# NOTE: RCCL is for ROCm (AMD GPUs), separate from NCCL (NVIDIA CUDA)
if(HAVE_RCCL)
add_executable(v2_integration_rccl_backend integration/collective/backends/Test__RCCLBackend.cpp)
target_link_libraries(v2_integration_rccl_backend 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_RCCLBackend
    COMMAND $<TARGET_FILE:v2_integration_rccl_backend>
    LABELS "V2;Integration;ROCm;RCCL;Collective;AllReduce;AllGather;Broadcast;MultiGPU"
    MPI_PROCS 1  # Tests RCCL locally - single rank
)
# RCCL/ROCm driver segfaults during static destructor teardown after all tests pass.
# Signal handlers can't help here since this binary uses gtest_main (no custom main).
if(TEST V2_Integration_RCCLBackend)
set_tests_properties(V2_Integration_RCCLBackend PROPERTIES
    PASS_REGULAR_EXPRESSION "\\[  PASSED  \\]"
)
endif()
message(STATUS "V2 Tests: Added RCCL backend integration tests")

# Test: RCCLAllreduceAccuracy - Focused numerical accuracy tests for RCCL allreduce
# Tests allreduce operations with inference-relevant sizes to isolate numerical issues
add_executable(v2_integration_rccl_allreduce_accuracy integration/collective/backends/Test__RCCLAllreduceAccuracy.cpp)
target_link_libraries(v2_integration_rccl_allreduce_accuracy 
    llaminar2_core 
    GTest::gtest 
)
add_v2_test(V2_Integration_RCCLAllreduceAccuracy
    COMMAND $<TARGET_FILE:v2_integration_rccl_allreduce_accuracy>
    LABELS "V2;Integration;ROCm;RCCL;Collective;AllReduce;Accuracy;Numerical;Inference"
    MPI_PROCS 1  # Tests RCCL locally - single rank
)
# RCCL driver has a known heap corruption bug ("double free or corruption")
# that intermittently triggers glibc abort() DURING test execution.
# glibc abort() forcibly resets SIGABRT to SIG_DFL before raising, so signal
# handlers cannot catch it. Use PASS_REGULAR_EXPRESSION as the only reliable
# workaround — CTest passes if GTest printed "[ PASSED ]" before the crash.
if(TEST V2_Integration_RCCLAllreduceAccuracy)
set_tests_properties(V2_Integration_RCCLAllreduceAccuracy PROPERTIES
    PASS_REGULAR_EXPRESSION "\\[  PASSED  \\]"
)
endif()
message(STATUS "V2 Tests: Added RCCL allreduce accuracy integration tests")
endif()

# Test: HeterogeneousBackend Phase 1 - Intra-Domain Reduce
# Tests parallel NCCL and RCCL reduce operations for heterogeneous GPU configurations
# Requires both CUDA and ROCm hardware
if(HAVE_CUDA AND HAVE_ROCM)
add_executable(v2_integration_heterogeneous_backend_phase1 integration/collective/backends/Test__HeterogeneousBackend_Phase1.cpp)
target_link_libraries(v2_integration_heterogeneous_backend_phase1 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_HeterogeneousBackend_Phase1
    COMMAND $<TARGET_FILE:v2_integration_heterogeneous_backend_phase1>
    LABELS "V2;Integration;CUDA;ROCm;Heterogeneous;Collective;Reduce;Phase1;CrossVendor"
    MPI_PROCS 1  # Single process with multiple GPUs
)
message(STATUS "V2 Tests: Added HeterogeneousBackend Phase 1 integration tests")

# Test: HeterogeneousBackend Phase 3 - Intra-Domain Broadcast
# Tests NCCL/RCCL broadcast from bridge devices to all domain devices
# Requires both CUDA and ROCm hardware
add_executable(v2_integration_heterogeneous_backend_phase3 integration/collective/backends/Test__HeterogeneousBackend_Phase3.cpp)
target_link_libraries(v2_integration_heterogeneous_backend_phase3 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_HeterogeneousBackend_Phase3
    COMMAND $<TARGET_FILE:v2_integration_heterogeneous_backend_phase3>
    LABELS "V2;Integration;CUDA;ROCm;Heterogeneous;Collective;Broadcast;Phase3;IntraDomain;NCCL;RCCL"
    MPI_PROCS 1  # Single process with multiple GPUs
)
message(STATUS "V2 Tests: Added HeterogeneousBackend Phase 3 integration tests")

# Test: HeterogeneousBackend Full AllReduce - 3-Phase AllReduce
# Tests complete 3-phase allreduce: intra-domain reduce, bridge exchange, intra-domain broadcast
# Requires both CUDA and ROCm hardware
add_executable(v2_integration_heterogeneous_backend_full_allreduce integration/collective/backends/Test__HeterogeneousBackend_FullAllReduce.cpp)
target_link_libraries(v2_integration_heterogeneous_backend_full_allreduce 
    llaminar2_core 
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_HeterogeneousBackend_FullAllReduce
    COMMAND $<TARGET_FILE:v2_integration_heterogeneous_backend_full_allreduce>
    LABELS "V2;Integration;CUDA;ROCm;Heterogeneous;Collective;AllReduce;FullPipeline;NCCL;RCCL"
    MPI_PROCS 1  # Single process with multiple GPUs
)
message(STATUS "V2 Tests: Added HeterogeneousBackend Full AllReduce integration tests")

endif()

# Test: CUDA Full Model Inference
# End-to-end GPU inference: model loading, forward pass, token prediction
# Tests GPU vs CPU parity for prefill and decode
add_executable(v2_integration_cuda_full_model_inference integration/kernels/cuda/Test__CUDAFullModelInference.cpp)
target_link_libraries(v2_integration_cuda_full_model_inference llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAFullModelInference
    COMMAND $<TARGET_FILE:v2_integration_cuda_full_model_inference>
    LABELS "V2;Integration;CUDA;Parity;FullModel;Pipeline;Prefill;Decode;E2E"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# =============================================================================
# PARITY TESTS (integration/parity/)
# =============================================================================
# Standardized parity tests comparing Llaminar backends against PyTorch ground truth.
# All tests use ParityTestBase for consistent metrics, visualization, and assertions.
#
# Test Organization:
#   - Test__CUDA_vs_PyTorch.cpp - CUDA pipeline vs PyTorch FP32 reference
#   - Test__CPU_vs_PyTorch.cpp  - CPU pipeline vs PyTorch FP32 reference
#   - Test__ROCm_vs_PyTorch.cpp - ROCm pipeline vs PyTorch FP32 reference
#
# Requirements:
#   - ZLIB (for cnpy .npy file loading)
#   - PyTorch snapshots generated by python/reference/generate_qwen_pipeline_snapshots.py

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    get_filename_component(WORKSPACE_ROOT_PARITY "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

    # Auto-clone cnpy if missing (small header + single .cpp for .npy file loading).
    # cnpy is .gitignored under external/ so fresh checkouts (and Docker builds that
    # COPY only src/+tests/) won't have it. Clone shallowly when needed.
    set(CNPY_DIR "${WORKSPACE_ROOT_PARITY}/external/cnpy")
    if(NOT EXISTS "${CNPY_DIR}/cnpy.cpp")
        find_package(Git REQUIRED)
        message(STATUS "V2 Tests: cnpy not found in external/, cloning https://github.com/rogersce/cnpy.git ...")
        if(NOT EXISTS "${WORKSPACE_ROOT_PARITY}/external")
            file(MAKE_DIRECTORY "${WORKSPACE_ROOT_PARITY}/external")
        endif()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone --depth 1 https://github.com/rogersce/cnpy.git cnpy
            WORKING_DIRECTORY "${WORKSPACE_ROOT_PARITY}/external"
            RESULT_VARIABLE CNPY_CLONE_RESULT
            ERROR_VARIABLE CNPY_CLONE_ERROR
        )
        if(NOT CNPY_CLONE_RESULT EQUAL 0)
            message(WARNING "V2 Tests: Failed to clone cnpy: ${CNPY_CLONE_ERROR}\n  Parity tests will be disabled.")
        endif()
    endif()

    if(NOT EXISTS "${CNPY_DIR}/cnpy.cpp")
        message(WARNING "V2 Tests: cnpy unavailable; skipping parity test executables")
    else()

    # =========================================================================
    # Qwen2 Parity Test Matrix (integration/parity/qwen2/)
    # =========================================================================
    # Split parity matrix: each category gets its own executable for faster
    # feedback, independent failure isolation, and cleaner CTest filtering.
    # =========================================================================

    # ---- Single-Device Parity (CPU, CUDA, ROCm) ----
    add_executable(v2_integration_parity_qwen2_single_device
        integration/parity/qwen2/Test__Qwen2_SingleDevice_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_single_device
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_single_device PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen2_single_device
        TEST_PREFIX V2_Integration_Parity_Qwen2_SingleDevice
        LABELS "V2;Integration;Parity;Qwen2;CPU;CUDA;ROCm;SingleDevice;PyTorch;GoldenReference;FullModel"
        MPI_PROCS 1
    )

    # ---- Local Tensor Parallelism (NCCL, RCCL, HOST) ----
    add_executable(v2_integration_parity_qwen2_local_tp
        integration/parity/qwen2/Test__Qwen2_LocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_local_tp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_local_tp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen2_local_tp
        TEST_PREFIX V2_Integration_Parity_Qwen2_LocalTP
        LABELS "V2;Integration;Parity;Qwen2;CUDA;ROCm;NCCL;RCCL;LOCAL;TensorParallel;PyTorch;GoldenReference;MultiDevice"
        MPI_PROCS 1
    )

    # ---- Local Pipeline Parallelism (NCCL, RCCL, HOST) ----
    add_executable(v2_integration_parity_qwen2_local_pp
        integration/parity/qwen2/Test__Qwen2_LocalPP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_local_pp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_local_pp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen2_local_pp
        TEST_PREFIX V2_Integration_Parity_Qwen2_LocalPP
        LABELS "V2;Integration;Parity;Qwen2;CUDA;ROCm;CPU;NCCL;RCCL;HOST;LOCAL;PipelineParallel;PyTorch;GoldenReference;MultiDevice"
        MPI_PROCS 1
    )

    # ---- Hybrid PP+TP (nested TP domains within PP stages) ----
    add_executable(v2_integration_parity_qwen2_hybrid_pptp
        integration/parity/qwen2/Test__Qwen2_HybridPPTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_hybrid_pptp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_hybrid_pptp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # Each test case runs in its own process to avoid ROCm CLR state
    # accumulation between RCCL init/destroy cycles (a known ROCm bug
    # that causes "Memobj map does not have ptr: 0x0" crashes when
    # multiple TP configs or test types run back-to-back in the same process).
    discover_v2_parity_tests(v2_integration_parity_qwen2_hybrid_pptp
        TEST_PREFIX V2_Integration_Parity_Qwen2_HybridPPTP
        LABELS "V2;Integration;Parity;Qwen2;CUDA;ROCm;CPU;NCCL;RCCL;LOCAL;TensorParallel;PipelineParallel;Hybrid;PyTorch;GoldenReference;MultiDevice"
        MPI_PROCS 1
    )

    # =========================================================================
    # Qwen3 Parity Test Matrix (integration/parity/qwen3/)
    # =========================================================================
    # Qwen3 architecture: per-head QK RMSNorm, no QKV biases.
    # Uses Qwen3-0.6B-Q8_0.gguf model.
    # =========================================================================

    # ---- Single-Device Parity (CPU, CUDA, ROCm) ----
    add_executable(v2_integration_parity_qwen3_single_device
        integration/parity/qwen3/Test__Qwen3_SingleDevice_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen3_single_device
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen3_single_device PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen3_single_device
        TEST_PREFIX V2_Integration_Parity_Qwen3_SingleDevice
        LABELS "V2;Integration;Parity;Qwen3;CPU;CUDA;ROCm;SingleDevice;PyTorch;GoldenReference;FullModel"
        MPI_PROCS 1
    )

    # ---- NodeLocalTP Parity (Multi-Socket CPU TP) ----
    add_executable(v2_integration_parity_qwen3_node_local_tp
        integration/parity/qwen3/Test__Qwen3_NodeLocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen3_node_local_tp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen3_node_local_tp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen3_node_local_tp
        TEST_PREFIX V2_Integration_Parity_Qwen3_NodeLocalTP
        LABELS "V2;Integration;Parity;Qwen3;NodeLocalTP;MPI;TensorParallel;UPI;CrossRank;CPU"
        MPI_PROCS 2
    )

    message(STATUS "V2: Qwen3 NodeLocalTP parity tests enabled (requires mpirun -np 2+)")

    # ---- NodeLocalPP Parity (Cross-Rank Pipeline Parallelism via GlobalOrchestrator) ----
    add_executable(v2_integration_parity_qwen3_node_local_pp
        integration/parity/qwen3/Test__Qwen3_NodeLocalPP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen3_node_local_pp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen3_node_local_pp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen3_node_local_pp
        TEST_PREFIX V2_Integration_Parity_Qwen3_NodeLocalPP
        LABELS "V2;Integration;Parity;Qwen3;NodeLocalPP;MPI;PipelineParallel;CrossRank;CPU;GlobalOrchestrator"
        MPI_PROCS 2
    )

    message(STATUS "V2: Qwen3 NodeLocalPP parity tests enabled (requires mpirun -np 2+)")

    # =========================================================================
    # Qwen3.5 Parity Test Matrix (integration/parity/qwen35/)
    # =========================================================================
    # Qwen3.5 architecture: hybrid GDN (Gated Delta Network) + full attention.
    # 75% GDN layers (conv1d + delta-rule recurrence) + 25% full attention (GQA).
    # Both layer types share: attention output gating, SwiGLU FFN, residuals.
    # Uses Qwen3.5-0.8B-Q4_0.gguf model.
    # =========================================================================

    # ---- Single-Device Parity (CPU, CUDA, ROCm) ----
    add_executable(v2_integration_parity_qwen35_single_device
        integration/parity/qwen35/Test__Qwen35_SingleDevice_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35_single_device
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35_single_device PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35_single_device
        TEST_PREFIX V2_Integration_Parity_Qwen35_SingleDevice
        LABELS "V2;Integration;Parity;Qwen35;GatedDeltaNet;CPU;CUDA;ROCm;SingleDevice;PyTorch;GoldenReference;FullModel"
        MPI_PROCS 1
    )

    # ---- Local Tensor Parallelism (HOST, NCCL, RCCL) ----
    add_executable(v2_integration_parity_qwen35_local_tp
        integration/parity/qwen35/Test__Qwen35_LocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35_local_tp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35_local_tp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35_local_tp
        TEST_PREFIX V2_Integration_Parity_Qwen35_LocalTP
        LABELS "V2;Integration;Parity;Qwen35;GatedDeltaNet;CUDA;ROCm;NCCL;RCCL;LOCAL;TensorParallel;PyTorch;GoldenReference;MultiDevice"
        MPI_PROCS 1
    )

    # ---- Local Pipeline Parallelism (HOST, NCCL, RCCL) ----
    add_executable(v2_integration_parity_qwen35_local_pp
        integration/parity/qwen35/Test__Qwen35_LocalPP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35_local_pp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35_local_pp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35_local_pp
        TEST_PREFIX V2_Integration_Parity_Qwen35_LocalPP
        LABELS "V2;Integration;Parity;Qwen35;GatedDeltaNet;CPU;CUDA;ROCm;NCCL;RCCL;HOST;LOCAL;PipelineParallel;PyTorch;GoldenReference;MultiDevice"
        MPI_PROCS 1
    )

    # =========================================================================
    # Test: LocalPP Layer-Partitioned Weights
    # Verifies PP stages only load their assigned layer weights (memory efficiency)
    # =========================================================================
    add_executable(v2_integration_localpp_layer_partitioned_weights
        integration/parity/Test__LocalPP_LayerPartitionedWeights.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_localpp_layer_partitioned_weights
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_localpp_layer_partitioned_weights PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
    )
    add_v2_test(V2_Integration_LocalPP_LayerPartitionedWeights
        COMMAND $<TARGET_FILE:v2_integration_localpp_layer_partitioned_weights>
        LABELS "V2;Integration;PipelineParallel;LayerPartitioned;Memory;WeightLoading"
        MPI_PROCS 1
    )

    # =========================================================================
    # Test: TopologyTreeParity (Phase 8 Global PP Integration)
    # Verifies ParallelismTree construction, parsing, and compilation
    # =========================================================================
    add_executable(v2_integration_topology_tree_parity
        integration/parity/topology/Test__TopologyTreeParity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_topology_tree_parity
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_topology_tree_parity PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
    )
    add_v2_test(V2_Integration_TopologyTreeParity
        COMMAND $<TARGET_FILE:v2_integration_topology_tree_parity>
        LABELS "V2;Integration;PipelineParallel;Topology;TreePP;ParallelismTree"
        MPI_PROCS 1
    )

    # NOTE: LocalPP Multi-Stage Parity and Unified PP vs Orchestrator tests removed
    # as part of Phase 6 (LocalPPOrchestrator sunset). LocalPP tests now use unified
    # graph approach via Qwen2ParityTestBase::setupLocalPPPipeline().
    # See: docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md

    message(STATUS "V2: Qwen2 parity tests enabled (SingleDevice, LocalTP, LocalPP, HybridPPTP)")
    message(STATUS "V2: LocalPP layer-partitioned weights test enabled")

    # =========================================================================
    # Test: Qwen3.5 NodeLocalTP Parity (Multi-Socket CPU TP)
    #
    # NodeLocalTP parity tests for Qwen3.5 GDN+FA hybrid architecture using MPI
    # for cross-rank tensor parallelism on the same physical node. CPU TP MUST
    # use NodeLocalTP because DeviceId::cpu() is a singleton — LocalTP cannot
    # distinguish CPU sockets.
    #
    # REQUIRES: mpirun -np 2 (or more) to run these tests
    #
    # Test configurations:
    #   - NodeLocalTP_2xMPI_CPU_08B: 2 MPI ranks, CPU, Qwen3.5-0.8B Q4_0
    #   - NodeLocalTP_2xMPI_CPU_4B:  2 MPI ranks, CPU, Qwen3.5-4B Q8_0
    # =========================================================================
    add_executable(v2_integration_parity_qwen35_node_local_tp
        integration/parity/qwen35/Test__Qwen35_NodeLocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    # Note: No gtest_main - test provides custom main() with MPI_Init
    target_link_libraries(v2_integration_parity_qwen35_node_local_tp
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35_node_local_tp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    discover_v2_parity_tests(v2_integration_parity_qwen35_node_local_tp
        TEST_PREFIX V2_Integration_Parity_Qwen35_NodeLocalTP
        LABELS "V2;Integration;Parity;Qwen35;GatedDeltaNet;NodeLocalTP;MPI;TensorParallel;UPI;CrossRank;CPU"
        MPI_PROCS 2
    )

    message(STATUS "V2: Qwen3.5 NodeLocalTP parity tests enabled (requires mpirun -np 2+)")

    # ---- NodeLocalPP Parity (Cross-Rank Pipeline Parallelism via GlobalOrchestrator) ----
    add_executable(v2_integration_parity_qwen35_node_local_pp
        integration/parity/qwen35/Test__Qwen35_NodeLocalPP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35_node_local_pp
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35_node_local_pp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35_node_local_pp
        TEST_PREFIX V2_Integration_Parity_Qwen35_NodeLocalPP
        LABELS "V2;Integration;Parity;Qwen35;GatedDeltaNet;NodeLocalPP;MPI;PipelineParallel;CrossRank;CPU;GlobalOrchestrator"
        MPI_PROCS 2
    )

    message(STATUS "V2: Qwen3.5 NodeLocalPP parity tests enabled (requires mpirun -np 2+)")

    # =========================================================================
    # Qwen3.6 Dense Prefix Cache / MTP PyTorch Parity Matrix
    # =========================================================================
    # Dense Qwen3.6 GGUFs are encoded as qwen35 with trailing nextn MTP sidecar
    # tensors, so this matrix uses the Qwen3.5 PyTorch reference generator.
    # The harness regenerates metadata-only PyTorch references when absent.
    # =========================================================================

    add_executable(v2_integration_parity_qwen36_single_device_math
        integration/parity/qwen36/Test__Qwen36_SingleDevice_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_single_device_math
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_single_device_math PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set(QWEN36_DENSE_SINGLE_DEVICE_PARITY_LABELS
        V2 Integration Parity Qwen36 Dense GatedDeltaNet
        CPU CUDA ROCm SingleDevice PyTorch GoldenReference FullModel
    )
    set(QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASES
        PrefillParity CPU  Qwen36Dense_27B_CPU_KV_FP16
        PrefillParity CUDA Qwen36Dense_27B_CUDA_KV_FP16
        PrefillParity ROCm Qwen36Dense_27B_ROCm_KV_FP16
        DecodeParity  CPU  Qwen36Dense_27B_CPU_KV_FP16
        DecodeParity  CUDA Qwen36Dense_27B_CUDA_KV_FP16
        DecodeParity  ROCm Qwen36Dense_27B_ROCm_KV_FP16
        SnapshotInfrastructure CPU  Qwen36Dense_27B_CPU_KV_FP16
        SnapshotInfrastructure CUDA Qwen36Dense_27B_CUDA_KV_FP16
        SnapshotInfrastructure ROCm Qwen36Dense_27B_ROCm_KV_FP16
    )
    list(LENGTH QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASES QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASE_COUNT)
    math(EXPR QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASE_LAST "${QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASE_COUNT} - 1")
    foreach(CASE_INDEX RANGE 0 ${QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASE_LAST} 3)
        math(EXPR BACKEND_INDEX "${CASE_INDEX} + 1")
        math(EXPR CONFIG_INDEX "${CASE_INDEX} + 2")
        list(GET QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASES ${CASE_INDEX} CASE_NAME)
        list(GET QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASES ${BACKEND_INDEX} CASE_BACKEND)
        list(GET QWEN36_DENSE_SINGLE_DEVICE_PARITY_CASES ${CONFIG_INDEX} CASE_CONFIG)
        add_v2_test(V2_Integration_Parity_Qwen36_SingleDevice_Math_${CASE_NAME}_${CASE_BACKEND}
            COMMAND $<TARGET_FILE:v2_integration_parity_qwen36_single_device_math>
                    --gtest_filter=Qwen36Dense/Qwen36DenseSingleDeviceParityTest.${CASE_NAME}/${CASE_CONFIG}
            LABELS ${QWEN36_DENSE_SINGLE_DEVICE_PARITY_LABELS}
            MPI_PROCS 1
            TIMEOUT 3600
        )
    endforeach()

    add_executable(v2_integration_parity_qwen36_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;ROCm;GPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 1800
    )
    add_v2_test(V2_Integration_Parity_Qwen36_SingleDevice_MTPGreedyDepth3GraphStreamStress
        COMMAND $<TARGET_FILE:v2_integration_parity_qwen36_single_device_prefix_mtp>
                --gtest_filter=Qwen36SingleDevicePrefixMTPParity.MTPGreedyDepth3MatchesPyTorchDecodeTokens
                --gtest_repeat=3
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;ROCm;GPU;SingleDevice;PyTorch;GraphCapture;Regression"
        MPI_PROCS 1
        TIMEOUT 1800
    )
    add_executable(v2_integration_parity_qwen36_cpu_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36_CPU_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_cpu_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_cpu_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_cpu_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36_CPU_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;CPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 2400
    )

    add_executable(v2_integration_parity_qwen36_rocm_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36_ROCm_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_rocm_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_rocm_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_rocm_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36_ROCm_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;ROCm;GPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 1800
    )
    add_v2_test(V2_Integration_Parity_Qwen36_ROCm_SingleDevice_MTPGreedyDepth3GraphStreamStress
        COMMAND $<TARGET_FILE:v2_integration_parity_qwen36_rocm_single_device_prefix_mtp>
                --gtest_filter=Qwen36ROCmSingleDevicePrefixMTPParity.MTPGreedyDepth3MatchesPyTorchDecodeTokens
                --gtest_repeat=3
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;ROCm;GPU;SingleDevice;PyTorch;GraphCapture;Regression"
        MPI_PROCS 1
        TIMEOUT 1800
    )
    # add_v2_test skips integration tests in Release/perf-only builds, so the
    # matching TEST property must be guarded by the same condition.
    if(NOT V2_PERF_TESTS_ONLY)
        set_property(TEST V2_Integration_Parity_Qwen36_ROCm_SingleDevice_MTPGreedyDepth3GraphStreamStress
            APPEND PROPERTY ENVIRONMENT
            "LLAMINAR_GPU_GRAPHS=1"
            "LLAMINAR_ROCM_CONCURRENT_DECODE=0"
            "LLAMINAR_ROCM_CONCURRENT_M2_ROWS=0"
        )
    endif()

    add_executable(v2_integration_parity_qwen36_cuda_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36_CUDA_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_cuda_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_cuda_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_cuda_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36_CUDA_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;CUDA;GPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 1800
    )
    add_v2_test(V2_Integration_Parity_Qwen36_CUDA_SingleDevice_MTPGreedyDepth3GraphStreamStress
        COMMAND $<TARGET_FILE:v2_integration_parity_qwen36_cuda_single_device_prefix_mtp>
                --gtest_filter=Qwen36CUDASingleDevicePrefixMTPParity.MTPGreedyDepth3MatchesPyTorchDecodeTokens
                --gtest_repeat=3
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;CUDA;GPU;SingleDevice;PyTorch;GraphCapture;Regression"
        MPI_PROCS 1
        TIMEOUT 1800
    )
    # add_v2_test skips integration tests in Release/perf-only builds, so the
    # matching TEST property must be guarded by the same condition.
    if(NOT V2_PERF_TESTS_ONLY)
        set_property(TEST V2_Integration_Parity_Qwen36_CUDA_SingleDevice_MTPGreedyDepth3GraphStreamStress
            APPEND PROPERTY ENVIRONMENT
            "LLAMINAR_GPU_GRAPHS=1"
            "LLAMINAR_ROCM_CONCURRENT_DECODE=0"
            "LLAMINAR_ROCM_CONCURRENT_M2_ROWS=0"
        )
    endif()
    add_executable(v2_integration_parity_qwen36_local_tp_prefix_mtp
        integration/parity/qwen36/Test__Qwen36_LocalTP_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_local_tp_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_local_tp_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_local_tp_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36_LocalTP
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;ROCm;GPU;LocalTP;RCCL;PyTorch"
        MPI_PROCS 1
        TIMEOUT 1800
    )

    add_executable(v2_integration_parity_qwen36_local_pp_prefix
        integration/parity/qwen36/Test__Qwen36_LocalPP_Prefix_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_local_pp_prefix
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_local_pp_prefix PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_local_pp_prefix
        TEST_PREFIX V2_Integration_Parity_Qwen36_LocalPP
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;RealModel;ROCm;GPU;LocalPP;PyTorch"
        MPI_PROCS 1
        TIMEOUT 1800
    )

    add_executable(v2_integration_parity_qwen36_node_local_tp_prefix
        integration/parity/qwen36/Test__Qwen36_NodeLocalTP_Prefix_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36_node_local_tp_prefix
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36_node_local_tp_prefix PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36_node_local_tp_prefix
        TEST_PREFIX V2_Integration_Parity_Qwen36_NodeLocalTP
        LABELS "V2;Integration;Parity;Qwen36;PrefixCache;MTP;RealModel;CPU;NodeLocalTP;MPI;PyTorch"
        MPI_PROCS 2
        TIMEOUT 2400
    )

    add_executable(v2_integration_parity_qwen36moe_cpu_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36MoE_CPU_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36moe_cpu_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36moe_cpu_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36moe_cpu_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36MoE_CPU_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36MoE;PrefixCache;MTP;RealModel;CPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 2400
    )

    add_executable(v2_integration_parity_qwen36moe_rocm_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36MoE_ROCm_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36moe_rocm_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36moe_rocm_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36moe_rocm_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36MoE_ROCm_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36MoE;PrefixCache;MTP;RealModel;ROCm;GPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 2400
    )

    add_executable(v2_integration_parity_qwen36moe_cuda_single_device_prefix_mtp
        integration/parity/qwen36/Test__Qwen36MoE_CUDA_SingleDevice_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36moe_cuda_single_device_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36moe_cuda_single_device_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36moe_cuda_single_device_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36MoE_CUDA_SingleDevice
        LABELS "V2;Integration;Parity;Qwen36MoE;PrefixCache;MTP;RealModel;CUDA;GPU;SingleDevice;PyTorch"
        MPI_PROCS 1
        TIMEOUT 2400
    )

    add_executable(v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp
        integration/parity/qwen36/Test__Qwen36MoE_ExpertOverlay_PrefixMTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp
        llaminar2_core GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp
        TEST_PREFIX V2_Integration_Parity_Qwen36MoE_ExpertOverlay
        LABELS "V2;Integration;Parity;Qwen36MoE;PrefixCache;MTP;RealModel;ExpertOverlay;SparseCollectives;ROCm;CPU;LocalTP;PyTorch"
        MPI_PROCS 1
        TIMEOUT 300
    )

    add_executable(v2_integration_parity_qwen36moe_single_device
        integration/parity/qwen36/Test__Qwen36MoE_SingleDevice_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen36moe_single_device
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen36moe_single_device PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set(QWEN36_MOE_SINGLE_DEVICE_PARITY_LABELS
        V2 Integration Parity Qwen36MoE MixtureOfExperts GatedDeltaNet
        CPU CUDA ROCm SingleDevice PyTorch GoldenReference FullModel
    )
    set(QWEN36_MOE_SINGLE_DEVICE_PARITY_CASES
        PrefillParity CPU  Qwen36MoE_35B_CPU_KV_FP16
        PrefillParity CUDA Qwen36MoE_35B_CUDA_KV_FP16
        PrefillParity ROCm Qwen36MoE_35B_ROCm_KV_FP16
        DecodeParity  CPU  Qwen36MoE_35B_CPU_KV_FP16
        DecodeParity  CUDA Qwen36MoE_35B_CUDA_KV_FP16
        DecodeParity  ROCm Qwen36MoE_35B_ROCm_KV_FP16
        SnapshotInfrastructure CPU  Qwen36MoE_35B_CPU_KV_FP16
        SnapshotInfrastructure CUDA Qwen36MoE_35B_CUDA_KV_FP16
        SnapshotInfrastructure ROCm Qwen36MoE_35B_ROCm_KV_FP16
    )
    list(LENGTH QWEN36_MOE_SINGLE_DEVICE_PARITY_CASES QWEN36_MOE_SINGLE_DEVICE_PARITY_CASE_COUNT)
    math(EXPR QWEN36_MOE_SINGLE_DEVICE_PARITY_CASE_LAST "${QWEN36_MOE_SINGLE_DEVICE_PARITY_CASE_COUNT} - 1")
    foreach(CASE_INDEX RANGE 0 ${QWEN36_MOE_SINGLE_DEVICE_PARITY_CASE_LAST} 3)
        math(EXPR BACKEND_INDEX "${CASE_INDEX} + 1")
        math(EXPR CONFIG_INDEX "${CASE_INDEX} + 2")
        list(GET QWEN36_MOE_SINGLE_DEVICE_PARITY_CASES ${CASE_INDEX} CASE_NAME)
        list(GET QWEN36_MOE_SINGLE_DEVICE_PARITY_CASES ${BACKEND_INDEX} CASE_BACKEND)
        list(GET QWEN36_MOE_SINGLE_DEVICE_PARITY_CASES ${CONFIG_INDEX} CASE_CONFIG)
        add_v2_test(V2_Integration_Parity_Qwen36MoE_SingleDevice_Math_${CASE_NAME}_${CASE_BACKEND}
            COMMAND $<TARGET_FILE:v2_integration_parity_qwen36moe_single_device>
                    --gtest_filter=Qwen36MoE/Qwen36MoESingleDeviceParityTest.${CASE_NAME}/${CASE_CONFIG}
            LABELS ${QWEN36_MOE_SINGLE_DEVICE_PARITY_LABELS}
            MPI_PROCS 1
            TIMEOUT 3600
        )
    endforeach()

    # =========================================================================
    # Test: Qwen3.5 MoE Parity (Mixture-of-Experts variant)
    #
    # Parity tests for Qwen3.5 MoE (35B-A3B) which replaces dense SwiGLU FFN
    # with SparseMoeBlock: router (256 experts, top-8) + shared expert.
    # Attention architecture (GDN + FA hybrid) is identical to dense Qwen3.5.
    #
    # Model: /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
    #   - Q4_K/Q5_K/Q6_K mixed quantization, 40 layers, 2048 hidden dim
    #   - 256 experts × 512 intermediate + 1 shared expert
    #
    # Validates:
    #   - MoE routing (softmax top-k expert selection)
    #   - Per-expert SwiGLU FFN (dequant from 3D packed tensors)
    #   - Shared expert + sigmoid gate
    #   - MoE combined output (routed + gated shared)
    #   - End-to-end token prediction quality
    # =========================================================================

    # ---- Single-Device MoE Parity (CPU, CUDA, ROCm) ----
    add_executable(v2_integration_parity_qwen35moe_single_device
        integration/parity/qwen35moe/Test__Qwen35MoE_SingleDevice_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_single_device
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_single_device PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35moe_single_device
        TEST_PREFIX V2_Integration_Parity_Qwen35MoE_SingleDevice
        LABELS "V2;Integration;Parity;Qwen35MoE;MixtureOfExperts;GatedDeltaNet;CPU;CUDA;ROCm;SingleDevice;PyTorch;GoldenReference;FullModel"
        MPI_PROCS 1
    )

    message(STATUS "V2: Qwen3.5 MoE SingleDevice parity tests enabled")

    # ---- Long-context MoE CPU-vs-ROCm numeric sweep ----
    add_executable(v2_integration_parity_qwen35moe_long_context
        integration/parity/qwen35moe/Test__Qwen35MoE_LongContext_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_long_context
        llaminar2_core v2_test_utils GTest::gtest ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_long_context PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35moe_long_context
        TEST_PREFIX V2_Integration_Parity_Qwen35MoE_LongContext
        LABELS "V2;Integration;Parity;Qwen35MoE;LongContext;CPU;ROCm;SingleDevice;FullModel"
        MPI_PROCS 1
        TIMEOUT 1800
    )

    message(STATUS "V2: Qwen3.5 MoE LongContext CPU-vs-ROCm parity sweep enabled")

    # ---- LocalTP MoE Parity (2x ROCm with Expert Parallelism + RCCL) ----
    add_executable(v2_integration_parity_qwen35moe_local_tp
        integration/parity/qwen35moe/Test__Qwen35MoE_LocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_local_tp
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_local_tp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    discover_v2_parity_tests(v2_integration_parity_qwen35moe_local_tp
        TEST_PREFIX V2_Integration_Parity_Qwen35MoE_LocalTP
        LABELS "V2;Integration;Parity;Qwen35MoE;MixtureOfExperts;LocalTP;RCCL;ROCm;TensorParallel;ExpertParallel"
        MPI_PROCS 1
    )

    message(STATUS "V2: Qwen3.5 MoE LocalTP parity tests enabled")

    # ---- Hybrid PP+TP MoE Parity (ROCm GPU cache + CPU routed expert domain) ----
    add_executable(v2_integration_parity_qwen35moe_hybrid_pptp
        integration/parity/qwen35moe/Test__Qwen35MoE_HybridPPTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_hybrid_pptp
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_hybrid_pptp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    discover_v2_parity_tests(v2_integration_parity_qwen35moe_hybrid_pptp
        TEST_PREFIX V2_Integration_Parity_Qwen35MoE_HybridPPTP
        LABELS "V2;Integration;Parity;Qwen35MoE;MixtureOfExperts;HybridPPTP;LocalPP;LocalTP;ROCm;CPU;ExpertMigration;GpuExpertCache"
        MPI_PROCS 2
    )

    message(STATUS "V2: Qwen3.5 MoE HybridPPTP parity tests enabled")

    # ---- Expert Overlay MoE Parity (same-layer hot/cold expert tiers) ----
    add_executable(v2_integration_parity_qwen35moe_expert_overlay
        integration/parity/qwen35moe/Test__Qwen35MoE_ExpertOverlay_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_expert_overlay
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_expert_overlay PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    discover_v2_parity_tests(v2_integration_parity_qwen35moe_expert_overlay
        TEST_PREFIX V2_Integration_Parity_Qwen35MoEExpertOverlay
        LABELS "V2;Integration;Parity;Qwen35MoE;ExpertOverlay;MixtureOfExperts;MPI;ROCm;CUDA;CPU;NodeLocalTP;ReplicatedExperts;ModelLight"
        MPI_PROCS 2
        TIMEOUT 600
    )

    message(STATUS "V2: Qwen3.5 MoE ExpertOverlay parity tests enabled (requires mpirun -np 2+)")

    # ---- Graph-native Expert Overlay parity (sparse collective tiers) ----
    add_executable(v2_integration_parity_qwen35moe_graphnative_rocm_hot_cpu_cold
        integration/parity/qwen35moe/Test__Qwen35MoE_GraphNative_RocmHotCpuCold_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_graphnative_rocm_hot_cpu_cold
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_graphnative_rocm_hot_cpu_cold PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35moe_graphnative_rocm_hot_cpu_cold
        TEST_PREFIX V2_Integration_Parity_Qwen35MoEGraphNative_RocmHotCpuCold
        LABELS "V2;Integration;Parity;Qwen35MoE;GraphNative;ExpertOverlay;SparseCollectives;ROCm;CPU;MPI"
        MPI_PROCS 2
        TIMEOUT 1200
    )

    add_executable(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm
        integration/parity/qwen35moe/Test__Qwen35MoE_GraphNative_CudaHotRocmWarm_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm
        TEST_PREFIX V2_Integration_Parity_Qwen35MoEGraphNative_CudaHotRocmWarm
        LABELS "V2;Integration;Parity;Qwen35MoE;GraphNative;ExpertOverlay;SparseCollectives;CUDA;ROCm;MPI"
        MPI_PROCS 2
        TIMEOUT 1200
    )

    add_executable(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm_cpu_cold
        integration/parity/qwen35moe/Test__Qwen35MoE_GraphNative_CudaHotRocmWarmCpuCold_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm_cpu_cold
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm_cpu_cold PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    discover_v2_parity_tests(v2_integration_parity_qwen35moe_graphnative_cuda_hot_rocm_warm_cpu_cold
        TEST_PREFIX V2_Integration_Parity_Qwen35MoEGraphNative_CudaHotRocmWarmCpuCold
        LABELS "V2;Integration;Parity;Qwen35MoE;GraphNative;ExpertOverlay;SparseCollectives;CUDA;ROCm;CPU;MPI"
        MPI_PROCS 3
        TIMEOUT 1800
    )

    message(STATUS "V2: Qwen3.5 MoE GraphNative parity tests enabled")

    # ---- NodeLocalTP MoE Parity (Cross-Socket CPU TP with Expert Parallelism) ----
    #
    # Tests Node-Local Tensor Parallelism for MoE models:
    #   - Expert Parallelism (EP): each rank computes a subset of experts
    #   - Shared Expert: Megatron TP (ColumnParallel gate/up, InputParallel down)
    #   - Single allreduce per layer on attn_proj
    #
    # REQUIRES: mpirun -np 2 (or more)
    add_executable(v2_integration_parity_qwen35moe_node_local_tp
        integration/parity/qwen35moe/Test__Qwen35MoE_NodeLocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen35moe_node_local_tp
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen35moe_node_local_tp PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    discover_v2_parity_tests(v2_integration_parity_qwen35moe_node_local_tp
        TEST_PREFIX V2_Integration_Parity_Qwen35MoE_NodeLocalTP
        LABELS "V2;Integration;Parity;Qwen35MoE;MixtureOfExperts;NodeLocalTP;MPI;TensorParallel;UPI;CrossRank;CPU;ExpertParallel"
        MPI_PROCS 2
    )

    message(STATUS "V2: Qwen3.5 MoE NodeLocalTP parity tests enabled (requires mpirun -np 2+)")

    # =========================================================================
    # Test: Qwen2 NodeLocalTP Parity (Multi-Socket CPU TP)
    #
    # NodeLocalTP parity tests using MPI for cross-rank tensor parallelism
    # on the same physical node. Unlike LocalTP (NCCL/RCCL/HOST),
    # NodeLocalTP uses MPI collectives for CPU sockets connected via UPI.
    #
    # REQUIRES: mpirun -np 2 (or more) to run these tests
    #
    # Test configurations:
    #   - NodeLocalTP_2xMPI_CPU: 2 MPI ranks, CPU devices, UPI/MPI backend
    #   - NodeLocalTP_4xMPI_CPU: 4 MPI ranks, CPU devices, UPI/MPI backend
    #
    # Validates:
    #   - GlobalTPContext initialization and collective operations
    #   - Cross-rank allreduce, broadcast, barrier
    #   - Prefill and decode parity with sharded weights
    # =========================================================================
    add_executable(v2_integration_node_local_tp_parity
        integration/parity/qwen2/Test__Qwen2_NodeLocalTP_Parity.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    # Note: No gtest_main - test provides custom main() with MPI_Init
    target_link_libraries(v2_integration_node_local_tp_parity
        llaminar2_core
        v2_test_utils
        GTest::gtest
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_node_local_tp_parity PRIVATE
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
        ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes
    )

    discover_v2_parity_tests(v2_integration_node_local_tp_parity
        TEST_PREFIX V2_Integration_Parity_NodeLocalTP
        LABELS "V2;Integration;Parity;Qwen2;NodeLocalTP;MPI;TensorParallel;UPI;CrossRank"
        MPI_PROCS 2
    )

    message(STATUS "V2: NodeLocalTP parity tests enabled (requires mpirun -np 2+)")

    endif() # cnpy available
else()
    message(STATUS "V2: Parity tests disabled (ZLIB not found)")
endif()

# Test: ComputeStage (device-agnostic kernel dispatch abstraction)
# Tests all compute stage types: RMSNorm, RoPE, Attention, SwiGLU, ResidualAdd, MoE stages
add_executable(v2_test_compute_stage unit/execution/compute_stages/Test__ComputeStage.cpp)
target_link_libraries(v2_test_compute_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeStage
    COMMAND $<TARGET_FILE:v2_test_compute_stage>
    LABELS "V2;Unit;Execution;ComputeStage;RMSNorm;RoPE;Attention;SwiGLU;MoE;KernelDispatch"
    MPI_PROCS 1  # Tests CPU stages locally, no MPI needed
)

# Test: GEMMStage (GEMM compute stage with workspace support)
# Tests construction, IWorkspaceConsumerStage delegation, params validation, metrics
add_executable(v2_test_gemm_stage unit/execution/compute_stages/stages/Test__GEMMStage.cpp)
target_link_libraries(v2_test_gemm_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GEMMStage
    COMMAND $<TARGET_FILE:v2_test_gemm_stage>
    LABELS "V2;Unit;Execution;ComputeStage;GEMM;Workspace;Validation"
    MPI_PROCS 1  # Tests stage construction and params, no MPI needed
)

# Test: MoEExpertDispatchStage (model-light expert-parallel routing descriptor builder)
add_executable(v2_test_moe_expert_dispatch_stage unit/execution/compute_stages/Test__MoEExpertDispatchStage.cpp)
target_include_directories(v2_test_moe_expert_dispatch_stage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_expert_dispatch_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoEExpertDispatchStage
    COMMAND $<TARGET_FILE:v2_test_moe_expert_dispatch_stage>
    LABELS "V2;Unit;Execution;ComputeStage;MoE;ExpertParallel;Dispatch"
    MPI_PROCS 1
)

# Test: MoEExpertParallelReduceStage (model-light dense partial reduce)
add_executable(v2_test_moe_expert_parallel_reduce_stage unit/execution/compute_stages/Test__MoEExpertParallelReduceStage.cpp)
target_include_directories(v2_test_moe_expert_parallel_reduce_stage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_expert_parallel_reduce_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoEExpertParallelReduceStage
    COMMAND $<TARGET_FILE:v2_test_moe_expert_parallel_reduce_stage>
    LABELS "V2;Unit;Execution;ComputeStage;MoE;ExpertParallel;Reduce"
    MPI_PROCS 1
)

# Test: ExecutionPolicy (declarative operation control)
# Tests factory methods, environment variable parsing, utility methods, comparison operators
add_executable(v2_test_execution_policy unit/execution/config/Test__ExecutionPolicy.cpp)
target_link_libraries(v2_test_execution_policy llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ExecutionPolicy
    COMMAND $<TARGET_FILE:v2_test_execution_policy>
    LABELS "V2;Unit;Execution;ExecutionPolicy;Configuration;EnvironmentVariables"
    MPI_PROCS 1
)

# Test: ResidualAddStage Precision Awareness
# Tests precision-aware residual addition: FP32, BF16, FP16, Q8_1
# Verifies Q8_1 uses native simd::q8_1_add_q8_1 (no explicit dequant)
add_executable(v2_test_residual_add_precision unit/execution/compute_stages/stages/Test__ResidualAddStage_Precision.cpp)
target_link_libraries(v2_test_residual_add_precision llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ResidualAddPrecision
    COMMAND $<TARGET_FILE:v2_test_residual_add_precision>
    LABELS "V2;Unit;Execution;ComputeStage;ResidualAdd;Precision;Q8_1;SIMD"
    MPI_PROCS 1
)

# Test: ComputeGraph Execution Flow
# Comprehensive tests for graph construction, dependency resolution, execution order,
# error handling, snapshot callbacks, statistics tracking, and multi-device execution
add_executable(v2_test_compute_graph_execution unit/execution/local_execution/graph/Test__ComputeGraphExecution.cpp)
target_link_libraries(v2_test_compute_graph_execution llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeGraphExecution
    COMMAND $<TARGET_FILE:v2_test_compute_graph_execution>
    LABELS "V2;Unit;Execution;ComputeGraph;TopologicalSort;DependencyResolution;SnapshotCallback;Statistics"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator Snapshot API
# Tests pipeline-compatible snapshot capture API (enableSnapshotCapture/getSnapshot/getSnapshotKeys)
# and stage name conversion for E2E test compatibility
add_executable(v2_test_device_graph_orchestrator_snapshots unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorSnapshots.cpp)
target_link_libraries(v2_test_device_graph_orchestrator_snapshots llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestratorSnapshots
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_snapshots>
    LABELS "V2;Unit;Execution;DeviceGraphOrchestrator;Snapshots;StageNameConversion"
    MPI_PROCS 1
)

# Test: LayerExecutor (layer-level execution orchestration)
# Tests compute graph construction, topological ordering, dependency tracking,
# attention/FFN/MoE graph building, and multi-device execution
add_executable(v2_test_layer_executor unit/execution/local_execution/model/Test__LayerExecutor.cpp)
target_link_libraries(v2_test_layer_executor llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LayerExecutor
    COMMAND $<TARGET_FILE:v2_test_layer_executor>
    LABELS "V2;Unit;Execution;LayerExecutor;ComputeGraph;Orchestration;Attention;FFN;MoE;MultiDevice"
    MPI_PROCS 1  # Tests execution orchestration locally
)

# Test: BufferRole (buffer role classification and descriptor types)
# Tests BufferRole enum, BufferDescriptor, StageBufferRequirements, and real-world usage patterns
# Part of Phase 1: DeviceGraphExecutor buffer management infrastructure
add_executable(v2_test_buffer_role unit/execution/debug/Test__BufferRole.cpp)
target_link_libraries(v2_test_buffer_role llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_BufferRole
    COMMAND $<TARGET_FILE:v2_test_buffer_role>
    LABELS "V2;Unit;Execution;BufferRole;BufferDescriptor;BufferManagement"
    MPI_PROCS 1
)

# Test: DomainAwareBufferManager (Phase 5.2: Domain-aware buffer allocation)
# Tests domain-aware buffer allocation based on layer placement configuration.
# GPU layers get GPU buffers, CPU layers get NUMA-local CPU buffers.
add_executable(v2_test_domain_aware_buffer_manager unit/execution/local_execution/device/Test__DomainAwareBufferManager.cpp)
target_link_libraries(v2_test_domain_aware_buffer_manager llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DomainAwareBufferManager
    COMMAND $<TARGET_FILE:v2_test_domain_aware_buffer_manager>
    LABELS "V2;Unit;Execution;DomainAwareBufferManager;BufferManagement;LayerPlacement;NUMA"
    MPI_PROCS 1
)

# Test: ComputeGraphMerge (graph merging, root/leaf detection)
# Tests merge(), getRootNodes(), getLeafNodes() for combining subgraphs
# Part of Phase 3: Declarative QwenStandardGraph refactoring
add_executable(v2_test_compute_graph_merge unit/execution/local_execution/graph/Test__ComputeGraphMerge.cpp)
target_link_libraries(v2_test_compute_graph_merge llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeGraphMerge
    COMMAND $<TARGET_FILE:v2_test_compute_graph_merge>
    LABELS "V2;Unit;Execution;ComputeGraph;GraphMerge;DAG"
    MPI_PROCS 1
)

# Test: IGraphBuilder interface and MockGraphBuilder
# Part of Phase 5: Declarative QwenStandardGraph refactoring - Interface extraction
add_executable(v2_test_igraph_builder unit/execution/local_execution/graph/Test__IGraphBuilder.cpp)
target_link_libraries(v2_test_igraph_builder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_IGraphBuilder
    COMMAND $<TARGET_FILE:v2_test_igraph_builder>
    LABELS "V2;Unit;Execution;IGraphBuilder;MockGraphBuilder;Interface"
    MPI_PROCS 1
)

# Test: TensorParallelConfig (proportional head assignment for heterogeneous TP)
# Tests equal splits, proportional splits (73%/27%), GQA alignment, d_ff alignment
# Part of Phase 1a: Heterogeneous Tensor Parallelism Infrastructure
add_executable(v2_test_tensor_parallel_config unit/config/Test__TensorParallelConfig.cpp)
target_link_libraries(v2_test_tensor_parallel_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TensorParallelConfig
    COMMAND $<TARGET_FILE:v2_test_tensor_parallel_config>
    LABELS "V2;Unit;Config;TensorParallelism;Heterogeneous;Sharding"
    MPI_PROCS 1
)

# Test: TensorParallelConfig_LocalTP (fromLocalTPContext factory method)
# Tests creation of TensorParallelConfig from ILocalTPContext for LOCAL TP weight sharding.
# Covers equal splits, proportional splits (73%/27%), 3-way splits, device lookup.
# Part of Phase 1: LOCAL TP Weight Sharding Infrastructure
add_executable(v2_test_tensor_parallel_config_localtp unit/config/Test__TensorParallelConfig_LocalTP.cpp)
target_link_libraries(v2_test_tensor_parallel_config_localtp llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TensorParallelConfig_LocalTP
    COMMAND $<TARGET_FILE:v2_test_tensor_parallel_config_localtp>
    LABELS "V2;Unit;Config;TensorParallelism;LocalTP;WeightSharding"
    MPI_PROCS 1
)

# Test: WeightManager_LocalTPSlicing (helper methods for device-aware weight slicing)
# Tests weight category detection (isQKVWeight, isFFNGateUpWeight, etc.) and sliceRowRange.
# Part of Phase 1: LOCAL TP Weight Sharding Infrastructure
add_executable(v2_test_weight_manager_localtp_slicing unit/loaders/Test__WeightManager_LocalTPSlicing.cpp)
target_link_libraries(v2_test_weight_manager_localtp_slicing llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_weight_manager_localtp_slicing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_v2_test(V2_Unit_WeightManager_LocalTPSlicing
    COMMAND $<TARGET_FILE:v2_test_weight_manager_localtp_slicing>
    LABELS "V2;Unit;Loaders;WeightManager;LocalTP;Slicing;WeightSharding"
    MPI_PROCS 1
)

# Test: PipelineParallelConfig (layer range assignment for pipeline parallelism)
# Tests equal splits, custom splits, topology methods, validation
# Part of Phase 2.3: Heterogeneous TP - Pipeline Parallelism Layer Assignment
add_executable(v2_test_pipeline_parallel_config unit/config/Test__PipelineParallelConfig.cpp)
target_link_libraries(v2_test_pipeline_parallel_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PipelineParallelConfig
    COMMAND $<TARGET_FILE:v2_test_pipeline_parallel_config>
    LABELS "V2;Unit;Config;PipelineParallelism;LayerAssignment;Topology"
    MPI_PROCS 1
)

# Test: LayerPlacementConfig (layer-to-device placement for CPU pipeline stages)
# Tests allOnDevice, cpuFirstLayers, cpuLastLayers, alternating, custom assignments,
# transition point detection, device inventory, and validation
# Part of Phase 3.1: CPU Pipeline Stage Support
add_executable(v2_test_layer_placement_config unit/config/Test__LayerPlacementConfig.cpp)
target_link_libraries(v2_test_layer_placement_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LayerPlacementConfig
    COMMAND $<TARGET_FILE:v2_test_layer_placement_config>
    LABELS "V2;Unit;Config;LayerPlacement;CPUOffload;HeterogeneousCompute"
    MPI_PROCS 1
)

# Test: LayerPlacementConfigExtended (extended methods for heterogeneous integration)
# Tests getDeviceTypeForLayer, isGPULayer/isCPULayer predicates, getGPULayers/getCPULayers,
# getDomainBoundaries for HeterogeneousLayerExecutor integration.
# Part of Phase 5.4: HeterogeneousLayerExecutor + LayerPlacementConfig Integration
add_executable(v2_test_layer_placement_config_extended unit/config/Test__LayerPlacementConfigExtended.cpp)
target_link_libraries(v2_test_layer_placement_config_extended llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LayerPlacementConfigExtended
    COMMAND $<TARGET_FILE:v2_test_layer_placement_config_extended>
    LABELS "V2;Unit;Config;LayerPlacement;HeterogeneousCompute"
    MPI_PROCS 1
)

# Test: TPDomain (multi-domain tensor parallelism)
# Tests TPDomain struct, MultiDomainTPConfig, domain type helpers, layer mapping
# Part of Phase 3.2b: TPDomain for Hybrid Parallelism
add_executable(v2_test_tpdomain unit/config/Test__TPDomain.cpp)
target_link_libraries(v2_test_tpdomain llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TPDomain
    COMMAND $<TARGET_FILE:v2_test_tpdomain>
    LABELS "V2;Unit;Config;TPDomain;TensorParallelism;HybridParallelism"
    MPI_PROCS 1
)

# Test: TPDomainConfig (TP domain configuration for unified PP graph architecture)
# Tests TPDomainConfig struct, accessors (primaryDevice, degree, isHomogeneous, etc.),
# validation, and factory method (createTPContext).
# Part of Phase 1.1: Unified PP Graph Architecture - TPDomainConfig data structure
add_executable(v2_test_tpdomain_config unit/config/Test__TPDomainConfig.cpp)
target_link_libraries(v2_test_tpdomain_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TPDomainConfig
    COMMAND $<TARGET_FILE:v2_test_tpdomain_config>
    LABELS "V2;Unit;Config;TPDomainConfig;TensorParallelism;UnifiedPPGraph"
    MPI_PROCS 1
)

# Test: BackendSelector (automatic backend selection for PP/TP based on device topology)
# Tests PP transfer selection (CUDA↔CUDA→NCCL, cross-vendor→PCIE_BAR, etc.),
# TP domain selection (homogeneous→NCCL/RCCL, mixed→HETEROGENEOUS, etc.),
# backend availability (isBackendCompiled, isBackendUsable), and utility methods
# (isHomogeneous, hasGPU, isCrossVendor, resolve).
add_executable(v2_test_backend_selector unit/config/Test__BackendSelector.cpp)
target_link_libraries(v2_test_backend_selector llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_BackendSelector
    COMMAND $<TARGET_FILE:v2_test_backend_selector>
    LABELS "V2;Unit;Config;BackendSelector;Heterogeneous;NCCL;RCCL;PCIE_BAR"
    MPI_PROCS 1
)

# Test: PPStageConfig (PP stage configuration for unified PP graph architecture)
# Tests PPStageConfig struct, accessors (numLayers, containsLayer, isFirstStage, isLastStage),
# validation, and factory methods (fullModel, firstStage, middleStage, lastStage).
# Part of Phase 1.2: Unified PP Graph Architecture - PPStageConfig data structure
add_executable(v2_test_ppstage_config unit/config/Test__PPStageConfig.cpp)
target_link_libraries(v2_test_ppstage_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PPStageConfig_UnifiedPP
    COMMAND $<TARGET_FILE:v2_test_ppstage_config>
    LABELS "V2;Unit;Config;PPStageConfig;PipelineParallelism;UnifiedPPGraph"
    MPI_PROCS 1
)

# Test: PipelineConfig (complete pipeline configuration for PP + TP composition)
# Tests PipelineConfig struct, lookup methods (getDomain, getStageForLayer, etc.),
# query methods (numStages, hasTP, hasPP, etc.), validation, and factory methods.
# Part of Phase 1.3: Unified PP Graph Architecture - PipelineConfig data structure
add_executable(v2_test_pipeline_config unit/config/Test__PipelineConfig.cpp)
target_link_libraries(v2_test_pipeline_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PipelineConfig
    COMMAND $<TARGET_FILE:v2_test_pipeline_config>
    LABELS "V2;Unit;Config;PipelineConfig;PipelineParallelism;TensorParallelism;UnifiedPPGraph"
    MPI_PROCS 1
)

# Test: GlobalDeviceAddress (fully-qualified device address across nodes)
# Tests factory methods, parsing (full/short forms), toString, predicates (isLocal, isGPU, etc.)
# Part of Phase 0.1: CLI and Configuration Architecture
add_executable(v2_test_global_device_address unit/backends/Test__GlobalDeviceAddress.cpp)
target_link_libraries(v2_test_global_device_address llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GlobalDeviceAddress
    COMMAND $<TARGET_FILE:v2_test_global_device_address>
    LABELS "V2;Unit;Backends;DeviceAddress;Heterogeneous;MPI"
    MPI_PROCS 1
)

# Test: DeviceRegistry (Phase 6 - centralized device discovery and management)
# Tests discovery of CPU/CUDA/ROCm devices, validation, memory queries, backend access, topology
add_executable(v2_test_device_registry unit/backends/Test__DeviceRegistry.cpp)
target_link_libraries(v2_test_device_registry llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceRegistry
    COMMAND $<TARGET_FILE:v2_test_device_registry>
    LABELS "V2;Unit;Backends;DeviceRegistry;Heterogeneous;DeviceManagement"
    MPI_PROCS 1
)

# Test: GPUDeviceContext - GPU device context infrastructure (pool, worker threads, streams/events)
# Tests IGPUDeviceContext interface, NvidiaDeviceContext, AMDDeviceContext, GPUDeviceContextPool singleton
add_v2_gpu_backend_test(
    v2_test_gpu_device_context
    unit/backends/Test__GPUDeviceContext.cpp
    V2_Integration_GPUDeviceContext
    V2 Integration Backends GPUDeviceContext WorkerThread Streams Events
)

# Test: DeviceAddressAdapter (Phase 6 - legacy-to-GlobalDeviceAddress bridge)
# Tests conversion between DeviceId, raw ordinals, and GlobalDeviceAddress
add_executable(v2_test_device_address_adapter unit/backends/Test__DeviceAddressAdapter.cpp)
target_link_libraries(v2_test_device_address_adapter llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceAddressAdapter
    COMMAND $<TARGET_FILE:v2_test_device_address_adapter>
    LABELS "V2;Unit;Backends;DeviceAddress;Migration;Heterogeneous"
    MPI_PROCS 1
)

# Test: OrchestrationConfig (configuration structs for heterogeneous orchestration)
# Tests enums, DomainDefinition, PPStageDefinition, OrchestrationConfig validation
# Part of Phase 0.2: CLI and Configuration Architecture
add_executable(v2_test_orchestration_config unit/config/Test__OrchestrationConfig.cpp)
target_link_libraries(v2_test_orchestration_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_OrchestrationConfig
    COMMAND $<TARGET_FILE:v2_test_orchestration_config>
    LABELS "V2;Unit;Config;Orchestration;Heterogeneous;DomainDefinition"
    MPI_PROCS 1
)

# Test: OrchestrationConfigParser (CLI and YAML parsing for orchestration config)
# Tests CLI parsing (--tp, --pp, --device, --define-domain, --pp-stage), YAML parsing
# Part of Phase 0.3: CLI and Configuration Architecture
add_executable(v2_test_orchestration_config_parser unit/config/Test__OrchestrationConfigParser.cpp)
target_link_libraries(v2_test_orchestration_config_parser llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_OrchestrationConfigParser
    COMMAND $<TARGET_FILE:v2_test_orchestration_config_parser>
    LABELS "V2;Unit;Config;CLI;YAML;Orchestration;Heterogeneous"
    MPI_PROCS 1
)

# Test: Prefix cache and MTP config feature gates
add_executable(v2_test_prefix_mtp_config unit/config/Test__PrefixMTPConfig.cpp)
target_link_libraries(v2_test_prefix_mtp_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixMTPConfig
    COMMAND $<TARGET_FILE:v2_test_prefix_mtp_config>
    LABELS "V2;Unit;Config;PrefixCache;MTP;CLI;YAML"
    MPI_PROCS 1
)

# Test: Prefix cache fingerprint key material
add_executable(v2_test_prefix_cache_fingerprint unit/prefix_cache/Test__PrefixCacheFingerprint.cpp)
target_link_libraries(v2_test_prefix_cache_fingerprint llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixCacheFingerprint
    COMMAND $<TARGET_FILE:v2_test_prefix_cache_fingerprint>
    LABELS "V2;Unit;PrefixCache;Fingerprint;MoE;MTP"
    MPI_PROCS 1
)

# Test: Prefix cache runtime state probe helpers
add_executable(v2_test_prefix_cache_state_probe unit/prefix_cache/Test__PrefixCacheStateProbe.cpp)
target_link_libraries(v2_test_prefix_cache_state_probe llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixCacheStateProbe
    COMMAND $<TARGET_FILE:v2_test_prefix_cache_state_probe>
    LABELS "V2;Unit;PrefixCache;StateProbe;KVCache"
    MPI_PROCS 1
)

# Test: Prefix lookup snapshot clamping
add_executable(v2_test_prefix_state_snapshot unit/prefix_cache/Test__PrefixStateSnapshot.cpp)
target_link_libraries(v2_test_prefix_state_snapshot llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixStateSnapshot
    COMMAND $<TARGET_FILE:v2_test_prefix_state_snapshot>
    LABELS "V2;Unit;PrefixCache;StateSnapshot;Hybrid"
    MPI_PROCS 1
)

# Test: Prefix cache participant coordination
add_executable(v2_test_prefix_cache_coordinator unit/prefix_cache/Test__PrefixCacheCoordinator.cpp)
target_link_libraries(v2_test_prefix_cache_coordinator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixCacheCoordinator
    COMMAND $<TARGET_FILE:v2_test_prefix_cache_coordinator>
    LABELS "V2;Unit;PrefixCache;Coordination;TP;PP;MoE"
    MPI_PROCS 1
)

# Test: Prefix cache participant coordination across MPI ranks
add_executable(v2_test_prefix_cache_coordinator_mpi
    unit/prefix_cache/Test__PrefixCacheCoordinatorMPI.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_test_prefix_cache_coordinator_mpi llaminar2_core GTest::gtest)
add_v2_test(V2_Unit_PrefixCacheCoordinatorMPI
    COMMAND $<TARGET_FILE:v2_test_prefix_cache_coordinator_mpi>
    LABELS "V2;Unit;PrefixCache;Coordination;MPI;GlobalTP;NodeLocalTP"
    MPI_PROCS 2
)

# Test: Prefix block hash helpers
add_executable(v2_test_prefix_block_hash unit/prefix_cache/Test__PrefixBlockHash.cpp)
target_link_libraries(v2_test_prefix_block_hash llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixBlockHash
    COMMAND $<TARGET_FILE:v2_test_prefix_block_hash>
    LABELS "V2;Unit;PrefixCache;Hash"
    MPI_PROCS 1
)

# Test: Prefix cache keys
add_executable(v2_test_prefix_cache_key unit/prefix_cache/Test__PrefixCacheKey.cpp)
target_link_libraries(v2_test_prefix_cache_key llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixCacheKey
    COMMAND $<TARGET_FILE:v2_test_prefix_cache_key>
    LABELS "V2;Unit;PrefixCache;Key"
    MPI_PROCS 1
)

# Test: Prefix cache RAM storage backend
add_executable(v2_test_ram_prefix_storage_backend unit/prefix_cache/Test__RamPrefixStorageBackend.cpp)
target_link_libraries(v2_test_ram_prefix_storage_backend llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RamPrefixStorageBackend
    COMMAND $<TARGET_FILE:v2_test_ram_prefix_storage_backend>
    LABELS "V2;Unit;PrefixCache;Storage;RAM"
    MPI_PROCS 1
)

# Test: Prefix state cache LRU accounting
add_executable(v2_test_prefix_state_cache_lru unit/prefix_cache/Test__PrefixStateCacheLRU.cpp)
target_link_libraries(v2_test_prefix_state_cache_lru llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixStateCacheLRU
    COMMAND $<TARGET_FILE:v2_test_prefix_state_cache_lru>
    LABELS "V2;Unit;PrefixCache;LRU;RAM"
    MPI_PROCS 1
)

# Test: Prefix cache disk storage backend
add_executable(v2_test_disk_prefix_storage_backend unit/prefix_cache/Test__DiskPrefixStorageBackend.cpp)
target_link_libraries(v2_test_disk_prefix_storage_backend llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DiskPrefixStorageBackend
    COMMAND $<TARGET_FILE:v2_test_disk_prefix_storage_backend>
    LABELS "V2;Unit;PrefixCache;Storage;Disk"
    MPI_PROCS 1
)

# Test: Prefix cache BufferArena staging ids
add_executable(v2_test_prefix_arena_staging unit/prefix_cache/Test__PrefixArenaStaging.cpp)
target_link_libraries(v2_test_prefix_arena_staging llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefixArenaStaging
    COMMAND $<TARGET_FILE:v2_test_prefix_arena_staging>
    LABELS "V2;Unit;PrefixCache;BufferArena"
    MPI_PROCS 1
)

# Test: Prefix cache OrchestrationRunner prefill control flow
add_executable(v2_test_prefix_cache_prefill_flow unit/prefix_cache/Test__PrefixCachePrefillFlow.cpp)
target_link_libraries(v2_test_prefix_cache_prefill_flow
    llaminar2_core
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)
add_v2_test(V2_Unit_PrefixCachePrefillFlow
    COMMAND $<TARGET_FILE:v2_test_prefix_cache_prefill_flow>
    LABELS "V2;Unit;PrefixCache;OrchestrationRunner;Prefill"
    MPI_PROCS 1
)

# Test: MTP adaptive draft-depth policy
add_executable(v2_test_mtp_depth_controller unit/mtp/Test__MTPDepthController.cpp)
target_link_libraries(v2_test_mtp_depth_controller llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_mtp_depth_controller PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPDepthController
    COMMAND $<TARGET_FILE:v2_test_mtp_depth_controller>
    LABELS "V2;Unit;MTP;DepthPolicy"
    MPI_PROCS 1
)

# Test: Shared MTP decode-equivalent catch-up contract
add_executable(v2_test_mtp_decode_catchup unit/mtp/Test__MTPDecodeCatchup.cpp)
target_link_libraries(v2_test_mtp_decode_catchup llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_decode_catchup PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPDecodeCatchup
    COMMAND $<TARGET_FILE:v2_test_mtp_decode_catchup>
    LABELS "V2;Unit;MTP;DecodeCatchup;Rollback"
    MPI_PROCS 1
)

# Test: Shared vLLM-style MTP rejection sampler contract
add_executable(v2_test_mtp_rejection_sampler unit/mtp/Test__MTPRejectionSampler.cpp)
target_link_libraries(v2_test_mtp_rejection_sampler llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_rejection_sampler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPRejectionSampler
    COMMAND $<TARGET_FILE:v2_test_mtp_rejection_sampler>
    LABELS "V2;Unit;MTP;SpecDecode;Sampler"
    MPI_PROCS 1
)

# Test: vLLM-style MTP spec-decode transaction metadata
add_executable(v2_test_mtp_spec_decode_transaction unit/mtp/Test__MTPSpecDecodeTransaction.cpp)
target_link_libraries(v2_test_mtp_spec_decode_transaction llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_spec_decode_transaction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPSpecDecodeTransaction
    COMMAND $<TARGET_FILE:v2_test_mtp_spec_decode_transaction>
    LABELS "V2;Unit;MTP;SpecDecode;Transaction"
    MPI_PROCS 1
)

# Test: vLLM-style MTP spec-decode graph-facing metadata buffers
add_executable(v2_test_mtp_spec_decode_metadata unit/mtp/Test__MTPSpecDecodeMetadata.cpp)
target_link_libraries(v2_test_mtp_spec_decode_metadata llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_spec_decode_metadata PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPSpecDecodeMetadata
    COMMAND $<TARGET_FILE:v2_test_mtp_spec_decode_metadata>
    LABELS "V2;Unit;MTP;SpecDecode;Metadata;Workspace"
    MPI_PROCS 1
)

# Test: vLLM-style MTP verifier graph-forward execution routing
add_executable(v2_test_mtp_verifier_forward_executor unit/mtp/Test__MTPVerifierForwardExecutor.cpp)
target_link_libraries(v2_test_mtp_verifier_forward_executor llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_verifier_forward_executor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPVerifierForwardExecutor
    COMMAND $<TARGET_FILE:v2_test_mtp_verifier_forward_executor>
    LABELS "V2;Unit;MTP;SpecDecode;Graph"
    MPI_PROCS 1
)

# Test: vLLM-style MTP request-batch scheduler admission
add_executable(v2_test_mtp_spec_request_batch_scheduler unit/mtp/Test__MTPSpecRequestBatchScheduler.cpp)
target_link_libraries(v2_test_mtp_spec_request_batch_scheduler llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_spec_request_batch_scheduler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPSpecRequestBatchScheduler
    COMMAND $<TARGET_FILE:v2_test_mtp_spec_request_batch_scheduler>
    LABELS "V2;Unit;MTP;SpecDecode;Scheduler"
    MPI_PROCS 1
)

# Test: vLLM-style MTP request-batch ownership and handoff
add_executable(v2_test_mtp_spec_request_batch_owner unit/mtp/Test__MTPSpecRequestBatchOwner.cpp)
target_link_libraries(v2_test_mtp_spec_request_batch_owner llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_spec_request_batch_owner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPSpecRequestBatchOwner
    COMMAND $<TARGET_FILE:v2_test_mtp_spec_request_batch_owner>
    LABELS "V2;Unit;MTP;SpecDecode;Scheduler;Ownership"
    MPI_PROCS 1
)

# Test: vLLM-style MTP spec-step state publication contract
add_executable(v2_test_mtp_spec_state_contract unit/mtp/Test__MTPSpecStateContract.cpp)
target_link_libraries(v2_test_mtp_spec_state_contract llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_spec_state_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPSpecStateContract
    COMMAND $<TARGET_FILE:v2_test_mtp_spec_state_contract>
    LABELS "V2;Unit;MTP;SpecDecode;StateContract"
    MPI_PROCS 1
)

# Test: vLLM-style MTP accepted-count KV publication contract
add_executable(v2_test_mtp_spec_kv_publisher unit/mtp/Test__MTPSpecKVPublisher.cpp)
target_link_libraries(v2_test_mtp_spec_kv_publisher llaminar2_core GTest::gtest GTest::gmock GTest::gtest_main)
target_include_directories(v2_test_mtp_spec_kv_publisher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPSpecKVPublisher
    COMMAND $<TARGET_FILE:v2_test_mtp_spec_kv_publisher>
    LABELS "V2;Unit;MTP;SpecDecode;KVPublisher"
    MPI_PROCS 1
)

# Test: MTP transaction state and rollback invariants
add_executable(v2_test_mtp_state_transaction unit/mtp/Test__MTPStateTransaction.cpp)
target_link_libraries(v2_test_mtp_state_transaction llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_mtp_state_transaction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPStateTransaction
    COMMAND $<TARGET_FILE:v2_test_mtp_state_transaction>
    LABELS "V2;Unit;MTP;StateTransaction;Rollback"
    MPI_PROCS 1
)

# Test: MTP verifier execution-policy selection
add_executable(v2_test_mtp_verifier_policy unit/mtp/Test__MTPVerifierPolicy.cpp)
target_link_libraries(v2_test_mtp_verifier_policy llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_mtp_verifier_policy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPVerifierPolicy
    COMMAND $<TARGET_FILE:v2_test_mtp_verifier_policy>
    LABELS "V2;Unit;MTP;VerifierPolicy;Rollback"
    MPI_PROCS 1
)

# Test: MTP sidecar weight discovery and graph-facing bindings
add_executable(v2_test_mtp_weight_manifest unit/mtp/Test__MTPWeightManifest.cpp)
target_link_libraries(v2_test_mtp_weight_manifest llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_mtp_weight_manifest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPWeightManifest
    COMMAND $<TARGET_FILE:v2_test_mtp_weight_manifest>
    LABELS "V2;Unit;MTP;Weights;PrefixCache"
    MPI_PROCS 1
)

# Test: MTP sidecar graph construction
add_executable(v2_test_mtp_graph_construction unit/mtp/Test__MTPGraphConstruction.cpp)
target_link_libraries(v2_test_mtp_graph_construction llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_mtp_graph_construction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_MTPGraphConstruction
    COMMAND $<TARGET_FILE:v2_test_mtp_graph_construction>
    LABELS "V2;Unit;MTP;Graph;PrefixCache"
    MPI_PROCS 1
)

# Test: Phase 0 real-runner state probe for GPU prefix-cache/MTP work
add_executable(v2_integration_prefix_cache_state_probe integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp)
target_include_directories(v2_integration_prefix_cache_state_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_integration_prefix_cache_state_probe PRIVATE
        llaminar2_core
        GTest::gtest
    )
elseif(UNIX AND NOT APPLE)
    set(_prefix_probe_gpu_backends "")
    if(HAVE_CUDA)
        list(APPEND _prefix_probe_gpu_backends -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
    endif()
    if(HAVE_ROCM)
        list(APPEND _prefix_probe_gpu_backends -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
    endif()
    target_link_libraries(v2_integration_prefix_cache_state_probe PRIVATE
        -Wl,--start-group
        llaminar2_core
        ${_prefix_probe_gpu_backends}
        -Wl,--end-group
        GTest::gtest
    )
else()
    target_link_libraries(v2_integration_prefix_cache_state_probe PRIVATE
        llaminar2_core
        GTest::gtest
    )
endif()
add_v2_test(V2_Integration_PrefixCacheStateProbe
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe>
        --gtest_filter=Test__KVPrefixMTPStateProbe.DenseQwen25_*:Test__KVPrefixMTPStateProbe.MTP_ModelInventoryWhenAvailable:Test__KVPrefixMTPStateProbe.MTP_ShiftedCacheCountProbeOnGPU:Test__KVPrefixMTPStateProbe.MTP_SidecarOneTokenExecutesOnGPU
    LABELS "V2;Integration;PrefixCache;StateProbe;MTP;GPU"
    MPI_PROCS 1
    TIMEOUT 300
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmMTPRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU"
    MPI_PROCS 1
    TIMEOUT 1200
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmMTPGpuGraphsRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;Regression"
    MPI_PROCS 1
    TIMEOUT 600
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36CUDAMTPGpuGraphsRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;CUDA;GPU;Regression"
    MPI_PROCS 1
    TIMEOUT 600
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmMTPGpuGraphsStochasticRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;Sampling;Regression"
    MPI_PROCS 1
    TIMEOUT 600
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsStochasticClearCacheRepeatabilityLong
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmMTPGpuGraphsStochasticClearCacheRepeatabilityLong
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;Sampling;Regression"
    MPI_PROCS 1
    TIMEOUT 1200
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsStochasticSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36CUDAMTPGpuGraphsStochasticRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;CUDA;GPU;Sampling;Regression"
    MPI_PROCS 1
    TIMEOUT 600
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36CUDAGpuGraphsStochasticClearCacheRepeatabilityLong
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36CUDAMTPGpuGraphsStochasticClearCacheRepeatabilityLong
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;CUDA;GPU;Sampling;Regression"
    MPI_PROCS 1
    TIMEOUT 1200
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsChainedDraftSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmMTPGpuGraphsChainedDraftRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;Regression"
    MPI_PROCS 1
    TIMEOUT 600
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmGpuGraphsBaselineThenMTPRegression
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmMTPGpuGraphsBaselineThenMTPRegression
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;Regression"
    MPI_PROCS 1
    TIMEOUT 900
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmPaddedPrefillBucketRegression
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmPaddedPrefillBucketGraphCaptureRegression
    LABELS "V2;Integration;PrefixCache;RealModel;Qwen36;ROCm;GPU;GraphCapture;Regression"
    MPI_PROCS 1
    TIMEOUT 900
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmPrefixSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmPrefixCacheMTPRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU"
    MPI_PROCS 1
    TIMEOUT 1200
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmLocalTPMTPRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;LocalTP"
    MPI_PROCS 1
    TIMEOUT 1200
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPSegmentedHardFail
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmLocalTPMTPSegmentedCollectiveHardFailsBeforeDraft
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;LocalTP;Regression"
    MPI_PROCS 1
    TIMEOUT 600
)
add_v2_test(V2_Integration_PrefixCacheMTP_Qwen36ROCmLocalTPPrefixSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen36ROCmLocalTPPrefixCacheMTPRealModelSmoke
    LABELS "V2;Integration;PrefixCache;MTP;RealModel;Qwen36;ROCm;GPU;LocalTP"
    MPI_PROCS 1
    TIMEOUT 1200
)
add_v2_test(V2_Integration_Parity_PrefixCacheMTP_Qwen35CPUPrefixSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen35CPUPrefixCacheMatchesPyTorchDecodeTokens
    LABELS "V2;Integration;Parity;PrefixCache;MTP;RealModel;Qwen35;CPU"
    MPI_PROCS 1
    TIMEOUT 900
)
add_v2_test(V2_Integration_Parity_PrefixCacheMTP_Qwen35CPUPartialPrefixSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen35CPUPartialPrefixCacheMatchesPyTorchDecodeTokens
    LABELS "V2;Integration;Parity;PrefixCache;MTP;RealModel;Qwen35;CPU"
    MPI_PROCS 1
    TIMEOUT 900
)
add_v2_test(V2_Integration_Parity_PrefixCacheMTP_Qwen35CPUSplitPrefillSmoke
    COMMAND $<TARGET_FILE:v2_integration_prefix_cache_state_probe> --gtest_filter=Test__KVPrefixMTPStateProbe.Qwen35CPUSplitPrefillMatchesPyTorchDecodeTokens
    LABELS "V2;Integration;Parity;PrefixCache;MTP;RealModel;Qwen35;CPU"
    MPI_PROCS 1
    TIMEOUT 900
)

# Test: ConfigValidator (declarative device selection validation rules)
# Tests mutual exclusion, co-requirements, intra-mode consistency, mode detection.
# Part of CLI Device Selection Cleanup
add_executable(v2_test_config_validator unit/config/Test__ConfigValidator.cpp)
target_link_libraries(v2_test_config_validator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ConfigValidator
    COMMAND $<TARGET_FILE:v2_test_config_validator>
    LABELS "V2;Unit;Config;Validation;CLI;DeviceSelection"
    MPI_PROCS 1
)

# Test: ParallelismTreeParser (YAML and CLI parsing for ParallelismTree construction)
# Tests YAML parsing, CLI inline format, error handling, round-trip serialization.
# Part of Phase 7: Global PP Implementation - ParallelismTree Configuration
add_executable(v2_test_parallelism_tree_parser unit/config/Test__ParallelismTreeParser.cpp)
target_link_libraries(v2_test_parallelism_tree_parser llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ParallelismTreeParser
    COMMAND $<TARGET_FILE:v2_test_parallelism_tree_parser>
    LABELS "V2;Unit;Config;CLI;YAML;Topology;ParallelismTree;Parser"
    MPI_PROCS 1
)

# Test: TPPPValidator (validates TP/PP configuration against model architecture)
# Tests head divisibility, dimension divisibility, layer assignment, proportional weights.
# Part of Phase 0.4: Model-Aware Configuration Validation
add_executable(v2_test_tppp_validator unit/config/Test__TPPPValidator.cpp)
target_link_libraries(v2_test_tppp_validator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TPPPValidator
    COMMAND $<TARGET_FILE:v2_test_tppp_validator>
    LABELS "V2;Unit;Config;Validation;TP;PP;Orchestration;ModelAware"
    MPI_PROCS 1
)

# Test: RankExecutionPlan (contract between Layer 2 and Layer 3)
# Tests validation, convenience methods, toString(), and integration scenarios.
# Part of Phase 1: Plan-to-Graph Translation
add_executable(v2_test_rank_execution_plan unit/execution/mpi_orchestration/Test__RankExecutionPlan.cpp)
target_link_libraries(v2_test_rank_execution_plan llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RankExecutionPlan
    COMMAND $<TARGET_FILE:v2_test_rank_execution_plan>
    LABELS "V2;Unit;Execution;Orchestration;Heterogeneous;PlanGeneration"
    MPI_PROCS 1
)

# Test: ExecutionPlanBuilder (builds execution plans from orchestration config)
# Tests single device, LOCAL TP, GLOBAL TP, PP, named domains, weight sharding.
# Part of Phase 1: Plan-to-Graph Translation
add_executable(v2_test_execution_plan_builder unit/execution/mpi_orchestration/Test__ExecutionPlanBuilder.cpp)
target_link_libraries(v2_test_execution_plan_builder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ExecutionPlanBuilder
    COMMAND $<TARGET_FILE:v2_test_execution_plan_builder>
    LABELS "V2;Unit;Execution;Orchestration;Heterogeneous;PlanGeneration;TP;PP"
    MPI_PROCS 1
)

# Test: LayerDevicePlacement (layer-to-device mapping for multi-device graph building)
# Tests SingleDevicePlacement, HybridCpuGpuPlacement, PipelineParallelPlacement,
# LocalTPPlacement, and factory method fromExecutionPlan().
# Part of Phase 2: Multi-Device Graph Building
add_executable(v2_test_layer_device_placement unit/execution/local_execution/device/Test__LayerDevicePlacement.cpp)
target_link_libraries(v2_test_layer_device_placement llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LayerDevicePlacement
    COMMAND $<TARGET_FILE:v2_test_layer_device_placement>
    LABELS "V2;Unit;Execution;Orchestration;Heterogeneous;LayerPlacement;TP;PP"
    MPI_PROCS 1
)

# Test: QwenStandardGraphConfigBuilder (builds QwenStandardGraphConfig from RankExecutionPlan)
# Tests config building for single device, PP layer ranges, LOCAL TP head distribution,
# proportional weights, and combined PP+TP scenarios.
# Part of Phase 2: Multi-Device Graph Building
add_executable(v2_test_qwen2_graph_config_builder unit/models/qwen/Test__QwenStandardGraphConfigBuilder.cpp)
target_link_libraries(v2_test_qwen2_graph_config_builder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_QwenStandardGraphConfigBuilder
    COMMAND $<TARGET_FILE:v2_test_qwen2_graph_config_builder>
    LABELS "V2;Unit;Models;Qwen2;GraphConfig;Orchestration;Heterogeneous;TP;PP"
    MPI_PROCS 1
)

# Test: HeterogeneousLayerExecutor (heterogeneous device layer execution)
# Tests device routing based on LayerPlacementConfig, statistics tracking,
# cross-domain transfer detection, and layer execution on GPU/CPU.
# Part of Phase 5.1: Heterogeneous Layer Execution
add_executable(v2_test_heterogeneous_layer_executor unit/execution/local_execution/model/Test__HeterogeneousLayerExecutor.cpp)
target_link_libraries(v2_test_heterogeneous_layer_executor llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_HeterogeneousLayerExecutor
    COMMAND $<TARGET_FILE:v2_test_heterogeneous_layer_executor>
    LABELS "V2;Unit;Execution;HeterogeneousCompute;LayerExecution;DeviceRouting"
    MPI_PROCS 1
)

# Test: StageBufferRequirements (IComputeStage::getBufferRequirements())
# Tests buffer requirement declarations for all compute stages (SwiGLU, ResidualAdd,
# RMSNorm, RoPE, GEMM, FusedQKVGEMM, FusedGateUpGEMM) for BufferArena integration
# Part of Phase 3: Stage Migration for buffer management infrastructure
add_executable(v2_test_stage_buffer_requirements unit/execution/compute_stages/Test__StageBufferRequirements.cpp)
target_link_libraries(v2_test_stage_buffer_requirements llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageBufferRequirements
    COMMAND $<TARGET_FILE:v2_test_stage_buffer_requirements>
    LABELS "V2;Unit;Execution;ComputeStage;BufferRequirements;BufferRole"
    MPI_PROCS 1
)

# Test: TensorVerification (unified tensor verification system)
# Tests VerificationResult, verifyRawBuffer, VerificationFailure, dumpStageBuffers.
# Part of Tensor Contracts and Validation Framework - Phase 1
add_executable(v2_test_tensor_verification unit/tensors/Test__TensorVerification.cpp)
target_link_libraries(v2_test_tensor_verification llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TensorVerification
    COMMAND $<TARGET_FILE:v2_test_tensor_verification>
    LABELS "V2;Unit;Utils;TensorVerification;Validation;Debug"
    MPI_PROCS 1
)

# Test: GraphValidator (buffer flow validation for compute graphs)
# Tests buffer producer/consumer ordering, orphan detection, type compatibility.
# Part of Buffer Contract Validation System - Phase 2: Buffer Contracts
add_executable(v2_test_graph_validator unit/execution/local_execution/graph/Test__GraphValidator.cpp)
target_link_libraries(v2_test_graph_validator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphValidator
    COMMAND $<TARGET_FILE:v2_test_graph_validator>
    LABELS "V2;Unit;Execution;Validation;GraphValidator;BufferContracts"
    MPI_PROCS 1
)

# Test: StageVerifier (stage output validation regressions)
add_executable(v2_test_stage_verifier unit/execution/local_execution/graph/Test__StageVerifier.cpp)
target_link_libraries(v2_test_stage_verifier llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageVerifier
    COMMAND $<TARGET_FILE:v2_test_stage_verifier>
    LABELS "V2;Unit;Execution;Validation;StageVerifier"
    MPI_PROCS 1
)

# Test: StageOutputManifest (getDeclaredOutputs for stage output contracts)
# Tests that stages correctly declare their output buffers with producer contracts.
# Part of Buffer Contract Validation System - Phase 3: Stage Output Manifest
add_executable(v2_test_stage_output_manifest unit/execution/debug/Test__StageOutputManifest.cpp)
target_link_libraries(v2_test_stage_output_manifest llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageOutputManifest
    COMMAND $<TARGET_FILE:v2_test_stage_output_manifest>
    LABELS "V2;Unit;Execution;Validation;StageOutputManifest;BufferContracts"
    MPI_PROCS 1
)

# Test: InferenceMode (centralized mode context for activation precision)
# Tests mode identification, buffer requirements, attention strategy, validation.
# Part of Buffer Contract Validation System - Phase 4: InferenceMode Context
add_executable(v2_test_inference_mode unit/execution/config/Test__InferenceMode.cpp)
target_link_libraries(v2_test_inference_mode llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_InferenceMode
    COMMAND $<TARGET_FILE:v2_test_inference_mode>
    LABELS "V2;Unit;Execution;Validation;InferenceMode;BufferContracts"
    MPI_PROCS 1
)

# Test: ValidationConfig (buffer validation environment variables and DeviceGraphExecutor integration)
# Tests LLAMINAR_VALIDATE_BUFFERS, LLAMINAR_FAIL_ON_ZERO, LLAMINAR_FAIL_ON_NAN.
# Part of Buffer Contract Validation System - Phase 5: Debug Validation Flag
add_executable(v2_test_validation_config unit/execution/config/Test__ValidationConfig.cpp)
target_link_libraries(v2_test_validation_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ValidationConfig
    COMMAND $<TARGET_FILE:v2_test_validation_config>
    LABELS "V2;Unit;Execution;Validation;ValidationConfig;BufferContracts;DebugEnv"
    MPI_PROCS 1
)

# Test: StreamingEnv (weight streaming environment variables and configuration)
# Tests LLAMINAR_WEIGHT_STREAMING, LLAMINAR_STREAM_MEMORY_MB, LLAMINAR_STREAM_PREFETCH_DEPTH,
# LLAMINAR_STREAM_EVICTION_POLICY, LLAMINAR_STREAM_VERBOSE.
# Part of Option B: Weight Streaming infrastructure
add_executable(v2_test_streaming_env unit/loaders/Test__StreamingEnv.cpp)
target_link_libraries(v2_test_streaming_env llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StreamingEnv
    COMMAND $<TARGET_FILE:v2_test_streaming_env>
    LABELS "V2;Unit;WeightStreaming;DebugEnv;Configuration"
    MPI_PROCS 1
)

# Test: MultiDomainOrchestrator (multi-domain tensor parallelism wrapper)
# Tests factory methods, domain access, delegation to inner orchestrator, and stats.
# Part of Phase 4.2: Multi-domain TP support for heterogeneous GPU systems
add_executable(v2_test_multi_domain_orchestrator unit/execution/local_execution/orchestrators/Test__MultiDomainOrchestrator.cpp)
target_link_libraries(v2_test_multi_domain_orchestrator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MultiDomainOrchestrator
    COMMAND $<TARGET_FILE:v2_test_multi_domain_orchestrator>
    LABELS "V2;Unit;MultiDomainTP;DeviceGraphOrchestrator;TensorParallelism;Execution"
    MPI_PROCS 1
)

# Test: RankOrchestrator (LOCAL tensor parallelism across multiple devices)
# Tests mock device runner and TP context, configuration validation, device coordination,
# forward/clear_cache delegation, logits retrieval, and synchronization.
# Part of LOCAL TP implementation for multi-GPU inference within single MPI rank.
add_executable(v2_test_multi_device_orchestrator unit/execution/local_execution/orchestrators/Test__RankOrchestrator.cpp)
target_include_directories(v2_test_multi_device_orchestrator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_multi_device_orchestrator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RankOrchestrator
    COMMAND $<TARGET_FILE:v2_test_multi_device_orchestrator>
    LABELS "V2;Unit;LocalTP;MultiDevice;IRankOrchestrator;TensorParallelism;Execution"
    MPI_PROCS 1
)

# Test: MDO Interface Decoupling (IInferenceRunner-based device_runners_)
# Verifies that RankOrchestrator works through IInferenceRunner interface
# without concrete DeviceGraphOrchestrator dependency. Tests primaryDeviceId(),
# hasLogitsLocal(), getLogitsLocalInfo(), and gatherLogits via CPU-only mocks.
add_executable(v2_test_mdo_interface_decoupling unit/execution/local_execution/orchestrators/Test__MDO_InterfaceDecoupling.cpp)
target_include_directories(v2_test_mdo_interface_decoupling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_mdo_interface_decoupling llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MDO_InterfaceDecoupling
    COMMAND $<TARGET_FILE:v2_test_mdo_interface_decoupling>
    LABELS "V2;Unit;LocalTP;MultiDevice;IInferenceRunner;InterfaceDecoupling;Execution"
    MPI_PROCS 1
)

# Test: MDO Delegation Coverage (comprehensive delegation pattern tests)
# Verifies that RankOrchestrator correctly delegates to device runners,
# TP context, and PP stage runners for all public API methods.
add_executable(v2_test_mdo_delegation_coverage unit/execution/local_execution/orchestrators/Test__MDO_DelegationCoverage.cpp)
target_include_directories(v2_test_mdo_delegation_coverage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_mdo_delegation_coverage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MDO_DelegationCoverage
    COMMAND $<TARGET_FILE:v2_test_mdo_delegation_coverage>
    LABELS "V2;Unit;LocalTP;MultiDevice;IInferenceRunner;Delegation;Execution"
    MPI_PROCS 1
)

# Test: LogitsGatherer (buffer allocation, skip-gather control, single-device gather, copyFromStage)
add_executable(v2_test_logits_gatherer unit/execution/local_execution/orchestrators/Test__LogitsGatherer.cpp)
target_include_directories(v2_test_logits_gatherer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_test_logits_gatherer llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LogitsGatherer
    COMMAND $<TARGET_FILE:v2_test_logits_gatherer>
    LABELS "V2;Unit;MultiDevice;LogitsGatherer;Execution"
    MPI_PROCS 1
)

# Test: DeviceSampler (edge cases, GPU-fallback paths)
add_executable(v2_test_device_sampler unit/execution/local_execution/orchestrators/Test__DeviceSampler.cpp)
target_include_directories(v2_test_device_sampler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_test_device_sampler llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceSampler
    COMMAND $<TARGET_FILE:v2_test_device_sampler>
    LABELS "V2;Unit;MultiDevice;DeviceSampler;Execution"
    MPI_PROCS 1
)

# Test: RankOrchestrator PP Config (Pipeline Parallelism configuration validation)
# Tests PPStageConfig validation, ParallelismMode detection, and Config::validate() for PP mode.
# Phase 1 of RankOrchestrator PP support implementation.
add_executable(v2_test_multi_device_orchestrator_pp_config unit/orchestrators/Test__RankOrchestrator_PPConfig.cpp)
target_link_libraries(v2_test_multi_device_orchestrator_pp_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RankOrchestrator_PPConfig
    COMMAND $<TARGET_FILE:v2_test_multi_device_orchestrator_pp_config>
    LABELS "V2;Unit;LocalPP;MultiDevice;IRankOrchestrator;PipelineParallelism;Config"
    MPI_PROCS 1
)

# Test: RankOrchestrator PP Interface (Hidden State API for PP nesting)
# Tests getHiddenState, setHiddenState, hasHiddenStateInput, clearHiddenStateInput
# for TP and PP modes to enable RankOrchestrator to be nested as a PP stage.
add_executable(v2_test_multi_device_orchestrator_pp_interface unit/orchestrators/Test__RankOrchestrator_PPInterface.cpp)
target_link_libraries(v2_test_multi_device_orchestrator_pp_interface llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RankOrchestrator_PPInterface
    COMMAND $<TARGET_FILE:v2_test_multi_device_orchestrator_pp_interface>
    LABELS "V2;Unit;LocalPP;MultiDevice;IRankOrchestrator;PipelineParallelism;HiddenState"
    MPI_PROCS 1
)

# Test: RankOrchestrator Nested TP (TP+PP hybrid mode nested MDO creation)
# Tests that PP stages configured as TP domains create nested RankOrchestrators
# Phase 5: TP+PP hybrid where each PP stage can be a TP domain with multiple devices.
add_executable(v2_test_multi_device_orchestrator_nested_tp unit/orchestrators/Test__RankOrchestrator_NestedTP.cpp)
target_link_libraries(v2_test_multi_device_orchestrator_nested_tp llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RankOrchestrator_NestedTP
    COMMAND $<TARGET_FILE:v2_test_multi_device_orchestrator_nested_tp>
    LABELS "V2;Unit;LocalPP;MultiDevice;IRankOrchestrator;PipelineParallelism;TP_PP;Hybrid"
    MPI_PROCS 1
)

# Test: RankOrchestrator getSnapshot in PP+TP mode (ownership-aware LM_HEAD retrieval)
# Tests that getSnapshot("LM_HEAD") respects hasLMHead() ownership in hybrid PP+TP mode.
# Bug Caught: Stage 0 returned stale combined_logits_ even though it didn't own LM_HEAD.
# Root Cause: Missing hasLMHead() check before returning combined_logits_.
add_executable(v2_test_multi_device_orchestrator_get_snapshot_pptp unit/orchestrators/Test__RankOrchestrator_GetSnapshotPPTP.cpp)
target_include_directories(v2_test_multi_device_orchestrator_get_snapshot_pptp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_multi_device_orchestrator_get_snapshot_pptp llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RankOrchestrator_GetSnapshotPPTP
    COMMAND $<TARGET_FILE:v2_test_multi_device_orchestrator_get_snapshot_pptp>
    LABELS "V2;Unit;LocalPP;MultiDevice;IRankOrchestrator;PipelineParallelism;TP_PP;Snapshot;LMHead"
    MPI_PROCS 1
)

# Test: RankOrchestrator PP Integration (Pipeline Parallelism end-to-end)
# Tests PP mode construction, forward execution, and parity with single-device mode.
# Phase 2 of RankOrchestrator PP support implementation.
add_executable(v2_integration_multi_device_orchestrator_pp integration/orchestrators/Test__RankOrchestrator_PP.cpp)
target_link_libraries(v2_integration_multi_device_orchestrator_pp llaminar2_core GTest::gtest)
add_v2_test(V2_Integration_RankOrchestrator_PP
    COMMAND $<TARGET_FILE:v2_integration_multi_device_orchestrator_pp>
    LABELS "V2;Integration;LocalPP;MultiDevice;IRankOrchestrator;PipelineParallelism"
    MPI_PROCS 1
    TIMEOUT 1800
)

# Test: EagerWeightValidator (schema-aware weight validation for model loading)
# Tests that missing optional weights (e.g., QKV biases in Qwen3) are skipped
# silently, while missing required weights cause a fatal validation failure.
add_executable(v2_unit_eager_weight_validator unit/execution/factory/Test__EagerWeightValidator.cpp)
target_include_directories(v2_unit_eager_weight_validator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_eager_weight_validator
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Unit_EagerWeightValidator
    COMMAND $<TARGET_FILE:v2_unit_eager_weight_validator>
    LABELS "V2;Unit;EagerWeightValidator;WeightLoading;Factory;Internal"
    MPI_PROCS 1
)

# Test: InferenceRunnerFactory MultiDevice (factory functions for RankOrchestrator)
# Tests createRankOrchestrator and createTestableRankOrchestrator
# factory functions with various configurations, validation, and error handling.
# Part of Phase 5: RankOrchestrator integration with InferenceRunnerFactory
add_executable(v2_unit_inference_runner_factory_multidevice unit/execution/factory/Test__InferenceRunnerFactory_MultiDevice.cpp)
target_include_directories(v2_unit_inference_runner_factory_multidevice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_inference_runner_factory_multidevice
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Unit_InferenceRunnerFactory_MultiDevice
    COMMAND $<TARGET_FILE:v2_unit_inference_runner_factory_multidevice>
    LABELS "V2;Unit;InferenceRunnerFactory;MultiDevice;LocalTP;Factory;Internal"
    MPI_PROCS 1
)

# Test: PPStageConfig (Pipeline Parallelism stage configuration struct)
# Tests PPStageConfig validation (isValid) and layer counting (layerCount).
# Part of Pipeline Parallelism implementation for multi-device inference.
add_executable(v2_unit_pp_stage_config unit/execution/factory/Test__PPStageConfig.cpp)
target_link_libraries(v2_unit_pp_stage_config
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PPStageConfig
    COMMAND $<TARGET_FILE:v2_unit_pp_stage_config>
    LABELS "V2;Unit;PPStageConfig;PipelineParallelism;Factory;PP"
    MPI_PROCS 1
)

# Test: StageCoherence (automatic device coherence at stage boundaries)
# Tests CoherencePolicy enum, cohereInputs/markOutputsDirty, buffer extraction,
# and stage coherence policy reporting.
# Part of Device Abstraction Phase 2: Automatic Stage Boundary Coherence
add_executable(v2_test_stage_coherence unit/execution/local_execution/coherence/Test__StageCoherence.cpp)
target_link_libraries(v2_test_stage_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageCoherence
    COMMAND $<TARGET_FILE:v2_test_stage_coherence>
    LABELS "V2;Unit;Coherence;DeviceGraphExecutor;DeviceAbstraction;Execution"
    MPI_PROCS 1
)

# Test: StreamCoherence (stream/event coherence interaction — regression tests)
# Tests for two critical coherence bugs in TP collective stages:
# Bug 1: Collective stages must use CoherencePolicy::OUTPUT (not NONE)
# Bug 2: mark_device_dirty_flags_only preserves stale completion events
# Uses MockCoherenceTensor to verify behavior without GPU hardware.
add_executable(v2_test_stream_coherence unit/execution/local_execution/coherence/Test__StreamCoherence.cpp)
target_link_libraries(v2_test_stream_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StreamCoherence
    COMMAND $<TARGET_FILE:v2_test_stream_coherence>
    LABELS "V2;Unit;Coherence;StreamCoherence;StaleEvent;TPAllreduce;AllGather;Regression"
    MPI_PROCS 1
)

# Test: CoherenceProtocol (comprehensive GPU tensor coherence protocol tests)
# Tests all major data paths: ensureOnDevice H2D, data() D2H trigger, allocateOnDevice,
# ensureOnHost no-op, mark_device_dirty_with_event lifecycle, full state machine, failures.
# Uses MockBackend DI injection for deterministic testing without GPU hardware.
add_executable(v2_test_coherence_protocol unit/execution/local_execution/coherence/Test__CoherenceProtocol.cpp)
target_link_libraries(v2_test_coherence_protocol llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CoherenceProtocol
    COMMAND $<TARGET_FILE:v2_test_coherence_protocol>
    LABELS "V2;Unit;Coherence;CoherenceProtocol;H2D;D2H;DeviceDirty;StateMachine;FailurePath"
    MPI_PROCS 1
)

# Test: CrossBackendCoherence (cross-backend safety in Pipeline Parallel mode)
# Validates that mark_device_dirty_with_event() detects cross-backend stream mismatches
# and skips event recording instead of crashing (e.g., CUDA stream → ROCm backend).
# Regression test for HybridPPTP segfault in hipEventRecord.
add_executable(v2_test_cross_backend_coherence unit/execution/local_execution/coherence/Test__CrossBackendCoherence.cpp)
target_link_libraries(v2_test_cross_backend_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CrossBackendCoherence
    COMMAND $<TARGET_FILE:v2_test_cross_backend_coherence>
    LABELS "V2;Unit;Coherence;CrossBackend;PipelineParallel;DeviceMigration;Regression"
    MPI_PROCS 1
)

# Test: CoherenceState (explicit coherence state machine)
# Tests all state transitions, invalid transitions, helper functions, and full lifecycle sequences.
add_executable(v2_test_coherence_state unit/tensors/Test__CoherenceState.cpp)
target_link_libraries(v2_test_coherence_state llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CoherenceState
    COMMAND $<TARGET_FILE:v2_test_coherence_state>
    LABELS "V2;Unit;Coherence;CoherenceState;StateMachine;Transitions"
    MPI_PROCS 1
)

# Test: CoherenceAuditLog (ring buffer recording coherence transitions)
# Tests recording, wrap-around, hasTransition(), dump() formatting.
add_executable(v2_test_coherence_audit_log unit/tensors/Test__CoherenceAuditLog.cpp)
target_link_libraries(v2_test_coherence_audit_log llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CoherenceAuditLog
    COMMAND $<TARGET_FILE:v2_test_coherence_audit_log>
    LABELS "V2;Unit;Coherence;AuditLog;Debugging"
    MPI_PROCS 1
)

# Test: TransferEngine (unified data movement planning and execution)
# Tests plan logic (pure, no GPU), mock execution via MockBackend, and error handling.
add_executable(v2_test_transfer_engine unit/transfer/Test__TransferEngine.cpp)
target_link_libraries(v2_test_transfer_engine llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TransferEngine
    COMMAND $<TARGET_FILE:v2_test_transfer_engine>
    LABELS "V2;Unit;Transfer;TransferEngine;Plan;Execute;BAR;HostBounce;Coherence"
    MPI_PROCS 1
)

# Test: QwenStandardGraphBufferManagement (BufferArena integration with DeviceGraphOrchestrator)
# Tests setTensorFactory/initializeBuffers workflow, buffer allocation/binding,
# statistics tracking, and release cleanup.
# Part of Phase 3: Buffer Management moved to DeviceGraphOrchestrator
add_executable(v2_test_device_graph_orchestrator_buffer_mgmt unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorBufferManagement.cpp)
target_link_libraries(v2_test_device_graph_orchestrator_buffer_mgmt llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestratorBufferManagement
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_buffer_mgmt>
    LABELS "V2;Unit;DeviceGraphOrchestrator;BufferManagement;BufferArena"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator (clean separation of graph building and execution)
# Tests the orchestrator's execution, caching, and device context management:
# - Construction with graph builder or config
# - Cache initialization and management
# - Device context lazy creation and caching
# - Cache statistics tracking
add_executable(v2_test_device_graph_orchestrator unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestrator.cpp)
target_link_libraries(v2_test_device_graph_orchestrator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestrator
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator>
    LABELS "V2;Unit;Qwen2;DeviceGraphOrchestrator;GraphExecution;DeclarativeDesign"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator Phase-Aware Weights (Gap 3 - CPU Decode Participation)
# Tests the phase-aware weight selection mechanism for "Option A: Selective Duplication":
# - Phase tracking (PREFILL vs DECODE)
# - WeightManager and PlacementMap integration
# - Phase-aware weight selection logic
# - CPU decode participation checks
add_executable(v2_test_device_graph_orchestrator_phase_aware_weights unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorPhaseAwareWeights.cpp)
target_include_directories(v2_test_device_graph_orchestrator_phase_aware_weights PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For mocks/ include path
)
target_link_libraries(v2_test_device_graph_orchestrator_phase_aware_weights llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestrator_PhaseAwareWeights
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_phase_aware_weights>
    LABELS "V2;Unit;DeviceGraphOrchestrator;PhaseAwareWeights;CPUDecodeParticipation;Gap3"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator Weight Streaming (Option B - On-Demand Weight Transfer)
# Tests the IWeightStreamer integration with DeviceGraphOrchestrator:
# - Phase transition notifications to streamer
# - Weight streaming hooks (ensure/prefetch/release) during layer execution
# - Streaming is optional (works without streamer set)
add_executable(v2_test_device_graph_orchestrator_weight_streaming unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorWeightStreaming.cpp)
target_link_libraries(v2_test_device_graph_orchestrator_weight_streaming llaminar2_core GTest::gtest GTest::gtest_main GTest::gmock)
add_v2_test(V2_Unit_DeviceGraphOrchestrator_WeightStreaming
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_weight_streaming>
    LABELS "V2;Unit;DeviceGraphOrchestrator;WeightStreaming;OptionB;IWeightStreamer"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator Domain Wiring (Phase 6.3 - Heterogeneous TP)
# Tests the MultiDomainTPConfig wiring in DeviceGraphOrchestrator:
# - setDomainConfig() and domainConfig() accessor
# - getDomainForLayer() routing logic
# - Domain config persistence across operations
# - Integration with AllreduceStage domain-aware communicator selection
add_executable(v2_test_device_graph_orchestrator_domain_wiring unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorDomainWiring.cpp)
target_link_libraries(v2_test_device_graph_orchestrator_domain_wiring llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestratorDomainWiring
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_domain_wiring>
    LABELS "V2;Unit;DeviceGraphOrchestrator;MultiDomainTP;HeterogeneousTP;Phase6.3"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator PP+TP initialization logic
# Tests that initializePPContexts() correctly:
# - Detects TP domains (degree > 1) and uses HierarchicalPPConfig
# - Falls back to flat LocalPPConfig when no TP domains
# - Wires TP contexts to PP contexts properly
add_executable(v2_test_device_graph_orchestrator_pptp_init unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorPPTPInit.cpp)
target_link_libraries(v2_test_device_graph_orchestrator_pptp_init llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestratorPPTPInit
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_pptp_init>
    LABELS "V2;Unit;DeviceGraphOrchestrator;PipelineParallel;TensorParallel;HierarchicalPPConfig;Phase6"
    MPI_PROCS 1
)

# Test: DeviceGraphOrchestrator Dependencies struct construction
# Validates the Phase 5 preferred constructor path using MockGraphBuilder
# (polymorphic IGraphBuilder, not QwenStandardGraph) and all Dependencies fields.
add_executable(v2_test_device_graph_orchestrator_deps_construction unit/execution/local_execution/orchestrators/Test__DeviceGraphOrchestratorDepsConstruction.cpp)
target_link_libraries(v2_test_device_graph_orchestrator_deps_construction llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceGraphOrchestratorDepsConstruction
    COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator_deps_construction>
    LABELS "V2;Unit;DeviceGraphOrchestrator;Dependencies;IGraphBuilder;DecouplingPhase5"
    MPI_PROCS 1
)

# Test: GraphResolver (declarative schema resolution for multi-model support)
# Tests the three-layer architecture: GraphSchema -> GraphResolver -> ComputeGraph
# Part of Phase 4: Schema-based graph building for model scalability
add_executable(v2_test_graph_resolver unit/execution/local_execution/graph/Test__GraphResolver.cpp)
target_link_libraries(v2_test_graph_resolver llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphResolver
    COMMAND $<TARGET_FILE:v2_test_graph_resolver>
    LABELS "V2;Unit;GraphResolver;GraphSchema;DeclarativeDesign;MultiModel"
    MPI_PROCS 1
)

# Test: GraphResolver ExpertParallel TP mode (Phase 4 MoE stage decomposition)
# Verifies TPMode::ExpertParallel inserts allreduce collectives
add_executable(v2_test_graph_resolver_expert_parallel unit/execution/local_execution/graph/Test__GraphResolver_ExpertParallel.cpp)
target_link_libraries(v2_test_graph_resolver_expert_parallel llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphResolver_ExpertParallel
    COMMAND $<TARGET_FILE:v2_test_graph_resolver_expert_parallel>
    LABELS "V2;Unit;GraphResolver;MoE;ExpertParallel"
    MPI_PROCS 1
)

# GraphBuilderRegistry: self-registration pattern for model architectures
add_executable(v2_test_graph_builder_registry unit/execution/local_execution/graph/Test__GraphBuilderRegistry.cpp)
target_link_libraries(v2_test_graph_builder_registry llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphBuilderRegistry
    COMMAND $<TARGET_FILE:v2_test_graph_builder_registry>
    LABELS "V2;Unit;GraphBuilderRegistry;Decoupling;MultiModel"
    MPI_PROCS 1
)

# SchemaFactoryRegistry: self-registration pattern for schema factories
add_executable(v2_test_schema_factory_registry unit/execution/local_execution/graph/Test__SchemaFactoryRegistry.cpp)
target_link_libraries(v2_test_schema_factory_registry llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_SchemaFactoryRegistry
    COMMAND $<TARGET_FILE:v2_test_schema_factory_registry>
    LABELS "V2;Unit;SchemaFactoryRegistry;Decoupling;MultiModel"
    MPI_PROCS 1
)

# ChatCompletionHandler: testable request handler for /v1/chat/completions
add_executable(v2_test_chat_completion_handler unit/app/modes/Test__ChatCompletionHandler.cpp)
target_include_directories(v2_test_chat_completion_handler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor  # For nlohmann/json.hpp
)
target_link_libraries(v2_test_chat_completion_handler llaminar2_core GTest::gtest GTest::gtest_main GTest::gmock)
add_v2_test(V2_Unit_ChatCompletionHandler
    COMMAND $<TARGET_FILE:v2_test_chat_completion_handler>
    LABELS "V2;Unit;Server;ChatCompletion;SamplingParams"
    MPI_PROCS 1
)

add_executable(v2_test_server_mode unit/app/modes/Test__ServerMode.cpp)
target_include_directories(v2_test_server_mode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor/cpp-httplib
)
target_link_libraries(v2_test_server_mode llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ServerMode
    COMMAND $<TARGET_FILE:v2_test_server_mode>
    LABELS "V2;Unit;Server;Threading;Memory"
    MPI_PROCS 1
)

# Completion and single-shot chat modes: MPI root/worker oneshot orchestration
add_executable(v2_test_completion_single_shot_modes unit/app/modes/Test__CompletionAndSingleShotModes.cpp)
target_include_directories(v2_test_completion_single_shot_modes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_completion_single_shot_modes
    llaminar2_core
    GTest::gtest
    GTest::gmock
    MPI::MPI_CXX
)
add_v2_test(V2_Unit_CompletionModeSingleShotChatMode
    COMMAND $<TARGET_FILE:v2_test_completion_single_shot_modes>
    LABELS "V2;Unit;App;CompletionMode;SingleShotChatMode;MPI;Oneshot"
    MPI_PROCS 1
)

# CLI Commands: DescribeCommand, PlanCommand, SubcommandRouter
add_executable(v2_test_commands unit/app/Test__Commands.cpp)
target_include_directories(v2_test_commands PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external/vendor  # For nlohmann/json.hpp
)
target_link_libraries(v2_test_commands llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_Commands
    COMMAND $<TARGET_FILE:v2_test_commands>
    LABELS "V2;Unit;CLI;Commands;DescribeCommand;PlanCommand"
    MPI_PROCS 1
)

# Test: QwenStandardGraphSchema (schema-based QwenStandardGraph building - Phase 4d)
# Tests the new buildForwardGraphFromSchema() method that uses
# Qwen2Schema + GraphResolver + GraphBuilder architecture.
# Validates that schema-based graphs match legacy imperative graphs.
add_executable(v2_test_qwen2_graph_schema unit/models/qwen/Test__QwenStandardGraphSchema.cpp)
target_link_libraries(v2_test_qwen2_graph_schema llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_QwenStandardGraphSchema
    COMMAND $<TARGET_FILE:v2_test_qwen2_graph_schema>
    LABELS "V2;Unit;Qwen2;GraphSchema;DeclarativeDesign;Phase4d"
    MPI_PROCS 1
)

# Test: Qwen3BufferSizes — Lock-in tests for Qwen3 schema buffer sizes
add_executable(v2_test_qwen3_buffer_sizes unit/models/qwen3/Test__Qwen3BufferSizes.cpp)
target_link_libraries(v2_test_qwen3_buffer_sizes llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_Qwen3BufferSizes
    COMMAND $<TARGET_FILE:v2_test_qwen3_buffer_sizes>
    LABELS "V2;Unit;Qwen3;GraphSchema;BufferSizes"
    MPI_PROCS 1
)

# Test: ExecutionConfigAutoEnable (regression test for empty graph bug)
# Tests that use_graph_buffer_management=true auto-enables all exec_* flags
# to prevent empty graphs (0 nodes) when graph buffer management is enabled.
# Covers the "repeating tokens" bug caused by graph having no computation nodes.
add_executable(v2_test_execution_config_auto_enable unit/execution/config/Test__ExecutionConfigAutoEnable.cpp)
target_link_libraries(v2_test_execution_config_auto_enable llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ExecutionConfigAutoEnable
    COMMAND $<TARGET_FILE:v2_test_execution_config_auto_enable>
    LABELS "V2;Unit;Execution;ExecutionConfig;GraphBufferManagement;Regression"
    MPI_PROCS 1
)

# Test: CPUKernelDeviceIndex (regression test for device_idx validation bug)
# Tests that CPU kernels accept any device_idx value (not just -1) for NUMA support.
add_executable(v2_test_cpu_kernel_device_index unit/kernels/cpu/Test__CPUKernelDeviceIndex.cpp)
target_link_libraries(v2_test_cpu_kernel_device_index llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CPUKernelDeviceIndex
    COMMAND $<TARGET_FILE:v2_test_cpu_kernel_device_index>
    LABELS "V2;Unit;Kernels;Attention;DeviceIndex;NUMA;Regression"
    MPI_PROCS 1
)

# Test: DecomposedAttentionRequirement (regression test for shared_from_this crash)
# Tests that graph buffer management requires decomposed attention path (KVCacheAppendStage +
# AttentionComputeStage).
# Covers the crash caused by unique_ptr buffers not supporting shared_from_this().

# Test: LayerExecutorInterface (ILayerExecutor interface and MockLayerExecutor)
# Tests polymorphic execution, MockLayerExecutor behavior, interface-based testing patterns,
# configuration, statistics, and multi-device execution through interface abstraction
add_executable(v2_test_layer_executor_interface unit/execution/local_execution/model/Test__LayerExecutorInterface.cpp)
target_link_libraries(v2_test_layer_executor_interface llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LayerExecutorInterface
    COMMAND $<TARGET_FILE:v2_test_layer_executor_interface>
    LABELS "V2;Unit;Execution;ILayerExecutor;MockLayerExecutor;Interface;Polymorphism;Testing"
    MPI_PROCS 1
)

# Test: StageParamsBase (device_id propagation via STAGE_PARAMS_COMMON_FIELDS macro)
# Validates that all stage Params satisfy StageParamsRequired concept, device_id defaults
# to CPU, and device() returns correct device from params - prevents bug where
# stages accidentally run on CPU when intended for GPU.
add_executable(v2_test_stage_params_base unit/execution/compute_stages/Test__StageParamsBase.cpp)
target_link_libraries(v2_test_stage_params_base llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageParamsBase
    COMMAND $<TARGET_FILE:v2_test_stage_params_base>
    LABELS "V2;Unit;ComputeStage;StageParams;DeviceId;Safety"
    MPI_PROCS 1
)

# Test: ComputeStages (individual compute stage unit tests)
# Tests GEMMStage, RMSNormStage, RoPEStage, SwiGLUStage, ResidualAddStage, etc.
# Verifies correctness, precision preservation, FLOP estimation, and backend compatibility
add_executable(v2_test_compute_stages unit/execution/compute_stages/Test__ComputeStages.cpp)
target_link_libraries(v2_test_compute_stages llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeStages
    COMMAND $<TARGET_FILE:v2_test_compute_stages>
    LABELS "V2;Unit;ComputeStage;GEMM;RMSNorm;RoPE;SwiGLU;ResidualAdd;Precision"
    MPI_PROCS 1
)

# Test: StageDumpInfo (getDumpInfo() snapshot support for all compute stages)
# Validates that each compute stage correctly reports inputs, outputs, weights, and scalars
add_executable(v2_test_stage_dump_info unit/execution/debug/Test__StageDumpInfo.cpp)
target_link_libraries(v2_test_stage_dump_info llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageDumpInfo
    COMMAND $<TARGET_FILE:v2_test_stage_dump_info>
    LABELS "V2;Unit;ComputeStage;Snapshot;DumpInfo;Debug"
    MPI_PROCS 1
)

# Test: KernelSnapshotInfo (getKernelSnapshotInfo() for all kernel implementations)
# Validates that each kernel correctly reports its I/O buffers via IKernelSnapshotCapable
add_executable(v2_test_kernel_snapshot_info unit/tensors/Test__KernelSnapshotInfo.cpp)
target_link_libraries(v2_test_kernel_snapshot_info llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_KernelSnapshotInfo
    COMMAND $<TARGET_FILE:v2_test_kernel_snapshot_info>
    LABELS "V2;Unit;Kernel;Snapshot;IKernelSnapshotCapable;Debug"
    MPI_PROCS 1
)

# Test: AllGatherStage (MPI collective for column-parallel LM head)
# Phase 5: Tests AllGatherStage for collecting distributed logits
add_executable(v2_test_allgather_stage unit/execution/compute_stages/stages/Test__AllGatherStage.cpp)
target_link_libraries(v2_test_allgather_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_AllGatherStage
    COMMAND $<TARGET_FILE:v2_test_allgather_stage>
    LABELS "V2;Unit;ComputeStage;AllGather;MPI;TensorParallel;LMHead"
    MPI_PROCS 1
)

# Test: AllGatherVStage (Variable-sized MPI collective for heterogeneous TP)
# Phase 1d: Tests AllGatherVStage for variable send counts per rank
add_executable(v2_test_allgatherv_stage unit/execution/compute_stages/stages/Test__AllGatherVStage.cpp)
target_link_libraries(v2_test_allgatherv_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_AllGatherVStage
    COMMAND $<TARGET_FILE:v2_test_allgatherv_stage>
    LABELS "V2;Unit;ComputeStage;AllGatherV;MPI;TensorParallel;HeterogeneousTP"
    MPI_PROCS 1
)

# Test: AllreduceStage (MPI collective for tensor parallel reductions)
# Phase 4.1a: Tests AllreduceStage including TPDomain support
add_executable(v2_test_allreduce_stage unit/execution/compute_stages/stages/Test__AllreduceStage.cpp)
target_link_libraries(v2_test_allreduce_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_AllreduceStage
    COMMAND $<TARGET_FILE:v2_test_allreduce_stage>
    LABELS "V2;Unit;ComputeStage;Allreduce;MPI;TensorParallel;TPDomain"
    MPI_PROCS 1
)

# =============================================================================
# Phase 4: LOCAL Tensor Parallelism Tests
# =============================================================================
# LOCAL TP enables sharding computations across multiple devices within a single
# MPI rank, decoupling TP degree from MPI world_size. This is essential for
# heterogeneous systems (e.g., 2 CUDA + 2 ROCm GPUs on one node).

# Test: ITPContext (Base tensor parallelism interface)
# Phase 1 Global TP Unified Interface: Tests the base ITPContext interface
# Validates LocalTPContext implements ITPContext correctly, isLocal()/isGlobal(),
# myIndex()/setCurrentDeviceIndex() functionality, and polymorphic usage.
add_executable(v2_test_itp_context unit/collective/Test__ITPContext.cpp)
target_link_libraries(v2_test_itp_context llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ITPContext
    COMMAND $<TARGET_FILE:v2_test_itp_context>
    LABELS "V2;Unit;ITPContext;TensorParallel;Interface;LocalTP;GlobalTP"
    MPI_PROCS 1
)

# Test: ICollectiveCoordinator (base interface for collective coordinators)
# Phase 3 Collective Coordinators: Tests the abstract base interface for coordinator
# threads that serialize collective operations. Tests template methods (submitAndWait,
# submitAsync), promise/future machinery, and exception propagation.
# CPU-only test using mock implementations - no GPU required.
add_executable(v2_test_icollective_coordinator unit/collective/Test__ICollectiveCoordinator.cpp)
target_link_libraries(v2_test_icollective_coordinator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ICollectiveCoordinator
    COMMAND $<TARGET_FILE:v2_test_icollective_coordinator>
    LABELS "V2;Unit;CollectiveCoordinator;Interface;Async;Threading"
    MPI_PROCS 1
)

# Test: GlobalTPContext (GLOBAL tensor parallelism context)
# Phase 2 Global TP Unified Interface: Tests IGlobalTPContext implementation
# for cross-MPI-rank tensor parallelism via UPI/MPI. Single-rank tests use MPI_COMM_SELF.
# Validates factory methods, accessors, move semantics, single-rank collectives, error handling.
# Note: Uses mpi_gtest_main.cpp for MPI initialization (needed for MPI_COMM_SELF tests).
add_executable(v2_test_global_tp_context
    unit/collective/Test__GlobalTPContext.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_test_global_tp_context llaminar2_core GTest::gtest)
add_v2_test(V2_Unit_GlobalTPContext
    COMMAND $<TARGET_FILE:v2_test_global_tp_context>
    LABELS "V2;Unit;GlobalTP;Context;TensorParallel;MPI;UPI;IGlobalTPContext"
    MPI_PROCS 1
)

# Test: TPContextFactory (factory for LOCAL and GLOBAL TP contexts)
# Phase 3 Global TP Unified Interface: Tests factory methods for creating appropriate
# ITPContext based on RankExecutionPlan, TPDomainParticipation, or explicit parameters.
# Validates local/global detection, backend selection, and plan-based creation.
add_executable(v2_test_tp_context_factory
    unit/collective/Test__TPContextFactory.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_test_tp_context_factory llaminar2_core GTest::gtest)
add_v2_test(V2_Unit_TPContextFactory
    COMMAND $<TARGET_FILE:v2_test_tp_context_factory>
    LABELS "V2;Unit;TPContextFactory;TensorParallel;Factory;LocalTP;GlobalTP;Phase3"
    MPI_PROCS 1
)

# Test: LocalTPContext (LOCAL tensor parallelism context)
# Phase 4: Tests ILocalTPContext implementation for device management and collectives
# Validates constructor, auto-detection, proportional weight distribution, head/row/col
# sharding queries, and collective operation interfaces.
add_executable(v2_test_local_tp_context unit/collective/Test__LocalTPContext.cpp)
target_link_libraries(v2_test_local_tp_context llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_LocalTPContext
    COMMAND $<TARGET_FILE:v2_test_local_tp_context>
    LABELS "V2;Integration;LocalTP;Context;TensorParallel;Heterogeneous;DeviceManagement"
    MPI_PROCS 1
)

# Test: LocalTPWeightSharder (weight sharding for LOCAL tensor parallelism)
# Phase 4: Tests ILocalTPWeightSharder implementation for column/row-parallel sharding
# Validates proportional weight distribution, equal weight handling, single-device
# extraction, and correct shard sizing based on device weights.
add_executable(v2_test_local_tp_weight_sharder unit/loaders/Test__LocalTPWeightSharder.cpp)
target_include_directories(v2_test_local_tp_weight_sharder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_local_tp_weight_sharder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LocalTPWeightSharder
    COMMAND $<TARGET_FILE:v2_test_local_tp_weight_sharder>
    LABELS "V2;Unit;LocalTP;WeightSharder;TensorParallel;ColumnParallel;RowParallel"
    MPI_PROCS 1
)

# Test: TPAllreduceStage (unified all-reduce stage for LOCAL + GLOBAL tensor parallelism)
# Phase 4 Global TP Unified Interface: Tests TPAllreduceStage with both LocalTPContext and
# GlobalTPContext. Validates ITPContext polymorphism, stage properties, buffer requirements,
# single-device no-op for both scopes, and is_global scalar in dump info.
add_executable(v2_test_tp_allreduce_stage unit/execution/compute_stages/stages/Test__TPAllreduceStage.cpp)
target_link_libraries(v2_test_tp_allreduce_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TPAllreduceStage
    COMMAND $<TARGET_FILE:v2_test_tp_allreduce_stage>
    LABELS "V2;Unit;TP;Allreduce;ComputeStage;TensorParallel;Collective;GlobalTP;LocalTP;ITPContext"
    MPI_PROCS 1
)

# Test: LocalPPTransferStage (activation transfer stage for LOCAL pipeline parallelism)
# Phase 6: Tests LocalPPTransferStage for intra-node GPU-to-GPU activation transfers
# Validates stage type/name, buffer requirements, same-stage no-op, mock transfer
# behavior, parameter validation, and dump info generation.
add_executable(v2_test_local_pp_transfer_stage unit/execution/compute_stages/stages/Test__LocalPPTransferStage.cpp)
target_link_libraries(v2_test_local_pp_transfer_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LocalPPTransferStage
    COMMAND $<TARGET_FILE:v2_test_local_pp_transfer_stage>
    LABELS "V2;Unit;LocalPP;Transfer;ComputeStage;PipelineParallel;ActivationTransfer"
    MPI_PROCS 1
)

# Test: LocalTPContext Barrier Synchronization
# Tests the NCCL-style barrier for host-staged heterogeneous allreduce
# Validates that multiple device threads rendezvous before data transfer
add_executable(v2_test_local_tp_context_barrier unit/collective/Test__LocalTPContextBarrier.cpp)
target_link_libraries(v2_test_local_tp_context_barrier PRIVATE llaminar2_core GTest::gtest_main)
add_v2_test(V2_Integration_LocalTPContextBarrier
    COMMAND $<TARGET_FILE:v2_test_local_tp_context_barrier>
    LABELS "V2;Integration;LocalTP;Barrier;Synchronization;Heterogeneous;Collective"
    MPI_PROCS 1
)

# Test: LocalTPContext CPU Barrier (HOST backend regression tests)
# Regression tests for CPU TP with HOST backend barrier-synchronized collectives.
# Covers: gpu_data_ptr nullptr crash, device_index slot collision, allgather deadlock.
add_executable(v2_test_local_tp_context_cpu_barrier unit/collective/Test__LocalTPContext_CpuBarrier.cpp)
target_link_libraries(v2_test_local_tp_context_cpu_barrier PRIVATE llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_LocalTPContext_CpuBarrier
    COMMAND $<TARGET_FILE:v2_test_local_tp_context_cpu_barrier>
    LABELS "V2;Unit;LocalTP;CpuBarrier;HOST;Allreduce;Allgather;Regression;Collective"
    MPI_PROCS 1
)

# Test: PPActivationStages (Pipeline Parallel activation stages)
# Phase 6.4: Tests SendActivationsStage and ReceiveActivationsStage interfaces
# Validates stage type, name generation, buffer requirements, dump info, and async support
# Note: Unit tests only - no actual MPI communication (use integration test for real P2P)
add_executable(v2_test_pp_activation_stages unit/execution/compute_stages/stages/Test__PPActivationStages.cpp)
target_link_libraries(v2_test_pp_activation_stages llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PPActivationStages
    COMMAND $<TARGET_FILE:v2_test_pp_activation_stages>
    LABELS "V2;Unit;ComputeStage;PipelineParallel;SendActivations;ReceiveActivations;MPI;P2P"
    MPI_PROCS 1
)

# Test: PipelineParallelGraphBuilder (PP stage insertion into compute graphs)
# Phase 3: Tests IPipelineParallelGraphBuilder interface for inserting Send/Recv stages
# Validates receive stage insertion (root node), send stage insertion (leaf node),
# graph dependency wiring, MPI tag uniqueness, and stage naming conventions.
# Note: Unit tests only - uses RankExecutionPlan configuration, no actual MPI
add_executable(v2_test_pipeline_parallel_graph_builder unit/execution/local_execution/graph/Test__PipelineParallelGraphBuilder.cpp)
target_link_libraries(v2_test_pipeline_parallel_graph_builder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PipelineParallelGraphBuilder
    COMMAND $<TARGET_FILE:v2_test_pipeline_parallel_graph_builder>
    LABELS "V2;Unit;PipelineParallel;GraphBuilder;SendActivations;ReceiveActivations;ComputeGraph"
    MPI_PROCS 1
)

# Test: PartialWeightLoader (layer-specific weight loading for pipeline parallelism)
# Phase 3: Tests IPartialWeightLoader interface for partial weight loading
# Validates weight name generation (attention/FFN), memory estimation, embedding/LM head handling,
# layer range filtering, and weight info construction.
# Part of memory-efficient PP where each rank loads only its assigned layers.
add_executable(v2_test_partial_weight_loader unit/loaders/Test__PartialWeightLoader.cpp)
target_link_libraries(v2_test_partial_weight_loader llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PartialWeightLoader
    COMMAND $<TARGET_FILE:v2_test_partial_weight_loader>
    LABELS "V2;Unit;PipelineParallel;WeightLoader;Memory;LayerAssignment"
    MPI_PROCS 1
)

# Test: WeightManager_LayerPartitioned (LAYER_PARTITIONED strategy for Pipeline Parallelism)
# Tests layer range filtering in WeightManager:
# - Embedding weight filtered by has_embedding flag
# - LM head weights filtered by has_lm_head flag  
# - Layer weights outside [first, last) return nullptr
# - Memory savings from per-stage weight loading
add_executable(v2_test_weightmanager_layer_partitioned unit/loaders/Test__WeightManager_LayerPartitioned.cpp)
target_link_libraries(v2_test_weightmanager_layer_partitioned llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_WeightManager_LayerPartitioned
    COMMAND $<TARGET_FILE:v2_test_weightmanager_layer_partitioned>
    LABELS "V2;Unit;PipelineParallel;WeightManager;LayerPartitioned;Memory"
    MPI_PROCS 1
)

# Test: CollectiveStageIntegration (AllreduceStage and AllGatherStage with CollectiveContext)
# Unit test for collective stages using the new collective infrastructure with mocks
add_executable(v2_unit_collective_stage_integration integration/collective/Test__CollectiveStageIntegration.cpp)
target_link_libraries(v2_unit_collective_stage_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CollectiveStageIntegration
    COMMAND $<TARGET_FILE:v2_unit_collective_stage_integration>
    LABELS "V2;Unit;ComputeStage;Collective;Allreduce;AllGather;MPI;TensorParallel"
    MPI_PROCS 1
)

# Test: HeterogeneousExecution - Unit test for mixed CPU/GPU execution across ranks
# Phase 5a testability refactoring: validates heterogeneous topology handling via mocks
# Note: Uses mock infrastructure, no real MPI/GPU required - runs as unit test
add_executable(v2_unit_heterogeneous_execution integration/execution/Test__HeterogeneousExecution.cpp)
target_include_directories(v2_unit_heterogeneous_execution PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_heterogeneous_execution 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HeterogeneousExecution 
    COMMAND $<TARGET_FILE:v2_unit_heterogeneous_execution>
    LABELS "V2;Unit;HeterogeneousExecution;Mock;MPI;Topology;DevicePlacement;CPU;GPU"
    MPI_PROCS 1
)

# Test: HeterogeneousLayerExecutorIntegration - Integration tests for executor + config
# Phase 5.4: Tests integration of HeterogeneousLayerExecutor with LayerPlacementConfig
# Verifies device routing, domain boundary detection, statistics tracking, and various
# placement patterns (first N, last N, alternating, multi-GPU, etc.)
# Note: Uses mocks, no real GPU required - runs as unit test
add_executable(v2_integration_heterogeneous_layer_executor integration/execution/model/Test__HeterogeneousLayerExecutorIntegration.cpp)
target_link_libraries(v2_integration_heterogeneous_layer_executor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HeterogeneousLayerExecutorPlacement
    COMMAND $<TARGET_FILE:v2_integration_heterogeneous_layer_executor>
    LABELS "V2;Unit;HeterogeneousExecution;LayerPlacement;DeviceRouting;DomainBoundary"
    MPI_PROCS 1
)

# =============================================================================
# Phase 5.5: End-to-End Heterogeneous Pipeline Integration Tests
# =============================================================================
# Comprehensive integration tests for the complete heterogeneous execution pipeline:
# - LayerPlacementConfig specifies which layers run where
# - HeterogeneousLayerExecutor routes execution correctly
# - CrossDomainTransfer handles activation transfers
# - DomainAwareBufferManager allocates buffers correctly

# Test: HeterogeneousPipeline - End-to-end heterogeneous pipeline tests
# Tests full pipeline with actual GPU hardware (graceful fallback to CPU-only)
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_heterogeneous_pipeline 
        integration/execution/model/Test__HeterogeneousPipeline.cpp
    )
    target_include_directories(v2_integration_heterogeneous_pipeline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    target_link_libraries(v2_integration_heterogeneous_pipeline 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_HeterogeneousPipeline 
        COMMAND $<TARGET_FILE:v2_integration_heterogeneous_pipeline>
        LABELS "V2;Integration;Heterogeneous;Pipeline;Phase5;GPU;CPU;Transfer;BufferManager"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added HeterogeneousPipeline end-to-end integration tests (GPU)")
else()
    # CPU-only version for systems without GPU
    add_executable(v2_integration_heterogeneous_pipeline 
        integration/execution/model/Test__HeterogeneousPipeline.cpp
    )
    target_include_directories(v2_integration_heterogeneous_pipeline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    target_link_libraries(v2_integration_heterogeneous_pipeline 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_HeterogeneousPipeline 
        COMMAND $<TARGET_FILE:v2_integration_heterogeneous_pipeline>
        LABELS "V2;Integration;Heterogeneous;Pipeline;Phase5;CPU"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added HeterogeneousPipeline end-to-end integration tests (CPU-only)")
endif()

# Test: HeterogeneousPipelineMPI - Multi-rank heterogeneous pipeline tests
# MPI-aware tests for cross-rank domain configuration and collective operations
add_executable(v2_integration_heterogeneous_pipeline_mpi 
    integration/execution/model/Test__HeterogeneousPipelineMPI.cpp
)
target_include_directories(v2_integration_heterogeneous_pipeline_mpi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
target_link_libraries(v2_integration_heterogeneous_pipeline_mpi 
    llaminar2_core 
    GTest::gtest
    # Note: No gtest_main - test has custom MPI main
)
add_v2_test(V2_Integration_HeterogeneousPipelineMPI 
    COMMAND $<TARGET_FILE:v2_integration_heterogeneous_pipeline_mpi>
    LABELS "V2;Integration;Heterogeneous;Pipeline;Phase5;MPI;CrossRank;TPDomain;Collective"
    MPI_PROCS 2
)
message(STATUS "V2 Tests: Added HeterogeneousPipelineMPI multi-rank integration tests")

# Test: WorkDistribution - Integration test for work distribution across ranks
# Phase 5b testability refactoring: validates tensor-parallel work distribution via mocks
# Note: Uses mock infrastructure, no real MPI required - runs as unit test
add_executable(v2_integration_work_distribution integration/execution/Test__WorkDistribution.cpp)
target_include_directories(v2_integration_work_distribution PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_work_distribution 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_WorkDistribution 
    COMMAND $<TARGET_FILE:v2_integration_work_distribution>
    LABELS "V2;Unit;WorkDistribution;TensorParallel;Mock;MPI;GEMM;Attention"
    MPI_PROCS 1
)

# Test: WeightStreaming - Unit test for weight streaming with prefetch/eviction
# Phase 5c testability refactoring: validates weight streaming via mocks
# Note: Uses mock infrastructure, no real GPU memory management - runs as unit test
add_executable(v2_integration_weight_streaming integration/loaders/Test__WeightStreaming.cpp)
target_include_directories(v2_integration_weight_streaming PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_weight_streaming 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_WeightStreaming 
    COMMAND $<TARGET_FILE:v2_integration_weight_streaming>
    LABELS "V2;Unit;WeightStreaming;Prefetch;Eviction;LRU;FIFO;Mock"
    MPI_PROCS 1
)

# Test: PhaseAwareExecution - Integration test for prefill vs decode device selection
# Phase 5d testability refactoring: validates phase-aware device selection via mocks
# Note: Uses mock infrastructure, no real MPI/GPU required - runs as unit test
add_executable(v2_integration_phase_aware_execution integration/execution/Test__PhaseAwareExecution.cpp)
target_include_directories(v2_integration_phase_aware_execution PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_phase_aware_execution 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PhaseAwareExecution 
    COMMAND $<TARGET_FILE:v2_integration_phase_aware_execution>
    LABELS "V2;Unit;PhaseAware;Prefill;Decode;DeviceSelection;Mock;MPI;Topology;CPU;GPU"
    MPI_PROCS 1
)

# =============================================================================
# Orchestration Scenario Tests (Declarative, Base Class Pattern)
# =============================================================================
# These tests follow the parity test pattern with OrchestrationTestBase
# providing common infrastructure and scenario files being declarative.
# See: tests/v2/integration/orchestration/README.md

# Test: Scenario 1 - Heterogeneous Single Machine (2 ranks, 1 machine, mixed CUDA/ROCm)
add_executable(v2_orchestration_scenario1
    integration/orchestration/scenarios/Test__Scenario1_HeterogeneousSingleMachine.cpp
)
target_include_directories(v2_orchestration_scenario1 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario1 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario1_HeterogeneousSingleMachine 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario1>
    LABELS "V2;Unit;Orchestration;Scenario;Heterogeneous;Placement;PlacementStrategy;Mock;MPI;Topology;CUDA;ROCm;CPU;GPU;Qwen2;7B;72B;235B"
    MPI_PROCS 1
)

# Test: Scenario 2 - Multi-Node Cluster (12 ranks, 6 machines, Infiniband, mixed CUDA/ROCm)
add_executable(v2_orchestration_scenario2
    integration/orchestration/scenarios/Test__Scenario2_MultiNodeCluster.cpp
)
target_include_directories(v2_orchestration_scenario2 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario2 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario2_MultiNodeCluster 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario2>
    LABELS "V2;Unit;Orchestration;Scenario;MultiNode;Cluster;Placement;PlacementStrategy;Mock;MPI;Topology;CUDA;ROCm;CPU;GPU;Infiniband;NVLink;xGMI;Qwen;7B;72B;235B;571B;1142B"
    MPI_PROCS 1
)

# Test: Scenario 3 - Single Machine with CPU Participation (1 GPU socket + 1 CPU-only socket)
add_executable(v2_orchestration_scenario3
    integration/orchestration/scenarios/Test__Scenario3_CPUParticipation.cpp
)
target_include_directories(v2_orchestration_scenario3 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario3 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario3_CPUParticipation 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario3>
    LABELS "V2;Unit;Orchestration;Scenario;CPUParticipation;Hybrid;Placement;PlacementStrategy;Mock;MPI;Topology;CUDA;CPU;GPU;Qwen2;0.5B;7B;72B;Heterogeneous"
    MPI_PROCS 1
)

# Test: Scenario 4 - Cross-Vendor TP (single socket, CUDA + ROCm, PCIe-based TP)
add_executable(v2_orchestration_scenario4
    integration/orchestration/scenarios/Test__Scenario4_CrossVendorTP.cpp
)
target_include_directories(v2_orchestration_scenario4 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario4 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario4_CrossVendorTP 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario4>
    LABELS "V2;Unit;Orchestration;Scenario;CrossVendor;TP;TensorParallel;Placement;PlacementStrategy;Mock;MPI;Topology;CUDA;ROCm;GPU;PCIe;Qwen2;0.5B;7B;72B;Heterogeneous"
    MPI_PROCS 1
)

# Test: Scenario 5 - Multi-Node Pipeline Parallel (2 nodes, HDR InfiniBand, PP)
add_executable(v2_orchestration_scenario5
    integration/orchestration/scenarios/Test__Scenario5_MultiNodePP.cpp
)
target_include_directories(v2_orchestration_scenario5 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario5 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario5_MultiNodePP 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario5>
    LABELS "V2;Unit;Orchestration;Scenario;MultiNode;PipelineParallel;PP;Placement;PlacementStrategy;Mock;MPI;Topology;CUDA;GPU;A100;Infiniband;NVLink;Qwen2;7B;72B;235B;571B"
    MPI_PROCS 1
)

# Test: Scenario 6 - Single Socket, Asymmetric VRAM (24GB RTX 4090 + 80GB A100)
add_executable(v2_orchestration_scenario6
    integration/orchestration/scenarios/Test__Scenario6_AsymmetricVRAM.cpp
)
target_include_directories(v2_orchestration_scenario6 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario6 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario6_AsymmetricVRAM 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario6>
    LABELS "V2;Unit;Orchestration;Scenario;AsymmetricVRAM;VRAM;Proportional;Placement;PlacementStrategy;Mock;MPI;Topology;CUDA;GPU;RTX4090;A100;NVLink;Qwen2;0.5B;7B;72B;Heterogeneous"
    MPI_PROCS 1
)

# Test: Scenario 7 - 2-Node Heterogeneous Pipeline Cluster (Cross-Vendor TP + Intra-Node PP)
add_executable(v2_orchestration_scenario7
    integration/orchestration/scenarios/Test__Scenario7_HeterogeneousPipelineCluster.cpp
)
target_include_directories(v2_orchestration_scenario7 PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}           # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_orchestration_scenario7 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Orchestration_Scenario7_HeterogeneousPipelineCluster 
    COMMAND $<TARGET_FILE:v2_orchestration_scenario7>
    LABELS "V2;Unit;Orchestration;Scenario;HeterogeneousPipeline;CrossVendor;IntraNode;PP;PipelineParallel;TP;TensorParallel;CUDA;ROCm;GPU;RTX3090;MI50;UPI;Infiniband;MultiNode;Qwen2;7B;72B;120B;NUMA"
    MPI_PROCS 1
)

# Test: PipelineExecutor (LayerExecutor-to-pipeline adapter)
# Test: GQAAttention Second Sequence Handling
# Tests that sequence 1 in batched attention is handled correctly
# Validates buffer offsets, mask application, and cross-sequence isolation

# Test: ModelLoader
add_executable(v2_test_model_loader integration/loaders/Test__ModelLoader.cpp)
target_link_libraries(v2_test_model_loader 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_ModelLoader 
    COMMAND $<TARGET_FILE:v2_test_model_loader>
    MPI_PROCS 1
    LABELS "V2;Integration;ModelLoading;GGUF;WeightLoading"
)

# Test: ModelLoader Split (multi-part GGUF)
add_executable(v2_test_model_loader_split integration/loaders/Test__ModelLoaderSplit.cpp)
target_link_libraries(v2_test_model_loader_split 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_ModelLoaderSplit 
    COMMAND $<TARGET_FILE:v2_test_model_loader_split>
    MPI_PROCS 1
    LABELS "V2;Integration;ModelLoading;GGUF;SplitGGUF"
)

# Test: ModelLoader Row Slice (memory-efficient tensor row slicing from GGUF)
add_executable(v2_test_model_loader_row_slice unit/loaders/Test__ModelLoaderRowSlice.cpp)
target_link_libraries(v2_test_model_loader_row_slice 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ModelLoaderRowSlice 
    COMMAND $<TARGET_FILE:v2_test_model_loader_row_slice>
    MPI_PROCS 1
    LABELS "V2;Unit;ModelLoading;GGUF;RowSlice;MemoryEfficient"
)

# Test: WeightManager Memory Efficient Slicing
add_executable(v2_test_weight_manager_memory_efficient integration/loaders/Test__WeightManagerMemoryEfficient.cpp)
target_link_libraries(v2_test_weight_manager_memory_efficient 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_WeightManagerMemoryEfficient 
    COMMAND $<TARGET_FILE:v2_test_weight_manager_memory_efficient>
    MPI_PROCS 2
    LABELS "V2;Integration;WeightManager;TensorSlice;MemoryEfficient;Sharding"
)

# Test: Unified GPU Pipeline (dense + MoE weights in single pipeline)
# Loads Qwen3.5-35B-A3B Q4_K model and verifies:
# - ExpertGemmRegistry populated for all layers/experts
# - Host data released for 3D MoE tensors
# - Dense and MoE weights in same VRAM pool
# - Expert GEMM engines produce correct matmul results
# Model: /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
add_executable(v2_integration_unified_gpu_pipeline integration/loaders/Test__UnifiedGPUPipeline.cpp)
target_include_directories(v2_integration_unified_gpu_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes
)
target_link_libraries(v2_integration_unified_gpu_pipeline
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_UnifiedGPUPipeline
    COMMAND $<TARGET_FILE:v2_integration_unified_gpu_pipeline>
    MPI_PROCS 1
    LABELS "V2;Integration;WeightManager;MoE;GPU;Pipeline"
)

# Integration Test: Unified GPU Pipeline with Floating-Point (BF16) Model
add_executable(v2_integration_unified_gpu_pipeline_fp integration/loaders/Test__UnifiedGPUPipeline_FP.cpp)
target_include_directories(v2_integration_unified_gpu_pipeline_fp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes
)
target_link_libraries(v2_integration_unified_gpu_pipeline_fp
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_UnifiedGPUPipelineFP
    COMMAND $<TARGET_FILE:v2_integration_unified_gpu_pipeline_fp>
    MPI_PROCS 1
    LABELS "V2;Integration;WeightManager;MoE;GPU;Pipeline;FP;BF16"
)

# Test: WeightManager Proportional Slicing (TensorParallelConfig support)
add_executable(v2_test_weight_manager_proportional integration/loaders/Test__WeightManagerProportional.cpp)
target_link_libraries(v2_test_weight_manager_proportional 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_WeightManagerProportional 
    COMMAND $<TARGET_FILE:v2_test_weight_manager_proportional>
    MPI_PROCS 1
    LABELS "V2;Integration;WeightManager;TensorParallel;Proportional;Heterogeneous"
)

# Test: WeightManager Multi-Device (device-aware features integration)
# Tests multi-device weight isolation, concurrent access, real GPU packing,
# and end-to-end preload workflow for LOCAL tensor parallelism
add_executable(v2_integration_weight_manager_multi_device integration/loaders/Test__WeightManager_MultiDevice.cpp)
target_include_directories(v2_integration_weight_manager_multi_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ and tests/v2/utils/ includes
)
target_link_libraries(v2_integration_weight_manager_multi_device 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_WeightManager_MultiDevice 
    COMMAND $<TARGET_FILE:v2_integration_weight_manager_multi_device>
    MPI_PROCS 1
    LABELS "V2;Integration;WeightManager;MultiDevice;LocalTP;TensorParallel;GPU;Concurrent"
)

# Test: ModelContext PP Stage Loading (Pipeline Parallelism layer filtering)
# Tests that ModelContext with ModelContextConfig correctly loads partial weights
# for Pipeline Parallelism stages, validating layer filtering and global weight filtering
add_executable(v2_integration_modelcontext_ppstage integration/loaders/Test__ModelContext_PPStage.cpp)
target_link_libraries(v2_integration_modelcontext_ppstage 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_ModelContext_PPStage 
    COMMAND $<TARGET_FILE:v2_integration_modelcontext_ppstage>
    MPI_PROCS 1
    LABELS "V2;Integration;Loaders;PipelineParallelism;ModelContext;LayerPartitioning"
)

# Test: QwenStandardGraph with TensorParallelConfig (Phase 1c: Proportional TP)
add_executable(v2_test_qwen_standard_graph_proportional_tp integration/models/qwen/Test__QwenStandardGraphProportionalTP.cpp)
target_link_libraries(v2_test_qwen_standard_graph_proportional_tp 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_QwenStandardGraphProportionalTP 
    COMMAND $<TARGET_FILE:v2_test_qwen_standard_graph_proportional_tp>
    MPI_PROCS 1
    LABELS "V2;Integration;QwenStandardGraph;TensorParallel;Proportional;Heterogeneous;Phase1c"
)

# Test: QwenStandardGraph with MultiDomainTPConfig (Phase 4.3: Heterogeneous TP Domains)
# Tests that QwenStandardGraph correctly passes TPDomain to AllreduceStage for
# domain-based collective routing in heterogeneous tensor parallelism scenarios.
add_executable(v2_test_qwen_standard_graph_domain unit/models/qwen/Test__QwenStandardGraphDomain.cpp)
target_link_libraries(v2_test_qwen_standard_graph_domain 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QwenStandardGraphDomain 
    COMMAND $<TARGET_FILE:v2_test_qwen_standard_graph_domain>
    MPI_PROCS 1
    LABELS "V2;Unit;QwenStandardGraph;TensorParallel;TPDomain;MultiDomain;Heterogeneous;Phase4_3"
)

# Test: QwenStandardGraph buildUnifiedPipelineGraph (Pipeline Parallelism graph building)
# Tests the unified PP graph builder that creates compute graphs for PP + TP composition.
add_executable(v2_test_qwen_standard_graph_pp unit/models/qwen/Test__QwenStandardGraph_PP.cpp)
target_link_libraries(v2_test_qwen_standard_graph_pp 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QwenStandardGraph_PP 
    COMMAND $<TARGET_FILE:v2_test_qwen_standard_graph_pp>
    MPI_PROCS 1
    LABELS "V2;Unit;Models;Qwen2;QwenStandardGraph;PipelineParallelism;UnifiedPPGraph"
)

# Test: PP absolute layer range validation (regression test for LocalPP crash)
add_executable(v2_test_qwen_standard_graph_pp_layer_range unit/models/qwen/Test__QwenStandardGraph_PPLayerRange.cpp)
target_link_libraries(v2_test_qwen_standard_graph_pp_layer_range 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QwenStandardGraph_PPLayerRange 
    COMMAND $<TARGET_FILE:v2_test_qwen_standard_graph_pp_layer_range>
    MPI_PROCS 1
    LABELS "V2;Unit;Models;Qwen2;QwenStandardGraph;PipelineParallelism;Regression"
)

# Test: PlacementStrategy and PlacementPlan (weight/compute placement algorithms)
add_executable(v2_test_placement_strategy unit/execution/mpi_orchestration/Test__PlacementStrategy.cpp)
target_link_libraries(v2_test_placement_strategy 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PlacementStrategy 
    COMMAND $<TARGET_FILE:v2_test_placement_strategy>
    MPI_PROCS 1
    LABELS "V2;Unit;Placement;Strategy;WeightDistribution;DeviceMapping"
)

# Test: MPITopology Capability Exchange (serialization/deserialization of RankInventory)
add_executable(v2_test_mpi_topology_capability_exchange unit/utils/Test__MPITopologyCapabilityExchange.cpp)
target_link_libraries(v2_test_mpi_topology_capability_exchange 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MPITopologyCapabilityExchange 
    COMMAND $<TARGET_FILE:v2_test_mpi_topology_capability_exchange>
    MPI_PROCS 1
    LABELS "V2;Unit;MPI;Topology;Serialization;DeviceInventory;Heterogeneous"
)

# Test: NodeDetection (canonical hostname-based node detection)
# Note: Uses mpi_gtest_main.cpp for MPI initialization (needed for detect() with MPI_COMM_SELF)
add_executable(v2_test_node_detection
    unit/utils/Test__NodeDetection.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_test_node_detection
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Unit_NodeDetection
    COMMAND $<TARGET_FILE:v2_test_node_detection>
    MPI_PROCS 1
    LABELS "V2;Unit;MPI;Topology;NodeDetection"
)

# Test: HeterogeneousMultiDomainStrategy (multi-domain TP + PP for heterogeneous clusters)
add_executable(v2_test_heterogeneous_multi_domain_strategy unit/execution/mpi_orchestration/placement/Test__HeterogeneousMultiDomainStrategy.cpp)
target_link_libraries(v2_test_heterogeneous_multi_domain_strategy 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HeterogeneousMultiDomainStrategy 
    COMMAND $<TARGET_FILE:v2_test_heterogeneous_multi_domain_strategy>
    MPI_PROCS 1
    LABELS "V2;Unit;Placement;Strategy;Heterogeneous;MultiDomain;TPDomain;PipelineParallel"
)

# Test: Phase 6 Heterogeneous Domain Integration (Phase 6.6)
# End-to-end integration tests for heterogeneous multi-domain parallelism:
# - Capability exchange (MPITopology)
# - HeterogeneousMultiDomainStrategy
# - Domain wiring in DeviceGraphOrchestrator
# - PP Send/Recv stages
# - Plan broadcast and deterministic computation
add_executable(v2_integration_phase6_domain integration/execution/Test__Phase6HeterogeneousDomainIntegration.cpp)
target_link_libraries(v2_integration_phase6_domain 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Phase6_HeterogeneousDomain 
    COMMAND $<TARGET_FILE:v2_integration_phase6_domain>
    MPI_PROCS 2
    LABELS "V2;Integration;Phase6;Heterogeneous;Domain;MultiDomain;TPDomain;PipelineParallel;MPI;CapabilityExchange"
)

# Test: PP+TP Buffer Pool Integration
# Test: Quantised GEMM Sliced Parity (verifies sliced GEMM across MPI ranks matches unsliced)
add_executable(v2_test_quantised_gemm_sliced_parity integration/kernels/cpu/gemm/Test__QuantisedGemmSlicedParity.cpp)
target_link_libraries(v2_test_quantised_gemm_sliced_parity 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_QuantisedGemmSlicedParity 
    COMMAND $<TARGET_FILE:v2_test_quantised_gemm_sliced_parity>
    MPI_PROCS 2
    LABELS "V2;Integration;GEMM;Slicing;MPI;Parity;TensorParallel"
)

# Test: Q16 MPI Allreduce (variable block size allreduce across 2 MPI ranks)
# Validates templated allreduce_q16_inplace<BlockType>() for BLOCK_32/64/128/192
add_executable(v2_integration_q16_mpi_allreduce integration/utils/mpi/Test__Q16MPI_Allreduce.cpp)
target_link_libraries(v2_integration_q16_mpi_allreduce 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q16MPI_Allreduce 
    COMMAND $<TARGET_FILE:v2_integration_q16_mpi_allreduce>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Q16;Allreduce;VariableBlockSize;TensorParallel"
)

# Test: Stage Dump Integrity
# Integration test that verifies stage dump infrastructure:
# - Layer filtering works correctly (only requested layers are dumped)
# - Dump file sizes match metadata byte_size
# - Block counts in metadata are correct for quantized formats
# - Metadata dimensions produce correct block counts
add_executable(v2_integration_stage_dump_integrity integration/execution/debug/Test__StageDumpIntegrity.cpp)
target_link_libraries(v2_integration_stage_dump_integrity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_StageDumpIntegrity
    COMMAND $<TARGET_FILE:v2_integration_stage_dump_integrity>
    MPI_PROCS 1
    LABELS "V2;Integration;StageDump;Debug;Infrastructure"
)

# Test: Qwen2Pipeline Integration Tests (full pipeline, autoregressive decode, bias validation)
# Validates end-to-end pipeline functionality including Q/K/V/O bias application

# Test: KVCache Integration Tests (KV cache behavior during autoregressive decode)
# Validates cache accumulation, data integrity, position tracking, and attention correctness

# Test: Graph Batched KVCache Integration Tests (Phase 3 of Graph Batching Implementation)
# Validates batched KV cache handling in Graph framework:
# - KVCacheAppendStage correctly appends to separate seq_idx per batch sequence
# - Batched prefill populates KV cache slots independently
# - Sequential execution matches batched execution for KV cache contents
# - Per-sequence cache isolation
add_executable(v2_integration_graph_batched_kvcache integration/execution/graph/Test__GraphBatchedKVCache.cpp)
target_link_libraries(v2_integration_graph_batched_kvcache
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_GraphBatchedKVCache
    COMMAND $<TARGET_FILE:v2_integration_graph_batched_kvcache>
    LABELS "V2;Integration;KVCache;Graph;Batched;ComputeStage;Phase3"
    MPI_PROCS 1
)

# Test: Q8_1 KV Cache Integration Tests
# Validates Q8_1 attention kernel correctly integrates with typed KV cache
# Tests: roundtrip, attention with cache, direct vs cached K/V, GQA path, block copy semantics

# Test: Decomposed Attention (Phase 9 KVCacheAppendStage + AttentionComputeStage)
# Validates the new decomposed attention path that routes through KernelFactory

# Test: Pipeline vs LayerExecutor FP32 Parity
# Validates that the new LayerExecutor framework produces equivalent results
# to the legacy imperative pipeline code when both use FP32 precision.
# Tests prefill, single decode, and multi-step decode scenarios.

# Test: GraphBufferManagement Parity
# Validates that graph-managed buffer allocation produces equivalent results
# to pipeline-managed buffer allocation. Part of Phase 5: QwenStandardGraph Integration.
# Both modes use LayerExecutor path, comparing LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=0 vs =1.

# Test: Full Graph Forward vs Legacy Parity
# Validates that the full graph forward path (orchestrator->executeForward())
# produces equivalent results to the legacy imperative pipeline code.
# Part of Phase 4: Integration Testing & Parity Verification for Full Graph Inference.

# Test: Pipeline vs LayerExecutor Q8_1 Parity
# Validates that the LayerExecutor framework produces equivalent results
# to the legacy pipeline code when BOTH use Q8_1 activation precision.
# Uses looser tolerances to account for quantization noise inherent to Q8_1.

# Test: Q8_1 MPI Tensor-Parallel Attention
# Tests the MPI tensor-parallel path for Q8_1 attention:
#   - Head slicing (sliceQ8_1HeadBlocks)
#   - K/V GQA broadcast (broadcastQ8_1KVHeads)
#   - Fused attention kernel execution
#   - Allgatherv result combination

# Test: Precision-Aware MPI Allreduce Integration Tests
# Tests actual MPI communication for FP32, FP16, BF16, Q8_1, and Q16_1 formats
# Validates correctness of partial sum accumulation across ranks
add_executable(v2_integration_mpi_precision_allreduce integration/utils/mpi/Test__MPI_PrecisionAwareAllreduce.cpp)
target_link_libraries(v2_integration_mpi_precision_allreduce
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_PrecisionAwareAllreduce
    COMMAND $<TARGET_FILE:v2_integration_mpi_precision_allreduce>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Allreduce;FP32;FP16;BF16;Q8_1;Q16_1;Bandwidth"
)

# Test: Point-to-Point MPI Operations (Pipeline Parallelism Foundation)
# Tests send/recv, isend/irecv, wait, waitAll, probe, iprobe operations:
# - Synchronous and async point-to-point communication
# - Large buffer transfers (1MB)
# - Multiple concurrent requests with waitAll
# - Message probing before receive
# - Tag-based message differentiation
# - Bidirectional communication patterns
add_executable(v2_integration_mpi_context_p2p 
    integration/utils/mpi/Test__MPIContextP2P.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_integration_mpi_context_p2p
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_ContextP2P
    COMMAND $<TARGET_FILE:v2_integration_mpi_context_p2p>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;P2P;Send;Recv;PipelineParallel"
)

# Test: Pipeline Activation Transfer (Phase 2.2 - P2P stages)
# Tests SendActivationsStage and ReceiveActivationsStage for pipeline parallelism:
# - Sync/async send/recv
# - Bidirectional exchange
# - Multi-layer sequence with tags
# - Error handling
add_executable(v2_integration_pipeline_activation_transfer
    integration/execution/Test__PipelineActivationTransfer.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_integration_pipeline_activation_transfer
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_PipelineActivationTransfer
    COMMAND $<TARGET_FILE:v2_integration_pipeline_activation_transfer>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;P2P;PipelineParallel;Stages;SendActivations;ReceiveActivations"
)

# Test: Row-Parallel Multi-Precision MPI Integration Tests
# Tests TensorSlice and k-sliced row-parallel GEMM with FP32, BF16, FP16, Q8_1
# Validates allgatherv (TensorSlice) and allreduce-sum (k-sliced) patterns

# Test: Column-Parallel QKV MPI Integration Tests (Phase 3)
# Tests column-parallel weight sharding for Q/K/V projections:
# - ShardingMode detection for attn_q/k/v weights and biases
# - Local head distribution calculation
# - Column-sliced Q/K/V weight GEMM correctness
# - Buffer dimension validation for local heads
add_executable(v2_integration_mpi_column_parallel_qkv integration/utils/mpi/Test__MPI_ColumnParallelQKV.cpp)
target_link_libraries(v2_integration_mpi_column_parallel_qkv
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_ColumnParallelQKV
    COMMAND $<TARGET_FILE:v2_integration_mpi_column_parallel_qkv>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;ColumnParallel;QKV;TensorParallel;Attention;Phase3"
)

# Test: Column-Parallel FFN (Phase 4: Gate/Up Column-Parallel, Down Input-Parallel)
# Tests column-parallel weight sharding for FFN projections:
# - ShardingMode detection for Gate/Up (COLUMN_PARALLEL) and Down (INPUT_PARALLEL)
# - Local FFN dimension calculation (d_ff_local = d_ff / world_size)
# - Column-sliced Gate/Up weight shapes and GEMM correctness
# - Input-sliced Down weight shape
# - Buffer dimension validation for local FFN
add_executable(v2_integration_mpi_column_parallel_ffn integration/utils/mpi/Test__MPI_ColumnParallelFFN.cpp)
target_link_libraries(v2_integration_mpi_column_parallel_ffn
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_ColumnParallelFFN
    COMMAND $<TARGET_FILE:v2_integration_mpi_column_parallel_ffn>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;ColumnParallel;FFN;TensorParallel;Phase4"
)

# Test: Column-Parallel LM Head (Phase 5: Output Weight Column-Parallel + AllGather)
# Tests column-parallel weight sharding for LM head projection:
# - ShardingMode detection for output.weight (COLUMN_PARALLEL)
# - Local vocab dimension calculation (vocab_local = vocab_size / world_size)
# - Column-sliced LM head weight shape
# - AllGather stage collects distributed logits
# - Full forward data flow with distributed LM head
add_executable(v2_integration_mpi_column_parallel_lmhead integration/utils/mpi/Test__MPI_ColumnParallelLMHead.cpp)
target_link_libraries(v2_integration_mpi_column_parallel_lmhead
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_ColumnParallelLMHead
    COMMAND $<TARGET_FILE:v2_integration_mpi_column_parallel_lmhead>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;ColumnParallel;LMHead;AllGather;TensorParallel;Phase5"
)

# Test: Sharded KV Cache (Phase 6 - Tensor Parallelism Memory Optimization)
# Tests sharded KV cache where each MPI rank stores only its local KV heads.
# Validates:
# - Each rank creates independent local cache (no MPI communication needed)
# - Tensor shapes are reduced proportionally (local_n_kv_heads * head_dim)
# - Append/clear operations work independently per rank
# - Memory usage is reduced by world_size factor
# - All precisions (FP32, BF16, FP16, Q8_1) work correctly
add_executable(v2_integration_mpi_sharded_kv_cache integration/utils/mpi/Test__MPI_ShardedKVCache.cpp)
target_link_libraries(v2_integration_mpi_sharded_kv_cache
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_ShardedKVCache
    COMMAND $<TARGET_FILE:v2_integration_mpi_sharded_kv_cache>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;KVCache;Sharding;TensorParallelism;Phase6;DistributedArchitecture"
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

# Integration Test: CPU NativeVNNI GEMV/GEMM Correctness (Q4_0, IQ4_NL)
add_executable(v2_integration_cpu_native_vnni_gemv integration/kernels/cpu/native_vnni/Test__CPUNativeVNNI_GEMV.cpp)
target_link_libraries(v2_integration_cpu_native_vnni_gemv
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_integration_cpu_native_vnni_gemv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes (TestTensorFactory)
)
add_v2_test(V2_Integration_CPUNativeVNNI_GEMV
    COMMAND $<TARGET_FILE:v2_integration_cpu_native_vnni_gemv>
    LABELS "V2;Integration;GEMM;GEMV;NativeVNNI;CPU;Q4_0;IQ4_NL;Decode"
    MPI_PROCS 1
)

# Performance Test: CPU NativeVNNI GEMV/GEMM Sweep (Q4_0, all model shapes)
add_executable(v2_perf_cpu_native_vnni_sweep performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_Sweep.cpp)
target_link_libraries(v2_perf_cpu_native_vnni_sweep
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_perf_cpu_native_vnni_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes (TestTensorFactory)
)
add_v2_test(V2_Perf_CPUNativeVNNI_Sweep
    COMMAND $<TARGET_FILE:v2_perf_cpu_native_vnni_sweep>
    LABELS "V2;Performance;GEMM;GEMV;NativeVNNI;CPU;Q4_0;Decode;Prefill;Qwen;Tuning"
    MPI_PROCS 1
)

# Performance Test: AVX2 vs AVX512 VNNI throughput comparison
add_executable(v2_perf_avx2_vs_avx512 performance/kernels/cpu/native_vnni/Perf__AVX2vsAVX512.cpp)
target_link_libraries(v2_perf_avx2_vs_avx512
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_perf_avx2_vs_avx512 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Perf_AVX2vsAVX512
    COMMAND $<TARGET_FILE:v2_perf_avx2_vs_avx512>
    LABELS "V2;Performance;GEMM;GEMV;NativeVNNI;CPU;AVX2;AVX512;Parity"
    MPI_PROCS 1
)

# Performance Test: CPU NativeVNNI Dispatch Parameter Sweep (tile heuristic tuning)
add_executable(v2_perf_cpu_native_vnni_dispatch_sweep performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_DispatchSweep.cpp)
target_link_libraries(v2_perf_cpu_native_vnni_dispatch_sweep
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_perf_cpu_native_vnni_dispatch_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes (TestTensorFactory)
)
add_v2_test(V2_Perf_CPUNativeVNNI_DispatchSweep
    COMMAND $<TARGET_FILE:v2_perf_cpu_native_vnni_dispatch_sweep>
    LABELS "V2;Performance;GEMM;GEMV;NativeVNNI;CPU;Dispatch;Tuning;Qwen"
    MPI_PROCS 1
)

# Performance Test: CPU NativeVNNI GEMV Decode Sweep (native Q8_0 vs packed VNNI, all models)
add_executable(v2_perf_cpu_native_vnni_gemv performance/kernels/cpu/native_vnni/Perf__CPUNativeVNNI_GEMV.cpp)
target_link_libraries(v2_perf_cpu_native_vnni_gemv
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_perf_cpu_native_vnni_gemv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ includes (TestTensorFactory)
)
add_v2_test(V2_Perf_CPUNativeVNNI_GEMV
    COMMAND $<TARGET_FILE:v2_perf_cpu_native_vnni_gemv>
    LABELS "V2;Performance;GEMV;NativeVNNI;CPU;Q8_0;Decode;Qwen;Tuning"
    MPI_PROCS 1
)

# Performance Test: CPU Flash Attention Sweep (all models, TP degrees, decode + prefill)
add_executable(v2_perf_cpu_flash_attention_sweep performance/kernels/cpu/attention/Perf__CpuFlashAttentionSweep.cpp)
target_link_libraries(v2_perf_cpu_flash_attention_sweep
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_CPUFlashAttentionSweep
    COMMAND $<TARGET_FILE:v2_perf_cpu_flash_attention_sweep>
    LABELS "V2;Performance;Attention;FlashAttention;CPU;Decode;Prefill;Qwen;Tuning"
    MPI_PROCS 1
)

# Performance Test: NativeVNNI vs oneDNN Reference (INT8 matmul baseline comparison)
if(ONEDNN_LIB)
    add_executable(v2_perf_onednn_reference performance/kernels/cpu/native_vnni/Perf__OneDNN_Reference.cpp)
    target_link_libraries(v2_perf_onednn_reference
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        ${ONEDNN_LIB}
    )
    target_include_directories(v2_perf_onednn_reference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_v2_test(V2_Perf_OneDNN_Reference
        COMMAND $<TARGET_FILE:v2_perf_onednn_reference>
        LABELS "V2;Performance;GEMV;NativeVNNI;CPU;OneDNN;Reference"
        MPI_PROCS 1
    )
    message(STATUS "oneDNN reference benchmark: ${ONEDNN_LIB}")
else()
    message(STATUS "oneDNN not available - skipping V2_Perf_OneDNN_Reference")
endif()

# Performance Test: Q8_1 Quantize SIMD (Scalar/AVX2/AVX512 throughput comparison)
add_executable(v2_perf_q8_1_quantize_simd performance/Perf__Q8_1_QuantizeSIMD.cpp)
target_link_libraries(v2_perf_q8_1_quantize_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_Q8_1_QuantizeSIMD 
    COMMAND $<TARGET_FILE:v2_perf_q8_1_quantize_simd>
    LABELS "V2;Performance;Quantization;Q8_1;SIMD;AVX2;AVX512"
)

# Performance Test: CPU sampler top-k distribution before/after fast path
add_executable(v2_perf_cpu_sampler_topk performance/Perf__CPUSamplerTopK.cpp)
target_link_libraries(v2_perf_cpu_sampler_topk
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_CPUSamplerTopK
    COMMAND $<TARGET_FILE:v2_perf_cpu_sampler_topk>
    LABELS "V2;Performance;Sampling;CPU;SIMD;AVX2;AVX512;TopK;MTP"
    MPI_PROCS 1
)

# Performance Test: MTP dynamic depth controller overhead
add_executable(v2_perf_mtp_depth_controller performance/Perf__MTPDepthController.cpp)
target_link_libraries(v2_perf_mtp_depth_controller
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_MTPDepthController
    COMMAND $<TARGET_FILE:v2_perf_mtp_depth_controller>
    LABELS "V2;Performance;MTP;DepthController;CPU"
    MPI_PROCS 1
)

# Test: IQ4_XS Tensor Views
add_executable(v2_test_iq4xs_views unit/tensors/Test__IQ4_XSTensor_Views.cpp)
target_link_libraries(v2_test_iq4xs_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_XS_Views 
    COMMAND $<TARGET_FILE:v2_test_iq4xs_views>
    LABELS "V2;Unit;TensorOperations;IQ4_XS;Quantization;ViewSupport;RowSlice;IQuant"
)

# Test: IQ2_XXS Tensor Views
add_executable(v2_test_iq2xxs_views unit/tensors/Test__IQ2_XXSTensor_Views.cpp)
target_link_libraries(v2_test_iq2xxs_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XXS_Views 
    COMMAND $<TARGET_FILE:v2_test_iq2xxs_views>
    LABELS "V2;Unit;TensorOperations;IQ2_XXS;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
)

# Test: IQ2_XS Tensor Views
add_executable(v2_test_iq2_xs_simd unit/tensors/Test__IQ2_XSTensor.cpp)
target_link_libraries(v2_test_iq2_xs_simd llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_IQ2_XSTensor
    COMMAND v2_test_iq2_xs_simd
    LABELS "V2;Unit;TensorOperations;Quantization;IQ2_XS"
)

# Test: IQ3_XXS Tensor Views
add_executable(v2_test_iq3_xxs_simd unit/tensors/Test__IQ3_XXSTensor.cpp)
target_link_libraries(v2_test_iq3_xxs_simd llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_IQ3_XXSTensor
    COMMAND v2_test_iq3_xxs_simd
    LABELS "V2;Unit;TensorOperations;Quantization;IQ3_XXS"
)

# Test: IQ1_S Tensor Tests
add_executable(v2_test_iq1_s_simd unit/tensors/Test__IQ1_STensor.cpp)
target_link_libraries(v2_test_iq1_s_simd llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_IQ1_STensor
    COMMAND v2_test_iq1_s_simd
    LABELS "V2;Unit;TensorOperations;Quantization;IQ1_S"
)



# Test: IQ1_M Tensor Tests
add_executable(v2_test_iq1_m_simd unit/tensors/Test__IQ1_MTensor.cpp)
target_link_libraries(v2_test_iq1_m_simd llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_IQ1_MTensor
    COMMAND v2_test_iq1_m_simd
    LABELS "V2;Unit;TensorOperations;Quantization;IQ1_M"
)

# Test: IQ3_S Tensor Tests
add_executable(v2_test_iq3_s_simd unit/tensors/Test__IQ3_STensor.cpp)
target_link_libraries(v2_test_iq3_s_simd llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_IQ3_STensor
    COMMAND v2_test_iq3_s_simd
    LABELS "V2;Unit;TensorOperations;Quantization;IQ3_S"
)

# IQ2_XXS Tensor Tests
# add_executable(v2_test_iq2_xxs_simd unit/tensors/Test__IQ2_XXSTensor.cpp)
# target_link_libraries(v2_test_iq2_xxs_simd llaminar2_core GTest::gtest_main)
# add_v2_test(V2_Unit_IQ2_XXSTensor
#     COMMAND v2_test_iq2_xxs_simd
#     LABELS "V2;Unit;TensorOperations;Quantization;IQ2_XXS"
# )

# IQ4_XS Tensor Tests
add_executable(v2_test_iq4_xs_simd unit/tensors/Test__IQ4_XSTensor.cpp)
target_link_libraries(v2_test_iq4_xs_simd llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_IQ4_XSTensor
    COMMAND v2_test_iq4_xs_simd
    LABELS "V2;Unit;TensorOperations;Quantization;IQ4_XS"
)

# =============================================================================
# E2E Parity Tests
# =============================================================================
# Full pipeline and kernel parity tests comparing different execution paths
# Organized by: what is being tested vs what reference
#
# Structure:
#   e2e/parity/pytorch/       - Against PyTorch ground truth
#   e2e/parity/internal/      - Llaminar modes compared against each other
#   e2e/parity/kernels/       - Individual kernel comparisons

# -----------------------------------------------------------------------------
# Integration Parity: MPI Tensor-Parallel vs Single-Rank
# -----------------------------------------------------------------------------
# MPIPipeline_vs_SingleRank removed — superseded by per-backend LocalTP/LocalPP/HybridPPTP parity tests

# -----------------------------------------------------------------------------
# DISABLED: E2E Parity tests moved to Integration or deprecated
# -----------------------------------------------------------------------------
# NOTE: FP32Pipeline_vs_PyTorch was removed - superseded by 
# integration/parity/qwen2/Test__Qwen2_CPU_vs_PyTorch.cpp which uses
# the standardized ParityTestBase infrastructure.
find_package(ZLIB)
if(ZLIB_FOUND)
    get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    
    if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR CMAKE_BUILD_TYPE MATCHES "Integration" OR ENABLE_SNAPSHOTS)
    else()
        message(STATUS "V2: Integration parity tests disabled (snapshots not available in Release build)")
        message(STATUS "     Use -DCMAKE_BUILD_TYPE=Integration or -DENABLE_SNAPSHOTS=ON to enable")
    endif()
else()
    message(WARNING "ZLIB not found - PyTorch parity tests disabled (requires cnpy for .npz loading)")
endif()

# Performance Test: GEMM Tuning Sweep
add_executable(v2_perf_gemm_tuning_sweep performance/kernels/cpu/gemm/Perf__GemmTuningSweep.cpp)
target_link_libraries(v2_perf_gemm_tuning_sweep 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Perf_GemmTuningSweep 
    COMMAND $<TARGET_FILE:v2_perf_gemm_tuning_sweep>
    LABELS "V2;Performance;GEMM;Tuning"
)

# Performance Test: KVCache
add_executable(v2_perf_kvcache performance/tensors/kvcache/Perf__KVCache.cpp)
target_link_libraries(v2_perf_kvcache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_KVCache 
    COMMAND $<TARGET_FILE:v2_perf_kvcache>
    LABELS "V2;Performance;KVCache;Memory"
)

# Performance Test: IINT8Unpackable (quantized tensor unpack throughput comparison)
add_executable(v2_perf_iint8_unpackable performance/tensors/Perf__IINT8Unpackable.cpp)
target_link_libraries(v2_perf_iint8_unpackable 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_IINT8Unpackable 
    COMMAND $<TARGET_FILE:v2_perf_iint8_unpackable>
    LABELS "V2;Performance;TensorOperations;Quantization;IINT8Unpackable;Unpack"
)

# Performance Test: Q3_K Unpack
add_executable(v2_perf_q3k_unpack performance/Perf__Q3K_Unpack.cpp)
target_link_libraries(v2_perf_q3k_unpack 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_Q3K_Unpack 
    COMMAND $<TARGET_FILE:v2_perf_q3k_unpack>
    LABELS "V2;Performance;Q3_K;Unpack"
)

# Performance Test: Q4_K Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q4_k_unpack performance/Perf__Q4_K_Unpack.cpp)
target_link_libraries(v2_perf_q4_k_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q4_K_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q4_k_unpack>
    LABELS "V2;Performance;Q4_K;Unpack"
)

# Performance Test: Q5_K Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q5_k_unpack performance/Perf__Q5_K_Unpack.cpp)
target_link_libraries(v2_perf_q5_k_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q5_K_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q5_k_unpack>
    LABELS "V2;Performance;Q5_K;Unpack"
)

# Performance Test: Q6_K Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q6_k_unpack performance/Perf__Q6_K_Unpack.cpp)
target_link_libraries(v2_perf_q6_k_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q6_K_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q6_k_unpack>
    LABELS "V2;Performance;Q6_K;Unpack"
)

# Performance Test: Q2_K Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q2_k_unpack performance/Perf__Q2_K_Unpack.cpp)
target_link_libraries(v2_perf_q2_k_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q2_K_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q2_k_unpack>
    LABELS "V2;Performance;Q2_K;Unpack"
)

# Performance Test: Q4_0 Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q4_0_unpack performance/Perf__Q4_0_Unpack.cpp)
target_link_libraries(v2_perf_q4_0_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q4_0_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q4_0_unpack>
    LABELS "V2;Performance;Q4_0;Unpack"
)

# Performance Test: Q4_1 Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q4_1_unpack performance/Perf__Q4_1_Unpack.cpp)
target_link_libraries(v2_perf_q4_1_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q4_1_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q4_1_unpack>
    LABELS "V2;Performance;Q4_1;Unpack"
)

# Performance Test: Q5_0 Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q5_0_unpack performance/Perf__Q5_0_Unpack.cpp)
target_link_libraries(v2_perf_q5_0_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q5_0_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q5_0_unpack>
    LABELS "V2;Performance;Q5_0;Unpack"
)

# Performance Test: Q5_1 Unpack (Scalar/AVX2/AVX-512)
add_executable(v2_perf_q5_1_unpack performance/Perf__Q5_1_Unpack.cpp)
target_link_libraries(v2_perf_q5_1_unpack llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_Q5_1_Unpack
    COMMAND $<TARGET_FILE:v2_perf_q5_1_unpack>
    LABELS "V2;Performance;Q5_1;Unpack"
)

# Performance Test: CPURMSNormKernelT (typed residuals performance)
add_executable(v2_perf_rmsnorm_typed performance/kernels/cpu/ops/Perf__CPURMSNormKernelT.cpp)
target_link_libraries(v2_perf_rmsnorm_typed 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_CPURMSNormKernelT 
    COMMAND $<TARGET_FILE:v2_perf_rmsnorm_typed>
    LABELS "V2;Performance;RMSNorm;TypedResiduals;Kernels"
    MPI_PROCS 1  # CPU-only perf test, single rank for accurate benchmarking
)

# Performance Test: CPUSoftmaxKernelT (typed softmax performance)
add_executable(v2_perf_softmax_typed performance/kernels/cpu/ops/Perf__CPUSoftmaxKernelT.cpp)
target_link_libraries(v2_perf_softmax_typed 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Perf_CPUSoftmaxKernelT 
    COMMAND $<TARGET_FILE:v2_perf_softmax_typed>
    LABELS "V2;Performance;Softmax;TypedResiduals;Kernels;FP32;BF16;FP16;Q8_1"
    MPI_PROCS 1  # CPU-only perf test, single rank for accurate benchmarking
)

# Performance Test: CPUSwiGLUKernelT (typed SwiGLU performance)
add_executable(v2_perf_swiglu_typed performance/kernels/cpu/ops/Perf__CPUSwiGLUKernelT.cpp)
target_link_libraries(v2_perf_swiglu_typed 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Perf_CPUSwiGLUKernelT 
    COMMAND $<TARGET_FILE:v2_perf_swiglu_typed>
    LABELS "V2;Performance;SwiGLU;TypedResiduals;Kernels;FP32;BF16;FP16;Q8_1"
    MPI_PROCS 1  # CPU-only perf test, single rank for accurate benchmarking
)

# Performance Test: RMSNorm Thread Sweep (optimal thread count discovery)
add_executable(v2_perf_rmsnorm_thread_sweep performance/kernels/cpu/ops/Perf__RMSNormThreadSweep.cpp)
target_link_libraries(v2_perf_rmsnorm_thread_sweep 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_RMSNormThreadSweep 
    COMMAND $<TARGET_FILE:v2_perf_rmsnorm_thread_sweep>
    LABELS "V2;Performance;RMSNorm;ThreadSweep;Kernels"
    MPI_PROCS 1
)

# Performance Test: CPURoPEKernel comparison (Legacy vs Typed FP32/BF16/FP16/Q8_1)
add_executable(v2_perf_rope_kernel performance/kernels/cpu/ops/Perf__CPURoPEKernel.cpp)
target_link_libraries(v2_perf_rope_kernel 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_CPURoPEKernel 
    COMMAND $<TARGET_FILE:v2_perf_rope_kernel>
    LABELS "V2;Performance;RoPE;TypedResiduals;Kernels;FP32;BF16;FP16;Q8_1"
    MPI_PROCS 1  # CPU-only perf test, single rank for accurate benchmarking
)

# Test: CPUFlashAttentionKernel mixed-precision tensor path (FP32 Q, Q8_1 K/V)
add_executable(v2_integration_cpu_flash_attention_kernel integration/kernels/cpu/attention/Test__CPUFlashAttentionKernel.cpp)
target_link_libraries(v2_integration_cpu_flash_attention_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_CPUFlashAttentionKernel
    COMMAND $<TARGET_FILE:v2_integration_cpu_flash_attention_kernel>
    LABELS "V2;Integration;CPU;FlashAttention;Q8_1;Decode;Prefill"
    MPI_PROCS 1
)

# Wo transpose benchmark - compares row-major vs column-major GEMV
add_executable(v2_perf_wo_transpose performance/Microbench__WoTranspose.cpp)
target_link_libraries(v2_perf_wo_transpose
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_compile_options(v2_perf_wo_transpose PRIVATE -mavx512f -mavx512dq -fopenmp)
target_link_libraries(v2_perf_wo_transpose gomp)
add_v2_test(V2_Perf_WoTranspose
    COMMAND $<TARGET_FILE:v2_perf_wo_transpose>
    LABELS "V2;Performance;Wo;Transpose;GEMV"
    MPI_PROCS 1
)

# =============================================================================
# Multi-Rank MPI Integration Tests for Distributed Architecture (Phase 1, 2, G0)
# =============================================================================

# Test: MPITopology Multi-Rank Integration Tests (Phase 1)
# Tests work distribution, capability exchange, and placement determinism
# across multiple MPI ranks.
# NOTE: These tests have custom main() with MPI_Init/MPI_Finalize
add_executable(v2_integration_mpi_topology integration/utils/mpi/Test__MPITopologyIntegration.cpp)
target_link_libraries(v2_integration_mpi_topology
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPITopology
    COMMAND $<TARGET_FILE:v2_integration_mpi_topology>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Topology;WorkDistribution;CapabilityExchange;Placement;Phase1"
)

# Test: Sliced GEMM MPI Integration Tests (Phase 2)
# Tests row-parallel GEMM with AllGatherv and column-parallel with AllReduce
# using the PreparedWeightStore ref-based sliced GEMM API.
add_executable(v2_integration_sliced_gemm_mpi integration/utils/mpi/Test__SlicedGEMM_MPI_Integration.cpp)
target_link_libraries(v2_integration_sliced_gemm_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_SlicedGEMM_MPI
    COMMAND $<TARGET_FILE:v2_integration_sliced_gemm_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;SlicedGEMM;RowParallel;ColumnParallel;AllGatherv;AllReduce;Phase2;TensorParallel"
)

# Test: PlacementStrategy MPI Integration Tests (Phase G0)
# Tests that PlacementStrategy produces identical plans across all ranks
# and integrates correctly with MPITopology.
add_executable(v2_integration_placement_strategy_mpi integration/utils/mpi/Test__PlacementStrategy_MPI_Integration.cpp)
target_link_libraries(v2_integration_placement_strategy_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_PlacementStrategy_MPI
    COMMAND $<TARGET_FILE:v2_integration_placement_strategy_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;PlacementStrategy;Determinism;CPUOnly;GPUFirst;PhaseG0"
)

# Test: Orchestrator Wiring Integration Tests (Phase G1 - TDD)
# TDD tests verifying that the orchestration system is properly wired:
#   1. MPITopology::computePlacement() is called during inference setup
#   2. PlacementStrategyFactory::autoSelect() selects the right strategy
#   3. HeterogeneousMultiDomainStrategy is used when appropriate
#   4. PlacementPlan flows through to DeviceGraphOrchestrator
add_executable(v2_integration_orchestrator_wiring integration/pipelines/Test__OrchestratorWiringIntegration.cpp)
target_include_directories(v2_integration_orchestrator_wiring PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_orchestrator_wiring
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_OrchestratorWiring
    COMMAND $<TARGET_FILE:v2_integration_orchestrator_wiring>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Orchestrator;Wiring;TDD;HeterogeneousMultiDomain;PhaseG1"
)

# Test: Orchestration Wiring Integration Tests (TDD - define expected behavior)
# These tests verify the EXPECTED correct behavior of orchestration wiring.
# They will FAIL until the wiring is properly implemented, then PASS.
# Purpose:
#   1. InferenceRunnerConfig should accept a PlacementPlan
#   2. createInferenceRunner() should use PlacementPlan for execution
#   3. MPIContext should expose topology() for placement computation
#   4. PlacementPlan should be honored during actual inference
add_executable(v2_integration_orchestration_wiring integration/pipelines/Test__OrchestrationWiring.cpp)
target_include_directories(v2_integration_orchestration_wiring PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_orchestration_wiring
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_OrchestrationWiring
    COMMAND $<TARGET_FILE:v2_integration_orchestration_wiring>
    LABELS "V2;Integration;Orchestrator;Wiring;TDD;PhaseG1"
)

# Test: Multi-turn inference session reset
# Verifies that clear_cache() correctly resets kernel dynamic state so that
# a second inference request produces correct output (not garbage from stale
# token IDs cached in embedding kernel GPU buffers).
# Requires: models/qwen2.5-0.5b-instruct-q4_0.gguf
add_executable(v2_integration_multi_turn_session_reset
    integration/pipelines/Test__MultiTurnSessionReset.cpp
    utils/TestOrchestrationHelper.cpp
)
target_include_directories(v2_integration_multi_turn_session_reset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ include path
)
target_link_libraries(v2_integration_multi_turn_session_reset
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MultiTurnSessionReset
    COMMAND $<TARGET_FILE:v2_integration_multi_turn_session_reset>
    LABELS "V2;Integration;Orchestrator;MultiTurn;SessionReset;Embedding;DynamicState"
    MPI_PROCS 1
)
# Override log level: DEBUG causes D2H syncs from GPU debug logging that
# crash under sustained multi-session GPU test loads (pre-existing issue).
if(NOT V2_PERF_TESTS_ONLY)
set_tests_properties(V2_Integration_MultiTurnSessionReset PROPERTIES
    ENVIRONMENT_MODIFICATION "LLAMINAR_LOG_LEVEL=set:WARN"
)
endif()

# Test: TPDomain MPI Integration Tests (Phase 3.2b)
# Tests TPDomainBuilder, MPI_Comm_split, cross-rank domain creation, and cleanup
# Verifies communicator management and domain consistency across ranks.
add_executable(v2_integration_tpdomain_mpi integration/collective/Test__TPDomainMPI.cpp)
target_link_libraries(v2_integration_tpdomain_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_TPDomainMPI
    COMMAND $<TARGET_FILE:v2_integration_tpdomain_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;TPDomain;TensorParallelism;HybridParallelism;Phase3"
)

# Test: UPIBackend MPI Integration Tests (Phase 3.3)
# Tests UPICollectiveBackend with actual MPI collectives over UPI fabric
# Verifies AllReduce, AllGather, AllGatherV, Broadcast, Barrier, and performance
add_executable(v2_integration_upi_backend_mpi integration/collective/backends/Test__UPIBackendMPI.cpp integration/mpi_gtest_main.cpp)
target_link_libraries(v2_integration_upi_backend_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_UPIBackendMPI
    COMMAND $<TARGET_FILE:v2_integration_upi_backend_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;UPI;Collective;TensorParallelism;Phase3"
)

# Test: GlobalTPContext MPI Integration Tests (Phase 2 Global TP Unified Interface)
# Tests multi-rank collective operations and point-to-point communication:
# - createWithSplit and domain setup
# - AllReduce, Broadcast, AllGather collectives
# - Send, Recv point-to-point
# - myIndex, degree, isGlobal domain ordering
# Requires mpirun -np 2 for execution
add_executable(v2_integration_global_tp_mpi
    integration/collective/Test__GlobalTPContext_MPI.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_integration_global_tp_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_GlobalTPContext_MPI
    COMMAND $<TARGET_FILE:v2_integration_global_tp_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;GlobalTP;MPI;TensorParallel;Collective;Phase2"
)

# Test: TPAllreduceStage MPI Integration Tests (Phase 4 Global TP Unified Interface)
# Tests TPAllreduceStage with GlobalTPContext across MPI ranks:
# - Stage construction with GlobalTPContext
# - Single-rank execute (no-op)
# - Two-rank allreduce correctness
# - ITPContext polymorphism with GlobalTPContext
# - Context switching (LOCAL to GLOBAL)
# Requires mpirun -np 2 for execution
add_executable(v2_integration_tp_allreduce_stage_mpi
    integration/execution/compute_stages/stages/Test__TPAllreduceStage_MPI.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_integration_tp_allreduce_stage_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_TPAllreduceStage_MPI
    COMMAND $<TARGET_FILE:v2_integration_tp_allreduce_stage_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;TPAllreduce;GlobalTP;MPI;TensorParallel;ComputeStage;Phase4"
)

add_executable(v2_integration_moe_overlay_sparse_transport_mpi
    integration/moe/Test__MoEOverlaySparseTransport_MPI.cpp
    integration/mpi_gtest_main.cpp
)
target_link_libraries(v2_integration_moe_overlay_sparse_transport_mpi
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_MoEOverlaySparseTransport_MPI
    COMMAND $<TARGET_FILE:v2_integration_moe_overlay_sparse_transport_mpi>
    LABELS "V2;Integration;MoE;Overlay;SparseCollective;MPI"
    MPI_PROCS 2
)

add_executable(v2_integration_moe_graph_native_prepared_expert_weights_mvp
    integration/moe/Test__MoEGraphNative_PreparedExpertWeights_MVP.cpp
    integration/mpi_gtest_main.cpp
)
target_include_directories(v2_integration_moe_graph_native_prepared_expert_weights_mvp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_integration_moe_graph_native_prepared_expert_weights_mvp
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_MoEGraphNative_PreparedExpertWeights_MVP
    COMMAND $<TARGET_FILE:v2_integration_moe_graph_native_prepared_expert_weights_mvp>
    LABELS "V2;Integration;MoE;Overlay;GraphNative;PreparedWeights"
    MPI_PROCS 1
)

# Test: TPAllreduceStage Local NCCL Integration
# Tests TPAllreduceStage with LocalTPContext using NCCL backend:
# - NCCL device validation (tensor must be on TP device)
# - Multi-GPU allreduce within single process
# Requires at least 2 CUDA GPUs
add_executable(v2_integration_tp_allreduce_stage_local_nccl
    integration/execution/compute_stages/stages/Test__TPAllreduceStage_LocalNCCL.cpp
)
target_link_libraries(v2_integration_tp_allreduce_stage_local_nccl
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_TPAllreduceStage_LocalNCCL
    COMMAND $<TARGET_FILE:v2_integration_tp_allreduce_stage_local_nccl>
    LABELS "V2;Integration;TPAllreduce;LocalTP;NCCL;CUDA;TensorParallel;ComputeStage"
)

# Test: Graph Snapshot Callback Invocation Integration Tests
# Verifies that snapshot callbacks are properly invoked for ALL stages during graph execution.
# Critical for E2E parity testing and debugging - ensures no stages silently skip callbacks.
# Tests:
#   - Basic stage types (RMSNorm, SwiGLU, ResidualAdd)
#   - Multi-stage graphs
#   - Full Qwen2 graph with DeviceGraphOrchestrator
#   - FusedAttentionWo context/output capture
#   - Decode phase callback invocation
add_executable(v2_test_graph_snapshot_callback_invocation integration/execution/graph/Test__GraphSnapshotCallbackInvocation.cpp)
target_link_libraries(v2_test_graph_snapshot_callback_invocation
    llaminar2_core
    GTest::gtest
    # NOTE: No gtest_main - this test has custom main() with MPI_Init
)
add_v2_test(V2_Integration_GraphSnapshotCallbackInvocation
    COMMAND $<TARGET_FILE:v2_test_graph_snapshot_callback_invocation>
    LABELS "V2;Integration;Graph;Snapshot;Callback"
    MPI_PROCS 1
)

# Test: VNNI Overflow Analysis
# Comprehensive analysis of INT32 overflow risks in Q×K^T and P×V paths
add_executable(v2_test_vnni_overflow_analysis unit/kernels/cpu/attention/Test__VNNIOverflowAnalysis.cpp)
target_link_libraries(v2_test_vnni_overflow_analysis
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_VNNIOverflowAnalysis
    COMMAND $<TARGET_FILE:v2_test_vnni_overflow_analysis>
    LABELS "V2;Unit;Microkernels;Q16_1;VNNI;Overflow;Analysis"
    MPI_PROCS 1
)

# =============================================================================
# ROCm Quantised GEMM Kernel Tests (INT8 + native-VNNI)
# =============================================================================
if(HAVE_ROCM)
    # Test: ROCm Quantised GEMM Kernel (INT8×INT8→INT32 VNNI)
    # Tests weight packing, activation quantization, VNNI GEMM, and output scaling
    add_executable(v2_unit_rocm_quantised_gemm_kernel 
        unit/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp
    )
    target_link_libraries(v2_unit_rocm_quantised_gemm_kernel 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CPUWeightPacking_ROCm
        COMMAND $<TARGET_FILE:v2_unit_rocm_quantised_gemm_kernel>
        LABELS "V2;Unit;Kernels;WeightPacking;INT8;Quantized"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CPU weight packing unit tests (ROCm format)")

    # Integration Test: ROCm Quantised GEMM Kernel Pipeline
    # Tests full pipeline: weight packing + activation quantization + realistic model dims
    # TODO(rocm-vnni): re-enable after CK references are stripped from this test.
    # The CK kernel was removed in the VNNI-only refactor but this test file
    # still references rocmQuantGemm_executeNoScale / executeTwoKernel / etc.
    if(FALSE)
    add_executable(v2_integration_rocm_quantised_gemm_kernel 
        integration/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp
    )
    target_link_libraries(v2_integration_rocm_quantised_gemm_kernel 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmQuantisedGemmKernel
        COMMAND $<TARGET_FILE:v2_integration_rocm_quantised_gemm_kernel>
        LABELS "V2;Integration;ROCm;GEMM;INT8;Quantized;gfx906;Qwen2"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Quantised GEMM kernel integration tests")
    endif()

    add_executable(v2_integration_rocm_quantised_gemm_small_m
        integration/kernels/rocm/Test__ROCmQuantisedGemmSmallM.cpp
    )
    target_link_libraries(v2_integration_rocm_quantised_gemm_small_m
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmQuantisedGemmSmallM
        COMMAND $<TARGET_FILE:v2_integration_rocm_quantised_gemm_small_m>
        LABELS "V2;Integration;ROCm;GEMM;MTP;SmallM"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added focused ROCm small-M GEMM integration tests")

    # Unit Test: Native-VNNI Packing (CPU-only)
    # Validates packNativeVNNI() output for all 16 supported quantization formats.
    # Tests codebook_id, payload/scales/mins sizes, FP16 scale validity, dimension sweeps.
    add_executable(v2_unit_native_vnni_packing
        unit/kernels/rocm/Test__NativeVNNI_Packing.cpp
    )
    target_link_libraries(v2_unit_native_vnni_packing
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_NativeVNNI_Packing
        COMMAND $<TARGET_FILE:v2_unit_native_vnni_packing>
        LABELS "V2;Unit;ROCm;GEMV;NativeVNNI;Quantized;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added native-VNNI packing unit tests (16 formats)")

    # Integration Test: Native-VNNI GEMV Accuracy
    # Runs M=1 GEMV on GPU for all 16 native-VNNI formats and compares output
    # against CPU FP32 reference (dequantized weights × FP32 input).
    add_executable(v2_integration_rocm_native_vnni_gemv
        integration/kernels/rocm/Test__NativeVNNI_GEMV.cpp
    )
    target_link_libraries(v2_integration_rocm_native_vnni_gemv
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCm_NativeVNNI_GEMV
        COMMAND $<TARGET_FILE:v2_integration_rocm_native_vnni_gemv>
        LABELS "V2;Integration;ROCm;GEMV;NativeVNNI;Quantized;gfx906;IQ;Accuracy"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added native-VNNI GEMV integration tests (16 formats)")

    # Integration Test: Native-VNNI GEMM — ALL dispatch paths
    # Exercises every kernel variant (streaming, N128, N64, M16/M32/M64, 2/3-wave)
    # against hipBLAS FP32 SGEMM reference for all 15 quantized formats across multiple M values and shapes.
    add_executable(v2_integration_rocm_native_vnni_gemm
        integration/kernels/rocm/Test__NativeVNNI_GEMM.cpp
    )
    target_link_libraries(v2_integration_rocm_native_vnni_gemm
        llaminar2_core
        roc::hipblas
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCm_NativeVNNI_GEMM
        COMMAND $<TARGET_FILE:v2_integration_rocm_native_vnni_gemm>
        LABELS "V2;Integration;ROCm;GEMM;NativeVNNI;Quantized;gfx906;Accuracy"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added native-VNNI GEMM integration tests (all dispatch paths)")

    # Performance Test: ROCm Quantised GEMM Kernel Benchmarks
    # Measures throughput and accuracy for Qwen model dimensions (0.5B, 7B, 14B, 32B)
    add_executable(v2_perf_rocm_quantised_gemm_kernel 
        performance/kernels/rocm/Perf__ROCmQuantisedGemmKernel.cpp
    )
    target_link_libraries(v2_perf_rocm_quantised_gemm_kernel 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmQuantisedGemmKernel
        COMMAND $<TARGET_FILE:v2_perf_rocm_quantised_gemm_kernel>
        LABELS "V2;Performance;ROCm;GEMM;INT8;Quantized;gfx906;Qwen2;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Quantised GEMM kernel performance benchmarks")

    # Performance Test: ROCm GEMV Kernel (bandwidth-optimized M=1 decode)
    # Benchmarks and validates the INT8 GEMV kernel against CPU reference
    # for Qwen2.5-0.5B and 7B decode workloads
    add_executable(v2_perf_rocm_gemv_kernel
        performance/kernels/rocm/Perf__ROCmGemvKernel.cpp
    )
    target_link_libraries(v2_perf_rocm_gemv_kernel
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmGemvKernel
        COMMAND $<TARGET_FILE:v2_perf_rocm_gemv_kernel>
        LABELS "V2;Performance;ROCm;GEMV;INT8;Decode;gfx906;Qwen2;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm GEMV kernel performance benchmarks")

    # Performance Test: ROCm TP Scaling Analysis
    # Benchmarks GEMV (decode M=1) and GEMM (prefill) at TP-sharded dimensions
    # for 2-way and 4-way tensor parallelism to identify dispatch/tiling regressions
    add_executable(v2_perf_rocm_tp_scaling
        performance/kernels/rocm/Perf__ROCmTPScaling.cpp
    )
    target_link_libraries(v2_perf_rocm_tp_scaling
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmTPScaling
        COMMAND $<TARGET_FILE:v2_perf_rocm_tp_scaling>
        LABELS "V2;Performance;ROCm;GEMV;GEMM;INT8;TP;Scaling;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm TP scaling performance analysis")

    # Performance Test: ROCm Flash Attention Decode — TP-sliced head count sweep
    # Measures flash decoding latency and CU occupancy across TP=1/2/4 head counts.
    # Includes split-K sweep to find optimal parallelism for low-head scenarios.
    add_executable(v2_perf_rocm_flash_attention_decode
        performance/kernels/rocm/Perf__ROCmFlashAttentionDecode.cpp
    )
    target_link_libraries(v2_perf_rocm_flash_attention_decode
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmFlashAttentionDecode
        COMMAND $<TARGET_FILE:v2_perf_rocm_flash_attention_decode>
        LABELS "V2;Performance;ROCm;Attention;FlashAttention;FlashDecode;TP;Scaling;SplitK;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Flash Attention decode TP scaling benchmarks")

    # Performance Test: ROCm Flash Attention Prefill — tile_q, KV type, and TP scaling sweep
    # Measures prefill attention latency across tile_q={16,32,64}, KV types (FP32/FP16/Q8_1),
    # and TP head count configurations. Unlike decode, prefill has no split-K; occupancy
    # comes from n_heads × num_q_tiles.
    add_executable(v2_perf_rocm_flash_attention_prefill
        performance/kernels/rocm/Perf__ROCmFlashAttentionPrefill.cpp
    )
    target_link_libraries(v2_perf_rocm_flash_attention_prefill
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmFlashAttentionPrefill
        COMMAND $<TARGET_FILE:v2_perf_rocm_flash_attention_prefill>
        LABELS "V2;Performance;ROCm;Attention;FlashAttention;Prefill;TP;Scaling;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Flash Attention prefill TP scaling benchmarks")

    # Performance Test: ROCm GEMV KB Scaling — KB (K-parallel split) sweep for decode GEMV
    # Measures INT8 VNNI GEMV latency across forced KB values and TP-derived N sizes.
    # Used to tune select_blockwise_qwo_outer_splits heuristic for TP configurations.
    add_executable(v2_perf_rocm_gemv_kb_scaling
        performance/kernels/rocm/Perf__ROCmGemvKBScaling.cpp
    )
    target_link_libraries(v2_perf_rocm_gemv_kb_scaling
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmGemvKBScaling
        COMMAND $<TARGET_FILE:v2_perf_rocm_gemv_kb_scaling>
        LABELS "V2;Performance;ROCm;GEMV;INT8;VNNI;KB;SplitK;TP;Scaling;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm GEMV KB scaling benchmarks")

    # Performance Test: ROCm IQ4_NL Decode Microbenchmark
    # Isolates IQ4_NL decode+dot strategy costs to compare kernel approach variants.
    set_source_files_properties(
        performance/kernels/rocm/Perf__ROCmIQ4DecodeMicrobench.hip
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_rocm_iq4_decode_microbench
        performance/kernels/rocm/Perf__ROCmIQ4DecodeMicrobench.hip
    )
    target_link_libraries(v2_perf_rocm_iq4_decode_microbench
        hip::host
        hip::device
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmIQ4DecodeMicrobench
        COMMAND $<TARGET_FILE:v2_perf_rocm_iq4_decode_microbench>
        LABELS "V2;Performance;ROCm;IQ4_NL;Decode;Microbenchmark;gfx906;Experiment"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm IQ4 decode microbenchmark")

    # Performance Test: Native-VNNI decode throughput/trainer (all NativeVNNI GPU formats)
    # Measures per-format bandwidth efficiency at model-realistic dimensions.
    add_executable(v2_perf_native_vnni_throughput
        performance/kernels/rocm/Perf__NativeVNNI_Throughput.cpp
        performance/kernels/rocm/GpuVerification.hip
    )
    target_link_libraries(v2_perf_native_vnni_throughput
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
        roc::hipblas
    )
    target_compile_options(v2_perf_native_vnni_throughput PRIVATE -fopenmp)
    target_link_options(v2_perf_native_vnni_throughput PRIVATE -fopenmp)
    add_v2_test(V2_Perf_NativeVNNI_Throughput
        COMMAND $<TARGET_FILE:v2_perf_native_vnni_throughput>
        LABELS "V2;Performance;ROCm;GEMV;Decode;NativeVNNI;Codebook;Quantized;gfx906;Benchmark;Bandwidth;Trainer"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added Native-VNNI decode throughput/trainer benchmark (all GPU codebooks, multi-GPU)")

    # Performance Test: Native-VNNI GEMM (Q4_0 + IQ4_NL, M>1 prefill)
    # Benchmarks native-VNNI GEMM kernel against INT8 V3/V7 baseline.
    # Reports speedup, correctness (cosine similarity), and GFLOPS.
    # GpuVerification.hip has the GPU cosine kernel (compiled by HIP compiler).
    add_executable(v2_perf_native_vnni_gemm
        performance/kernels/rocm/Perf__NativeVNNI_GEMM.cpp
        performance/kernels/rocm/GpuVerification.hip
    )
    target_link_libraries(v2_perf_native_vnni_gemm
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
        roc::hipblas
    )
    target_compile_options(v2_perf_native_vnni_gemm PRIVATE -fopenmp)
    target_link_options(v2_perf_native_vnni_gemm PRIVATE -fopenmp)
    add_v2_test(V2_Perf_NativeVNNI_GEMM
        COMMAND $<TARGET_FILE:v2_perf_native_vnni_gemm>
        LABELS "V2;Performance;ROCm;GEMM;NativeVNNI;Quantized;gfx906;Benchmark;Prefill"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added Native-VNNI GEMM benchmark (Q4_0 + IQ4_NL)")

    # Performance Test: Native-VNNI prefill sweep/trainer (all NativeVNNI GPU formats)
    # Per-shape tuning sweep over N_TILE, M_TILE, MIN_BLOCKS, UNROLL_G.
    # Outputs codebook-tagged candidate rows and best variant per shape for dispatch heuristic tuning.
    add_executable(v2_perf_native_vnni_sweep
        performance/kernels/rocm/Perf__NativeVNNI_Sweep.cpp
        performance/kernels/rocm/GpuVerification.hip
    )
    target_link_libraries(v2_perf_native_vnni_sweep
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
        roc::hipblas
    )
    target_compile_options(v2_perf_native_vnni_sweep PRIVATE -fopenmp)
    target_link_options(v2_perf_native_vnni_sweep PRIVATE -fopenmp)
    add_v2_test(V2_Perf_NativeVNNI_Sweep
        COMMAND $<TARGET_FILE:v2_perf_native_vnni_sweep>
        LABELS "V2;Performance;ROCm;GEMM;Prefill;NativeVNNI;Codebook;Quantized;gfx906;Benchmark;Sweep;Trainer"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added Native-VNNI prefill sweep benchmark (all GPU codebooks)")

    # Performance Test: Q8_0 Native-VNNI GEMV Tuning
    # Compares INT8-VNNI vs Native-VNNI decode paths for Q8_0 weights.
    # Sweeps TARGET_WAVES_PER_CU and KB to find optimal dispatch for Q8_0.
    add_executable(v2_perf_q8_0_native_vnni_gemv_tuning
        performance/kernels/rocm/Perf__Q8_0_NativeVNNI_GemvTuning.cpp
        performance/kernels/rocm/GpuVerification.hip
    )
    target_link_libraries(v2_perf_q8_0_native_vnni_gemv_tuning
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
        roc::hipblas
    )
    target_compile_options(v2_perf_q8_0_native_vnni_gemv_tuning PRIVATE -fopenmp)
    target_link_options(v2_perf_q8_0_native_vnni_gemv_tuning PRIVATE -fopenmp)
    add_v2_test(V2_Perf_Q8_0_NativeVNNI_GemvTuning
        COMMAND $<TARGET_FILE:v2_perf_q8_0_native_vnni_gemv_tuning>
        LABELS "V2;Performance;ROCm;GEMV;NativeVNNI;Q8_0;Quantized;gfx906;Benchmark;Tuning"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added Q8_0 Native-VNNI GEMV tuning benchmark")

    # Performance Microbenchmark: IQ Decode Strategy Lab
    # Isolated decode-only benchmark for IQ grid-index quantization formats.
    # Tests multiple decode strategies (baseline, vsub4, LDS cache, ksigns64)
    # head-to-head without the full GEMV pipeline.
    set_source_files_properties(
        performance/kernels/rocm/Microbench__IQDecodeLab.hip
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_iq_decode_lab
        performance/kernels/rocm/Microbench__IQDecodeLab.hip
    )
    target_link_libraries(v2_perf_iq_decode_lab
        hip::host
        hip::device
        llaminar2_core
    )
    add_v2_test(V2_Perf_IQDecodeLab
        COMMAND $<TARGET_FILE:v2_perf_iq_decode_lab>
        LABELS "V2;Performance;ROCm;GEMV;IQ;Decode;Microbenchmark;gfx906;Experiment"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added IQ decode strategy lab microbenchmark")

    # TODO(rocm-vnni): re-enable after CK references are stripped from this microbench.
    if(FALSE)
    set_source_files_properties(
        performance/kernels/rocm/Microbench__ROCmPrefillStrategyLab.hip
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_rocm_prefill_strategy_lab
        performance/kernels/rocm/Microbench__ROCmPrefillStrategyLab.hip
    )
    target_link_libraries(v2_perf_rocm_prefill_strategy_lab
        hip::host
        hip::device
        llaminar2_core
        OpenMP::OpenMP_CXX
    )
    target_compile_options(v2_perf_rocm_prefill_strategy_lab PRIVATE -fopenmp)
    target_link_options(v2_perf_rocm_prefill_strategy_lab PRIVATE -fopenmp)
    add_v2_test(V2_Perf_ROCmPrefillStrategyLab
        COMMAND $<TARGET_FILE:v2_perf_rocm_prefill_strategy_lab>
        LABELS "V2;Performance;ROCm;GEMM;Prefill;Microbenchmark;StrategyLab;gfx906;Experiment"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added standalone ROCm prefill strategy lab microbenchmark")
    endif()

    set_source_files_properties(
        performance/kernels/rocm/Microbench__GeminiFFNKernelPlayground.hip
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_rocm_gemini_ffn_kernel_playground
        performance/kernels/rocm/Microbench__GeminiFFNKernelPlayground.hip
    )
    target_link_libraries(v2_perf_rocm_gemini_ffn_kernel_playground
        hip::host
        hip::device
        llaminar2_core
        OpenMP::OpenMP_CXX
    )
    target_compile_options(v2_perf_rocm_gemini_ffn_kernel_playground PRIVATE -fopenmp)
    target_link_options(v2_perf_rocm_gemini_ffn_kernel_playground PRIVATE -fopenmp)
    add_v2_test(V2_Perf_ROCmGeminiFFNKernelPlayground
        COMMAND $<TARGET_FILE:v2_perf_rocm_gemini_ffn_kernel_playground>
        LABELS "V2;Performance;ROCm;GEMM;Prefill;Microbenchmark;Gemini;FFN;gfx906;Experiment"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added Gemini FFN kernel playground microbenchmark")

    add_executable(v2_perf_rocm_prefill_dispatch_comparison
        performance/kernels/rocm/Perf__ROCmPrefillDispatchComparison.cpp
    )
    target_link_libraries(v2_perf_rocm_prefill_dispatch_comparison
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_ROCmPrefillDispatchComparison
        COMMAND $<TARGET_FILE:v2_perf_rocm_prefill_dispatch_comparison>
        LABELS "V2;Performance;ROCm;GEMM;INT8;Prefill;Dispatch;Comparison;gfx906;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm prefill dispatch comparison performance benchmark")

    # Performance Test: Blockwise VNNI GEMM per-shape tuning
    # Benchmarks V3/V7 blockwise GEMM variants with different M_TILE values
    # across all Qwen shapes (0.5B, 3B, 7B) with correctness gate.
    add_executable(v2_perf_blockwise_vnni_gemm
        performance/kernels/rocm/Perf__BlockwiseVNNI_GEMM.cpp
        performance/kernels/rocm/GpuVerification.hip
    )
    target_link_libraries(v2_perf_blockwise_vnni_gemm
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
        roc::hipblas
    )
    target_compile_options(v2_perf_blockwise_vnni_gemm PRIVATE -fopenmp)
    target_link_options(v2_perf_blockwise_vnni_gemm PRIVATE -fopenmp)
    add_v2_test(V2_Perf_BlockwiseVNNI_GEMM
        COMMAND $<TARGET_FILE:v2_perf_blockwise_vnni_gemm>
        LABELS "V2;Performance;ROCm;GEMM;INT8;Blockwise;Prefill;Dispatch;Tuning;gfx906;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added blockwise VNNI GEMM per-shape tuning benchmark")

    # Performance Test: Blockwise Native-VNNI (sub-Q8) correctness & throughput
    # Compares blockwise activation quantization native-VNNI (Q4_0 through IQ1_M)
    # against INT8-VNNI baseline across all Qwen shapes for decode (GEMV M=1) and
    # prefill (GEMM M=32..256). Reports correctness (cosine vs FP32 hipBLAS ref),
    # throughput (μs), and speedup.
    add_executable(v2_perf_blockwise_native_vnni
        performance/kernels/rocm/Perf__BlockwiseNativeVNNI.cpp
        performance/kernels/rocm/GpuVerification.hip
    )
    target_link_libraries(v2_perf_blockwise_native_vnni
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        OpenMP::OpenMP_CXX
        roc::hipblas
    )
    target_compile_options(v2_perf_blockwise_native_vnni PRIVATE -fopenmp)
    target_link_options(v2_perf_blockwise_native_vnni PRIVATE -fopenmp)
    add_v2_test(V2_Perf_BlockwiseNativeVNNI
        COMMAND $<TARGET_FILE:v2_perf_blockwise_native_vnni>
        LABELS "V2;Performance;ROCm;GEMV;GEMM;NativeVNNI;Blockwise;Quantized;gfx906;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added blockwise native-VNNI correctness & throughput benchmark")

    # Performance Test: Blockwise activation quantization kernel tuning
    # Benchmarks 5 kernel variants for FP32->INT8 blockwise quantization
    # across all Qwen activation sizes with correctness gate.
    add_executable(v2_perf_blockwise_quant_kernel
        performance/kernels/rocm/Perf__BlockwiseQuantKernel.cpp
    )
    target_link_libraries(v2_perf_blockwise_quant_kernel
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_BlockwiseQuantKernel
        COMMAND $<TARGET_FILE:v2_perf_blockwise_quant_kernel>
        LABELS "V2;Performance;ROCm;Quantization;Blockwise;INT8;Tuning;gfx906;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added blockwise activation quantization kernel tuning benchmark")

    # NOTE: FP16 vs INT8 benchmark removed (January 2026) - INT8 proven faster for all sizes.
    # Results documented in changelog/2026-01-10-rocm-int8-gemm-optimization.md

    # Performance Test: hipBLAS INT8 GEMM Roofline
    # Measures raw hipBLAS INT8 GEMM throughput to establish roofline on MI50
    set_source_files_properties(
        performance/rocm/Perf__HipblasInt8Roofline.hip
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_hipblas_int8_roofline 
        performance/rocm/Perf__HipblasInt8Roofline.hip
    )
    target_link_libraries(v2_perf_hipblas_int8_roofline 
        hip::host
        hip::device
        hipblas
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_HipblasInt8Roofline
        COMMAND $<TARGET_FILE:v2_perf_hipblas_int8_roofline>
        LABELS "V2;Performance;ROCm;GEMM;INT8;hipBLAS;Roofline;gfx906;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added hipBLAS INT8 GEMM roofline benchmarks")

    # =============================================================================
    # ROCm Ring Buffer KV Cache Tests
    # =============================================================================

    # Integration Test: ROCm Ring Buffer KV Cache
    # Tests append, wrap-around, eviction, linearization, FP32/FP16 precision
    add_executable(v2_integration_rocm_ring_kv_cache 
        integration/kernels/rocm/Test__ROCmRingKVCache.cpp
    )
    target_link_libraries(v2_integration_rocm_ring_kv_cache 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRingKVCache
        COMMAND $<TARGET_FILE:v2_integration_rocm_ring_kv_cache>
        LABELS "V2;Integration;ROCm;KVCache;RingBuffer;Eviction;SlidingWindow;FP32;FP16;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Ring KV Cache integration tests")

    # Integration Test: ROCm Ring Buffer KV Cache Sharding
    # Tests sharded KV cache for tensor parallelism (stub tests until sharding implemented)
    add_executable(v2_integration_rocm_ring_kv_cache_sharding 
        integration/kernels/rocm/Test__ROCmRingKVCache_Sharding.cpp
    )
    target_link_libraries(v2_integration_rocm_ring_kv_cache_sharding 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRingKVCache_Sharding
        COMMAND $<TARGET_FILE:v2_integration_rocm_ring_kv_cache_sharding>
        LABELS "V2;Integration;ROCm;KVCache;RingBuffer;Sharding;TensorParallel;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Ring KV Cache sharding integration tests")

    # Integration Test: ROCm Ring Buffer KV Cache Comprehensive
    # Comprehensive edge-case tests: metadata, eviction, single-token, multi-seq, wrapping stress
    add_executable(v2_integration_rocm_ring_kv_cache_comprehensive 
        integration/kernels/rocm/Test__ROCmRingKVCache_Comprehensive.cpp
    )
    target_link_libraries(v2_integration_rocm_ring_kv_cache_comprehensive 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRingKVCache_Comprehensive
        COMMAND $<TARGET_FILE:v2_integration_rocm_ring_kv_cache_comprehensive>
        LABELS "V2;Integration;ROCm;KVCache;RingBuffer;Eviction;Metadata;EdgeCases;FP32;FP16;BF16;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Ring KV Cache comprehensive edge-case tests")

    # Integration Test: ROCm Ring Buffer KV Cache Logical Block IO
    # Proves host-backed prefix payload export/import/truncate over real ROCm storage.
    add_executable(v2_integration_rocm_ring_kv_cache_logical_block_io
        integration/kernels/rocm/Test__ROCmRingKVCache_LogicalBlockIO.cpp
    )
    target_link_libraries(v2_integration_rocm_ring_kv_cache_logical_block_io
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRingKVCache_LogicalBlockIO
        COMMAND $<TARGET_FILE:v2_integration_rocm_ring_kv_cache_logical_block_io>
        LABELS "V2;Integration;ROCm;KVCache;RingBuffer;PrefixCache;LogicalBlockIO;FP32;Q8_1;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Ring KV Cache logical block IO integration tests")

    # Integration Test: ROCm Hybrid KV Cache Reset
    # Verifies GDN GPU kernel state resets through standard clear()/clear_layer()
    # semantics without destroying/recreating the hybrid cache object.
    add_executable(v2_integration_rocm_hybrid_kv_cache_reset
        integration/kernels/rocm/Test__ROCmHybridKVCacheReset.cpp
    )
    target_link_libraries(v2_integration_rocm_hybrid_kv_cache_reset
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmHybridKVCacheReset
        COMMAND $<TARGET_FILE:v2_integration_rocm_hybrid_kv_cache_reset>
        LABELS "V2;Integration;ROCm;KVCache;Hybrid;GatedDeltaNet;Reset;Regression;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Hybrid KV Cache reset regression tests")

    # Integration Test: ROCm Padded GDN Real-Length Semantics
    # Verifies effective-length recurrence and short-conv prefill preserve decode state.
    add_executable(v2_integration_rocm_gdn_padded_real_length
        integration/kernels/rocm/Test__ROCmGDNPaddedRealLength.cpp
    )
    target_link_libraries(v2_integration_rocm_gdn_padded_real_length
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmGDNPaddedRealLength
        COMMAND $<TARGET_FILE:v2_integration_rocm_gdn_padded_real_length>
        LABELS "V2;Integration;ROCm;GatedDeltaNet;ShortConv;Padding;Phase6;RealLength;FP32;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm padded GDN real-length integration tests")

    # Integration Test: ROCm TurboQuant KV Cache
    # Tests ROCmRingKVCacheTQ: TQ8 K / TQ4 V quantized storage, RoPE-on-read, eviction, shadows
    add_executable(v2_integration_rocm_ring_kv_cache_tq 
        integration/kernels/rocm/Test__ROCmRingKVCacheTQ.cpp
    )
    target_link_libraries(v2_integration_rocm_ring_kv_cache_tq 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRingKVCacheTQ
        COMMAND $<TARGET_FILE:v2_integration_rocm_ring_kv_cache_tq>
        LABELS "V2;Integration;ROCm;KVCache;TurboQuant;TQ8;TQ4;RoPE;Quantization;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm TurboQuant KV Cache integration tests")

    # =============================================================================
    # ROCm SwiGLU Parity Tests
    # =============================================================================

    # Integration Test: ROCm SwiGLU Parity
    # Tests ROCm SwiGLU kernels (FP32, BF16, FP16) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_swiglu_parity 
        integration/kernels/rocm/Test__ROCmSwiGLUParity.cpp
    )
    target_link_libraries(v2_integration_rocm_swiglu_parity 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmSwiGLUParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_swiglu_parity>
        LABELS "V2;Integration;ROCm;Parity;SwiGLU;FP32;BF16;FP16;CosineSimilarity;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm SwiGLU parity tests")

    # ROCm RoPE Parity Tests
    # =============================================================================

    # Integration Test: ROCm RoPE Parity
    # Tests ROCm RoPE kernels (FP32, BF16, FP16) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_rope_parity 
        integration/kernels/rocm/Test__ROCmRoPEParity.cpp
    )
    target_link_libraries(v2_integration_rocm_rope_parity 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRoPEParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_rope_parity>
        LABELS "V2;Integration;ROCm;Parity;RoPE;FP32;BF16;FP16;CosineSimilarity;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm RoPE parity tests")

    # ROCm RMSNorm Parity Tests
    # =============================================================================

    # Integration Test: ROCm RMSNorm Parity
    # Tests ROCm RMSNorm kernels (FP32, BF16, FP16) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_rmsnorm_parity 
        integration/kernels/rocm/Test__ROCmRMSNormParity.cpp
    )
    target_link_libraries(v2_integration_rocm_rmsnorm_parity 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRMSNormParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_rmsnorm_parity>
        LABELS "V2;Integration;ROCm;Parity;RMSNorm;FP32;BF16;FP16;CosineSimilarity;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm RMSNorm parity tests")

    # ROCm Embedding Parity Tests
    # =============================================================================

    # Integration Test: ROCm Embedding Parity
    # Tests ROCm Embedding kernels (FP32, BF16, FP16, Q8_1) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_embedding_parity 
        integration/kernels/rocm/Test__ROCmEmbeddingParity.cpp
    )
    target_link_libraries(v2_integration_rocm_embedding_parity 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmEmbeddingParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_embedding_parity>
        LABELS "V2;Integration;ROCm;Parity;Embedding;FP32;BF16;FP16;Q8_1;CosineSimilarity;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Embedding parity tests")

    # ROCm Flash Attention Parity Tests
    # =============================================================================

    # Integration Test: ROCm Flash Attention Parity
    # Tests Flash Attention 2 (prefill) and Flash Decoding (decode) ROCm kernels
    # against CPU attention reference with cosine similarity metrics
    add_executable(v2_integration_rocm_flash_attention_parity 
        integration/kernels/rocm/Test__ROCmFlashAttentionParity.cpp
    )
    target_link_libraries(v2_integration_rocm_flash_attention_parity 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmFlashAttentionParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_flash_attention_parity>
        LABELS "V2;Integration;ROCm;Parity;Attention;FlashAttention;FlashDecode;GQA;FP32;CosineSimilarity;gfx906"
        MPI_PROCS 1
    )
    add_v2_test(V2_Integration_ROCmFlashAttentionExternalMask
        COMMAND $<TARGET_FILE:v2_integration_rocm_flash_attention_parity> --gtest_filter=Test__ROCmFlashAttentionParity.RocblasPrefillHonorsExternalMask
        LABELS "V2;Integration;ROCm;Attention;ExternalMask;MTP;FP32;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Flash Attention parity tests")

    # Integration Test: ROCm attention padding parity for Phase 6 bucket safety
    # Verifies causal real-row outputs are unchanged by hostile bucket-tail padding
    add_executable(v2_integration_rocm_attention_padding_parity
        integration/kernels/rocm/Test__ROCmAttentionPaddingParity.cpp
    )
    target_link_libraries(v2_integration_rocm_attention_padding_parity
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmAttentionPaddingParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_attention_padding_parity>
        LABELS "V2;Integration;ROCm;Attention;Padding;CausalMask;Phase6;Parity;FP32;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm attention padding parity tests")

    # Integration Test: ROCm HiddenStateRowSelectStage graph capture/replay
    # Verifies bucketed-prefill LM-head row selection can update real length across HIP graph replays.
    add_executable(v2_integration_rocm_hidden_state_row_select_stage
        integration/kernels/rocm/Test__ROCmHiddenStateRowSelectStage.cpp
    )
    target_link_libraries(v2_integration_rocm_hidden_state_row_select_stage
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmHiddenStateRowSelectStage
        COMMAND $<TARGET_FILE:v2_integration_rocm_hidden_state_row_select_stage>
        LABELS "V2;Integration;ROCm;GraphCapture;RowSelect;LMHead;Phase6;FP32;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm hidden-state row-select graph replay tests")

    # Integration Test: ROCm RoPE graph-capture dynamic metadata
    # Verifies contiguous RoPE replay uses pre-uploaded device params and explicit
    # non-contiguous position IDs fail before H2D during graph capture.
    add_executable(v2_integration_rocm_rope_graph_capture_no_h2d
        integration/kernels/Test__RoPEGraphCaptureNoH2D.cpp
    )
    target_compile_definitions(v2_integration_rocm_rope_graph_capture_no_h2d PRIVATE GPU_CONTEXT_TEST_BACKEND_ROCM=1)
    target_include_directories(v2_integration_rocm_rope_graph_capture_no_h2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(v2_integration_rocm_rope_graph_capture_no_h2d
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmRoPEGraphCaptureNoH2D
        COMMAND $<TARGET_FILE:v2_integration_rocm_rope_graph_capture_no_h2d>
        LABELS "V2;Integration;ROCm;GraphCapture;RoPE;NoH2D;FP32;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm RoPE graph-capture no-H2D tests")

    # ROCm ResidualAdd Parity Tests
    # =============================================================================

    # Integration Test: ROCm ResidualAdd Parity
    # Tests ROCm ResidualAdd kernels (FP32, BF16, FP16) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_residual_add_parity 
        integration/kernels/rocm/Test__ROCmResidualAddParity.cpp
    )
    target_link_libraries(v2_integration_rocm_residual_add_parity 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmResidualAddParity
        COMMAND $<TARGET_FILE:v2_integration_rocm_residual_add_parity>
        LABELS "V2;Integration;ROCm;Parity;ResidualAdd;FP32;BF16;FP16;CosineSimilarity;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm ResidualAdd parity tests")

    # ROCm Event Synchronization Tests
    # =============================================================================
    # Critical regression test for event-based synchronization fix.
    # Validates that waitForEvent() uses hipEventSynchronize() instead of
    # hipStreamSynchronize(), which is essential for snapshot capture performance.

    add_executable(v2_integration_rocm_event_synchronization
        integration/backends/rocm/Test__ROCmEventSynchronization.cpp
    )
    target_link_libraries(v2_integration_rocm_event_synchronization
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmEventSynchronization
        COMMAND $<TARGET_FILE:v2_integration_rocm_event_synchronization>
        LABELS "V2;Integration;ROCm;Backend;EventSync;Coherence;Performance;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Event Synchronization tests")

    # Performance Test: HIP Graph Launch Overhead
    # =============================================================================
    # Measures hipGraphLaunch() overhead vs direct kernel dispatch across varying
    # graph sizes (1-732 nodes). Isolates whether GPU graph capture is a net win
    # or loss for decode-step-sized workloads on gfx906 (MI50).
    set_source_files_properties(
        performance/Perf__HIPGraphLaunchOverhead.cpp
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_hip_graph_launch_overhead
        performance/Perf__HIPGraphLaunchOverhead.cpp
    )
    target_link_libraries(v2_perf_hip_graph_launch_overhead
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_HIPGraphLaunchOverhead
        COMMAND $<TARGET_FILE:v2_perf_hip_graph_launch_overhead>
        LABELS "V2;Performance;ROCm;HIPGraph;GraphCapture;LaunchOverhead;gfx906;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added HIP Graph Launch Overhead performance benchmarks")

    # Performance Test: RCCL AllReduce Minimum Latency
    # =============================================================================
    # Measures the minimum achievable RCCL allreduce latency for small transfers
    # (~14 KB = Qwen2.5-7B d_model × sizeof(float)). Explores:
    #   - Latency vs message size (256B → 1MB)
    #   - Group vs no-group overhead
    #   - Batched allreduce (ncclGroupStart/End fusion)
    #   - Sequential vs fused allreduce (56×14KB vs 1×784KB)
    #   - In-place vs out-of-place
    #   - FP16 vs FP32
    #   - Rapid-fire sequential (simulates real decode pipeline)
    # Results guide TP efficiency optimization for local tensor parallelism.
    set_source_files_properties(
        performance/Perf__RCCLAllreduceLatency.cpp
        PROPERTIES LANGUAGE HIP
    )
    add_executable(v2_perf_rccl_allreduce_latency
        performance/Perf__RCCLAllreduceLatency.cpp
    )
    target_link_libraries(v2_perf_rccl_allreduce_latency
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_RCCLAllreduceLatency
        COMMAND $<TARGET_FILE:v2_perf_rccl_allreduce_latency>
        LABELS "V2;Performance;ROCm;RCCL;AllReduce;Latency;TensorParallelism;gfx906;Benchmark"
        MPI_PROCS 1
        TIMEOUT 600
    )
    message(STATUS "V2 Tests: Added RCCL AllReduce Latency performance benchmarks")
endif()

# =============================================================================
# ROCm MoE Kernel Integration Tests
# =============================================================================
if(HAVE_ROCM)
    add_executable(v2_integration_rocm_moe_kernel
        integration/kernels/rocm/Test__ROCmMoEKernel.cpp
    )
    target_link_libraries(v2_integration_rocm_moe_kernel
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmMoEKernel
        COMMAND $<TARGET_FILE:v2_integration_rocm_moe_kernel>
        LABELS "V2;Integration;ROCm;MoE;Routing;Gather;Scatter;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm MoE kernel integration tests")
endif()

# =============================================================================
# GPU Expert Transfer Unit Tests
# =============================================================================
# Tests GPU↔GPU (peer DMA or host-staged) transfer of packed MoE expert
# weight arrays between ROCm devices. Requires ≥2 ROCm GPUs.

if(HAVE_ROCM)
    add_executable(v2_unit_gpu_expert_transfer
        unit/moe/Test__GPUExpertTransfer.cpp
    )
    target_link_libraries(v2_unit_gpu_expert_transfer
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_GPUExpertTransfer
        COMMAND $<TARGET_FILE:v2_unit_gpu_expert_transfer>
        LABELS "V2;Integration;MoE;GPUTransfer;ROCm"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added GPU Expert Transfer integration tests")
endif()

# =============================================================================
# GPU Device Context Integration Tests
# =============================================================================
# Tests GPUDeviceContextPool and IGPUDeviceContext implementations for both
# CUDA (NvidiaDeviceContext) and ROCm (AMDDeviceContext). Validates:
# - Memory allocation/deallocation through contexts
# - Host-to-device and device-to-host transfers
# - Event creation, recording, and synchronization
# - Concurrent access from multiple threads
# - Multi-device scenarios

if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_gpu_device_context
        integration/backends/Test__GPUDeviceContextIntegration.cpp
    )
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_gpu_device_context PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    elseif(UNIX AND NOT APPLE)
        # Must include cuda_backend/rocm_backend with --whole-archive to ensure static
        # initializers (factory registrars) are included even if not directly referenced.
        # Without --whole-archive, the linker won't include AMDContextFactory.cpp.o or
        # NvidiaContextFactory.cu.o since they only contain static initializers.
        set(_gpu_backends_whole "")
        if(HAVE_CUDA)
            list(APPEND _gpu_backends_whole -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
        endif()
        if(HAVE_ROCM)
            list(APPEND _gpu_backends_whole -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
        endif()
        target_link_libraries(v2_integration_gpu_device_context PRIVATE
            -Wl,--start-group
            llaminar2_core
            ${_gpu_backends_whole}
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_integration_gpu_device_context PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_GPUDeviceContextIntegration
        COMMAND $<TARGET_FILE:v2_integration_gpu_device_context>
        LABELS "V2;Integration;Backend;GPUDeviceContext;CUDA;ROCm;Threading;WorkerThread;Memory;Events"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added GPU Device Context Integration tests")
endif()

# =============================================================================
# CollectiveContext GPU Integration Tests
# =============================================================================
# CollectiveContext GPU integration tests
# Split into separate binaries per vendor to avoid ROCm CLR state corruption
# from repeated RCCL init/destroy cycles within a single process.

# --- CUDA-specific CollectiveContext tests ---
if(HAVE_CUDA)
    add_executable(v2_integration_collective_context_cuda
        integration/collective/Test__CollectiveContext_CUDA.cpp
    )
    target_link_libraries(v2_integration_collective_context_cuda PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
    )
    add_v2_test(V2_Integration_CollectiveContext_CUDA
        COMMAND $<TARGET_FILE:v2_integration_collective_context_cuda>
        LABELS "V2;Integration;Collective;CollectiveContext;GPU;CUDA;NCCL;AllReduce;AllGather;Broadcast"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CollectiveContext CUDA integration tests")
endif()

# --- ROCm-specific CollectiveContext tests ---
if(HAVE_ROCM)
    add_executable(v2_integration_collective_context_rocm
        integration/collective/Test__CollectiveContext_ROCm.cpp
    )
    target_link_libraries(v2_integration_collective_context_rocm PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        hip::host
    )
    add_v2_test(V2_Integration_CollectiveContext_ROCm
        COMMAND $<TARGET_FILE:v2_integration_collective_context_rocm>
        LABELS "V2;Integration;Collective;CollectiveContext;GPU;ROCm;RCCL;AllReduce;AllGather;Broadcast"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CollectiveContext ROCm integration tests")
endif()

# --- Metadata / routing tests (backend-agnostic) ---
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_collective_context_gpu
        integration/collective/Test__CollectiveContext_GPU_Metadata.cpp
    )
    target_link_libraries(v2_integration_collective_context_gpu PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    if(HAVE_CUDA)
        target_link_libraries(v2_integration_collective_context_gpu PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_link_libraries(v2_integration_collective_context_gpu PRIVATE hip::host)
    endif()
    add_v2_test(V2_Integration_CollectiveContext_GPU
        COMMAND $<TARGET_FILE:v2_integration_collective_context_gpu>
        LABELS "V2;Integration;Collective;CollectiveContext;GPU;Metadata;Router"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CollectiveContext GPU metadata integration tests")
endif()

# --- HostBackend copy integration tests (CPU↔CPU, CUDA↔ROCm) ---
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_host_backend_copy
        integration/collective/Test__HostBackendCopy.cpp
    )
    target_link_libraries(v2_integration_host_backend_copy PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_HostBackendCopy
        COMMAND $<TARGET_FILE:v2_integration_host_backend_copy>
        LABELS "V2;Integration;Collective;HostBackend;Copy;CrossVendor"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added HostBackend copy integration tests")
endif()

# =============================================================================
# NCCLCoordinator Integration Tests
# =============================================================================
# Tests NCCLCoordinator with actual NVIDIA GPU hardware.
# Validates:
# - Single/multi-GPU initialization
# - AllReduce, AllGather, Broadcast operations
# - Thread safety (concurrent submissions)
# - Shutdown idempotency

if(HAVE_CUDA)
    add_executable(v2_integration_nccl_coordinator
        integration/collective/Test__NCCLCoordinator.cpp
    )
    target_link_libraries(v2_integration_nccl_coordinator PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
    )
    add_v2_test(V2_Integration_NCCLCoordinator
        COMMAND $<TARGET_FILE:v2_integration_nccl_coordinator>
        LABELS "V2;Integration;Collective;NCCLCoordinator;NCCL;CUDA;AllReduce;AllGather;Broadcast;ThreadSafety"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added NCCLCoordinator integration tests")
endif()

# =============================================================================
# RCCLCoordinator Integration Tests
# =============================================================================
# Tests RCCLCoordinator with actual AMD ROCm GPU hardware.
# Validates:
# - Single/multi-GPU initialization
# - AllReduce, AllGather, Broadcast operations
# - Thread safety (concurrent submissions)
# - Shutdown idempotency

if(HAVE_ROCM)
    add_executable(v2_integration_rccl_coordinator
        integration/collective/Test__RCCLCoordinator.cpp
    )
    target_link_libraries(v2_integration_rccl_coordinator PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        hip::host
    )
    add_v2_test(V2_Integration_RCCLCoordinator
        COMMAND $<TARGET_FILE:v2_integration_rccl_coordinator>
        LABELS "V2;Integration;Collective;RCCLCoordinator;RCCL;ROCm;AllReduce;AllGather;Broadcast;ThreadSafety"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added RCCLCoordinator integration tests")
endif()

# =============================================================================
# Backend-Coordinator Integration Tests
# =============================================================================
# Tests that NCCLBackend and RCCLBackend properly delegate to their respective
# coordinators (NCCLCoordinator, RCCLCoordinator) for multi-GPU single-process
# collective operations.
# Validates:
# - Backend initialization with multi-GPU DeviceGroup
# - Backend lifecycle (initialize → operations → shutdown)
# - Proper delegation of allreduceMulti, allgatherMulti, broadcastMulti
# - Error handling when coordinator fails

if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_backend_coordinator
        integration/collective/Test__BackendCoordinatorIntegration.cpp
    )
    target_link_libraries(v2_integration_backend_coordinator PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    # Conditionally link GPU libraries
    if(HAVE_CUDA)
        target_link_libraries(v2_integration_backend_coordinator PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_link_libraries(v2_integration_backend_coordinator PRIVATE hip::host)
    endif()
    add_v2_test(V2_Integration_BackendCoordinatorIntegration
        COMMAND $<TARGET_FILE:v2_integration_backend_coordinator>
        LABELS "V2;Integration;Collective;NCCLBackend;RCCLBackend;NCCLCoordinator;RCCLCoordinator;NCCL;RCCL;CUDA;ROCm;AllReduce;AllGather;Broadcast;MultiGPU"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added Backend-Coordinator integration tests")
endif()

# =============================================================================
# LOCAL TP Multi-Device Integration Tests
# =============================================================================
# Tests RankOrchestrator with real GPU hardware for LOCAL tensor parallelism.
# Validates:
# - Hardware detection (CUDA/ROCm device enumeration)
# - 2-GPU CUDA LOCAL TP (NCCL backend)
# - 2-GPU ROCm LOCAL TP (RCCL backend)
# - Heterogeneous CUDA+ROCm LOCAL TP (HOST backend)
# - Logit parity between single-device and multi-device execution

if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_local_tp_multi_device
        integration/collective/Test__LocalTPMultiDevice.cpp
    )
    target_link_libraries(v2_integration_local_tp_multi_device PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    # Conditionally link GPU libraries
    if(HAVE_CUDA)
        target_link_libraries(v2_integration_local_tp_multi_device PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_link_libraries(v2_integration_local_tp_multi_device PRIVATE hip::host)
    endif()
    add_v2_test(V2_Integration_LocalTP_MultiDevice
        COMMAND $<TARGET_FILE:v2_integration_local_tp_multi_device>
        LABELS "V2;Integration;LocalTP;MultiDevice;RankOrchestrator;TensorParallelism;GPU;NCCL;RCCL;CUDA;ROCm;Heterogeneous"
        MPI_PROCS 1  # LOCAL TP runs within a single MPI rank
    )
    message(STATUS "V2 Tests: Added LOCAL TP Multi-Device integration tests")
endif()

# =============================================================================
# ITPContext LocalTP Unit Tests (CPU-only, no GPU required)
# =============================================================================
# Tests the ITPContext interface with LocalTPContext implementation using HOST
# backend. These tests validate interface compliance, polymorphism, and device
# index management - not actual collective tensor operations.

add_executable(v2_unit_itpcontext_local_tp
    unit/collective/Test__ITPContext_LocalTP.cpp
)
target_link_libraries(v2_unit_itpcontext_local_tp PRIVATE
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ITPContext_LocalTP
    COMMAND $<TARGET_FILE:v2_unit_itpcontext_local_tp>
    LABELS "V2;Unit;ITPContext;LocalTP;Interface;Polymorphism;DeviceIndex"
    MPI_PROCS 1  # LOCAL TP runs within a single MPI rank
)
message(STATUS "V2 Tests: Added ITPContext LocalTP unit tests (CPU-only)")

# =============================================================================
# LOCAL TP Backend Behavior Integration Tests
# =============================================================================
# Tests for LocalTPContext backend auto-detection, barrier synchronization,\n# and tensor management. Migrated from unit tests that require real hardware.

if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_local_tp_backend_behavior
        integration/collective/Test__LocalTPBackendBehavior.cpp
    )
    target_link_libraries(v2_integration_local_tp_backend_behavior PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    # Conditionally link GPU libraries
    if(HAVE_CUDA)
        target_link_libraries(v2_integration_local_tp_backend_behavior PRIVATE CUDA::cudart)
    endif()
    if(HAVE_ROCM)
        target_link_libraries(v2_integration_local_tp_backend_behavior PRIVATE hip::host)
    endif()
    add_v2_test(V2_Integration_LocalTP_BackendBehavior
        COMMAND $<TARGET_FILE:v2_integration_local_tp_backend_behavior>
        LABELS "V2;Integration;LocalTP;BackendBehavior;AutoDetection;Barrier;BARManagement;CUDA;ROCm;Heterogeneous"
        MPI_PROCS 1  # LOCAL TP runs within a single MPI rank
    )
    message(STATUS "V2 Tests: Added LOCAL TP Backend Behavior integration tests")
endif()

# =============================================================================
# CUDA/HIP Kernel Coexistence Test
# =============================================================================
# Minimal test to verify CUDA kernels work after HIP/ROCm operations.
# Used to diagnose "context is destroyed" (error 709) issues.

if(HAVE_CUDA AND HAVE_ROCM)
    add_executable(v2_integration_cuda_hip_coexistence
        diagnostics/Test__CUDAHIPKernelCoexistence.cpp
    )
    target_link_libraries(v2_integration_cuda_hip_coexistence PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
        hip::host
    )
    add_v2_test(V2_Integration_CUDAHIPCoexistence
        COMMAND $<TARGET_FILE:v2_integration_cuda_hip_coexistence>
        LABELS "V2;Integration;CUDA;ROCm;Heterogeneous;Diagnostics"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CUDA/HIP Kernel Coexistence diagnostic test")
endif()

# =============================================================================
# Phase 5: Prefill→Decode Transition Unit Tests (Regression)
# =============================================================================
# Regression test for the prefill→decode last-token double-processing bug.
# Uses a mock IInferenceRunner to verify that the first decodeStep() samples
# from prefill logits instead of re-feeding the last prompt token.
add_executable(v2_unit_prefill_decode_transition
    unit/execution/runner/Test__PrefillDecodeTransition.cpp
)
target_link_libraries(v2_unit_prefill_decode_transition
    llaminar2_core
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)
add_v2_test(V2_Unit_PrefillDecodeTransition
    COMMAND $<TARGET_FILE:v2_unit_prefill_decode_transition>
    LABELS "V2;Unit;Orchestration;OrchestrationRunner;Regression;PrefillDecode"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added prefill→decode transition regression tests")

# MPI Coordinated Mode unit tests
# Tests the MPI worker loop protocol where rank 0 broadcasts commands
# to non-root ranks for server/interactive mode inference coordination.
add_executable(v2_unit_mpi_coordinated_mode
    unit/execution/runner/Test__MPICoordinatedMode.cpp
)
target_link_libraries(v2_unit_mpi_coordinated_mode
    llaminar2_core
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)
add_v2_test(V2_Unit_MPICoordinatedMode
    COMMAND $<TARGET_FILE:v2_unit_mpi_coordinated_mode>
    LABELS "V2;Unit;Orchestration;OrchestrationRunner;MPI;CoordinatedMode"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added MPI coordinated mode unit tests")

# MoE overlay plan freezing unit tests
# Locks in that OrchestrationRunner uses model metadata to freeze YAML overlay
# plans before root/endpoint graph construction.
add_executable(v2_unit_orchestration_runner_moe_overlay_plan
    unit/execution/runner/Test__OrchestrationRunnerMoEOverlayPlan.cpp
)
target_include_directories(v2_unit_orchestration_runner_moe_overlay_plan PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_unit_orchestration_runner_moe_overlay_plan
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OrchestrationRunner_MoEOverlayPlan
    COMMAND $<TARGET_FILE:v2_unit_orchestration_runner_moe_overlay_plan>
    LABELS "V2;Unit;Orchestration;OrchestrationRunner;MoE;Overlay;ExpertParallel"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added OrchestrationRunner MoE overlay plan unit tests")

# Model file validation unit tests
# Locks in that OrchestrationRunner hard-fails on missing/invalid model files
# instead of silently falling back to defaults.
add_executable(v2_unit_orchestration_runner_model_validation
    unit/execution/runner/Test__OrchestrationRunnerModelValidation.cpp
)
target_include_directories(v2_unit_orchestration_runner_model_validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(v2_unit_orchestration_runner_model_validation
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OrchestrationRunner_ModelValidation
    COMMAND $<TARGET_FILE:v2_unit_orchestration_runner_model_validation>
    LABELS "V2;Unit;Orchestration;OrchestrationRunner;ModelValidation"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added OrchestrationRunner model validation unit tests")

# Node-leader page cache coordination unit tests
# Tests the per-node-leader page cache prepopulation logic using MockMPITopology.
add_executable(v2_unit_node_leader_page_cache
    unit/execution/runner/Test__NodeLeaderPageCache.cpp
)
target_include_directories(v2_unit_node_leader_page_cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_node_leader_page_cache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_NodeLeaderPageCache
    COMMAND $<TARGET_FILE:v2_unit_node_leader_page_cache>
    LABELS "V2;Unit;Orchestration;OrchestrationRunner;PageCache;Topology"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added node-leader page cache coordination tests")

# =============================================================================
# Phase 5: OrchestrationRunner Integration Tests
# =============================================================================
# Tests the end-to-end OrchestrationRunner that wires all Phase 0-4 components.
# Validates factory creation, configuration, execution plan building, and inference lifecycle.

add_executable(v2_integration_orchestration_runner
    integration/execution/runner/Test__OrchestrationRunner.cpp
)
target_include_directories(v2_integration_orchestration_runner PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_integration_orchestration_runner
    llaminar2_core
    GTest::gtest
    GTest::gmock
)
add_v2_test(V2_Integration_OrchestrationRunner
    COMMAND $<TARGET_FILE:v2_integration_orchestration_runner>
    LABELS "V2;Integration;Orchestration;OrchestrationRunner;Phase5;Factory;Lifecycle"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added OrchestrationRunner integration tests")

# Test: Scenario 7 Multi-Domain Pipeline Parallel
# Validates end-to-end integration for multi-domain PP with heterogeneous devices.
# Hardware: GPU domain (CUDA) -> GPU domain (ROCm) -> CPU domain (2 sockets)
add_executable(v2_integration_scenario7_multidomain_pp
    integration/execution/runner/Test__Scenario7_MultiDomainPP.cpp
)
target_include_directories(v2_integration_scenario7_multidomain_pp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
    ${CMAKE_CURRENT_SOURCE_DIR}/integration/orchestration  # For OrchestrationTestBase.h
)
target_link_libraries(v2_integration_scenario7_multidomain_pp
    llaminar2_core
    GTest::gtest
    GTest::gmock
)
add_v2_test(V2_Integration_Scenario7_MultiDomainPP
    COMMAND $<TARGET_FILE:v2_integration_scenario7_multidomain_pp>
    LABELS "V2;Integration;Orchestration;Scenario7;MultiDomainPP;PipelineParallel;Heterogeneous;CUDA;ROCm;CPU;Phase5"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added Scenario 7 Multi-Domain PP integration tests")

# Test: PPStageRunner Integration (createPPStageRunner factory function)
# Validates Pipeline Parallelism stage runner creation with various configurations:
# - First stage (embedding + initial layers)
# - Middle stage (layers only)
# - Last stage (final layers + LM head)
# - Invalid configuration rejection
add_executable(v2_integration_ppstage_runner
    integration/execution/runner/Test__PPStageRunner.cpp
)
target_include_directories(v2_integration_ppstage_runner PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/ include path
)
target_link_libraries(v2_integration_ppstage_runner
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_PPStageRunner
    COMMAND $<TARGET_FILE:v2_integration_ppstage_runner>
    LABELS "V2;Integration;PipelineParallel;PPStageRunner;Factory;Execution"
    MPI_PROCS 1
    TIMEOUT 1800
)
message(STATUS "V2 Tests: Added PPStageRunner integration tests")

# =============================================================================
# Performance: CUDA Vector Add Kernels (used by host-staged reductions)
# =============================================================================
# Benchmarks the in-place vector addition kernels that perform GPU-side reductions
# during host-staged allreduce operations. Tests realistic sizes (hidden_size, intermediate_size,
# vocab_size) and measures throughput (GB/s), latency, and kernel launch overhead.
if(HAVE_CUDA)
add_executable(v2_perf_cuda_vector_add performance/kernels/cuda/ops/Perf__CUDAVectorAddKernels.cpp)
target_link_libraries(v2_perf_cuda_vector_add llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Perf_CUDAVectorAddKernels
    COMMAND $<TARGET_FILE:v2_perf_cuda_vector_add>
    LABELS "V2;Performance;CUDA;Kernels;VectorAdd;Allreduce;Throughput;Latency"
    MPI_PROCS 1
)
message(STATUS "V2 Tests: Added CUDA Vector Add kernel performance benchmarks")
endif()

# =============================================================================
# Global Pipeline Parallelism (Global PP) Tests
# =============================================================================

# Test: GlobalPPTransferStage
# Unit tests for the MPI send/recv compute stage used for cross-rank activation
# transfer in Global PP. Validates stage type, naming, coherence policy,
# buffer requirements, dump info, parameter validation. No MPI required (mock context).
add_executable(v2_test_global_pp_transfer_stage unit/execution/compute_stages/stages/Test__GlobalPPTransferStage.cpp)
target_link_libraries(v2_test_global_pp_transfer_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GlobalPPTransferStage
    COMMAND $<TARGET_FILE:v2_test_global_pp_transfer_stage>
    LABELS "V2;Unit;GlobalPP;Transfer;ComputeStage;PipelineParallel;MPI"
    MPI_PROCS 1
)

# Test: GlobalPPTopology + GlobalPPRankPlanBuilder
# Unit tests for topology data structures, validation (layer coverage, contiguity,
# embedding/lm_head placement, rank bounds), transfer derivation (single→single,
# single→globalTP fan-out, same-rank no-op), rank plan building (step ordering,
# weight shard info, transfer direction matching). Pure data-structure tests.
add_executable(v2_test_global_pp_topology unit/execution/global_pp/Test__GlobalPPTopology.cpp)
target_link_libraries(v2_test_global_pp_topology llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GlobalPPTopology
    COMMAND $<TARGET_FILE:v2_test_global_pp_topology>
    LABELS "V2;Unit;GlobalPP;Topology;RankPlan;PipelineParallel;TensorParallel"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added Global PP unit tests")

# Test: GlobalOrchestrator
# Unit tests for the cross-machine MPI cluster inference orchestrator (Phase 1).
# Tests construction, validation, forward delegation, logits access (tail rank only),
# clear_cache with barrier, greedy/full sampling with MPI_Bcast, pipeline head/tail
# detection, global TP and PP topologies. Uses MockMPIContext and MockDeviceRunner.
add_executable(v2_test_global_orchestrator unit/execution/global/Test__GlobalOrchestrator.cpp)
target_include_directories(v2_test_global_orchestrator PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_global_orchestrator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GlobalOrchestrator
    COMMAND $<TARGET_FILE:v2_test_global_orchestrator>
    LABELS "V2;Unit;GlobalOrchestrator;GlobalPP;PipelineParallel;TensorParallel;MPI"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added GlobalOrchestrator unit tests")

# Test: StageRunnerFactory
# Unit tests for StageRunnerFactory (Phase 3 of Multi-Domain Pipeline Plan).
# Tests single-device, local TP, and global TP stage building via model-light
# MockModelContext. Global TP uses injected GlobalTPContext (MPI_COMM_SELF, 1 rank).
# Uses mpi_gtest_main.cpp because GlobalTPContext::createForTest calls MPI_Comm_rank.
add_executable(v2_test_stage_runner_factory
    unit/execution/global/Test__StageRunnerFactory.cpp
    integration/mpi_gtest_main.cpp)
target_include_directories(v2_test_stage_runner_factory PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_stage_runner_factory llaminar2_core GTest::gtest MPI::MPI_CXX)
add_v2_test(V2_Unit_StageRunnerFactory
    COMMAND $<TARGET_FILE:v2_test_stage_runner_factory>
    LABELS "V2;Unit;StageRunnerFactory;GlobalPP;PipelineParallel;TensorParallel;DomainComm"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added StageRunnerFactory unit tests")

# Test: DomainCommunicatorRegistry (MPI integration, 3 ranks)
# Integration tests for DomainCommunicatorRegistry (Phase 3).
# Two global TP stages with overlapping participant sets {0,1} and {1,2}.
# Asserts: rank 0 gets only stage 0 ctx, rank 1 gets both, rank 2 gets only stage 1.
# Verifies worldRanks(), degree(), domainId(), myIndex() correctness.
add_executable(v2_test_domain_communicator_registry_mpi
    integration/execution/Test__DomainCommunicatorRegistry_MPI.cpp
    integration/mpi_gtest_main.cpp)
target_link_libraries(v2_test_domain_communicator_registry_mpi llaminar2_core GTest::gtest MPI::MPI_CXX)
add_v2_test(V2_Integration_DomainCommunicatorRegistry_MPI
    COMMAND $<TARGET_FILE:v2_test_domain_communicator_registry_mpi>
    LABELS "V2;Integration;DomainCommunicatorRegistry;GlobalTP;MPI;PipelineParallel"
    MPI_PROCS 3
)

message(STATUS "V2 Tests: Added DomainCommunicatorRegistry MPI integration tests")

# Test: GlobalOrchestratorRunner
# Unit tests for IOrchestrationRunner adapter wrapping GlobalOrchestrator (Phase 4).
# Verifies lifecycle (init/shutdown), inference delegation (prefill/decodeStep/generate),
# sampling state, stop tokens, GPU sampling delegation, PP topologies.
add_executable(v2_test_global_orchestrator_runner unit/execution/global/Test__GlobalOrchestratorRunner.cpp)
target_include_directories(v2_test_global_orchestrator_runner PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_global_orchestrator_runner llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GlobalOrchestratorRunner
    COMMAND $<TARGET_FILE:v2_test_global_orchestrator_runner>
    LABELS "V2;Unit;GlobalOrchestratorRunner;GlobalOrchestrator;GlobalPP;IOrchestrationRunner"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added GlobalOrchestratorRunner unit tests")

# Test: ParallelismTree
# Unit tests for recursive parallelism tree data structure:
# - Fluent builder construction (PP, TP, Device)
# - Layer assignment (equal split, odd split, nested PP/TP)
# - Validation (TP child count, rank bounds, zero layers, empty children)
# - Query methods (leafRanks, leafDevices, isCrossRank, leafCount)
# - toString visualization
# - Edge cases (single device, pure TP, deep nesting, equality)
add_executable(v2_test_parallelism_tree unit/execution/parallelism_tree/Test__ParallelismTree.cpp)
target_link_libraries(v2_test_parallelism_tree llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ParallelismTree
    COMMAND $<TARGET_FILE:v2_test_parallelism_tree>
    LABELS "V2;Unit;ParallelismTree;PipelineParallel;TensorParallel;Topology"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added ParallelismTree unit tests")

# Test: TransferSpec
# Unit tests for transfer mechanism derivation between adjacent PP stages:
# - LOCAL_PP: Same-rank transfer (NCCL/HOST)
# - MPI_INTRAHOST: Cross-rank on same machine
# - MPI_INTERHOST: Cross-rank on different machines
# - Tag assignment and rank selection
add_executable(v2_test_transfer_spec unit/execution/parallelism_tree/Test__TransferSpec.cpp)
target_link_libraries(v2_test_transfer_spec llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TransferSpec
    COMMAND $<TARGET_FILE:v2_test_transfer_spec>
    LABELS "V2;Unit;Topology;PipelineParallel;TensorParallel;TransferSpec"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added TransferSpec unit tests")

# Test: CompoundShardInfo
# Unit tests for multi-level shard descriptor:
# - Layer range from PP assignment
# - TP shard index and total (compound for nested TP)
# - Work fraction (for proportional TP)
# - Validation of shard info consistency
add_executable(v2_test_compound_shard_info unit/execution/parallelism_tree/Test__CompoundShardInfo.cpp)
target_link_libraries(v2_test_compound_shard_info llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CompoundShardInfo
    COMMAND $<TARGET_FILE:v2_test_compound_shard_info>
    LABELS "V2;Unit;Topology;PipelineParallel;TensorParallel;ShardInfo"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added CompoundShardInfo unit tests")

# Test: HierarchicalCommBuilder
# Unit tests for MPI communicator hierarchy construction:
# - needsCommunicator() for device/TP/PP nodes
# - calculateTPColor() for domain membership
# - CommHierarchy RAII wrapper (add, get, move, toString)
# - Hierarchy building for various topologies
# - Edge cases (rank not in domain, PP-only, mixed local+cross-rank)
add_executable(v2_test_hierarchical_comm_builder unit/execution/parallelism_tree/Test__HierarchicalCommBuilder.cpp)
target_link_libraries(v2_test_hierarchical_comm_builder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_HierarchicalCommBuilder
    COMMAND $<TARGET_FILE:v2_test_hierarchical_comm_builder>
    LABELS "V2;Unit;Topology;PipelineParallel;TensorParallel;MPI"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added HierarchicalCommBuilder unit tests")

# Test: PipelineRunner
# Unit tests for cross-rank pipeline runner:
# - Construction with valid/invalid parameters
# - myStageIndex() for various rank configurations
# - vocab_size(), executionPath(), architecture() accessors
# - clear_cache() propagation to owned stages
# - forward() updates position and calls stage forward
# - hasEmbedding() / hasLMHead() reflect stage config
add_executable(v2_test_pipeline_runner unit/execution/parallelism_tree/Test__PipelineRunner.cpp)
target_link_libraries(v2_test_pipeline_runner llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PipelineRunner
    COMMAND $<TARGET_FILE:v2_test_pipeline_runner>
    LABELS "V2;Unit;Topology;PipelineParallel"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added PipelineRunner unit tests")

# Test: TreeToRunnerCompiler
# Unit tests for tree-to-runner compilation:
# - shouldCompile() logic for DEVICE/TP/PP nodes
# - isCrossRankPP() detection
# - Compile with mock factories (device, TP, PP)
# - Cross-rank PP → PipelineRunner creation
# - Nested TP+PP compilation
# - Layer range propagation to runner configs
add_executable(v2_test_tree_to_runner_compiler unit/execution/parallelism_tree/Test__TreeToRunnerCompiler.cpp)
target_link_libraries(v2_test_tree_to_runner_compiler llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TreeToRunnerCompiler
    COMMAND $<TARGET_FILE:v2_test_tree_to_runner_compiler>
    LABELS "V2;Unit;Topology;PipelineParallel;TensorParallel"
    MPI_PROCS 1
)

message(STATUS "V2 Tests: Added TreeToRunnerCompiler unit tests")

# Test: HIP Graph Capture - ROCm graph capture/replay infrastructure
# Tests IGPUGraphCapture interface via HIPGraphCapture: lifecycle, error paths, ownership
if(HAVE_ROCM)
    add_executable(v2_integration_hip_graph_capture integration/backends/Test__HIPGraphCapture.cpp)
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_hip_graph_capture PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_integration_hip_graph_capture PRIVATE
            -Wl,--start-group
            llaminar2_core
            -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_HIPGraphCapture
        COMMAND $<TARGET_FILE:v2_integration_hip_graph_capture>
        LABELS "V2;Integration;Backends;GraphCapture;ROCm;HIP"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added HIP Graph Capture integration tests (ROCm)")
endif()

# Test: CUDA Graph Capture - NVIDIA graph capture/replay infrastructure
# Tests IGPUGraphCapture interface via CUDAGraphCapture: lifecycle, error paths, ownership
if(HAVE_CUDA)
    add_executable(v2_integration_cuda_graph_capture integration/backends/Test__CUDAGraphCapture.cpp)
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_cuda_graph_capture PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_integration_cuda_graph_capture PRIVATE
            -Wl,--start-group
            llaminar2_core
            -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_CUDAGraphCapture
        COMMAND $<TARGET_FILE:v2_integration_cuda_graph_capture>
        LABELS "V2;Integration;Backends;GraphCapture;CUDA"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added CUDA Graph Capture integration tests")
endif()

# =============================================================================
# Unit: SnapshotCapture (Phase 2 DGO refactor extraction)
# =============================================================================

add_executable(v2_test_snapshot_capture unit/snapshots/Test__SnapshotCapture.cpp)
target_link_libraries(v2_test_snapshot_capture llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_SnapshotCapture
    COMMAND $<TARGET_FILE:v2_test_snapshot_capture>
    LABELS "V2;Unit;Snapshot;SnapshotCapture;DGORefactor"
    MPI_PROCS 1
)

# =============================================================================
# Unit: ForwardGraphTypes (Phase 3 DGO refactor extraction)
# =============================================================================

add_executable(v2_test_forward_graph_types unit/execution/local_execution/engine/Test__ForwardGraphTypes.cpp)
target_link_libraries(v2_test_forward_graph_types llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ForwardGraphTypes
    COMMAND $<TARGET_FILE:v2_test_forward_graph_types>
    LABELS "V2;Unit;Execution;ForwardGraph;DGORefactor"
    MPI_PROCS 1
)

# =============================================================================
# Unit: ForwardExecutionEngine (Phase 3 DGO refactor extraction)
# =============================================================================

add_executable(v2_test_forward_execution_engine unit/execution/local_execution/engine/Test__ForwardExecutionEngine.cpp)
target_link_libraries(v2_test_forward_execution_engine llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ForwardExecutionEngine
    COMMAND $<TARGET_FILE:v2_test_forward_execution_engine>
    LABELS "V2;Unit;Execution;ForwardExecutionEngine;DGORefactor"
    MPI_PROCS 1
)

# Unit: ForwardExecutionEngine Advanced (host callbacks, phase transitions, PP/TP handoff)
# =============================================================================

add_executable(v2_test_forward_execution_engine_advanced unit/execution/local_execution/engine/Test__ForwardExecutionEngineAdvanced.cpp)
target_link_libraries(v2_test_forward_execution_engine_advanced llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ForwardExecutionEngineAdvanced
    COMMAND $<TARGET_FILE:v2_test_forward_execution_engine_advanced>
    LABELS "V2;Unit;Execution;ForwardExecutionEngine;HostCallbacks;PhaseTransitions;PPHandoff"
    MPI_PROCS 1
)

# Test: GPUGraphCaptureExecution - Integration test for GPU graph capture in DeviceGraphExecutor
add_v2_gpu_backend_test(
    v2_test_gpu_graph_capture_execution
    integration/execution/graph/Test__GPUGraphCaptureExecution.cpp
    V2_Integration_GPUGraphCaptureExecution
    V2 Integration Execution GraphCapture
)
message(STATUS "V2 Tests: Added GPUGraphCaptureExecution integration tests")

# Tests: PrefillGraphCacheExecution - real GPU exact-bucket prefill graph cache lifecycle.
# These are explicit per-backend integration tests rather than a macro-generated
# pair so CTest output is direct: one ROCm test and one CUDA test.
if(HAVE_ROCM)
    add_executable(v2_test_prefill_graph_cache_execution_rocm
        integration/execution/graph/Test__ROCmPrefillGraphCacheExecution.cpp
    )
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_test_prefill_graph_cache_execution_rocm PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_test_prefill_graph_cache_execution_rocm PRIVATE
            -Wl,--start-group
            llaminar2_core
            -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_PrefillGraphCacheExecution_ROCm
        COMMAND $<TARGET_FILE:v2_test_prefill_graph_cache_execution_rocm>
        LABELS "V2;Integration;Execution;GraphCapture;PrefillGraphCache;Buckets;ROCm"
        MPI_PROCS 1
    )
endif()

if(HAVE_CUDA)
    add_executable(v2_test_prefill_graph_cache_execution_cuda
        integration/execution/graph/Test__CUDAPrefillGraphCacheExecution.cpp
    )
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_test_prefill_graph_cache_execution_cuda PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_test_prefill_graph_cache_execution_cuda PRIVATE
            -Wl,--start-group
            llaminar2_core
            -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_PrefillGraphCacheExecution_CUDA
        COMMAND $<TARGET_FILE:v2_test_prefill_graph_cache_execution_cuda>
        LABELS "V2;Integration;Execution;GraphCapture;PrefillGraphCache;Buckets;CUDA"
        MPI_PROCS 1
    )
endif()

foreach(_prefill_graph_cache_test
        V2_Integration_PrefillGraphCacheExecution_ROCm
        V2_Integration_PrefillGraphCacheExecution_CUDA)
    if(TEST ${_prefill_graph_cache_test})
        set_tests_properties(${_prefill_graph_cache_test} PROPERTIES
            ENVIRONMENT "LLAMINAR_VALIDATE_BUFFERS=0;LLAMINAR_VALIDATE_INPUTS=0;LLAMINAR_FAIL_ON_ZERO=0"
        )
    endif()
endforeach()
message(STATUS "V2 Tests: Added PrefillGraphCacheExecution integration tests")

# Test: SegmentedGraphCaptureExecution - Integration test for segmented GPU graph capture/replay
add_executable(v2_test_segmented_graph_capture_execution integration/execution/graph/Test__SegmentedGraphCaptureExecution.cpp)
if(LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_test_segmented_graph_capture_execution PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
elseif(UNIX AND NOT APPLE AND (HAVE_CUDA OR HAVE_ROCM))
    set(_gpu_backends_whole_sgce "")
    if(HAVE_CUDA)
        list(APPEND _gpu_backends_whole_sgce -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
    endif()
    if(HAVE_ROCM)
        list(APPEND _gpu_backends_whole_sgce -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
    endif()
    target_link_libraries(v2_test_segmented_graph_capture_execution PRIVATE
        -Wl,--start-group
        llaminar2_core
        ${_gpu_backends_whole_sgce}
        -Wl,--end-group
        GTest::gtest
        GTest::gtest_main
    )
else()
    target_link_libraries(v2_test_segmented_graph_capture_execution PRIVATE
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
endif()
add_v2_test(V2_Integration_SegmentedGraphCaptureExecution
    COMMAND $<TARGET_FILE:v2_test_segmented_graph_capture_execution>
    LABELS "V2;Integration;Execution;GraphCapture;Segmented;CUDA;ROCm"
    MPI_PROCS 1
)
if(NOT V2_PERF_TESTS_ONLY)
set_tests_properties(V2_Integration_SegmentedGraphCaptureExecution PROPERTIES
    ENVIRONMENT "LLAMINAR_VALIDATE_BUFFERS=0;LLAMINAR_VALIDATE_INPUTS=0;LLAMINAR_FAIL_ON_ZERO=0"
)
endif()
message(STATUS "V2 Tests: Added SegmentedGraphCaptureExecution integration tests")

# =============================================================================
# GPU Sampling Kernels Integration Tests (argmaxF32, topKF32)
# =============================================================================
# Validates GPU-side sampling primitives for both CUDA and ROCm backends.
# Covers: argmax correctness, top-k ranking, numerical stability, edge cases,
# large vocabulary (Qwen2 151936), and stress tests.
# Mirrors the CPU sampler unit tests (Test__Sampler.cpp) for the GPU kernel API.

if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_gpu_sampling_kernels
        integration/backends/Test__GPUSamplingKernels.cpp
    )
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_gpu_sampling_kernels PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    elseif(UNIX AND NOT APPLE)
        set(_gpu_backends_whole_sampling "")
        if(HAVE_CUDA)
            list(APPEND _gpu_backends_whole_sampling -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
        endif()
        if(HAVE_ROCM)
            list(APPEND _gpu_backends_whole_sampling -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
        endif()
        target_link_libraries(v2_integration_gpu_sampling_kernels PRIVATE
            -Wl,--start-group
            llaminar2_core
            ${_gpu_backends_whole_sampling}
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_integration_gpu_sampling_kernels PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_GPUSamplingKernels
        COMMAND $<TARGET_FILE:v2_integration_gpu_sampling_kernels>
        LABELS "V2;Integration;Backend;GPU;CUDA;ROCm;Sampling;Argmax;TopK"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added GPU Sampling Kernels integration tests")
endif()

# =============================================================================
# E2E Server Integration Tests (REST API)
# =============================================================================
# Shell-based tests that spin up llaminar2 --serve and test via curl.
# Bypasses mpirun wrapper since the script manages its own server processes.

if(NOT V2_PERF_TESTS_ONLY)
add_test(
    NAME V2_E2E_Server_MultiTurn
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/e2e/server/test_server_e2e.sh
        --binary $<TARGET_FILE:llaminar2>
)
set_tests_properties(V2_E2E_Server_MultiTurn PROPERTIES
    LABELS "V2;E2E;Server;REST;MultiTurn;CPU;CUDA;ROCm"
    TIMEOUT 1200
    ENVIRONMENT "LLAMINAR_LOG_LEVEL=WARN"
)
message(STATUS "V2 Tests: Added E2E Server multi-turn integration tests")
endif()

# =============================================================================
# Performance: TurboQuant TQ4 (quantize / dequantize / native dot)
# =============================================================================

add_executable(v2_perf_turboquant performance/kernels/cpu/turboquant/Perf__TurboQuant.cpp)
target_link_libraries(v2_perf_turboquant
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_perf_turboquant PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Perf_TurboQuant
    COMMAND $<TARGET_FILE:v2_perf_turboquant>
    LABELS "V2;Performance;TurboQuant;TQ4;TQ8;Quantize;Dequantize;NativeDot;KVCache;SIMD;AVX512"
    MPI_PROCS 1
)

# =============================================================================
# Python Reference Module Tests (PyTorch parity infrastructure)
# =============================================================================
# The python/reference/ package provides the PyTorch reference implementation
# used by C++ parity tests. Pytest auto-discovers all test_*.py files under
# python/reference/tests/, so new pure-unit test files are picked up
# automatically without editing this CMakeLists.txt.
#
# Only fast unit tests run here (no GGUF model loading, ~15s total).
# Integration tests (model loading, inference) are excluded via --ignore
# for pure-integration files and -k filters for mixed files.
#
# Python3 is already found by the parent CMakeLists.txt (src/v2/CMakeLists.txt).

set(PYTHON_REFERENCE_ROOT "${CMAKE_SOURCE_DIR}/../..")

if(NOT V2_PERF_TESTS_ONLY)
# Auto-discovers every test_*.py in python/reference/tests/ EXCEPT:
#   --ignore: pure-integration files (all classes load GGUF models)
#   -k:       integration classes in mixed files (GGUFLoading, Inference, etc.)
add_test(
    NAME V2_Unit_Python_Reference_All
    COMMAND ${Python3_EXECUTABLE} -m pytest
        python/reference/tests/
        --ignore=python/reference/tests/test_gguf_parser.py
        --ignore=python/reference/tests/test_gguf_inference.py
        -v --tb=short
        -k "not (GGUFLoading or Inference or GGUFCompatLoading or GGUFDimensionOrder or DequantizationVsLlamaCpp)"
)
set_tests_properties(V2_Unit_Python_Reference_All PROPERTIES
    LABELS "V2;Unit;Python;Reference;Dequantize;GGUFParser;Qwen35;MoE;GatedDeltaNet"
    WORKING_DIRECTORY "${PYTHON_REFERENCE_ROOT}"
    TIMEOUT 900
)
endif()

# =============================================================================
# MoE Infrastructure Unit Tests
# =============================================================================

# Test: MoERouter — SoftmaxTopKRouter: softmax, top-k selection, normalization
add_executable(v2_test_moe_router unit/moe/Test__MoERouter.cpp)
target_link_libraries(v2_test_moe_router
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoERouter
    COMMAND $<TARGET_FILE:v2_test_moe_router>
    LABELS "V2;Unit;MoE;Router;Softmax"
)

# Test: MoEDispatcher — StandardMoEDispatcher: token grouping by expert
add_executable(v2_test_moe_dispatcher unit/moe/Test__MoEDispatcher.cpp)
target_link_libraries(v2_test_moe_dispatcher
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEDispatcher
    COMMAND $<TARGET_FILE:v2_test_moe_dispatcher>
    LABELS "V2;Unit;MoE;Dispatcher"
)

# Test: MoEExpertExecutor — CPUMoEExpertExecutor: SwiGLU FFN correctness
add_executable(v2_test_moe_expert_executor unit/moe/Test__MoEExpertExecutor.cpp)
target_link_libraries(v2_test_moe_expert_executor
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEExpertExecutor
    COMMAND $<TARGET_FILE:v2_test_moe_expert_executor>
    LABELS "V2;Unit;MoE;Expert;SwiGLU;FFN"
)

# Test: MoECombiner — CPUMoECombiner: weighted combination of expert outputs
add_executable(v2_test_moe_combiner unit/moe/Test__MoECombiner.cpp)
target_link_libraries(v2_test_moe_combiner
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoECombiner
    COMMAND $<TARGET_FILE:v2_test_moe_combiner>
    LABELS "V2;Unit;MoE;Combiner"
)

# Test: ExpertPlacementMap — placement strategies, usage tracking, rebalancing
add_executable(v2_test_expert_placement_map unit/moe/Test__ExpertPlacementMap.cpp)
target_link_libraries(v2_test_expert_placement_map
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpertPlacementMap
    COMMAND $<TARGET_FILE:v2_test_expert_placement_map>
    LABELS "V2;Unit;MoE;Placement;WeightPlacement"
)

# Test: ExpertWeightCache — LRU/FIFO eviction, mock IExpertWeightStorage
add_executable(v2_test_expert_weight_cache unit/moe/Test__ExpertWeightCache.cpp)
target_link_libraries(v2_test_expert_weight_cache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpertWeightCache
    COMMAND $<TARGET_FILE:v2_test_expert_weight_cache>
    LABELS "V2;Unit;MoE;Cache;LRU;WeightCache"
)

# Test: ExpertWeightPayloadProvider — metadata-based host retention lifecycle
add_executable(v2_test_expert_weight_payload_provider unit/moe/Test__ExpertWeightPayloadProvider.cpp)
target_link_libraries(v2_test_expert_weight_payload_provider
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpertWeightPayloadProvider
    COMMAND $<TARGET_FILE:v2_test_expert_weight_payload_provider>
    LABELS "V2;Unit;MoE;WeightLifecycle;PayloadProvider"
)

# Test: ExpertRebalancer — histogram-based dynamic weight movement
add_executable(v2_test_expert_rebalancer unit/moe/Test__ExpertRebalancer.cpp)
target_link_libraries(v2_test_expert_rebalancer
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpertRebalancer
    COMMAND $<TARGET_FILE:v2_test_expert_rebalancer>
    LABELS "V2;Unit;MoE;Rebalancer;Histogram;DynamicPlacement"
)

# Test: DecodeExpertHistogram — per-layer decode expert utilization tracking
add_executable(v2_test_decode_expert_histogram unit/moe/Test__DecodeExpertHistogram.cpp)
target_link_libraries(v2_test_decode_expert_histogram
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DecodeExpertHistogram
    COMMAND $<TARGET_FILE:v2_test_decode_expert_histogram>
    LABELS "V2;Unit;MoE;Histogram;DecodeTracking;DynamicPlacement"
)

# Test: MoERuntimeTable — graph-facing runtime placement tables and bank flips
add_executable(v2_test_moe_runtime_table unit/execution/moe/Test__MoERuntimeTable.cpp)
target_link_libraries(v2_test_moe_runtime_table
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoERuntimeTable
    COMMAND $<TARGET_FILE:v2_test_moe_runtime_table>
    LABELS "V2;Unit;MoE;RuntimeTable;GraphCapture;DynamicPlacement"
)

# Test: SocketAwareRebalancer — socket-load-aware expert swap proposals
add_executable(v2_test_socket_aware_rebalancer unit/moe/Test__SocketAwareRebalancer.cpp)
target_link_libraries(v2_test_socket_aware_rebalancer
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SocketAwareRebalancer
    COMMAND $<TARGET_FILE:v2_test_socket_aware_rebalancer>
    LABELS "V2;Unit;MoE;Rebalancer;SocketAware;DynamicPlacement"
)

# Test: MoERebalanceController — histogram + rebalancer orchestration
add_executable(v2_test_moe_rebalance_controller unit/moe/Test__MoERebalanceController.cpp)
target_link_libraries(v2_test_moe_rebalance_controller
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoERebalanceController
    COMMAND $<TARGET_FILE:v2_test_moe_rebalance_controller>
    LABELS "V2;Unit;MoE;Rebalancer;Controller;DynamicPlacement"
)

# Test: MoEConfig — configuration validation, isMoELayer()
add_executable(v2_test_moe_config unit/moe/Test__MoEConfig.cpp)
target_link_libraries(v2_test_moe_config
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEConfig
    COMMAND $<TARGET_FILE:v2_test_moe_config>
    LABELS "V2;Unit;MoE;Config"
)

# Test: MoEPipeline — full pipeline dry-run with mock model + rebalancing loop
add_executable(v2_test_moe_pipeline unit/moe/Test__MoEPipeline.cpp)
target_link_libraries(v2_test_moe_pipeline
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEPipeline
    COMMAND $<TARGET_FILE:v2_test_moe_pipeline>
    LABELS "V2;Unit;MoE;Pipeline;MockModel;Rebalancing"
)

# Test: MoEExpertWeightService — weight lifecycle service extracted from MoEFFNStage
add_executable(v2_test_moe_expert_weight_service unit/moe/Test__MoEExpertWeightService.cpp)
target_include_directories(v2_test_moe_expert_weight_service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_expert_weight_service
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEExpertWeightService
    COMMAND $<TARGET_FILE:v2_test_moe_expert_weight_service>
    LABELS "V2;Unit;MoE;WeightService;WeightLifecycle"
)

# Test: MoEPhaseCStoreResolution — prepared expert slab reuse/no-repack guardrails
add_executable(v2_test_moe_phase_c_store_resolution unit/moe/Test__MoEPhaseC_StoreResolution.cpp)
target_include_directories(v2_test_moe_phase_c_store_resolution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_phase_c_store_resolution
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MoEPhaseCStoreResolution
    COMMAND $<TARGET_FILE:v2_test_moe_phase_c_store_resolution>
    LABELS "V2;Unit;MoE;WeightService;PreparedWeightStore;PhaseC"
)

# Test: MoERoutingToCompute — end-to-end two-stage pipeline (MoERoutingStage → MoEExpertComputeStage)
add_executable(v2_test_moe_routing_to_compute unit/moe/Test__MoERoutingToCompute_Integration.cpp)
target_include_directories(v2_test_moe_routing_to_compute PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_routing_to_compute llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoERoutingToCompute
    COMMAND $<TARGET_FILE:v2_test_moe_routing_to_compute>
    LABELS "V2;Unit;MoE;Router;Stages;Pipeline;Integration"
    MPI_PROCS 1
)

# Test: HeterogeneousLayers — Phase B: named_templates, per-layer dispatch,
# AttentionOutputGateStage, GatedRMSNormStage, HybridCacheManager, GraphConfig
add_executable(v2_test_heterogeneous_layers unit/execution/Test__HeterogeneousLayers.cpp)
target_link_libraries(v2_test_heterogeneous_layers
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HeterogeneousLayers
    COMMAND $<TARGET_FILE:v2_test_heterogeneous_layers>
    LABELS "V2;Unit;HeterogeneousLayers;GraphSchema;GDN;PhaseB"
    MPI_PROCS 1
)

# Regression: AVX512 exp() polynomial bug in GatedRMSNormStage and AttentionOutputGateStage
# The polynomial used 2^f Taylor coefficients but was fed f=x-n*ln2 instead of f=x*log2e-n,
# causing 3-11% SiLU/sigmoid error that compounded across 24 GDN layers.
add_executable(v2_test_exp_polynomial_regression unit/execution/Test__ExpPolynomialRegression.cpp)
target_link_libraries(v2_test_exp_polynomial_regression
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExpPolynomialRegression
    COMMAND $<TARGET_FILE:v2_test_exp_polynomial_regression>
    LABELS "V2;Unit;Regression;GDN;SiLU;Sigmoid;AVX512"
    MPI_PROCS 1
)

# Regression: GDNRecurrenceStage QKV deinterleave repeat_interleave for n_k_heads < n_v_heads
add_executable(v2_test_gdn_repeat_interleave_regression unit/execution/Test__GDNRepeatInterleaveRegression.cpp)
target_link_libraries(v2_test_gdn_repeat_interleave_regression
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GDNRepeatInterleaveRegression
    COMMAND $<TARGET_FILE:v2_test_gdn_repeat_interleave_regression>
    LABELS "V2;Unit;Regression;GDN;RepeatInterleave"
    MPI_PROCS 1
)

# Regression: GDNRecurrenceStage + ShortConv1dStage updateDynamicParams override
add_executable(v2_test_gdn_dynamic_params_regression unit/execution/Test__GDNDynamicParamsRegression.cpp)
target_link_libraries(v2_test_gdn_dynamic_params_regression
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GDNDynamicParamsRegression
    COMMAND $<TARGET_FILE:v2_test_gdn_dynamic_params_regression>
    LABELS "V2;Unit;Regression;GDN;DynamicParams"
    MPI_PROCS 1
)

# Regression: Qwen35Graph global_v_head_offset under TP sharding.
# Locks in the fix that sets the offset in ALL TP modes where V is sharded
# (selection / identity / expansion), not only selection/identity. The
# expansion branch was missed, causing rank>0 to pick the wrong K-heads
# on 27B Qwen-3.5 under 2xMPI CPU.
add_executable(v2_test_gdn_tp_offset_regression unit/execution/Test__GDNTPOffsetRegression.cpp)
target_link_libraries(v2_test_gdn_tp_offset_regression
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GDNTPOffsetRegression
    COMMAND $<TARGET_FILE:v2_test_gdn_tp_offset_regression>
    LABELS "V2;Unit;Regression;GDN;TensorParallel;Qwen35"
    MPI_PROCS 1
)

# Test: GDN Kernels — Phase C: GDNProjectionStage, ShortConv1dStage,
# GDNRecurrenceStage (recurrent step + chunk-parallel prefill)
add_executable(v2_test_gdn_kernels unit/execution/Test__GDNKernels.cpp)
target_link_libraries(v2_test_gdn_kernels
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Unit_GDNKernels
    COMMAND $<TARGET_FILE:v2_test_gdn_kernels>
    LABELS "V2;Unit;GDNKernels;GDN;PhaseC"
    MPI_PROCS 1
)

# Test: GDN Mathematical Correctness — Deep formula-level verification of
# recurrence, gate computation, softplus edge cases, GatedRMSNorm per-head
# norm_dim, Conv1d bias, AVX512 scalar tail handling, multi-step accumulation
add_executable(v2_test_gdn_mathematical_correctness unit/execution/Test__GDNMathematicalCorrectness.cpp)
target_link_libraries(v2_test_gdn_mathematical_correctness
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)
add_v2_test(V2_Unit_GDNMathematicalCorrectness
    COMMAND $<TARGET_FILE:v2_test_gdn_mathematical_correctness>
    LABELS "V2;Unit;GDNKernels;GDN;MathCorrectness"
    MPI_PROCS 1
)

# Test: Qwen35Schema — Phase D: Qwen 3.5 Dense schema, config builder,
# graph builder, and architecture registration
add_executable(v2_test_qwen35_schema unit/models/qwen35/Test__Qwen35Schema.cpp)
target_link_libraries(v2_test_qwen35_schema
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Qwen35Schema
    COMMAND $<TARGET_FILE:v2_test_qwen35_schema>
    LABELS "V2;Unit;Qwen35;GraphSchema;GDN;PhaseD"
    MPI_PROCS 1
)

# Test: Qwen35MoEGraph — MoE graph construction regressions
add_executable(v2_test_qwen35moe_graph unit/models/qwen35moe/Test__Qwen35MoEGraph.cpp)
target_link_libraries(v2_test_qwen35moe_graph
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_test_qwen35moe_graph PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_Qwen35MoEGraph
    COMMAND $<TARGET_FILE:v2_test_qwen35moe_graph>
    LABELS "V2;Unit;Qwen35MoE;Graph;MixtureOfExperts;TensorParallel"
    MPI_PROCS 1
)

# Test: Qwen35MoEGraphNativeProductionLowering — graph-native overlay lowering regressions
add_executable(v2_test_qwen35moe_graph_native_production_lowering unit/models/qwen35moe/Test__Qwen35MoEGraphNativeProductionLowering.cpp)
target_link_libraries(v2_test_qwen35moe_graph_native_production_lowering
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(v2_test_qwen35moe_graph_native_production_lowering PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_v2_test(V2_Unit_Qwen35MoEGraphNativeProductionLowering
    COMMAND $<TARGET_FILE:v2_test_qwen35moe_graph_native_production_lowering>
    LABELS "V2;Unit;Qwen35MoE;Graph;MixtureOfExperts;GraphNativeOverlay"
    MPI_PROCS 1
)

# Test: Qwen35BufferSizes — Lock-in tests for Qwen3.5 schema buffer sizes
add_executable(v2_test_qwen35_buffer_sizes unit/models/qwen35/Test__Qwen35BufferSizes.cpp)
target_link_libraries(v2_test_qwen35_buffer_sizes
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Qwen35BufferSizes
    COMMAND $<TARGET_FILE:v2_test_qwen35_buffer_sizes>
    LABELS "V2;Unit;Qwen35;GraphSchema;BufferSizes;GDN"
    MPI_PROCS 1
)

# Test: HybridAllreducePrecision — Layer-type-aware allreduce precision for Qwen3.5
# FA layers always use FP32, GDN layers follow count-based fp32/fp16 policy.
add_executable(v2_test_hybrid_allreduce_precision unit/models/qwen35/Test__HybridAllreducePrecision.cpp)
target_link_libraries(v2_test_hybrid_allreduce_precision
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_HybridAllreducePrecision
    COMMAND $<TARGET_FILE:v2_test_hybrid_allreduce_precision>
    LABELS "V2;Unit;Qwen35;Allreduce;Precision;GDN"
    MPI_PROCS 1
)

# Test: ActivationRotation — Block-diagonal rotation for kurtosis reduction
# Tests orthogonality (norm preservation), identity (rotate+inverse), kurtosis
# reduction, GEMM invariance (X@W^T == X@R@(W@R)^T), ModelWeightRotation
# weight replacement and metadata wiring.
add_executable(v2_test_activation_rotation unit/kernels/rotation/Test__ActivationRotation.cpp)
target_link_libraries(v2_test_activation_rotation
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ActivationRotation
    COMMAND $<TARGET_FILE:v2_test_activation_rotation>
    LABELS "V2;Unit;Kernels;Rotation;ActivationRotation"
    MPI_PROCS 1
)

# Test: DynamicBufferExtensions — Dynamic extension buffer wiring chain
# Tests BufferArena::forEachRegistered(), ActivationBuffers::extensions/get(),
# InferenceState::extension_buffers, toModelBuffers() propagation, and
# end-to-end auto-discovery with mock model fictitious BufferIds.
add_executable(v2_test_dynamic_buffer_extensions unit/execution/local_execution/orchestrators/Test__DynamicBufferExtensions.cpp)
target_link_libraries(v2_test_dynamic_buffer_extensions llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DynamicBufferExtensions
    COMMAND $<TARGET_FILE:v2_test_dynamic_buffer_extensions>
    LABELS "V2;Unit;BufferArena;DynamicExtensions;BufferWiring;InferenceState"
    MPI_PROCS 1
)

# =============================================================================
# Weight Coherence Contract Tests
# =============================================================================

# Test: StageWeightContracts — Verifies all weight-bearing stages declare weights
# in bufferContract().addWeight(). Locks in contract-only coherence paradigm.
add_executable(v2_test_stage_weight_contracts unit/execution/compute_stages/Test__StageWeightContracts.cpp)
target_include_directories(v2_test_stage_weight_contracts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ include path
)
target_link_libraries(v2_test_stage_weight_contracts llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageWeightContracts
    COMMAND $<TARGET_FILE:v2_test_stage_weight_contracts>
    LABELS "V2;Unit;Execution;ComputeStage;BufferContracts;WeightCoherence"
    MPI_PROCS 1
)

# Test: ExecutorWeightCoherence — Tests DeviceGraphExecutor weight coherence system:
# session-level flag, per-node flags, contract-only processing, policy gating.
add_executable(v2_test_executor_weight_coherence unit/execution/local_execution/graph/Test__ExecutorWeightCoherence.cpp)
target_include_directories(v2_test_executor_weight_coherence PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_test_executor_weight_coherence
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ExecutorWeightCoherence
    COMMAND $<TARGET_FILE:v2_test_executor_weight_coherence>
    LABELS "V2;Unit;DeviceGraphExecutor;Execution;WeightCoherence;BufferContracts"
    MPI_PROCS 1
)

# Test: PreparedEmbeddingWeightsLifecycle — PreparedWeightStore-owned embedding
# lifecycle: creation, lookup, device separation, clear, move semantics.
add_executable(v2_test_prepared_embedding_lifecycle unit/kernels/Test__PreparedEmbeddingWeightsLifecycle.cpp)
target_link_libraries(v2_test_prepared_embedding_lifecycle llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_prepared_embedding_lifecycle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ include path
)
add_v2_test(V2_Unit_PreparedEmbeddingWeightsLifecycle
    COMMAND $<TARGET_FILE:v2_test_prepared_embedding_lifecycle>
    LABELS "V2;Unit;Kernels;Embedding;PreparedWeights;Lifecycle"
    MPI_PROCS 1
)

# Test: WeightManagerHostRelease — WeightManager host memory release decision logic:
# releaseAllHostWeightData and releaseHostResidentWeightData with per-tensor
# prepared-device-state metadata.
add_executable(v2_test_weight_manager_host_release unit/loaders/Test__WeightManagerHostRelease.cpp)
target_link_libraries(v2_test_weight_manager_host_release llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_weight_manager_host_release PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
add_v2_test(V2_Unit_WeightManagerHostRelease
    COMMAND $<TARGET_FILE:v2_test_weight_manager_host_release>
    LABELS "V2;Unit;WeightManager;HostRelease;Memory;Lifecycle"
    MPI_PROCS 1
)

# Test: WeightManagerCpuHostRelease — Verifies that quantized TensorSlice raw data
# is released after binding-driven CPU VNNI packing, while FP32, embeddings, norms,
# and biases are retained. Locks in the memory optimization fix.
add_executable(v2_test_weight_manager_cpu_host_release unit/loaders/Test__WeightManagerCpuHostRelease.cpp)
target_link_libraries(v2_test_weight_manager_cpu_host_release llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_weight_manager_cpu_host_release PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
add_v2_test(V2_Unit_WeightManagerCpuHostRelease
    COMMAND $<TARGET_FILE:v2_test_weight_manager_cpu_host_release>
    LABELS "V2;Unit;WeightManager;HostRelease;Memory;Lifecycle;CPU"
    MPI_PROCS 1
)

# Test: EmbeddingKernelPreparedPath — Embedding kernel workspace requirements and
# prepared vs fallback path selection. GPU tests skip if no CUDA/ROCm available.
add_executable(v2_test_embedding_kernel_prepared_path unit/kernels/Test__EmbeddingKernelPreparedPath.cpp)
target_link_libraries(v2_test_embedding_kernel_prepared_path llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_embedding_kernel_prepared_path PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ include path
)
add_v2_test(V2_Unit_EmbeddingKernelPreparedPath
    COMMAND $<TARGET_FILE:v2_test_embedding_kernel_prepared_path>
    LABELS "V2;Unit;Kernels;Embedding;Workspace;PreparedWeights"
    MPI_PROCS 1
)

# Test: VocabParallelEmbeddingSharding — Non-GPU logic for Megatron-style
# vocab-parallel embedding sharding: WeightShardingConfig for token_embd.weight,
# TensorParallelConfig vocab splits, WeightSlicer boundaries, EmbedQ8Repack
# vocab-range overload, PreparedEmbeddingWeights metadata, schema TPMode.
add_executable(v2_test_vocab_parallel_embedding_sharding unit/loaders/Test__VocabParallelEmbeddingSharding.cpp)
target_link_libraries(v2_test_vocab_parallel_embedding_sharding llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_vocab_parallel_embedding_sharding PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/utils/ include path
)
add_v2_test(V2_Unit_VocabParallelEmbeddingSharding
    COMMAND $<TARGET_FILE:v2_test_vocab_parallel_embedding_sharding>
    LABELS "V2;Unit;Loaders;WeightSharding;Embedding;VocabParallel;TensorParallelism"
    MPI_PROCS 1
)

# Test: WeightManager_PPSafety — PP pipeline safety fixes:
# isWeightInLayerRange tied embeddings check, tryReleaseReclaimHostRawData
# CPU-only tensor safety, default WeightPrecision is NATIVE.
add_executable(v2_test_weight_manager_pp_safety unit/loaders/Test__WeightManager_PPSafety.cpp)
target_link_libraries(v2_test_weight_manager_pp_safety llaminar2_core GTest::gtest GTest::gtest_main)
target_include_directories(v2_test_weight_manager_pp_safety PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
add_v2_test(V2_Unit_WeightManagerPPSafety
    COMMAND $<TARGET_FILE:v2_test_weight_manager_pp_safety>
    LABELS "V2;Unit;WeightManager;PipelineParallel;Safety;LayerRange"
    MPI_PROCS 1
)

# =============================================================================
# Memory Planning Tests
# =============================================================================

# Test: ModelMemoryProfile — GGUF metadata extraction for memory planning
add_executable(v2_test_model_memory_profile unit/planning/Test__ModelMemoryProfile.cpp)
target_link_libraries(v2_test_model_memory_profile llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ModelMemoryProfile
    COMMAND $<TARGET_FILE:v2_test_model_memory_profile>
    LABELS "V2;Unit;Planning;ModelMemoryProfile;GGUF"
    MPI_PROCS 1
)

# Test: WeightMemoryEstimator — weight sizing with TP sharding and PP slicing
add_executable(v2_test_weight_memory_estimator unit/planning/Test__WeightMemoryEstimator.cpp)
target_link_libraries(v2_test_weight_memory_estimator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_WeightMemoryEstimator
    COMMAND $<TARGET_FILE:v2_test_weight_memory_estimator>
    LABELS "V2;Unit;Planning;WeightMemoryEstimator;TensorParallelism"
    MPI_PROCS 1
)

# Test: KVCacheMemoryEstimator — KV cache sizing for fp16/fp32/q8_1/tq
add_executable(v2_test_kvcache_memory_estimator unit/planning/Test__KVCacheMemoryEstimator.cpp)
target_link_libraries(v2_test_kvcache_memory_estimator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_KVCacheMemoryEstimator
    COMMAND $<TARGET_FILE:v2_test_kvcache_memory_estimator>
    LABELS "V2;Unit;Planning;KVCacheMemoryEstimator;KVCache"
    MPI_PROCS 1
)

# Test: ActivationMemoryEstimator — activation buffer peak estimation
add_executable(v2_test_activation_memory_estimator unit/planning/Test__ActivationMemoryEstimator.cpp)
target_link_libraries(v2_test_activation_memory_estimator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ActivationMemoryEstimator
    COMMAND $<TARGET_FILE:v2_test_activation_memory_estimator>
    LABELS "V2;Unit;Planning;ActivationMemoryEstimator;Activations"
    MPI_PROCS 1
)

# Test: WorkspaceMemoryEstimator — GPU kernel workspace budget estimation
add_executable(v2_test_workspace_memory_estimator unit/planning/Test__WorkspaceMemoryEstimator.cpp)
target_link_libraries(v2_test_workspace_memory_estimator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_WorkspaceMemoryEstimator
    COMMAND $<TARGET_FILE:v2_test_workspace_memory_estimator>
    LABELS "V2;Unit;Planning;WorkspaceMemoryEstimator;Workspace"
    MPI_PROCS 1
)

# Test: MemoryPlanner — end-to-end planning with single/multi-device configs
add_executable(v2_test_memory_planner unit/planning/Test__MemoryPlanner.cpp)
target_link_libraries(v2_test_memory_planner llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MemoryPlanner
    COMMAND $<TARGET_FILE:v2_test_memory_planner>
    LABELS "V2;Unit;Planning;MemoryPlanner;MemoryBudget"
    MPI_PROCS 1
)

# Test: ClusterInventoryGatherer — single-rank inventory gathering
add_executable(v2_test_cluster_inventory_gatherer unit/planning/Test__ClusterInventoryGatherer.cpp)
target_link_libraries(v2_test_cluster_inventory_gatherer llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ClusterInventoryGatherer
    COMMAND $<TARGET_FILE:v2_test_cluster_inventory_gatherer>
    LABELS "V2;Unit;Planning;ClusterInventory"
    MPI_PROCS 1
)

# Test: KVCacheConfig::estimateBytes() — bridge between KernelFactory config and planning
add_executable(v2_test_kvcache_config_estimate unit/planning/Test__KVCacheConfigEstimate.cpp)
target_link_libraries(v2_test_kvcache_config_estimate llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_KVCacheConfigEstimate
    COMMAND $<TARGET_FILE:v2_test_kvcache_config_estimate>
    LABELS "V2;Unit;Planning;KVCache;KernelFactory"
    MPI_PROCS 1
)

# Test: DeviceMemoryPlan — fits/deficit/remaining/summary struct methods
add_executable(v2_test_device_memory_plan unit/planning/Test__DeviceMemoryPlan.cpp)
target_link_libraries(v2_test_device_memory_plan llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_DeviceMemoryPlan
    COMMAND $<TARGET_FILE:v2_test_device_memory_plan>
    LABELS "V2;Unit;Planning;MemoryPlan"
    MPI_PROCS 1
)

# =============================================================================
# MoE Stage Tests
# =============================================================================

# Test: MoEExpertComputeStage — Router + Expert SwiGLU + Combine (Q4_K/Q5_K 3D weights)
# Tests: MoEExpertComputeStage (routing, expert FFN, combine), SharedExpertFFNStage (dense SwiGLU),
# SharedExpertGateStage (sigmoid gating), isNonGemmWeight regression for 3D expert tensors
add_executable(v2_test_moe_expert_compute_stage unit/stages/Test__MoEExpertComputeStage.cpp)
target_include_directories(v2_test_moe_expert_compute_stage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_expert_compute_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoEExpertComputeStage
    COMMAND $<TARGET_FILE:v2_test_moe_expert_compute_stage>
    LABELS "V2;Unit;MoE;Stages;Kernels;Q4_K;Q5_K"
    MPI_PROCS 1
)

# Test: MoERoutingStage — extracted router stage (softmax top-k → expert indices + weights)
add_executable(v2_test_moe_routing_stage unit/stages/Test__MoERoutingStage.cpp)
target_include_directories(v2_test_moe_routing_stage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_routing_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoERoutingStage
    COMMAND $<TARGET_FILE:v2_test_moe_routing_stage>
    LABELS "V2;Unit;MoE;Router;Stages"
    MPI_PROCS 1
)

add_executable(v2_test_moe_local_expert_stage_prepared_weights
    unit/execution/compute_stages/Test__MoELocalExpertStage_PreparedWeights.cpp)
target_include_directories(v2_test_moe_local_expert_stage_prepared_weights PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_local_expert_stage_prepared_weights llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoELocalExpertStage_PreparedWeights
    COMMAND $<TARGET_FILE:v2_test_moe_local_expert_stage_prepared_weights>
    LABELS "V2;Unit;MoE;Stages;PreparedWeights;RuntimeTable;GraphCapture"
    MPI_PROCS 1
)

add_executable(v2_test_moe_local_expert_stage_params
    unit/execution/compute_stages/Test__MoELocalExpertStage_Params.cpp)
target_include_directories(v2_test_moe_local_expert_stage_params PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_moe_local_expert_stage_params llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_MoELocalExpertStage_Params
    COMMAND $<TARGET_FILE:v2_test_moe_local_expert_stage_params>
    LABELS "V2;Unit;MoE;Stages;Params;RuntimeTable"
    MPI_PROCS 1
)

add_executable(v2_test_prefill_graph_capturability unit/stages/Test__PrefillGraphCapturability.cpp)
target_include_directories(v2_test_prefill_graph_capturability PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_prefill_graph_capturability llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefillGraphCapturability
    COMMAND $<TARGET_FILE:v2_test_prefill_graph_capturability>
    LABELS "V2;Unit;MoE;GDN;Stages;GraphCapture"
    MPI_PROCS 1
)

add_executable(v2_test_prefill_graph_capture_guards unit/stages/Test__PrefillGraphCaptureGuards.cpp)
target_include_directories(v2_test_prefill_graph_capture_guards PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_prefill_graph_capture_guards llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefillGraphCaptureGuards
    COMMAND $<TARGET_FILE:v2_test_prefill_graph_capture_guards>
    LABELS "V2;Unit;MoE;GDN;Stages;GraphCapture"
    MPI_PROCS 1
)

add_executable(v2_test_gpu_workspace_allocation_policy unit/stages/Test__GpuWorkspaceAllocationPolicy.cpp)
target_include_directories(v2_test_gpu_workspace_allocation_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(v2_test_gpu_workspace_allocation_policy
    PRIVATE LLAMINAR_REPO_ROOT="${CMAKE_SOURCE_DIR}/../..")
target_link_libraries(v2_test_gpu_workspace_allocation_policy llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GpuWorkspaceAllocationPolicy
    COMMAND $<TARGET_FILE:v2_test_gpu_workspace_allocation_policy>
    LABELS "V2;Unit;CUDA;ROCm;Stages;GraphCapture;Workspace"
    MPI_PROCS 1
)

add_executable(v2_test_prefill_graph_capture_dynamic_params unit/stages/Test__PrefillGraphCaptureDynamicParams.cpp)
target_include_directories(v2_test_prefill_graph_capture_dynamic_params PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_prefill_graph_capture_dynamic_params llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefillGraphCaptureDynamicParams
    COMMAND $<TARGET_FILE:v2_test_prefill_graph_capture_dynamic_params>
    LABELS "V2;Unit;Embedding;KVCache;Stages;GraphCapture"
    MPI_PROCS 1
)

add_executable(v2_test_hidden_state_row_select_stage unit/stages/Test__HiddenStateRowSelectStage.cpp)
target_include_directories(v2_test_hidden_state_row_select_stage PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_hidden_state_row_select_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_HiddenStateRowSelectStage
    COMMAND $<TARGET_FILE:v2_test_hidden_state_row_select_stage>
    LABELS "V2;Unit;Stages;GraphCapture;RowSelect;LMHead;Phase6"
    MPI_PROCS 1
)

add_executable(v2_test_prefill_graph_cache unit/stages/Test__PrefillGraphCache.cpp)
target_include_directories(v2_test_prefill_graph_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_prefill_graph_cache llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefillGraphCache
    COMMAND $<TARGET_FILE:v2_test_prefill_graph_cache>
    LABELS "V2;Unit;GraphCapture;Stages"
    MPI_PROCS 1
)

add_executable(v2_test_prefill_graph_cache_integration unit/stages/Test__PrefillGraphCacheIntegration.cpp)
target_include_directories(v2_test_prefill_graph_cache_integration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(v2_test_prefill_graph_cache_integration llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_PrefillGraphCacheIntegration
    COMMAND $<TARGET_FILE:v2_test_prefill_graph_cache_integration>
    LABELS "V2;Unit;GraphCapture;Stages"
    MPI_PROCS 1
)

# =============================================================================
# Collective Backend Tests
# =============================================================================

# Test: ShmemSpinBackend — Shared-memory spin-wait allreduce for 2-rank intra-node CPU TP
# Single-process: arena layout, fast-path detection, identity checks
# Two-process: allreduce correctness, repeated allreduce, boundary counts, fallback delegation, stress
add_executable(v2_test_shmem_spin_backend unit/collective/Test__ShmemSpinBackend.cpp)
target_include_directories(v2_test_shmem_spin_backend PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_shmem_spin_backend
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
)
add_v2_test(V2_Unit_ShmemSpinBackend
    COMMAND $<TARGET_FILE:v2_test_shmem_spin_backend>
    LABELS "V2;Unit;Collective;ShmemSpin;Allreduce"
    MPI_PROCS 2
    TIMEOUT 30
    RESOURCE_LOCK "ShmemSpin"
)
add_v2_test(V2_Unit_ShmemSpinBackend_NP4
    COMMAND $<TARGET_FILE:v2_test_shmem_spin_backend>
    LABELS "V2;Unit;Collective;ShmemSpin;Allreduce"
    MPI_PROCS 4
    TIMEOUT 30
    RESOURCE_LOCK "ShmemSpin"
)
add_v2_test(V2_Unit_ShmemSpinBackend_NP8
    COMMAND $<TARGET_FILE:v2_test_shmem_spin_backend>
    LABELS "V2;Unit;Collective;ShmemSpin;Allreduce"
    MPI_PROCS 8
    TIMEOUT 30
    RESOURCE_LOCK "ShmemSpin"
)
if(TEST V2_Unit_ShmemSpinBackend)
    set_tests_properties(
        V2_Unit_ShmemSpinBackend
        V2_Unit_ShmemSpinBackend_NP4
        V2_Unit_ShmemSpinBackend_NP8
        PROPERTIES
            RUN_SERIAL TRUE
    )
endif()

# =============================================================================
# GPU Weight Loading Pipeline - Phase 1 Foundation
# =============================================================================

add_executable(v2_test_weight_vram_pool unit/Test__WeightVRAMPool.cpp)
target_include_directories(v2_test_weight_vram_pool PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_weight_vram_pool
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_WeightVRAMPool
    COMMAND $<TARGET_FILE:v2_test_weight_vram_pool>
    LABELS "V2;Unit;WeightLoading;GpuPipeline"
    MPI_PROCS 1
)

add_executable(v2_test_pinned_ring_buffer unit/Test__PinnedRingBuffer.cpp)
target_include_directories(v2_test_pinned_ring_buffer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_pinned_ring_buffer
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PinnedRingBuffer
    COMMAND $<TARGET_FILE:v2_test_pinned_ring_buffer>
    LABELS "V2;Unit;WeightLoading;GpuPipeline"
    MPI_PROCS 1
)

add_executable(v2_test_load_orchestrator unit/Test__LoadOrchestrator.cpp)
target_include_directories(v2_test_load_orchestrator PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_load_orchestrator
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_LoadOrchestrator
    COMMAND $<TARGET_FILE:v2_test_load_orchestrator>
    LABELS "V2;Unit;WeightLoading;GpuPipeline"
    MPI_PROCS 1
)

# GPU Weight Loading Pipeline - Phase 2 GPU Repack Kernels
add_executable(v2_test_vnni_repack_kernels unit/Test__VnniRepackKernels.cpp)
target_include_directories(v2_test_vnni_repack_kernels PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_vnni_repack_kernels
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
if(HAVE_ROCM AND NOT LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_test_vnni_repack_kernels rocm_backend)
endif()
if(HAVE_ROCM)
    target_compile_definitions(v2_test_vnni_repack_kernels PRIVATE __HIP_PLATFORM_AMD__=1)
    target_link_libraries(v2_test_vnni_repack_kernels hip::host)
endif()
add_v2_test(V2_Unit_VnniRepackKernels
    COMMAND $<TARGET_FILE:v2_test_vnni_repack_kernels>
    LABELS "V2;Unit;WeightLoading;GpuPipeline;ROCm;HIP"
    MPI_PROCS 1
)

# GPU Weight Loading Pipeline - VNNI Unpack (reverse repack) round-trip tests
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_integration_vnni_unpack_kernels
        integration/Test__VnniUnpackKernels.cpp
    )
    if(LLAMINAR_SHARED_CORE)
        target_link_libraries(v2_integration_vnni_unpack_kernels PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    elseif(UNIX AND NOT APPLE)
        set(_gpu_backends_whole_unpack "")
        if(HAVE_CUDA)
            list(APPEND _gpu_backends_whole_unpack -Wl,--whole-archive cuda_backend -Wl,--no-whole-archive)
        endif()
        if(HAVE_ROCM)
            list(APPEND _gpu_backends_whole_unpack -Wl,--whole-archive rocm_backend -Wl,--no-whole-archive)
        endif()
        target_link_libraries(v2_integration_vnni_unpack_kernels PRIVATE
            -Wl,--start-group
            llaminar2_core
            ${_gpu_backends_whole_unpack}
            -Wl,--end-group
            GTest::gtest
            GTest::gtest_main
        )
    else()
        target_link_libraries(v2_integration_vnni_unpack_kernels PRIVATE
            llaminar2_core
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    add_v2_test(V2_Integration_VnniUnpackKernels
        COMMAND $<TARGET_FILE:v2_integration_vnni_unpack_kernels>
        LABELS "V2;Integration;WeightLoading;GpuPipeline;GPU;CUDA;ROCm;Repack"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added VNNI unpack (reverse repack) integration tests")
endif()

# GPU Weight Loading Pipeline - Phase 3 Pipeline Integration
add_executable(v2_test_device_load_pipeline unit/Test__DeviceLoadPipeline.cpp)
target_include_directories(v2_test_device_load_pipeline PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_device_load_pipeline
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
if(HAVE_ROCM AND NOT LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_test_device_load_pipeline rocm_backend)
endif()
if(HAVE_ROCM)
    target_compile_definitions(v2_test_device_load_pipeline PRIVATE __HIP_PLATFORM_AMD__=1)
    target_link_libraries(v2_test_device_load_pipeline hip::host)
endif()
add_v2_test(V2_Unit_DeviceLoadPipeline
    COMMAND $<TARGET_FILE:v2_test_device_load_pipeline>
    LABELS "V2;Unit;WeightLoading;GpuPipeline;ROCm;HIP"
    MPI_PROCS 1
)

# GPU Weight Loading Pipeline - FP Weight Pipeline Unit Tests
add_executable(v2_test_fp_weight_pipeline unit/Test__FPWeightPipeline.cpp)
target_include_directories(v2_test_fp_weight_pipeline PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(v2_test_fp_weight_pipeline
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
if(HAVE_ROCM AND NOT LLAMINAR_SHARED_CORE)
    target_link_libraries(v2_test_fp_weight_pipeline rocm_backend)
endif()
if(HAVE_ROCM)
    target_compile_definitions(v2_test_fp_weight_pipeline PRIVATE __HIP_PLATFORM_AMD__=1)
    target_link_libraries(v2_test_fp_weight_pipeline hip::host)
endif()
add_v2_test(V2_Unit_FPWeightPipeline
    COMMAND $<TARGET_FILE:v2_test_fp_weight_pipeline>
    LABELS "V2;Unit;WeightLoading;GpuPipeline;FP"
    MPI_PROCS 1
)
