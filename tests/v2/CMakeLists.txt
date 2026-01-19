# Llaminar V2 Test Suite
# Test categories: unit, integration, e2e, pytorch_parity

# Note: enable_testing() is called in parent CMakeLists.txt (src/v2/CMakeLists.txt)

# =============================================================================
# HybridQ16 Test Suite Toggle
# =============================================================================
# HybridQ16 project is on hold - these tests are disabled by default until
# we resume Q16_1 typed residual development. Enable with -DENABLE_HYBRIDQ16_TESTS=ON
option(ENABLE_HYBRIDQ16_TESTS "Enable HybridQ16/Q16_1 pipeline tests (project on hold)" OFF)
if(NOT ENABLE_HYBRIDQ16_TESTS)
    message(STATUS "V2: HybridQ16 tests DISABLED (project on hold, use -DENABLE_HYBRIDQ16_TESTS=ON to enable)")
endif()

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
        "HWLOC_COMPONENTS=-gl,-opencl"
        "OMP_NUM_THREADS=${CORES_PER_SOCKET}"
        "OMP_PLACES=sockets"
        "OMP_PROC_BIND=close"
        "OMP_NESTED=false"
        "OMP_DYNAMIC=false"
        "KMP_AFFINITY=granularity=fine,compact,1,0"
        "KMP_BLOCKTIME=0"
        "OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET}"
        "GOTO_NUM_THREADS=${CORES_PER_SOCKET}"
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
    
    # Add RESOURCE_LOCK if specified (prevents parallel execution of tests needing same resource)
    if(ARG_RESOURCE_LOCK)
        set_tests_properties(${TEST_NAME} PROPERTIES
            RESOURCE_LOCK "${ARG_RESOURCE_LOCK}"
        )
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

# Test: CacheInfo - CPU cache detection and batch size optimization
add_executable(v2_test_cache_info unit/Test__CacheInfo.cpp)
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
add_executable(v2_test_stage_dump_config unit/Test__StageDumpConfig.cpp)
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

# Test: DeviceGroup - Device group construction and predicates for collective operations
add_executable(v2_unit_device_group unit/Test__DeviceGroup.cpp)
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

# Test: DeviceBenchmark - Lightweight device benchmarking infrastructure
add_executable(v2_unit_device_benchmark unit/Test__DeviceBenchmark.cpp)
target_link_libraries(v2_unit_device_benchmark 
    llaminar2_core 
    device_benchmarks
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_DeviceBenchmark 
    COMMAND $<TARGET_FILE:v2_unit_device_benchmark>
    LABELS "V2;Unit;DeviceManagement;Benchmark;Performance"
    MPI_PROCS 1
)

# Test: DirectP2P - Cross-vendor P2P via PCIe BAR mapping
# Requires both CUDA and ROCm - tests true zero-copy P2P between GPU vendors
if(HAVE_CUDA AND HAVE_ROCM)
    add_executable(v2_integration_direct_p2p integration/Test__DirectP2P.cpp)
    target_link_libraries(v2_integration_direct_p2p 
        llaminar2_core 
        device_benchmarks
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_DirectP2P 
        COMMAND $<TARGET_FILE:v2_integration_direct_p2p>
        LABELS "V2;Integration;DeviceManagement;P2P;PCIeBAR;CUDA;ROCm;CrossVendor"
        MPI_PROCS 1
    )

    # Test: CrossVendorTensorParallel - Real tensor parallel inference across NVIDIA+AMD
    # Tests FP32/Q8_1 tensor transfers, column/row parallel GEMM, AllReduce semantics
    add_executable(v2_integration_cross_vendor_tp integration/Test__CrossVendorTensorParallel.cpp)
    target_link_libraries(v2_integration_cross_vendor_tp 
        llaminar2_core 
        device_benchmarks
        GTest::gtest
        GTest::gtest_main
    )
    target_include_directories(v2_integration_cross_vendor_tp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    add_v2_test(V2_Integration_CrossVendorTensorParallel 
        COMMAND $<TARGET_FILE:v2_integration_cross_vendor_tp>
        LABELS "V2;Integration;TensorParallel;P2P;CUDA;ROCm;CrossVendor;GEMM;AllReduce"
        MPI_PROCS 1
        RESOURCE_LOCK "PCIeBAR_P2P"
    )

    # Test: PCIeBARBackend Integration - End-to-end collective operations via PCIe BAR P2P
    add_executable(v2_integration_pcie_bar_backend integration/Test__PCIeBARBackendIntegration.cpp)
    target_link_libraries(v2_integration_pcie_bar_backend 
        llaminar2_core 
        device_benchmarks
        GTest::gtest
        GTest::gtest_main
    )
    target_include_directories(v2_integration_pcie_bar_backend PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
    )
    add_v2_test(V2_Integration_PCIeBARBackend 
        COMMAND $<TARGET_FILE:v2_integration_pcie_bar_backend>
        LABELS "V2;Integration;Collective;PCIeBARBackend;P2P;CUDA;ROCm;CrossVendor;TensorParallel;AllReduce"
        MPI_PROCS 1
        RESOURCE_LOCK "PCIeBAR_P2P"
    )

    # Test: CUDA Vector Add Kernels - Accuracy tests for FP32/FP16/BF16/INT8/INT32 reduction kernels
    add_executable(v2_integration_cuda_vector_add_kernels integration/Test__CUDAVectorAddKernels.cpp)
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
endif()

# Test: BackendRouter - Backend selection and routing for collective operations
add_executable(v2_unit_backend_router unit/Test__BackendRouter.cpp)
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
add_executable(v2_unit_buffer_registration unit/Test__IBufferRegistration.cpp)
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
add_executable(v2_unit_host_backend unit/Test__HostBackend.cpp)
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
add_executable(v2_unit_cpu_backend unit/Test__CPUBackend.cpp)
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
add_executable(v2_unit_device_workspace_manager unit/Test__DeviceWorkspaceManager.cpp)
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

# Test: IWorkspaceConsumerStage - Mixin for stages that delegate workspace to kernels
add_executable(v2_unit_workspace_consumer_stage unit/Test__IWorkspaceConsumerStage.cpp)
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
add_executable(v2_unit_rocm_gemm_workspace unit/Test__ROCmQuantisedGemmKernel_Workspace.cpp)
target_link_libraries(v2_unit_rocm_gemm_workspace 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ROCmQuantisedGemmKernel_Workspace 
    COMMAND $<TARGET_FILE:v2_unit_rocm_gemm_workspace>
    LABELS "V2;Unit;Kernels;ROCm;GEMM;Workspace;IWorkspaceConsumer"
    MPI_PROCS 1
)

# Test: MPIBackend - MPI-based collective backend for inter-node communication
add_executable(v2_unit_mpi_backend unit/Test__MPIBackend.cpp)
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

# Test: PCIeBARBackend - PCIe BAR direct P2P collective backend (CUDA + ROCm)
if(HAVE_CUDA AND HAVE_ROCM)
    add_executable(v2_unit_pcie_bar_backend unit/Test__PCIeBARBackend.cpp)
    target_link_libraries(v2_unit_pcie_bar_backend 
        llaminar2_core 
        device_benchmarks
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_PCIeBARBackend 
        COMMAND $<TARGET_FILE:v2_unit_pcie_bar_backend>
        LABELS "V2;Unit;Collective;PCIeBARBackend;P2P;CUDA;ROCm;CrossVendor"
        MPI_PROCS 1
    )
endif()

# Test: CollectiveContext - Runtime context for collective operations
add_executable(v2_unit_collective_context unit/Test__CollectiveContext.cpp)
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

# Test: MockGraphBufferManager - Mock graph buffer manager for unit testing without tensor allocation
add_executable(v2_unit_mock_graph_buffer_manager unit/mocks/Test__MockGraphBufferManager.cpp)
target_include_directories(v2_unit_mock_graph_buffer_manager PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For tests/v2/mocks/ include path
)
target_link_libraries(v2_unit_mock_graph_buffer_manager 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MockGraphBufferManager 
    COMMAND $<TARGET_FILE:v2_unit_mock_graph_buffer_manager>
    LABELS "V2;Unit;Mock;GraphBufferManager;Testing"
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

# Test: JIT Register Allocation Framework
# Compile-time safe register allocation for JIT code generation
add_executable(v2_test_register_allocation unit/Test__RegisterAllocation.cpp)
target_link_libraries(v2_test_register_allocation 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RegisterAllocation 
    COMMAND $<TARGET_FILE:v2_test_register_allocation>
    LABELS "V2;Unit;JIT;RegisterAllocation;AVX512"
)

# Test: RegisterGuard - RAII-based register conflict detection
add_executable(v2_test_register_guard unit/kernels/cpu/jit/Test__RegisterGuard.cpp)
target_link_libraries(v2_test_register_guard 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RegisterGuard 
    COMMAND $<TARGET_FILE:v2_test_register_guard>
    LABELS "V2;Unit;JIT;RegisterGuard;AVX512"
)

# Test: RegisterEnforcement - Compile-time register type enforcement
add_executable(v2_test_register_enforcement unit/jit/Test__RegisterEnforcement.cpp)
target_link_libraries(v2_test_register_enforcement 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RegisterEnforcement 
    COMMAND $<TARGET_FILE:v2_test_register_enforcement>
    LABELS "V2;Unit;JIT;RegisterEnforcement;TypeSafety"
)

# Test: JitWoProjectionOptimized - BLAS-style Wo projection microkernels
add_executable(v2_test_jit_wo_projection_optimized unit/jit/Test__JitWoProjectionOptimized.cpp)
target_link_libraries(v2_test_jit_wo_projection_optimized 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitWoProjectionOptimized 
    COMMAND $<TARGET_FILE:v2_test_jit_wo_projection_optimized>
    LABELS "V2;Unit;JIT;GEMM;GEMV;Wo;Attention;AVX512"
)

# Test: JitQ16MicrokernelParity - JIT vs Reference Q16 microkernel parity tests
add_executable(v2_test_jit_q16_microkernel_parity unit/jit/Test__JitQ16MicrokernelParity.cpp)
target_link_libraries(v2_test_jit_q16_microkernel_parity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitQ16MicrokernelParity 
    COMMAND $<TARGET_FILE:v2_test_jit_q16_microkernel_parity>
    LABELS "V2;Unit;JIT;Q16;Attention;Parity;AVX512;VNNI"
)

# Test: SwiGLU formula correctness (regression test for gate/up swap bug)
# This test verifies that SwiGLU computes silu(gate) * up, NOT gate * silu(up)
add_executable(v2_test_swiglu_formula unit/Test__SwiGLUFormula.cpp)
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

# Test: Q16 Fixed-Scale SIMD Quantization (VNNI-safe clipping)
# Tests SIMD implementations (Scalar, AVX2, AVX512) of fixed-scale quantization
add_executable(v2_test_q16_fixed_scale_simd unit/tensors/Test__Q16FixedScaleSIMD.cpp)
target_link_libraries(v2_test_q16_fixed_scale_simd
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16FixedScaleSIMD
    COMMAND $<TARGET_FILE:v2_test_q16_fixed_scale_simd>
    LABELS "V2;Unit;Quantization;Q16_1;SIMD;AVX2;AVX512;VNNI;FixedScale;Safety"
    MPI_PROCS 1
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

# Performance: Q16 Dynamic-Scale RoPE (Scalar vs AVX2 vs AVX512)
add_executable(v2_perf_q16_dynamic_scale_rope performance/kernels/cpu/primitives/Perf__Q16_DynamicScaleRoPE.cpp)
target_link_libraries(v2_perf_q16_dynamic_scale_rope
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_Q16_DynamicScaleRoPE
    COMMAND $<TARGET_FILE:v2_perf_q16_dynamic_scale_rope>
    LABELS "V2;Performance;RoPE;Q16_1;SIMD;AVX2;AVX512;DynamicScale;IntegerAttention"
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
add_executable(v2_test_tensor_factory_device_idx unit/Test__TensorFactory_DeviceIndex.cpp)
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
add_executable(v2_test_tensor_factory_q16_block_size unit/Test__TensorFactory_Q16BlockSize.cpp)
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
add_executable(v2_test_itensor unit/Test__ITensor.cpp)
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
add_executable(v2_test_attention_mode unit/Test__AttentionMode.cpp)
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
add_executable(v2_test_q8_1_residual_layer21 unit/Test__Q8_1_ResidualAdd_Layer21.cpp)
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
add_executable(v2_test_int8_tensor unit/Test__INT8Tensor.cpp)
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
add_executable(v2_test_q8_1_pure_integer_rmsnorm unit/Test__Q8_1PureIntegerRMSNorm.cpp)
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
add_executable(v2_test_q16_1_rmsnorm unit/Test__Q16_1RMSNorm.cpp)
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

# Test: Q8_1 OnlineSoftmax JIT Kernel Unit Tests (fast, no model loading)
# Tests the Q8_1 x Q8_1 attention kernel with fused online softmax
add_executable(v2_test_q8_1_online_softmax_unit unit/kernels/cpu/gemm_v4/Test__Q8_1_OnlineSoftmax_Unit.cpp)
target_link_libraries(v2_test_q8_1_online_softmax_unit 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_OnlineSoftmax 
    COMMAND $<TARGET_FILE:v2_test_q8_1_online_softmax_unit>
    LABELS "V2;Unit;Kernels;GEMM;Q8_1;OnlineSoftmax;Attention;JIT;AVX512"
    MPI_PROCS 1  # JIT kernel unit tests are CPU-only, single rank
)

# Test: Q8_1 Fused Attention JIT Kernel Unit Tests
# Tests the fully fused Q8_1 attention kernel with dynamic head_dim support
add_executable(v2_test_xbyak_vpmovdb unit/Test__Xbyak_VPMOVDB.cpp)
target_link_libraries(v2_test_xbyak_vpmovdb llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_Xbyak_VPMOVDB
    COMMAND v2_test_xbyak_vpmovdb
    LABELS "V2;Unit;Xbyak;VPMOVDB"
)

add_executable(v2_test_q8_1_fused_attention_unit unit/kernels/cpu/gemm_v4/Test__QuantisedAttentionJit_Q8_1_Fused.cpp)
target_link_libraries(v2_test_q8_1_fused_attention_unit 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1_FusedAttention 
    COMMAND $<TARGET_FILE:v2_test_q8_1_fused_attention_unit>
    LABELS "V2;Unit;Kernels;GEMM;Q8_1;FusedAttention;Attention;JIT;AVX512;StackLayout"
    MPI_PROCS 1  # JIT kernel unit tests are CPU-only, single rank
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
add_executable(v2_test_q3k_bug unit/Test__Q3K_Bug.cpp)
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

# Test: QuantisedGemmKernel Mins Flag Verification
add_executable(v2_test_gemm_mins_flag unit/kernels/cpu/gemm_v4/Test__QuantisedGemmKernel_MinsFlag.cpp)
target_link_libraries(v2_test_gemm_mins_flag 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Gemm_MinsFlag 
    COMMAND $<TARGET_FILE:v2_test_gemm_mins_flag>
    LABELS "V2;Unit;Kernels;GEMM;Mins"
    MPI_PROCS 1
)

# Test: Q4_1 Integration (GEMM)
add_executable(v2_test_q4_1_integration integration/Test__Q4_1Tensor.cpp)
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
add_executable(v2_test_q2_k_integration integration/Test__Q2_KTensor.cpp)
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
add_executable(v2_test_q3_k_integration integration/Test__Q3_KTensor.cpp)
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
add_executable(v2_test_q4_k_integration integration/Test__Q4_KTensor.cpp)
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
add_executable(v2_test_q6_k_integration integration/Test__Q6_KTensor.cpp)
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
add_executable(v2_test_iq4_nl_integration integration/Test__IQ4_NLTensor.cpp)
target_link_libraries(v2_test_iq4_nl_integration 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_IQ4_NLTensor 
    COMMAND $<TARGET_FILE:v2_test_iq4_nl_integration>
    LABELS "V2;Integration;TensorOperations;Quantization;IQ4_NL;GEMM"
)

# Test: FusedGEMM kernel (generic N-projection fused GEMM)
add_executable(v2_test_fused_gemm unit/Test__FusedGEMM.cpp)
target_link_libraries(v2_test_fused_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedGEMM
    COMMAND $<TARGET_FILE:v2_test_fused_gemm>
    LABELS "V2;Unit;Kernels;FusedKernels;GEMM;Quantization;OperatorFusion;FFN;Attention;AVX512;AVX2;CPU"
    MPI_PROCS 1  # CPU-only fused kernel, single rank
)

# Test: FusedQKVGEMMStage compute stage (regression test for QKV bias bug)
add_executable(v2_test_fused_qkv_gemm_stage unit/Test__FusedQKVGEMMStage.cpp)
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
add_executable(v2_test_fused_gate_up_gemm_stage unit/Test__FusedGateUpGEMMStage.cpp)
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
add_executable(v2_test_device_index_cpu_kernels unit/Test__DeviceIndexCPUKernels.cpp)
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
add_executable(v2_test_kernel_factory unit/Test__KernelFactory.cpp)
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

# Test: KernelFactory sliced GEMM for tensor parallelism (Phase 2.1)
add_executable(v2_test_kernel_factory_sliced unit/Test__KernelFactorySliced.cpp)
target_link_libraries(v2_test_kernel_factory_sliced
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelFactorySliced
    COMMAND $<TARGET_FILE:v2_test_kernel_factory_sliced>
    LABELS "V2;Unit;Kernels;KernelFactory;TensorParallelism;SlicedGEMM;Phase2"
    MPI_PROCS 1  # Sliced kernel tests, single rank
)

# Test: KernelFactory automatic cache invalidation on tensor destruction
add_executable(v2_test_kernel_factory_cache_invalidation unit/Test__KernelFactoryCacheInvalidation.cpp)
target_link_libraries(v2_test_kernel_factory_cache_invalidation
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelFactoryCacheInvalidation
    COMMAND $<TARGET_FILE:v2_test_kernel_factory_cache_invalidation>
    LABELS "V2;Unit;Kernels;KernelFactory;CacheInvalidation;MemorySafety"
    MPI_PROCS 1  # Cache invalidation tests, single rank
)

# Test: KernelFactory TensorSlice unwrapping in CUDA dispatch path
# Regression test for: "unsupported tensor type 10 for device CUDA:0"
# Root cause: dynamic_cast<Q4_0Tensor*>(TensorSlice*) returns nullptr
# Solution: Unwrap TensorSlice via slice->inner() before type dispatch
add_executable(v2_test_kernel_factory_tensorslice_cuda unit/Test__KernelFactoryTensorSliceCUDA.cpp)
target_link_libraries(v2_test_kernel_factory_tensorslice_cuda
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_KernelFactoryTensorSliceCUDA
    COMMAND $<TARGET_FILE:v2_test_kernel_factory_tensorslice_cuda>
    LABELS "V2;Unit;Kernels;KernelFactory;TensorSlice;CUDA;TensorParallelism;Regression"
    MPI_PROCS 1  # TensorSlice tests, single rank
)

# Test: NUMATopology NUMA detection (local node, GPU affinity) - INTEGRATION (requires real GPUs)
add_executable(v2_integration_numa_topology integration/Test__NUMATopology.cpp)
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
add_executable(v2_test_kernel_factory_ops unit/Test__KernelFactoryOps.cpp)
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

# Test: RMSNorm precision variants (FP32/BF16/FP16/INT32 scalar/AVX2/AVX512 parity)
add_executable(v2_test_rmsnorm_precision unit/Test__RMSNormPrecision.cpp)
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
add_executable(v2_test_rmsnorm_int32 unit/Test__RMSNormINT32.cpp)
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
add_executable(v2_test_softmax_primitives unit/Test__SoftmaxPrimitives.cpp)
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
add_executable(v2_test_softmax_inf unit/Test__SoftmaxInf.cpp)
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
add_executable(v2_test_q8_1_softmax_primitives unit/Test__Q8_1SoftmaxPrimitives.cpp)
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
add_executable(v2_test_index_softmax unit/Test__IndexSoftmax.cpp)
target_link_libraries(v2_test_index_softmax
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IndexSoftmax
    COMMAND $<TARGET_FILE:v2_test_index_softmax>
    LABELS "V2;Unit;Primitives;Softmax;IntegerDomain;IndexSoftmax;LUT;Quantization"
    MPI_PROCS 1
)

# Test: Quantised JIT GEMM kernel
add_executable(v2_test_quantised_gemm_kernel unit/kernels/gemm/Test__QuantisedGemmKernel.cpp)
target_link_libraries(v2_test_quantised_gemm_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QuantisedGemmKernel
    COMMAND $<TARGET_FILE:v2_test_quantised_gemm_kernel>
    LABELS "V2;Unit;Kernels;GEMM;Quantised;JIT;CPU"
    MPI_PROCS 1
)

# Test: Quantised GEMM Kernel Reproduction (Asymmetric/Tail bugs)
add_executable(v2_test_quantised_gemm_kernel_repro unit/kernels/Test__QuantisedGemmKernel_Repro.cpp)
target_link_libraries(v2_test_quantised_gemm_kernel_repro
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QuantisedGemmKernel_Repro
    COMMAND $<TARGET_FILE:v2_test_quantised_gemm_kernel_repro>
    LABELS "V2;Unit;Kernels;GEMM;Quantised;Repro"
    MPI_PROCS 1
)

# Test: Quantised GEMM Kernel Q8_1 Output Epilogue
add_executable(v2_test_quantised_gemm_q8_1_output unit/kernels/gemm/Test__QuantisedGemmKernel_Q8_1_Output.cpp)
target_link_libraries(v2_test_quantised_gemm_q8_1_output
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QuantisedGemmKernel_Q8_1_Output
    COMMAND $<TARGET_FILE:v2_test_quantised_gemm_q8_1_output>
    LABELS "V2;Unit;Kernels;GEMM;Quantised;Q8_1;Epilogue"
    MPI_PROCS 1
)

# Test: Quantised GEMM Kernel Q16_1 Output Epilogue
add_executable(v2_test_quantised_gemm_q16_1_output unit/kernels/gemm/Test__QuantisedGemmKernel_Q16_1_Output.cpp)
target_link_libraries(v2_test_quantised_gemm_q16_1_output
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QuantisedGemmKernel_Q16_1_Output
    COMMAND $<TARGET_FILE:v2_test_quantised_gemm_q16_1_output>
    LABELS "V2;Unit;Kernels;GEMM;Quantised;Q16_1;Epilogue"
    MPI_PROCS 1
)

# Test: Quantised JIT GEMM Fused Operations (Bias, Mask, Softmax)
add_executable(v2_test_quantised_gemm_fused unit/kernels/gemm/Test__QuantisedGemmFused.cpp)
target_link_libraries(v2_test_quantised_gemm_fused
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_QuantisedGemmFused
    COMMAND $<TARGET_FILE:v2_test_quantised_gemm_fused>
    LABELS "V2;Unit;Kernels;GEMM;Quantised;Fused;CPU"
    MPI_PROCS 1
)

# Test: FloatingPointGemmKernel (OneDNN-based FP32/BF16/FP16 GEMM)
add_executable(v2_test_fp_gemm_kernel unit/kernels/gemm/Test__FloatingPointGemmKernel.cpp)
target_link_libraries(v2_test_fp_gemm_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FloatingPointGemmKernel
    COMMAND $<TARGET_FILE:v2_test_fp_gemm_kernel>
    LABELS "V2;Unit;Kernels;GEMM;FloatingPoint;OneDNN;FP32;Strided;Softmax;CausalMask;CPU"
    MPI_PROCS 1
)

# Test: ActivationTraits (trait dispatch to softmax primitives, GEMM, workspace allocation)
add_executable(v2_test_activation_traits unit/Test__ActivationTraits.cpp)
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
add_executable(v2_test_rmsnorm_typed_kernel unit/Test__CPURMSNormKernelT.cpp)
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
add_executable(v2_test_rope_typed_kernel unit/Test__CPURoPEKernelT.cpp)
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
add_executable(v2_test_rope_q16_block_sizes unit/Test__CPURoPEKernelT_Q16_1_BlockSizes.cpp)
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
add_executable(v2_test_q16_head_normalization unit/Test__Q16HeadNormalization.cpp)
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
add_executable(v2_test_q16_vnni_safe_normalization unit/Test__Q16_VNNI_Safe_Normalization.cpp)
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

# Test: CPUAttentionKernelT (Enum-based attention kernel via Factory)
add_executable(v2_test_cpu_attention_kernel_t unit/Test__CPUAttentionKernelT.cpp)
target_link_libraries(v2_test_cpu_attention_kernel_t
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUAttentionKernelT
    COMMAND $<TARGET_FILE:v2_test_cpu_attention_kernel_t>
    LABELS "V2;Unit;Kernels;Attention;FP32;BF16;FP16;TypedKernel;CPU;FactoryWiring"
    MPI_PROCS 1
)

# Test: AttentionComputeStage (Phase 9 - ComputeStage for attention via KernelFactory)
add_executable(v2_test_attention_compute_stage unit/Test__AttentionComputeStage.cpp)
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
add_executable(v2_test_cpu_softmax_kernel_typed unit/Test__CPUSoftmaxKernelT.cpp)
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
add_executable(v2_test_bf16_simd_accuracy unit/Test__BF16_SIMD_Accuracy.cpp)
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
add_executable(v2_test_precision_aware_allreduce unit/mpi/Test__PrecisionAwareAllreduce.cpp)
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
add_executable(v2_test_rope_primitives unit/Test__RoPEPrimitives.cpp)
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
add_executable(v2_test_rope_precision unit/Test__RoPEPrecisionCorrectness.cpp)
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
add_executable(v2_test_q16_1_rope unit/Test__Q16_1RoPE.cpp)
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
add_executable(v2_test_q16_variable_block_simd unit/Test__Q16VariableBlockSIMD.cpp)
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
add_executable(v2_test_q8_1_to_q16_rope_blocksizes unit/kernels/Test__Q8_1_to_Q16_RoPE_BlockSizes.cpp)
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
add_executable(v2_test_q8_1_rope unit/Test__Q8_1RoPE.cpp)
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
add_executable(v2_test_q8_1_to_q16_rope_fixed_scale unit/Test__Q8_1_to_Q16_RoPE_FixedScale.cpp)
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

# Test: Fixed-Scale Q16 to Q16 RoPE (for HybridQ16 K precision fix)
add_executable(v2_test_q16_to_q16_rope_fixed_scale unit/Test__Q16_to_Q16_RoPE_FixedScale.cpp)
target_link_libraries(v2_test_q16_to_q16_rope_fixed_scale 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16_to_Q16_RoPE_FixedScale 
    COMMAND $<TARGET_FILE:v2_test_q16_to_q16_rope_fixed_scale>
    LABELS "V2;Unit;Primitives;RoPE;Q16_1;FixedScale;IntegerAttention;HybridQ16;Correctness;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)

# Test: Dynamic-Scale Q16 to Q16 RoPE (Phase 12 - preserves spiky K projection) [DISABLED - project on hold]
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_q16_to_q16_rope_dynamic_scale unit/Test__Q16_to_Q16_RoPE_DynamicScale.cpp)
    target_link_libraries(v2_test_q16_to_q16_rope_dynamic_scale 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Q16_to_Q16_RoPE_DynamicScale 
        COMMAND $<TARGET_FILE:v2_test_q16_to_q16_rope_dynamic_scale>
        LABELS "V2;Unit;Primitives;RoPE;Q16_1;DynamicScale;IntegerAttention;HybridQ16;Correctness;CPU;Phase12"
        MPI_PROCS 1  # RoPE is CPU-only, single rank
    )
endif()

# Test: HybridQ16 K Precision Fix Regression Tests (Phase 8 bugs) [DISABLED - project on hold]
# Covers: single-block RoPE, Q8_1 field names, block_size constraints, FP32→Q8 quantization
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_hybridq16_k_regression unit/Test__HybridQ16_K_Precision_Fix_Regressions.cpp)
    target_link_libraries(v2_test_hybridq16_k_regression 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_HybridQ16_K_Precision_Fix_Regressions 
        COMMAND $<TARGET_FILE:v2_test_hybridq16_k_regression>
        LABELS "V2;Unit;Regression;RoPE;GEMM;Q16_1;Q8_1;IntegerAttention;HybridQ16;Correctness;CPU;Phase8"
        MPI_PROCS 1  # CPU-only, single rank
    )
endif()

# Test: CPURoPEKernelT class interface
add_executable(v2_test_cpu_rope_kernel unit/kernels/Test__CPURoPEKernelT.cpp)
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
add_executable(v2_test_cpu_swiglu_kernel unit/kernels/Test__CPUSwiGLUKernelT.cpp)
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
add_executable(v2_test_cpu_swiglu_kernel_typed unit/Test__CPUSwiGLUKernelT.cpp)
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

# Test: Native Format Decode (Q4_0→int8 without requantization)
add_executable(v2_test_native_format_decode unit/kernels/Test__NativeFormatDecode.cpp)
target_link_libraries(v2_test_native_format_decode
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_NativeFormatDecode
    COMMAND $<TARGET_FILE:v2_test_native_format_decode>
    LABELS "V2;Unit;Kernels;Quantization;NativeDecode;Q4_0;INT8;PrecisionPreservation"
    MPI_PROCS 1
)

# Test: BPETokenizer (native tokenizer implementation)
add_executable(v2_test_bpe_tokenizer integration/Test__BPETokenizer.cpp)
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

# Test: Stop token detection (template-specific stop tokens for generation)
add_executable(v2_test_stop_tokens integration/Test__StopTokens.cpp)
target_link_libraries(v2_test_stop_tokens 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_StopTokens 
    COMMAND $<TARGET_FILE:v2_test_stop_tokens>
    LABELS "V2;Integration;Tokenization;StopTokens;Generation;ChatML"
    MPI_PROCS 1  # Tokenizer is single-rank only
)

# Test: Sampler (token sampling strategies for generation)
add_executable(v2_test_sampler unit/Test__Sampler.cpp)
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

# Test: ChatTemplate (chat template detection and formatting)
add_executable(v2_test_chat_template unit/Test__ChatTemplate.cpp)
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

# Test: ArgParser (command-line argument parsing and validation)
add_executable(v2_test_arg_parser unit/Test__ArgParser.cpp)
target_link_libraries(v2_test_arg_parser 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ArgParser 
    COMMAND $<TARGET_FILE:v2_test_arg_parser>
    LABELS "V2;Unit;CLI;ArgParser;Validation;Configuration"
    MPI_PROCS 1  # ArgParser is single-rank only
)

# Test: BPETokenizer byte encoding (GPT-2 style byte-level encoding/decoding)
add_executable(v2_test_bpe_byte_encoding unit/Test__BPETokenizerByteEncoding.cpp)
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
add_executable(v2_test_attention_dynamic_kvlen unit/Test__AttentionComputeStage_DynamicKVLen.cpp)
target_link_libraries(v2_test_attention_dynamic_kvlen 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionComputeStage_DynamicKVLen 
    COMMAND $<TARGET_FILE:v2_test_attention_dynamic_kvlen>
    LABELS "V2;Unit;Attention;ComputeStage;DynamicKVLen;GraphBufferManagement;Phase9;CPU"
    MPI_PROCS 1  # Single rank for unit testing
)

# Test: SwiGLU parity regression tests (locks in correct formula: gate * silu(up))
add_executable(v2_test_swiglu_parity unit/Test__SwiGLUParity.cpp)
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

# Test: Attention mask application - verify masks are applied before softmax
add_executable(v2_test_attention_mask_application unit/Test__AttentionMaskApplication.cpp)
target_link_libraries(v2_test_attention_mask_application 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionMaskApplication 
    COMMAND $<TARGET_FILE:v2_test_attention_mask_application>
    LABELS "V2;Unit;Attention;MaskApplication;Softmax;PaddingMask;AttentionWeights"
    MPI_PROCS 1  # Unit test for mask application in attention, single rank
)

# Test: Sequential vs Batched Attention Comparison
add_executable(v2_test_attention_sequential_vs_batched unit/Test__AttentionSequentialVsBatched.cpp)
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
add_executable(v2_test_attention_mask_diagnostic unit/Test__AttentionMaskDiagnostic.cpp)
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

# Test: CPUKVCache (supports both single-sequence and batched modes, all precisions)
add_executable(v2_test_cpu_kv_cache unit/tensors/Test__CPUKVCache.cpp)
target_link_libraries(v2_test_cpu_kv_cache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUKVCache 
    COMMAND $<TARGET_FILE:v2_test_cpu_kv_cache>
    LABELS "V2;Unit;KVCache;Batched;SingleSequence;FP32;BF16;FP16;Q8_1;Q16_1;Eviction"
    MPI_PROCS 1
)

# Test: KernelFactory KVCache methods (unified factory interface for KV cache creation)
add_executable(v2_test_kernelfactory_kvcache unit/Test__KernelFactory_KVCache.cpp)
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
add_executable(v2_test_sharded_kv_cache unit/Test__ShardedKVCache.cpp)
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
add_executable(v2_test_iq2xs_decode_vectorization unit/Test__IQ2_XS_DecodeVectorization.cpp)
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
add_executable(v2_test_iq2_xs_unpack_vectorization unit/Test__IQ2_XS_UnpackVectorization.cpp)
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
add_executable(v2_test_iq1m_decode_vectorization unit/Test__IQ1_M_DecodeVectorization.cpp)
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
add_executable(v2_test_iq3s_decode_vectorization unit/Test__IQ3_S_DecodeVectorization.cpp)
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
add_executable(v2_test_iq3xxs_tensor unit/Test__IQ3_XXSTensor.cpp)
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
add_executable(v2_test_iq3s_tensor unit/Test__IQ3_STensor.cpp)
target_link_libraries(v2_test_iq3s_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_S_Tensor 
    COMMAND $<TARGET_FILE:v2_test_iq3s_tensor>
    LABELS "V2;Unit;TensorOperations;IQ3_S;Quantization;IINT8Unpackable"
)

# Test: IQ3_XXS GEMM Integration
add_executable(v2_test_iq3xxs_gemm integration/Test__IQ3_XXS_GEMM.cpp)
target_link_libraries(v2_test_iq3xxs_gemm 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ3_XXS_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq3xxs_gemm>
    LABELS "V2;Integration;GEMM;IQ3_XXS;Quantization"
)

# Test: IQ3_S GEMM Integration
add_executable(v2_test_iq3s_gemm integration/Test__IQ3_S_GEMM.cpp)
target_link_libraries(v2_test_iq3s_gemm 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ3_S_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq3s_gemm>
    LABELS "V2;Integration;GEMM;IQ3_S;Quantization"
)

# Test: IQ2_XXS Tensor (IINT8Unpackable)
add_executable(v2_test_iq2xxs_tensor unit/Test__IQ2_XXSTensor.cpp)
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
add_executable(v2_test_iq2_xxs_unpack_vectorization unit/Test__IQ2_XXS_UnpackVectorization.cpp)
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

# Test: IQ2_XXS GEMM Integration
add_executable(v2_test_iq2xxs_gemm_integration integration/Test__IQ2_XXS_GEMM_Integration.cpp)
target_link_libraries(v2_test_iq2xxs_gemm_integration 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ2_XXS_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq2xxs_gemm_integration>
    LABELS "V2;Integration;GEMM;IQ2_XXS;Quantization"
)


# Test: IQ2_XS GEMM Integration
add_executable(v2_test_iq2xs_gemm_integration integration/Test__IQ2_XS_GEMM_Integration.cpp)
target_link_libraries(v2_test_iq2xs_gemm_integration 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ2_XS_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq2xs_gemm_integration>
    LABELS "V2;Integration;GEMM;IQ2_XS;Quantization"
)

# Test: IQ4_XS GEMM Integration
add_executable(v2_test_iq4xs_gemm_integration integration/Test__IQ4_XS_GEMM_Integration.cpp)
target_link_libraries(v2_test_iq4xs_gemm_integration 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ4_XS_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq4xs_gemm_integration>
    LABELS "V2;Integration;GEMM;IQ4_XS;Quantization"
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

# Test: IQ2_S GEMM Integration
add_executable(v2_test_iq2s_gemm_integration integration/Test__IQ2_S_GEMM_Integration.cpp)
target_link_libraries(v2_test_iq2s_gemm_integration 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ2_S_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq2s_gemm_integration>
    LABELS "V2;Integration;GEMM;IQ2_S;Quantization"
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

# =============================================================================
# SIMD Path Equivalency Tests (Phase 1: Q4_0, Q4_1, Q5_0, Q5_1, Q6_K)
# =============================================================================
# These tests validate that scalar, AVX2, and AVX512 dequantization paths
# produce identical results for quantized tensors with SIMD optimizations.
# Tests use direct method calls (decodeBlockScalar, decodeBlockAVX2, decodeBlockAVX512)
# with synthetic test data to ensure numerical equivalency across all code paths.

# Test: Q4_0 SIMD Equivalency
add_executable(v2_test_q4_0_simd integration/Test__Q4_0Tensor.cpp)
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
add_executable(v2_test_q5_0_simd integration/Test__Q5_0Tensor.cpp)
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

# Test: Q5_0 GEMM Integration
add_executable(v2_test_q5_0_gemm integration/Test__Q5_0_GEMM.cpp)
target_link_libraries(v2_test_q5_0_gemm 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q5_0_GEMM 
    COMMAND $<TARGET_FILE:v2_test_q5_0_gemm>
    MPI_PROCS 1
    LABELS "V2;Integration;TensorOperations;Q5_0;Quantization;GEMM;IINT8Unpackable"
)

# Test: Q5_1 GEMM Integration
add_executable(v2_test_q5_1_gemm integration/Test__Q5_1_GEMM.cpp)
target_link_libraries(v2_test_q5_1_gemm 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q5_1_GEMM 
    COMMAND $<TARGET_FILE:v2_test_q5_1_gemm>
    MPI_PROCS 1
    LABELS "V2;Integration;TensorOperations;Q5_1;Quantization;GEMM;IINT8Unpackable"
)

# Test: Q5_K GEMM Integration
add_executable(v2_test_q5_k_gemm integration/Test__Q5_K_GEMM.cpp)
target_link_libraries(v2_test_q5_k_gemm 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q5_K_GEMM 
    COMMAND $<TARGET_FILE:v2_test_q5_k_gemm>
    MPI_PROCS 1
    LABELS "V2;Integration;TensorOperations;Q5_K;Quantization;GEMM;IINT8Unpackable"
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
add_executable(v2_test_q8_1_quantize_simd unit/Test__Q8_1_QuantizeSIMD.cpp)
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
add_executable(v2_test_q8_0_simd integration/Test__Q8_0Tensor.cpp)
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

# Test: WeightPlacementMap
# WeightPlacementMap Tests (Tensor-Level Device Placement - Phase 2)
add_executable(v2_test_weight_placement_map unit/loaders/Test__WeightPlacementMap.cpp)
target_link_libraries(v2_test_weight_placement_map llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightPlacementMap
    COMMAND $<TARGET_FILE:v2_test_weight_placement_map>
    LABELS "V2;Unit;DeviceManagement;WeightPlacement;Heterogeneous;MoE"
)

# Test: WeightManager Sharding
# Tests weight sharding for tensor parallelism (Phase 1: row-parallel only)
add_executable(v2_test_weight_manager_sharding unit/loaders/Test__WeightManagerSharding.cpp)
target_link_libraries(v2_test_weight_manager_sharding llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightManagerSharding
    COMMAND $<TARGET_FILE:v2_test_weight_manager_sharding>
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
add_executable(v2_test_work_distributor unit/Test__WorkDistributor.cpp)
target_link_libraries(v2_test_work_distributor llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_WorkDistributor
    COMMAND $<TARGET_FILE:v2_test_work_distributor>
    LABELS "V2;Unit;Execution;WorkDistribution;MPI;DeviceManagement;HierarchicalSlicing"
    MPI_PROCS 1  # Tests slice computation, not actual MPI communication
)

# Test: DeviceContext (abstract execution context for CPU/GPU)
# Tests CPUDeviceContext memory allocation, parallel execution, and workspace management
add_executable(v2_test_device_context unit/Test__DeviceContext.cpp)
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
add_v2_test(V2_Unit_MultiGPU
    COMMAND $<TARGET_FILE:v2_test_multi_gpu>
    LABELS "V2;Unit;DeviceManagement;MultiGPU;CUDA;ROCm;HeterogeneousBackends;TensorTransfer;KernelFallback"
    MPI_PROCS 1  # Tests device management locally, no MPI needed
)

# Test: MultiGPU Real Model (Integration tests with actual GGUF model data)
# Tests GPU transfers of real weight tensors, data integrity verification,
# cross-backend transfers (CUDA ↔ ROCm), and multi-tensor GPU staging
add_executable(v2_test_multi_gpu_real_model integration/Test__MultiGPU_RealModel.cpp)
target_link_libraries(v2_test_multi_gpu_real_model llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_MultiGPU_RealModel
    COMMAND $<TARGET_FILE:v2_test_multi_gpu_real_model>
    LABELS "V2;Integration;DeviceManagement;MultiGPU;CUDA;ROCm;ModelLoading;TensorTransfer;WeightManagement"
    MPI_PROCS 1  # Tests tensor transfers locally, no MPI needed
)

# =============================================================================
# CUDA Kernel Infrastructure Tests (Phase 1)
# =============================================================================
# Test: CUDA Tensor Transfer (Phase 1 - GPU upload/download via TensorBase::ensureOnDevice)
# Tests FP32, BF16, FP16 tensor upload/download, round-trip verification, edge cases
add_executable(v2_test_cuda_tensor_transfer unit/Test__CUDATensorTransfer.cpp)
target_link_libraries(v2_test_cuda_tensor_transfer llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CUDATensorTransfer
    COMMAND $<TARGET_FILE:v2_test_cuda_tensor_transfer>
    LABELS "V2;Unit;CUDA;TensorTransfer;FP32;BF16;FP16;GPUMemory;DeviceManagement"
    MPI_PROCS 1  # Tests GPU transfers locally, no MPI needed
)

# Test: CUDA Flash Attention Decode KV Length Bug (Bug fix validation)
# Bug discovered 2026-01-07: compute() hardcoded kv_len = seq_len. For decode mode
# with seq_len=1 but kv_len=7, GPU only attended to 1 token instead of 7.
# Fix: Use compute_decode() which takes separate seq_len and kv_len parameters.
add_executable(v2_unit_cuda_flash_attention_decode_kvlen unit/kernels/cuda/Test__CUDAFlashAttention_DecodeKVLen.cpp)
target_link_libraries(v2_unit_cuda_flash_attention_decode_kvlen llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CUDAFlashAttention_DecodeKVLen
    COMMAND $<TARGET_FILE:v2_unit_cuda_flash_attention_decode_kvlen>
    LABELS "V2;Unit;CUDA;Attention;FlashAttention;FlashDecode;DecodeMode;KVCache;BugFix"
    MPI_PROCS 1  # Tests GPU attention locally, no MPI needed
)

# Test: cuBLAS GEMM (Phase 1 - dense GEMM via cuBLAS for FP32/FP16/BF16)
# Tests correctness against CPU reference, various matrix sizes, transpose variants
add_executable(v2_test_cublas_gemm unit/Test__CuBLASGemm.cpp)
target_link_libraries(v2_test_cublas_gemm llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CuBLASGemm
    COMMAND $<TARGET_FILE:v2_test_cublas_gemm>
    LABELS "V2;Unit;CUDA;cuBLAS;GEMM;FP32;FP16;BF16;DenseGEMM;KernelCorrectness"
    MPI_PROCS 1  # Tests GPU GEMM locally, no MPI needed
)

# Test: hipBLAS GEMM (ROCm - dense GEMM via hipBLAS for FP32/FP16)
# Tests ROCmFloatingPointGemmKernel correctness against CPU reference, various matrix sizes
# MI50: 13.4 TFLOPS FP32, 26.5 TFLOPS FP16 (no native BF16)
add_executable(v2_test_hipblas_gemm unit/kernels/rocm/Test__ROCmFloatingPointGemmKernel.cpp)
target_link_libraries(v2_test_hipblas_gemm llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_HipBLASGemm
    COMMAND $<TARGET_FILE:v2_test_hipblas_gemm>
    LABELS "V2;Unit;ROCm;hipBLAS;GEMM;FP32;FP16;DenseGEMM;KernelCorrectness;MI50"
    MPI_PROCS 1  # Tests GPU GEMM locally, no MPI needed
)

# Test: CUDA Basic Pipeline (Phase 1 - end-to-end KernelFactory + tensor transfer)
# Tests KernelFactory dispatch to cuBLAS, tensor upload/download in GEMM context
add_executable(v2_integration_cuda_basic_pipeline integration/Test__CUDABasicPipeline.cpp)
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
add_executable(v2_integration_tensor_coherence integration/Test__TensorCoherence.cpp)
target_link_libraries(v2_integration_tensor_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_TensorCoherence
    COMMAND $<TARGET_FILE:v2_integration_tensor_coherence>
    LABELS "V2;Integration;CUDA;TensorTransfer;Coherence;StateMachine;FP32;BF16;FP16;GPUMemory;DeviceManagement"
    MPI_PROCS 1  # Tests GPU coherence locally, no MPI needed
)

# Test: CUDA GEMM Parity (FP32 and IQ4_NL: CUDA vs CPU kernel comparison)
# Tests CUDAFloatingPointGemmKernel and CUDAQuantisedGemmKernel through KernelFactory
# with cosine similarity metrics against CPU reference kernels
add_executable(v2_integration_cuda_gemm_parity integration/Test__CUDAGemmParity.cpp)
target_link_libraries(v2_integration_cuda_gemm_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAGemmParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_gemm_parity>
    LABELS "V2;Integration;CUDA;GEMM;Parity;FP32;IQ4_NL;Quantized;KernelFactory;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# Test: CUDA Ops Parity (RMSNorm, SwiGLU, RoPE)
# Tests CUDA kernel implementations of normalization and activation functions
# against CPU reference kernels with cosine similarity metrics
add_executable(v2_integration_cuda_ops_parity integration/Test__CUDAOpsParity.cpp)
target_link_libraries(v2_integration_cuda_ops_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAOpsParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_ops_parity>
    LABELS "V2;Integration;CUDA;Parity;RMSNorm;SwiGLU;RoPE;FP32;BF16;FP16;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

# Test: CUDA Flash Attention Parity
# Tests Flash Attention 2 (prefill) and Flash Decoding (decode) CUDA kernels
# against CPU attention reference with cosine similarity metrics
add_executable(v2_integration_cuda_flash_attention_parity integration/Test__CUDAFlashAttentionParity.cpp)
target_link_libraries(v2_integration_cuda_flash_attention_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDAFlashAttentionParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_flash_attention_parity>
    LABELS "V2;Integration;CUDA;Parity;Attention;FlashAttention;FlashDecode;GQA;FP32;CosineSimilarity"
    MPI_PROCS 1  # Tests GPU vs CPU parity locally, no MPI needed
)

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

# Test: CUDA Ring Buffer KV Cache Parity
# Tests ring buffer KV cache with O(1) eviction, wrap-around, batched gather, FP32/FP16/BF16
if(HAVE_CUDA)
add_executable(v2_integration_cuda_ring_kv_cache_parity integration/Test__CUDARingKVCacheParity.cpp)
target_link_libraries(v2_integration_cuda_ring_kv_cache_parity llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_CUDARingKVCacheParity
    COMMAND $<TARGET_FILE:v2_integration_cuda_ring_kv_cache_parity>
    LABELS "V2;Integration;CUDA;Parity;KVCache;RingBuffer;Eviction;SlidingWindow;FP32;FP16;BF16"
    MPI_PROCS 1  # Tests GPU KV cache locally, no MPI needed
)
endif()

# Test: CUDA Full Model Inference
# End-to-end GPU inference: model loading, forward pass, token prediction
# Tests GPU vs CPU parity for prefill and decode
add_executable(v2_integration_cuda_full_model_inference integration/Test__CUDAFullModelInference.cpp)
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
#   - PyTorch snapshots generated by python/reference/generate_qwen2_pipeline_snapshots.py

find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    get_filename_component(WORKSPACE_ROOT_PARITY "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

    # =========================================================================
    # Qwen2 Parity Tests (integration/parity/qwen2/)
    # =========================================================================

    # Qwen2 CUDA vs PyTorch Parity Test
    add_executable(v2_integration_parity_qwen2_cuda_vs_pytorch 
        integration/parity/qwen2/Test__Qwen2_CUDA_vs_PyTorch.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_cuda_vs_pytorch 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_cuda_vs_pytorch PRIVATE 
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
    )
    add_v2_test(V2_Integration_Parity_Qwen2_CUDA_vs_PyTorch
        COMMAND $<TARGET_FILE:v2_integration_parity_qwen2_cuda_vs_pytorch>
        LABELS "V2;Integration;Parity;Qwen2;CUDA;PyTorch;GoldenReference;FullModel;Pipeline"
        MPI_PROCS 1
    )

    # Qwen2 CPU vs PyTorch Parity Test
    add_executable(v2_integration_parity_qwen2_cpu_vs_pytorch 
        integration/parity/qwen2/Test__Qwen2_CPU_vs_PyTorch.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_cpu_vs_pytorch 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_cpu_vs_pytorch PRIVATE 
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
    )
    add_v2_test(V2_Integration_Parity_Qwen2_CPU_vs_PyTorch
        COMMAND $<TARGET_FILE:v2_integration_parity_qwen2_cpu_vs_pytorch>
        LABELS "V2;Integration;Parity;Qwen2;CPU;PyTorch;GoldenReference;FullModel;Pipeline"
        MPI_PROCS 1
    )

    # Qwen2 ROCm vs PyTorch Parity Test
    add_executable(v2_integration_parity_qwen2_rocm_vs_pytorch 
        integration/parity/qwen2/Test__Qwen2_ROCm_vs_PyTorch.cpp
        ${WORKSPACE_ROOT_PARITY}/external/cnpy/cnpy.cpp
    )
    target_link_libraries(v2_integration_parity_qwen2_rocm_vs_pytorch 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
        ZLIB::ZLIB
    )
    target_include_directories(v2_integration_parity_qwen2_rocm_vs_pytorch PRIVATE 
        ${WORKSPACE_ROOT_PARITY}/external/cnpy
    )
    add_v2_test(V2_Integration_Parity_Qwen2_ROCm_vs_PyTorch
        COMMAND $<TARGET_FILE:v2_integration_parity_qwen2_rocm_vs_pytorch>
        LABELS "V2;Integration;Parity;Qwen2;ROCm;PyTorch;GoldenReference;FullModel;Pipeline"
        MPI_PROCS 1
    )

    message(STATUS "V2: Qwen2 parity tests enabled (CUDA, CPU, ROCm vs PyTorch)")
else()
    message(STATUS "V2: Parity tests disabled (ZLIB not found)")
endif()

# Test: ComputeStage (device-agnostic kernel dispatch abstraction)
# Tests all compute stage types: RMSNorm, RoPE, Attention, SwiGLU, ResidualAdd, MoE stages
add_executable(v2_test_compute_stage unit/Test__ComputeStage.cpp)
target_link_libraries(v2_test_compute_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeStage
    COMMAND $<TARGET_FILE:v2_test_compute_stage>
    LABELS "V2;Unit;Execution;ComputeStage;RMSNorm;RoPE;Attention;SwiGLU;MoE;KernelDispatch"
    MPI_PROCS 1  # Tests CPU stages locally, no MPI needed
)

# Test: GEMMStage (GEMM compute stage with workspace support)
# Tests construction, IWorkspaceConsumerStage delegation, params validation, metrics
add_executable(v2_test_gemm_stage unit/Test__GEMMStage.cpp)
target_link_libraries(v2_test_gemm_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GEMMStage
    COMMAND $<TARGET_FILE:v2_test_gemm_stage>
    LABELS "V2;Unit;Execution;ComputeStage;GEMM;Workspace;Validation"
    MPI_PROCS 1  # Tests stage construction and params, no MPI needed
)

# Test: ExecutionPolicy (declarative operation control)
# Tests factory methods, environment variable parsing, utility methods, comparison operators
add_executable(v2_test_execution_policy unit/Test__ExecutionPolicy.cpp)
target_link_libraries(v2_test_execution_policy llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ExecutionPolicy
    COMMAND $<TARGET_FILE:v2_test_execution_policy>
    LABELS "V2;Unit;Execution;ExecutionPolicy;Configuration;EnvironmentVariables"
    MPI_PROCS 1
)

# Test: EmbeddingStage Q16_1 Output
# Tests that EmbeddingStage correctly produces Q16_1 output for HybridQ16 mode [DISABLED - project on hold]
# Verifies FP32 → Q16_1 quantization preserves embedding values
# NOTE: This is an Integration test (slow, requires model loading)
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_embedding_stage_q16_1 unit/stages/Test__EmbeddingStage_Q16_1Output.cpp)
    target_link_libraries(v2_test_embedding_stage_q16_1 llaminar2_core GTest::gtest GTest::gtest_main)
    add_v2_test(V2_Integration_EmbeddingStage_Q16_1Output
        COMMAND $<TARGET_FILE:v2_test_embedding_stage_q16_1>
        LABELS "V2;Integration;Execution;ComputeStage;Embedding;Q16_1;HybridQ16;Quantization"
        MPI_PROCS 1
    )
endif()

# Test: HybridQ16 Layer 0 Dataflow Diagnostics [DISABLED - project on hold]
# Traces data flow from embedding through RMSNorm in HybridQ16 mode
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_hybridq16_layer0_dataflow unit/stages/Test__HybridQ16_Layer0_Dataflow.cpp)
    target_link_libraries(v2_test_hybridq16_layer0_dataflow llaminar2_core GTest::gtest GTest::gtest_main)
    add_v2_test(V2_Unit_HybridQ16_Layer0_Dataflow
        COMMAND $<TARGET_FILE:v2_test_hybridq16_layer0_dataflow>
        LABELS "V2;Unit;Execution;ComputeStage;HybridQ16;Dataflow;Diagnostics"
        MPI_PROCS 1
    )
endif()

# Test: FusedAttentionWoStage with Q16_INTEGER backend
# Tests stage construction, type, backend support, and dump info with Q16FusedAttentionKernel
add_executable(v2_test_fused_attention_wo_stage_q16 unit/stages/Test__FusedAttentionWoStage_Q16Backend.cpp)
target_link_libraries(v2_test_fused_attention_wo_stage_q16 llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_FusedAttentionWoStage_Q16Backend
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_stage_q16>
    LABELS "V2;Unit;Execution;ComputeStage;FusedAttentionWo;Q16_INTEGER;Q16FusedAttentionKernel"
    MPI_PROCS 1
)

# Test: KVCacheAppendStage kv_cache_scale Propagation [DISABLED - project on hold]
# Locks in the VNNI-safe quantization contract for Q16_1 KV cache
# Tests default scale (8.0), custom scale propagation, and VNNI safety limits
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_kv_cache_scale_propagation unit/stages/Test__KVCacheScalePropagation.cpp)
    target_link_libraries(v2_test_kv_cache_scale_propagation llaminar2_core GTest::gtest GTest::gtest_main)
    add_v2_test(V2_Unit_KVCacheScalePropagation
        COMMAND $<TARGET_FILE:v2_test_kv_cache_scale_propagation>
        LABELS "V2;Unit;Execution;ComputeStage;KVCache;VNNI;Quantization;Q16_1;Scale"
        MPI_PROCS 1
    )
endif()

# Test: ResidualAddStage Precision Awareness
# Tests precision-aware residual addition: FP32, BF16, FP16, Q8_1
# Verifies Q8_1 uses native simd::q8_1_add_q8_1 (no explicit dequant)
add_executable(v2_test_residual_add_precision unit/Test__ResidualAddStage_Precision.cpp)
target_link_libraries(v2_test_residual_add_precision llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ResidualAddPrecision
    COMMAND $<TARGET_FILE:v2_test_residual_add_precision>
    LABELS "V2;Unit;Execution;ComputeStage;ResidualAdd;Precision;Q8_1;SIMD"
    MPI_PROCS 1
)

# Test: Pure Integer Residual Add (Q16_1 + Q16_1 → Q16_1)
# Phase 6 of Q16 Integer Attention: Full integer residual stream
# Tests block sizes 32/64/128, edge cases, error handling
add_executable(v2_test_residual_add_q16_1 unit/stages/Test__ResidualAddStage_Q16_1.cpp)
target_link_libraries(v2_test_residual_add_q16_1 llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ResidualAddQ16_1
    COMMAND $<TARGET_FILE:v2_test_residual_add_q16_1>
    LABELS "V2;Unit;Execution;ComputeStage;ResidualAdd;Q16_1;IntegerAttention;Phase6"
    MPI_PROCS 1
)

# Test: ComputeGraph Execution Flow
# Comprehensive tests for graph construction, dependency resolution, execution order,
# error handling, snapshot callbacks, statistics tracking, and multi-device execution
add_executable(v2_test_compute_graph_execution unit/Test__ComputeGraphExecution.cpp)
target_link_libraries(v2_test_compute_graph_execution llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeGraphExecution
    COMMAND $<TARGET_FILE:v2_test_compute_graph_execution>
    LABELS "V2;Unit;Execution;ComputeGraph;TopologicalSort;DependencyResolution;SnapshotCallback;Statistics"
    MPI_PROCS 1
)

# Test: GraphOrchestrator Snapshot API
# Tests pipeline-compatible snapshot capture API (enableSnapshotCapture/getSnapshot/getSnapshotKeys)
# and stage name conversion for E2E test compatibility
add_executable(v2_test_graph_orchestrator_snapshots unit/Test__GraphOrchestratorSnapshots.cpp)
target_link_libraries(v2_test_graph_orchestrator_snapshots llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphOrchestratorSnapshots
    COMMAND $<TARGET_FILE:v2_test_graph_orchestrator_snapshots>
    LABELS "V2;Unit;Execution;GraphOrchestrator;Snapshots;StageNameConversion"
    MPI_PROCS 1
)

# Test: LayerExecutor (layer-level execution orchestration)
# Tests compute graph construction, topological ordering, dependency tracking,
# attention/FFN/MoE graph building, and multi-device execution
add_executable(v2_test_layer_executor unit/Test__LayerExecutor.cpp)
target_link_libraries(v2_test_layer_executor llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LayerExecutor
    COMMAND $<TARGET_FILE:v2_test_layer_executor>
    LABELS "V2;Unit;Execution;LayerExecutor;ComputeGraph;Orchestration;Attention;FFN;MoE;MultiDevice"
    MPI_PROCS 1  # Tests execution orchestration locally
)

# Test: BufferRole (buffer role classification and descriptor types)
# Tests BufferRole enum, BufferDescriptor, StageBufferRequirements, and real-world usage patterns
# Part of Phase 1: GraphExecutor buffer management infrastructure
add_executable(v2_test_buffer_role unit/Test__BufferRole.cpp)
target_link_libraries(v2_test_buffer_role llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_BufferRole
    COMMAND $<TARGET_FILE:v2_test_buffer_role>
    LABELS "V2;Unit;Execution;BufferRole;BufferDescriptor;BufferManagement"
    MPI_PROCS 1
)

# Test: GraphBufferManager (centralized buffer allocation and management)
# Tests buffer allocation, retrieval, statistics, graph allocation, and executor integration
# Part of Phase 2: GraphExecutor buffer management infrastructure
add_executable(v2_test_graph_buffer_manager unit/Test__GraphBufferManager.cpp)
target_link_libraries(v2_test_graph_buffer_manager llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphBufferManager
    COMMAND $<TARGET_FILE:v2_test_graph_buffer_manager>
    LABELS "V2;Unit;Execution;GraphBufferManager;BufferManagement;TensorAllocation"
    MPI_PROCS 1
)

# Test: CollectiveBufferAllocation (Phase 3 Buffer Registration API)
# Tests integration of BAR-region allocator with buffer management
# for collective operations (AllReduce, AllGather)
add_executable(v2_test_collective_buffer_allocation unit/Test__CollectiveBufferAllocation.cpp)
target_link_libraries(v2_test_collective_buffer_allocation llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CollectiveBufferAllocation
    COMMAND $<TARGET_FILE:v2_test_collective_buffer_allocation>
    LABELS "V2;Unit;Execution;Collective;BufferRegistration;PCIeBAR;CrossVendor"
    MPI_PROCS 1
)

# Test: ComputeGraphMerge (graph merging, root/leaf detection)
# Tests merge(), getRootNodes(), getLeafNodes() for combining subgraphs
# Part of Phase 3: Declarative Qwen2Graph refactoring
add_executable(v2_test_compute_graph_merge unit/Test__ComputeGraphMerge.cpp)
target_link_libraries(v2_test_compute_graph_merge llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeGraphMerge
    COMMAND $<TARGET_FILE:v2_test_compute_graph_merge>
    LABELS "V2;Unit;Execution;ComputeGraph;GraphMerge;DAG"
    MPI_PROCS 1
)

# Test: IGraphBuilder interface and MockGraphBuilder
# Part of Phase 5: Declarative Qwen2Graph refactoring - Interface extraction
add_executable(v2_test_igraph_builder unit/Test__IGraphBuilder.cpp)
target_link_libraries(v2_test_igraph_builder llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_IGraphBuilder
    COMMAND $<TARGET_FILE:v2_test_igraph_builder>
    LABELS "V2;Unit;Execution;IGraphBuilder;MockGraphBuilder;Interface"
    MPI_PROCS 1
)

# Test: HybridPrecisionConfig (per-buffer precision configuration for Hybrid mode)
# Tests buffer precision resolution, default configuration, and requiresSeparateBuffer
# Part of Hybrid Activation Precision Mode implementation
add_executable(v2_test_hybrid_precision_config unit/Test__HybridPrecisionConfig.cpp)
target_link_libraries(v2_test_hybrid_precision_config llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_HybridPrecisionConfig
    COMMAND $<TARGET_FILE:v2_test_hybrid_precision_config>
    LABELS "V2;Unit;Execution;HybridPrecision;BufferPrecision;ActivationPrecision"
    MPI_PROCS 1
)

# Test: StageBufferRequirements (IComputeStage::getBufferRequirements())
# Tests buffer requirement declarations for all compute stages (SwiGLU, ResidualAdd,
# RMSNorm, RoPE, GEMM, FusedQKVGEMM, FusedGateUpGEMM) for GraphBufferManager integration
# Part of Phase 3: Stage Migration for buffer management infrastructure
add_executable(v2_test_stage_buffer_requirements unit/Test__StageBufferRequirements.cpp)
target_link_libraries(v2_test_stage_buffer_requirements llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageBufferRequirements
    COMMAND $<TARGET_FILE:v2_test_stage_buffer_requirements>
    LABELS "V2;Unit;Execution;ComputeStage;BufferRequirements;BufferRole"
    MPI_PROCS 1
)

# Test: TensorVerification (unified tensor verification system)
# Tests VerificationResult, verifyRawBuffer, VerificationFailure, dumpStageBuffers.
# Part of Tensor Contracts and Validation Framework - Phase 1
add_executable(v2_test_tensor_verification unit/utils/Test__TensorVerification.cpp)
target_link_libraries(v2_test_tensor_verification llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TensorVerification
    COMMAND $<TARGET_FILE:v2_test_tensor_verification>
    LABELS "V2;Unit;Utils;TensorVerification;Validation;Debug"
    MPI_PROCS 1
)

# Test: GraphValidator (buffer flow validation for compute graphs)
# Tests buffer producer/consumer ordering, orphan detection, type compatibility.
# Part of Buffer Contract Validation System - Phase 2: Buffer Contracts
add_executable(v2_test_graph_validator unit/Test__GraphValidator.cpp)
target_link_libraries(v2_test_graph_validator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphValidator
    COMMAND $<TARGET_FILE:v2_test_graph_validator>
    LABELS "V2;Unit;Execution;Validation;GraphValidator;BufferContracts"
    MPI_PROCS 1
)

# Test: StageOutputManifest (getDeclaredOutputs for stage output contracts)
# Tests that stages correctly declare their output buffers with producer contracts.
# Part of Buffer Contract Validation System - Phase 3: Stage Output Manifest
add_executable(v2_test_stage_output_manifest unit/Test__StageOutputManifest.cpp)
target_link_libraries(v2_test_stage_output_manifest llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageOutputManifest
    COMMAND $<TARGET_FILE:v2_test_stage_output_manifest>
    LABELS "V2;Unit;Execution;Validation;StageOutputManifest;BufferContracts"
    MPI_PROCS 1
)

# Test: InferenceMode (centralized mode context for activation precision)
# Tests mode identification, buffer requirements, attention strategy, validation.
# Part of Buffer Contract Validation System - Phase 4: InferenceMode Context
add_executable(v2_test_inference_mode unit/Test__InferenceMode.cpp)
target_link_libraries(v2_test_inference_mode llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_InferenceMode
    COMMAND $<TARGET_FILE:v2_test_inference_mode>
    LABELS "V2;Unit;Execution;Validation;InferenceMode;BufferContracts"
    MPI_PROCS 1
)

# Test: ValidationConfig (buffer validation environment variables and GraphExecutor integration)
# Tests LLAMINAR_VALIDATE_BUFFERS, LLAMINAR_FAIL_ON_ZERO, LLAMINAR_FAIL_ON_NAN.
# Part of Buffer Contract Validation System - Phase 5: Debug Validation Flag
add_executable(v2_test_validation_config unit/Test__ValidationConfig.cpp)
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
add_executable(v2_test_streaming_env unit/Test__StreamingEnv.cpp)
target_link_libraries(v2_test_streaming_env llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StreamingEnv
    COMMAND $<TARGET_FILE:v2_test_streaming_env>
    LABELS "V2;Unit;WeightStreaming;DebugEnv;Configuration"
    MPI_PROCS 1
)

# Test: StageCoherence (automatic device coherence at stage boundaries)
# Tests CoherencePolicy enum, cohereInputs/markOutputsDirty, buffer extraction,
# and stage coherence policy reporting.
# Part of Device Abstraction Phase 2: Automatic Stage Boundary Coherence
add_executable(v2_test_stage_coherence unit/Test__StageCoherence.cpp)
target_link_libraries(v2_test_stage_coherence llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageCoherence
    COMMAND $<TARGET_FILE:v2_test_stage_coherence>
    LABELS "V2;Unit;Coherence;GraphExecutor;DeviceAbstraction;Execution"
    MPI_PROCS 1
)

# Test: LivenessAnalyzer (buffer lifetime analysis for memory optimization)
# Tests buffer liveness tracking, aliasing group computation, memory savings calculation.
# Part of Phase 4: Liveness Analysis for buffer management infrastructure
add_executable(v2_test_liveness_analyzer unit/Test__LivenessAnalyzer.cpp)
target_link_libraries(v2_test_liveness_analyzer llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_LivenessAnalyzer
    COMMAND $<TARGET_FILE:v2_test_liveness_analyzer>
    LABELS "V2;Unit;Execution;LivenessAnalyzer;BufferAliasing;MemoryOptimization"
    MPI_PROCS 1
)

# Test: Qwen2GraphBufferManagement (GraphBufferManager integration with GraphOrchestrator)
# Tests setTensorFactory/initializeBuffers workflow, buffer allocation/binding,
# statistics tracking, and release cleanup.
# Part of Phase 3: Buffer Management moved to GraphOrchestrator
add_executable(v2_test_graph_orchestrator_buffer_mgmt unit/Test__GraphOrchestratorBufferManagement.cpp)
target_link_libraries(v2_test_graph_orchestrator_buffer_mgmt llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphOrchestratorBufferManagement
    COMMAND $<TARGET_FILE:v2_test_graph_orchestrator_buffer_mgmt>
    LABELS "V2;Unit;GraphOrchestrator;BufferManagement;GraphBufferManager"
    MPI_PROCS 1
)

# Test: GraphOrchestrator (clean separation of graph building and execution)
# Tests the orchestrator's execution, caching, and device context management:
# - Construction with graph builder or config
# - Cache initialization and management
# - Device context lazy creation and caching
# - Cache statistics tracking
add_executable(v2_test_graph_orchestrator unit/Test__GraphOrchestrator.cpp)
target_link_libraries(v2_test_graph_orchestrator llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphOrchestrator
    COMMAND $<TARGET_FILE:v2_test_graph_orchestrator>
    LABELS "V2;Unit;Qwen2;GraphOrchestrator;GraphExecution;DeclarativeDesign"
    MPI_PROCS 1
)

# Test: GraphOrchestrator Phase-Aware Weights (Gap 3 - CPU Decode Participation)
# Tests the phase-aware weight selection mechanism for "Option A: Selective Duplication":
# - Phase tracking (PREFILL vs DECODE)
# - WeightManager and PlacementMap integration
# - Phase-aware weight selection logic
# - CPU decode participation checks
add_executable(v2_test_graph_orchestrator_phase_aware_weights unit/Test__GraphOrchestratorPhaseAwareWeights.cpp)
target_link_libraries(v2_test_graph_orchestrator_phase_aware_weights llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphOrchestrator_PhaseAwareWeights
    COMMAND $<TARGET_FILE:v2_test_graph_orchestrator_phase_aware_weights>
    LABELS "V2;Unit;GraphOrchestrator;PhaseAwareWeights;CPUDecodeParticipation;Gap3"
    MPI_PROCS 1
)

# Test: GraphOrchestrator Weight Streaming (Option B - On-Demand Weight Transfer)
# Tests the IWeightStreamer integration with GraphOrchestrator:
# - Phase transition notifications to streamer
# - Weight streaming hooks (ensure/prefetch/release) during layer execution
# - Streaming is optional (works without streamer set)
add_executable(v2_test_graph_orchestrator_weight_streaming unit/Test__GraphOrchestratorWeightStreaming.cpp)
target_link_libraries(v2_test_graph_orchestrator_weight_streaming llaminar2_core GTest::gtest GTest::gtest_main GTest::gmock)
add_v2_test(V2_Unit_GraphOrchestrator_WeightStreaming
    COMMAND $<TARGET_FILE:v2_test_graph_orchestrator_weight_streaming>
    LABELS "V2;Unit;GraphOrchestrator;WeightStreaming;OptionB;IWeightStreamer"
    MPI_PROCS 1
)

# Test: GraphResolver (declarative schema resolution for multi-model support)
# Tests the three-layer architecture: GraphSchema -> GraphResolver -> ComputeGraph
# Part of Phase 4: Schema-based graph building for model scalability
add_executable(v2_test_graph_resolver unit/Test__GraphResolver.cpp)
target_link_libraries(v2_test_graph_resolver llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GraphResolver
    COMMAND $<TARGET_FILE:v2_test_graph_resolver>
    LABELS "V2;Unit;GraphResolver;GraphSchema;DeclarativeDesign;MultiModel"
    MPI_PROCS 1
)

# Test: Qwen2GraphSchema (schema-based Qwen2Graph building - Phase 4d)
# Tests the new buildForwardGraphFromSchema() method that uses
# Qwen2Schema + GraphResolver + GraphBuilder architecture.
# Validates that schema-based graphs match legacy imperative graphs.
add_executable(v2_test_qwen2_graph_schema unit/Test__Qwen2GraphSchema.cpp)
target_link_libraries(v2_test_qwen2_graph_schema llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_Qwen2GraphSchema
    COMMAND $<TARGET_FILE:v2_test_qwen2_graph_schema>
    LABELS "V2;Unit;Qwen2;GraphSchema;DeclarativeDesign;Phase4d"
    MPI_PROCS 1
)

# Test: ExecutionConfigAutoEnable (regression test for empty graph bug)
# Tests that use_graph_buffer_management=true auto-enables all exec_* flags
# to prevent empty graphs (0 nodes) when graph buffer management is enabled.
# Covers the "repeating tokens" bug caused by graph having no computation nodes.
add_executable(v2_test_execution_config_auto_enable unit/Test__ExecutionConfigAutoEnable.cpp)
target_link_libraries(v2_test_execution_config_auto_enable llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ExecutionConfigAutoEnable
    COMMAND $<TARGET_FILE:v2_test_execution_config_auto_enable>
    LABELS "V2;Unit;Execution;ExecutionConfig;GraphBufferManagement;Regression"
    MPI_PROCS 1
)

# Test: CPUKernelDeviceIndex (regression test for device_idx validation bug)
# Tests that CPU kernels accept any device_idx value (not just -1) for NUMA support.
# Covers the bug where CPUAttentionKernelT rejected device_idx=0 or 1.
add_executable(v2_test_cpu_kernel_device_index unit/Test__CPUKernelDeviceIndex.cpp)
target_link_libraries(v2_test_cpu_kernel_device_index llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_CPUKernelDeviceIndex
    COMMAND $<TARGET_FILE:v2_test_cpu_kernel_device_index>
    LABELS "V2;Unit;Kernels;Attention;DeviceIndex;NUMA;Regression"
    MPI_PROCS 1
)

# Test: DecomposedAttentionRequirement (regression test for shared_from_this crash)
# Tests that graph buffer management requires decomposed attention path (KVCacheAppendStage +
# AttentionComputeStage) instead of AttentionWithKVCacheStage which uses create_view().
# Covers the crash caused by unique_ptr buffers not supporting shared_from_this().

# Test: LayerExecutorInterface (ILayerExecutor interface and MockLayerExecutor)
# Tests polymorphic execution, MockLayerExecutor behavior, interface-based testing patterns,
# configuration, statistics, and multi-device execution through interface abstraction
add_executable(v2_test_layer_executor_interface unit/Test__LayerExecutorInterface.cpp)
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
add_executable(v2_test_stage_params_base unit/Test__StageParamsBase.cpp)
target_link_libraries(v2_test_stage_params_base llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageParamsBase
    COMMAND $<TARGET_FILE:v2_test_stage_params_base>
    LABELS "V2;Unit;ComputeStage;StageParams;DeviceId;Safety"
    MPI_PROCS 1
)

# Test: ComputeStages (individual compute stage unit tests)
# Tests GEMMStage, RMSNormStage, RoPEStage, SwiGLUStage, ResidualAddStage, etc.
# Verifies correctness, precision preservation, FLOP estimation, and backend compatibility
add_executable(v2_test_compute_stages unit/Test__ComputeStages.cpp)
target_link_libraries(v2_test_compute_stages llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeStages
    COMMAND $<TARGET_FILE:v2_test_compute_stages>
    LABELS "V2;Unit;ComputeStage;GEMM;RMSNorm;RoPE;SwiGLU;ResidualAdd;Precision"
    MPI_PROCS 1
)

# Test: StageDumpInfo (getDumpInfo() snapshot support for all compute stages)
# Validates that each compute stage correctly reports inputs, outputs, weights, and scalars
add_executable(v2_test_stage_dump_info unit/Test__StageDumpInfo.cpp)
target_link_libraries(v2_test_stage_dump_info llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_StageDumpInfo
    COMMAND $<TARGET_FILE:v2_test_stage_dump_info>
    LABELS "V2;Unit;ComputeStage;Snapshot;DumpInfo;Debug"
    MPI_PROCS 1
)

# Test: StageDumperNativeFormat (native tensor format preservation in stage dumps) [DISABLED - project on hold]
# Validates that addInput(TensorBase*)/addOutput(TensorBase*) preserve native format
# (Q8_1, Q16_1, FP32) instead of dequantizing to FP32 - critical for replay testing
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_stage_dumper_native_format unit/execution/Test__StageDumperNativeFormat.cpp)
    target_link_libraries(v2_test_stage_dumper_native_format llaminar2_core GTest::gtest GTest::gtest_main)
    add_v2_test(V2_Unit_StageDumperNativeFormat
        COMMAND $<TARGET_FILE:v2_test_stage_dumper_native_format>
        LABELS "V2;Unit;ComputeStage;StageDump;NativeFormat;Q8_1;Q16_1;Debug"
        MPI_PROCS 1
    )
endif()

# Test: KernelSnapshotInfo (getKernelSnapshotInfo() for all kernel implementations)
# Validates that each kernel correctly reports its I/O buffers via IKernelSnapshotCapable
add_executable(v2_test_kernel_snapshot_info unit/Test__KernelSnapshotInfo.cpp)
target_link_libraries(v2_test_kernel_snapshot_info llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_KernelSnapshotInfo
    COMMAND $<TARGET_FILE:v2_test_kernel_snapshot_info>
    LABELS "V2;Unit;Kernel;Snapshot;IKernelSnapshotCapable;Debug"
    MPI_PROCS 1
)

# Test: AllGatherStage (MPI collective for column-parallel LM head)
# Phase 5: Tests AllGatherStage for collecting distributed logits
add_executable(v2_test_allgather_stage unit/Test__AllGatherStage.cpp)
target_link_libraries(v2_test_allgather_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_AllGatherStage
    COMMAND $<TARGET_FILE:v2_test_allgather_stage>
    LABELS "V2;Unit;ComputeStage;AllGather;MPI;TensorParallel;LMHead"
    MPI_PROCS 1
)

# Test: CollectiveStageIntegration (AllreduceStage and AllGatherStage with CollectiveContext)
# Unit test for collective stages using the new collective infrastructure with mocks
add_executable(v2_unit_collective_stage_integration integration/Test__CollectiveStageIntegration.cpp)
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
add_executable(v2_unit_heterogeneous_execution integration/Test__HeterogeneousExecution.cpp)
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

# Test: WorkDistribution - Integration test for work distribution across ranks
# Phase 5b testability refactoring: validates tensor-parallel work distribution via mocks
# Note: Uses mock infrastructure, no real MPI required - runs as unit test
add_executable(v2_integration_work_distribution integration/Test__WorkDistribution.cpp)
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
add_executable(v2_integration_weight_streaming integration/Test__WeightStreaming.cpp)
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
add_executable(v2_integration_phase_aware_execution integration/Test__PhaseAwareExecution.cpp)
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

# Test: PipelineExecutor (LayerExecutor-to-pipeline adapter)
# Test: GQAAttention Second Sequence Handling
# Tests that sequence 1 in batched attention is handled correctly
# Validates buffer offsets, mask application, and cross-sequence isolation

# Test: ModelLoader
add_executable(v2_test_model_loader integration/Test__ModelLoader.cpp)
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

# Test: ModelLoader Row Slice (memory-efficient tensor row slicing from GGUF)
add_executable(v2_test_model_loader_row_slice unit/Test__ModelLoaderRowSlice.cpp)
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
add_executable(v2_test_weight_manager_memory_efficient integration/Test__WeightManagerMemoryEfficient.cpp)
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

# Test: PlacementStrategy and PlacementPlan (weight/compute placement algorithms)
add_executable(v2_test_placement_strategy unit/Test__PlacementStrategy.cpp)
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

# Test: Quantised GEMM Sliced Parity (verifies sliced GEMM across MPI ranks matches unsliced)
add_executable(v2_test_quantised_gemm_sliced_parity integration/Test__QuantisedGemmSlicedParity.cpp)
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
add_executable(v2_integration_q16_mpi_allreduce integration/Test__Q16MPI_Allreduce.cpp)
target_link_libraries(v2_integration_q16_mpi_allreduce 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q16MPI_Allreduce 
    COMMAND $<TARGET_FILE:v2_integration_q16_mpi_allreduce>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Q16;Allreduce;VariableBlockSize;TensorParallel"
)

# Test: QuantisedGemmKernel Fused Operations (Bias, Softmax, SwiGLU)
# Validates mathematical correctness of fused ops against separate primitives
add_executable(v2_integration_quantised_gemm_fused integration/Test__QuantisedGemmFused.cpp)
target_link_libraries(v2_integration_quantised_gemm_fused 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_QuantisedGemmFused 
    COMMAND $<TARGET_FILE:v2_integration_quantised_gemm_fused>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;GEMM;FusedOps;Softmax;SwiGLU;Bias;Correctness"
)

# Test: Q8_1 OnlineSoftmax Kernel Correctness (Q8_1 x Q8_1 attention + fused softmax)
# Validates correctness across model sizes (Qwen 0.5B/7B/32B) and sequence lengths
add_executable(v2_integration_q8_1_online_softmax integration/Test__Q8_1_OnlineSoftmax.cpp)
target_link_libraries(v2_integration_q8_1_online_softmax 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_OnlineSoftmax 
    COMMAND $<TARGET_FILE:v2_integration_q8_1_online_softmax>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;GEMM;Attention;Q8_1;OnlineSoftmax;Correctness"
)

# Test: Q8_1 Fully Fused Attention Kernel Correctness
# Validates the fused Q8_1 attention kernel (Q@K^T + softmax + @V + requant)
# against pure FP32 reference implementation (FP32 GEMM + softmax + FP32 GEMM)
add_executable(v2_integration_q8_1_fused_attention integration/Test__Q8_1_FusedAttention.cpp)
target_link_libraries(v2_integration_q8_1_fused_attention 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_FusedAttention 
    COMMAND $<TARGET_FILE:v2_integration_q8_1_fused_attention>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;Attention;Q8_1;FusedOps;Correctness"
)

# NOTE: Q8_1 vs FP32 attention parity tests have been moved to e2e/parity/kernels/attention/
# See: V2_E2E_Parity_SimpleAttentionQ8_1_vs_FP32 and V2_E2E_Parity_JitFusedAttentionWo_vs_FP32

# Test: Q8_1 Isolated Attention Kernel Test
# Captures attention inputs from pipeline, compares FP32 vs Q8_1 kernel in isolation
# Used for debugging attention kernel divergence without full pipeline complexity

# Test: Q8_1 Attention JIT Replay Test
# Saves attention inputs to disk and replays them through the JIT kernel in isolation
# Used for reproducible debugging of the JIT attention kernel

# Test: Q8_1 Attention Replay from Captured Dump
# Loads actual Q/K/V blocks captured from E2E pipeline via LLAMINAR_DUMP_ATTENTION_INPUTS
# Compares JIT kernel output to C++ reference to isolate kernel bugs from precision loss
add_executable(v2_integration_q8_1_attention_replay_dump integration/Test__Q8_1_AttentionReplayFromDump.cpp)
target_link_libraries(v2_integration_q8_1_attention_replay_dump 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_AttentionReplayDump
    COMMAND $<TARGET_FILE:v2_integration_q8_1_attention_replay_dump>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;Attention;Q8_1;JIT;Debug;Replay;E2E"
)

# Test: HybridQ16 Attention Replay from StageDump [DISABLED - project on hold]
# Loads Q/K tensors and attention context captured from FusedAttentionWoStage via StageDumper
# Analyzes Q16 quantization error and attention output statistics
# Run V2_Integration_HybridQ16Pipeline_vs_FP32 with LLAMINAR_STAGE_DUMP_ENABLED=1 first
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_integration_hybridq16_attention_replay integration/replay/Test__HybridQ16AttentionReplay.cpp)
    target_link_libraries(v2_integration_hybridq16_attention_replay 
        llaminar2_core 
        GTest::gtest
    )
    add_v2_test(V2_Integration_HybridQ16_AttentionReplay
        COMMAND $<TARGET_FILE:v2_integration_hybridq16_attention_replay>
        MPI_PROCS 1
        LABELS "V2;Integration;Kernels;Attention;Q16;HybridQ16;Debug;Replay"
    )
endif()

# Test: OnlineSoftmax V2 Replay from StageDump [DISABLED - project on hold]
# Compares V1 (running-average) vs V2 (deferred-normalization) online softmax
# using real Q/K/V tensors captured from FusedAttentionWoStage
# Validates deferred normalization produces equivalent results to running average
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_integration_online_softmax_v2_replay integration/replay/Test__OnlineSoftmaxV2_Replay.cpp)
    target_link_libraries(v2_integration_online_softmax_v2_replay 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_OnlineSoftmaxV2_Replay
        COMMAND $<TARGET_FILE:v2_integration_online_softmax_v2_replay>
        MPI_PROCS 1
        LABELS "V2;Integration;Kernels;Attention;OnlineSoftmax;Debug;Replay"
    )
endif()

# Test: Q16 Attention Live Kernel Replay [DISABLED - project on hold]
# Calls the actual flash_decode_process_kv_block() from OnlineSoftmax.cpp
# using dumped Q16_1/Q8_1 tensors to reproduce V1 divergence issues.
# Compares V1 online softmax against FP32 reference to demonstrate the problem.
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_integration_q16_attention_live_kernel integration/replay/Test__Q16AttentionLiveKernel.cpp)
    target_link_libraries(v2_integration_q16_attention_live_kernel 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_Q16AttentionLiveKernel
        COMMAND $<TARGET_FILE:v2_integration_q16_attention_live_kernel>
        MPI_PROCS 1
        LABELS "V2;Integration;Kernels;Attention;Q16;OnlineSoftmax;Debug;Replay"
    )
endif()

# Test: Stage Dump Integrity
# Integration test that verifies stage dump infrastructure:
# - Layer filtering works correctly (only requested layers are dumped)
# - Dump file sizes match metadata byte_size
# - Block counts in metadata are correct for quantized formats
# - Metadata dimensions produce correct block counts
add_executable(v2_integration_stage_dump_integrity integration/Test__StageDumpIntegrity.cpp)
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

# Test: Q8_1 Fused Attention + Wo Projection with Tensor Parallelism (Prefill Mode)
# Validates the JIT prefill kernel with VNNI packed Wo weights under MPI tensor parallelism
# Key validations:
# - Context buffer uses local_dim (not d_model) for query stride
# - Stack allocation handles TP local dimensions correctly
# - No memory corruption from buffer overflows
add_executable(v2_integration_q8_1_fused_attention_wo_prefill_tp integration/Test__Q8_1_FusedAttentionWo_Prefill_TP.cpp)
target_link_libraries(v2_integration_q8_1_fused_attention_wo_prefill_tp 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_FusedAttentionWo_Prefill_TP
    COMMAND $<TARGET_FILE:v2_integration_q8_1_fused_attention_wo_prefill_tp>
    MPI_PROCS 2
    LABELS "V2;Integration;Kernels;Attention;Q8_1;VNNI;Prefill;TP;MPI"
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
add_executable(v2_integration_graph_batched_kvcache integration/Test__GraphBatchedKVCache.cpp)
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

# Test: Q8_1 vs FP32 Activation Precision Parity
# Validates Q8_1 activation precision produces equivalent results to FP32
# Tests Top-5 token overlap, KL divergence, and greedy sampling equivalence
# NOTE: Runs with MPI_PROCS 1 since it compares precision modes, not MPI parallelism

# Test: LayerExecutor Q8_1 vs FP32 Activation Precision Parity
# Validates Qwen2LayerExecutor produces equivalent results with Q8_1 vs FP32 activations
# Tests single layer execution using mock weights (faster than full pipeline test)
# Uses decomposed attention path (Phase 9 ComputeStage architecture)
add_executable(v2_integration_layer_executor_q8_1_vs_fp32_parity integration/Test__LayerExecutor_Q8_1_vs_FP32_Parity.cpp)
target_link_libraries(v2_integration_layer_executor_q8_1_vs_fp32_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_LayerExecutor_Q8_1_vs_FP32_Parity
    COMMAND $<TARGET_FILE:v2_integration_layer_executor_q8_1_vs_fp32_parity>
    LABELS "V2;Integration;LayerExecutor;ActivationPrecision;Q8_1;FP32;Parity;ComputeStage;Phase9"
    MPI_PROCS 1
)

# Integration: HybridQ16 Pipeline vs FP32 (Layer-by-Layer) [DISABLED - project on hold]
# -----------------------------------------------------------------------------
# Validates that HybridQ16 precision mode (Q16_1 typed residuals) achieves
# ≥0.995 cosine similarity with FP32 at residual stages, measuring improvement
# over Hybrid mode (Q8_1 residuals). Tests both prefill and decode phases.
# Requires ENABLE_PIPELINE_SNAPSHOTS for snapshot capture.
if(ENABLE_PIPELINE_SNAPSHOTS AND ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_integration_hybridq16_vs_fp32_pipeline 
        integration/Test__HybridQ16Pipeline_vs_FP32_LayerByLayer.cpp
    )
    target_link_libraries(v2_integration_hybridq16_vs_fp32_pipeline 
        llaminar2_core 
        GTest::gtest 
    )
    add_v2_test(V2_Integration_HybridQ16Pipeline_vs_FP32
        COMMAND $<TARGET_FILE:v2_integration_hybridq16_vs_fp32_pipeline>
        LABELS "V2;Integration;Parity;HybridQ16;Q16_1;FP32;LayerByLayer;Snapshots"
        MPI_PROCS 1
    )
    message(STATUS "V2: Integration HybridQ16 vs FP32 pipeline parity test enabled")
endif()

# Test: Pipeline vs LayerExecutor FP32 Parity
# Validates that the new LayerExecutor framework produces equivalent results
# to the legacy imperative pipeline code when both use FP32 precision.
# Tests prefill, single decode, and multi-step decode scenarios.

# Test: GraphBufferManagement Parity
# Validates that graph-managed buffer allocation produces equivalent results
# to pipeline-managed buffer allocation. Part of Phase 5: Qwen2Graph Integration.
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
add_executable(v2_integration_mpi_precision_allreduce integration/Test__MPI_PrecisionAwareAllreduce.cpp)
target_link_libraries(v2_integration_mpi_precision_allreduce
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_PrecisionAwareAllreduce
    COMMAND $<TARGET_FILE:v2_integration_mpi_precision_allreduce>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Allreduce;FP32;FP16;BF16;Q8_1;Q16_1;Bandwidth"
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
add_executable(v2_integration_mpi_column_parallel_qkv integration/Test__MPI_ColumnParallelQKV.cpp)
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
add_executable(v2_integration_mpi_column_parallel_ffn integration/Test__MPI_ColumnParallelFFN.cpp)
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
add_executable(v2_integration_mpi_column_parallel_lmhead integration/Test__MPI_ColumnParallelLMHead.cpp)
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
add_executable(v2_integration_mpi_sharded_kv_cache integration/Test__MPI_ShardedKVCache.cpp)
target_link_libraries(v2_integration_mpi_sharded_kv_cache
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_ShardedKVCache
    COMMAND $<TARGET_FILE:v2_integration_mpi_sharded_kv_cache>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;KVCache;Sharding;TensorParallelism;Phase6;DistributedArchitecture"
)

# ============================================================================
# Test: E2E Tensor-Parallel Inference (Capstone Integration Test)
# ============================================================================
# Full end-to-end test of the distributed inference pipeline exercising:
#   - MPITopology: Device capability exchange, work distribution
#   - WeightManager: Column/row-parallel weight sharding
#   - Qwen2Graph: Graph building with TP configuration
#   - GraphOrchestrator: Full forward execution with all stages
#   - Sharded KV Cache: Per-rank cache with local heads
#   - AllReduceStage: After Wo projection, FFN down projection
#   - AllGatherStage: After LM head for logits aggregation
add_executable(v2_integration_mpi_e2e_tp_inference integration/Test__MPI_E2E_TensorParallelInference.cpp)
target_link_libraries(v2_integration_mpi_e2e_tp_inference
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_E2E_TensorParallelInference
    COMMAND $<TARGET_FILE:v2_integration_mpi_e2e_tp_inference>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;E2E;TensorParallelism;DistributedArchitecture;Inference;GraphOrchestrator"
)

# Test: FloatingPointGemmKernel Integration Tests (thorough OneDNN FP32/BF16/FP16 GEMM tests)
# Large matrices, mixed precision, strided layouts, fused softmax, causal masking
add_executable(v2_integration_fp_gemm integration/Test__FloatingPointGemmKernel.cpp)
target_link_libraries(v2_integration_fp_gemm
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_FloatingPointGemmKernel
    COMMAND $<TARGET_FILE:v2_integration_fp_gemm>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;GEMM;FP32;BF16;FP16;OneDNN;Softmax;Attention;MixedPrecision"
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

# Performance Test: QuantisedGemmKernel (Q8_1 weights via M1/M2 JIT kernels)
add_executable(v2_perf_quantised_gemm_kernel performance/kernels/cpu/gemm/gemm_v4/Perf__QuantisedGemmKernel.cpp)
target_link_libraries(v2_perf_quantised_gemm_kernel 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_QuantisedGemmKernel 
    COMMAND $<TARGET_FILE:v2_perf_quantised_gemm_kernel>
    LABELS "V2;Performance;GEMM;Q8_1;JIT;Projection"
)

# Performance Test: QuantisedGemmKernel with Q8_1 Online Softmax (Attention kernel)
add_executable(v2_perf_quantised_gemm_kernel_q8_1_online_softmax performance/kernels/cpu/gemm/gemm_v4/Perf__QuantisedGemmKernel__Q8_1_OnlineSoftmax.cpp)
target_link_libraries(v2_perf_quantised_gemm_kernel_q8_1_online_softmax 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_QuantisedGemmKernel_Q8_1_OnlineSoftmax 
    COMMAND $<TARGET_FILE:v2_perf_quantised_gemm_kernel_q8_1_online_softmax>
    LABELS "V2;Performance;GEMM;Q8_1;JIT;OnlineSoftmax;Attention"
)

# Performance Test: QuantisedGemmKernel parameter sweep
add_executable(v2_perf_quantised_gemm_kernel_sweep performance/kernels/cpu/gemm/gemm_v4/Perf__QuantisedGemmKernel__Sweep.cpp)
target_link_libraries(v2_perf_quantised_gemm_kernel_sweep 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_QuantisedGemmKernel_Sweep 
    COMMAND $<TARGET_FILE:v2_perf_quantised_gemm_kernel_sweep>
    LABELS "V2;Performance;GEMM;Sweep"
)

# Performance Test: QuantisedGemmKernel with Q8_0 weights
add_executable(v2_perf_quantised_gemm_kernel_q8_0 performance/kernels/cpu/gemm/gemm_v4/Perf__QuantisedGemmKernel__Q8_0.cpp)
target_link_libraries(v2_perf_quantised_gemm_kernel_q8_0 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_QuantisedGemmKernel_Q8_0 
    COMMAND $<TARGET_FILE:v2_perf_quantised_gemm_kernel_q8_0>
    LABELS "V2;Performance;GEMM;Q8_0"
)

# Performance Test: QuantisedGemmKernel with Q4_0 weights
add_executable(v2_perf_quantised_gemm_kernel_q4_0 performance/kernels/cpu/gemm/gemm_v4/Perf__QuantisedGemmKernel__Q4_0.cpp)
target_link_libraries(v2_perf_quantised_gemm_kernel_q4_0 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_QuantisedGemmKernel_Q4_0 
    COMMAND $<TARGET_FILE:v2_perf_quantised_gemm_kernel_q4_0>
    LABELS "V2;Performance;GEMM;Q4_0"
)

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

# Test: IQ1_S GEMM Integration
add_executable(v2_test_iq1s_gemm_integration integration/Test__IQ1_S_GEMM_Integration.cpp)
target_link_libraries(v2_test_iq1s_gemm_integration 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ1_S_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq1s_gemm_integration>
    LABELS "V2;Integration;GEMM;IQ1_S;Quantization"
)

# Test: IQ1_M GEMM Integration
add_executable(v2_test_iq1m_gemm_integration integration/Test__IQ1_M_GEMM_Integration.cpp)
target_link_libraries(v2_test_iq1m_gemm_integration 
    llaminar2_core 
    GTest::gtest
    # No gtest_main, uses custom main
)
add_v2_test(V2_Integration_IQ1_M_GEMM 
    COMMAND $<TARGET_FILE:v2_test_iq1m_gemm_integration>
    LABELS "V2;Integration;GEMM;IQ1_M;Quantization"
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
# Compares distributed execution against single-node to ensure TP correctness
if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR CMAKE_BUILD_TYPE MATCHES "Integration" OR ENABLE_SNAPSHOTS)
    add_executable(v2_integration_parity_mpi_vs_single_rank 
        e2e/parity/internal/mpi_vs_single_rank/Test__MPIPipeline_vs_SingleRank.cpp
        e2e/parity/gtest_mpi_main.cpp
    )
    target_link_libraries(v2_integration_parity_mpi_vs_single_rank llaminar2_core GTest::gtest MPI::MPI_CXX)

    add_v2_test(V2_Integration_Parity_MPIPipeline_vs_SingleRank
        COMMAND $<TARGET_FILE:v2_integration_parity_mpi_vs_single_rank>
        LABELS "V2;Integration;Parity;MPI;TensorParallel"
        MPI_PROCS 2
    )
else()
    message(STATUS "V2: Integration MPI parity test disabled (snapshots not available in Release build)")
    message(STATUS "     Use -DCMAKE_BUILD_TYPE=Integration or -DENABLE_SNAPSHOTS=ON to enable")
endif()

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
        # -----------------------------------------------------------------------------
        # DISABLED: Legacy E2E Parity tests
        # -----------------------------------------------------------------------------
        # The following tests are disabled as part of test suite cleanup.
        # - Q8_1 vs FP32 Pipeline: Superseded by unit tests and integration stage tests
        # - Hybrid Pipeline: Project on hold (ENABLE_HYBRIDQ16_TESTS)
        # - Kernel-level attention parity: Covered by integration attention tests
        # To re-enable, set ENABLE_DEPRECATED_E2E_TESTS=ON
        if(ENABLE_DEPRECATED_E2E_TESTS)
            # E2E Parity: Q8_1 Pipeline vs FP32 Pipeline (Layer-by-Layer)
            add_executable(v2_e2e_parity_q8_1_vs_fp32_pipeline 
                e2e/parity/internal/q8_1_vs_fp32/Test__Q8_1Pipeline_vs_FP32Pipeline_LayerByLayer.cpp
            )
            target_link_libraries(v2_e2e_parity_q8_1_vs_fp32_pipeline 
                llaminar2_core 
                GTest::gtest 
            )
            add_v2_test(V2_E2E_Parity_Q8_1Pipeline_vs_FP32Pipeline
                COMMAND $<TARGET_FILE:v2_e2e_parity_q8_1_vs_fp32_pipeline>
                LABELS "V2;E2E;Parity;Internal;Q8_1;FP32;LayerByLayer"
                MPI_PROCS 1
            )
            message(STATUS "V2: E2E Q8_1 vs FP32 pipeline parity test enabled (deprecated)")
        endif()

        # E2E Parity: Hybrid Pipeline vs FP32 and Q8_1 (Layer-by-Layer) [DISABLED - project on hold]
        if(ENABLE_HYBRIDQ16_TESTS)
            add_executable(v2_e2e_parity_hybrid_vs_fp32_q8_1_pipeline 
                e2e/parity/internal/hybrid_vs_fp32_q8_1/Test__HybridPipeline_vs_FP32_Q8_1_LayerByLayer.cpp
            )
            target_link_libraries(v2_e2e_parity_hybrid_vs_fp32_q8_1_pipeline 
                llaminar2_core 
                GTest::gtest 
            )
            add_v2_test(V2_E2E_Parity_HybridPipeline_vs_FP32_Q8_1
                COMMAND $<TARGET_FILE:v2_e2e_parity_hybrid_vs_fp32_q8_1_pipeline>
                LABELS "V2;E2E;Parity;Internal;Hybrid;FP32;Q8_1;LayerByLayer"
                MPI_PROCS 1
            )
            message(STATUS "V2: E2E Hybrid vs FP32/Q8_1 pipeline parity test enabled")
        endif()
    else()
        message(STATUS "V2: Integration parity tests disabled (snapshots not available in Release build)")
        message(STATUS "     Use -DCMAKE_BUILD_TYPE=Integration or -DENABLE_SNAPSHOTS=ON to enable")
    endif()
else()
    message(WARNING "ZLIB not found - PyTorch parity tests disabled (requires cnpy for .npz loading)")
endif()

# -----------------------------------------------------------------------------
# DISABLED: E2E Parity Kernel-Level Tests (Attention)
# -----------------------------------------------------------------------------
# These tests are disabled as part of test suite cleanup.
# Kernel-level attention parity is now covered by integration tests:
# - V2_Integration_Q8_1_FusedAttention
# - V2_Integration_Q8_1_OnlineSoftmax
# To re-enable, set ENABLE_DEPRECATED_E2E_TESTS=ON

if(ENABLE_DEPRECATED_E2E_TESTS)
    # JitFusedAttentionWo (FA2 Q8_1 kernel) vs FP32 Decomposed Attention
    add_executable(v2_e2e_parity_jit_fused_attention_vs_fp32 
        e2e/parity/kernels/attention/Test__JitFusedAttentionWo_vs_FP32Attention.cpp
    )
    target_link_libraries(v2_e2e_parity_jit_fused_attention_vs_fp32 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_E2E_Parity_JitFusedAttentionWo_vs_FP32
        COMMAND $<TARGET_FILE:v2_e2e_parity_jit_fused_attention_vs_fp32>
        LABELS "V2;E2E;Parity;Kernels;Attention;Q8_1;FA2;JIT"
        MPI_PROCS 1
    )

    # QuantisedAttentionJit_Q8_1_Fused (Simple Q8_1 kernel) vs FP32 Decomposed Attention  
    add_executable(v2_e2e_parity_simple_attention_q8_1_vs_fp32 
        e2e/parity/kernels/attention/Test__SimpleAttentionQ8_1_vs_FP32Attention.cpp
    )
    target_link_libraries(v2_e2e_parity_simple_attention_q8_1_vs_fp32 
        llaminar2_core 
        GTest::gtest
    )
    add_v2_test(V2_E2E_Parity_SimpleAttentionQ8_1_vs_FP32
        COMMAND $<TARGET_FILE:v2_e2e_parity_simple_attention_q8_1_vs_fp32>
        LABELS "V2;E2E;Parity;Kernels;Attention;Q8_1;Simple"
        MPI_PROCS 1
    )
endif()

# JitFusedAttentionWo with Q16_1 Fused Residual vs FP32 Reference [DISABLED - project on hold]
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_e2e_parity_jit_fused_attention_q16_1_vs_fp32 
        e2e/parity/kernels/attention/Test__JitFusedAttentionWo_Q16_1Fused_vs_FP32.cpp
    )
    target_link_libraries(v2_e2e_parity_jit_fused_attention_q16_1_vs_fp32 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_E2E_Parity_JitFusedAttentionWo_Q16_1Fused_vs_FP32
        COMMAND $<TARGET_FILE:v2_e2e_parity_jit_fused_attention_q16_1_vs_fp32>
        LABELS "V2;E2E;Parity;Kernels;Attention;Q16_1;FusedResidual;JIT"
        MPI_PROCS 1
    )
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

# Performance Test: QuantisedGemmKernel Packing (constructor time)
add_executable(v2_perf_quantised_gemm_packing performance/kernels/Perf__QuantisedGemmPacking.cpp)
target_link_libraries(v2_perf_quantised_gemm_packing 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_QuantisedGemmPacking 
    COMMAND $<TARGET_FILE:v2_perf_quantised_gemm_packing>
    LABELS "V2;Performance;Kernels;GEMM;Packing"
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

# Performance Test: Q16→Q16 Fixed-Scale RoPE SIMD
add_executable(v2_perf_q16_to_q16_rope_fixed_scale performance/kernels/cpu/primitives/Perf__Q16_to_Q16_RoPE_FixedScale.cpp)
target_link_libraries(v2_perf_q16_to_q16_rope_fixed_scale 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Perf_Q16_to_Q16_RoPE_FixedScale 
    COMMAND $<TARGET_FILE:v2_perf_q16_to_q16_rope_fixed_scale>
    LABELS "V2;Performance;RoPE;Q16_1;FixedScale;SIMD;AVX2;AVX512;HybridQ16;Primitives"
    MPI_PROCS 1  # CPU-only perf test, single rank for accurate benchmarking
)

# =============================================================================
# Microkernels for Fused Attention + Wo Projection
# =============================================================================
# These are composable, independently-testable units for the fused attention kernel.
# See docs/v2/FUSED_ATTENTION_WO_PROJECT_PLAN.md for design documentation.

# Test: μK1 - Q8_1 Dot Product Microkernel
add_executable(v2_test_microkernel_q8_dot_product unit/microkernels/Test__Q8DotProduct.cpp)
target_link_libraries(v2_test_microkernel_q8_dot_product 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_Q8DotProduct 
    COMMAND $<TARGET_FILE:v2_test_microkernel_q8_dot_product>
    LABELS "V2;Unit;Microkernels;Q8_1;DotProduct;VNNI;AVX512;FusedAttention"
    MPI_PROCS 1
)

# Test: μK2 - Online Softmax Microkernel
add_executable(v2_test_microkernel_online_softmax unit/microkernels/Test__OnlineSoftmax.cpp)
target_link_libraries(v2_test_microkernel_online_softmax 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_OnlineSoftmax 
    COMMAND $<TARGET_FILE:v2_test_microkernel_online_softmax>
    LABELS "V2;Unit;Microkernels;Softmax;Online;StreamingSoftmax;FlashAttention;FusedAttention"
    MPI_PROCS 1
)

# Test: μK2-V2 - Online Softmax V2 (Deferred Normalization) Unit Tests
# Proves mathematical correctness of V2 implementation before integration:
# - __int128 rescale correctness
# - Chunked accumulation overflow safety
# - Deferred normalization equivalence to running average
# - Weight sum tracking exactness
add_executable(v2_test_microkernel_online_softmax_v2 unit/microkernels/Test__OnlineSoftmaxV2_Unit.cpp)
target_link_libraries(v2_test_microkernel_online_softmax_v2 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_OnlineSoftmaxV2 
    COMMAND $<TARGET_FILE:v2_test_microkernel_online_softmax_v2>
    LABELS "V2;Unit;Microkernels;Softmax;Online;DeferredNorm;VNNI;Integer;FusedAttention"
    MPI_PROCS 1
)

# Test: μK2b - Q16 Online Softmax K-Scales Microkernel (HybridQ16 K precision fix)
add_executable(v2_test_microkernel_q16_online_softmax_kscales unit/microkernels/Test__Q16OnlineSoftmaxKScales.cpp)
target_link_libraries(v2_test_microkernel_q16_online_softmax_kscales 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_Q16OnlineSoftmaxKScales 
    COMMAND $<TARGET_FILE:v2_test_microkernel_q16_online_softmax_kscales>
    LABELS "V2;Unit;Microkernels;Softmax;Q16;KScales;HybridQ16;FusedAttention"
    MPI_PROCS 1
)

# Test: Q16 Attention with Realistic Pipeline Data
# Tests attention kernel using actual data captured from pipeline stage dumps.
# Exposes the K precision bug where K values exceed representable range (max_abs ~130 > 128).
add_executable(v2_test_microkernel_q16_attention_realistic unit/microkernels/Test__Q16AttentionRealisticData.cpp)
target_link_libraries(v2_test_microkernel_q16_attention_realistic 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_Q16AttentionRealistic 
    COMMAND $<TARGET_FILE:v2_test_microkernel_q16_attention_realistic>
    LABELS "V2;Unit;Microkernels;Q16;Attention;HybridQ16;Realistic;FusedAttention"
    MPI_PROCS 1
)

# Test: OnlineSoftmax Per-Position K Scales Bug Reproduction
# Unit tests to isolate and reproduce the HybridQ16 OnlineSoftmax per-position K scale bug.
# The tests demonstrate that mixing different alpha values in the scaled score domain
# leads to incorrect softmax weights because the domains are incompatible.
add_executable(v2_test_microkernel_online_softmax_k_scales unit/microkernels/Test__OnlineSoftmaxPerPositionKScales.cpp)
target_link_libraries(v2_test_microkernel_online_softmax_k_scales 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_OnlineSoftmaxKScales 
    COMMAND $<TARGET_FILE:v2_test_microkernel_online_softmax_k_scales>
    LABELS "V2;Unit;Microkernels;Q16;OnlineSoftmax;HybridQ16;KScales;FusedAttention;BugRepro"
    MPI_PROCS 1
)

# Test: μK3 - V Weighted Accumulation Microkernel
add_executable(v2_test_microkernel_v_weighted_accum unit/microkernels/Test__VWeightedAccum.cpp)
target_link_libraries(v2_test_microkernel_v_weighted_accum 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_VWeightedAccum 
    COMMAND $<TARGET_FILE:v2_test_microkernel_v_weighted_accum>
    LABELS "V2;Unit;Microkernels;Attention;Accumulation;AVX512;Q8_1;FusedAttention"
    MPI_PROCS 1
)

# Test: μK4 - Wo Projection Microkernel
add_executable(v2_test_microkernel_wo_projection unit/microkernels/Test__WoProjection.cpp)
target_link_libraries(v2_test_microkernel_wo_projection 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_WoProjection 
    COMMAND $<TARGET_FILE:v2_test_microkernel_wo_projection>
    LABELS "V2;Unit;Microkernels;Projection;GEMV;Wo;FP32;Q8_1;AVX512;FusedAttention"
    MPI_PROCS 1
)

# Test: μK5 - Fast Exp Microkernel
add_executable(v2_test_microkernel_fast_exp unit/microkernels/Test__FastExp.cpp)
target_link_libraries(v2_test_microkernel_fast_exp 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_FastExp 
    COMMAND $<TARGET_FILE:v2_test_microkernel_fast_exp>
    LABELS "V2;Unit;Microkernels;FastExp;Softmax;Polynomial;AVX512;FusedAttention"
    MPI_PROCS 1
)

# =============================================================================
# Q16_1 Integer Attention Microkernels (TRUE INTEGER PIPELINE v2)
# =============================================================================
# These are the integer-only microkernels for the Q16_1 attention path.
# See docs/v2/PROJECT_Q16_INTEGER_ATTENTION.md for design documentation.

# Test: Q16 μK1 - Q16 Dot Product Microkernel (Q×K^T)
add_executable(v2_test_microkernel_q16_dot_product unit/microkernels/Test__Q16DotProduct.cpp)
target_link_libraries(v2_test_microkernel_q16_dot_product 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_Q16DotProduct 
    COMMAND $<TARGET_FILE:v2_test_microkernel_q16_dot_product>
    LABELS "V2;Unit;Microkernels;Q16_1;DotProduct;Integer;VNNI;FusedAttention"
    MPI_PROCS 1
)

# Test: Q16 μK2 - P×V Accumulation Microkernel
add_executable(v2_test_microkernel_pv_accumulate unit/microkernels/Test__PVAccumulate.cpp)
target_link_libraries(v2_test_microkernel_pv_accumulate 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_PVAccumulate 
    COMMAND $<TARGET_FILE:v2_test_microkernel_pv_accumulate>
    LABELS "V2;Unit;Microkernels;Q16_1;PVAccumulate;Integer;VNNI;FusedAttention"
    MPI_PROCS 1
)

# Test: Q16 μK3 - Wo Projection Microkernel (INT32 → Q16_1)
add_executable(v2_test_microkernel_q16_wo_projection unit/microkernels/Test__Q16WoProjection.cpp)
target_link_libraries(v2_test_microkernel_q16_wo_projection 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Microkernel_Q16WoProjection 
    COMMAND $<TARGET_FILE:v2_test_microkernel_q16_wo_projection>
    LABELS "V2;Unit;Microkernels;Q16_1;WoProjection;Integer;VNNI;FusedAttention"
    MPI_PROCS 1
)

# =============================================================================
# Phase 2: Fused Attention + Wo Reference Kernel (Composed from Microkernels)
# =============================================================================

# Test: FusedAttentionWoRef - Reference implementation using all microkernels
add_executable(v2_test_fused_attention_wo_ref integration/Test__FusedAttentionWoRef.cpp)
target_link_libraries(v2_test_fused_attention_wo_ref 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_FusedAttentionWoRef 
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_ref>
    LABELS "V2;Integration;FusedAttention;Q8_1;Wo;GEMM;GQA;Causal;KVCache"
    MPI_PROCS 1
)

# Test: FusedAttentionWoRef Robustness - Edge cases from real pipeline issues
# Tests for: softmax saturation, catastrophic cancellation, quantization clipping,
#            scale disparity, sign flips, residual collapse, extreme outliers
add_executable(v2_test_fused_attention_wo_robustness integration/Test__FusedAttentionWoRef_Robustness.cpp)
target_link_libraries(v2_test_fused_attention_wo_robustness 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_FusedAttentionWo_Robustness 
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_robustness>
    LABELS "V2;Integration;FusedAttention;Robustness;Saturation;Quantization;EdgeCases"
    MPI_PROCS 1
)

# Test: FusedAttentionWoRef Batch - Batched execution modes
# Tests for: prefill, incremental decode, batched decode uniform/variable KV
add_executable(v2_test_fused_attention_wo_batch integration/Test__FusedAttentionWoRef_Batch.cpp)
target_link_libraries(v2_test_fused_attention_wo_batch 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_FusedAttentionWo_Batch 
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_batch>
    LABELS "V2;Integration;FusedAttention;Batch;Prefill;Decode;VariableKV"
    MPI_PROCS 1
)

# Test: FusedAttentionWoTiled - Cache-blocked tiled implementation (Phase 3)
# Tests for: tile size computation, tiled vs reference parity, various configs
add_executable(v2_test_fused_attention_wo_tiled unit/attention/Test__FusedAttentionWoTiled.cpp)
target_link_libraries(v2_test_fused_attention_wo_tiled 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedAttentionWo_Tiled 
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_tiled>
    LABELS "V2;Unit;FusedAttention;Tiled;CacheBlocking;FlashAttention;Parity"
    MPI_PROCS 1
)


# Test: JIT Microkernels - Modular JIT attention microkernels (Phase 4)
# Tests for: JitFastExp, JitQ8DotProduct, JitOnlineSoftmax, JitVWeightedAccum, JitWoProjection
add_executable(v2_test_jit_microkernels unit/attention/Test__JitMicrokernels.cpp)
target_link_libraries(v2_test_jit_microkernels 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitMicrokernels 
    COMMAND $<TARGET_FILE:v2_test_jit_microkernels>
    LABELS "V2;Unit;JIT;Microkernels;FusedAttention;AVX512;Xbyak;FastExp;Q8DotProduct;OnlineSoftmax"
    MPI_PROCS 1
)

# Test: JIT Fused Attention + Wo - Composed JIT kernel (Phase 4)
# Tests the complete fused attention + Wo projection JIT kernel
add_executable(v2_test_jit_fused_attention_wo unit/attention/Test__JitFusedAttentionWo.cpp)
target_link_libraries(v2_test_jit_fused_attention_wo 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitFusedAttentionWo 
    COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_wo>
    LABELS "V2;Unit;JIT;FusedAttention;WoProjection;AVX512;Xbyak;Q8_1"
    MPI_PROCS 1
)

# Test: JIT Fused Attention + Wo - Numerical Correctness Tests
# Comprehensive tests comparing JIT kernel against reference implementation
# Covers Qwen 0.5B, 7B, 32B model configurations with various sequence lengths
add_executable(v2_test_jit_fused_attention_wo_correctness unit/attention/Test__JitFusedAttentionWo_Correctness.cpp)
target_link_libraries(v2_test_jit_fused_attention_wo_correctness 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitFusedAttentionWo_Correctness 
    COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_wo_correctness>
    LABELS "V2;Unit;JIT;FusedAttention;Correctness;AVX512;Q8_1;NumericalParity"
    MPI_PROCS 1
)

# Test: JIT Fused Attention + Wo - Q16_1 Residual Fusion Tests [DISABLED - project on hold]
# Tests the fused residual addition path for Q16_1 typed residual streams
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_jit_fused_attention_wo_q16_1_residual unit/attention/Test__JitFusedAttentionWo_Q16_1Residual.cpp)
    target_link_libraries(v2_test_jit_fused_attention_wo_q16_1_residual 
        llaminar2_core
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_JitFusedAttentionWo_Q16_1Residual 
        COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_wo_q16_1_residual>
        LABELS "V2;Unit;JIT;FusedAttention;Q16_1;Residual;TypedResiduals"
        MPI_PROCS 1
    )
endif()

# Test: JIT Fused Attention Manual Verification
# Tests JIT kernel against MANUALLY COMPUTED ground truth (not C++ reference).
# This catches bugs where both JIT and reference have the same error.
add_executable(v2_test_jit_fused_attention_manual_verify unit/attention/Test__JitFusedAttentionWo_ManualVerification.cpp)
target_link_libraries(v2_test_jit_fused_attention_manual_verify 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitFusedAttention_ManualVerify
    COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_manual_verify>
    LABELS "V2;Unit;JIT;FusedAttention;ManualVerification;Q8_1;GroundTruth"
    MPI_PROCS 1
)

# Test: JIT Wo Projection Isolated
# Tests the Wo projection logic in isolation to debug 1.0 output issue
add_executable(v2_test_jit_wo_projection_isolated unit/attention/Test__JitWoProjectionIsolated.cpp)
target_link_libraries(v2_test_jit_wo_projection_isolated 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitWoProjectionIsolated
    COMMAND $<TARGET_FILE:v2_test_jit_wo_projection_isolated>
    LABELS "V2;Unit;JIT;WoProjection;Isolated"
    MPI_PROCS 1
)

# Test: JIT Fused Attention + Wo - Prefill Mode Tests
# Tests the prefill kernel (H→KV→Q loop order) for K/V cache reuse optimization
# Validates prefill vs decode parity, tile boundary handling, causal masking
add_executable(v2_test_jit_fused_attention_wo_prefill unit/attention/Test__JitFusedAttentionWo_Prefill.cpp)
target_link_libraries(v2_test_jit_fused_attention_wo_prefill 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitFusedAttentionWo_Prefill 
    COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_wo_prefill>
    LABELS "V2;Unit;JIT;FusedAttention;Prefill;AVX512;Q8_1;KVCacheReuse"
    MPI_PROCS 1
)

# Debug test for prefill alternating zeros issue
add_executable(v2_test_jit_fused_attention_wo_debug unit/attention/Test__JitFusedAttentionWo_Debug.cpp)
target_link_libraries(v2_test_jit_fused_attention_wo_debug 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitFusedAttentionWo_Debug 
    COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_wo_debug>
    LABELS "V2;Unit;JIT;FusedAttention;Debug"
    MPI_PROCS 1
)

# Test: JIT Fused Attention + Wo - FA2 vs FA1 Parity Tests
# Tests that FA2 tiling (use_fa2_tiling=true) produces identical output to FA1
# Validates tile boundary handling, remainder processing, various KV lengths
add_executable(v2_test_jit_fused_attention_wo_fa2parity unit/attention/Test__JitFusedAttentionWo_FA2Parity.cpp)
target_link_libraries(v2_test_jit_fused_attention_wo_fa2parity 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitFusedAttentionWo_FA2Parity 
    COMMAND $<TARGET_FILE:v2_test_jit_fused_attention_wo_fa2parity>
    LABELS "V2;Unit;JIT;FusedAttention;FA2;Parity;AVX512;Q8_1;FlashAttention"
    MPI_PROCS 1
)

# Test: JIT Wo Projection - Tests the Wo projection implementation in JIT kernel
# Validates FP32, FP16, BF16, Q8_1 Wo formats against REFERENCE backend
add_executable(v2_test_jit_wo_projection unit/attention/Test__JitWoProjection.cpp)
target_link_libraries(v2_test_jit_wo_projection 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Unit_JitWoProjection 
    COMMAND $<TARGET_FILE:v2_test_jit_wo_projection>
    LABELS "V2;Unit;JIT;WoProjection;FusedAttention;AVX512;Q8_1;FP16;BF16;NumericalParity"
    MPI_PROCS 1
)

# Q16 INT16 VNNI Wo Projection Unit Tests
# Tests the Wo projection implementation in Q16IntegerAttentionRef:
# - wo_projection_vnni_int16() for decode (single token GEMV)
# - wo_projection_vnni_int16_batched() for prefill (batched GEMM)
# - FP32 reference comparison for correctness validation
add_executable(v2_test_q16_wo_projection unit/attention/Test__Q16WoProjection.cpp)
target_link_libraries(v2_test_q16_wo_projection
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16WoProjection
    COMMAND $<TARGET_FILE:v2_test_q16_wo_projection>
    LABELS "V2;Unit;Q16;WoProjection;VNNI;INT16;NumericalParity"
    MPI_PROCS 1
)

# Q16 Attention Output Composition Test (Level 2 Macrokernel)
# Tests the FULL attention output path: Context → Wo Projection → Residual Add → Q16_1
# This is the bridge between microkernel unit tests and full pipeline integration tests.
# See: docs/v2/PLAN_FIXED_SCALE_ROPE_Q16.md "Addendum 2"
add_executable(v2_test_q16_attention_output_composition unit/attention/Test__Q16AttentionOutput_Composition.cpp)
target_link_libraries(v2_test_q16_attention_output_composition
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16AttentionOutput_Composition
    COMMAND $<TARGET_FILE:v2_test_q16_attention_output_composition>
    LABELS "V2;Unit;Q16;WoProjection;ResidualAdd;Composition;Macrokernel;NumericalParity"
    MPI_PROCS 1
)

# =============================================================================
# Phase 5: Pipeline Integration - FusedAttentionWoKernel wrapper
# =============================================================================

# Test: FusedAttentionWoKernel - Pipeline-compatible wrapper for fused attention
# Tests all backends (Reference, Tiled, JIT) and verifies cross-backend parity
add_executable(v2_test_fused_attention_wo_kernel integration/attention/Test__FusedAttentionWoKernel.cpp)
target_link_libraries(v2_test_fused_attention_wo_kernel 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Integration_FusedAttentionWoKernel 
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_kernel>
    LABELS "V2;Integration;FusedAttention;PipelineWrapper;JIT;Tiled;Reference;Q8_1"
    MPI_PROCS 1
)

# =============================================================================
# Phase 8.2: Performance Benchmarking - Fused Attention + Wo
# =============================================================================
# Compares fused JIT attention+Wo vs unfused Reference attention+Wo
# Measures: prefill throughput, decode latency, speedup percentages

add_executable(v2_perf_fused_attention_wo performance/kernels/cpu/attention/Perf__FusedAttentionWo.cpp)
target_link_libraries(v2_perf_fused_attention_wo 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Perf_FusedAttentionWo 
    COMMAND $<TARGET_FILE:v2_perf_fused_attention_wo>
    LABELS "V2;Performance;FusedAttention;JIT;Reference;Tiled;Prefill;Decode;Throughput"
    MPI_PROCS 1
)

# Standalone microbench (no gtest) - intended for perf stat cache profiling
add_executable(v2_microbench_fused_attention_wo performance/kernels/cpu/attention/Microbench__FusedAttentionWo.cpp)
target_link_libraries(v2_microbench_fused_attention_wo
    llaminar2_core
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
# Phase 8.2b: Performance Benchmarking - JIT Microkernels
# =============================================================================
# Benchmarks individual JIT microkernels in isolation

add_executable(v2_perf_jit_microkernels performance/kernels/cpu/attention/Perf__JitMicrokernels.cpp)
target_link_libraries(v2_perf_jit_microkernels 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Perf_JitMicrokernels 
    COMMAND $<TARGET_FILE:v2_perf_jit_microkernels>
    LABELS "V2;Performance;JIT;Microkernels;Q8_1"
    MPI_PROCS 1
)

add_executable(v2_perf_fused_attention_wo_tuning performance/kernels/cpu/attention/Perf__FusedAttentionWo_Tuning.cpp)
target_link_libraries(v2_perf_fused_attention_wo_tuning 
    llaminar2_core
    GTest::gtest 
    GTest::gtest_main
)
add_v2_test(V2_Perf_FusedAttentionWo_Tuning 
    COMMAND $<TARGET_FILE:v2_perf_fused_attention_wo_tuning>
    LABELS "V2;Performance;FusedAttention;Tuning;Qwen"
    MPI_PROCS 1
)

# =============================================================================
# Multi-Rank MPI Integration Tests for Distributed Architecture (Phase 1, 2, G0)
# =============================================================================

# Test: MPITopology Multi-Rank Integration Tests (Phase 1)
# Tests work distribution, capability exchange, and placement determinism
# across multiple MPI ranks.
# NOTE: These tests have custom main() with MPI_Init/MPI_Finalize
add_executable(v2_integration_mpi_topology integration/Test__MPITopologyIntegration.cpp)
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
# using the new KernelFactory::getOrCreateGemmSliced() API.
add_executable(v2_integration_sliced_gemm_mpi integration/Test__SlicedGEMM_MPI_Integration.cpp)
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
add_executable(v2_integration_placement_strategy_mpi integration/Test__PlacementStrategy_MPI_Integration.cpp)
target_link_libraries(v2_integration_placement_strategy_mpi
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_PlacementStrategy_MPI
    COMMAND $<TARGET_FILE:v2_integration_placement_strategy_mpi>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;PlacementStrategy;Determinism;CPUOnly;GPUFirst;PhaseG0"
)


# Test: Qwen2Graph Fused Attention Integration
# Verifies that the graph builder correctly switches between fused and decomposed paths
add_executable(v2_test_qwen2_graph_fused_attention integration/Test__Qwen2Graph_FusedAttention.cpp)
target_link_libraries(v2_test_qwen2_graph_fused_attention 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Qwen2Graph_FusedAttention 
    COMMAND $<TARGET_FILE:v2_test_qwen2_graph_fused_attention>
    LABELS "V2;Integration;Graph;Attention;FusedAttention"
)

# Test: Graph Snapshot Callback Invocation Integration Tests
# Verifies that snapshot callbacks are properly invoked for ALL stages during graph execution.
# Critical for E2E parity testing and debugging - ensures no stages silently skip callbacks.
# Tests:
#   - Basic stage types (RMSNorm, SwiGLU, ResidualAdd)
#   - Multi-stage graphs
#   - Full Qwen2 graph with GraphOrchestrator
#   - FusedAttentionWo context/output capture
#   - Decode phase callback invocation
add_executable(v2_test_graph_snapshot_callback_invocation integration/Test__GraphSnapshotCallbackInvocation.cpp)
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
# =============================================================================
# Q16 Integer Attention v2 Tests
# =============================================================================

# Test: Exp2FixedSoftmax microkernel
# Integer-only softmax using exp2 LUT approximation (2048 entries)
# Produces INT16 weights in [0, 32767] for VNNI compatibility
add_executable(v2_test_exp2_fixed_softmax unit/kernels/cpu/q16_attention/Test__Exp2FixedSoftmax.cpp)
target_link_libraries(v2_test_exp2_fixed_softmax
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Exp2FixedSoftmax
    COMMAND $<TARGET_FILE:v2_test_exp2_fixed_softmax>
    LABELS "V2;Unit;Kernels;Attention;Q16;Softmax;Integer"
)

# Test: Exp2Core primitives
# Core exp2 LUT primitives shared by Exp2FixedSoftmax and OnlineSoftmax
# Tests LUT initialization, AdaptiveAlphaConfig, weight/rescale computation
add_executable(v2_test_exp2_core unit/kernels/cpu/q16_attention/Test__Exp2Core.cpp)
target_link_libraries(v2_test_exp2_core
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Exp2Core
    COMMAND $<TARGET_FILE:v2_test_exp2_core>
    LABELS "V2;Unit;Kernels;Attention;Q16;Exp2;Primitives;Integer"
)

# Test: Q16 OnlineSoftmax microkernel (FlashDecode)
# Block-wise online softmax with rescaling for streaming KV processing
add_executable(v2_test_q16_online_softmax unit/kernels/cpu/q16_attention/Test__Q16OnlineSoftmax.cpp)
target_link_libraries(v2_test_q16_online_softmax
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16OnlineSoftmax
    COMMAND $<TARGET_FILE:v2_test_q16_online_softmax>
    LABELS "V2;Unit;Kernels;Attention;Q16;OnlineSoftmax;FlashDecode;Integer"
)

# Test: Q16 Integer Attention (Full Kernel) [DISABLED - project on hold]
# Complete attention kernel including Flash Decode and FA2 Prefill paths
# Tests parameter validation, snapshot capture, and multi-head/GQA configurations
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_q16_integer_attention unit/kernels/cpu/q16_attention/Test__Q16IntegerAttention.cpp)
    target_link_libraries(v2_test_q16_integer_attention
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Q16IntegerAttention
        COMMAND $<TARGET_FILE:v2_test_q16_integer_attention>
        LABELS "V2;Unit;Kernels;Attention;Q16;Integer;FlashDecode;FA2;Prefill"
        MPI_PROCS 1
    )
endif()

# Test: Q16 Integer Attention Parity (vs FP32 Reference) [DISABLED - project on hold]
# Compares Q16 attention output against FP32 reference implementation
# Validates cosine similarity, RMSE, and softmax weight normalization
# NOTE: This is an integration test due to long runtime (~270s)
if(ENABLE_HYBRIDQ16_TESTS)
    add_executable(v2_test_q16_integer_attention_parity unit/kernels/cpu/q16_attention/Test__Q16IntegerAttentionParity.cpp)
    target_link_libraries(v2_test_q16_integer_attention_parity
        llaminar2_core
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_Q16IntegerAttentionParity
        COMMAND $<TARGET_FILE:v2_test_q16_integer_attention_parity>
        LABELS "V2;Integration;Kernels;Attention;Q16;Parity;FP32;Cosine;RMSE"
        MPI_PROCS 1
    )
endif()
# Test: MLA (Multi-head Latent Attention) Microkernels - DeepSeek V3/Kimi K2
# Split NOPE/ROPE Q×K^T with combined scoring for latent attention architectures
add_executable(v2_test_mla_attention unit/microkernels/Test__MLAAttention.cpp)
target_link_libraries(v2_test_mla_attention
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MLAAttention
    COMMAND $<TARGET_FILE:v2_test_mla_attention>
    LABELS "V2;Unit;Microkernels;Q16_1;MLA;DeepSeekV3;KimiK2;NOPE;ROPE;Integer"
    MPI_PROCS 1
)

# Test: Q16 Fixed-Scale Quantization Accuracy
# Measures precision impact of VNNI-safe fixed-scale vs adaptive quantization
add_executable(v2_test_q16_fixed_scale_accuracy unit/microkernels/Test__Q16FixedScaleAccuracy.cpp)
target_link_libraries(v2_test_q16_fixed_scale_accuracy
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16FixedScaleAccuracy
    COMMAND $<TARGET_FILE:v2_test_q16_fixed_scale_accuracy>
    LABELS "V2;Unit;Microkernels;Q16_1;Quantization;Accuracy;VNNI"
    MPI_PROCS 1
)

# Test: VNNI Overflow Analysis
# Comprehensive analysis of INT32 overflow risks in Q×K^T and P×V paths
add_executable(v2_test_vnni_overflow_analysis unit/microkernels/Test__VNNIOverflowAnalysis.cpp)
target_link_libraries(v2_test_vnni_overflow_analysis
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_VNNIOverflowAnalysis
    COMMAND $<TARGET_FILE:v2_test_vnni_overflow_analysis>
    LABELS "V2;Unit;Microkernels;Q16_1;VNNI;Overflow;Analysis"
    MPI_PROCS 1
)

# Test: Q16 Fixed-Scale Quantization with VNNI-safe clipping
# Tests copyFrom_fp32_fixed_scale() and copyFrom_fp32_rows_fixed_scale()
add_executable(v2_test_q16_fixed_scale_quantization unit/kernels/Test__Q16FixedScaleQuantization.cpp)
target_link_libraries(v2_test_q16_fixed_scale_quantization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16FixedScaleQuantization
    COMMAND $<TARGET_FILE:v2_test_q16_fixed_scale_quantization>
    LABELS "V2;Unit;Kernels;Q16_1;Quantization;VNNI;Safety"
    MPI_PROCS 1
)

# =============================================================================
# ROCm Quantised GEMM Kernel Tests (ComposableKernel INT8)
# =============================================================================
if(HAVE_ROCM)
    # Test: ROCm Quantised GEMM Kernel (INT8×INT8→INT32 via ComposableKernel)
    # Tests weight packing, activation quantization, CK GEMM, and output scaling
    add_executable(v2_unit_rocm_quantised_gemm_kernel 
        unit/kernels/rocm/Test__ROCmQuantisedGemmKernel.cpp
    )
    target_link_libraries(v2_unit_rocm_quantised_gemm_kernel 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_ROCmQuantisedGemmKernel
        COMMAND $<TARGET_FILE:v2_unit_rocm_quantised_gemm_kernel>
        LABELS "V2;Unit;ROCm;GEMM;INT8;Quantized;ComposableKernel;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Quantised GEMM kernel unit tests")

    # Integration Test: ROCm Quantised GEMM Kernel Pipeline
    # Tests full pipeline: weight packing + activation quantization + realistic model dims
    add_executable(v2_integration_rocm_quantised_gemm_kernel 
        integration/Test__ROCmQuantisedGemmKernel.cpp
    )
    target_link_libraries(v2_integration_rocm_quantised_gemm_kernel 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Integration_ROCmQuantisedGemmKernel
        COMMAND $<TARGET_FILE:v2_integration_rocm_quantised_gemm_kernel>
        LABELS "V2;Integration;ROCm;GEMM;INT8;Quantized;ComposableKernel;gfx906;Qwen2"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Quantised GEMM kernel integration tests")

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
        LABELS "V2;Performance;ROCm;GEMM;INT8;Quantized;ComposableKernel;gfx906;Qwen2;Benchmark"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Quantised GEMM kernel performance benchmarks")

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

    # Unit Test: ROCm Ring Buffer KV Cache
    # Tests append, wrap-around, eviction, linearization, FP32/FP16 precision
    add_executable(v2_unit_rocm_ring_kv_cache 
        unit/Test__ROCmRingKVCache.cpp
    )
    target_link_libraries(v2_unit_rocm_ring_kv_cache 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_ROCmRingKVCache
        COMMAND $<TARGET_FILE:v2_unit_rocm_ring_kv_cache>
        LABELS "V2;Unit;ROCm;KVCache;RingBuffer;Eviction;SlidingWindow;FP32;FP16;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Ring KV Cache unit tests")

    # Unit Test: ROCm Ring Buffer KV Cache Sharding
    # Tests sharded KV cache for tensor parallelism (stub tests until sharding implemented)
    add_executable(v2_unit_rocm_ring_kv_cache_sharding 
        unit/Test__ROCmRingKVCache_Sharding.cpp
    )
    target_link_libraries(v2_unit_rocm_ring_kv_cache_sharding 
        llaminar2_core 
        GTest::gtest 
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_ROCmRingKVCache_Sharding
        COMMAND $<TARGET_FILE:v2_unit_rocm_ring_kv_cache_sharding>
        LABELS "V2;Unit;ROCm;KVCache;RingBuffer;Sharding;TensorParallel;gfx906"
        MPI_PROCS 1
    )
    message(STATUS "V2 Tests: Added ROCm Ring KV Cache sharding unit tests")

    # =============================================================================
    # ROCm SwiGLU Parity Tests
    # =============================================================================

    # Integration Test: ROCm SwiGLU Parity
    # Tests ROCm SwiGLU kernels (FP32, BF16, FP16) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_swiglu_parity 
        integration/Test__ROCmSwiGLUParity.cpp
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
        integration/Test__ROCmRoPEParity.cpp
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
        integration/Test__ROCmRMSNormParity.cpp
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
        integration/Test__ROCmEmbeddingParity.cpp
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
        integration/Test__ROCmFlashAttentionParity.cpp
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
    message(STATUS "V2 Tests: Added ROCm Flash Attention parity tests")

    # ROCm ResidualAdd Parity Tests
    # =============================================================================

    # Integration Test: ROCm ResidualAdd Parity
    # Tests ROCm ResidualAdd kernels (FP32, BF16, FP16) against CPU reference
    # with cosine similarity and L2 error metrics
    add_executable(v2_integration_rocm_residual_add_parity 
        integration/Test__ROCmResidualAddParity.cpp
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
endif()