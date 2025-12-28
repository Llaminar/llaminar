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
    if(CMAKE_BUILD_TYPE MATCHES "Debug" AND ("${ARG_LABELS}" MATCHES "E2E" OR "${ARG_LABELS}" MATCHES "Integration"))
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
    
    # Set environment variables for optimal performance + logging
    # Working directory: project root (so model paths like "models/..." work)
    # CMAKE_SOURCE_DIR is /workspaces/llaminar/src/v2, so go up two levels to reach project root
    set_tests_properties(${TEST_NAME} PROPERTIES
        FIXTURES_REQUIRED V2_Models                    # Requires V2_Models fixture
        LABELS "${ARG_LABELS}"
        ENVIRONMENT "LLAMINAR_LOG_LEVEL=${DEFAULT_LOG_LEVEL};HWLOC_COMPONENTS=-gl,-opencl;OMP_NUM_THREADS=${CORES_PER_SOCKET};OMP_PLACES=sockets;OMP_PROC_BIND=close;OMP_NESTED=false;OMP_DYNAMIC=false;KMP_AFFINITY=granularity=fine,compact,1,0;KMP_BLOCKTIME=0;OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET};GOTO_NUM_THREADS=${CORES_PER_SOCKET};MKL_NUM_THREADS=${CORES_PER_SOCKET};MKL_DYNAMIC=false;OMPI_MCA_mpi_leave_pinned=1;OMPI_MCA_btl_vader_single_copy_mechanism=none;OMPI_MCA_btl_openib_allow_ib=1"
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

# Enable pipeline snapshot capture for test builds (matches core library)
# This compile definition enables the snapshot infrastructure in PipelineBase.
# In release builds (without this flag), all snapshot code compiles away to NOOPs.
# CRITICAL: Must match core library definition to avoid ODR violations!
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
elseif(CMAKE_BUILD_TYPE MATCHES "E2ERelease")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
elseif(ENABLE_SNAPSHOTS)
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
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

# Test: Q16 Microkernels (DotProduct, PVAccumulate, WoProjectionVNNI)
# Note: Integer softmax tested separately via Exp2FixedSoftmax
add_executable(v2_test_q16_microkernels unit/Test__Q16Microkernels.cpp)
target_link_libraries(v2_test_q16_microkernels
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16Microkernels
    COMMAND $<TARGET_FILE:v2_test_q16_microkernels>
    LABELS "V2;Unit;Attention;Q16;Microkernels;DotProduct;WoProjection;VPDPWSSD"
    MPI_PROCS 1
)

# Test: Q16 Integer-Domain Attention Parity vs FP32 Reference
add_executable(v2_test_q16_integer_domain_parity unit/Test__Q16IntegerDomainParity.cpp)
target_link_libraries(v2_test_q16_integer_domain_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16IntegerDomainParity
    COMMAND $<TARGET_FILE:v2_test_q16_integer_domain_parity>
    LABELS "V2;Unit;Attention;Q16;IntegerDomain;Parity;FP32Reference;Exp2Softmax"
    MPI_PROCS 1
)

# Test: Q16 Integer-Domain + VPDPWSSD Wo Projection Parity vs FP32 Reference
add_executable(v2_test_q16_integer_domain_wo_projection unit/Test__Q16IntegerDomainWoProjection.cpp)
target_link_libraries(v2_test_q16_integer_domain_wo_projection
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16IntegerDomainWoProjection
    COMMAND $<TARGET_FILE:v2_test_q16_integer_domain_wo_projection>
    LABELS "V2;Unit;Attention;Q16;IntegerDomain;WoProjection;VPDPWSSD;Parity;FP32Reference"
    MPI_PROCS 1
)

# Test: Q16 Full Pipeline Parity (end-to-end PURE INTEGER vs FP32)
add_executable(v2_test_q16_full_pipeline_parity unit/Test__Q16FullPipelineParity.cpp)
target_link_libraries(v2_test_q16_full_pipeline_parity
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q16FullPipelineParity
    COMMAND $<TARGET_FILE:v2_test_q16_full_pipeline_parity>
    LABELS "V2;Unit;Attention;Q16;IntegerDomain;WoProjection;ResidualAdd;FullPipeline;Parity;FP32Reference"
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

# Test: Precision-aware allreduce N-way sum functions (FP16, BF16, Q8_1)
add_executable(v2_test_precision_aware_allreduce unit/mpi/Test__PrecisionAwareAllreduce.cpp)
target_link_libraries(v2_test_precision_aware_allreduce 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PrecisionAwareAllreduce
    COMMAND $<TARGET_FILE:v2_test_precision_aware_allreduce>
    LABELS "V2;Unit;MPI;Allreduce;SIMD;FP16;BF16;Q8_1;AVX512;Precision"
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

# Test: UnifiedKVCache (supports both single-sequence and batched modes, all precisions)
add_executable(v2_test_unified_kv_cache unit/tensors/Test__UnifiedKVCache.cpp)
target_link_libraries(v2_test_unified_kv_cache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_UnifiedKVCache 
    COMMAND $<TARGET_FILE:v2_test_unified_kv_cache>
    LABELS "V2;Unit;KVCache;Batched;SingleSequence;FP32;BF16;FP16;Q8_1;Q16_1;Eviction"
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

# Test: ComputeStage (device-agnostic kernel dispatch abstraction)
# Tests all compute stage types: RMSNorm, RoPE, Attention, SwiGLU, ResidualAdd, MoE stages
add_executable(v2_test_compute_stage unit/Test__ComputeStage.cpp)
target_link_libraries(v2_test_compute_stage llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ComputeStage
    COMMAND $<TARGET_FILE:v2_test_compute_stage>
    LABELS "V2;Unit;Execution;ComputeStage;RMSNorm;RoPE;Attention;SwiGLU;MoE;KernelDispatch"
    MPI_PROCS 1  # Tests CPU stages locally, no MPI needed
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
# Tests that EmbeddingStage correctly produces Q16_1 output for HybridQ16 mode
# Verifies FP32 → Q16_1 quantization preserves embedding values
# NOTE: This is an Integration test (slow, requires model loading)
add_executable(v2_test_embedding_stage_q16_1 unit/stages/Test__EmbeddingStage_Q16_1Output.cpp)
target_link_libraries(v2_test_embedding_stage_q16_1 llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Integration_EmbeddingStage_Q16_1Output
    COMMAND $<TARGET_FILE:v2_test_embedding_stage_q16_1>
    LABELS "V2;Integration;Execution;ComputeStage;Embedding;Q16_1;HybridQ16;Quantization"
    MPI_PROCS 1
)

# Test: HybridQ16 Layer 0 Dataflow Diagnostics
# Traces data flow from embedding through RMSNorm in HybridQ16 mode
add_executable(v2_test_hybridq16_layer0_dataflow unit/stages/Test__HybridQ16_Layer0_Dataflow.cpp)
target_link_libraries(v2_test_hybridq16_layer0_dataflow llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_HybridQ16_Layer0_Dataflow
    COMMAND $<TARGET_FILE:v2_test_hybridq16_layer0_dataflow>
    LABELS "V2;Unit;Execution;ComputeStage;HybridQ16;Dataflow;Diagnostics"
    MPI_PROCS 1
)

# Test: FusedAttentionWoStage with Q16_INTEGER backend
# Tests stage construction, type, backend support, and dump info with Q16FusedAttentionKernel
add_executable(v2_test_fused_attention_wo_stage_q16 unit/stages/Test__FusedAttentionWoStage_Q16Backend.cpp)
target_link_libraries(v2_test_fused_attention_wo_stage_q16 llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_FusedAttentionWoStage_Q16Backend
    COMMAND $<TARGET_FILE:v2_test_fused_attention_wo_stage_q16>
    LABELS "V2;Unit;Execution;ComputeStage;FusedAttentionWo;Q16_INTEGER;Q16FusedAttentionKernel"
    MPI_PROCS 1
)

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

# Test: TensorValidation (debug-only tensor validation utilities)
# Tests zero detection, NaN/Inf detection, and validation assertions.
# Part of Buffer Contract Validation System - Phase 1: Quick Wins
add_executable(v2_test_tensor_validation unit/Test__TensorValidation.cpp)
target_link_libraries(v2_test_tensor_validation llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_TensorValidation
    COMMAND $<TARGET_FILE:v2_test_tensor_validation>
    LABELS "V2;Unit;TensorOperations;Validation;Debug"
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
# Tests actual MPI communication for FP32, FP16, BF16, and Q8_1 formats
# Validates correctness of partial sum accumulation across ranks
add_executable(v2_integration_mpi_precision_allreduce integration/Test__MPI_PrecisionAwareAllreduce.cpp)
target_link_libraries(v2_integration_mpi_precision_allreduce
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_PrecisionAwareAllreduce
    COMMAND $<TARGET_FILE:v2_integration_mpi_precision_allreduce>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;Allreduce;FP32;FP16;BF16;Q8_1;Bandwidth"
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
# E2E Parity: MPI Tensor-Parallel vs Single-Rank
# -----------------------------------------------------------------------------
# Compares distributed execution against single-node to ensure TP correctness
if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR ENABLE_SNAPSHOTS)
    add_executable(v2_e2e_parity_mpi_vs_single_rank 
        e2e/parity/internal/mpi_vs_single_rank/Test__MPIPipeline_vs_SingleRank.cpp
        e2e/parity/gtest_mpi_main.cpp
    )
    target_link_libraries(v2_e2e_parity_mpi_vs_single_rank llaminar2_core GTest::gtest MPI::MPI_CXX)

    add_v2_test(V2_E2E_Parity_MPIPipeline_vs_SingleRank
        COMMAND $<TARGET_FILE:v2_e2e_parity_mpi_vs_single_rank>
        LABELS "V2;E2E;Parity;Internal;MPI;TensorParallel"
        MPI_PROCS 2
    )
else()
    message(STATUS "V2: E2E MPI parity test disabled (snapshots not available in Release build)")
    message(STATUS "     Use -DCMAKE_BUILD_TYPE=E2ERelease or -DENABLE_SNAPSHOTS=ON to enable")
endif()

# -----------------------------------------------------------------------------
# E2E Parity: Llaminar FP32 Pipeline vs PyTorch Reference
# -----------------------------------------------------------------------------
# Ground truth validation against PyTorch Qwen2 implementation
find_package(ZLIB)
if(ZLIB_FOUND)
    get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    
    if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR ENABLE_SNAPSHOTS)
        add_executable(v2_e2e_parity_fp32_vs_pytorch 
            e2e/parity/pytorch/qwen2/Test__FP32Pipeline_vs_PyTorch.cpp
            ${WORKSPACE_ROOT}/external/cnpy/cnpy.cpp
        )
        target_link_libraries(v2_e2e_parity_fp32_vs_pytorch 
            llaminar2_core 
            GTest::gtest 
            GTest::gtest_main
            ZLIB::ZLIB
        )
        target_include_directories(v2_e2e_parity_fp32_vs_pytorch PRIVATE 
            ${WORKSPACE_ROOT}/external/cnpy
        )
        
        add_v2_test(V2_E2E_Parity_FP32Pipeline_vs_PyTorch
            COMMAND $<TARGET_FILE:v2_e2e_parity_fp32_vs_pytorch>
            LABELS "V2;E2E;Parity;PyTorch;FP32;Qwen2"
            MPI_PROCS 1
        )
        message(STATUS "V2: E2E FP32 vs PyTorch parity test enabled")

        # -----------------------------------------------------------------------------
        # E2E Parity: Q8_1 Pipeline vs FP32 Pipeline (Layer-by-Layer)
        # -----------------------------------------------------------------------------
        # Identifies exactly where Q8_1 diverges from FP32 using snapshot comparison
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
        message(STATUS "V2: E2E Q8_1 vs FP32 pipeline parity test enabled")

        # -----------------------------------------------------------------------------
        # E2E Parity: Hybrid Pipeline vs FP32 and Q8_1 (Layer-by-Layer)
        # -----------------------------------------------------------------------------
        # Validates that Hybrid precision mode achieves ≥0.995 cosine similarity
        # with FP32, measuring improvement over full Q8_1 mode
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

        # -----------------------------------------------------------------------------
        # E2E Parity: HybridQ16 Pipeline vs FP32 (Layer-by-Layer)
        # -----------------------------------------------------------------------------
        # Validates that HybridQ16 precision mode (Q16_1 typed residuals) achieves
        # ≥0.995 cosine similarity with FP32 at residual stages, measuring improvement
        # over Hybrid mode (Q8_1 residuals). Tests both prefill and decode phases.
        add_executable(v2_e2e_parity_hybridq16_vs_fp32_pipeline 
            e2e/parity/internal/hybridq16_vs_fp32/Test__HybridQ16Pipeline_vs_FP32_LayerByLayer.cpp
        )
        target_link_libraries(v2_e2e_parity_hybridq16_vs_fp32_pipeline 
            llaminar2_core 
            GTest::gtest 
        )
        add_v2_test(V2_E2E_Parity_HybridQ16Pipeline_vs_FP32
            COMMAND $<TARGET_FILE:v2_e2e_parity_hybridq16_vs_fp32_pipeline>
            LABELS "V2;E2E;Parity;Internal;HybridQ16;Q16_1;FP32;LayerByLayer"
            MPI_PROCS 1
        )
        message(STATUS "V2: E2E HybridQ16 vs FP32 pipeline parity test enabled")
    else()
        message(STATUS "V2: E2E parity tests disabled (snapshots not available in Release build)")
        message(STATUS "     Use -DCMAKE_BUILD_TYPE=E2ERelease or -DENABLE_SNAPSHOTS=ON to enable")
    endif()
else()
    message(WARNING "ZLIB not found - PyTorch parity tests disabled (requires cnpy for .npz loading)")
endif()

# -----------------------------------------------------------------------------
# E2E Parity: Kernel-Level Tests (Attention)
# -----------------------------------------------------------------------------
# Compare individual kernels against FP32 reference implementations
# These run without model loading and test kernel correctness in isolation

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

# JitFusedAttentionWo with Q16_1 Fused Residual vs FP32 Reference
# Tests Phase 5 of Q16_1 Typed Residual Stream: fused Attention + Wo + Residual Add
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

# Test: JIT Fused Attention + Wo - Q16_1 Residual Fusion Tests
# Tests the fused residual addition path for Q16_1 typed residual streams
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
