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

# Test: CpuAttentionKernelT (template-based attention kernel)
add_executable(v2_test_cpu_attention_kernel_t unit/Test__CpuAttentionKernelT.cpp)
target_link_libraries(v2_test_cpu_attention_kernel_t
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CpuAttentionKernelT
    COMMAND $<TARGET_FILE:v2_test_cpu_attention_kernel_t>
    LABELS "V2;Unit;Kernels;Attention;FP32;TemplateKernel;CPU;MHA;GQA;CausalMasking"
    MPI_PROCS 1  # CpuAttentionKernelT tests are CPU-only, single rank
)

# Test: CPUAttentionKernelTyped (Enum-based attention kernel via Factory)
add_executable(v2_test_cpu_attention_kernel_typed unit/Test__CPUAttentionKernelTyped.cpp)
target_link_libraries(v2_test_cpu_attention_kernel_typed
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUAttentionKernelTyped
    COMMAND $<TARGET_FILE:v2_test_cpu_attention_kernel_typed>
    LABELS "V2;Unit;Kernels;Attention;FP32;BF16;FP16;TypedKernel;CPU;FactoryWiring"
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

# Test: CpuAttentionKernelT Batch Attention (comprehensive batch testing)
add_executable(v2_test_cpu_attention_kernel_t_batch unit/Test__CpuAttentionKernelT_Batch.cpp)
target_link_libraries(v2_test_cpu_attention_kernel_t_batch
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CpuAttentionKernelT_Batch
    COMMAND $<TARGET_FILE:v2_test_cpu_attention_kernel_t_batch>
    LABELS "V2;Unit;Kernels;Attention;FP32;BatchAttention;Equivalence;Independence;GQA;CausalMasking"
    MPI_PROCS 1  # Batch attention tests are CPU-only, single rank
)

# Test: CpuAttentionKernelT Padding Mask (critical batch padding bug tests)
add_executable(v2_test_cpu_attention_kernel_t_padding_mask unit/Test__CpuAttentionKernelT_PaddingMask.cpp)
target_link_libraries(v2_test_cpu_attention_kernel_t_padding_mask
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CpuAttentionKernelT_PaddingMask
    COMMAND $<TARGET_FILE:v2_test_cpu_attention_kernel_t_padding_mask>
    LABELS "V2;Unit;Kernels;Attention;FP32;BatchAttention;PaddingMask;MaskApplication;OutputZeroing;SequentialVsBatch"
    MPI_PROCS 1  # CPU-only padding mask tests, single rank
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
add_executable(v2_test_attention_parity unit/Test__AttentionParity.cpp)
target_link_libraries(v2_test_attention_parity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_AttentionParity 
    COMMAND $<TARGET_FILE:v2_test_attention_parity>
    LABELS "V2;Unit;Attention;RegressionTests;RoPE;CausalMasking;GQA;ParityTesting;CPU"
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
add_executable(v2_test_batch_attention_correctness unit/Test__BatchAttentionCorrectness.cpp)
target_link_libraries(v2_test_batch_attention_correctness 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BatchAttentionCorrectness 
    COMMAND $<TARGET_FILE:v2_test_batch_attention_correctness>
    LABELS "V2;Unit;Attention;BatchProcessing;MaskGeneration;PaddingMask;CausalMask;BlockDiagonal"
    MPI_PROCS 1  # Unit test for mask utilities, single rank
)

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
add_executable(v2_test_gqa_attention_batched_path unit/Test__GQAAttention_BatchedPath.cpp)
target_link_libraries(v2_test_gqa_attention_batched_path 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GQAAttention_BatchedPath
    COMMAND $<TARGET_FILE:v2_test_gqa_attention_batched_path>
    LABELS "V2;Unit;Attention;GQA;Batched;Causal;Padding;HeadBroadcasting"
    MPI_PROCS 1  # Unit test for GQAAttention compute_batch()
)

# Test: GQAAttention padding mask correctness
add_executable(v2_test_gqa_attention_padding_mask unit/Test__GQAAttention_PaddingMask.cpp)
target_link_libraries(v2_test_gqa_attention_padding_mask 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GQAAttention_PaddingMask
    COMMAND $<TARGET_FILE:v2_test_gqa_attention_padding_mask>
    LABELS "V2;Unit;Attention;GQA;Padding;Masking;BatchProcessing;SequenceIsolation"
    MPI_PROCS 1  # Unit test for padding mask construction and application
)

# Test: GQAAttention mask passing verification (critical bug investigation)
add_executable(v2_test_gqa_attention_mask_passing unit/Test__GQAAttention_MaskPassing.cpp)
target_link_libraries(v2_test_gqa_attention_mask_passing 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GQAAttention_MaskPassing
    COMMAND $<TARGET_FILE:v2_test_gqa_attention_mask_passing>
    LABELS "V2;Unit;Attention;GQA;MaskPassing;BufferLayout;PaddingZeroing;SequentialVsBatch"
    MPI_PROCS 1  # Critical tests for batch padding divergence bug
)

# Test: MpiAttentionOrchestrator mask passing verification
add_executable(v2_test_mpi_attention_orchestrator_mask_passing unit/Test__MpiAttentionOrchestrator_MaskPassing.cpp)
target_link_libraries(v2_test_mpi_attention_orchestrator_mask_passing 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MpiAttentionOrchestrator_MaskPassing
    COMMAND $<TARGET_FILE:v2_test_mpi_attention_orchestrator_mask_passing>
    LABELS "V2;Unit;Attention;MpiOrchestrator;MaskPassing;BatchProcessing;MPIDispatch;SequentialVsBatch"
    MPI_PROCS 1  # Tests orchestrator layer mask handling (next layer up from GQA)
)

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

# Test: WeightManager Sharding
# Tests weight sharding for tensor parallelism (Phase 1: row-parallel only)
add_executable(v2_test_weight_manager_sharding unit/loaders/Test__WeightManagerSharding.cpp)
target_link_libraries(v2_test_weight_manager_sharding llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightManagerSharding
    COMMAND $<TARGET_FILE:v2_test_weight_manager_sharding>
    LABELS "V2;Unit;TensorParallelism;WeightSharding;RowParallel"
)

# Test: PipelineBase Device Detection (Phase 3)
add_executable(v2_test_pipeline_device_detection
    unit/pipelines/Test__PipelineBase_DeviceDetection.cpp
)
target_link_libraries(v2_test_pipeline_device_detection llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PipelineDeviceDetection
    COMMAND $<TARGET_FILE:v2_test_pipeline_device_detection>
    LABELS "V2;Unit;PipelineExecution;DeviceManagement;MoE;Phase3"
)

# Test: PipelineBase Precision Mode
# Tests WeightPrecision and ActivationPrecision configuration flow through PipelineBase
add_executable(v2_test_pipeline_precision_mode unit/pipelines/Test__PipelineBase_PrecisionMode.cpp)
target_link_libraries(v2_test_pipeline_precision_mode llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PipelinePrecisionMode
    COMMAND $<TARGET_FILE:v2_test_pipeline_precision_mode>
    LABELS "V2;Unit;PipelineExecution;PrecisionConfiguration;WeightPrecision;ActivationPrecision"
)

# Test: ResidualOpTyped (typed residual connections FP32/BF16/FP16/Q8_1)
# Tests the typed residual connection operations that add projection outputs back to residual stream
# Critical for verifying Q8_1 native residual addition (dequant-add-requant via SIMD)
add_executable(v2_test_residual_op_typed unit/pipelines/ops/Test__ResidualOpTyped.cpp)
target_link_libraries(v2_test_residual_op_typed llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_ResidualOpTyped
    COMMAND $<TARGET_FILE:v2_test_residual_op_typed>
    LABELS "V2;Unit;PipelineOps;ResidualConnection;Q8_1;FP32;TypedResiduals;Quantization;SIMD"
    MPI_PROCS 1
)

# Test: RMSNormOpTyped (Q8_1 native RMSNorm)
# Critical for verifying Q8_1 native RMSNorm (dequant-normalize-requant)
add_executable(v2_test_rmsnorm_op_typed unit/pipelines/ops/Test__RMSNormOpTyped.cpp)
target_link_libraries(v2_test_rmsnorm_op_typed llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_RMSNormOpTyped
    COMMAND $<TARGET_FILE:v2_test_rmsnorm_op_typed>
    LABELS "V2;Unit;PipelineOps;RMSNorm;Q8_1;FP32;TypedNorm;Quantization;SIMD"
    MPI_PROCS 1
)

# Test: GemmOpTyped (Q8_1 native GEMM for QKV projection)
# Critical for verifying Q8_1 native GEMM (Q8_1→Q8_1 and FP32→Q8_1 paths)
add_executable(v2_test_gemm_op_typed unit/pipelines/ops/Test__GemmOpTyped.cpp)
target_link_libraries(v2_test_gemm_op_typed llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GemmOpTyped
    COMMAND $<TARGET_FILE:v2_test_gemm_op_typed>
    LABELS "V2;Unit;PipelineOps;GEMM;Q8_1;FP32;TypedGemm;Quantization;QKVProjection"
    MPI_PROCS 1
)

# Test: GQAAttention Second Sequence Handling
# Tests that sequence 1 in batched attention is handled correctly
# Validates buffer offsets, mask application, and cross-sequence isolation
add_executable(v2_test_gqa_attention_second_sequence unit/Test__GQAAttention_SecondSequence.cpp)
target_link_libraries(v2_test_gqa_attention_second_sequence llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_GQAAttention_SecondSequence
    COMMAND $<TARGET_FILE:v2_test_gqa_attention_second_sequence>
    LABELS "V2;Unit;Attention;GQA;Batched;SecondSequence;BufferOffsets"
)

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

# Test: Q8_1 Isolated Attention Kernel Test
# Captures attention inputs from pipeline, compares FP32 vs Q8_1 kernel in isolation
# Used for debugging attention kernel divergence without full pipeline complexity
add_executable(v2_integration_q8_1_attention_kernel integration/Test__Q8_1_AttentionKernel.cpp)
target_link_libraries(v2_integration_q8_1_attention_kernel 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_AttentionKernel
    COMMAND $<TARGET_FILE:v2_integration_q8_1_attention_kernel>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;Attention;Q8_1;Parity;Debug"
)

# Test: Q8_1 Attention JIT Replay Test
# Saves attention inputs to disk and replays them through the JIT kernel in isolation
# Used for reproducible debugging of the JIT attention kernel
add_executable(v2_integration_q8_1_attention_jit_replay integration/Test__Q8_1_AttentionJitReplay.cpp)
target_link_libraries(v2_integration_q8_1_attention_jit_replay 
    llaminar2_core 
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_AttentionJitReplay
    COMMAND $<TARGET_FILE:v2_integration_q8_1_attention_jit_replay>
    MPI_PROCS 1
    LABELS "V2;Integration;Kernels;Attention;Q8_1;JIT;Debug;Replay"
)

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

# Test: Qwen2Pipeline Integration Tests (full pipeline, autoregressive decode, bias validation)
# Validates end-to-end pipeline functionality including Q/K/V/O bias application
add_executable(v2_integration_qwen2_pipeline integration/Test__Qwen2Pipeline.cpp)
target_link_libraries(v2_integration_qwen2_pipeline
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Qwen2Pipeline
    COMMAND $<TARGET_FILE:v2_integration_qwen2_pipeline>
    LABELS "V2;Integration;Pipeline;Qwen2;AutoregressiveDecode;Bias;Attention;E2E"
)

# Test: KVCache Integration Tests (KV cache behavior during autoregressive decode)
# Validates cache accumulation, data integrity, position tracking, and attention correctness
add_executable(v2_integration_kv_cache integration/Test__KVCacheIntegration.cpp)
target_link_libraries(v2_integration_kv_cache
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_KVCache
    COMMAND $<TARGET_FILE:v2_integration_kv_cache>
    LABELS "V2;Integration;KVCache;AutoregressiveDecode;Attention;Pipeline"
)

# Test: Q8_1 KV Cache Integration Tests
# Validates Q8_1 attention kernel correctly integrates with typed KV cache
# Tests: roundtrip, attention with cache, direct vs cached K/V, GQA path, block copy semantics
add_executable(v2_integration_q8_1_kvcache integration/Test__Q8_1_KVCache_Integration.cpp)
target_link_libraries(v2_integration_q8_1_kvcache
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_KVCache
    COMMAND $<TARGET_FILE:v2_integration_q8_1_kvcache>
    MPI_PROCS 1
    LABELS "V2;Integration;KVCache;Q8_1;Attention;Pipeline;Correctness"
)

# Test: Q8_1 vs FP32 Activation Precision Parity
# Validates Q8_1 activation precision produces equivalent results to FP32
# Tests Top-5 token overlap, KL divergence, and greedy sampling equivalence
# NOTE: Runs with MPI_PROCS 1 since it compares precision modes, not MPI parallelism
add_executable(v2_integration_q8_1_vs_fp32_parity integration/Test__Q8_1_vs_FP32_Parity.cpp)
target_link_libraries(v2_integration_q8_1_vs_fp32_parity
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_vs_FP32_Parity
    COMMAND $<TARGET_FILE:v2_integration_q8_1_vs_fp32_parity>
    LABELS "V2;Integration;Pipeline;ActivationPrecision;Q8_1;FP32;Parity;TopK;KLDivergence"
    MPI_PROCS 1
)

# Test: Q8_1 MPI Tensor-Parallel Attention
# Tests the MPI tensor-parallel path for Q8_1 attention:
#   - Head slicing (sliceQ8_1HeadBlocks)
#   - K/V GQA broadcast (broadcastQ8_1KVHeads)
#   - Fused attention kernel execution
#   - Allgatherv result combination
add_executable(v2_integration_q8_1_mpi_tp integration/Test__Q8_1_MPI_TensorParallel.cpp)
target_link_libraries(v2_integration_q8_1_mpi_tp
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_Q8_1_MPI_TensorParallel
    COMMAND $<TARGET_FILE:v2_integration_q8_1_mpi_tp>
    MPI_PROCS 2
    LABELS "V2;Integration;Attention;MPI;TensorParallel;Q8_1;GQA;Orchestration"
)

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
add_executable(v2_integration_mpi_row_parallel_multiprecision integration/Test__MPI_RowParallelMultiPrecision.cpp)
target_link_libraries(v2_integration_mpi_row_parallel_multiprecision
    llaminar2_core
    GTest::gtest
)
add_v2_test(V2_Integration_MPI_RowParallelMultiPrecision
    COMMAND $<TARGET_FILE:v2_integration_mpi_row_parallel_multiprecision>
    MPI_PROCS 2
    LABELS "V2;Integration;MPI;RowParallel;TensorSlice;KSliced;Allgatherv;Allreduce;FP32;BF16;FP16;Q8_1;TensorParallel"
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
# E2E Tests
# =============================================================================
# Full pipeline tests with real models and inference
# Tests complete workflows: load -> prefill -> decode -> validate output

# Qwen2 MPI Rank Parity (single-rank vs multi-rank comparison)
# These tests compare MPI ranks against each other, NOT against PyTorch ground truth.
# Only build for Debug or E2ERelease configurations (requires ENABLE_PIPELINE_SNAPSHOTS)
if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR ENABLE_SNAPSHOTS)
    add_executable(v2_test_qwen2_mpi_rank_parity 
        e2e/qwen2/Test__Qwen2MPIRankParity.cpp
        e2e/gtest_mpi_main.cpp
    )
    target_link_libraries(v2_test_qwen2_mpi_rank_parity llaminar2_core GTest::gtest MPI::MPI_CXX)

    add_v2_test(V2_E2E_Qwen2MPIRankParity
        COMMAND $<TARGET_FILE:v2_test_qwen2_mpi_rank_parity>
        LABELS "V2;E2E;Qwen;MPI;RankParity;Models"
        MPI_PROCS 2
    )
else()
    message(STATUS "V2: E2E MPI rank parity test disabled (snapshots not available in Release build)")
    message(STATUS "     Use -DCMAKE_BUILD_TYPE=E2ERelease or -DENABLE_SNAPSHOTS=ON to enable")
endif()

# =============================================================================
# PyTorch Parity Tests
# =============================================================================
# Ground truth validation against PyTorch reference implementation
# Compares layer-by-layer outputs, logits, attention scores, etc.

# Qwen2 FP32 Parity (PyTorch ground truth validation)
# Requires: PyTorch snapshots generated by python/reference/generate_qwen2_pipeline_snapshots.py
# Requires: cnpy library for NumPy .npz file loading
find_package(ZLIB)
if(ZLIB_FOUND)
    # Get the workspace root (parent of src/v2)
    get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    
    # E2E parity test requires ENABLE_PIPELINE_SNAPSHOTS (Debug or E2ERelease builds)
    # Skip in standard Release builds to avoid linking errors
    if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR ENABLE_SNAPSHOTS)
        add_executable(v2_test_qwen2_fp32_parity 
            e2e/qwen2/Test__Qwen2FP32Parity.cpp
            ${WORKSPACE_ROOT}/external/cnpy/cnpy.cpp
        )
        target_link_libraries(v2_test_qwen2_fp32_parity 
            llaminar2_core 
            GTest::gtest 
            GTest::gtest_main
            ZLIB::ZLIB
        )
        target_include_directories(v2_test_qwen2_fp32_parity PRIVATE 
            ${WORKSPACE_ROOT}/external/cnpy
        )
        
        add_v2_test(V2_E2E_Qwen2FP32Parity
            COMMAND $<TARGET_FILE:v2_test_qwen2_fp32_parity>
            LABELS "V2;E2E;Qwen;Parity;PyTorch;FP32"
            MPI_PROCS 1
        )
        message(STATUS "V2: E2E parity test enabled (snapshots available)")
        
        # E2E Test: Q8_1 vs FP32 Layer-by-Layer Divergence Analysis
        # Compares intermediate activations at each pipeline stage between Q8_1 and FP32
        # Uses snapshot capture to identify exactly where divergence occurs
        add_executable(v2_test_q8_1_layer_divergence 
            e2e/qwen2/Test__Q8_1_LayerByLayer_Divergence.cpp
        )
        target_link_libraries(v2_test_q8_1_layer_divergence 
            llaminar2_core 
            GTest::gtest 
        )
        add_v2_test(V2_E2E_Q8_1_LayerByLayer_Divergence
            COMMAND $<TARGET_FILE:v2_test_q8_1_layer_divergence>
            LABELS "V2;E2E;Qwen;Parity;Q8_1;FP32;Divergence"
            MPI_PROCS 1
        )
        message(STATUS "V2: Q8_1 layer divergence test enabled (snapshots available)")
    else()
        message(STATUS "V2: E2E parity test disabled (snapshots not available in Release build)")
        message(STATUS "     Use -DCMAKE_BUILD_TYPE=E2ERelease or -DENABLE_SNAPSHOTS=ON to enable")
    endif()
else()
    message(WARNING "ZLIB not found - Qwen2 FP32 parity test disabled (requires cnpy for .npz loading)")
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

