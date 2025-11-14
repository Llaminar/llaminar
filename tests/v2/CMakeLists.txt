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

# Test: INT32Tensor class (accumulator tensor for INT8 pipelines)
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

# Test: INT32 RMSNorm kernel (INT32→INT8 normalization for full INT8 pipelines)
add_executable(v2_test_int32_rmsnorm unit/Test__INT32RMSNorm.cpp)
target_link_libraries(v2_test_int32_rmsnorm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_INT32RMSNorm 
    COMMAND $<TARGET_FILE:v2_test_int32_rmsnorm>
    LABELS "V2;Unit;Kernels;RMSNorm;INT32;INT8Pipeline;Quantization;AVX512;CPU"
    MPI_PROCS 1  # RMSNorm is CPU-only, single rank
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

# Test: RMSNorm interface (demonstrates clean interface design)
add_executable(v2_test_rmsnorm_interface unit/Test__RMSNormInterface.cpp)
target_link_libraries(v2_test_rmsnorm_interface
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RMSNormInterface
    COMMAND $<TARGET_FILE:v2_test_rmsnorm_interface>
    LABELS "V2;Unit;Kernels;RMSNorm;InterfaceDesign;FP32;BF16;FP16;INT32;INT8;TypeSafety;CPU"
    MPI_PROCS 1  # Interface tests are CPU-only, single rank
)

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

# Test: CPUAttentionT (template-based attention kernel)
add_executable(v2_test_cpu_attention_t unit/Test__CPUAttentionT.cpp)
target_link_libraries(v2_test_cpu_attention_t
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUAttentionT
    COMMAND $<TARGET_FILE:v2_test_cpu_attention_t>
    LABELS "V2;Unit;Kernels;Attention;FP32;TemplateKernel;CPU;MHA;GQA;CausalMasking"
    MPI_PROCS 1  # CPUAttentionT tests are CPU-only, single rank
)

# Test: FusedGemmSoftmax (Stage 1 optimization - fused attention scores)
# add_executable(v2_test_fused_gemm_softmax unit/Test__FusedGemmSoftmax.cpp)
# target_link_libraries(v2_test_fused_gemm_softmax
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_FusedGemmSoftmax
#     COMMAND $<TARGET_FILE:v2_test_fused_gemm_softmax>
#     LABELS "V2;Unit;Kernels;Attention;FusedKernel;CPU;CacheOptimization;MemoryEfficiency"
#     MPI_PROCS 1  # CPU-only kernel, single rank
# )

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

# Test: TiledGemmSoftmax (tile-level fusion for attention scores)
# add_executable(v2_test_tiled_gemm_softmax unit/Test__TiledGemmSoftmax.cpp)
# target_link_libraries(v2_test_tiled_gemm_softmax
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_TiledGemmSoftmax
#     COMMAND $<TARGET_FILE:v2_test_tiled_gemm_softmax>
#     LABELS "V2;Unit;Kernels;Attention;TileFusion;GEMM;Softmax;CacheOptimization;CPU;AVX2;AVX512"
#     MPI_PROCS 1  # CPU-only kernel, single rank
# )

# Test: IntegerGemm (integer-domain quantized GEMM with AVX512-VNNI, llama.cpp strategy)
add_executable(v2_test_integer_gemm unit/Test__IntegerGemm.cpp)
target_link_libraries(v2_test_integer_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IntegerGemm
    COMMAND $<TARGET_FILE:v2_test_integer_gemm>
    LABELS "V2;Unit;Kernels;GEMM;Quantization;INT8;AVX512VNNI;IQ4_NL;Q8_0;IntegerArithmetic"
    MPI_PROCS 1  # CPU-only VNNI kernel, single rank
)

# Test: ParameterizedInt8Gemm (template-based microkernel with variable MR×NR, OneDNN-inspired)
add_executable(v2_test_parameterized_int8_gemm unit/Test__ParameterizedInt8Gemm.cpp)
target_link_libraries(v2_test_parameterized_int8_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_ParameterizedInt8Gemm
    COMMAND $<TARGET_FILE:v2_test_parameterized_int8_gemm>
    LABELS "V2;Unit;Kernels;GEMM;Quantization;INT8;AVX512VNNI;ParameterizedKernel;OneDNN;Performance"
    MPI_PROCS 1  # CPU-only VNNI kernel, single rank
)

# Test: FusedSoftmaxGemm (fused softmax+GEMM microkernel for attention mechanism)
add_executable(v2_test_fused_softmax_gemm unit/Test__FusedSoftmaxGemm.cpp)
target_link_libraries(v2_test_fused_softmax_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedSoftmaxGemm
    COMMAND $<TARGET_FILE:v2_test_fused_softmax_gemm>
    LABELS "V2;Unit;Kernels;GEMM;FusedOperations;Softmax;Attention;INT8;AVX512VNNI;ModularKernel"
    MPI_PROCS 1  # CPU-only fused kernel, single rank
)

# Test: VectorizedSoftmax (ISA parity and performance for AVX512/AVX2/Scalar)
add_executable(v2_test_vectorized_softmax unit/Test__VectorizedSoftmax.cpp)
target_link_libraries(v2_test_vectorized_softmax
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_VectorizedSoftmax
    COMMAND $<TARGET_FILE:v2_test_vectorized_softmax>
    LABELS "V2;Unit;Kernels;Softmax;Vectorization;ISA;AVX512;AVX2;Performance;Correctness"
    MPI_PROCS 1  # CPU-only SIMD kernel, single rank
)

# Test: Q8_0DecodeVectorization (SIMD correctness for IQ4_NL/FP32→Q8_0 decode)
add_executable(v2_test_q8_0_decode_vectorization unit/Test__Q8_0DecodeVectorization.cpp)
target_link_libraries(v2_test_q8_0_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_0DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q8_0_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q8_0;IQ4_NL;FP32;SIMD;AVX2;AVX512;VectorizationCorrectness"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: TensorDecodeToQ8_0 (tensor decode_to_q8_0() correctness and integration)
add_executable(v2_test_tensor_decode_to_q8_0 unit/Test__TensorDecodeToQ8_0.cpp)
target_link_libraries(v2_test_tensor_decode_to_q8_0
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_TensorDecodeToQ8_0
    COMMAND $<TARGET_FILE:v2_test_tensor_decode_to_q8_0>
    LABELS "V2;Unit;TensorOperations;Quantization;Q8_0;DecodeAPI;AccessorIntegration;ThreadSafety"
    MPI_PROCS 1  # CPU-only tensor decode, single rank
)

# Test: Q8_0_SIMD_Accuracy (SIMD variant accuracy for FP32/FP16/BF16/Q8_K→Q8_0)
add_executable(v2_test_q8_0_simd_accuracy unit/Test__Q8_0_SIMD_Accuracy.cpp)
target_link_libraries(v2_test_q8_0_simd_accuracy
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_0_SIMD_Accuracy
    COMMAND $<TARGET_FILE:v2_test_q8_0_simd_accuracy>
    LABELS "V2;Unit;TensorOperations;Quantization;Q8_0;FP32;FP16;BF16;Q8_K;SIMD;AVX2;AVX512;AccuracyTest"
    MPI_PROCS 1  # CPU-only SIMD accuracy testing, single rank
)

# Test: IQ8_1Decodable (Q8_1 quantization SIMD accuracy and performance)
add_executable(v2_test_iq8_1_decodable unit/Test__IQ8_1Decodable.cpp)
target_link_libraries(v2_test_iq8_1_decodable
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ8_1Decodable
    COMMAND $<TARGET_FILE:v2_test_iq8_1_decodable>
    LABELS "V2;Unit;TensorOperations;Quantization;Q8_1;FP32;FP16;BF16;SIMD;AVX2;AVX512;AccuracyTest;PerformanceTest;PreComputedSum"
    MPI_PROCS 1  # CPU-only Q8_1 quantization testing, single rank
)

# Test: Q8_1GemmKernel (Q8_1 × Q8_0 → FP32 GEMM mathematical correctness)
add_executable(v2_test_q8_1_gemm_kernel unit/Test__Q8_1GemmKernel.cpp)
target_link_libraries(v2_test_q8_1_gemm_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1GemmKernel
    COMMAND $<TARGET_FILE:v2_test_q8_1_gemm_kernel>
    LABELS "V2;Unit;Kernels;GEMM;Q8_1;Q8_0;FP32;FP16;BF16;IQ8_1Decodable;AccuracyTest;PreComputedSum;Correctness"
    MPI_PROCS 1  # CPU-only GEMM correctness testing, single rank
)

# Test: Q8_1GemmKernel_JRBatch (JR_BATCH parameterization correctness)
add_executable(v2_test_q8_1_gemm_jr_batch unit/Test__Q8_1GemmKernel_JRBatch.cpp)
target_link_libraries(v2_test_q8_1_gemm_jr_batch
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_1GemmKernel_JRBatch
    COMMAND $<TARGET_FILE:v2_test_q8_1_gemm_jr_batch>
    LABELS "V2;Unit;Kernels;GEMM;Q8_1;Parameterization;JR_BATCH;Correctness;RegisterPressure;Vectorization"
    MPI_PROCS 1  # CPU-only JR_BATCH validation, single rank
)

# Test: Q6_K_DecodeVectorization (SIMD correctness for Q6_K→Q8_0 decode)
add_executable(v2_test_q6_k_decode_vectorization unit/Test__Q6_K_DecodeVectorization.cpp)
target_link_libraries(v2_test_q6_k_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q6_K_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q6_k_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q6_K;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: Q4_0_DecodeVectorization (SIMD correctness for Q4_0→Q8_0 decode)
add_executable(v2_test_q4_0_decode_vectorization unit/Test__Q4_0_DecodeVectorization.cpp)
target_link_libraries(v2_test_q4_0_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_0_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q4_0_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q4_0;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: Q4_1_DecodeVectorization (SIMD correctness for Q4_1→Q8_0 decode)
add_executable(v2_test_q4_1_decode_vectorization unit/Test__Q4_1_DecodeVectorization.cpp)
target_link_libraries(v2_test_q4_1_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_1_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q4_1_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q4_1;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: Q5_0_DecodeVectorization (SIMD correctness for Q5_0→Q8_0 decode)
add_executable(v2_test_q5_0_decode_vectorization unit/Test__Q5_0_DecodeVectorization.cpp)
target_link_libraries(v2_test_q5_0_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_0_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q5_0_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q5_0;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: Q5_1_DecodeVectorization (SIMD correctness for Q5_1→Q8_0 decode)
add_executable(v2_test_q5_1_decode_vectorization unit/Test__Q5_1_DecodeVectorization.cpp)
target_link_libraries(v2_test_q5_1_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_1_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q5_1_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q5_1;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: Q4_K_DecodeVectorization (SIMD correctness for Q4_K→Q8_0 decode)
add_executable(v2_test_q4_k_decode_vectorization unit/Test__Q4_K_DecodeVectorization.cpp)
target_link_libraries(v2_test_q4_k_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_K_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q4_k_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q4_K;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting;KQuant"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: Q5_K_DecodeVectorization (SIMD correctness for Q5_K→Q8_0 decode)
add_executable(v2_test_q5_k_decode_vectorization unit/Test__Q5_K_DecodeVectorization.cpp)
target_link_libraries(v2_test_q5_k_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_K_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_q5_k_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;Q5_K;Q8_0;SIMD;AVX2;AVX512;VectorizationCorrectness;FuzzTesting;KQuant"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ4_NL decode vectorization (lookup table based quantization, tests SIMD correctness)
add_executable(v2_test_iq4nl_decode_vectorization unit/Test__IQ4_NL_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq4nl_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_NL_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq4nl_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ4_NL;Q8_0;SIMD;AVX2;AVX512;LookupTable;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ4_XS decode vectorization (4-bit extra-small with per-sub-block scales)
add_executable(v2_test_iq4xs_decode_vectorization unit/Test__IQ4_XS_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq4xs_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_XS_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq4xs_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ4_XS;Q8_0;SIMD;AVX2;AVX512;LookupTable;SuperBlock;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ2_XXS decode vectorization (2-bit extra-extra-small with grid lookup)
add_executable(v2_test_iq2xxs_decode_vectorization unit/Test__IQ2_XXS_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq2xxs_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XXS_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq2xxs_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ2_XXS;Q8_0;SIMD;AVX2;AVX512;GridLookup;SuperBlock;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ2_XS decode vectorization (2-bit extra-small with grid lookup and scales)
add_executable(v2_test_iq2xs_decode_vectorization unit/Test__IQ2_XS_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq2xs_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XS_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq2xs_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ2_XS;Q8_0;SIMD;AVX2;AVX512;GridLookup;SuperBlock;ScalesArray;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ3_XXS decode_to_q8_0 vectorization (scalar/AVX2/AVX-512 parity)
add_executable(v2_test_iq3xxs_decode_vectorization unit/Test__IQ3_XXS_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq3xxs_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_XXS_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq3xxs_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ3_XXS;Q8_0;SIMD;AVX2;AVX512;GridLookup;SuperBlock;ScalesArray;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ2_S decode vectorization (scalar/AVX2/AVX-512 parity)
add_executable(v2_test_iq2s_decode_vectorization unit/Test__IQ2_S_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq2s_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_S_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq2s_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ2_S;Q8_0;SIMD;AVX2;AVX512;GridLookup;HighBits;ScalesArray;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ3_S decode vectorization (scalar/AVX2/AVX-512 parity)
add_executable(v2_test_iq3s_decode_vectorization unit/Test__IQ3_S_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq3s_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_S_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq3s_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ3_S;Q8_0;SIMD;AVX2;AVX512;GridLookup;SignsArray;ScalesArray;VectorizationCorrectness;FuzzTesting"
    MPI_PROCS 1  # CPU-only SIMD decode, single rank
)

# Test: IQ1_S decode_to_q8_0 vectorization (1-bit quantization with IQ1S_DELTA offset)
add_executable(v2_test_iq1s_decode_vectorization unit/Test__IQ1_S_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq1s_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_S_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq1s_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ1_S;Q8_0;SIMD;AVX2;AVX512;GridLookup;DeltaOffset;VectorizationCorrectness;FuzzTesting;OneBitQuantization"
    MPI_PROCS 1  # Single rank (no MPI needed for decode tests)
)

# Test: IQ1_M decode_to_q8_0 vectorization (1-bit quantization with multiple scales)
add_executable(v2_test_iq1m_decode_vectorization unit/Test__IQ1_M_DecodeVectorization.cpp)
target_link_libraries(v2_test_iq1m_decode_vectorization
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_M_DecodeVectorization
    COMMAND $<TARGET_FILE:v2_test_iq1m_decode_vectorization>
    LABELS "V2;Unit;TensorOperations;Quantization;IQ1_M;Q8_0;SIMD;AVX2;AVX512;GridLookup;MultipleScales;VectorizationCorrectness;FuzzTesting;OneBitQuantization"
    MPI_PROCS 1  # Single rank (no MPI needed for decode tests)
)

# Test: BF16PackedGemm (auto-tuned BF16×BF16→FP32 GEMM with micro-kernels)
# add_executable(v2_test_bf16_packed_gemm unit/Test__BF16PackedGemm.cpp)
# target_link_libraries(v2_test_bf16_packed_gemm
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_BF16PackedGemm
#     COMMAND $<TARGET_FILE:v2_test_bf16_packed_gemm>
#     LABELS "V2;Unit;Kernels;GEMM;BF16;AutoTuning;MicroKernels;CacheBlocking;CPU;AVX2;AVX512"
#     MPI_PROCS 1  # CPU-only kernel, single rank
# )

# Test: INT8PackedGemm (auto-tuned INT8×INT8→INT32 GEMM with AVX512 VNNI)
# add_executable(v2_test_int8_packed_gemm unit/Test__INT8PackedGemm.cpp)
# target_link_libraries(v2_test_int8_packed_gemm
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_INT8PackedGemm
#     COMMAND $<TARGET_FILE:v2_test_int8_packed_gemm>
#     LABELS "V2;Unit;Kernels;GEMM;INT8;VNNI;AutoTuning;MicroKernels;CacheBlocking;CPU;AVX512;AVX512VNNI;Quantization"
#     MPI_PROCS 1  # CPU-only kernel, single rank
# )

# Test: INT8GemmImpl (multi-ISA INT8 GEMM implementations - AVX512/AVX2/Scalar correctness)
# add_executable(v2_test_int8_gemm_impl unit/Test__INT8GemmImpl.cpp)
# target_link_libraries(v2_test_int8_gemm_impl
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_INT8GemmImpl
#     COMMAND $<TARGET_FILE:v2_test_int8_gemm_impl>
#     LABELS "V2;Unit;Kernels;GEMM;INT8;MultiISA;AVX512VNNI;AVX2;Scalar;ImplementationParity;CPU"
#     MPI_PROCS 1  # CPU-only multi-ISA test, single rank
# )

# Test: FP16GemmImpl (multi-ISA FP16×FP16→FP32 GEMM implementations - AVX512F/F16C/Scalar correctness)
add_executable(v2_test_fp16_gemm_impl unit/Test__FP16Gemm.cpp)
target_link_libraries(v2_test_fp16_gemm_impl
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FP16GemmImpl
    COMMAND $<TARGET_FILE:v2_test_fp16_gemm_impl>
    LABELS "V2;Unit;Kernels;GEMM;FP16;MultiISA;AVX512F;F16C;AVX2;Scalar;ImplementationParity;CPU;HalfPrecision"
    MPI_PROCS 1  # CPU-only multi-ISA test, single rank
)

# Test: FP16PackedGemm (autotuner integration with micro-kernel variants)
add_executable(v2_test_fp16_packed_gemm unit/Test__FP16PackedGemm.cpp)
target_link_libraries(v2_test_fp16_packed_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FP16PackedGemm
    COMMAND $<TARGET_FILE:v2_test_fp16_packed_gemm>
    LABELS "V2;Unit;Kernels;GEMM;FP16;AutoTuner;MicroKernels;CacheBlocking;PackedGEMM;CPU;HalfPrecision"
    MPI_PROCS 1  # CPU-only autotuner test, single rank
)

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

# Test: BPETokenizer (native tokenizer implementation)
add_executable(v2_test_bpe_tokenizer unit/Test__BPETokenizer.cpp)
target_link_libraries(v2_test_bpe_tokenizer 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_BPETokenizer 
    COMMAND $<TARGET_FILE:v2_test_bpe_tokenizer>
    LABELS "V2;Unit;Tokenization;BPE;NLP;TextProcessing"
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

# Test: FP32 Attention kernel parity test against PyTorch
add_executable(v2_test_fp32_attention_parity unit/Test__FP32AttentionParity.cpp)
target_link_libraries(v2_test_fp32_attention_parity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FP32AttentionParity 
    COMMAND $<TARGET_FILE:v2_test_fp32_attention_parity>
    LABELS "V2;Unit;Kernels;Attention;FP32;PyTorchParity;GroundTruth;MultiHead;GQA;CPU"
    MPI_PROCS 1  # Parity test is CPU-only, single rank
)

# Test: INT8 Attention kernel (multi-head attention with INT8 Q/K/V, INT32 accumulators)
# add_executable(v2_test_int8_attention unit/Test__INT8AttentionKernel.cpp)
# target_link_libraries(v2_test_int8_attention 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_INT8Attention 
#     COMMAND $<TARGET_FILE:v2_test_int8_attention>
#     LABELS "V2;Unit;Kernels;Attention;INT8;INT32;INT8Pipeline;Quantization;MultiHead;CPU"
#     MPI_PROCS 1  # Attention is CPU-only, single rank
# )

# Test: INT8 SwiGLU kernel (SwiGLU activation with INT8 gate/up projections)
add_executable(v2_test_int8_swiglu unit/Test__INT8SwiGLUKernel.cpp)
target_link_libraries(v2_test_int8_swiglu 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_INT8SwiGLU 
    COMMAND $<TARGET_FILE:v2_test_int8_swiglu>
    LABELS "V2;Unit;Kernels;SwiGLU;INT8;INT8Pipeline;Quantization;Activation;CPU"
    MPI_PROCS 1  # SwiGLU is CPU-only, single rank
)

# Test: INT8GemmKernel (OneDNN per-channel quantization)
add_executable(v2_test_int8_gemm unit/Test__INT8GemmKernel.cpp)
target_link_libraries(v2_test_int8_gemm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_INT8GemmKernel 
    COMMAND $<TARGET_FILE:v2_test_int8_gemm>
    LABELS "V2;Unit;Kernels;INT8;Quantization;GEMM;OneDNN;CPU"
    MPI_PROCS 1  # INT8 GEMM is CPU-only, single rank
)

# Test: INT8GemmKernel INT32 Output (full INT8 pipeline foundation) (DISABLED - missing INT8GemmKernel.h)
# add_executable(v2_test_int8_gemm_int32 unit/Test__INT8GemmKernel__INT32Output.cpp)
# target_link_libraries(v2_test_int8_gemm_int32 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_INT8GemmKernel_INT32Output 
#     COMMAND $<TARGET_FILE:v2_test_int8_gemm_int32>
#     LABELS "V2;Unit;Kernels;INT8;Quantization;GEMM;OneDNN;CPU;INT32;Pipeline"
#     MPI_PROCS 1  # INT8 GEMM is CPU-only, single rank
# )

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
add_v2_test(V2_Unit_IQ4NL_Views 
    COMMAND $<TARGET_FILE:v2_test_iq4nl_views>
    LABELS "V2;Unit;TensorOperations;IQ4_NL;Quantization;ViewSupport;RowSlice;IQuant"
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
add_executable(v2_test_q4_0_simd unit/tensors/Test__Q4_0Tensor.cpp)
target_link_libraries(v2_test_q4_0_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q4_0_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q4_0_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q4_0;Quantization;SIMD;AVX2;AVX512;Equivalency"
)

# Test: Q4_0 Fused Dequant+GEMM Parity
add_executable(v2_test_q4_0_gemm_parity unit/tensors/Test__Q4_0Gemm_DequantParity.cpp)
target_link_libraries(v2_test_q4_0_gemm_parity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    ${BLAS_LIBRARIES}
)
add_v2_test(V2_Unit_Q4_0_Gemm_DequantParity 
    COMMAND $<TARGET_FILE:v2_test_q4_0_gemm_parity>
    MPI_PROCS 1
    LABELS "V2;Unit;Kernels;GEMM;Q4_0;Quantization;DequantParity;FusedKernels"
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
add_executable(v2_test_q5_0_simd unit/tensors/Test__Q5_0Tensor.cpp)
target_link_libraries(v2_test_q5_0_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q5_0_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q5_0_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q5_0;Quantization;SIMD;AVX2;AVX512;Equivalency;HighBit"
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

# Test: Q8_0 SIMD Equivalency
add_executable(v2_test_q8_0_simd unit/tensors/Test__Q8_0Tensor.cpp)
target_link_libraries(v2_test_q8_0_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_0_SIMD 
    COMMAND $<TARGET_FILE:v2_test_q8_0_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Q8_0;Quantization;SIMD;AVX2;AVX512;Equivalency"
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
    LABELS "V2;Unit;TensorOperations;Q4_K;Quantization;SIMD;AVX2;AVX512;Equivalency;KQuant;SuperBlock;ScaleMin"
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

# Test: Qwen2 E2E Correctness (MPI Vectorized Attention Integration)





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

# Test: PipelineBase Precision Mode
# Tests WeightPrecision and ActivationPrecision configuration flow through PipelineBase
add_executable(v2_test_pipeline_precision_mode unit/pipelines/Test__PipelineBase_PrecisionMode.cpp)
target_link_libraries(v2_test_pipeline_precision_mode llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_PipelinePrecisionMode
    COMMAND $<TARGET_FILE:v2_test_pipeline_precision_mode>
    LABELS "V2;Unit;PipelineExecution;PrecisionConfiguration;WeightPrecision;ActivationPrecision"
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

# Test: IQ4_XS SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ4_XS SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ4_XS: 4-bit extra-small with per-sub-block scales, uses kvalues_iq4nl lookup
add_executable(v2_test_iq4xs_tensor unit/tensors/Test__IQ4_XSTensor.cpp)
target_link_libraries(v2_test_iq4xs_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ4_XSTensor 
    COMMAND $<TARGET_FILE:v2_test_iq4xs_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ4_XS;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;LookupTable;SubBlockScales"
)

# Test: IQ2_XXS SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ2_XXS SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ2_XXS: 2-bit extra-extra-small with grid lookup, uses iq2xxs_grid[256] table
add_executable(v2_test_iq2xxs_tensor unit/tensors/Test__IQ2_XXSTensor.cpp)
target_link_libraries(v2_test_iq2xxs_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XXSTensor 
    COMMAND $<TARGET_FILE:v2_test_iq2xxs_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ2_XXS;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;SignMask"
)

# Test: IQ2_XS SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ2_XS SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ2_XS: 2-bit extra-small with per-sub-block scales, uses iq2xs_grid[512] table (9-bit indices)
add_executable(v2_test_iq2xs_tensor unit/tensors/Test__IQ2_XSTensor.cpp)
target_link_libraries(v2_test_iq2xs_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_XSTensor 
    COMMAND $<TARGET_FILE:v2_test_iq2xs_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ2_XS;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;SubBlockScales;9BitIndex"
)

# Test: IQ3_XXS SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ3_XXS SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ3_XXS: 3-bit extra-extra-small with grid lookup, uses iq3xxs_grid[512] table
add_executable(v2_test_iq3xxs_tensor unit/tensors/Test__IQ3_XXSTensor.cpp)
target_link_libraries(v2_test_iq3xxs_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_XXSTensor 
    COMMAND $<TARGET_FILE:v2_test_iq3xxs_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ3_XXS;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;3Bit"
)

# Test: IQ2_S SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ2_S SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ2_S: 2-bit small with high bits and scales, uses iq2s_grid[1024] table (10-bit indices)
add_executable(v2_test_iq2s_tensor unit/tensors/Test__IQ2_STensor.cpp)
target_link_libraries(v2_test_iq2s_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ2_STensor 
    COMMAND $<TARGET_FILE:v2_test_iq2s_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ2_S;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;HighBits;10BitIndex"
)

# Test: IQ3_S SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ3_S SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ3_S: 3-bit small with signs and scales, uses iq3s_grid[512] table (9-bit indices)
add_executable(v2_test_iq3s_tensor unit/tensors/Test__IQ3_STensor.cpp)
target_link_libraries(v2_test_iq3s_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ3_STensor 
    COMMAND $<TARGET_FILE:v2_test_iq3s_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ3_S;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;9BitIndex;SignArray"
)

# Test: IQ1_M SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ1_M SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ1_M: 1-bit medium with packed scales and delta, uses iq1s_grid[2048] table (11-bit indices)
add_executable(v2_test_iq1m_tensor unit/tensors/Test__IQ1_MTensor.cpp)
target_link_libraries(v2_test_iq1m_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_MTensor 
    COMMAND $<TARGET_FILE:v2_test_iq1m_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ1_M;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;11BitIndex;PackedScales"
)

# Test: IQ1_S SIMD equivalency (scalar vs AVX2 vs AVX512)
# Purpose: Validate IQ1_S SIMD implementations produce identical results
# Expected: 8 tests total (3 equivalency + 5 edge cases)
# IQ1_S: 1-bit small with sub-scales and delta, uses iq1s_grid[2048] table (11-bit indices)
add_executable(v2_test_iq1s_tensor unit/tensors/Test__IQ1_STensor.cpp)
target_link_libraries(v2_test_iq1s_tensor 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ1_STensor 
    COMMAND $<TARGET_FILE:v2_test_iq1s_tensor>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;IQ1_S;Quantization;SIMD;AVX2;AVX512;Equivalency;IQQuant;GridLookup;11BitIndex;SubScales"
)

# Test: IQ SIMD equivalency tests for all IQ formats
# Purpose: Validate scalar decode implementations and prepare for SIMD optimization
# Coverage: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS
# Expected: 2-3 tests per format (basic decode, zero scale, determinism)
add_executable(v2_test_iq_simd unit/tensors/Test__IQ_SIMD.cpp)
target_link_libraries(v2_test_iq_simd 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_IQ_SIMD 
    COMMAND $<TARGET_FILE:v2_test_iq_simd>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Quantization;SIMD;IQVariants;GridQuantization;LookupTable"
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

# Test: IQ4_NL Alignment and Layout Diagnostics (DISABLED - outdated test expecting old API)
# add_executable(v2_test_iq4nl_alignment_layout unit/Test__IQ4_NL_Alignment_Layout.cpp)
# target_link_libraries(v2_test_iq4nl_alignment_layout
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_IQ4_NL_Alignment_Layout
#     COMMAND $<TARGET_FILE:v2_test_iq4nl_alignment_layout>
#     MPI_PROCS 1
#     LABELS "V2;Unit;TensorOperations;IQ4_NL;Diagnostics;Alignment;Layout;Performance"
#     FIXTURES_REQUIRED "V2_Models"
# )

# Test: Q8_0 Conversion Accuracy (all activation tensor types)
add_executable(v2_test_q8_0_conversion unit/Test__Q8_0_Conversion.cpp)
target_link_libraries(v2_test_q8_0_conversion
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Q8_0_Conversion
    COMMAND $<TARGET_FILE:v2_test_q8_0_conversion>
    MPI_PROCS 1
    LABELS "V2;Unit;TensorOperations;Quantization;Q8_0;Conversion;FP32;BF16;FP16;INT32"
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

# Test: GQAAttention
add_executable(v2_test_gqa_attention unit/pipelines/Test__GQAAttention.cpp)
target_link_libraries(v2_test_gqa_attention 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GQAAttention 
    COMMAND $<TARGET_FILE:v2_test_gqa_attention>
    LABELS "V2;Unit;PipelineExecution;Attention;GQA;MQA;MHA"
)

# Test: CPUAttention kernel interface
add_executable(v2_test_cpu_attention unit/kernels/Test__CPUAttention.cpp)
target_link_libraries(v2_test_cpu_attention 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_CPUAttention 
    COMMAND $<TARGET_FILE:v2_test_cpu_attention>
    LABELS "V2;Unit;Kernels;Attention;CPUKernel;InterfaceTest"
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

# Test: GPU Backend Memory Allocation/Transfer (CUDA/ROCm)
# Only build when GPU backend is enabled
if(HAVE_CUDA OR HAVE_ROCM)
    add_executable(v2_test_gpu_backend_memory unit/Test__GPUBackendMemory.cpp)
    target_link_libraries(v2_test_gpu_backend_memory 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_GPUBackendMemory 
        COMMAND $<TARGET_FILE:v2_test_gpu_backend_memory>
        LABELS "V2;Unit;GPUBackend;MemoryManagement;CUDA;ROCm;DataTransfer"
        MPI_PROCS 1
    )
endif()

# Test: CUDA IQ4_NL GEMM Kernel Correctness
# Only build when CUDA is enabled
if(HAVE_CUDA)
    add_executable(v2_test_cuda_gemm unit/Test__CUDAGemm.cpp)
    target_link_libraries(v2_test_cuda_gemm 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CUDAGemm 
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm>
        LABELS "V2;Unit;Kernels;GEMM;IQ4_NL;CUDA;Quantization"
        MPI_PROCS 1
    )
    
    # Test: CUDA GEMM Auto-Tuner
    add_executable(v2_test_cuda_gemm_autotuner unit/kernels/cuda/Test__CudaGemmAutoTuner.cpp)
    target_include_directories(v2_test_cuda_gemm_autotuner PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(v2_test_cuda_gemm_autotuner 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CudaGemmAutoTuner 
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm_autotuner>
        LABELS "V2;Unit;AutoTuning;CUDA;GEMM;Heuristics;ConfigGeneration;CacheManagement"
        MPI_PROCS 1
    )
    
    # Test: CUDA CuTe Tensor Core GEMM
    # Phase 1: Minimal template using CUTLASS CuTe for FP16 Tensor Cores
    add_executable(v2_test_cuda_gemm_cute unit/Test__CudaGemmCuTe.cpp)
    target_include_directories(v2_test_cuda_gemm_cute PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_cuda_gemm_cute 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CudaGemmCuTe 
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm_cute>
        LABELS "V2;Unit;Kernels;GEMM;IQ4_NL;CUDA;TensorCores;CuTe"
        MPI_PROCS 1
    )
    
    # Phase 3: Large tile optimization (128×128×64) for 2-3× speedup
    add_executable(v2_test_cuda_gemm_phase3 unit/Test__CudaGemmPhase3.cpp)
    target_include_directories(v2_test_cuda_gemm_phase3 PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_cuda_gemm_phase3 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CudaGemmPhase3 
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm_phase3>
        LABELS "V2;Unit;Kernels;GEMM;IQ4_NL;CUDA;TensorCores;CuTe;LargeTiles;Performance"
        MPI_PROCS 1
    )
    
    # Phase 3 Part 2: Pipelined large tiles for 900-1,000 GFLOPS
    add_executable(v2_test_cuda_gemm_phase3_pipelined unit/Test__CudaGemmPhase3Pipelined.cpp)
    target_include_directories(v2_test_cuda_gemm_phase3_pipelined PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_cuda_gemm_phase3_pipelined 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CudaGemmPhase3Pipelined 
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm_phase3_pipelined>
        LABELS "V2;Unit;Kernels;GEMM;IQ4_NL;CUDA;TensorCores;CuTe;LargeTiles;Pipelining;Performance"
        MPI_PROCS 1
    )
    
    # Phase 4: Quick wins (swizzled shared memory + cp.async) for 7.6 TFLOPS target
    add_executable(v2_test_cuda_gemm_phase4_quickwins unit/Test__CudaGemmPhase4QuickWins.cpp)
    target_include_directories(v2_test_cuda_gemm_phase4_quickwins PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_cuda_gemm_phase4_quickwins 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CudaGemmPhase4QuickWins 
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm_phase4_quickwins>
        LABELS "V2;Unit;Kernels;GEMM;IQ4_NL;CUDA;TensorCores;CuTe;SwizzledMemory;AsyncCopy;Performance"
        MPI_PROCS 1
    )
    
    # Phase 4: Profiling (manual timing breakdown)
    add_executable(v2_test_phase4_profiling unit/Test__Phase4Profiling.cpp)
    target_include_directories(v2_test_phase4_profiling PRIVATE
        ${CMAKE_SOURCE_DIR}/src/v2/tensors
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase4_profiling 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Phase4Profiling
        COMMAND $<TARGET_FILE:v2_test_phase4_profiling>
        LABELS "V2;Unit;Profiling;Performance;CUDA"
        MPI_PROCS 1
    )
    
    # Phase 5A: Streaming dequantization (overlap decode with MMA)
    add_executable(v2_test_cuda_gemm_phase5a unit/Test__CudaGemmPhase5A.cpp)
    target_include_directories(v2_test_cuda_gemm_phase5a PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_cuda_gemm_phase5a 
        llaminar2_core
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_CudaGemmPhase5A
        COMMAND $<TARGET_FILE:v2_test_cuda_gemm_phase5a>
        LABELS "V2;Unit;Kernels;GEMM;IQ4_NL;CUDA;TensorCores;CuTe;StreamingDequant;InstructionParallelism;Performance"
        MPI_PROCS 1
    )
    
    # Phase 5: JIT Auto-Tuning Sweep
    # Configuration space exploration for Phase 5 streaming dequant
    # Tests hypothesis that single buffering may outperform double buffering
    add_executable(v2_test_phase5_sweep cuda/Test__Phase5Sweep.cpp)
    target_include_directories(v2_test_phase5_sweep PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_sweep 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5Sweep
        COMMAND $<TARGET_FILE:v2_test_phase5_sweep>
        LABELS "V2;Performance;AutoTuning;JIT;CUDA;ConfigSpaceSweep;OccupancyOptimization"
        MPI_PROCS 1
    )
    
    # Test: Phase 5 JIT Parity
    # Validates JIT-compiled Phase 5A achieves same performance as nvcc baseline
    add_executable(v2_test_phase5_parity cuda/Test__Phase5Parity.cpp)
    target_include_directories(v2_test_phase5_parity PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_parity 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5Parity
        COMMAND $<TARGET_FILE:v2_test_phase5_parity>
        LABELS "V2;Performance;Parity;JIT;CUDA;Baseline;Phase5A"
        MPI_PROCS 1
    )
    
    # Test: Phase 5 SUB_K Sweep
    # Sweeps SUB_K values (16, 32, 64) to find optimal configuration for memory coalescing
    add_executable(v2_test_phase5_sub_k_sweep cuda/Test__Phase5_SUB_K_Sweep.cpp)
    target_include_directories(v2_test_phase5_sub_k_sweep PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_sub_k_sweep 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5_SUB_K_Sweep
        COMMAND $<TARGET_FILE:v2_test_phase5_sub_k_sweep>
        LABELS "V2;Performance;Sweep;SUB_K;CUDA;Phase5;Coalescing"
        MPI_PROCS 1
    )
    
    # Test: Phase 5 SUB_K=64 Profiling
    # Single-configuration test optimized for NCU profiling of SUB_K=64 coalescing
    add_executable(v2_test_phase5_sub_k_64_profile cuda/Test__Phase5_SUB_K_64_Profile.cpp)
    target_include_directories(v2_test_phase5_sub_k_64_profile PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_sub_k_64_profile 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5_SUB_K_64_Profile
        COMMAND $<TARGET_FILE:v2_test_phase5_sub_k_64_profile>
        LABELS "V2;Performance;Profile;SUB_K;CUDA;Phase5;Coalescing"
        MPI_PROCS 1
    )
    
    # Test: Phase 5 Transpose B Layout Benchmark
    # Validates NCU-guided hypothesis that transposing B from [N][K_blocks] to [K_blocks][N]
    # improves memory coalescing from 27% to 87-100% utilization (62% speedup potential)
    add_executable(v2_test_phase5_transpose_benchmark cuda/Test__Phase5_Transpose_Benchmark.cpp)
    target_include_directories(v2_test_phase5_transpose_benchmark PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_transpose_benchmark 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5_Transpose_Benchmark
        COMMAND $<TARGET_FILE:v2_test_phase5_transpose_benchmark>
        LABELS "V2;Performance;CUDA;Phase5;Coalescing;Transpose;MemoryLayout"
        MPI_PROCS 1
    )
    
    # Test: Phase 5 Transpose Large Problem
    # Larger problem size (2048×2048) for accurate timing of transpose benefit
    add_executable(v2_test_phase5_transpose_large cuda/Test__Phase5_Transpose_Large.cpp)
    target_include_directories(v2_test_phase5_transpose_large PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_transpose_large 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5_Transpose_Large
        COMMAND $<TARGET_FILE:v2_test_phase5_transpose_large>
        LABELS "V2;Performance;CUDA;Phase5;Coalescing;Transpose;LargeProblem"
        MPI_PROCS 1
    )
    
    # Test: Phase 5 Direct NVRTC (bypass Jitify)
    # Tests Phase 5 kernel compiled directly with NVRTC to isolate Jitify overhead
    add_executable(v2_test_phase5_direct_nvrtc cuda/Test__Phase5DirectNVRTC.cpp)
    target_include_directories(v2_test_phase5_direct_nvrtc PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase5_direct_nvrtc 
        llaminar2_core
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase5DirectNVRTC
        COMMAND $<TARGET_FILE:v2_test_phase5_direct_nvrtc>
        LABELS "V2;Performance;NVRTC;CUDA;Phase5;Debug"
        MPI_PROCS 1
    )
    
    # Phase 6: Int8 DP4A Kernel (llama.cpp-inspired) - Minimal compilation test
    # Replaces Tensor Cores with int8 DP4A for better memory coalescing
    # Expected: 50-90 TFLOPS (2.9-5.1× improvement over Phase 5's 17.5 TFLOPS)
    add_executable(v2_test_phase6_int8_minimal cuda/Test__Phase6_Int8_Minimal.cpp)
    target_include_directories(v2_test_phase6_int8_minimal PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase6_int8_minimal 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Phase6_Int8_Minimal
        COMMAND $<TARGET_FILE:v2_test_phase6_int8_minimal>
        LABELS "V2;Unit;CUDA;Phase6;Int8;DP4A;CompilationTest"
        MPI_PROCS 1
    )
    
    # Phase 6: Int8 DP4A Kernel - Functional correctness tests
    # Tests kernel correctness with CPU reference implementation
    # Validates: small matrices (4×4), medium matrices (64×64), identity patterns
    add_executable(v2_test_phase6_int8_functional cuda/Test__Phase6_Int8_Functional.cpp)
    target_include_directories(v2_test_phase6_int8_functional PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase6_int8_functional 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Phase6_Int8_Functional
        COMMAND $<TARGET_FILE:v2_test_phase6_int8_functional>
        LABELS "V2;Unit;CUDA;Phase6;Int8;DP4A;FunctionalTest;Correctness"
        MPI_PROCS 1
    )
    
    # Phase 6: Int8 DP4A Kernel - Debug test with intermediate value printing
    # Helps diagnose why GPU results differ from CPU reference
    add_executable(v2_test_phase6_int8_debug cuda/Test__Phase6_Int8_Debug.cpp)
    target_include_directories(v2_test_phase6_int8_debug PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase6_int8_debug 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Phase6_Int8_Debug
        COMMAND $<TARGET_FILE:v2_test_phase6_int8_debug>
        LABELS "V2;Unit;CUDA;Phase6;Int8;DP4A;Debug"
        MPI_PROCS 1
    )

    # Phase 6: Int8 DP4A Kernel - Performance benchmark
    # Measures TFLOPS on large matrices (2048×2048) and compares against Phase 5 baseline
    add_executable(v2_test_phase6_int8_performance cuda/Test__Phase6_Int8_Performance.cpp)
    target_include_directories(v2_test_phase6_int8_performance PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase6_int8_performance 
        llaminar2_core
        cuda_backend
        CUDA::nvrtc
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase6_Int8_Performance
        COMMAND $<TARGET_FILE:v2_test_phase6_int8_performance>
        LABELS "V2;Performance;CUDA;Phase6;Int8;DP4A;Benchmark"
        MPI_PROCS 1
    )

    # Phase 7: CUTLASS Int8 GEMM - Functional correctness tests (simplified)
    # COMMENTED OUT: API has changed, needs update
    # Uses NVIDIA CUTLASS library with direct IQ4_NL→INT8 conversion (no dequant/requant)
    # Expected performance: 50-100+ TFLOPS (vs Phase 6's 0.34 TFLOPS)
    # add_executable(v2_test_phase7_cutlass_functional cuda/Test__Phase7_CUTLASS_Functional_Simple.cpp)
    # target_include_directories(v2_test_phase7_cutlass_functional PRIVATE
    #     ${CMAKE_SOURCE_DIR}/kernels/cuda
    #     /opt/cutlass/include
    # )
    # target_link_libraries(v2_test_phase7_cutlass_functional 
    #     llaminar2_core
    #     cuda_backend
    #     CUDA::cudart
    #     CUDA::cuda_driver
    #     GTest::gtest
    #     GTest::gtest_main
    # )
    # add_v2_test(V2_Unit_Phase7_CUTLASS_Functional
    #     COMMAND $<TARGET_FILE:v2_test_phase7_cutlass_functional>
    #     LABELS "V2;Unit;CUDA;Phase7;CUTLASS;Int8;GEMM;Functional;DirectConversion"
    #     MPI_PROCS 1
    # )

    # Phase 7: CUTLASS Int8 GEMM - Simple correctness test (no IQ4_NL)
    # Direct int8×int8→int32 GEMM to verify CUTLASS integration works
    add_executable(v2_test_phase7_cutlass_simple cuda/Test__Phase7_CUTLASS_Simple.cu)
    target_include_directories(v2_test_phase7_cutlass_simple PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_test_phase7_cutlass_simple 
        CUDA::cudart
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_Phase7_CUTLASS_Simple
        COMMAND $<TARGET_FILE:v2_test_phase7_cutlass_simple>
        LABELS "V2;Unit;CUDA;Phase7;CUTLASS;Int8;GEMM;Simple"
        MPI_PROCS 1
    )

    # Phase 7: CUTLASS Int8 GEMM - Performance benchmark
    # Measures throughput (GFLOPS) for various matrix sizes
    # Expected: 50-100+ TFLOPS (vs Phase 6's 0.34 TFLOPS disaster)
    add_executable(v2_perf_phase7_cutlass cuda/Perf__Phase7_CUTLASS.cu)
    target_include_directories(v2_perf_phase7_cutlass PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
        /opt/cutlass/include
    )
    target_link_libraries(v2_perf_phase7_cutlass 
        llaminar2_core
        cuda_backend
        CUDA::cudart
        CUDA::cuda_driver
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_Phase7_CUTLASS
        COMMAND $<TARGET_FILE:v2_perf_phase7_cutlass>
        LABELS "V2;Performance;CUDA;Phase7;CUTLASS;Int8;GEMM;Throughput"
        MPI_PROCS 1
    )

    # Test: Driver API Launch Overhead
    # Measures the overhead difference between Driver API (cuLaunchKernel) and Runtime API (<<<>>>)
    # This test confirms that JIT kernels have higher launch overhead (~900 μs) due to Driver API
    add_executable(v2_test_driver_api_launch_overhead cuda/Test__DriverAPILaunchOverhead.cu)
    target_link_libraries(v2_test_driver_api_launch_overhead 
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_DriverAPILaunchOverhead
        COMMAND $<TARGET_FILE:v2_test_driver_api_launch_overhead>
        LABELS "V2;Performance;CUDA;DriverAPI;LaunchOverhead"
        MPI_PROCS 1
    )
    
    # Test: Jitify Overhead Analysis
    # Measures overhead at each stage: raw NVRTC, Jitify compilation, cached launch, parameter setup
    add_executable(v2_test_jitify_overhead cuda/Test__JitifyOverhead.cu)
    target_include_directories(v2_test_jitify_overhead PRIVATE
        ${CMAKE_SOURCE_DIR}/kernels/cuda
    )
    target_link_libraries(v2_test_jitify_overhead 
        llaminar2_core 
        cuda_backend
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Perf_JitifyOverhead
        COMMAND $<TARGET_FILE:v2_test_jitify_overhead>
        LABELS "V2;Performance;CUDA;Jitify;JIT;Overhead;LaunchPerformance"
        MPI_PROCS 1
    )
    
    # Test: IQ4_NL BF16 Decoder Correctness
    # Tests BF16 decoder method for IQ4_NL quantized tensors on Ampere+ GPUs
    add_executable(v2_test_iq4nl_bf16_decoder unit/Test__IQ4_NL_BF16_Decoder.cu)
    target_link_libraries(v2_test_iq4nl_bf16_decoder 
        llaminar2_core 
        cuda_backend
        GTest::gtest
        GTest::gtest_main
    )
    add_v2_test(V2_Unit_IQ4_NL_BF16_Decoder 
        COMMAND $<TARGET_FILE:v2_test_iq4nl_bf16_decoder>
        LABELS "V2;Unit;Kernels;IQ4_NL;BF16;CUDA;Quantization;Ampere"
        MPI_PROCS 1
    )
endif()

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
# add_executable(v2_test_vectorized_primitives Test__VectorizedPrimitives.cpp)
# target_link_libraries(v2_test_vectorized_primitives 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_VectorizedPrimitives 
#     COMMAND $<TARGET_FILE:v2_test_vectorized_primitives>
#     LABELS "V2;Unit;Kernels;VectorizedPrimitives;RoPE;RMSNorm;Softmax;AVX512;AVX2"
#     MPI_PROCS 1
# )

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
# add_executable(v2_test_cpu_kernels Test__CPUKernels.cpp)
# target_link_libraries(v2_test_cpu_kernels 
#     llaminar2_core 
#     ${OPENBLAS_LIBRARIES}
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_CPUKernels 
#     COMMAND $<TARGET_FILE:v2_test_cpu_kernels>
#     LABELS "V2;Unit;Kernels;CPU;ComputeBackends"
#     MPI_PROCS 1
# )

# Test: GEMM Auto-Tuner Integration
add_executable(v2_test_gemm_autotuner_integration unit/Test__GemmAutoTunerIntegration.cpp)
target_link_libraries(v2_test_gemm_autotuner_integration 
    llaminar2_core 
    ${OPENBLAS_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_GemmAutoTunerIntegration 
    COMMAND $<TARGET_FILE:v2_test_gemm_autotuner_integration>
    LABELS "V2;Unit;Kernels;AutoTuning;GEMM;QuantizedTensors"
    MPI_PROCS 1
)

# Test: GEMM Tiling Debug (isolate 2-tile column bug)
# DISABLED: Work-in-progress debugging test with known failures
# add_executable(v2_test_tiling_debug unit/Test__TilingDebug.cpp)
# target_link_libraries(v2_test_tiling_debug 
#     llaminar2_core 
#     ${OPENBLAS_LIBRARIES}
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_TilingDebug 
#     COMMAND $<TARGET_FILE:v2_test_tiling_debug>
#     LABELS "V2;Unit;Kernels;GEMM;Tiling;Debug"
#     MPI_PROCS 1
# )

# Test: GEMM Auto-Tuner Correctness (validates all kernel variants)
# add_executable(v2_test_gemm_autotuner_correctness unit/Test__GemmAutoTunerCorrectness.cpp)
# target_link_libraries(v2_test_gemm_autotuner_correctness 
#     llaminar2_core 
#     ${OPENBLAS_LIBRARIES}
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_GemmAutoTunerCorrectness 
#     COMMAND $<TARGET_FILE:v2_test_gemm_autotuner_correctness>
#     LABELS "V2;Unit;Kernels;AutoTuning;GEMM;Correctness;AllVariants"
#     MPI_PROCS 1
# )

# Test: ML-Based Tile Autotuner (Phase 3, November 2025)
# COMMENTED OUT: Requires generated ML model files (GemmAutoTunerML.h) that don't exist yet
# add_executable(v2_test_gemm_autotuner_ml unit/Test__GemmAutoTunerML.cpp)
# target_link_libraries(v2_test_gemm_autotuner_ml 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_GemmAutoTunerML 
#     COMMAND $<TARGET_FILE:v2_test_gemm_autotuner_ml>
#     LABELS "V2;Unit;Kernels;AutoTuning;MachineLearning;TilePrediction;Phase3"
#     MPI_PROCS 1
# )

# Test: GEMM Template Infrastructure (Phase 1 validation)
# add_executable(v2_test_gemm_template_infrastructure unit/Test__GemmTemplateInfrastructure.cpp)
# target_link_libraries(v2_test_gemm_template_infrastructure 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_GemmTemplateInfrastructure 
#     COMMAND $<TARGET_FILE:v2_test_gemm_template_infrastructure>
#     LABELS "V2;Unit;Kernels;GEMM;Templates;SimdTraits;MicroKernel"
#     MPI_PROCS 1
# )

# Test: GEMM Macro vs Template (Phase 2 validation - correctness + performance)
# add_executable(v2_test_gemm_macro_vs_template unit/Test__GemmMacroVsTemplate.cpp)
# target_link_libraries(v2_test_gemm_macro_vs_template 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_GemmMacroVsTemplate 
#     COMMAND $<TARGET_FILE:v2_test_gemm_macro_vs_template>
#     LABELS "V2;Unit;Kernels;GEMM;Templates;Validation;Correctness;Performance"
#     MPI_PROCS 1
# )

# Test: MicroKernel Adapter Packing (cache-blocked buffer management) (DISABLED - missing GemmMicroKernelAdapter.h)
# add_executable(v2_test_microkernel_adapter_packing unit/Test__MicroKernelAdapter_Packing.cpp)
# target_link_libraries(v2_test_microkernel_adapter_packing 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_MicroKernelAdapterPacking
#     COMMAND $<TARGET_FILE:v2_test_microkernel_adapter_packing>
#     LABELS "V2;Unit;Kernels;MicroKernel;CacheBlocking;Packing;BufferManagement"
#     MPI_PROCS 1
# )

# Test: MicroKernel Padding (unit test for required buffer size)
add_executable(v2_test_microkernel_padding unit/Test__MicroKernelPadding.cpp)
target_link_libraries(v2_test_microkernel_padding 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MicroKernelPadding
    COMMAND $<TARGET_FILE:v2_test_microkernel_padding>
    LABELS "V2;Unit;Kernels;MicroKernel;Padding;BufferManagement;SIMD;BoundsChecking"
    MPI_PROCS 1
)

# Test: Micro-Kernel Write Bounds Verification
add_executable(v2_test_microkernel_bounds unit/Test__MicroKernelBounds.cpp)
target_link_libraries(v2_test_microkernel_bounds 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MicroKernelBounds
    COMMAND $<TARGET_FILE:v2_test_microkernel_bounds>
    LABELS "V2;Unit;Kernels;MicroKernel;BoundsChecking;WriteVerification"
    MPI_PROCS 1
)

# Test: Micro-Kernel Mathematical Correctness
# Validates numerical correctness against reference implementations (naive GEMM, OpenBLAS)
# Tests: identity, zero, diagonal, random matrices, edge cases, all tile variants
# add_executable(v2_test_microkernel_correctness unit/Test__MicroKernelCorrectness.cpp)
# target_link_libraries(v2_test_microkernel_correctness 
#     llaminar2_core 
#     ${OPENBLAS_LIBRARIES}
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_MicroKernelCorrectness
#     COMMAND $<TARGET_FILE:v2_test_microkernel_correctness>
#     LABELS "V2;Unit;Kernels;MicroKernel;GEMM;Correctness;NumericalAccuracy;Parity"
#     MPI_PROCS 1
# )

# Test: GEMM Debug (minimal test to debug kernel issues)
# NOTE: Commented out - Test__GemmDebug.cpp not found
# add_executable(v2_test_gemm_debug unit/Test__GemmDebug.cpp)
# target_link_libraries(v2_test_gemm_debug 
#     llaminar2_core 
#     ${OPENBLAS_LIBRARIES}
#     GTest::gtest
#     GTest::gtest_main
# )
# add_v2_test(V2_Unit_GemmDebug 
#     COMMAND $<TARGET_FILE:v2_test_gemm_debug>
#     LABELS "V2;Unit;Kernels;GEMM;Debug"
#     MPI_PROCS 1
# )

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

# MPI Staging Tests - Validates MPIStager utility for GPU↔Host transfers
add_executable(v2_test_mpi_staging 
    integration/Test__MPIStaging.cpp
)
target_link_libraries(v2_test_mpi_staging 
    llaminar2_core 
    GTest::gtest  # Provides its own MPI-aware main()
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_MPIStaging 
    COMMAND $<TARGET_FILE:v2_test_mpi_staging>
    LABELS "V2;Integration;MPI;Staging;CPUTensors;ErrorHandling;Allreduce;Allgather"
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

# Test: MicroKernel AutoTuner Integration (October 2025)
# add_executable(v2_test_microkernel_autotuner_integration 
#     integration/test_microkernel_autotuner_integration.cpp)
# target_link_libraries(v2_test_microkernel_autotuner_integration 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
add_v2_test(V2_Integration_MicroKernelAutoTuner
    COMMAND $<TARGET_FILE:v2_test_microkernel_autotuner_integration>
    MPI_PROCS 1
    LABELS "V2;Integration;MicroKernel;AutoTuner;GEMM;Performance"
)

# Test: Dequant Equivalency - Verify IBlockDecoder matches llama.cpp dequantization (October 2025)
add_executable(v2_test_dequant_equivalency
    integration/Test__DequantEquivalency.cpp
)

# Link against llaminar2_core, GTest, and ggml from llama.cpp
target_link_libraries(v2_test_dequant_equivalency
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    ggml  # llama.cpp's ggml library provides dequant functions
)

# Include llama.cpp headers for GGML dequant functions
target_include_directories(v2_test_dequant_equivalency PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../external/llama.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../external/llama.cpp/ggml/include
    ${CMAKE_CURRENT_LIST_DIR}/../../external/llama.cpp/ggml/src
)

add_v2_test(V2_Integration_DequantEquivalency
    COMMAND $<TARGET_FILE:v2_test_dequant_equivalency>
    MPI_PROCS 1
    LABELS "V2;Integration;Quantization;IBlockDecoder;LlamaCppParity;Correctness"
)

# Test: ML AutoTuner E2E - Validate ML-based tile predictor integration (November 2025)
# COMMENTED OUT: Requires generated ML model files (not yet available)
# if(HAVE_CUDA)
#     add_executable(v2_test_ml_autotuner_e2e
#         integration/Test__MLAutoTunerE2E.cpp
#     )
#     
#     target_link_libraries(v2_test_ml_autotuner_e2e
#         llaminar2_core
#         cuda_backend
#         GTest::gtest
#         GTest::gtest_main
#         CUDA::cudart
#     )
#     
#     add_v2_test(V2_Integration_MLAutoTuner
#         COMMAND $<TARGET_FILE:v2_test_ml_autotuner_e2e>
#         MPI_PROCS 1
#         LABELS "V2;Integration;AutoTuner;MachineLearning;GEMM;Performance;Correctness"
#     )
# endif()

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

# Qwen2 FP32 Parity (PyTorch ground truth validation)
# Requires: PyTorch snapshots generated by python/reference/generate_qwen2_pipeline_snapshots.py
# Requires: cnpy library for NumPy .npz file loading
find_package(ZLIB)
if(ZLIB_FOUND)
    # Get the workspace root (parent of src/v2)
    get_filename_component(WORKSPACE_ROOT "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)
    
    # E2E parity test requires ENABLE_PIPELINE_SNAPSHOTS (Debug or E2ERelease builds)
    # Skip in standard Release builds to avoid linking errors
    if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR ENABLE_SNAPSHOTS)
        add_executable(v2_test_qwen2_fp32_parity 
            e2e/Test__Qwen2FP32Parity.cpp
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
    else()
        message(STATUS "V2: E2E parity test disabled (snapshots not available in Release build)")
        message(STATUS "     Use -DCMAKE_BUILD_TYPE=E2ERelease or -DENABLE_SNAPSHOTS=ON to enable")
    endif()
else()
    message(WARNING "ZLIB not found - Qwen2 FP32 parity test disabled (requires cnpy for .npz loading)")
endif()
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
    # ARG_COMMAND already contains the full executable path (from $<TARGET_FILE:...>)
    add_test(NAME ${TEST_NAME} COMMAND ${MPI_CMD} ${ARG_COMMAND})
    
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

# Performance Test: Q8_0 × Q8_0 GEMM (8×8 microkernel with inline compensation)
# Benchmarks Q8_0 quantized GEMM using standard Q8_0Tensor interface
# Test cases: SingleToken, SmallBatch, LargePrefill
# Measures: Time, GFLOPS, comparison with dense INT8 baseline
add_executable(v2_perf_q8_0_gemm performance/Perf__Q8_0Gemm.cpp)
target_link_libraries(v2_perf_q8_0_gemm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_q8_0_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Q8_0_GEMM
    COMMAND $<TARGET_FILE:v2_perf_q8_0_gemm>
    LABELS "V2;Performance;GEMM;Q8_0;Quantization;Throughput"
    MPI_PROCS 1  # Single rank (no MPI distribution for this kernel)
)

# Performance Test: Q8_1 × Q8_0 GEMM (32×64 microkernel with dpbusd + compensation)
# Benchmarks Q8_1 activations × Q8_0 weights using dpbusd instruction
# Test cases: LargeBatchedPrefill (4096 tokens)
# Measures: Time, GFLOPS, comparison with Q8_0×Q8_0 baseline
add_executable(v2_perf_q8_1_gemm performance/Perf__Q8_1Gemm.cpp)
target_link_libraries(v2_perf_q8_1_gemm 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_q8_1_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Q8_1_GEMM
    COMMAND $<TARGET_FILE:v2_perf_q8_1_gemm>
    LABELS "V2;Performance;GEMM;Q8_1;Quantization;Throughput;dpbusd"
    MPI_PROCS 1  # Single rank (no MPI distribution for this kernel)
)

# Alternative non-MPI test for Q8_1 GEMM (for ComprehensiveParameterSweep)
add_test(NAME V2_Perf_Q8_1_GEMM_NoMPI
    COMMAND $<TARGET_FILE:v2_perf_q8_1_gemm> --gtest_filter=Q8_1GemmPerformance.ComprehensiveParameterSweep
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../.."
)
set_tests_properties(V2_Perf_Q8_1_GEMM_NoMPI PROPERTIES
    LABELS "V2;Performance;GEMM;Q8_1;ParameterSweep;NoMPI"
    ENVIRONMENT "LLAMINAR_LOG_LEVEL=INFO"
)

# Performance Test: Q8_0 GEMM Microkernel Size Comparison
# Benchmarks different microkernel sizes (8×8, 12×12, 16×16, 8×16)
# Test cases: Large prefill workload (M=4096, N=896, K=896)
# Measures: Time, GFLOPS, speedup vs 8×8 baseline
add_executable(v2_perf_q8_0_gemm_microkernel_size performance/Perf__Q8_0Gemm_MicrokernelSize.cpp)
target_link_libraries(v2_perf_q8_0_gemm_microkernel_size 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_q8_0_gemm_microkernel_size PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Q8_0_GEMM_MicrokernelSize
    COMMAND $<TARGET_FILE:v2_perf_q8_0_gemm_microkernel_size>
    LABELS "V2;Performance;GEMM;Q8_0;MicrokernelOptimization;Tuning"
    MPI_PROCS 1  # Single rank (no MPI distribution for this kernel)
)

# Performance Test: Q8_0 Conversion with SIMD Variants
# Benchmarks vectorized Q8_0 quantization (AVX512/AVX2/scalar) and OpenMP scaling
# Test cases: SIMD variants, OpenMP thread scaling, all tensor types, large-scale conversion
# Measures: Time, ops/sec, SIMD speedup, parallel efficiency, throughput (MB/s)
add_executable(v2_perf_q8_0_conversion performance/Perf__Q8_0_Conversion.cpp)
target_link_libraries(v2_perf_q8_0_conversion 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_q8_0_conversion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Q8_0_Conversion
    COMMAND $<TARGET_FILE:v2_perf_q8_0_conversion>
    LABELS "V2;Performance;Quantization;Q8_0;SIMD;AVX512;AVX2;OpenMP;Vectorization;Parallelization"
    MPI_PROCS 1  # Single rank (tests CPU SIMD + OpenMP)
)

# Performance Test: INT8 GEMM M-Scaling Analysis
# Benchmarks how INT8 GEMM performance scales with increasing M dimension
# Test cases: M from 128 to 4096, single-thread vs multi-thread, thread count scaling
# Identifies: Parallelization breakpoint, optimal thread count, efficiency metrics
add_executable(v2_perf_int8_gemm_m_scaling performance/Perf__Int8Gemm_MScaling.cpp)
target_link_libraries(v2_perf_int8_gemm_m_scaling 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_int8_gemm_m_scaling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Int8Gemm_MScaling
    COMMAND $<TARGET_FILE:v2_perf_int8_gemm_m_scaling>
    LABELS "V2;Performance;GEMM;INT8;Scaling;Parallelization;OpenMP;Efficiency"
    MPI_PROCS 1  # Single rank (tests OpenMP scaling)
)

# Performance Test: INT8 GEMM Prefetch Distance Tuning
# Benchmarks impact of OneDNN-style aggressive prefetching vs conservative vs none
# Test cases: Prefetch distances (0,0,0), (64,64,32), (160,128,64), (256,192,96)
# Expected: 2-4× improvement from OneDNN prefetch (160,128,64) over no prefetch
# Validates: Memory bandwidth utilization improvement (3.6% → 10-15%)
add_executable(v2_perf_int8_gemm_prefetch performance/Perf__Int8Gemm_Prefetch.cpp)
target_link_libraries(v2_perf_int8_gemm_prefetch 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_int8_gemm_prefetch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Int8Gemm_Prefetch
    COMMAND $<TARGET_FILE:v2_perf_int8_gemm_prefetch>
    LABELS "V2;Performance;GEMM;INT8;Prefetch;MemoryBandwidth;OneDNN;Optimization"
    MPI_PROCS 1  # Single rank (tests prefetch impact)
)

# Performance Test: INT8 GEMM Microkernel Size Comparison
# Compares different microkernel dimensions (48×8 vs 32×8 vs 16×8 vs 16×16)
# Test cases: M-scaling (512, 2048, 8192), single-thread vs multi-thread
# Expected: 48×8 (OneDNN's choice) provides 20-30% improvement over 16×16
# Validates: Better M-dimension amortization and register utilization
add_executable(v2_perf_int8_gemm_microkernel_size performance/Perf__Int8Gemm_MicrokernelSize.cpp)
target_link_libraries(v2_perf_int8_gemm_microkernel_size 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_int8_gemm_microkernel_size PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Int8Gemm_MicrokernelSize
    COMMAND $<TARGET_FILE:v2_perf_int8_gemm_microkernel_size>
    LABELS "V2;Performance;GEMM;INT8;Microkernel;OneDNN;RegisterAllocation;MAmortization"
    MPI_PROCS 1  # Single rank (tests microkernel sizes)
)

# Performance Test: IQ4_NL GEMM Implementation Comparison
# Compares IQ4_NLQuantizedGemm (specialized) vs AutoTunedQuantizedGemm (generic)
# Test cases: Single token, small/medium/large batch, Q-projection and FFN shapes
# Measures: Time, GFLOPS, relative speedup between implementations
add_executable(v2_perf_iq4nl_gemm_comparison performance/Perf__IQ4_NL_GEMM_Comparison.cpp)
target_link_libraries(v2_perf_iq4nl_gemm_comparison 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_iq4nl_gemm_comparison PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_IQ4NL_GEMM_Comparison
    COMMAND $<TARGET_FILE:v2_perf_iq4nl_gemm_comparison>
    LABELS "V2;Performance;GEMM;IQ4_NL;Comparison;AutoTuner"
    MPI_PROCS 2  # Match baseline configuration
)

# Performance Test: Fused GEMM+Softmax (Stage 1 Optimization) (DISABLED - missing FusedGemmSoftmax.h)
# Validates Stage 1 fusion claims: 47% memory reduction, 5-15% speedup
# Test cases: seq_len 128/256/512/1024/2048, causal masking, Qwen 7B config
# add_executable(v2_perf_fused_gemm_softmax performance/Perf__FusedGemmSoftmax.cpp)
# target_link_libraries(v2_perf_fused_gemm_softmax 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# set_target_properties(v2_perf_fused_gemm_softmax PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )
# 
# add_v2_perf_test(V2_Perf_FusedGemmSoftmax
#     COMMAND $<TARGET_FILE:v2_perf_fused_gemm_softmax>
#     LABELS "V2;Performance;Attention;FusedKernel;CacheOptimization;MemoryEfficiency"
#     MPI_PROCS 1  # Attention is single-rank (no MPI distribution)
# )

# Performance Test: Multi-Precision GEMM (FP32/BF16/FP16/INT8) (DISABLED - missing GemmKernelTemplate.h)
# Compares pure GEMM performance across all precision types (symmetric, not mixed)
# Test cases: Small/Medium/Large batch sizes (32/128/512 tokens)
# Measures: GFLOPS, bandwidth, conversion overhead, efficiency vs FP32 baseline
# add_executable(v2_perf_multiprecision_gemm performance/Perf__MultiPrecision_GEMM.cpp)
# target_link_libraries(v2_perf_multiprecision_gemm
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
#     MPI::MPI_CXX
#     OpenMP::OpenMP_CXX
#     ${BLAS_LIBRARIES}
# )
# set_target_properties(v2_perf_multiprecision_gemm PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )
# 
# add_v2_perf_test(V2_Perf_MultiPrecisionGEMM
#     COMMAND $<TARGET_FILE:v2_perf_multiprecision_gemm>
#     LABELS "V2;Performance;GEMM;MultiPrecision;FP32;BF16;FP16;INT8;Throughput"
#     MPI_PROCS 2  # Symmetric multi-rank for MPI barrier synchronization
# )

# Performance Test: BF16 Conversion Profiling (for linux perf analysis)
# Isolates BF16 conversion overhead for detailed profiling with perf record/report/annotate
# Tests: Pure conversion time vs GEMM time to identify bottlenecks
add_executable(v2_perf_bf16_profile performance/Perf__BF16_Profile.cpp)
target_link_libraries(v2_perf_bf16_profile 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
set_target_properties(v2_perf_bf16_profile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_BF16_Profile
    COMMAND $<TARGET_FILE:v2_perf_bf16_profile>
    LABELS "V2;Performance;BF16;Conversion;Profiling;Bandwidth"
    MPI_PROCS 2  # Match production configuration
)

# Performance Test: Auto-tuned GEMM Cache Profiling
# Micro-benchmark for auto-tuned IQ4_NL GEMM with L1/L2/L3 cache profiling
# Test cases: Prefill batch sizes 32/128/512/1024/2048/4096 tokens
# Profiles: Cache miss rates, IPC/CPI, memory bandwidth, GFLOPS
# Uses: Linux perf_event_open for hardware performance counters
add_executable(v2_perf_autotuned_gemm_cache performance/Perf__AutoTunedGemm_CacheProfiling.cpp)
target_link_libraries(v2_perf_autotuned_gemm_cache 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_autotuned_gemm_cache PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_AutoTunedGemm_CacheProfiling
    COMMAND $<TARGET_FILE:v2_perf_autotuned_gemm_cache>
    LABELS "V2;Performance;GEMM;AutoTuner;CacheProfiling;L1;L2;L3;IQ4_NL;Prefill"
    MPI_PROCS 1  # Single rank for pure performance (no MPI overhead)
)

# Performance Test: Exhaustive Autotuner Sweep
# Tests ALL 1225 microkernel configurations to find globally optimal settings
# Sweeps entire parameter space: ISA × MR × NR × UNROLL_K × PREFETCH_DIST
# Goal: Bypass autotuner heuristic and find true performance ceiling
add_executable(v2_perf_exhaustive_autotuner_sweep performance/Perf__Exhaustive_Autotuner_Sweep.cpp)
target_link_libraries(v2_perf_exhaustive_autotuner_sweep
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_exhaustive_autotuner_sweep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_ExhaustiveAutotunerSweep
    COMMAND $<TARGET_FILE:v2_perf_exhaustive_autotuner_sweep>
    LABELS "V2;Performance;GEMM;IQ4_NL;AutoTuner;Optimization;ConfigurationSweep"
    MPI_PROCS 2  # 2 ranks to distribute workload, each with OpenMP parallelization
)

# Performance Test: Integer GEMM Configuration Space Sweep (ML Training Data Collection)
# Comprehensive sweep of ALL 8000 Integer GEMM kernel configurations across realistic
# Qwen 0.5B workloads to gather training data for ML-based kernel auto-tuning.
#
# Test scope:
# - Configurations: 8000 variants (ISA × MR × UNROLL_K × PREFETCH_DIST × MC × KC × NC)
# - Workloads: Qwen 0.5B Q/K/V/O projections, FFN gate/up/down
# - Batch sizes: Single token (1), Small (8), Medium (32), Large (128)
# - Metrics: INT8 GFLOPS, time (ms)
# - Top-K profiling: Re-run top 10 configs per workload with perf stats
#   (L1/L2/L3 cache misses, instructions, cycles, IPC)
# Performance Test: Integer GEMM Minimal Profiling Harness
# Focused benchmark for iterative bug fixing and profiling.
# Reproduces the catastrophic performance bug (1.1% efficiency).
#
# Purpose:
#   - Rapid iteration on performance fixes
#   - Clean profiling with perf/vtune/gdb
#   - Test individual configurations without full sweep overhead
#
# Usage:
#   # Default (128×896×896, MR=16, shows bug)
#   ./performance/v2_perf_integer_gemm_minimal
#
#   # Test MR=1 (less overhead)
#   ./performance/v2_perf_integer_gemm_minimal --mr 1
#
#   # Profile with perf
#   perf record -g ./performance/v2_perf_integer_gemm_minimal --m 512 --iters 100
#   perf report
#
#   # Large matrix for sustained profiling
#   ./performance/v2_perf_integer_gemm_minimal --m 512 --n 4096 --k 4096 --iters 1000
#
add_executable(v2_perf_integer_gemm_minimal performance/cpu/kernels/gemm/Perf__IntegerGEMM_Minimal.cpp)
target_link_libraries(v2_perf_integer_gemm_minimal
    llaminar2_core
)
set_target_properties(v2_perf_integer_gemm_minimal PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

# NOTE: Not registered as CTest (no add_v2_perf_test) - this is a manual profiling tool
# Run directly: ./build_v2_release/performance/v2_perf_integer_gemm_minimal

# Performance Test: Qwen 0.5B Pipeline Profiling
# Benchmarks representative operations from Qwen 0.5B across decode and prefill scenarios.
# Tests 16 operations covering Q/K/V projections, attention output, and FFN layers
# with appropriate kernel configurations per operation size.
add_executable(v2_perf_integer_gemm_qwen_profile performance/cpu/kernels/gemm/Perf__IntegerGEMM_QwenProfile.cpp)
target_link_libraries(v2_perf_integer_gemm_qwen_profile
    llaminar2_core
)
set_target_properties(v2_perf_integer_gemm_qwen_profile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

# Register as CTest with optimal OMP settings
add_v2_perf_test(V2_Perf_IntegerGEMM_QwenProfile
    COMMAND $<TARGET_FILE:v2_perf_integer_gemm_qwen_profile> --simd
    LABELS "V2;Performance;Kernels;GEMM;IntegerGEMM;QwenProfile;AVX512VNNI"
    MPI_PROCS 1  # Single rank for pure performance measurement
)

# Performance Test: Register-Blocked INT8 GEMM (Phase 1 - OneDNN Architecture)
# Implements 6×16 microkernel with register blocking following OneDNN design
# Target: 2,000-3,000 GOPS (4-6× speedup over baseline 472 GOPS)
add_executable(v2_perf_register_blocked_int8_gemm performance/cpu/kernels/gemm/Perf__RegisterBlockedInt8Gemm.cpp)
target_link_libraries(v2_perf_register_blocked_int8_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_register_blocked_int8_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

# Register as CTest with optimal OMP settings
add_v2_perf_test(V2_Perf_RegisterBlockedInt8Gemm
    COMMAND $<TARGET_FILE:v2_perf_register_blocked_int8_gemm>
    LABELS "V2;Performance;Kernels;GEMM;RegisterBlocked;Phase1;OneDNN;AVX512VNNI"
    MPI_PROCS 1  # Single rank for pure performance measurement
)

# ============================================================================
# Phase 1: Register-Blocked Microkernel (6×16 tiles)
# ============================================================================
# OneDNN-inspired optimization: Register blocking for 31× reduction in memory traffic
# Target: 2,000-3,000 GOPS (4-6× speedup over baseline 472 GOPS)
# See: src/v2/kernels/cpu/gemm_v2/INT8_GEMM_QUICK_ACTION_PLAN.md
add_executable(v2_perf_register_blocked_microkernel performance/cpu/kernels/gemm/Perf__RegisterBlockedMicroKernel.cpp)
target_link_libraries(v2_perf_register_blocked_microkernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_register_blocked_microkernel PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

# Register as CTest
add_v2_perf_test(V2_Perf_RegisterBlockedMicroKernel
    COMMAND $<TARGET_FILE:v2_perf_register_blocked_microkernel>
    LABELS "V2;Performance;Kernels;GEMM;Phase1;RegisterBlocking;AVX512VNNI"
    MPI_PROCS 1
)

# Performance Test: Simple INT8 GEMM Baseline (No Abstractions)
# Clean-sheet design to establish baseline performance without microkernel complexity.
# Tests pure VNNI performance on Qwen 0.5B operations with minimal overhead.
add_executable(v2_perf_simple_int8_gemm performance/cpu/kernels/gemm/Perf__SimpleInt8Gemm.cpp)
target_link_libraries(v2_perf_simple_int8_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_simple_int8_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_SimpleInt8Gemm
    COMMAND $<TARGET_FILE:v2_perf_simple_int8_gemm>
    LABELS "V2;Performance;Kernels;GEMM;IntegerGEMM;Baseline;SimpleDesign;AVX512VNNI"
    MPI_PROCS 1
)

# Performance Test: Simple INT8 GEMM - FFN_down Prefill-4096 (Optimization Target)
# Focused benchmark for cache blocking optimization.
# Single test case: M=4096, N=896, K=4864 (35.6B ops)
# Target: 6,610 GOPS (OneDNN baseline, 84% efficiency)
add_executable(v2_perf_simple_int8_gemm_ffn_down_4096 performance/cpu/kernels/gemm/Perf__SimpleInt8Gemm_FFNDown4096.cpp)
target_link_libraries(v2_perf_simple_int8_gemm_ffn_down_4096
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_simple_int8_gemm_ffn_down_4096 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_SimpleInt8Gemm_FFNDown4096
    COMMAND $<TARGET_FILE:v2_perf_simple_int8_gemm_ffn_down_4096>
    LABELS "V2;Performance;Kernels;GEMM;IntegerGEMM;FFNDown;Prefill4096;CacheBlocking;OptimizationTarget"
    MPI_PROCS 1
)

# Performance Test: Packed INT8 GEMM - Pre-expansion Strategy
# Tests whether pre-packing B matrix into plain INT8 format improves performance
# vs decoding Q8 blocks on-the-fly.
add_executable(v2_perf_packed_int8_gemm performance/cpu/kernels/gemm/Perf__PackedInt8Gemm.cpp)
target_link_libraries(v2_perf_packed_int8_gemm
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_packed_int8_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_PackedInt8Gemm
    COMMAND $<TARGET_FILE:v2_perf_packed_int8_gemm>
    LABELS "V2;Performance;Kernels;GEMM;IntegerGEMM;DataPacking;PreExpansion;FFNDown"
    MPI_PROCS 1
)

# Performance Test: Integer GEMM Tile Size Parameter Sweep
# OBSOLETE: Comprehensive sweep of MR, UNROLL_K, PREFETCH_DIST, cache blocking params
# Superseded by Perf__IntegerGEMM_FullSweep_Registry.cpp (registry-based, no template explosion)
# Conclusion: MR=8 found optimal
# add_executable(v2_perf_integer_gemm_tile_sweep performance/cpu/kernels/gemm/Perf__IntegerGEMM_TileSweep.cpp.old)
# target_link_libraries(v2_perf_integer_gemm_tile_sweep
#     llaminar2_core
#     GTest::gtest
#     GTest::gtest_main
# )
# set_target_properties(v2_perf_integer_gemm_tile_sweep PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )

# NOTE: Not registered as CTest - outputs CSV for analysis
# Run: ./build_v2_release/performance/v2_perf_integer_gemm_tile_sweep --gtest_filter="*QuickSweep" > sweep_results.csv

# Complete configuration space sweep for autotuner training data
# Tests all parameter combinations across all Qwen workloads
# REGISTRY VERSION: Uses IntegerGemmKernelRegistry for fast compilation
add_executable(v2_perf_integer_gemm_full_sweep performance/cpu/kernels/gemm/Perf__IntegerGEMM_FullSweep_Registry.cpp)
target_link_libraries(v2_perf_integer_gemm_full_sweep
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_integer_gemm_full_sweep PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

# Register with CTest for proper OMP/MPI environment setup
# Run modes (via CTest for correct environment):
#   Full sweep:   ./run_integer_gemm_autotuner_data.sh  (uses CTest internally)
#   Single token: ctest -R "V2_Perf_IntegerGEMM_FullSweep_SingleToken" --verbose
#   For perf:     ctest -R "V2_Perf_IntegerGEMM_FullSweep_SingleConfig_M512" --verbose

add_v2_perf_test(V2_Perf_IntegerGEMM_FullSweep_AllConfigs
    COMMAND $<TARGET_FILE:v2_perf_integer_gemm_full_sweep> --gtest_filter=IntegerGEMM_FullSweep.AllConfigsAllWorkloads
    LABELS "V2;Performance;GEMM;IntegerGEMM;Autotuner;DataCollection;FullSweep"
    MPI_PROCS 1  # Single-rank pure CPU performance
)

add_v2_perf_test(V2_Perf_IntegerGEMM_FullSweep_SingleToken
    COMMAND $<TARGET_FILE:v2_perf_integer_gemm_full_sweep> --gtest_filter=IntegerGEMM_FullSweep.AllConfigsSingleToken
    LABELS "V2;Performance;GEMM;IntegerGEMM;Autotuner;QuickTest"
    MPI_PROCS 1
)

# Individual configs for perf stat profiling (registered by autotuner script)
# Example: V2_Perf_IntegerGEMM_FullSweep_SingleConfig_M512_N896_K896_MR8_KB4_UK2_P1_CB0

# Unit Test: Integer GEMM V2 Basic Correctness
# Tests new V2 Integer GEMM rewrite against scalar reference
# Validates small, medium matrices before performance testing
add_executable(v2_test_integer_gemm_v2_basic performance/cpu/kernels/gemm/Test__IntegerGEMM_V2_Basic.cpp)
target_link_libraries(v2_test_integer_gemm_v2_basic
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_test_integer_gemm_v2_basic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

add_v2_test(V2_Unit_IntegerGEMM_V2_Basic
    COMMAND v2_test_integer_gemm_v2_basic
    LABELS "V2;Unit;Kernels;IntegerGEMM;Correctness"
    MPI_PROCS 1
)

# Performance Test: CPU GEMM Heuristic Validation Suite
# Comprehensive test suite for ML heuristic training data collection.
# Covers all major model sizes from Qwen 0.5B to DeepSeek 671B (53 tests total).
# Used to generate exhaustive benchmarking data for neural network training.
add_executable(v2_perf_cpu_gemm_validation performance/Perf__CpuGemmHeuristicValidation.cpp)
target_link_libraries(v2_perf_cpu_gemm_validation
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
    MPI::MPI_CXX
)
set_target_properties(v2_perf_cpu_gemm_validation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_CpuGemmValidation
    COMMAND $<TARGET_FILE:v2_perf_cpu_gemm_validation>
    LABELS "V2;Performance;GEMM;MLHeuristic;Validation;Qwen;DeepSeek;IQ4_NL"
    MPI_PROCS 1  # Single rank for consistent benchmarking
)

# Performance Test: BF16 Conversion Optimization (side-by-side comparison)
# Tests baseline vs optimized BF16 conversion implementations
# Optimizations: loop unrolling, prefetching, aligned loads, streaming stores
add_executable(v2_perf_bf16_conversion_opt performance/Perf__BF16_Conversion_Optimization.cpp)
target_link_libraries(v2_perf_bf16_conversion_opt 
    llaminar2_core 
    GTest::gtest
    MPI::MPI_CXX
)
set_target_properties(v2_perf_bf16_conversion_opt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_BF16_ConversionOpt
    COMMAND $<TARGET_FILE:v2_perf_bf16_conversion_opt>
    LABELS "V2;Performance;BF16;Conversion;Optimization;Bandwidth;SIMD"
    MPI_PROCS 2  # Match production configuration
)

# NOTE: Removed experimental Phase 2.7 pipelining tests
# - Perf__CuTePipelining.cu (references non-existent CudaGemmKernelTensorCorePipelined.cuh)
# - Perf__Phase2_7_TensorCore_Pipeline.cu (references non-existent CudaGemmKernelTensorCorePipeline.cuh)
# These were experimental explorations of software pipelining with CuTe that are not
# part of the current production kernel architecture.

# NOTE: Removed CUDAvsGPU_GEMM performance test
# Reason: Compared GPU against single-threaded CPU, which is not representative
# of production performance and took 6+ minutes to run. Use V2_Perf_IQ4NL_GEMM
# for proper multi-rank CPU benchmarks, and CUDA unit tests verify GPU correctness.

# Performance Test: Phase 1 Standalone CUDA Kernel Benchmark
# Direct baseline vs optimized kernel comparison with manual CUDA memory allocation
# Bypasses pipeline/autotuner to measure pure kernel performance
if(HAVE_CUDA)
    add_executable(v2_perf_phase1_standalone performance/Perf__Phase1_Standalone.cu)
    target_link_libraries(v2_perf_phase1_standalone
        cuda_backend
        GTest::gtest
        CUDA::cudart
    )
    set_target_properties(v2_perf_phase1_standalone PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    add_v2_perf_test(V2_Perf_Phase1_Standalone
        COMMAND $<TARGET_FILE:v2_perf_phase1_standalone>
        LABELS "V2;Performance;CUDA;Phase1;Standalone;GEMM"
        MPI_PROCS 1  # Single GPU, no MPI
    )
    
    # Performance Test: Phase 1 Debug - Separate configs for baseline vs optimized
    # Tests each kernel with its own optimal configuration
    add_executable(v2_perf_phase1_debug performance/Perf__Phase1_Debug.cu)
    target_link_libraries(v2_perf_phase1_debug
        cuda_backend
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
    )
    set_target_properties(v2_perf_phase1_debug PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    add_v2_perf_test(V2_Perf_Phase1_Debug
        COMMAND $<TARGET_FILE:v2_perf_phase1_debug>
        LABELS "V2;Performance;CUDA;Phase1;Debug;GEMM"
        MPI_PROCS 1  # Single GPU, no MPI
    )
    
    # Performance Test: Phase 2 Tensor Core - wmma API with FP16 compute
    # Validates 3-4× speedup over Phase 1 using Tensor Cores
    # COMMENTED OUT: Depends on obsolete CudaGemmVariantsTensorCore.h file
    # add_executable(v2_perf_phase2_tensorcore performance/Perf__Phase2_TensorCore.cu)
    # target_link_libraries(v2_perf_phase2_tensorcore
    #     cuda_backend
    #     GTest::gtest
    #     GTest::gtest_main
    #     CUDA::cudart
    # )
    # set_target_properties(v2_perf_phase2_tensorcore PROPERTIES
    #     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
    #     CUDA_SEPARABLE_COMPILATION ON
    # )
    # 
    # add_v2_perf_test(V2_Perf_Phase2_TensorCore
    #     COMMAND $<TARGET_FILE:v2_perf_phase2_tensorcore>
    #     LABELS "V2;Performance;CUDA;Phase2;TensorCore;FP32Input;GEMM"
    #     MPI_PROCS 1  # Single GPU, no MPI
    # )
    
    # Phase 2.5: FP16 input with async copy
    # COMMENTED OUT: Depends on obsolete CudaGemmKernel.cuh file
    # add_executable(v2_perf_phase2_5_tensorcore_fp16 performance/Perf__Phase2_5_TensorCore_FP16.cu)
    # target_link_libraries(v2_perf_phase2_5_tensorcore_fp16
    #     cuda_backend
    #     GTest::gtest
    #     GTest::gtest_main
    #     CUDA::cudart
    # )
    # set_target_properties(v2_perf_phase2_5_tensorcore_fp16 PROPERTIES
    #     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
    #     CUDA_SEPARABLE_COMPILATION ON
    # )
    
    # add_v2_perf_test(V2_Perf_Phase2_5_TensorCore_FP16
    #     COMMAND $<TARGET_FILE:v2_perf_phase2_5_tensorcore_fp16>
    #     LABELS "V2;Performance;CUDA;Phase2_5;TensorCore;FP16Input;AsyncCopy;GEMM"
    #     MPI_PROCS 1  # Single GPU, no MPI
    # )
    
    # Phase 3: Tile size tuning sweep
    # COMMENTED OUT: Depends on obsolete CudaGemmKernel.cuh file
    # add_executable(v2_perf_phase3_tile_sweep performance/Perf__Phase3_TileSizeSweep.cu)
    # target_link_libraries(v2_perf_phase3_tile_sweep
    #     cuda_backend
    #     GTest::gtest
    #     GTest::gtest_main
    #     CUDA::cudart
    # )
    # set_target_properties(v2_perf_phase3_tile_sweep PROPERTIES
    #     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
    #     CUDA_SEPARABLE_COMPILATION ON
    # )
    
    # add_v2_perf_test(V2_Perf_Phase3_TileSizeSweep
    #     COMMAND $<TARGET_FILE:v2_perf_phase3_tile_sweep>
    #     LABELS "V2;Performance;CUDA;Phase3;TensorCore;TileSizeTuning;Optimization;GEMM"
    #     MPI_PROCS 1  # Single GPU, no MPI
    # )
endif()

# Performance Test: 64×32 Tile Size Comparison (DISABLED - missing GemmAutoTuner.h)
# Compares V1's optimal 64×32 tile against auto-selected tiles for 1024 tokens
# add_executable(v2_perf_tile64x32_comparison performance/Perf__Tile64x32Comparison.cpp)
# target_link_libraries(v2_perf_tile64x32_comparison 
#     llaminar2_core 
#     GTest::gtest
#     GTest::gtest_main
# )
# set_target_properties(v2_perf_tile64x32_comparison PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )
# 
# add_v2_perf_test(V2_Perf_Tile64x32Comparison
#     COMMAND $<TARGET_FILE:v2_perf_tile64x32_comparison>
#     LABELS "V2;Performance;GEMM;TileSizeOptimization;AutoTuning"
#     MPI_PROCS 1  # Single rank benchmark
# )

# Performance Test: GEMM Tile Size Scaling (Phase 4) (DISABLED - missing GemmKernelTemplate.h)
# Benchmarks all tile sizes (4, 8, 16, 32) across matrix dimensions
# Validates larger tiles provide speedup for appropriate problem sizes
# add_executable(v2_perf_gemm_tile_scaling 
#     performance/Perf__GemmTileSizeScaling.cpp
#     e2e/gtest_mpi_main.cpp  # MPI-aware GTest main
# )
# target_link_libraries(v2_perf_gemm_tile_scaling 
#     llaminar2_core 
#     GTest::gtest
#     MPI::MPI_CXX
# )
# set_target_properties(v2_perf_gemm_tile_scaling PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )
# 
# add_v2_perf_test(V2_Perf_GemmTileScaling
#     COMMAND $<TARGET_FILE:v2_perf_gemm_tile_scaling>
#     LABELS "V2;Performance;GEMM;Templates;TileSize;Scaling;Phase4"
#     MPI_PROCS 2  # Use 2 ranks for consistency
# )

# Performance Test: CUDA GEMM Heuristic Validation (Phase 6.11)
# Benchmarks ALL kernel variants for 0.5B/4B model matrix shapes
# Compares heuristic ranking vs empirical performance
# Validates auto-selection logic makes good choices
# NOTE: Excluded from default ALL target to avoid running on every build
#       Build manually with: cmake --build . --target v2_perf_cuda_heuristic_validation
if(HAVE_CUDA)
    add_executable(v2_perf_cuda_heuristic_validation 
        performance/Perf__CudaGemmHeuristicValidation.cpp
    )
    target_link_libraries(v2_perf_cuda_heuristic_validation 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    set_target_properties(v2_perf_cuda_heuristic_validation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
        EXCLUDE_FROM_ALL TRUE  # Don't build as part of default ALL target
    )
    
    add_v2_perf_test(V2_Perf_CudaHeuristicValidation
        COMMAND $<TARGET_FILE:v2_perf_cuda_heuristic_validation>
        LABELS "V2;Performance;CUDA;GEMM;AutoTuning;HeuristicValidation;ModelSizes"
        MPI_PROCS 1  # Single rank GPU benchmark
    )
    
    # Custom target: Train ML heuristic from benchmark data
    # This regenerates cuda_heuristic_weights.h in the source tree
    # Run with: cmake --build . --target train_cuda_heuristic
    add_custom_target(train_cuda_heuristic
        COMMAND ${CMAKE_COMMAND} -E echo "Training ML heuristic from benchmark data..."
        COMMAND Python3::Interpreter 
                ${PROJECT_SOURCE_DIR}/train_cuda_heuristic.py
                --input ${CMAKE_BINARY_DIR}/cuda_gemm_benchmark_data.csv
                --output-dir ${CMAKE_SOURCE_DIR}/src/v2/kernels/cuda
        COMMAND ${CMAKE_COMMAND} -E echo "Generated: src/v2/kernels/cuda/cuda_heuristic_weights.h"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Training CUDA GEMM ML heuristic and generating C++ header"
        VERBATIM
    )
    
    # Make training depend on the benchmark executable
    add_dependencies(train_cuda_heuristic v2_perf_cuda_heuristic_validation)
    
    # Hash-based automatic retraining (only retrains if CSV changed)
    # Run with: cmake --build . --target train_cuda_heuristic_auto
    add_custom_target(train_cuda_heuristic_auto
        COMMAND ${CMAKE_COMMAND} -E echo "Checking if ML retraining needed (hash-based)..."
        COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/check_cuda_heuristic_needs_retrain.sh && 
                bash ${PROJECT_SOURCE_DIR}/scripts/train_cuda_heuristic.sh || 
                ${CMAKE_COMMAND} -E echo "Heuristic weights already up-to-date"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Auto-train CUDA GEMM heuristic (only if CSV hash changed)"
        VERBATIM
    )
    
    # Make auto-training depend on the benchmark executable
    add_dependencies(train_cuda_heuristic_auto v2_perf_cuda_heuristic_validation)
    
    # Optional: Auto-train after running performance test
    # Uncomment to automatically regenerate weights when benchmark data changes
    # add_custom_command(
    #     OUTPUT ${CMAKE_SOURCE_DIR}/src/v2/kernels/cuda/cuda_heuristic_weights.h
    #     COMMAND $<TARGET_FILE:v2_perf_cuda_heuristic_validation>
    #     COMMAND Python3::Interpreter 
    #             ${PROJECT_SOURCE_DIR}/train_cuda_heuristic.py
    #             --input ${CMAKE_BINARY_DIR}/cuda_gemm_benchmark_data.csv
    #             --output-dir ${CMAKE_SOURCE_DIR}/src/v2/kernels/cuda
    #     DEPENDS v2_perf_cuda_heuristic_validation
    #     COMMENT "Running benchmark and training ML heuristic"
    #     VERBATIM
    # )
    
    # Performance Test: CUDA GEMM Heuristic Canary (Generalization Validation)
    # Tests ML heuristic with UNTRAINED matrix shapes to detect overfitting
    # Does NOT export training data - purely validation/testing
    # Expected: >70% top-20 hit rate on unseen configurations
    # NOTE: Excluded from default ALL target to avoid running on every build
    #       Build manually with: cmake --build . --target v2_perf_cuda_heuristic_canary
    add_executable(v2_perf_cuda_heuristic_canary 
        performance/Perf__CudaGemmHeuristicCanary.cpp
    )
    target_link_libraries(v2_perf_cuda_heuristic_canary 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    set_target_properties(v2_perf_cuda_heuristic_canary PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
        EXCLUDE_FROM_ALL TRUE  # Don't build as part of default ALL target
    )
    set_target_properties(v2_perf_cuda_heuristic_canary PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
    )
    
    add_v2_perf_test(V2_Perf_CudaHeuristicCanary
        COMMAND $<TARGET_FILE:v2_perf_cuda_heuristic_canary>
        LABELS "V2;Performance;CUDA;GEMM;AutoTuning;HeuristicCanary;GeneralizationTest;NoTrainingData"
        MPI_PROCS 1  # Single rank GPU benchmark
    )

    # Performance Test: Batch Prefill GFLOPS Benchmark
    # Measures peak GFLOPS for batch prefill scenarios (Qwen 0.5B)
    # Tests batch sizes from 1 to 1024 to find optimal throughput
    # Useful for understanding batch scaling behavior
    add_executable(v2_perf_batch_prefill_gflops 
        performance/Perf__BatchPrefillGflops.cpp
    )
    target_link_libraries(v2_perf_batch_prefill_gflops 
        llaminar2_core 
        GTest::gtest
        GTest::gtest_main
    )
    set_target_properties(v2_perf_batch_prefill_gflops PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
        EXCLUDE_FROM_ALL TRUE  # Build manually
    )
    
    add_v2_perf_test(V2_Perf_BatchPrefillGflops
        COMMAND $<TARGET_FILE:v2_perf_batch_prefill_gflops>
        LABELS "V2;Performance;CUDA;GEMM;BatchPrefill;Throughput;Qwen"
        MPI_PROCS 1  # Single rank GPU benchmark
    )

    # Performance Test: Tensor Core Heuristic Validation (Phase 3+)
    # COMPREHENSIVE benchmarking of ALL Tensor Core variants
    # Tests 0.5B, 4B, 7B, 14B model sizes with batch sizes 1, 8, 32, 128
    # Exports data for ML heuristic training: tensorcore_benchmark_data.csv
    # Validates Phase 3 optimization (32×64×16 for m=32)
    # COMMENTED OUT: Depends on obsolete CudaGemmVariantsTensorCore.h file
    # add_executable(v2_perf_tensorcore_heuristic_validation 
    #     performance/Perf__TensorCoreHeuristicValidation.cpp
    # )
    # target_link_libraries(v2_perf_tensorcore_heuristic_validation 
    #     llaminar2_core 
    #     GTest::gtest
    # )
    # set_target_properties(v2_perf_tensorcore_heuristic_validation PROPERTIES
    #     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
    # )
    # 
    # add_v2_perf_test(V2_Perf_TensorCoreHeuristicValidation
    #     COMMAND $<TARGET_FILE:v2_perf_tensorcore_heuristic_validation>
    #     LABELS "V2;Performance;CUDA;TensorCore;AutoTuning;HeuristicValidation;ModelSizes;Comprehensive"
    #     MPI_PROCS 1  # Single rank GPU benchmark
    # )
    
    # Custom target: Train ML heuristic from Tensor Core benchmark data
    # This regenerates tensorcore_heuristic_weights.h in the source tree
    # Run with: cmake --build . --target train_tensorcore_heuristic
    add_custom_target(train_tensorcore_heuristic
        COMMAND ${CMAKE_COMMAND} -E echo "Training Tensor Core ML heuristic from benchmark data..."
        COMMAND Python3::Interpreter 
                ${PROJECT_SOURCE_DIR}/train_tensorcore_heuristic.py
                --input ${CMAKE_BINARY_DIR}/tensorcore_benchmark_data.csv
                --output-dir ${CMAKE_SOURCE_DIR}/src/v2/kernels/cuda
        COMMAND ${CMAKE_COMMAND} -E echo "Generated: src/v2/kernels/cuda/tensorcore_heuristic_weights.h"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Training Tensor Core GEMM ML heuristic and generating C++ header"
        VERBATIM
    )
    
    # Make training depend on the benchmark executable
    # COMMENTED OUT: v2_perf_tensorcore_heuristic_validation is disabled
    # add_dependencies(train_tensorcore_heuristic v2_perf_tensorcore_heuristic_validation)
endif()

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

# Perf: Auto-tuner validation (verifies optimal kernel selection) (DISABLED - missing GemmAutoTuner.h)
# add_executable(v2_perf_gemm_autotuner performance/Perf__GemmAutoTuner.cpp)
# target_link_libraries(v2_perf_gemm_autotuner 
#     llaminar2_core 
#     GTest::gtest
# )
# set_target_properties(v2_perf_gemm_autotuner PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
# )
# add_v2_perf_test(V2_Perf_GemmAutoTuner
#     COMMAND $<TARGET_FILE:v2_perf_gemm_autotuner>
#     LABELS "V2;Performance;Kernels;GEMM;AutoTuning;KernelSelection;TileSizeSweep"
#     MPI_PROCS 1
# )

# Perf: Micro-Kernel Scaling (2 MPI ranks for multi-socket performance) (DISABLED - missing GemmAutoTuner.h)
# add_executable(v2_perf_microkernel_scaling performance/Perf__MicroKernelScaling.cpp)
# target_link_libraries(v2_perf_microkernel_scaling 
#     llaminar2_core 
#     GTest::gtest
#     MPI::MPI_CXX
# )
# add_v2_perf_test(V2_Perf_MicroKernelScaling
#     COMMAND $<TARGET_FILE:v2_perf_microkernel_scaling>
#     LABELS "V2;Performance;Kernels;GEMM;MicroKernel;MultiSocket;MPIScaling;Throughput"
#     MPI_PROCS 2
# )

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


# Debug test: 64-byte vs 128-byte comparison
add_executable(v2_debug_integer_gemm_64vs128
    performance/cpu/kernels/gemm/Debug__IntegerGEMM_64vs128.cpp
)
target_link_libraries(v2_debug_integer_gemm_64vs128
    PRIVATE llaminar2_core
    PRIVATE GTest::gtest
    PRIVATE GTest::gtest_main
)
