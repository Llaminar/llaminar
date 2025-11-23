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

# Test: FusedRMSNormQuantize kernel (operator fusion framework - Phase 1)
add_executable(v2_test_fused_rmsnorm_quantize unit/Test__FusedRMSNormQuantize.cpp)
target_link_libraries(v2_test_fused_rmsnorm_quantize
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_FusedRMSNormQuantize
    COMMAND $<TARGET_FILE:v2_test_fused_rmsnorm_quantize>
    LABELS "V2;Unit;Kernels;FusedKernels;RMSNorm;Quantization;INT8;OperatorFusion;AVX512;AVX2;CPU"
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

# Test: OneDNN GEMM kernel fused matmul + softmax path
add_executable(v2_test_onednn_gemm_kernel unit/kernels/gemm/Test__OneDNNGemmKernel.cpp)
target_link_libraries(v2_test_onednn_gemm_kernel
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OneDNNGemmKernel
    COMMAND $<TARGET_FILE:v2_test_onednn_gemm_kernel>
    LABELS "V2;Unit;Kernels;GEMM;Softmax;OneDNN;CPU"
    MPI_PROCS 1  # OneDNN fused kernel executes on CPU only
)

# Test: OneDNN GEMM kernel strided/causal support
add_executable(v2_test_onednn_gemm_kernel_strided unit/kernels/gemm/Test__OneDNNGemmKernel_Strided.cpp)
target_link_libraries(v2_test_onednn_gemm_kernel_strided
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OneDNNGemmKernel_Strided
    COMMAND $<TARGET_FILE:v2_test_onednn_gemm_kernel_strided>
    LABELS "V2;Unit;Kernels;GEMM;Softmax;OneDNN;CPU;Strided"
    MPI_PROCS 1
)

# Test: OneDNN GEMM kernel typed interface (BF16/FP16/FP32)
add_executable(v2_test_onednn_gemm_kernel_typed unit/kernels/gemm/Test__OneDNNGemmKernel_Typed.cpp)
target_link_libraries(v2_test_onednn_gemm_kernel_typed
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OneDNNGemmKernel_Typed
    COMMAND $<TARGET_FILE:v2_test_onednn_gemm_kernel_typed>
    LABELS "V2;Unit;Kernels;GEMM;OneDNN;Typed;BF16;FP16;CPU"
    MPI_PROCS 1
)

# Test: OneDNN GEMM kernel scratch buffer functionality
add_executable(v2_test_onednn_gemm_kernel_scratch_buffers unit/kernels/gemm/Test__OneDNNGemmKernel_ScratchBuffers.cpp)
target_link_libraries(v2_test_onednn_gemm_kernel_scratch_buffers
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OneDNNGemmKernel_ScratchBuffers
    COMMAND $<TARGET_FILE:v2_test_onednn_gemm_kernel_scratch_buffers>
    LABELS "V2;Unit;Kernels;GEMM;OneDNN;ScratchBuffers;MemoryManagement;ThreadLocal;Performance;CPU"
    MPI_PROCS 1
)

# Test: OneDNN GEMM adapter (INT8 path with tensor interface)
add_executable(v2_test_onednn_gemm_adapter unit/kernels/gemm/Test__OneDNNGemmAdapter.cpp)
target_link_libraries(v2_test_onednn_gemm_adapter
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_OneDNNGemmAdapter
    COMMAND $<TARGET_FILE:v2_test_onednn_gemm_adapter>
    LABELS "V2;Unit;Kernels;GEMM;OneDNN;INT8;Adapter;CPU"
    MPI_PROCS 1  # Adapter operates purely on CPU tensors, single rank
)

# Test: WeightPack cache functionality (lazy INT8 repacking, memory deallocation)
add_executable(v2_test_weight_pack_cache unit/Test__WeightPackCache.cpp)
target_link_libraries(v2_test_weight_pack_cache
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_WeightPackCache
    COMMAND $<TARGET_FILE:v2_test_weight_pack_cache>
    LABELS "V2;Unit;TensorOperations;WeightCaching;INT8;MemoryOptimization;OneDNN;CPU"
    MPI_PROCS 1  # Cache tests are single-rank CPU operations
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
add_v2_test(V2_Unit_CPURoPEKernelT 
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

# Test: Qwen2Pipeline INT8 Buffer Bug (demonstrates total_rows vs effective_seq_len bug)
add_executable(v2_test_qwen2_int8_buffer_bug unit/Test__Qwen2Pipeline_INT8BufferBug.cpp)
target_link_libraries(v2_test_qwen2_int8_buffer_bug 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_Qwen2INT8BufferBug
    COMMAND $<TARGET_FILE:v2_test_qwen2_int8_buffer_bug>
    LABELS "V2;Unit;Pipelines;Qwen2;INT8;BufferManagement;BatchPadding;BugDemonstration"
    MPI_PROCS 1  # Single rank unit test
)

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

add_v2_test(V2_Unit_IQ4_XS_Views 
    COMMAND $<TARGET_FILE:v2_test_iq4xs_views>
    LABELS "V2;Unit;TensorOperations;IQ4_XS;Quantization;ViewSupport;RowSlice;IQuant"
)

# Test: IQ2_XXS Tensor Views
add_executable(v2_test_iq4xs_views unit/tensors/Test__IQ4_XSTensor_Views.cpp)
target_link_libraries(v2_test_iq4xs_views 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
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
add_v2_test(V2_Unit_IQ2_XXS_Tensor 
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
add_v2_test(V2_Unit_IQ2_XS_Tensor 
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
add_v2_test(V2_Unit_IQ2_S_Tensor 
    COMMAND $<TARGET_FILE:v2_test_iq2s_tensor>
    LABELS "V2;Unit;TensorOperations;IQ2_S;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
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
add_v2_test(V2_Unit_IQ3_S_Tensor 
    COMMAND $<TARGET_FILE:v2_test_iq3s_tensor>
    LABELS "V2;Unit;TensorOperations;IQ3_S;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
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
    LABELS "V2;Unit;TensorOperations;IQ1_M;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
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
    LABELS "V2;Unit;TensorOperations;IQ1_S;Quantization;ViewSupport;RowSlice;IQuant;SuperBlock"
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
add_executable(v2_test_qwen2_null_mpi_context integration/Test__Qwen2NullMPIContext.cpp)
target_link_libraries(v2_test_qwen2_null_mpi_context
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Qwen2NullMPIContext
    COMMAND $<TARGET_FILE:v2_test_qwen2_null_mpi_context>
    MPI_PROCS 1
    LABELS "V2;Integration;PipelineExecution;Qwen;SingleRank;Regression;Models"
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

# Test: PipelineBase logits() override (regression test for decode segfault)
add_executable(v2_test_pipeline_logits_override unit/pipelines/Test__PipelineBase_LogitsOverride.cpp)
target_link_libraries(v2_test_pipeline_logits_override
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_PipelineLogitsOverride
    COMMAND $<TARGET_FILE:v2_test_pipeline_logits_override>
    LABELS "V2;Unit;PipelineExecution;RegressionTest;LogitsBug;Qwen2;NullPointer"
)

# Test: MpiAttentionOrchestrator
add_executable(v2_test_mpi_attention_orchestrator unit/pipelines/Test__MpiAttentionOrchestrator.cpp)
target_link_libraries(v2_test_mpi_attention_orchestrator 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_MpiAttentionOrchestrator 
    COMMAND $<TARGET_FILE:v2_test_mpi_attention_orchestrator>
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
    
    # Phase 3 Part 2: Pipelined large tiles for 900-1,000 GFLOPS target
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
    add_executable(v2_test_phase5_sweep cuda/Perf__Phase5Sweep.cpp)
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
    add_v2_perf_test(V2_Perf_Phase5Sweep
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
    add_v2_perf_test(V2_Perf_Phase5Parity
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
    add_v2_perf_test(V2_Perf_Phase5_SUB_K_Sweep
        COMMAND $<TARGET_FILE:v2_test_phase5_sub_k_sweep>
        LABELS "V2;Performance;CUDA;Phase5;TensorCore;TileSizeTuning;Optimization;GEMM"
        MPI_PROCS 1  # Single GPU, no MPI
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
    add_v2_perf_test(V2_Perf_Phase5_SUB_K_64_Profile
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
    add_v2_perf_test(V2_Perf_Phase5_Transpose_Benchmark
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
    add_v2_perf_test(V2_Perf_Phase5_Transpose_Large
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
    add_v2_perf_test(V2_Perf_Phase6_Int8_Performance
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
    add_v2_perf_test(V2_Perf_Phase7_CUTLASS
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
# COMMENTED OUT: API changed with ActivationTraits/WeightAccessor refactor
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
# COMMENTED OUT: API changed with ActivationTraits/WeightAccessor refactor
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

# NOTE: MicroKernel tests removed - depend on obsolete gemm implementations
# (MicroKernelAdapter_Packing, MicroKernelPadding, MicroKernelBounds, MicroKernelCorrectness)

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

# Test: Qwen2Pipeline INT8 Dequantization Regression (prevents recurrence of 97B% divergence bug)
# Loads full model, runs complete forward passes - too slow for unit test suite
add_executable(v2_test_qwen2_int8_dequant_regression integration/Test__Qwen2Pipeline_INT8DequantRegression.cpp)
target_link_libraries(v2_test_qwen2_int8_dequant_regression 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Integration_Qwen2INT8DequantRegression
    COMMAND $<TARGET_FILE:v2_test_qwen2_int8_dequant_regression>
    LABELS "V2;Integration;Pipelines;Qwen2;INT8;Regression;BatchPadding;Dequantization"
    MPI_PROCS 1  # Single rank integration test
)

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

# NOTE: MicroKernel AutoTuner Integration test removed - depends on obsolete gemm implementations

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

# Test: Qwen2Pipeline Integration - Comprehensive pipeline validation with numerical sanity checks
add_executable(v2_test_qwen2_pipeline integration/Test__Qwen2Pipeline.cpp)
target_link_libraries(v2_test_qwen2_pipeline 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    MPI::MPI_CXX
)
add_v2_test(V2_Integration_Qwen2Pipeline
    COMMAND $<TARGET_FILE:v2_test_qwen2_pipeline>
    MPI_PROCS 2
    LABELS "V2;Integration;PipelineExecution;Qwen;Batching;Models;NumericalSanity"
)

# Test: Batch Padding Divergence - Minimal reproducer for batch padding bug
add_executable(v2_test_batch_padding_divergence unit/Test__BatchPaddingDivergence.cpp)
target_link_libraries(v2_test_batch_padding_divergence 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
    MPI::MPI_CXX
)
add_v2_test(V2_Unit_BatchPaddingDivergence
    COMMAND $<TARGET_FILE:v2_test_batch_padding_divergence>
    MPI_PROCS 2
    LABELS "V2;Unit;PipelineExecution;Batching;BugReproducer;Correctness"
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
# Only build for Debug or E2ERelease configurations (requires ENABLE_PIPELINE_SNAPSHOTS)
# Requires: PyTorch snapshots generated by python/reference/generate_qwen2_pipeline_snapshots.py
# Requires: cnpy library for NumPy .npz file loading
find_package(ZLIB)
if(ZLIB_FOUND)
    # Get the workspace root (parent of tests/v2)
    get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    
    # E2E parity test requires ENABLE_PIPELINE_SNAPSHOTS (Debug or E2ERelease builds)
    # Skip in standard Release builds to avoid linking errors
    if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "E2ERelease" OR ENABLE_SNAPSHOTS)
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
    else()
        message(STATUS "V2: E2E correctness test disabled (snapshots not available in Release build)")
        message(STATUS "     Use -DCMAKE_BUILD_TYPE=E2ERelease or -DENABLE_SNAPSHOTS=ON to enable")
    endif()
else()
    message(WARNING "ZLIB not found - Qwen2 FP32 parity test disabled (requires cnpy for .npz file loading)")
endif()
# )

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
    # Get the workspace root (parent of tests/v2)
    get_filename_component(WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
    
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
        
        # Pass Python executable path to the test for snapshot generation
        if(Python3_EXECUTABLE)
            target_compile_definitions(v2_test_qwen2_fp32_parity PRIVATE
                PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
            )
        endif()

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
    message(WARNING "ZLIB not found - Qwen2 FP32 parity test disabled (requires cnpy for .npz file loading)")
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
        -x ONEDNN_VERBOSE=0
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
        ENVIRONMENT "LLAMINAR_LOG_LEVEL=INFO;OMP_NUM_THREADS=${CORES_PER_SOCKET};OMP_PLACES=sockets;OMP_PROC_BIND=close;OMP_NESTED=false;OMP_DYNAMIC=false;KMP_AFFINITY=granularity=fine,compact,1,0;KMP_BLOCKTIME=0;OPENBLAS_NUM_THREADS=${CORES_PER_SOCKET};GOTO_NUM_THREADS=${CORES_PER_SOCKET};MKL_NUM_THREADS=${CORES_PER_SOCKET};MKL_DYNAMIC=false;ONEDNN_VERBOSE=0;OMPI_MCA_mpi_leave_pinned=1;OMPI_MCA_btl_vader_single_copy_mechanism=none;OMPI_MCA_btl_openib_allow_ib=1"
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../.."
    )
    
    # Note: Performance tests do NOT require V2_Models fixture
    # They control model loading themselves for benchmarking
endfunction()

# NOTE: IQ4_NL GEMM v4 performance test removed - depended on obsolete
# weight-owned GEMM paths; IQ4_NL is now exercised only via activation-centric
# kernels and higher-level pipelines.

# NOTE: Q8_0 Conversion performance test removed - gemm_v2 is obsolete

# NOTE: IQ4_NL GEMM comparison test removed - depends on obsolete QuantizedGemm

# NOTE: VNNIGemm Qwen Profile test removed - gemm_v3 is obsolete

# Performance Test: OneDNN GEMM Qwen Profile (gemm_v4)
# Uses the new gemm_v4 adapters that wrap OneDNN matmul to gauge achievable throughput.
add_executable(v2_perf_onednn_gemm_qwen_profile performance/kernels/cpu/gemm/gemm_v4/Perf__OneDNNGemm_QwenProfile.cpp)
target_link_libraries(v2_perf_onednn_gemm_qwen_profile 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_onednn_gemm_qwen_profile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_OneDNNGemm_QwenProfile
    COMMAND $<TARGET_FILE:v2_perf_onednn_gemm_qwen_profile>
    LABELS "V2;Performance;GEMM;OneDNN;FP32;Qwen;AdapterOverhead;gemm_v4"
    MPI_PROCS 1
)

# Performance Test: OneDNN GEMM Overhead (M=1)
add_executable(v2_perf_onednn_gemm_overhead performance/kernels/cpu/gemm/gemm_v4/Perf__OneDNNGemm_Overhead.cpp)
target_link_libraries(v2_perf_onednn_gemm_overhead 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_onednn_gemm_overhead PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_OneDNNGemm_Overhead
    COMMAND $<TARGET_FILE:v2_perf_onednn_gemm_overhead>
    LABELS "V2;Performance;GEMM;OneDNN;FP32;Qwen;Overhead;gemm_v4"
    MPI_PROCS 1
)

# Performance Test: Fused OneDNN matmul + softmax diagnostic
add_executable(v2_perf_onednn_gemm_fused_softmax performance/kernels/cpu/gemm/gemm_v4/Perf__OneDNNGemm_FusedSoftmax.cpp)
target_link_libraries(v2_perf_onednn_gemm_fused_softmax
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_onednn_gemm_fused_softmax PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_OneDNNGemm_FusedSoftmax
    COMMAND $<TARGET_FILE:v2_perf_onednn_gemm_fused_softmax>
    LABELS "V2;Performance;Attention;GEMM;Softmax;OneDNN;gemm_v4"
    MPI_PROCS 1
)

# Performance Test: INT8 GEMM + streaming softmax comparison
add_executable(v2_perf_onednn_gemm_int8_streaming_softmax performance/kernels/cpu/gemm/gemm_v4/Perf__OneDNNGemm_Int8StreamingSoftmax.cpp)
target_link_libraries(v2_perf_onednn_gemm_int8_streaming_softmax
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_onednn_gemm_int8_streaming_softmax PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_OneDNNGemm_Int8StreamingSoftmax
    COMMAND $<TARGET_FILE:v2_perf_onednn_gemm_int8_streaming_softmax>
    LABELS "V2;Performance;Attention;GEMM;Softmax;OneDNN;INT8;gemm_v4"
    MPI_PROCS 1
)

# Performance Test: OneDNN Conversion (FP32->INT8)
add_executable(v2_perf_onednn_conversion performance/kernels/cpu/gemm/gemm_v4/Perf__OneDNN_Conversion.cpp)
target_link_libraries(v2_perf_onednn_conversion
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_onednn_conversion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_OneDNN_Conversion
    COMMAND $<TARGET_FILE:v2_perf_onednn_conversion>
    LABELS "V2;Performance;GEMM;OneDNN;Conversion;FP32"
    MPI_PROCS 1
)

# Performance Test: Qwen2 Single Token Profile
# Runs a tight decode loop for profiling with perf
add_executable(v2_perf_qwen2_single_token performance/pipelines/qwen/Perf__Qwen2_SingleToken_Profile.cpp)
target_link_libraries(v2_perf_qwen2_single_token
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_qwen2_single_token PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Qwen2_SingleToken
    COMMAND $<TARGET_FILE:v2_perf_qwen2_single_token>
    LABELS "V2;Performance;Pipeline;Qwen2;Profiling"
    MPI_PROCS 1
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

# NOTE: Auto-tuned GEMM cache profiling test removed - depends on obsolete GemmAutoTuner

# NOTE: Exhaustive autotuner sweep test removed - depends on obsolete GemmAutoTuner

# NOTE: CPU GEMM heuristic validation suite removed - depended on obsolete
# GEMM v4 weight-owned APIs and has been superseded by activation-centric
# performance tests.

# Performance Test: BF16 Conversion Optimization (side-by-side comparison)

# Performance Test: CpuAttentionKernelT (FP32)
# Benchmarks CPU attention kernel latency and throughput
add_executable(v2_perf_cpu_attention_fp32 performance/kernels/cpu/Perf__CpuAttentionKernelT.cpp)
target_link_libraries(v2_perf_cpu_attention_fp32
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
set_target_properties(v2_perf_cpu_attention_fp32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_CpuAttention_FP32
    COMMAND $<TARGET_FILE:v2_perf_cpu_attention_fp32>
    LABELS "V2;Performance;Attention;FP32;Latency"
    MPI_PROCS 1
)

# Performance Test: CPURoPEKernel (FP32)
# Benchmarks CPU RoPE kernel latency and throughput
add_executable(v2_perf_cpu_rope_fp32 performance/kernels/cpu/Perf__CPURoPEKernel.cpp)
target_link_libraries(v2_perf_cpu_rope_fp32
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
set_target_properties(v2_perf_cpu_rope_fp32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_CpuRoPE_FP32
    COMMAND $<TARGET_FILE:v2_perf_cpu_rope_fp32>
    LABELS "V2;Performance;RoPE;FP32;Latency"
    MPI_PROCS 1
)

# Performance Test: CPUSwiGLUKernel (FP32)
# Benchmarks CPU SwiGLU kernel latency and throughput
add_executable(v2_perf_cpu_swiglu_fp32 performance/kernels/cpu/Perf__CPUSwiGLUKernel.cpp)
target_link_libraries(v2_perf_cpu_swiglu_fp32
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
set_target_properties(v2_perf_cpu_swiglu_fp32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_CpuSwiGLU_FP32
    COMMAND $<TARGET_FILE:v2_perf_cpu_swiglu_fp32>
    LABELS "V2;Performance;SwiGLU;FP32;Latency"
    MPI_PROCS 1
)

# Performance Test: Qwen2Pipeline
add_executable(v2_perf_qwen2_pipeline performance/pipelines/qwen/Perf__Qwen2Pipeline.cpp)
target_link_libraries(v2_perf_qwen2_pipeline
    llaminar2_core
    GTest::gtest
    GTest::gtest_main
)
set_target_properties(v2_perf_qwen2_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_Qwen2Pipeline
    COMMAND $<TARGET_FILE:v2_perf_qwen2_pipeline>
    LABELS "V2;Performance;Pipeline;Qwen2"
    MPI_PROCS 1
)

# Performance Test: CPURMSNormKernel (FP32)
# Benchmarks CPU RMSNorm kernel latency and throughput
add_executable(v2_perf_cpu_rmsnorm_fp32 performance/kernels/cpu/Perf__CPURMSNormKernel.cpp)
target_link_libraries(v2_perf_cpu_rmsnorm_fp32
    llaminar2_core
    GTest::gtest
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
set_target_properties(v2_perf_cpu_rmsnorm_fp32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/performance"
)

add_v2_perf_test(V2_Perf_CpuRMSNorm_FP32
    COMMAND $<TARGET_FILE:v2_perf_cpu_rmsnorm_fp32>
    LABELS "V2;Performance;RMSNorm;FP32;Latency"
    MPI_PROCS 1
)

add_executable(v2_test_mpi_attention_orchestrator_masking unit/pipelines/Test__MpiAttentionOrchestrator_Masking.cpp)
target_link_libraries(v2_test_mpi_attention_orchestrator_masking llaminar2_core GTest::gtest_main)

add_v2_test(V2_Unit_MpiAttentionOrchestrator_Masking
    COMMAND v2_test_mpi_attention_orchestrator_masking
    LABELS "V2;Unit;PipelineExecution;Orchestration;Attention"
    MPI_PROCS 1
)

# Test: Qwen2Pipeline Batch Handling
add_executable(v2_test_qwen2_pipeline_batch_handling unit/pipelines/Test__Qwen2Pipeline_BatchHandling.cpp)
target_link_libraries(v2_test_qwen2_pipeline_batch_handling llaminar2_core GTest::gtest_main)

add_v2_test(V2_Unit_Qwen2Pipeline_BatchHandling
    COMMAND v2_test_qwen2_pipeline_batch_handling
    LABELS "V2;Unit;PipelineExecution;Qwen2;BatchProcessing;ResidualConnections"
    MPI_PROCS 1
)
