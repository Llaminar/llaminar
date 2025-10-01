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
add_subdirectory(llama.cpp)



# Source files for the core library
set(LLAMINAR_CORE_SOURCES
    src/common.cpp
    src/argument_parser.cpp
    src/topology_manager.cpp
    src/graph_compute.cpp
    src/mpi_kernel_base.cpp
    src/cosma_prefill_manager.cpp
    src/utils/debug_env.cpp
    src/kernels/MPILinearKernel.cpp
    src/kernels/MPIRMSNormKernel.cpp
    src/kernels/MPIAttentionKernel.cpp
    src/kernels/MPIEmbeddingKernel.cpp
    src/kernels/MPIMLPKernel.cpp
    src/kernels/MPIResidualKernel.cpp
    src/kernels/MPIRoPEKernel.cpp
    src/kernels/MPISwiGLUKernel.cpp
    src/kernels/common/attention_primitives.cpp
    src/kernels/common/normalization.cpp
    src/kernels/common/softmax.cpp
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
    ggml
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
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/llama.cpp/ggml
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
    )

    set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
endfunction()

# Test sources and executables
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic llaminar_core GTest::gtest GTest::gtest_main)
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
    add_llaminar_mpi_test(MPILinearKernelTest 2 test_mpi_linear_kernel)

    # Linear orientation parity test (naive reference vs MPILinearKernel)
    add_executable(test_linear_orientation_parity tests/test_linear_orientation_parity.cpp)
    target_link_libraries(test_linear_orientation_parity llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_linear_orientation_parity GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_linear_orientation_parity PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_linear_orientation_parity ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(LinearOrientationParityTest 2 test_linear_orientation_parity)
    set_tests_properties(LinearOrientationParityTest PROPERTIES TIMEOUT 30)
    set_tests_properties(LinearOrientationParityTest PROPERTIES LABELS "linear;parity;kernel")

    # Minimal MPI finalize probe test (diagnostic for finalize hang)
    add_executable(test_mpi_finalize_probe tests/test_mpi_finalize_probe.cpp)
    target_link_libraries(test_mpi_finalize_probe llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_finalize_probe GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_finalize_probe PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_finalize_probe ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPIFinalizeProbeTest 2 test_mpi_finalize_probe)

    add_executable(test_mpi_rmsnorm_kernel tests/test_mpi_rmsnorm_kernel.cpp)
    target_link_libraries(test_mpi_rmsnorm_kernel llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_rmsnorm_kernel GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_rmsnorm_kernel PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_rmsnorm_kernel ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPIRMSNormKernelTest 2 test_mpi_rmsnorm_kernel)

    # Reconfigure test_mpi_attention_kernel as a standalone minimal MPI binary (no GTest) to avoid
    # pre-main static initialization hang (see tests/AGENTS.md Case Study 11).
    add_executable(test_mpi_attention_kernel tests/test_mpi_attention_kernel.cpp)
    target_compile_definitions(test_mpi_attention_kernel PRIVATE LLAMINAR_STANDALONE_MPI_ATTENTION_TEST=1)
    target_link_libraries(test_mpi_attention_kernel PRIVATE llaminar_core MPI::MPI_CXX)
    # Register as CTest with explicit mpiexec invocation.
    add_test(NAME MPIAttentionKernelTest COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mpi_attention_kernel>)
    set_tests_properties(MPIAttentionKernelTest PROPERTIES TIMEOUT 45 LABELS "attention;mpi;kernel" RUN_SERIAL TRUE)

    # Minimal standalone MPI probe (no GTest, used to debug hangs prior to test harness)
    add_executable(minimal_mpi_probe tests/minimal_mpi_probe.cpp)
    target_link_libraries(minimal_mpi_probe llaminar_core MPI::MPI_CXX)

    # Parity test wrapper integration: use the standalone binary with parameter variants.
    # We keep the binary (test_attention_shard_parity) but add several CTest entries.
    # Variant matrix chosen to exercise small vs medium shapes and uneven head distribution.
    add_test(NAME AttentionShardParity_Small COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_parity> --heads=4 --head-dim=16 --seq-len=5)
    set_tests_properties(AttentionShardParity_Small PROPERTIES TIMEOUT 30 LABELS "attention;parity;sharding")
    add_test(NAME AttentionShardParity_Medium COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_parity> --heads=8 --head-dim=8 --seq-len=32)
    set_tests_properties(AttentionShardParity_Medium PROPERTIES TIMEOUT 30 LABELS "attention;parity;sharding")
    add_test(NAME AttentionShardParity_Uneven COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_parity> --heads=6 --head-dim=12 --seq-len=9)
    set_tests_properties(AttentionShardParity_Uneven PROPERTIES TIMEOUT 30 LABELS "attention;parity;sharding")
    add_test(NAME AttentionShardParity_Stress COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_parity> --heads=12 --head-dim=16 --seq-len=24)

    # --- MLP shard parity tests ---
    add_executable(test_mlp_shard_parity tests/test_mlp_shard_parity.cpp)
    target_link_libraries(test_mlp_shard_parity llaminar_core)
    set_target_properties(test_mlp_shard_parity PROPERTIES OUTPUT_NAME test_mlp_shard_parity)

    # Small (even-ish) partition
    add_test(NAME MLPShardParity_Small COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mlp_shard_parity> --seq-len=5 --d-model=48 --d-ff=64)
    # Medium uneven d_ff
    add_test(NAME MLPShardParity_Medium COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mlp_shard_parity> --seq-len=9 --d-model=64 --d-ff=96)
    # Stress larger dims (still quick)
    add_test(NAME MLPShardParity_Stress COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_mlp_shard_parity> --seq-len=16 --d-model=80 --d-ff=128)

    # --- End-to-End shard parity ---
    add_executable(test_end2end_shard_parity tests/test_end2end_shard_parity.cpp)
    target_link_libraries(test_end2end_shard_parity llaminar_core)
    set_target_properties(test_end2end_shard_parity PROPERTIES OUTPUT_NAME test_end2end_shard_parity)
    add_test(NAME End2EndShardParity_Small COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_end2end_shard_parity> --heads=4 --head-dim=8 --seq-len=6 --d-ff=48)
    add_test(NAME End2EndShardParity_UnevenHeads COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_end2end_shard_parity> --heads=6 --head-dim=8 --seq-len=5 --d-ff=56)
    set_tests_properties(AttentionShardParity_Stress PROPERTIES TIMEOUT 60 LABELS "attention;parity;sharding;stress")

    # Diagnostic path test: force loader sharded-mode + shard diagnostics for a tiny config to exercise verification logic.
    add_test(NAME AttentionShardLoad_Diag
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_parity> --heads=4 --head-dim=8 --seq-len=4)
    # Environment variables enabling shard-mode and diagnostic logging
    set_tests_properties(AttentionShardLoad_Diag PROPERTIES
        TIMEOUT 40
        LABELS "attention;parity;sharding;diagnostic"
        ENVIRONMENT "LLAMINAR_SHARD_LOAD_DIAG=1")

    # Uneven head distribution diagnostic (5 heads -> ranks get 3 and 2) to stress offset logic
    add_test(NAME AttentionShardLoad_Diag_Uneven
             COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_shard_parity> --heads=5 --head-dim=8 --seq-len=6)
    set_tests_properties(AttentionShardLoad_Diag_Uneven PROPERTIES
        TIMEOUT 50
        LABELS "attention;parity;sharding;diagnostic;uneven"
        ENVIRONMENT "LLAMINAR_SHARD_LOAD_DIAG=1")

    # --- Role tagging synthetic log capture test (no MPI) ---
    add_executable(test_role_tagging_logs tests/test_role_tagging_logs.cpp)
    target_link_libraries(test_role_tagging_logs llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_role_tagging_logs GTest::gtest GTest::gtest_main)
    endif()
    add_test(NAME RoleTaggingSyntheticTest COMMAND test_role_tagging_logs)
    set_tests_properties(RoleTaggingSyntheticTest PROPERTIES TIMEOUT 15 LABELS "logging;roles;synthetic")

    # --- Role tagging integration tests (per quant group via GTest discovery) ---
    add_executable(test_model_load_role_tags tests/test_model_load_role_tags.cpp)
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
    add_executable(test_attention_shard_parity tests/test_attention_shard_parity.cpp)
    add_executable(test_attention_missing_reduction_negative tests/test_attention_missing_reduction_negative.cpp)
    target_link_libraries(test_attention_shard_parity llaminar_core)
    target_link_libraries(test_attention_missing_reduction_negative llaminar_core)
    # Standalone binary (no GTest) used for debugging; do not register with add_llaminar_mpi_test macro.
    add_test(NAME AttentionShardParityStandalone 
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_attention_shard_parity>)
    set_tests_properties(AttentionShardParityStandalone PROPERTIES TIMEOUT 30)
    set_tests_properties(AttentionShardParityStandalone PROPERTIES LABELS "attention;parity;sharding;standalone")

    # Negative misuse test: ensures missing reduction leads to detectable mismatch
    add_test(NAME AttentionShardMissingReduction_Negative
        COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:test_attention_missing_reduction_negative> --heads=4 --head-dim=16 --seq-len=5)
    set_tests_properties(AttentionShardMissingReduction_Negative PROPERTIES TIMEOUT 30 LABELS "attention;negative;sharding")

    # Attention micro parity test (single small config, scalar reference comparison)
    add_executable(test_attention_micro tests/test_attention_micro.cpp)
    target_link_libraries(test_attention_micro llaminar_core)
    add_executable(test_attention_primitives tests/test_attention_primitives.cpp)
    target_link_libraries(test_attention_primitives llaminar_core)
    if (GTest_FOUND)
        target_link_libraries(test_attention_primitives GTest::gtest GTest::gtest_main pthread)
    else()
        target_include_directories(test_attention_primitives PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_primitives ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    # Single-rank pure math primitives test (no MPI). Register with CTest explicitly.
    # Executable may live either in top-level or tests/ depending on generator/output rules.
    if (EXISTS ${CMAKE_BINARY_DIR}/tests/test_attention_primitives)
        add_test(NAME AttentionPrimitivesTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_attention_primitives)
    else()
        add_test(NAME AttentionPrimitivesTest COMMAND ${CMAKE_BINARY_DIR}/test_attention_primitives)
    endif()
    set_tests_properties(AttentionPrimitivesTest PROPERTIES TIMEOUT 5)
    if(GTest_FOUND)
        target_link_libraries(test_attention_micro GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_attention_micro PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_attention_micro ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(AttentionMicroTest 2 test_attention_micro)
    set_tests_properties(AttentionMicroTest PROPERTIES TIMEOUT 30)
    set_tests_properties(AttentionMicroTest PROPERTIES LABELS "attention;parity;kernel")

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
    add_executable(test_mpi_embedding_kernel tests/test_mpi_embedding_kernel.cpp)
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
    add_executable(test_mpi_embedding_kernel_extended tests/test_mpi_embedding_kernel.cpp)
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
    
    # MPI Transformer Pipeline test
    add_executable(test_mpi_transformer_pipeline tests/test_mpi_transformer_pipeline.cpp)
    target_link_libraries(test_mpi_transformer_pipeline llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_mpi_transformer_pipeline GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_mpi_transformer_pipeline PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_mpi_transformer_pipeline ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(MPITransformerPipelineTest 2 test_mpi_transformer_pipeline)
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
    add_llaminar_mpi_test(AdaptiveMatMulTest 2 test_adaptive_matmul)
    
    # Minimal COSMA Custom Layout test
    add_executable(test_cosma_minimal tests/test_cosma_minimal.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_minimal llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_minimal GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_minimal PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_minimal ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_llaminar_mpi_test(CosmaMinimalTest 2 test_cosma_minimal)

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
    add_llaminar_mpi_test(CosmaPrefillManagerMultiTest 2 test_cosma_prefill_manager_multi)
    set_tests_properties(CosmaPrefillManagerMultiTest PROPERTIES TIMEOUT 60)

    # Cosma Prefill Manager distributed full-path (large GEMM) correctness test
    add_executable(test_cosma_prefill_manager_distributed tests/test_cosma_prefill_manager_distributed.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_distributed PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaPrefillManagerDistributedFullTest 2 test_cosma_prefill_manager_distributed)
    set_tests_properties(CosmaPrefillManagerDistributedFullTest PROPERTIES TIMEOUT 60)

    add_executable(test_cosma_prefill_manager_paths tests/test_cosma_prefill_manager_paths.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_paths PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaPrefillManagerPathsTest 2 test_cosma_prefill_manager_paths)
    set_tests_properties(CosmaPrefillManagerPathsTest PROPERTIES TIMEOUT 60)

    add_executable(test_cosma_prefill_attention_integration tests/test_cosma_prefill_attention_integration.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_attention_integration PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaPrefillAttentionIntegrationTest 2 test_cosma_prefill_attention_integration)
    set_tests_properties(CosmaPrefillAttentionIntegrationTest PROPERTIES TIMEOUT 120)

    add_executable(test_cosma_prefill_manager_orientation_autofix tests/test_cosma_prefill_manager_orientation_autofix.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_orientation_autofix llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager_orientation_autofix GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager_orientation_autofix PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager_orientation_autofix ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerOrientationAutoFixTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager_orientation_autofix)
    set_tests_properties(CosmaPrefillManagerOrientationAutoFixTest PROPERTIES TIMEOUT 60)

    add_executable(test_cosma_prefill_manager_env_audit tests/test_cosma_prefill_manager_env_audit.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_env_audit llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager_env_audit GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager_env_audit PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager_env_audit ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerEnvAuditTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager_env_audit)

    add_executable(test_cosma_prefill_manager_stats tests/test_cosma_prefill_manager_stats.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_stats llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager_stats GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager_stats PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager_stats ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerStatsTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager_stats)
    set_tests_properties(CosmaPrefillManagerStatsTest PROPERTIES TIMEOUT 60)

    add_executable(test_cosma_prefill_manager_stats_dump tests/test_cosma_prefill_manager_stats_dump.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_manager_stats_dump llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_prefill_manager_stats_dump GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_prefill_manager_stats_dump PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_prefill_manager_stats_dump ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaPrefillManagerStatsDumpTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_prefill_manager_stats_dump)

    # Population regression (destination-local round-trip invariants)
    add_executable(test_cosma_prefill_population_regression tests/test_cosma_prefill_population_regression.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_population_regression PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaPrefillManagerPopulationRegressionTest 2 test_cosma_prefill_population_regression)
    set_tests_properties(CosmaPrefillManagerPopulationRegressionTest PROPERTIES TIMEOUT 60)

    # Population performance benchmark (env-gated)
    add_executable(test_cosma_prefill_population_perf tests/test_cosma_prefill_population_perf.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_prefill_population_perf PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaPrefillManagerPopulationPerfTest 2 test_cosma_prefill_population_perf)
    set_tests_properties(CosmaPrefillManagerPopulationPerfTest PROPERTIES TIMEOUT 120)

    add_executable(test_prefill_attention_golden tests/test_prefill_attention_golden.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_prefill_attention_golden PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(PrefillAttentionGoldenTest 2 test_prefill_attention_golden)
    add_executable(test_tp_partition tests/test_tp_partition.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_tp_partition PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_test(NAME TPPartitionSpecTest COMMAND test_tp_partition)

    # Embedding orientation diagnostic test (non-MPI critical, rank 0 does work)
    add_executable(test_embedding_orientation tests/test_embedding_orientation.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_embedding_orientation PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
    target_include_directories(test_embedding_orientation PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/src)
    add_test(NAME EmbeddingOrientationTest COMMAND test_embedding_orientation)

    # Dedicated fused RMSNorm+QKV correctness test (isolates CosmaPrefillManager::fused_rmsnorm_qkv)
    add_executable(test_cosma_fused_rmsnorm_qkv tests/test_cosma_fused_rmsnorm_qkv.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_fused_rmsnorm_qkv PRIVATE llaminar_core llama GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaFusedRmsnormQkvTest 2 test_cosma_fused_rmsnorm_qkv)
    set_tests_properties(PrefillAttentionGoldenTest PROPERTIES TIMEOUT 180)

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
    target_link_libraries(test_dequant_golden llaminar_core llama)
    if(GTest_FOUND)
        target_link_libraries(test_dequant_golden GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_dequant_golden PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_dequant_golden ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME DequantGoldenTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_dequant_golden)

    # Model loader alignment regression test (ensures offset chain + alignment invariants)
    add_executable(test_model_loader_alignment tests/test_model_loader_alignment.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
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
    add_executable(test_integration_q4k tests/test_integration_q4k.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_integration_q4k llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_integration_q4k GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_integration_q4k PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_integration_q4k ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME IntegrationQ4K COMMAND ${CMAKE_BINARY_DIR}/tests/test_integration_q4k)

    # Golden model loader comparison (float tensors vs llama.cpp reference)
    add_executable(test_model_loader_golden tests/test_model_loader_golden.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_model_loader_golden PRIVATE llaminar_core llama GTest::gtest_main)
    # Executable ended up at top-level build dir (not in tests/); reference directly
    if (EXISTS ${CMAKE_BINARY_DIR}/tests/test_model_loader_golden)
        add_test(NAME ModelLoaderGoldenTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_model_loader_golden)
    else()
        add_test(NAME ModelLoaderGoldenTest COMMAND ${CMAKE_BINARY_DIR}/test_model_loader_golden)
    endif()
    set_tests_properties(ModelLoaderGoldenTest PROPERTIES TIMEOUT 60)
    
    # In-layout elementwise (RMSNorm + SwiGLU) single-rank correctness test
    add_executable(test_cosma_in_layout_elementwise tests/test_cosma_in_layout_elementwise.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_in_layout_elementwise llaminar_core)
    if(GTest_FOUND)
        target_link_libraries(test_cosma_in_layout_elementwise GTest::gtest GTest::gtest_main)
    else()
        target_include_directories(test_cosma_in_layout_elementwise PRIVATE ${GTEST_INCLUDE_DIR})
        target_link_libraries(test_cosma_in_layout_elementwise ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
    endif()
    add_test(NAME CosmaInLayoutElementwiseTest COMMAND ${CMAKE_BINARY_DIR}/tests/test_cosma_in_layout_elementwise)
    set_tests_properties(CosmaInLayoutElementwiseTest PROPERTIES TIMEOUT 30)

    # Multi-rank in-layout elementwise (RMSNorm + SwiGLU) correctness
    add_executable(test_cosma_in_layout_elementwise_multi tests/test_cosma_in_layout_elementwise_multi.cpp $<TARGET_OBJECTS:test_logging_bootstrap>)
    target_link_libraries(test_cosma_in_layout_elementwise_multi PRIVATE llaminar_core GTest::gtest_main MPI::MPI_CXX)
    add_llaminar_mpi_test(CosmaInLayoutElementwiseMultiTest 2 test_cosma_in_layout_elementwise_multi)

    # Quant / Dequant parity tests (single-rank deterministic)
    add_executable(test_quant_dequant_parity tests/test_quant_dequant_parity.cpp)
    target_link_libraries(test_quant_dequant_parity llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME QuantDequantParityTest COMMAND test_quant_dequant_parity)

    # Quantized shard regression test (validates column/row shard parity for quant tensors)
    add_executable(test_model_loader_quant_shard_regression tests/test_model_loader_quant_shard_regression.cpp)
    target_link_libraries(test_model_loader_quant_shard_regression llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME ModelLoaderQuantShardRegression COMMAND test_model_loader_quant_shard_regression)
    add_executable(test_quant_shard_cache_stats tests/test_quant_shard_cache_stats.cpp)
    target_link_libraries(test_quant_shard_cache_stats llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME QuantShardCacheStatsTest COMMAND test_quant_shard_cache_stats)
    add_executable(test_partial_decode_q4_0 tests/test_partial_decode_q4_0.cpp)
    target_link_libraries(test_partial_decode_q4_0 llaminar_core GTest::gtest GTest::gtest_main)
    add_test(NAME PartialDecodeQ4_0Test COMMAND test_partial_decode_q4_0)
    # Multi-format partial decode parity (legacy formats Q4_1/Q5_1/Q8_1 removed)
    if (EXISTS ${CMAKE_SOURCE_DIR}/tests/test_partial_decode_multi_formats.cpp)
        add_executable(test_partial_decode_multi_formats tests/test_partial_decode_multi_formats.cpp)
        target_link_libraries(test_partial_decode_multi_formats llaminar_core GTest::gtest GTest::gtest_main)
        add_test(NAME PartialDecodeMultiFormatsTest COMMAND test_partial_decode_multi_formats)
    endif()
    # Aggregated parity test iterating present supported formats (Q4_0, Q5_0, Q8_0, K variants)
    if (EXISTS ${CMAKE_SOURCE_DIR}/tests/test_partial_decode_aggregated.cpp)
        add_executable(test_partial_decode_aggregated tests/test_partial_decode_aggregated.cpp)
        target_link_libraries(test_partial_decode_aggregated llaminar_core GTest::gtest GTest::gtest_main)
        add_test(NAME PartialDecodeAggregatedTest COMMAND test_partial_decode_aggregated)
    endif()
    if (EXISTS ${CMAKE_SOURCE_DIR}/tests/test_partial_decode_k_formats.cpp)
        add_executable(test_partial_decode_k_formats tests/test_partial_decode_k_formats.cpp)
        target_link_libraries(test_partial_decode_k_formats llaminar_core GTest::gtest GTest::gtest_main)
        add_test(NAME PartialDecodeKFormatsTest COMMAND test_partial_decode_k_formats)
    endif()

    # (Removed) Q5_1 layout micro test – deprecated quantization formats are no longer supported.
    # K-format (Q4_K / Q4_K_M) partial decode parity (added once implementation matures)
    # Placeholder future extension: we could add a dedicated test when models contain those tensors.
    set_tests_properties(QuantDequantParityTest PROPERTIES TIMEOUT 10)

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
        test_linear_orientation_parity
        test_mpi_rmsnorm_kernel
        test_mpi_attention_kernel
    test_attention_micro
    test_attention_primitives
        test_mpi_transformer_pipeline
        test_adaptive_matmul
        test_cosma_minimal
        test_cosma_prefill_manager
        test_cosma_prefill_manager_multi
        test_cosma_prefill_manager_distributed
        test_cosma_prefill_manager_paths
        test_cosma_prefill_manager_orientation_autofix
        test_cosma_prefill_manager_env_audit
        test_cosma_prefill_manager_stats
        test_cosma_prefill_manager_stats_dump
        test_dequant
        test_dequant_extra
        test_dequant_golden
        test_integration_q4k
        test_cosma_in_layout_elementwise
        test_cosma_in_layout_elementwise_multi)
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
