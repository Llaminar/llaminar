# Replay the actual immutable-owner state machine without a backend or driver.
add_executable(v2_unit_rocm_embedding_execution_binding
    unit/backends/Test__ROCmEmbeddingExecutionBinding.cpp)
target_include_directories(v2_unit_rocm_embedding_execution_binding PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/v2")
target_compile_features(v2_unit_rocm_embedding_execution_binding PRIVATE cxx_std_20)
target_link_libraries(v2_unit_rocm_embedding_execution_binding PRIVATE GTest::gtest_main)
add_v2_test(V2_Unit_ROCmEmbeddingExecutionBinding
    COMMAND $<TARGET_FILE:v2_unit_rocm_embedding_execution_binding>
    LABELS "V2;Unit;DeviceFree;ROCm;Embedding;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS)
add_v2_test(V2_Integration_ROCmEmbeddingExecutionBindingReplay
    COMMAND $<TARGET_FILE:v2_unit_rocm_embedding_execution_binding>
    LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;ROCm;Embedding;Ownership;Replay;Regression"
    MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
    V2_Integration_ROCmEmbeddingExecutionBindingReplay)
if(TARGET v2_integration_rocm_embedding_grouped_verifier)
    add_v2_test(V2_Integration_ROCmEmbeddingOwnerCapturedReplay
        COMMAND $<TARGET_FILE:v2_integration_rocm_embedding_grouped_verifier>
            --gtest_filter=Test__ROCmEmbeddingGroupedVerifier.ImmutableOwnerResetRebindRetainsCapturedBytes
        LABELS "V2;Integration;ProductionTestPreflight;ROCm;Embedding;Ownership;Captured;AllFormats;Replay;Regression"
        MPI_PROCS 1 NO_MODELS RESOURCE_LOCK rocm_gpu_0 TIMEOUT 120)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_ROCmEmbeddingOwnerCapturedReplay)
endif()
