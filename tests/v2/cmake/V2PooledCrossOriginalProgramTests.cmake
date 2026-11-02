# @file V2PooledCrossOriginalProgramTests.cmake
# @brief Source-vs-storage role and exact two-channel budget replay before original same-PMA admission.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE unit/kernels/qwen4exp/Test__PooledCrossOriginalProgramPlan.cpp)
    add_v2_test(V2_Unit_PooledCrossOriginalProgramPlan
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=PooledCrossOriginalProgramPlan.*
        LABELS "V2;Unit;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    add_v2_test(V2_Integration_PooledCrossOriginalProgramPlanReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=PooledCrossOriginalProgramPlan.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCrossOriginalProgramPlanReplay)
endif()

# Missing sparse oracle rows reject before any native publication in the actual fixture helper.
if(TARGET v2_test_physical_memory_authority)
    add_v2_test(V2_Integration_PooledCrossSparseOracleInputReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority>
            --gtest_filter=PooledCrossOriginalProgramPlan.SparseOracleRejectsUnauthoredRowsBeforeNativeInput --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ngram;Source;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledCrossSparseOracleInputReplay)
endif()
