# @file V2PooledProviderOriginalProgramTests.cmake
# @brief Actual provider declaration envelopes and separately retained native obligations.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE unit/kernels/qwen4exp/Test__PooledProviderOriginalProgramPlan.cpp)
    add_v2_test(V2_Unit_PooledProviderOriginalProgramPlan
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=PooledProviderOriginalProgramPlan.*
        LABELS "V2;Unit;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    add_v2_test(V2_Integration_PooledProviderOriginalProgramPlanReplay
        COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=PooledProviderOriginalProgramPlan.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression" MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledProviderOriginalProgramPlanReplay)
endif()
