# Exact original prepared birth metadata; no provider probing or key cache.
if(TARGET v2_test_qwen4exp_primitives)
    target_sources(v2_test_qwen4exp_primitives PRIVATE unit/backends/Test__OriginalPreparedBirthKey.cpp)
    add_v2_test(V2_Integration_OriginalPreparedBirthKeyReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives> --gtest_filter=OriginalPreparedBirthKey.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_OriginalPreparedBirthKeyReplay)
endif()
