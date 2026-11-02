# Exact original opaque event groups survive canonical Adaptive plan reconstruction.
if(TARGET v2_test_memory_planner)
    target_sources(v2_test_memory_planner PRIVATE unit/planning/Test__ImmutableOrderingReplan.cpp)
    add_v2_test(V2_Integration_ImmutableOrderingReplanReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner> --gtest_filter=ImmutableOrderingReplanLocal.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ImmutableOrderingReplanReplay)
endif()
