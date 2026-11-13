# Original source-worker/PMA replay uses recorded native bodies, never accelerators.
if(TARGET v2_test_memory_planner)
    target_sources(v2_test_memory_planner PRIVATE
        unit/planning/Test__GPUCacheSourceVectorPlanner.cpp)
    add_v2_test(V2_Integration_GPUCacheSourceVectorPlannerReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUCacheSourceVectorPlanner.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_GPUCacheSourceVectorPlannerReplay)
endif()

if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_RecordedRemoteHostPlanningReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=GPUCacheSourceVectorPlanner.OriginalTwoWorkerSourceFeedsFinalQFNPlanAtC0AndBoundedCapacity --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Planning;Qwen4Exp;Admission;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RecordedRemoteHostPlanningReplay)
endif()
