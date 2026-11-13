# Exact setup snapshot survives mutable diagnostic controls after admission.
if(TARGET v2_test_memory_planner)
    add_v2_test(V2_Integration_ReadAheadRuntimeSnapshotReplay
        COMMAND $<TARGET_FILE:v2_test_memory_planner>
            --gtest_filter=RankMemoryPlanInputs.ReadAheadSelectionSnapshotSurvivesDebugMutationAndAdapters
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_ReadAheadRuntimeSnapshotReplay)
endif()
