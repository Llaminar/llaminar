# @file V2NativeOrderingUnattemptedTests.cmake
# @brief Focused production regressions for cancellation before any native event creator.
if(TARGET v2_test_native_ordering_event_group)
    add_v2_test(V2_Integration_NativeOrderingUnattemptedReservationReplay
        COMMAND $<TARGET_FILE:v2_test_native_ordering_event_group>
            --gtest_filter=NativeOrderingEventGroup.UnattemptedClaimsCancelWithoutPhysicalRetirement --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_NativeOrderingUnattemptedReservationReplay)
endif()
if(TARGET v2_test_pooled_ordering_root_lifetime)
    add_v2_test(V2_Integration_PooledUninitializedRootReservationReplay
        COMMAND $<TARGET_FILE:v2_test_pooled_ordering_root_lifetime>
            --gtest_filter=PooledImmutableOrderingRootLifetime.UninitializedRootsCancelGroupBeforeReset --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledUninitializedRootReservationReplay)
endif()
