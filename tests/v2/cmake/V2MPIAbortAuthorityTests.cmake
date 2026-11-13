# @file V2MPIAbortAuthorityTests.cmake
# @brief Verify fatal command cleanup through the original coordinated rank authority.
if(TARGET v2_unit_mpi_coordinated_mode)
    add_v2_test(V2_Integration_MPIAbortAuthorityLifecycle
        COMMAND $<TARGET_FILE:v2_unit_mpi_coordinated_mode>
            --gtest_filter=Test__MPICoordinatedMode.AbortAuthorityPublishesOriginalShutdownExactlyOnce:Test__MPICoordinatedMode.AbortFollowerPreservesFailureWithoutRootPublication --gtest_brief=1
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;MPI;Orchestration;Lifecycle;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;OMP_NUM_THREADS=1;OMP_DYNAMIC=false")
    if(TEST V2_Integration_MPIAbortAuthorityLifecycle)
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_MPIAbortAuthorityLifecycle)
    endif()
endif()
