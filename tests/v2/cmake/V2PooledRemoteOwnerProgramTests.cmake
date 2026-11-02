# @file V2PooledRemoteOwnerProgramTests.cmake
# @brief Fast device-free source-author/actual-launch-plan declaration regressions.
if(TARGET v2_test_physical_memory_authority)
    target_sources(v2_test_physical_memory_authority PRIVATE
        unit/kernels/qwen4exp/Test__PooledImmutableOwnerLaunchPlan.cpp
        unit/kernels/qwen4exp/Test__PooledImmutableDirectedOwnerProgram.cpp)
    foreach(_case IN ITEMS
        PooledImmutableOwnerLaunchPlan.ActualOwnerDispatchAndEnumHolesReplay
        PooledImmutableDirectedOwnerProgram.ResetPartialAndPlacementClassReplay
        PooledImmutableDirectedOwnerProgram.CompiledRowKeyABIChangesPortableScope
        PooledImmutableDirectedOwnerProgram.UnsupportedOwnerCannotDeclareNativeScope)
        string(REPLACE "." "_" _name "${_case}")
        add_v2_test(V2_Unit_${_name}
            COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=${_case}
            LABELS "V2;Unit;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
        add_v2_test(V2_Integration_${_name}Replay
            COMMAND $<TARGET_FILE:v2_test_physical_memory_authority> --gtest_filter=${_case} --gtest_repeat=20
            LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Ownership;Ngram;TraceReplay;Regression"
            MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
            ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
        list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_${_name}Replay)
    endforeach()
endif()
