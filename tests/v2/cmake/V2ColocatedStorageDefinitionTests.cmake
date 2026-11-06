#[[
@file V2ColocatedStorageDefinitionTests.cmake
@brief Register graph-only storage lifetime and finite native-frontier replay prerequisites.

These device-free gates prove original capture/definition ownership only.
Colocated runtime activation remains guarded until its complete original source,
parent transaction and terminal/cancellation composition is implemented.
]]
if(TARGET v2_test_pooled_remote_graph_host)
    add_v2_test(V2_Integration_RecordedStorageDefinitionOriginalSourceReplay
        COMMAND $<TARGET_FILE:v2_test_pooled_remote_graph_host>
            --gtest_filter=PooledRemoteGraphHost.RecordedStorageBodyAuthenticatesSourceAndRetiresAfterBorrow
            --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Ownership;GraphCapture;TraceReplay;Regression"
        NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1;LLAMINAR_SKIP_CUDA_STARTUP=1;LLAMINAR_SKIP_ROCM_STARTUP=1;GTEST_DEATH_TEST_STYLE=threadsafe")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_RecordedStorageDefinitionOriginalSourceReplay)
endif()
if(TARGET v2_test_compute_graph_execution)
    add_v2_test(V2_Integration_FiniteStorageParallelBodyFrontierReplay
        COMMAND $<TARGET_FILE:v2_test_compute_graph_execution>
            --gtest_filter=NativeParallelGraphBranch.* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;GraphCapture;Ownership;TraceReplay;Regression"
        NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FiniteStorageParallelBodyFrontierReplay)
endif()

# Exact original main/finite-body terminal joins; native GPU runs remain separate.
if(TARGET v2_test_compute_graph_execution)
    add_v2_test(V2_Integration_FiniteChildOriginalTerminalJoinReplay
        COMMAND $<TARGET_FILE:v2_test_compute_graph_execution>
            --gtest_filter=NativeParallelGraphBranch.FiniteChildJoin* --gtest_repeat=20
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;GraphCapture;Ownership;TraceReplay;Regression"
        NO_MPI NO_MODELS MPI_PROCS 1 TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_FiniteChildOriginalTerminalJoinReplay)
endif()
