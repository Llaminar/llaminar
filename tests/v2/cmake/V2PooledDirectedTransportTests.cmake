# Focused source-owned directed geometry and event-free cold channel lifecycle.
if(TARGET v2_test_qwen4exp_primitives)
    target_sources(v2_test_qwen4exp_primitives PRIVATE
        unit/kernels/qwen4exp/Test__PooledImmutableDirectedTransport.cpp)
    add_v2_test(V2_Integration_PooledImmutableDirectedTransportReplay
        COMMAND $<TARGET_FILE:v2_test_qwen4exp_primitives>
            --gtest_filter=PooledImmutableDirectedTransport.*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;LiveExtent;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_PooledImmutableDirectedTransportReplay)
endif()
if(TARGET v2_test_captured_transfer_channel_binding)
    add_v2_test(V2_Integration_CapturedChannelColdInitializationReplay
        COMMAND $<TARGET_FILE:v2_test_captured_transfer_channel_binding>
            --gtest_filter=CapturedTransferChannelBindingTest.ColdInitialization*:CapturedTransferChannelBindingTest.PartialSetup*
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;Memory;Transfer;Ownership;TraceReplay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=1" "LLAMINAR_SKIP_CUDA_STARTUP=1" "LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS V2_Integration_CapturedChannelColdInitializationReplay)
endif()
