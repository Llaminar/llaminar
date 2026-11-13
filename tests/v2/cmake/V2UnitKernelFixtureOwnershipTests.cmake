# Real factory construction and actual orchestrator cache replay stay device-free.
if(HAVE_ROCM AND TARGET v2_test_kernel_factory_ops)
    add_v2_test(V2_Integration_ROCmEmbeddingFactoryOwnerContractReplay
        COMMAND $<TARGET_FILE:v2_test_kernel_factory_ops>
            --gtest_filter=Test__KernelFactoryOps.CreateEmbedding_*_ROCm:Test__KernelFactoryOps.CreateEmbedding_ROCmRequiresExplicitImmutableOwner
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;ROCm;Embedding;Factory;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_ROCmEmbeddingFactoryOwnerContractReplay)
endif()
if(TARGET v2_test_device_graph_orchestrator)
    add_v2_test(V2_Integration_DeviceGraphContextCacheIdentityReplay
        COMMAND $<TARGET_FILE:v2_test_device_graph_orchestrator>
            --gtest_filter=Test__DeviceGraphOrchestrator.GetDeviceContextMultipleDevices
        LABELS "V2;Integration;ProductionTestPreflight;DeviceFree;CPU;Graph;ContextCache;Ownership;Replay;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 30)
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_DeviceGraphContextCacheIdentityReplay)
endif()
