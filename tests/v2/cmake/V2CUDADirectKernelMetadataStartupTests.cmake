# Explicit first-use native admission replay, isolated from other fixture cases.
if(TARGET v2_integration_cuda_native_vnni_prefill_staging)
    # The selected case authenticates first capture on every visible ordinal.
    add_v2_test(V2_Integration_CUDANativeVNNIPrefillMetadataFirstUse
        COMMAND $<TARGET_FILE:v2_integration_cuda_native_vnni_prefill_staging>
            --gtest_filter=CUDANativeVNNIPrefillStaging.BK256CapturedLaunchOwnsEveryDeviceContext
        LABELS "V2;Integration;ProductionTestPreflight;CUDA;NativeVNNI;HardwareMetadata;Startup;GraphCapture;Regression"
        MPI_PROCS 1 NO_MPI NO_MODELS TIMEOUT 90
        RESOURCE_LOCK "cuda_gpu_0;cuda_gpu_1"
        ENVIRONMENT "LLAMINAR_FORCE_CPU_ONLY_STARTUP=0;LLAMINAR_SKIP_CUDA_STARTUP=0;LLAMINAR_SKIP_ROCM_STARTUP=1")
    list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS
        V2_Integration_CUDANativeVNNIPrefillMetadataFirstUse)
endif()
